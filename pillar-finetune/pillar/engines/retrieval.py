import time
from collections import OrderedDict, defaultdict

import torch
import wandb
from tqdm import tqdm
import rve

from pillar.utils.logging import logger
from pillar.utils.engine import gather_predictions_dict, prefix_dict, to_cpu_recursive
from pillar.utils.misc import AverageMeter, Summary, ProgressMeter, get_is_master
from pillar.utils.windowing import batch_apply_ct_windowing

from .base import Engine

from pillar import augmentations


def get_augmentations(image_augmentations, args):
    augmentations_list = []
    for augmentation in image_augmentations:
        name = augmentation["type"]
        kwargs = augmentation["kwargs"] if "kwargs" in augmentation else {}
        augmentation = augmentations.__dict__[name](args, **kwargs)
        augmentations_list.append(augmentation)

    return augmentations_list


class Retrieval(Engine):
    def __init__(self, *args, log_interval=None, log_loss_components=None, **kwargs):
        super().__init__(*args, **kwargs)

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        dataloader,
        optimizer,
        device,
        epoch,
        loss_scaler,
        lr_scheduler,
        args,
        log_interval=50,
        clip_grad=None,
        log_loss_components=False,
    ):
        model.train()

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")

        lr = AverageMeter("lr", ":.4e", summary_type=Summary.NONE)

        max_mem = AverageMeter("Max mem", ":.0f", summary_type=Summary.NONE)
        cur_mem = AverageMeter("Cur mem", ":.0f", summary_type=Summary.NONE)  # 추가
        progress = ProgressMeter(len(dataloader), [batch_time, data_time, lr, losses, max_mem, cur_mem], prefix="Epoch: [{}]".format(epoch))        
        epoch_metrics_configured = len(self.get_epoch_metrics(split="train")) > 0

        # NOTE: 루프가 한 번도 안 돌 경우를 대비한 안전한 기본값
        # (dataloader가 비었거나 limit_num_batches == 0인 극단적 케이스)
        lr_value = optimizer.param_groups[0]["lr"]
        has_data = False

        end = time.time()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training", disable=not get_is_master())):
            data_time.update(time.time() - end)
            max_mem.update(torch.cuda.max_memory_allocated() / (1024 * 1024))
            cur_mem.update(torch.cuda.memory_allocated() / (1024 * 1024))  # 추가
            if batch_idx == self.limit_num_batches:
                break

            if batch is None:
                # Potentially corrupted data
                # NOTE: end를 갱신해 타이밍 통계가 스킵된 시간을 포함하지 않도록 함
                end = time.time()
                continue

            has_data = True

            # From MAE
            # we use a per iteration (instead of per epoch) lr scheduler
            if (lr_scheduler is not None) and (batch_idx % self.accum_iter == 0):
                lr_scheduler.adjust_learning_rate(batch_idx / len(dataloader) + epoch)

            result = OrderedDict()

            with torch.amp.autocast("cuda", dtype=self.amp_precision, enabled=self.amp_precision is not None):
                loss, logging_dict, predictions_dict = self.step(model, batch, batch_idx, epoch=epoch, split="train", device=device)

            loss /= self.accum_iter
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                clip_grad=clip_grad,
                create_graph=False,
                need_update=(batch_idx + 1) % self.accum_iter == 0,
            )

            # gradient accumulation
            if (batch_idx + 1) % self.accum_iter == 0:
                optimizer.zero_grad()
                self.global_step += 1

            losses.update(loss.item(), batch["x"].size(0))

            # lr logging
            lr_value = optimizer.param_groups[0]["lr"]
            lr.update(lr_value)

            # train_ 접두사는 epoch metric 설정과 **무관하게** 항상 붙인다.
            # 이전에는 이 두 줄이 아래 if 블록 안에 있어서, epoch_metrics.train: [] 인
            # config(retrieval 런이 CPU RAM OOM 때문에 쓰는 설정)에서는 아래 wandb 루프가
            # 접두사 없는 sim_loss/cls_loss/... 로 찍었다. val_* 와 이름 규칙이 갈려서
            # train/val 성분을 나란히 비교하는 것이 조용히 어려워진다(실제로 job 27890344
            # 진단 때 "train 성분이 안 찍힌다"고 잘못 판단할 뻔했다).
            # logging is not synchronized across processes
            logging_dict = prefix_dict(logging_dict, "train_")
            logging_dict["train_loss"] = loss.detach()

            # If there are no epoch metrics configured, we will not save results to `training_step_outputs`
            if epoch_metrics_configured:
                result["logs"] = logging_dict

                if self.args.main.multi_gpu:
                    predictions_dict = gather_predictions_dict(predictions_dict)

                # gather 이후에 CPU로 이동 (튜플 안의 텐서까지 재귀적으로 이동시켜야 GPU 메모리 누수 방지)
                predictions_dict = {k: to_cpu_recursive(v) for k, v in predictions_dict.items()}

                result.update(predictions_dict)
                self.training_step_outputs.append(result)

            # collect runtimes
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % log_interval == 0:
                if get_is_master() and not self.args.main.disable_wandb:
                    wandb.log({"train/loss": loss.detach(), "lr": lr_value}, step=self.global_step)
                progress.display(batch_idx + 1, tqdm_write=True)
                if log_loss_components == True and get_is_master() and not self.args.main.disable_wandb:
                    for k, v in logging_dict.items():
                        wandb.log({k: v}, step=self.global_step)

        if not has_data:
            logger.warning(f"Epoch {epoch}: train_one_epoch에서 처리된 배치가 없습니다. "
                            f"dataloader가 비어있거나 모든 배치가 corrupted이거나 limit_num_batches=0일 수 있습니다.")

        # log epoch metrics
        self.on_epoch_end(split="train", device=device, epoch=epoch, metadata={"lr": lr_value})

    def evaluate(
        self,
        model,
        dataloader,
        device,
        epoch=None,
        split="val",
        gather_predictions=False,
        log_loss_components=False,
        ckpt_dir=None,
    ):
        model.eval()

        losses = AverageMeter(f"{split}_loss", ":.4e")
        # NOTE: loss 컴포넌트별 평균을 위한 누적기. 마지막 배치 값만 로깅되던 버그 수정.
        component_meters = defaultdict(lambda: AverageMeter("", ":.6e"))
        has_data = False

        # tqdm progress bar
        desc = "Evaluation" if split == "val" else "Testing"
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} {desc}" if epoch else desc, disable=not get_is_master())):
            if batch_idx == self.limit_num_batches:
                break

            if batch is None:
                # Potentially corrupted data
                continue

            has_data = True
            result = OrderedDict()

            # NOTE: step() 내부에서 amp_precision 기준 autocast를 다시 열기 때문에
            # 여기서 enabled=False로 감싸는 건 실질적으로 무효였음.
            # eval에서 amp를 정말로 끄고 싶다면 step()이 split을 보고 분기하도록 바꿔야 함.
            # 우선 train과 동일하게 self.amp_precision을 그대로 따르도록 정리함.
            with torch.no_grad():
                loss, logging_dict, predictions_dict = self.step(model, batch, batch_idx, epoch=epoch, split=split, device=device)

            # log metrics
            batch_size = batch["x"].size(0)
            losses.update(loss.item(), batch_size)

            # 컴포넌트별 loss도 배치 크기 가중 평균으로 누적
            for k, v in logging_dict.items():
                v_scalar = v.item() if torch.is_tensor(v) else v
                component_meters[k].update(v_scalar, batch_size)

            result["logs"] = {f"{split}_loss": loss.detach().cpu()}
            if gather_predictions:
                # We need to gather the predictions if we use Multi-GPU eval
                predictions_dict = gather_predictions_dict(predictions_dict)
            # CPU로 이동하지 않으면 eval loop 내내 GPU에 예측값(및 튜플 안 텐서)이 쌓여
            # validation/test 배치 수가 많을 때 OOM으로 이어짐 (on_epoch_end에서 clear될 때까지 유지됨)
            predictions_dict = {k: to_cpu_recursive(v) for k, v in predictions_dict.items()}
            result.update(predictions_dict)
            if split == "test":
                self.test_step_outputs.append(result)
            else:
                self.validation_step_outputs.append(result)

        if not has_data:
            logger.warning(f"{split} evaluate: 처리된 배치가 없습니다. "
                            f"dataloader가 비어있거나 모든 배치가 corrupted이거나 limit_num_batches=0일 수 있습니다.")

        if get_is_master() and not self.args.main.disable_wandb:
            if has_data:
                wandb.log({f"{split}_loss": losses.avg}, step=self.global_step)
                if log_loss_components == True:
                    for k, meter in component_meters.items():
                        wandb.log({f"{split}_" + k: meter.avg}, step=self.global_step)
            else:
                logger.warning(f"{split} evaluate: 로깅할 데이터가 없어 wandb.log를 건너뜁니다.")

        # log epoch metrics
        epoch_metrics = self.on_epoch_end(split=split, device=device, epoch=epoch, ckpt_dir=ckpt_dir)
        return epoch_metrics

    def preprocess_batch(self, batch, device="cuda", train=True):
        # move all keys to cuda
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            else:
                batch[k] = v
        batch["x"] = batch["x"].to(dtype=torch.float32)

        if self.args.engine.kwargs.use_gpu_augs:
            if train:
                gpu_augs = get_augmentations(self.train_gpu_augs, self.args)
            else:
                gpu_augs = get_augmentations(self.test_gpu_augs, self.args)
            inputs = {"input": batch["x"],}
            if "mask" in batch:
                inputs["mask"] = batch["mask"]
            for transform in gpu_augs:
                inputs = transform(inputs)
            batch["x"] = inputs["input"]

        if self.args.dataset.shared_dataset_kwargs.windows:
            inputs = self.batch_windowing(batch["x"], modality=self.args.dataset.modality, windows_type=self.args.dataset.shared_dataset_kwargs.windows,)
            batch["x"] = inputs

        return batch

    def batch_windowing(self, batch: torch.Tensor, modality: str = "CT", windows_type="all") -> torch.Tensor:
        """Apply windowing and normalization transforms to CT volumes."""

        if len(batch.shape) == 4:
            B, D, H, W = batch.shape
            C = 1
        else:
            B, C, D, H, W = batch.shape

        device = batch.device

        # Vectorized windowing implementation
        if windows_type == "all":
            # Get all available windows
            windows = rve.get_available_windows(modality)
        elif isinstance(windows_type, str):
            windows = [windows_type]
        elif isinstance(windows_type, list):
            windows = windows_type
        else:
            raise ValueError(f"Invalid windows type: {windows_type}")

        if modality == "CT":
            # "minmax" window는 배치 전체가 아니라 샘플별로 min/max를 계산해야 함
            # (자세한 이유는 pillar/utils/windowing.py 참고) — 그 로직을 공용 유틸로 뺌.
            batch = batch_apply_ct_windowing(batch, windows_type)
        elif modality == "MR":
            # If modality is MR we need to apply each window to each channel of each item in the batch separately (due to percentile windowing)
            windowed = torch.zeros((B, len(windows) * C, D, H, W), device=device, dtype=batch.dtype)
            for n in range(B):
                for i, window in enumerate(windows):
                    for j in range(C):
                        windowed[n, i * C + j] = rve.apply_windowing(batch[n, j], window, modality)

            # This is hard coded for UCSF BMR
            batch = (windowed - 0.041) / 0.073
        else:
            raise ValueError(f"Invalid modality: {modality}")
        return batch

    # def _extract_required_data(self, batch, model_output, target_keys, pred_keys):
    #     """Extract required data from batch and model_output for metrics computation"""
    #     data = OrderedDict()
    #     data['accession'] = batch['accession']
    #     # Extract target data
    #     for key in sorted(target_keys):
    #         if key in batch:
    #             data[key] = batch[key]
    #         else:
    #             raise ValueError(f"Target key {key} not found in batch.")

    #     # Get prediction data from model_output
    #     for key in sorted(pred_keys):
    #         # Check if this is a nested key (contains a dot)
    #         if "." in key:
    #             parent_key, child_key = key.split(".", 1)
    #             if parent_key in model_output and child_key in model_output[parent_key]:
    #                 data[key] = model_output[parent_key][child_key]
    #             else:
    #                 raise ValueError(f"Nested prediction key {key} not found in model_output.")
    #         elif key in model_output:
    #             data[key] = model_output[key]
    #         else:
    #             raise ValueError(f"Prediction key {key} not found in model_output.")

    #     return data

    def _extract_required_data(self, batch, model_output, target_keys, pred_keys):
            """Extract required data from batch and model_output for metrics computation"""
            data = OrderedDict()
            data['accession'] = batch['accession']
    
            def _detach(v):
                if isinstance(v, torch.Tensor):
                    return v.detach()
                if isinstance(v, (list, tuple)):
                    return type(v)(_detach(vv) for vv in v)
                return v
    
            # Extract target data
            for key in sorted(target_keys):
                if key in batch:
                    data[key] = _detach(batch[key])
                else:
                    raise ValueError(f"Target key {key} not found in batch.")
    
            # Get prediction data from model_output
            for key in sorted(pred_keys):
                if "." in key:
                    parent_key, child_key = key.split(".", 1)
                    if parent_key in model_output and child_key in model_output[parent_key]:
                        data[key] = _detach(model_output[parent_key][child_key])
                    else:
                        raise ValueError(f"Nested prediction key {key} not found in model_output.")
                elif key in model_output:
                    data[key] = _detach(model_output[key])   # ← retrieval 키가 (sent_feats, topic_preds) 튜플이므로 재귀 detach 필요
                else:
                    raise ValueError(f"Prediction key {key} not found in model_output.")
    
            return data    

    def step(self, model, batch, batch_idx, epoch=None, split="train", device="cuda"):
        # Get required keys for current split
        target_keys, pred_keys = self.get_required_metric_keys(split)

        batch = self.preprocess_batch(batch, device=device, train=split == "train")

        with torch.amp.autocast("cuda", dtype=self.amp_precision, enabled=self.amp_precision is not None):
            model_output = model(batch["x"], batch=batch, split=split)

            # Extract required data
            predictions_dict = self._extract_required_data(batch, model_output, target_keys, pred_keys)
            # Also include accession identifiers for saving alongside predictions

            metric_input = {"batch": batch, "model_output": model_output,}

            loss, logging_dict = self.compute_step_metrics(loss_input=metric_input, metric_input=metric_input, train=(split == "train"))

        return loss, logging_dict, predictions_dict