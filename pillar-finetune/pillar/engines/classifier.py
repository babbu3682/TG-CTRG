import time
from collections import OrderedDict

import torch
import wandb
from tqdm import tqdm
import rve

from pillar.utils.logging import logger
from pillar.utils.engine import gather_predictions_dict, prefix_dict
from pillar.utils.misc import AverageMeter, Summary, ProgressMeter, get_is_master
from timm.data.mixup import Mixup

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


class Classifier(Engine):
    def __init__(
        self, *args, binary_pred=False, multi_head_pred=False, log_interval=None, log_loss_components=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.binary_pred = binary_pred
        self.multi_head_pred = multi_head_pred

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

        # # lr for the first param group
        # lr_value = [param_group["lr"] for param_group in optimizer.param_groups][0]

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")

        lr = AverageMeter("lr", ":.4e", summary_type=Summary.NONE)

        max_mem = AverageMeter("Max mem", ":.0f", summary_type=Summary.NONE)
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, lr, losses, max_mem],
            prefix="Epoch: [{}]".format(epoch),
        )
        epoch_metrics_configured = len(self.get_epoch_metrics(split="train")) > 0

        end = time.time()
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"Epoch {epoch} Training",
                disable=not get_is_master(),
            )
        ):
            data_time.update(time.time() - end)
            max_mem.update(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if batch_idx == self.limit_num_batches:
                break

            if batch is None:
                # Potentially corrupted data
                continue

            # From MAE
            # we use a per iteration (instead of per epoch) lr scheduler
            if (lr_scheduler is not None) and (batch_idx % self.accum_iter == 0):
                lr_scheduler.adjust_learning_rate(batch_idx / len(dataloader) + epoch)

            result = OrderedDict()

            with torch.amp.autocast("cuda", dtype=self.amp_precision, enabled=self.amp_precision is not None):
                loss, logging_dict, predictions_dict = self.step(
                    model, batch, batch_idx, epoch=epoch, split="train", device=device
                )

            loss /= self.accum_iter
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                clip_grad=clip_grad,
                create_graph=False,
                need_update=(batch_idx + 1) % self.accum_iter == 0,
            )

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        logger.warning(f"NaN gradients in {name}")
                    elif torch.isinf(param.grad).any():
                        logger.warning(f"Inf gradients in {name}")

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
        # tqdm progress bar
        desc = "Evaluation" if split == "val" else "Testing"
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"Epoch {epoch} {desc}" if epoch else desc,
                disable=not get_is_master(),
            )
        ):
            if batch_idx == self.limit_num_batches:
                break

            result = OrderedDict()

            # NOTE: 원래 이 자리(그리고 step() 직후)에 torch.cuda.empty_cache()가 배치마다
            # 두 번 호출되고 있었는데, 이건 캐싱 할당자가 들고 있던 블록을 통째로 반환시켜
            # 다음 배치마다 비싼 cudaMalloc을 다시 하게 만든다(동기화까지 동반). 실측:
            # 학습은 forward+backward를 다 하고도 1.855s/batch인데, forward만 하는 검증이
            # 3.94s/batch(205s / 52 batch)로 오히려 2배 느렸음. 제거해도 메모리는 안전하다 —
            # eval은 gradient를 안 들고 있어서 같은 batch_size의 학습보다 항상 덜 쓴다.
            #
            # 그리고 여기 있던 torch.cuda.amp.autocast(enabled=False)도 같이 제거함:
            # step()이 내부에서 self.amp_precision으로 autocast를 다시 열기 때문에(아래 step()
            # 참고) 바깥에서 enabled=False로 감싸는 건 실질적으로 무효였다. retrieval.py에도
            # 동일한 코드가 있었고 이미 같은 이유로 정리된 상태.
            with torch.no_grad():
                loss, logging_dict, predictions_dict = self.step(
                    model, batch, batch_idx, epoch=epoch, split=split, device=device
                )

            # log metrics
            result["logs"] = {f"{split}_loss": loss.detach().cpu()}
            if gather_predictions:
                # We need to gather the predictions if we use Multi-GPU eval
                predictions_dict = gather_predictions_dict(predictions_dict)
            result.update(predictions_dict)
            if split == "test":
                self.test_step_outputs.append(result)
            else:
                self.validation_step_outputs.append(result)

            # (배치 끝의 세 번째 torch.cuda.empty_cache()도 위와 같은 이유로 제거 —
            #  캐싱 할당자를 매 배치 비우면 다음 배치가 cudaMalloc부터 다시 해야 한다.)

        if get_is_master() and not self.args.main.disable_wandb:
            wandb.log({f"{split}_loss": loss.detach().cpu()}, step=self.global_step)
            if log_loss_components == True:
                for k, v in logging_dict.items():
                    wandb.log({k: v}, step=self.global_step)
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
            inputs = {
                "input": batch["x"],
            }
            if "mask" in batch:
                inputs["mask"] = batch["mask"]
            for transform in gpu_augs:
                inputs = transform(inputs)
            batch["x"] = inputs["input"]

        if self.args.dataset.shared_dataset_kwargs.windows:
            inputs = self.batch_windowing(
                batch["x"],
                modality=self.args.dataset.modality,
                windows_type=self.args.dataset.shared_dataset_kwargs.windows,
            )
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
            # If modality is CT we can apply each window to all items in a batch at once

            windowed = torch.zeros((B, len(windows), D, H, W), device=device, dtype=batch.dtype)
            for i, window in enumerate(windows):
                if C == 1:
                    windowed[:, i] = rve.apply_windowing(batch.squeeze(1), window, modality)
                else:
                    windowed[:, i] = rve.apply_windowing(batch[:, 0], window, modality)
            batch = windowed
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

    def _extract_required_data(self, batch, model_output, target_keys, pred_keys):
        """Extract required data from batch and model_output for metrics computation"""
        data = OrderedDict()
        data['accession'] = batch['accession']
        # Extract target data
        for key in sorted(target_keys):
            if key in batch:
                data[key] = batch[key]
            else:
                raise ValueError(f"Target key {key} not found in batch.")

        # Get prediction data from model_output
        for key in sorted(pred_keys):
            # Check if this is a nested key (contains a dot)
            if "." in key:
                parent_key, child_key = key.split(".", 1)
                if parent_key in model_output and child_key in model_output[parent_key]:
                    data[key] = model_output[parent_key][child_key]
                else:
                    raise ValueError(f"Nested prediction key {key} not found in model_output.")
            elif key in model_output:
                data[key] = model_output[key]
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
            if "exam" in batch:
                predictions_dict["exam"] = batch["exam"]

            metric_input = {
                "batch": batch,
                "model_output": model_output,
            }

            loss, logging_dict = self.compute_step_metrics(
                loss_input=metric_input, metric_input=metric_input, train=(split == "train")
            )
        return loss, logging_dict, predictions_dict
