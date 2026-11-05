import json
import logging
import math
import os
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import rve

# Suppress warnings about non-writable tensors
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")

try:
    import wandb
except ImportError:
    wandb = None


from miniclip import get_input_dtype

from .data import MultiDatasetDataloader
from .distributed import is_master

from .precision import get_autocast
from .utils import AverageMeter, compute_weight_norms, compute_gradient_norms


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2],
    }


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def save_step_checkpoint(args, model, optimizer, scaler, epoch, step):
    """Save a step-level checkpoint. This is purely a preemption/crash safety net -- resume only
    ever looks for step_latest.pt (see main.py), so we only write that single, overwritten-in-
    place file. A uniquely-named step_{step}.pt used to also be written on every trigger, which
    silently doubled the I/O cost of every save and accumulated an ever-growing pile of
    multi-GB checkpoints on disk that nothing ever read back."""
    save_frequency_step = getattr(args, "save_frequency_step", 0)
    if not args.save_logs or save_frequency_step <= 0 or not args.save_most_recent:
        return

    if (step + 1) % save_frequency_step != 0:
        return

    checkpoint_dict = {
        "epoch": epoch,
        "step": step,
        "name": args.name,
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()

    # Write to a tmp file and atomically rename, so a crash/preemption mid-write can't corrupt
    # the checkpoint that resume depends on (matches maybe_save_checkpoint's pattern).
    latest_path = os.path.join(args.checkpoint_path, "step_latest.pt")
    tmp_path = os.path.join(args.checkpoint_path, "step_latest.pt.tmp")
    torch.save(checkpoint_dict, tmp_path)
    os.replace(tmp_path, latest_path)

    logging.info(f"Saved step checkpoint (step_latest.pt) at step {step}")


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

# 핵심
def train_multimodal_one_epoch_lapa(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    args,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    if isinstance(data["train"], MultiDatasetDataloader):
        # If MultiDatasetDataloader, we need to set the epoch for each dataset
        for dl in data["train"].dataloaders.values():
            if hasattr(dl, "set_epoch"):
                dl.set_epoch(epoch)
        dataloader = data["train"]
        num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    else:
        # If not MultiDatasetDataloader, we can set the epoch directly
        if hasattr(data["train"], "set_epoch"):
            data["train"].set_epoch(epoch)
        dataloader = data["train"].dataloader
        num_batches_per_epoch = dataloader.num_batches // args.accum_freq

    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = {}
    modality_losses_m = defaultdict(lambda: AverageMeter())  # Track modality-specific losses
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # --- Accumulators using dictionaries to respect modality structure ---
    # For caching the original inputs needed for the gradient-enabled second pass
    accum_inputs = defaultdict(lambda: {"images": [], "texts": [], "querys": [], "counts": [], "labels": [], "volume_labels": [], "phrase_ids": []})
    # For caching the detached features from the first no_grad pass
    accum_features = defaultdict(lambda: {"local_image_features": [], "key_phrase_features": []})
    
    # Check if we're resuming from a step checkpoint
    resume_step = None
    step_in_epoch = 0  # Track position within current epoch
    if args.resume and os.path.exists(args.resume):
        try:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if "step" in checkpoint and checkpoint.get("epoch", -1) == epoch:
                resume_step = checkpoint["step"]  # This is the global step number
                
                # Calculate position within current epoch
                step_in_epoch = resume_step % num_batches_per_epoch
                
                logging.info(f"Resuming from global step {resume_step} (step {step_in_epoch} in epoch {epoch})")
                
                # Calculate remaining batches in this epoch
                if isinstance(dataloader, MultiDatasetDataloader):
                    # Each step processes args.accum_freq raw batches
                    completed_batches = step_in_epoch * args.accum_freq
                    remaining_batches = dataloader.num_batches - completed_batches
                    if remaining_batches > 0:
                        # Log modality status at resume point
                        modality_status = dataloader.get_modality_status_at_batch(completed_batches)
                        logging.info(f"Modality status at batch {completed_batches}:")
                        for modality, status in modality_status.items():
                            logging.info(f"  {modality}: {status}")
                        
                        dataloader.set_remaining_batches(remaining_batches)
                        logging.info(f"Set MultiDatasetDataloader to yield {remaining_batches} remaining batches")
                    else:
                        # Already completed this epoch
                        logging.info(f"Epoch {epoch} already completed (step_in_epoch={step_in_epoch}), skipping")
                        return {}  # Return empty metrics since epoch is complete
        except Exception:
            pass

    for i, multimodal_batch in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        end = time.time()

        query = None
        count = None
        label = None
        volume_label = None
        phrase_ids = None
        optimizer.zero_grad()

        # --- Pass 1: Cache inputs and features into modality-keyed dictionaries ---
        # This pass runs with no_grad for efficiency.
        with torch.no_grad():
            with autocast():
                for modality, batch in multimodal_batch.items():
                    if len(batch) == 7:
                        images, texts, query, count, label, volume_label, phrase_ids = batch # [B,C,H,W,D]
                    elif len(batch) == 6:
                        images, texts, query, count, label, volume_label = batch # [B,C,H,W,D]
                    elif len(batch) == 5:
                        images, texts, query, count, label = batch # [B,C,H,W,D]
                    elif len(batch) == 2:
                        images, texts = batch
                    else:
                        images, texts = batch["images"], batch["texts"]
                    
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)

                    # Apply GPU-based 3D rotation augmentation if enabled
                    if getattr(args, 'enable_gpu_rotation', False) and model.training:
                        from trainer.gpu_augment import apply_3d_rotation_gpu
                        images = apply_3d_rotation_gpu(images.to(torch.float32), degrees=getattr(args, 'gpu_rotation_degrees', 10.0), p=getattr(args, 'gpu_rotation_p', 0.5), training=model.training)
                    
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    # Apply GPU transforms for Merlin CT data if configured
                    if modality in ['merlin_abd_ct', 'chest_ct', 'brain_ct', 'abdomen_ct'] and hasattr(args, 'window_type') and args.window_type is not None:
                        images = rve.batch_apply_windowing_vectorized(images, windows=args.window_type, modality='CT')
                    elif modality in ['breast_mr', 'prostate_mr']:
                        if modality == 'prostate_mr':
                            window_type = "znorm"
                        else:
                            window_type = "high_contrast"                        
                        from trainer.merlin_abd_ct import apply_merlin_gpu_transforms
                        images = apply_merlin_gpu_transforms(images, window_type=window_type, modality='MR', normalize_mean=getattr(args, 'image_mean', 0.289), normalize_std=getattr(args, 'image_std', 0.198), normalize=False)

                    images_dict = {modality: images}
                    if isinstance(texts, dict):
                        # For CustomTextCLIP
                        texts = {k: v.to(device=device, non_blocking=True) for k, v in texts.items()}
                    else:
                        texts = texts.to(device=device, non_blocking=True)

                    # logging.info("moving time: " + str(time.time() - end))   
                    # end = time.time()

                    # --- Cache inputs for the second pass ---
                    accum_inputs[modality]["images"].append(images_dict)
                    accum_inputs[modality]["texts"].append(texts)
                    if query is not None:
                        accum_inputs[modality]["querys"].append(query)
                    if count is not None:
                        accum_inputs[modality]["counts"].append(count)
                    if label is not None:
                        accum_inputs[modality]["labels"].append(label)
                    if volume_label is not None:
                        accum_inputs[modality]["volume_labels"].append(volume_label)
                    if phrase_ids is not None:
                        accum_inputs[modality]["phrase_ids"].append(phrase_ids)

                    # --- Compute and cache features ---
                    # model forward (핵심)
                    model_out = model(images_dict, texts, query=query, count=count)
                    # logging.info("model time: " + str(time.time() - end))
                    # end = time.time()

                    # Store features as normal tensors (not inference tensors)
                    # to allow concatenation with grad-enabled tensors in pass 2
                    # local-only variant: no global image feature and no report-level text
                    # feature exist here (see CustomTextCLIP_PhraseAlign / PhraseAlignLoss).
                    local_image_feats      = model_out["local_image_features"]
                    key_phrase_feats       = model_out["key_phrase_features"]

                    if isinstance(local_image_feats, dict):
                        local_image_feats = {k: v.detach().clone() for k, v in local_image_feats.items()}
                    else:
                        local_image_feats = local_image_feats.detach().clone()

                    if isinstance(key_phrase_feats, dict):
                        key_phrase_feats = {k: v.detach().clone() for k, v in key_phrase_feats.items()}
                    else:
                        key_phrase_feats = key_phrase_feats.detach().clone()

                    accum_features[modality]["local_image_features"].append(local_image_feats)
                    accum_features[modality]["key_phrase_features"].append(key_phrase_feats)

        # --- Accumulation Check ---
        # Continue accumulating batches until we reach the desired frequency.
        if ((i + 1) % args.accum_freq) != 0:
            continue

        # --- Pass 2: Re-compute all with gradients for a single, global loss ---
        # This block executes only once every `args.accum_freq` steps.
        i_accum = i // args.accum_freq
        
        # Calculate global step, accounting for resume
        if resume_step is not None:
            # When resuming, continue from the global step number
            # i_accum=0 corresponds to the next step after resume_step
            step = resume_step + 1 + i_accum
        else:
            step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        optimizer.zero_grad()

        # Re-run the forward pass for ALL cached inputs to build the computation graph
        with autocast():
            batch_sizes = {}
            for modality in accum_inputs:
                # Get the batch size for this modality
                batch_sizes[modality] = len(accum_inputs[modality]['images'][0][modality])

            # Accumulates a (weighted_sum, weight) pair per loss key across every
            # (modality, micro-batch) pass below, so the value logged to wandb for this
            # optimizer step is a true weighted average -- not just whichever (modality, j)
            # happened to run last.
            step_losses_accum = defaultdict(lambda: [0.0, 0])

            # Total number of (modality, micro-batch) passes in this optimizer step, so we know
            # which one is last (see no_sync note below).
            total_jobs = sum(len(d["images"]) for d in accum_inputs.values())
            job_idx = 0

            # Iterate through each modality we've accumulated
            for modality, inputs_dict in accum_inputs.items():
                # Iterate through each micro-batch within that modality
                num_micro_batches = len(inputs_dict["images"])
                for j in range(num_micro_batches):
                    images = inputs_dict["images"][j]
                    texts  = inputs_dict["texts"][j]
                    query  = inputs_dict["querys"][j] if inputs_dict["querys"] else None
                    count  = inputs_dict["counts"][j] if inputs_dict.get("counts") else None

                    # DDP all-reduces gradients on every backward() by default, which would mean
                    # accum_freq (20x here) all-reduces per optimizer step instead of one. Skip
                    # the all-reduce (no_sync) on every pass except the last.
                    job_idx += 1
                    is_last_job = job_idx == total_jobs
                    sync_context = model.no_sync() if (hasattr(model, "no_sync") and not is_last_job) else nullcontext()

                    with sync_context:
                        # model forward (핵심)
                        model_out = model(images, texts, query=query, count=count)

                        inputs_no_accum = {}
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                        # lapa: the VL-CABS attention temperature is an nn.Parameter and
                        # is returned by forward() precisely so DDP sees it used there. Reading
                        # unwrap_model(model).temperature instead would take it OUTSIDE the
                        # autograd graph DDP tracks and trip "Expected to mark a variable ready
                        # only once" -- the reason it used to be a frozen buffer.
                        inputs_no_accum["temperature"] = model_out.pop("temperature")
                        if "logit_bias" in model_out:
                            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                        inputs = {}
                        for key, val in accum_features[modality].items():
                            accumulated = accum_features[modality][key]
                            if type(accumulated[0]) is dict:
                                # If the feature is a dict (e.g., for query), concatenate appropriately
                                captions_dict = {}
                                keylist = list(accumulated[0].keys())
                                for k in keylist:
                                    accs = [acc[k] for acc in accumulated]
                                    captions_dict[k] = torch.cat(accs[:j] + [model_out[key][k]] + accs[j + 1 :])
                                inputs[key] = captions_dict

                            else:
                                # Otherwise, concatenate the tensors directly
                                inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1 :])

                        # NOTE: was hardcoded to accum_inputs['chest_ct'] regardless of which
                        # modality this pass is for -- harmless while chest_ct is the only
                        # modality in use, but would silently apply the wrong modality's
                        # counts/labels once a second modality is added. Use `modality` instead.
                        # temperature: VL-CABS now runs inside the loss, so the model's
                        # attention temperature has to be handed over explicitly (it is a
                        # buffer, not a Parameter -- see CustomTextCLIP_PhraseAlign.__init__).
                        # volume_label is intentionally NOT passed: it only ever protected
                        # clip_loss's negatives, and this variant has no clip_loss.
                        losses = loss(**inputs, **inputs_no_accum, output_dict=True, counts=accum_inputs[modality]['counts'], label_map=accum_inputs[modality]['labels'], phrase_ids=accum_inputs[modality]['phrase_ids'] or None)
                        del inputs
                        del inputs_no_accum
                        # PhraseAlignLoss's output_dict mixes the real (already correctly-weighted)
                        # "total_loss" together with detached, log-only diagnostic scalars
                        # (num_label_pos, known_phrase_ratio, ...) that can be arbitrarily large
                        # counts. sum(losses.values()) was adding those into the training loss
                        # value -- it didn't corrupt gradients (the diagnostics are detached) but
                        # it corrupted every logged/backward()'d loss magnitude. Use the already-
                        # computed total_loss directly instead of re-summing the whole dict.
                        total_loss = losses.pop("total_loss")
                        losses["loss"] = total_loss

                        # Track modality-specific loss
                        modality_losses_m[modality].update(total_loss.item(), batch_sizes[modality])

                        for key, val in losses.items():
                            step_losses_accum[key][0] += val.item() * batch_sizes[modality]
                            step_losses_accum[key][1] += batch_sizes[modality]

                        backward(total_loss, scaler)

        # logging.info("loss time: " + str(time.time() - end))   
        # end = time.time()

        # Track gradient norms BEFORE optimizer.zero_grad() or scaler operations
        grad_norm_info = None
        if args.track_gradient_norms:
            grad_norm_info = compute_gradient_norms(unwrap_model(model), args.norm_type)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # logging.info("optimizer time: " + str(time.time() - end))   
        # end = time.time()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_inputs = defaultdict(lambda: {"images": [], "texts": [], "querys": [], "counts": [], "labels": [], "volume_labels": [], "phrase_ids": []})
            # For caching the detached features from the first no_grad pass
            accum_features = defaultdict(lambda: {"local_image_features": [], "key_phrase_features": []})

        # Track weight norms after optimizer.step()
        weight_norm_info = None
        if args.track_weight_norms:
            with torch.no_grad():
                weight_norm_info = compute_weight_norms(unwrap_model(model), args.norm_type)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
            # self.temperature is the VL-CABS attention divisor (attn / temperature.exp()).
            # In this variant it IS a trainable nn.Parameter (unlike phrase_align, where it is a
            # frozen buffer), so this clamp is doing real work: gradient descent can push it
            # toward -inf and blow the attention scores up into inf/NaN.
            if hasattr(unwrap_model(model), "temperature"):
                unwrap_model(model).temperature.clamp_(math.log(0.01), math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        
        # Calculate batch count within current epoch
        if resume_step is not None:
            # When resuming, add the steps already completed in this epoch
            batch_count = step_in_epoch + 1 + i_accum
        else:
            batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            # batch_size = batch_sizes[dataloader.shortest_modal] if hasattr(dataloader, 'shortest_modal') else len(next(iter(images.values())))
            batch_size = batch_sizes[dataloader.longest_modal] if hasattr(dataloader, 'longest_modal') else len(next(iter(images.values())))
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            
            # NOTE loss is coarsely sampled, just master node and per log update.
            # Log the weighted average across every (modality, micro-batch) pass of this
            # optimizer step (see step_losses_accum above), not just the last one processed.
            for key, (weighted_sum, weight) in step_losses_accum.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(weighted_sum / weight, weight)

            logit_scale_scalar = logit_scale.item()

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            
            # Add modality-specific losses to log
            if modality_losses_m:
                modality_log = " ".join(
                    [
                        f"{modality}: {loss_m.avg:#.5g}"
                        for modality, loss_m in modality_losses_m.items()
                    ]
                )
                loss_log += f" | Modality Losses: {modality_log}"
            # args.batch_size 는 CLI 기본값(64)이고 실제 배치는 data config 의 yaml 값이라,
            # 그걸 쓰면 처리량이 16배로 부풀려 찍힌다. 이번 step 이 실제로 본 배치를 쓴다.
            _bs = sum(batch_sizes.values()) if batch_sizes else args.batch_size
            samples_per_second = (args.accum_freq * _bs * args.world_size / batch_time_m.val)
            samples_per_second_per_gpu = (args.accum_freq * _bs / batch_time_m.val)
            # Peak allocated vs card capacity. Without this there is no way to tell from the log
            # whether batch_size is leaving the GPU half empty -- and per-step memory here is
            # deterministic (volumes are a fixed 11 x 256^3, the loss window is a fixed
            # world_size x batch_size x accum_freq), so the first few steps already tell you the
            # value for the whole run.
            mem_log = ""
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 2**30
                total = torch.cuda.get_device_properties(device).total_memory / 2**30
                mem_log = f"Mem: {peak:.1f}/{total:.0f}GiB ({100 * peak / total:.0f}%) "
            # VL-CABS attention temperature -- trainable in this variant, so it should MOVE.
            # If tau stays pinned at 0.0700 the parameter is not receiving gradient.
            tau_log = ""
            if hasattr(unwrap_model(model), "temperature"):
                tau_log = f"Tau: {unwrap_model(model).temperature.exp().item():.4f} "
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"{mem_log}"
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + tau_log + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "gpu_mem_peak_gib": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
                "attn_tau": unwrap_model(model).temperature.exp().item() if hasattr(unwrap_model(model), "temperature") else 0.0,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})
            
            # Add modality-specific losses
            for modality, loss_m in modality_losses_m.items():
                log_data[f"{modality}_loss"] = loss_m.avg

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                # Unified x-axis across all 4 ablation variants (see wandb.define_metric in
                # maybe_init_wandb): fractional dataset passes, independent of accum_freq/batch_size.
                log_data["epoch"] = epoch + batch_count / num_batches_per_epoch

                # Add norm data to wandb logging
                if args.track_weight_norms and weight_norm_info:
                    log_data["train/weight_norm"] = weight_norm_info.get('total_weight_norm', weight_norm_info.get('total_norm', 0))
                    log_data["train/weight_norm_param_count"] = weight_norm_info.get('param_count', 0)
                    # Log fine-grained weight norms
                    for key, value in weight_norm_info.items():
                        if key.startswith('weight_norm/'):
                            log_data[f"train/{key}"] = value
                
                if args.track_gradient_norms and grad_norm_info:
                    log_data["train/grad_norm"] = grad_norm_info.get('total_grad_norm', grad_norm_info.get('total_norm', 0))
                    log_data["train/grad_norm_param_count"] = grad_norm_info.get('param_count', 0)
                    # Log fine-grained gradient norms
                    for key, value in grad_norm_info.items():
                        if key.startswith('grad_norm/'):
                            log_data[f"train/{key}"] = value
                
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
        
        # Save step checkpoint
        save_step_checkpoint(args, model, optimizer, scaler, epoch, step)
    # end for


def _consolidate_features(feature_list):
    """Consolidate a list of features, handling both tensors and dicts of tensors."""
    if isinstance(feature_list[0], dict):
        # Handle a list of dictionaries of features
        text_features_dict = defaultdict(list)
        for features in feature_list:
            for key, val in features.items():
                text_features_dict[key].append(val)
        return {k: torch.cat(v, dim=0) for k, v in text_features_dict.items()}
    else:
        # Handle a list of tensors
        return torch.cat(feature_list, dim=0)


def evaluate_multimodal_lapa(model, data, loss, epoch, args, tb_writer=None, tokenizer=None, label=None):
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if "val" not in data or not (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        return metrics
    
    # Dataloader setup
    wandb_prefix = ""
    logging_prefix = ""
    if label is not None:
        logging_prefix = label.upper() + " "
        wandb_prefix = label + "_"
        val_source = data["val_labels"][label]
    else:
        val_source = data["val"]
    dataloader = val_source.dataloader if hasattr(val_source, 'dataloader') else val_source

    all_modalities_in_run = set()

    # Val loss is computed in windows the same size as one train optimizer step (accum_freq
    # micro-batches) instead of accumulating the whole (or a capped) val set into one giant
    # pool. This matches how the upstream pillar-pretrain reference evaluates -- per-window
    # loss, running sample-weighted average, no retrieval metrics -- and puts val loss on the
    # SAME scale as train loss so the two are directly comparable for spotting overfitting.
    # Retrieval R@k is intentionally not computed: this early in pretraining it sits at chance
    # level and adds cost without adding signal -- loss is the more sensitive check. label_map/
    # counts are lists (gather_features_vlcabs concats/flattens them internally, same pattern
    # as train loop's accum_inputs[modality]['labels']/['counts']).
    #
    # The window flush is also what keeps memory bounded: local_image_features carries a huge
    # per-image local-token dimension (multiscale patch tokens), so concatenating +
    # all_gathering the WHOLE val set at once would OOM even on an 80GB GPU.
    window_size = max(1, getattr(args, "accum_freq", 1) or 1)
    window_buffer = defaultdict(lambda: {
        "local_image_features": [], "key_phrase_features": [], "label_map": [], "counts": [],
        "phrase_ids": [],
    })
    logit_scale_buffer = {}
    loss_sum = defaultdict(float)
    loss_weight = defaultdict(float)

    # Configure loss for distributed evaluation -- gather_features() will all_gather across ranks
    # inside each loss() call below.
    loss.world_size = args.world_size
    loss.rank = args.rank

    def _flush_window(modality):
        buf = window_buffer[modality]
        if not buf["local_image_features"]:
            return
        local_image_features = torch.cat(buf["local_image_features"], dim=0)
        key_phrase_features = torch.cat(buf["key_phrase_features"], dim=0)

        losses = loss(
            local_image_features=local_image_features,
            key_phrase_features=key_phrase_features,
            logit_scale=logit_scale_buffer[modality],
            output_dict=True,
            label_map=buf["label_map"],
            counts=buf["counts"],
            temperature=temperature,
            phrase_ids=buf["phrase_ids"] or None,
        )
        total_loss = losses["total_loss"]
        n = local_image_features.shape[0]
        loss_sum[modality] += float(total_loss.item()) * n
        loss_weight[modality] += n

        window_buffer[modality] = {
            "local_image_features": [], "key_phrase_features": [], "label_map": [], "counts": [],
            "phrase_ids": [],
        }

    with torch.no_grad():
        # Compute display batches considering optional cap
        total_batches = getattr(dataloader, 'num_batches', None)
        val_batches = total_batches
        if getattr(args, "max_val_batches", None) is not None and total_batches is not None:
            val_batches = min(total_batches, args.max_val_batches)
        if is_master(args):
            if val_batches is not None:
                logging.info(f"Eval Epoch: {epoch}, num batches: {val_batches}")
            else:
                logging.info(f"Eval Epoch: {epoch}")

        # VL-CABS runs inside the loss, so it needs the model's attention temperature. In
        # lapa it is a trainable nn.Parameter; under no_grad here that distinction
        # does not matter, and reading it off the module is safe outside the DDP graph.
        temperature = unwrap_model(model).temperature

        # ── 1단계: 전체 val 데이터 feature/label/count 추출 ──────────────────
        for i, multimodal_batch in enumerate(dataloader):
            with autocast():
                for modality, batch in multimodal_batch.items():
                    all_modalities_in_run.add(modality)
                    query = None
                    count = None
                    volume_label = None
                    phrase_ids = None
                    if len(batch) == 7:
                        images, texts, query, count, label, volume_label, phrase_ids = batch
                    elif len(batch) == 6:
                        images, texts, query, count, label, volume_label = batch
                    elif len(batch) == 5:
                        images, texts, query, count, label = batch
                    else:
                        images, texts = batch

                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    if modality in ["merlin_abd_ct", "chest_ct", "brain_ct", "abdomen_ct"] and hasattr(args, "window_type") and args.window_type:
                        images = rve.batch_apply_windowing_vectorized(images.to(torch.float32), windows=args.window_type, modality='CT')
                        # from trainer.merlin_abd_ct import apply_merlin_gpu_transforms
                        # images = apply_merlin_gpu_transforms(images, window_type=args.window_type, modality='CT', normalize_mean=getattr(args, 'image_mean', 0.289), normalize_std=getattr(args, 'image_std', 0.198))
                    elif modality in ['breast_mr', 'prostate_mr']:
                        from trainer.merlin_abd_ct import apply_merlin_gpu_transforms
                        if modality == 'prostate_mr':
                            window_type = "znorm"
                        else:
                            window_type = "high_contrast"
                        images = apply_merlin_gpu_transforms(images, window_type=window_type, modality='MR', normalize_mean=getattr(args, 'image_mean', 0.289), normalize_std=getattr(args, 'image_std', 0.198), normalize=False)

                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)

                    if isinstance(texts, dict):
                        texts = {k: v.to(device=device, non_blocking=True) for k, v in texts.items()}
                    else:
                        texts = texts.to(device=device, non_blocking=True)

                    # model forward (핵심)
                    model_out = model({modality: images}, texts, query=query, count=count)

                    wbuf = window_buffer[modality]
                    wbuf["local_image_features"].append(model_out["local_image_features"])
                    wbuf["key_phrase_features"].append(model_out["key_phrase_features"])
                    wbuf["label_map"].append(label)
                    wbuf["counts"].append(count)
                    if phrase_ids is not None:
                        wbuf["phrase_ids"].append(phrase_ids)
                    logit_scale_buffer[modality] = model_out["logit_scale"]  # 마지막 값 사용

            # Flush every window_size batches so the loss pool stays the same size as one train
            # optimizer step.
            if (i + 1) % window_size == 0:
                for modality in list(window_buffer.keys()):
                    _flush_window(modality)

            # Early stop evaluation if max-val-batches is set
            if getattr(args, "max_val_batches", None) is not None and (i + 1) >= args.max_val_batches:
                if is_master(args):
                    logging.info(f"Stopping eval early at {i + 1} batches due to --max-val-batches={args.max_val_batches}")
                break

        # Flush any leftover partial window (same on every rank, since the loop above iterates
        # the same number of times on every rank).
        for modality in list(window_buffer.keys()):
            _flush_window(modality)

    # loss_sum/loss_weight are local-rank running totals, but each window's total_loss already
    # reflects the full cross-rank gathered pool (loss()'s own gather_features all_gathers inside
    # every call), so only master needs to log the result.
    if is_master(args):
        for modality in sorted(all_modalities_in_run):
            n = loss_weight.get(modality, 0)
            if n > 0:
                metrics[f"{wandb_prefix}{modality}_clip_val_loss"] = loss_sum[modality] / n
                metrics[f"{wandb_prefix}{modality}_num_samples"] = n

        if not metrics:
            return metrics

        logging.info(f"Eval Epoch: {epoch} " + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

        log_data = {"val/" + name: val for name, val in metrics.items()}
        if args.save_logs:
            if tb_writer:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, epoch)
            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        if args.wandb:
            assert wandb is not None, "Please install wandb."
            step = None
            if "train" in data:
                step = (data["train"].num_batches // args.accum_freq) * epoch
            log_data["epoch"] = float(epoch)
            wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    if isinstance(text_features, dict):
        metrics = {}
        for key, text_feature_tensor in text_features.items():
            # Recursively call with each text feature tensor
            sub_metrics = get_clip_metrics(image_features, text_feature_tensor, logit_scale)
            for metric_name, val in sub_metrics.items():
                metrics[f"{key}_{metric_name}"] = val
        return metrics

    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)
 
    for name, logit in logits.items():
         ranking = torch.argsort(logit, descending=True)
         preds = torch.where(ranking == ground_truth)[1]
         preds = preds.detach().cpu().numpy()
         metrics[f"{name}_mean_rank"] = preds.mean() + 1
         metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
         for k in [1, 5, 10]:
             metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
