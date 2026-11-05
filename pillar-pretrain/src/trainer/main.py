import copy
import glob
import logging
import os
import random
import re
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from miniclip import (
    create_loss,
    create_model_and_transforms,
    get_tokenizer,
    trace_model,
)
from torch import optim
from transformers import AutoModel

from .data import get_data, MultiDatasetDataloader
from .distributed import (
    broadcast_object,
    init_distributed_device,
    is_master,
)
from .file_utils import (
    pt_load,
    remote_sync,
    start_sync_process,
)
from .logger import setup_logging
from .params import parse_args
from .scheduler import const_lr, const_lr_cooldown, cosine_lr
from .train import evaluate, train_one_epoch
from .train_multimodal import evaluate_multimodal, train_multimodal_one_epoch
from .train_multimodal_siglip import evaluate_multimodal_siglip, train_multimodal_siglip_one_epoch
from .train_textdino import train_textdino_one_epoch, evaluate_textdino

from .train_multimodal_phrase_align import evaluate_multimodal_phrase_align, train_multimodal_one_epoch_phrase_align
from .train_multimodal_lapa import evaluate_multimodal_lapa, train_multimodal_one_epoch_lapa
from .train_multimodal_phrase_align_split import evaluate_multimodal_phrase_align_split, train_multimodal_one_epoch_phrase_align_split

from .utils import maybe_copy_codebase

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [
            os.path.join(path, x.split(" ")[-1])
            for x in result.stdout.decode().split("\n")[:-1]
        ]
    else:
        checkpoints = glob.glob(path + "**/*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def maybe_run_remote_sync_final(args, remote_sync_process):
    if remote_sync_process is None:
        return

    # run a final sync
    logging.info("Final remote sync.")
    remote_sync_process.terminate()
    result = remote_sync(
        os.path.join(args.logs, args.name),
        os.path.join(args.remote_sync, args.name),
        args.remote_sync_protocol,
    )
    if result:
        logging.info("Final remote sync successful.")
    else:
        logging.info("Final remote sync failed.")


def maybe_save_checkpoint(args, original_model, optimizer, scaler, completed_epoch):
    if args.save_logs:
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": original_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
        if args.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(
                args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt"
            )
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

        if args.save_most_recent:
            # try not to corrupt the latest checkpoint if save fails
            tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
            latest_save_path = os.path.join(
                args.checkpoint_path, LATEST_CHECKPOINT_NAME
            )
            torch.save(checkpoint_dict, tmp_save_path)
            os.replace(tmp_save_path, latest_save_path)


def handle_flash_attn(args):
    sm = torch.cuda.get_device_capability(0)
    # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    enable_flashattn = sm[0] >= 8 or (sm[0] == 7 and sm[1] >= 5)

    print(f"enable_flashattn: {enable_flashattn}")

    if args.flash_attn == "fa3":
        print("Flash attention 3 library enabled")

        # This requies installing the hopper directory in https://github.com/Dao-AILab/flash-attention/tree/v2.7.2.post1

        assert sm[0] >= 9, "Flash attn requires compute capabilities 9.0"

        import flash_attn_interface

        torch_scaled_dot_product_attention = F.scaled_dot_product_attention

        def scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        ):
            # torch convention: B, num heads, seq len, C
            # print(f"Using flash attention, query: {query.shape}, key: {key.shape}, value: {value.shape}")
            assert attn_mask is None, attn_mask
            head_dim = query.shape[-1]
            if head_dim > 256:
                return torch_scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
            assert dropout_p == 0, "Dropout not supported in flash attn 3"

            # `flash_attn_interface.flash_attn_func`'s return value is out, lse (logsumexp). We only take `out`.
            out = torch.permute(
                flash_attn_interface.flash_attn_func(
                    torch.permute(query, [0, 2, 1, 3]),
                    torch.permute(key, [0, 2, 1, 3]),
                    torch.permute(value, [0, 2, 1, 3]),
                    causal=is_causal,
                )[0],
                [0, 2, 1, 3],
            )
            return out

        F.scaled_dot_product_attention = scaled_dot_product_attention

        # Use memory efficient attention as a fallback
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    elif args.flash_attn == "fa2":
        print("Flash attention 2 library enabled")

        # This requies installing https://github.com/Dao-AILab/flash-attention/tree/v2.2.3

        assert enable_flashattn, "Flash attn requires compute capabilities >= 7.5"

        import flash_attn

        torch_scaled_dot_product_attention = F.scaled_dot_product_attention

        def scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs
        ):
            # conver to bfloat16 if needed
            if query.dtype == torch.float32 or key.dtype == torch.float32 or value.dtype == torch.float32:
                query = query.to(torch.bfloat16)
                key = key.to(torch.bfloat16)
                value = value.to(torch.bfloat16)

            # torch convention: B, num heads, seq len, C
            # print(f"Using flash attention, query: {query.shape}, key: {key.shape}, value: {value.shape}")
            # assert attn_mask is None, attn_mask
            if query.shape[-1] > 256:
                return torch_scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
            return torch.permute(
                flash_attn.flash_attn_func(
                    torch.permute(query, [0, 2, 1, 3]),
                    torch.permute(key, [0, 2, 1, 3]),
                    torch.permute(value, [0, 2, 1, 3]),
                    dropout_p=dropout_p,
                    causal=is_causal,
                ),
                [0, 2, 1, 3],
            )

        F.scaled_dot_product_attention = scaled_dot_product_attention

        # Use memory efficient attention as a fallback
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        # If compute capabilities >= 7.5, flash sdp will be enabled. On torch >= 2.2, it's still fa2. Otherwise it's fa.
        print(
            "Flash attention 2 library is not enabled. Using built-in attention implementation (F.scaled_dot_product_attention)."
        )
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)


def maybe_sanitize_model_name(args):
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )
        return name
    return args.name


def maybe_init_wandb(args, data, model, params_file):
    # initialize wandb
    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting wandb.")
        # train_sz / val_sz accounting may be reintroduced later if required for analytics.

        # you will have to configure this for your project!
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume="auto" if getattr(args, "resume_latest", False) else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log="all")
        wandb.save(params_file)

        # Unify the x-axis across every chart to fractional epoch (dataset passes), instead of
        # raw optimizer step. accum_freq/batch_size differ across the 4 ablation variants, so raw
        # step count means a different amount of data seen per step in each -- "epoch" is the one
        # unit that means the same thing (one full pass over the same training set) in all of
        # them, and lets train/eval curves line up directly. Every wandb.log() call still passes
        # an integer step= (required for wandb's own ordering/resume bookkeeping) and additionally
        # logs an "epoch" field; this makes every chart default to that field for its x-axis.
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        logging.debug("Finished loading wandb.")


def _vision_layer_buckets(named_parameters, num_buckets_hint=None):
    """Map every trainable parameter name to a depth bucket, shallow -> deep.

    Bucket 0 is the patch embedding; bucket i+1 is atlas stage i. Anything outside the staged
    hierarchy (projectors, logit_scale, temperature, text tower) returns None and is treated
    as a head by the caller -- never silently filed into bucket 0, which would give the most
    general, most expensive-to-relearn weights the head learning rate.
    """
    def stage_of(name):
        if "atlas_models." in name:
            try:
                return int(name.split("atlas_models.")[1].split(".")[0])
            except (IndexError, ValueError):
                return None
        return None

    def bucket_of(name):
        n = name.replace("module.", "", 1)
        if not n.startswith("visual."):
            return None
        if "patch_embeds" in n:
            return 0
        s = stage_of(n)
        return None if s is None else s + 1

    buckets = {n: bucket_of(n) for n, _ in named_parameters}
    stages = [b for b in buckets.values() if b is not None]
    n_buckets = (max(stages) + 1) if stages else 0
    if num_buckets_hint is not None:
        n_buckets = max(n_buckets, num_buckets_hint)
    return buckets, n_buckets


def _build_layer_decay_param_groups(args, model):
    """BEiT/MAE-style layer-wise lr decay for a pretrained vision backbone.

    Why: initializing from Pillar0-ChestCT and then training every parameter at the flat
    pretraining lr (2.5e-4) lets the patch embedding move as fast as the deepest attention
    stage, on 23,850 volumes rather than the corpus Pillar0 was pretrained on. The reference
    stage-1 runs used deepest-stage 3e-5 with decay 0.8 (patch embed 1.54e-5) and head 1e-4.

    Implemented through the existing "lr_scale" convention: scheduler.assign_learning_rate()
    multiplies each group's base lr by its lr_scale on every step, so warmup/cosine survive.
    Weight decay follows the same rule as the flat path (none for 1-D params).
    """
    named_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    buckets, n_buckets = _vision_layer_buckets(named_parameters)

    head_lr = args.head_lr if args.head_lr is not None else args.lr
    is_text_tower = lambda n: n.replace("module.", "", 1).startswith("text.")
    text_lr = args.lr * args.text_lr_multiplier

    groups = {}          # (lr, weight_decay, lr_scale) -> [params]
    bucket_report = {}
    for name, param in named_parameters:
        b = buckets[name]
        if b is not None:
            # deepest bucket keeps args.lr; each step shallower multiplies by layer_decay
            scale = args.layer_decay ** (n_buckets - 1 - b)
            lr = args.lr * scale
            bucket_report.setdefault(b, [lr, 0])
            bucket_report[b][1] += 1
        elif is_text_tower(name):
            lr, scale = text_lr, args.text_lr_multiplier
        else:
            lr, scale = head_lr, head_lr / args.lr if args.lr else 1.0
        wd = 0.0 if param.ndim < 2 else args.wd
        groups.setdefault((lr, wd, scale), []).append(param)

    param_groups = [
        {"params": ps, "lr": lr, "weight_decay": wd, "lr_scale": scale}
        for (lr, wd, scale), ps in sorted(groups.items(), key=lambda kv: -kv[0][0])
    ]

    if is_master(args):
        logging.info(
            f"Layer-wise lr decay {args.layer_decay} over {n_buckets} vision buckets "
            f"({len(param_groups)} param groups; head lr {head_lr:.3e})"
        )
        for b in sorted(bucket_report):
            lr, n = bucket_report[b]
            tag = "patch_embeds" if b == 0 else f"atlas stage {b - 1}"
            logging.info(f"  vision bucket {b} ({tag}): lr={lr:.3e}, {n} tensors")

    return param_groups


def build_optimizer(args, model, device):
    optimizer, scaler = None, None
    assert not args.trace, "Cannot train with traced model"

    opt = getattr(args, "opt", "adamw").lower()
    if opt.startswith("timm/"):
        from timm.optim import create_optimizer_v2

        timm_opt = opt.split("timm/")[-1]
        opt_kwargs = {}
        assert (args.beta1 is None) == (args.beta2 is None), (
            "When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified)."
        )
        if args.beta1 is not None:
            opt_kwargs["betas"] = (args.beta1, args.beta2)
        if args.momentum is not None:
            opt_kwargs["momentum"] = args.momentum
        optimizer = create_optimizer_v2(
            model,
            timm_opt,
            lr=args.lr,
            weight_decay=args.wd,
            eps=args.eps,
            **opt_kwargs,
        )
    else:
        # If some params are not passed, we use the default values based on model name.
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)
        # Text tower (Y-Encoder) params get their own LR via --text-lr-multiplier (default 1.0,
        # i.e. no-op). The VL-JEPA paper (Table 7b) found a 0.05-0.10 multiplier optimal for a
        # jointly-trained text encoder -- training it at the full predictor LR measurably hurt
        # performance. --lock-text can't substitute for this: it's a no-op for CacheTextEncoder.
        is_text_tower = lambda n: n.replace("module.", "", 1).startswith("text.")

        named_parameters = list(model.named_parameters())

        def _select(match_fn, text_tower: bool):
            return [
                p for n, p in named_parameters
                if match_fn(n, p) and is_text_tower(n) == text_tower and p.requires_grad
            ]

        gain_or_bias_params = _select(exclude, False)
        rest_params = _select(include, False)
        text_gain_or_bias_params = _select(exclude, True)
        text_rest_params = _select(include, True)
        text_lr = args.lr * args.text_lr_multiplier

        # "lr_scale" is read by scheduler.assign_learning_rate() on every scheduler step, so the
        # multiplier survives warmup/cosine/etc. instead of being overwritten on the first step.
        if opt == "adamw" and getattr(args, "layer_decay", None):
            param_groups = _build_layer_decay_param_groups(args, model)
            optimizer = optim.AdamW(
                param_groups,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif opt == "adamw":
            param_groups = [
                {"params": gain_or_bias_params, "weight_decay": 0.0, "lr": args.lr, "lr_scale": 1.0},
                {"params": rest_params, "weight_decay": args.wd, "lr": args.lr, "lr_scale": 1.0},
            ]
            if text_gain_or_bias_params:
                param_groups.append({
                    "params": text_gain_or_bias_params, "weight_decay": 0.0,
                    "lr": text_lr, "lr_scale": args.text_lr_multiplier,
                })
            if text_rest_params:
                param_groups.append({
                    "params": text_rest_params, "weight_decay": args.wd,
                    "lr": text_lr, "lr_scale": args.text_lr_multiplier,
                })
            optimizer = optim.AdamW(
                param_groups,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        else:
            assert False, f"Unknown optimizer {opt}"

    if is_master(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults["weight_decay"] = args.wd
        defaults = ", ".join([f"{k}: {v}" for k, v in defaults.items()])
        logging.info(
            f"Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}"
        )

    scaler = None
    if args.precision == "amp":
        try:
            scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    return optimizer, scaler


def maybe_compile_model(args, original_model):
    if args.torchcompile:
        logging.info("Compiling model...")

        if args.grad_checkpointing and args.distributed:
            logging.info("Disabling DDP dynamo optimizer when grad checkpointing enabled.")
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)
    else:
        model = original_model
    return model


def maybe_create_scheduler(args, data, optimizer):
    # create scheduler if train
    scheduler = None
    if "train" in data and optimizer is not None:
        if isinstance(data["train"], MultiDatasetDataloader):
            num_batches = data["train"].num_batches
        else:
            num_batches = data["train"].dataloader.num_batches
        total_steps = (num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, (
                "Please specify the number of cooldown epochs for this lr schedule."
            )
            cooldown_steps = (num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer,
                args.lr,
                args.warmup,
                total_steps,
                cooldown_steps,
                args.lr_cooldown_power,
                args.lr_cooldown_end,
            )
        else:
            logging.error(
                f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown."
            )
            exit(1)
    return scheduler


def maybe_init_remote_sync(args):
    # start the sync process if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info("remote sync successful.")
        else:
            logging.info("Error: remote sync failed. Exiting.")
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        remote_sync_process.start()
    return remote_sync_process


def maybe_resume_from_checkpoint(args, model, optimizer, scaler):
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location="cpu")
        if "epoch" in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            sd = {k: v for k, v in sd.items() if "relative_bias" not in k}
            
            # Handle module prefix mismatch between checkpoint and model
            if sd:  # Only check if state dict is not empty
                checkpoint_has_module = next(iter(sd.items()))[0].startswith("module.")
                if args.distributed and not checkpoint_has_module:
                    # Model is DDP (expects module.) but checkpoint doesn't have it
                    sd = {f"module.{k}": v for k, v in sd.items()}
                elif not args.distributed and checkpoint_has_module:
                    # Model is not DDP but checkpoint has module.
                    sd = {k[len("module."):]: v for k, v in sd.items()}
            
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            logging.info(
                f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})"
            )
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    return start_epoch, model, optimizer, scaler


def setup_cuda():
    if torch.cuda.is_available():
        # TF32 is validated on Ampere/Lovelace and should be revisited on H100s once profiling is complete.
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def maybe_load_pretrained_safetensors(args, model):
    """Initialize from a raw .safetensors checkpoint, bypassing AutoModel/trust_remote_code.

    The released Pillar0-ChestCT repo ships its own copies of atlas_encoders.py /
    multimodal_atlas.py / multimodal_msa.py for `AutoModel.from_pretrained(...,
    trust_remote_code=True)`. Going through that path would build the model with THAT code --
    including the patch-embed stride bug this variant exists to fix -- and it needs network/
    git resolution for `revision`. Only the tensors are wanted here, so read them directly.

    Keys are matched after stripping one leading 'model.' ('model.visual.*' -> 'visual.*').
    Shape mismatches are reported and skipped rather than silently dropped.
    """
    path = getattr(args, "pretrained_safetensors", "")
    if not path:
        return model

    from safetensors.torch import load_file

    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(f"--pretrained-safetensors: no such file: {path}")

    src = load_file(path, device="cpu")
    tgt = model.state_dict()

    loaded, mismatched = [], []
    for src_key, value in src.items():
        key = src_key[len("model."):] if src_key.startswith("model.") else src_key
        if key not in tgt:
            continue
        if tgt[key].shape == value.shape:
            tgt[key] = value
            loaded.append(key)
        else:
            mismatched.append((key, tuple(tgt[key].shape), tuple(value.shape)))

    not_loaded = sorted(set(tgt.keys()) - set(loaded))
    logging.info(
        f"=> safetensors load: {len(loaded)}/{len(tgt)} model tensors initialized from '{path}'"
    )
    if mismatched:
        logging.warning(
            f"=> safetensors load: {len(mismatched)} keys SKIPPED on shape mismatch and left at "
            f"random init: {mismatched}"
        )
    if not_loaded:
        logging.info(
            f"=> safetensors load: {len(not_loaded)} model tensors absent from the checkpoint "
            f"(left at current init): {not_loaded}"
        )
    if not loaded:
        raise RuntimeError(
            f"--pretrained-safetensors matched ZERO tensors in {path}. Checkpoint keys look "
            f"like {list(src)[:3]}, model keys like {list(tgt)[:3]} -- the prefix convention "
            f"must have changed."
        )

    model.load_state_dict(tgt, strict=False)
    return model


def maybe_load_pretrained_model(args, model, imagenet_pretrained=False):
    if len(args.pretrained_model) > 0:
        pretrained_model = AutoModel.from_pretrained(
            args.pretrained_model, 
            revision=args.revision, 
            trust_remote_code=True
        )

        if not imagenet_pretrained:
            model_state_dict = model.state_dict()
            pretrained_state_dict = pretrained_model.state_dict()
            loaded_keys = []
            shape_mismatch_keys = []
            for src_key, value in pretrained_state_dict.items():
                key = src_key.replace("model.", "")
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        model_state_dict[key] = value
                        loaded_keys.append(key)
                    else:
                        shape_mismatch_keys.append(
                            (key, tuple(model_state_dict[key].shape), tuple(value.shape))
                        )
            # NOTE: model_state_dict started as model.state_dict(), so it always contains every
            # key the model has -- load_state_dict's own missing_keys/unexpected_keys are always
            # empty and cannot be used to detect a partial load. Diff loaded_keys against the
            # model's own key set instead.
            not_loaded_keys = sorted(set(model_state_dict.keys()) - set(loaded_keys))
            logging.info(
                f"=> pretrained load: {len(loaded_keys)}/{len(model_state_dict)} model keys loaded "
                f"from '{args.pretrained_model}' (revision={args.revision})"
            )
            if shape_mismatch_keys:
                logging.warning(
                    f"=> pretrained load: {len(shape_mismatch_keys)} keys skipped due to shape "
                    f"mismatch and left RANDOMLY INITIALIZED: {shape_mismatch_keys}"
                )
            if not_loaded_keys:
                logging.info(
                    f"=> pretrained load: {len(not_loaded_keys)} model keys not present in checkpoint "
                    f"(left at current init): {not_loaded_keys}"
                )
            model.load_state_dict(model_state_dict, strict=False)
            return model


        ## specific to imagenet pretrained model
        ## pretrained_model -> model -> patch_embeds -> chest_xray_single_view
        ## model -> visual -> patch_embeds -> [chest_xray_single_view, chest_xray_multi_view]
        ## load the pretrained model patch-embeds/chest_xray_single_view into all the model's patch_embeds
        ## load the rest of the pretrained_model -> model keys into model -> visual as long as the key is not in model -> patch_embeds and match exists

        pretrained_state_dict = pretrained_model.state_dict()
        model_state_dict = model.state_dict()
        # Untouched reference to the model's own (freshly initialized) parameters, taken before
        # any pretrained values are written into model_state_dict below -- used to revert BN
        # stats that get paired with a conv layer we end up leaving randomly initialized.
        fresh_state_dict = model.state_dict()
        # pretrained_model.model.visual.patch_embeds.chest_ct.ds1


        # 1. Load patch_embeds/chest_xray_single_view weights into all model's patch_embeds
        # Find all keys in the pretrained model related to patch_embeds/chest_xray_single_view
        patch_embed_prefix = "patch_embeds.chest_xray_single_view"
        patch_embed_keys = [k for k in pretrained_state_dict if patch_embed_prefix in k]
        loaded_key_list = []
        missed_patch_embed_keys = []

        # 1. Load patch_embeds/chest_xray_single_view weights into all model's patch_embeds
        for src_key in patch_embed_keys:
            # For each patch_embed in your model, if the subkey matches, copy it
            # subkey = src_key[len(patch_embed_prefix):]  # e.g., '.weight' or '.bias'
            subkey = src_key.split(patch_embed_prefix)[-1]
            if "patch_embeds" not in src_key:
                continue
            for tgt_key in model_state_dict:
                if "patch_embeds" in tgt_key and tgt_key.endswith(subkey):
                    if model_state_dict[tgt_key].shape == pretrained_state_dict[src_key].shape:
                        model_state_dict[tgt_key] = pretrained_state_dict[src_key]
                        loaded_key_list.append(tgt_key)
                    # elif tgt_key == "visual.patch_embeds.chest_ct.conv_down.0.weight":
                    #     # take mean pooling of the source key
                    #     model_state_dict[tgt_key] = pretrained_state_dict[src_key].mean(dim=1).unsqueeze(1) 
                    #     loaded_key_list.append(tgt_key)

                    ## if the source can be inflated to the target, then load the source into the target
                    # elif model_state_dict[tgt_key].shape[0] % pretrained_state_dict[src_key].shape[0] == 0:
                    #     # General inflation: compute inflation factor for each dimension
                    #     source_shape = pretrained_state_dict[src_key].shape
                    #     target_shape = model_state_dict[tgt_key].shape
                        
                    #     # Check if all dimensions can be inflated (target must be divisible by source)
                    #     can_inflate = True
                    #     inflation_factors = []
                        
                    #     for i, (src_dim, tgt_dim) in enumerate(zip(source_shape, target_shape)):
                    #         if tgt_dim % src_dim != 0:
                    #             can_inflate = False
                    #             break
                    #         inflation_factors.append(tgt_dim // src_dim)
                        
                    #     if can_inflate:
                    #         # Inflate the kernel by repeating it according to inflation factors for each dimension
                    #         inflated_kernel = pretrained_state_dict[src_key]
                    #         for i, factor in enumerate(inflation_factors):
                    #             if factor > 1:
                    #                 # Create repeat pattern: repeat only the current dimension
                    #                 repeat_pattern = [1] * len(source_shape)
                    #                 repeat_pattern[i] = factor
                    #                 inflated_kernel = inflated_kernel.repeat(*repeat_pattern)
                            
                    #         model_state_dict[tgt_key] = inflated_kernel
                    #         loaded_key_list.append(tgt_key)
                    #         print(f"Inflated kernel for {tgt_key} with factors {inflation_factors}")
                    #     else:
                    #         missed_patch_embed_keys.append((tgt_key, src_key))

                    else:
                        missed_patch_embed_keys.append(
                            (tgt_key, src_key,
                             tuple(model_state_dict[tgt_key].shape),
                             tuple(pretrained_state_dict[src_key].shape))
                        )

        if missed_patch_embed_keys:
            logging.warning(
                f"=> imagenet-pretrained load: {len(missed_patch_embed_keys)} patch_embed keys skipped "
                f"due to shape mismatch against source '{patch_embed_prefix}' and left RANDOMLY "
                f"INITIALIZED (target/source shape shown per entry): {missed_patch_embed_keys}"
            )

        # 1b. A skipped conv_down.0 (the stem conv that touches the raw input and is the only
        #     layer whose in_channels can differ, e.g. chest_xray=3 vs chest_ct=11) leaves that
        #     conv randomly initialized. The BatchNorm right after it (conv_down.1) was loaded in
        #     step 1 too (its shape [C] doesn't depend on in_channels, so it always "matches"),
        #     but its running stats were calibrated on activations from the pretrained conv that
        #     we are NOT using here. Loading them onto a fresh conv's output is wrong (stale
        #     mean/var -> bad normalization -> a likely source of early-training instability), so
        #     revert those specific BN keys back to the model's own fresh init.
        reverted_bn_keys = []
        for tgt_key, _src_key, _tgt_shape, _src_shape in missed_patch_embed_keys:
            if tgt_key.endswith("conv_down.0.weight") or tgt_key.endswith("conv_down.0.bias"):
                bn_prefix = tgt_key.rsplit(".0.", 1)[0] + ".1."
                for k in list(loaded_key_list):
                    if k.startswith(bn_prefix):
                        model_state_dict[k] = fresh_state_dict[k]
                        loaded_key_list.remove(k)
                        reverted_bn_keys.append(k)
        if reverted_bn_keys:
            logging.warning(
                f"=> imagenet-pretrained load: reverted {len(reverted_bn_keys)} BatchNorm keys paired "
                f"with a shape-mismatched conv_down.0 back to fresh init (pretrained BN running "
                f"stats were calibrated for a different, discarded conv layer): {reverted_bn_keys}"
            )

        # 2. Load the rest of the pretrained_model -> model keys into model -> visual,
        #    as long as the key is not a patch_embed key (handled above) and a shape-matching
        #    target exists that hasn't already been loaded.
        for src_key, value in pretrained_state_dict.items():
            if patch_embed_prefix in src_key:
                continue  # already handled above
            # Map to model.visual.* if such a key exists in your model
            visual_key = src_key.replace("model.", "visual.", 1)
            if visual_key in loaded_key_list:
                continue
            if visual_key in model_state_dict and model_state_dict[visual_key].shape == value.shape:
                model_state_dict[visual_key] = value
                loaded_key_list.append(visual_key)

        # 3. Load the updated state dict into your model.
        # NOTE: model_state_dict started as model.state_dict(), so it always contains every key
        # the model has -- load_state_dict's own missing_keys/unexpected_keys are always empty
        # and cannot be used to detect a partial load. Diff loaded_key_list against the model's
        # own key set instead.
        not_loaded_keys = sorted(set(model_state_dict.keys()) - set(loaded_key_list))
        logging.info(
            f"=> imagenet-pretrained load: {len(loaded_key_list)}/{len(model_state_dict)} model keys "
            f"loaded from '{args.pretrained_model}' (revision={args.revision})"
        )
        if not_loaded_keys:
            logging.info(
                f"=> imagenet-pretrained load: {len(not_loaded_keys)} model keys NOT loaded from "
                f"checkpoint (left at current init): {not_loaded_keys}"
            )
        model.load_state_dict(model_state_dict, strict=False)
    return model



def main(args):
    args = parse_args(args)

    # setup cuda, distributed device, flash attn
    setup_cuda()
    handle_flash_attn(args)
    device = init_distributed_device(args)

    # get the name of the experiments
    args.name = maybe_sanitize_model_name(args)

    resume_latest = args.resume == "latest"
    # args.resume gets overwritten below (maybe_resume_from_checkpoint) from the literal string
    # "latest" to the actual resolved checkpoint path, so maybe_init_wandb can no longer check
    # `args.resume == "latest"` by the time it runs. Stash the original intent here instead.
    args.resume_latest = resume_latest
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print("Error. Experiment already exists. Use --name {} to specify a new experiment.")
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb and checkpoint logging
    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print("Error. Cannot use save-most-recent with remote_sync and resume latest.")
                return -1
            if args.remote_sync_protocol != "s3":
                print("Error. Sync protocol not supported when using resume latest.")
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                # First check for step_latest.pt, then epoch_latest.pt
                step_latest = os.path.join(checkpoint_path, "step_latest.pt")
                if os.path.exists(step_latest):
                    resume_from = step_latest
                else:
                    resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                    if not os.path.exists(resume_from):
                        # If no latest checkpoint has been saved yet, don't try to resume
                        resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    # copy codebase
    maybe_copy_codebase(args)
    remote_sync_process = maybe_init_remote_sync(args)

    ## check precision and distributed mode
    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train."
        )
    if args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    # set random seed
    random_seed(args.seed, 0)

    ## build model kwargs
    model_kwargs = {}
    if args.siglip:
        model_kwargs["siglip"] = True
        model_kwargs["init_logit_scale"] = np.log(10)  # different from CLIP
        model_kwargs["init_logit_bias"] = -10.0

    if args.textdino:
        model_kwargs["textdino"] = True

    if args.phrase_align:
        model_kwargs["phrase_align"] = True

    if args.lapa or getattr(args, 'phrase_align_polar', False):
        # phrase_align_polar changes only the phrase labels and the loss; the model
        # (CustomTextCLIP_LAPA) and the training loop are the lapa ones.
        model_kwargs["lapa"] = True

    if args.phrase_align_split:
        model_kwargs["phrase_align_split"] = True

    ## create model and transforms
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        multimodal_config=args.multimodal_config,
        **model_kwargs,
    )

    model = maybe_load_pretrained_model(args, model, imagenet_pretrained=args.imagenet_pretrained)
    model = maybe_load_pretrained_safetensors(args, model)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991 (실행 안함.)
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats,
        )
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm,
        )

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    else:
        params_file = None

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {"find_unused_parameters": True}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args["static_graph"] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    # create optimizer and scaler
    optimizer, scaler = build_optimizer(args, model, device)

    # resume from checkpoint
    start_epoch, model, optimizer, scaler = maybe_resume_from_checkpoint(args, model, optimizer, scaler)
    # breakpoint()
    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=tokenizer,)
    # breakpoint()
    assert len(data), "At least one train or eval dataset must be specified."

    # build scheduler
    scheduler = maybe_create_scheduler(args, data, optimizer)

    # determine if this worker should save logs and checkpoints
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)

    # initialize wandb
    maybe_init_wandb(args, data, model, params_file)

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    model = maybe_compile_model(args, original_model)

    ## build loss function
    loss = create_loss(args)

    ## build train and validate functions
    if args.multimodal_config is not None and args.textdino:
        train_one_epoch_fn = train_textdino_one_epoch
        validate_fn = evaluate_textdino
    elif args.multimodal_config is not None and args.siglip:
        train_one_epoch_fn = train_multimodal_siglip_one_epoch
        validate_fn = evaluate_multimodal_siglip
    elif args.multimodal_config is not None and args.phrase_align_split:
        train_one_epoch_fn = train_multimodal_one_epoch_phrase_align_split
        validate_fn = evaluate_multimodal_phrase_align_split
    elif args.multimodal_config is not None and (args.lapa or getattr(args, 'phrase_align_polar', False)):
        # Shared with lapa: the loop pops "total_loss" and logs every other
        # output_dict entry via .item(), so PhraseAlignPolarLoss's extra diagnostics
        # (source_valid_row_frac) flow through to wandb without any change here.
        train_one_epoch_fn = train_multimodal_one_epoch_lapa
        validate_fn = evaluate_multimodal_lapa
    elif args.multimodal_config is not None and args.phrase_align:
        train_one_epoch_fn = train_multimodal_one_epoch_phrase_align
        validate_fn = evaluate_multimodal_phrase_align
    elif args.multimodal_config is not None:
        train_one_epoch_fn = train_multimodal_one_epoch
        validate_fn = evaluate_multimodal
    else:
        train_one_epoch_fn = train_one_epoch
        validate_fn = evaluate

    # run validation only if --eval-only is set
    if getattr(args, 'eval_only', False):
        validate_fn(model, data, loss, start_epoch, args, tokenizer=tokenizer)
        return

    #########################################################
    ##############      Training Loop          ##############
    #########################################################

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        completed_epoch = epoch + 1

        train_one_epoch_fn(
            model,
            data,
            loss,
            epoch,
            optimizer,
            scaler,
            scheduler,
            args,
        )

        if any(v in data for v in ("val", "imagenet-val", "imagenet-v2")):
            validate_fn(
                model,
                data,
                loss,
                completed_epoch,
                args,
                tokenizer=tokenizer,
            )

        maybe_save_checkpoint(args, original_model, optimizer, scaler, completed_epoch)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync
    maybe_run_remote_sync_final(args, remote_sync_process)


if __name__ == "__main__":
    main(sys.argv[1:])
