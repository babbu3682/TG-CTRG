#!/usr/bin/env python3
"""
Standalone inference script for Pillar retrieval (TranSQ / Simple_TranSQ) models.

Mirrors scripts/inference.py (the abn/classification inference script) in
structure and CLI style, but adapted to the retrieval task:

- Model output is (sent_feats, topic_preds) instead of a single logit tensor.
- There is no "label" to predict directly; instead, each predicted semantic
  query is matched against a fixed sentence gallery (RA-RRG style, see
  Simple_TranSQ.retrieve()) to produce a free-text report made of retrieved
  key phrases.
- If ground-truth key phrase embeddings are present in the dataset
  (sent_embeds / sent_num, as produced by retrieval_collate_fn), Recall@K /
  MRR metrics are computed via the project's own RetrievalMetric — the same
  metric class used during training/validation.

This script is single-GPU only: it never initializes a process group or
wraps the model in DistributedDataParallel, and always uses a
SequentialSampler regardless of the config's dataloader.multi_gpu_eval value.

It works whether or not the dataset provides ground-truth key phrases:
  - If present, Recall@K / MRR metrics are computed and written to CSV.
  - If absent, only the retrieved report (+ accession) is written.

Usage:
    # Single GPU, default config-driven gallery + threshold
    python scripts/inference_retrieval.py configs/ctrate_retrieval_finetune_lz4_pillar_opensource.yaml \
        --ckpt logs/CTRATE/CTRATE_FT_Atlas_unimodal_retrieval_original_pillar/checkpoints/epoch=2.ckpt \
        --output logs/inference/retrieval_preds.csv

    # With uv, custom threshold, dev split
    uv run scripts/inference_retrieval.py configs/ctrate_retrieval_finetune_lz4_pillar_opensource.yaml \
        --ckpt logs/CTRATE/.../checkpoints/epoch=2.ckpt \
        --output logs/inference/retrieval_preds.csv \
        --split dev --threshold 0.5

    # Report only, skip metrics for speed
    uv run scripts/inference_retrieval.py configs/ctrate_retrieval_finetune_lz4_pillar_opensource.yaml \
        --ckpt logs/CTRATE/.../checkpoints/epoch=2.ckpt \
        --output preds.csv --no-metrics

    # Custom output locations
    uv run scripts/inference_retrieval.py configs/ctrate_retrieval_finetune_lz4_pillar_opensource.yaml \
        --ckpt logs/CTRATE/.../checkpoints/epoch=2.ckpt \
        --output logs/inference/preds.csv \
        --metrics-output results/metrics/retrieval_metrics.csv \
        --report-output results/reports/retrieval_reports.json
"""

import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pillar import datasets, models
from pillar.utils.parsing import parse_args
from pillar.utils.loading import get_eval_dataset_loader
from pillar.utils.logging import logger
from pillar.metrics.retrieval_metric import RetrievalMetric
from pillar.engines.base import parse_amp_precision


# --------------------------------------------------------------------------- #
# Config / CLI
# --------------------------------------------------------------------------- #
def parse_inference_args():
    """
    Parses the shared project config (same YAML/--opts system as train.py)
    plus a handful of inference-only flags.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.ckpt) to load")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file (per-sample predictions)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"], help="Which dataset split to run inference on")
    parser.add_argument("--batch-limit", type=int, default=None, help="Optional cap on number of batches (debugging)")
    parser.add_argument("--no-metrics", action="store_true", help="Skip metric computation (Recall@K / MRR via RetrievalMetric) even if ground-truth key phrases are found")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (single device only, e.g. 'cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--no-report", action="store_true", help="Skip gallery-retrieval report generation")
    parser.add_argument("--report-output", type=str, default=None, help="Path to output JSON file for generated reports (default: <output>_report.json)")
    parser.add_argument("--metrics-output", type=str, default=None, help="Path to output CSV for computed metrics (default: <output>_metrics.csv)")
    # 여러 값을 주면 **인코더 forward를 한 번만** 하고 threshold마다 리포트를 따로 쓴다.
    # RA-RRG의 0.4는 논문이 자기 데이터(N=50, 평균 7.16 phrase)에서 example-F1을 최대화해
    # 고른 값이다(Appendix F.3). 우리 세팅은 N=64 / 평균 15.2문장이라 최적점이 같을 이유가
    # 없고, 실측으로 0.4에서 GT의 1.85배를 발화한다(26.3 vs 15.2) — 그래서 재튜닝이 필요하고,
    # 인코더가 전체 비용의 대부분이므로 한 번의 GPU 패스로 sweep하는 것이 맞다.
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.4],
                        help="Sigmoid probability threshold(s) for selecting which semantic queries "
                             "fire. Multiple values sweep in a single encoder pass; each gets its own "
                             "report file (suffixed _th<value>).")

    known_infer_args, remaining = parser.parse_known_args()

    # Delegate the rest (config file + --opts overrides) to the project's own parser
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()

    return args, known_infer_args


# --------------------------------------------------------------------------- #
# Builders (mirrors train.py, trimmed to what inference needs)
# --------------------------------------------------------------------------- #
def build_dataset(args, split):
    logger.info(f"Building {split} dataset...")

    dataset_type = args.dataset.type
    if dataset_type not in datasets.__dict__:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataset_cls = datasets.__dict__[dataset_type]
    augmentations = []

    shared_kwargs = dict(args.dataset.shared_dataset_kwargs) if hasattr(args.dataset, "shared_dataset_kwargs") else {}

    if split == "train":
        split_kwargs = dict(args.dataset.dataset_train_kwargs) if hasattr(args.dataset, "dataset_train_kwargs") else {}
    elif split == "dev":
        split_kwargs = dict(args.dataset.dataset_dev_kwargs) if hasattr(args.dataset, "dataset_dev_kwargs") else {}
    elif split == "test":
        split_kwargs = dict(args.dataset.dataset_test_kwargs) if hasattr(args.dataset, "dataset_test_kwargs") else {}
    else:
        split_kwargs = {}

    dataset_kwargs = {**shared_kwargs, **split_kwargs}
    dataset = dataset_cls(args=args, augmentations=augmentations, split_group=split, **dataset_kwargs)

    logger.info(f"{split} dataset size: {len(dataset)}")
    return dataset


def build_dataloader(args, dataset):
    logger.info("Building inference dataloader...")

    if getattr(args.dataloader, "use_null_collate", False):
        raise ValueError(
            "args.dataloader.use_null_collate is True — this collate function always returns None "
            "and is only intended for cache-generation runs. Inference cannot proceed with it; "
            "set dataloader.use_null_collate=False for inference runs."
        )

    if args.dataset.mode != "retrieval":
        raise ValueError(
            f"inference_retrieval.py expects dataset.mode == 'retrieval', got '{args.dataset.mode}'. "
            f"Use scripts/inference.py for 'abn' mode."
        )

    # This script is single-GPU only: multi_gpu_eval (DistributedSampler) is
    # intentionally never used here, regardless of what the config says, since
    # DistributedSampler requires a process group to be initialized and pads/
    # drops samples in ways that would silently affect Recall@K/MRR.
    if getattr(args.dataloader, "multi_gpu_eval", False):
        logger.warning(
            "args.dataloader.multi_gpu_eval is True in the config, but inference_retrieval.py "
            "always runs single-GPU inference — ignoring multi_gpu_eval and using a "
            "SequentialSampler instead."
        )

    # Same loader factory (and therefore the same ignore_None_collate) that
    # scripts/train.py uses for its dev/test loaders, so batching here is identical to
    # the validation path that produced the numbers being reproduced.
    loader = get_eval_dataset_loader(args, dataset, shuffle=False, multi_gpu_eval=False)

    logger.info(f"Inference dataloader: {len(loader)} batches")
    return loader


def build_model(args):
    logger.info("Building model...")

    model_type = args.model.type
    if model_type not in models.__dict__:
        raise ValueError(f"Unknown model type: {model_type}")

    model_cls = models.__dict__[model_type]
    model_kwargs = dict(args.model.kwargs) if hasattr(args.model, "kwargs") and args.model.kwargs else {}

    model = model_cls(args=args, **model_kwargs)
    return model


def load_checkpoint_weights(ckpt_path, model, device):
    logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint at {ckpt_path} has no 'model' key. Keys found: {list(checkpoint.keys())}")

    state_dict = checkpoint["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

    epoch = checkpoint.get("epoch", None)
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    return model


def resolve_logit_key(args):
    """
    Figures out which model_output key holds the (sent_feats, topic_preds)
    tuple, by reusing whatever the configured metrics/losses already declare.
    Falls back to "retrieval" (the default used throughout the retrieval
    config/engine/loss/metric classes).
    """
    logit_key = None

    if hasattr(args, "metrics") and "epoch_metrics" in args.metrics:
        for split_metrics in args.metrics.epoch_metrics.values():
            for metric_info in split_metrics:
                kwargs = metric_info.get("kwargs", {})
                if "logit_key" in kwargs:
                    logit_key = kwargs["logit_key"]

    return logit_key or "retrieval"


def find_retrieval_metric_info(args, split):
    """
    Finds the RetrievalMetric entry whose kwargs carry the gallery paths, preferring
    the requested split. Returns (metric_info, source_split), or (None, None).

    Why the cross-split fallback is needed: this project's retrieval config
    deliberately sets `metrics.epoch_metrics.train: []` because keeping epoch metrics
    on during *training* accumulated (B, 64, 4096) tensors across every train batch
    until host RAM ran out (see the comment on that config key). Inference over the
    train split still needs the gallery — it is what turns semantic queries into
    text — and this script accumulates only one small per-batch stats dict, so
    borrowing another split's entry is both safe here and the only way `--split train`
    can run at all.
    """
    if not (hasattr(args, "metrics") and "epoch_metrics" in args.metrics):
        return None, None

    epoch_metrics = args.metrics.epoch_metrics
    candidates = [split] + [s for s in ("dev", "test", "val", "train") if s != split]
    for candidate in candidates:
        for metric_info in epoch_metrics.get(candidate, []):
            if "gallery_embeddings" in metric_info.get("kwargs", {}):
                return metric_info, candidate
    return None, None


def load_gallery(gallery_embeddings_path, gallery_sentences_path, device):
    """
    gallery_embeddings.pt: (N_gallery, d_target) sentence embeddings.
    gallery_sentences.pt : List[str] of length N_gallery, same order as embeddings.
    """
    gallery_embeds = torch.load(gallery_embeddings_path, weights_only=False)
    if not torch.is_tensor(gallery_embeds):
        gallery_embeds = torch.as_tensor(gallery_embeds)
    gallery_embeds = F.normalize(gallery_embeds.float(), p=2, dim=-1).to(device)

    gallery_texts = torch.load(gallery_sentences_path, weights_only=False)
    if len(gallery_texts) != gallery_embeds.size(0):
        raise ValueError(
            f"Gallery size mismatch: {len(gallery_texts)} sentences vs "
            f"{gallery_embeds.size(0)} embeddings."
        )
    return gallery_embeds, gallery_texts


def get_retrieval_head(model, logit_key):
    """
    Returns the head module (Simple_TranSQ instance) responsible for the
    logit_key output, so we can call its .retrieve() method directly.

    Why this is needed: model(...) only runs the forward pass and returns
    tensors (sent_feats, topic_preds) inside model_output. The gallery-search
    logic (.retrieve()) is an instance method defined on Simple_TranSQ itself,
    not on the tensor output, so it has to be called on the head module that
    produced those tensors — not re-run, just referenced for post-processing.

    Falls back to the model itself if head_models isn't exposed as expected
    (e.g. a bespoke model wrapper).
    """
    head_models = getattr(model, "head_models", None)

    if head_models is not None and logit_key in head_models:
        head = head_models[logit_key]
    else:
        head = None
    
    if head is None:
        logger.warning(
            f"Could not find model.head_models['{logit_key}'] — falling back to using "
            f"the full model as the retrieval head. This requires the model itself to "
            f"expose a .retrieve(...) method."
        )
        head = model
    return head


def normalize_sent_texts(sent_texts, batch_size, expected_num_sentences):
    """
    Returns the GT sentences as a per-sample list (length batch_size) of list[str].

    Why this is needed: ignore_None_collate() runs default_collate over the
    per-sample list[str] that CTRATE_RET_Dataset puts in "sent_texts", and
    default_collate TRANSPOSES a batch of equal-length string lists — B samples x S
    sentences comes back as S entries of B strings each. Indexing that by sample
    index would silently mix different patients' sentences into one report (the same
    per-sample-vs-structural confusion documented for gather_predictions_dict in
    CLAUDE.md). Every sample is padded to max_sentences by the dataset, so S is known
    and the transpose is unambiguous even when B happens to equal S.

    Args:
        sent_texts: whatever the collate produced for "sent_texts"
        batch_size: B
        expected_num_sentences: S (dataset's max_sentences), or None if unknown
    Returns:
        list of length B, each a list[str] of length S — or None if unrecognizable.
    """
    if sent_texts is None:
        return None

    def is_str_seq(v):
        return isinstance(v, (list, tuple)) and all(isinstance(s, str) for s in v)

    if not isinstance(sent_texts, (list, tuple)) or not sent_texts:
        return None

    outer = len(sent_texts)
    inner = len(sent_texts[0]) if is_str_seq(sent_texts[0]) else None
    if inner is None:
        return None

    # Transposed (collate output): outer == S, inner == B.
    transposed = (expected_num_sentences is not None and outer == expected_num_sentences) or (
        expected_num_sentences is None and outer != batch_size and inner == batch_size
    )
    if transposed:
        if inner != batch_size:
            logger.warning(
                f"'sent_texts' looks transposed ({outer} x {inner}) but its inner length "
                f"{inner} != batch size {batch_size}; skipping GT text alignment."
            )
            return None
        return [[sent_texts[s][b] for s in range(outer)] for b in range(batch_size)]

    if outer != batch_size:
        logger.warning(
            f"'sent_texts' has outer length {outer}, which matches neither the batch size "
            f"({batch_size}) nor max_sentences ({expected_num_sentences}); skipping GT text alignment."
        )
        return None
    return [list(v) for v in sent_texts]


def normalize_accessions(accessions, batch_size):
    """
    ignore_None_collate leaves a per-sample list of strings intact for 'accession',
    so this is mostly a defensive pass-through + length check.
    """
    if isinstance(accessions, torch.Tensor):
        accessions = accessions.tolist()
    elif isinstance(accessions, (list, tuple)):
        accessions = list(accessions)
    else:
        accessions = [accessions]

    if len(accessions) != batch_size:
        raise ValueError(
            f"Collated 'accession' field has length {len(accessions)} but batch size is "
            f"{batch_size}. retrieval_collate_fn is expected to always produce a plain "
            f"list matching batch size; a custom collate_fn may be in use."
        )
    return accessions


def preprocess_batch(args, batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
        else:
            batch[k] = v
    batch["x"] = batch["x"].to(dtype=torch.float32)

    if hasattr(args.engine, "kwargs") and getattr(args.engine.kwargs, "use_gpu_augs", False):
        # Inference always uses eval-time (test) augmentations, never train-time ones.
        from pillar import augmentations as augmentations_module

        gpu_augs_cfg = getattr(args.engine.kwargs, "test_gpu_augs", [])
        gpu_augs = []
        for aug in gpu_augs_cfg:
            name = aug["type"]
            kwargs = aug["kwargs"] if "kwargs" in aug else {}
            gpu_augs.append(augmentations_module.__dict__[name](args, **kwargs))

        inputs = {"input": batch["x"]}
        if "mask" in batch:
            inputs["mask"] = batch["mask"]
        for transform in gpu_augs:
            inputs = transform(inputs)
        batch["x"] = inputs["input"]

    if args.dataset.shared_dataset_kwargs.windows:
        import rve
        from pillar.utils.windowing import batch_apply_ct_windowing

        windows_type = args.dataset.shared_dataset_kwargs.windows
        modality = args.dataset.modality
        x = batch["x"]

        if modality == "CT":
            # "minmax" window는 배치 전체가 아니라 샘플별로 min/max를 계산해야 함
            # (자세한 이유는 pillar/utils/windowing.py 참고) — 그 로직을 공용 유틸로 뺌.
            batch["x"] = batch_apply_ct_windowing(x, windows_type)
        elif modality == "MR":
            if len(x.shape) == 4:
                B, D, H, W = x.shape
                C = 1
            else:
                B, C, D, H, W = x.shape

            if windows_type == "all":
                windows = rve.get_available_windows(modality)
            elif isinstance(windows_type, str):
                windows = [windows_type]
            else:
                windows = windows_type

            windowed = torch.zeros((B, len(windows) * C, D, H, W), device=x.device, dtype=x.dtype)
            for n in range(B):
                for i, window in enumerate(windows):
                    for j in range(C):
                        windowed[n, i * C + j] = rve.apply_windowing(x[n, j], window, modality)
            batch["x"] = (windowed - 0.041) / 0.073
        else:
            raise ValueError(f"Invalid modality: {modality}")

    return batch


def extract_gt_key_phrases(gt_embeds_i, gt_num_i, gallery_embeds, gallery_texts):
    """
    Fallback: batch에 실제 원문(sent_texts)이 없을 때만 쓰는 근사치.

    CTRATE_RET_Dataset은 이제 원본 json row의 "sentences"를 sent_texts로 batch에
    실어주지만(row["key_phrase"]가 곧 그 텍스트이고 text_cache는 그걸 임베딩한 것뿐이라
    순서가 1:1로 동일함), 혹시 그 필드가 없는 데이터셋 변형이나 과거 캐시를 쓸 경우를
    대비해 RetrievalMetric/Simple_TranSQ.retrieve()와 동일한 방식으로 GT 임베딩을
    gallery에서 코사인 유사도가 가장 큰 문장에 매핑해 "근사 원문"을 만든다. 이 경우 gallery가
    train corpus라서 val/test의 실제 원문과 다를 수 있다는 점에 유의 — 정량 지표(recall@k
    등)에는 맞는 방식이지만 사람이 읽는 비교용으로는 sent_texts가 있으면 그쪽을 우선한다.

    Args:
        gt_embeds_i: (max_sentences, D) 이 샘플의 (패딩 포함) sent_embeds 슬라이스
        gt_num_i: 이 샘플의 실제(패딩 제외) 문장 개수
        gallery_embeds: (N_gallery, D) L2-normalized gallery 임베딩
        gallery_texts: length N_gallery, gallery_embeds와 같은 순서의 문장 리스트
    Returns:
        list[str] (중복 제거, 등장 순서 유지) — 실제 문장이 0개면 빈 리스트
    """
    n = int(gt_num_i)
    if n == 0:
        return []

    gt_norm = F.normalize(gt_embeds_i[:n].detach().float(), p=2, dim=-1)
    idx = (gt_norm @ gallery_embeds.T).argmax(dim=-1).tolist()
    phrases = [gallery_texts[i] for i in idx]
    return list(dict.fromkeys(phrases))


# --------------------------------------------------------------------------- #
# Main inference loop
# --------------------------------------------------------------------------- #
def run_inference(args, infer_args):
    # Single-GPU only: even if multiple GPUs are visible (e.g. CUDA_VISIBLE_DEVICES
    # lists several), this script always targets exactly one device and never
    # spawns a process group or wraps the model in DistributedDataParallel.
    if infer_args.device.startswith("cuda") and torch.cuda.is_available():
        if ":" not in infer_args.device:
            infer_args.device = "cuda:0"
        device = torch.device(infer_args.device)
        torch.cuda.set_device(device)
        if torch.cuda.device_count() > 1:
            logger.info(
                f"{torch.cuda.device_count()} GPUs are visible, but inference_retrieval.py is "
                f"single-GPU only — pinning to {device}. Set CUDA_VISIBLE_DEVICES to restrict "
                f"visibility if needed."
            )
    else:
        device = torch.device("cpu")

    dataset = build_dataset(args, infer_args.split)
    loader = build_dataloader(args, dataset)

    model = build_model(args)
    model = load_checkpoint_weights(infer_args.ckpt, model, device)
    model = model.to(device)
    model.eval()

    logit_key = resolve_logit_key(args)
    logger.info(f"Retrieval logit key: {logit_key}")

    # Padded GT-sentence count per sample, used to undo default_collate's transpose of
    # "sent_texts" (see normalize_sent_texts).
    max_sentences = getattr(args.dataset.shared_dataset_kwargs, "max_sentences", None)

    # Gallery is required to turn semantic queries into text — pulled from the
    # same config path used by RetrievalMetric during training/validation.
    # Resolved ONCE here and reused for both the report and the metric below, so the
    # gallery the reports are built from can never silently diverge from the gallery
    # the Recall@K numbers are scored against.
    metric_info, gal_split = find_retrieval_metric_info(args, infer_args.split)
    if metric_info is None:
        raise ValueError(
            f"Could not find gallery_embeddings/gallery_sentences under "
            f"args.metrics.epoch_metrics['{infer_args.split}'][*].kwargs. "
            f"These are required to run retrieval inference."
        )
    if gal_split != infer_args.split:
        logger.warning(
            f"metrics.epoch_metrics['{infer_args.split}'] declares no RetrievalMetric with a "
            f"gallery — borrowing the entry from '{gal_split}' instead (see "
            f"find_retrieval_metric_info for why this is safe at inference time)."
        )
    gal_kwargs = metric_info["kwargs"]

    gallery_embeds, gallery_texts = load_gallery(gal_kwargs["gallery_embeddings"], gal_kwargs["gallery_sentences"], device)
    logger.info(f"Loaded gallery: {len(gallery_texts)} sentences from {gal_kwargs['gallery_embeddings']}")

    head = get_retrieval_head(model, logit_key)

    # Ground-truth metric (Recall@K / MRR) reuses the project's own RetrievalMetric,
    # same as train.py's evaluate() path, so scoring is identical to training/val.
    retrieval_metric = None
    if not infer_args.no_metrics and metric_info["type"] == "RetrievalMetric":
        retrieval_metric = RetrievalMetric(args, dataset_info=None, split=infer_args.split, **gal_kwargs)

    # 학습/평가와 동일한 AMP 정밀도를 config에서 그대로 읽어온다(위 forward의 주석 참고).
    amp_dtype = parse_amp_precision(getattr(args.engine.kwargs, "precision", "bf16-mixed"))
    logger.info(f"AMP precision for inference: {amp_dtype} "
                f"(from engine.kwargs.precision={getattr(args.engine.kwargs, 'precision', None)})")

    pred_rows = []       # per-sample fired-query stats (for --output CSV)
    thresholds = sorted(set(float(t) for t in infer_args.threshold))
    # threshold -> per-sample retrieved reports (for --report-output JSON)
    report_records = {thr: [] for thr in thresholds}
    metric_stats_list = []  # per-batch RetrievalMetric stats_dict, averaged at the end
    has_labels = None
    skipped_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Inference [{infer_args.split}]")):
            if infer_args.batch_limit is not None and batch_idx >= infer_args.batch_limit:
                break

            if batch is None:
                # retrieval_collate_fn returns None when every sample in the batch failed to load.
                skipped_batches += 1
                continue

            try:
                batch = preprocess_batch(args, batch, device)
            except Exception as e:
                logger.warning(f"Skipping batch {batch_idx} due to preprocessing error: {e}")
                skipped_batches += 1
                continue

            batch_size = batch["x"].shape[0]
            accessions = normalize_accessions(batch["accession"], batch_size)
            sent_texts = normalize_sent_texts(batch.get("sent_texts"), batch_size, max_sentences)

            # ⚠️ dtype을 반드시 명시한다. `torch.amp.autocast("cuda")`의 기본값은 **fp16**
            # (max 65,504)이고, 학습/평가 엔진은 config의 `engine.kwargs.precision`
            # (= bf16-mixed, max 3.4e38)을 쓴다. 이 불일치 때문에 phrase_align_split
            # 체크포인트를 채점할 때 activ 32,768 토큰이 cross-attention에서 fp16 범위를
            # 넘겨 inf -> NaN이 되고, 1,552례 전부 0문장이 나왔다(job 28024889).
            # lapa는 activ가 1,024 토큰뿐이라 우연히 살아남아 문제가 안 보였다.
            # 즉 채점 경로가 학습 때와 다른 수치 정밀도로 돌고 있었다.
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=device.type == "cuda" and amp_dtype is not None):
                model_output = model(batch["x"], batch=batch, split="test")

            sent_feats, topic_preds = model_output[logit_key]
            # aux_loss models return (L, B, N_q, ...) — only the final decoder layer is used at inference.
            if sent_feats.dim() == 4:
                sent_feats, topic_preds = sent_feats[-1], topic_preds[-1]

            if has_labels is None:
                has_labels = ("sent_embeds" in batch) and ("sent_num" in batch)
                if has_labels:
                    logger.info("Ground-truth key phrase embeddings detected in batch — will compute Recall@K/MRR.")
                else:
                    logger.info("No ground-truth key phrases found in batch — writing retrieved reports only.")

            # --- Ground-truth metrics (RetrievalMetric), same code path as train.py's evaluate() --- #
            if has_labels and retrieval_metric is not None:
                try:
                    stats = retrieval_metric(**{
                        retrieval_metric.logit_key: (sent_feats, topic_preds),
                        retrieval_metric.gt_embeds_key: batch[retrieval_metric.gt_embeds_key],
                        retrieval_metric.gt_len_key: batch[retrieval_metric.gt_len_key],
                    })
                    metric_stats_list.append(stats)
                except Exception as e:
                    logger.warning(f"RetrievalMetric failed on batch {batch_idx}: {e}")

            # --- Gallery retrieval report (RA-RRG style, reuses Simple_TranSQ.retrieve) --- #
            # threshold별로 retrieve를 반복하되 인코더 forward는 위에서 이미 한 번만 했다.
            # retrieve는 (발화 query 수 x 79,166) matmul 하나라 인코더에 비해 무시할 수준.
            prob = topic_preds.float().sigmoid()
            phrase_lists_by_thr = {}
            if not infer_args.no_report:
                for thr in thresholds:
                    phrase_lists_by_thr[thr] = head.retrieve(
                        sent_feats, topic_preds, gallery_embeds, gallery_texts,
                        prob_threshold=thr,
                    )

            for i in range(batch_size):
                row = OrderedDict(accession=accessions[i])
                for thr in thresholds:
                    row[f"num_fired@{thr:g}"] = int((prob[i] > thr).sum().item())
                if len(thresholds) == 1:
                    # 단일 threshold일 때는 기존 컬럼명을 유지한다(하위 호환:
                    # ret_reports_to_reportgen_csv.py / 기존 분석 노트가 이 이름을 읽는다).
                    row["num_fired"] = row.pop(f"num_fired@{thresholds[0]:g}")
                pred_rows.append(row)


                if not infer_args.no_report:
                    if has_labels:
                        gt_num_phrases = int(batch["sent_num"][i].item())
                        # Drop the padding the dataset added up to max_sentences: only the
                        # first sent_num entries are this patient's real sentences.
                        real_texts_i = None
                        if sent_texts is not None:
                            real_texts_i = [t for t in sent_texts[i][:gt_num_phrases] if t]
                        if real_texts_i:
                            # dataset이 원문을 실어준 경우 (정상 경로) — gallery 근사 없이 그대로 사용
                            gt_key_phrases = real_texts_i
                            gt_key_phrases_source = "raw_text"
                        else:
                            # fallback: 원문이 없는 경우에만 gallery 최근접 근사 사용
                            gt_key_phrases = extract_gt_key_phrases(
                                batch["sent_embeds"][i], batch["sent_num"][i], gallery_embeds, gallery_texts
                            )
                            gt_key_phrases_source = "gallery_nearest_neighbor_approx"
                    else:
                        gt_num_phrases = None
                        gt_key_phrases = None
                        gt_key_phrases_source = None

                    for thr in thresholds:
                        report_records[thr].append({
                            "accession": accessions[i],
                            # prediction
                            "pred_threshold": thr,
                            "pred_num_fired": int((prob[i] > thr).sum().item()),
                            "pred_key_phrases": phrase_lists_by_thr[thr][i],
                            # ground truth
                            "gt_key_phrases_source": gt_key_phrases_source,
                            "gt_num_phrases": gt_num_phrases,
                            "gt_key_phrases": gt_key_phrases,
                        })

    if skipped_batches:
        logger.warning(f"Skipped {skipped_batches} batch(es) that were entirely filtered out by the collate function (e.g. all samples failed to load).")

    if not pred_rows:
        logger.warning("No predictions were produced (empty dataloader or all batches corrupted).")
        return None, None, has_labels, metric_stats_list

    df = pd.DataFrame(pred_rows)
    output_path = Path(infer_args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} predictions to {output_path}")

    report_dfs = {thr: pd.DataFrame(recs) for thr, recs in report_records.items() if recs} or None

    return df, report_dfs, has_labels, metric_stats_list


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def maybe_compute_metrics(metric_stats_list, has_labels, infer_args):
    """
    RetrievalMetric is computed per-batch (it needs the raw sent_feats/topic_preds
    tensors and performs Hungarian matching internally, so it cannot be
    recomputed later from flattened CSV columns the way AbnMetric can).
    metric_stats_list already holds one stats_dict per batch; here we
    macro-average across batches (consistent with how AverageMeter-based
    logging works elsewhere in the codebase) and write the result to CSV.
    """
    if not has_labels or infer_args.no_metrics:
        return

    if not metric_stats_list:
        logger.warning("has_labels=True but no RetrievalMetric stats were collected; skipping metrics output.")
        return

    keys = metric_stats_list[0].keys()
    averaged = OrderedDict()
    for k in keys:
        values = [s[k] for s in metric_stats_list if k in s and not (isinstance(s[k], float) and np.isnan(s[k]))]
        averaged[k] = float(np.mean(values)) if values else float("nan")

    logger.info("=" * 60)
    logger.info("Metrics (via RetrievalMetric: Recall@K / MRR)")
    logger.info("=" * 60)
    for k, v in averaged.items():
        logger.info(f"{k}: {v:.4f}")

    metrics_df = pd.DataFrame(list(averaged.items()), columns=["metric", "value"])
    metrics_path = Path(infer_args.metrics_output) if infer_args.metrics_output else Path(infer_args.output).with_name(Path(infer_args.output).stem + "_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved {len(metrics_df)} metrics to {metrics_path}")


# --------------------------------------------------------------------------- #
# Report saving (JSON, fixed format — no CSV option)
# --------------------------------------------------------------------------- #
def save_reports(report_df, report_path):
    import json

    report_path.parent.mkdir(parents=True, exist_ok=True)
    records = report_df.to_dict(orient="records")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(records)} text reports to {report_path}")


def join_phrases(phrases):
    out = []
    for p in phrases:
        p = str(p).strip()
        if not p:
            continue
        if not p.endswith((".", "?", "!")):
            p += "."
        out.append(p)
    return " ".join(out)


def main():
    args, infer_args = parse_inference_args()

    logger.info("=" * 80)
    logger.info("Starting retrieval inference (single-GPU)")
    logger.info(f"Checkpoint: {infer_args.ckpt}")
    logger.info(f"Split: {infer_args.split}")
    logger.info(f"Output: {infer_args.output}")
    logger.info(f"Threshold(s): {infer_args.threshold}")
    logger.info("=" * 80)

    df, report_dfs, has_labels, metric_stats_list = run_inference(args, infer_args)

    if df is not None:
        maybe_compute_metrics(metric_stats_list, has_labels, infer_args)

        if not infer_args.no_report and report_dfs:
            base = Path(infer_args.report_output) if infer_args.report_output else Path(
                infer_args.output).with_name(Path(infer_args.output).stem + "_report.json")
            multi = len(report_dfs) > 1
            for thr, report_df in sorted(report_dfs.items()):
                # threshold가 하나면 파일명을 그대로 둔다(기존 산출물/스크립트 호환).
                path = base if not multi else base.with_name(f"{base.stem}_th{thr:g}{base.suffix}")
                save_reports(report_df, path)

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()



# metric
# uv run scripts/ret_inference.py configs/ctrate_retrieval_finetune_lz4_pillar_opensource.yaml \
#     --ckpt /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/checkpoints/best.ckpt \
#     --split test \
#     --output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/train_ret_results.csv \
#     --report-output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/train_ret_reports.json \
#     --metrics-output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/train_ret_metrics.csv \
#     --threshold 0.4


# report 만

# val
# uv run scripts/ret_inference.py configs/ctrate_retrieval_finetune_lz4_pillar_opensource.yaml \
#     --ckpt /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/checkpoints/best.ckpt \
#     --split test \
#     --output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/valid_ret_results.csv \
#     --report-output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/valid_ret_reports.json \
#     --metrics-output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/valid_ret_metrics.csv \
#     --threshold 0.4
#     --no-metrics

# train
# uv run scripts/ret_inference.py configs/ctrate_retrieval_finetune_lz4_pillar_opensource.yaml \
#     --ckpt /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/checkpoints/best.ckpt \
#     --split test \
#     --output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/train_ret_results.csv \
#     --report-output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/train_ret_reports.json \
#     --metrics-output /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze/train_ret_metrics.csv \
#     --threshold 0.4
#     --no-metrics