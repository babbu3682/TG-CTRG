#!/usr/bin/env python3
"""
Pre-extract frozen backbone "pooled" features for the abn multi-label task.

Why this exists: with augmentation off (engine.kwargs.use_gpu_augs: False) and
the backbone fully frozen, the pooled feature for a given sample is now
identical every epoch — the abn head is the only thing that actually learns.
Re-running the (expensive) 3D backbone forward every epoch just to feed a
single Linear(1152, 18) head is wasted compute. Extract once, cache to disk,
then train the probe directly on the cached tensors (see
scripts/train_abn_probe.py). This mirrors the pillar-0 authors' own
rate-evals recipe (rate-extract -> cached embeddings -> fast linear probe).

Always forces center-crop (regardless of split), because
CTRATE_ABN_Dataset uses crop_mode='random' for is_train=True — a random
crop is exactly the kind of noise a *fixed* cached feature must not have.

This script is backbone-agnostic (it reads backbone_model_type/backbone_kwargs
straight from the config), so it's meant to be re-run once per pretrained
checkpoint you want to compare — each run identified by --tag. Outputs land
under <cache-root>/<tag>/{split}.pt so results from different pretrain
variants never collide and stay easy to diff (e.g. --tag original_pillar for
the official pillar-0 HF checkpoint, --tag <your_other_checkpoint_name> for
each of the other pretrain runs you're comparing against it).

Usage:
    uv run scripts/extract_abn_features.py configs/ctrate_abnclass_finetune_lz4_pillar_opensource.yaml \
        --tag original_pillar \
        --splits train,dev

무엇을 캐시할 수 있고 무엇은 못 하는가 (--save):
    'pooled'  기본값. (N, 1152) — 샘플당 4.5 KB, train 105 MB. probe 승자(0.8692)가 쓴 것.
    'activ'   공간 토큰 맵. 풀 해상도는 샘플당 72 MB(32x32x32, self 계열) ~ 144 MB
              (32x32x64, patch [8,8,4]인 오픈소스)라 train split이 1.6~3.3 TB가 된다.
              그래서 --activ-pool로 공간 축을 줄여야만 저장된다 (4를 주면 8x8x8, 즉 64배
              축소 -> float16으로 train 약 26 GB).
    그 외      backbone이 내주는 아무 키. 예: MultimodalAtlasVLCABS의 'vlcabs_logits' (18차원).

    ⚠️ 다운샘플한 activ는 attnpool(MultiAttentionPool)이나 retrieval(Simple_TranSQ)을
    "재현"하는 데는 쓸 수 없다 — 두 head 모두 풀 해상도 토큰으로 학습됐으므로 입력 분포가
    달라진다. 새 실험을 싸게 프로토타이핑하는 용도로만 쓸 것.

    ⚠️ retrieval head 학습은 근본적으로 이 캐시로 대체할 수 없다. Simple_TranSQ는
    apply_pooling/use_pooled_features가 둘 다 False라서 매 스텝 풀 해상도 activ를
    cross-attention 입력으로 받는다.
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pillar import datasets, models
from pillar.utils.parsing import parse_args
from pillar.utils.loading import get_eval_dataset_loader
from pillar.utils.logging import logger
from pillar.utils.windowing import batch_apply_ct_windowing
from pillar.utils.vlcabs import (
    build_vision_projector_from_checkpoint,
    resolve_checkpoint_path,
)

# 여러 pretrain checkpoint(현재 4개)를 비교할 예정이라, 모든 추출 결과를 한 root 아래
# <tag>별 하위 폴더로 모아둔다. --output-dir을 명시하면 그걸로 override.
DEFAULT_CACHE_ROOT = "/data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-pretrain/data/abn_probe_cache"

# 한 split 파일의 예상 용량이 이 값을 넘으면 추출 전에 중단한다(--force-size로 우회).
# activ를 무심코 풀 해상도로 요청하면 train split이 TB 단위가 되어 파일시스템을 채운다.
SIZE_GUARD_GB = 100


def parse_extract_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tag", type=str, required=True, help="이 pretrain checkpoint/backbone을 구분하는 이름 (예: original_pillar). 출력이 <cache-root>/<tag>/ 아래 저장됨")
    parser.add_argument("--cache-root", type=str, default=DEFAULT_CACHE_ROOT, help="여러 --tag 결과를 함께 모아두는 상위 디렉토리")
    parser.add_argument("--splits", type=str, default="train,dev", help="Comma-separated splits to extract (train/dev/test)")
    parser.add_argument("--output-dir", type=str, default=None, help="<cache-root>/<tag> 대신 쓸 출력 디렉토리 (직접 지정 시 override)")
    parser.add_argument("--batch-size", type=int, default=None, help="Overrides dataloader.eval_batch_size for extraction (no backward pass, can usually go higher)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on number of samples per split (for fast iteration/debugging; omit for a full extraction)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--z-offset", type=int, default=None,
                        help="TTA: center crop 대신 z 축으로 n 슬라이스 이동한 크롭을 쓴다 "
                             "(추론 시점 증강. 학습된 head 는 그대로 두고 예측만 평균한다)")
    parser.add_argument("--save", type=str, default="pooled",
                        help="캐시할 backbone 출력들 (콤마 구분). 'pooled'(1152차원), 'activ'(공간 토큰 — "
                             "--activ-pool 필수, 아래 참고), 그리고 backbone이 내주는 다른 키(예: "
                             "MultimodalAtlasVLCABS의 'vlcabs_logits'). 기본값 'pooled'로 기존 동작 유지.")
    parser.add_argument("--activ-pool", type=int, default=None,
                        help="activ를 저장하기 전에 공간 축마다 이 배수로 average-pool한다 (예: 4 -> "
                             "32x32x32이 8x8x8로). 풀 해상도 activ는 샘플당 72~144MB라 train split이 "
                             "1.6~3.3TB가 되므로, activ를 저장하려면 이 값을 반드시 지정해야 한다. "
                             "1을 주면 풀 해상도이며 --force-size 없이는 거부된다.")
    parser.add_argument("--vision-projector", action="store_true",
                        help="체크포인트의 vision_projector(atlas_dim -> embed_dim, bias=False)를 "
                             "spatial token 에 적용해 `pooled_proj` 키를 추가로 만든다: "
                             "maxpool(normalize(proj(token))) -> normalize, 즉 VL-CABS zero-shot 이 "
                             "쓰는 것과 **같은 토큰 표현**에 attention 대신 고정 maxpool 을 건 것. "
                             "zero-shot 이 full training 을 넘은 원인이 '폭/투영'인지 "
                             "'attention pooling'인지 갈라내기 위한 arm (lapa 전용).")
    parser.add_argument("--cache-dtype", type=str, default="float32", choices=["float32", "float16"],
                        help="디스크에 저장할 dtype. activ를 저장할 때는 float16을 권장(용량 절반).")
    parser.add_argument("--force-size", action="store_true",
                        help=f"예상 용량이 {SIZE_GUARD_GB} GB를 넘어도 진행한다. 기본은 쓰기 전에 중단.")

    known_extract_args, remaining = parser.parse_known_args()

    # Delegate the rest (config file + --opts overrides) to the project's own parser
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()

    # build_dataset 은 args 만 받으므로 TTA z-offset 을 여기에 실어 나른다.
    args._z_offset = known_extract_args.z_offset

    return args, known_extract_args


def build_dataset(args, split):
    logger.info(f"Building {split} dataset...")

    dataset_type = args.dataset.type
    if dataset_type not in datasets.__dict__:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    dataset_cls = datasets.__dict__[dataset_type]

    shared_kwargs = dict(args.dataset.shared_dataset_kwargs) if hasattr(args.dataset, "shared_dataset_kwargs") else {}

    split_key = {"train": "dataset_train_kwargs", "dev": "dataset_dev_kwargs", "test": "dataset_test_kwargs"}.get(split)
    if split_key is None or not hasattr(args.dataset, split_key):
        raise ValueError(f"No dataset kwargs found for split '{split}' (expected args.dataset.{split_key})")
    split_kwargs = dict(getattr(args.dataset, split_key))

    dataset_kwargs = {**shared_kwargs, **split_kwargs}
    dataset = dataset_cls(args=args, augmentations=[], split_group=split, **dataset_kwargs)

    # 매 epoch 다른 depth 위치를 뽑는 random crop은 "frozen backbone 출력은 샘플당 하나로 고정"이라는
    # 이번 추출의 전제와 어긋난다. split과 무관하게 항상 center-crop으로 고정해서 뽑는다.
    if getattr(dataset, "is_train", False):
        logger.info(f"[{split}] forcing dataset.is_train=False so volume loading uses center-crop (deterministic)")
        dataset.is_train = False

    z_off = getattr(args, "_z_offset", None)
    if z_off is not None:
        dataset.crop_mode_override = f"offset:{z_off}"
        logger.info(f"[{split}] TTA: z 크롭을 center 에서 {z_off:+d} 슬라이스 이동 "
                    f"(crop_mode={dataset.crop_mode_override})")

    logger.info(f"[{split}] dataset size: {len(dataset)}")
    return dataset


def build_backbone(args, device):
    logger.info("Building backbone (head is not needed for feature extraction)...")

    backbone_type = args.model.kwargs.backbone_model_type
    if backbone_type not in models.__dict__:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    backbone_kwargs = dict(args.model.kwargs.backbone_kwargs) if hasattr(args.model.kwargs, "backbone_kwargs") else {}

    backbone = models.__dict__[backbone_type](args=args, **backbone_kwargs)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def batch_windowing(args, x):
    """CT-only. Uses the shared per-sample-safe helper (see pillar/utils/windowing.py) —
    rve.apply_windowing's "minmax" window computes min/max over whatever tensor it's given,
    so calling it on a stacked batch mixes statistics across unrelated samples."""
    modality = args.dataset.modality
    if modality != "CT":
        raise NotImplementedError(f"extract_abn_features.py currently only supports CT modality, got '{modality}'")

    return batch_apply_ct_windowing(x, args.dataset.shared_dataset_kwargs.windows)


def maybe_pool_activ(activ, factor):
    """activ (B, C, D, H, W)를 공간 축마다 factor배로 average-pool. factor=1이면 그대로."""
    if factor is None or factor == 1:
        return activ
    return torch.nn.functional.avg_pool3d(activ, kernel_size=factor, stride=factor)


def check_projected_size(name, per_sample_bytes, n_samples, force):
    """추출을 시작하기 전에 예상 용량을 알려주고, 가드를 넘으면 중단한다."""
    total_gb = per_sample_bytes * n_samples / 1024 ** 3
    per_mb = per_sample_bytes / 1024 ** 2
    logger.info(f"  '{name}': 샘플당 {per_mb:.2f} MB x {n_samples} = {total_gb:.2f} GB")
    if total_gb > SIZE_GUARD_GB and not force:
        raise RuntimeError(
            f"'{name}' 캐시가 {total_gb:.1f} GB로 예상되어 가드({SIZE_GUARD_GB} GB)를 넘습니다. "
            f"activ라면 --activ-pool을 키우거나(예: 4 -> 공간 64배 축소) --cache-dtype float16을 "
            f"쓰거나 --max-samples로 줄이세요. 정말 이 용량을 쓰려면 --force-size."
        )


@torch.no_grad()
def extract_split(args, backbone, split, device, batch_size, max_samples=None,
                  save_keys=("pooled",), activ_pool=None, cache_dtype=torch.float32,
                  force_size=False, vision_projector=None):
    dataset = build_dataset(args, split)

    if max_samples is not None and max_samples < len(dataset):
        logger.info(f"[{split}] capping dataset to first {max_samples}/{len(dataset)} samples (--max-samples)")
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    # DDP 초기화를 거치지 않는 단일 프로세스 실행이므로 rank 관련 속성을 직접 채워준다.
    if not hasattr(args.main, "global_rank"):
        args.main.global_rank = 0
    if not hasattr(args.main, "world_size"):
        args.main.world_size = 1
    if batch_size is not None:
        args.dataloader.eval_batch_size = batch_size

    loader = get_eval_dataset_loader(args, dataset, shuffle=False, multi_gpu_eval=False)
    logger.info(f"[{split}] {len(loader)} batches")

    buffers = {k: [] for k in save_keys}
    labels, sample_names = [], []
    skipped = 0
    size_checked = False

    for batch in tqdm(loader, desc=f"Extracting [{split}]"):
        if batch is None:
            skipped += 1
            continue

        x = batch["x"].to(device, non_blocking=True).to(torch.float32)
        if args.dataset.shared_dataset_kwargs.windows:
            x = batch_windowing(args, x)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            out = backbone(x, batch=batch)

        if vision_projector is not None:
            # zero-shot VL-CABS 와 같은 토큰 표현: proj -> per-token L2 -> 그 위에 고정 maxpool.
            # pretrain 의 CustomTextCLIP_LAPA.encode_image 가
            # F.normalize(vision_projector(visual(x))) 이므로 projection 이 먼저다.
            # maxpool 과 linear 는 교환되지 않으므로 pooled(384-d)에 proj 를 걸면 안 된다
            # (그건 head 앞의 선형변환이라 MLP 가 흡수해 버려 아무 정보도 추가하지 않는다).
            tok = rearrange(out["activ"].float(), "b c d h w -> b (d h w) c")
            tok = F.normalize(vision_projector(tok), p=2, dim=-1)
            pooled_proj = tok.transpose(1, 2).amax(dim=2)
            out = dict(out)
            out["pooled_proj"] = F.normalize(pooled_proj, p=2, dim=-1)

        missing = [k for k in save_keys if k not in out]
        if missing:
            raise KeyError(
                f"backbone {type(backbone).__name__}의 출력에 {missing} 키가 없습니다. "
                f"사용 가능한 키: {sorted(out.keys())}"
            )

        # 첫 배치에서 실제 shape을 보고 예상 용량을 점검한다 — 쓰기 시작 전에 중단하기 위함.
        if not size_checked:
            logger.info(f"[{split}] 캐시 예상 용량 (dtype={cache_dtype}):")
            for k in save_keys:
                t = maybe_pool_activ(out[k], activ_pool) if k == "activ" else out[k]
                per_sample = t[0].numel() * torch.empty(0, dtype=cache_dtype).element_size()
                logger.info(f"    {k}: shape/sample {tuple(t.shape[1:])}")
                check_projected_size(k, per_sample, len(dataset), force_size)
            size_checked = True

        for k in save_keys:
            t = out[k]
            if k == "activ":
                t = maybe_pool_activ(t, activ_pool)
            buffers[k].append(t.detach().to(cache_dtype).cpu())

        labels.append(batch["label"].detach().float().cpu())
        sample_names.extend(batch["sample_name"])

    if skipped:
        logger.warning(f"[{split}] skipped {skipped} fully-empty batch(es) (all samples in the batch failed to load)")

    if not labels:
        raise RuntimeError(f"[{split}] no features extracted — every batch was empty or the dataset is empty.")

    result = {"labels": torch.cat(labels, dim=0), "sample_names": sample_names}
    for k in save_keys:
        result[k] = torch.cat(buffers[k], dim=0)
        logger.info(f"[{split}] {k}: {tuple(result[k].shape)} ({result[k].dtype})")

    # 하위 호환: 기존 소비자(train_abn_probe.py, build_cls_report_templates.py)가 읽는
    # "features" 키는 pooled를 가리키도록 유지한다.
    if "pooled" in result:
        result["features"] = result["pooled"]

    if activ_pool and "activ" in save_keys:
        result["activ_pool_factor"] = activ_pool

    logger.info(f"[{split}] extracted {result['labels'].shape[0]} samples")
    return result


def main():
    args, extract_args = parse_extract_args()
    device = torch.device(extract_args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(extract_args.output_dir) if extract_args.output_dir else Path(extract_args.cache_root) / extract_args.tag

    logger.info("=" * 70)
    logger.info(f"tag={extract_args.tag}  backbone={args.model.kwargs.backbone_model_type}  device={device}")
    logger.info(f"output_dir={output_dir}")
    logger.info("=" * 70)

    backbone = build_backbone(args, device)

    vision_projector = None
    if extract_args.vision_projector:
        ckpt_path, ckpt_prefix = resolve_checkpoint_path(args)
        vision_projector = build_vision_projector_from_checkpoint(
            ckpt_path, device=device, prefix=(ckpt_prefix or None))
        logger.info(f"vision_projector: {vision_projector.weight.shape[1]} -> "
                    f"{vision_projector.weight.shape[0]} (pooled_proj 키를 만듭니다)")

    output_dir.mkdir(parents=True, exist_ok=True)

    save_keys = tuple(k.strip() for k in extract_args.save.split(",") if k.strip())
    if "pooled_proj" in save_keys and vision_projector is None:
        raise SystemExit("--save 에 pooled_proj 를 넣으려면 --vision-projector 도 주어야 합니다.")
    if "activ" in save_keys and extract_args.activ_pool is None:
        raise SystemExit(
            "--save에 activ를 넣으려면 --activ-pool을 반드시 지정해야 합니다. 풀 해상도 activ는 "
            "샘플당 72~144 MB(인코더의 patch_size에 따라 32x32x32 또는 32x32x64 토큰 x 1152채널)라 "
            "train split이 1.6~3.3 TB가 됩니다. 예: --activ-pool 4 --cache-dtype float16 "
            "(공간 64배 축소). 풀 해상도가 정말 필요하면 --activ-pool 1 --force-size."
        )
    cache_dtype = getattr(torch, extract_args.cache_dtype)
    logger.info(f"저장할 키: {save_keys} | activ_pool={extract_args.activ_pool} | dtype={cache_dtype}")

    for split in extract_args.splits.split(","):
        split = split.strip()
        result = extract_split(
            args, backbone, split, device, extract_args.batch_size, extract_args.max_samples,
            save_keys=save_keys, activ_pool=extract_args.activ_pool,
            cache_dtype=cache_dtype, force_size=extract_args.force_size,
            vision_projector=vision_projector,
        )
        out_path = output_dir / f"{split}.pt"
        torch.save(result, out_path)
        logger.info(f"[{extract_args.tag}/{split}] saved to {out_path}")

    logger.info(f"Extraction completed for tag='{extract_args.tag}'!")


if __name__ == "__main__":
    main()
