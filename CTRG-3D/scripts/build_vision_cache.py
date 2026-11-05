"""Precompute the frozen visual features the GRPO arm trains against.

    uv run --no-project --python .venv/bin/python python scripts/build_vision_cache.py \
        --split valid --limit 64        # smoke first, then drop --limit

Writes ``[N, image_num, hidden]`` float16 plus an index, one file pair per split:

    data/vision_cache/<ckpt_tag>/<split>.npy          memory-mappable
    data/vision_cache/<ckpt_tag>/<split>_index.json   {sample_name: row}
    data/vision_cache/<ckpt_tag>/<split>_meta.json    provenance + timing

WHAT IS BEING FROZEN, AND WHY THAT IS A CHOICE
----------------------------------------------
The cached tensor is the output of ``fc(perceiver(atlas(volume)))`` -- i.e. the
whole visual path up to the point where features enter the language model. RL
then trains the LoRA adapter only, so the Perceiver and ``fc`` are frozen for
the duration.

That is NOT free: ``training_stage="joint"`` leaves the Perceiver and ``fc``
trainable (``_setup_trainable_params`` does nothing in that branch; only the
Atlas encoder is frozen, via ``finetune_ct_feature_extractor=False``). Caching
after ``fc`` therefore removes them from the optimisation. It is the standard
configuration for multimodal RL post-training -- a scalar reward per rollout is
a poor signal for a perception module, and freezing is what makes precomputation
possible at all -- but it means RL can only change what the model SAYS about a
volume, never what it SEES. Caching before the Perceiver instead would preserve
that, at 151 MB/volume (3.6 TB) rather than 1 MB/volume.

THE CACHE IS BOUND TO ONE CHECKPOINT
------------------------------------
Perceiver and ``fc`` weights come from ``--ckpt``, so features computed from one
checkpoint are meaningless for another. The checkpoint's own tag goes in the
output directory name and its sha-ish identity in ``*_meta.json``; the GRPO run
re-checks both before it starts. Point the RL at a different starting policy and
the cache must be rebuilt.

Crops are CENTER crops (``is_train=False``), matching how the eval path produced
the numbers we compare against. The cost, stated because it is real: the run
loses the random-crop augmentation the SFT path used -- one fixed view per
volume for the whole of RL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

DEFAULTS = {
    "train": dict(
        json_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_train_filtered_prob_half.json",
        rve_manifest="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/train/mapping_path.csv",
        cls_template="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_ABN_cls_report_templates_p90n20/cls_report_train.json",
        ret_template="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden/train_ret_reports.json",
    ),
    "valid": dict(
        json_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_valid_filtered_prob_half.json",
        rve_manifest="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/valid/mapping_path.csv",
        cls_template="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_ABN_cls_report_templates_p90n20/cls_report_dev.json",
        ret_template="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden/valid_ret_reports.json",
    ),
}
DEFAULT_CKPT = (
    "/data/drdcad/sunggu/0.Challenge/PETRG-3D/scripts/outputs/ep15/"
    "ctrg3d-qwen3-8B-pillar0os-lz4-vt128/checkpoint-16412"
)
DEFAULT_LANG = "/data/drdcad/sunggu/llm_models/Qwen3-8B"
DEFAULT_YAML = str(_REPO / "src/Model/ctrate_pillar0_chest_opensource.yaml")


def resolve_ckpt_file(path: str) -> str:
    """Accept either a checkpoint file or a Trainer checkpoint directory."""
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin", "mp_rank_00_model_states.pt"):
            cand = p / name
            if cand.is_file():
                return str(cand)
        shards = sorted(p.glob("*.safetensors"))
        if shards:
            return str(shards[0])
    raise FileNotFoundError(
        f"no loadable checkpoint at {path}; expected a .safetensors/.bin file or a "
        f"directory containing one")


@torch.no_grad()
def encode_batch(embedding_layer, vision_x: dict) -> torch.Tensor:
    """The visual half of MyEmbedding.forward, and nothing else.

    Deliberately mirrors ``pillar_embedding_layer.MyEmbedding.forward`` line for
    line rather than calling it: that method also builds the text embedding
    table and validates token ids, neither of which exists here. If that method
    changes, this must change with it -- the shape assertion below is the guard.
    """
    import rve

    ct = vision_x["ct_image"]
    ct = rve.batch_apply_windowing_vectorized(ct, windows="all", modality="CT")
    feat = embedding_layer._encode_volume(ct, embedding_layer.perceiver)
    return embedding_layer.fc(feat)  # [B, image_num, embedding_dim]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("train", "valid"), required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--lang_encoder_path", default=DEFAULT_LANG)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--model_config_yaml", default=DEFAULT_YAML)
    ap.add_argument("--num_visual_tokens", type=int, default=128)
    ap.add_argument("--vis_dim", type=int, default=1152)
    ap.add_argument("--max_seq", type=int, default=2688)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_root", default=str(_REPO / "data/vision_cache"))
    ap.add_argument("--limit", type=int, default=0, help="smoke run: first N volumes only")
    ap.add_argument("--json_path", default=None)
    ap.add_argument("--rve_manifest", default=None)
    args = ap.parse_args()

    d = DEFAULTS[args.split]
    json_path = args.json_path or d["json_path"]
    rve_manifest = args.rve_manifest or d["rve_manifest"]
    tokenizer_path = args.tokenizer_path or args.lang_encoder_path
    ckpt_file = resolve_ckpt_file(args.ckpt)
    ckpt_tag = Path(args.ckpt).name if Path(args.ckpt).is_dir() else Path(args.ckpt).stem
    out_dir = Path(args.out_root) / ckpt_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file as load_safetensors
    from torch.utils.data import DataLoader

    from Dataset.ctrate_dataset_test import CTRATEDataset_Test
    from Model.CTRG_3D import CTRG3D

    print(f"[cfg] split={args.split}  ckpt={ckpt_file}\n      out={out_dir}", flush=True)

    ds = CTRATEDataset_Test(
        tokenizer_path=tokenizer_path,
        json_path=json_path,
        rve_manifest_path=rve_manifest,
        cls_template_path=d["cls_template"],
        ret_template_path=d["ret_template"],
        image_num=args.num_visual_tokens,
        max_seq=args.max_seq,
        use_fast=False,
        use_cls_template=True,
        use_ret_template=True,
    )
    n_total = len(ds)
    n = min(args.limit, n_total) if args.limit else n_total
    print(f"[data] {n} of {n_total} volumes", flush=True)

    model = CTRG3D(
        text_tokenizer_path=tokenizer_path,
        lang_model_path=args.lang_encoder_path,
        pretrained_visual_encoder="",
        pretrained_adapter="",
        use_fast=False,
        enable_thinking=False,
        model_config_yaml=args.model_config_yaml,
        vis_dim=args.vis_dim,
        image_num=args.num_visual_tokens,
    ).bfloat16()

    ckpt = (load_safetensors(ckpt_file, device="cpu") if ckpt_file.endswith(".safetensors")
            else torch.load(ckpt_file, map_location="cpu", weights_only=False))
    if "model" in ckpt:
        ckpt = ckpt["model"]
    # strict=True on purpose: a silently partial load would leave the Perceiver
    # at its random init and produce a cache of noise that trains perfectly
    # happily and scores like garbage.
    model.load_state_dict(ckpt, strict=True)
    model = model.cuda().eval()
    emb = model.embedding_layer
    hidden = emb.fc.out_features
    print(f"[model] loaded; caching [{args.num_visual_tokens}, {hidden}] per volume", flush=True)

    loader = DataLoader(
        torch.utils.data.Subset(ds, list(range(n))) if args.limit else ds,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda b: b, drop_last=False,
    )

    suffix = f"{args.split}_smoke{n}" if args.limit else args.split
    npy_path = out_dir / f"{suffix}.npy"
    store = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float16,
        shape=(n, args.num_visual_tokens, hidden))
    index: "dict[str, int]" = {}

    row = 0
    t0 = time.time()
    t_first100 = None
    for batch in loader:
        names = [s["sample_name"] for s in batch]
        vision = {"ct_image": torch.stack([s["vision_x"]["ct_image"] for s in batch]).cuda()}
        out = encode_batch(emb, vision).float()
        if out.shape[1:] != (args.num_visual_tokens, hidden):
            raise RuntimeError(
                f"encoder returned {tuple(out.shape)}, expected [B, {args.num_visual_tokens}, "
                f"{hidden}] -- MyEmbedding.forward has changed and encode_batch has not")
        peak = out.abs().max().item()
        if peak > 60000:
            raise RuntimeError(f"|feature| max {peak:.1f} overflows float16 storage")
        store[row:row + out.shape[0]] = out.cpu().numpy().astype(np.float16)
        for k, name in enumerate(names):
            if name in index:
                raise RuntimeError(f"duplicate sample_name {name}; the index would be ambiguous")
            index[name] = row + k
        row += out.shape[0]
        if row >= 100 and t_first100 is None:
            t_first100 = time.time() - t0
            print(f"[timing] first {row} volumes in {t_first100:.1f}s "
                  f"= {t_first100/row:.3f}s/volume -> {t_first100/row*n_total/3600:.2f}h "
                  f"for the full {n_total}", flush=True)
        if row % 1000 < args.batch_size:
            el = time.time() - t0
            print(f"  {row}/{n}  {el/60:.1f} min  eta {(el/row*(n-row))/60:.1f} min", flush=True)

    store.flush()
    if row != n:
        raise RuntimeError(f"wrote {row} rows but allocated {n}")

    (out_dir / f"{suffix}_index.json").write_text(json.dumps(index))
    meta = {
        "split": args.split, "n": n, "n_total": n_total,
        "checkpoint": ckpt_file, "checkpoint_tag": ckpt_tag,
        "checkpoint_mtime": os.path.getmtime(ckpt_file),
        "json_path": json_path, "rve_manifest": rve_manifest,
        "num_visual_tokens": args.num_visual_tokens, "hidden": hidden,
        "dtype": "float16", "crop": "center",
        "seconds": round(time.time() - t0, 1),
        "seconds_per_volume": round((time.time() - t0) / max(1, row), 4),
    }
    (out_dir / f"{suffix}_meta.json").write_text(json.dumps(meta, indent=2))

    gb = npy_path.stat().st_size / 1e9
    print(f"\n[out] {npy_path}  ({gb:.1f} GB, {row} rows)")
    print(f"[out] {out_dir / (suffix + '_index.json')}")
    print(f"[out] {out_dir / (suffix + '_meta.json')}")
    print(f"[done] {meta['seconds']/60:.1f} min  ({meta['seconds_per_volume']:.3f} s/volume)")


if __name__ == "__main__":
    main()
