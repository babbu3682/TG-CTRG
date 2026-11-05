"""Stage A vision encoder — lapa, pillar-finetune port (fp32 weights).

WHICH PORT, AND WHY IT MATTERS
------------------------------
Two ports of the same tower exist and they are NOT numerically interchangeable:

  * ``pillar_backbone`` (this module)  — pillar-finetune's
    ``pillar/models/backbones/mmatlas_self_lapa.py``. fp32 parameters,
    forward run under ``autocast(bf16)``. This is what produced the abn probe
    cache (``abn_probe_cache/lapa_lz4_half``) and what the retrieval
    head was trained on, so the cls probe and the Simple_TranSQ head are only
    valid on ITS features.
  * ``Model.pillar_mmatlas_lapa`` (see ``visual_cache.py``) — PETRG-3D's
    port, materialised in bf16 because CTRG3D casts the whole model with
    ``.bfloat16()``. That is what the CTRG-3D checkpoint was trained with.

They differ in parameter dtype (fp32-under-autocast vs bf16 weights) and in
``multimodal_msa.py``'s relative-coordinate table (``.float()`` vs
``.to(torch.bfloat16)``). Mixing them loads without error — the shapes are
identical — and silently shifts every feature. So stage A runs BOTH towers,
each feeding only the heads that were trained on it. Two forwards per volume,
one volume read.

Forward recipe mirrors pillar-finetune ``scripts/extract_abn_features.py``:
    float32 HU volume -> 11-ch windowing -> autocast(bf16) -> {"pooled", "activ"}
with ``pooled`` = L2-normalised maxpool over the 1,024 scale-1 tokens (384-d)
and ``activ`` = those tokens on their native (384, 16, 8, 8) grid.
"""
from __future__ import annotations

from pathlib import Path

import torch
from easydict import EasyDict

from .windowing import batch_apply_ct_windowing

ENCODER_CKPT_NAME = "lapa_epoch12.pt"

# Copied VERBATIM from pillar-finetune
# configs/ctrate_abnclass_finetune_lz4_pillar_self_lapa_maxpool.yaml
# (model.kwargs.backbone_kwargs.model_config), which mirrors the pretrain run's
# model_configs json. patch_size is [8, 8, 4], NOT [8, 8, 8].
MODEL_CONFIG = {
    "modalities": {
        "chest_ct": {
            "image_size": [256, 256, 256],
            "in_chans": 11,
            "local2global": [4, 4, 4],
            "merge_ratio": [4, 4, 4],
            "num_features": 192,
            "patch_size": [8, 8, 4],
        }
    },
    "model": {
        "atlas_args": {
            "attn_drop": 0.0,
            "class_token": False,
            "dim": 384,
            "drop": 0.0,
            "drop_path_rate": 0.3,
            "embed_dim": 384,
            "global_pool": "avg",
            "in_chans": 1,
            "in_dim": 64,
            "mlp_ratio": 4.0,
            "num_classes": 0,
            "num_features": 192,
            "num_heads": 6,
            "num_scales": 3,
            "pre_norm": False,
            "qk_scale": False,
            "qkv_bias": True,
        },
        "stages": [2, 2, 8],
    },
}

ACTIV_SHAPE = (384, 16, 8, 8)


def load_encoder(weights_dir: str | Path, device: str = "cuda"):
    """Build MultimodalAtlasSelfLAPA and strict-load lapa epoch_12."""
    from pillar_backbone import MultimodalAtlasSelfLAPA

    ckpt = Path(weights_dir) / ENCODER_CKPT_NAME
    if not ckpt.exists():
        raise FileNotFoundError(f"stage-A encoder checkpoint not found: {ckpt}")

    backbone = MultimodalAtlasSelfLAPA(
        args=EasyDict(),            # only stashed as self.args; never read
        embed_dim=384,
        image_size=256,
        multiscale_feats=True,
        scale_index=1,
        model_config=MODEL_CONFIG,
        device=device,
        pretrained=str(ckpt),
        strict_load=True,
    )

    # setup_model() ends in self.train(); DropPath 0.3 would corrupt features.
    # train(False) rather than .eval() — some Atlas variants override eval() as
    # a no-op — and then verify.
    backbone.train(False)
    for p in backbone.parameters():
        p.requires_grad_(False)
    assert not any(m.training for m in backbone.modules()), \
        "stage-A encoder still has modules in training mode — DropPath would fire"
    return backbone


@torch.no_grad()
def extract_features(encoder, volume_tensor: torch.Tensor,
                     device: str = "cuda") -> dict:
    """(1, D, H, W) float32 HU tensor -> {"pooled": (1,384), "activ": (1,384,16,8,8)}.

    Both come back as fp32 CPU tensors. ``pooled`` feeds the cls probe (the
    training cache's ``features`` column is bit-identical to it) and ``activ``
    feeds Simple_TranSQ at full resolution (``apply_pooling: False`` in the
    retrieval config).
    """
    x = volume_tensor.unsqueeze(0).to(device)          # (1, 1, D, H, W)
    x = batch_apply_ct_windowing(x, "all")             # (1, 11, D, H, W)

    with torch.amp.autocast("cuda", enabled=device.startswith("cuda"),
                            dtype=torch.bfloat16):
        out = encoder(x, batch={"anatomy": ["chest_ct"]})

    activ = out["activ"]
    if tuple(activ.shape[1:]) != ACTIV_SHAPE:
        raise RuntimeError(
            f"stage-A activ grid {tuple(activ.shape[1:])} != {ACTIV_SHAPE}; the "
            f"encoder config or scale_index drifted")
    return {
        "pooled": out["pooled"].detach().float().cpu(),   # (1, 384), ||.|| == 1
        "activ": activ.detach().float().cpu(),            # (1, 384, 16, 8, 8)
    }
