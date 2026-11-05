"""Stage A' — the CTRG-3D vision path, run once per volume.

WHAT THIS IS FOR
----------------
The ensemble draws 10 reports per exam. Every draw shares the same visual
input, because GRPO trained LoRA adapters on ``lang_model`` only: the vision
encoder, the PerceiverResampler and the projection ``fc`` are frozen SFT
weights identical across all ten. Running them once per volume instead of once
per draw turns the marginal cost of a draw into pure LLM generation.

The cached tensor is exactly what ``MyEmbedding.forward`` would have handed the
LLM — ``fc(perceiver(activ))``, shape (128, 4096) — which is also the layout
the training-side offline cache uses
(``PETRG-3D/data/vision_cache/newencB_ep15_ckpt7455/valid.npy``:
[N, 128, 4096] float16, 1 MB per volume). That makes the two directly
comparable, which is how this wiring gets verified.

WHY IT DOES NOT REUSE STAGE A's ENCODER OUTPUT
----------------------------------------------
Stage A runs pillar-finetune's port (fp32 parameters under autocast(bf16));
CTRG-3D was trained with PETRG-3D's port, materialised in bf16 by CTRG3D's
``.bfloat16()`` cast, and its ``multimodal_msa`` builds the relative-coordinate
table in bf16 rather than fp32. Same shapes, same weights, different numerics —
feeding one tower's ``activ`` to the other's Perceiver would load cleanly and
be quietly wrong. So this module owns a second tower, and both live on the GPU
at once (~1.5 GB together) so the volume is still read only once.

The weights come from the CTRG-3D checkpoint itself
(``embedding_layer.vision_encoder.* / .perceiver.* / .fc.*``, split out by
make_weights.sh) rather than from a standalone encoder file — that way "is this
the encoder the checkpoint was trained with?" is true by construction.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from einops import rearrange

VISION_WEIGHTS_NAME = "ctrg3d/vision.safetensors"

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
ENCODER_YAML = (VENDOR_DIR / "petrg" / "Model"
                / "ctrate_pillar0_chest_lapa.yaml")

NUM_VISUAL_TOKENS = 128
VIS_DIM = 384                 # single scale, native resolution
HIDDEN = 4096                 # Qwen3-8B embedding width

_PREFIX_ENC = "embedding_layer.vision_encoder."
_PREFIX_PER = "embedding_layer.perceiver."
_PREFIX_FC = "embedding_layer.fc."


class CTRG3DVisionPath(nn.Module):
    """vision_encoder -> perceiver -> fc, i.e. MyEmbedding's visual branch."""

    def __init__(self, perceiver_num: int = NUM_VISUAL_TOKENS,
                 vis_dim: int = VIS_DIM, hidden: int = HIDDEN):
        super().__init__()
        from Model.helpers import PerceiverResampler
        from Model.pillar_mmatlas_lapa import MultimodalAtlasLAPA

        with open(ENCODER_YAML) as f:
            cfg = yaml.safe_load(f)
        # The tensors arrive from the checkpoint; self-loading the pretrain .pt
        # would be both redundant and a second source of truth.
        cfg["vision_cfg"]["model_config"]["pretrained"] = "None"
        model_config = cfg["vision_cfg"]["model_config"]
        atlas_args = model_config["model"]["atlas_args"]

        self.vision_encoder = MultimodalAtlasLAPA(
            scale_index=int(cfg["vision_cfg"].get("scale_index", 1)),
            expect_tokens=int(cfg["vision_cfg"].get("expect_tokens", 0)),
            embed_dim=atlas_args["embed_dim"],
            num_classes=atlas_args["num_classes"],
            image_size=model_config["modalities"]["chest_ct"]["image_size"],
            multiscale_feats=True,
            model_config=model_config,
            device="cuda",
            pretrained="None",
        )
        self.perceiver = PerceiverResampler(dim=vis_dim,
                                            num_latents=perceiver_num)
        self.fc = nn.Linear(vis_dim, hidden)

    @torch.no_grad()
    def forward(self, ct_image: torch.Tensor) -> torch.Tensor:
        """(1, 1, D, H, W) HU volume -> (1, 128, 4096).

        Line for line what ``MyEmbedding.forward`` +
        ``MyEmbedding._encode_volume`` do, minus the text branch.
        """
        import rve

        x = rve.batch_apply_windowing_vectorized(ct_image, windows="all",
                                                 modality="CT")
        x = x.to(torch.bfloat16)
        out = self.vision_encoder(x)
        feat = rearrange(out["activ"], "b c d h w -> b (d h w) c")
        feat = feat.unsqueeze(1).unsqueeze(1)          # (B, 1, 1, N, C)
        feat = self.perceiver(feat).squeeze(1)         # (B, 128, vis_dim)
        return self.fc(feat)                           # (B, 128, hidden)


def load_vision_path(weights_dir: str | Path,
                     num_visual_tokens: int = NUM_VISUAL_TOKENS,
                     device: str = "cuda") -> CTRG3DVisionPath:
    """Build the vision path and strict-load it from the split-out tensors."""
    from safetensors.torch import load_file as load_safetensors

    path = Path(weights_dir) / VISION_WEIGHTS_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — make_weights.sh splits the vision tensors out "
            f"of the CTRG-3D checkpoint; re-run it")

    module = CTRG3DVisionPath(perceiver_num=num_visual_tokens).bfloat16()

    raw = load_safetensors(str(path), device="cpu")
    state = {}
    for k, v in raw.items():
        if k.startswith(_PREFIX_ENC):
            state["vision_encoder." + k[len(_PREFIX_ENC):]] = v
        elif k.startswith(_PREFIX_PER):
            state["perceiver." + k[len(_PREFIX_PER):]] = v
        elif k.startswith(_PREFIX_FC):
            state["fc." + k[len(_PREFIX_FC):]] = v
        else:
            raise RuntimeError(f"{path}: unexpected tensor {k!r}")
    module.load_state_dict(state, strict=True)

    module = module.to(device)
    # MultimodalAtlas overrides .eval() as a no-op in some variants, and
    # DropPath 0.3 would fire in train mode; use train(False) and verify.
    module.train(False)
    assert not any(m.training for m in module.modules()), \
        "vision path still has modules in training mode"
    for p in module.parameters():
        p.requires_grad_(False)
    return module


@torch.no_grad()
def encode_visual(vision_path: CTRG3DVisionPath, volume_tensor: torch.Tensor,
                  device: str = "cuda") -> torch.Tensor:
    """(1, D, H, W) fp32 HU tensor -> (128, 4096) float16 CPU tensor.

    float16 matches the training-side cache's dtype
    (``valid_meta.json``: "dtype": "float16"), so the verification diff is
    against like for like.
    """
    ct = volume_tensor.unsqueeze(0).to(torch.bfloat16).to(device)   # (1,1,D,H,W)
    emb = vision_path(ct)
    if tuple(emb.shape[1:]) != (NUM_VISUAL_TOKENS, HIDDEN):
        raise RuntimeError(
            f"visual embedding {tuple(emb.shape)} != (1, {NUM_VISUAL_TOKENS}, "
            f"{HIDDEN}); NUM_VISUAL_TOKENS and the checkpoint disagree")
    return emb[0].detach().to(torch.float16).cpu()
