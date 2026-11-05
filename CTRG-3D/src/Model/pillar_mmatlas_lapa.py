"""Vision tower for the ``lapa`` encoder.

Ported from ``pillar-pretrain/src/miniclip/multimodal_atlas_lapa.py``, the
implementation the checkpoint at
``pillar-pretrain/logs/lapa_pillar0_lz4_v1/checkpoints/epoch_12.pt`` was
pretrained with (that run continued from ``checkpoints/Pillar0-ChestCT``; see its
``params.txt``). Kept in its own module so ``pillar_mmatlas.py`` stays exactly as
the old arms saw it.

TWO CHANGES AGAINST ``MultimodalAtlas``, BOTH BUG FIXES
-------------------------------------------------------
1. **Corrected patch-embed axis order** -- ``ChestCTEmbed3D_Pillar0`` instead of
   ``ChestCTEmbed3D``. See ``atlas_encoders_pillar0.py``: the old stride puts the
   anisotropic factor on W instead of D, so the token grid the attention windows
   are laid out on is transposed relative to the feature map. Weight shapes are
   identical, which is why it never crashed.

2. **One scale at native resolution, not three interpolated and concatenated.**
   ``MultimodalAtlas.forward`` upsamples every scale to a common grid with
   ``mode="nearest"`` and channel-concatenates -> 65,536 tokens x 3*384 = 1152.
   Two problems: it is expensive, and the interpolate call mixes up axis order --
   ``res`` is built ``[image_size[i] // patch_size[i]]`` in (H, W, D) order while
   the tensor it resizes is (D, H, W). This tower returns scale ``scale_index``
   (default 1 = 1,024 tokens x 384) at its native grid, which is the readout the
   reference uses and what the Pillar0 report-generation paper calls h^(2).

The 64x token reduction is the point: the PerceiverResampler in
``pillar_embedding_layer`` cross-attends over the token axis, so its cost drops with
it.

WHAT THIS RETURNS, AND WHY IT IS A 5-D ``activ``
------------------------------------------------
The reference returns ``(B, N, C)`` tokens directly. This returns the same tokens
reshaped to ``(B, C, d, h, w)`` under the key ``activ``, because that is what
``MyEmbedding._encode_volume`` already consumes:

    feat = rearrange(outputs['activ'], 'b c d h w -> b (d h w) c')

Going out as 5-D and straight back to (B, N, C) leaves ``_encode_volume``
untouched -- no second opt-in branch in a file the old arms share. The round trip
permutes the token ORDER relative to the reference's ``(b nw) k c -> b (nw k) c``
flatten, which is harmless here: the Perceiver adds no positional encoding to its
input tokens, so its output is invariant to their order. If a positional input
embedding is ever added to the Perceiver, this note stops being true and the
ordering has to be matched to the reference.
"""

import logging
import math

import torch
import torch.nn as nn
from einops import rearrange

from .atlas_encoders import (AbdomenCTEmbed3D, BMRMultiConvEmbed3D,
                             BrainCTMultiConvEmbed3D, ConvCXREmbed,
                             MultiViewConvCXR, ProstateMRMultiConvEmbed3D)
from .atlas_encoders_pillar0 import ChestCTEmbed3D_Pillar0
from .pillar_mmatlas import MultimodalAtlas

logger = logging.getLogger(__name__)


class MultimodalAtlasLAPA(MultimodalAtlas):
    """``MultimodalAtlas`` with the corrected stem and a single-scale readout."""

    #: ChestCTEmbed3D_Pillar0 derives ds1's stride on the (D, H, W) axis order, so
    #: the parent's patch-grid guard has to state its expectation the same way.
    #: Without this the guard rejects the corrected map: it computed
    #: [32, 32, 64] while the (fixed) stem produces [64, 32, 32].
    stem_axis_order_corrected = True

    def __init__(self, *args, scale_index: int = 1, expect_tokens: int = 0, **kwargs):
        # Stashed before super().__init__ because that runs _init_patch_embeds().
        self.scale_index = int(scale_index)
        # A wrong scale_index yields a VALID shape (65,536 or 16 tokens instead of
        # 1,024), so nothing would crash -- it would just train on the wrong
        # readout. Declaring the expected count turns that into a first-forward
        # failure. 0 disables the check.
        self.expect_tokens = int(expect_tokens)
        self._logged_shape = False
        super().__init__(*args, **kwargs)

    def _init_patch_embeds(self):
        """Same as the parent, except chest_ct uses the corrected stem."""
        MODALITY_TO_PATCH_EMBED = {
            "chest_xray": MultiViewConvCXR,
            "chest_xray_single_view": ConvCXREmbed,
            "chest_ct": ChestCTEmbed3D_Pillar0,      # <-- the one substitution
            "abdomen_ct": AbdomenCTEmbed3D,
            "breast_mr": BMRMultiConvEmbed3D,
            "brain_ct": BrainCTMultiConvEmbed3D,
            "prostate_mr": ProstateMRMultiConvEmbed3D,
        }
        modalities = self.model_config["modalities"]
        self.patch_embeds = nn.ModuleDict()
        for modality in modalities:
            cfg = modalities[modality]
            image_size = cfg.get("image_size", self.image_size)
            patch_size = cfg.get("patch_size", 16)
            in_chans = cfg.get("in_chans", 1)
            num_features = cfg.get("num_features", 192)
            logger.info(
                f"[lapa] patch embed for {modality}: image_size={image_size}, "
                f"patch_size(H,W,D)={patch_size}, in_chans={in_chans}, "
                f"num_features={num_features}")
            self.patch_embeds[modality] = MODALITY_TO_PATCH_EMBED[modality](
                vol_size=image_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=num_features,
            )

    def forward(self, x, modality="chest_ct"):
        image = x
        bsz = image.shape[0]
        x, _ = self.build_scales(image, modality)

        modal_config = self.model_config["modalities"][modality]
        multiscale_layout = self.prepare_multiscale_layout(
            img_size=modal_config["image_size"],
            merge_ratio=modal_config["merge_ratio"],
            local2global=modal_config["local2global"],
            patch_size=modal_config["patch_size"],
        )
        grid_sizes = [layout["grid_size"] for layout in multiscale_layout]

        multiscale_feats = []
        for level, atlas_model in enumerate(self.atlas_models):
            is_last = len(x) == 1 or level == len(self.atlas_models) - 1
            x = atlas_model(
                x,
                grid_sizes=grid_sizes[level:],
                multiscale_layout=multiscale_layout[level:],
                merge_ratio=modal_config["merge_ratio"],
                local2global=modal_config["local2global"],
                modality=modality,
            )
            if not is_last:
                multiscale_feats.append(x[0])
                x = x[1:]
        multiscale_feats.append(x[0])

        if not 0 <= self.scale_index < len(multiscale_feats):
            raise IndexError(
                f"scale_index={self.scale_index} but this configuration produced "
                f"{len(multiscale_feats)} scales")

        idx = self.scale_index
        layout = multiscale_layout[idx]
        tokens = multiscale_feats[idx]

        # Scales with more than one attention window come back as (B*NW, K, C).
        if math.prod(grid_sizes[idx]) > 1 and tokens.dim() == 3:
            tokens = rearrange(tokens, "(b nw) k c -> b (nw k) c", b=bsz)

        m0, m1, m2 = layout["window_dims"]
        g0, g1, g2 = layout["grid_size"]
        expected = g0 * g1 * g2 * m0 * m1 * m2
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"scale {idx} has {tokens.shape[1]} tokens but its layout describes "
                f"{expected} (grid {g0}x{g1}x{g2}, window {m0}x{m1}x{m2}); the tower "
                f"and prepare_multiscale_layout disagree")

        activ = rearrange(
            tokens, "b (g2 g0 g1 m2 m0 m1) c -> b c (g2 m2) (g0 m0) (g1 m1)",
            g2=g2, g1=g1, g0=g0, m2=m2, m1=m1, m0=m0)

        if not self._logged_shape:
            self._logged_shape = True
            print(f"INFO: [lapa] scale_index={idx} -> activ {tuple(activ.shape)} "
                  f"= {tokens.shape[1]} tokens x {tokens.shape[2]} dim", flush=True)
        if self.expect_tokens and tokens.shape[1] != self.expect_tokens:
            raise RuntimeError(
                f"scale_index={idx} produced {tokens.shape[1]} tokens but the config "
                f"declared expect_tokens={self.expect_tokens}. A wrong scale trains on "
                f"the wrong readout without crashing, which is why this is checked.")

        pooled = self.maxpool(tokens.transpose(1, 2)).squeeze(2)
        return {
            "activ": activ,
            "pooled": nn.functional.normalize(pooled, dim=-1),
        }
