"""Backbone for `lapa` self-pretrained checkpoints (e.g.
logs/lapa_pillar0_lz4_v1). Reuses the validated MultimodalAtlasSelf body
(mmatlas_self.py) and applies the two architecture changes that pretrain variant made —
both must be mirrored here or the frozen encoder's features are not the ones the
checkpoint was trained to produce (see pillar-pretrain
src/miniclip/multimodal_atlas_lapa.py, which this class deliberately parallels):

1. **Stride-corrected patch embed** (`ChestCTEmbed3D_Pillar0`): strides derived from
   `patch_size` and permuted (H, W, D) -> (D, H, W) once. With patch (8, 8, 4) a 256^3
   volume produces a (D, H, W) = (64, 32, 32) feature map — exactly what
   prepare_multiscale_layout computes — instead of the released stem's transposed grid.
   Parameter names/shapes are identical to ChestCTEmbed3D, so strict checkpoint loading
   still verifies every tensor.

2. **Single-scale ("intermediate hidden") readout.** The pretrain tower returns ONLY
   scale `scale_index` (default 1 = 1024 tokens x atlas_dim 384, what a published
   Pillar0 report-generation paper calls h^(2)) — it never concatenates the three
   scales. So here:
     - `pooled`  = L2-normalized maxpool over that one scale's tokens -> 384-dim
                   (the same fixed-maxpool+normalize recipe as the winning readout,
                   applied to the representation this encoder was actually trained on;
                   NOT the 3-scale 1152-dim concat other self backbones expose).
     - `activ`   = that scale's tokens reshaped to their (B, C, D, H, W) grid
                   (16x8x8 for scale 1), no cross-scale interpolation.
     - `hidden_dim` = atlas_args["dim"] (384), not dim * num_scales.

   The checkpoint's `vision_projector` (384->1152, CLIP text-alignment head living
   OUTSIDE `visual.`) is intentionally NOT loaded: this backbone is for encoder-quality
   comparisons, and the other maxpool/attnpool runs likewise use the encoder's native
   feature, not the text-projected one.
"""

import math

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .logging_shim import logger

from .atlas_encoders import (
    AbdomenCTEmbed3D,
    BMRMultiConvEmbed3D,
    BrainCTMultiConvEmbed3D,
    ChestCTEmbed3D_Pillar0,
    ConvCXREmbed,
    MultiViewConvCXR,
    ProstateMRMultiConvEmbed3D,
)
from .mmatlas_self import MultimodalAtlas as _MultimodalAtlasSelf


class MultimodalAtlasSelfLAPA(_MultimodalAtlasSelf):
    def __init__(self, args, scale_index: int = 1, **kwargs):
        # Plain-int attribute; safe to stash before nn.Module.__init__ (mirrors the
        # pretrain tower, where _init_* hooks run inside super().__init__).
        self.scale_index = int(scale_index)
        super().__init__(args, **kwargs)

        # Single-scale readout: hidden_dim is one scale's width, not dim * num_scales.
        # Must be overridden AFTER super().__init__ (which sets the 3-scale value and
        # then loads the checkpoint — loading is unaffected by this attribute).
        self.hidden_dim = self.model_config["model"]["atlas_args"]["dim"]

    def _init_patch_embeds(self):
        modalities = self.model_config["modalities"]
        MODALITY_TO_PATCH_EMBED = {
            "chest_xray_single_view": ConvCXREmbed,
            "chest_xray_two_view": MultiViewConvCXR,
            # the only difference from the parent: the stride-corrected stem
            "chest_ct": ChestCTEmbed3D_Pillar0,
            "abdomen_ct": AbdomenCTEmbed3D,
            "breast_mr": BMRMultiConvEmbed3D,
            "brain_ct": BrainCTMultiConvEmbed3D,
            "prostate_mr": ProstateMRMultiConvEmbed3D,
        }

        self.patch_embeds = nn.ModuleDict()
        for modality in modalities:
            image_size = modalities[modality].get("image_size", self.image_size)
            patch_size = modalities[modality].get("patch_size", 16)
            in_chans = modalities[modality].get("in_chans", 1)
            num_features = modalities[modality].get("num_features", 192)

            logger.info(
                f"[lapa] patch embed for {modality}: image_size={image_size}, "
                f"patch_size(H,W,D)={patch_size}, in_chans={in_chans}, num_features={num_features}"
            )

            self.patch_embeds[modality] = MODALITY_TO_PATCH_EMBED[modality](
                vol_size=image_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=num_features,
            )

    def forward(self, x, batch=None):
        modality, image = batch["anatomy"][0], x
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

        if not (0 <= self.scale_index < len(multiscale_feats)):
            raise IndexError(
                f"scale_index={self.scale_index} but this configuration produced "
                f"{len(multiscale_feats)} scales."
            )

        tokens = multiscale_feats[self.scale_index]
        # Scales with more than one attention window come back as (B*NW, K, C); flatten
        # the window axis back into the token axis so downstream always sees (B, N, C) —
        # same rearrange the pretrain tower and the parent's forward use.
        if math.prod(grid_sizes[self.scale_index]) > 1 and tokens.dim() == 3:
            tokens = rearrange(tokens, "(b nw) k c -> b (nw k) c", b=bsz)

        pooled = self.maxpool(tokens.transpose(1, 2)).squeeze(2)  # (B, C)

        layout = multiscale_layout[self.scale_index]
        m0, m1, m2 = layout["window_dims"]
        g0, g1, g2 = layout["grid_size"]
        activ = rearrange(
            tokens,
            "b (g2 g0 g1 m2 m0 m1) c -> b c (g2 m2) (g0 m0) (g1 m1)",
            g2=g2, g1=g1, g0=g0, m2=m2, m1=m1, m0=m0,
        )

        return {
            "activ": activ,
            "pooled": F.normalize(pooled, dim=-1),
        }
