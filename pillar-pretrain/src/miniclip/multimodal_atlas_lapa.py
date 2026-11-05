"""Vision tower for the `lapa` variant.

Two changes against `MultiModalAtlas_PhraseAlign`, both taken from the reference stage-1
implementation (`ct-rate-rrg/stage1_phrase_align_ct`):

1. **Corrected patch-embed axis order.** Uses `ChestCTEmbed3D_Pillar0`, whose strides are
   derived from `patch_size` and permuted (H, W, D) -> (D, H, W) once. The released
   Pillar0-ChestCT stem hard-codes `ds1 stride=(2,2,1)`, which puts the anisotropic factor
   on W instead of D; see that class's docstring. With the fix, a 256^3 volume at
   patch (8, 8, 4) produces a (D, H, W) = (64, 32, 32) feature map -- exactly what
   `prepare_multiscale_layout` computes -- and the native scales come out at
   65536 / 1024 / 16 tokens x 384 dim.

2. **One scale, not all three.** `MultiModalAtlas_PhraseAlign` interpolates every scale to a
   common grid with `mode="nearest"` and channel-concatenates, producing 32768 tokens of
   width 3*384. That is expensive (VL-CABS softmaxes over the token axis for every
   (image, phrase) pair) and the interpolate call also mixes up axis order -- `res` is
   built in (H, W, D) order while the tensor it resizes is (D, H, W). Here the tower simply
   returns scale `scale_index` (default 1 = 1024 tokens), which is the same readout the
   reference uses and what a published Pillar0 report-generation paper calls h^(2). The
   64x token reduction is what buys the larger batch.
"""

import logging
import math

import torch.nn as nn
from einops import rearrange

from .atlas_encoders import (
    AbdomenCTEmbed3D,
    BMRMultiConvEmbed3D,
    BrainCTMultiConvEmbed3D,
    ChestCTEmbed3D_Pillar0,
    ConvCXREmbed,
    MultiViewConvCXR,
    ProstateMRMultiConvEmbed3D,
)
from .multimodal_atlas_phrase_align import MultiModalAtlas_PhraseAlign

logger = logging.getLogger(__name__)


class MultiModalAtlas_LAPA(MultiModalAtlas_PhraseAlign):
    def __init__(self, *args, scale_index: int = 1, **kwargs):
        # scale_index has to be stashed before super().__init__ runs the _init_* hooks.
        self.scale_index = int(scale_index)
        super().__init__(*args, **kwargs)

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

            logging.info(
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
        modality, image = list(x.items())[0]
        bsz = image.shape[0]

        x, _ = self.build_scales({modality: image}, batch)

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
        # Scales with more than one attention window come back as (B*NW, K, C); flatten the
        # window axis back into the token axis so downstream always sees (B, N, C).
        if math.prod(grid_sizes[self.scale_index]) > 1 and tokens.dim() == 3:
            tokens = rearrange(tokens, "(b nw) k c -> b (nw k) c", b=bsz)
        return tokens
