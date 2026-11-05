"""Chest-CT patch embedding with the anisotropic stride on the CORRECT axis.

Ported from the pretrain repo's reference implementation
(``pillar-pretrain/src/miniclip/atlas_encoders.py::ChestCTEmbed3D_Pillar0``), which is
what the ``lapa`` encoder -- ``logs/lapa_pillar0_lz4_v1/checkpoints/
epoch_12.pt`` -- was pretrained with. Kept in its own module so
``atlas_encoders.py`` stays exactly as the old arms saw it.

THE BUG THIS FIXES
------------------
Everything declares ``patch_size`` in **(H, W, D)** order, which is what
``MultimodalAtlas.prepare_multiscale_layout`` assumes:

    H, W, D = img_size;  pH, pW, pD = patch_size;  h, w, d = H//pH, W//pW, D//pD

But the volume tensor is **(B, C, D, H, W)** and ``nn.Conv3d``'s stride tuple is
**(stride_D, stride_H, stride_W)**. ``ChestCTEmbed3D._resolve_ds1_stride`` returns
``tuple(p // 4 for p in patch_size)`` -- i.e. it feeds the (H, W, D)-ordered
patch_size straight into a (D, H, W)-ordered stride. For patch_size (8, 8, 4) that
gives ds1 stride (2, 2, 1): the anisotropic factor lands on **W instead of D**.

Because pH == pW the token *count* is unchanged (65,536 either way), so nothing
ever crashed. But the token grid the attention windows are laid out on
(``prepare_multiscale_layout`` -> ``grid_size``, used for windowing and
local2global) ends up transposed relative to the actual feature map. That
disagreement is invisible: ``_build_multiscale_tokens`` computes its own
grid_size and ``forward`` discards it in favour of the layout's.

Here the stride is permuted (H, W, D) -> (D, H, W) once, explicitly. With the fix a
256^3 volume at patch (8, 8, 4) produces a (D, H, W) = (64, 32, 32) feature map --
exactly what ``prepare_multiscale_layout`` computes -- and the native scales come
out at 65,536 / 1,024 / 16 tokens x 384 dim.

**Weight-compatible on purpose.** Only ``ds1``'s stride changes, and a stride is not
a learned tensor, so parameter names and shapes stay byte-identical to
``ChestCTEmbed3D`` (``conv_down.*`` / ``conv1.*`` / ``ds1.*``). Pillar0-ChestCT and
lapa checkpoints both load into this module unchanged -- which also means a
stride mismatch loads cleanly and shows up only as a silently transposed grid. That
is exactly why this class exists rather than a flag on the old one.
"""

from .atlas_encoders import ChestCTEmbed3D


class ChestCTEmbed3D_Pillar0(ChestCTEmbed3D):
    """``ChestCTEmbed3D`` with ds1's stride derived on the (D, H, W) axis order."""

    @staticmethod
    def _resolve_ds1_stride(patch_size):
        """(H, W, D) patch_size -> (D, H, W) stride, so conv_down (4x) * ds1 == patch_size."""
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        patch_size = tuple(int(p) for p in patch_size)
        if len(patch_size) != 3:
            raise ValueError(f"chest_ct patch_size must have 3 entries, got {patch_size}")
        for name, p in zip(("pH", "pW", "pD"), patch_size):
            if p % 4 != 0 or p < 4:
                raise ValueError(
                    f"patch_size {patch_size} is not supported: {name}={p} must be a "
                    f"multiple of 4 (conv_down contributes a fixed 4x per axis)."
                )
        pH, pW, pD = patch_size
        # nn.Conv3d indexes (D, H, W); patch_size is declared (H, W, D).
        return (pD // 4, pH // 4, pW // 4)
