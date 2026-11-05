"""Per-sample-safe 11-channel CT windowing.

Verbatim copy of pillar-finetune ``pillar/utils/windowing.py`` (the exact
function the abn-probe features and the retrieval head were trained with).
Kept as a standalone module because importing the full ``pillar`` package
would pull in kornia/flash_attn.

Note: PETRG-3D's embedding layer instead calls
``rve.batch_apply_windowing_vectorized`` internally; the two were verified
equal channel-for-channel except a ~8e-4 numerical difference on the minmax
channel (see pillar-finetune CLAUDE.md), and with batch size 1 the minmax
statistics are per-sample in both.
"""
import torch
import rve


def batch_apply_ct_windowing(x: torch.Tensor, windows_type="all") -> torch.Tensor:
    """x: (B, D, H, W) or (B, 1, D, H, W). Returns (B, num_windows, D, H, W)."""
    if x.dim() == 5:
        x = x[:, 0]

    if windows_type == "all":
        windows = rve.get_available_windows("CT")
    elif isinstance(windows_type, str):
        windows = [windows_type]
    else:
        windows = windows_type

    outs = []
    for window in windows:
        if window == "minmax":
            per_sample_min = x.amin(dim=(1, 2, 3), keepdim=True)
            per_sample_max = x.amax(dim=(1, 2, 3), keepdim=True)
            w = (x - per_sample_min) / (per_sample_max - per_sample_min + 1e-8)
            w = w.clamp(0, 1)
        else:
            w = rve.apply_windowing(x, window, "CT")
        outs.append(w)
    return torch.stack(outs, dim=1)
