import torch
import rve


def batch_apply_ct_windowing(x: torch.Tensor, windows_type="all") -> torch.Tensor:
    """
    x: (B, D, H, W) or (B, 1, D, H, W). Returns (B, num_windows, D, H, W).

    rve.apply_windowing computes any batch-level statistic (e.g. the "minmax"
    window's min/max) over the *entire* tensor it's given. Calling it once with
    a stacked (B, D, H, W) batch mixes that statistic across unrelated samples
    — a sample's "minmax" channel then depends on which other samples happen
    to share its batch (e.g. one sample with a metal implant compresses every
    other sample's minmax channel toward the low end). Every other CT window
    (lung/mediastinum/bone/...) is a fixed HU center/width clip — purely
    elementwise, safe to vectorize across the batch. Only "minmax" needs
    per-sample statistics, computed here directly instead of looping in
    Python per sample (rve's own per-sample-safe approach, but slower).
    """
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
