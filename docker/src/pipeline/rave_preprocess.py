"""Stage-0 preprocessing: challenge ``*.nii.gz`` -> RAVE chest-CT cache volume.

Reproduces exactly what built the ``ctrate_cache_1.25mm_lz4`` training cache
(rad-vision-engine ``CTProcessor`` + ``configs/ct_chest.yaml`` + LZ4 exporter),
minus the tar/LZ4 round-trip: the LZ4Exporter ``np.save``s
``processed_series.numpy_volume`` verbatim and ``rve.load_sample`` reads it
back unchanged, so handing the in-memory ``numpy_volume`` (D, H, W int16 HU)
straight to the consumers is bit-identical to going through the cache file.

ct_chest.yaml settings baked in below (kept as module constants so drift from
the repo config is visible in one place):
    resampling.target_spacing: [1.25, 1.25, 1.25]  (x, y, z mm)
    slice_selection: middle 256 slices
    crop_pad.size: [256, 256]  (H, W)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# ct_chest.yaml (rad-vision-engine) — must match the training cache recipe.
TARGET_SPACING = [1.25, 1.25, 1.25]   # (x, y, z) mm
TARGET_HW = [256, 256]                # crop/pad target (H, W)
NUM_SLICES = 256                      # middle-slice selection (D)

_PROCESSOR = None


def _get_processor():
    """Build the real vision_engine CTProcessor once (thread-unsafe by design:
    the whole pipeline is sequential per volume)."""
    global _PROCESSOR
    if _PROCESSOR is None:
        from vision_engine.core.config import Config
        from vision_engine.processing.ct_processor import CTProcessor

        config = Config(
            modality="CT",
            anatomy="chest",
            processing={
                "crop_pad": {"size": list(TARGET_HW)},
                "resampling": {"target_spacing": list(TARGET_SPACING)},
                "conversion_backend": "sitk",
                "slice_selection": {"enabled": True, "slices": NUM_SLICES},
            },
            # Only used by Config validation — no exporter runs in this
            # pipeline (the volume is consumed in memory; "lz4" mirrors the
            # cache-recipe ct_chest.yaml + lz4_fast.yaml pairing).
            exporter={"compression": "lz4"},
        )
        _PROCESSOR = CTProcessor(config)
    return _PROCESSOR


def preprocess_nii(nii_path: str | Path, sample_name: str) -> np.ndarray:
    """Run the RAVE chest-CT pipeline on one NIfTI file.

    Returns the cache-equivalent volume: (D, H, W) int16 numpy array in raw
    Hounsfield units (D <= 256 middle slices, H = W = 256, 1.25 mm iso).
    """
    from vision_engine.core.data_structures import SeriesInfo

    nii_path = str(nii_path)
    info = SeriesInfo(
        accession=sample_name,
        series_uid=sample_name,
        study_path=nii_path,
        series_number=0,
        series_description="challenge chest CT",
        slice_count=0,
        modality="CT",
        anatomy="chest",
    )
    # SeriesInfo.get_dicom_path() honors this override for direct-file inputs
    # (same mechanism vision_engine's series_paths_loader uses for NIfTI rows).
    info._direct_path = nii_path

    processed = _get_processor().process_series(info)
    return processed.numpy_volume


def volume_to_model_tensor(volume: np.ndarray,
                           target_d: int = 256,
                           target_hw: int = 256,
                           pad_value: float = -1024.0) -> torch.Tensor:
    """Cache volume -> the (1, D, H, W) float32 tensor every model consumer uses.

    Mirrors the shared recipe of pillar-finetune ``CTRATE_ABN/RET_Dataset`` and
    PETRG-3D ``CTRATEDataset_Train/Test``:
        load_rve_volume(target 256^3, pad -1024, center crop)  ->  flip(dims=[1])
        ->  float32
    The datasets all call ``torch.flip(volume, dims=[1])`` (D axis) right after
    loading — keep that here so the tensor matches training bit-for-bit.
    """
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(volume)).float()
    if t.dim() == 3:
        t = t.unsqueeze(0)                      # (1, D, H, W)

    # D: pad (never crop here — the RAVE step already selected 256 slices; a
    # deeper volume would mean the preprocessing config drifted, so fail loud).
    d = t.shape[1]
    if d > target_d:
        raise ValueError(f"volume has {d} slices > target_d={target_d}; "
                         f"RAVE slice_selection should have capped this")
    if d < target_d:
        diff = target_d - d
        before = diff // 2
        t = F.pad(t, (0, 0, 0, 0, before, diff - before),
                  mode="constant", value=pad_value)

    # H/W: the RAVE step already crop/padded to 256x256; verify instead of
    # silently re-cropping.
    if t.shape[2] != target_hw or t.shape[3] != target_hw:
        raise ValueError(f"unexpected H/W {tuple(t.shape[2:])}, expected "
                         f"{target_hw}x{target_hw}")

    t = torch.flip(t, dims=[1])                 # same as every training dataset
    return t.contiguous()
