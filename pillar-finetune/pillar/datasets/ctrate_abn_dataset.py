import os
import hashlib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rve

from pillar.utils.logging import logger
from .hf_cache_utils import create_embedder_from_hub


def load_rve_volume(filepath: str,
                    target_slices: Optional[int] = None,
                    target_h: Optional[int] = None,
                    target_w: Optional[int] = None,
                    pad_value: float = -1024.0,
                    crop_mode: str = 'center') -> torch.Tensor:
    """
    Load a 3D volume from an RVE tar file with optional cropping/padding.
    Returns: (1, D, H, W)
    """
    try:
        volume = rve.load_sample(filepath, use_hardware_acceleration=False)

        if volume.dim() == 3:
            volume = volume.unsqueeze(0)          # (D,H,W) -> (1,D,H,W)
        elif volume.dim() != 4:
            raise ValueError(f"Unexpected volume shape from RVE: {volume.shape}")

        # D
        if target_slices is not None and volume.shape[1] < target_slices:
            z_diff = target_slices - volume.shape[1]
            z0 = z_diff // 2
            volume = torch.nn.functional.pad(
                volume, (0, 0, 0, 0, z0, z_diff - z0),
                mode='constant', value=pad_value)
        elif target_slices is not None and volume.shape[1] > target_slices:
            max_start = volume.shape[1] - target_slices
            if crop_mode == 'random':
                start_d = torch.randint(0, max_start + 1, (1,)).item()
            elif isinstance(crop_mode, str) and crop_mode.startswith('offset:'):
                # TTA 용: center 에서 z 축으로 n 슬라이스 이동한 크롭 (범위 밖은 clamp).
                start_d = min(max(max_start // 2 + int(crop_mode.split(':', 1)[1]), 0), max_start)
            else:
                start_d = max_start // 2
            volume = volume[:, start_d:start_d + target_slices]

        # H
        if target_h is not None:
            cur_h = volume.shape[2]
            if cur_h < target_h:
                h_diff = target_h - cur_h
                h0 = h_diff // 2
                volume = torch.nn.functional.pad(
                    volume, (0, 0, h0, h_diff - h0, 0, 0),
                    mode='constant', value=pad_value)
            elif cur_h > target_h:
                start_h = (cur_h - target_h) // 2
                volume = volume[:, :, start_h:start_h + target_h, :]

        # W
        if target_w is not None:
            cur_w = volume.shape[3]
            if cur_w < target_w:
                w_diff = target_w - cur_w
                w0 = w_diff // 2
                volume = torch.nn.functional.pad(
                    volume, (w0, w_diff - w0, 0, 0, 0, 0),
                    mode='constant', value=pad_value)
            elif cur_w > target_w:
                start_w = (cur_w - target_w) // 2
                volume = volume[:, :, :, start_w:start_w + target_w]

        return volume
    except Exception as e:
        # 원 예외를 그대로 올려 상위 retry가 판단하도록 함
        logger.warning(f"Failed to load RVE volume from {filepath}: {e}")
        raise


class CTRATE_ABN_Dataset(torch.utils.data.Dataset):
    """
    CSV/JSONL 기반 retrieval 파인튜닝 데이터셋.

    반환 dict:
      - x:          (C=1, D, H, W)  CT 볼륨
      - key_phrase: (N_sentence, d_target)  L2 정규화된 문장 임베딩
      - accession / sample_name / anatomy: 메타데이터
    실패 시 None을 반환하여 collate에서 걸러지도록 함(ignore_None 철학과 정합).
    """

    def __init__(self, args, split_group, augmentations,
                 json_path: str = None,
                 rve_manifest_path: str = None, anatomy: Optional[str] = None,
                 windows: Optional[str] = "default", target_d: int = 256,
                 target_hw: int = 256, pad_value: int = -1024,
                 is_train: bool = True,
                 rve_root: str = "", label_csv: str = "") -> None:
        self.split_group = split_group
        self.json_path = json_path
        self.rve_manifest_path = rve_manifest_path
        self.augmentations = augmentations
        self.anatomy = anatomy
        self.windows = windows
        self.target_d = target_d
        self.target_hw = target_hw
        self.pad_value = pad_value
        self.is_train = is_train

        # add
        self.rve_root = rve_root
        self.df = pd.read_json(json_path, lines=True)
        label_df = pd.read_csv(label_csv)
        # VolumeName -> 18개 라벨 값(np.ndarray). 매 __getitem__마다 47k행을 선형 스캔하던 것을 O(1) 조회로 변경.
        label_values = label_df.iloc[:, 1:].to_numpy(dtype=np.float32)
        self.label_map = dict(zip(label_df["VolumeName"], label_values))
        rve_df = pd.read_csv(self.rve_manifest_path)
        self.rve_manifest = rve_df.set_index('sample_name')['image_cache_path'].to_dict()
        self.weights = torch.ones(len(self.df), dtype=torch.float32)


    def __len__(self) -> int:
        return len(self.df)

    def _load_one(self, row) -> Dict[str, Any]:
        """단일 row 로드. 실패 시 예외를 올림."""
        sample_name = row.get('sample_name', None)
        if sample_name is None:
            raise ValueError("row에 'sample_name'이 없습니다.")

        # 매니페스트 조회 (None 방어)
        cache_path = self.rve_manifest.get(sample_name)
        if cache_path is None:
            raise KeyError(f"rve_manifest에 '{sample_name}' 없음")
        rve_path = os.path.join(self.rve_root, cache_path)

        volume = load_rve_volume(
            rve_path,
            target_slices=self.target_d,
            target_h=self.target_hw,
            target_w=self.target_hw,
            pad_value=self.pad_value,
            crop_mode=(self.crop_mode_override
                       if getattr(self, 'crop_mode_override', None)
                       else ('random' if self.is_train else 'center')),
        )

        if volume.shape[-2] != volume.shape[-1]:
            raise ValueError(f"Unexpected volume shape: {tuple(volume.shape)}")            

        volume = torch.flip(volume, dims=[1]).to(torch.float32)

        # Class Label
        label = self.label_map.get(sample_name + ".nii.gz")
        if label is None:
            raise KeyError(f"label_map에 '{sample_name}' 없음")

        batch: Dict[str, Any] = {
            "accession": sample_name,
            "sample_name": sample_name,
            "x": volume,
            "label": torch.from_numpy(label).clone(),
            "anatomy": self.anatomy,
        }
        return batch

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """해당 인덱스의 샘플을 한 번 로드. 실패 시 None을 반환해 collate에서 걸러지도록 함
        (단일 손상 샘플 때문에 DataLoader worker/DDP 전체가 죽는 것을 방지)."""
        row = self.df.iloc[int(idx)]

        try:
            return self._load_one(row)
        except Exception:
            logger.exception(f"샘플 로드 실패: idx={idx} (건너뜀)")
            return None
