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
            if crop_mode == 'random':
                start_d = torch.randint(0, volume.shape[1] - target_slices + 1, (1,)).item()
            else:
                start_d = (volume.shape[1] - target_slices) // 2
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


class CTRATE_RET_Dataset(torch.utils.data.Dataset):
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
                 is_train: bool = True, text_cache_dir: str = None,
                 rve_root: str = "",
                 max_sentences: Optional[int] = None,
                 text_proj_ckpt: Optional[str] = None) -> None:
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
        self.text_cache_dir = text_cache_dir
        self.max_sentences = max_sentences

        # add
        self.rve_root = rve_root
        self.df = pd.read_json(json_path, lines=True)
        rve_df = pd.read_csv(self.rve_manifest_path)
        self.rve_manifest = rve_df.set_index('sample_name')['image_cache_path'].to_dict()
        self.weights = torch.ones(len(self.df), dtype=torch.float32)

        # None(기본) = 캐시된 raw Qwen 임베딩(4096-d)을 그대로 타깃으로 쓴다.
        # 경로를 주면 그 체크포인트의 text.proj(4096 -> 2624 -> 1152)를 태워서,
        # **비전 인코더가 실제로 정렬된 공간**으로 타깃을 옮긴다. gallery도 같은
        # 투영을 거친 파일을 써야 한다(scripts/project_gallery_textproj.py).
        self.text_proj_ckpt = text_proj_ckpt
        self._text_proj = None
        if text_proj_ckpt is not None:
            from pillar.utils.vlcabs import build_text_proj_from_checkpoint

            self._text_proj = build_text_proj_from_checkpoint(text_proj_ckpt, device="cpu")
            logger.info(f"CTRATE_RET_Dataset: 타깃 임베딩을 text.proj로 투영합니다 ({text_proj_ckpt})")

        self.build_text_cache()

    def __len__(self) -> int:
        return len(self.df)

    def build_text_cache(self):
        if self.text_cache_dir is None:
            self.text_embedder = None
        else:
            self.text_embedder = create_embedder_from_hub(local_cache_dir=self.text_cache_dir, revision=None, token=None, force_download=False,)

    def _get_cache_key(self, sample: Dict[str, Any], key: str = "sample_name") -> str:
        """캐시 키 생성. 값 유효성 검사를 '해시 전에' 수행."""
        raw = sample.get(key, None)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"데이터셋의 '{key}' 값이 비어있거나 무효합니다: {raw!r}")
        return hashlib.sha256(str(raw).encode()).hexdigest()

    def _load_one(self, row) -> Dict[str, Any]:
        """단일 row 로드. 실패 시 예외를 올림."""
        sample_name = row.get('sample_name', None)
        if sample_name is None:
            raise ValueError("row에 'sample_name'이 없습니다.")

        # sentence
        # sentence = row.get('sentence', None)

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
            crop_mode='random' if self.is_train else 'center',
        )

        if volume.shape[-2] != volume.shape[-1]:
            raise ValueError(f"Unexpected volume shape: {tuple(volume.shape)}")   

        volume = torch.flip(volume, dims=[1]).to(torch.float32)

        # 텍스트 임베딩
        if self.text_embedder is None:
            raise RuntimeError("text_embedder가 None입니다. text_cache_dir를 설정하세요.")
        text_cache_key = self._get_cache_key(row, key="sample_name")
        text_features = self.text_embedder.load_embedding_from_cache_key(text_cache_key)
        key_phrase = text_features['key_phrase']  # (N_sentence, d_target)
        if self._text_proj is not None:
            # raw Qwen(4096) -> text.proj -> 1152. 투영 후에 정규화해야 gallery와
            # 같은 단위구 위에 놓인다(gallery 쪽도 투영 후 정규화한다).
            with torch.no_grad():
                key_phrase = self._text_proj(key_phrase.float())
        key_phrase = F.normalize(key_phrase, p=2, dim=-1)

        # 원문 문장(gt_key_phrases 리포트 비교/디버깅용). json row의 "sentences"는
        # key_phrase 임베딩과 순서가 1:1로 동일하다 (row["key_phrase"]가 곧 이 텍스트이고,
        # text_cache는 sample_name 해시로 그 텍스트를 임베딩한 결과이므로) — 그래서 아래
        # 캡핑에서 embedding에 쓰는 idx를 텍스트에도 그대로 재사용해 정렬을 맞춘다.
        sentences = row.get("sentences", None)
        if isinstance(sentences, (list, tuple)) and len(sentences) != key_phrase.shape[0]:
            logger.warning(
                f"{sample_name}: sentences 개수({len(sentences)})가 key_phrase 개수"
                f"({key_phrase.shape[0]})와 달라 gt 텍스트 정렬이 안 맞을 수 있습니다."
            )
            sentences = None
        elif not isinstance(sentences, (list, tuple)):
            sentences = None
        else:
            sentences = list(sentences)

        # === outlier 캡핑 ===
        # max_sentences는 단순 메모리 절약용이 아니라(sent_embeds 텐서 자체는 activ spatial map
        # 대비 무시할 수준으로 작음), multi-GPU 학습에서 NCCL all_gather가 랭크 간 동일한 텐서
        # shape을 요구하기 때문에 필요한 "전역 고정 상한"이다. num_queries보다 크게 잡아봐야
        # Hungarian 매칭이 어차피 min(num_queries, n_tgt)개만 매칭하므로 의미가 없다.
        # 실제로 이 상한에 걸리면(=한 환자의 문장 수가 상한을 넘으면) 정답 일부가 유실되는
        # 것이므로, 조용히 넘어가지 않고 warning으로 남긴다.
        n = key_phrase.shape[0]
        if self.max_sentences is not None and n > self.max_sentences:
            logger.warning(
                f"{sample_name}: key_phrase 개수({n})가 max_sentences({self.max_sentences})를 "
                f"초과해 일부 문장이 잘립니다. max_sentences를 올리는 것을 고려하세요."
            )
            if self.is_train:
                idx = torch.randperm(n)[:self.max_sentences]
            else:
                idx = torch.arange(self.max_sentences)  # val/test는 결정적으로
            key_phrase = key_phrase[idx]
            if sentences is not None:
                sentences = [sentences[j] for j in idx.tolist()]
            n = self.max_sentences

        # sent_num: 패딩 전 실제 문장 개수. 모델/손실(TranSQLoss, RetrievalMetric)이
        # target_embeds_key="sent_embeds"/target_len_key="sent_num" 고정 크기 텐서를
        # 기대하므로, 여기서 max_sentences까지 0-padding해 배치 내(및 랭크 간) shape을
        # 맞춘다 — 그렇지 않으면 default_collate가 샘플마다 다른 문장 수 때문에 실패한다.
        sent_num = n
        if sentences is None:
            sentences = [""] * n
        if self.max_sentences is not None and n < self.max_sentences:
            pad_len = self.max_sentences - n
            key_phrase = torch.nn.functional.pad(key_phrase, (0, 0, 0, pad_len), mode="constant", value=0.0)
            sentences = sentences + [""] * pad_len

        batch: Dict[str, Any] = {
            "accession": sample_name,
            "sent_texts": sentences,  # list[str] | None — 원문 GT 문장 (key_phrase와 순서 동일, "" 패딩 포함)
            "sample_name": sample_name,
            "x": volume,
            "anatomy": self.anatomy,
            "sent_embeds": key_phrase,  # (max_sentences, d_target), 0-padded
            "sent_num": sent_num,       # 패딩 전 실제 문장 개수
        }
        return batch

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """해당 인덱스의 샘플을 한 번 로드하고, 실패하면 즉시 예외를 전파합니다."""
        row = self.df.iloc[int(idx)]

        try:
            return self._load_one(row)
        except Exception:
            logger.exception(f"샘플 로드 실패: idx={idx}")
            raise