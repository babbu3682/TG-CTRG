"""Multimodal Atlas implementation."""

import torch
import numpy as np
from transformers import AutoModel, AutoConfig
from torchvision.transforms import Compose, Normalize
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Any, Dict, Optional
from easydict import EasyDict
from pathlib import Path
import yaml
import json
from huggingface_hub import hf_hub_download
from pillar.utils.logging import logger


def setup_device(device_spec: Optional[str] = None) -> torch.device:
    """
    Set up and validate device for model/data operations.

    Args:
        device_spec: Device specification ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        Torch device object
    """
    if device_spec is None:
        device_spec = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return device


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load model-specific configuration.

    Args:
        model_name: Name of the model (e.g., 'medgemma', 'medimageinsights')

    Returns:
        Model configuration dictionary
    """
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "models" / f"{model_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    args = EasyDict(config)
    return args


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get value from config using dot notation (e.g., 'data.root_dir').

    Args:
        config: Configuration dictionary
        key: Dot-separated key path
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


class MultimodalAtlas(nn.Module):
    """Multimodal Atlas medical image analysis."""

    def __init__(
        self,
        args,
        device="cuda",
        model_repo_id=None,
        model_revision=None,
        pretrained=False,
        activ_scale=None,
    ):
        super().__init__()
        self.args = args

        # Load model-specific config and setup device
        self.device = setup_device(device)
        self.model_repo_id = model_repo_id
        self.model_revision = model_revision
        self.pretrained = pretrained
        # None(기본) = 기존 동작: 3개 스케일을 전부 최정밀 해상도로 nearest 업샘플한 뒤
        #   채널로 concat해서 `activ`를 만든다 (chest_ct 기준 (B,1152,32,32,64)).
        # 정수 k = 스케일 k의 native 토큰만 업샘플/concat 없이 그대로 내보낸다
        #   (chest_ct scale 1 기준 (B,384,16,8,8) = 1024토큰). forward의 주석 참고.
        self.activ_scale = activ_scale

        # Setup model
        self.setup_model()

        logger.info(f"Initialized MultimodalAtlas on device: {self.device}")

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""

        print(f"Loading {self.model_repo_id} with revision {self.model_revision} from HuggingFace")
        logger.info("Loading model from HuggingFace")

        # self.model is CLIPMultimodalAtlas
        self.model = AutoModel.from_pretrained(self.model_repo_id, revision=self.model_revision, trust_remote_code=True)
        # self.visual is MultiModalAtlas
        if hasattr(self.model.model, "visual"):
            self.visual = self.model.model.visual
        else:
            self.visual = self.model.model

        self.model.to(self.device)

        # Get hidden_dim from the model - check various possible attributes
        if hasattr(self.visual, "embed_dim"):
            self.hidden_dim = self.visual.embed_dim
        else:
            # Default fallback
            self.hidden_dim = 1152
            logger.warning(f"Could not determine hidden_dim from model, using default: {self.hidden_dim}")

        self.model.train()

        logger.info(f"Model loaded successfully on device: {self.device}")

    def preprocess_single(self, image):
        """
        Preprocess a single exam for your model (for use in dataset __getitem__).

        Argscn:
            image: Tensor from dataset (normalized to [0,1] range)

        Returns:
            Preprocessed tensor ready for your model
        """
        ## hard
        # normalize_mean = self.normalize_mean
        # normalize_std = self.normalize_std
        # self.transform = Compose(
        #     [
        #         Normalize(
        #             mean=normalize_mean, std=normalize_std
        #         ),  # Normalize for single channel
        #     ]
        # )

        # image = self.transform(image)
        return image

    def extract_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        """
        Extract features from input images/volumes.

        Args:
            inputs: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D

        Returns:
            Feature embeddings as numpy array of shape (B, feature_dim)
        """
        inputs = inputs.to(self.device)
        inputs_as_dict = {modality: inputs}

        with torch.no_grad():
            # Check if we have the HuggingFace model with extract_vision_feats method
            if hasattr(self.model, "extract_vision_feats"):
                features = self.model.extract_vision_feats(inputs)
            else:
                # Fallback to forward pass for feature extraction
                output = self.forward(inputs)
                if isinstance(output, dict) and "features" in output:
                    features = output["features"]
                elif isinstance(output, torch.Tensor):
                    features = output
                else:
                    raise ValueError(f"Cannot extract features from model output of type {type(output)}")

            # Convert to numpy for multiprocessing compatibility
            features = features.cpu().numpy().astype(np.float32)

        return features

    def eval(self):
        """Set model to evaluation mode."""
        if hasattr(self, "model"):
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor, batch=None) -> dict:
        visual = self.visual

        modality, image = batch["anatomy"][0], x
        bsz = image.shape[0]
        logger.debug(f"Entering MultiModalAtlas forward with modality: {modality}, batch size: {bsz}")
        x, grid_sizes = visual.build_scales({modality: x}, batch)

        modal_config = visual.model_config["modalities"][modality]
        image_size = modal_config["image_size"]
        merge_ratio = modal_config["merge_ratio"]
        local2global = modal_config["local2global"]
        patch_size = modal_config["patch_size"]
        multiscale_layout = visual.prepare_multiscale_layout(
            img_size=image_size,
            merge_ratio=merge_ratio,
            local2global=local2global,
            patch_size=patch_size,
        )
        grid_sizes = [layout["grid_size"] for layout in multiscale_layout]

        multiscale_feats = []
        for level, atlas_model in enumerate(visual.atlas_models):
            is_last = len(x) == 1 or level == len(visual.atlas_models) - 1
            x = atlas_model(
                x,
                grid_sizes=grid_sizes[level:],
                multiscale_layout=multiscale_layout[level:],
                merge_ratio=merge_ratio,
                local2global=local2global,
                modality=modality,
            )
            if not is_last:
                multiscale_feats.append(x[0])
                x = x[1:]

        multiscale_feats.append(x[0])

        feats_for_head = []
        for i, scale_tokens in enumerate(multiscale_feats):
            # Need to know if scale_tokens is windowed (B*NW, K, C) or flattened (B, N, C)
            # Assuming it's (B*NW, K, C) if NW > 1 based on layout
            if math.prod(grid_sizes[i]) > 1 and len(scale_tokens.shape) == 3:  # Check if likely windowed format
                rearranged_scale = rearrange(scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz)
                logger.debug(f"  Rearranging scale {i} (windowed) {scale_tokens.shape} -> {rearranged_scale.shape}")
                feats_for_head.append(rearranged_scale)
            elif len(scale_tokens.shape) == 3 and scale_tokens.shape[0] == bsz:  # Already (B, N, C)
                logger.debug(f"  Using scale {i} (already BNC) shape: {scale_tokens.shape}")
                feats_for_head.append(scale_tokens)
            else:
                logger.warning(f"  Unexpected shape for scale {i}: {scale_tokens.shape}. Attempting BNC rearrange.")
                # Try a generic rearrange, might fail if batch dim isn't divisible
                try:
                    rearranged_scale = rearrange(scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz)
                    feats_for_head.append(rearranged_scale)
                except Exception as e:
                    logger.error(f"    Failed to rearrange scale {i}: {e}. Skipping this scale for readout.")
                    continue

        if visual.multiscale_feats:
            x = []
            for i, scale in enumerate(feats_for_head):
                feats = visual.maxpool(scale.transpose(1, 2)).squeeze(2)
                x.append(feats)
            x = torch.cat(x, dim=1)
        else:
            ## return maxpool on last scale features only
            x = feats_for_head[0]
            x = visual.maxpool(x.transpose(1, 2)).squeeze(2)

        # finetune UPSAMPLE
        import torch.nn.functional as F

        res = [image_size[i] // patch_size[i] for i in range(3)]

        def _to_grid(idx):
            """스케일 idx의 window-major 토큰을 (B, C, d, h, w) 격자로 되돌린다.

            `prepare_multiscale_layout`이 grid_size=[h//mH, w//mW, d//mD],
            window_dims=[mH, mW, mD] 로 **[h, w, d] 순서**를 쓰므로
            g0↔h, g1↔w, g2↔d 이고, 아래 rearrange 결과의 축 순서는 (d, h, w) 이다.
            """
            layout = multiscale_layout[idx]
            m0, m1, m2 = layout["window_dims"]
            g0, g1, g2 = layout["grid_size"]
            return rearrange(
                feats_for_head[idx],
                "b (g2 g0 g1 m2 m0 m1) c -> b c (g2 m2) (g0 m0) (g1 m1)",
                g2=g2,
                g1=g1,
                g0=g0,
                m2=m2,
                m1=m1,
                m0=m0,
            )

        if self.activ_scale is not None:
            # 단일 native 스케일만 그대로 내보낸다 — 업샘플도 concat도 하지 않으므로
            # 아래 기본 경로의 축 불일치(주석 참고)와 채널 부풀리기가 둘 다 사라진다.
            # chest_ct(patch [8,8,4]) scale 1: (B, 384, 16, 8, 8) = 1024 토큰.
            if not 0 <= self.activ_scale < len(multiscale_layout):
                raise ValueError(
                    f"activ_scale={self.activ_scale} 는 이 modality의 스케일 범위 "
                    f"[0, {len(multiscale_layout) - 1}] 를 벗어납니다."
                )
            outs = _to_grid(self.activ_scale)
        else:
            # ⚠️ 기본(레거시) 경로. 여기서 `res`는 [H//pH, W//pW, D//pD] 즉 (h, w, d)
            # 순서인데 `_to_grid`의 결과는 (d, h, w) 순서다. patch_size의 세 값이 모두
            # 같으면(예: self-pretrained의 [8,8,8]) 격자가 정육면체라 무해하지만,
            # 오픈소스 Pillar0-ChestCT의 [8,8,4]에서는 (64,32,32) 텐서를 (32,32,64)로
            # interpolate하게 되어 깊이축이 절반으로 솎이고 폭축이 복제된다.
            # 진행 중인 실험들과의 일관성 때문에 지금 고치지 않고 그대로 둔다 —
            # 새 실험은 activ_scale 을 지정해서 이 경로를 아예 타지 않게 할 것.
            outs = [
                nn.functional.interpolate(_to_grid(idx), size=res, mode="nearest")
                for idx in range(len(multiscale_layout))
            ]
            outs = torch.cat(outs, dim=1)  # pillar-0 chest ct weight: (B, 1152, 32, 32, 64)

        outputs = {
            "activ": outs,
            # could also consider taking scale 2 and scale 3 and interpolate up then concat along channels
            # can modify atlas for the cleanest API
            "pooled": F.normalize(x, dim=-1),
        }
        return outputs
