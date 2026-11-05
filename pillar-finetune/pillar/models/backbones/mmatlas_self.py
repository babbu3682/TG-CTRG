"""Multimodal Atlas implementation."""
import logging
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
from pillar.utils.logging import logger

from .atlas_encoders import ConvCXREmbed, MultiViewConvCXR, AbdomenCTEmbed3D, BMRMultiConvEmbed3D, BrainCTMultiConvEmbed3D, ProstateMRMultiConvEmbed3D, ChestCTEmbed3D
from .multimodal_msa import AtlasStage

# NOTE: do NOT re-bind `logger` to logging.getLogger(__name__) here. This module
# used to do exactly that right after importing pillar's configured logger,
# which silently routed every logger.info() in this file (including the
# checkpoint-load summary below) to an unconfigured logger that prints nothing
# under the project's logging setup — so a job log would show no evidence of
# what was or wasn't loaded.


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

class MultimodalAtlas(nn.Module):
    """Multimodal Atlas medical image analysis."""

    def __init__(
        self,
        args,
        embed_dim=384,
        num_classes=100,
        image_size=256,
        multiscale_feats=True,
        model_config={},
        device="cuda",
        pretrained="None",
        strict_load=True,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.multiscale_feats = multiscale_feats
        self.embed_dim = embed_dim
        self.model_config = model_config
        # Load model-specific config and setup device
        self.device = setup_device(device)
        self.pretrained = pretrained
        # Raise (instead of warn) if the checkpoint doesn't line up with the
        # architecture built from model_config. See setup_model() for why the
        # permissive default was dangerous.
        self.strict_load = strict_load

        self._init_patch_embeds()
        self._init_atlas_stages()
        self._init_pools()

        # Multi-scale activ/pooled outputs concat one `atlas_args["dim"]`-wide feature
        # per scale (see forward()'s torch.cat over `feats_for_head`/multiscale_layout) —
        # matches mmatlas.py's self.hidden_dim, which downstream code (MultiStage's
        # pool/head construction) reads directly.
        atlas_args = self.model_config["model"]["atlas_args"]
        num_scales = len(self.model_config["model"]["stages"])
        self.hidden_dim = atlas_args["dim"] * num_scales

        # Setup model
        self.setup_model()
        logger.info(f"Initialized MultimodalAtlas on device: {self.device}")

    def setup_model(self) -> None:
        """Load pretrained weights into the current module."""

        if self.pretrained is None or self.pretrained == "None":
            logger.info("No pretrained checkpoint provided. Using randomly initialized weights.")
            self.to(self.device)
            self.train()
            return

        ckpt = torch.load(self.pretrained, map_location="cpu", weights_only=False)

        # Lightning / 일반 ckpt 둘 다 대응
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            state_dict = ckpt

        ckpt_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None

        # DDP / DataParallel prefix 제거 (앞부분만)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[len("module."):] if k.startswith("module.") else k
            cleaned_state_dict[new_k] = v

        # visual prefix 추출 (startswith으로 안전하게).
        # 이 체크포인트는 CLIP 전체(vision + text tower)를 담고 있으므로, vision
        # encoder에 해당하는 "visual.*"만 골라내야 한다 (나머지 logit_scale/text
        # tower 키는 이 클래스에 대응 파라미터가 없음).
        visual_state_dict = {}
        for k, v in cleaned_state_dict.items():
            if k.startswith("visual."):
                new_k = k[len("visual."):]
                visual_state_dict[new_k] = v

        if not visual_state_dict:
            raise ValueError(
                "No 'visual.' keys found in checkpoint. "
                f"Available keys (first 10): {list(cleaned_state_dict.keys())[:10]}"
            )

        # 항상 strict=False로 로드한 뒤 결과를 직접 판정한다. strict=True로 바로
        # 던지면 missing/unexpected 목록을 볼 수 없어 디버깅이 어렵기 때문.
        missing, unexpected = self.load_state_dict(visual_state_dict, strict=False)

        # 로드 결과는 항상 남긴다 — "조용히 성공한 것처럼 보이는 실패"가 가장 위험하다.
        # missing key는 그 파라미터가 체크포인트 값이 아니라 **랜덤 초기화 상태로 남아
        # 있다**는 뜻이다. backbone을 freeze해서 쓰는 지금 구조에선 그런 파라미터가
        # 학습으로 교정될 방법이 아예 없으므로, encoder 품질을 비교하는 실험에서
        # "이 체크포인트는 성능이 나쁘다"는 잘못된 결론으로 직행하게 된다.
        logger.info(
            f"Loading self-pretrained checkpoint: {self.pretrained}"
            + (f" (epoch {ckpt_epoch})" if ckpt_epoch is not None else "")
        )
        logger.info(
            f"  checkpoint keys: {len(state_dict)} total, {len(visual_state_dict)} under 'visual.' "
            f"-> model params: {len(self.state_dict())}, missing: {len(missing)}, unexpected: {len(unexpected)}"
        )

        if missing or unexpected:
            detail = (
                f"Checkpoint/architecture mismatch for {self.pretrained}: "
                f"{len(missing)} missing key(s) (left RANDOMLY INITIALIZED), "
                f"{len(unexpected)} unexpected key(s) (DISCARDED).\n"
                f"  missing (first 20): {missing[:20]}\n"
                f"  unexpected (first 20): {unexpected[:20]}\n"
                "This usually means backbone_kwargs.model_config does not match the "
                "architecture this checkpoint was trained with (patch_size / stages / "
                "dim / num_heads / in_chans ...). Copy model_config verbatim from the "
                "pretrain run's own model_configs/*.json rather than hand-writing it. "
                "Pass strict_load=False in backbone_kwargs to downgrade this to a warning."
            )
            if self.strict_load:
                raise RuntimeError(detail)
            logger.warning(detail)
        else:
            logger.info("  all checkpoint tensors matched the model exactly (0 missing / 0 unexpected)")

        self.to(self.device)
        self.train()
        logger.info(f"Model loaded successfully on device: {self.device}")

    def extract_features(self, x, batch=None) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            output = self.forward(x=x, batch=batch)
        return output

    # NOTE: no custom eval()/train() override here (unlike mmatlas.py's HF-wrapped
    # variant, which dispatches to an inner self.model). This class's submodules
    # (patch_embeds, atlas_models) are direct children, so nn.Module's own eval()/
    # train() (which recursively sets .training on self and every submodule) is
    # already correct. A prior override here checked `hasattr(self, "model")` — an
    # attribute that never exists on this class — making it a silent no-op: calling
    # .eval() left self.training (and every DropPath/Dropout inside) stuck at True.



    def _init_patch_embeds(self):
        modalities = self.model_config["modalities"]
        MODALITY_TO_PATCH_EMBED = {
            "chest_xray_single_view": ConvCXREmbed,
            "chest_xray_two_view": MultiViewConvCXR,
            "chest_ct": ChestCTEmbed3D,
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

            logging.info(f"Initializing patch embed for modality: {modality}, image_size: {image_size}, patch_size: {patch_size}, in_chans: {in_chans}, num_features: {num_features}")

            self.patch_embeds[modality] = MODALITY_TO_PATCH_EMBED[modality](
                vol_size=image_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=num_features,
            )

    def _init_atlas_stages(self):
        atlas_models = []
        atlas_args = self.model_config["model"]["atlas_args"]
        num_scales = len(self.model_config["model"]["stages"])
        for stage_idx, depth in enumerate(self.model_config["model"]["stages"]):
            atlas_args["depth"] = depth
            atlas_args["num_scales"] = num_scales
            atlas_models.append(AtlasStage(**atlas_args))

        self.atlas_models = nn.ModuleList(atlas_models)

    def _init_pools(self):
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def _build_multiscale_tokens(self, x_BCDHW, merge_ratio, local2global):
        seqlen = x_BCDHW.shape[2] * x_BCDHW.shape[3] * x_BCDHW.shape[4]
        downsampling_ratio = local2global[0] * local2global[1] * local2global[2]
        kernel_size = (1, 1, 1)

        min_seqlen = merge_ratio[0] * merge_ratio[1] * merge_ratio[2]

        num_stages = (math.ceil((math.log2(seqlen / min_seqlen) / math.log2(downsampling_ratio))) + 1)

        stages = []
        grid_sizes = []

        for scale in range(num_stages):
            local2global_op = nn.MaxPool3d(kernel_size=kernel_size)
            x_scale_BCDHW = local2global_op(x_BCDHW)
            kernel_size = tuple(k * l for k, l in zip(kernel_size, local2global))

            # check if windowing is possible
            b, c, d, h, w = x_scale_BCDHW.shape

            ## if exact merge is possible, then dont add another scale
            if d == merge_ratio[0] and h == merge_ratio[1] and w == merge_ratio[2]:
                x_scale_win = rearrange(x_scale_BCDHW, "b c d h w -> b (d h w) c")
                grid_size = [1, 1, 1]
            ## only reduce in dimensions where merge is possible
            elif d >= merge_ratio[0] and h >= merge_ratio[1] and w >= merge_ratio[2]:
                # run windowing
                x_scale_win = rearrange(
                    x_scale_BCDHW,
                    "b c (d m0) (h m1) (w m2) -> b d h w m0 m1 m2 c",
                    m0=merge_ratio[0],
                    m1=merge_ratio[1],
                    m2=merge_ratio[2],
                )
                grid_size = [
                    x_scale_win.shape[2],
                    x_scale_win.shape[3],
                    x_scale_win.shape[1],
                ]
                x_scale_win = rearrange(x_scale_win, "b d h w m0 m1 m2 c -> (b d h w) (m0 m1 m2) c")
            else:
                x_scale_win = rearrange(x_scale_BCDHW, "b c d h w -> b (d h w) c")
                grid_size = [1, 1, 1]

            stages.append(x_scale_win)
            grid_sizes.append(grid_size)

            if math.prod(grid_size) == 1:
                break

        return stages, grid_sizes
    
    def build_scales(self, image, modality):
        """
        Expects input to be dictionary of tensors,
        """
        modal_config = self.model_config["modalities"]

        patch_embed_fn = self.patch_embeds[modality] # image = torch.Size([4, 11, 256, 256, 256])
        patch_embed_xBCDHW = patch_embed_fn(image)   # patch_embed_xBCDHW = torch.Size([4, 384, 64, 32, 32])
        multiscale_feats, grid_sizes = self._build_multiscale_tokens(
            patch_embed_xBCDHW,
            merge_ratio=modal_config[modality]["merge_ratio"],
            local2global=modal_config[modality]["local2global"],
        )
        return multiscale_feats, grid_sizes

    def prepare_multiscale_layout(self, img_size, merge_ratio, local2global, patch_size):
        """
        given the input size, merge_ratio and local2global
        prepare the layout for multiscale attention and config
        for the architecture
        """
        H, W, D = img_size
        pH, pW, pD = patch_size
        lD, lH, lW = local2global
        mD, mH, mW = merge_ratio

        h, w, d = H // pH, W // pW, D // pD
        seqlen = h * w * d

        downsampling_ratio = lD * lH * lW
        min_seqlen = mD * mH * mW

        kD, kH, kW = 1, 1, 1

        if seqlen <= min_seqlen:
            return [
                {
                    "grid_size": [1, 1, 1],
                    "window_size": seqlen,
                    "num_windows": 1,
                    "seq_length": seqlen,
                    "window_dims": [h, w, d],
                }
            ]
        num_stages = (
            math.ceil((math.log2(seqlen / min_seqlen) / math.log2(downsampling_ratio)))
            + 1
        )

        multiscale_layout = []

        for scale in range(num_stages):
            h0, w0, d0 = h // kH, w // kW, d // kD

            # if h0 <= mH or w0 <= mW or d0 <= mD:
            if h0 == mH and w0 == mW and d0 == mD:
                # if exact merge is possible, then dont add another scale
                multiscale_layout.append(
                    {
                        "grid_size": [1, 1, 1],
                        "window_size": h0 * w0 * d0,
                        "num_windows": 1,
                        "seq_length": h0 * w0 * d0,
                        "window_dims": [h0, w0, d0],
                    }
                )
                break
            elif h0 >= mH and w0 >= mW and d0 >= mD:
                grid_size = [h0 // mH, w0 // mW, d0 // mD]
                multiscale_layout.append(
                    {
                        "grid_size": grid_size,
                        "window_size": mH * mW * mD,
                        "num_windows": grid_size[0] * grid_size[1] * grid_size[2],
                        "seq_length": h0 * w0 * d0,
                        "window_dims": [mH, mW, mD],
                    }
                )
            else:
                multiscale_layout.append(
                    {
                        "grid_size": [1, 1, 1],
                        "window_size": h0 * w0 * d0,
                        "num_windows": 1,
                        "seq_length": h0 * w0 * d0,
                        "window_dims": [h0, w0, d0],
                    }
                )
                break

            kD *= lD
            kH *= lH
            kW *= lW
        return multiscale_layout

    def forward(self, x, batch=None):
        modality, image = batch["anatomy"][0], x
        bsz = image.shape[0]
        logger.debug(f"Entering MultiModalAtlas forward with modality: {modality}, batch size: {bsz}")
        x, grid_sizes = self.build_scales(image, modality)

        modal_config = self.model_config["modalities"][modality]
        image_size = modal_config["image_size"]
        merge_ratio = modal_config["merge_ratio"]
        local2global = modal_config["local2global"]
        patch_size = modal_config["patch_size"]
        multiscale_layout = self.prepare_multiscale_layout(
            img_size=image_size,
            merge_ratio=merge_ratio,
            local2global=local2global,
            patch_size=patch_size,
        )
        grid_sizes = [layout["grid_size"] for layout in multiscale_layout]

        multiscale_feats = []
        for level, atlas_model in enumerate(self.atlas_models):
            is_last = len(x) == 1 or level == len(self.atlas_models) - 1
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
            if (math.prod(grid_sizes[i]) > 1 and len(scale_tokens.shape) == 3):  # Check if likely windowed format
                rearranged_scale = rearrange(scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz)
                logger.debug(f"  Rearranging scale {i} (windowed) {scale_tokens.shape} -> {rearranged_scale.shape}")
                feats_for_head.append(rearranged_scale)
            elif (len(scale_tokens.shape) == 3 and scale_tokens.shape[0] == bsz):  # Already (B, N, C)
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
        
        if self.multiscale_feats:
            x = []
            for i, scale in enumerate(feats_for_head):
                feats = self.maxpool(scale.transpose(1, 2)).squeeze(2)
                x.append(feats)
            x = torch.cat(x, dim=1)
        else:
            ## return maxpool on last scale features only
            x = feats_for_head[0]
            x = self.maxpool(x.transpose(1, 2)).squeeze(2)

        # For multiscale
        res = [image_size[i] // patch_size[i] for i in range(3)]
        outs = []

        for idx in range(len(multiscale_layout)):
            layout = multiscale_layout[idx]
            scale = feats_for_head[idx]
            m0, m1, m2 = layout["window_dims"]
            g0, g1, g2 = layout["grid_size"]
            scale = rearrange(scale, "b (g2 g0 g1 m2 m0 m1) c -> b c (g2 m2) (g0 m0) (g1 m1)", g2=g2, g1=g1, g0=g0, m2=m2, m1=m1, m0=m0)
            # interpolate to the same size
            scale = nn.functional.interpolate(scale, size=res, mode="nearest")
            outs.append(scale)
        outs = torch.cat(outs, dim=1)

        outputs = {
            "activ": outs,
            # could also consider taking scale 2 and scale 3 and interpolate up then concat along channels
            # can modify atlas for the cleanest API
            "pooled": nn.functional.normalize(x, dim=-1),
        }

        return outputs
