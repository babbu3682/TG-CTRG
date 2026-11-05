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
import logging

from .atlas_encoders import ConvCXREmbed, MultiViewConvCXR, AbdomenCTEmbed3D, BMRMultiConvEmbed3D, BrainCTMultiConvEmbed3D, ProstateMRMultiConvEmbed3D, ChestCTEmbed3D
from .multimodal_msa import AtlasStage

logger = logging.getLogger(__name__)


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
        embed_dim=384,
        num_classes=100,
        image_size=256,
        multiscale_feats=True,
        model_config={},
        device="cuda",
        pretrained="None",        
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes        
        self.image_size = image_size
        self.multiscale_feats = multiscale_feats
        self.embed_dim = embed_dim
        self.model_config = model_config
        # Load model-specific config and setup device
        self.device = setup_device(device)
        self.pretrained = pretrained

        self._init_patch_embeds()
        self._init_atlas_stages()
        self._init_pools()

        # Setup model
        self.setup_model()
        logger.info(f"Initialized MultimodalAtlas on device: {self.device}")

    @staticmethod
    def _read_checkpoint(path: str) -> Dict[str, torch.Tensor]:
        """Read a raw state dict from a ``.pt``/``.bin`` file, a ``.safetensors``
        file, or a HuggingFace-style directory containing ``model.safetensors``.

        The open-source Pillar-0 releases (e.g. ``checkpoints/Pillar0-ChestCT``)
        ship as the latter, whereas the in-house PILLAR0 pretraining runs write
        plain ``epoch_*.pt`` files -- both are accepted here so that
        ``model_config_yaml``'s ``pretrained:`` field can point at either.
        """
        p = Path(path)
        if p.is_dir():
            candidates = [p / "model.safetensors", p / "pytorch_model.bin"]
            for candidate in candidates:
                if candidate.exists():
                    p = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No model.safetensors / pytorch_model.bin found in {path!r}."
                )

        if p.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(p), device="cpu")

        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            return ckpt.get("state_dict", ckpt.get("model", ckpt))
        return ckpt

    def setup_model(self) -> None:
        """Load pretrained weights into the current module."""

        if self.pretrained is None or self.pretrained == "None":
            logger.info("No pretrained checkpoint provided. Using randomly initialized weights.")
            self.to(self.device)
            self.train()
            return

        state_dict = self._read_checkpoint(self.pretrained)

        # DDP / DataParallel prefix 제거 (앞부분만)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[len("module."):] if k.startswith("module.") else k
            cleaned_state_dict[new_k] = v

        # This module *is* the visual tower, so strip whatever wrapper prefix the
        # checkpoint used: the in-house CLIP checkpoints store it as ``visual.*``
        # while the open-source HF export nests it one level deeper as
        # ``model.visual.*`` (CLIPMultimodalAtlas.model.visual).
        visual_state_dict = {}
        for prefix in ("visual.", "model.visual."):
            visual_state_dict = {
                k[len(prefix):]: v
                for k, v in cleaned_state_dict.items()
                if k.startswith(prefix)
            }
            if visual_state_dict:
                logger.info(f"Loading vision weights from {self.pretrained!r} (prefix {prefix!r}).")
                break

        if not visual_state_dict:
            raise ValueError(
                "No 'visual.' / 'model.visual.' keys found in checkpoint "
                f"{self.pretrained!r}. Available keys (first 10): "
                f"{list(cleaned_state_dict.keys())[:10]}"
            )

        # strict=False로 불일치 허용 + 로깅. print()도 같이 쓰는 이유: 학습 런처가
        # logging을 설정하지 않는 경우가 많아 warning이 조용히 사라지고, 그러면
        # "가중치가 거의 안 실린" 인코더로 며칠짜리 학습이 돌아가버린다.
        missing, unexpected = self.load_state_dict(visual_state_dict, strict=False)
        loaded = len(self.state_dict()) - len(missing)
        print(
            f"INFO: vision encoder loaded {loaded}/{len(self.state_dict())} tensors "
            f"from {self.pretrained!r} (missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing}")
            print(f"WARNING: missing keys ({len(missing)}), first 10: {missing[:10]}")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected}")
            print(f"WARNING: unexpected keys ({len(unexpected)}), first 10: {unexpected[:10]}")
        if loaded < 0.5 * len(self.state_dict()):
            raise ValueError(
                f"Only {loaded}/{len(self.state_dict())} vision-encoder tensors were "
                f"filled from {self.pretrained!r} -- the checkpoint almost certainly "
                f"does not match model_config_yaml's architecture."
            )

        self.to(self.device)
        self.train()
        logger.info(f"Model loaded successfully on device: {self.device}")

    def extract_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        inputs = inputs.to(self.device)
        inputs_as_dict = {modality: inputs}

        self.eval()
        with torch.no_grad():
            output = self.forward(inputs_as_dict)

            if isinstance(output, dict) and "pooled" in output:
                features = output["pooled"]
            elif isinstance(output, dict) and "activ" in output:
                features = output["activ"]
            else:
                raise ValueError(f"Unexpected output type: {type(output)}")

        return features.cpu().numpy().astype(np.float32)

    def eval(self):
        """Set model to evaluation mode."""
        if hasattr(self, "model"):
            self.model.eval()
        return self







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
        # breakpoint()
        modal_config = self.model_config["modalities"]

        patch_embed_fn = self.patch_embeds[modality] # image = torch.Size([4, 11, 256, 256, 256])
        patch_embed_xBCDHW = patch_embed_fn(image)   # patch_embed_xBCDHW = torch.Size([4, 384, 32, 32, 32])

        # ``prepare_multiscale_layout()`` and the ``activ`` upsampling in
        # forward() both derive the token grid from the configured patch_size,
        # while the number of tokens actually produced above is fixed by the
        # patch embed's conv strides. If the two disagree the multiscale
        # rearrange fails with an opaque einops error several frames later, so
        # catch it here -- this is the guard for using a checkpoint whose
        # patch_size (8x8x8 in-house vs 8x8x4 open-source Pillar-0) does not
        # match the yaml.
        # patch_size is declared (H, W, D) everywhere, but `image` is
        # (B, C, D, H, W) -- so the zip below is only correct for an isotropic
        # patch. For (8, 8, 4) it computes [D//pH, H//pW, W//pD] = [32, 32, 64],
        # which happens to match ChestCTEmbed3D's own transposed stride
        # (ds1 (2, 2, 1), the anisotropic factor on W instead of D). The two
        # errors cancel, which is why this guard passed for the old arms.
        # ChestCTEmbed3D_Pillar0 fixes the stride, so its (D, H, W) = (64, 32, 32)
        # map needs the expectation stated in the same order. Subclasses that use
        # the corrected stem set `stem_axis_order_corrected = True`; the default
        # keeps the old comparison byte for byte.
        _ps = list(modal_config[modality]["patch_size"])
        if getattr(self, "stem_axis_order_corrected", False):
            pH, pW, pD = _ps
            expected_grid = [image.shape[2] // pD,      # D
                             image.shape[3] // pH,      # H
                             image.shape[4] // pW]      # W
        else:
            expected_grid = [image.shape[2 + i] // _ps[i] for i in range(3)]
        actual_grid = list(patch_embed_xBCDHW.shape[2:])
        if actual_grid != expected_grid:
            raise ValueError(
                f"patch embed for modality {modality!r} produced grid {actual_grid} "
                f"but model_config's patch_size "
                f"{list(modal_config[modality]['patch_size'])} implies "
                f"{expected_grid} for input {list(image.shape)}. Fix patch_size in "
                f"model_config_yaml so it matches the checkpoint "
                f"([8, 8, 8] for the in-house PILLAR0 runs, [8, 8, 4] for the "
                f"open-source Pillar0-ChestCT release)."
            )

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

    def forward(self, x, modality="chest_ct"):
        # breakpoint()
        image = x
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
