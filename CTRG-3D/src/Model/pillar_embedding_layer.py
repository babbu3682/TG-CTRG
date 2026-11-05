"""Multimodal embedding layer used by :class:`Model.CTRG_3D.CTRG3D`.

The layer maintains a learnable embedding table that concatenates:

1. The LLM's own input embeddings (tied via ``self.weight``).
2. Four pairs of learned embeddings for the wrapper tokens
   ``<ct>/</ct>``, ``<region>/</region>`` and
   ``<template>/</template>``.
3. Per-forward CT visual features that fill the positions of the
   dynamic ``<image_ct*>`` placeholder tokens.

The tokenizer IDs and the row order of this logical table are kept in sync, so
the forward pass resolves each position with an embedding lookup / gather
depending on which of the three ranges its ID falls in. (It used to build a
``(B, L, vocab + 6 + num_image_tokens)`` one-hot tensor and matmul it against
the table, which is mathematically identical but costs gigabytes once the
sequence length or the visual-token count grows.)
"""

from typing import Dict, Optional
import yaml
from easydict import EasyDict
import torch
import torch.nn as nn
from einops import rearrange

from .helpers import PerceiverResampler
from .vit_3d import ViT
from .pillar_mmatlas import MultimodalAtlas
import rve

class MyEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pretrained_visual_encoder: Optional[str] = None,
        pretrained_adapter: Optional[str] = None,
        perceiver_num: int = 128,
        core_vocab_size: Optional[int] = None,
        vis_dim: int = 768,
        patch_size: int = 32,
        frame_patch_size: int = 4,
        finetune_ct_feature_extractor: bool = False,
        model_config_yaml: str = None,
    ):
        super().__init__()
        self.finetune_ct_feature_extractor = finetune_ct_feature_extractor
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Row layout of the embedding lookup built in forward():
        #   [0, core_vocab_size)                     -> the LLM's own embeddings
        #   core_vocab_size + 0 .. +1                -> <ct>, </ct>
        #   core_vocab_size + 2 .. +5                -> <cls_template> .. </ret_template>
        #   core_vocab_size + 6 .. +5+num_image_tokens -> this forward's CT features
        # This MUST equal the token ids the dataset's tokenizer hands out, i.e.
        # core_vocab_size == len(tokenizer) *before* the placeholder tokens are
        # added. CTRG3D resizes the LLM's embedding matrix to len(tokenizer)
        # first (Qwen3 ships 151936 rows for a 151669-token vocabulary), which
        # is what makes num_embeddings and len(tokenizer) coincide -- if that
        # resize is ever dropped, every id from <ct> onwards would silently
        # address an unused LLM vocabulary row instead of the visual features,
        # so the id range is asserted in forward() rather than assumed.
        self.core_vocab_size = num_embeddings if core_vocab_size is None else core_vocab_size
        self.num_wrapper_tokens = 6
        self.num_image_tokens = perceiver_num

        # Tied to ``lang_model.get_input_embeddings().weight`` in the parent
        # module, so initialization here is only a placeholder.
        self.weight = nn.Parameter(torch.randn((num_embeddings, embedding_dim)), requires_grad=True)

        # Learnable embeddings for the wrapper special tokens. The order below
        # MUST match the order in which they are added to the tokenizer
        # (``<ct>/</ct>`` then ``<cls_template>/</cls_template>``
        # then ``<ret_template>/</ret_template>``) -- see
        # Model.CTRG_3D.CTRG3D.__init__ and
        # Dataset.ctrate_dataset_train.CTRATEDataset_Train.__init__.
        self.image_token_weight = nn.Parameter(torch.randn((2, embedding_dim)), requires_grad=True)

        # <cls_template>, </cls_template>, <ret_template>, </ret_template>
        self.template_token_weight = nn.Parameter(torch.randn((4, embedding_dim)), requires_grad=True)

        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.vis_dim = vis_dim

        # 3D ViT backbone shared between CT (only the Perceiver + FC)
        # heads are modality-specific).

        with open(model_config_yaml, "r") as f:
            config = yaml.safe_load(f)
        yaml_config = EasyDict(config)
        model_config = yaml_config.vision_cfg.model_config
        # Which tower to build is declared by the yaml, and every yaml that
        # existed before this said "MultimodalAtlas" -- so the old arms keep the
        # old encoder byte for byte. Only ctrate_pillar0_chest_lapa.yaml
        # asks for the corrected stem + single-scale readout; see
        # Model/pillar_mmatlas_lapa.py for what changes and why.
        _tower = yaml_config.vision_cfg.get("model_name", "MultimodalAtlas")
        _common = dict(embed_dim=model_config.model.atlas_args.embed_dim,
                       num_classes=model_config.model.atlas_args.num_classes,
                       image_size=model_config.modalities.chest_ct.image_size,
                       multiscale_feats=True,
                       model_config=model_config,
                       device="cuda",
                       pretrained=model_config.pretrained)
        if _tower == "MultimodalAtlasLAPA":
            from .pillar_mmatlas_lapa import MultimodalAtlasLAPA
            _scale_index = int(yaml_config.vision_cfg.get("scale_index", 1))
            _expect = int(yaml_config.vision_cfg.get("expect_tokens", 0))
            self.vision_encoder = MultimodalAtlasLAPA(scale_index=_scale_index,
                                                            expect_tokens=_expect,
                                                            **_common)
            print(f"INFO: vision tower MultimodalAtlasLAPA "
                  f"(scale_index={_scale_index}, corrected chest_ct stem)")
        elif _tower == "MultimodalAtlas":
            self.vision_encoder = MultimodalAtlas(**_common)
        else:
            # Refusing beats defaulting: a typo here would silently train on a
            # different encoder than the config asked for.
            raise ValueError(
                f"{model_config_yaml}: vision_cfg.model_name={_tower!r} is not a "
                f"known tower (MultimodalAtlas | MultimodalAtlasLAPA)")
        
        if pretrained_visual_encoder:
            # Legacy RadFM-era loading path. The Atlas vision encoder above
            # already self-loads its own checkpoint from
            # ``model_config_yaml``'s ``pretrained:`` field, so this only
            # applies (and should only be set) when overriding that with a
            # separately saved ``visual.*``-prefixed state dict.
            ckpt = torch.load(pretrained_visual_encoder, map_location="cpu", weights_only=False)

            # Lightning / 일반 ckpt 둘 다 대응
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
            else:
                state_dict = ckpt

            # DDP / DataParallel prefix 제거 (앞부분만)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_k = k[len("module."):] if k.startswith("module.") else k
                cleaned_state_dict[new_k] = v

            # visual prefix 추출 (startswith으로 안전하게)
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

            # Load
            missing, unexpected = self.vision_encoder.load_state_dict(visual_state_dict, strict=False)
            if missing:
                print(f"Missing keys ({len(missing)}): {missing}")
            if unexpected:
                print(f"Unexpected keys ({len(unexpected)}): {unexpected}")

        if not self.finetune_ct_feature_extractor:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("INFO: CT feature extractor (vision_encoder) is frozen.")
        else:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            print("INFO: CT feature extractor (vision_encoder) is trainable.")

        self.perceiver = PerceiverResampler(dim=self.vis_dim, num_latents=perceiver_num)
        # NOTE: ``pretrained_adapter`` (RadFM's perceiver/fc weights) is no longer
        # loaded because we increased ``perceiver_num`` from 32 to 128, which is
        # incompatible in shape. The adapter is re-trained from scratch.
        print(
            f"INFO: PerceiverResampler (CT) initialized from scratch with "
            f"{perceiver_num} latents."
        )

        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)


    def _encode_volume(self, vol: torch.Tensor, perceiver: nn.Module) -> torch.Tensor:
        """Encode a (B, 11, H, W, D) volume and return (B, S*num_latents, vis_dim)."""
        # B, S = vol.shape[0], vol.shape[1]
        # feat = rearrange(vol, "b S c h w d -> (b S) c h w d")
        vol = vol.to(torch.bfloat16)
        outputs = self.vision_encoder(vol) # 여기 손봐야함. activ, pooled

        feat = rearrange(outputs['activ'], 'b c d h w -> b (d h w) c').unsqueeze(1).unsqueeze(1)  # Modified for 3D data

        feat = perceiver(feat) # [B, T(=1), 128, 1152]
        feat = feat.squeeze(1) # [B, 128, 1152]

        return feat

    def forward(
        self,
        vision_x: Dict[str, torch.Tensor],
        text_input: torch.Tensor,
    ) -> torch.Tensor:
        """Run the multimodal embedding lookup.

        Args:
            vision_x: ``{"ct_image": tensor}`` where each
                tensor has shape ``(B, S, 3, H, W, D)`` (``S == 1`` in this
                release -- the whole volume is fed as a single chunk).
            text_input: ``(B, L)`` LongTensor of token IDs.

        Returns:
            ``(B, L, embedding_dim)`` mixed textual + visual embeddings suitable
            for ``LLM(inputs_embeds=...)``.
        """
        ct_input = vision_x["ct_image"]

        # VISION
        ct_input = rve.batch_apply_windowing_vectorized(ct_input, windows="all", modality='CT') # windowing, ct_input = [B, C, D, H, W]
        ct_feat = self._encode_volume(ct_input, self.perceiver)
        image_embedding = self.fc(ct_feat)  # (B, num_image_tokens, embedding_dim)

        core = self.core_vocab_size
        n_wrap = self.num_wrapper_tokens
        n_img = image_embedding.shape[1]
        max_id = int(text_input.max().item())
        if max_id >= core + n_wrap + n_img:
            raise ValueError(
                f"token id {max_id} has no embedding row: rows are "
                f"[0, {core}) LLM vocab + {n_wrap} wrapper tokens + {n_img} visual "
                f"slots. Either the dataset's image_num and this layer's "
                f"perceiver_num disagree, or the tokenizer's vocabulary is larger "
                f"than the LLM embedding matrix that was resized to match it."
            )

        # Gather instead of one-hot @ table. The old formulation built a
        # (B, L, core + 6 + n_img) one-hot tensor -- ~3.7 GB at B=4/L=3072 with
        # Qwen3's 151,669-row vocabulary, and linear in both L and the number of
        # visual tokens, which is exactly what we want to scale up. The three
        # branches below reproduce it exactly at a few hundred MB.
        text_embedding = torch.nn.functional.embedding(
            torch.where(text_input < core, text_input, torch.zeros_like(text_input)),
            self.weight,
        )

        # <ct>, </ct> then the four cls/ret template tags -- same order as the
        # tokenizer adds them.
        wrapper_weight = torch.cat([self.image_token_weight, self.template_token_weight], dim=0)
        is_wrapper = (text_input >= core) & (text_input < core + n_wrap)
        wrapper_embedding = torch.nn.functional.embedding(
            (text_input - core).clamp_(0, n_wrap - 1), wrapper_weight
        )

        # Per-sample visual features: id core+n_wrap+k -> image_embedding[b, k].
        is_image = text_input >= core + n_wrap
        image_index = (text_input - (core + n_wrap)).clamp_(0, n_img - 1)
        visual_embedding = torch.gather(
            image_embedding, 1, image_index.unsqueeze(-1).expand(-1, -1, image_embedding.shape[-1])
        )

        out = torch.where(is_wrapper.unsqueeze(-1), wrapper_embedding, text_embedding)
        out = torch.where(is_image.unsqueeze(-1), visual_embedding, out)
        return out  # (B, L, embedding_dim)
