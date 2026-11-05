"""VL-CABS-logit backbone: frozen encoder + frozen VL-CABS text-side, exposing 18 raw
CT-RATE-class logits as `outputs['pooled']` for a trainable MLP head to consume.

Motivation: scripts/zeroshot_abn.py's --image-readout phrase_align_phrase --positive-only
sigmoids these same 18 logits directly to get a zero-shot macro AUROC (see CLAUDE.md
"VL-CABS phrase 결과" -- phrase_align_split scored highest among every zero-shot variant tried,
0.8446, beating even the opensource checkpoint's pooled+MLP zero-shot). This backbone lets
`scripts/train.py` train a supervised MLP on top of that same 18-dim feature instead of
just sigmoid'ing it directly, for a fair comparison against the existing pooled-attnpool
route on the identical checkpoint
(configs/ctrate_abnclass_finetune_lz4_pillar_self_phrase_align_split_attnpool.yaml, 1152-dim
input) -- same backbone, same frozen weights, only the readout differs, mirroring how
zeroshot_abn.py itself isolates readout choice from checkpoint choice.

Body (patch_embeds/atlas_stages/forward) reused verbatim from the already-validated
MultimodalAtlasSelf (mmatlas_self.py) -- same reuse pattern as mmatlas_self_jepa.py, to
avoid drift between backbone variants. The text side (text.proj/logit_scale/logit_bias/
temperature loading, VL-CABS attention formula) reuses pillar/utils/vlcabs.py, the same
module scripts/zeroshot_abn.py uses -- single source of truth for the PhraseAlignJepaLoss.get_logits
formula (see that module's docstring for the derivation against pillar-pretrain).

Cost note: the 18 positive CT-RATE class prompts are loaded from
scripts/zeroshot_abn.py's existing prompt cache (raw Qwen embeddings, pre-text.proj) and
projected through this checkpoint's own text.proj ONCE at __init__ -- never re-run per
forward, and Qwen3-Embedding-8B itself is never loaded into this training job. If the cache
is missing or was built from different report_templates sentences, __init__ raises rather
than silently falling back to re-encoding (that's scripts/zeroshot_abn.py's job, run ahead
of time) -- see pillar/utils/vlcabs.py::load_cached_positive_prompts.
"""

import torch
import torch.nn.functional as F

from pillar.utils.logging import logger
from pillar.utils.vlcabs import load_text_tower, load_cached_positive_prompts, vlcabs_cos_sim
from .mmatlas_self import MultimodalAtlas as _MultimodalAtlasSelf

DEFAULT_PROMPT_CACHE = "/data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-pretrain/data/abn_probe_cache/zeroshot_prompt_embeds.pt"


class MultimodalAtlasVLCABS(_MultimodalAtlasSelf):
    def __init__(self, args, prompt_cache=DEFAULT_PROMPT_CACHE, **kwargs):
        # Builds patch_embeds/atlas_stages and loads visual.* from `pretrained` exactly like
        # MultimodalAtlasSelf (strict by default -- see mmatlas_self.py::setup_model).
        super().__init__(args, **kwargs)

        device = self.device
        text_proj, logit_scale, logit_bias, temperature = load_text_tower(args, device)
        for p in text_proj.parameters():
            p.requires_grad_(False)
        text_proj.eval()

        if temperature is None:
            raise ValueError(
                "MultimodalAtlasVLCABS needs the checkpoint's `temperature` parameter (VL-CABS "
                "attention scaling) -- this backbone only makes sense for phrase_align-family "
                "checkpoints. Use MultimodalAtlasSelf + attnpool for non-phrase_align checkpoints."
            )

        raw_pos = load_cached_positive_prompts(args, prompt_cache).to(device)
        with torch.no_grad():
            pos_embeds = F.normalize(text_proj(raw_pos), p=2, dim=-1)  # [18, D], frozen

        # Buffers (not parameters): frozen, but must still move with .to(device)/.cuda()/.eval()
        # calls the same way the encoder's own weights do.
        self.register_buffer("pos_embeds", pos_embeds, persistent=False)
        self.register_buffer("logit_scale", logit_scale, persistent=False)
        self.register_buffer("temperature", temperature, persistent=False)
        self._has_logit_bias = logit_bias is not None
        if self._has_logit_bias:
            self.register_buffer("logit_bias", logit_bias, persistent=False)

        # MultiStage reads .hidden_dim off the backbone in some code paths (e.g. the default
        # pool_kwargs sizing). This backbone's usable output is the 18-dim VL-CABS logit
        # vector, not the 1152-dim pooled/activ features MultimodalAtlasSelf normally
        # advertises -- configs using this backbone must set pool_name: (omitted) and the
        # head's input_dim: 18 explicitly (use_pooled_features: True bypasses self.pool
        # entirely, per codebase pitfall #2 in CLAUDE.md), so this is informational only.
        self.hidden_dim = pos_embeds.shape[0]

        logger.info(
            f"MultimodalAtlasVLCABS: {pos_embeds.shape[0]} positive class prompts loaded, "
            f"text_proj frozen, logit_scale.exp()={logit_scale.exp().item():.3f}"
        )

    def forward(self, x, batch=None):
        outputs = super().forward(x, batch=batch)  # {'activ': ..., 'pooled': ...} from the validated body

        cos_sim = vlcabs_cos_sim(outputs, self.pos_embeds, self.temperature)  # [B, 18]
        bias = self.logit_bias if self._has_logit_bias else 0.0
        # PhraseAlignJepaLoss.get_logits, exactly: logit_scale.exp() * cos_sim [+ logit_bias]. No
        # negative prompt anywhere in this path -- see pillar/utils/vlcabs.py's docstring.
        vlcabs_logits = self.logit_scale.exp() * cos_sim + bias

        outputs["vlcabs_logits"] = vlcabs_logits  # explicit name, for clarity/debugging
        outputs["pooled"] = vlcabs_logits  # what a head with use_pooled_features: True consumes
        return outputs
