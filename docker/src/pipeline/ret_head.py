"""Retrieval (Simple_TranSQ) head -> per-sample ret_report template entry.

Reproduces pillar-finetune's
``CTRATE_FT_Atlas_retrieval_self_lapa/templates_ep32_pp`` — the ret
template set every GRPO draw was prompted with
(``generate_batched.py`` SPLIT_PATHS["valid_newenc"]["ret"]).

WHAT CHANGED FROM THE OPEN-SOURCE-ENCODER VERSION, AND WHY
----------------------------------------------------------
1. ``input_dim`` 1152 -> **384**. Not a tuning choice: the lapa tower
   emits one scale at native resolution, so ``activ`` is (B, 384, 16, 8, 8) and
   ``input_proj``'s in_channels is forced to 384
   (configs/ctrate_retrieval_finetune_lz4_pillar_self_lapa.yaml:228).
2. checkpoint -> ``CTRATE_FT_Atlas_retrieval_self_lapa/checkpoints/best.ckpt``
   (= epoch 32), confirmed by that run's ``reportgen_best_ep32_train/inference.log``.
   NOT epoch=36 — ep36 templates belong to the newtemplate2 SFT arm.
3. **fp32, no autocast.** The previous version ran the head under
   ``torch.amp.autocast("cuda")`` (fp16 by default). Both an fp32 and an fp16
   inference run of this very checkpoint survive on disk, so the question was
   settled by replaying ``apply_display_overrides`` on each and diffing against
   the shipped templates:
       reportgen_best_ep32       (fp32)  1552/1552 exams reproduce
       reportgen_best_ep32_fp16          955/1552
   Keeping the autocast would have silently changed the ret template on 38% of
   exams — the near-duplicate gallery makes argmax ties dtype-sensitive.
4. ``_pp`` post-processing is applied HERE. The scored recipe feeds CTRG-3D the
   post-processed phrases (``dev_ret_reports_pp.json``), not the raw ones.

Gallery and threshold are unchanged from the previous container:
``gallery_embeddings_lasthidden.pt`` + ``gallery_sentences.pt``, threshold 0.4.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from pillar_min import Simple_TranSQ

# From configs/ctrate_retrieval_finetune_lz4_pillar_self_lapa.yaml:222-235.
RET_HEAD_KWARGS = dict(
    input_dim=384,
    d_target=4096,
    num_queries=64,
    num_decoder_layers=6,
    aux_loss=False,
)
RET_PROB_THRESHOLD = 0.4          # inference.log: "Threshold(s): [0.4]"
HEAD_PREFIX = "head_models.retrieval."

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DISPLAY_OVERRIDES = ASSETS_DIR / "display_overrides.json"


def load_ret_head(ckpt_path: str | Path, device: str = "cuda"):
    """Load only the Simple_TranSQ tensors out of a checkpoint.

    Accepts the full training checkpoint (``best.ckpt``, with a "model" key) or
    a pre-stripped head-only state dict (make_weights.sh strips it so
    weights.zip does not carry optimizer state / a duplicated backbone).
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    head_state = {k[len(HEAD_PREFIX):]: v for k, v in state.items()
                  if k.startswith(HEAD_PREFIX)}
    if not head_state:
        head_state = dict(state)       # already stripped

    head = Simple_TranSQ(args=None, **RET_HEAD_KWARGS)
    head.load_state_dict(head_state, strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head.to(device)


def load_gallery(weights_dir: str | Path, device: str = "cuda"):
    """gallery_embeddings_lasthidden.pt (N, 4096) + gallery_sentences.pt (N strs)."""
    weights_dir = Path(weights_dir)
    embeds = torch.load(weights_dir / "gallery_embeddings_lasthidden.pt",
                        map_location="cpu", weights_only=False)
    if not torch.is_tensor(embeds):
        embeds = torch.as_tensor(embeds)
    embeds = F.normalize(embeds.float(), p=2, dim=-1).to(device)

    texts = torch.load(weights_dir / "gallery_sentences.pt", weights_only=False)
    if len(texts) != embeds.size(0):
        raise ValueError(f"gallery size mismatch: {len(texts)} sentences vs "
                         f"{embeds.size(0)} embeddings")
    return embeds, texts


def load_display_overrides(path: str | Path = DISPLAY_OVERRIDES) -> dict:
    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)
    if not overrides:
        raise ValueError(f"display_overrides is empty: {path}")
    return overrides


def apply_display_overrides(phrases, overrides: dict) -> list[str]:
    """Verbatim port of pillar-finetune
    ``scripts/ret_reports_to_reportgen_csv.apply_display_overrides``.

    Strip the trailing period, substitute surface forms, then de-duplicate.
    De-duplication is deliberately conditional on ``overrides`` being non-empty:
    before substitution two distinct gallery sentences carry information, after
    it several surface forms collapse onto one phrase and genuinely repeat.
    """
    cleaned = []
    for p in phrases:
        p = (p or "").strip().rstrip(".").strip()
        if p:
            cleaned.append(p)
    if overrides:
        cleaned = [overrides.get(s, s) for s in cleaned]
        cleaned = list(dict.fromkeys(cleaned))
    return cleaned


@torch.no_grad()
def build_ret_entry(head, gallery_embeds, gallery_texts, activ: torch.Tensor,
                    sample_name: str, overrides: dict, device: str = "cuda",
                    prob_threshold: float = RET_PROB_THRESHOLD) -> dict:
    """activ: (1, 384, 16, 8, 8) fp32 CPU tensor -> post-processed ret_report row.

    Schema matches ``dev_ret_reports_pp.json``, which is what CTRG-3D's
    ``_load_key_phrase_template`` consumed for every draw.

    fp32 on purpose — see the module docstring; do not wrap this in autocast.
    """
    vis = activ.to(device).float()
    sent_feats, topic_preds = head(vis)
    if sent_feats.dim() == 4:          # aux_loss checkpoints: final decoder layer
        sent_feats, topic_preds = sent_feats[-1], topic_preds[-1]

    phrase_lists = head.retrieve(
        sent_feats.float(), topic_preds.float(),
        gallery_embeds, gallery_texts, prob_threshold=prob_threshold,
    )
    raw = phrase_lists[0]
    pp = apply_display_overrides(raw, overrides)
    prob = topic_preds.float().sigmoid()
    return {
        "accession": str(sample_name),
        "pred_key_phrases": pp,
        "pred_threshold": prob_threshold,
        "pred_num_fired": int((prob[0] > prob_threshold).sum().item()),
        "pred_num_phrases_before_pp": len(raw),
    }
