"""PhraseAlign text-tower loading and VL-CABS scoring, shared between scripts/zeroshot_abn.py
(eval-time, positive-vs-negative or --positive-only) and
pillar/models/backbones/mmatlas_self_vlcabs.py (a trainable head built on frozen VL-CABS
logits). Single source of truth for the two pieces that must stay bit-identical between
those two consumers: how the checkpoint's text tower (text.proj/logit_scale/logit_bias/
temperature) is rebuilt, and the VL-CABS attention formula itself.

See scripts/zeroshot_abn.py's module docstring for the full derivation against
pillar-pretrain/src/miniclip/model.py and loss_phrase_align.py.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pillar.metrics.abn_metric import LABEL_COLS
from pillar.utils.logging import logger


def get_prompt_pairs(args):
    """Pull the positive/negative sentence pair for each class out of the config's
    report_templates, ordered to match AbnMetric's LABEL_COLS (which is also the column
    order of the CT-RATE predicted_labels.csv the dataset reads labels from).
    """
    by_name = {entry["name"]: entry for entry in args.report_templates["labels"]}
    missing = [c for c in LABEL_COLS if c not in by_name]
    if missing:
        raise ValueError(f"config report_templates is missing prompts for: {missing}")

    positives = [by_name[c]["positive"] for c in LABEL_COLS]
    negatives = [by_name[c]["negative"] for c in LABEL_COLS]
    return positives, negatives


def resolve_checkpoint_path(args):
    """The backbone classes only ever load `visual.*`, so the text tower has to be pulled from
    the same checkpoint separately. Returns (path, key_prefix)."""
    bk = args.model.kwargs.backbone_kwargs

    pretrained = getattr(bk, "pretrained", None)
    if isinstance(pretrained, str) and pretrained.endswith(".pt"):
        return Path(pretrained), ""

    repo_id = getattr(bk, "model_repo_id", None)
    if repo_id:
        st = Path(repo_id) / "model.safetensors"
        if not st.exists():
            raise FileNotFoundError(f"expected {st} for the HF-packaged checkpoint")
        return st, "model."

    raise ValueError("could not resolve a checkpoint path from backbone_kwargs (need `pretrained: *.pt` or `model_repo_id`)")


def build_text_proj_from_checkpoint(ckpt_path, device="cpu", prefix=None):
    """체크포인트 경로만으로 text.proj 모듈을 복원한다 (config/args 불필요).

    `load_text_tower()`는 config(args)에서 경로를 유도하고 logit_scale/temperature까지
    같이 꺼내는 zero-shot 전용 진입점이라, 경로 하나만 아는 호출자(retrieval 데이터셋,
    gallery 투영 스크립트)가 쓰기엔 맞지 않아 별도로 둔다.

    ⚠️ text.proj의 구조(Linear -> GELU -> Linear, 전부 bias 없음)는 두 함수가 동일하게
    가정한다. miniclip CacheTextEncoder(proj_type='mlp')가 바뀌면 둘 다 고칠 것.

    prefix=None이면 'text.proj.0.weight'를 그대로 찾고, 없으면 'model.' 접두사를
    자동으로 시도한다 (오픈소스 HF 체크포인트가 이 형태).
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(ckpt_path))
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    if prefix is None:
        prefix = "" if "text.proj.0.weight" in sd else "model."
    if prefix:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if "text.proj.0.weight" not in sd:
        raise KeyError(
            f"{ckpt_path} 에서 text.proj 가중치를 찾지 못했습니다 "
            f"(prefix={prefix!r}). 이 체크포인트에 텍스트 타워가 있는지 확인하세요."
        )

    w0, w2 = sd["text.proj.0.weight"], sd["text.proj.2.weight"]
    proj = nn.Sequential(
        nn.Linear(w0.shape[1], w0.shape[0], bias=False),
        nn.GELU(),
        nn.Linear(w2.shape[1], w2.shape[0], bias=False),
    )
    proj[0].weight.data.copy_(w0)
    proj[2].weight.data.copy_(w2)
    proj.to(device).eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    logger.info(f"text.proj 복원: {w0.shape[1]} -> {w0.shape[0]} -> {w2.shape[0]} ({ckpt_path})")
    return proj


def load_text_tower(args, device):
    """Rebuild the checkpoint's text projection (miniclip CacheTextEncoder, proj_type='mlp':
    Linear -> GELU -> Linear, all bias-free) and load its weights, plus logit_scale, logit_bias
    (usually absent) and (phrase_align family only) temperature.

    logit_scale doesn't change AUROC (monotonic per class) but does move the sigmoid, so it
    matters for the threshold-0.5 precision/recall/f1/accuracy numbers.

    temperature is a *different* parameter that only the phrase_align variants have: it divides the
    phrase-vs-token similarities before the softmax in VL-CABS attention, so unlike logit_scale
    it is inside a non-monotonic transform and does change AUROC/ranking for VL-CABS scoring.
    Returned separately for that reason.
    """
    ckpt_path, prefix = resolve_checkpoint_path(args)
    logger.info(f"Loading text tower from {ckpt_path} (key prefix '{prefix}')")

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(ckpt_path))
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    if prefix:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    w0 = sd["text.proj.0.weight"]
    w2 = sd["text.proj.2.weight"]
    proj = nn.Sequential(
        nn.Linear(w0.shape[1], w0.shape[0], bias=False),
        nn.GELU(),
        nn.Linear(w2.shape[1], w2.shape[0], bias=False),
    )
    proj[0].weight.data.copy_(w0)
    proj[2].weight.data.copy_(w2)
    proj.to(device).eval()

    logit_scale = sd.get("logit_scale")
    logit_scale = logit_scale.to(device) if logit_scale is not None else None
    temperature = sd.get("temperature")
    temperature = temperature.to(device) if temperature is not None else None
    # logit_bias: additive term in PhraseAlignJepaLoss.get_logits (logits_per_image = logit_scale *
    # cos_sim + logit_bias). None of the checkpoints used so far have this key (CustomTextCLIP_
    # *_PhraseAlign's init_logit_bias defaults to unset), but load it if present rather than silently
    # dropping it.
    logit_bias = sd.get("logit_bias")
    logit_bias = logit_bias.to(device) if logit_bias is not None else None
    logger.info(
        f"  text.proj: {w0.shape[1]} -> {w0.shape[0]} -> {w2.shape[0]}"
        + (f", logit_scale.exp()={logit_scale.exp().item():.3f}" if logit_scale is not None else ", no logit_scale")
        + (f", temperature.exp()={temperature.exp().item():.4f}" if temperature is not None else ", no temperature")
        + (f", logit_bias={logit_bias.item():.4f}" if logit_bias is not None else ", no logit_bias")
    )
    return proj, logit_scale, logit_bias, temperature


def load_cached_positive_prompts(args, cache_path):
    """Read the raw (pre-text.proj) Qwen embeddings for this config's 18 positive class
    prompts from scripts/zeroshot_abn.py's prompt cache. Strict on purpose: a training job is
    the wrong place to silently fall back to loading Qwen3-Embedding-8B and re-encoding (that's
    scripts/zeroshot_abn.py's job, run once, ahead of time) -- if the cache is missing or was
    built from different sentences, raise with a clear fix instead of paying that cost inline
    or silently training on stale prompts.
    """
    positives, _ = get_prompt_pairs(args)
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"No prompt cache at {cache_path}. This backbone reuses the 18 positive CT-RATE "
            "class prompts scripts/zeroshot_abn.py already encodes with Qwen3-Embedding-8B -- "
            "run that script once (any config sharing this report_templates block) to build "
            "the cache before training with this backbone."
        )
    cached = torch.load(cache_path, map_location="cpu")
    if cached.get("positives") != positives:
        raise ValueError(
            f"Prompt cache at {cache_path} was built from different sentences than this "
            "config's report_templates -- re-run scripts/zeroshot_abn.py to refresh it before "
            "training with this backbone."
        )
    return cached["pos"]  # [18, 4096] raw Qwen embeddings, CPU, pre-text.proj


def normalized_tokens(out, vision_projector=None):
    """[B, N, D] spatial tokens, L2-normalized per token, from a backbone's outputs['activ'].

    outputs['activ'] is (B, C, D, H, W) with the multiscale features already interpolated to a
    common grid and concatenated on C -- the same tensor the pretraining vision towers
    rearrange to (B, N, C) before handing to the phrase_align losses.

    ⚠️ **The flatten order does NOT match** (measured 2026-08-27, contrary to what this
    docstring claimed before). The backbone builds `activ` with
    `b (g2 g0 g1 m2 m0 m1) c -> b c (g2 m2) (g0 m0) (g1 m1)`, and flattening that back with
    `b c d h w -> b (d h w) c` yields `g2 m2 g0 m0 g1 m1` instead of the pretrain tower's
    `g2 g0 g1 m2 m0 m1`. It is the same SET of 1024 tokens in a permuted order.
    That is harmless for everything here -- VL-CABS (softmax over tokens + weighted sum) and
    maxpool are both permutation-invariant, verified numerically at 1.8e-7 max abs diff --
    but any future readout that depends on token position (positional embedding, spatial
    crop, per-region pooling) would be silently wrong. Re-derive the order first.

    `vision_projector` is required for single-scale towers (`lapa`): that variant's
    tower returns one scale's native width (atlas_dim 384) and a separate
    `vision_projector` (384 -> 1152, bias=False) maps it into the shared CLIP width. The
    pretrain path is `F.normalize(vision_projector(visual(image)))` --
    `CustomTextCLIP_LAPA.encode_image`, i.e. **project first, normalize after**, so
    that order is reproduced here rather than normalizing the 384-d tokens.
    """
    tokens = rearrange(out["activ"], "b c d h w -> b (d h w) c")
    if vision_projector is not None:
        tokens = vision_projector(tokens)
    return F.normalize(tokens, p=2, dim=-1)


def build_vision_projector_from_checkpoint(ckpt_path, device="cpu", prefix=None):
    """`vision_projector` (Linear(atlas_dim -> embed_dim, bias=False)) 를 체크포인트에서 복원한다.

    `build_text_proj_from_checkpoint` 와 대칭. 이 텐서는 `visual.` 밖에 살기 때문에 백본
    클래스가 로드하지 않는다 (mmatlas_self_lapa.py 의 docstring 이 명시적으로
    "intentionally NOT loaded" 라고 적은 그 텐서다 -- abn head 는 인코더 native feature 를
    쓰지만, VL-CABS 는 텍스트와 같은 공간이어야 하므로 zero-shot 에서는 반드시 필요하다).
    """
    import torch.nn as nn

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if prefix is None:
        prefix = "module." if any(k.startswith("module.") for k in sd) else ""
    key = f"{prefix}vision_projector.weight"
    if key not in sd:
        raise RuntimeError(
            f"{ckpt_path} has no `{key}`. Only the single-scale lapa/phrase_align_polar "
            f"checkpoints carry a vision_projector; available non-visual keys: "
            f"{[k for k in sd if 'visual' not in k]}"
        )
    w = sd[key]
    proj = nn.Linear(w.shape[1], w.shape[0], bias=False)
    proj.weight.data.copy_(w)
    return proj.to(device).eval()


def vlcabs_cos_sim(out, prompts, temperature, vision_projector=None):
    """VL-CABS: each prompt attends independently over the volume's spatial tokens
    (temperature-scaled softmax), and the attention-aggregated, re-normalized vector is dotted
    with that same prompt -- raw cosine similarity, [B, C_prompts]. Caller applies
    logit_scale/logit_bias on top to reproduce PhraseAlignJepaLoss.get_logits exactly. Mirrors the
    per-phrase branch of CustomTextCLIP_PhraseAlign.forward in pillar-pretrain.
    """
    if temperature is None:
        raise ValueError(
            "VL-CABS scoring needs the checkpoint's `temperature` parameter, which only the "
            "phrase_align-family checkpoints have."
        )
    tokens = normalized_tokens(out, vision_projector=vision_projector)  # [B, N, D]
    if tokens.shape[-1] != prompts.shape[-1]:
        raise RuntimeError(
            f"VL-CABS dim mismatch: image tokens are {tokens.shape[-1]}-d but prompts are "
            f"{prompts.shape[-1]}-d. Single-scale towers (lapa) need the "
            f"checkpoint's vision_projector -- pass --apply-vision-projector."
        )
    attn = torch.einsum("bnd,cd->bcn", tokens, prompts) / temperature.exp()
    weights = attn.softmax(dim=-1)                                     # [B, C, N]
    agg = torch.einsum("bcn,bnd->bcd", weights, tokens)                # [B, C, D]
    agg = F.normalize(agg, p=2, dim=-1)
    return (agg * prompts.unsqueeze(0)).sum(dim=-1)
