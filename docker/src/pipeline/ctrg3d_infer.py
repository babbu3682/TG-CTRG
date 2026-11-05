"""CTRG-3D stage B — batched, cache-fed generation with hot-swapped LoRA draws.

This is a port of PETRG-3D ``scripts/generate_batched.py``, which is the script
that produced every draw the submitted ensemble is built from. It replaces the
previous container's batch-1 ``src/test.py`` path.

THREE THINGS MAKE THE ENSEMBLE AFFORDABLE
-----------------------------------------
1. **The visual features are computed once** (stage A', ``visual_cache.py``) and
   fed in through a replacement input-embedding module. GRPO trained LoRA on
   ``lang_model`` only, so all ten draws share the same 128x4096 tensor.
2. **The base model is loaded once** and each draw only overwrites the 144
   ``lora_*`` tensors (15.3 MB). The other 2.48 GB of an ``adapter_model.safetensors``
   is ``embed_tokens``/``lm_head``, frozen during RL and therefore identical to
   the base — loading it would only risk clobbering the wrapper.
3. **Generation is batched** with left padding.

FAITHFULNESS NOTES — each of these was a bug in an earlier attempt upstream
--------------------------------------------------------------------------
* ``max_seq`` is **2688**, ``generate_batched.py``'s default, because
  ``run_sbatch_grpo_score_newencB`` never passes ``--max_seq``. The previous
  container used 3072; that truncates the prompt somewhere else.
* ``inputs_embeds``, never ``input_ids``. The prompt carries visual ids above
  the tied ``lm_head``'s width, and ``repetition_penalty`` gathers over the
  whole sequence — passing ids trips a device-side assert.
* Left padding, and ``pad_token_id`` must stay below ``core_vocab_size`` or
  padded positions get counted as visual slots.
* The prompt carries a ``Case %06d.`` marker, because both GRPO training and
  the draw generation inserted one (``grpo_crg.CASE_MARKER``). The policy has
  only ever seen prompts with it. The number is a row index and was arbitrary
  with respect to content upstream too, so here it is the volume's position in
  the sorted input list.
* Sampler defaults are ``test.py``'s: do_sample, top_p 0.9, temperature 0.7,
  repetition_penalty 1.05, max_new_tokens 2048, seed 42.

WHAT IS DELIBERATELY SIMPLER THAN UPSTREAM
------------------------------------------
``generate_batched.py`` identifies a prompt's cache row by hashing the 16
tokens after the visual block, because its dataset and its cache are built by
separate programs. Here one loop produces both, so the row travels with the
sample and the key-window machinery (and its collision failure mode) is gone.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

CASE_MARKER = "</ct>\nCase {row:06d}.\n"    # grpo_crg.CASE_MARKER, verbatim

DEFAULT_MAX_SEQ = 2688
DEFAULT_VIS_DIM = 384
DEFAULT_SAMPLER = dict(temperature=0.7, top_p=0.9, repetition_penalty=1.05,
                       max_new_tokens=2048)


class CachedVisualEmbedding(nn.Module):
    """Drop-in for ``embed_tokens``: fills the visual slots from a cache array.

    Reproduces the id layout ``MyEmbedding.forward`` builds, minus the encoder:

        [0, core)                   the LLM's own vocabulary -> inner embedding
        [core, core+n_wrap)         wrapper tokens           -> wrapper parameter
        [core+n_wrap, ...)          128 visual slots         -> cache[row]

    ``set_rows`` names the cache row for each row of the next forward; decode
    steps after the prefill carry no visual ids and fall through untouched.
    """

    def __init__(self, inner: nn.Module, wrapper_weight: torch.Tensor,
                 cache: np.ndarray, core: int, n_wrap: int, n_vis: int):
        super().__init__()
        self.inner = inner
        self.register_buffer("wrapper_weight", wrapper_weight.clone(),
                             persistent=False)
        self.cache = cache                      # np.memmap [N, n_vis, hidden]
        self.core, self.n_wrap, self.n_vis = core, n_wrap, n_vis
        self.vis_lo = core + n_wrap
        self._rows: Optional[Sequence[int]] = None
        self.n_injected = 0

    # the tied lm_head reads these off the input embedding
    @property
    def weight(self):
        return self.inner.weight

    @property
    def num_embeddings(self):
        return self.inner.num_embeddings

    @property
    def embedding_dim(self):
        return self.inner.embedding_dim

    def set_rows(self, rows: Sequence[int]) -> None:
        self._rows = list(rows)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        safe = input_ids.clamp_max(self.core - 1)
        emb = self.inner(safe)

        is_wrapper = (input_ids >= self.core) & (input_ids < self.vis_lo)
        if is_wrapper.any():
            idx = (input_ids - self.core).clamp(0, self.n_wrap - 1)
            emb = torch.where(
                is_wrapper.unsqueeze(-1),
                nn.functional.embedding(idx, self.wrapper_weight.to(emb.dtype)),
                emb)

        is_visual = input_ids >= self.vis_lo
        if not is_visual.any():
            return emb
        if self._rows is None:
            raise RuntimeError(
                "visual placeholders present but set_rows() was never called — "
                "the batch would be embedded blind")
        if len(self._rows) != input_ids.shape[0]:
            raise RuntimeError(
                f"set_rows() named {len(self._rows)} rows for a batch of "
                f"{input_ids.shape[0]}")

        for b in range(input_ids.shape[0]):
            pos = torch.nonzero(is_visual[b], as_tuple=False).flatten()
            if pos.numel() == 0:
                continue
            if pos.numel() != self.n_vis:
                raise RuntimeError(
                    f"row {b} holds {pos.numel()} visual placeholders, expected "
                    f"{self.n_vis}; the prompt and image_num disagree")
            slot = (input_ids[b, pos] - self.vis_lo).to(torch.long)
            feats = torch.as_tensor(np.asarray(self.cache[self._rows[b]]))
            emb[b, pos] = feats.to(emb.device, emb.dtype)[slot]
            self.n_injected += 1
        return emb


def load_ctrg3d(weights_dir: str | Path,
                cache: np.ndarray,
                num_visual_tokens: int = 128,
                vis_dim: int = DEFAULT_VIS_DIM,
                max_seq: int = DEFAULT_MAX_SEQ,
                use_fast: bool = False,
                enable_thinking: bool = False,
                device: str = "cuda"):
    """Build CTRG3D, strict-load the checkpoint, and wire in the visual cache.

    Returns ``(lang_model, prompt_builder, cached_embedding)``. The vision tower
    inside the checkpoint is deleted after loading: stage A' already produced
    everything it would have computed, and it is ~725 MB of VRAM.
    """
    from safetensors.torch import load_file as load_safetensors
    from Model.CTRG_3D import CTRG3D

    weights_dir = Path(weights_dir)
    lang_dir = weights_dir / "Qwen3-8B"
    ckpt_path = weights_dir / "ctrg3d" / "model.safetensors"

    yaml_path = _patched_encoder_yaml()

    # CTRG3D.__init__ loads the LLM in fp32 and the caller casts to bf16 right
    # after (PETRG-3D does the same fp32->bf16 dance). Loading bf16 directly is
    # numerically identical (the shards are bf16 on disk) and halves peak host
    # RAM, which matters under the platform's container memory limit.
    import transformers
    _orig = transformers.AutoModelForCausalLM.from_pretrained.__func__

    def _bf16_from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        kwargs.setdefault("low_cpu_mem_usage", True)
        return _orig(cls, *args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = classmethod(_bf16_from_pretrained)
    try:
        model = CTRG3D(
            text_tokenizer_path=str(lang_dir),
            lang_model_path=str(lang_dir),
            pretrained_visual_encoder="",
            pretrained_adapter="",
            use_fast=use_fast,
            enable_thinking=enable_thinking,
            model_config_yaml=yaml_path,
            vis_dim=vis_dim,
            image_num=num_visual_tokens,
        )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = classmethod(_orig)

    model = model.bfloat16()
    print(f"[ctrg3d] loading checkpoint {ckpt_path}", flush=True)
    model.load_state_dict(load_safetensors(str(ckpt_path), device="cpu"),
                          strict=True)

    core = model.core_vocab_size
    n_wrap = model.embedding_layer.num_wrapper_tokens
    tok = model.text_tokenizer
    tok.padding_side = "left"                 # non-negotiable for batched decode
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if not tok.pad_token_id < core:
        raise RuntimeError(
            f"pad_token_id {tok.pad_token_id} is not below core_vocab_size "
            f"{core}; padded positions would be read as visual slots")

    lm = model.lang_model                     # PeftModel, LoRA already attached
    wrapper_w = torch.cat([model.embedding_layer.image_token_weight,
                           model.embedding_layer.template_token_weight],
                          dim=0).detach()
    cached = CachedVisualEmbedding(
        lm.get_base_model().get_input_embeddings(), wrapper_w, cache,
        core, n_wrap, num_visual_tokens)
    lm.get_base_model().set_input_embeddings(cached)

    del model.embedding_layer.vision_encoder
    torch.cuda.empty_cache()

    lm = lm.to(device)
    lm.eval()

    builder = _PromptBuilder(model, max_seq=max_seq,
                             num_visual_tokens=num_visual_tokens,
                             enable_thinking=enable_thinking)
    return lm, builder, cached


def _patched_encoder_yaml() -> str:
    """lapa yaml with ``pretrained: None``.

    The tower is deleted right after the checkpoint load, but CTRG3D still
    builds it, and letting it self-load a 725 MB file we are about to throw away
    is pure latency.
    """
    import tempfile

    import yaml

    src = (Path(__file__).resolve().parent.parent / "vendor" / "petrg" / "Model"
           / "ctrate_pillar0_chest_lapa.yaml")
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["vision_cfg"]["model_config"]["pretrained"] = "None"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                      prefix="ctrg3d_encoder_", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def load_lora_adapter(lm, adapter_path: str | Path) -> int:
    """Overwrite the live ``lora_*`` tensors from a saved adapter slice.

    Verbatim port of ``generate_batched.py``'s adapter block, checks included:
    a tensor with no destination, a shape mismatch, or a count that differs from
    the live LoRA parameter count all raise. A partially loaded adapter would
    generate from a silent mixture of the RL and SFT policies — which is exactly
    the failure this ensemble cannot detect downstream, because every draw would
    still look like a plausible report.
    """
    from safetensors.torch import load_file as load_safetensors

    adapter_path = Path(adapter_path)
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    raw = load_safetensors(str(adapter_path), device="cpu")
    lora = {k: v for k, v in raw.items() if "lora_" in k}
    if not lora:
        raise RuntimeError(f"{adapter_path} holds no lora_* tensors")

    own = {k for k, _ in lm.named_parameters() if "lora_" in k}
    sd = dict(lm.named_parameters())
    loaded, missing = 0, []
    with torch.no_grad():
        for k, v in lora.items():
            # PEFT saves "...lora_A.weight"; the live module is
            # "...lora_A.default.weight". Normalise, then require a hit.
            cand = [k, k.replace(".lora_A.weight", ".lora_A.default.weight")
                        .replace(".lora_B.weight", ".lora_B.default.weight")]
            tgt = next((c for c in cand if c in sd), None)
            if tgt is None:
                missing.append(k)
                continue
            if tuple(sd[tgt].shape) != tuple(v.shape):
                raise RuntimeError(f"{tgt}: live {tuple(sd[tgt].shape)} vs "
                                   f"saved {tuple(v.shape)}")
            sd[tgt].copy_(v.to(sd[tgt].device).to(sd[tgt].dtype))
            loaded += 1
    if missing:
        raise RuntimeError(
            f"{len(missing)} adapter tensors had no destination, e.g. "
            f"{missing[:3]}. The live LoRA layout differs from the saved one — "
            f"refusing to generate from a partially loaded adapter.")
    if loaded != len(own):
        raise RuntimeError(
            f"loaded {loaded} LoRA tensors but the model has {len(own)}; a "
            f"partial load would mix the RL and SFT policies.")
    return loaded


class _PromptBuilder:
    """Builds the prompt exactly like ``CTRATEDataset_Test.__getitem__``.

    Reuses the model's own tokenizer, which already carries the special tokens
    in the canonical order, and then inserts the case marker the cached-vision
    path has always carried.
    """

    def __init__(self, model, max_seq: int, num_visual_tokens: int,
                 enable_thinking: bool):
        from Dataset.ctrate_dataset_train import SYSTEM_PROMPT, build_instruction
        self._system_prompt = SYSTEM_PROMPT
        self._build_instruction = build_instruction
        # Pin the CTRG3D wrapper alive. load_ctrg3d returns only lang_model, and
        # although the LLM survives on its own reference, letting the parent be
        # collected mid-run would also drop the tokenizer's owner. Cheap
        # insurance against a lifetime bug that would only show up under memory
        # pressure, hours into a run.
        self._model = model
        self.tokenizer = model.text_tokenizer
        self.image_padding_tokens = model.image_padding_tokens
        self.max_seq = max_seq
        self.enable_thinking = enable_thinking

    def __call__(self, cls_template: Optional[str], ret_template: Optional[str],
                 row: int) -> str:
        instruction = self._build_instruction(cls_template=cls_template,
                                              ret_template=ret_template)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",
             "content": f"<ct>{self.image_padding_tokens[0]}</ct>\n{instruction}"},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                conversation=messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=self.enable_thinking)
        except Exception:
            text = self.tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True)

        if "</ct>\n" not in text:
            raise RuntimeError("prompt has no '</ct>' boundary to anchor the "
                               "case marker to")
        text = text.replace("</ct>\n", CASE_MARKER.format(row=row), 1)

        n_tokens = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens > self.max_seq:
            raise RuntimeError(
                f"prompt is {n_tokens} tokens, over max_seq {self.max_seq}; "
                f"truncation would push the visual slots out of the window")
        return text


def render_cls_template(cls_entry: Optional[dict]) -> Optional[str]:
    """cls_report row -> newline-joined sentence block (or None if empty)."""
    if not cls_entry:
        return None
    sentences = [s.strip() for s in cls_entry.get("report", []) if s and s.strip()]
    return "\n".join(sentences) if sentences else None


def render_ret_template(ret_entry: Optional[dict]) -> Optional[str]:
    """ret_report row -> newline-joined pred_key_phrases block (or None)."""
    if not ret_entry:
        return None
    phrases = [p.strip() for p in ret_entry.get("pred_key_phrases", [])
               if p and p.strip()]
    return "\n".join(phrases) if phrases else None


@torch.no_grad()
def generate_batch(lm, cached: CachedVisualEmbedding, tokenizer,
                   prompts: Sequence[str], rows: Sequence[int],
                   sampler: dict, device: str = "cuda") -> list[str]:
    """Generate one batch of reports. Returns raw decoded strings."""
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True,
                    add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    cached.set_rows(rows)
    embeds = lm.get_base_model().get_input_embeddings()(ids)
    out = lm.generate(
        inputs_embeds=embeds, attention_mask=mask,
        do_sample=True,
        top_p=sampler["top_p"],
        temperature=sampler["temperature"],
        repetition_penalty=sampler["repetition_penalty"],
        max_new_tokens=sampler["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # With inputs_embeds and no input_ids, HF returns ONLY the new tokens. Assert
    # it rather than assume: a transformers change that starts echoing the prompt
    # would otherwise prepend garbage to every prediction.
    if out.shape[1] == 0:
        raise RuntimeError("generate returned no new tokens")
    if out.shape[1] > sampler["max_new_tokens"]:
        raise RuntimeError(
            f"generate returned {out.shape[1]} tokens with max_new_tokens="
            f"{sampler['max_new_tokens']} — it is echoing the "
            f"{ids.shape[1]}-token prompt")
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def postprocess_report(text: str) -> str:
    """Challenge-format cleanup.

    Generation used inputs_embeds, so the decode holds only new tokens (already
    skip_special_tokens=True). Defensively strip a leaked ``<think>`` block:
    enable_thinking=False normally leaves an empty one that the special-token
    filter removes.
    """
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text
