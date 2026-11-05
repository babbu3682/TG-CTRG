"""Text-only draw for the ensemble (no visual input at all).

Why this exists
---------------
The ten CTRG-3D draws are one policy family sampled ten ways, so their errors
correlate. This draw is a *different model class*: stock Qwen3-8B + a LoRA
trained on cls/ret templates alone, with no vision tower and no visual tokens.
Its labels come from a channel the other ten do not share, which is the whole
reason it is worth a vote.

It is a VOTER, not a donor of choice. Standalone it scores BLEU 0.19-0.27
against the ensemble's 0.372 because it writes long and repeats; the merge's
donor rule (`min(label count, length)`) therefore almost never picks its
sentences, and the measured effect of adding it is macro-F1 0.6123 -> 0.6126.
Do not promote it to skeleton.

Contract, verbatim from ``src/test_textonly.py`` (the script whose outputs were
scored):

* stock Qwen3-8B tokenizer -- **no** added specials. The CTRG-3D tokenizer
  carries ``<ct>``/``<image_ct*>`` and a resized embedding; reusing it here
  would shift every id.
* ``padding_side="left"`` for batched generation.
* prompt = SYSTEM_PROMPT + build_instruction_textonly, chat-templated with
  ``add_generation_prompt=True`` and ``enable_thinking=False``.
* generation over ``input_ids`` (not ``inputs_embeds``): with no out-of-range
  visual ids there is nothing for ``repetition_penalty``'s gather to trip on,
  and HF then echoes the prompt, so the new tokens must be sliced off.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

# SYSTEM_PROMPT is shared with the MLLM prompt so the two cannot drift.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "petrg"))
from Dataset.ctrate_dataset_train import SYSTEM_PROMPT  # noqa: E402

# Verbatim from PETRG-3D src/Dataset/ctrate_dataset_textonly.py. Copied rather
# than imported because that module pulls in torch datasets and the RVE loader.
BLOCK_LABELS = {
    "cls": "per-abnormality determinations for this study",
    "ret": "sentences retrieved from reports of similar cases -- phrasing reference only",
}

DEFAULT_SAMPLER = dict(temperature=0.7, top_p=0.9, repetition_penalty=1.05,
                       max_new_tokens=2048)
MAX_SEQ = 2432          # measured: train MAX 2301, valid MAX 1961 (design doc G3)


def build_instruction_textonly(cls_template: Optional[str],
                               ret_template: Optional[str]) -> str:
    """User-turn text. Byte-identical to the training/inference builder."""
    blocks = []
    if cls_template:
        blocks.append(f"<cls_template> ({BLOCK_LABELS['cls']})\n{cls_template.strip()}\n</cls_template>")
    if ret_template:
        blocks.append(f"<ret_template> ({BLOCK_LABELS['ret']})\n{ret_template.strip()}\n</ret_template>")

    if not blocks:
        raise ValueError(
            "textonly prompts must carry at least one template block: with no "
            "images and no blocks the model would be asked to write a report "
            "from nothing.")

    return (
        "Write the radiology report for this chest CT.\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        "Rules:\n"
        "- The determinations above are for this study. Report every finding "
        "listed as present, and do not report any finding listed as absent.\n"
        "- Do not copy the wording of the lists. Write in the style of a "
        "radiology report, while preserving exactly which findings are present "
        "and which are absent.\n"
        "- Do not take laterality, segments, sizes, or measurements from the "
        "retrieved sentences; they describe other patients.\n\n"
        "Output exactly two lines, in English:\n"
        "Findings: detailed description of the imaging observations.\n"
        "Impression: concise clinical interpretation and conclusion."
    )


def load_textonly(weights_dir: str | Path, adapter_dir: str | Path,
                  device: str = "cuda"):
    """Stock Qwen3-8B + the text-only LoRA. Returns (model, tokenizer).

    The base LLM is the same ``Qwen3-8B/`` already in weights.zip for CTRG-3D,
    so this ships only the 15 MB adapter.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = Path(weights_dir) / "Qwen3-8B"
    adapter_dir = Path(adapter_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"base LLM not found: {base}")
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / f).is_file():
            raise FileNotFoundError(f"{adapter_dir / f} missing")

    tok = AutoTokenizer.from_pretrained(str(base), use_fast=True)
    # The CTRG-3D tokenizer adds 134 specials; this one must not. If a future
    # sync copies the wrong tokenizer in, every id shifts and the adapter is
    # silently applied to a different vocabulary.
    if any(t.startswith("<image_ct") for t in tok.get_vocab()):
        raise RuntimeError(
            "the tokenizer at %s carries CTRG-3D visual specials; the text-only "
            "adapter was trained on the stock vocabulary" % base)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval().to(device)

    n_lora = sum(1 for k, _ in model.named_parameters() if "lora_" in k)
    if n_lora == 0:
        raise RuntimeError("no lora_* parameters after PeftModel.from_pretrained")
    print(f"[textonly] Qwen3-8B + LoRA ({n_lora} lora_* params) from {adapter_dir.name}",
          flush=True)
    return model, tok


def build_prompt(tok, cls_block: Optional[str], ret_block: Optional[str]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_instruction_textonly(cls_block, ret_block)},
    ]
    kw = dict(conversation=messages, tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(**kw)


@torch.no_grad()
def generate_batch(model, tok, prompts: Sequence[str], sampler: dict,
                   max_seq: int = MAX_SEQ, device: str = "cuda") -> list[str]:
    """One batch. Mirrors src/test_textonly.py's loop exactly."""
    enc = tok(list(prompts), return_tensors="pt", padding=True,
              truncation=True, max_length=max_seq)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        do_sample=True,
        top_p=sampler["top_p"],
        temperature=sampler["temperature"],
        repetition_penalty=sampler["repetition_penalty"],
        max_new_tokens=sampler["max_new_tokens"],
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    # input_ids were passed, so HF echoes the prompt: slice it off.
    new = out[:, enc["input_ids"].shape[1]:]
    if new.shape[1] == 0:
        raise RuntimeError("textonly generate returned no new tokens")
    return tok.batch_decode(new, skip_special_tokens=True)
