#!/usr/bin/env python3
"""Batched report generation from the offline vision cache -- for scoring GRPO
checkpoints (and any SFT checkpoint the cache was built from) on the full valid
split without paying test.py's batch-1 cost.

    # baseline reference (no RL adapter) -- score this too, or the comparison is not paired
    python scripts/generate_batched.py --out results/grpo_pilot/base.csv

    # a GRPO checkpoint
    python scripts/generate_batched.py \
        --adapter outputs/grpo_crg_f1bleu/checkpoint-500 \
        --out results/grpo_pilot/step500.csv

The CSV columns are exactly test.py's (SampleName / Question / GT_report /
Pred_report), so the existing, already-validated scorer runs on it unchanged:

    python src/evaluate_reportgen.py --results_csv <csv> --output_dir <dir> \
        --radbert_model_path ... --radbert_dir ...

WHY THIS FILE EXISTS
--------------------
The GRPO pilot (job 28075386) finished 500 steps and left checkpoint-100..500,
but "how much did it gain over the baseline?" could not be answered: scoring a
checkpoint meant test.py at batch 1, measured 5:52:30 for the 1,552-volume valid
split. Five checkpoints plus the reference = ~35 h, which is why the design
doc's 100-step gate never fired. Batch 32 generation was measured at 0.75 s/seq
(docs/2026-08-21-grpo-crg-design.md 6), i.e. ~20 min per checkpoint.

Nothing in src/ is touched -- src/test.py and src/train.py are left alone, per
the grpo_crg keyword convention.

WHAT MAKES THIS CORRECT RATHER THAN JUST FAST
---------------------------------------------
1. GENERATION SETTINGS ARE test.py's, VERBATIM. CTRG3D.generate uses
   do_sample=True, top_p=0.9, temperature=0.7, repetition_penalty=1.05,
   max_new_tokens=2048 (Model/CTRG_3D.py, enable_thinking=False branch). The
   baseline numbers this run is compared against (CRG 0.4551 / macro-F1 0.4528 /
   BLEU_mean 0.3194, results/ep15best/summary.csv) were produced with those, so
   they are reproduced here exactly. Do not "improve" them: a different sampler
   makes the delta meaningless. max_new_tokens=2048 costs nothing in practice --
   ep15best predictions max out at 756 tokens -- because generate stops once
   every sequence in the batch has emitted EOS.

2. LEFT PADDING. Batched decoder-only generation needs the prompts
   right-aligned, or the model continues from pad tokens. tokenizer.padding_side
   is forced to "left" here and asserted after generation by checking that the
   sliced-off prefix length equals the padded prompt width for every row.

3. PADDING CANNOT BE MISTAKEN FOR A VISUAL SLOT. CachedVisualEmbedding treats
   ids in [core+6, core+6+n_vis) as visual placeholders and raises if a sequence
   does not hold exactly n_vis of them. Verified for this tokenizer:
       core_vocab_size = 151,669   visual ids = 151,675 .. 151,802
       pad_token_id    = 151,643   -> below core, takes the ordinary path
   So padding is safe. If a future tokenizer moves pad above core this assert
   fires rather than silently shifting the visual slots.

4. ONLY THE LoRA KEYS ARE LOADED FROM THE ADAPTER. GRPOTrainer saved
   adapter_model.safetensors with save_embedding_layers=True, so it holds 144
   lora_* tensors (3,833,856 params -- exactly the trainable count the run
   reported) plus embed_tokens.inner.weight and lm_head.weight (1.24 B params).
   Those two were FROZEN during RL, so they are identical to the base
   checkpoint's and loading them would only risk clobbering the wrapper. They
   are skipped, and the skip is checked: the LoRA key count must match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_HERE))

# The cached-embedding machinery is shared with the RL script on purpose: if the
# prompt or the key-window logic ever diverges between training and scoring, the
# cache lookup silently returns the wrong volume's features.
from grpo_crg import CASE_MARKER, KEY_WINDOW, CachedVisualEmbedding, build_key_lookup

# The baseline's cls/ret template set. These MUST be the ones checkpoint-16412
# was trained with (qwen3_joint_template.sh), not the lapa ones -- the
# prompt has to match what the cache was built from or build_key_lookup raises.
CLS_DEV = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
           "CTRATE_ABN_cls_report_templates_p90n20/cls_report_dev.json")
RET_DEV = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
           "CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden/valid_ret_reports.json")
VALID_JSON = ("/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/"
              "merged_valid_filtered_prob_half.json")
VALID_RVE = ("/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/"
             "ctrate_cache_1.25mm_lz4/valid/mapping_path.csv")

# `--split train` was already in `choices` but every dataset path above is the
# valid one, so it would have read the TRAIN cache while building VALID prompts
# and build_key_lookup would have raised (or worse, matched the wrong volume).
# These are the train counterparts, taken verbatim from grpo_crg.py's dataset
# construction so the prompt is byte-identical to the one the RL path builds.
# `--split valid` is untouched: SPLIT_PATHS["valid"] is exactly the four
# constants this file already used.
TRAIN_JSON = ("/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/"
              "merged_train_filtered_prob_half.json")
TRAIN_RVE = ("/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/"
             "ctrate_cache_1.25mm_lz4/train/mapping_path.csv")
CLS_TRAIN = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
             "CTRATE_ABN_cls_report_templates_p90n20/cls_report_train.json")
RET_TRAIN = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
             "CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden/train_ret_reports.json")

# The newenc (lapa) SFT runs were trained on DIFFERENT cls/ret templates
# and, for the gtweight arm, a slightly smaller split. Scoring or RL-ing those
# checkpoints through the old-template prompts is a silent distribution shift --
# the policy would see prompts it never trained on. These two entries are purely
# additive: "valid"/"train" above are untouched, so every existing invocation is
# byte-identical.
_NEW_CLS = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
            "CTRATE_ABN_cls_report_templates_lapa_f1max")
_NEW_RET = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
            "CTRATE_FT_Atlas_retrieval_self_lapa/templates_ep32_pp")
_GTCOV = "/data/drdcad/sunggu/0.Challenge/PETRG-3D/data/sentence_labels"

# The FINAL SFT runs (newtemplate2, gtw10 / gtw30) trained on the OPTIMISED
# template set -- maxmean_f1max + templates_ep36_pp -- which is neither the
# baseline p90n20 set nor the _NEW_* (f1max / ep32_pp) set above. Scoring them
# through either of those is a silent prompt shift, so they get their own key.
_NT2_CLS = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
            "CTRATE_ABN_cls_report_templates_lapa_maxmean_f1max")
_NT2_RET = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
            "CTRATE_FT_Atlas_retrieval_self_lapa/templates_ep36_pp")

# "cache" is the basename of the cached-feature files, which is NOT the same
# string as the split key. build_vision_cache.py only ever writes "train"/"valid"
# (its --split choices), so a key like "train_newenc" -- which selects a prompt
# template set, not a different cache -- still reads train.npy / train_meta.json.
# Conflating the two is what made the 2026-08-26 reference job die on a missing
# "train_newenc_meta.json" after the 36-minute cache stage had already succeeded.
SPLIT_PATHS = {
    "valid": {"cache": "valid",
              "json": VALID_JSON, "rve": VALID_RVE, "cls": CLS_DEV, "ret": RET_DEV},
    "train": {"cache": "train",
              "json": TRAIN_JSON, "rve": TRAIN_RVE, "cls": CLS_TRAIN, "ret": RET_TRAIN},
    # newenc B (gtweight) -- the gtcov split it was actually trained on.
    "train_newenc": {"cache": "train",
                     "json": f"{_GTCOV}/merged_train_gtcov.json", "rve": TRAIN_RVE,
                     "cls": f"{_NEW_CLS}/cls_report_train.json",
                     "ret": f"{_NEW_RET}/train_ret_reports_pp.json"},
    # Scoring stays on the STANDARD full valid 1,552 so the numbers compare with
    # every other case in results/all_experiments_ranking.csv.
    "valid_newenc": {"cache": "valid",
                     "json": VALID_JSON, "rve": VALID_RVE,
                     "cls": f"{_NEW_CLS}/cls_report_dev.json",
                     "ret": f"{_NEW_RET}/dev_ret_reports_pp.json"},
    # newtemplate2 final SFT runs (gtw10 / gtw30). Scoring stays on the STANDARD
    # full valid 1,552 even though these trained on the gtcov split, so the
    # numbers compare with every other case in the ranking.
    "valid_nt2": {"cache": "valid",
                  "json": VALID_JSON, "rve": VALID_RVE,
                  "cls": f"{_NT2_CLS}/cls_report_dev.json",
                  "ret": f"{_NT2_RET}/dev_ret_reports_pp.json"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(
        _REPO / "scripts/outputs/ep15/ctrg3d-qwen3-8B-pillar0os-lz4-vt128/checkpoint-16412"),
        help="the SFT checkpoint the cache was built from")
    ap.add_argument("--adapter", default="",
                    help="GRPO checkpoint dir; omit to score the SFT policy itself")
    ap.add_argument("--cache_dir", default=str(_REPO / "data/vision_cache/checkpoint-16412"))
    # choices 를 손으로 적어 두면 SPLIT_PATHS 에 항목을 추가할 때마다 여기가
    # 어긋나 argparse 에서 죽는다(실제로 valid_nt2 를 추가하고 겪었다).
    ap.add_argument("--split", default="valid", choices=sorted(SPLIT_PATHS))
    ap.add_argument("--lang_encoder_path", default="/data/drdcad/sunggu/llm_models/Qwen3-8B")
    ap.add_argument("--model_config_yaml", default=str(
        _REPO / "src/Model/ctrate_pillar0_chest_opensource.yaml"))
    ap.add_argument("--num_visual_tokens", type=int, default=128)
    ap.add_argument("--vis_dim", type=int, default=1152)
    ap.add_argument("--max_seq", type=int, default=2688)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--names_json", default="",
                    help="Optional JSON list of sample_names to generate, in place of "
                         "the split order. Used by the crgf1bleu reward build so the "
                         "reference generation covers EXACTLY the RL prompt subset. "
                         "Inert when omitted.")
    ap.add_argument("--out", required=True)
    # test.py's sampler, verbatim. See the docstring.
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from safetensors.torch import load_file as load_safetensors

    from Dataset.ctrate_dataset_test import CTRATEDataset_Test
    from Model.CTRG_3D import CTRG3D

    torch.manual_seed(args.seed)
    split = args.split
    paths = SPLIT_PATHS[split]
    cache_split = paths["cache"]          # NOT `split`; see SPLIT_PATHS comment
    cache_dir = Path(args.cache_dir)

    # ---- cache, and the check that it belongs to this checkpoint -------------
    meta_path = cache_dir / f"{cache_split}_meta.json"
    if not meta_path.exists():
        have = sorted(p.name for p in cache_dir.glob("*_meta.json")) or ["(none)"]
        raise FileNotFoundError(
            f"--split {split} reads cache basename '{cache_split}', but "
            f"{meta_path} does not exist. {cache_dir} has: {', '.join(have)}. "
            f"Build it with scripts/build_vision_cache.py --split {cache_split}.")
    meta = json.loads(meta_path.read_text())
    if meta["checkpoint_tag"] != Path(args.ckpt).name:
        raise RuntimeError(
            f"cache was built from {meta['checkpoint_tag']} but --ckpt is "
            f"{Path(args.ckpt).name}. The perceiver/fc weights differ, so the cached "
            f"features do not belong to this model. Rebuild the cache.")
    index = json.loads((cache_dir / f"{cache_split}_index.json").read_text())
    cache = np.load(cache_dir / f"{cache_split}.npy", mmap_mode="r")
    print(f"[cache] {cache.shape} {cache.dtype} from {cache_dir} "
          f"({cache_split}.npy, --split {split})", flush=True)

    # ---- prompts, built exactly as the RL path builds them ------------------
    ds = CTRATEDataset_Test(
        tokenizer_path=args.lang_encoder_path,
        json_path=paths["json"], rve_manifest_path=paths["rve"],
        cls_template_path=paths["cls"], ret_template_path=paths["ret"],
        image_num=args.num_visual_tokens, max_seq=args.max_seq, use_fast=False,
        use_cls_template=True, use_ret_template=True,
    )
    tok = ds.tokenizer
    tok.padding_side = "left"                       # non-negotiable, see docstring
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    names = [n for n in ds.df["sample_name"].tolist() if n in index]
    if args.names_json:
        want = json.loads(Path(args.names_json).read_text())
        have = set(names)
        absent = [n for n in want if n not in have]
        if absent:
            raise RuntimeError(
                f"{len(absent)} of {len(want)} requested names are not in the {split} "
                f"split's cached volumes, e.g. {absent[:3]}. The reference generation "
                f"must cover the RL subset exactly or the BLEU floor has holes.")
        names = list(want)
    if args.n:
        names = names[:args.n]
    by_name = {n: i for i, n in enumerate(ds.df["sample_name"].tolist())}
    print(f"[data] {len(names)} of {len(index)} cached {split} volumes", flush=True)

    prompts, prompt_ids, gts, questions = {}, {}, {}, {}
    for n in names:
        s = ds[by_name[n]]
        text = s["question"]
        if "</ct>\n" not in text:
            raise RuntimeError(f"{n}: prompt has no '</ct>' boundary")
        text = text.replace("</ct>\n", CASE_MARKER.format(row=index[n]), 1)
        prompts[n] = text
        questions[n] = s["question"]
        gts[n] = s["gt_report"]
        prompt_ids[n] = tok(text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids[n]) > args.max_seq:
            raise RuntimeError(
                f"{n}: prompt is {len(prompt_ids[n])} tokens, over --max_seq {args.max_seq}")

    # ---- model --------------------------------------------------------------
    model = CTRG3D(
        text_tokenizer_path=args.lang_encoder_path,
        lang_model_path=args.lang_encoder_path,
        pretrained_visual_encoder="", pretrained_adapter="",
        use_fast=False, enable_thinking=False,
        model_config_yaml=args.model_config_yaml,
        vis_dim=args.vis_dim, image_num=args.num_visual_tokens,
    ).bfloat16()
    ck = next(Path(args.ckpt).glob("model.safetensors"))
    model.load_state_dict(load_safetensors(str(ck), device="cpu"), strict=True)

    core, n_wrap = model.core_vocab_size, model.embedding_layer.num_wrapper_tokens
    vis_lo, vis_hi = core + n_wrap, core + n_wrap + args.num_visual_tokens
    # (3) in the docstring: padding must not land inside the visual id range.
    if not (tok.pad_token_id < core):
        raise RuntimeError(
            f"pad_token_id {tok.pad_token_id} is not below core_vocab_size {core}; "
            f"padded positions would be counted as visual slots. Pick a pad id < {core}.")
    print(f"[vocab] core {core}  visual ids {vis_lo}..{vis_hi - 1}  "
          f"pad {tok.pad_token_id}", flush=True)

    lm = model.lang_model                      # PeftModel, LoRA already attached

    # ---- the RL adapter, LoRA keys only (see (4) in the docstring) ----------
    if args.adapter:
        ad = Path(args.adapter) / "adapter_model.safetensors"
        if not ad.is_file():
            raise FileNotFoundError(ad)
        raw = load_safetensors(str(ad), device="cpu")
        lora = {k: v for k, v in raw.items() if "lora_" in k}
        skipped = sorted(set(raw) - set(lora))
        if not lora:
            raise RuntimeError(f"{ad} holds no lora_* tensors")
        own = {k for k, _ in lm.named_parameters() if "lora_" in k}
        loaded, missing = 0, []
        sd = dict(lm.named_parameters())
        with torch.no_grad():
            for k, v in lora.items():
                # PEFT saves "...lora_A.weight"; the live module is
                # "...lora_A.default.weight". Normalise, then require a hit.
                cand = [k, k.replace(".lora_A.weight", ".lora_A.default.weight")
                            .replace(".lora_B.weight", ".lora_B.default.weight")]
                tgt = next((c for c in cand if c in sd), None)
                if tgt is None:
                    missing.append(k); continue
                if tuple(sd[tgt].shape) != tuple(v.shape):
                    raise RuntimeError(f"{tgt}: live {tuple(sd[tgt].shape)} vs "
                                       f"saved {tuple(v.shape)}")
                sd[tgt].copy_(v.to(sd[tgt].dtype))
                loaded += 1
        if missing:
            raise RuntimeError(
                f"{len(missing)} adapter tensors had no destination, e.g. {missing[:3]}. "
                f"The live LoRA layout differs from the saved one -- refusing to score a "
                f"partially loaded adapter.")
        if loaded != len(own):
            raise RuntimeError(
                f"loaded {loaded} LoRA tensors but the model has {len(own)}; a partial "
                f"load would score a mixture of the RL and SFT policies.")
        print(f"[adapter] {loaded} LoRA tensors from {args.adapter}; "
              f"skipped (frozen in RL): {skipped}", flush=True)
    else:
        print("[adapter] none -- scoring the SFT policy itself (the reference)", flush=True)

    lookup = build_key_lookup(prompt_ids, index, vis_lo, vis_hi, args.num_visual_tokens)
    print(f"[keys] {len(lookup)} unique prompt keys (window={KEY_WINDOW})", flush=True)
    wrapper_w = torch.cat([model.embedding_layer.image_token_weight,
                           model.embedding_layer.template_token_weight], dim=0).detach()
    lm.get_base_model().set_input_embeddings(CachedVisualEmbedding(
        lm.get_base_model().get_input_embeddings(), wrapper_w, cache, lookup,
        core, n_wrap, args.num_visual_tokens))
    del model.embedding_layer.vision_encoder
    lm = lm.cuda().eval()

    # ---- generate -----------------------------------------------------------
    # Longest-first so a batch's slowest row does not stall a batch of short ones.
    order = sorted(names, key=lambda n: -len(prompt_ids[n]))
    rows, done = [], 0
    import time
    t0 = time.time()
    for i in range(0, len(order), args.batch_size):
        chunk = order[i:i + args.batch_size]
        enc = tok([prompts[n] for n in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False)
        ids = enc["input_ids"].cuda()
        mask = enc["attention_mask"].cuda()
        # PASSING `inputs_embeds` (NOT `input_ids`) IS LOAD-BEARING.
        #
        # The prompt carries the 128 visual placeholder ids, which live ABOVE the
        # tied lm_head's output width (`core`). `repetition_penalty=1.05` installs
        # RepetitionPenaltyLogitsProcessor, which does
        # `torch.gather(scores, 1, input_ids)` over the WHOLE sequence -- prompt
        # included -- so a visual id indexes past `scores.size(1)`:
        #
        #   ScatterGatherKernel.cu:203: Assertion `idx_dim < index_size` failed
        #   torch.AcceleratorError: CUDA error: device-side assert triggered
        #
        # That killed job 28097124 after 3 min. CTRG3D.generate() never hit it
        # because it feeds `inputs_embeds`, which leaves generate's `input_ids` at
        # width 0 -- the penalty then sees only the tokens the model itself emitted,
        # all of which are < core. Embedding the prompt here ourselves reproduces
        # test.py's sampler EXACTLY rather than merely approximating it.
        with torch.no_grad():
            embeds = lm.get_base_model().get_input_embeddings()(ids)
            out = lm.generate(
                inputs_embeds=embeds, attention_mask=mask,
                do_sample=True, top_p=args.top_p, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            )
        # With `inputs_embeds` and no `input_ids`, HF returns ONLY the new tokens,
        # so there is no prompt prefix to slice off. Assert that, rather than
        # assuming it: a transformers change that starts echoing the prompt would
        # otherwise silently prepend garbage to every prediction.
        if out.shape[1] == 0:
            raise RuntimeError("generate returned no new tokens")
        if out.shape[1] > args.max_new_tokens:
            raise RuntimeError(
                f"generate returned {out.shape[1]} tokens with max_new_tokens="
                f"{args.max_new_tokens} -- it is echoing the {ids.shape[1]}-token "
                f"prompt; the slicing below would leak it into every prediction.")
        preds = tok.batch_decode(out, skip_special_tokens=True)
        for n, p in zip(chunk, preds):
            rows.append({"SampleName": n, "Question": questions[n],
                         "GT_report": gts[n], "Pred_report": p.strip()})
        done += len(chunk)
        el = time.time() - t0
        print(f"[gen] {done}/{len(order)}  {el/done:.2f} s/vol  "
              f"eta {(len(order) - done) * el / done / 60:.1f} min", flush=True)

    emb = lm.get_base_model().get_input_embeddings()
    print(f"[check] visual injections {emb.n_injected}  misses {emb.n_missing}", flush=True)
    if emb.n_injected == 0:
        raise RuntimeError("no visual features were injected -- the run was blind")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Restore the split's own order so two CSVs line up row for row.
    df = pd.DataFrame(rows).set_index("SampleName").loc[names].reset_index()
    df.to_csv(out_path, index=False)
    empty = int((df["Pred_report"].str.len() == 0).sum())
    print(f"[done] {out_path}  {len(df)} rows  empty predictions {empty}", flush=True)
    print(f"       mean pred chars {df['Pred_report'].str.len().mean():.0f}  "
          f"mean GT chars {df['GT_report'].str.len().mean():.0f}", flush=True)


if __name__ == "__main__":
    main()
