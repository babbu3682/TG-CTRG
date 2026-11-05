"""GRPO post-training against the challenge metrics (macro-F1 + BLEU floor).

    # smoke (minutes), then the real thing -- run_sbatch_grpo does both:
    python scripts/grpo_crg.py --smoke
    python scripts/grpo_crg.py --max_steps 500

Design and evidence: docs/2026-08-21-grpo-crg-design.md. Mechanism and tensor
shapes: docs/2026-08-21-grpo-mechanism.html.

HOW THE CACHED VISUAL FEATURES REACH THE MODEL
----------------------------------------------
By replacing the language model's input embedding module, and nothing else. No
TRL subclass, no `generate` override, no custom trainer. Every forward inside
the LM goes through `embed_tokens(input_ids)` -- the generation prefill, each
decode step, the policy log-prob pass and the reference log-prob pass -- so a
custom embedding is the one hook TRL's internal re-batching cannot bypass. This
was verified before any of this was written, with an adversarial alignment
assert, in a 2-step run against TRL 1.10 (see design doc §8.1).

Three constraints came out of that experiment and are all load-bearing here:

1. PEFT must be the OUTERMOST wrapper. If it is not, `is_peft_model()` is False
   and TRL builds a separate reference model from `config._name_or_path` -- a
   second full 8B (16 GB), or a crash if that path is empty. With PEFT outside,
   `ref_model is None` and TRL gets reference log-probs by disabling the
   adapter, which is both cheaper and exactly what we want.
2. The visual placeholders must be real tokenizer tokens. TRL's dataset carries
   the prompt as TEXT, so ids only survive the round-trip if the tokenizer knows
   the strings. Ours does (`<image_ct*>`; that is what the core+6+128 vocabulary
   assertion in CTRG3D is about).
3. The model must not delegate `generate`. Replacing the embedding sidesteps
   this entirely -- there is no wrapper to delegate from.

WHICH SAMPLE IS THIS ROW? -- the lookup key
-------------------------------------------
The 128 placeholder ids are identical for every sample, so they cannot identify
one, and neither can the prompt text: measured over 400 valid prompts, even the
256 tokens following the visual block are non-unique (392/400). Different
volumes genuinely produce byte-identical prompts, because the cls/ret templates
are drawn from a small fixed pool of sentences. No token-derived key can
separate those, so the prompt is made unique BY CONSTRUCTION: a 9-token case
marker (`Case 000123.`) naming the cache row is inserted right after `</ct>`.
Measured: 400/400 unique with a 16-token key window, and the whole prompt grows
by 9 tokens out of ~2560 (0.35%).

The key is then the 16 tokens immediately after the last placeholder. That
window sits at a position found by locating the placeholders, so it is immune to
left padding; it is inside the prompt, so it is identical across the rollouts of
a group and across generation and both log-prob passes. Uniqueness over the
whole split is asserted when the lookup is built.

DEVIATION, stated because it is real: the RL prompt is 9 tokens longer than the
SFT prompt. Keep the marker in the eval prompts too, so the policy always sees
the same format as it was trained with -- the mismatch to guard against is
train/eval, not RL/SFT.

WHY THE REWARD IS macro-F1 AND NOT CRG
--------------------------------------
Measured: asserting the 5 most prevalent labels unconditionally moves CRG
+0.0082 and macro-F1 -0.0223, and raises the official corpus BLEU_mean by
+0.0193. So CRG has a free gaming direction and BLEU rewards rather than
punishes it -- `CRG + BLEU` is the one pair with nothing opposing prevalence
gaming. macro-F1 weights labels equally and has no free direction, and pulls CRG
along anyway (r = 0.747). See scripts/reward_challenge_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_HERE))

KEY_WINDOW = 16   # tokens after the last visual placeholder that identify a sample
CASE_MARKER = "</ct>\nCase {row:06d}.\n"   # see WHICH SAMPLE IS THIS ROW below


# ---------------------------------------------------------------------------
# the one hook
# ---------------------------------------------------------------------------

class CachedVisualEmbedding(nn.Module):
    """Drop-in for `embed_tokens`: fills the visual slots from the offline cache.

    Reproduces the id layout `MyEmbedding.forward` builds, minus the vision
    encoder:

        [0, core)                 the LLM's own vocabulary  -> inner embedding
        [core, core+n_wrap)       wrapper tokens            -> wrapper parameter
        [core+n_wrap, ...)        128 visual slots          -> cache[row]

    Rows with no visual slots (every decode step after the prefill) fall through
    untouched.
    """

    def __init__(self, inner: nn.Module, wrapper_weight: torch.Tensor,
                 cache: np.ndarray, key_to_row: dict, core: int, n_wrap: int,
                 n_vis: int):
        super().__init__()
        self.inner = inner
        self.register_buffer("wrapper_weight", wrapper_weight.clone(), persistent=False)
        self.cache = cache                      # np.memmap [N, n_vis, hidden]
        self.key_to_row = key_to_row
        self.core, self.n_wrap, self.n_vis = core, n_wrap, n_vis
        self.vis_lo = core + n_wrap
        self.vis_hi = core + n_wrap + n_vis
        self.n_injected = 0
        self.n_missing = 0

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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        safe = input_ids.clamp_max(self.core - 1)
        emb = self.inner(safe)

        flat_ids = input_ids.reshape(-1, input_ids.shape[-1])
        # .clone() IS LOAD-BEARING -- do not remove it to save a tensor.
        #
        # GRPOConfig(gradient_checkpointing=True) on a PEFT model makes
        # transformers install enable_input_require_grads(), a forward hook on the
        # input embedding that calls output.requires_grad_(True). Because the base
        # embedding is frozen (LoRA only), that output is not the result of a
        # differentiable op, so requires_grad_(True) turns it into a LEAF that
        # requires grad -- and emb.reshape(...) is a VIEW of that leaf. Writing the
        # cached visual rows into a view of a leaf is forbidden:
        #
        #   RuntimeError: a view of a leaf Variable that requires grad is being
        #                 used in an in-place operation
        #
        # Generation never hit this because TRL rolls out under no_grad; the first
        # policy update did (job 28074796, died 78 s into the smoke stage).
        # clone() gives the destination its own storage, is differentiable, and
        # leaves the gradient path intact: text positions keep grad 1, and the
        # injected visual/wrapper positions correctly get grad 0 -- they are
        # constants from the offline cache, which is exactly the frozen-perception
        # design this pilot chose.
        flat_emb = emb.reshape(flat_ids.shape[0], flat_ids.shape[1], -1).clone()

        for b in range(flat_ids.shape[0]):
            ids = flat_ids[b]
            wrap = ((ids >= self.core) & (ids < self.vis_lo)).nonzero(as_tuple=True)[0]
            if wrap.numel():
                flat_emb[b, wrap] = self.wrapper_weight[
                    ids[wrap] - self.core].to(flat_emb.dtype)

            pos = ((ids >= self.vis_lo) & (ids < self.vis_hi)).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue                                   # decode step
            if pos.numel() != self.n_vis:
                raise RuntimeError(
                    f"{pos.numel()} visual slots in a sequence, expected {self.n_vis}; "
                    f"the prompt was truncated or the tokenizer disagrees with the model")
            end = int(pos[-1].item()) + 1
            key = tuple(int(x) for x in ids[end:end + KEY_WINDOW].tolist())
            row = self.key_to_row.get(key)
            if row is None:
                self.n_missing += 1
                raise KeyError(
                    "no cache row for this prompt: the key window does not match any "
                    "sample. The prompt built here differs from the one the lookup was "
                    "built from (template dropout must stay OFF).")
            flat_emb[b, pos] = torch.as_tensor(
                np.asarray(self.cache[row]), device=flat_emb.device).to(flat_emb.dtype)
            self.n_injected += 1

        return flat_emb.reshape(emb.shape)


def build_key_lookup(prompt_ids_per_sample, index, vis_lo, vis_hi, n_vis):
    """{key window -> cache row}, with a loud uniqueness check."""
    lookup, seen = {}, {}
    for name, ids in prompt_ids_per_sample.items():
        arr = torch.tensor(ids)
        pos = ((arr >= vis_lo) & (arr < vis_hi)).nonzero(as_tuple=True)[0]
        if pos.numel() != n_vis:
            raise RuntimeError(f"{name}: {pos.numel()} visual slots, expected {n_vis}")
        end = int(pos[-1].item()) + 1
        key = tuple(ids[end:end + KEY_WINDOW])
        if key in seen:
            raise RuntimeError(
                f"key collision between {seen[key]} and {name}: the {KEY_WINDOW} tokens "
                f"after the visual block are identical, so their features would be "
                f"swapped. Widen KEY_WINDOW.")
        seen[key] = name
        lookup[key] = index[name]
    return lookup


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default=str(
        _REPO / "data/vision_cache/checkpoint-16412"))
    ap.add_argument("--ckpt", default=str(
        _REPO / "scripts/outputs/ep15/ctrg3d-qwen3-8B-pillar0os-lz4-vt128/checkpoint-16412"))
    ap.add_argument("--lang_encoder_path", default="/data/drdcad/sunggu/llm_models/Qwen3-8B")
    ap.add_argument("--model_config_yaml", default=str(
        _REPO / "src/Model/ctrate_pillar0_chest_opensource.yaml"))
    ap.add_argument("--num_visual_tokens", type=int, default=128)
    ap.add_argument("--vis_dim", type=int, default=1152)
    ap.add_argument("--max_seq", type=int, default=2688)
    ap.add_argument("--output_dir", default=str(_REPO / "outputs/grpo_crg"))
    # the pilot's settings, fixed in the design doc
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=512)
    ap.add_argument("--learning_rate", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--lambda_bleu", type=float, default=2.0)
    ap.add_argument("--n_prompts", type=int, default=0, help="0 = whole train split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="2 steps, 4 prompts, 32 new tokens -- proves the wiring")
    args = ap.parse_args()

    if args.smoke:
        args.max_steps, args.num_generations = 2, 2
        args.per_device_train_batch_size, args.max_completion_length = 2, 32
        args.n_prompts, args.output_dir = 8, args.output_dir + "_smoke"

    import datasets
    from safetensors.torch import load_file as load_safetensors
    from trl import GRPOConfig, GRPOTrainer

    from Dataset.ctrate_dataset_test import CTRATEDataset_Test
    from Model.CTRG_3D import CTRG3D
    from reward_challenge_metrics import (ChallengeReward, MacroF1Marginal,
                                          N_LABELS, build_bleu_reference,
                                          load_organizer_labels,
                                          load_reference_reports, _classify_counts)

    cache_dir = Path(args.cache_dir)
    meta = json.loads((cache_dir / "train_meta.json").read_text())
    if Path(meta["checkpoint"]).parent.name != Path(args.ckpt).name and \
            meta["checkpoint_tag"] != Path(args.ckpt).name:
        raise RuntimeError(
            f"cache was built from {meta['checkpoint']} but the policy starts from "
            f"{args.ckpt}. Perceiver/fc weights differ, so the cached features do not "
            f"belong to this model. Rebuild the cache.")
    index = json.loads((cache_dir / "train_index.json").read_text())
    cache = np.load(cache_dir / "train.npy", mmap_mode="r")
    print(f"[cache] {cache.shape} {cache.dtype} from {cache_dir}", flush=True)

    # ---- dataset: prompts only; the answer is what GRPO generates -----------
    ds_src = CTRATEDataset_Test(
        tokenizer_path=args.lang_encoder_path,
        json_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_train_filtered_prob_half.json",
        rve_manifest_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/train/mapping_path.csv",
        cls_template_path="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_ABN_cls_report_templates_p90n20/cls_report_train.json",
        ret_template_path="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden/train_ret_reports.json",
        image_num=args.num_visual_tokens, max_seq=args.max_seq, use_fast=False,
        use_cls_template=True, use_ret_template=True,
    )
    tok = ds_src.tokenizer
    names_all = [n for n in ds_src.df["sample_name"].tolist() if n in index]
    rng = np.random.default_rng(args.seed)
    picked = list(rng.permutation(names_all))
    if args.n_prompts:
        picked = picked[:args.n_prompts]
    print(f"[data] {len(picked)} prompts of {len(names_all)} cached", flush=True)

    by_name = {n: i for i, n in enumerate(ds_src.df["sample_name"].tolist())}
    prompts, prompt_ids = {}, {}
    for n in picked:
        s = ds_src[by_name[n]]            # the volume load is wasted work here, but
        text = s["question"]              # it keeps the prompt identical to the SFT path
        if "</ct>\n" not in text:
            raise RuntimeError(
                f"{n}: prompt has no '</ct>' boundary to anchor the case marker to")
        text = text.replace("</ct>\n", CASE_MARKER.format(row=index[n]), 1)
        prompts[n] = text
        prompt_ids[n] = tok(text, add_special_tokens=False)["input_ids"]
        # TRL 1.10.0 dropped GRPOConfig.max_prompt_length, so nothing truncates the
        # prompt for us -- and nothing warns either. An over-long prompt would push
        # the visual slots past the model's window and the run would go blind
        # silently, so the check the config used to do is done here instead.
        if len(prompt_ids[n]) > args.max_seq:
            raise RuntimeError(
                f"{n}: prompt is {len(prompt_ids[n])} tokens, over --max_seq "
                f"{args.max_seq}. TRL no longer truncates prompts; re-measure the "
                f"templates for this arm instead of raising max_seq blindly.")

    model = CTRG3D(
        text_tokenizer_path=args.lang_encoder_path,
        lang_model_path=args.lang_encoder_path,
        pretrained_visual_encoder="", pretrained_adapter="",
        use_fast=False, enable_thinking=False,
        model_config_yaml=args.model_config_yaml,
        vis_dim=args.vis_dim, image_num=args.num_visual_tokens,
    ).bfloat16()
    ck = args.ckpt if args.ckpt.endswith((".safetensors", ".bin")) else \
        str(next(Path(args.ckpt).glob("model.safetensors")))
    sd = load_safetensors(ck, device="cpu") if ck.endswith(".safetensors") else \
        torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=True)

    core, n_wrap = model.core_vocab_size, model.embedding_layer.num_wrapper_tokens
    lookup = build_key_lookup(prompt_ids, index, core + n_wrap,
                              core + n_wrap + args.num_visual_tokens,
                              args.num_visual_tokens)
    print(f"[keys] {len(lookup)} unique prompt keys (window={KEY_WINDOW})", flush=True)

    lm = model.lang_model                     # already a PeftModel -> PEFT outermost
    wrapper_w = torch.cat([model.embedding_layer.image_token_weight,
                           model.embedding_layer.template_token_weight], dim=0).detach()
    lm.get_base_model().set_input_embeddings(CachedVisualEmbedding(
        lm.get_base_model().get_input_embeddings(), wrapper_w, cache, lookup,
        core, n_wrap, args.num_visual_tokens))
    del model.embedding_layer.vision_encoder  # not used again; free the memory
    lm = lm.cuda()

    # ---- reward -------------------------------------------------------------
    gt_labels = load_organizer_labels(
        "/data/drdcad/sunggu/0.Challenge/pillar-pretrain/train_predicted_labels.csv")
    gt_reports = {n: ds_src.df.iloc[by_name[n]]["report_metadata"] for n in picked}
    ref_reports = {n: prompts[n] for n in picked}   # placeholder, replaced below
    ref_json = _REPO / "results/ep15best/ctrg3d-qwen3-8B-pillar0os-lz4-vt128/vlm3d_eval_full/predictions.json"
    if ref_json.exists():
        # BLEU floor from the starting policy where we have its outputs (valid);
        # on train volumes we have none, so the floor is that policy's mean.
        vr = load_reference_reports(str(ref_json))
        vg = {k: v for k, v in gt_reports.items() if k in vr}
        base_mean = float(np.mean(list(build_bleu_reference(
            {k: gt_reports[k] for k in vg}, {k: vr[k] for k in vg}).values()))) \
            if vg else 0.2966
    else:
        base_mean = 0.2966
    bleu_ref = {n: base_mean for n in picked}
    print(f"[reward] BLEU floor {base_mean:.4f} (starting policy's mean)", flush=True)

    tp = [0] * N_LABELS; fp = [0] * N_LABELS; fn = [0] * N_LABELS
    for n in picked:                       # seed the corpus context with the GT prior
        dtp, dfp, dfn = _classify_counts(gt_labels[n], gt_labels[n])
        for j in range(N_LABELS):
            tp[j] += dtp[j]; fp[j] += dfp[j]; fn[j] += dfn[j]
    reward = ChallengeReward(gt_labels, gt_reports, bleu_ref, (tp, fp, fn),
                             lambda_bleu=args.lambda_bleu).with_radbert()

    class CommitOnStep(GRPOTrainer):
        """Advance the reward's corpus context once per optimiser step, and log
        the reward's two terms separately.

        Never per rollout: the 8 rollouts of a group must all be scored against
        the same counts or the group-relative advantage is polluted by ordering.

        WHY THE TERM SPLIT IS LOGGED
        ----------------------------
        reward = d_f1 - lambda_B * max(0, floor - BLEU)/sigma_B, and TRL only
        logs the sum. After the 500-step pilot (job 28075386) the sum sat at
        -0.85 for the whole run and there was no way to tell whether the F1 term
        was doing anything or the BLEU penalty was simply always on -- which is
        exactly the ratio the design doc says is the only thing that matters
        (docs/2026-08-21-grpo-crg-design.md 2.2). Without it, "loosen beta" and
        "the penalty is decorative" are indistinguishable, and both are
        three-day decisions. ChallengeReward already stores the split in
        `last_terms`; nothing was reading it.
        """
        def training_step(self, *a, **k):
            out = super().training_step(*a, **k)
            terms = getattr(reward, "last_terms", None)
            if terms:
                n = len(terms)
                self.log({
                    "reward_f1": sum(t["f1"] for t in terms) / n,
                    "reward_bleu_pen": sum(t["bleu_pen"] for t in terms) / n,
                    "reward_bleu_raw": sum(t["bleu"] for t in terms) / n,
                    # how often the floor actually bites; 0 means the penalty is
                    # decorative and beta is the only thing defending BLEU.
                    "reward_bleu_pen_frac": sum(1 for t in terms if t["bleu_pen"] > 0) / n,
                })
            reward.commit()
            return out

    ds = datasets.Dataset.from_dict({"prompt": [prompts[n] for n in picked],
                                     "sample_name": picked})
    cfg = GRPOConfig(
        output_dir=args.output_dir, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # NOTE: no max_prompt_length -- removed in TRL 1.10.0. Prompt length is
        # asserted at build time above instead.
        learning_rate=args.learning_rate,
        beta=args.beta, temperature=args.temperature, top_p=args.top_p,
        bf16=True, logging_steps=1, save_strategy="steps", save_steps=100,
        report_to=["wandb"], run_name=Path(args.output_dir).name, seed=args.seed,
        gradient_checkpointing=True, log_completions=True,
    )
    trainer = CommitOnStep(model=lm, reward_funcs=reward, args=cfg, train_dataset=ds,
                           processing_class=tok)
    if getattr(trainer, "ref_model", None) is not None:
        raise RuntimeError(
            "TRL built a separate reference model -- PEFT is not the outermost "
            "wrapper. That is a second full 8B in memory and the wrong KL baseline.")
    print("[trl] ref_model is None (adapter-disable KL path)", flush=True)

    trainer.train()

    emb = lm.get_base_model().get_input_embeddings()
    print(f"\n[check] visual injections: {emb.n_injected}  misses: {emb.n_missing}")
    if emb.n_injected == 0:
        raise RuntimeError("no visual features were ever injected -- the run was blind")
    print("[done]", args.output_dir)


if __name__ == "__main__":
    main()
