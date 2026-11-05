"""GRPO on top of newenc B (gtweight w=5) -- the SFT best. LoRA only.

scripts/grpo_crgf1bleu.py IS NOT MODIFIED: it is what job 28101808 ran and the
base -> ckpt-1000 trajectory in docs/2026-08-25-grpo-improvements.md has to stay
reproducible. This file is a minimal derivation of it. The REWARD is imported,
not copied, so the three terms are literally the same code.

What differs from round 2, and only this:
  1. starting policy   newenc B ckpt-7455 (lapa encoder, vis_dim 384)
  2. prompt templates  lapa_f1max / retrieval_self_lapa, and the
     gtcov train split -- i.e. EXACTLY what B was trained on. Feeding B the old
     p90n20 / pillar_freeze templates would be a silent distribution shift.
  3. subset 12,000 -> max_steps 1500 = still exactly 1 epoch at 8 prompts/step.
     Round 2 ended 1 epoch still improving (F1 +0.0080 over the last 100 steps)
     but with entropy 0.687 -> 0.392 and frac_reward_zero_std leaving 0 after
     step 601. A SECOND pass over the same 8,000 risks group-variance collapse;
     doubling the subset buys the same extra updates on fresh prompts instead.
  4. --resume_from_checkpoint, because 1500 steps x 170 s = 71 h is right at the
     3-day walltime. Submit the launcher again and it continues. Under the
     2026-08-30 deadline the run need not finish: save_steps=100, so the best
     checkpoint scored before the deadline is what ships.

Original round-2 docstring follows.

GRPO round 2 (crgf1bleu): CRG + macro-F1 + paired BLEU floor, LoRA only.

    # smoke uses the REAL batch, so it also proves the memory envelope:
    python scripts/grpo_crgf1bleu.py --smoke
    python scripts/grpo_crgf1bleu.py            # 1000 steps = 1 epoch of the subset

Diagnosis of the 500-step pilot and every decision made from it:
docs/2026-08-22-grpo-review.md. Mechanism and tensor shapes are unchanged from
round 1: docs/2026-08-21-grpo-mechanism.html.

scripts/grpo_crg.py IS NOT MODIFIED. It is what job 28075386 ran and it stays
reproducible; the cached-embedding machinery is imported from it so training and
scoring cannot drift apart in how they build a prompt or key a cache row.

WHAT CHANGED FROM THE PILOT, AND WHY
------------------------------------
reward      3 terms, all normalised to unit spread on the starting policy:
              alpha*s_i/sigma_S + gamma*dF1_i/sigma_F1 - lambda*hinge/sigma_B
            The pilot's effective ratio was F1 1 : BLEU 9.4 -- it was optimising
            BLEU, not macro-F1. See scripts/reward_crgf1bleu.py.

corpus      FROZEN at the starting policy's own predictions on the subset.
            The pilot advanced it every step from a GT-perfect seed, so the F1
            term was differentiated at macro-F1 = 1.0 and shrank as the run went.

floor       PAIRED, per volume. The pilot's single constant put the starting
            policy itself in penalty on 65.5% of samples.

prompts/step  8, not 1. TRL's per_device_train_batch_size counts COMPLETIONS,
            so the pilot's 8 with num_generations 8 meant one group per step --
            the update direction was decided by whichever single volume came up.
            64/8 = 8 groups. KV cache 32.6 GB of 80.

cap         768, not 512. Measured on 4,000 train GT reports: p99 = 762, and
            512 covers only 89.4%. The pilot truncated 12% of rollouts, which
            rewards writing less because unwritten findings cannot be FP.
            mask_truncated_completions keeps whatever still truncates out of the
            loss.

lr          1e-5 with 20 warmup steps then CONSTANT, not 2e-6 decaying to 0.
            The pilot's total budget was 500 x 2e-6 = 1e-3 against SFT's 0.82.
            Constant keeps kl/entropy interpretable against step number.

beta        0. The adapter-disable reference is the PRE-SFT base model, so the
            KL term pulled toward undoing the SFT, and at beta*kl = 2.8e-4 --
            0.2% of the loss -- it was never a defence. BLEU's paired floor is
            the defence now.

clip_ratio  Do not read it as a diagnostic. With num_iterations=1 and no
            gradient accumulation TRL sets old_per_token_logps=None, so the
            ratio is exp(x - x.detach()) == 1 exactly and the clip can never
            fire (grpo_trainer.py:3107). The pilot's headers read its constant 0
            as "the policy never moved" and four follow-up jobs were built on
            that. Movement shows in kl, entropy and completion length.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_HERE))

from grpo_crg import CASE_MARKER, KEY_WINDOW, CachedVisualEmbedding, build_key_lookup

REF_DIR = _REPO / "data/grpo_ref_newencB"
_NEW_CLS = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
            "CTRATE_ABN_cls_report_templates_lapa_f1max")
_NEW_RET = ("/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/"
            "CTRATE_FT_Atlas_retrieval_self_lapa/templates_ep32_pp")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default=str(
        _REPO / "data/vision_cache/newencB_ep15_ckpt7455"))
    # RadBERT 체크포인트 경로. 기본은 with_radbert() 의 MIDS 경로이고, 그것을
    # 읽을 수 없는 계정(ryuj4 는 MIDS 그룹이 아니다)에서만 drdcad 사본을 넘긴다.
    ap.add_argument("--radbert_ckpt", default="")
    ap.add_argument("--ckpt", default=str(
        _REPO / "scripts/outputs/preserved/newencB_ep15_ckpt7455"))
    ap.add_argument("--lang_encoder_path", default="/data/drdcad/sunggu/llm_models/Qwen3-8B")
    ap.add_argument("--model_config_yaml", default=str(
        _REPO / "src/Model/ctrate_pillar0_chest_lapa.yaml"))
    ap.add_argument("--subset_json", default=str(REF_DIR / "subset12000.json"))
    ap.add_argument("--reference_json", default=str(REF_DIR / "reference.json"))
    ap.add_argument("--train_json_path", default=str(
        _REPO / "data/sentence_labels/merged_train_gtcov.json"))
    ap.add_argument("--cls_template_path", default=f"{_NEW_CLS}/cls_report_train.json")
    ap.add_argument("--ret_template_path", default=f"{_NEW_RET}/train_ret_reports_pp.json")
    ap.add_argument("--num_visual_tokens", type=int, default=128)
    ap.add_argument("--vis_dim", type=int, default=384)
    ap.add_argument("--max_seq", type=int, default=2688)
    ap.add_argument("--output_dir", default=str(_REPO / "outputs/grpo_newencB"))
    # round-2 settings, fixed in docs/2026-08-22-grpo-review.md 6
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--resume_from_checkpoint", default="",
                    help="a checkpoint-N dir; HF fast-forwards the dataloader and "
                         "restores optimizer/scheduler/step. max_steps must match "
                         "the interrupted run or the schedule changes mid-flight.")
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--per_device_train_batch_size", type=int, default=64)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="TRL GRPO 에서 generation_batch_size = per_device x accum 이므로, "
                         "per_device 를 낮추고 이 값을 같은 배수로 올리면 스텝당 completion "
                         "수와 옵티마이저 스텝 수가 모두 보존된다. 순수하게 loss 패스의 "
                         "logits 피크를 쪼개는 장치다 (job 28098315 는 accum 1 / pd 64 로 "
                         "logits 21.15 GiB 를 한 번에 요구해 smoke 에서 OOM).")
    ap.add_argument("--max_completion_length", type=int, default=768)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lambda_bleu", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompts_only", action="store_true",
                    help="프롬프트를 만들 때 CT 볼륨을 읽지 않는다. RL 은 시각 특징을 "
                         "오프라인 캐시에서 받으므로 dataset 의 ct_image 를 쓰지 않는데, "
                         "기본 경로는 샘플당 ~25 MB 를 읽고 버린다(subset 12,000 이면 "
                         "~300 GB). Biowulf 스토리지에서는 안 보이지만 storm 의 sshfs "
                         "에서는 3.5 시간이다. 출력은 바이트 단위로 동일하다.")
    ap.add_argument("--smoke", action="store_true",
                    help="2 steps at the REAL batch -- proves wiring AND memory")
    args = ap.parse_args()

    if args.smoke:
        # Batch, cap and generation count stay at their real values on purpose:
        # the pilot's 2x2 smoke passed and then the run died on the first policy
        # update. Only the number of steps and prompts shrinks.
        args.max_steps = 2
        args.output_dir = args.output_dir + "_smoke"

    import datasets
    from safetensors.torch import load_file as load_safetensors
    from trl import GRPOConfig, GRPOTrainer

    from Dataset.ctrate_dataset_test import CTRATEDataset_Test
    from Model.CTRG_3D import CTRG3D

    from reward_challenge_metrics import load_organizer_labels
    from reward_crgf1bleu import CRGF1BLEUReward

    if args.per_device_train_batch_size % args.num_generations:
        raise ValueError(
            f"per_device_train_batch_size {args.per_device_train_batch_size} is not a "
            f"multiple of num_generations {args.num_generations}; TRL counts COMPLETIONS "
            f"here, and the leftover would silently drop a partial group.")
    # TRL: generation_batch_size = per_device x num_processes x steps_per_generation,
    # steps_per_generation 은 gradient_accumulation_steps 를 따라간다
    # (grpo_config.py:1083). 프롬프트/스텝은 per_device 가 아니라 이 값으로 정해진다 --
    # pd 를 내리고 accum 을 같은 배수로 올려도 실험이 바뀌지 않는 이유가 이것이다.
    generation_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if generation_batch % args.num_generations:
        raise ValueError(
            f"generation batch {generation_batch} (= per_device "
            f"{args.per_device_train_batch_size} x accum {args.gradient_accumulation_steps}) "
            f"is not a multiple of num_generations {args.num_generations}.")
    prompts_per_step = generation_batch // args.num_generations
    print(f"[batch] per_device {args.per_device_train_batch_size} x accum "
          f"{args.gradient_accumulation_steps} -> generation batch {generation_batch} "
          f"completions = {prompts_per_step} prompts/step; "
          f"{args.max_steps} steps = {args.max_steps * prompts_per_step} prompts", flush=True)

    cache_dir = Path(args.cache_dir)
    meta = json.loads((cache_dir / "train_meta.json").read_text())
    if meta["checkpoint_tag"] != Path(args.ckpt).name:
        raise RuntimeError(
            f"cache was built from {meta['checkpoint_tag']} but the policy starts from "
            f"{Path(args.ckpt).name}. Perceiver/fc weights differ, so the cached "
            f"features do not belong to this model. Rebuild the cache.")
    index = json.loads((cache_dir / "train_index.json").read_text())
    cache = np.load(cache_dir / "train.npy", mmap_mode="r")
    print(f"[cache] {cache.shape} {cache.dtype} from {cache_dir}", flush=True)

    # ---- the frozen subset -- NOT the whole split --------------------------
    # The pilot built prompts for all 23,869 volumes and then used 500 of them.
    # The subset is also the definition domain of the BLEU floor and the F1
    # corpus, so it has to be this exact list, in this exact order.
    picked = json.loads(Path(args.subset_json).read_text())
    if args.smoke:
        # More prompts than the 2 steps consume, deliberately: TRL's sampler may
        # drop a short last batch, and a dataset sized to the step count exactly
        # would then fail for a reason that has nothing to do with the wiring.
        picked = picked[:max(64, prompts_per_step * args.max_steps * 2)]
    missing = [n for n in picked if n not in index]
    if missing:
        raise RuntimeError(f"{len(missing)} subset volumes are not in the cache index, "
                           f"e.g. {missing[:3]}")

    ds_src = CTRATEDataset_Test(
        tokenizer_path=args.lang_encoder_path,
        # B 가 학습한 그 경로 그대로. 옛 템플릿을 주면 B 가 본 적 없는 프롬프트가 되고,
        # 그건 loss 곡선에 아무 표시도 남기지 않는 분포 이동이다.
        json_path=args.train_json_path,
        rve_manifest_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/train/mapping_path.csv",
        cls_template_path=args.cls_template_path,
        ret_template_path=args.ret_template_path,
        image_num=args.num_visual_tokens, max_seq=args.max_seq, use_fast=False,
        use_cls_template=True, use_ret_template=True,
        prompts_only=args.prompts_only,
    )
    tok = ds_src.tokenizer
    by_name = {n: i for i, n in enumerate(ds_src.df["sample_name"].tolist())}
    print(f"[data] {len(picked)} subset prompts ({prompts_per_step}/step x "
          f"{args.max_steps} steps = {prompts_per_step * args.max_steps} draws)", flush=True)

    prompts, prompt_ids, gt_reports = {}, {}, {}
    for n in picked:
        s = ds_src[by_name[n]]
        text = s["question"]
        if "</ct>\n" not in text:
            raise RuntimeError(f"{n}: prompt has no '</ct>' boundary for the case marker")
        text = text.replace("</ct>\n", CASE_MARKER.format(row=index[n]), 1)
        prompts[n] = text
        gt_reports[n] = ds_src.df.iloc[by_name[n]]["report_metadata"]
        prompt_ids[n] = tok(text, add_special_tokens=False)["input_ids"]
        # TRL 1.10 dropped GRPOConfig.max_prompt_length and nothing warns, so an
        # over-long prompt would push the visual slots out of the window and the
        # run would go blind silently.
        if len(prompt_ids[n]) > args.max_seq:
            raise RuntimeError(
                f"{n}: prompt is {len(prompt_ids[n])} tokens, over --max_seq "
                f"{args.max_seq}. Re-measure the templates instead of raising it blindly.")

    model = CTRG3D(
        text_tokenizer_path=args.lang_encoder_path,
        lang_model_path=args.lang_encoder_path,
        pretrained_visual_encoder="", pretrained_adapter="",
        use_fast=False, enable_thinking=False,
        model_config_yaml=args.model_config_yaml,
        vis_dim=args.vis_dim, image_num=args.num_visual_tokens,
    ).bfloat16()
    ck = str(next(Path(args.ckpt).glob("model.safetensors")))
    model.load_state_dict(load_safetensors(ck, device="cpu"), strict=True)

    core, n_wrap = model.core_vocab_size, model.embedding_layer.num_wrapper_tokens
    vis_lo, vis_hi = core + n_wrap, core + n_wrap + args.num_visual_tokens
    lookup = build_key_lookup(prompt_ids, index, vis_lo, vis_hi, args.num_visual_tokens)
    print(f"[keys] {len(lookup)} unique prompt keys (window={KEY_WINDOW})", flush=True)

    lm = model.lang_model                     # already a PeftModel -> PEFT outermost
    wrapper_w = torch.cat([model.embedding_layer.image_token_weight,
                           model.embedding_layer.template_token_weight], dim=0).detach()
    lm.get_base_model().set_input_embeddings(CachedVisualEmbedding(
        lm.get_base_model().get_input_embeddings(), wrapper_w, cache, lookup,
        core, n_wrap, args.num_visual_tokens))
    del model.embedding_layer.vision_encoder
    lm = lm.cuda()

    # ---- reward -------------------------------------------------------------
    gt_labels = load_organizer_labels(
        "/data/drdcad/sunggu/0.Challenge/pillar-pretrain/train_predicted_labels.csv")
    reward = CRGF1BLEUReward(
        args.reference_json, gt_labels, gt_reports,
        alpha=args.alpha, gamma=args.gamma, lambda_bleu=args.lambda_bleu).with_radbert(
            **({"ckpt": args.radbert_ckpt} if args.radbert_ckpt else {}))
    print(f"[reward] corpus macro-F1={reward.base_f1:.4f}  r={reward.r:.4f}  "
          f"sigma_s={reward.sigma_s:.4f} sigma_f1={reward.sigma_f1:.4e} "
          f"sigma_bleu={reward.sigma_bleu:.4f}", flush=True)
    holes = [n for n in picked if n not in reward.bleu_floor]
    if holes:
        raise RuntimeError(
            f"{len(holes)} subset volumes have no BLEU floor, e.g. {holes[:3]}. "
            f"The reference build did not cover the subset.")

    ds = datasets.Dataset.from_dict({"prompt": [prompts[n] for n in picked],
                                     "sample_name": picked})
    cfg = GRPOConfig(
        output_dir=args.output_dir, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # Whatever still truncates at 768 contributes no tokens to the loss;
        # a cut-off report has fewer FPs than it earned.
        mask_truncated_completions=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup", warmup_steps=args.warmup_steps,
        beta=args.beta, temperature=args.temperature, top_p=args.top_p,
        bf16=True, logging_steps=1, save_strategy="steps", save_steps=100,
        report_to=["wandb"], run_name=Path(args.output_dir).name, seed=args.seed,
        gradient_checkpointing=True, log_completions=True,
    )
    trainer = GRPOTrainer(model=lm, reward_funcs=reward, args=cfg, train_dataset=ds,
                          processing_class=tok)
    if getattr(trainer, "ref_model", None) is not None:
        raise RuntimeError(
            "TRL built a separate reference model -- PEFT is not the outermost "
            "wrapper. That is a second full 8B in memory and the wrong KL baseline.")
    print(f"[trl] ref_model is None; beta={args.beta} "
          f"({'KL off' if args.beta == 0 else 'KL on'})", flush=True)

    resume = args.resume_from_checkpoint or None
    if resume:
        st = Path(resume) / "trainer_state.json"
        if not st.is_file():
            raise RuntimeError(f"no trainer_state.json in {resume}")
        done = json.loads(st.read_text())["global_step"]
        print(f"[resume] {resume}  global_step={done} -> max_steps={args.max_steps} "
              f"({args.max_steps - done} steps left)", flush=True)
        if done >= args.max_steps:
            raise RuntimeError(
                f"checkpoint is already at step {done} of {args.max_steps}; raise "
                f"--max_steps or there is nothing to resume.")
    trainer.train(resume_from_checkpoint=resume)

    emb = lm.get_base_model().get_input_embeddings()
    print(f"\n[check] visual injections: {emb.n_injected}  misses: {emb.n_missing}")
    if emb.n_injected == 0:
        raise RuntimeError("no visual features were ever injected -- the run was blind")
    print(f"[check] reward scored {reward.n_scored} rollouts")
    print("[done]", args.output_dir)


if __name__ == "__main__":
    main()
