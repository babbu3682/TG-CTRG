#!/bin/bash
# ==============================================================================
# PETRG-3D joint training with the Qwen3-8B backbone.
#
# "joint" means all modules are trained simultaneously:
#   * 3D ViT visual encoder  (frozen by default -- set
#     ``finetune_ct_feature_extractor=True`` to unfreeze it)
#   * CT / PET Perceiver resampler + projection heads
#   * LoRA adapter on the LLM
# ==============================================================================

# ==============================================================================
# Ablation knobs (overridable from the environment; run_sbatch_vt sets them).
#
# num_visual_tokens = PerceiverResampler latents = number of <image_ct*>
# placeholders in the prompt. The Atlas encoder emits 65,536 tokens
# (32x32x64), so 128 latents is a 512:1 compression; this is the variable the
# vt128/vt512/vt1024 ablation sweeps. src/train.py feeds the single value to
# both the model and the dataset, and CTRG3D raises if the tokenizer's
# vocabulary does not come out at core + 6 + num_visual_tokens.
# ==============================================================================
num_visual_tokens="${NUM_VISUAL_TOKENS:-128}"

# Renamed vs the previous run ("ctrg3d-qwen3-8B-joint") because the vision
# encoder, the cls/ret templates and the prompt all changed -- keeping the old
# name would drop new checkpoints into the same outputs/ dir (and the same W&B
# run name) as the 10-epoch run whose checkpoint-11936 it supersedes.
experiment_name="${EXPERIMENT_NAME:-ctrg3d-qwen3-8B-pillar0os-lz4-vt${num_visual_tokens}}"
bf16=True
# Derived from the job id so that two ablation arms landing on the same node
# cannot collide on torchrun's rendezvous port (a fixed port made them fight).
master_port="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-5335} % 20000))}"

# W&B: run_name below (= experiment_name) is what shows up as the run name;
# set WANDB_API_KEY / WANDB_PROJECT in run_sbatch (or your shell) before
# launching. train.py no longer force-disables W&B.
report_to="wandb"

# -------------------- LLM backbone --------------------
lang_encoder_path="${LANG_ENCODER_PATH:-/path/to/Qwen3-8B}"
tokenizer_path="${TOKENIZER_PATH:-${lang_encoder_path}}"

# -------------------- Visual encoder (Atlas) --------------------
# The Atlas vision encoder self-loads its checkpoint from model_config_yaml's
# `pretrained:` field below, so this is left empty. Only set
# PRETRAINED_VISUAL_ENCODER if you need to override with a separately saved
# `visual.*`-prefixed state dict (legacy RadFM ViT-3D loading path).
pretrained_visual_encoder="${PRETRAINED_VISUAL_ENCODER:-}"
pretrained_adapter=""

# -------------------- Training strategy --------------------
training_stage="joint"
finetune_ct_feature_extractor=False

# -------------------- Data --------------------
petct_image_folder="${PETRG_LYM_TRAIN_IMAGES:-/path/to/petrg_lym/train/images_mv_leg}"
petct_report_folder="${PETRG_LYM_TRAIN_REPORTS:-/path/to/petrg_lym/train/reports}"
template_path="${PETRG_LYM_TEMPLATE:-/path/to/petrg_lym/template.json}"

monai_cache_dir="/tmp/monai_cache_persistent/${experiment_name}"
output_dir="${PETRG3D_OUTPUT_DIR:-./outputs}/${experiment_name}"
deepspeed_config="../ds_configs/deepspeed_zero2.json"

mkdir -p "${monai_cache_dir}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${experiment_name}"
mkdir -p "${TRITON_CACHE_DIR}"

# -------------------- Optimizer / schedule --------------------
# 5e-5 at global batch 16, unchanged by the 15 -> 30 epoch ceiling, and that is
# a consequence of the schedule rather than an omission: lr_scheduler_type is
# constant_with_warmup, whose LR at step k depends only on k and warmup_steps --
# there is no total-steps term for the horizon to enter. A decay schedule would
# have to be re-tuned for 30 epochs; this one does not, and with early stopping
# ending the run mid-horizon a decay schedule would never reach its endpoint
# anyway, so its only effect would be to make the effective LR trajectory depend
# on a horizon we do not intend to reach.
#
# Scale it with the batch, not with the epochs: derived configs that raise the
# global batch use sqrt scaling (5e-5 * sqrt(global/16)) -- see
# qwen3_joint_template_new_encoder_template.sh.
#
# If a run's eval_loss is still falling when it hits epoch 30, the lever is a
# LOWER LR (3e-5 was the value considered), not more epochs -- but that is an
# experiment with its own arm, not a default, because it invalidates comparison
# with every number measured at 5e-5.
learning_rate=5e-5
# Effective global batch size is held at 16 across all three ablation arms
# (per_device * 2 GPUs * accum), so the LR schedule and optimization dynamics
# stay comparable. Note accumulation does NOT change memory -- it only splits
# the global batch into sequential micro-steps -- so filling the card means
# maximizing per_device and leaving accum at 1. With global batch 16 on 2 GPUs,
# per_device=8/accum=1 is therefore the maximum, and it now fits every arm:
#
#   per-GPU budget at per_device=8 (measured parts marked *):
#     bf16 weights                              ~17 GB
#     perceiver fwd+bwd over 65,536 keys*    17.9 / 18.5 / 19.3 GB (128/512/1024)
#     LLM activations (grad ckpt, ~ bs*max_seq) ~12 / 13.5 / 16 GB
#     activ + encoder transients                 ~2-10 GB (frozen, no graph)
#     ------------------------------------------------------------------
#     total                                     ~48 / 50 / 53 GB of 80
#
# What made 8 possible is the SDPA rewrite of PerceiverAttention (see
# Model/helpers.py): the old explicit einsum kept the full
# (heads, latents, 65,536) score matrix for backward, which cost 12.7 GB at
# per_device=4/128 latents and scaled linearly with the latent count (~45 GB at
# 1024). Post-rewrite, perceiver memory is essentially independent of the
# latent count, so all three arms share one batch configuration.
#
# Gradient checkpointing on the LLM stays on: turning it off would spend the
# freed memory many times over (~30 GB of hidden states alone at bs8/L3072).
per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
# 30 epochs as a CEILING, with EarlyStoppingCallback (patience 5) deciding the
# actual end. Standing setting as of 2026-08-24; was 15.
#
# Why a ceiling and not a plan. Measured on the ep15 sweep at global batch 16
# (22,380 steps), eval_loss bottoms at ep11 / ep10 / ep9 for vt128 / vt512 /
# vt1024 and then rebounds +0.025 / +0.049 / +0.037 by ep15. Patience 5 would
# have stopped those runs at ep14-16, i.e. well inside a 30-epoch ceiling. So
# raising 15 -> 30 costs nothing on those settings; it only removes the ceiling
# as the thing that ends a run.
#
# It is NOT cosmetic, because a smaller global batch (or a lower LR) moves the
# minimum later and 15 was already binding there: the newenc runs at global 48
# did 7,470 steps in 15 epochs -- a THIRD of the ep15 sweep's updates -- and
# their eval_loss was still falling monotonically at ep15 (best == last, gap
# only +0.029). Those two runs stopped because they ran out of epochs, not
# because they converged. 30 is what makes the stopping criterion the metric
# rather than the calendar.
#
# Disk: save_total_limit below caps what is kept, so a 30-epoch ceiling does not
# multiply storage. Checkpoints are still written every epoch, so a timeout
# leaves usable ones.
num_train_epochs="${NUM_TRAIN_EPOCHS:-30}"
# 5 evaluations (= 5 epochs, evaluation_strategy=epoch) without improvement.
# Late enough to ride out one-epoch plateaus (vt512's ep10 bottom followed a
# 0.0003 dip), early enough to skip the overfitting tail.
early_stopping_patience="${EARLY_STOPPING_PATIENCE:-5}"
# Eval loop: eval_loss per epoch, keep the best checkpoint. evaluation_strategy
# must match save_strategy for load_best_model_at_end to work.
evaluation_strategy="epoch"
save_strategy="epoch"
# 10, not 5: the overfitting sweep needs every epoch's weights kept, because the
# question it answers -- does eval_loss's minimum coincide with the best
# generation quality? -- can only be settled by scoring several epochs with the
# VLM3D scorer, and the ablation could not answer it precisely because
# save_total_limit=1 there had already discarded the intermediate checkpoints.
# Each checkpoint is ~49 GB (18 GB model.safetensors + ~31 GB of DeepSpeed
# optimizer state), so 10 costs ~490 GB per run.
# Must stay > early_stopping_patience so the rotation cannot outrun the stopping
# window; 10 > 5 holds. (Trainer also protects best_model_checkpoint from
# rotation while load_best_model_at_end=True, so the selected epoch survives
# regardless -- this bound is about keeping the neighbours for scoring.)
save_total_limit="${SAVE_TOTAL_LIMIT:-10}"
per_device_eval_batch_size=4
load_best_model_at_end=True
metric_for_best_model="eval_loss"
greater_is_better=False
weight_decay=0.0
# 100 steps of warmup then constant, matching the PETRG-3D paper's recipe
# (supplementary B.1). Was 30; at global batch 16 an epoch is 1,492 steps, so
# either value is a small fraction of training, but there is no reason to
# deviate from the reference recipe.
#
# STEPS, deliberately, not warmup_ratio, and unchanged by the 30-epoch ceiling.
# Warmup here is a startup transient that protects the freshly-initialised
# perceiver latents from the first large updates; that job is done after a fixed
# number of updates and has nothing to do with how long the run then goes.
# warmup_ratio would tie it to total steps, so doubling the ceiling would double
# the warmup -- 15 -> 30 epochs at global 16 would take it from ~1,340 to ~2,690
# steps, changing early optimisation on every run for no reason.
warmup_steps=100
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=8
dataloader_pin_memory=False
logging_steps=1

# -------------------- Data preprocessing --------------------
z_chunk_size=64
z_chunk_stride=32
xy_size=256
# max_seq is derived from num_visual_tokens rather than set independently: each
# visual token occupies one LLM position, and the dataset pads every sample to
# max_length, so the LLM pays for the full max_seq on every step whatever the
# real length. Pairing the two by hand is how a run ends up either truncating
# answers or burning ~25% of its compute on padding.
#
# Measured by replicating CTRATEDataset_Train.__getitem__'s text path exactly
# (same registered special tokens so each <image_ct*> counts as one token;
# Qwen3 chat template; no sentence dropout = worst case of system + <ct> +
# instruction + full cls/ret templates + answer) over every sample of both
# splits. use_fast=False/True agree token-for-token on the 50 longest.
#
#   visual tokens |  train mean / p99 / max  |  valid max  |  max_seq
#   --------------+--------------------------+-------------+---------
#        128      |     1097 / 1641 / 2414   |    2087     |   2688
#        512      |     1481 / 2025 / 2798   |    2471     |   3072
#       1024      |     1993 / 2537 / 3310   |    2983     |   3584
#
# Each max_seq is the measured max plus >=256 tokens of slack, rounded up to a
# multiple of 64. Nothing is truncated in either split. Re-run the measurement
# (scratchpad maxlen_vt.py) if the cls/ret template files or the prompt change.
case "$num_visual_tokens" in
    128)  max_seq="${MAX_SEQ:-2688}" ;;
    512)  max_seq="${MAX_SEQ:-3072}" ;;
    1024) max_seq="${MAX_SEQ:-3584}" ;;
    *)
        if [ -z "${MAX_SEQ:-}" ]; then
            echo "num_visual_tokens=${num_visual_tokens} has no measured max_seq;" \
                 "measure it and pass MAX_SEQ=<value> explicitly." >&2
            exit 1
        fi
        max_seq="$MAX_SEQ"
        ;;
esac

# -------------------- Prompting --------------------
use_template=True
enable_thinking=False
use_fast=False
enable_augmentation=False

# cls/ret auxiliary-prediction templates (see build_instruction() in
# ctrate_dataset_train.py). Both are the *updated* template sets, regenerated
# with the open-source Pillar-0 chest CT encoder on the lz4 half-split cache
# (i.e. the same encoder configured below), replacing the earlier
# CTRATE_FT_Atlas_abn_bce_freeze_original_pillar / ..._retrieval_pillar_freeze
# outputs -- those directories no longer exist on disk anyway.
#
#   * cls: CTRATE_ABN_cls_report_templates_p90n20 -- threshold-based report
#     templates (positive >= 0.90, negative <= 0.20, everything else omitted;
#     train macro-AUROC 0.885 / PPV 0.819, dev 0.869 / 0.807, ~9.0 sentences
#     per sample). Unlike the old train_abn_reports.json, whose
#     pred_key_phrases was empty for all 23,869 train samples, this file has
#     real sentences for train *and* dev, so train and eval_loss finally see
#     the same kind of cls content. Its records use a "report" field (a list
#     of finished sentences) instead of "pred_key_phrases";
#     _load_key_phrase_template() accepts either.
#   * ret: CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden -- retrieval
#     predictions from the last-hidden-state variant, same
#     accession/pred_key_phrases schema as before.
#
# Both files cover 23,869 train / 1,552 dev accessions, matching
# merged_{train,valid}_filtered_prob_half.json 1:1.
use_cls_template=True
use_ret_template=True
cls_template_path="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_ABN_cls_report_templates_p90n20/cls_report_train.json"
ret_template_path="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden/train_ret_reports.json"

# -------- Add ---------
# NOTE: this must be the repo's own reshaped copy (vision_cfg.model_config...
# layout expected by Model/pillar_embedding_layer.py), NOT the
# pillar-finetune training config of the same basename -- that one has a
# completely different top-level structure (experiment/main/engine/dataset/
# model/...) and will crash with AttributeError: no attribute 'vision_cfg'.
#
# Switched from ctrate_abnclass_finetune.yaml (in-house PILLAR0
# ...clip_merlin_chest_ct_r384... epoch_50.pt, patch_size 8x8x8) to the
# open-source Pillar-0 chest CT release -- the same encoder that generated the
# cls/ret templates above. Two consequences worth knowing:
#   * the release's patch embed is 8x8x4, so `activ` is (B, 1152, 32, 32, 64) =
#     65,536 visual tokens instead of 32,768. The PerceiverResampler still
#     emits 128 latents, so max_seq / the LLM side is unchanged; only the
#     perceiver's activation memory grows (measured on an A100: 6.5 -> 12.7 GB
#     at per_device_train_batch_size=4, and the previous run peaked at
#     32.6/80 GB, so batch size 4 is kept).
#   * epoch_50.pt has since been deleted from pillar-pretrain/logs, so the old
#     yaml no longer loads at all.
model_config_yaml="/data/drdcad/sunggu/0.Challenge/PETRG-3D/src/Model/ctrate_pillar0_chest_opensource.yaml"
# Matches the Atlas vision encoder's actual output channel count (empirically
# verified: model_config's 3 multiscale levels concat to 1152 channels), not
# ModelArguments' generic default of 768. Unchanged by the encoder swap.
vis_dim=1152
# Data: same "half" split jsons as before, but read from the lz4 RVE cache
# (ctrate_cache_1.25mm_lz4) rather than the video cache. That cache holds
# exactly the half split (23,880 train / 1,552 valid .tar.lz4 files, covering
# every sample_name in the jsons) and its manifests store *absolute*
# image_cache_paths -- see _resolve_rve_path() in ctrate_dataset_train.py.
train_json_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_train_filtered_prob_half.json"
train_rve_manifest_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/train/mapping_path.csv"

# Validation split (eval_loss per epoch, see evaluation_strategy above).
valid_json_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_valid_filtered_prob_half.json"
valid_rve_manifest_path="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/valid/mapping_path.csv"
valid_cls_template_path="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_ABN_cls_report_templates_p90n20/cls_report_dev.json"
valid_ret_template_path="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden/valid_ret_reports.json"