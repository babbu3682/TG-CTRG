#!/bin/bash
# ==============================================================================
# new-encoder-template arm.
#
#   ./train.sh qwen3_joint_template_new_encoder_template
#
# Differs from configs/train/qwen3_joint_template.sh in THREE factors, all
# belonging to the same upgrade -- named honestly rather than called
# "new-template", because the templates were produced BY the new encoder and
# using one without the other is not an experiment anyone wants:
#
#   1. cls/ret prompt templates  -> the lapa_f1max / retrieval_self ones
#   2. vision encoder weights    -> lapa_pillar0_lz4_v1/epoch_12.pt
#   3. vision tower              -> corrected chest_ct stem + single-scale readout
#
# (2) and (3) are BUG FIXES, not tuning knobs -- see
# Model/atlas_encoders_pillar0.py (the anisotropic stride sat on W instead of D,
# so the attention-window grid was transposed relative to the feature map) and
# Model/pillar_mmatlas_lapa.py (the three-scale nearest-interpolate also
# mixed up axis order). The user has already verified the encoder's effect
# separately, so this arm does not try to attribute the gain between the three.
#
# Everything else -- visual tokens, max_seq, schedule, template dropout,
# 15 epochs, early stopping, the splits -- is inherited by sourcing the baseline,
# so this arm cannot drift from it by accident. Batch and LR are the exception and
# are NOT a fourth factor: the card is always filled (see the block at the bottom),
# and LR follows the batch. That is standing practice, not part of the experiment.
#
#   old:  CTRATE_ABN_cls_report_templates_p90n20                 (cls)
#         CTRATE_FT_Atlas_retrieval_pillar_freeze_lasthidden     (ret)
#   new:  CTRATE_ABN_cls_report_templates_lapa_f1max      (cls)
#         CTRATE_FT_Atlas_retrieval_self_lapa/templates_ep32_pp  (ret)
#
# MAX_SEQ WAS RE-MEASURED, NOT ASSUMED
# ------------------------------------
# Changing the templates changes the prompt length, and the answer sits at the
# END of the sequence -- so a too-small max_seq eats the training target first
# while every loss curve still looks normal. Measured over both splits in full
# with scripts/verify_maxseq_newtemplate.py (log:
# scripts/outputs/maxseq_newtemplate.log):
#
#   split   n       mean   p99    max      cls tpl (mean/max)  ret tpl (mean/max)
#   train   23,869  1157   1728   2535     144 / 148           313 / 634
#   valid    1,552  1157   1732   2098     144 / 148           311 / 598
#
#   vt128 worst case 2535  vs configured 2688  ->  153 tokens of slack, no truncation
#   vt512 worst case 2919  vs configured 3072  ->  153
#   vt1024 worst case 3431 vs configured 3584  ->  153
#
# The new templates come out SHORTER than the old ones despite the new cls file
# emitting a sentence for all 18 labels rather than only the fired ones -- those
# sentences are short (cls 144 tokens on average), and the post-processed ret
# file is tighter. So max_seq is left at the inherited value on purpose: the
# measured-to-fit batch/accumulation ladder in the parent config is calibrated
# against 2688, and lowering max_seq to the measured 2624 would buy nothing
# while invalidating that calibration.
#
# A bonus worth knowing: the new cls file covers 23,869/23,869 train samples with
# a non-empty template. The old one left 19 empty, which silently dropped the
# <cls_template> block from those prompts.
# ==============================================================================

# Distinct output dir / W&B run name. Exported BEFORE sourcing because the base
# config reads ${EXPERIMENT_NAME:-...} and derives output_dir and the caches
# from it -- without this the run would land on top of the baseline's outputs.
_vt="${NUM_VISUAL_TOKENS:-128}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ctrg3d-qwen3-8B-lapa-lz4-vt${_vt}-new-encoder-template}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/qwen3_joint_template.sh"

# -------------------- the one factor this arm changes --------------------
_cls_dir="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_ABN_cls_report_templates_lapa_f1max"
_ret_dir="/data/drdcad/sunggu/0.Challenge/pillar-finetune/logs/CTRATE/CTRATE_FT_Atlas_retrieval_self_lapa/templates_ep32_pp"

cls_template_path="${CLS_TEMPLATE_PATH:-${_cls_dir}/cls_report_train.json}"
valid_cls_template_path="${VALID_CLS_TEMPLATE_PATH:-${_cls_dir}/cls_report_dev.json}"
ret_template_path="${RET_TEMPLATE_PATH:-${_ret_dir}/train_ret_reports_pp.json}"
valid_ret_template_path="${VALID_RET_TEMPLATE_PATH:-${_ret_dir}/dev_ret_reports_pp.json}"

# Fail here rather than 40 minutes into a queued job.
for _f in "$cls_template_path" "$valid_cls_template_path" \
          "$ret_template_path" "$valid_ret_template_path"; do
    if [ ! -s "$_f" ]; then
        echo "qwen3_joint_template_newtemplate: missing or empty ${_f}" >&2
        exit 1
    fi
done
# -------------------- factor 2+3: the encoder --------------------
# vis_dim MUST move with the tower: one scale of 384, not three concatenated
# into 1152. Getting this wrong is a shape error at the Perceiver, not a silent
# regression, but state it anyway.
model_config_yaml="${MODEL_CONFIG_YAML:-/data/drdcad/sunggu/0.Challenge/PETRG-3D/src/Model/ctrate_pillar0_chest_lapa.yaml}"
vis_dim="${VIS_DIM:-384}"

if [ ! -s "$model_config_yaml" ]; then
    echo "qwen3_joint_template_new_encoder_template: missing ${model_config_yaml}" >&2
    exit 1
fi
_enc="/data/drdcad/sunggu/0.Challenge/pillar-pretrain/logs/lapa_pillar0_lz4_v1/checkpoints/epoch_12.pt"
if [ ! -s "$_enc" ]; then
    echo "qwen3_joint_template_new_encoder_template: missing encoder ${_enc}" >&2
    exit 1
fi

unset _f _cls_dir _ret_dir _enc
# -------------------- filling the card (not a factor, a default) --------------------
# The single-scale readout frees the parent config's largest batch-dependent
# term: perceiver fwd+bwd over 65,536 keys cost 17.9 GB per GPU at per_device=8,
# and 1,024 keys at width 384 costs ~1. The encoder forward itself is NOT
# cheaper -- the volume still goes through the whole Atlas; only the readout
# changed -- so the saving is that one term.
#
#   per-GPU estimate       weights  perceiver  LLM act  encoder transients  total
#   pd=8, old encoder        17       17.9       12         ~2-10            ~48
#   pd=8, this encoder       17       ~1         12         ~2-10            ~31
#   pd=16, this encoder      17       ~1         24         ~4-20            ~55
#   pd=24, this encoder      17       ~2         36         ~6-30            ~74   <- here
#
# pd=24 rather than 16 because leaving 25 GB of an 80 GB card unused is waste, not
# caution: the card gets filled, and the OOM ladder in run_sbatch_newenc makes
# being wrong cost minutes rather than a resubmission. Note that only per_device
# moves memory -- gradient_accumulation splits the optimiser step and changes
# nothing about the peak, so it cannot spend what the readout freed.
#
# There is no gradient checkpointing anywhere in this repo, so the LLM-activation
# term really is linear in per_device; that is why the table above scales the way
# it does and why 24 is the last rung that fits.
#
# LR MOVES WITH IT. 5e-5 was set at global batch 16; pd=24 x 2 GPUs x accum 1 is
# global 48. Tripling the batch cuts gradient noise by sqrt(3), so sqrt scaling for
# AdamW gives 5e-5 * sqrt(3) = 8.66e-5. Linear scaling (1.5e-4) is the SGD-regime
# rule and is aggressive for a constant-schedule fine-tune. Override with
# LEARNING_RATE= to hold the old value -- defensible, it just under-steps.
#
# Epochs are data passes, so 15 still sees the same data; it is now 498 optimiser
# steps per epoch instead of 1,492. warmup_steps=100 stays a small fraction.
per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-24}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
learning_rate="${LEARNING_RATE:-8.66e-5}"

# Pure throughput, no optimisation state touched. The workers matter more than
# usual here: with the perceiver 64x cheaper the step may become dataloader-bound,
# and then workers -- not batch -- is the knob that helps. 2 GPUs x 12 = 24 of 32.
per_device_eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE:-12}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-12}"

