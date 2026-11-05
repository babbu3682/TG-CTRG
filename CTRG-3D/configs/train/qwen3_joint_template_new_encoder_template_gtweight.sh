#!/bin/bash
# ==============================================================================
# new-encoder-template arm PLUS the GT-sentence token-weighted loss.
#
#   ./train.sh qwen3_joint_template_new_encoder_template_gtweight
#   LABEL_TOKEN_WEIGHT=3.73 ./train.sh qwen3_joint_template_new_encoder_template_gtweight
#
# This is the "+" arm: the base setting is
# configs/train/qwen3_joint_template_new_encoder_template.sh (ctrg-best with the
# lapa encoder and the templates that encoder produced), and the ONE thing
# layered on top is the colleague's idea -- upweight the loss on report tokens that
# belong to a sentence carrying an abnormality label.
#
# So the pair that decides anything is:
#
#   base   ctrg3d-qwen3-8B-lapa-lz4-vt128-new-encoder-template
#   +this  ctrg3d-qwen3-8B-lapa-lz4-vt128-new-encoder-template-gtw5.0
#
# and the comparison is CRG / macro-F1 / BLEU from generated reports on the FULL
# valid split via test.py -- not eval_loss, which is a weighted mean here and a
# plain one there and therefore not comparable across the two.
#
# WHY THIS SOURCES THE BASE ARM AND RESTATES FOUR VARIABLES
# ---------------------------------------------------------
# The obvious composition -- source qwen3_joint_template_gtweight.sh -- does not
# work. That file sources qwen3_joint_template_tokenweight.sh, which sources
# qwen3_joint_template.sh, which assigns the encoder yaml, vis_dim, batch and LR
# unconditionally. Sourcing it after the base arm would silently put this run back
# on the OLD encoder with the OLD batch: exactly the failure this arm exists to
# avoid, and invisible in every log. So the base arm is sourced (it is the thing
# that must not be overridden) and the four variables the tokenweight config adds
# are restated below. They are four lines and they are checked against their
# source in the assertion at the bottom of this file.
#
# WHY THE CALIBRATION NUMBERS CARRY OVER
# --------------------------------------
# qwen3_joint_template_gtweight.sh calibrates w from f, the label-bearing share of
# SUPERVISED tokens, measured at f = 38.49% (3,128,897 of 8,128,940) over all
# 23,850 train samples. Supervised tokens are the answer only -- the prompt is
# masked with label[:prompt_len] = -100 -- and the answer is the GT report, which
# the template change does not touch. The cls/ret templates live entirely in the
# prompt. So f is template-independent and the calibration transfers:
#
#     w = 1.00  ->  38.5% of the loss   (weighting disabled)
#     w = 3.73  ->  70.0%
#     w = 5.00  ->  75.8%   <- default, same as the old-base arm
#     w = 6.39  ->  80.0%
#
# That is an argument, not a measurement, so it is checked rather than assumed:
# scripts/audit_token_weight.py is re-run against THIS config and its log lands in
# scripts/outputs/audit_gtweight_newenc.log. If f has moved, w moves with it.
#
# THE SPLITS ARE 20 SAMPLES SHORTER, AND THAT IS DELIBERATE
# ----------------------------------------------------------
# The GT spans cover 23,850 of 23,869 train samples (1,551 of 1,552 valid), so this
# arm trains on 0.08% fewer volumes than the base. Accepted for this experiment to
# match the label author's N exactly; see the header of
# configs/train/qwen3_joint_template_gtweight.sh for the alternatives that were
# rejected. It does mean base-vs-this is not strictly single-factor.
# ==============================================================================

_vt="${NUM_VISUAL_TOKENS:-128}"
_w="${LABEL_TOKEN_WEIGHT:-5.0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ctrg3d-qwen3-8B-lapa-lz4-vt${_vt}-new-encoder-template-gtw${_w}}"

# The base arm. Everything it sets -- encoder yaml, vis_dim, templates, batch, LR
# -- must survive this file, so nothing below may re-source the baseline.
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/qwen3_joint_template_new_encoder_template.sh"

# -------------------- the one factor this arm adds --------------------
# Restated from qwen3_joint_template_tokenweight.sh (see header for why it is not
# sourced). label_token_weight is what turns the weighting on; weight_eval_loss
# keeps eval_loss on the same objective as train_loss so early stopping is not
# selecting against the thing being optimised.
label_token_weight="${_w}"
weight_eval_loss="${WEIGHT_EVAL_LOSS:-True}"

# The label author's spans, re-keyed by sample_name, and the splits restricted to
# the samples those spans cover (so the dataset's construction-time coverage check
# passes without being weakened). Built by scripts/build_gt_sentence_weights.py --
# see the header of qwen3_joint_template_gtweight.sh for the exact invocation.
_gt_dir="${GT_LABEL_DIR:-/data/drdcad/sunggu/0.Challenge/PETRG-3D/data/sentence_labels}"
sentence_labels_path="${SENTENCE_LABELS_PATH:-${_gt_dir}/train_gt_sentence_labels.json}"
valid_sentence_labels_path="${VALID_SENTENCE_LABELS_PATH:-${_gt_dir}/valid_gt_sentence_labels.json}"
train_json_path="${TRAIN_JSON_PATH:-${_gt_dir}/merged_train_gtcov.json}"
valid_json_path="${VALID_JSON_PATH:-${_gt_dir}/merged_valid_gtcov.json}"

# Fail here rather than 40 minutes into a queued job.
for _f in "$sentence_labels_path" "$valid_sentence_labels_path" \
          "$train_json_path" "$valid_json_path"; do
    if [ ! -s "$_f" ]; then
        echo "qwen3_joint_template_new_encoder_template_gtweight: missing or empty ${_f}" >&2
        echo "  run scripts/build_gt_sentence_weights.py first" >&2
        exit 1
    fi
done
unset _f _gt_dir

# The whole point of this file is that the base arm's encoder and templates are
# still in force. A future edit to the source chain that quietly reverts either is
# the one mistake that would not show up in any log, so it fails here instead.
case "$model_config_yaml" in
    *ctrate_pillar0_chest_lapa.yaml) ;;
    *) echo "new_encoder_template_gtweight: model_config_yaml reverted to ${model_config_yaml}" >&2
       exit 1 ;;
esac
case "$cls_template_path" in
    *lapa_f1max*) ;;
    *) echo "new_encoder_template_gtweight: cls_template_path reverted to ${cls_template_path}" >&2
       exit 1 ;;
esac
case "$ret_template_path" in
    *retrieval_self_lapa*) ;;
    *) echo "new_encoder_template_gtweight: ret_template_path reverted to ${ret_template_path}" >&2
       exit 1 ;;
esac
if [ "$vis_dim" != "384" ]; then
    echo "new_encoder_template_gtweight: vis_dim is ${vis_dim}, expected 384 for the single-scale tower" >&2
    exit 1
fi
