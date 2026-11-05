#!/bin/bash
#
# Launch PETRG-3D training.
#
# Usage:
#   ./train.sh <config_name>
#
# where <config_name> is the basename of a shell file under
# ``configs/train/`` (e.g. ``qwen3_joint_template`` for
# ``configs/train/qwen3_joint_template.sh``).
# The config file is ``source``'d before launching ``torchrun`` and is
# expected to export all the paths / hyper-parameters referenced below.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_name>" >&2
    exit 1
fi

config_file="$1"
script_name=$(basename "$0" .sh)

# shellcheck disable=SC1090
source "../configs/${script_name}/${config_file}.sh"

# Decide world size from the current CUDA_VISIBLE_DEVICES / SLURM env, unless
# the config already decided (the ablation config does, so that the batch/accum
# split it computes and the world size torchrun launches can never disagree).
#
# The last resort is nvidia-smi, not a hardcoded 2: an sinteractive allocation
# sets neither CUDA_VISIBLE_DEVICES nor SLURM_GPUS_ON_NODE, and assuming 2 there
# starts a rank on cuda:1 that does not exist ("device_id cuda:1 is out of
# range. Please use a device index less than the number of accelerators
# available: 1").
if [ -z "${nproc_per_node:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        nproc_per_node="$SLURM_GPUS_ON_NODE"
    elif [ -n "${SLURM_GPUS:-}" ]; then
        nproc_per_node="$SLURM_GPUS"
    else
        nproc_per_node=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
        [ "${nproc_per_node:-0}" -ge 1 ] || nproc_per_node=1
    fi
fi
echo "train.sh: launching torchrun with nproc_per_node=${nproc_per_node}"

# Forward the optional model-side flags only when the config sets them.
model_extra_args=""
if [ -n "${finetune_ct_feature_extractor:-}" ]; then
    model_extra_args="$model_extra_args --finetune_ct_feature_extractor ${finetune_ct_feature_extractor}"
fi
if [ -n "${training_stage:-}" ]; then
    model_extra_args="$model_extra_args --training_stage ${training_stage}"
fi
if [ -n "${pretrained_model_path:-}" ]; then
    model_extra_args="$model_extra_args --pretrained_model_path ${pretrained_model_path}"
fi
if [ -n "${report_to:-}" ]; then
    model_extra_args="$model_extra_args --report_to ${report_to}"
fi

# DeepSpeed is optional. Set ``deepspeed_config=""`` in the config (or export
# DEEPSPEED_CONFIG="") to launch plain torchrun instead -- DeepSpeed builds CUDA
# ops at start-up and dies with MissingCUDAException on hosts without
# CUDA_HOME, which is every bare node here, so the batch jobs run inside the
# CUDA devel container while an interactive smoke/ablation run can just skip it.
# Only worth skipping on a single GPU: without ZeRO-2 there is no optimizer or
# gradient sharding, and the model trains in pure bf16 (no fp32 master weights).
ds_extra_args=""
if [ -n "${deepspeed_config:-}" ]; then
    ds_extra_args="--deepspeed ${deepspeed_config}"
fi

# Resume support. Set RESUME_FROM_CHECKPOINT to a checkpoint-* directory, or to
# "True" to let the Trainer pick the newest one under output_dir. Needed after a
# late crash: without it, relaunching the same output_dir restarts at step 0 and
# overwrites the per-epoch checkpoints already written.
resume_extra_args=""
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    resume_extra_args="--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
    echo "train.sh: resuming from ${RESUME_FROM_CHECKPOINT}"
fi

# Forward the optional eval-loop flags only when the config sets a
# validation split (valid_json_path). Standard transformers.TrainingArguments
# fields (per_device_eval_batch_size, load_best_model_at_end, ...) are
# accepted by HfArgumentParser without needing a dedicated dataclass field.
eval_extra_args=""
if [ -n "${valid_json_path:-}" ]; then
    eval_extra_args="$eval_extra_args --valid_json_path ${valid_json_path}"
    eval_extra_args="$eval_extra_args --valid_rve_manifest_path ${valid_rve_manifest_path:-}"
    eval_extra_args="$eval_extra_args --valid_cls_template_path ${valid_cls_template_path:-}"
    eval_extra_args="$eval_extra_args --valid_ret_template_path ${valid_ret_template_path:-}"
    eval_extra_args="$eval_extra_args --per_device_eval_batch_size ${per_device_eval_batch_size:-1}"
    eval_extra_args="$eval_extra_args --load_best_model_at_end ${load_best_model_at_end:-True}"
    eval_extra_args="$eval_extra_args --metric_for_best_model ${metric_for_best_model:-eval_loss}"
    eval_extra_args="$eval_extra_args --greater_is_better ${greater_is_better:-False}"
    eval_extra_args="$eval_extra_args --early_stopping_patience ${early_stopping_patience:-0}"
fi

# Token-weighted loss. Forwarded only when the config sets a spans file, so
# every existing config produces a byte-identical command line and train.py
# falls through to the language model's own uniform cross-entropy.
tw_extra_args=""
if [ -n "${sentence_labels_path:-}" ]; then
    tw_extra_args="$tw_extra_args --sentence_labels_path ${sentence_labels_path}"
    tw_extra_args="$tw_extra_args --valid_sentence_labels_path ${valid_sentence_labels_path:-}"
    tw_extra_args="$tw_extra_args --label_token_weight ${label_token_weight:-1.0}"
    tw_extra_args="$tw_extra_args --weight_eval_loss ${weight_eval_loss:-True}"
fi

torchrun --nproc_per_node="$nproc_per_node" --master-port="$master_port" \
    "../src/${script_name}.py" \
    --bf16 "$bf16" \
    --lang_encoder_path "$lang_encoder_path" \
    --tokenizer_path "$tokenizer_path" \
    --pretrained_visual_encoder "$pretrained_visual_encoder" \
    --pretrained_adapter "$pretrained_adapter" \
    --petct_image_folder "$petct_image_folder" \
    --petct_report_folder "$petct_report_folder" \
    --model_config_yaml "${model_config_yaml:-}" \
    --vis_dim "${vis_dim:-768}" \
    --num_visual_tokens "${num_visual_tokens:-128}" \
    --train_json_path "${train_json_path:-}" \
    --train_rve_manifest_path "${train_rve_manifest_path:-}" \
    --output_dir "$output_dir" \
    --z_chunk_size "$z_chunk_size" \
    --z_chunk_stride "$z_chunk_stride" \
    --xy_size "$xy_size" \
    --max_seq "$max_seq" \
    --pet_clip_percentile "${pet_clip_percentile:-99.5}" \
    --pet_apply_log "${pet_apply_log:-True}" \
    --use_fast "${use_fast:-False}" \
    --use_cls_template "${use_cls_template:-False}" \
    --use_ret_template "${use_ret_template:-False}" \
    --cls_template_path "${cls_template_path:-}" \
    --ret_template_path "${ret_template_path:-}" \
    --cls_dropout_prob "${cls_dropout_prob:-0.30}" \
    --ret_dropout_prob "${ret_dropout_prob:-0.30}" \
    --template_full_drop_prob "${template_full_drop_prob:-0.05}" \
    --train_subsample_fraction "${train_subsample_fraction:-1.0}" \
    --valid_subsample_fraction "${valid_subsample_fraction:-1.0}" \
    --subsample_seed "${subsample_seed:-42}" \
    --enable_thinking "${enable_thinking:-False}" \
    --enable_augmentation "${enable_augmentation:-False}" \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --num_train_epochs "$num_train_epochs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --eval_strategy "$evaluation_strategy" \
    --save_strategy "$save_strategy" \
    --save_total_limit "$save_total_limit" \
    --learning_rate "$learning_rate" \
    --weight_decay "$weight_decay" \
    --warmup_steps "$warmup_steps" \
    --lr_scheduler_type "$lr_scheduler_type" \
    --dataloader_num_workers "$dataloader_num_workers" \
    --run_name "$experiment_name" \
    --logging_steps "$logging_steps" \
    $ds_extra_args \
    $model_extra_args \
    $eval_extra_args \
    $tw_extra_args \
    $resume_extra_args
