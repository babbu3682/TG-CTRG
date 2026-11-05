#!/bin/bash

# Training script for the BCE-loss abn classification run.
# Multi-GPU training using uv and torchrun.
#
# Usage:
#   bash scripts/run_ctrate_abnclass_bce.sh                    # Run with default config
#   bash scripts/run_ctrate_abnclass_bce.sh custom_config.yaml # Run with custom config

# Configuration
# Canonical abn task launcher. Default config: frozen backbone, activ feature ->
# MultiAttentionPool -> MLP[1024] head, MultiLabelWeightedBCELoss, lz4 cache,
# early stopping. Pass a different config as $1 to compare against, e.g. the
# plain-linear-probe-style config:
#   bash scripts/run_ctrate_abnclass_bce.sh configs/ctrate_abnclass_finetune_lz4_pillar_opensource.yaml
#
# exp_name ($2) also determines the checkpoints/log directory
# (logs/CTRATE/<exp_name>/) — always pass a distinct one (along with a
# distinct --job-name at sbatch submission time) when running a variant you
# want to compare side-by-side, otherwise it silently shares (and overwrites)
# the same checkpoint directory as any other run using the default below.
CONFIG_FILE="${1:-configs/ctrate_abnclass_finetune_lz4_pillar_opensource_attnpool.yaml}"
EXP_NAME="${2:-CTRATE_FT_Atlas_abn_bce_attnpool}"
NOTES="${3:-pillar-0 pretrained (frozen, eval-pinned), activ -> MultiAttentionPool -> MLP[1024], MultiLabelWeightedBCELoss, lz4 cache, early stopping}"
TAGS='["attention-pooling", "ABN", "rve", "bce", "pillar0", "lz4"]'

# Define Variables
# ⚠️ WANDB_API_KEY는 저장소에 두지 않는다 — origin이 공용 저장소(YalaLab/pillar-finetune)라
# 커밋하면 그대로 공개된다. /data/drdcad/sunggu/.wandb_key 에 export WANDB_API_KEY="..." 를
# 한 줄로 두고(chmod 600) 여기서 읽는다. 없으면 조용히 넘어가지 않고 즉시 죽는다.
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/data/drdcad/sunggu/.wandb_key}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$WANDB_KEY_FILE" ]; then . "$WANDB_KEY_FILE"; fi
: "${WANDB_API_KEY:?WANDB_API_KEY 없음 — $WANDB_KEY_FILE 을 만들거나 환경변수로 주세요}"
export WANDB_API_KEY

###### Multi-GPU training with uv and torchrun

NUM_GPUS=1
MASTER_PORT=56790
OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=$OMP_NUM_THREADS \
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    $CONFIG_FILE \
    --opts \
    experiment.name "$EXP_NAME" \
    experiment.tags "$TAGS" \
    experiment.notes "$NOTES" \
    engine.max_epochs 50
