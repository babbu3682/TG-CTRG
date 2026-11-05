#!/bin/bash

# Training script
# Multi-GPU training using uv and torchrun
#
# Usage:
#   bash scripts/run.sh                    # Run with default config
#   bash scripts/run.sh custom_config.yaml # Run with custom config

# Configuration
# Usage: bash scripts/run_ctrate_retrieval.sh [config_file] [exp_name] [notes]
#   exp_name also determines the checkpoints/log directory
#   (logs/CTRATE/<exp_name>/) — always pass a distinct one when running a
#   variant you want to compare side-by-side against another run (e.g. a
#   different text embedding pooling method), otherwise it silently shares
#   (and overwrites) the same checkpoint directory as any other run using
#   the default exp_name below.
CONFIG_FILE="${1:-configs/ctrate_retrieval_finetune_lz4_pillar_opensource.yaml}"
EXP_NAME="${2:-CTRATE_FT_Atlas_retrieval_pillar_freeze}"
NOTES="${3:-}"
TAGS='["apply-pooling", "Retrieval", "rve"]'

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
MASTER_PORT=12345
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



###### Single-GPU training with uv and torchrun

# NUM_GPUS=1
# MASTER_PORT=12345
# OMP_NUM_THREADS=2

# CUDA_VISIBLE_DEVICES=0 \
# OMP_NUM_THREADS=$OMP_NUM_THREADS \
# uv run torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=$MASTER_PORT \
#     scripts/train.py \
#     $CONFIG_FILE \
#     --opts \
#     experiment.name "$EXP_NAME" \
#     experiment.tags "$TAGS" \
#     experiment.notes "$NOTES" \
#     engine.max_epochs 50
