#!/bin/bash
#SBATCH --job-name=stageA_regress
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=1:00:00
#SBATCH --output=${LOG_DIR:-./logs}/stageA_regress-%j.out
set -euo pipefail
DOCKER=/vf/users/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_pillar_ctrg3d_docker
PY=/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D/.venv/bin/python
RUN=/lscratch/$SLURM_JOB_ID/regress
mkdir -p "$RUN/checkpoint" "$RUN/output"
cd "$DOCKER"
PYTHONPATH="$DOCKER/src:$DOCKER/src/vendor:$DOCKER/src/vendor/petrg:$DOCKER/src/vendor/rad-vision-engine" \
INPUT_DIR="$DOCKER/verify/run100/input" OUTPUT_DIR="$RUN/output" \
CHECKPOINT_DIR="$RUN/checkpoint" WEIGHTS_DIR="$DOCKER/weights" \
STAGE_DIR=/lscratch/$SLURM_JOB_ID/stage_in \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" verify/regress_stage_a.py
