#!/bin/bash
#SBATCH --job-name=reportgen_ens_replay_gb
#SBATCH --partition=norm
#SBATCH --cpus-per-task=32
#SBATCH --mem=100g
#SBATCH --time=6:00:00
#SBATCH --output=${LOG_DIR:-./logs}/reportgen_ens_replay_gb-%j.out

# GBF5I0_v gate: does the CONTAINER's ensemble code reproduce the scored CSV
# exactly? Draw order is passed explicitly because cluster_block() keeps the
# first draw's sentence as each cluster's representative -- reordering changes
# the prose that ships, not just tie-breaks.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
cd /vf/users/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_pillar_ctrg3d_docker
D="grpo_newencB/700 grpo_newencB/800 grpo_B_a2from700/1000_t10p95 grpo_newencB/900_t10p95 grpo_newencB/800_rep100_t10p95 grpo_B_g3/800 grpo_B_a2from700/1000 grpo_newencB/1100_rep100 grpo_B_nobleu/700 grpo_B_g3/700"
/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D/.venv/bin/python verify/ensemble_replay.py \
    --labels scorer --draws "$D" --backbone 6 \
    --reference GBF5I0_v --merged_ref GBF5I0 \
    --limit "${LIMIT:-0}"
