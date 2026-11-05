#!/bin/bash
#SBATCH --job-name=reportgen_stageA_parity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=1:00:00
#SBATCH --output=${LOG_DIR:-./logs}/reportgen_stageA_parity-%j.out

set -euo pipefail
nvidia-smi -L
cd /vf/users/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_pillar_ctrg3d_docker
N_EXAMS=${N_EXAMS:-32} /vf/users/drdcad/sunggu/0.Challenge/PETRG-3D/.venv/bin/python verify/stage_a_parity.py
