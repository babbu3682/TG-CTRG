#!/bin/bash
#SBATCH --job-name=reportgen_run100_e2e
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=8:00:00
#SBATCH --output=${LOG_DIR:-./logs}/reportgen_run100_e2e-%j.out

set -euo pipefail
nvidia-smi -L
bash /vf/users/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_pillar_ctrg3d_docker/verify/run100_e2e.sh
