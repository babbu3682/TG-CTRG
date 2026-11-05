#!/bin/bash
#SBATCH --job-name=reportgen_ens_replay
#SBATCH --partition=norm
#SBATCH --cpus-per-task=32
#SBATCH --mem=100g
#SBATCH --time=8:00:00
#SBATCH --output=${LOG_DIR:-./logs}/reportgen_ens_replay-%j.out

# CPU only: the ten draws already exist, so this replays stages C/D/E and diffs
# the result against bb1000F5I0_v.csv. norm has no GPU queue wait.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
cd /vf/users/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_pillar_ctrg3d_docker
/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D/.venv/bin/python verify/ensemble_replay.py --labels "${LABELS:-scorer}"
