#!/bin/bash
#SBATCH --job-name=stageA_profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=1:30:00
#SBATCH --output=${LOG_DIR:-./logs}/stageA_profile-%j.out

# Why: the submitted container ran stage A at ~40 s/vol on gpu-a100-40 while the
# local 100-exam end-to-end run implies ~2 s/vol. Two variants below isolate
# whether the gap is CPU starvation (the platform container's core count is
# unknown; this job has 16) or something intrinsic to the GPU path.
set -euo pipefail
nvidia-smi -L

DOCKER=/vf/users/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_pillar_ctrg3d_docker
PY=/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D/.venv/bin/python
export PYTHONPATH="$DOCKER/src:$DOCKER/src/vendor:$DOCKER/src/vendor/petrg:$DOCKER/src/vendor/rad-vision-engine"
export WEIGHTS_DIR="$DOCKER/weights"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$DOCKER"

N=${N:-6}

echo "############ variant 1: all 16 cores ############"
PROFILE_WORK=/lscratch/$SLURM_JOB_ID/p16 \
    "$PY" verify/profile_stage_a.py "$N" cpu16

echo
echo "############ variant 2: pinned to 4 cores ############"
PROFILE_WORK=/lscratch/$SLURM_JOB_ID/p4 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    taskset -c 0-3 "$PY" verify/profile_stage_a.py "$N" cpu4

echo
echo "############ variant 3: 16 cores, /checkpoint on NFS ############"
PROFILE_WORK=$DOCKER/verify/_profile_nfs \
    "$PY" verify/profile_stage_a.py "$N" nfs16
rm -rf "$DOCKER/verify/_profile_nfs"
