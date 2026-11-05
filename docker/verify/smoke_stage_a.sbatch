#!/bin/bash
#SBATCH --job-name=stageA_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:50
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=0:40:00
#SBATCH --output=${LOG_DIR:-./logs}/stageA_smoke-%j.out
set -euo pipefail
DOCKER=/vf/users/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_pillar_ctrg3d_docker
PY=/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D/.venv/bin/python
RUN=/lscratch/$SLURM_JOB_ID/smoke
MAPPING=/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/valid/mapping_path.csv
mkdir -p "$RUN/input" "$RUN/output" "$RUN/checkpoint"
"$PY" - "$MAPPING" "$RUN/input" <<'PYEOF'
import csv, os, sys
src = {r["sample_name"]: r["source_path"] for r in csv.DictReader(open(sys.argv[1]))}
for n in sorted(src)[:5]:
    os.symlink(src[n], os.path.join(sys.argv[2], f"{n}.nii.gz"))
print("linked 5")
PYEOF
cd "$DOCKER"
# N_DRAWS=0 is rejected (BACKBONE_IDX check), so run stage A only by letting it
# fail at stage B: we just need the [host] block and the per-volume profile.
PYTHONPATH="$DOCKER/src:$DOCKER/src/vendor:$DOCKER/src/vendor/petrg:$DOCKER/src/vendor/rad-vision-engine" \
INPUT_DIR="$RUN/input" OUTPUT_DIR="$RUN/output" CHECKPOINT_DIR="$RUN/checkpoint" \
WEIGHTS_DIR="$DOCKER/weights" STAGE_DIR=/lscratch/$SLURM_JOB_ID/stage_in \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
STAGE_A_ONLY=1 \
    timeout 900 "$PY" predict.py || echo "(stopped after stage A / stage B start — expected)"
