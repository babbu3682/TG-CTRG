#!/usr/bin/env bash
# Local smoke test: mounts ./test as /input and ./weights as /weights, runs
# the container, and pretty-prints /output/predictions.json.
#
# Populate ./weights first (see make_weights.sh) and drop at least one
# *.nii.gz under ./test/.
#
# GPU SELECTION -- read this before blaming the container for an OOM.
# Stage B moves an 8B bf16 model onto the GPU and needs ~16.5 GB free BEFORE
# generation starts, then peaks around 30 GB at GEN_BATCH=24. storm's A100s are
# 40 GB and usually already host a draw-generation container, so `--gpus all`
# lands us on an occupied card and dies in `lm.to(device)` with
#     "Process <pid> has 29.41 GiB memory in use"
# which is someone else's memory, not a leak here. Pick a free card:
#
#     nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
#     GPU=2 ./test.sh          # or GPU=all to keep the old behaviour
#
# The submission itself is unaffected -- the platform gives the container a
# dedicated GPU.
set -euo pipefail
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
./build.sh

# GPU=auto picks the emptiest card, which is what you want on a box whose other
# cards are hosting draw-generation containers. GPU=<index> pins one; GPU=all
# keeps docker's default (and probes card 0).
GPU="${GPU:-auto}"
NEED_MIB="${NEED_MIB:-20000}"

if command -v nvidia-smi >/dev/null 2>&1; then
    if [ "$GPU" = auto ]; then
        GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
              | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')
        echo "[test] GPU=auto selected card ${GPU}"
    fi
    if [ "$GPU" = all ]; then PROBE=0; else PROBE="$GPU"; fi
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$PROBE" 2>/dev/null | head -1 | tr -d ' ' || echo "")
    if [ -n "$FREE" ] && [ "$FREE" -lt "$NEED_MIB" ]; then
        echo "ERROR: GPU $PROBE has only ${FREE} MiB free; stage B needs ~16.5 GB to" >&2
        echo "       load the model and ~30 GB at GEN_BATCH=24." >&2
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >&2
        echo "       Free a card (a draw-generation container is probably holding it)," >&2
        echo "       then:  GPU=auto ./test.sh" >&2
        exit 1
    fi
    echo "[test] using GPU ${GPU} (${FREE:-?} MiB free)"
fi

if [ "$GPU" = all ]; then
    GPU_ARG="--gpus all"
else
    GPU_ARG="--gpus device=${GPU}"
fi

MEM_LIMIT="${MEM_LIMIT:-48g}"
# Recreate the output volume from scratch: a volume left over from a failed
# run may be root-owned (pre-fix image) and predictions.json writes would fail.
docker volume rm -f reportgen-pillar-output >/dev/null 2>&1 || true
docker volume create reportgen-pillar-output
# Clean the volume up even if the run below fails, so re-runs start fresh.
trap 'docker volume rm -f reportgen-pillar-output >/dev/null 2>&1 || true' EXIT

docker run --rm \
        ${GPU_ARG} \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --shm-size="512m" \
        -v "$SCRIPTPATH/test/:/input/:ro" \
        -v "$SCRIPTPATH/weights/:/weights/:ro" \
        -v reportgen-pillar-output:/output/ \
        reportgen-pillar-ctrg3d

# Pretty-print inside the container — the host may not have a `python` binary.
docker run --rm -v reportgen-pillar-output:/output/:ro \
        python:3.10-slim python -m json.tool /output/predictions.json
