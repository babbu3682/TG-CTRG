#!/usr/bin/env bash
# Submission container entrypoint.
# - Forwards SIGTERM to predict.py so its handler can flush state.
# - On exit, makes sure /output/predictions.json exists (empty schema if not).

set -uo pipefail

INPUT_DIR="${INPUT_DIR:-/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/checkpoint}"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

echo "[entrypoint] Pillar-0 + CTRG-3D reportgen container starting"
echo "[entrypoint]   INPUT_DIR         = $INPUT_DIR"
echo "[entrypoint]   OUTPUT_DIR        = $OUTPUT_DIR"
echo "[entrypoint]   CHECKPOINT_DIR    = $CHECKPOINT_DIR"
echo "[entrypoint]   WEIGHTS_DIR       = ${WEIGHTS_DIR:-/weights}"
echo "[entrypoint]   NUM_VISUAL_TOKENS = ${NUM_VISUAL_TOKENS:-128}"

python /opt/app/predict.py &
PID=$!

term_handler() {
    echo "[entrypoint] SIGTERM received, forwarding to predict.py (pid=$PID)"
    kill -SIGTERM "$PID" 2>/dev/null || true
    # Wait up to 25s for predict to flush; the platform gives 30s.
    for _ in $(seq 1 25); do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 1
    done
}
trap term_handler SIGTERM

wait "$PID"
RC=$?

if [ ! -f "$OUTPUT_DIR/predictions.json" ]; then
    echo "[entrypoint] predictions.json missing — writing empty schema"
    cat > "$OUTPUT_DIR/predictions.json" <<EOF
[{"outputs": [{"value": {"name": "Generated reports", "type": "Report generation", "version": "1.0", "generated_reports": []}}]}]
EOF
fi

echo "[entrypoint] exit $RC"
exit "$RC"
