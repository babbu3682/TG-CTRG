#!/usr/bin/env bash
# Build the Pillar-0 + CTRG-3D reportgen submission image.
set -euo pipefail
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

if [ ! -d "$SCRIPTPATH/src/vendor/petrg" ]; then
    echo "FATAL: src/vendor/petrg is missing (it is part of the repository)" >&2; exit 1
fi

# Snapshot the build host's CA bundle (includes any TLS-inspecting proxy CA)
# so in-container pip can verify certificates normally. Empty file = disabled.
cp /etc/ssl/certs/ca-certificates.crt "$SCRIPTPATH/host-ca-bundle.crt" 2>/dev/null \
    || : > "$SCRIPTPATH/host-ca-bundle.crt"

docker build -t reportgen-pillar-ctrg3d \
    --build-arg NUM_VISUAL_TOKENS="${NUM_VISUAL_TOKENS:-128}" \
    "$SCRIPTPATH"
