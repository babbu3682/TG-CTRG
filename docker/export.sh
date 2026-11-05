#!/usr/bin/env bash
# Package the image for upload via `forithmus submit`.
set -euo pipefail
./build.sh
docker save reportgen-pillar-ctrg3d | gzip -c > reportgen-pillar-ctrg3d.tar.gz
echo "Wrote reportgen-pillar-ctrg3d.tar.gz — submit with:"
echo "  forithmus submit reportgen-pillar-ctrg3d.tar.gz --phase <phase> --tier gpu-l4-xl --weights weights.zip -t 1200"

