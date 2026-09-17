#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/experiment-scripts/ci:/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
test "$(git -C /opt/tt-metal rev-parse HEAD)" = 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
timeout -k 5 20 python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
timeout -k 10 180 python3 /experiment-scripts/ci/history-append-probe.py \
    --hardware --output /experiment/results/history-append-hardware.json
