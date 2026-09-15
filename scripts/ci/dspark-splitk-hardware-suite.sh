#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/experiment-scripts/ci:/speculative-decoding/harness:/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
timeout -k 5 20 python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
timeout -k 10 360 python3 /experiment-scripts/ci/dspark_splitk_hardware_build.py
timeout -k 10 120 python3 /experiment-scripts/ci/dspark-splitk-hardware-probe.py \
    --hardware --output /experiment/results/dspark-splitk-hardware.json
