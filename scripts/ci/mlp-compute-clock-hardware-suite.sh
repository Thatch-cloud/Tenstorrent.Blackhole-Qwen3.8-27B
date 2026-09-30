#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export PYTHONPATH=/experiment-scripts/ci:/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
test "$(git -C /opt/tt-metal rev-parse HEAD)" = 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
test -z "${TT_METAL_SIMULATOR:-}"
test -z "${TT_METAL_DEVICE_PROFILER:-}"
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/mlp_compute_clock_hardware.py verify --directory /experiment-scripts/ci
status=0
timeout -k 10 210 python3 -u /experiment-scripts/ci/fused-batch-probe.py \
    --fixture /fixture-mlp --hardware --device-weight-check --trace-replay --trace-t16 --target-math \
    --output /experiment/results/fused-batch.json 2>&1 | tee /experiment/results/probe.log || status=$?
printf '%s\n' "$status" > /experiment/results/fused-batch.exit-status
test "$status" = 0
python3 /experiment-scripts/ci/mlp_compute_clock_hardware.py validate --directory /experiment-scripts/ci \
    --evidence /experiment/results/simulator --output /experiment/results
