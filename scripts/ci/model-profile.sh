#!/usr/bin/env bash
set -euo pipefail
cd /opt/tt-metal
output=/experiment/results/model-profile
mkdir -p "$output"
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
unset TT_METAL_PROFILER_MID_RUN_DUMP
timeout -k 30 1200 python3 -m tracy -p -r --disable-device-data-dump-to-files --op-support-count 10000 -o "$output" \
    /experiment-scripts/ci/model-profile.py 2>&1 | tee "$output/console.log"
python3 /experiment-scripts/ci/check-model-profile.py "$output"
