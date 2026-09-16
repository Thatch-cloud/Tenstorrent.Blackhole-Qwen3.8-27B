#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export PYTHONPATH=/experiment-scripts/ci:/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
test "$(git -C /opt/tt-metal rev-parse HEAD)" = 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
test -z "${TT_METAL_SIMULATOR:-}"
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/frozen_wait_zone_hardware.py verify-sources --directory /experiment-scripts/ci
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
export TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
unset TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES
cd /opt/tt-metal
preserve_profiler() {
    python3 - <<'PY'
from pathlib import Path
import shutil
from tracy.common import PROFILER_LOGS_DIR
source = Path(PROFILER_LOGS_DIR)
if source.is_dir():
    shutil.copytree(source, '/experiment/results/raw-profiler', dirs_exist_ok=True)
PY
}
trap preserve_profiler EXIT
status=0
timeout -k 10 210 python3 -u /experiment-scripts/ci/fused-batch-probe.py \
    --fixture /fixture-mlp --hardware --device-weight-check --trace-replay --trace-t16 --target-math \
    --output /experiment/results/fused-batch.json 2>&1 | tee /experiment/results/probe.log || status=$?
printf '%s\n' "$status" > /experiment/results/fused-batch.exit-status
preserve_profiler
test "$status" = 0
python3 /experiment-scripts/ci/frozen_wait_zone_hardware.py validate --directory /experiment-scripts/ci \
    --evidence /experiment/results/simulator --output /experiment/results
