#!/usr/bin/env bash
set -euo pipefail
mkdir -p /experiment/results
test "${QWEN_HARDWARE_TESTS:-0}" = 1
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${QWEN_PROJECTION_LINKS:-0}" = 4
test -z "${TT_METAL_SIMULATOR:-}"
test -z "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}"
test -z "${TT_METAL_SLOW_DISPATCH_MODE:-}"
export PYTHONPATH=/experiment-scripts/ci:/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
build_started=$SECONDS
bash /experiment-scripts/ci/ccl-links-build.sh
printf '{"build_seconds":%s,"scope":"isolated runtime build; excluded from kernel timing"}\n' "$((SECONDS-build_started))" \
    > /experiment/results/dspark-build-time.json
set +e
timeout -k 20 3000 python3 -u /experiment-scripts/ci/dspark-pipeline-hardware.py \
    --checkpoint /dspark/model.safetensors --config /dspark/config.json \
    --output /experiment/results/dspark-pipeline-hardware.json 2>&1 | tee /experiment/results/dspark-pipeline-hardware.log
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > /experiment/results/dspark-pipeline-hardware.exit-status
test "$status" = 0
if grep -q 'Failed to discover available ethernet links' /experiment/results/dspark-pipeline-hardware.log; then
    echo 'Explicit four-link integration unexpectedly invoked fallback discovery' >&2
    exit 1
fi
