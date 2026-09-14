#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${QWEN_HARDWARE_TESTS:-0}" = 1
test "${QWEN_LADDER_BACKEND:-}" = hardware
test "${QWEN_LADDER_CONTEXT:-}" = 65536
test -z "${TT_METAL_SIMULATOR:-}"
test -z "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}"
export PYTHONPATH=/experiment-scripts/ci:/opt/tt-metal/ttnn:/opt/tt-metal
printf 'hardware result write preflight\n' > /experiment/results/ladder-write-preflight.txt
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
python3 /experiment-scripts/ci/dspark_native_restore.py
ln -s /experiment-optimisation /optimisation
export QWEN_DRAFT_FP32_CONTROL=0
timeout -k 30 1900 python3 -u /experiment-scripts/ci/dspark_ladder_build.py
export QWEN_DRAFT_FP32_INTERMEDIATES=1
export TT_METAL_DPRINT_CORES='(0,0)'
export TT_METAL_DPRINT_RISCVS=TR0
export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1
status=0
started=$SECONDS
timeout -k 15 300 python3 -u /experiment-scripts/ci/dspark-ladder-attention-probe.py \
    --hardware --output /experiment/results/dspark-ladder-hardware-65536.json || status=$?
printf '%s\n' "$status" > /experiment/results/dspark-ladder-hardware-65536.exit-status
printf '%s\n' "$((SECONDS - started))" > /experiment/results/dspark-ladder-hardware-65536.elapsed-seconds
exit "$status"
