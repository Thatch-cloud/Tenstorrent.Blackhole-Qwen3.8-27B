#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${QWEN_HARDWARE_TESTS:-0}" = 1
test "${QWEN_LADDER_BACKEND:-}" = hardware
test "${QWEN_LADDER_CONTEXT:-}" = 65536
test "${QWEN_DSPARK_REQUEST_CONTEXT:-}" = 65536
test -z "${TT_METAL_SIMULATOR:-}"
test -z "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}"
test -z "${TT_METAL_SLOW_DISPATCH_MODE:-}"
export PYTHONPATH=/experiment-scripts/ci:/opt/tt-metal/ttnn:/opt/tt-metal
printf 'frozen hardware result write preflight\n' > /experiment/results/frozen-hardware-write-preflight.txt
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
packer=/opt/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h
sha256sum "$packer" > /experiment/results/hardware-packer.sha256
printf '%s  %s\n' 87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181 "$packer" | sha256sum -c -
python3 /experiment-scripts/ci/dspark_native_restore.py
export QWEN_DRAFT_FP32_CONTROL=0
build_status=0
build_started=$SECONDS
timeout -k 30 2100 python3 -u /experiment-scripts/ci/frozen_hardware_build.py || build_status=$?
printf '%s\n' "$build_status" > /experiment/results/frozen-hardware-build.exit-status
printf '%s\n' "$((SECONDS - build_started))" > /experiment/results/frozen-hardware-build.elapsed-seconds
test "$build_status" = 0
export QWEN_DRAFT_FP32_INTERMEDIATES=1
probe_status=0
probe_started=$SECONDS
timeout -k 15 600 python3 -u /experiment-scripts/ci/dspark-native-8k-attention-probe.py \
    --hardware --output /experiment/results/dspark-native-8k-attention-hardware-65536.json || probe_status=$?
printf '%s\n' "$probe_status" > /experiment/results/dspark-native-8k-attention-hardware-65536.exit-status
printf '%s\n' "$((SECONDS - probe_started))" > /experiment/results/dspark-native-8k-attention-hardware-65536.elapsed-seconds
exit "$probe_status"
