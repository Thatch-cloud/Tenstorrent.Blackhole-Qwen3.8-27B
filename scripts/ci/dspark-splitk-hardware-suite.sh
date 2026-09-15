#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/experiment-scripts/ci:/speculative-decoding/harness:/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
timeout -k 5 20 python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
build=dspark_splitk_hardware_build.py
probe=matched-context-attention-probe.py
if [ "${QWEN_SPLITK_MAXIMA:-0}" = 1 ]; then
    build=dspark_splitk_maxima_hardware.py
    probe=matched-context-maxima-probe.py
fi
timeout -k 10 360 python3 "/experiment-scripts/ci/$build"
if [ "${QWEN_MATCHED_TARGET:-0}" = 1 ]; then
    timeout -k 10 120 python3 /experiment-scripts/ci/matched-context-target-probe.py --hardware \
        --output "/experiment/results/matched-context-target-${QWEN_MATCHED_CONTEXT}.json"
    exit "$?"
fi
if [ "${QWEN_MATCHED_CONTEXT:-0}" != 0 ]; then
    timeout -k 10 120 python3 "/experiment-scripts/ci/$probe" \
        --output "/experiment/results/matched-context-attention-${QWEN_MATCHED_CONTEXT}.json"
    exit "$?"
fi
timeout -k 10 120 python3 /experiment-scripts/ci/dspark-splitk-hardware-probe.py \
    --hardware --output /experiment/results/dspark-splitk-hardware.json
