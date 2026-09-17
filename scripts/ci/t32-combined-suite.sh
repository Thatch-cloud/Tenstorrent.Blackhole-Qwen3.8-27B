#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/experiment-scripts/ci:/speculative-decoding/harness:/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /experiment/results
test "${QWEN_T32_FUSED_SCORE_HARDWARE:-0}" = 1
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
python3 /experiment-scripts/ci/dspark_native_restore.py
ln -s /experiment-optimisation /optimisation
python3 /experiment-scripts/ci/dspark_runtime_cache.py
arguments=(--checkpoint /dspark/model.safetensors --config /dspark/config.json --target "$MODEL_WEIGHTS_DIR"
    --proposal-evidence /evidence/proposal/t32-combined.json
    --score-evidence /evidence/score/t32-markov.json
    --attention-evidence /evidence/attention/t32-attention.json --commit-evidence /evidence/commit)
python3 -u /experiment-scripts/ci/t32-combined-hardware.py "${arguments[@]}" \
    --preflight --output /experiment/results/t32-preflight.json
set +e
timeout -k 20 600 python3 -u /experiment-scripts/ci/t32-combined-hardware.py "${arguments[@]}" \
    --output /experiment/results/t32-combined-hardware.json 2>&1 | tee /experiment/results/t32-combined-hardware.log
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > /experiment/results/t32-combined-hardware.exit-status
test "$status" = 0
if grep -q 'Failed to discover available ethernet links' /experiment/results/t32-combined-hardware.log; then
    echo 'Four-link configuration unexpectedly reached fallback discovery' >&2
    exit 1
fi
