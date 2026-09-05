#!/usr/bin/env bash
set -euo pipefail
export QWEN_RUN_MODE=interleave
for ratio in 0 1 2 4; do
    export QWEN_INTERLEAVE_RATIO="$ratio"
    timeout 1600 bash scripts/ci/run-baseline.sh
    if [ "$ratio" != 0 ]; then
        python3 scripts/ci/compare-interleave.py experiment-results/interleave-0 "experiment-results/interleave-$ratio"
    fi
done
