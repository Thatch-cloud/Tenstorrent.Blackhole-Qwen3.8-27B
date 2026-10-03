#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
export QWEN_RUN_MODE=full-norm-engine
export QWEN_CODING_REQUEST=1
export QWEN_FABRIC_LINK_PROBE=1
export QWEN_LOOKUP_CAP_ABBA=0
export QWEN_PREFIX_ZERO_REUSE=0
export QWEN_LEARNED_STACK=0
export QWEN_SIM_ONLY=0
exec bash scripts/ci/run-baseline.sh
