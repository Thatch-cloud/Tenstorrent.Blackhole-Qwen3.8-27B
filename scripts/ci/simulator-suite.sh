#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
test "${QWEN_SIM_ONLY:-0}" = 1
test ! -e /dev/tenstorrent
unset QWEN_HARDWARE_TESTS QWEN_CARDS_ALLOCATED TT_METAL_SLOW_DISPATCH_MODE TT_MESH_GRAPH_DESC_PATH
mkdir -p /tmp/ttsim /experiment/results
cp /simulator-assets/libttsim_bh_x2.so /tmp/ttsim/
cp /simulator-assets/cluster_descriptor.yaml /tmp/ttsim/
cp /opt/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml /tmp/ttsim/soc_descriptor.yaml
export TT_METAL_HOME=/opt/tt-metal
export PYTHONPATH=/opt/tt-metal:/opt/tt-metal/ttnn:/experiment-scripts/ci
export TT_METAL_SIMULATOR=/tmp/ttsim/libttsim_bh_x2.so
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/tmp/ttsim/cluster_descriptor.yaml
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_CACHE=/tmp/ttsim-kernel-cache
export QWEN_SIM_SHARED_BDF=1
export MESH_DEVICE=P300
git -C /opt/tt-metal rev-parse HEAD > /experiment/results/simulator-runtime.txt
test "$(cat /experiment/results/simulator-runtime.txt)" = 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
cd /opt/tt-metal
if [ "${QWEN_SIM_CASE:-stack}" = t32-commit ]; then
    ln -s /experiment-scripts /scripts
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export QWEN_SIM_REPORT=/experiment/results/t32-commit.json
    python3 -B -m unittest test_gdn_device_loop_state
    status=0
    timeout -k 15 9000 python3 -u /optimisation/sim/gdn-multitoken.py \
        --source-root /opt/tt-metal --rows 32 --norm-gate --conv --batch-conv \
        --dma-windows --packed-checkpoints --model-adapter --continuation \
        --compact-prologue --norm-batch-layer --defer-conv-publication --commit-only-gdn || status=$?
    printf '%s\n' "$status" > /experiment/results/t32-commit.exit-status
    exit "$status"
fi
if [[ "${QWEN_SIM_CASE:-stack}" = t32-markov* ]]; then
    weight_flags=()
    if [ "$QWEN_SIM_CASE" = t32-markov-learned ]; then
        weight_flags=(--checkpoint /dspark-model.safetensors)
    fi
    python3 -B -m unittest test_dspark_t32_weights test_dspark_t32_reference
    status=0
    timeout -k 15 9000 python3 -u /experiment-scripts/ci/dspark-t32-markov-probe.py \
        --native-reference "${weight_flags[@]}" --output /experiment/results/t32-markov.json || status=$?
    printf '%s\n' "$status" > /experiment/results/t32-markov.exit-status
    exit "$status"
fi
if [[ "${QWEN_SIM_CASE:-stack}" = fusion-t16* || "${QWEN_SIM_CASE:-stack}" = t32-*attention || "${QWEN_SIM_CASE:-stack}" = t32-combined || "${QWEN_SIM_CASE:-stack}" = t32-publication || "${QWEN_SIM_CASE:-stack}" = t32-request ]]; then
    mkdir -p /optimisation
    ln -s /simulator-support /optimisation/sim
    if [ "$QWEN_SIM_CASE" = t32-draft-attention ]; then
        python3 -B -m unittest test_t32_ci_runtime test_dspark_t32_draft_probe test_dspark_t32_attention
    fi
    python3 - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('compatibility', '/simulator-support/run-native-fixed-attention.py')
compatibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compatibility)
packer = Path('/opt/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h')
patch = Path('/simulator-support/blackhole-packer-zero-flags.patch').read_bytes().replace(b'\r\n', b'\n')
packer.write_bytes(compatibility.patched_bytes(packer.read_bytes(), patch))
PY
    export QWEN_SIM_PACKER_ZERO_GRAFT=1
    if [ "$QWEN_SIM_CASE" = t32-request ]; then
        export PYTHONPATH="/speculative-decoding/harness:$PYTHONPATH"
        python3 -B -m unittest test_t32_request_summary test_full_dspark_request test_t32_request_cli
        status=0
        QWEN_T32_SFPU_SUM=1 timeout -k 15 9000 python3 -u /experiment-scripts/ci/t32-full-request-probe.py \
            --checkpoint /dspark-model.safetensors --config /dspark-config.json \
            --target /target/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
            --output /experiment/results/t32-request.json || status=$?
        printf '%s\n' "$status" > /experiment/results/t32-request.exit-status
        exit "$status"
    fi
    if [[ "$QWEN_SIM_CASE" = t32-combined || "$QWEN_SIM_CASE" = t32-publication ]]; then
        python3 -B -m unittest test_dspark_t32_prepared test_t32_sim_target test_t32_target_weights test_t32_combined_upload test_dspark_publication_trace
        publication_args=()
        if [ "$QWEN_SIM_CASE" = t32-publication ]; then publication_args+=(--publication-only); fi
        status=0
        QWEN_T32_SFPU_SUM=1 timeout -k 15 9000 python3 -u /experiment-scripts/ci/t32-combined-proposal-probe.py \
            --checkpoint /dspark-model.safetensors --config /dspark-config.json \
            --target /target/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
            "${publication_args[@]}" --output "/experiment/results/$QWEN_SIM_CASE.json" || status=$?
        printf '%s\n' "$status" > "/experiment/results/$QWEN_SIM_CASE.exit-status"
        exit "$status"
    fi
    if [ "$QWEN_SIM_CASE" = t32-draft-attention ]; then
        status=0
        QWEN_T32_SFPU_SUM=1 timeout -k 15 9000 python3 -u /experiment-scripts/ci/dspark-t32-draft-attention-probe.py \
            --key-chunk-size 64 --output /experiment/results/t32-draft-attention.json || status=$?
        printf '%s\n' "$status" > /experiment/results/t32-draft-attention.exit-status
        exit "$status"
    fi
    if [ "$QWEN_SIM_CASE" = t32-context-attention ]; then
        python3 -B -m unittest test_target_t16_context_cli
        status=0
        timeout -k 15 9000 python3 -u /experiment-scripts/ci/target-t16-context-probe.py \
            --context "${QWEN_SIM_CONTEXT:-2048}" --output /experiment/results/t16-context-attention.json || status=$?
        printf '%s\n' "$status" > /experiment/results/t16-context-attention.exit-status
        exit "$status"
    fi
    if [ "$QWEN_SIM_CASE" = t32-attention ]; then
        python3 -B -m unittest test_target_t32_attention_gate test_target_t16_attention_gate
        status=0
        timeout -k 15 9000 python3 -u /experiment-scripts/ci/target-t32-attention-probe.py \
            --output /experiment/results/t32-attention.json || status=$?
        printf '%s\n' "$status" > /experiment/results/t32-attention.exit-status
        exit "$status"
    fi
    math_flags=()
    if [ "$QWEN_SIM_CASE" = fusion-t16-target ]; then math_flags=(--target-math); fi
    status=0
    timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py \
        --fixture /fixture-mlp --device-weight-check --trace-replay --trace-t16 "${math_flags[@]}" \
        --output /experiment/results/fused-batch.json || status=$?
    printf '%s\n' "$status" > /experiment/results/fused-batch.exit-status
    exit "$status"
fi
if [ "${QWEN_CCL_LAZY_BUILD:-0}" = 1 ]; then
    bash /experiment-scripts/ci/ccl-links-build.sh
    export QWEN_PROJECTION_LINKS=1
    timeout -k 15 3200 python3 -u /experiment-scripts/ci/ccl-link-probe.py \
        --output /experiment/results/ccl-link-simulator.json 2>&1 | tee /experiment/results/ccl-link-simulator.log
    if grep -q 'Failed to discover available ethernet links' /experiment/results/ccl-link-simulator.log; then
        echo 'Explicit-link collective still invoked fallback discovery' >&2
        exit 1
    fi
    exit 0
fi
if [ "${QWEN_SIM_CASE:-stack}" = shortlist ]; then
    for width in 32768 65536; do
        timeout -k 15 3200 python3 -u /experiment-scripts/ci/draft-shortlist-probe.py \
            --width "$width" --output "/experiment/results/draft-shortlist-$width.json"
    done
    exit 0
fi
test "${QWEN_SIM_CASE:-stack}" = stack
timeout -k 15 6600 python3 -u /experiment-scripts/ci/learned-attention-probe.py \
    --fixture /fixture-attention --convolution-fixture /fixture-convolution --mlp-fixture /fixture-mlp \
    --stack-fixtures /fixture-stack --stack-layers 5 --selector-fixture /fixture-selector \
    --fp32-rope --explicit-softmax --fused-row-sum --fused-dots --cache-dot-tiles --captured-stack \
    --output /experiment/results/learned-attention-simulator.json
