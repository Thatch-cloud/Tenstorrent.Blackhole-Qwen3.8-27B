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
if [[ "${QWEN_SIM_CASE:-stack}" = markov-cache-control ]]; then
    status=0
    timeout -k 15 600 python3 -u /experiment-scripts/ci/markov-cache-control-probe.py \
        --output /experiment/results/markov-cache-control.json || status=$?
    printf '%s\n' "$status" > /experiment/results/markov-cache-control.exit-status
    exit "$status"
fi
if [[ "${QWEN_SIM_CASE:-stack}" = fusion-t16* || "${QWEN_SIM_CASE:-stack}" = markov-sparse-dot || "${QWEN_SIM_CASE:-stack}" = gdn-output-* || "${QWEN_SIM_CASE:-stack}" = gdn-copy-pairs || "${QWEN_SIM_CASE:-stack}" = gdn-outer-add || "${QWEN_SIM_CASE:-stack}" = dspark-ladder-attention || "${QWEN_SIM_CASE:-stack}" = dspark-native-8k-attention || "${QWEN_SIM_CASE:-stack}" = target-t16-attention-8k || "${QWEN_SIM_CASE:-stack}" = gdn-shared-recurrence || "${QWEN_SIM_CASE:-stack}" = gdn-shared-qk ]]; then
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
    if [[ "${QWEN_SIM_CASE:-stack}" = markov-sparse-dot || "$QWEN_SIM_CASE" = gdn-output-* || "$QWEN_SIM_CASE" = gdn-copy-pairs || "$QWEN_SIM_CASE" = gdn-outer-add || "$QWEN_SIM_CASE" = dspark-ladder-attention || "$QWEN_SIM_CASE" = dspark-native-8k-attention || "$QWEN_SIM_CASE" = target-t16-attention-8k || "$QWEN_SIM_CASE" = gdn-shared-recurrence || "$QWEN_SIM_CASE" = gdn-shared-qk ]]; then
        [[ "${QWEN_SIM_CASE:-stack}" = markov-sparse-dot || "$QWEN_SIM_CASE" = gdn-output-l1 || "$QWEN_SIM_CASE" = gdn-output-grid || "$QWEN_SIM_CASE" = gdn-copy-pairs || "$QWEN_SIM_CASE" = gdn-outer-add || "$QWEN_SIM_CASE" = dspark-ladder-attention || "$QWEN_SIM_CASE" = dspark-native-8k-attention || "$QWEN_SIM_CASE" = target-t16-attention-8k || "$QWEN_SIM_CASE" = gdn-shared-recurrence || "$QWEN_SIM_CASE" = gdn-shared-qk ]]
        status=0
        limit=3600
        if [[ "$QWEN_SIM_CASE" = dspark-ladder-attention ]]; then
            export QWEN_SIM_BOUNDED_MEMORY=1
            mkdir -p /optimisation
            ln -s /simulator-support /optimisation/sim
            export QWEN_DRAFT_FP32_CONTROL=0
            build_started=$SECONDS
            printf 'ladder build started %s\n' "$(date -u +%FT%TZ)"
            if [ "${QWEN_SCORE_BITWISE:-0}" = 1 ]; then
                build_script=dspark_sim_build_cache.py
                if [ "${QWEN_SCORE_SFPU:-0}" = 1 ]; then build_script=dspark_score_sfpu_build.py; fi
                if [ "${QWEN_SUM_SFPU:-0}" = 1 ]; then build_script=dspark_sum_sfpu_build.py; fi
                if [ "${QWEN_SPLITK_ATTENTION:-0}" = 1 ]; then build_script=dspark_splitk_fp32_build.py; fi
                timeout -k 15 330 python3 -u "/experiment-scripts/ci/$build_script"
            else
                timeout -k 30 1900 python3 -u /experiment-scripts/ci/dspark_ladder_build.py
            fi
            printf 'ladder build completed elapsed_seconds=%s\n' "$((SECONDS - build_started))"
            export QWEN_DRAFT_FP32_INTERMEDIATES=1
            unset TT_METAL_DPRINT_CORES TT_METAL_DPRINT_RISCVS TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC TT_METAL_DPRINT_FILE
            export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=60000
            if [ "${QWEN_SCORE_BITWISE:-0}" = 1 ]; then
                score_name=dspark-score-bitwise
                if [ "${QWEN_SCORE_SFPU:-0}" = 1 ]; then
                    score_name=dspark-score-sfpu
                    export TT_METAL_DPRINT_CORES='(0,0)'
                    export TT_METAL_DPRINT_RISCVS=TR0
                    export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1
                fi
                if [ "${QWEN_SUM_SFPU:-0}" = 1 ]; then score_name=dspark-sum-sfpu; fi
                if [ "${QWEN_MASK_BITS:-0}" = 1 ]; then score_name=dspark-mask-bits; fi
                if [ "${QWEN_ATTENTION_BOUNDARY:-0}" = 1 ]; then score_name=dspark-attention-boundary; fi
                if [ "${QWEN_DIRECT_FP32_STAGE:-0}" = 1 ]; then score_name=dspark-direct-fp32-stage; fi
                if [ "${QWEN_NORMALIZATION_DIRECT_STAGE:-0}" = 1 ]; then score_name=dspark-normalization-direct-stage; fi
                if [ "${QWEN_CENTER_TILE_FILL:-0}" = 1 ]; then score_name=dspark-center-tile-fill; fi
                if [ "${QWEN_SPLITK_ATTENTION:-0}" = 1 ]; then score_name=dspark-splitk; fi
                QWEN_LADDER_CONTEXT=128 QWEN_LADDER_SCORE_SMOKE=1 timeout -k 15 165 python3 -u \
                    "/experiment-scripts/ci/$score_name-probe.py" \
                    --output "/experiment/results/$score_name.json"
                exit 0
            fi
            smoke_status=0
            smoke_started=$SECONDS
            export TT_METAL_DPRINT_CORES='(0,0)'
            export TT_METAL_DPRINT_RISCVS=TR0
            export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1
            printf 'ladder score smoke started %s\n' "$(date -u +%FT%TZ)"
            QWEN_LADDER_CONTEXT=128 QWEN_LADDER_SCORE_SMOKE=1 timeout -k 15 600 python3 -u \
                /experiment-scripts/ci/dspark-ladder-attention-probe.py \
                --output /experiment/results/dspark-ladder-score-smoke.json || smoke_status=$?
            printf '%s\n' "$smoke_status" > /experiment/results/dspark-ladder-score-smoke.exit-status
            printf '%s\n' "$((SECONDS - smoke_started))" > /experiment/results/dspark-ladder-score-smoke.elapsed-seconds
            printf 'ladder score smoke completed status=%s elapsed_seconds=%s\n' "$smoke_status" "$((SECONDS - smoke_started))"
            if [ "$smoke_status" != 0 ]; then exit "$smoke_status"; fi
            for context in 65536 32768 128 4096 8192; do
                unset TT_METAL_DPRINT_CORES TT_METAL_DPRINT_RISCVS TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC TT_METAL_DPRINT_FILE
                if [ "$context" = 65536 ]; then
                    export TT_METAL_DPRINT_CORES='(0,0)'
                    export TT_METAL_DPRINT_RISCVS=TR0
                    export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1
                fi
                context_status=0
                context_started=$SECONDS
                printf 'ladder context=%s started %s\n' "$context" "$(date -u +%FT%TZ)"
                QWEN_LADDER_CONTEXT="$context" timeout -k 15 1800 python3 -u \
                    /experiment-scripts/ci/dspark-ladder-attention-probe.py \
                    --output "/experiment/results/dspark-ladder-attention-$context.json" || context_status=$?
                printf '%s\n' "$context_status" > "/experiment/results/dspark-ladder-attention-$context.exit-status"
                printf '%s\n' "$((SECONDS - context_started))" > "/experiment/results/dspark-ladder-attention-$context.elapsed-seconds"
                printf 'ladder context=%s completed status=%s elapsed_seconds=%s\n' "$context" "$context_status" "$((SECONDS - context_started))"
                if [ "$context_status" != 0 ]; then exit "$context_status"; fi
            done
            exit 0
        fi
        if [[ "$QWEN_SIM_CASE" = markov-sparse-dot ]]; then
            timeout -k 30 1900 python3 -u /experiment-scripts/ci/markov_sparse_build.py
        fi
        if [[ "$QWEN_SIM_CASE" = dspark-native-8k-attention ]]; then
            export QWEN_SIM_BOUNDED_MEMORY=1
            mkdir -p /optimisation
            ln -s /simulator-support /optimisation/sim
            export QWEN_DRAFT_FP32_CONTROL=0
            timeout -k 30 1900 python3 -u /experiment-scripts/ci/dspark_fp32_build.py
            export QWEN_DRAFT_FP32_INTERMEDIATES=1
        fi
        if [[ "$QWEN_SIM_CASE" = markov-sparse-dot || "$QWEN_SIM_CASE" = gdn-copy-pairs || "$QWEN_SIM_CASE" = gdn-outer-add || "$QWEN_SIM_CASE" = dspark-ladder-attention || "$QWEN_SIM_CASE" = dspark-native-8k-attention || "$QWEN_SIM_CASE" = target-t16-attention-8k || "$QWEN_SIM_CASE" = gdn-shared-recurrence || "$QWEN_SIM_CASE" = gdn-shared-qk ]]; then limit=900; fi
        if [ "${QWEN_TARGET_T16_64K:-0}" = 1 ]; then
            test "$QWEN_SIM_CASE" = target-t16-attention-8k
            QWEN_SIM_CASE=target-t16-attention-64k
            limit=360
        fi
        timeout -k 15 "$limit" python3 -u "/experiment-scripts/ci/$QWEN_SIM_CASE-probe.py" \
            --output "/experiment/results/$QWEN_SIM_CASE.json" || status=$?
        printf '%s\n' "$status" > "/experiment/results/$QWEN_SIM_CASE.exit-status"
        if [[ "$QWEN_SIM_CASE" = markov-sparse-dot && "$status" = 0 ]]; then
            timeout -k 15 600 python3 -u /experiment-scripts/ci/markov-cache-pipeline-probe.py \
                --output /experiment/results/markov-cache-pipeline.json || status=$?
            printf '%s\n' "$status" > /experiment/results/markov-cache-pipeline.exit-status
            for width in 4992 3712; do
                if [[ "$status" != 0 ]]; then break; fi
                timeout -k 15 600 python3 -u /experiment-scripts/ci/markov-cache-pipeline-probe.py \
                    --width "$width" --output "/experiment/results/markov-cache-pipeline-$width.json" || status=$?
                printf '%s\n' "$status" > "/experiment/results/markov-cache-pipeline-$width.exit-status"
            done
            if [[ "$status" = 0 ]]; then
                timeout -k 15 900 python3 -u /experiment-scripts/ci/dspark-cached-markov-probe.py \
                    --output /experiment/results/dspark-cached-markov.json || status=$?
                printf '%s\n' "$status" > /experiment/results/dspark-cached-markov.exit-status
            fi
        fi
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
