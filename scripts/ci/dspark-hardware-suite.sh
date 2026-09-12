#!/usr/bin/env bash
set -euo pipefail
mkdir -p /experiment/results
test "${QWEN_HARDWARE_TESTS:-0}" = 1
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${QWEN_PROJECTION_LINKS:-0}" = 4
test -z "${TT_METAL_SIMULATOR:-}"
test -z "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}"
test -z "${TT_METAL_SLOW_DISPATCH_MODE:-}"
export PYTHONPATH=/experiment-scripts/ci:/speculative-decoding/harness:/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
python3 /experiment-scripts/ci/dspark_native_restore.py
mode=${QWEN_DSPARK_MODE:-backbone}
[[ "$mode" = backbone || "$mode" = target || "$mode" = request || "$mode" = request-variants || "$mode" = request-native-attention || "$mode" = request-combined || "$mode" = request-target-attention || "$mode" = request-norm-scatter || "$mode" = request-verifier-profile ]]
if [[ "$mode" = request-variants || "$mode" = request-native-attention || "$mode" = request-combined || "$mode" = request-target-attention || "$mode" = request-norm-scatter || "$mode" = request-verifier-profile ]]; then
    test -f /experiment-optimisation/sim/gdn-multitoken.py
    ln -s /experiment-optimisation /optimisation
    test -f /experiment-scripts/ci/../../optimisation/sim/gdn-multitoken.py
fi
probe=dspark-pipeline-hardware
request_options=()
if [ "$mode" != backbone ]; then
    probe=dspark-target-hardware
    export MODEL_WEIGHTS_DIR=/models/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    export HF_MODEL="$MODEL_WEIGHTS_DIR"
fi
report_name=$probe
if [[ "$mode" = request || "$mode" = request-variants || "$mode" = request-native-attention || "$mode" = request-combined || "$mode" = request-target-attention || "$mode" = request-norm-scatter || "$mode" = request-verifier-profile ]]; then
    report_name=dspark-request-hardware
    request_options=(--request)
fi
if [ "$mode" = request-variants ]; then
    report_name=dspark-request-variants-hardware
    request_options+=(--request-variants)
fi
if [ "$mode" = request-native-attention ]; then
    report_name=dspark-native-attention-request-hardware
    request_options+=(--native-attention-variants)
fi
if [ "$mode" = request-target-attention ]; then
    report_name=dspark-target-attention-request-hardware
    request_options+=(--target-attention-variants)
    if [ "${QWEN_DSPARK_SCORE_LAYOUT:-0}" = 1 ]; then
        test "${QWEN_DSPARK_MLP_DOWN:-0}" = 0
        report_name=dspark-score-layout-request-hardware
        request_options+=(--score-layout)
        if [ "${QWEN_DSPARK_FUSION_T16:-0}" = 1 ]; then
            report_name=dspark-fusion-request-hardware
            request_options+=(--fused-t16-mlp)
            if [ "${QWEN_DSPARK_CAPTURED_PUBLICATION:-0}" = 1 ]; then
                report_name=dspark-captured-publication-request-hardware
                request_options+=(--captured-publication)
            fi
        fi
        if [ "${QWEN_DSPARK_NATIVE_SLOT:-0}" = 1 ]; then
            report_name=dspark-native-slot-request-hardware
            request_options+=(--native-slot-gdn)
        fi
        if [ "${QWEN_DSPARK_BANKED_PROPOSAL:-0}" = 1 ]; then
            report_name=dspark-banked-request-hardware
            request_options+=(--banked-proposal)
        fi
    fi
    if [ "${QWEN_DSPARK_MLP_DOWN:-0}" = 1 ]; then
        report_name=dspark-mlp-down-request-hardware
        request_options+=(--mlp-down)
        if [ "${QWEN_DSPARK_MLP_FOOTPRINT:-0}" = 1 ]; then
            report_name=dspark-mlp-footprint-request-hardware
            request_options+=(--mlp-equal-footprint)
        fi
    fi
fi
if [ "$mode" = request-combined ]; then
    report_name=dspark-combined-request-hardware
    request_options+=(--combined-variants)
fi
if [ "$mode" = request-norm-scatter ]; then
    report_name=dspark-norm-scatter-request-hardware
    request_options+=(--norm-scatter-variants)
fi
if [ "$mode" = request-verifier-profile ]; then
    report_name=dspark-verifier-profile-hardware
    request_options+=(--profile-verifier)
    if [ "${QWEN_DSPARK_DRAFT_PROFILE:-0}" = 1 ]; then
        report_name=dspark-draft-profile-hardware
        request_options=(--request --profile-drafter)
    fi
fi
python3 "/experiment-scripts/ci/$probe.py" --preflight "${request_options[@]}" \
    --checkpoint /dspark/model.safetensors --config /dspark/config.json \
    --output /experiment/results/dspark-python-preflight.json
build_started=$SECONDS
python3 /experiment-scripts/ci/dspark_runtime_cache.py
printf '{"build_seconds":%s,"scope":"isolated runtime build; excluded from kernel timing"}\n' "$((SECONDS-build_started))" \
    > /experiment/results/dspark-build-time.json
set +e
runner=(timeout -k 20 3000 python3 -u "/experiment-scripts/ci/$probe.py" "${request_options[@]}"
    --checkpoint /dspark/model.safetensors --config /dspark/config.json
    --output "/experiment/results/$report_name.json")
if [ "$mode" = request-verifier-profile ]; then
    runner=(bash /experiment-scripts/ci/dspark-request-profile.sh)
fi
"${runner[@]}" 2>&1 | tee "/experiment/results/$report_name.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "/experiment/results/$report_name.exit-status"
test "$status" = 0
if grep -q 'Failed to discover available ethernet links' "/experiment/results/$report_name.log"; then
    echo 'Explicit four-link integration unexpectedly invoked fallback discovery' >&2
    exit 1
fi
