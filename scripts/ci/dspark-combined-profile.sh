#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_COMBINED_TRACE_PROFILE:-0}" = 1
test "${QWEN_DSPARK_MODE:-}" = request-target-attention
test "${QWEN_DSPARK_CAPTURED_PUBLICATION:-0}" = 1
test "${QWEN_DSPARK_FUSION_T16:-0}" = 1
test "${QWEN_DSPARK_SCORE_LAYOUT:-0}" = 1
cd /opt/tt-metal
output=/experiment/results/combined-verifier-profile
report=/experiment/results/dspark-captured-publication-request-hardware.json
mkdir -p "$output"
preserve_metadata() {
    mkdir -p "$output/metadata"
    for name in tracy_ops_data.csv cpp_device_perf_report.csv; do
        if [ -f "$output/.logs/$name" ]; then cp "$output/.logs/$name" "$output/metadata/$name"; fi
    done
    if [ -f "$report" ]; then cp "$report" "$output/request.json"; fi
}
trap preserve_metadata EXIT
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
timeout -k 30 4200 python3 -m tracy -p --check-exit-code --disable-device-data-dump-to-files \
    --disable-device-data-push-to-tracy --dump-device-data-mid-run --op-support-count 20000 -o "$output" \
    /experiment-scripts/ci/dspark-target-hardware.py --request --target-attention-variants \
    --score-layout --fused-t16-mlp --captured-publication --max-new-tokens 64 \
    --checkpoint /dspark/model.safetensors --config /dspark/config.json --output "$report" \
    2>&1 | tee "$output/console.log"
preserve_metadata
python3 /experiment-scripts/ci/request_verifier_profile_report.py "$output" dspark
