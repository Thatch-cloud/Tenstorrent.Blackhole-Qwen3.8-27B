#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_DSPARK_COMBINED_DEVICE_PROFILE:-0}" = 1
test "${QWEN_DSPARK_64K_TRIAL:-0}" = 1
test "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1
output=/experiment/results/combined-draft-device-profile
mkdir -p "$output"
preserve_metadata() {
    mkdir -p "$output/metadata"
    for name in tracy_ops_data.csv cpp_device_perf_report.csv; do
        if [ -f "$output/.logs/$name" ]; then cp "$output/.logs/$name" "$output/metadata/$name"; fi
    done
}
trap preserve_metadata EXIT
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
python3 -m tracy -p --check-exit-code --disable-device-data-dump-to-files \
    --disable-device-data-push-to-tracy --dump-device-data-mid-run --op-support-count 20000 -o "$output" \
    "$@" 2>&1 | tee "$output/console.log"
preserve_metadata
test -s "$output/metadata/tracy_ops_data.csv"
test -s "$output/metadata/cpp_device_perf_report.csv"
