#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_TENSIX_MLP_PROFILE:-0}" = 1
producers=${QWEN_TENSIX_MLP_PRODUCERS:-8}
suffix=''
if [ "$producers" = 16 ]; then suffix=-16; else test "$producers" = 8; fi
cd /opt/tt-metal
output=/experiment/results/tensix-mlp-profile
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
timeout -k 30 1200 python3 -m tracy -p --check-exit-code --disable-device-data-dump-to-files \
    --disable-device-data-push-to-tracy --dump-device-data-mid-run --op-support-count 20000 -o "$output" \
    /experiment-scripts/ci/tensix-stream-mlp-hardware.py --profile \
    --producers "$producers" \
    --simulator-report "/experiment-scripts/ci/tensix-mlp-simulator$suffix.json" \
    --simulator-exit-status "/experiment-scripts/ci/tensix-mlp-simulator$suffix.exit-status" \
    --output "$output/mlp.json" 2>&1 | tee "$output/console.log"
preserve_metadata
python3 /experiment-scripts/ci/tensix_mlp_profile_report.py "$output"
