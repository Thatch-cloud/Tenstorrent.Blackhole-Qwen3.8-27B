#!/usr/bin/env bash
set -euo pipefail
cd /opt/tt-metal
output=/experiment/results/model-profile
mkdir -p "$output"
preserve_metadata() {
    mkdir -p "$output/metadata"
    for name in tracy_ops_data.csv cpp_device_perf_report.csv; do
        if [ -f "$output/.logs/$name" ]; then cp "$output/.logs/$name" "$output/metadata/$name"; fi
    done
}
trap preserve_metadata EXIT
python3 /experiment-scripts/ci/stage-model-profile.py > "$output/report-stage.json"
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
unset TT_METAL_PROFILER_MID_RUN_DUMP
timeout -k 30 1200 python3 -m tracy -p -r --disable-device-data-dump-to-files --op-support-count 10000 -o "$output" \
    /experiment-scripts/ci/model-profile.py 2>&1 | tee "$output/console.log"
python3 /experiment-scripts/ci/check-model-profile.py "$output"
python3 /experiment-scripts/ci/summarize-model-shapes.py "$output"
python3 /experiment-scripts/ci/summarize-gdn-state.py "$output"
