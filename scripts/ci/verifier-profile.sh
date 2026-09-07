#!/usr/bin/env bash
set -euo pipefail
cd /opt/tt-metal
output=/experiment/results/verifier-profile
mkdir -p "$output"
preserve_metadata() {
    mkdir -p "$output/metadata"
    for name in tracy_ops_data.csv cpp_device_perf_report.csv; do
        if [ -f "$output/.logs/$name" ]; then cp "$output/.logs/$name" "$output/metadata/$name"; fi
    done
}
trap preserve_metadata EXIT
export QWEN_PROFILE_VERIFIER=1
python3 /experiment-scripts/ci/stage-model-profile.py > "$output/report-stage.json"
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
unset TT_METAL_PROFILER_MID_RUN_DUMP
timeout -k 30 5400 python3 -m tracy -p -r --disable-device-data-dump-to-files --op-support-count 10000 -o "$output" \
    /experiment-scripts/ci/full-prefix.py --device-profile --max-rows 32 --batch --coding-cost --serial-sdpa \
    --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn \
    --compact-prologue --batch-conv --packed-checkpoints --ordered-cache 2>&1 | tee "$output/console.log"
python3 /experiment-scripts/ci/check-verifier-profile.py "$output" /experiment/results/full-gdn-device-loop.json
