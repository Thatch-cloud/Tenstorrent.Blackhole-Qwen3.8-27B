#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_DFLASH_VERIFIER_PROFILE:-0}" = 1
cd /opt/tt-metal
output=/experiment/results/request-verifier-profile
mkdir -p "$output"
preserve_metadata() {
    mkdir -p "$output/metadata"
    for name in tracy_ops_data.csv cpp_device_perf_report.csv; do
        if [ -f "$output/.logs/$name" ]; then cp "$output/.logs/$name" "$output/metadata/$name"; fi
    done
    for name in memory.current memory.peak memory.events cpu.stat; do
        if [ -r "/sys/fs/cgroup/$name" ]; then cat "/sys/fs/cgroup/$name" > "$output/$name.txt"; fi
    done
    if [ -f /experiment/results/full-dflash-request.json ]; then
        cp /experiment/results/full-dflash-request.json "$output/request.json"
    fi
}
trap preserve_metadata EXIT
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1
unset TT_METAL_PROFILER_MID_RUN_DUMP
arguments=(--max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input
    --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv
    --packed-checkpoints --ordered-cache --device-selection --request-pilot --norm-batch)
timeout -k 30 4200 python3 -m tracy -p --check-exit-code --disable-device-data-dump-to-files \
    --disable-device-data-push-to-tracy --op-support-count 20000 -o "$output" \
    /experiment-scripts/ci/full-prefix.py "${arguments[@]}" 2>&1 | tee "$output/console.log"
preserve_metadata
python3 /experiment-scripts/ci/request_verifier_profile_report.py "$output"
