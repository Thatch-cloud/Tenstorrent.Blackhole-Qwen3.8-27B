#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_DSPARK_MODE:-}" = request-verifier-profile
cd /opt/tt-metal
output=/experiment/results/request-verifier-profile
report_name=dspark-verifier-profile-hardware
profile_option=--profile-verifier
if [ "${QWEN_DSPARK_DRAFT_PROFILE:-0}" = 1 ]; then
    output=/experiment/results/request-draft-profile
    report_name=dspark-draft-profile-hardware
    profile_option=--profile-drafter
fi
mkdir -p "$output"
for name in memory.current memory.peak memory.events; do
    if [ -r "/sys/fs/cgroup/$name" ]; then cat "/sys/fs/cgroup/$name" > "$output/$name.before.txt"; fi
done
preserve_metadata() {
    mkdir -p "$output/metadata"
    for name in tracy_ops_data.csv cpp_device_perf_report.csv; do
        if [ -f "$output/.logs/$name" ]; then cp "$output/.logs/$name" "$output/metadata/$name"; fi
    done
    for name in memory.current memory.peak memory.events memory.max cpu.stat; do
        if [ -r "/sys/fs/cgroup/$name" ]; then cat "/sys/fs/cgroup/$name" > "$output/$name.txt"; fi
    done
    if [ -f "/experiment/results/$report_name.json" ]; then
        cp "/experiment/results/$report_name.json" "$output/request.json"
    fi
}
trap preserve_metadata EXIT
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
arguments=(--request "$profile_option" --checkpoint /dspark/model.safetensors --config /dspark/config.json
    --output "/experiment/results/$report_name.json")

timeout -k 30 4200 python3 -m tracy -p --check-exit-code --disable-device-data-dump-to-files \
    --disable-device-data-push-to-tracy --dump-device-data-mid-run --op-support-count 20000 -o "$output" \
    /experiment-scripts/ci/dspark-target-hardware.py "${arguments[@]}" 2>&1 | tee "$output/console.log"
preserve_metadata
if [ "${QWEN_DSPARK_DRAFT_PROFILE:-0}" = 1 ]; then
    python3 /experiment-scripts/ci/dspark_draft_profile_report.py "$output"
else
    python3 /experiment-scripts/ci/request_verifier_profile_report.py "$output" dspark
fi
