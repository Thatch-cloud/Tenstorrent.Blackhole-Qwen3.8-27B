#!/usr/bin/env bash
set -euo pipefail
cd /opt/tt-metal
output=/experiment/results/verifier-profile
mkdir -p "$output"
preserve_metadata() {
    for directory in "$output" "$output"/context-*; do
        [ -d "$directory" ] || continue
        mkdir -p "$directory/metadata"
        for name in tracy_ops_data.csv cpp_device_perf_report.csv; do
            if [ -f "$directory/.logs/$name" ]; then cp "$directory/.logs/$name" "$directory/metadata/$name"; fi
        done
    done
    for name in memory.current memory.peak memory.events; do
        if [ -r "/sys/fs/cgroup/$name" ]; then cat "/sys/fs/cgroup/$name" > "$output/$name.txt"; fi
    done
}
trap preserve_metadata EXIT
arguments=(--max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input
    --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv
    --packed-checkpoints --ordered-cache --norm-batch --grouped-attention --attention-dma
    --attention-parallel --attention-tree --prefix-zero-reuse)
unset TTNN_OP_PROFILER TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_TRACE_TRACKING
timeout -k 30 1920 bash /experiment-scripts/ci/sdpa-tree-build.sh
export QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
timeout -k 30 2700 python3 /experiment-scripts/ci/full-prefix.py --correctness-only "${arguments[@]}" \
    2>&1 | tee "$output/correctness-console.log"
cp /experiment/results/full-gdn-device-loop.json "$output/correctness.json"
export TTNN_OP_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1
unset TT_METAL_PROFILER_MID_RUN_DUMP
for context in 4095 16383; do
    directory="$output/context-$context"
    mkdir -p "$directory"
    timeout -k 30 1500 python3 -m tracy -p --check-exit-code --disable-device-data-dump-to-files --disable-device-data-push-to-tracy --op-support-count 20000 -o "$directory" \
        /experiment-scripts/ci/full-prefix.py --device-profile --profile-context "$context" "${arguments[@]}" 2>&1 | tee "$directory/console.log"
    cp /experiment/results/full-gdn-device-loop.json "$directory/generation.json"
done
python3 /experiment-scripts/ci/check-verifier-profile.py "$output"
