#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
mode=${QWEN_RUN_MODE:-baseline}
ratio=${QWEN_INTERLEAVE_RATIO:-0}
[[ "$mode" = baseline || "$mode" = interleave || "$mode" = profile || "$mode" = model-profile || "$mode" = mlp-sweep || "$mode" = mlp-packing || "$mode" = mlp-fusion || "$mode" = projection-1d || "$mode" = full-model-fusion || "$mode" = full-prefix || "$mode" = full-batch || "$mode" = full-gdn-row-layout || "$mode" = full-gdn-row-clones || "$mode" = full-gdn-input-reuse || "$mode" = full-compact-gdn || "$mode" = full-coding-cost || "$mode" = full-batch-attribution || "$mode" = attention-batch || "$mode" = attention-timing || "$mode" = gdn-prefix || "$mode" = gdn-block || "$mode" = gdn-active || "$mode" = gdn-multitoken || "$mode" = gdn-multitoken-norm || "$mode" = device-readback || "$mode" = gdn-checkpoint-dma || "$mode" = gdn-checkpoint-cost || "$mode" = gdn-inplace-timing || "$mode" = gdn-inplace || "$mode" = gdn-direct || "$mode" = sampling-kernel || "$mode" = sampling || "$mode" = sampling-extended ]]
[[ "$ratio" = 0 || "$ratio" = 1 || "$ratio" = 2 || "$ratio" = 4 ]]
output=experiment-results
if [ "$mode" = interleave ]; then output="$output/interleave-$ratio"; fi
mkdir -p "$output"
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
test_id=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$test_id" ]; then
        docker logs "$test_id" > "$output/baseline-container.log" 2>&1 || true
        docker cp "$test_id:/experiment/results/." "$output/" || true
        docker rm -f "$test_id" >/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cache=/home/thatch/hf-cache
printf 'image=%s\ncache=%s\n' "$image" "$cache" | tee "$output/baseline-host.log"
volume=qwen-experiments-f1e9b1a64b4f
if docker volume inspect "$volume" >/dev/null 2>&1; then
    test "$(docker volume inspect --format '{{index .Labels "thatch.qwen.experiment-cache"}}' "$volume")" = true
else
    docker volume create --label thatch.qwen.experiment-cache=true "$volume" >/dev/null
fi
test_id=$(docker create --network none --hostname qwen-experiment --add-host qwen-experiment:127.0.0.1 \
    --cap-drop ALL --cap-add SYS_NICE \
    --security-opt no-new-privileges --pids-limit 4096 --memory 96g --cpus 24 --shm-size 8g \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount "type=bind,src=$cache/hub/models--Qwen--Qwen3.8-27B,dst=/models/hub/models--Qwen--Qwen3.8-27B,readonly" \
    --mount "type=volume,src=$volume,dst=/experiment-cache" \
    --label thatch.qwen.baseline=true --workdir /opt/vllm-tt-plugin \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/models -e HF_HUB_CACHE=/models/hub -e TT_METAL_HOME=/opt/tt-metal \
    -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels \
    -e MESH_DEVICE=P300 -e VLLM_PLUGINS=tt,tt_model_registry -e VLLM_RPC_TIMEOUT=100000 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto \
    -e QWEN36_BATCHED_DECODE_MODE=host -e QWEN36_SHARD_GREEDY=0 \
    -e QWEN_PREFILL_CONTINUATION=0 -e TT_PREFILL_DECODE_INTERLEAVE=0 \
    -e "QWEN_RUN_MODE=$mode" -e "QWEN_INTERLEAVE_RATIO=$ratio" \
    -e "QWEN_BOUNDARY_DIAGNOSTICS=${QWEN_BOUNDARY_DIAGNOSTICS:-0}" \
    -e "QWEN_INTERLEAVE_MIXED=${QWEN_INTERLEAVE_MIXED:-0}" \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/baseline-suite.sh)
docker cp scripts "$test_id:/experiment-scripts"
docker cp optimisation "$test_id:/experiment-optimisation"
docker start -a "$test_id" | tee "$output/baseline-console.log"
test "$(docker inspect --format '{{.State.ExitCode}}' "$test_id")" = 0
