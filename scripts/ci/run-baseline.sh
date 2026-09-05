#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
mkdir -p experiment-results
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
probe_id=''
test_id=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$test_id" ]; then
        docker logs "$test_id" > experiment-results/baseline-container.log 2>&1 || true
        docker cp "$test_id:/experiment/results/." experiment-results/ || true
        docker rm -f "$test_id" >/dev/null || true
    fi
    if [ -n "$probe_id" ]; then docker rm -f "$probe_id" >/dev/null || true; fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cache=''
for candidate in /home/thatch/.cache/huggingface /home/thatch/hf-cache /root/hf-cache /root/.cache/huggingface /srv/thatch/hf-cache; do
    if ! probe_id=$(docker create --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
        --mount "type=bind,src=$candidate,dst=/models,readonly" --entrypoint /bin/bash "$image" \
        -c 'test -f /models/hub/models--Qwen--Qwen3.8-27B/refs/main && test -d /models/hub/models--Qwen--Qwen3.8-27B/snapshots'); then
        probe_id=''
        continue
    fi
    docker start -a "$probe_id" || true
    status=$(docker inspect --format '{{.State.ExitCode}}' "$probe_id")
    docker rm "$probe_id" >/dev/null
    probe_id=''
    if [ "$status" = 0 ]; then cache=$candidate; break; fi
done
test -n "$cache" || { echo 'No reviewed local Qwen cache found; no downloads attempted' >&2; exit 1; }
printf 'image=%s\ncache=%s\n' "$image" "$cache" | tee experiment-results/baseline-host.log
volume=qwen-experiments-f1e9b1a64b4f
if docker volume inspect "$volume" >/dev/null 2>&1; then
    test "$(docker volume inspect --format '{{index .Labels "thatch.qwen.experiment-cache"}}' "$volume")" = true
else
    docker volume create --label thatch.qwen.experiment-cache=true "$volume" >/dev/null
fi
test_id=$(docker create --network none --cap-drop ALL --cap-add SYS_NICE \
    --security-opt no-new-privileges --pids-limit 4096 --memory 96g --cpus 24 --shm-size 8g \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount "type=bind,src=$cache,dst=/models,readonly" \
    --mount "type=volume,src=$volume,dst=/experiment-cache" \
    --label thatch.qwen.baseline=true --workdir /opt/vllm-tt-plugin \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/models -e HF_HUB_CACHE=/models/hub -e TT_METAL_HOME=/opt/tt-metal \
    -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels \
    -e MESH_DEVICE=P300 -e VLLM_PLUGINS=tt,tt_model_registry -e VLLM_RPC_TIMEOUT=100000 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto \
    -e QWEN36_BATCHED_DECODE_MODE=host -e QWEN36_SHARD_GREEDY=0 \
    -e QWEN_PREFILL_CONTINUATION=0 -e TT_PREFILL_DECODE_INTERLEAVE=0 \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/baseline-suite.sh)
docker cp scripts "$test_id:/experiment-scripts"
docker start -a "$test_id" | tee experiment-results/baseline-console.log
test "$(docker inspect --format '{{.State.ExitCode}}' "$test_id")" = 0
