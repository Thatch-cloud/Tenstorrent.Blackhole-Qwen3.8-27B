#!/usr/bin/env bash
set -euo pipefail
test "${RUNNER_NAME:-}" = thatch-build-amd64-02-cp-temp
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test -z "${TT_METAL_SIMULATOR:-}"
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
output=$(realpath -e experiment-results)
evidence=$(mktemp -d "$RUNNER_TEMP/qwen-splitk-evidence.XXXXXX")
timeout -k 5 45 gh run download 34913565056 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
    --name qwen-hardware-inventory-34913565056 --dir "$evidence"
cp "$evidence/dspark-splitk.json" scripts/ci/dspark-splitk-simulator.json
PYTHONPATH=scripts/ci python3 -c 'from dspark_splitk_sim_gate import qualify; qualify("scripts/ci", "scripts/ci/dspark-splitk-simulator.json")'
volume=qwen-experiments-f1e9b1a64b4f
if docker volume inspect "$volume" >/dev/null 2>&1; then
    test "$(docker volume inspect --format '{{index .Labels "thatch.qwen.experiment-cache"}}' "$volume")" = true
else
    docker volume create --label thatch.qwen.experiment-cache=true "$volume" >/dev/null
fi
container=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$container" ]; then
        timeout -k 1 5 docker kill "$container" >/dev/null 2>&1 || true
        docker logs "$container" > "$output/splitk-container.log" 2>&1 || true
        docker inspect --format '{{json .State}}' "$container" > "$output/splitk-container-state.json" || true
        timeout -k 1 10 docker rm -f "$container" >/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
container=$(docker create --network none --hostname qwen-experiment --add-host qwen-experiment:127.0.0.1 \
    --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
    --pids-limit 4096 --memory 96g --cpus 24 --shm-size 8g \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount "type=bind,src=$output,dst=/experiment/results" --group-add "$(stat -c %g "$output")" \
    --mount "type=volume,src=$volume,dst=/experiment-cache" \
    --label thatch.qwen.baseline=true --label "thatch.qwen.workflow-run=$GITHUB_RUN_ID" \
    --label "thatch.qwen.source-revision=$GITHUB_SHA" --workdir /opt/vllm-tt-plugin \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 \
    -e QWEN_SPLITK_ATTENTION=1 -e QWEN_LADDER_BACKEND=hardware -e QWEN_LADDER_CONTEXT=65536 \
    -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/experiment-cache/kernels -e MESH_DEVICE=P300 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/dspark-splitk-hardware-suite.sh)
docker cp scripts "$container:/experiment-scripts"
docker cp optimisation "$container:/optimisation"
docker cp speculative-decoding "$container:/speculative-decoding"
timeout -k 10 510 docker start -a "$container"
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
