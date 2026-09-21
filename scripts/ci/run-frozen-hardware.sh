#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${RUNNER_NAME:-}" = thatch-build-amd64-02-cp-temp
test -z "${TT_METAL_SIMULATOR:-}"
mkdir -p experiment-results
results=$(cd experiment-results && pwd -P)
results_gid=$(stat -c %g "$results")
chmod g+rwx "$results"
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
container=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$container" ]; then
        timeout -k 5 20 docker inspect --format '{{json .State}}' "$container" > "$results/frozen-hardware-container-state.json" || true
        timeout -k 5 20 docker rm -f "$container" >/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
container=$(docker create --network none --hostname qwen-experiment --add-host qwen-experiment:127.0.0.1 \
    --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges --group-add "$results_gid" \
    --pids-limit 4096 --memory 96g --cpus 24 --shm-size 8g \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount "type=bind,src=$results,dst=/experiment/results" \
    --label thatch.qwen.baseline=true --label "thatch.qwen.workflow-run=${GITHUB_RUN_ID:-untracked}" \
    --label "thatch.qwen.source-revision=${GITHUB_SHA:-untracked}" \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 \
    -e QWEN_LADDER_BACKEND=hardware -e QWEN_LADDER_CONTEXT=65536 \
    -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/tmp/frozen-hardware-kernels -e MESH_DEVICE=P300 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/frozen-hardware-suite.sh)
docker cp scripts "$container:/experiment-scripts"
docker start -a "$container" 2>&1 | tee "$results/frozen-hardware-console.log"
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
