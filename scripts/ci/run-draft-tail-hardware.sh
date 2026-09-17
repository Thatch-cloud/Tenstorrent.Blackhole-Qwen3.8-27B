#!/usr/bin/env bash
set -euo pipefail
test "${RUNNER_NAME:-}" = thatch-build-amd64-02-cp-temp
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
output=$(realpath -e experiment-results)
command -v fuser >/dev/null
sudo -n true
if sudo -n fuser /dev/tenstorrent/0 /dev/tenstorrent/2 > "$output/device-owners.txt" 2>&1; then
    echo 'Cards have an owner; refusing diagnostic' >&2
    exit 1
else
    test "$?" = 1
fi
container=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$container" ]; then
        timeout -k 1 5 docker kill "$container" >/dev/null 2>&1 || true
        timeout -k 1 5 docker logs "$container" > "$output/container.log" 2>&1 || true
        timeout -k 1 5 docker inspect --format '{{json .State}}' "$container" > "$output/container-state.json" 2>&1 || true
        timeout -k 1 10 docker rm -f "$container" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
container=$(timeout -k 5 30 docker create --network none --hostname qwen-experiment --add-host qwen-experiment:127.0.0.1 \
    --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
    --pids-limit 4096 --memory 32g --cpus 16 --shm-size 4g \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount "type=bind,src=$output,dst=/experiment/results" --group-add "$(stat -c %g "$output")" \
    --label thatch.qwen.baseline=true --label "thatch.qwen.workflow-run=$GITHUB_RUN_ID" \
    --label "thatch.qwen.source-revision=$GITHUB_SHA" --workdir /opt/tt-metal \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e TT_METAL_HOME=/opt/tt-metal -e MESH_DEVICE=P300 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/draft-tail-hardware-suite.sh)
docker cp scripts "$container:/experiment-scripts"
timeout -k 10 260 docker start -a "$container"
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
