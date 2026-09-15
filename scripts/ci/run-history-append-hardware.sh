#!/usr/bin/env bash
set -euo pipefail
test "${RUNNER_NAME:-}" = thatch-build-amd64-02-cp-temp
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test -z "${TT_METAL_SIMULATOR:-}"
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
output=$(realpath -e experiment-results)
timeout -k 5 45 gh api repos/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/artifacts/10418288112/zip > "$output/simulator.zip"
python3 -c 'import pathlib,sys,zipfile; root=pathlib.Path(sys.argv[1]); (root/"history-append-simulator.json").write_bytes(zipfile.ZipFile(root/"simulator.zip").read("history-append.json"))' "$output"
PYTHONPATH=scripts/ci python3 -c 'from history_append_sim_gate import qualify; qualify("scripts/ci", "experiment-results/history-append-simulator.json")'
container=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$container" ]; then
        timeout -k 1 5 docker kill "$container" >/dev/null 2>&1 || true
        timeout -k 1 5 docker logs "$container" > "$output/history-container.log" 2>&1 || true
        timeout -k 1 5 docker inspect --format '{{json .State}}' "$container" > "$output/history-container-state.json" 2>&1 || true
        timeout -k 1 10 docker rm -f "$container" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
container=$(timeout -k 5 45 docker create --network none --hostname qwen-experiment --add-host qwen-experiment:127.0.0.1 \
    --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
    --pids-limit 4096 --memory 96g --cpus 24 --shm-size 8g \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount "type=bind,src=$output,dst=/experiment/results" --group-add "$(stat -c %g "$output")" \
    --label thatch.qwen.baseline=true --label "thatch.qwen.workflow-run=$GITHUB_RUN_ID" \
    --label "thatch.qwen.source-revision=$GITHUB_SHA" --workdir /opt/vllm-tt-plugin \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 \
    -e TT_METAL_HOME=/opt/tt-metal -e MESH_DEVICE=P300 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/history-append-hardware-suite.sh)
docker cp scripts "$container:/experiment-scripts"
timeout -k 10 240 docker start -a "$container"
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
