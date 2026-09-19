#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${RUNNER_NAME:-}" = thatch-build-amd64-02-cp-temp
test -z "${TT_METAL_SIMULATOR:-}"
test -f experiment-evidence/proposal/t32-combined.json
test -f experiment-evidence/score/t32-markov.json
test -f experiment-evidence/attention/t32-attention.json
test -f experiment-evidence/commit/t32-commit.json
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
fixture=/home/thatch/.cache/qwen-experiments/dspark-b9a5dbdf03bc999c6c73c426b19c2d9041cea393
target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
test -f "$fixture/model.safetensors"
test -f "$fixture/config.json"
test -d "$target/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
volume=qwen-experiments-f1e9b1a64b4f
test "$(docker volume inspect --format '{{index .Labels "thatch.qwen.experiment-cache"}}' "$volume")" = true
mkdir -p experiment-results
container=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$container" ]; then
        docker stop -t 15 "$container" >/dev/null 2>&1 || true
        docker cp "$container:/experiment/results/." experiment-results/ || true
        docker rm "$container" >/dev/null || true
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
    --mount "type=bind,src=$fixture,dst=/dspark,readonly" \
    --mount "type=bind,src=$target,dst=/models/hub/models--Qwen--Qwen3.8-27B,readonly" \
    --mount "type=bind,src=$PWD/experiment-evidence,dst=/evidence,readonly" \
    --mount "type=volume,src=$volume,dst=/experiment-cache" \
    --label thatch.qwen.baseline=true --workdir /opt/vllm-tt-plugin \
    --label "thatch.qwen.workflow-run=$GITHUB_RUN_ID" --label "thatch.qwen.source-revision=$GITHUB_SHA" \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 -e QWEN_CCL_LAZY_BUILD=1 \
    -e QWEN_T32_FUSED_SCORE_HARDWARE=1 -e QWEN_DSPARK_REQUEST_CONTEXT=4096 \
    -e "QWEN_T32_TIMED=${QWEN_T32_TIMED:-0}" \
    -e "QWEN_SOURCE_REVISION=$GITHUB_SHA" -e "QWEN_WORKFLOW_RUN=$GITHUB_RUN_ID" \
    -e MODEL_WEIGHTS_DIR=/models/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TT_METAL_HOME=/opt/tt-metal \
    -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels -e MESH_DEVICE=P300 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/t32-combined-suite.sh)
docker cp scripts "$container:/experiment-scripts"
docker cp speculative-decoding "$container:/speculative-decoding"
docker cp optimisation "$container:/experiment-optimisation"
docker cp optimisation/sim/sdpa-graft-registration.patch "$container:/tmp/ccl-graft-registration.patch"
docker start -a "$container" | tee experiment-results/console.log
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
