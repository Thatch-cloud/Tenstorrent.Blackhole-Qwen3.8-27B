#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${RUNNER_NAME:-}" = thatch-build-amd64-02-cp-temp
test -z "${TT_METAL_SIMULATOR:-}"
mode=${QWEN_DSPARK_MODE:-backbone}
draft_profile=${QWEN_DSPARK_DRAFT_PROFILE:-0}
[[ "$draft_profile" = 0 || "$draft_profile" = 1 ]]
if [ "$draft_profile" = 1 ]; then test "$mode" = request-verifier-profile; fi
mlp_down=${QWEN_DSPARK_MLP_DOWN:-0}
score_layout=${QWEN_DSPARK_SCORE_LAYOUT:-0}
banked_proposal=${QWEN_DSPARK_BANKED_PROPOSAL:-0}
native_slot=${QWEN_DSPARK_NATIVE_SLOT:-0}
fusion=${QWEN_DSPARK_FUSION_T16:-0}
publication=${QWEN_DSPARK_CAPTURED_PUBLICATION:-0}
[[ "$publication" = 0 || "$publication" = 1 ]]
if [ "$publication" = 1 ]; then test "$fusion" = 1; test "$mode" = request-target-attention; fi
publication_report=''
if [ "$publication" = 1 ]; then
    evidence=$(mktemp -d "$RUNNER_TEMP/qwen-publication-evidence.XXXXXX")
    gh run download 34677941763 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34677941763 --dir "$evidence"
    publication_report="$evidence/t32-publication.json"
    printf '%s  %s\n' 4bd749d6381cb7e1f5be69276d5a5011c9e6cfa7c30182b44dd09f3d1b115914 "$publication_report" | sha256sum -c -
fi
[[ "$fusion" = 0 || "$fusion" = 1 ]]
if [ "$fusion" = 1 ]; then test "$score_layout" = 1; test "$native_slot" = 0; test "$banked_proposal" = 0; fi
[[ "$native_slot" = 0 || "$native_slot" = 1 ]]
if [ "$native_slot" = 1 ]; then test "$score_layout" = 1; test "$banked_proposal" = 0; fi
[[ "$banked_proposal" = 0 || "$banked_proposal" = 1 ]]
if [ "$banked_proposal" = 1 ]; then test "$score_layout" = 1; fi
[[ "$score_layout" = 0 || "$score_layout" = 1 ]]
if [ "$score_layout" = 1 ]; then
    test "$mode" = request-target-attention
    test "$mlp_down" = 0
    test "$draft_profile" = 0
fi
mlp_footprint=${QWEN_DSPARK_MLP_FOOTPRINT:-0}
[[ "$mlp_footprint" = 0 || "$mlp_footprint" = 1 ]]
if [ "$mlp_footprint" = 1 ]; then test "$mlp_down" = 1; fi
[[ "$mlp_down" = 0 || "$mlp_down" = 1 ]]
if [ "$mlp_down" = 1 ]; then test "$mode" = request-target-attention; fi
task=${QWEN_DSPARK_CODING_TASK:-merge_intervals}
case "$task" in
    merge_intervals) ;;
    stable_unique_v1|run_length_encode_v1|rotate_right_v1) test "$mode" = request-target-attention ;;
    *) exit 64 ;;
esac
[[ "$mode" = backbone || "$mode" = target || "$mode" = request || "$mode" = request-variants || "$mode" = request-native-attention || "$mode" = request-combined || "$mode" = request-target-attention || "$mode" = request-norm-scatter || "$mode" = request-verifier-profile ]]
target_mount=()
if [ "$mode" != backbone ]; then
    target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
    test -d "$target/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    target_mount=(--mount "type=bind,src=$target,dst=/models/hub/models--Qwen--Qwen3.8-27B,readonly")
fi
output=experiment-results
mkdir -p "$output"
PYTHONPATH=scripts/ci python3 -c \
    'import json; from dspark_hardware_gate import simulator_preflight; print(json.dumps(simulator_preflight("scripts/ci"),indent=2))' \
    > "$output/dspark-simulator-preflight.json"
fixture=/home/thatch/.cache/qwen-experiments/dspark-b9a5dbdf03bc999c6c73c426b19c2d9041cea393
timeout -k 10 1200 python3 scripts/ci/dspark-hardware-fixtures.py --output "$fixture" \
    > "$output/dspark-checkpoint-manifest.json"
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
if [ "$fusion" = 1 ]; then
    timeout -k 15 300 docker run --rm --network none --cap-drop ALL \
        --security-opt no-new-privileges --memory 4g --cpus 2 \
        --mount "type=bind,src=$PWD,dst=/source,readonly" --workdir /source \
        -e PYTHONPATH=/source/scripts/ci:/source/speculative-decoding/harness \
        -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 "$image" -B -m unittest \
        test_packed_weight_check test_dspark_fusion_variants test_fused_t16_scope test_fused_t16_admission \
        test_full_dspark_request test_dspark_score_layout_variants \
        2>&1 | tee "$output/fusion-host-tests.log"
fi
test_id=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$test_id" ]; then
        docker logs "$test_id" > "$output/dspark-container.log" 2>&1 || true
        docker cp "$test_id:/experiment/results/." "$output/" || true
        docker inspect --format '{{json .State}}' "$test_id" > "$output/dspark-container-state.json" || true
        docker rm -f "$test_id" >/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
volume=qwen-experiments-f1e9b1a64b4f
if docker volume inspect "$volume" >/dev/null 2>&1; then
    test "$(docker volume inspect --format '{{index .Labels "thatch.qwen.experiment-cache"}}' "$volume")" = true
else
    docker volume create --label thatch.qwen.experiment-cache=true "$volume" >/dev/null
fi
test_id=$(docker create --network none --hostname qwen-experiment --add-host qwen-experiment:127.0.0.1 \
    --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
    --pids-limit 4096 --memory 96g --cpus 24 --shm-size 8g \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount "type=bind,src=$fixture,dst=/dspark,readonly" \
    "${target_mount[@]}" \
    --mount "type=volume,src=$volume,dst=/experiment-cache" \
    --label thatch.qwen.baseline=true --workdir /opt/vllm-tt-plugin \
    --label "thatch.qwen.workflow-run=${GITHUB_RUN_ID:-untracked}" \
    --label "thatch.qwen.source-revision=${GITHUB_SHA:-untracked}" \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 -e QWEN_CCL_LAZY_BUILD=1 \
    -e "QWEN_DSPARK_MODE=$mode" \
    -e "QWEN_DSPARK_DRAFT_PROFILE=$draft_profile" \
    -e "QWEN_DSPARK_MLP_DOWN=$mlp_down" \
    -e "QWEN_DSPARK_SCORE_LAYOUT=$score_layout" \
    -e "QWEN_DSPARK_BANKED_PROPOSAL=$banked_proposal" \
    -e "QWEN_DSPARK_NATIVE_SLOT=$native_slot" \
    -e "QWEN_DSPARK_FUSION_T16=$fusion" \
    -e "QWEN_DSPARK_CAPTURED_PUBLICATION=$publication" \
    -e "QWEN_DSPARK_MLP_FOOTPRINT=$mlp_footprint" \
    -e "QWEN_DSPARK_CODING_TASK=$task" \
    -e "QWEN_SOURCE_REVISION=${GITHUB_SHA:-untracked}" -e "QWEN_WORKFLOW_RUN=${GITHUB_RUN_ID:-untracked}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TT_METAL_HOME=/opt/tt-metal \
    -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels -e MESH_DEVICE=P300 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/dspark-hardware-suite.sh)
docker cp scripts "$test_id:/experiment-scripts"
if [ "$publication" = 1 ]; then
    docker cp "$publication_report" "$test_id:/experiment-scripts/ci/dspark-publication-simulator.json"
fi
docker cp optimisation "$test_id:/experiment-optimisation"
if [[ "$mode" = request || "$mode" = request-variants || "$mode" = request-native-attention || "$mode" = request-combined || "$mode" = request-target-attention || "$mode" = request-norm-scatter || "$mode" = request-verifier-profile ]]; then
    docker cp speculative-decoding "$test_id:/speculative-decoding"
fi
docker cp optimisation/sim/sdpa-graft-registration.patch "$test_id:/tmp/ccl-graft-registration.patch"
docker start -a "$test_id" | tee "$output/dspark-console.log"
test "$(docker inspect --format '{{.State.ExitCode}}' "$test_id")" = 0
