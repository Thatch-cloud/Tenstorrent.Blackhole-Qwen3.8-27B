#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
mode=${QWEN_RUN_MODE:-baseline}
mtp_drafts=0
dflash_drafts=0
dflash_capture=0
dflash_commit_abba=0
if [ "$mode" = full-dflash-commit-request ]; then
    dflash_commit_abba=1
    mode=full-dflash-trace-request
fi
if [[ "$mode" = full-dflash-trace-request || "$mode" = full-dflash-wide-trace-request ]]; then
    dflash_capture=1
    mode=${mode/-trace/}
fi
if [[ "$mode" = full-dflash-request || "$mode" = full-dflash-wide-request ]]; then
    [[ "${QWEN_LOOKUP_CAP_ABBA:-0}" = 0 && "${QWEN_LEARNED_STACK:-0}" = 0 && "${QWEN_PREFIX_ZERO_REUSE:-0}" = 0 ]]
    dflash_drafts=7
    if [ "$mode" = full-dflash-wide-request ]; then dflash_drafts=31; fi
    mode=full-norm-engine
    export QWEN_CODING_REQUEST=1 QWEN_FABRIC_LINK_PROBE=1
fi
if [ "$mode" = full-mtp-request ]; then
    [[ "${QWEN_LOOKUP_CAP_ABBA:-0}" = 0 && "${QWEN_LEARNED_STACK:-0}" = 0 && "${QWEN_PREFIX_ZERO_REUSE:-0}" = 0 ]]
    mtp_drafts=7
    mode=full-norm-engine
    export QWEN_CODING_REQUEST=1 QWEN_FABRIC_LINK_PROBE=1
fi
[[ "${QWEN_FABRIC_LINK_PROBE:-0}" = 0 || ( "${QWEN_FABRIC_LINK_PROBE:-0}" = 1 &&
    ( "$mode" = sampling-kernel || "$mode" = learned-attention || ( "$mode" = full-norm-engine && "${QWEN_CODING_REQUEST:-0}" = 1 && "${QWEN_LOOKUP_CAP_ABBA:-0}" = 0 ) ) ) ]]
descriptor=p300_mesh_graph_descriptor.textproto
if [ "${QWEN_FABRIC_LINK_PROBE:-0}" = 1 ]; then descriptor=p150_x2_mesh_graph_descriptor.textproto; fi
projection_links=1
ccl_build=0
if [[ "$mode" = learned-attention && "${QWEN_FABRIC_LINK_PROBE:-0}" = 1 ]]; then ccl_build=1; fi
if [ "$mtp_drafts" != 0 ]; then ccl_build=1; fi
if [ "$dflash_drafts" != 0 ]; then ccl_build=1; projection_links=4; fi
if [[ "$mode" = learned-attention || "$mode" = learned-mlp || "$mode" = feature-projection || "$mode" = feature-projection-full ]]; then
    descriptor=p150_x2_mesh_graph_descriptor.textproto
    projection_links=4
fi
if [[ "${QWEN_CODING_REQUEST:-0}" != 0 && !( "${QWEN_CODING_REQUEST:-0}" = 1 && "$mode" = full-norm-engine ) ]]; then
    echo 'Short coding workload requires full-norm-engine; long-context replay is not qualified' >&2
    exit 2
fi
[[ "${QWEN_LOOKUP_CAP_ABBA:-0}" = 0 || ( "${QWEN_LOOKUP_CAP_ABBA:-0}" = 1 && "${QWEN_CODING_REQUEST:-0}" = 1 && "$mode" = full-norm-engine ) ]]
ratio=${QWEN_INTERLEAVE_RATIO:-0}
[[ "$mode" = learned-mlp || "$mode" = learned-attention || "$mode" = learned-convolution || "$mode" = feature-projection || "$mode" = feature-projection-full ]] ||
[[ "$mode" = baseline || "$mode" = interleave || "$mode" = profile || "$mode" = model-profile || "$mode" = verifier-profile || "$mode" = mlp-sweep || "$mode" = mlp-packing || "$mode" = mlp-fusion || "$mode" = projection-1d || "$mode" = full-model-fusion || "$mode" = target-feature-prefill || "$mode" = target-feature-prefix || "$mode" = target-feature-replay || "$mode" = target-feature-batch || "$mode" = target-features || "$mode" = full-prefix || "$mode" = full-batch || "$mode" = full-gdn-row-layout || "$mode" = full-gdn-device-loop || "$mode" = full-attention-engine-wide || "$mode" = full-attention-engine || "$mode" = full-attention-replay || "$mode" = full-attention-mask-once || "$mode" = full-attention-tree-replay || "$mode" = full-attention-tree || "$mode" = full-attention-parallel || "$mode" = full-attention-dma || "$mode" = full-attention-groups || "$mode" = full-norm-batch || "$mode" = full-norm-replay || "$mode" = full-norm-selection || "$mode" = full-norm-engine || "$mode" = full-verifier-replay || "$mode" = full-verifier-engine || "$mode" = full-verifier-selection || "$mode" = full-gdn-row-clones || "$mode" = full-gdn-input-reuse || "$mode" = full-compact-gdn || "$mode" = full-coding-cost || "$mode" = full-batch-attribution || "$mode" = attention-batch || "$mode" = attention-timing || "$mode" = attention-tree-layer || "$mode" = attention-tree-replay || "$mode" = attention-tree-parallel || "$mode" = attention-tree-scratch || "$mode" = attention-replay || "$mode" = attention-mask-replay || "$mode" = attention-parallel-groups || "$mode" = attention-dma-layer || "$mode" = attention-group-dma || "$mode" = attention-group-layer || "$mode" = attention-groups || "$mode" = gdn-prefix || "$mode" = gdn-block || "$mode" = gdn-active || "$mode" = gdn-multitoken || "$mode" = gdn-value-split || "$mode" = gdn-value-split-timing || "$mode" = gdn-value-split-prefetch || "$mode" = gdn-value-split-norm-batch || "$mode" = gdn-norm-batch-layer || "$mode" = gdn-multitoken-norm || "$mode" = gdn-multitoken-conv || "$mode" = device-readback || "$mode" = gdn-checkpoint-dma || "$mode" = gdn-checkpoint-cost || "$mode" = gdn-inplace-timing || "$mode" = gdn-inplace || "$mode" = gdn-direct || "$mode" = sampling-kernel || "$mode" = sampling || "$mode" = sampling-extended ]]
[[ "$ratio" = 0 || "$ratio" = 1 || "$ratio" = 2 || "$ratio" = 4 ]]
output=experiment-results
[[ "${QWEN_PREFIX_ZERO_REUSE:-0}" = 0 || ( "${QWEN_PREFIX_ZERO_REUSE:-0}" = 1 && "$mode" = full-attention-tree ) ]]
[[ "${QWEN_LEARNED_STACK:-0}" = 0 || ( "${QWEN_LEARNED_STACK:-0}" = 1 && "$mode" = learned-attention ) ]]
if [ "$mode" = interleave ]; then output="$output/interleave-$ratio"; fi
mkdir -p "$output"
if [ "$dflash_drafts" != 0 ]; then
    source scripts/ci/dflash-fixtures.sh
    prepare_dflash_fixtures
fi
projection_fixture="$output/draft-projection-fixture"
if [ "$mode" = learned-mlp ]; then
    projection_fixture=/home/thatch/.cache/qwen-experiments/dflash2-mlp-dedf8df68adfb1afeaf7b7480c0a0243108177b4
    timeout -k 10 1200 python3 scripts/ci/draft_mlp_fixture.py --reuse-verified --output "$projection_fixture"
    cp "$projection_fixture/manifest.json" "$output/draft-mlp-manifest.json"
fi
if [[ "$mode" = learned-mlp || "$mode" = learned-attention ]]; then
    convolution_fixture=/home/thatch/.cache/qwen-experiments/dflash2-convolution-dedf8df68adfb1afeaf7b7480c0a0243108177b4
    timeout -k 10 600 python3 scripts/ci/draft_convolution_fixture.py --reuse-verified --output "$convolution_fixture"
    cp "$convolution_fixture/manifest.json" "$output/draft-convolution-manifest.json"
fi
if [ "$mode" = learned-attention ]; then
    projection_fixture=/home/thatch/.cache/qwen-experiments/dflash2-attention-dedf8df68adfb1afeaf7b7480c0a0243108177b4
    timeout -k 10 900 python3 scripts/ci/draft_attention_fixture.py --reuse-verified --output "$projection_fixture"
    cp "$projection_fixture/manifest.json" "$output/draft-attention-manifest.json"
    mlp_fixture=/home/thatch/.cache/qwen-experiments/dflash2-mlp-dedf8df68adfb1afeaf7b7480c0a0243108177b4
    timeout -k 10 1200 python3 scripts/ci/draft_mlp_fixture.py --reuse-verified --output "$mlp_fixture"
    cp "$mlp_fixture/manifest.json" "$output/draft-mlp-manifest.json"
fi
if [ "$mode" = learned-convolution ]; then
    projection_fixture=/home/thatch/.cache/qwen-experiments/dflash2-convolution-dedf8df68adfb1afeaf7b7480c0a0243108177b4
    timeout -k 10 600 python3 scripts/ci/draft_convolution_fixture.py --reuse-verified --output "$projection_fixture"
    cp "$projection_fixture/manifest.json" "$output/draft-convolution-manifest.json"
fi
if [ "${QWEN_LEARNED_STACK:-0}" = 1 ]; then
    stack_fixture=/home/thatch/.cache/qwen-experiments/dflash2-stack-dedf8df68adfb1afeaf7b7480c0a0243108177b4
    selector_fixture=/home/thatch/.cache/qwen-experiments/dflash2-selector-dedf8df68adfb1afeaf7b7480c0a0243108177b4
    timeout -k 10 4800 python3 -u scripts/ci/draft_remaining_layers_fixture.py --reuse-verified --output "$stack_fixture"
    timeout -k 10 900 python3 scripts/ci/draft_selector_fixture.py --reuse-verified --output "$selector_fixture"
    for layer in 1 2 3 4; do cp "$stack_fixture/layer-$layer/manifest.json" "$output/draft-layer-$layer-manifest.json"; done
    cp "$selector_fixture/manifest.json" "$output/draft-selector-manifest.json"
fi
if [ "$mode" = feature-projection ]; then
    timeout -k 10 120 python3 scripts/ci/draft_projection_fixture.py --output "$output/draft-projection-fixture"
fi
if [ "$mode" = feature-projection-full ]; then
    projection_fixture=/home/thatch/.cache/qwen-experiments/dflash2-projection-dedf8df68adfb1afeaf7b7480c0a0243108177b4
    timeout -k 10 900 python3 scripts/ci/draft_projection_full_fixture.py --reuse-verified --output "$projection_fixture"
    cp "$projection_fixture/manifest.json" "$output/draft-projection-manifest.json"
fi
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
    --label "thatch.qwen.workflow-run=${GITHUB_RUN_ID:-untracked}" \
    --label "thatch.qwen.source-revision=${GITHUB_SHA:-untracked}" \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/models -e HF_HUB_CACHE=/models/hub -e TT_METAL_HOME=/opt/tt-metal \
    -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels \
    -e MESH_DEVICE=P300 -e VLLM_PLUGINS=tt,tt_model_registry -e VLLM_RPC_TIMEOUT=100000 \
    -e "TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/$descriptor" \
    -e "QWEN_FABRIC_LINK_PROBE=${QWEN_FABRIC_LINK_PROBE:-0}" \
    -e "QWEN_PROJECTION_LINKS=$projection_links" \
    -e "QWEN_CCL_LAZY_BUILD=$ccl_build" \
    -e "QWEN_MTP_DRAFTS=$mtp_drafts" \
    -e "QWEN_DFLASH_DRAFTS=$dflash_drafts" \
    -e "QWEN_DFLASH_CAPTURE=$dflash_capture" \
    -e "QWEN_DFLASH_COMMIT_ABBA=$dflash_commit_abba" \
    -e QWEN36_BATCHED_DECODE_MODE=host -e QWEN36_SHARD_GREEDY=0 \
    -e QWEN_PREFILL_CONTINUATION=0 -e TT_PREFILL_DECODE_INTERLEAVE=0 \
    -e "QWEN_RUN_MODE=$mode" -e "QWEN_INTERLEAVE_RATIO=$ratio" \
    -e "QWEN_CODING_REQUEST=${QWEN_CODING_REQUEST:-0}" \
    -e "QWEN_LOOKUP_CAP_ABBA=${QWEN_LOOKUP_CAP_ABBA:-0}" \
    -e "QWEN_PREFIX_ZERO_REUSE=${QWEN_PREFIX_ZERO_REUSE:-0}" \
    -e "QWEN_LEARNED_STACK=${QWEN_LEARNED_STACK:-0}" \
    -e "QWEN_BOUNDARY_DIAGNOSTICS=${QWEN_BOUNDARY_DIAGNOSTICS:-0}" \
    -e "QWEN_INTERLEAVE_MIXED=${QWEN_INTERLEAVE_MIXED:-0}" \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/baseline-suite.sh)
docker cp scripts "$test_id:/experiment-scripts"
if [ "$dflash_drafts" != 0 ]; then copy_dflash_fixtures; fi
if [ "$ccl_build" = 1 ]; then
    docker cp optimisation/sim/sdpa-graft-registration.patch "$test_id:/tmp/ccl-graft-registration.patch"
fi
if [ "${QWEN_LEARNED_STACK:-0}" = 1 ]; then
    docker cp "$stack_fixture" "$test_id:/experiment-stack-fixture"
    docker cp "$selector_fixture" "$test_id:/experiment-selector-fixture"
fi
if [ "$mode" = learned-attention ]; then
    docker cp "$mlp_fixture" "$test_id:/experiment-mlp-fixture"
fi
if [[ "$mode" = learned-mlp || "$mode" = learned-attention ]]; then
    docker cp "$convolution_fixture" "$test_id:/experiment-convolution-fixture"
fi
if [[ "$mode" = learned-mlp || "$mode" = learned-attention || "$mode" = learned-convolution || "$mode" = feature-projection || "$mode" = feature-projection-full ]]; then
    docker cp "$projection_fixture" "$test_id:/experiment-projection-fixture"
fi
docker cp optimisation "$test_id:/experiment-optimisation"
docker cp speculative-decoding/harness "$test_id:/experiment-speculative"
docker start -a "$test_id" | tee "$output/baseline-console.log"
test "$(docker inspect --format '{{.State.ExitCode}}' "$test_id")" = 0
