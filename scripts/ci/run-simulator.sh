#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_SIM_ONLY:-0}" = 1
test "${QWEN_LEARNED_STACK:-0}" = 1
score_bitwise=0
target_64k=0
if [ "${QWEN_SIM_CASE:-stack}" = target-t16-attention-64k ]; then
    target_64k=1
    export QWEN_SIM_CASE=target-t16-attention-8k
fi
score_sfpu=0
sum_sfpu=0
mask_bits=0
attention_boundary=0
direct_fp32_stage=0
normalization_direct_stage=0
center_tile_fill=0
splitk_attention=0
if [ "${QWEN_SIM_CASE:-stack}" = dspark-splitk ]; then
    splitk_attention=1
    export QWEN_SIM_CASE=dspark-center-tile-fill
fi
if [ "${QWEN_SIM_CASE:-stack}" = dspark-center-tile-fill ]; then
    center_tile_fill=1
    export QWEN_SIM_CASE=dspark-normalization-direct-stage
fi
if [ "${QWEN_SIM_CASE:-stack}" = dspark-normalization-direct-stage ]; then
    normalization_direct_stage=1
    export QWEN_SIM_CASE=dspark-direct-fp32-stage
fi
if [ "${QWEN_SIM_CASE:-stack}" = dspark-direct-fp32-stage ]; then
    direct_fp32_stage=1
    export QWEN_SIM_CASE=dspark-attention-boundary
fi
if [ "${QWEN_SIM_CASE:-stack}" = dspark-attention-boundary ]; then
    attention_boundary=1
    export QWEN_SIM_CASE=dspark-mask-bits
fi
if [ "${QWEN_SIM_CASE:-stack}" = dspark-mask-bits ]; then
    mask_bits=1
    export QWEN_SIM_CASE=dspark-sum-sfpu
fi
if [ "${QWEN_SIM_CASE:-stack}" = dspark-sum-sfpu ]; then
    sum_sfpu=1
    export QWEN_SIM_CASE=dspark-score-sfpu
fi
if [ "${QWEN_SIM_CASE:-stack}" = dspark-score-sfpu ]; then
    score_sfpu=1
    export QWEN_SIM_CASE=dspark-score-bitwise
fi
if [ "${QWEN_SIM_CASE:-stack}" = dspark-score-bitwise ]; then
    score_bitwise=1
    export QWEN_SIM_CASE=dspark-ladder-attention
fi
case "${QWEN_SIM_CASE:-stack}" in dspark-ladder-attention|markov-sparse-dot|markov-cache-control|stack|shortlist|fusion-t16|fusion-t16-target|gdn-output-l1|gdn-output-grid|gdn-copy-pairs|gdn-outer-add|gdn-shared-qk|gdn-shared-recurrence|target-t16-attention-8k|dspark-native-8k-attention) ;; *) exit 2 ;; esac
mkdir -p experiment-results
results=$(cd experiment-results && pwd -P)
assets=$(mktemp -d "$RUNNER_TEMP/qwen-simulator.XXXXXX")
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
results_gid=$(stat -c %g "$results")
[[ "$results_gid" =~ ^[0-9]+$ ]]
chmod g+rwx "$results"
timeout -k 5 30 docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
    --group-add "$results_gid" \
    --mount "type=bind,src=$results,dst=/experiment/results" --entrypoint /bin/bash "$image" \
    -c 'printf "container_uid=%s\n" "$(id -u)" > /experiment/results/result-write-preflight.txt'
test -r "$results/result-write-preflight.txt"
cache=/home/thatch/.cache/qwen-experiments
revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
kinds='attention convolution mlp stack selector'
if [[ "${QWEN_SIM_CASE:-stack}" = markov-cache-control ]]; then kinds=''; fi
if [[ "${QWEN_SIM_CASE:-stack}" = fusion-t16* ]]; then kinds=mlp; fi
if [[ "${QWEN_SIM_CASE:-stack}" = markov-sparse-dot || "${QWEN_SIM_CASE:-stack}" = gdn-output-* || "${QWEN_SIM_CASE:-stack}" = gdn-copy-pairs || "${QWEN_SIM_CASE:-stack}" = gdn-outer-add || "${QWEN_SIM_CASE:-stack}" = dspark-ladder-attention || "${QWEN_SIM_CASE:-stack}" = dspark-native-8k-attention || "${QWEN_SIM_CASE:-stack}" = target-t16-attention-8k || "${QWEN_SIM_CASE:-stack}" = gdn-shared-recurrence || "${QWEN_SIM_CASE:-stack}" = gdn-shared-qk ]]; then kinds=''; fi
mounts=()
for kind in $kinds; do
    test -d "$cache/dflash2-$kind-$revision"
    mounts+=(--mount "type=bind,src=$cache/dflash2-$kind-$revision,dst=/fixture-$kind,readonly")
done
curl --fail --location --max-time 180 https://github.com/tenstorrent/ttsim/releases/download/v1.10.3/libttsim_bh_x2.so -o "$assets/libttsim_bh_x2.so"
curl --fail --location --max-time 180 https://raw.githubusercontent.com/tenstorrent/tt-umd/115b809170ff762182f925a07636887e5afb910e/tests/cluster_descriptor_examples/blackhole_P300_both_mmio.yaml -o "$assets/cluster_descriptor.yaml"
printf '%s\n' "79287bd7cc1fc0fab28ca7b82567c39311f0dcc6ec2704ab7c4386dfc71abfd4  $assets/libttsim_bh_x2.so" \
    "27b7ec074f81fe4b4a1be89a5c39fd6c7eaf2c2d68da8f49bd8faf02a7d3df15  $assets/cluster_descriptor.yaml" | sha256sum -c -
sha256sum "$assets/"* > experiment-results/simulator-assets.sha256
chmod 0755 "$assets"
chmod 0644 "$assets/libttsim_bh_x2.so" "$assets/cluster_descriptor.yaml"
container=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$container" ]; then
        if [[ "$score_bitwise" = 1 || "$target_64k" = 1 ]] && [ "$status" != 0 ]; then
            timeout -k 1 5 docker kill "$container" >/dev/null 2>&1 || true
        fi
        timeout -k 5 20 docker logs "$container" > experiment-results/simulator-final-container.log 2>&1 || true
        timeout -k 5 20 docker rm -f "$container" >/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
memory_options=()
build_cache_mount=()
if [ "$score_bitwise" = 1 ]; then
    volume=qwen-simulator-factory-f1e9b1a64b4f
    if ! docker volume inspect "$volume" >/dev/null 2>&1; then
        docker volume create --label thatch.qwen.simulator-cache=true "$volume" >/dev/null
    fi
    test "$(docker volume inspect --format '{{index .Labels "thatch.qwen.simulator-cache"}}' "$volume")" = true
    build_cache_mount=(--mount "type=volume,src=$volume,dst=/simulator-build-cache")
fi
if [[ "${QWEN_SIM_CASE:-stack}" = dspark-ladder-attention || "${QWEN_SIM_CASE:-stack}" = dspark-native-8k-attention ]]; then memory_options=(--memory-swap 64g); fi
container=$(docker create --network none --cap-drop ALL --security-opt no-new-privileges \
    --group-add "$results_gid" \
    --pids-limit 4096 --memory 64g --cpus 16 --shm-size 8g \
    "${memory_options[@]}" \
    "${build_cache_mount[@]}" \
    --mount "type=bind,src=$assets,dst=/simulator-assets,readonly" \
    --mount "type=bind,src=$results,dst=/experiment/results" \
    "${mounts[@]}" \
    -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e QWEN_SIM_ONLY=1 \
    -e "QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}" \
    -e "QWEN_SCORE_BITWISE=$score_bitwise" \
    -e "QWEN_TARGET_T16_64K=$target_64k" \
    -e "QWEN_SCORE_SFPU=$score_sfpu" \
    -e "QWEN_SUM_SFPU=$sum_sfpu" \
    -e "QWEN_MASK_BITS=$mask_bits" \
    -e "QWEN_ATTENTION_BOUNDARY=$attention_boundary" \
    -e "QWEN_DIRECT_FP32_STAGE=$direct_fp32_stage" \
    -e "QWEN_NORMALIZATION_DIRECT_STAGE=$normalization_direct_stage" \
    -e "QWEN_CENTER_TILE_FILL=$center_tile_fill" \
    -e "QWEN_SPLITK_ATTENTION=$splitk_attention" \
    -e "QWEN_CCL_LAZY_BUILD=${QWEN_CCL_LAZY_BUILD:-0}" \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/simulator-suite.sh)
docker cp scripts "$container:/experiment-scripts"
if [[ "${QWEN_SIM_CASE:-stack}" = fusion-t16* || "${QWEN_SIM_CASE:-stack}" = markov-sparse-dot || "${QWEN_SIM_CASE:-stack}" = gdn-output-* || "${QWEN_SIM_CASE:-stack}" = gdn-copy-pairs || "${QWEN_SIM_CASE:-stack}" = gdn-outer-add || "${QWEN_SIM_CASE:-stack}" = dspark-ladder-attention || "${QWEN_SIM_CASE:-stack}" = dspark-native-8k-attention || "${QWEN_SIM_CASE:-stack}" = target-t16-attention-8k || "${QWEN_SIM_CASE:-stack}" = gdn-shared-recurrence || "${QWEN_SIM_CASE:-stack}" = gdn-shared-qk ]]; then
    docker cp optimisation/sim "$container:/simulator-support"
fi
if [ "${QWEN_CCL_LAZY_BUILD:-0}" = 1 ]; then
    docker cp optimisation/sim/sdpa-graft-registration.patch "$container:/tmp/ccl-graft-registration.patch"
fi
docker start -a "$container" 2>&1 | tee experiment-results/simulator-container.log
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
