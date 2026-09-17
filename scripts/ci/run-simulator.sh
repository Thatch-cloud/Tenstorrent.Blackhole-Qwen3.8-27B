#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_SIM_ONLY:-0}" = 1
test "${QWEN_LEARNED_STACK:-0}" = 1
case "${QWEN_SPLITK_MAXIMA:-0}" in 0|1) ;; *) exit 2 ;; esac
case "${QWEN_SPLITK_WORKERS:-8}" in 8|16) ;; *) exit 2 ;; esac
if [ "${QWEN_SPLITK_WORKERS:-8}" = 16 ]; then test "${QWEN_SPLITK_MAXIMA:-0}" = 1; fi
if [ "${QWEN_SPLITK_MAXIMA:-0}" = 1 ]; then
    test "${QWEN_SIM_CASE:-stack}" = dspark-splitk
    test "${QWEN_SPLITK_ROW_DIAGNOSTIC:-0}" = 0
fi
case "${QWEN_SPLITK_ROW_DIAGNOSTIC:-0}" in 0|1) ;; *) exit 2 ;; esac
if [ "${QWEN_SPLITK_ROW_DIAGNOSTIC:-0}" = 1 ]; then
    test "${QWEN_SIM_CASE:-stack}" = dspark-splitk
fi
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
case "${QWEN_SIM_CASE:-stack}" in t32-markov-fused|t32-markov|t32-markov-learned|t32-attention|t32-draft-attention|t32-commit|t32-combined|t32-publication|t32-context-attention|ladder-cache|draft-tail|history-append|dspark-ladder-attention|markov-sparse-dot|markov-cache-control|stack|shortlist|fusion-t16|fusion-t16-target|gdn-output-l1|gdn-output-grid|gdn-copy-pairs|gdn-outer-add|gdn-shared-qk|gdn-shared-recurrence|target-t16-attention-8k|dspark-native-8k-attention) ;; *) printf 'Unsupported QWEN_SIM_CASE: %s\n' "${QWEN_SIM_CASE:-stack}" >&2; exit 2 ;; esac
mkdir -p experiment-results
case "${QWEN_T32_FUSED_SCORE:-0}" in 0|1) ;; *) exit 2 ;; esac
if [ "${QWEN_T32_FUSED_SCORE:-0}" = 1 ]; then test "${QWEN_SIM_CASE:-stack}" = t32-combined; fi
results=$(cd experiment-results && pwd -P)
assets=$(mktemp -d "$RUNNER_TEMP/qwen-simulator.XXXXXX")
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
results_gid=$(stat -c %g "$results")
[[ "$results_gid" =~ ^[0-9]+$ ]]
chmod g+rwx "$results"
preflight_name="qwen-sim-preflight-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
preflight_cleanup() {
    status=$?
    trap - EXIT
    timeout -k 1 5 docker inspect --format '{{json .State}}' "$preflight_name" > "$results/preflight-state.json" 2>&1 || true
    timeout -k 1 5 docker logs "$preflight_name" > "$results/preflight-container.log" 2>&1 || true
    if [ "$status" -ne 0 ]; then
        free -m > "$results/preflight-memory.txt" || true
        cat /proc/pressure/cpu /proc/pressure/memory /proc/pressure/io > "$results/preflight-pressure.txt" || true
        timeout -k 1 5 docker ps --format '{{.Names}} {{.Status}}' > "$results/preflight-containers.txt" 2>&1 || true
    fi
    timeout -k 1 10 docker rm -f "$preflight_name" >/dev/null 2>&1 || true
    exit "$status"
}
trap preflight_cleanup EXIT
printf 'preflight_create\n' | tee "$results/preflight-stage.txt"
timeout -k 5 30 docker create --name "$preflight_name" --network none --cap-drop ALL --security-opt no-new-privileges \
    --group-add "$results_gid" \
    --mount "type=bind,src=$results,dst=/experiment/results" --entrypoint /bin/bash "$image" \
    -c 'printf "container_uid=%s\n" "$(id -u)" > /experiment/results/result-write-preflight.txt'
printf 'preflight_start\n' | tee "$results/preflight-stage.txt"
timeout -k 5 30 docker start -a "$preflight_name"
test "$(timeout -k 1 5 docker inspect --format '{{.State.ExitCode}}' "$preflight_name")" = 0
test -r "$results/result-write-preflight.txt"
timeout -k 1 10 docker rm "$preflight_name" >/dev/null
trap - EXIT
printf 'preflight_complete\n' | tee "$results/preflight-stage.txt"
cache=/home/thatch/.cache/qwen-experiments
revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
kinds='attention convolution mlp stack selector'
if [[ "${QWEN_SIM_CASE:-stack}" = history-append || "${QWEN_SIM_CASE:-stack}" = draft-tail || "${QWEN_SIM_CASE:-stack}" = ladder-cache ]]; then kinds=''; fi
if [[ "${QWEN_SIM_CASE:-stack}" = markov-cache-control ]]; then kinds=''; fi
if [[ "${QWEN_SIM_CASE:-stack}" = fusion-t16* ]]; then kinds=mlp; fi
if [[ "${QWEN_SIM_CASE:-stack}" = markov-sparse-dot || "${QWEN_SIM_CASE:-stack}" = gdn-output-* || "${QWEN_SIM_CASE:-stack}" = gdn-copy-pairs || "${QWEN_SIM_CASE:-stack}" = gdn-outer-add || "${QWEN_SIM_CASE:-stack}" = dspark-ladder-attention || "${QWEN_SIM_CASE:-stack}" = dspark-native-8k-attention || "${QWEN_SIM_CASE:-stack}" = target-t16-attention-8k || "${QWEN_SIM_CASE:-stack}" = gdn-shared-recurrence || "${QWEN_SIM_CASE:-stack}" = gdn-shared-qk ]]; then kinds=''; fi
if [[ "${QWEN_SIM_CASE:-stack}" = t32-* ]]; then kinds=''; fi
mounts=()
if [[ "${QWEN_SIM_CASE:-stack}" = t32-markov-learned || "${QWEN_SIM_CASE:-stack}" = t32-combined || "${QWEN_SIM_CASE:-stack}" = t32-publication ]]; then
    checkpoint="$cache/dspark-b9a5dbdf03bc999c6c73c426b19c2d9041cea393/model.safetensors"
    test -f "$checkpoint"
    mounts+=(--mount "type=bind,src=$checkpoint,dst=/dspark-model.safetensors,readonly")
fi
if [[ "${QWEN_SIM_CASE:-stack}" = t32-combined || "${QWEN_SIM_CASE:-stack}" = t32-publication ]]; then
    config="$cache/dspark-b9a5dbdf03bc999c6c73c426b19c2d9041cea393/config.json"
    target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
    test -f "$config"
    test -d "$target/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    mounts+=(--mount "type=bind,src=$config,dst=/dspark-config.json,readonly")
    mounts+=(--mount "type=bind,src=$target,dst=/target,readonly")
fi
for kind in $kinds; do
    test -d "$cache/dflash2-$kind-$revision"
    mounts+=(--mount "type=bind,src=$cache/dflash2-$kind-$revision,dst=/fixture-$kind,readonly")
done
fetch_simulator_asset() {
    local url=$1 digest=$2 destination=$3 cached temporary
    mkdir -p "$cache/simulator-assets"
    cached="$cache/simulator-assets/$digest"
    if [ -f "$cached" ] && printf '%s  %s\n' "$digest" "$cached" | sha256sum -c - >/dev/null 2>&1; then
        cp -- "$cached" "$destination"
        return
    fi
    temporary=$(mktemp "$cache/simulator-assets/download.XXXXXX")
    if ! curl --fail --location --connect-timeout 10 --max-time 30 --retry 2 --retry-max-time 65 \
            "$url" -o "$temporary"; then
        rm -f -- "$temporary"
        return 1
    fi
    if ! printf '%s  %s\n' "$digest" "$temporary" | sha256sum -c -; then
        rm -f -- "$temporary"
        return 1
    fi
    chmod 0644 "$temporary"
    mv -- "$temporary" "$cached"
    cp -- "$cached" "$destination"
}
fetch_simulator_asset https://github.com/tenstorrent/ttsim/releases/download/v1.10.3/libttsim_bh_x2.so \
    79287bd7cc1fc0fab28ca7b82567c39311f0dcc6ec2704ab7c4386dfc71abfd4 "$assets/libttsim_bh_x2.so"
fetch_simulator_asset https://raw.githubusercontent.com/tenstorrent/tt-umd/115b809170ff762182f925a07636887e5afb910e/tests/cluster_descriptor_examples/blackhole_P300_both_mmio.yaml \
    27b7ec074f81fe4b4a1be89a5c39fd6c7eaf2c2d68da8f49bd8faf02a7d3df15 "$assets/cluster_descriptor.yaml"
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
    -e "QWEN_T32_FUSED_SCORE=${QWEN_T32_FUSED_SCORE:-0}" \
    -e "QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}" \
    -e "QWEN_SIM_CONTEXT=${QWEN_SIM_CONTEXT:-2048}" \
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
    -e "QWEN_SPLITK_MAXIMA=${QWEN_SPLITK_MAXIMA:-0}" \
    -e "QWEN_LADDER_CONTEXT=${QWEN_LADDER_CONTEXT:-}" \
    -e "QWEN_SPLITK_WORKERS=${QWEN_SPLITK_WORKERS:-8}" \
    -e "QWEN_SPLITK_ROW_DIAGNOSTIC=${QWEN_SPLITK_ROW_DIAGNOSTIC:-0}" \
    -e "QWEN_CCL_LAZY_BUILD=${QWEN_CCL_LAZY_BUILD:-0}" \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/simulator-suite.sh)
docker cp scripts "$container:/experiment-scripts"
if [ "${QWEN_SIM_CASE:-stack}" = t32-commit ]; then
    docker cp optimisation "$container:/optimisation"
fi
if [[ "${QWEN_SIM_CASE:-stack}" = t32-*attention || "${QWEN_SIM_CASE:-stack}" = t32-combined || "${QWEN_SIM_CASE:-stack}" = t32-publication || "${QWEN_SIM_CASE:-stack}" = ladder-cache || "${QWEN_SIM_CASE:-stack}" = draft-tail || "${QWEN_SIM_CASE:-stack}" = fusion-t16* || "${QWEN_SIM_CASE:-stack}" = markov-sparse-dot || "${QWEN_SIM_CASE:-stack}" = gdn-output-* || "${QWEN_SIM_CASE:-stack}" = gdn-copy-pairs || "${QWEN_SIM_CASE:-stack}" = gdn-outer-add || "${QWEN_SIM_CASE:-stack}" = dspark-ladder-attention || "${QWEN_SIM_CASE:-stack}" = dspark-native-8k-attention || "${QWEN_SIM_CASE:-stack}" = target-t16-attention-8k || "${QWEN_SIM_CASE:-stack}" = gdn-shared-recurrence || "${QWEN_SIM_CASE:-stack}" = gdn-shared-qk ]]; then
    docker cp optimisation/sim "$container:/simulator-support"
fi
if [ "${QWEN_CCL_LAZY_BUILD:-0}" = 1 ]; then
    docker cp optimisation/sim/sdpa-graft-registration.patch "$container:/tmp/ccl-graft-registration.patch"
fi
docker start -a "$container" 2>&1 | tee experiment-results/simulator-container.log
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
