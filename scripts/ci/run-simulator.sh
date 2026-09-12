#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_SIM_ONLY:-0}" = 1
test "${QWEN_LEARNED_STACK:-0}" = 1
case "${QWEN_SIM_CASE:-stack}" in stack|shortlist|fusion-t16|fusion-t16-target|t32-markov|t32-markov-learned|t32-attention|t32-draft-attention|t32-commit|t32-combined|t32-publication) ;; *) exit 2 ;; esac
mkdir -p experiment-results
assets=$(mktemp -d "$RUNNER_TEMP/qwen-simulator.XXXXXX")
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
cache=/home/thatch/.cache/qwen-experiments
revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
kinds='attention convolution mlp stack selector'
if [[ "${QWEN_SIM_CASE:-stack}" = fusion-t16* ]]; then kinds=mlp; fi
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
        docker logs "$container" > experiment-results/simulator-container.log 2>&1 || true
        docker cp "$container:/experiment/results/." experiment-results/ || true
        docker rm -f "$container" >/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
container=$(docker create --network none --cap-drop ALL --security-opt no-new-privileges \
    --pids-limit 4096 --memory 64g --cpus 16 --shm-size 8g \
    --mount "type=bind,src=$assets,dst=/simulator-assets,readonly" \
    "${mounts[@]}" \
    -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e QWEN_SIM_ONLY=1 \
    -e "QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}" \
    -e "QWEN_CCL_LAZY_BUILD=${QWEN_CCL_LAZY_BUILD:-0}" \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/simulator-suite.sh)
docker cp scripts "$container:/experiment-scripts"
if [ "${QWEN_SIM_CASE:-stack}" = t32-commit ]; then
    docker cp optimisation "$container:/optimisation"
fi
if [[ "${QWEN_SIM_CASE:-stack}" = fusion-t16* || "${QWEN_SIM_CASE:-stack}" = t32-*attention || "${QWEN_SIM_CASE:-stack}" = t32-combined || "${QWEN_SIM_CASE:-stack}" = t32-publication ]]; then
    docker cp optimisation/sim "$container:/simulator-support"
fi
if [ "${QWEN_CCL_LAZY_BUILD:-0}" = 1 ]; then
    docker cp optimisation/sim/sdpa-graft-registration.patch "$container:/tmp/ccl-graft-registration.patch"
fi
docker start -a "$container"
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
