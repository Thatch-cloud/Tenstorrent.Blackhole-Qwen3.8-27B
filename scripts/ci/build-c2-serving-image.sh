#!/usr/bin/env bash
# Build the C2 serving base image on the rig (docker/qwen-c2-serving.Dockerfile).
# Usage: build-c2-serving-image.sh <context.tgz> <tag>
#   context.tgz holds Dockerfile, ci/, graft/, graft.sha256, source.sha256 (from the repo and
#   the v235 graft artifact); this script adds what only the rig has: the K64i op graft, the
#   DFlash2 fixtures and draft config, and a persistent copy of the gate's cache volume
#   under the platform's /models mount (~/hf-cache/hub/.qwen-c2).
set -euo pipefail
context_tgz=$1
tag=$2
image=zot.thatch.local:5000/tt-vllm:qwen38-c2-$tag
revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
graft=/home/thatch/opgraft-K64i
fixtures=/home/thatch/.cache/qwen-experiments
ctx=/home/thatch/c2-serving-ctx

rm -rf "$ctx"
mkdir -p "$ctx/fixture" "$ctx/draft-config"
tar -xzf "$context_tgz" -C "$ctx"
cp -al "$graft" "$ctx/opgraft-K64i"
for component in attention convolution mlp projection selector; do
  cp -al "$fixtures/dflash2-$component-$revision" "$ctx/fixture/$component"
done
for layer in 1 2 3 4; do
  cp -al "$fixtures/dflash2-stack-$revision/layer-$layer" "$ctx/fixture/layer-$layer"
done
curl -fsSL --max-time 30 "https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/$revision/config.json" \
  > "$ctx/draft-config/config.json"

# The gate arm's kernel cache key (lever_n_m3native_run_arm.sh), so the served process reads
# the same warm JIT cache the gate built.
kernels="$graft/sdpa_decode/device/kernels"
others=$(cd "$kernels" && find . -type f -name '*qwen*.cpp' ! -path ./dataflow/reader_decode_qwen.cpp \
  ! -path ./compute/sdpa_flash_decode_qwen.cpp | LC_ALL=C sort | while IFS= read -r file; do
    printf '%s %s\n' "$(sha256sum < "$file" | cut -c1-64)" "$file"
  done)
key=$({ cat "$kernels/dataflow/reader_decode_qwen.cpp" "$kernels/compute/sdpa_flash_decode_qwen.cpp"; \
  printf '%s' "$others"; } | sha256sum | cut -c1-12)
pf=$(cd "$graft/sdpa" && find . -type f | LC_ALL=C sort | while IFS= read -r file; do
    printf '%s %s\n' "$(sha256sum < "$file" | cut -c1-64)" "$file"
  done | sha256sum | cut -c1-12)
kernel_cache=/experiment-cache/kernels-qwen-$key-pf-$pf
echo "kernel cache: $kernel_cache"

persistent=/home/thatch/hf-cache/hub/.qwen-c2
if [ ! -d "$persistent" ]; then
  volume=$(docker volume inspect qwen-experiments-f1e9b1a64b4f --format '{{.Mountpoint}}')
  echo "copying the gate cache volume $volume -> $persistent"
  sudo -n cp -a "$volume" "$persistent.partial"
  sudo -n mv "$persistent.partial" "$persistent"
fi
sudo -n du -sh "$persistent" || true
sudo -n test -d "$persistent/${kernel_cache#/experiment-cache/}" && echo "warm kernel cache present" \
  || echo "kernel cache ${kernel_cache#/experiment-cache/} absent: the first start compiles it"

DOCKER_BUILDKIT=1 docker build -f "$ctx/Dockerfile" --build-arg "KERNEL_CACHE=$kernel_cache" -t "$image" "$ctx"
echo "built $image $(docker image inspect "$image" --format '{{.Id}}')"

# Boot check: the contract rewrites a platform-shaped argv and names the profile.
docker run --rm --network none -v /home/thatch/hf-cache/hub:/models:ro --entrypoint python3 "$image" \
  -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
  --port 8001 --max-model-len 65536 --help 2>&1 >/dev/null | grep -F '[QWEN-C2]'
rm -rf "$ctx"
