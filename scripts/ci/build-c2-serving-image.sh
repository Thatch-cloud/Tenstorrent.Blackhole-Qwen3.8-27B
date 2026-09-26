#!/usr/bin/env bash
# Build the C2 serving base image on the rig (docker/qwen-c2-serving.Dockerfile).
# Usage: build-c2-serving-image.sh <context.tgz> <tag>
#   context.tgz is what `python3 scripts/ci/c2_overlay.py stage --repo . --out <dir>` stages
#   (tar -czf <tgz> -C <dir> .): the Dockerfile, the overlay manifest (docker/qwen-c2-overlay.txt)
#   and overlay/ tree it names, the overlay and provenance tools, THIS script, the v235 model graft
#   (graft/, graft.sha256, source.sha256), the v235 environment, source-revision and layers.json.
#   Run the context's own copy of this script, from outside $ctx (below):
#     tar -xzf <tgz> -O ./build-c2-serving-image.sh > /tmp/build-c2.sh && bash /tmp/build-c2.sh <tgz> <tag>
#   An older copy writes no build-stamp, so the Dockerfile's last COPY fails its build; a copy
#   that is not the context's is refused below, and G1 checks the stamp again.
#   This script checks the context against the manifest and adds what only the rig has: the K64j
#   op graft (checked against its MANIFEST.sha256 and its _ttnncpp.so pin first), the DFlash2 fixtures
#   and draft config, and a persistent copy of the gate's cache volume under the platform's /models
#   mount (~/hf-cache/hub/.qwen-c2).
#   The image is tagged only after G1 provenance (c2_image_provenance.py) passes; a failed build
#   is removed (C2_KEEP_FAILED_IMAGE=1 tags it <image>-g1-failed instead).
#   Optional env: C2_CHECKOUT (a checkout to compare the image's trees with, informational),
#   C2_PROVENANCE_REPORT (where to write the provenance JSON).
#   The graft is K64j (S2, design W9): its _ttnncpp.so must be $graft_sha, and G1 holds every
#   QWEN_ / [QWEN- string of the previous graft (K64i, $previous_graft) to be in it too (a graft .so
#   replaces the whole binary: memory graft-so-drops-image-patches). The kernel-cache key below hashes
#   the graft's own *qwen*.cpp, so K64j's first start compiles every kernel (TT_METAL_CACHE is that
#   whole directory); never seed it from K64i's.
set -euo pipefail
context_tgz=$1
tag=$2
image=zot.thatch.local:5000/tt-vllm:qwen38-c2-$tag
revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
graft=/home/thatch/opgraft-K64j
graft_name=opgraft-K64j
graft_sha=152951c1c0de5c9dfad2d62c295393a43b2ecf353965c55c709da7e539b975b7
previous_graft=/home/thatch/opgraft-K64i
fixtures=/home/thatch/.cache/qwen-experiments
models=/home/thatch/hf-cache/hub
ctx=/home/thatch/c2-serving-ctx

self=$(readlink -f "${BASH_SOURCE[0]}")
case "$self" in
  "$ctx"/*) echo "run a copy of this script from outside $ctx: it deletes $ctx first" >&2; exit 2 ;;
esac
stamp=$(sha256sum < "$self" | cut -c1-64)

rm -rf "$ctx"
mkdir -p "$ctx"
tar -xzf "$context_tgz" -C "$ctx"
python3 -B "$ctx/c2_overlay.py" check --context "$ctx"
if [ "$stamp" != "$(sha256sum < "$ctx/build-c2-serving-image.sh" | cut -c1-64)" ]; then
  echo "this script ($self) is not the context's build-c2-serving-image.sh: run that one" >&2
  exit 2
fi
printf '%s\n' "$stamp" > "$ctx/build-stamp"
source_revision=$(cat "$ctx/source-revision")
echo "context staged from $source_revision"
mkdir -p "$ctx/fixture" "$ctx/draft-config"
(cd "$graft" && sha256sum -c --quiet MANIFEST.sha256) || { echo "$graft does not verify against its MANIFEST.sha256" >&2; exit 2; }
if [ "$(sha256sum < "$graft/_ttnncpp.so" | cut -c1-64)" != "$graft_sha" ]; then
  echo "$graft/_ttnncpp.so is not the K64j binary $graft_sha" >&2
  exit 2
fi
test -f "$previous_graft/_ttnncpp.so" || { echo "$previous_graft/_ttnncpp.so is missing: G1 compares the graft's strings with it" >&2; exit 2; }
cp -al "$graft" "$ctx/$graft_name"
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

# Until G1 passes, the built image is untagged and belongs to this script: any failure from here
# on removes it (the rig is the production control plane; a dangling multi-GB image is not free).
iid=$(mktemp)
built=
cleanup() {
  local status=$?
  rm -f "$iid"
  if [ "$status" -ne 0 ] && [ -n "$built" ]; then
    if [ "${C2_KEEP_FAILED_IMAGE:-0}" = 1 ]; then
      docker tag "$built" "$image-g1-failed" && echo "kept the failed build as $image-g1-failed" >&2
    else
      docker rmi "$built" >/dev/null 2>&1 && echo "removed the failed build $built" >&2 \
        || echo "could not remove the failed build $built (another tag or container holds it)" >&2
    fi
  fi
  return "$status"
}
trap cleanup EXIT
DOCKER_BUILDKIT=1 docker build -f "$ctx/Dockerfile" --build-arg "KERNEL_CACHE=$kernel_cache" \
  --build-arg "SOURCE_REVISION=$source_revision" --iidfile "$iid" "$ctx"
built=$(cat "$iid")
echo "built $built: G1 provenance before it is tagged $image"

# G1 provenance: (a) the installed binaries and op directories are the K64j graft's, no QWEN_
# flag the image sets lost its patch, and every QWEN_ / [QWEN- string of the previous graft is still
# there (plus K64j's own literals), (b) every overlaid file's sha256 in the image is its source's
# and the image names the staged commit and this script, (c) the boot argv of every profile is the
# contract's and exact's environment is the v235 gate's, (d) the image's trees match the layer
# model, (e) every frozen-recipe pin holds. Any problem exits non-zero here, so the image never
# gets its tag.
provenance=(--image "$built" --context "$ctx" --models "$models" --previous-graft "$previous_graft")
if [ -n "${C2_CHECKOUT:-}" ]; then provenance+=(--checkout "$C2_CHECKOUT"); fi
if [ -n "${C2_PROVENANCE_REPORT:-}" ]; then provenance+=(--report "$C2_PROVENANCE_REPORT"); fi
python3 -B "$ctx/c2_image_provenance.py" "${provenance[@]}"
docker tag "$built" "$image"
built=
echo "built $image $(docker image inspect "$image" --format '{{.Id}}')"
rm -rf "$ctx"
