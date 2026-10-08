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
#   The image is tagged <image> only after G1 provenance (c2_image_provenance.py) passes; a failed build
#   is removed (C2_KEEP_FAILED_IMAGE=1 tags it <image>-g1-failed instead). Until then it is NOT dangling:
#   the build tags it <image>-unverified itself (--tag), because the rig's other CI runs `docker image
#   prune` on the shared daemon and a dangling image id vanished between the build and provenance
#   ('No such image: sha256:...'). The provisional tag is removed on success and on failure.
#   Optional env: C2_CHECKOUT (a checkout to compare the image's trees with, informational),
#   C2_PROVENANCE_REPORT (where to write the provenance JSON).
#   The graft is K64j (S2, design W9): its _ttnncpp.so must be $graft_sha, and G1 holds every
#   QWEN_ / [QWEN- string of the previous graft (K64i, $previous_graft, itself checked against its
#   MANIFEST.sha256 and $previous_graft_sha first) to be in it too (a graft .so replaces the whole
#   binary: memory graft-so-drops-image-patches). The kernel-cache key below hashes
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
kvread=/home/thatch/opgraft-KVR
kvread_name=opgraft-KVR
kvread_sha=5b2ad8d72bf134f1ef6413994d2b75511779660555cf0251d2b132773dcbdd8a
previous_graft=/home/thatch/opgraft-K64i
previous_graft_sha=cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4
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
mkdir -p "$ctx/fixture" "$ctx/draft-config" "$ctx/fixtures" "$ctx/draft-configs"
# Drafter candidates (docs/tp4-combined-window.md): C2_DRAFTER_CANDIDATES="<id> <id>" stages each pinned id from the cache (drafter-candidates/<id>/fixture and
# config.json) beside the default checkpoint. Both directories always exist (the Dockerfile copies them), holding a placeholder when no candidate is listed; a
# candidate's bytes are checked against scripts/ci/drafter_checkpoints.json at every boot (drafter_checkpoint.attach_check), so nothing is trusted here.
touch "$ctx/fixtures/.placeholder" "$ctx/draft-configs/.placeholder"
for candidate in ${C2_DRAFTER_CANDIDATES:-}; do
  case "$candidate" in
    [a-z0-9]*) ;;
    *) echo "drafter candidate id $candidate is not a plain lowercase id" >&2; exit 2 ;;
  esac
  case "$candidate" in *[!a-z0-9.-]*) echo "drafter candidate id $candidate is not a plain lowercase id" >&2; exit 2 ;; esac
  test -d "$fixtures/drafter-candidates/$candidate/fixture" || { echo "no staged fixture for drafter candidate $candidate" >&2; exit 2; }
  test -s "$fixtures/drafter-candidates/$candidate/config.json" || { echo "no staged config for drafter candidate $candidate" >&2; exit 2; }
  cp -al "$fixtures/drafter-candidates/$candidate/fixture" "$ctx/fixtures/$candidate"
  mkdir -p "$ctx/draft-configs/$candidate"
  cp "$fixtures/drafter-candidates/$candidate/config.json" "$ctx/draft-configs/$candidate/config.json"
done
echo "${C2_DRAFTER_CANDIDATES:-}" > "$ctx/drafter-candidates.txt"
(cd "$graft" && sha256sum -c --quiet MANIFEST.sha256) || { echo "$graft does not verify against its MANIFEST.sha256" >&2; exit 2; }
if [ "$(sha256sum < "$graft/_ttnncpp.so" | cut -c1-64)" != "$graft_sha" ]; then
  echo "$graft/_ttnncpp.so is not the K64j binary $graft_sha" >&2
  exit 2
fi
# G1 holds the graft's strings to the previous graft's, so that graft must be K64i as built: its MANIFEST.sha256
# verifies and its _ttnncpp.so is the v235 gate's binary (a replaced or rebuilt ~/opgraft-K64i would make the
# superset vacuous).
test -f "$previous_graft/_ttnncpp.so" || { echo "$previous_graft/_ttnncpp.so is missing: G1 compares the graft's strings with it" >&2; exit 2; }
(cd "$previous_graft" && sha256sum -c --quiet MANIFEST.sha256) || { echo "$previous_graft does not verify against its MANIFEST.sha256" >&2; exit 2; }
if [ "$(sha256sum < "$previous_graft/_ttnncpp.so" | cut -c1-64)" != "$previous_graft_sha" ]; then
  echo "$previous_graft/_ttnncpp.so is not the K64i binary $previous_graft_sha" >&2
  exit 2
fi
cp -al "$graft" "$ctx/$graft_name"
# The region-read extension (optimisation/ttnn-op/kv_region_read/build_kv_read.sh): replaces no binary, so it needs no strings superset, only its own
# manifest and pin. The prefix audit reads the blocks a request names through it (docs/prefix-audit-cost.md); an image without it audits at 8 minutes a request.
(cd "$kvread" && sha256sum -c --quiet MANIFEST.sha256) || { echo "$kvread does not verify against its MANIFEST.sha256" >&2; exit 2; }
if [ "$(sha256sum < "$kvread/qwen_kv_read.so" | cut -c1-64)" != "$kvread_sha" ]; then
  echo "$kvread/qwen_kv_read.so is not the pinned build $kvread_sha" >&2
  exit 2
fi
cp -al "$kvread" "$ctx/$kvread_name"
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

# Until G1 passes, the built image carries only the provisional tag below and belongs to this script: any failure from here
# on removes it (the rig is the production control plane; a multi-GB image nobody owns is not free). The provisional tag keeps
# it from being dangling, so another CI job's `docker image prune` on the shared daemon cannot delete it mid-provenance.
provisional="$image-unverified"
# A build killed outright (SIGKILL, a cancelled job) never ran the cleanup below and leaves this tag behind; clear it first.
docker rmi "$provisional" >/dev/null 2>&1 || true
iid=$(mktemp)
built=
cleanup() {
  local status=$?
  rm -f "$iid"
  if [ "$status" -ne 0 ] && [ -n "$built" ]; then
    if [ "${C2_KEEP_FAILED_IMAGE:-0}" = 1 ]; then
      docker tag "$built" "$image-g1-failed" && echo "kept the failed build as $image-g1-failed" >&2
    fi
    docker rmi "$provisional" >/dev/null 2>&1 && echo "removed the provisional tag $provisional" >&2 \
      || echo "could not remove the provisional tag $provisional (another tag or container holds it)" >&2
  fi
  return "$status"
}
trap cleanup EXIT
# The eight-seat variant: C2_BAKE_DEFAULT_PROFILE=<profile> bakes that profile as the image's serving default (ENV QWEN_C2_PROFILE) and
# THATCH_SERVING_SESSION_CAP as its max-num-seqs, from the context's own profiles file. Unset, nothing is baked (production's four-seat default).
bake=(--build-arg "C2_BAKE_PROFILE=" --build-arg "C2_BAKE_SESSION_CAP=")
baked_profile=${C2_BAKE_DEFAULT_PROFILE:-}
baked_cap=
if [ -n "$baked_profile" ]; then
  baked_cap=$(python3 -B - "$ctx/overlay/scripts/ci/qwen_c2_profiles.json" "$baked_profile" <<'PY'
import json, sys
entry = json.load(open(sys.argv[1], encoding='utf-8'))['profiles'].get(sys.argv[2])
if entry is None or entry.get('gate_only') is True or entry.get('mesh_device') != 'P150x4':
    sys.exit('C2_BAKE_DEFAULT_PROFILE %s is not a four-card serving profile of this context' % sys.argv[2])
print(entry['engine']['max-num-seqs'])
PY
  )
  test -n "$baked_cap"
  bake=(--build-arg "C2_BAKE_PROFILE=$baked_profile" --build-arg "C2_BAKE_SESSION_CAP=$baked_cap")
  echo "baking the serving default $baked_profile with a session cap of $baked_cap"
fi
DOCKER_BUILDKIT=1 docker build -f "$ctx/Dockerfile" --build-arg "KERNEL_CACHE=$kernel_cache" \
  --build-arg "SOURCE_REVISION=$source_revision" "${bake[@]}" --tag "$provisional" --iidfile "$iid" "$ctx"
built=$(cat "$iid")
# Belt and braces for a builder that ignores --tag: the provisional tag must exist before anything else runs.
docker tag "$built" "$provisional"
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
# The stall watch (stall_watch.py) runs the pinned tt-metal triage tools from INSIDE the serving container, whose root filesystem is
# read-only, so the tools and their ttexalens dependency must already be in the image. Informational (a missing tool never fails the
# build: the watch still dumps every stack), but the log says so, and B4's reading checks the line. ttexalens is asked of the interpreter
# the tools run under (stall_watch.triage_python): the image's isolated /opt/triage-venv when it exists, else this one.
triage_probe='import importlib.util, os, subprocess, sys
root = os.environ.get("QWEN_FAST_TRIAGE_ROOT", "/opt/tt-metal/tools/triage")
tools = ("dump_running_operations", "dump_callstacks", "check_binary_integrity", "check_noc_status", "dump_fast_dispatch", "check_eth_status")
missing = [tool for tool in tools if not (os.path.isfile(os.path.join(root, tool + ".py")) or os.path.isfile(os.path.join(root, "triage.py")))]
venv = "/opt/triage-venv/bin/python3"
if os.path.isfile(venv):
    exalens = subprocess.run([venv, "-c", "import ttexalens"], capture_output=True).returncode == 0
else:
    exalens = importlib.util.find_spec("ttexalens") is not None
print("[TRIAGE-CHECK] root=%s present=%s ttexalens=%s missing=%s" % (root, os.path.isdir(root), exalens, ",".join(missing) or "none"))
sys.exit(0 if os.path.isdir(root) and exalens and not missing else 3)'
if ! docker run --rm --network none --entrypoint python3 "$built" -c "$triage_probe"; then
  echo "[TRIAGE-CHECK] WARNING: the triage tools or ttexalens are not usable in this image: a stall will give stacks but no device triage" >&2
fi
# What was baked is what the image carries (G1 provenance checked the pair; this is the build's own read-back).
got=$(docker image inspect "$built" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(QWEN_C2_PROFILE|THATCH_SERVING_SESSION_CAP)=' | LC_ALL=C sort | tr '\n' ' ')
want="QWEN_C2_PROFILE=$baked_profile THATCH_SERVING_SESSION_CAP=$baked_cap "
if [ "$got" != "$want" ]; then
  echo "the built image carries '$got', the build asked for '$want'" >&2
  exit 2
fi
docker tag "$built" "$image"
docker rmi "$provisional" >/dev/null || echo "could not remove the provisional tag $provisional; $image is tagged" >&2
built=
echo "built $image $(docker image inspect "$image" --format '{{.Id}}')"
rm -rf "$ctx"
