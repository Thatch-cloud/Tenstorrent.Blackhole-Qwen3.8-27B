#!/usr/bin/env bash
# Build the two-card image chain from this repository, with no internal registry and no prebuilt binary.
#
#   scripts/two-card-build.sh [--tier ops|k64j|both] [--to STAGE] [--from STAGE] [--jobs N]
#
# Stages (each FROM the previous; --tier chooses what stage 5/6 build):
#   1 bringup   docker/tenstorrent-bringup.Dockerfile              tt-bringup:v0.77.0-rc1
#   2 prstack   docker/two-card/10-tt-prstack.Dockerfile           tt-bringup:v0.77.0-rc1-prstack   (vendored PR diffs)
#   3 serving   docker/tenstorrent-serving.Dockerfile              tt-serving:v0.77.0-rc1-prstack
#   4 plugin    docker/tenstorrent-vllm-plugin.Dockerfile          tt-vllm:v0.77.0-rc1-prstack-plugin
#   5 ops       docker/two-card/40-qwen-ops.Dockerfile             tt-vllm:qwen38-k                 (tier ops)
#   6 k64j      docker/two-card/50-qwen-k64j.Dockerfile            tt-vllm:qwen38-k64j              (tier k64j)
#
# Environment: REGISTRY (default localhost:5000; only a name prefix, nothing has to listen there), TAG
# (default v0.77.0-rc1), JOBS (default 48). Needs docker with BuildKit (the default `docker build`).
# Do not use a buildx container driver: it would try to pull the local base images.
#
# At the end it prints RUNTIME_BINARY_SHA256 for the tier(s) built: the hash of YOUR _ttnncpp.so, which the
# serving Dockerfiles take as a build argument (the maintainers' hash is not published and would not match).
set -euo pipefail

REGISTRY=${REGISTRY:-localhost:5000}
TAG=${TAG:-v0.77.0-rc1}
JOBS=${JOBS:-48}
TIER=k64j
FROM_N=1
TO_N=6

stage_n() { case "$1" in 1|bringup) echo 1 ;; 2|prstack) echo 2 ;; 3|serving) echo 3 ;; 4|plugin) echo 4 ;;
                          5|ops) echo 5 ;; 6|k64j) echo 6 ;; *) echo "unknown stage: $1" >&2; exit 2 ;; esac; }

while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --from) FROM_N=$(stage_n "$2"); shift 2 ;;
    --to)   TO_N=$(stage_n "$2"); shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$TIER" in ops|k64j|both) ;; *) echo "--tier takes ops, k64j or both" >&2; exit 2 ;; esac

[ -f docker/two-card/10-tt-prstack.Dockerfile ] || { echo "run me from the repository root" >&2; exit 1; }

want() { [ "$1" -ge "$FROM_N" ] && [ "$1" -le "$TO_N" ]; }
BRINGUP="${REGISTRY}/tt-bringup:${TAG}"
PRSTACK="${BRINGUP}-prstack"
SERVING="${REGISTRY}/tt-serving:${TAG}-prstack"
PLUGIN="${REGISTRY}/tt-vllm:${TAG}-prstack-plugin"

empty=$(mktemp -d); bundle_ctx=$(mktemp -d)
trap 'rm -rf "$empty" "$bundle_ctx"' EXIT
cp -a tt-metal-custom-ops "$bundle_ctx/"

build() { echo "=== $1 -> $2"; shift 2; docker build "$@"; }

if want 1; then build "1 bringup" "$BRINGUP" -f docker/tenstorrent-bringup.Dockerfile \
    --build-arg TT_METAL_REF="$TAG" --build-arg JOBS="$JOBS" -t "$BRINGUP" "$empty"; fi
if want 2; then build "2 prstack" "$PRSTACK" -f docker/two-card/10-tt-prstack.Dockerfile \
    --build-arg BASE="$BRINGUP" --build-arg JOBS="$JOBS" -t "$PRSTACK" "$bundle_ctx"; fi
if want 3; then build "3 serving" "$SERVING" -f docker/tenstorrent-serving.Dockerfile \
    --build-arg BASE="$PRSTACK" -t "$SERVING" "$empty"; fi
if want 4; then build "4 plugin" "$PLUGIN" -f docker/tenstorrent-vllm-plugin.Dockerfile \
    --build-arg BASE="$SERVING" -t "$PLUGIN" "$empty"; fi
if want 5 && [ "$TIER" != k64j ]; then build "5 ops" "${REGISTRY}/tt-vllm:qwen38-k" -f docker/two-card/40-qwen-ops.Dockerfile \
    --build-arg BASE="$PLUGIN" --build-arg JOBS="$JOBS" -t "${REGISTRY}/tt-vllm:qwen38-k" "$bundle_ctx"; fi
if want 6 && [ "$TIER" != ops ]; then build "6 k64j" "${REGISTRY}/tt-vllm:qwen38-k64j" -f docker/two-card/50-qwen-k64j.Dockerfile \
    --build-arg BASE="$PLUGIN" --build-arg JOBS="$JOBS" -t "${REGISTRY}/tt-vllm:qwen38-k64j" "$bundle_ctx"; fi

for img in "${REGISTRY}/tt-vllm:qwen38-k" "${REGISTRY}/tt-vllm:qwen38-k64j"; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    sum=$(docker run --rm --entrypoint /bin/sh "$img" -c 'sha256sum < /opt/tt-metal/build_Release/lib/_ttnncpp.so | cut -c1-64')
    echo "$img  RUNTIME_BINARY_SHA256=$sum"
  fi
done
