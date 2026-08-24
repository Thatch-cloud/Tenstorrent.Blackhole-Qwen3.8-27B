#!/usr/bin/env bash
# Build the Tenstorrent image chain on the host with the cards, from the repo's own
# Dockerfiles.
#
# WHY THIS EXISTS
#
# The chain was previously built by pasting a serving Dockerfile inline into an
# ssh heredoc. That copy silently dropped
#     RUN --mount=type=cache,target=/root/.cache/pip
# from docker/tenstorrent-serving.Dockerfile, so every rebuild re-downloaded the
# whole of requirements-dev.txt (torch, transformers, and ~200 more) instead of
# hitting the cache — minutes of avoidable network on each iteration, and a
# divergence between what the repo says the image is and what was actually built.
#
# Both files already take BASE as a build arg precisely so they can be layered
# without duplication. Use them. Do not inline Dockerfiles.
#
# USAGE (run from the repo root, on the host with the cards)
#   scripts/tt-build-images.sh                        # plain chain at v0.77.0-rc1
#   scripts/tt-build-images.sh --prstack              # + the upstream TP=2 fix stack
#   scripts/tt-build-images.sh --prstack --vllm       # full chain
#
# STAGE SELECTION — build only what you need.
#   --from serving      skip bring-up; start at the serving layer
#   --from vllm         skip both; build only the vLLM layer
#
#   # e.g. add vLLM on top of an already-built prstack serving image (~2 min):
#   scripts/tt-build-images.sh --prstack --vllm --from vllm
#
# Docker's layer cache makes a full re-run cheap when nothing changed (none of
# these Dockerfiles COPY from the context, so context contents do not invalidate
# it). --from exists for when you do NOT want to risk it: a cache miss on the
# bring-up layer means a multi-hour tt-metal compile.
#
# The chain, and what each layer is for:
#   tt-bringup    tt-metal + ttnn + test_system_health     diagnostics
#   tt-serving    + torch/transformers/pytest              demos and PCC
#   tt-vllm       + stock vLLM + the TT plugin              an actual endpoint
#
# The vLLM layer is built from tenstorrent/vllm-tt-plugin (stock vLLM + a
# plugin-only repo), which is where Tenstorrent are moving. The old fork-based
# docker/tenstorrent-vllm.Dockerfile has been removed; see
# https://github.com/tenstorrent/vllm/issues/473 for the maintainer's guidance.
set -euo pipefail

REG=${REG:-localhost:5000}
TAG=${TAG:-v0.77.0-rc1}
JOBS=${JOBS:-48}
PRSTACK=0
VLLM=0
FROM_STAGE=bringup

while [ $# -gt 0 ]; do
  case "$1" in
    --prstack) PRSTACK=1 ;;
    --vllm)    VLLM=1 ;;
    --from)
      shift
      case "${1:-}" in
        bringup|serving|vllm) FROM_STAGE="$1" ;;
        *) echo "--from takes: bringup | serving | vllm" >&2; exit 2 ;;
      esac ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# Numeric ordering so "should I run stage X" is a comparison, not a nest of ifs.
stage_num() { case "$1" in bringup) echo 0 ;; serving) echo 1 ;; vllm) echo 2 ;; esac; }
FROM_N=$(stage_num "$FROM_STAGE")
should_run() { [ "$(stage_num "$1")" -ge "$FROM_N" ]; }

[ -f docker/tenstorrent-bringup.Dockerfile ] || {
  echo "run me from the repo root (docker/ not found)" >&2; exit 1; }

SUFFIX=""
[ "$PRSTACK" = 1 ] && SUFFIX="-prstack"
BRINGUP="${REG}/tt-bringup:${TAG}"
SERVING="${REG}/tt-serving:${TAG}${SUFFIX}"
VLLMIMG="${REG}/tt-vllm:${TAG}${SUFFIX}-plugin"

if should_run bringup; then
  echo "=== 1/3  bring-up  ${BRINGUP} ==="
  docker build -f docker/tenstorrent-bringup.Dockerfile \
    --build-arg TT_METAL_REF="${TAG}" --build-arg JOBS="${JOBS}" \
    -t "${BRINGUP}" .

  if [ "$PRSTACK" = 1 ]; then
    echo "=== 1b/3  + upstream fix stack  ${BRINGUP}${SUFFIX} ==="
    # Layers on the BUILT image so the tt-metal rebuild is incremental — only the
    # four patched C++ files recompile. From-scratch this step is hours; layered it
    # is roughly a minute plus the image export.
    docker build -f docker/tenstorrent-bringup-prstack.Dockerfile \
      --build-arg BASE="${BRINGUP}" --build-arg JOBS="${JOBS}" \
      -t "${BRINGUP}${SUFFIX}" .
  fi
else
  echo "=== 1/3  bring-up  SKIPPED (--from ${FROM_STAGE}) ==="
fi

if should_run serving; then
  echo "=== 2/3  serving  ${SERVING} ==="
  # The repo file, NOT an inline copy — this is the one that carries the pip cache
  # mount. See the header.
  docker build -f docker/tenstorrent-serving.Dockerfile \
    --build-arg BASE="${BRINGUP}${SUFFIX}" \
    -t "${SERVING}" .
else
  echo "=== 2/3  serving  SKIPPED (--from ${FROM_STAGE}) ==="
fi

if [ "$VLLM" = 1 ] && should_run vllm; then
  echo "=== 3/3  vllm  ${VLLMIMG} ==="
  docker build -f docker/tenstorrent-vllm-plugin.Dockerfile \
    --build-arg BASE="${SERVING}" \
    -t "${VLLMIMG}" .
fi

# Assert on whatever we ended up with, not on what we happened to build. An image
# that silently lost the FIR patch produces wrong output for every batched
# request past the first user — not something to find out from a serving log.
if [ "$PRSTACK" = 1 ]; then
  FIR=/opt/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py
  IMGS="${SERVING}"
  [ "$VLLM" = 1 ] && IMGS="${IMGS} ${VLLMIMG}"
  for img in ${IMGS}; do
    echo "--- asserting FIR batch fix in ${img} ---"
    # if/else, not A && B || C: with the latter a failing echo would silently
    # take the error branch and report a good image as broken.
    if docker run --rm --entrypoint /bin/bash "$img" -lc "grep -q '(B, k + T, D)' $FIR"; then
      echo "    OK: batch bound is B, not 1"
    else
      echo "    FAIL: ${img} still has the batch-truncating slice" >&2
      exit 1
    fi
  done
fi

if [ "$VLLM" = 1 ]; then
  echo "--- asserting the vLLM plugin loads in ${VLLMIMG} ---"
  docker run --rm --entrypoint /bin/bash "${VLLMIMG}" -lc \
    'python3 -c "import vllm, vllm_tt_plugin, ttnn, torch, transformers; \
      print(\"    vllm\", vllm.__version__, \"| torch\", torch.__version__, \
            \"| transformers\", transformers.__version__)"'
fi

echo
echo "=== built ==="
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep -E "tt-(bringup|serving|vllm)" | sort
