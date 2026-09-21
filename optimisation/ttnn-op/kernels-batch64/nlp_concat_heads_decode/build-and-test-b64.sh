#!/usr/bin/env bash
#
# Graft the batch-64 nlp_concat_heads_decode over the image's copy of the op, rebuild the two
# .so in the ttbuild container, and run test_concat_heads_b64.py on card M with the watcher on.
#
# Run this ON THE RIG (thatch-control-plane-prod) after the op dir has been shipped there
# (base64 tar through plink, per tt-rig-access.md), from wherever it landed, e.g.
#   bash ~/kwork64/nlp_concat_heads_decode/build-and-test-b64.sh
#
# nlp_concat_heads_decode is an UPSTREAM op: it is already in the experimental/transformer
# CMakeLists glob, so nothing has to be registered - the copy alone makes ninja rebuild it. It
# lives under operations/EXPERIMENTAL/transformer, a different root from the kwork ops.
#
# The host .cpp changes need the ninja rebuild (~25 s incremental); the python binding that
# results is build_Release/ttnn/_ttnn.so, NOT the stale ttnn/ttnn/_ttnn.so in the source tree.
# The two device kernels are JIT-compiled from source at dispatch, so they take effect through
# the runtime bind-mount of the op dir.
#
# This script builds ONLY this op, so an in-flight patch to another K64 op cannot break it. For
# the full batch-64 stack run ~/kwork64/build-k64.sh instead (it stages attn_prep too and
# assembles the whole graft), then re-run this script - it will reuse ~/kwork64/test-k64.sh.
#
# On a hang: docker rm -f k64concat (timeout does not stop the container), then reset the card
# with ~/.local/bin/tt-smi -r (not on the ssh PATH). The watcher log is copied out below.

set -euo pipefail

OP=nlp_concat_heads_decode
SRC=$(cd "$(dirname "$0")" && pwd)
XOPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer
BUILDER=${BUILDER:-ttbuild}
KWORK=${KWORK:-$HOME/kwork64}
GRAFT=${GRAFT:-$HOME/opgraft-K64}
IMAGE=${IMAGE:-zot.thatch.local:5000/tt-serving:v0.77.0-rc1-prstack}
CARD=${CARD:-/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D}
CACHE=${CACHE:-$HOME/ttcache}
NAME=${NAME:-k64concat}
TEST=${1:-$SRC/test_concat_heads_b64.py}

echo "== staging the 13 op sources from $SRC (PATCH-NOTES.md, the test and this script excluded)"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$OP/device/kernels/dataflow"
cp "$SRC"/*.cpp "$SRC"/*.hpp "$STAGE/$OP/"
cp "$SRC"/device/*.cpp "$SRC"/device/*.hpp "$STAGE/$OP/device/"
cp "$SRC"/device/kernels/dataflow/*.cpp "$STAGE/$OP/device/kernels/dataflow/"
find "$STAGE/$OP" -type f | sort

mkdir -p "$KWORK"
rm -rf "${KWORK:?}/$OP"
cp -r "$STAGE/$OP" "$KWORK/$OP"

echo "== grafting into the $BUILDER source tree and rebuilding"
docker exec "$BUILDER" rm -rf "$XOPS/$OP"
docker cp "$STAGE/$OP" "$BUILDER:$XOPS/$OP"
docker exec -w /opt/tt-metal "$BUILDER" ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so

echo "== refreshing $GRAFT"
mkdir -p "$GRAFT"
docker cp "$BUILDER:/opt/tt-metal/build_Release/ttnn/_ttnn.so" "$GRAFT/_ttnn.so"
docker cp "$BUILDER:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so" "$GRAFT/_ttnncpp.so"
rm -rf "${GRAFT:?}/$OP"
cp -r "$STAGE/$OP" "$GRAFT/$OP"
ls -l "$GRAFT/_ttnn.so" "$GRAFT/_ttnncpp.so"
ls -d "$GRAFT/$OP"

if [ -f "$KWORK/test-k64.sh" ]; then
    echo "== running $TEST through $KWORK/test-k64.sh (card M, timeout 900, TT_METAL_WATCHER=5)"
    exec bash "$KWORK/test-k64.sh" "$TEST"
fi

echo "== running $TEST directly (fallback: these docker flags are not the proven test-k64.sh set)"
mkdir -p "$CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true
set +e
timeout 900 docker run --name "$NAME" \
    --device "$CARD" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "$CACHE:/ttcache" \
    -v "$GRAFT/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro" \
    -v "$GRAFT/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro" \
    -v "$GRAFT/$OP:$XOPS/$OP:ro" \
    -v "$TEST:/work/test_concat_heads_b64.py:ro" \
    -e TT_METAL_WATCHER=5 \
    -e TT_METAL_CACHE=/ttcache \
    -w /opt/tt-metal \
    "$IMAGE" python3 /work/test_concat_heads_b64.py
rc=$?
set -e

docker cp "$NAME:/opt/tt-metal/generated/watcher/watcher.log" "./watcher-$NAME.log" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true

if [ "$rc" -eq 124 ]; then
    echo "TIMED OUT after 900s: the kernel hung. Reset the card with ~/.local/bin/tt-smi -r"
    echo "Watcher waypoints, if they were copied out: ./watcher-$NAME.log"
fi
echo "exit $rc"
exit "$rc"
