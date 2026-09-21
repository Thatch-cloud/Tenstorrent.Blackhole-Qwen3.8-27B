#!/usr/bin/env bash
#
# Build and test the batch-64 attn_prep graft on the rig. Run this ON the rig
# (thatch-control-plane-prod), not from the Windows side.
#
# Prerequisite: this directory must already be staged on the rig at $SRC_DIR
# (default ~/kwork64/attn_prep), shipped with the base64-through-plink pattern.
# It deliberately does NOT touch ~/kwork/attn_prep or ~/opgraft-K, so the
# production graft keeps working while this is A/B'd.
#
# What it does:
#   1. rm -rf the op dir inside the ttbuild container, then docker cp this one
#      over it (docker cp NESTS when the target exists, hence the rm -rf).
#   2. delete the three non-source files this directory carries, inside the
#      container, before the build sees them.
#   3. ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so (~25-30 s
#      incremental). The Python binding that matters is build_Release/ttnn/
#      _ttnn.so, NOT the stale ttnn/ttnn/_ttnn.so in the source tree.
#   4. refresh ~/opgraft-K64 = both .so + the production graft's op dirs, with
#      attn_prep replaced by this one. This is ADDITIVE: it never deletes the
#      graft, so an op another agent staged there (nlp_concat_heads_decode,
#      which lives under the experimental ops root and needs its own -v mount
#      line in the runner) survives.
#   5. run test_attn_prep_b64.py in the serving image on card M with
#      TT_METAL_WATCHER=5, so a hang leaves generated/watcher/watcher.log
#      waypoints, and copy that log out before the container is removed.
#
# If it hangs: this script kills the container itself at $TIMEOUT seconds --
# `timeout` on `docker run` does NOT stop the container, which is why the run
# is detached and polled. After a hung kernel the card needs a reset:
#
#     ~/.local/bin/tt-smi -r
#
# tt-smi is not on the ssh PATH; use the full path. Reset only after the
# container is gone (this script does `docker rm -f` on timeout).
#
# The rig is a production control plane and card M is a shared resource. Do not
# run this while a serving benchmark is live.

set -euo pipefail

OP=attn_prep
SRC_DIR=${SRC_DIR:-$HOME/kwork64/attn_prep}
BUILD_CONTAINER=${BUILD_CONTAINER:-ttbuild}
METAL=${METAL:-/opt/tt-metal}
BUILD=${BUILD:-build_Release}
OPS=$METAL/ttnn/cpp/ttnn/operations/transformer
GRAFT=${GRAFT:-$HOME/opgraft-K64}
PROD_GRAFT=${PROD_GRAFT:-$HOME/opgraft-K}
TEST_IMAGE=${TEST_IMAGE:-zot.thatch.local:5000/tt-serving:v0.77.0-rc1-prstack}
CARD=${CARD:-/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D}
CACHE=${CACHE:-/ttcache}
CONTAINER=${CONTAINER:-k64test}
TEST_PY=${TEST_PY:-test_attn_prep_b64.py}
WATCHER=${WATCHER:-5}
TIMEOUT=${TIMEOUT:-900}
OUT_DIR=${OUT_DIR:-$HOME/k64-evidence}
OPS_LIST=${OPS_LIST:-"gdn_conv_gates gdn_norm_gate attn_prep decode_gated_delta_rule"}
EXTRA_FILES="PATCH-NOTES.md $TEST_PY build-and-test-b64.sh"

say() { printf '\n== %s\n' "$*"; }

say "checking the staged op dir"
test -d "$SRC_DIR" || { echo "missing $SRC_DIR"; exit 2; }
for f in device/attn_prep_program_factory.cpp \
         device/kernels/dataflow/reader_attn_prep.cpp \
         device/kernels/dataflow/writer_attn_prep.cpp \
         device/kernels/compute/attn_prep.cpp \
         "$TEST_PY"; do
    test -f "$SRC_DIR/$f" || { echo "missing $SRC_DIR/$f"; exit 2; }
done
grep -q "cbap::cpy" "$SRC_DIR/device/attn_prep_program_factory.cpp" || { echo "factory is not the batch-64 patch"; exit 2; }
grep -q "cb_cpy" "$SRC_DIR/device/kernels/dataflow/reader_attn_prep.cpp" || { echo "reader is not the batch-64 patch"; exit 2; }
grep -q "cb_cpy" "$SRC_DIR/device/kernels/dataflow/writer_attn_prep.cpp" || { echo "writer is not the batch-64 patch"; exit 2; }

say "grafting $OP into $BUILD_CONTAINER:$OPS"
docker exec "$BUILD_CONTAINER" rm -rf "$OPS/$OP"
docker cp "$SRC_DIR" "$BUILD_CONTAINER:$OPS/$OP"
for f in $EXTRA_FILES; do
    docker exec "$BUILD_CONTAINER" rm -f "$OPS/$OP/$f"
done
docker exec "$BUILD_CONTAINER" ls "$OPS/$OP" "$OPS/$OP/device" "$OPS/$OP/device/kernels/compute" "$OPS/$OP/device/kernels/dataflow"

say "ninja incremental build"
docker exec "$BUILD_CONTAINER" ninja -C "$METAL/$BUILD" ttnn/_ttnncpp.so ttnn/_ttnn.so

say "refreshing $GRAFT (additive: op dirs another agent staged there are left alone)"
mkdir -p "$GRAFT"
if [ -d "$PROD_GRAFT" ]; then
    for op in $OPS_LIST; do
        if [ -d "$PROD_GRAFT/$op" ] && [ ! -d "$GRAFT/$op" ]; then
            cp -a "$PROD_GRAFT/$op" "$GRAFT/$op"
        fi
    done
fi
docker cp "$BUILD_CONTAINER:$METAL/$BUILD/ttnn/_ttnn.so" "$GRAFT/_ttnn.so"
docker cp "$BUILD_CONTAINER:$METAL/$BUILD/ttnn/_ttnncpp.so" "$GRAFT/_ttnncpp.so"
rm -rf "${GRAFT:?}/$OP"
cp -a "$SRC_DIR" "$GRAFT/$OP"
for f in $EXTRA_FILES; do
    rm -f "$GRAFT/$OP/$f"
done
ls -la "$GRAFT"

say "running $TEST_PY on card M with TT_METAL_WATCHER=$WATCHER"
mkdir -p "$OUT_DIR"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
KM=()
for op in $OPS_LIST; do
    if [ -d "$GRAFT/$op" ]; then
        KM+=( -v "$GRAFT/$op:$OPS/$op:ro" )
    fi
done
docker run -d --name "$CONTAINER" \
    --device "$CARD" \
    -v "$GRAFT/_ttnn.so:$METAL/ttnn/ttnn/_ttnn.so:ro" \
    -v "$GRAFT/_ttnncpp.so:$METAL/$BUILD/ttnn/_ttnncpp.so:ro" \
    ${KM[@]+"${KM[@]}"} \
    -v "$SRC_DIR/$TEST_PY:/work/$TEST_PY:ro" \
    -v "$CACHE:$CACHE" \
    -e "TT_METAL_CACHE=$CACHE" \
    -e "TT_METAL_WATCHER=$WATCHER" \
    -e "TT_METAL_HOME=$METAL" \
    -w "$METAL" \
    "$TEST_IMAGE" \
    python3 "/work/$TEST_PY" >/dev/null

docker logs -f "$CONTAINER" 2>&1 | tee "$OUT_DIR/b64-test.log" &
LOG_PID=$!

rc=""
deadline=$(( $(date +%s) + TIMEOUT ))
while true; do
    state=$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo gone)
    if [ "$state" != "true" ]; then
        rc=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || echo 1)
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "TIMEOUT after ${TIMEOUT}s: the op is hung, killing the container"
        rc=124
        break
    fi
    sleep 5
done
kill "$LOG_PID" >/dev/null 2>&1 || true
wait "$LOG_PID" 2>/dev/null || true

say "collecting the watcher log"
docker cp "$CONTAINER:$METAL/generated/watcher/watcher.log" "$OUT_DIR/watcher.log" >/dev/null 2>&1 \
    || docker cp "$CONTAINER:/work/generated/watcher/watcher.log" "$OUT_DIR/watcher.log" >/dev/null 2>&1 \
    || echo "no watcher.log found in the container"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

if [ -f "$OUT_DIR/watcher.log" ]; then
    echo "watcher log: $OUT_DIR/watcher.log"
    grep -c CWFW "$OUT_DIR/watcher.log" || true
fi
echo "test log: $OUT_DIR/b64-test.log"

if [ "$rc" != "0" ]; then
    echo
    echo "FAILED (exit $rc). If it hung, reset the card before anything else runs on it:"
    echo "    ~/.local/bin/tt-smi -r"
fi
exit "$rc"
