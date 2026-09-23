#!/usr/bin/env bash
# Run test_sdpa_decode_qwen_card_m.py on card M, in the serving image.
#
#   bash run_card_m.sh reference   # stock image, no graft: legacy calls only, records output shas
#   bash run_card_m.sh candidate   # the graft (K64f by default) mounted exactly as
#                                  # lever_n_m3native_run_arm.sh mounts it; compares every qwen
#                                  # mode against legacy, and legacy against the reference
#   WATCHER=1 bash run_card_m.sh candidate
#                                  # THE FIRST HARDWARE PASS of a new graft (spec section 8, N6):
#                                  # TT_METAL_WATCHER=5 (NoC sanitiser, waypoints) over one full pass
#                                  # at 33,024 and 2,304 keys, seed 0, no timing, a per-call
#                                  # watchdog (WATCHDOG_S, default 120 s) that os._exit(3)s the
#                                  # test, and a 900 s container timeout. The watcher's own log
#                                  # lands in $RESULTS/watcher-<stamp>/.
#
# Env: K64F_SRC (~/kwork64/k64f: this directory), KOPGRAFT64 (~/opgraft-K64f; ~/opgraft-K64e
# still works - the test reads the stage off the binary), IMAGE (the arm's default serving
# image), RESULTS (~/kwork64/k64f/card-m), CARD_M_ARGS (extra args, appended last, e.g.
# "--seeds 0 --capacities 33024" for a quick pass), WATCHER=1, WATCHDOG_S.
# Both roles set QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 as the arm does: the legacy G8 (PNHt=3) calls
# do not fit L1 with the full tree scratch.
#
# Every run gets a fresh kernel cache. ON A HANG (the WATCHDOG line, exit 3, or the timeout's
# 124): the EXIT trap removes the container; then reset CARD M ONLY with ~/.local/bin/tt-smi -r
# <card M's index> - never the serving cards. This script never resets anything itself.
set -euo pipefail
mode=${1:?usage: run_card_m.sh reference|candidate}
S=${K64F_SRC:-$HOME/kwork64/k64f}
G=${KOPGRAFT64:-$HOME/opgraft-K64f}
IMAGE=${IMAGE:-sha256:e41ef884f4c8e07ce6632511fac19364af35dc73d601f05f5191784a64a3a768}
R=${RESULTS:-$S/card-m}
CARD_M=/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D
name=qwen-sdpa-card-m
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=1800

node=$(readlink -f "$CARD_M")
test -c "$node" || { echo "card M ($CARD_M) is not a device node here" >&2; exit 1; }
# Never share the card: refuse if any running container can reach a Tenstorrent device.
for id in $(docker ps -q); do
  if docker inspect "$id" --format '{{json .HostConfig.Devices}} {{json .Mounts}}' | grep -q tenstorrent; then
    echo "refusing: container $(docker inspect "$id" --format '{{.Name}}') holds a Tenstorrent device" >&2
    exit 1
  fi
done
mkdir -p "$R" "$R/kcache-$mode-$stamp"
chmod 0777 "$R" "$R/kcache-$mode-$stamp"

args=(--out "/results/$mode-$stamp.json")
KM=()
WM=()
case "$mode" in
  reference)
    args+=(--legacy-only)
    ;;
  candidate)
    for part in _ttnn.so _ttnncpp.so attn_prep nlp_concat_heads_decode sdpa_decode; do
      test -e "$G/$part" || { echo "$G/$part missing: build it with build_k64f.sh" >&2; exit 1; }
    done
    KM+=(-v "$G/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro")
    KM+=(-v "$G/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro")
    KM+=(-v "$G/_ttnncpp.so:/opt/tt-metal/build_Release/lib/_ttnncpp.so:ro")
    KM+=(-v "$G/attn_prep:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/attn_prep:ro")
    KM+=(-v "$G/nlp_concat_heads_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode:ro")
    KM+=(-v "$G/sdpa_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode:ro")
    reference=$(ls -t "$R"/reference-*.json 2>/dev/null | head -1 || true)
    if [ -n "$reference" ]; then
      args+=(--reference "/results/$(basename "$reference")")
      echo "legacy outputs are compared against $reference"
    else
      echo "no reference report in $R: run 'run_card_m.sh reference' first for the legacy-regression check"
    fi
    ;;
  *)
    echo "unknown mode $mode" >&2
    exit 1
    ;;
esac
if [ "${WATCHER:-}" = "1" ]; then
  # One full pass under the NoC sanitiser: every check at the early-return and a multi-chunk
  # capacity (two variants, two starts), the full 1,000-call alternation and 200 trace replays;
  # timing is meaningless here. CARD_M_ARGS, appended last, can widen it.
  timeout_s=900
  args+=(--capacities 2304,33024 --seeds 0 --variants normal,zeroq --starts 0,240 --no-timing --watchdog "${WATCHDOG_S:-120}")
  mkdir -p "$R/watcher-$stamp"
  chmod 0777 "$R/watcher-$stamp"
  WM+=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  echo "WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog ${WATCHDOG_S:-120} s, container timeout ${timeout_s} s"
else
  args+=(--watchdog "${WATCHDOG_S:-300}")
fi
# shellcheck disable=SC2206
extra=(${CARD_M_ARGS:-})

echo "### card M $mode $stamp node=$node image=${IMAGE:7:12} graft=$G"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" docker run --rm --name "$name" --network none \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g \
  --device "$node" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$S/test_sdpa_decode_qwen_card_m.py,dst=/bench/test_sdpa_decode_qwen_card_m.py,readonly" \
  --mount "type=bind,src=$R,dst=/results" \
  --mount "type=bind,src=$R/kcache-$mode-$stamp,dst=/kcache" \
  "${KM[@]}" \
  "${WM[@]}" \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8 \
  -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 \
  --entrypoint python3 "$IMAGE" -B /bench/test_sdpa_decode_qwen_card_m.py "${args[@]}" "${extra[@]}" \
  2>&1 | tee "$R/$mode-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status; report $R/$mode-$stamp.json; native log $R/$mode-$stamp.json.native.log"
case "$status" in
  3|124|137)
    echo "HANG SUSPECTED (exit $status): the container is removed on exit; reset card M only:" \
         "~/.local/bin/tt-smi -r <card M index> (CEF5729692C19E6D). Never the serving cards." >&2
    ;;
esac
exit "$status"
