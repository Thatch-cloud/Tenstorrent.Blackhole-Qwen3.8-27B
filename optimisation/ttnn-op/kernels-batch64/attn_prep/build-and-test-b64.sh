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
#   5. run test_attn_prep_b64.py in the serving image on the qualification
#      card (QUAL_CARD, a board id; default card B, blackhole-F36F768B9A5CAFA0)
#      with TT_METAL_WATCHER=5, so a hang leaves generated/watcher/watcher.log
#      waypoints, and copy that log out before the container is removed. The
#      node is resolved by board id right before the run, and the run is
#      refused while a container or a host process can reach it.
#
# If it hangs: this script kills the container itself at $TIMEOUT seconds --
# `timeout` on `docker run` does NOT stop the container, which is why the run
# is detached and polled. After a hung kernel the target card needs a reset:
# the script prints a tt-smi -r command that resolves its board id when run, and its
# PCI address for tt-smi -ls. A bare `tt-smi -r` resets EVERY board, the serving pair
# included, and a bare number is tt-smi's own board index: never use either.
# tt-smi is not on the ssh PATH; use the full path. Reset only after the
# container is gone (this script does `docker rm -f` on timeout).
#
# The rig is a production control plane. Card M and card A are the serving pair:
# QUAL_CARD may name one only with ALLOW_SERVING_CARD=1 (a loud warning follows).

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
CACHE=${CACHE:-/ttcache}
TEST_PY=${TEST_PY:-test_attn_prep_b64.py}
WATCHER=${WATCHER:-5}
TIMEOUT=${TIMEOUT:-900}
OUT_DIR=${OUT_DIR:-$HOME/k64-evidence}
OPS_LIST=${OPS_LIST:-"gdn_conv_gates gdn_norm_gate attn_prep decode_gated_delta_rule"}
EXTRA_FILES="PATCH-NOTES.md $TEST_PY build-and-test-b64.sh"
if [ -n "${CARD:-}" ]; then
    echo "refusing: CARD is retired (it took any path); set QUAL_CARD to a board id under /dev/tenstorrent/by-id" >&2
    exit 2
fi
# >>> qual_card.sh: which board a qualification harness runs on (canonical copy scripts/ci/qual_card.sh)
# Every single-card harness under optimisation/ttnn-op embeds this block byte for byte (the scripts in
# scripts/ci source the file); scripts/ci/test_qual_card.py fails when a copy drifts, and
# `py -3.11 -B scripts/ci/test_qual_card.py --sync` re-copies the canonical text into every harness.
#
# The rig has three p150a. Card M (blackhole-CEF5729692C19E6D) and card A (blackhole-3707293C249A5E67)
# are the serving pair: Ethernet-linked, mounted by every CI gate arm (lever_n_m3native_run_arm.sh), and
# reset together by the gate. Card B (blackhole-F36F768B9A5CAFA0, PCIe only) is the qualification card.
# /dev/tenstorrent/N numbers change across resets and switch power-cycles, and tt-smi's own board index
# is a different numbering again, so nothing here hard-codes either: the target is a board id, its node
# is resolved with readlink -f at launch and again right before the container starts, and the reset
# hint prints commands that resolve the board id when they are run (never a bare index) plus the PCI
# address that identifies the board's row in tt-smi -ls.
#
# Only the m3native gate (qwen-lever-n-m3native-gate.yml) is scoped to the serving pair. qwen-card-reset.yml
# and most other qwen-* hardware workflows still act on every node, or on fixed node numbers, in the same
# qwen-two-p150a-exclusive group: check gh run list for that group before and during a card-B session.
#
#   QUAL_CARD=<board id>    the target, a name under /dev/tenstorrent/by-id (default: card B)
#   ALLOW_SERVING_CARD=1    required to target card M or card A (half of the serving pair); loud warning
#
#   qual_card_select      sets QUAL_CARD, QUAL_BYID, QUAL_TAG, QUAL_SERVING; refuses a serving card
#                         without the override. Touches no device: dry runs call it too.
#   qual_card_resolve     sets QUAL_NODE (readlink -f, now) and QUAL_PCI; refuses a missing board and
#                         treats a board id that resolves to a serving card's node as that serving card.
#   qual_refuse_holders   refuses while a container or a host process can reach the target.
#   qual_card_recheck     readlink -f again right before the container starts; refuses if the node moved.
#   qual_reset_hint       the recovery lines after a hang, on stdout; it never resets anything itself.
QUAL_CARD_B=blackhole-F36F768B9A5CAFA0
QUAL_SERVING_CARDS='blackhole-CEF5729692C19E6D blackhole-3707293C249A5E67'
QUAL_TT_ROOT=/dev/tenstorrent
QUAL_BYID_ROOT=$QUAL_TT_ROOT/by-id
QUAL_SYS_ROOT=/sys

qual_card_label() {
  case ${1:-$QUAL_CARD} in
    blackhole-CEF5729692C19E6D) echo 'card M, half of the serving pair' ;;
    blackhole-3707293C249A5E67) echo 'card A, half of the serving pair' ;;
    "$QUAL_CARD_B") echo 'card B, the qualification card' ;;
    *) echo 'a board this harness does not name' ;;
  esac
}

qual_card_select() {
  QUAL_CARD=${QUAL_CARD:-$QUAL_CARD_B}
  case $QUAL_CARD in
    .*|*/*|*[!A-Za-z0-9._-]*)
      echo "refusing: QUAL_CARD=$QUAL_CARD is not a board id under $QUAL_BYID_ROOT (default $QUAL_CARD_B, card B)" >&2
      exit 1 ;;
  esac
  QUAL_BYID=$QUAL_BYID_ROOT/$QUAL_CARD
  case $QUAL_CARD in
    blackhole-CEF5729692C19E6D) QUAL_TAG=card-m ;;
    blackhole-3707293C249A5E67) QUAL_TAG=card-a ;;
    "$QUAL_CARD_B") QUAL_TAG=card-b ;;
    *) QUAL_TAG=$QUAL_CARD ;;
  esac
  QUAL_SERVING=0
  case " $QUAL_SERVING_CARDS " in
    *" $QUAL_CARD "*) qual_serving_override ;;
  esac
}

qual_serving_override() {
  QUAL_SERVING=1
  if [ "${ALLOW_SERVING_CARD:-0}" != 1 ]; then
    echo "refusing: QUAL_CARD=$QUAL_CARD is $(qual_card_label), which the CI gate and the endpoint use." >&2
    echo "  Qualify on card B (unset QUAL_CARD, or QUAL_CARD=$QUAL_CARD_B); ALLOW_SERVING_CARD=1 overrides." >&2
    exit 1
  fi
  echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' >&2
  echo "!!! WARNING: ALLOW_SERVING_CARD=1: this run is on $QUAL_CARD, $(qual_card_label)." >&2
  echo '!!! A CI gate that starts meanwhile fails its holder check, or resets the pair under this run before' >&2
  echo '!!! it opens the card; a hang here keeps the pair down until card M and card A are reset together.' >&2
  echo '!!! Check gh run list for the qwen-two-p150a-exclusive group before and after.' >&2
  echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' >&2
}

qual_is_char() {
  test -c "$1"
}

qual_card_resolve() {
  local card node
  QUAL_NODE=$(readlink -f -- "$QUAL_BYID" 2>/dev/null || true)
  if [ -z "$QUAL_NODE" ] || ! qual_is_char "$QUAL_NODE"; then
    echo "refusing: $QUAL_CARD ($(qual_card_label)) has no device node here: $QUAL_BYID does not resolve to one" >&2
    exit 1
  fi
  if [ "$QUAL_SERVING" != 1 ]; then
    for card in $QUAL_SERVING_CARDS; do
      node=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
      if [ "$node" = "$QUAL_NODE" ]; then
        echo "### $QUAL_CARD resolves to $node, the node of $card ($(qual_card_label "$card"))" >&2
        qual_serving_override
      fi
    done
  fi
  QUAL_PCI=$(qual_pci_of "$QUAL_NODE")
  echo "### target card: $QUAL_CARD ($(qual_card_label)) -> $QUAL_NODE, PCI ${QUAL_PCI:-unknown}"
  if [ "$QUAL_SERVING" != 1 ]; then
    echo "### note: only the m3native gate spares this card; qwen-card-reset.yml and most other hardware workflows"
    echo "###   still act on every node - check gh run list for the qwen-two-p150a-exclusive group"
  fi
}

# A path's device numbers (major:minor, hex, following symlinks), or nothing.
qual_majmin_of() {
  stat -L -c '%t:%T' -- "$1" 2>/dev/null || true
}

# A device node's PCI address (0000:f4:00.0) from sysfs, or nothing.
qual_pci_of() {
  local majmin dev
  majmin=$(qual_majmin_of "${1:-}")
  case $majmin in
    *[!0-9a-f:]*|:*|*:|*:*:*) return 0 ;;
    *:*) ;;
    *) return 0 ;;
  esac
  dev=$(readlink -e -- "$QUAL_SYS_ROOT/dev/char/$((16#${majmin%%:*})):$((16#${majmin##*:}))/device" 2>/dev/null) || return 0
  case ${dev##*/} in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f]:[0-9a-f][0-9a-f]:[0-9a-f][0-9a-f].[0-7]) echo "${dev##*/}" ;;
  esac
}

# Why a container (qual_refuse_holders' docker inspect lines) can reach the target; nothing when it cannot.
# A non-privileged container reaches a device only through its device list, its device cgroup rules and
# its device requests: a bind mount of /dev or /dev/tenstorrent shows node files it cannot open (every CI
# gate arm mounts /dev/tenstorrent read-only for its board-mapping check and is given card M and card A
# only). So for a non-serving target it names: --privileged; any device cgroup rule; a device request
# naming Tenstorrent (CDI); a mapped device path that is the target's node, has the target's device
# numbers, is a directory holding the node, or is a /dev/tenstorrent path that is not a device node now
# (which board it was given is unknowable); a mount of the target's node itself. For a serving target
# (ALLOW_SERVING_CARD=1) it names any container that can reach any Tenstorrent device: --privileged, a
# device cgroup rule, a Tenstorrent device request, or /dev or a tenstorrent path among its devices or
# mounts.
qual_container_reach() {
  local head=${1%%$'\n'*} line kind path resolved
  case $head in
    *' true') echo 'is --privileged (it can open every device)'; return 0 ;;
  esac
  while IFS= read -r line; do
    kind=${line%% *}
    path=${line#* }
    case $kind in
      rule) echo "has a device cgroup rule ($path)"; return 0 ;;
      req)
        case $path in
          *[Tt]enstorrent*) echo "has a device request for a Tenstorrent device ($path)"; return 0 ;;
        esac
        continue ;;
      dev|mnt) ;;
      *) continue ;;
    esac
    resolved=$(readlink -f -- "$path" 2>/dev/null || true)
    if [ "$QUAL_SERVING" = 1 ]; then
      case "$path $resolved" in
        "${QUAL_TT_ROOT%/*} "*|"${QUAL_TT_ROOT%/*}/ "*|*" ${QUAL_TT_ROOT%/*}"|*tenstorrent*)
          echo "has $path among its ${kind}s (a Tenstorrent device; the target is a serving card)"; return 0 ;;
      esac
      continue
    fi
    if [ "$resolved" = "$QUAL_NODE" ]; then
      echo "has $path among its ${kind}s, which is $QUAL_NODE (the target)"; return 0
    fi
    [ "$kind" = dev ] || continue
    if [ -n "${QUAL_MAJMIN:-}" ] && [ "$(qual_majmin_of "$path")" = "$QUAL_MAJMIN" ]; then
      echo "is given $path, a device with the target's numbers ($QUAL_MAJMIN)"; return 0
    fi
    case $QUAL_NODE in
      "$resolved"/*) echo "is given $path, a directory holding the target's node"; return 0 ;;
    esac
    case $path in
      "$QUAL_TT_ROOT"|"$QUAL_TT_ROOT"/*)
        if [ -z "$resolved" ] || ! qual_is_char "$resolved"; then
          echo "is given $path, which is not a device node now (it may be the target)"; return 0
        fi ;;
    esac
  done <<< "${1#*$'\n'}"
}

# Refuses while anything else can reach the target. Containers: qual_container_reach. Host processes:
# fuser on the target's node (with sudo -n when that works, else this user's processes, said so) - which
# also sees a process in a container holding it, however the container was given it - up to five tries
# two seconds apart (the rig's telemetry exporter holds every card for a moment every 30 s), refusing
# while any holder persists.
qual_refuse_holders() {
  local id info why st out try scope
  local pre=()
  QUAL_MAJMIN=$(qual_majmin_of "$QUAL_NODE")
  for id in $(docker ps -q); do
    info=$(docker inspect "$id" --format '{{.Name}} {{.HostConfig.Privileged}}{{println}}{{range .HostConfig.Devices}}dev {{println .PathOnHost}}{{end}}{{range .HostConfig.DeviceCgroupRules}}rule {{println .}}{{end}}{{range .HostConfig.DeviceRequests}}req {{.Driver}} {{println .DeviceIDs}}{{end}}{{range .Mounts}}mnt {{println .Source}}{{end}}') || continue
    why=$(qual_container_reach "$info")
    if [ -n "$why" ]; then
      echo "refusing: container ${info%% *} $why" >&2
      exit 1
    fi
  done
  echo "### containers: none can reach $QUAL_NODE ($QUAL_CARD)"
  if ! command -v fuser >/dev/null 2>&1; then
    echo "WARN: fuser is not installed; host processes holding $QUAL_NODE were not checked" >&2
    return 0
  fi
  scope="this user's processes only (no passwordless sudo)"
  if [ "$(id -u)" = 0 ]; then
    scope=all
  elif sudo -n true >/dev/null 2>&1; then
    pre=(sudo -n)
    scope=all
  fi
  for try in 1 2 3 4 5; do
    st=0
    out=$(${pre[@]+"${pre[@]}"} fuser -v "$QUAL_NODE" 2>&1) || st=$?
    if [ "$st" != 0 ] && [ -z "$out" ]; then
      echo "### device holders on $QUAL_NODE: none ($scope)"
      return 0
    fi
    [ "$try" = 5 ] || sleep 2
  done
  echo "refusing: host processes hold $QUAL_NODE (or fuser failed):" >&2
  echo "$out" >&2
  exit 1
}

# readlink -f again right before the container starts: refuses when a board is not on the node the holder
# check cleared (the boards re-enumerated in between - a switch event, or the gate resetting card A on the
# switch card B shares - so the old node may now be another board's). Args: [board id] [node]; the
# target by default.
qual_card_recheck() {
  local card=${1:-$QUAL_CARD} was=${2:-$QUAL_NODE} now
  now=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
  if [ -z "$now" ] || [ "$now" != "$was" ] || ! qual_is_char "$now"; then
    echo "refusing: $card ($(qual_card_label "$card")) was $was at the holder check and is ${now:-gone} now" >&2
    echo "  (the boards re-enumerated); nothing was launched - check the boards, then run again" >&2
    exit 1
  fi
  echo "### $card is still $now"
}

# The recovery lines after a hang. The node and PCI address are looked up again now (nodes renumber),
# and the reset command resolves the board id again when it is run: tt-smi -r takes a /dev/tenstorrent
# path, while a bare number there is tt-smi's own board index, which renumbers too - so none is printed.
# The command runs tt-smi only when readlink -e resolved every board id (an empty argument would make
# tt-smi -r reset every board). The PCI address is what identifies the board's row in tt-smi -ls.
qual_reset_hint() {
  local node pci card n p var vars='m a' cmd= args=
  node=$(readlink -f -- "$QUAL_BYID" 2>/dev/null || true)
  pci=$(qual_pci_of "$node")
  echo "HANG RECOVERY for $QUAL_CARD ($(qual_card_label)), once the container is gone; nothing here resets a card."
  echo "  It is ${node:-absent} now, PCI ${pci:-unknown}; nodes renumber, so the reset below resolves the board id"
  echo "  when it is run. Never a bare number: tt-smi -r reads one as tt-smi's own board index, which renumbers too."
  if [ "$QUAL_SERVING" = 1 ]; then
    for card in $QUAL_SERVING_CARDS; do
      n=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
      p=$(qual_pci_of "$n")
      echo "  $card ($(qual_card_label "$card")) is ${n:-absent} now, PCI ${p:-unknown}."
      var=${vars%% *}
      vars=${vars#* }
      cmd="$cmd$var=\$(readlink -e $QUAL_BYID_ROOT/$card) && "
      args="$args \"\$$var\""
    done
    echo "  It is half of the serving pair: one link end reset alone leaves the mesh at 1x1. Check gh run list for the"
    echo "  qwen-two-p150a-exclusive group, confirm both PCI addresses in ~/.local/bin/tt-smi -ls, then reset card M and"
    echo "  card A TOGETHER, in one call:"
    echo "    $cmd~/.local/bin/tt-smi -r$args"
    echo "  then a passing smoke run."
  else
    if [ -n "$pci" ]; then
      echo "  CONFIRM its row (PCI BDF $pci): ~/.local/bin/tt-smi -ls | grep -i '${pci#0000:}'; then reset it alone:"
    else
      echo "  CONFIRM in ~/.local/bin/tt-smi -ls which row is ${node:-this board} (its PCI address is unknown here); then reset it alone:"
    fi
    echo "    n=\$(readlink -e $QUAL_BYID) && ~/.local/bin/tt-smi -r \"\$n\""
    echo "  then a passing smoke run. Never card M or card A: they are the serving pair, and this card needs neither."
  fi
}
# <<< qual_card.sh
qual_card_select
CONTAINER=${CONTAINER:-k64test-$QUAL_TAG}

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

say "running $TEST_PY on $QUAL_CARD ($(qual_card_label)) with TT_METAL_WATCHER=$WATCHER"
qual_card_resolve
qual_refuse_holders
mkdir -p "$OUT_DIR"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
KM=()
for op in $OPS_LIST; do
    if [ -d "$GRAFT/$op" ]; then
        KM+=( -v "$GRAFT/$op:$OPS/$op:ro" )
    fi
done
qual_card_recheck   # the board is still on the node the holder check cleared
docker run -d --name "$CONTAINER" \
    --device "$QUAL_NODE" \
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
    echo "FAILED (exit $rc). If it hung, reset $QUAL_CARD before anything else runs on it:"
    qual_reset_hint
fi
exit "$rc"
