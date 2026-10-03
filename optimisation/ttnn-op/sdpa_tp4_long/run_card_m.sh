#!/usr/bin/env bash
# Run sdpa_tp4_long.py (the long-context SDPA decode sweep: the served K64j G8B2 0x23 call at the FOUR-CARD per-chip shape, 6 query
# heads on 1 KV head, bf8 paged KV, timed against every candidate configuration at 32k / 64k / 131k / 262k and byte-compared with it
# row by row) on the card QUAL_CARD names (no default; card B is refused), in a serving image that CONTAINS THE SERVED SDPA GRAFT.
# One card, eager and traced, no mesh and no collectives, so the run is the cardm step's: C2_ACTIONS=cardm with this harness. NOT a CI
# workflow of its own.
#
#   bash run_card_m.sh                                     # every arm, extents 33024 65792 131328 262400, 4 users
#   CARD_B_ARGS="--arms served,multi --extents 262400" bash run_card_m.sh
#   CARD_B_ARGS="--timing none" bash run_card_m.sh         # exactness only
#
# Needs no weights, no fixtures and no model directory: the sweep makes its own seeded K/V. The scripts (sdpa_tp4_long.py,
# k64j_nkv1_spike.py for the mask and position helpers, test_sdpa_decode_qwen_card_m.py for the native-log capture and the factory
# line parser, sdpa_long_tp.py for the arm names and grids, tp_shapes.py) are mounted from THIS checkout, one file each: no image
# build. The op graft is NOT mounted: the image carries it (the build script installs the K64j graft's binaries and op trees), and
# the sweep records the installed _ttnncpp.so's sha256 and holds it to the graft the checkout's build script pins (override with
# EXPECT_TTNNCPP_SHA256; an empty value there skips the check, said so). Every launched program must also have its [QWEN-SDPA]
# factory line in the native log, so a graft that is installed but not executed fails the run.
#
# The image: IMAGE (a full image reference) if set, else the local image whose tag is qwen38-c2-$IMAGE_TAG (IMAGE_TAG comes from the
# job's C2_CARDM_ENV: the window's tag, e.g. tp4-next-3a), looked up in the local store at run time so no registry name is written
# here. Any serving image from the K64j graft on (every four-card image) qualifies.
#
# Env: REPO (this checkout; default three levels up), IMAGE or IMAGE_TAG, RESULTS (default $HOME/sdpalong/<card tag>), CARD_B_ARGS
# (extra sweep arguments, appended last), SWEEP_TIMEOUT_S (container timeout, default 3000; the sweep stops cleanly 600 s before it),
# EXPECT_TTNNCPP_SHA256, QUAL_CARD (the target board id under /dev/tenstorrent/by-id; required, no default: card B is reserved for
# another project and refused; card M or card A, the serving pair, is refused unless ALLOW_SERVING_CARD=1, which prints a loud
# warning).
#
# Read: results/sdpalong-<stamp>.json (per case and arm: exact / differing_rows, the paired ratio against the served arm per round,
# GB/s, trace_equal; fits: fixed_us and per_chunk_us per arm; winners: the fastest exact arm per case kind) and the verdict line
# 'SDPA_TP4_LONG verdict=...' in the log; sdpalong-<stamp>.json.native.log holds the factory lines.
#
# ON A HANG (the timeout's 124 / 137, or the sweep's own watchdog exit 124): the EXIT trap removes the container; then reset THE TARGET
# CARD ONLY with the hint printed below (a tt-smi -r command that resolves the board id when it is run, and the PCI address that
# identifies its row in tt-smi -ls; never a bare index). This script never resets anything itself.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$here/../../.." && pwd)}

# >>> qual_card.sh: which board a qualification harness runs on (canonical copy scripts/ci/qual_card.sh)
# Every single-card harness under optimisation/ttnn-op embeds this block byte for byte (the scripts in
# scripts/ci source the file); scripts/ci/test_qual_card.py fails when a copy drifts, and
# `py -3.11 -B scripts/ci/test_qual_card.py --sync` re-copies the canonical text into every harness.
#
# The rig has three p150a. Card M (blackhole-CEF5729692C19E6D) and card A (blackhole-3707293C249A5E67)
# are the serving pair: Ethernet-linked, mounted by every CI gate arm (lever_n_m3native_run_arm.sh), and
# reset together by the gate. Card B (blackhole-F36F768B9A5CAFA0) is reserved for another project: no
# harness may select it, so there is no default target, and nothing here opens, maps, resets or waits
# on it. /dev/tenstorrent/N numbers change across resets and switch power-cycles, and tt-smi's own board
# index is a different numbering again, so nothing here hard-codes either: the target is a board id, its
# node is resolved with readlink -f at launch and again right before the container starts, and the
# reset hint prints commands that resolve the board id when they are run (never a bare index) plus the
# PCI address that identifies the board's row in tt-smi -ls.
#
# The qwen-* hardware workflows in the qwen-two-p150a-exclusive group act on card M and card A: a run on
# either must not overlap them - check gh run list for that group before and during it.
#
#   QUAL_CARD=<board id>    the target, a name under /dev/tenstorrent/by-id (required: there is no default)
#   ALLOW_SERVING_CARD=1    required to target card M or card A (half of the serving pair); loud warning
#
#   qual_card_select      sets QUAL_CARD, QUAL_BYID, QUAL_TAG, QUAL_SERVING; refuses an unset QUAL_CARD,
#                         card B, and a serving card without the override. Touches no device: dry runs
#                         call it too.
#   qual_card_resolve     sets QUAL_NODE (readlink -f, now) and QUAL_PCI; refuses a missing board and a
#                         board id that resolves to card B's node, and treats one that resolves to a
#                         serving card's node as that serving card.
#   qual_refuse_holders   refuses while a container or a host process can reach the target (for a serving
#                         target, either end of the pair).
#   qual_card_recheck     readlink -f again right before the container starts; refuses if the node moved.
#   qual_reset_hint       the recovery lines after a hang, on stdout; it never resets anything itself.
QUAL_RESERVED_CARD=blackhole-F36F768B9A5CAFA0
QUAL_SERVING_CARDS='blackhole-CEF5729692C19E6D blackhole-3707293C249A5E67'
QUAL_TT_ROOT=/dev/tenstorrent
QUAL_BYID_ROOT=$QUAL_TT_ROOT/by-id
QUAL_SYS_ROOT=/sys

qual_card_label() {
  case ${1:-$QUAL_CARD} in
    blackhole-CEF5729692C19E6D) echo 'card M, half of the serving pair' ;;
    blackhole-3707293C249A5E67) echo 'card A, half of the serving pair' ;;
    "$QUAL_RESERVED_CARD") echo 'card B, reserved for another project' ;;
    *) echo 'a board this harness does not name' ;;
  esac
}

qual_card_select() {
  if [ -z "${QUAL_CARD:-}" ]; then
    echo "refusing: QUAL_CARD is not set; name the target's board id under $QUAL_BYID_ROOT. There is no default:" >&2
    echo "  card B is reserved for another project, and card M or card A needs ALLOW_SERVING_CARD=1." >&2
    exit 1
  fi
  case $QUAL_CARD in
    .*|*/*|*[!A-Za-z0-9._-]*)
      echo "refusing: QUAL_CARD=$QUAL_CARD is not a board id under $QUAL_BYID_ROOT" >&2
      exit 1 ;;
  esac
  if [ "$QUAL_CARD" = "$QUAL_RESERVED_CARD" ]; then
    echo "refusing: QUAL_CARD=$QUAL_CARD is $(qual_card_label); no qualification harness may use it." >&2
    echo "  Run on card M instead (QUAL_CARD=card M's board id, ALLOW_SERVING_CARD=1: qwen-c2-serving.yml's cardm)." >&2
    exit 1
  fi
  QUAL_BYID=$QUAL_BYID_ROOT/$QUAL_CARD
  case $QUAL_CARD in
    blackhole-CEF5729692C19E6D) QUAL_TAG=card-m ;;
    blackhole-3707293C249A5E67) QUAL_TAG=card-a ;;
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
    echo "  ALLOW_SERVING_CARD=1 overrides (card B is reserved for another project: it is never an alternative)." >&2
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
  node=$(readlink -f -- "$QUAL_BYID_ROOT/$QUAL_RESERVED_CARD" 2>/dev/null || true)
  if [ -n "$node" ] && [ "$node" = "$QUAL_NODE" ]; then
    echo "refusing: $QUAL_CARD resolves to the node of $(qual_card_label "$QUAL_RESERVED_CARD")" >&2
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

# The nodes nothing else may be able to reach while the target runs (qual_refuse_holders sets them): the
# target's, and for a serving target its Ethernet partner's as well (a program on either end of the pair
# reaches the other across the link), each with what it is and its device numbers. Any other board's node
# (card B's included) is never among them. A serving target whose partner has no node here is refused:
# no container's devices could then be told apart from the partner's.
QUAL_GUARD_NODES=()
QUAL_GUARD_WHAT=()
QUAL_GUARD_MAJMIN=()

qual_guard_set() {
  local card node
  QUAL_GUARD_NODES=("$QUAL_NODE")
  QUAL_GUARD_WHAT=('the target')
  if [ "$QUAL_SERVING" = 1 ]; then
    for card in $QUAL_SERVING_CARDS; do
      node=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
      [ "$node" != "$QUAL_NODE" ] || continue
      if [ -z "$node" ] || ! qual_is_char "$node"; then
        echo "refusing: $card ($(qual_card_label "$card")), the target's Ethernet partner, has no device node here:" >&2
        echo "  a container's devices cannot be told apart from it" >&2
        exit 1
      fi
      QUAL_GUARD_NODES+=("$node")
      QUAL_GUARD_WHAT+=("$(qual_card_label "$card"), the target's Ethernet partner")
    done
  fi
  QUAL_GUARD_MAJMIN=()
  for node in "${QUAL_GUARD_NODES[@]}"; do
    QUAL_GUARD_MAJMIN+=("$(qual_majmin_of "$node")")
  done
}

# Why a container (qual_refuse_holders' docker inspect lines) can reach a guarded node; nothing when it
# cannot. It names: --privileged; any device cgroup rule; a device request naming Tenstorrent (CDI); a
# device or a mount that is a guarded node, a directory holding one (/dev, /dev/tenstorrent - every CI
# gate arm mounts /dev/tenstorrent read-only for its board-mapping check, beside card M and card A), or
# the host's root; a device with a guarded node's numbers; a /dev/tenstorrent path that is not a device
# node now (which board it was given is unknowable). Anything else - a container given only card B's node,
# or any other board's - cannot reach the target and is not looked at further. With no guarded node set,
# every container is refused.
qual_container_reach() {
  local head=${1%%$'\n'*} line kind path resolved i
  if [ "${#QUAL_GUARD_NODES[@]}" = 0 ]; then
    echo 'was inspected before any target node was set (qual_guard_set)'; return 0
  fi
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
    if [ "$resolved" = / ]; then
      echo "has $path among its ${kind}s, the host's root (it holds every device node)"; return 0
    fi
    for i in "${!QUAL_GUARD_NODES[@]}"; do
      if [ "$resolved" = "${QUAL_GUARD_NODES[$i]}" ]; then
        echo "has $path among its ${kind}s, which is ${QUAL_GUARD_NODES[$i]} (${QUAL_GUARD_WHAT[$i]})"; return 0
      fi
      if [ -n "$resolved" ]; then
        case ${QUAL_GUARD_NODES[$i]} in
          "$resolved"/*)
            echo "has $path among its ${kind}s, a directory holding ${QUAL_GUARD_NODES[$i]} (${QUAL_GUARD_WHAT[$i]})"
            return 0 ;;
        esac
      fi
      if [ "$kind" = dev ] && [ -n "${QUAL_GUARD_MAJMIN[$i]}" ] \
          && [ "$(qual_majmin_of "$path")" = "${QUAL_GUARD_MAJMIN[$i]}" ]; then
        echo "is given $path, a device with the numbers of ${QUAL_GUARD_NODES[$i]} (${QUAL_GUARD_WHAT[$i]}," \
          "${QUAL_GUARD_MAJMIN[$i]})"
        return 0
      fi
    done
    case $path in
      "$QUAL_TT_ROOT"|"$QUAL_TT_ROOT"/*)
        if [ -z "$resolved" ] || ! qual_is_char "$resolved"; then
          echo "has $path among its ${kind}s, which is not a device node now (it may be the target)"; return 0
        fi ;;
    esac
  done <<< "${1#*$'\n'}"
}

# Refuses while anything else can reach the target. Containers: qual_container_reach, against the guarded
# nodes (qual_guard_set). Host processes: fuser on the guarded nodes (with sudo -n when that works, else
# this user's processes, said so) - which also sees a process in a container holding one, however the
# container was given it - up to five tries two seconds apart (the rig's telemetry exporter holds every
# card for a moment every 30 s), refusing while any holder persists.
qual_refuse_holders() {
  local id info why st out try scope
  local pre=()
  qual_guard_set
  for id in $(docker ps -q); do
    info=$(docker inspect "$id" --format '{{.Name}} {{.HostConfig.Privileged}}{{println}}{{range .HostConfig.Devices}}dev {{println .PathOnHost}}{{end}}{{range .HostConfig.DeviceCgroupRules}}rule {{println .}}{{end}}{{range .HostConfig.DeviceRequests}}req {{.Driver}} {{println .DeviceIDs}}{{end}}{{range .Mounts}}mnt {{println .Source}}{{end}}') || continue
    why=$(qual_container_reach "$info")
    if [ -n "$why" ]; then
      echo "refusing: container ${info%% *} $why" >&2
      exit 1
    fi
  done
  echo "### containers: none can reach ${QUAL_GUARD_NODES[*]} ($QUAL_CARD)"
  if ! command -v fuser >/dev/null 2>&1; then
    echo "WARN: fuser is not installed; host processes holding ${QUAL_GUARD_NODES[*]} were not checked" >&2
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
    out=$(${pre[@]+"${pre[@]}"} fuser -v "${QUAL_GUARD_NODES[@]}" 2>&1) || st=$?
    if [ "$st" != 0 ] && [ -z "$out" ]; then
      echo "### device holders on ${QUAL_GUARD_NODES[*]}: none ($scope)"
      return 0
    fi
    [ "$try" = 5 ] || sleep 2
  done
  echo "refusing: host processes hold ${QUAL_GUARD_NODES[*]} (or fuser failed):" >&2
  echo "$out" >&2
  exit 1
}

# readlink -f again right before the container starts: refuses when a board is not on the node the holder
# check cleared (the boards re-enumerated in between - a switch event, or the gate resetting the pair - so
# the old node may now be another board's). Args: [board id] [node]; the target by default.
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
    echo "  then a passing smoke run. Never card M or card A (the serving pair), and never card B (reserved)."
  fi
}
# <<< qual_card.sh
qual_card_select
R=${RESULTS:-$HOME/sdpalong/$QUAL_TAG}
name=qwen-sdpalong-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=${SWEEP_TIMEOUT_S:-3000}
case $timeout_s in ''|*[!0-9]*) echo "refusing: SWEEP_TIMEOUT_S=$timeout_s is not a number of seconds" >&2; exit 1 ;; esac
test "$timeout_s" -gt 1200 || { echo "refusing: SWEEP_TIMEOUT_S=$timeout_s leaves no room under the 600 s margin" >&2; exit 1; }

# The target's node, resolved by board id now; never shared: refuse while a container or a host
# process can reach it (a container on another board does not block).
qual_card_resolve
node=$QUAL_NODE
qual_refuse_holders

IMAGE=${IMAGE:-}
if [ -z "$IMAGE" ]; then
  test -n "${IMAGE_TAG:-}" || { echo "set IMAGE (an image reference) or IMAGE_TAG (the tag of a local qwen38-c2-<tag> image)" >&2; exit 1; }
  case $IMAGE_TAG in *[!A-Za-z0-9._-]*) echo "refusing: IMAGE_TAG=$IMAGE_TAG is not a plain tag" >&2; exit 1 ;; esac
  IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E ":qwen38-c2-${IMAGE_TAG//./\.}\$" | head -n 1 || true)
  test -n "$IMAGE" || { echo "no local image tagged qwen38-c2-$IMAGE_TAG" >&2; exit 1; }
fi

# The graft the image must carry: EXPECT_TTNNCPP_SHA256 if set (empty skips), else the K64j pin in this checkout's build script.
if [ -z "${EXPECT_TTNNCPP_SHA256+x}" ]; then
  EXPECT_TTNNCPP_SHA256=$(sed -n 's/^graft_sha=\([0-9a-f]\{64\}\)$/\1/p' "$REPO/scripts/ci/build-c2-serving-image.sh" | head -n 1)
fi
if [ -z "$EXPECT_TTNNCPP_SHA256" ]; then
  echo "WARN: no expected _ttnncpp.so sha256 (EXPECT_TTNNCPP_SHA256 empty and no graft_sha in the build script): the graft check is skipped" >&2
fi

SM=()
for pair in "$here/sdpa_tp4_long.py:sdpa_tp4_long.py" "$here/../k64j/k64j_nkv1_spike.py:k64j_nkv1_spike.py" \
            "$here/../sdpa_decode_qwen/test_sdpa_decode_qwen_card_m.py:test_sdpa_decode_qwen_card_m.py" \
            "$REPO/scripts/ci/sdpa_long_tp.py:sdpa_long_tp.py" "$REPO/scripts/ci/tp_shapes.py:tp_shapes.py"; do
  src=${pair%%:*}
  test -s "$src" || { echo "$src missing" >&2; exit 1; }
  SM+=(--mount "type=bind,src=$(cd "$(dirname "$src")" && pwd)/$(basename "$src"),dst=/bench/${pair##*:},readonly")
done

mkdir -p "$R" "$R/kcache-$stamp"
chmod 0777 "$R" "$R/kcache-$stamp"
# shellcheck disable=SC2206
extra=(${CARD_B_ARGS:-})
sweep=(--out "/results/sdpalong-$stamp.json" --device-id 0 --deadline-s "$((timeout_s - 600))")
if [ -n "$EXPECT_TTNNCPP_SHA256" ]; then
  sweep+=(--expect-binary-sha256 "$EXPECT_TTNNCPP_SHA256")
fi

echo "### sdpalong $stamp card=$QUAL_CARD ($QUAL_TAG) node=$node image=$IMAGE"
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" docker run --rm --name "$name" --network none \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g \
  --device "$node" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  "${SM[@]}" \
  --mount "type=bind,src=$R,dst=/results" \
  --mount "type=bind,src=$R/kcache-$stamp,dst=/kcache" \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8 \
  -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 \
  --workdir /bench \
  --entrypoint python3 "$IMAGE" -B /bench/sdpa_tp4_long.py \
  "${sweep[@]}" "${extra[@]}" \
  2>&1 | tee "$R/sdpalong-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status; report $R/sdpalong-$stamp.json"
case "$status" in
  124|137)
    echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
    qual_reset_hint >&2
    ;;
esac
exit "$status"
