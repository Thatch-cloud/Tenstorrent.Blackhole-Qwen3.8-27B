#!/usr/bin/env bash
# Run gdn_wy_card_m.py: the NON-EXACT window-WY probe (two 16-row windows of the WY / UT form against K5-A run twice) on the card QUAL_CARD names,
# in the serving image, four users x 32 rows at the four-card geometry. One card, a 1x1 mesh, no model weights and no collectives, so the run is
# the cardm step's: C2_CARDS=pair C2_ACTIONS=cardm with this harness. NOT a CI workflow of its own. It never runs while the four cards are held by
# a quad step. Research only: no verdict of this probe licenses serving (docs/gdn-wy-probe.md).
#
#   QUAL_CARD=<board id> IMAGE_TAG=<tag> bash run_card_m.sh                       # the full probe
#   CARD_B_ARGS="--sections selftest,causality --commits 1,16" QUAL_CARD=... bash run_card_m.sh    # a reduced scope (never CONTINUE)
#
# The kernel module scripts/ci/gdn_wy_block.py does not exist yet: without it the run is the BASELINE half (selftest, the K5-A causality
# controls, timing of A and A2) and ends NO-DECISION (exit 4). If scripts/ci/gdn_wy_block* exist in this checkout they are mounted too.
#
# Target card: selected by the embedded scripts/ci/qual_card.sh block (the library every single-card harness uses): QUAL_CARD (a board id under /dev/tenstorrent/by-id,
# required, no default), the reserved board refused, a serving card refused unless ALLOW_SERVING_CARD=1. Nodes renumber, so the id is resolved to
# its node at launch and again right before the container starts.
#
# The image: IMAGE (a full image reference) if set, else the local image whose tag is qwen38-c2-$IMAGE_TAG (IMAGE_TAG comes from the job's
# C2_CARDM_ENV, a plain tag such as the window's), looked up in the local store at run time so no registry name is written here.
#
# Env: REPO (this checkout; default three levels up), IMAGE or IMAGE_TAG, RESULTS (default ~/kwork64/wy-probe/<card tag>), CARD_B_ARGS (extra probe
# arguments, appended last), WATCHER (0 or 1: TT_METAL_WATCHER=10).
#
# Read: the last stdout line (one JSON object, kind gdn-wy-probe) and the GDN_WY verdict line above it; results/gdn-wy-<stamp>.json holds every
# case, the timing arms, the SRAM figures and the causality and packing cases.
#
# ON A HANG (the timeout's 124 / 137, or the probe's own watchdog exit 3): the EXIT trap removes the container; then reset THE TARGET CARD ONLY
# with the hint printed below. This script never resets anything itself.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$here/../../.." && pwd)}

# The card first, before docker or any file is touched: the canonical library, embedded byte for byte (scripts/ci/test_qual_card.py --sync keeps it).
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
R=${RESULTS:-$HOME/kwork64/wy-probe/$QUAL_TAG}
name=qwen-wy-probe-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=7200
WATCHER=${WATCHER:-0}
case $WATCHER in 0|1) ;; *) echo "refusing: WATCHER=$WATCHER is not 0 or 1" >&2; exit 1 ;; esac

# The library echoes the board id, node and PCI address on stdout; this repo is public and the step tees stdout into a public log, so those lines go to
# /dev/null here (its refusals are on stderr and stay). Nothing below prints a board id, node, PCI address or image reference.
qual_card_resolve >/dev/null
node=$QUAL_NODE
qual_refuse_holders >/dev/null

IMAGE=${IMAGE:-}
if [ -z "$IMAGE" ]; then
  test -n "${IMAGE_TAG:-}" || { echo "set IMAGE (an image reference) or IMAGE_TAG (the tag of a local qwen38-c2-<tag> image)" >&2; exit 1; }
  case $IMAGE_TAG in *[!A-Za-z0-9._-]*) echo "refusing: IMAGE_TAG=$IMAGE_TAG is not a plain tag" >&2; exit 1 ;; esac
  IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E ":qwen38-c2-${IMAGE_TAG//./\.}\$" | head -n 1 || true)
  test -n "$IMAGE" || { echo "no local image tagged qwen38-c2-$IMAGE_TAG" >&2; exit 1; }
fi

SM=()
mount_file() {
  local src=$1 dst=$2
  test -s "$src" || { echo "$src missing" >&2; exit 1; }
  if grep -q $'\r' "$src"; then echo "refusing: $src has CR line endings; its sha256 is its identity" >&2; exit 1; fi
  SM+=(--mount "type=bind,src=$src,dst=/bench/$dst,readonly")
}
# gdn_v5_card_m.py imports gdn_seq_block_split at module level, so the split build's files ride along (never launched here).
for file in gdn_seq_block.py gdn_seq_block_compute.cpp gdn_seq_block_reader.cpp gdn_seq_block_writer.cpp \
    gdn_seq_block_split.py gdn_seq_block_split_compute.cpp gdn_seq_block_split_reader.cpp gdn_seq_block_split_writer.cpp \
    gdn_seq_block_device_test.py gdn_tp4_card_test.py gdn_user_batch.py gdn_user_batch_tp.py gdn_multitoken.py tp_shapes.py \
    verify_trace_t1.py gdn_wy_model.py; do
  mount_file "$REPO/scripts/ci/$file" "$file"
done
mount_file "$REPO/optimisation/ttnn-op/v5split/gdn_v5_card_m.py" gdn_v5_card_m.py
mount_file "$REPO/optimisation/ttnn-op/wy_probe/gdn_wy_card_m.py" gdn_wy_card_m.py
kernel_files=0
for src in "$REPO"/scripts/ci/gdn_wy_block*; do
  [ -e "$src" ] || continue
  mount_file "$src" "$(basename "$src")"
  kernel_files=$((kernel_files + 1))
done
echo "### window kernel files in this checkout: $kernel_files (0 = baseline half only)"

mkdir -p "$R" "$R/kcache-$stamp"
chmod 0777 "$R" "$R/kcache-$stamp"
extra=(${CARD_B_ARGS:-})
watch=()
if [ "$WATCHER" = 1 ]; then
  watch=(-e TT_METAL_WATCHER=10 -e TT_METAL_WATCHER_APPEND=1)
fi

echo "### wy-probe $stamp card=$QUAL_TAG image-tag=${IMAGE##*:} watcher=$WATCHER"   # the tag only: never the resolved reference, board id, node or address (public log)
qual_card_recheck >/dev/null   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" docker run --rm --name "$name" --network none \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 1024 --memory 32g --cpus 8 --shm-size 4g \
  --device "$node" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  "${SM[@]}" \
  --mount "type=bind,src=$R,dst=/results" \
  --mount "type=bind,src=$R/kcache-$stamp,dst=/kcache" \
  -e QWEN_FAST_TP=4 -e QWEN_FAST_VERIFY_T1=1 \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8 \
  ${watch[@]+"${watch[@]}"} \
  --workdir /bench \
  -e QWEN_C2_SERVING=0 --entrypoint env "$IMAGE" -u TT_MESH_GRAPH_DESC_PATH python3 -B /bench/gdn_wy_card_m.py \
  --out "/results/gdn-wy-$stamp.json" "${extra[@]}" \
  2>&1 | tee "$R/gdn-wy-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status (0 CONTINUE, 10 KILL, 3 watchdog, 4 NO-DECISION; 1 is a launcher refusal or a crash, never a finding); report gdn-wy-$stamp.json in the results directory"
case "$status" in
  3|124|137)
    echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
    qual_reset_hint > "$R/reset-hint-$stamp.txt" 2>&1   # names the board: kept on the rig, not printed into the public log
    echo "recovery commands for the target card are in reset-hint-$stamp.txt in the results directory on the rig" >&2
    ;;
esac
exit "$status"
