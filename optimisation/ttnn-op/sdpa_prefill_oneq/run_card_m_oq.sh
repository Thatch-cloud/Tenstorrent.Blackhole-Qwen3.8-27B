#!/usr/bin/env bash
# The oneq card-M job (S1 of the prefill programme): oneq_card_m.py on the card QUAL_CARD names (no default; card B is refused), in the serving image,
# with graft K64j-OQ (build_k64j_oq.sh) mounted exactly as the K64j card tests mount K64j: _ttnn.so, _ttnncpp.so (both paths), attn_prep,
# nlp_concat_heads_decode, sdpa_decode and sdpa. No model, no weights: the harness calls the served causal chunked paged SDPA at the TP4 shape directly
# (6 Q heads, 1 KV head, 2048/1024/512 rows, a 4096-block page table) in four programs - stock (no program word), served (0x5EFA0003, the production
# chain), oneq (0x5EFA000B) and oneq_noc (0x5EFA000F) - and compares bytes against the stock path at every case from context 2k to 254k, then times the
# arms (see oneq_card_m.py for the matrix, the refusals, the cache check, the stress, the TP2 head-count and the 11 x 10 grid passes).
#
#   WATCHER=1 bash run_card_m_oq.sh     # THE FIRST HARDWARE PASS of the oneq program: TT_METAL_WATCHER=5 (NoC sanitiser, waypoints, asserts), rows 2048 and 1024,
#                                       # five starts, one seed, no timing, a 2,700 s container timeout; the watcher log lands in $RESULTS/watcher-<stamp>/
#   bash run_card_m_oq.sh               # the full matrix and the timing, a 5,400 s container timeout
#   OQ_CARD_DRY_RUN=1 bash run_card_m_oq.sh   # print the launch argv and exit: no node resolution, no holder check, no docker (the graft is checked when it exists)
#
# Through CI (the cardm action of qwen-c2-serving.yml; it sets QUAL_CARD, ALLOW_SERVING_CARD and RESULTS itself):
#   C2_CARDM_HARNESS=optimisation/ttnn-op/sdpa_prefill_oneq/run_card_m_oq.sh
#   C2_CARDM_ENV=IMAGE=<the served image> KOPGRAFT64=<the K64j-OQ graft dir> EXPECT_TTNNCPP_SHA256=<the K64J_OQ_TTNNCPP_SHA256 line build_k64j_oq.sh printed> [WATCHER=1]
#   C2_CARDM_ARGS=<extra oneq_card_m.py arguments, e.g. --rows 2048 --time-q-memory dram>
# By hand: QUAL_CARD=<board id> [ALLOW_SERVING_CARD=1] IMAGE=... KOPGRAFT64=... EXPECT_TTNNCPP_SHA256=... bash run_card_m_oq.sh
# Card M (the serving pair's half) needs ALLOW_SERVING_CARD=1 and a look at gh run list for the qwen-two-p150a-exclusive group before and after; a hang's
# hint resets card M and card A TOGETHER. The node is resolved by board id at launch and rechecked right before the docker run; the holder check refuses
# while a container or a host process can reach the card.
#
# Before anything is launched the graft must verify against its MANIFEST.sha256, its _ttnncpp.so must be EXPECT_TTNNCPP_SHA256 (REQUIRED for a real run)
# and carry the K64j runtime-extent literal, the chain factory's and the oneq edits' literals, and its chain reader must be the recorded one. Inside the
# container the harness checks the mapped binary again (both markers, the sha).
#
# Env: KOPGRAFT64 (default ~/opgraft-K64j-OQ), EXPECT_TTNNCPP_SHA256, IMAGE (required for a real run: no default digest), RESULTS (~/kwork64/k64j-oq/<card tag>),
# OQ_ARGS (extra harness args, appended last; CARD_B_ARGS is read the same way), WATCHER=1, WATCHDOG_S, OQ_CARD_DRY_RUN=1, QUAL_CARD, ALLOW_SERVING_CARD.
# QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is set as the K64j arm sets it. Every run gets a fresh kernel cache.
#
# ON A HANG (the WATCHDOG line and exit 3; exit 1 with 'Timeout (' in the log; or the timeout's 124 / 137): the EXIT trap removes the container. Then reset THE
# TARGET CARD ONLY with the hint printed below. This script never resets.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
chain_dir=$(cd "$here/../sdpa_prefill_chain" 2>/dev/null && pwd || echo "$here/../sdpa_prefill_chain")
G=${KOPGRAFT64:-$HOME/opgraft-K64j-OQ}
IMAGE=${IMAGE:-}
EXPECT=${EXPECT_TTNNCPP_SHA256:-}
EXTENT_MARKER='[QWEN-SDPA] runtime-extent entries='
CHAIN_MARKER='[QWEN-SDPA-PF] flags='
ONEQ_MARKER='[QWEN-SDPA-PF] oneq needs one q chunk per core'
CHAIN_READER_SHA=eecc1166a209e61dc8498149b5a4338cc278d7106f4ca68d6434f620e942f8d8

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
R=${RESULTS:-$HOME/kwork64/k64j-oq/$QUAL_TAG}
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
DRY=${OQ_CARD_DRY_RUN:-0}
case $DRY in
  0|1) ;;
  *) echo "refusing: OQ_CARD_DRY_RUN=$DRY (0 or 1)" >&2; exit 1 ;;
esac
name=qwen-oneq-card-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=5400   # the full matrix (3 row counts x 2 Q memories x 2 seeds x 2 variants x 15 starts x 4 arms), the stress, the passes and the timing

if [ "$DRY" != 1 ] && [ -z "$EXPECT" ]; then
  echo "refusing: EXPECT_TTNNCPP_SHA256 is required (the K64J_OQ_TTNNCPP_SHA256 line build_k64j_oq.sh printed for $G)" >&2
  exit 1
fi
if [ "$DRY" != 1 ] && [ -z "$IMAGE" ]; then
  echo "refusing: IMAGE is required (the serving image the graft is mounted into; there is no default digest)" >&2
  exit 1
fi
[ -n "$IMAGE" ] || IMAGE=dry-run-image

# The target's node, resolved by board id now; never shared: refuse while a container or a host
# process can reach it (a container on another board - a CI gate on the serving pair - does not block).
if [ "$DRY" = 1 ]; then
  node=$QUAL_BYID
else
  qual_card_resolve
  node=$QUAL_NODE
  qual_refuse_holders
fi

HARNESS_FILES=("$here/oneq_card_m.py" "$here/oneq_planner.py" "$here/oneq_report.py" "$here/apply_factory_ps.py")
for file in "${HARNESS_FILES[@]}" "$chain_dir/test_sdpa_prefill_chain_card_m.py" "$chain_dir/apply_factory_pf.py"; do
  test -s "$file" || { echo "refusing: $file missing (ship sdpa_prefill_oneq and sdpa_prefill_chain side by side)" >&2; exit 1; }
done

# Graft K64j-OQ, checked before anything is launched (in a dry run, only when it exists here).
if [ "$DRY" = 1 ] && [ ! -e "$G" ]; then
  echo "### dry run: $G does not exist here; the graft was not checked"
else
  for part in _ttnn.so _ttnncpp.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
    test -e "$G/$part" || { echo "refusing: $G/$part missing (the harness needs graft K64j-OQ, build_k64j_oq.sh)" >&2; exit 1; }
  done
  (cd "$G" && sha256sum -c --quiet MANIFEST.sha256 >&2) \
    || { echo "refusing: $G/MANIFEST.sha256 does not verify (not the graft its build script made)" >&2; exit 1; }
  so_sha=$(sha256sum "$G/_ttnncpp.so" | cut -c1-64)
  if [ -n "$EXPECT" ] && [ "$so_sha" != "$EXPECT" ]; then
    echo "refusing: $G/_ttnncpp.so is ${so_sha:0:16}, not ${EXPECT:0:16}" >&2
    exit 1
  fi
  for marker in "$EXTENT_MARKER" "$CHAIN_MARKER" "$ONEQ_MARKER"; do
    grep -a -q -F -- "$marker" "$G/_ttnncpp.so" \
      || { echo "refusing: $G/_ttnncpp.so lacks '$marker' (not a K64j-OQ binary)" >&2; exit 1; }
  done
  reader_sha=$(sha256sum "$G/sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp" 2>/dev/null | cut -c1-64 || true)
  if [ "$reader_sha" != "$CHAIN_READER_SHA" ]; then
    echo "refusing: $G/sdpa/.../reader_interleaved_qwen_chain.cpp is ${reader_sha:-missing}, not the chain reader $CHAIN_READER_SHA" >&2
    exit 1
  fi
  echo "### graft $G: _ttnncpp.so ${so_sha:0:16} with the K64j, chain and oneq literals, the recorded chain reader, manifest verified"
fi
KM=(--mount "type=bind,src=$G/_ttnn.so,dst=/opt/tt-metal/ttnn/ttnn/_ttnn.so,readonly"
    --mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so,readonly"
    --mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly")
for op in attn_prep:transformer/attn_prep nlp_concat_heads_decode:experimental/transformer/nlp_concat_heads_decode \
          sdpa_decode:transformer/sdpa_decode sdpa:transformer/sdpa; do
  KM+=(--mount "type=bind,src=$G/${op%%:*},dst=$OPS/${op#*:},readonly")
done

if [ "$DRY" != 1 ]; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-$stamp"
  chmod 0777 "$R" "$R/kcache-$stamp"
fi

args=(--out "/results/oneq-$stamp.json" --expect-binary-sha256 "$EXPECT")
BM=(--mount "type=bind,src=$here,dst=/bench,readonly" --mount "type=bind,src=$chain_dir,dst=/bench_pf,readonly")
WM=()
if [ "${WATCHER:-}" = "1" ]; then
  # One pass under the NoC sanitiser over every new program kind (oneq, oneq_noc) and the refusals; timing is meaningless here. OQ_ARGS, appended last, can widen it.
  timeout_s=2700
  args+=(--rows 2048,1024 --starts 0,128,2048,65536,251904 --seeds 0 --variants normal --q-memory dram --alternations 0 --no-timing
         --grids native --watchdog "${WATCHDOG_S:-120}")
  WM=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  if [ "$DRY" != 1 ]; then
    mkdir -p "$R/watcher-$stamp"
    chmod 0777 "$R/watcher-$stamp"
  fi
  echo "WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog ${WATCHDOG_S:-120} s, container timeout ${timeout_s} s"
else
  args+=(--watchdog "${WATCHDOG_S:-300}")
fi
# shellcheck disable=SC2206
extra=(${OQ_ARGS:-} ${CARD_B_ARGS:-})

# The container's script: record the binaries and the harness it runs, then the harness. One line (printf %q of a newline is $'...', which the runner test's
# shlex cannot read). unset TT_MESH_GRAPH_DESC_PATH and QWEN_C2_SERVING=0: a served image's ENV turns on the C2 boot hook, which applies the four-card mesh
# descriptor, and this ONE-card open then dies in the topology mapper.
inner='unset TT_MESH_GRAPH_DESC_PATH; sha256sum /opt/tt-metal/build_Release/lib/_ttnncpp.so /opt/tt-metal/build_Release/ttnn/_ttnncpp.so '
inner+="$OPS/transformer/sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp /bench/oneq_card_m.py /bench/oneq_planner.py "
inner+='/bench/apply_factory_ps.py 2>&1; exec python3 -B /bench/oneq_card_m.py "$@"'

argv=(docker run --rm --name "$name" --network none
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g
  --device "$node"
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G
  "${BM[@]}"
  --mount "type=bind,src=$R,dst=/results"
  --mount "type=bind,src=$R/kcache-$stamp,dst=/kcache"
  "${KM[@]}"
  ${WM[@]+"${WM[@]}"}
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  ${QUAL_TT_GRID:+-e "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=$QUAL_TT_GRID"}
  -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 -e QWEN_C2_SERVING=0
  --entrypoint sh "$IMAGE" -c "$inner"
  oneq "${args[@]}" ${extra[@]+"${extra[@]}"})
echo "### oneq-card $stamp card=$QUAL_CARD ($QUAL_TAG) node=$node image=${IMAGE:0:19} graft=$G watcher=${WATCHER:-0}"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/oneq-$stamp.log"
status=${PIPESTATUS[0]}
log=$R/oneq-$stamp.log
echo "### exit $status; report $R/oneq-$stamp.json; native log $R/oneq-$stamp.json.native.log"
if [ "${WATCHER:-}" = "1" ]; then
  wlog=$R/watcher-$stamp/watcher.log
  if [ -s "$wlog" ]; then
    echo "### watcher log $wlog: $(grep -ciE 'error|assert|tripped|sanitiz' "$wlog" || true) error/assert lines"
    grep -iE 'error|assert|tripped|sanitiz' "$wlog" | head -20 || true
  else
    echo "### no watcher log at $wlog"
  fi
fi
# The verdict and the timing lines (anchored: the harness prints them at column 0).
echo "### $(grep -E '^ONEQ_CARD_M ' "$log" | tail -1 || echo 'no ONEQ_CARD_M line')"
grep -E '^ONEQ TIME_VERDICT ' "$log" || true
# A hang: the watchdog's exit 3, the container timeout (124 / 137), or the faulthandler backstop (exit 1 with 'Timeout (' when a blocking call held the GIL).
hung=0
case "$status" in
  3|124|137) hung=1 ;;
  1) grep -qF 'Timeout (' "$log" && hung=1 ;;   # the faulthandler backstop (GIL held)
esac
if [ "$hung" = 1 ]; then
  echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
  qual_reset_hint >&2
fi
exit "$status"
