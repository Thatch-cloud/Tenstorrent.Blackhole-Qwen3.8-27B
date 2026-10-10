#!/usr/bin/env bash
# Build ~/opgraft-K64j-OQ on the rig HOST: the K64j graft (~/opgraft-K64j: the binary the serving images bake) plus ONE change, the prefill SDPA factory's
# oneq edits (apply_factory_ps.py PS0-PS5: flag 0x8 of the per-call word gives every core ONE q chunk, no causal pairing), through the ttbuild container.
# It is the experiment's graft, never a serving binary: the oneq card job (run_card_m_oq.sh) is what mounts it. It opens no device and launches no device container.
#
#   K64j-OQ = K64j's contents, byte for byte (attn_prep, nlp_concat_heads_decode, sdpa_decode, sdpa, _ttnn.so, ...)
#           + a _ttnncpp.so rebuilt in ttbuild from K64j's OWN source state (the decode factory at K64j, the prefill factory as K64g/K64i/K64j built it, the four
#             K64j kernels and the chain reader: build_k64j.sh's staging, step for step) with the prefill factory replaced by the oneq one (apply_factory_ps.py).
#   Nothing else changes: the chain reader, the compute and writer kernels and every header of sdpa/ are K64j's (the oneq edits are host-side only), so the graft's
#   sdpa/ directory is K64j's byte for byte.
#   A graft .so replaces the WHOLE binary (memory graft-so-drops-image-patches), so this build starts from K64j's exact source tree and patches, adds only the oneq
#   edits, and REFUSES to publish unless the result is K64j with that change and nothing else (the checks, in order):
#     1. the base is K64j: its MANIFEST verifies and its _ttnncpp.so is the sha packed_any_admission.K64J_TTNNCPP_SHA256 records (the image's binary);
#     2. apply_factory_ps.py is the recorded script, the saved fd8c0676 prefill factory through apply_factory_pf.py gives PF_FACTORY (K64j's) and through
#        apply_factory_ps.py gives PS_FACTORY (this build's);
#     3. REPRODUCTION (unless OQ_SKIP_REPRODUCE=1): K64j alone, rebuilt in this ttbuild, is compared with the base _ttnncpp.so. Byte-identical means the only
#        delta of the result is the patch; not identical is a WARNING (OQ_REQUIRE_REPRODUCE=1 makes it fatal) and checks 4-6 carry the weight;
#     4. every distinct QWEN_ / [QWEN- string of the base survives in the result, and the ONLY new ones are the two literals the patch adds (each present, none in the
#        base; every other new line must contain one of them); the K64j, K64i-slice and prefill-chain markers and the tree-scratch factory are all present;
#     5. the exported symbols (nm -D) of the result equal the base's, where nm exists;
#     6. the graft is the base directory with exactly one file replaced (_ttnncpp.so): _ttnn.so and every op directory are identical, and the new manifest
#        differs from the base's in that one line; with OQ_IMAGES set, every QWEN string of each image's _ttnncpp.so is in the result too.
#   Then ttbuild is restored (the factories from the saved bases, the kernels), rebuilt, and checked clean.
#
# THROUGH CI (the owner is offsite): the cardm action of qwen-c2-serving.yml
#   C2_CARDM_HARNESS=optimisation/ttnn-op/sdpa_prefill_oneq/build_k64j_oq.sh
#   C2_CARDM_ENV=OQ_BUILD_DRY_RUN=1      # first: every check the host can make, every command printed, nothing run
#   C2_CARDM_ENV=                        # then the build: about three ninja runs of 30 to 60 s
# runs it on the rig host from the checkout root with QUAL_CARD = card M and ALLOW_SERVING_CARD=1 (the card is context only: nothing here opens a device).
# BY HAND on the rig: QUAL_CARD=<board id> [ALLOW_SERVING_CARD=1] bash optimisation/ttnn-op/sdpa_prefill_oneq/build_k64j_oq.sh [graft dir]   (no default board, card B refused)
#
# Env (C2_CARDM_ENV carries NAME=value words without spaces):
#   OQ_BASE_GRAFT        the K64j graft to layer on (~/opgraft-K64j). NEVER written, never moved. OQ_BASE_SHA256: its _ttnncpp.so (default: the sha
#                        scripts/ci/packed_any_admission.py records; empty skips, last resort)
#   OQ_GRAFT             the graft to write (~/opgraft-K64j-OQ; a first argument also sets it). Built in <it>.partial and moved into place only after every
#                        check; an existing one is refused (move it away yourself): no existing graft is ever overwritten
#   OQ_WORK_ROOT         work-<stamp>/ (backups, copies out of ttbuild, strings lists): ~/kwork64/k64j-oq
#   K64J_DECODE_BASE_DIR / K64J_PF_BASE_DIR   the saved 3e0a69af decode and fd8c0676 prefill factories (build_k64j.sh's: ~/kwork64/k64f, ~/kwork64/k64g)
#   OQ_IMAGES            images (comma or space separated) whose _ttnncpp.so QWEN strings must also survive; default none (the base graft IS the image's binary)
#   OQ_SKIP_REPRODUCE=1  skip the K64j-alone build (one ninja run less); OQ_REQUIRE_REPRODUCE=1  a K64j rebuild that is not byte-identical is fatal
#   OQ_ALLOW_SYMBOL_DELTA=1  a different exported-symbol set is a warning (it should be identical: the patch adds no function)
#   OQ_KEEP_STAGED=1     leave ttbuild staged; OQ_BUILD_DRY_RUN=1  print every docker, cp, mv, rm, touch, mkdir and generator command as '### run: ...' and
#                        run none of them (the read-only checks of the checkout, the base and the saved bases still run)
#   QUAL_CARD            only validated (qual_card_select): this build never opens a card
set -euo pipefail
export LC_ALL=C

# ---- K64j's recorded shas, kernel lists and markers: ONE source, build_k64j.sh's own header (a copy would drift) ----
S=$(cd "$(dirname "$0")" && pwd)
K64J_DIR=$S/../k64j
K64J_BUILD=$K64J_DIR/build_k64j.sh
test -s "$K64J_BUILD" || { echo "refusing: $K64J_BUILD missing (the checkout carries k64j and sdpa_prefill_oneq side by side)" >&2; exit 1; }
eval "$(sed -n '/^# ---- recorded shas/,/^# >>> qual/p' "$K64J_BUILD" | sed '$d')"
test -n "${FACTORY_K64J:-}" && test -n "${PF_FACTORY:-}" && [ "${#DECODE_KERNELS[@]}" = 4 ] \
  || { echo "refusing: could not read K64j's recorded shas from $K64J_BUILD" >&2; exit 1; }

# ---- this build's own recorded shas (test_build_k64j_oq.py keeps them equal to apply_factory_ps.py and the patched file) ----
PS_FACTORY=37b9d966b9c85a09eb1e8941b6af789b5a2c9263fea9a8b929284560ede689ce
PS_SCRIPT=d85fc5f0130a292da0e34072248690c45294d13bc7e98be2e2b7ba03b2b48b12
OQ_LITERAL_1='[QWEN-SDPA-PF] oneq needs one q chunk per core: {} q chunks on {} cores'
OQ_LITERAL_2='[QWEN-SDPA-PF] oneq=1 q_chunks={} cores={} chunks_per_core=1 chains={} members={}'
OQ_LITERALS=("$OQ_LITERAL_1" "$OQ_LITERAL_2")

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
DRY=${OQ_BUILD_DRY_RUN:-0}
case $DRY in
  0|1) ;;
  *) echo "refusing: OQ_BUILD_DRY_RUN=$DRY (0 or 1)" >&2; exit 1 ;;
esac
if [ -n "${CARD_B_ARGS:-}" ]; then
  echo "refusing: CARD_B_ARGS='$CARD_B_ARGS': build_k64j_oq.sh takes no arguments from the job file; set OQ_* in C2_CARDM_ENV" >&2
  exit 1
fi
if [ $# -gt 1 ]; then
  echo "refusing: build_k64j_oq.sh takes at most one argument, the graft directory (got $#)" >&2
  exit 1
fi

CHECKOUT=$(cd "$S/../../.." && pwd)
for dir in sdpa_decode_slice sdpa_decode_qwen sdpa_prefill_chain; do
  test -d "$S/../$dir" || { echo "refusing: $S/../$dir missing (the checkout carries k64j and its siblings)" >&2; exit 1; }
done
SL=$(cd "$S/../sdpa_decode_slice" && pwd)
DS=$(cd "$S/../sdpa_decode_qwen" && pwd)
PS=$(cd "$S/../sdpa_prefill_chain" && pwd)
KS=$DS/stage3
KJ=$K64J_DIR/kernels
PATCHER=$S/apply_factory_ps.py
ADMISSION=$CHECKOUT/scripts/ci/packed_any_admission.py
DBASE=${K64J_DECODE_BASE_DIR:-$HOME/kwork64/k64f}
PBASE=${K64J_PF_BASE_DIR:-$HOME/kwork64/k64g}
BASE=${OQ_BASE_GRAFT:-$HOME/opgraft-K64j}
GRAFT=${1:-${OQ_GRAFT:-$HOME/opgraft-K64j-OQ}}
G=$GRAFT.partial
WROOT=${OQ_WORK_ROOT:-$HOME/kwork64/k64j-oq}
IMAGES=${OQ_IMAGES:-}
IMAGES=${IMAGES//,/ }
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
D=$OPS/sdpa_decode
F=$D/device/sdpa_decode_program_factory.cpp
KD=$D/device/kernels
P=$OPS/sdpa
PF=$P/device/sdpa_program_factory.cpp
PK=$P/device/kernels
TB_SO=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so
SAVED_DECODE=$DBASE/sdpa_decode_program_factory.cpp.$FACTORY_BASE
SAVED_PF=$PBASE/sdpa_program_factory.cpp.$PF_BASE
# The expected base binary: the one the repo records for the served image, unless the caller gives another (empty skips).
if [ -n "${OQ_BASE_SHA256+x}" ]; then
  BASE_SHA=$OQ_BASE_SHA256
else
  BASE_SHA=$(grep -oE "^K64J_TTNNCPP_SHA256 = '[0-9a-f]{64}'" "$ADMISSION" 2>/dev/null | grep -oE '[0-9a-f]{64}' || true)
  test -n "$BASE_SHA" || { echo "refusing: no K64J_TTNNCPP_SHA256 in $ADMISSION (give OQ_BASE_SHA256)" >&2; exit 1; }
fi

canon() {  # one absolute spelling of a path
  local path=$1
  if command -v cygpath >/dev/null 2>&1; then
    path=$(cygpath -u -- "$path")
  fi
  readlink -m -- "$path"
}
under() {  # path root -> true when path is root or inside it
  local path root
  path=$(canon "$1")
  root=$(canon "$2")
  case "$path/" in "$root/"*) return 0 ;; esac
  return 1
}
case $GRAFT in
  ''|/|.|..) echo "refusing: graft directory '$GRAFT'" >&2; exit 1 ;;
esac
if under "$GRAFT" "$BASE" || under "$BASE" "$GRAFT"; then
  echo "refusing: the graft $GRAFT and the base $BASE overlap (the base is K64j: it is copied, never written)" >&2
  exit 1
fi
for path in "$GRAFT" "$WROOT"; do
  if under "$path" "$CHECKOUT"; then
    echo "refusing: $path is inside the checkout $CHECKOUT (the next checkout cleans it)" >&2
    exit 1
  fi
done
if [ -e "$GRAFT" ]; then
  echo "refusing: $GRAFT exists (a graft is never overwritten: move it away, or name another with OQ_GRAFT)" >&2
  exit 1
fi

stamp=$(date +%Y%m%dT%H%M%S)
exec 3>&1
x() {
  if [ "$DRY" = 1 ]; then
    echo "### run: $(printf '%q ' "$@")" >&3
    return 0
  fi
  "$@"
}
fail() {
  echo "FAIL: $*" >&2
  exit 1
}
need() {  # label actual expected (an empty actual in a dry run is a command that was not run)
  if [ "$DRY" = 1 ] && [ -z "$2" ]; then
    echo "dry   $1: must be ${3:0:16}"
    return 0
  fi
  if [ "$2" != "$3" ]; then
    echo "FAIL: $1 is ${2:-missing}, expected $3" >&2
    exit 1
  fi
  echo "ok    $1 ${2:0:16}"
}
hsha() { sha256sum "$1" 2>/dev/null | cut -c1-64 || true; }
csha() { x docker exec ttbuild sha256sum "$1" | cut -c1-64; }
cexists() { x docker exec ttbuild test -e "$1"; }
count() { strings "$1" | grep -cF -- "$2" || true; }
qwen_strings() { strings "$1" | grep -E 'QWEN_|\[QWEN-' | sort -u || true; }
symbols() { nm -D --defined-only "$1" 2>/dev/null | awk '{print $NF}' | sort -u || true; }

nearest() {
  local path
  path=$(canon "$1")
  while [ ! -e "$path" ] && [ "$path" != / ]; do
    path=$(dirname "$path")
  done
  echo "$path"
}
echo "### host $(hostname 2>/dev/null || echo unknown) user $(id -un 2>/dev/null || echo unknown) HOME $HOME"
for path in "$GRAFT" "$WROOT" "$DBASE" "$PBASE"; do
  at=$(nearest "$path")
  free=$(df -Pk "$at" 2>/dev/null | awk 'NR == 2 { printf "%d MB free", $4 / 1024 }' || true)
  if [ -w "$at" ]; then
    echo "### $path: its nearest existing directory $at is writable, ${free:-free space unknown}"
  elif [ "$DRY" = 1 ]; then
    echo "### $path: its nearest existing directory $at is NOT WRITABLE by $(id -un 2>/dev/null) (the build would stop here)"
  else
    fail "$path: its nearest existing directory $at is not writable by $(id -un 2>/dev/null)"
  fi
done

if [ "$DRY" = 1 ]; then
  W=$(mktemp -d "${TMPDIR:-/tmp}/k64j-oq-dry.XXXXXX")
  echo "### build_k64j_oq DRY RUN $(date -Is) graft=$GRAFT base=$BASE card context $QUAL_CARD ($QUAL_TAG): nothing is built"
  x mkdir -p "$WROOT/work-$stamp/backup"
else
  command -v docker >/dev/null 2>&1 || fail "docker is not on PATH"
  command -v strings >/dev/null 2>&1 || fail "strings (binutils) is not on PATH: the QWEN-string checks need it"
  W=$WROOT/work-$stamp
  mkdir -p "$W/backup"
  echo "### build_k64j_oq $(date -Is) work=$W graft=$GRAFT base=$BASE card context $QUAL_CARD ($QUAL_TAG): no device is opened"
fi
command -v python3 >/dev/null 2>&1 || fail "python3 is not on PATH: the factory generators are python"
HAVE_NM=0
command -v nm >/dev/null 2>&1 && HAVE_NM=1
[ "$HAVE_NM" = 1 ] || echo "WARN  nm is not on PATH: check 5 (the exported symbols) will be skipped"

# The EXIT trap: ttbuild's sources back (from the first staging copy on), and the dry run's scratch removed.
armed=0
restored=0
built=0
rebuilt=0
restore_ttbuild() {
  if [ "$armed" = "0" ] || [ "$restored" = "1" ] || [ "${OQ_KEEP_STAGED:-}" = "1" ]; then
    return 0
  fi
  x docker cp "$SAVED_DECODE" "ttbuild:$F"
  x docker cp "$SAVED_PF" "ttbuild:$PF"
  for kernel in "${DECODE_KERNELS[@]}"; do
    if [ -e "$W/backup/$(basename "$kernel").replaced" ]; then
      x docker cp "$W/backup/$(basename "$kernel").replaced" "ttbuild:$KD/$kernel"
    else
      x docker exec ttbuild rm -f "$KD/$kernel"
    fi
  done
  if [ -e "$W/backup/reader_interleaved_qwen_chain.cpp.replaced" ]; then
    x docker cp "$W/backup/reader_interleaved_qwen_chain.cpp.replaced" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
  else
    x docker exec ttbuild rm -f "$PK/dataflow/reader_interleaved_qwen_chain.cpp"
  fi
  # docker cp keeps the saved files' old mtimes: touch, or ninja calls the unity TUs up to date.
  x docker exec ttbuild touch "$F" "$PF"
  restored=1
  if [ "$DRY" = 1 ]; then
    echo "ttbuild restore: the commands above (the factories from the saved bases, the qwen kernel files removed or put back)"
  else
    echo "ttbuild restored: decode $(csha $F | cut -c1-16), prefill $(csha $PF | cut -c1-16) (touched)"
  fi
}
on_exit() {
  local status=$?
  restore_ttbuild || echo "WARN: ttbuild restore failed; check $F and $PF in ttbuild" >&2
  if [ "$DRY" != 1 ] && [ "$built" = "1" ] && [ "$rebuilt" = "0" ] && [ "${OQ_KEEP_STAGED:-}" != "1" ]; then
    echo "WARN  ttbuild build_Release still holds the staged objects; the next ninja run recompiles the touched factories, but do not copy a .so out of build_Release before one" >&2
  fi
  if [ "$DRY" = 1 ]; then
    rm -rf "$W"
  elif [ "$status" != 0 ] && [ -e "$G" ]; then
    echo "note  $G is left for inspection; the next run removes it" >&2
  fi
}
trap on_exit EXIT

# ---------- 1. preflight: the checkout (the K64j sources, the generators) ----------
for path in "$KJ/dataflow/reader_decode_qwen.cpp" "$KJ/dataflow/reader_decode_qwen_slice.cpp" \
            "$KJ/compute/sdpa_flash_decode_qwen.cpp" "$KJ/dataflow/writer_decode_qwen_slice.cpp" \
            "$K64J_DIR/apply_factory_k64j.py" "$PS/apply_factory_pf.py" "$PS/reader_interleaved_qwen_chain.cpp" \
            "$PATCHER" "$ADMISSION"; do
  test -s "$path" || fail "$path missing"
  if grep -q $'\r' "$path"; then fail "$path has CRLF line endings"; fi
done
for kernel in "${DECODE_KERNELS[@]}"; do
  need "shipped K64j $kernel" "$(hsha "$KJ/$kernel")" "$(k64j_sha "$kernel")"
done
need "shipped reader_interleaved_qwen_chain.cpp" "$(hsha "$PS/reader_interleaved_qwen_chain.cpp")" $PF_READER
need "shipped apply_factory_ps.py" "$(hsha "$PATCHER")" $PS_SCRIPT

# ---------- 2. preflight: the base graft (K64j; read-only) ----------
base_so=
if [ "$DRY" = 1 ] && [ ! -e "$BASE" ]; then
  echo "### dry run: $BASE does not exist here; the base graft was not checked"
else
  for part in _ttnncpp.so _ttnn.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
    test -e "$BASE/$part" || fail "$BASE/$part missing (the base is K64j, build_k64j.sh)"
  done
  (cd "$BASE" && sha256sum -c --quiet MANIFEST.sha256) || fail "$BASE/MANIFEST.sha256 does not verify"
  echo "ok    $BASE/MANIFEST.sha256 verifies"
  base_so=$(hsha "$BASE/_ttnncpp.so")
  if [ -n "$BASE_SHA" ]; then
    need "base graft _ttnncpp.so (K64j, the served image's)" "$base_so" "$BASE_SHA"
  else
    echo "WARN  OQ_BASE_SHA256 is empty: the base graft is not checked against the recorded K64j binary"
  fi
  BK=$BASE/sdpa_decode/device/kernels
  for kernel in "${DECODE_KERNELS[@]}"; do
    need "base graft $kernel (K64j)" "$(hsha "$BK/$kernel")" "$(k64j_sha "$kernel")"
  done
  need "base graft sdpa/ chain reader" "$(hsha "$BASE/sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp")" $PF_READER
  if command -v strings >/dev/null 2>&1; then
    for marker in "${K64J_MARKERS[@]}" "${STAGE4_MARKERS[@]}" "${PF_MARKERS[@]}" QWEN_SDPA_TREE_SCRATCH_ROUNDS; do
      test "$(count "$BASE/_ttnncpp.so" "$marker")" -ge 1 || fail "$BASE/_ttnncpp.so lacks '$marker' (not K64j)"
    done
    for literal in "${OQ_LITERALS[@]}"; do
      test "$(count "$BASE/_ttnncpp.so" "$literal")" -eq 0 || fail "$BASE/_ttnncpp.so already has '$literal' (it is not K64j)"
    done
    qwen_strings "$BASE/_ttnncpp.so" > "$W/base-qwen-strings.txt"
    echo "ok    base graft ${base_so:0:16}: K64j markers present, $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings recorded"
  fi
  if [ "$HAVE_NM" = 1 ]; then
    symbols "$BASE/_ttnncpp.so" > "$W/base-symbols.txt"
    echo "ok    base graft: $(wc -l < "$W/base-symbols.txt") exported symbols recorded"
  fi
fi

# ---------- 3. preflight: the saved factory bases, ttbuild ----------
for pair in "decode|$SAVED_DECODE|$FACTORY_BASE" "prefill|$SAVED_PF|$PF_BASE"; do
  label=${pair%%|*}
  rest=${pair#*|}
  path=${rest%|*}
  if [ -e "$path" ]; then
    need "saved $label base factory" "$(hsha "$path")" "${rest##*|}"
  else
    echo "note  no saved $label base at $path yet: step 4 saves it from ttbuild when ttbuild holds the base, else fails"
  fi
done
if [ "$DRY" = 1 ]; then
  x docker ps --format '{{.Names}}'
else
  docker ps --format '{{.Names}}' | grep -qx ttbuild || fail "ttbuild container is not running"
  echo "ok    ttbuild is running"
fi
dcur=$(csha $F)
pcur=$(csha $PF)
if [ "$DRY" != 1 ]; then
  case "$dcur" in
    "$FACTORY_BASE"|"$FACTORY_QWEN_STAGE1"|"$FACTORY_QWEN_STAGE3"|"$FACTORY_QWEN_STAGE4"|"$FACTORY_K64J") ;;
    "$FACTORY_UNPATCHED") fail "ttbuild decode factory is the unpatched 05708e6d; run the K64d build (tree-scratch) first" ;;
    *) fail "unexpected ttbuild decode factory ${dcur:-missing}" ;;
  esac
  case "$pcur" in
    "$PF_BASE"|"$PF_FACTORY") ;;
    "$PS_FACTORY") echo "note  ttbuild prefill factory is already the oneq one (an OQ_KEEP_STAGED run); it is restored from the saved base at the end" ;;
    *) fail "unexpected ttbuild prefill factory ${pcur:-missing}" ;;
  esac
  echo "ok    ttbuild: decode ${dcur:0:16}, prefill ${pcur:0:16}"
fi
need "ttbuild reader_decode_all.cpp" "$(csha $KD/dataflow/reader_decode_all.cpp)" $READER_ALL
need "ttbuild sdpa_flash_decode.cpp" "$(csha $KD/compute/sdpa_flash_decode.cpp)" $COMPUTE_ALL
need "ttbuild writer_decode_all.cpp" "$(csha $KD/dataflow/writer_decode_all.cpp)" $WRITER_ALL
need "ttbuild decode dataflow_common.hpp" "$(csha $KD/dataflow/dataflow_common.hpp)" $DATAFLOW_COMMON
need "ttbuild rt_args_common.hpp" "$(csha $KD/rt_args_common.hpp)" $RT_ARGS_COMMON
need "ttbuild reader_interleaved.cpp" "$(csha $PK/dataflow/reader_interleaved.cpp)" $PF_READER_BASE
if [ -n "$IMAGES" ]; then
  for image in $IMAGES; do
    if [ "$DRY" = 1 ]; then
      x docker image inspect "$image"
    else
      docker image inspect "$image" >/dev/null 2>&1 || fail "image $image is not present"
    fi
  done
fi

# ---------- 4. stage K64j exactly as build_k64j.sh does, and the oneq prefill factory (backing up anything replaced) ----------
x docker cp ttbuild:$F "$W/backup/sdpa_decode_program_factory.cpp.${dcur:0:8}"
x docker cp ttbuild:$PF "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}"
x mkdir -p "$DBASE" "$PBASE"
if [ "$dcur" = "$FACTORY_BASE" ] && [ ! -e "$SAVED_DECODE" ]; then
  x cp "$W/backup/sdpa_decode_program_factory.cpp.${dcur:0:8}" "$SAVED_DECODE"
fi
if [ "$pcur" = "$PF_BASE" ] && [ ! -e "$SAVED_PF" ]; then
  x cp "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}" "$SAVED_PF"
fi
if [ "$DRY" != 1 ]; then
  test -s "$SAVED_DECODE" || fail "no saved 3e0a69af decode factory in $DBASE (and ttbuild does not hold it)"
  test -s "$SAVED_PF" || fail "no saved fd8c0676 prefill factory in $PBASE (and ttbuild does not hold it)"
fi
need "saved decode base factory" "$( [ "$DRY" = 1 ] || hsha "$SAVED_DECODE")" $FACTORY_BASE
need "saved prefill base factory" "$( [ "$DRY" = 1 ] || hsha "$SAVED_PF")" $PF_BASE
x python3 -B "$K64J_DIR/apply_factory_k64j.py" "$SAVED_DECODE" --out "$W/sdpa_decode_program_factory.cpp"
need "patched decode factory (K64j)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_decode_program_factory.cpp")" $FACTORY_K64J
x python3 -B "$PS/apply_factory_pf.py" "$SAVED_PF" --out "$W/sdpa_program_factory.pf.cpp"
need "patched prefill factory (K64j's)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_program_factory.pf.cpp")" $PF_FACTORY
x python3 -B "$PATCHER" "$SAVED_PF" --out "$W/sdpa_program_factory.oq.cpp"
need "oneq prefill factory (PS0-PS5 over the saved base)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_program_factory.oq.cpp")" $PS_FACTORY
x python3 -B "$PATCHER" "$W/sdpa_program_factory.pf.cpp" --out "$W/sdpa_program_factory.oq2.cpp"
need "oneq prefill factory (PS0-PS5 over K64j's PF factory)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_program_factory.oq2.cpp")" $PS_FACTORY

armed=1
x docker cp "$W/sdpa_decode_program_factory.cpp" "ttbuild:$F"
x docker cp "$W/sdpa_program_factory.pf.cpp" "ttbuild:$PF"      # K64j's prefill factory first: build A is K64j alone
x docker exec ttbuild touch "$F" "$PF"
need "ttbuild decode factory now" "$(csha $F)" $FACTORY_K64J
need "ttbuild prefill factory now (K64j's)" "$(csha $PF)" $PF_FACTORY
for kernel in "${DECODE_KERNELS[@]}"; do
  if cexists "$KD/$kernel"; then
    x docker cp "ttbuild:$KD/$kernel" "$W/backup/$(basename "$kernel").replaced"
  fi
  x docker cp "$KJ/$kernel" "ttbuild:$KD/$kernel"
done
if cexists "$PK/dataflow/reader_interleaved_qwen_chain.cpp"; then
  x docker cp "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp" "$W/backup/reader_interleaved_qwen_chain.cpp.replaced"
fi
x docker cp "$PS/reader_interleaved_qwen_chain.cpp" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
for kernel in "${DECODE_KERNELS[@]}"; do
  need "ttbuild $kernel" "$(csha "$KD/$kernel")" "$(k64j_sha "$kernel")"
done
need "ttbuild reader_interleaved_qwen_chain.cpp" "$(csha $PK/dataflow/reader_interleaved_qwen_chain.cpp)" $PF_READER

# ---------- 5. build A: K64j alone (the reproduction of the base) ----------
built=1
a_sha=
if [ "${OQ_SKIP_REPRODUCE:-}" = "1" ]; then
  echo "WARN  OQ_SKIP_REPRODUCE=1: K64j was not rebuilt alone; the delta to the base is judged by checks 4-6 only"
else
  echo "### ninja A (K64j alone) start $(date -Is)"
  x docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
  echo "### ninja A done $(date -Is)"
  x docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$W/A-ttnncpp.so"
  if [ "$DRY" != 1 ]; then
    a_sha=$(hsha "$W/A-ttnncpp.so")
    if [ "$a_sha" = "$base_so" ]; then
      echo "ok    REPRODUCED: K64j rebuilt alone in this ttbuild is byte-identical to the base graft's _ttnncpp.so (${a_sha:0:16}): the only delta of the result is the patch"
    elif [ "${OQ_REQUIRE_REPRODUCE:-}" = "1" ]; then
      fail "K64j rebuilt alone (${a_sha:0:16}) is not byte-identical to the base graft's (${base_so:0:16}) and OQ_REQUIRE_REPRODUCE=1"
    else
      echo "WARN  K64j rebuilt alone (${a_sha:0:16}) is NOT byte-identical to the base graft's (${base_so:0:16}): the toolchain, paths or timestamps differ; checks 4-6 carry the weight"
    fi
  fi
fi

# ---------- 6. build B: K64j + the oneq prefill factory ----------
x docker cp "$W/sdpa_program_factory.oq.cpp" "ttbuild:$PF"
x docker exec ttbuild touch "$PF"
need "ttbuild prefill factory now (oneq)" "$(csha $PF)" $PS_FACTORY
echo "### ninja B (K64j + oneq) start $(date -Is)"
x docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
echo "### ninja B done $(date -Is)"
x docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$W/B-ttnncpp.so"
x docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnn.so "$W/B-ttnn.so"

# ---------- 7. assemble: the base directory with exactly one file replaced ----------
x rm -rf "$G"
x mkdir -p "$G"
x cp -a "$BASE/." "$G/"
x rm -f "$G/MANIFEST.sha256"
x cp "$W/B-ttnncpp.so" "$G/_ttnncpp.so"

# ---------- 8. restore ttbuild (before the verify; the EXIT trap covers a failure) ----------
if [ "${OQ_KEEP_STAGED:-}" = "1" ]; then
  echo "note  OQ_KEEP_STAGED=1: ttbuild keeps the staged factories and kernels"
else
  restore_ttbuild
  need "ttbuild decode factory restored" "$(csha $F)" $FACTORY_BASE
  need "ttbuild prefill factory restored" "$(csha $PF)" $PF_BASE
  echo "### ninja (restore build_Release to the audited sources) start $(date -Is)"
  x docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
  echo "### ninja (restore) done $(date -Is)"
  tb_flags=$(x docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA] flags=' $TB_SO || true")
  tb_extent=$(x docker exec ttbuild bash -c "grep -caF -- '$EXTENT_MARKER' $TB_SO || true")
  tb_pf=$(x docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA-PF]' $TB_SO || true")
  tb_oq=$(x docker exec ttbuild bash -c "grep -caF -- 'oneq needs one q chunk per core' $TB_SO || true")
  tb_scratch=$(x docker exec ttbuild bash -c "grep -caF -- QWEN_SDPA_TREE_SCRATCH_ROUNDS $TB_SO || true")
  if [ "$DRY" != 1 ]; then
    test "$tb_flags" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA] branch; build_Release is dirty"
    test "$tb_extent" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the K64j runtime extent; build_Release is dirty"
    test "$tb_pf" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA-PF] branch; build_Release is dirty"
    test "$tb_oq" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the oneq edits; build_Release is dirty"
    test "$tb_scratch" -ge 1 || fail "ttbuild's rebuilt _ttnncpp.so lost the tree-scratch factory"
    echo "ok    ttbuild build_Release rebuilt from the audited sources (no qwen branch, no oneq, tree scratch present)"
  fi
  rebuilt=1
fi

# ---------- 9. verify: K64j plus the oneq edits and nothing else ----------
echo "### verify $G"
if [ "$DRY" = 1 ]; then
  echo "### dry run: step 9 reads the assembled graft; its checks were not run"
else
  so=$G/_ttnncpp.so
  test "$(hsha "$so")" != "$base_so" || fail "the build produced the base graft's _ttnncpp.so unchanged (the patch did not reach the binary)"
  if [ -n "$a_sha" ]; then
    test "$(hsha "$so")" != "$a_sha" || fail "build B equals build A: the patch did not reach the binary"
  fi
  qwen_strings "$so" > "$W/graft-qwen-strings.txt"
  lost=$(comm -23 "$W/base-qwen-strings.txt" "$W/graft-qwen-strings.txt")
  if [ -n "$lost" ]; then
    echo "FAIL: QWEN strings of $BASE/_ttnncpp.so missing from the new .so (a graft .so replaces the whole binary):" >&2
    echo "$lost" >&2
    exit 1
  fi
  echo "ok    every one of the base's $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings is in the new .so ($(wc -l < "$W/graft-qwen-strings.txt") now)"
  for literal in "${OQ_LITERALS[@]}"; do
    test "$(count "$so" "$literal")" -ge 1 || fail "the new _ttnncpp.so lacks '$literal'"
  done
  added=$(comm -13 "$W/base-qwen-strings.txt" "$W/graft-qwen-strings.txt")
  unexpected=
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    known=0
    for literal in "${OQ_LITERALS[@]}"; do
      case $line in *"$literal"*) known=1 ;; esac
    done
    [ "$known" = 1 ] || unexpected="$unexpected$line"$'\n'
  done <<< "$added"
  if [ -n "$unexpected" ]; then
    echo "FAIL: the new .so's QWEN strings beyond the base's are not the oneq literals:" >&2
    printf '%s' "$unexpected" >&2
    exit 1
  fi
  echo "ok    the only new QWEN strings are the two oneq literals ($(printf '%s\n' "$added" | grep -c . || true) lines)"
  for marker in "${K64J_MARKERS[@]}" "${STAGE4_MARKERS[@]}" "${PF_MARKERS[@]}" '[QWEN-SDPA] flags=' QWEN_SDPA_TREE_SCRATCH_ROUNDS \
                reader_decode_qwen.cpp "$SHARE_MARKER"; do
    test "$(count "$so" "$marker")" -ge 1 || fail "the new _ttnncpp.so lacks '$marker'"
  done
  echo "ok    strings: every K64j, K64i-slice, prefill-chain and tree-scratch marker of the base is present"
  test "$(count "$so" "$STAGE1_REFUSAL")" -eq 0 || fail "the new _ttnncpp.so refuses KV share (the stage-1 decode factory was linked)"
  test "$(count "$so" qwen_draft_fp32_intermediates)" = "$(count "$BASE/_ttnncpp.so" qwen_draft_fp32_intermediates)" \
    || fail "the combined prefill factory's qwen_draft_fp32_intermediates count moved from $BASE's"
  if [ "$HAVE_NM" = 1 ]; then
    symbols "$so" > "$W/graft-symbols.txt"
    if ! cmp -s "$W/base-symbols.txt" "$W/graft-symbols.txt"; then
      if [ "${OQ_ALLOW_SYMBOL_DELTA:-}" = "1" ]; then
        echo "WARN  the exported symbols of the new .so differ from the base's (OQ_ALLOW_SYMBOL_DELTA=1):"
        diff "$W/base-symbols.txt" "$W/graft-symbols.txt" | head -20 || true
      else
        echo "FAIL: the exported symbols of the new .so differ from $BASE's (the patch adds no function):" >&2
        diff "$W/base-symbols.txt" "$W/graft-symbols.txt" | head -20 >&2 || true
        exit 1
      fi
    else
      echo "ok    the $(wc -l < "$W/graft-symbols.txt") exported symbols equal the base's"
    fi
  fi
  # The directory is the base's with one file replaced: _ttnn.so and every op directory identical.
  cmp -s "$BASE/_ttnn.so" "$G/_ttnn.so" || fail "$G/_ttnn.so differs from $BASE/_ttnn.so"
  cmp -s "$BASE/_ttnn.so" "$W/B-ttnn.so" && echo "ok    the relinked _ttnn.so is byte-identical to the base's" \
    || echo "note  the relinked _ttnn.so differs from the base's; the graft keeps the BASE's (the patch changes no ABI)"
  for op in attn_prep nlp_concat_heads_decode sdpa sdpa_decode; do
    diff -r "$BASE/$op" "$G/$op" >/dev/null || fail "$G/$op differs from $BASE/$op"
  done
  echo "ok    _ttnn.so, attn_prep, nlp_concat_heads_decode, sdpa and sdpa_decode identical to $BASE"
  size_base=$(stat -c %s "$BASE/_ttnncpp.so")
  size_new=$(stat -c %s "$so")
  echo "ok    _ttnncpp.so size $size_base -> $size_new ($((size_new - size_base)) bytes)"
  for image in $IMAGES; do
    cid=$(docker create --network none --entrypoint true "$image")
    docker cp -L "$cid:/opt/tt-metal/build_Release/lib/_ttnncpp.so" "$W/image-ttnncpp.so"
    docker rm "$cid" >/dev/null
    qwen_strings "$W/image-ttnncpp.so" > "$W/image-${image//[^A-Za-z0-9]/_}-qwen-strings.txt"
    image_lost=$(comm -23 "$W/image-${image//[^A-Za-z0-9]/_}-qwen-strings.txt" "$W/graft-qwen-strings.txt")
    if [ -n "$image_lost" ]; then
      echo "FAIL: QWEN strings of image $image's _ttnncpp.so missing from the new .so:" >&2
      echo "$image_lost" >&2
      exit 1
    fi
    echo "ok    every QWEN string of image $image's _ttnncpp.so is in the new .so"
    rm -f "$W/image-ttnncpp.so"
  done
fi
if [ "$DRY" = 1 ]; then
  echo "### run: (cd $(printf '%q' "$G") && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > $(printf '%q' "$G/MANIFEST.sha256")"
else
  (cd "$G" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > "$G/MANIFEST.sha256"
  # The manifest differs from the base's in exactly one line: ./_ttnncpp.so.
  changed=$(diff <(sed 's/^[0-9a-f]\{64\}  //' "$BASE/MANIFEST.sha256" | sort) <(sed 's/^[0-9a-f]\{64\}  //' "$G/MANIFEST.sha256" | sort) || true)
  test -z "$changed" || fail "the new manifest lists other files than the base's: $changed"
  differing=$(join -j 2 <(sort -k 2 "$BASE/MANIFEST.sha256") <(sort -k 2 "$G/MANIFEST.sha256") | awk '$2 != $3 {print $1}')
  test "$differing" = "./_ttnncpp.so" || fail "the new manifest differs from the base's in more than ./_ttnncpp.so: $differing"
  echo "ok    MANIFEST.sha256 differs from the base's in exactly ./_ttnncpp.so"
fi
x mv "$G" "$GRAFT"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing built"
  exit 0
fi
so_sha=$(hsha "$GRAFT/_ttnncpp.so")
echo "### summary"
echo "ttbuild: sources restored (decode 3e0a69af, prefill fd8c0676, touched) and build_Release rebuilt from them"
sha256sum "$GRAFT/_ttnncpp.so" "$BASE/_ttnncpp.so"
echo "graft files: $(wc -l < "$GRAFT/MANIFEST.sha256") (manifest $GRAFT/MANIFEST.sha256); $BASE was not touched"
if [ -n "${RESULTS:-}" ]; then
  {
    mkdir -p "$RESULTS"
    cp "$GRAFT/MANIFEST.sha256" "$RESULTS/k64j-oq-MANIFEST.sha256"
    cp "$W"/*-qwen-strings.txt "$RESULTS/"
    [ ! -e "$W/base-symbols.txt" ] || cp "$W/base-symbols.txt" "$W/graft-symbols.txt" "$RESULTS/"
    printf 'graft=%s\nbase=%s\nK64J_OQ_TTNNCPP_SHA256=%s\nK64J_TTNNCPP_SHA256=%s\nreproduced=%s\nwork=%s\n' \
      "$GRAFT" "$BASE" "$so_sha" "$base_so" "$([ "$a_sha" = "$base_so" ] && echo yes || echo no)" "$W" > "$RESULTS/k64j-oq-build-summary.txt"
  } || echo "WARN  could not copy the manifest and strings lists into $RESULTS" >&2
fi
echo "next: the oneq card job (run_card_m_oq.sh) mounts $GRAFT and checks its _ttnncpp.so sha256 $so_sha (EXPECT_TTNNCPP_SHA256)"
echo "K64J_OQ_TTNNCPP_SHA256=$so_sha"
echo "### build_k64j_oq done $(date -Is)"
