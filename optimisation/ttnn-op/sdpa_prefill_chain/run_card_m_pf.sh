#!/usr/bin/env bash
# Run test_sdpa_prefill_chain_card_m.py (Q1 of sdpa-prefill-share-spec.md 6.2) on the qualification card
# (QUAL_CARD, default card B), in the model-gate image, mirroring ../sdpa_decode_qwen/run_card_m.sh.
# The file names keep 'card_m' from when card M was the only bench card.
#
#   bash run_card_m_pf.sh reference            # stock image, no graft: every served output's sha
#   WATCHER=1 bash run_card_m_pf.sh candidate  # THE FIRST HARDWARE PASS of K64g (Q1.2): TT_METAL_WATCHER=5
#                                              # (NoC sanitiser, waypoints, asserts), flags 0x1/0x3/0x5/0x7,
#                                              # rows 2048, starts 0 and 2048, seed 0, no stress, 120 s
#                                              # per-call watchdog, 900 s container cap
#   bash run_card_m_pf.sh candidate            # the full Q1 sweep + cache + refusals + M-A + stress + trace
#   bash run_card_m_pf.sh hang                 # Q1.3, LAST in its session (it hangs the card on purpose):
#                                              # a served pre-check, then flags 0x203; PASS = the planted call
#                                              # never returns (WATCHDOG on it, exit 3; or the faulthandler
#                                              # backstop, exit 1 + 'Timeout ('; or the container cap)
#   WATCHER=1 bash run_card_m_pf.sh hang       # Q1.3 with the watcher: PASS = QWDC / QWDV, or an assert in
#                                              # reader_interleaved_qwen_chain, in the watcher log
#
# Env: PF_SRC (~/kwork64/k64g: this directory), KOPGRAFT_PF (~/opgraft-K64g), IMAGE (the model-gate
# image A'' eceb2daa, qwen-lever-n-m3native-gate.yml v128+; spec 6: the image the model gate will use,
# and re-take the stock reference in it), RESULTS ($PF_SRC/<card tag>: card-b, card-m, card-a - one
# directory per board, so a candidate only ever picks a reference taken on its own board), REFERENCE (a
# reference report to use instead of the automatic pick), CARD_M_ARGS (extra args, appended last; the
# historical name), WATCHER=1, WATCHDOG_S (per-call seconds, default 120), PF_DRY_RUN=1 (print the docker
# argv only), QUAL_CARD (the target board id; default card B, blackhole-F36F768B9A5CAFA0; card M or card
# A, the serving pair, only with ALLOW_SERVING_CARD=1).
#
# The candidate's reference is the newest reference-*.json in $RESULTS that PASSED, ran in the same
# image, and (for the full candidate) covers the full matrix; the full candidate refuses to start
# without one. The test also fails every swept case the reference lacks.
#
# One card, resolved by board id at launch (the embedded qual_card.sh block). Refuses while a running
# container can reach the target's node (--privileged, a device cgroup rule, or given that node or a
# directory holding it; a container on another board, such as a CI gate on the serving pair, does not
# block) or a host process holds it (fuser; with sudo -n when that works, else this user's processes
# only, said so). Every run gets a fresh kernel cache. The graft is mounted exactly as
# ../sdpa_prefill_bench/run_m1.sh KOPGRAFT_PF mounts it (the arm's targets). candidate and hang set
# QWEN_SDPA_PF_TEST=1 (M-A, the test-flag refusal, 0x203).
# ON A HANG (exit 3, 124 or 137, or 1 with 'Timeout (' in the log): the EXIT trap removes the
# container; then reset THE TARGET CARD ONLY with the printed hint - a tt-smi -r command that resolves
# its board id when run, never a bare index - then a passing stock smoke call (run_m1.sh,
# same QUAL_CARD, with M1_ARGS="--arms baseline --starts 0,2048 --rounds 1") before the next run.
set -euo pipefail
role=${1:?usage: run_card_m_pf.sh reference|candidate|hang}
S=${PF_SRC:-$HOME/kwork64/k64g}
G=${KOPGRAFT_PF:-$HOME/opgraft-K64g}
IMAGE=${IMAGE:-sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9}
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
R=${RESULTS:-$S/$QUAL_TAG}
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
DRY=${PF_DRY_RUN:-0}
name=qwen-sdpa-pf-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=7200   # the full candidate: ~8,600 matrix calls (4 flag sets), 1,000 stress calls, 200 replays, cold JIT
WD=${WATCHDOG_S:-120}

test -s "$S/test_sdpa_prefill_chain_card_m.py" || { echo "$S/test_sdpa_prefill_chain_card_m.py missing (set PF_SRC)" >&2; exit 1; }
if [ "$DRY" = 1 ]; then
  node=$QUAL_BYID
else
  qual_card_resolve
  node=$QUAL_NODE
  qual_refuse_holders
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-$role-$stamp"
  chmod 0777 "$R" "$R/kcache-$role-$stamp"
fi

args=(--role "$role" --out "/results/$role-$stamp.json" --watchdog "$WD" --image "$IMAGE")
KM=()
EM=()
case "$role" in
  reference)
    ;;
  candidate|hang)
    for part in _ttnn.so _ttnncpp.so sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp; do
      test -s "$G/$part" || { echo "$G/$part missing: build it with build_k64g.sh" >&2; exit 1; }
    done
    KM+=(--mount "type=bind,src=$G/_ttnn.so,dst=/opt/tt-metal/ttnn/ttnn/_ttnn.so,readonly")
    KM+=(--mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so,readonly")
    KM+=(--mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly")
    for op in attn_prep:transformer/attn_prep nlp_concat_heads_decode:experimental/transformer/nlp_concat_heads_decode \
              sdpa_decode:transformer/sdpa_decode sdpa:transformer/sdpa; do
      if [ -d "$G/${op%%:*}" ]; then
        KM+=(--mount "type=bind,src=$G/${op%%:*},dst=$OPS/${op#*:},readonly")
      fi
    done
    EM+=(-e QWEN_SDPA_PF_TEST=1)
    if [ "$role" = candidate ]; then
      # The newest reference that passed, ran in this image and (full candidate) covers the full matrix.
      reference=
      if [ -n "${REFERENCE:-}" ]; then
        reference=$REFERENCE
      elif compgen -G "$R/reference-*.json" >/dev/null; then
        reference=$("${PF_PYTHON:-python3}" -B - "$R" "$IMAGE" "${WATCHER:-0}" <<'PY' || true
import json
import sys
from pathlib import Path

root, image, watcher = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == '1'
paths = sorted(root.glob('reference-*.json'), key=lambda path: path.stat().st_mtime, reverse=True) if root.is_dir() else []
for path in paths:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    if data.get('passed') is True and data.get('image') == image and (watcher or data.get('narrow') is False):
        print(path.as_posix())
        break
PY
)
      fi
      if [ -n "$reference" ]; then
        test -s "$reference" || { echo "refusing: reference $reference is missing or empty" >&2; exit 1; }
        case "$(readlink -f "$reference")" in
          "$(readlink -f "$R")"/*) ;;
          *) echo "refusing: reference $reference is not in $R (the container sees /results only)" >&2; exit 1 ;;
        esac
        args+=(--reference "/results/$(basename "$reference")")
        echo "served outputs are compared against $reference"
      elif [ "$DRY" = 1 ]; then
        echo "no passing reference report for this image in $R (dry run: continuing)"
      else
        echo "refusing: no passing $( [ "${WATCHER:-}" = 1 ] || echo 'full-matrix ')reference report for image ${IMAGE:7:12} in $R;" \
             "run 'run_card_m_pf.sh reference' first (Q1.6 needs it) or set REFERENCE" >&2
        exit 1
      fi
    fi
    ;;
  *)
    echo "unknown role $role" >&2
    exit 1
    ;;
esac
if [ "$role" = hang ]; then
  timeout_s=900
fi
if [ "${WATCHER:-}" = "1" ]; then
  timeout_s=900
  if [ "$role" = candidate ]; then
    args+=(--rows 2048 --starts 0,2048 --seeds 0 --variants normal --tables perm --widths fit --q-memory dram
           --flags 0x1,0x3,0x5,0x7 --alternations 0 --trace-replays 0 --no-mutation --no-controls)
  elif [ "$role" = reference ]; then
    args+=(--rows 2048 --starts 0,2048 --seeds 0 --variants normal --tables perm --widths fit --q-memory dram)
  fi
  WM=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  if [ "$DRY" != 1 ]; then
    mkdir -p "$R/watcher-$stamp"
    chmod 0777 "$R/watcher-$stamp"
  fi
  echo "WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog ${WD} s, container cap ${timeout_s} s, log $R/watcher-$stamp"
else
  WM=()
fi
# shellcheck disable=SC2206
extra=(${CARD_M_ARGS:-})

argv=(docker run --rm --name "$name" --network none
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g
  --device "$node"
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G
  --mount "type=bind,src=$S/test_sdpa_prefill_chain_card_m.py,dst=/bench/test_sdpa_prefill_chain_card_m.py,readonly"
  --mount "type=bind,src=$R,dst=/results"
  --mount "type=bind,src=$R/kcache-$role-$stamp,dst=/kcache"
  ${KM[@]+"${KM[@]}"}
  ${WM[@]+"${WM[@]}"}
  ${EM[@]+"${EM[@]}"}
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  --entrypoint sh "$IMAGE" -c
  "sha256sum /opt/tt-metal/build_Release/lib/_ttnncpp.so $OPS/transformer/sdpa/device/kernels/dataflow/reader_interleaved*.cpp $OPS/transformer/sdpa/device/sdpa_program_factory.cpp 2>&1; exec python3 -B /bench/test_sdpa_prefill_chain_card_m.py \"\$@\""
  card-m "${args[@]}" ${extra[@]+"${extra[@]}"})
echo "### pf $role $stamp card=$QUAL_CARD ($QUAL_TAG) node=$node image=${IMAGE:7:12} graft=$([ "$role" = reference ] && echo none || echo "$G") watcher=${WATCHER:-0}"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/$role-$stamp.log"
status=${PIPESTATUS[0]}
log=$R/$role-$stamp.log
echo "### exit $status; report $R/$role-$stamp.json; native log $R/$role-$stamp.json.native.log"
wlog=$R/watcher-$stamp/watcher.log
if [ "${WATCHER:-}" = "1" ]; then
  if [ -s "$wlog" ]; then
    echo "### watcher log $wlog: $(grep -ciE 'error|assert|tripped|sanitiz' "$wlog" || true) error/assert lines; QWDC/QWDV lines: $(grep -cE 'QWD[CV]' "$wlog" || true)"
    grep -iE 'error|assert|tripped|sanitiz|QWD[CV]' "$wlog" | head -20 || true
  else
    echo "### no watcher log at $wlog"
  fi
fi
if [ "$role" = hang ]; then
  if grep -qF 'planted hang RETURNED' "$log" "$R/$role-$stamp.json" 2>/dev/null; then
    echo "HANG TEST FAIL (K-7): the planted hang returned a result" >&2
    qual_reset_hint >&2
    exit 1
  fi
  verdict=2
  if ! grep -qF 'planted hang: armed' "$log"; then
    echo "HANG TEST INCONCLUSIVE (exit $status): the planted hang was never armed - the device open, an upload or" \
         "the served pre-check did not complete, so $QUAL_CARD is suspect before the test began" >&2
  elif [ "${WATCHER:-}" = "1" ]; then
    if grep -qE 'QWD[CV]' "$wlog" 2>/dev/null \
       || { grep -qF 'tripped an assert' "$wlog" 2>/dev/null && grep -qF reader_interleaved_qwen_chain "$wlog"; }; then
      echo "HANG TEST PASS: the bounded wait fired under the watcher (QWDC/QWDV, or an assert with the chain reader loaded); exit $status"
      verdict=0
    else
      echo "HANG TEST INCONCLUSIVE (exit $status): the watcher log names neither QWDC/QWDV nor a chain-reader assert" >&2
    fi
  elif [ "$status" = 3 ] && grep -qE "WATCHDOG: '(planted hang 0x203|planted hang read back)'" "$log"; then
    echo "HANG TEST PASS: the planted call never returned; the host watchdog exited 3"
    verdict=0
  elif [ "$status" = 1 ] && grep -qF 'Timeout (' "$log"; then
    echo "HANG TEST PASS: the planted call never returned; the faulthandler backstop exited 1 (the Python watchdog" \
         "thread never ran: a blocking ttnn call held the GIL)"
    verdict=0
  elif [ "$status" = 124 ] || [ "$status" = 137 ]; then
    echo "HANG TEST PASS: the planted call never returned (container cap, exit $status). WARN: neither the host" \
         "watchdog nor its faulthandler backstop fired - fix that before relying on them in a candidate run" >&2
    verdict=0
  else
    echo "HANG TEST INCONCLUSIVE (exit $status): the watchdog did not name the planted call or its read back" >&2
  fi
  echo "NOW RECOVER $QUAL_CARD: the container is removed on exit." >&2
  qual_reset_hint >&2
  exit "$verdict"
fi
hung=0
case "$status" in
  3|124|137) hung=1 ;;
  1) grep -qF 'Timeout (' "$log" && hung=1 ;;
esac
if [ "$hung" = 1 ]; then
  echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
  qual_reset_hint >&2
fi
exit "$status"
