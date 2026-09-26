#!/usr/bin/env bash
# K64j K1-K4 (c2-serve-for-real-plan.md section 2.3): run k64j_card_b.py on the qualification card (QUAL_CARD,
# default card B) in the C2 gate image (P8) with graft K64j mounted exactly as lever_n_m3native_run_arm.sh mounts
# a KOPGRAFT64 graft: _ttnn.so, _ttnncpp.so (both paths), attn_prep, nlp_concat_heads_decode, sdpa_decode and sdpa.
# No build, no weights: the harness runs the runtime-extent (0x20) programs against K64i's compile-time served
# programs of the same binary. Queue it on the card-B runner (docs/card-b-runner.md:
# CARD_B_HARNESS=optimisation/ttnn-op/k64j/run_card_b.sh) or run it by hand on the rig.
#
#   WATCHER=1 bash run_card_b.sh          # THE FIRST HARDWARE PASS of the 0x20 kernels: TT_METAL_WATCHER=5,
#                                         # extents 2,304 and 33,024, seed 0, G4B3 0x21 / 0x23 and G8B2 0x27, 8 trace
#                                         # families, no timing, a per-call watchdog (WATCHDOG_S, default 120 s), a
#                                         # 2,700 s container timeout; the watcher's own log lands in
#                                         # $RESULTS/watcher-<stamp>/
#   bash run_card_b.sh                    # every section (N, X, M, K, L, T) at K1's six extents, seeds 0-2, every
#                                         # combo, and the timing; a 5,400 s container timeout, --deadline-s 600 s
#                                         # before it; CARD_B_ARGS="--seeds 0,1,2,3,4" is K1's full seed list
#   K64J_CARD_DRY_RUN=1 bash run_card_b.sh   # print the launch argv and exit: no node resolution, no holder check,
#                                         # no docker, nothing launched (the graft is checked when it exists)
#
# Before anything is launched the graft must verify against its MANIFEST.sha256, its _ttnncpp.so must be
# EXPECT_TTNNCPP_SHA256 (REQUIRED for a real run: K64j's sha is the K64J_TTNNCPP_SHA256 line build_k64j.sh prints)
# and carry the F22 literal '[QWEN-SDPA] runtime-extent entries=', its four qwen decode kernels must be K64j's
# (make_k64j_kernels.OUTPUTS) and its five STOCK decode sources the stock ones, so a wrong graft fails before it
# costs card time. Inside the container the harness checks the mapped binary and the kernels again.
#
# The container timeout SIGTERMs `docker run`, which forwards it to the harness (PID 1 in the container, by exec):
# the harness writes its partial report at once and unwinds. A 124 exit prints the hang hint below unless the log
# shows the harness finished that unwinding ('ERROR terminated by signal', printed after the device closed).
#
# Env: KOPGRAFT64 (default ~/opgraft-K64j), EXPECT_TTNNCPP_SHA256, IMAGE (default image P8), RESULTS
# (~/kwork64/k64j/<card tag>), CARD_B_ARGS (extra harness args, appended last, e.g. "--sections X,M --seeds 0"),
# WATCHER=1, WATCHDOG_S, K64J_CARD_DRY_RUN=1, QUAL_CARD (the target board id under /dev/tenstorrent/by-id; default
# card B, blackhole-F36F768B9A5CAFA0; card M or card A, the serving pair, is refused unless ALLOW_SERVING_CARD=1,
# which prints a loud warning). QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is set as the arm sets it. Every run gets a fresh
# kernel cache.
#
# ON A HANG (the WATCHDOG line and exit 3; exit 1 with 'Timeout (' in the log; or the timeout's 124 / 137): the EXIT
# trap removes the container. Then reset THE TARGET CARD ONLY with the hint printed below. This script never resets.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
probe_dir=$(cd "$here/../k64j_probe" 2>/dev/null && pwd || echo "$here/../k64j_probe")
qwen=$(cd "$here/../sdpa_decode_qwen" 2>/dev/null && pwd || echo "$here/../sdpa_decode_qwen")
G=${KOPGRAFT64:-$HOME/opgraft-K64j}
IMAGE=${IMAGE:-sha256:57cb699489436842d7e7bdd5ab917d509b4492fc95f6f083ef7d2865b488c2ef}
EXPECT=${EXPECT_TTNNCPP_SHA256:-}
EXTENT_MARKER='[QWEN-SDPA] runtime-extent entries='
# The K64j qwen kernels (make_k64j_kernels.OUTPUTS; k64j_card_b.K64J_KERNELS).
READER_QWEN=adb6091878ba3f0a0805846ff56f05352610d7fe779b5ae96320c437c095db49
READER_SLICE=518d8096e3cceb160eaef8ab4f0ae976ccbffd3904d31176b7f9d02828c37f8a
COMPUTE_QWEN=409a1aafc3ffaaca2c6afba0e999525b7d141491f0a52e5583efa70447ee5c0e
WRITER_SLICE=642c36f809be0f1ad1664deb405dc310dabaa32d5628710320a118eb37a6cc7a
# The stock decode sources (tt-metal 9f9cd4fd; probe_k64j_card_b.STOCK_KERNELS, the vendored ../k64j_probe/fixtures).
READER_ALL=49a05926b437e2ca90d7e01c60e85a6b11f333375a6159fa02c9ff6f78af764e
WRITER_ALL=734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8
COMPUTE_ALL=d24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f
DATAFLOW_COMMON=e4623a2254559eaec4450ebfab0f9c5732e02acfe4d8126bd5eeb7efe0fdc608
RT_ARGS_COMMON=1b52c60d78ada6f08effd326c2ed2407b3a74cf0db2353fadbe51b088610aec8

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
R=${RESULTS:-$HOME/kwork64/k64j/$QUAL_TAG}
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
KD=$OPS/transformer/sdpa_decode/device/kernels
DRY=${K64J_CARD_DRY_RUN:-0}
name=qwen-k64j-card-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=5400   # N, X (6 combos x 6 extents x 5 starts x 3 seeds), M, K, L, T (two 64-family traces), timing

if [ "$DRY" != 1 ] && [ -z "$EXPECT" ]; then
  echo "refusing: EXPECT_TTNNCPP_SHA256 is required (the K64J_TTNNCPP_SHA256 line build_k64j.sh printed for $G)" >&2
  exit 1
fi

# The target's node, resolved by board id now; never shared: refuse while a container or a host
# process can reach it (a container on another board - a CI gate on the serving pair - does not block).
if [ "$DRY" = 1 ]; then
  node=$QUAL_BYID
else
  qual_card_resolve
  node=$QUAL_NODE
  qual_refuse_holders
fi

HARNESS=("$here/k64j_card_b.py" "$probe_dir/probe_k64j_card_b.py" "$probe_dir/split_model.py"
         "$qwen/test_sdpa_decode_qwen_card_m.py" "$qwen/probe_k1_card_b.py")
for file in "${HARNESS[@]}"; do
  test -s "$file" || { echo "refusing: $file missing (ship k64j, k64j_probe and sdpa_decode_qwen side by side)" >&2; exit 1; }
done

# Graft K64j, checked before anything is launched (in a dry run, only when it exists here).
if [ "$DRY" = 1 ] && [ ! -e "$G" ]; then
  echo "### dry run: $G does not exist here; the graft was not checked"
else
  for part in _ttnn.so _ttnncpp.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
    test -e "$G/$part" || { echo "refusing: $G/$part missing (the harness needs graft K64j, build_k64j.sh)" >&2; exit 1; }
  done
  (cd "$G" && sha256sum -c --quiet MANIFEST.sha256 >&2) \
    || { echo "refusing: $G/MANIFEST.sha256 does not verify (not the graft its build script made)" >&2; exit 1; }
  so_sha=$(sha256sum "$G/_ttnncpp.so" | cut -c1-64)
  if [ -n "$EXPECT" ] && [ "$so_sha" != "$EXPECT" ]; then
    echo "refusing: $G/_ttnncpp.so is ${so_sha:0:16}, not ${EXPECT:0:16}" >&2
    exit 1
  fi
  grep -a -q -F -- "$EXTENT_MARKER" "$G/_ttnncpp.so" \
    || { echo "refusing: $G/_ttnncpp.so lacks '$EXTENT_MARKER' (not a K64j binary)" >&2; exit 1; }
  pairs=("dataflow/reader_decode_qwen.cpp:$READER_QWEN" "compute/sdpa_flash_decode_qwen.cpp:$COMPUTE_QWEN"
         "dataflow/reader_decode_qwen_slice.cpp:$READER_SLICE" "dataflow/writer_decode_qwen_slice.cpp:$WRITER_SLICE")
  stock=("dataflow/reader_decode_all.cpp:$READER_ALL" "dataflow/writer_decode_all.cpp:$WRITER_ALL"
         "compute/sdpa_flash_decode.cpp:$COMPUTE_ALL" "dataflow/dataflow_common.hpp:$DATAFLOW_COMMON"
         "rt_args_common.hpp:$RT_ARGS_COMMON")
  for pair in "${pairs[@]/#/K64j:}" "${stock[@]/#/stock:}"; do
    kind=${pair%%:*}
    pair=${pair#*:}
    file=$G/sdpa_decode/device/kernels/${pair%%:*}
    got=$(sha256sum "$file" 2>/dev/null | cut -c1-64 || true)
    if [ "$got" != "${pair#*:}" ]; then
      echo "refusing: $file is ${got:-missing}, not the $kind ${pair#*:}" >&2
      exit 1
    fi
  done
  echo "### graft $G: _ttnncpp.so ${so_sha:0:16} with the F22 literal, ${#pairs[@]} K64j qwen kernels, ${#stock[@]} stock" \
    "decode sources, manifest verified"
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

args=(--out "/results/card-$stamp.json" --expect-binary-sha256 "$EXPECT")
BM=()
for file in "${HARNESS[@]}"; do
  BM+=(--mount "type=bind,src=$file,dst=/bench/$(basename "$file"),readonly")
done
WM=()
if [ "${WATCHER:-}" = "1" ]; then
  # One pass under the NoC sanitiser over every new program kind (0x21, share, slice) and the trace; timing is
  # meaningless here. CARD_B_ARGS, appended last, can widen it.
  timeout_s=2700
  args+=(--extents 2304,33024 --seeds 0 --variants normal --combos G4B3:0x21,G4B3:0x23,G8B2:0x27
         --trace-combos G4B3:0x21,G8B2:0x27 --trace-families 8 --trace-references 2 --no-timing
         --watchdog "${WATCHDOG_S:-120}")
  WM=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  if [ "$DRY" != 1 ]; then
    mkdir -p "$R/watcher-$stamp"
    chmod 0777 "$R/watcher-$stamp"
  fi
  echo "WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog ${WATCHDOG_S:-120} s, container timeout ${timeout_s} s"
else
  args+=(--watchdog "${WATCHDOG_S:-300}")
fi
# Stop cleanly 600 s (DEADLINE_MARGIN_S) before the container timeout: the rest is listed, not lost.
args+=(--deadline-s "$((timeout_s - 600))")
# shellcheck disable=SC2206
extra=(${CARD_B_ARGS:-})

# The container's script: record the binaries and the decode sources it runs, then the harness. One line (printf %q
# of a newline is $'...', which the runner test's shlex cannot read).
inner='sha256sum /opt/tt-metal/build_Release/lib/_ttnncpp.so /opt/tt-metal/build_Release/ttnn/_ttnncpp.so '
inner+="$KD/dataflow/reader_decode_qwen.cpp $KD/dataflow/reader_decode_qwen_slice.cpp $KD/compute/sdpa_flash_decode_qwen.cpp "
inner+="$KD/dataflow/writer_decode_qwen_slice.cpp $KD/dataflow/reader_decode_all.cpp $KD/dataflow/writer_decode_all.cpp "
inner+="$KD/compute/sdpa_flash_decode.cpp $KD/dataflow/dataflow_common.hpp $KD/rt_args_common.hpp 2>&1; "
inner+='exec python3 -B /bench/k64j_card_b.py "$@"'

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
  -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
  --entrypoint sh "$IMAGE" -c "$inner"
  card "${args[@]}" ${extra[@]+"${extra[@]}"})
echo "### k64j-card $stamp card=$QUAL_CARD ($QUAL_TAG) node=$node image=${IMAGE:7:12} graft=$G watcher=${WATCHER:-0}"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/card-$stamp.log"
status=${PIPESTATUS[0]}
log=$R/card-$stamp.log
echo "### exit $status; report $R/card-$stamp.json; native log $R/card-$stamp.json.native.log"
if [ "${WATCHER:-}" = "1" ]; then
  wlog=$R/watcher-$stamp/watcher.log
  if [ -s "$wlog" ]; then
    echo "### watcher log $wlog: $(grep -ciE 'error|assert|tripped|sanitiz' "$wlog" || true) error/assert lines"
    grep -iE 'error|assert|tripped|sanitiz' "$wlog" | head -20 || true
  else
    echo "### no watcher log at $wlog"
  fi
fi
# Anchored: the summary line 'SDPA_K64J_CARD passed=...' comes after the verdict and also contains 'K64J_CARD '.
echo "### $(grep -E '^K64J_CARD ' "$log" | tail -1 || echo 'no K64J_CARD line')"
# >>> hang: status, log -> hung
hung=0
case "$status" in
  3|124|137) hung=1 ;;
  1) grep -qF 'Timeout (' "$log" && hung=1 ;;   # the faulthandler backstop (GIL held)
esac
# The container timeout (124) after the harness caught SIGTERM and unwound: its 'ERROR terminated by signal' line is
# printed only after the device closed, so this is an overrun (see its partial report), not a hang.
if [ "$status" = 124 ] && grep -q '^ERROR terminated by signal' "$log" && ! grep -qF 'Timeout (' "$log"; then
  hung=0
  echo "### TIMEOUT (exit 124): the harness caught SIGTERM, wrote its partial report and closed the device; not a" \
    "hang. Raise the timeout or split the run (--sections)." >&2
fi
# <<< hang
if [ "$hung" = 1 ]; then
  echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
  qual_reset_hint >&2
fi
exit "$status"
