#!/usr/bin/env bash
# K0 (sdpa-prefill-share-spec.md 7.1): the no-build compute-floor probes, one session on the qualification
# card (QUAL_CARD, default card B; card M or card A only with ALLOW_SERVING_CARD=1 - run_m1.sh's rules).
#
#   bash k0_session.sh                    # from the staged directory: sdpa_prefill_bench.py, run_m1.sh, k0/
#   K0_Q4096=1 bash k0_session.sh         # also the q256_4096 second opinion on t_c (a seventh, separate run)
#
# Six container runs through run_m1.sh, in this order: stock (the image's own reader), k0a, k0b4,
# k0b32, k0c (M1_READER=k0/reader_<v>.cpp, one file mounted over the image's reader), then stock2
# (stock again: the drift bracket across the session, stock's reproducibility across processes, and
# the control for K0c's compiled-reader proof). Each is the baseline arm only, chunk_start 0 / 32768
# / 65536 / 126976, 7 interleaved rounds, --sha, --no-fallback-scalar (a failing tensor-path pair is
# an error, never silently timed on the scalar path), --kernel-elf reader_interleaved (the digest of
# the compiled reader), a per-call host watchdog (120 s; sdpa_prefill_bench.py gives each arm's first
# warmup +780 s for the fresh-cache JIT compile, open_device/build_inputs/other warmups +240 s), and
# M1_REQUIRE_SOURCES=1 (the seven sdpa sources in the container must be the probe-v25 ones, the
# reader replaced by the mounted file). The stock run also captures the worker-coordinate fixture
# (spec 7.1 / 5.1): fixtures/cardm_worker_coords.json when the session ran on card M (the board the
# model's G6 groups are costed on), else fixtures/worker_coords-<card tag>.json (harvesting may differ).
#
# Image (spec 6 ground rules: the image the model gate will use): image A', the image of the
# v117-v127 model gates, whose sdpa sources are the probe-v25 ones. IMAGE=<id> overrides; the image is
# printed for every run and in the summary.
#
# STOPS at the first run that does not exit 0 (a clean K0 run always exits 0). The session's exit:
#   3   a hang: the run printed WATCHDOG or a faulthandler 'Timeout (', or exited 3/124/137
#   97  the container's sdpa sources, bench or reader mount are not what was asked (checked before
#       the bench starts: nothing ran on the device)
#   4   any other failure after a container was launched on the card
#   1   a failure before any container was launched (a refusal, a missing image, or docker could not
#       start the container: exit 125)
# On 3 and 4 it prints the recovery (spec 6: docker rm -f, a reset of the target card only - its tt-smi
# index derived from its PCI address - then a passing stock smoke run) without running it. It never
# resets a card itself.
#
# Summary: slope per run, stock drift (stock2/stock), t_c = K0a slope x 0.064 ms, K0b4/K0a,
# K0b32/K0a, K0b4/stock, K0c/stock; the per-variant proof that the mounted reader ran (K0a/b4/b32:
# output differs from stock at every start; K0c, whose output equals stock by design: its reader ELF
# differs from stock's while stock's equals stock2's); K0c == stock at every start; and the
# pre-registered rules K-1..K-5 (spec 7.2), each printed only when the runs it reads are proven to
# have run, and flagged when a ratio lies within the stock drift of its threshold.
#
# Env: M1_SRC (default: this script's directory), K0_DIR (default $M1_SRC/k0), RESULTS (default
# $M1_SRC/results), IMAGE (default image A' 1b9b6445), K0_ROUNDS (7), K0_STARTS
# (0,32768,65536,126976), K0_WATCHDOG_S (120), K0_ONLY (e.g. "stock k0c stock2"), K0_Q4096 (0),
# K0_DRY_RUN (1: run_m1.sh prints its argv only), QUAL_CARD and ALLOW_SERVING_CARD (passed to every run).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=${M1_SRC:-$HERE}
K0_DIR=${K0_DIR:-$S/k0}
R=${RESULTS:-$S/results}
RUN_M1=${K0_RUN_M1:-$HERE/run_m1.sh}   # K0_RUN_M1: a stand-in runner, for the CPU tests only
ROUNDS=${K0_ROUNDS:-7}
STARTS=${K0_STARTS:-0,32768,65536,126976}
WATCHDOG_S=${K0_WATCHDOG_S:-120}
# Image A' (fast-serving-image-v96, commit 8dab9ae8): the v117-v127 model gates' image
# (qwen-lever-n-m3native-gate.yml); its sdpa/ tree is the probe-v25 one run_m1.sh checks.
IMAGE_A1=sha256:1b9b644549d4409c4fc80e2f92c183e665e7e693c37cccc95b4bb760a6d7d537
IMAGE=${IMAGE:-$IMAGE_A1}
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
KERNEL=reader_interleaved
RUNS=(stock k0a k0b4 k0b32 k0c stock2)
IFS=, read -r -a STARTS_A <<< "$STARTS"
TOTAL=${#STARTS_A[@]}
# Recorded reader shas (make_k0_readers.py OUTPUTS; test_k0_readers.py keeps these equal).
declare -A READER_SHA=(
  [k0a]=fbbd325213d0f3d77c4ea700679112bb959d7f4182cacab815a848dbbb78df38
  [k0b4]=0fe631aaae0a9e894c865a1623cde031b7828ee0eef1f531ea0d648a0d25fd06
  [k0b32]=c66b03c6c9fc8739443addf81058579c0a3a04b47b1874ffbc760d8645681504
  [k0c]=817bf36b9a28fffe603c012c6a32cca7cd5e709469173ded98ac1cc895f14747
)
sstamp=$(date +%Y%m%dT%H%M%S)
L=$R/k0-$sstamp
COORDS=worker_coords-$QUAL_TAG-$sstamp.json
# The committed fixture the G6 noc-order test reads is card M's; another board's goes beside it.
FIXTURE=cardm_worker_coords.json
[ "$QUAL_TAG" = card-m ] || FIXTURE=worker_coords-$QUAL_TAG.json

test -s "$RUN_M1" || { echo "run_m1.sh missing next to $0" >&2; exit 1; }
test -s "$S/sdpa_prefill_bench.py" || { echo "$S/sdpa_prefill_bench.py missing (set M1_SRC)" >&2; exit 1; }
for v in k0a k0b4 k0b32 k0c; do
  f=$K0_DIR/reader_$v.cpp
  got=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
  if [ "$got" != "${READER_SHA[$v]}" ]; then
    echo "refusing: $f sha256=${got:-missing} is not the recorded ${READER_SHA[$v]} (regenerate with make_k0_readers.py)" >&2
    exit 1
  fi
done
mkdir -p "$L"
echo "### K0 session $sstamp: card $QUAL_CARD ($(qual_card_label)); image $IMAGE; runs ${K0_ONLY:-${RUNS[*]}}; starts $STARTS; rounds $ROUNDS; watchdog ${WATCHDOG_S}s; logs $L"

recovery() {   # [log of the failed run]
  local log=${1:-} stamp
  echo "### recovery (spec 6), $QUAL_CARD ($(qual_card_label)) only:"
  echo "###   timeout 20 docker rm -f qwen-sdpa-m1-$QUAL_TAG"
  qual_reset_hint | sed 's/^/###   /'
  echo "###   QUAL_CARD=$QUAL_CARD ALLOW_SERVING_CARD=${ALLOW_SERVING_CARD:-0} IMAGE=$IMAGE M1_SRC=$S RESULTS=$R M1_ARGS='--arms baseline --starts 0 --rounds 1 --watchdog-s 120 --no-fallback-scalar' bash $RUN_M1   # stock smoke; must exit 0"
  if [ -n "$log" ] && grep -q -E '^WATCHDOG: .* warmup' "$log" 2>/dev/null; then
    stamp=$(sed -n -E 's/^### M1 ([0-9]{8}T[0-9]{6}) node=.*/\1/p' "$log" | head -n 1)
    echo "###   (it fired in a warmup, whose budget includes the JIT compile: the newest file under"
    echo "###    $R/kcache-m1-${stamp:-<stamp>} against the WATCHDOG time shows whether the compile was still running)"
  fi
}

slope_of() {   # the baseline row of the bench table: ... <ms/1k keys> <per 2048 rows>
  awk '$1 == "baseline" && NF >= 6 { s = $(NF - 1) } END { print (s == "" ? "n/a" : s) }' "$1" 2>/dev/null || echo n/a
}

sha_of() {     # M1 SHA baseline@<start> <hex> stable=<0|1>
  awk -v k="baseline@$2" '$1 == "M1" && $2 == "SHA" && $3 == k { print $4 }' "$1" 2>/dev/null
}

stable_of() {
  awk -v k="baseline@$2" '$1 == "M1" && $2 == "SHA" && $3 == k { sub("stable=", "", $5); print $5 }' "$1" 2>/dev/null
}

elf_of() {     # M1 KERNEL_ELF reader_interleaved <hex|none> files=<n>; empty when none
  awk -v k="$KERNEL" '$1 == "M1" && $2 == "KERNEL_ELF" && $3 == k && $4 != "none" { print $4 }' "$1" 2>/dev/null
}

compare_shas() {   # <run>: "<differ> <equal> <missing>" of its output shas against stock's, over the starts
  local s a b d=0 e=0 m=0
  for s in "${STARTS_A[@]}"; do
    a=$(sha_of "$L/stock.log" "$s"); b=$(sha_of "$L/$1.log" "$s")
    if [ -z "$a" ] || [ -z "$b" ]; then m=$((m + 1))
    elif [ "$a" = "$b" ]; then e=$((e + 1))
    else d=$((d + 1)); fi
  done
  echo "$d $e $m"
}

num() { [[ "$1" =~ ^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; }

ratio() {      # a / b, or n/a
  if num "$1" && num "$2"; then awk -v a="$1" -v b="$2" 'BEGIN { if (b + 0 > 0) printf "%.3f", a / b; else print "n/a" }'
  else echo n/a; fi
}

near() {       # <value> <threshold> <drift fraction>: a note when |value / threshold - 1| < drift
  if num "$1" && num "$3"; then
    awk -v v="$1" -v t="$2" -v d="$3" 'BEGIN { r = v / t - 1; if (r < 0) r = -r;
      if (r < d) printf "      NOT DECISIVE: %s is within the stock drift (%.1f%%) of the %s threshold; repeat the runs it reads\n", v, 100 * d, t }'
  fi
}

summary() {
  local v s d e m why elf
  local -A SL ELF PROOF
  echo
  echo "=================== K0 SUMMARY $sstamp (logs $L) ==================="
  echo "  image $IMAGE"
  for v in "${RUNS[@]}"; do
    SL[$v]=$(slope_of "$L/$v.log")
    ELF[$v]=$(elf_of "$L/$v.log")
    elf=${ELF[$v]:-none}
    printf '  %-6s slope %-10s ms/1k keys   status %-8s reader ELF %s\n' "$v" "${SL[$v]}" \
      "$(cat "$L/$v.status" 2>/dev/null || echo not-run)" "${elf:0:16}"
  done

  # Drift across the session: stock at the start, stock2 at the end.
  local drift=n/a drift_frac=n/a
  if num "${SL[stock]}" && num "${SL[stock2]}"; then
    drift=$(ratio "${SL[stock2]}" "${SL[stock]}")
    drift_frac=$(awk -v a="${SL[stock2]}" -v b="${SL[stock]}" 'BEGIN { d = a / b - 1; if (d < 0) d = -d; printf "%.4f", d }')
    echo "  stock drift across the session: stock2/stock slope = $drift (|drift| $(awk -v d="$drift_frac" 'BEGIN { printf "%.1f", 100 * d }')%)"
  else
    echo "  stock drift: n/a (stock2 not run or no slope); no ratio can be judged against the drift"
  fi

  # Stock reproducibility: within the run (the bench's two fresh calls) and across processes (stock2).
  local unstable=0 repro=incomplete
  for s in "${STARTS_A[@]}"; do
    [ "$(stable_of "$L/stock.log" "$s")" = 1 ] || unstable=$((unstable + 1))
  done
  read -r d e m <<< "$(compare_shas stock2)"
  if [ "$m" = 0 ]; then if [ "$d" = 0 ]; then repro=yes; else repro=NO; fi; fi
  echo "  stock stable call to call at $((TOTAL - unstable))/$TOTAL starts; stock2 == stock at $e/$TOTAL starts ($m missing): reproducible across processes = $repro"
  [ "$unstable" = 0 ] || echo "  WARNING: stock output not stable call to call at $unstable/$TOTAL starts; the K0c compare needs a stable stock"

  for s in "${STARTS_A[@]}"; do
    echo "  sha @$s: stock  $(sha_of "$L/stock.log" "$s" | grep . || echo none)"
    echo "  ${s//?/ }       stock2 $(sha_of "$L/stock2.log" "$s" | grep . || echo none)"
    echo "  ${s//?/ }       k0c    $(sha_of "$L/k0c.log" "$s" | grep . || echo none)"
  done

  # The proof that each mounted reader ran. K0a/b4/b32: the output differs from stock at every
  # start (a start where it EQUALS stock is evidence against, whatever else is missing).
  for v in k0a k0b4 k0b32; do
    read -r d e m <<< "$(compare_shas "$v")"
    if [ "$d" = "$TOTAL" ]; then PROOF[$v]=yes
    elif [ "$e" != 0 ]; then PROOF[$v]=NO
    else PROOF[$v]=incomplete; fi
    echo "  $v ran (output differs from stock at every start): ${PROOF[$v]}   ($d differ, $e EQUAL, $m missing of $TOTAL)"
  done
  # K0c: its output equals stock by design, so the proof is the compiled reader: k0c's ELF differs
  # from stock's, and stock's is reproducible (stock == stock2) so that a difference means a
  # different source. An ELF EQUAL to stock's means the mounted reader was not what compiled.
  local se=${ELF[stock]} s2e=${ELF[stock2]} ce=${ELF[k0c]} elf_repro=unproven
  if [ -n "$se" ] && [ -n "$s2e" ]; then if [ "$se" = "$s2e" ]; then elf_repro=yes; else elf_repro=NO; fi; fi
  why=
  if [ -z "$ce" ] || [ -z "$se" ]; then PROOF[k0c]=unproven; why="no reader ELF digest for stock or k0c"
  elif [ "$ce" = "$se" ]; then PROOF[k0c]=NO; why="k0c's reader ELF equals stock's: the mounted reader was not compiled"
  elif [ "$elf_repro" = yes ]; then PROOF[k0c]=yes
  elif [ "$elf_repro" = NO ]; then PROOF[k0c]=unproven; why="stock's own reader ELF differs from stock2's, so a difference proves nothing"
  else PROOF[k0c]=unproven; why="no stock2 reader ELF to show the build is reproducible"; fi
  echo "  k0c reader compiled (its ELF != stock's, stock's == stock2's): ${PROOF[k0c]}${why:+ ($why)}"

  local k0c_exact=incomplete
  read -r d e m <<< "$(compare_shas k0c)"
  if [ "$m" = 0 ]; then if [ "$e" = "$TOTAL" ]; then k0c_exact=yes; else k0c_exact=no; fi; fi
  echo "  K0c == stock byte for byte at $e/$TOTAL starts ($m missing): $k0c_exact"

  local k0a=${SL[k0a]} tc=n/a tag=
  if num "$k0a"; then tc=$(awk -v s="$k0a" 'BEGIN { printf "%.2f", s * 0.064 * 1000 }'); fi
  [ "${PROOF[k0a]}" = yes ] || tag="   UNPROVEN: k0a is not shown to have run"
  echo "  t_c = K0a slope x 0.064 ms = $tc us per step   (served step 21.76 us at slope 0.340)$tag"
  local b4 b32 b4s
  b4=$(ratio "${SL[k0b4]}" "$k0a"); b32=$(ratio "${SL[k0b32]}" "$k0a"); b4s=$(ratio "${SL[k0b4]}" "${SL[stock]}")
  echo "  K0b4/K0a = $b4   K0b32/K0a = $b32   K0b4/stock = $b4s   K0c/stock = $(ratio "${SL[k0c]}" "${SL[stock]}")   K0a/stock = $(ratio "$k0a" "${SL[stock]}")"

  echo "  rules (spec 7.2, pre-registered; each only when the runs it reads are proven to have run):"
  if [ "${PROOF[k0a]}" != yes ]; then
    echo "    K-1/K-2 withheld: k0a ran = ${PROOF[k0a]}"
  elif ! num "$tc"; then
    echo "    K-1/K-2: no t_c (K0a slope missing)"
  else
    awk -v t="$tc" 'BEGIN {
      if (t >= 18.5) print "    K-1 TRIGGERED: t_c >= 18.5 us -> stop all K/V-sharing work (G6 and Plan B); redirect to compute";
      else if (t > 15) print "    K-2 TRIGGERED: 15 < t_c < 18.5 us -> build only if nothing better is queued (3.4-6.6 s)";
      else print "    K-1/K-2 clear: t_c < 15 us";
    }'
    near "$tc" 18.5 "$drift_frac"; near "$tc" 15 "$drift_frac"
  fi
  if [ "${PROOF[k0a]}" != yes ] || [ "${PROOF[k0b32]}" != yes ]; then
    echo "    K-3 withheld: k0a ran = ${PROOF[k0a]}, k0b32 ran = ${PROOF[k0b32]}"
  elif ! num "$b32"; then
    echo "    K-3: no ratio"
  else
    awk -v r="$b32" 'BEGIN { if (r > 1.10) print "    K-3 TRIGGERED: K0b32 slope > 1.10 x K0a -> one injector per 6 cannot feed the group; stop G6 as specified";
                             else print "    K-3 clear: K0b32/K0a <= 1.10" }'
    near "$b32" 1.10 "$drift_frac"
  fi
  if [ "${PROOF[k0a]}" != yes ] || [ "${PROOF[k0b4]}" != yes ] || [ "${PROOF[k0b32]}" != yes ]; then
    echo "    K-4 withheld: k0a ran = ${PROOF[k0a]}, k0b4 ran = ${PROOF[k0b4]}, k0b32 ran = ${PROOF[k0b32]}"
  else
    echo "    K-4 (read): K0b4/stock = $b4s (near 1: the baseline), K0b32/K0a = $b32 (near 1); both together confirm flag 0x2 is mandatory"
  fi
  case "$k0c_exact" in
    incomplete) echo "    K-5 incomplete: stock or K0c shas missing" ;;
    yes)
      if [ "${PROOF[k0c]}" = yes ] && [ "$repro" = yes ]; then
        echo "    K-5 clear: K0c output equals stock at every start (K0c's reader compiled; stock reproducible across processes)"
      else
        echo "    K-5 unproven: K0c equals stock, but k0c reader compiled = ${PROOF[k0c]}, stock reproducible across processes = $repro (a reader that never ran gives the same equality)"
      fi ;;
    no)
      if [ "$repro" = yes ]; then
        echo "    K-5 TRIGGERED: K0c output != stock (stock reproducible) -> stop, the batching premise (flag 0x2 exactness) is wrong; investigate before any build"
      elif [ "$repro" = NO ]; then
        echo "    K-5 inconclusive: stock itself differs between stock and stock2, so K0c's mismatch proves nothing"
      else
        echo "    K-5 provisional TRIGGER: K0c output != stock, but stock2 did not run to rule out a non-reproducible stock; re-run stock before acting"
      fi ;;
  esac
  if [ -s "$L/stock-q256_4096.log" ]; then
    local q4096 r
    q4096=$(awk '$1 == "q256_4096" && NF >= 6 { s = $NF } END { print (s == "" ? "n/a" : s) }' "$L/stock-q256_4096.log")
    r=$(ratio "$q4096" "${SL[stock]}")
    if num "$r"; then
      echo "  q256_4096 per-2048-row slope $q4096 -> r = $r; second opinion t_c ~ 22 x r = $(awk -v r="$r" 'BEGIN { printf "%.1f", 22 * r }') us (optimistic; valid for r > 0.5)"
    else
      echo "  q256_4096: no per-token slope"
    fi
  fi
  if [ -s "$S/fixtures/$FIXTURE" ]; then echo "  coordinate fixture ($QUAL_CARD): $S/fixtures/$FIXTURE"; fi
  echo "======================================================================"
}

run_one() {    # <label> <reader file or empty> <bench args>
  local label=$1 reader=$2 args=$3 log=$L/$1.log status hang=0
  echo
  echo "### K0 run $label: reader=${reader:-served} image=$IMAGE args: $args"
  QUAL_CARD=$QUAL_CARD ALLOW_SERVING_CARD=${ALLOW_SERVING_CARD:-0} IMAGE=$IMAGE M1_SRC=$S RESULTS=$R M1_REQUIRE_SOURCES=1 M1_READER=$reader M1_ARGS=$args M1_DRY_RUN=${K0_DRY_RUN:-0} \
    bash "$RUN_M1" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  echo "$status" > "$L/$label.status"
  if grep -q -E '^WATCHDOG|^Timeout \(' "$log"; then hang=1; fi
  case $status in 3|124|137) hang=1 ;; esac
  if [ "$hang" = 1 ]; then
    echo "### K0 STOPPED at $label: a hang (exit $status): $(grep -m 1 -E '^WATCHDOG|^Timeout \(' "$log" || echo 'no WATCHDOG line; the container was killed')"
    echo "### Nothing further runs on $QUAL_CARD."
    recovery "$log"
    summary
    exit 3
  fi
  [ "$status" = 0 ] && return 0
  if [ "$status" = 97 ]; then
    echo "### K0 STOPPED at $label: the container's sdpa sources, bench or reader mount are not what was asked (exit 97; checked before the bench, nothing ran on the device)"
    summary
    exit 97
  fi
  if [ "$status" = 125 ] || ! grep -q -E '^### M1 [0-9]{8}T[0-9]{6} node=' "$log"; then
    echo "### K0 STOPPED at $label: exit $status before any container ran on $QUAL_CARD (see $log); nothing to recover"
    summary
    exit 1
  fi
  echo "### K0 STOPPED at $label: exit $status after a container ran on $QUAL_CARD. Not a recognised hang, but every clean K0 run exits 0; the card may be in any state."
  recovery "$log"
  summary
  exit 4
}

common="--arms baseline --starts $STARTS --rounds $ROUNDS --sha --watchdog-s $WATCHDOG_S --no-fallback-scalar --kernel-elf $KERNEL"
for v in "${RUNS[@]}"; do
  case " ${K0_ONLY:-${RUNS[*]}} " in *" $v "*) ;; *) continue ;; esac
  case $v in
    stock)
      run_one stock "" "$common --coords-out /results/$COORDS"
      if [ -s "$R/$COORDS" ]; then
        mkdir -p "$S/fixtures" && cp "$R/$COORDS" "$S/fixtures/$FIXTURE" \
          && echo "### coordinate fixture ($QUAL_CARD) -> $S/fixtures/$FIXTURE (copy it into the repo)"
      fi ;;
    stock2) run_one stock2 "" "$common" ;;
    *) run_one "$v" "$K0_DIR/reader_$v.cpp" "$common" ;;
  esac
done
if [ "${K0_Q4096:-0}" = 1 ]; then
  # Separate run: adding the 4096-row arm to the stock pass would resize the shared K/V pool and change
  # every stock sha, so it could no longer be compared with K0c.
  run_one stock-q256_4096 "" "--arms q256_4096 --starts $STARTS --rounds $ROUNDS --watchdog-s $WATCHDOG_S --no-fallback-scalar"
  echo "### q256_4096: per-token slope ratio r vs stock gives t_c ~ 22 x r us for r > 0.5 (optimistic second opinion)"
fi
summary
exit 0
