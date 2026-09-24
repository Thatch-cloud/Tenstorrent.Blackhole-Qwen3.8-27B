#!/usr/bin/env bash
# M1 of the prefill ranking: the chunked-SDPA isolation bench (sdpa_prefill_bench.py) on the
# qualification card (QUAL_CARD, default card B), in the serving image, no model load, no graft.
#
#   bash run_m1.sh              # all seven arms at chunk_start 0 / 32k / 64k / 126k
#   M1_ARGS="--arms baseline,bf16_kv --rounds 3" bash run_m1.sh     # a quick pass
#   M1_READER=$HOME/kwork-m1/k0/reader_k0a.cpp M1_ARGS="--arms baseline --sha" bash run_m1.sh   # a K0 probe
#
# Env: M1_SRC (the directory holding sdpa_prefill_bench.py; default ~/kwork-m1), IMAGE (the
# serving image the 131k prefill numbers were measured on, sha256:0648ca9a...), RESULTS
# (default $M1_SRC/results), M1_ARGS (extra args), QUAL_CARD (the target board id under
# /dev/tenstorrent/by-id; default card B, blackhole-F36F768B9A5CAFA0; card M or card A, the serving
# pair, is refused unless ALLOW_SERVING_CARD=1). The node is resolved by board id at launch; every run
# gets a fresh kernel cache. Before the bench it greps
# the image's own SDPA factory for the three facts the ranking read from the TT-Sim tree (the
# Q-chunk pair distribution, zigzag balancing, and the chain-forward gate), so the verdict and
# the lever #1 build plan rest on the served source, not the simulator's.
#
# K0 (sdpa-prefill-share-spec.md 3.7 / 7.1):
#   M1_READER=<file>       bind-mounts that ONE file, read-only, over the image's
#                          sdpa/device/kernels/dataflow/reader_interleaved.cpp (never a directory);
#                          the served factory JIT-compiles whatever is there. Inside the container,
#                          before the bench, the mounted file must hash to the host file (else exit 97).
#   M1_REQUIRE_SOURCES=1   also refuse (exit 97) unless the other six served sdpa sources equal the
#                          probe-v25 shas below (and the reader too when M1_READER is unset).
#   M1_DRY_RUN=1           print the docker argv and exit 0: no device, holder or image check, no docker.
# The in-container sha256 of all seven served sdpa sources and of the mounted bench is printed before
# every run (the bench must hash to the host file, else exit 97), and the full launched docker argv is
# echoed ("read the launched argv"). The '### M1 <stamp> node=' line (it names the card too) is printed
# only once every pre-launch check has passed (k0_session.sh reads it as "a container was launched on
# the card"). On a hang: docker rm -f qwen-sdpa-m1-<card tag>, then the printed reset hint (a tt-smi -r
# command that resolves the target's board id when run; that card only).
#
# Prefill lever #1 (sdpa-prefill-share-spec.md 3.7, optimisation/ttnn-op/sdpa_prefill_chain):
#   KOPGRAFT_PF=<dir>      the K64g graft, mounted as lever_n_m3native_run_arm.sh mounts a graft: its
#                          _ttnn.so, its _ttnncpp.so (build_Release/ttnn and build_Release/lib) and every
#                          op directory it carries (attn_prep, nlp_concat_heads_decode, sdpa_decode, sdpa),
#                          read-only. Its sdpa/ must hold reader_interleaved_qwen_chain.cpp; inside the
#                          container the mounted .so files must hash to the host files (else exit 97), and
#                          the chain reader joins the source list (its recorded sha below). Refused together
#                          with M1_READER (a file mount inside a directory mount). IMAGE must be set
#                          explicitly to one of PF_GRAFT_IMAGES, the images build_k64g.sh compared the
#                          graft's sdpa/ against (never this script's 0648ca9a default), and the seven
#                          sdpa sources are then enforced (M1_REQUIRE_SOURCES=1).
#   WATCHER=1              TT_METAL_WATCHER=5 (NoC sanitiser, waypoints, asserts), the watcher log in
#                          $RESULTS/watcher-<stamp>/, and a 900 s container cap instead of 1800 s.
#   QWEN_SDPA_PF_TEST=1    passed into the container: the factory then accepts the test-only flags
#                          (0x100 mutation, 0x200 planted hang) of --program-word arms.
# Device holders (qual_refuse_holders): refuses if a running container can reach the target's node
# (--privileged, a device cgroup rule, or given that node or a directory holding it) or a host process
# holds it (fuser). A container on another board - a CI gate on the serving pair while card B is
# qualified - does not block; with a serving card as the target, any container reaching any card does.
set -euo pipefail
S=${M1_SRC:-$HOME/kwork-m1}
IMAGE_EXPLICIT=${IMAGE:+1}
IMAGE=${IMAGE:-sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae}
# build_k64g.sh's K64G_IMAGES default (A'' eceb2daa, A' 1b9b6445, e41ef884): a KOPGRAFT_PF run is refused
# in any other image (the CPU tests keep the two lists equal).
PF_GRAFT_IMAGES="sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9 sha256:1b9b644549d4409c4fc80e2f92c183e665e7e693c37cccc95b4bb760a6d7d537 sha256:e41ef884f4c8e07ce6632511fac19364af35dc73d601f05f5191784a64a3a768"
REQUIRE_SOURCES=${M1_REQUIRE_SOURCES:-0}
R=${RESULTS:-$S/results}
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
SDPA=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device
FACTORY=$SDPA/sdpa_program_factory.cpp
READER_DST=$SDPA/kernels/dataflow/reader_interleaved.cpp
# The seven served sdpa sources (probe-v25 = images 1b9b6445 / 0648ca9a); the reader is index 1.
SERVED=(
  "fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783  $FACTORY"
  "f97f5490cf476db92d575de33c85f8707f96d7474896ee086ee864c23a3efa27  $READER_DST"
  "a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0  $SDPA/kernels/compute/sdpa.cpp"
  "3fb5da2440c3bf90ebceb8acd55424c7739339de4c6b02db83836e1e3414fa19  $SDPA/kernels/compute/compute_common.hpp"
  "554a0b282d2a36c7b129eef04df67a5becbb14f98220246e43e47fdd56118aa6  $SDPA/kernels/dataflow/dataflow_common.hpp"
  "43a24c466c97beb5d34269630fde6a7831a23246d9df99a03f8c88265044cd9f  $SDPA/kernels/dataflow/chain_link.hpp"
  "2a0959ff7cce507eab39f2e063f4dbc264d8b51f4db9f479b8c2419435c745c2  $SDPA/kernels/dataflow/writer_interleaved.cpp"
)
DRY=${M1_DRY_RUN:-0}
name=qwen-sdpa-m1-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)

test -s "$S/sdpa_prefill_bench.py" || { echo "$S/sdpa_prefill_bench.py missing (set M1_SRC)" >&2; exit 1; }
bench_sha=$(sha256sum "$S/sdpa_prefill_bench.py" | cut -d' ' -f1)
echo "### bench $S/sdpa_prefill_bench.py sha256=$bench_sha -> /bench/sdpa_prefill_bench.py (read-only)"

# M1_READER: exactly one regular file, mounted read-only over the image's reader. Never a directory.
expect=("${SERVED[@]}")
reader_mount=()
if [ -n "${M1_READER:-}" ]; then
  reader_src=$(readlink -f -- "$M1_READER") || { echo "refusing: M1_READER=$M1_READER does not resolve" >&2; exit 1; }
  if [ -d "$reader_src" ] || [ ! -f "$reader_src" ] || [ ! -s "$reader_src" ]; then
    echo "refusing: M1_READER=$M1_READER is not one non-empty regular file (a directory is never mounted)" >&2
    exit 1
  fi
  case "$reader_src" in
    *,*) echo "refusing: M1_READER path contains a comma (breaks --mount): $reader_src" >&2; exit 1 ;;
  esac
  reader_sha=$(sha256sum "$reader_src" | cut -d' ' -f1)
  expect[1]="$reader_sha  $READER_DST"
  reader_mount=(--mount "type=bind,src=$reader_src,dst=$READER_DST,readonly")
  echo "### M1_READER $reader_src sha256=$reader_sha -> $READER_DST (one file, read-only)"
fi

# KOPGRAFT_PF: the K64g graft, exactly the arm's mount targets.
CHAIN_READER=$SDPA/kernels/dataflow/reader_interleaved_qwen_chain.cpp
CHAIN_READER_SHA=eecc1166a209e61dc8498149b5a4338cc278d7106f4ca68d6434f620e942f8d8
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
graft_mounts=()
graft_lines=()
if [ -n "${KOPGRAFT_PF:-}" ]; then
  if [ -n "${M1_READER:-}" ]; then
    echo "refusing: M1_READER and KOPGRAFT_PF together (a file mount inside the graft's sdpa/ mount)" >&2
    exit 1
  fi
  if [ -z "$IMAGE_EXPLICIT" ]; then
    echo "refusing: KOPGRAFT_PF needs IMAGE set explicitly to one of the images build_k64g.sh compared: $PF_GRAFT_IMAGES" >&2
    exit 1
  fi
  case " $PF_GRAFT_IMAGES " in
    *" $IMAGE "*) ;;
    *) echo "refusing: KOPGRAFT_PF in image $IMAGE, which build_k64g.sh did not compare (allowed: $PF_GRAFT_IMAGES)" >&2; exit 1 ;;
  esac
  REQUIRE_SOURCES=1
  graft=$(readlink -f -- "$KOPGRAFT_PF") || { echo "refusing: KOPGRAFT_PF=$KOPGRAFT_PF does not resolve" >&2; exit 1; }
  case "$graft" in
    *,*) echo "refusing: KOPGRAFT_PF path contains a comma (breaks --mount): $graft" >&2; exit 1 ;;
  esac
  for part in _ttnn.so _ttnncpp.so sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp; do
    test -s "$graft/$part" || { echo "refusing: $graft/$part missing (build it with build_k64g.sh)" >&2; exit 1; }
  done
  graft_mounts+=(--mount "type=bind,src=$graft/_ttnn.so,dst=/opt/tt-metal/ttnn/ttnn/_ttnn.so,readonly")
  graft_mounts+=(--mount "type=bind,src=$graft/_ttnncpp.so,dst=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so,readonly")
  graft_mounts+=(--mount "type=bind,src=$graft/_ttnncpp.so,dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly")
  for op in attn_prep:transformer/attn_prep nlp_concat_heads_decode:experimental/transformer/nlp_concat_heads_decode \
            sdpa_decode:transformer/sdpa_decode sdpa:transformer/sdpa; do
    if [ -d "$graft/${op%%:*}" ]; then
      graft_mounts+=(--mount "type=bind,src=$graft/${op%%:*},dst=$OPS/${op#*:},readonly")
    fi
  done
  so_sha=$(sha256sum "$graft/_ttnn.so" | cut -d' ' -f1)
  cpp_sha=$(sha256sum "$graft/_ttnncpp.so" | cut -d' ' -f1)
  graft_lines=("$so_sha  /opt/tt-metal/ttnn/ttnn/_ttnn.so"
               "$cpp_sha  /opt/tt-metal/build_Release/ttnn/_ttnncpp.so"
               "$cpp_sha  /opt/tt-metal/build_Release/lib/_ttnncpp.so")
  expect+=("$CHAIN_READER_SHA  $CHAIN_READER")
  echo "### KOPGRAFT_PF $graft _ttnn.so=${so_sha:0:16} _ttnncpp.so=${cpp_sha:0:16} (+ its op directories, read-only)"
fi

# WATCHER=1: the NoC sanitiser pass (spec 6.2 Q1.2); a shorter container cap.
timeout_s=1800
watcher=()
if [ "${WATCHER:-}" = 1 ]; then
  timeout_s=900
  watcher=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  echo "### WATCHER=1: TT_METAL_WATCHER=5, watcher log $R/watcher-$stamp, container cap ${timeout_s} s"
fi
pf_test=()
if [ "${QWEN_SDPA_PF_TEST:-}" = 1 ]; then
  pf_test=(-e QWEN_SDPA_PF_TEST=1)
  echo "### QWEN_SDPA_PF_TEST=1: the factory accepts the test-only flags 0x100 / 0x200"
fi

if [ "$DRY" = 1 ]; then
  node=$QUAL_BYID
else
  # The target's node, resolved by board id now; never shared: refuse while a container or a host
  # process can reach it (a container on another board does not block).
  qual_card_resolve
  node=$QUAL_NODE
  qual_refuse_holders
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-m1-$stamp"
  chmod 0777 "$R" "$R/kcache-m1-$stamp"
  if [ "${WATCHER:-}" = 1 ]; then
    mkdir -p "$R/watcher-$stamp"
    chmod 0777 "$R/watcher-$stamp"
  fi
  # What the container checks before the bench: every source line, the mounted bench, and
  # (M1_READER) the mounted reader.
  printf '%s\n' "${expect[@]}" > "$R/m1-$stamp.sources.sha256"
  printf '%s\n' "$bench_sha  /bench/sdpa_prefill_bench.py" > "$R/m1-$stamp.bench.sha256"
  if [ -n "${M1_READER:-}" ]; then
    printf '%s\n' "${expect[1]}" > "$R/m1-$stamp.reader.sha256"
  fi
  if [ -n "${KOPGRAFT_PF:-}" ]; then
    printf '%s\n' "${graft_lines[@]}" > "$R/m1-$stamp.graft.sha256"
    chmod 0644 "$R/m1-$stamp.graft.sha256"
  fi
  chmod 0644 "$R/m1-$stamp.sources.sha256" "$R/m1-$stamp.bench.sha256" "$R/m1-$stamp.reader.sha256" 2>/dev/null || true

  echo "### factory facts in ${IMAGE:7:12} ($FACTORY)"
  timeout -k 10 120 docker run --rm --network none --entrypoint sh "$IMAGE" -c \
    "sha256sum $FACTORY; grep -n -E 'global_q_pair_distribute|use_zigzag_balancing|is_causal && !is_chunked|q_num_chunks % 2' $FACTORY" \
    2>&1 | tee "$R/m1-$stamp.factory.txt" || true
fi

# The in-container preamble: the seven shas (the reader line is the mounted file under M1_READER),
# the mount proof, then exec the bench so its exit status is the container's.
paths=("${expect[@]#*  }")
pre="echo '### in-container sha256 of the served sdpa sources (reader_interleaved.cpp = the M1_READER file when set) and the mounted bench'; "
pre+="sha256sum ${paths[*]} /bench/sdpa_prefill_bench.py || echo '### some sdpa sources are missing in this image'; "
pre+="sha256sum -c /results/m1-$stamp.bench.sha256 || { echo '### REFUSING: the in-container bench is not the host file'; exit 97; }; "
if [ -n "${KOPGRAFT_PF:-}" ]; then
  pre+="echo '### in-container sha256 of the mounted graft binaries'; "
  pre+="sha256sum -c /results/m1-$stamp.graft.sha256 || { echo '### REFUSING: the in-container graft .so files are not the KOPGRAFT_PF files'; exit 97; }; "
fi
if [ -n "${M1_READER:-}" ]; then
  pre+="grep -F reader_interleaved.cpp /proc/self/mountinfo || echo '### no mountinfo line for reader_interleaved.cpp'; "
  pre+="sha256sum -c /results/m1-$stamp.reader.sha256 || { echo '### REFUSING: the in-container reader is not the M1_READER file'; exit 97; }; "
fi
pre+="if sha256sum -c --quiet /results/m1-$stamp.sources.sha256; then echo '### sdpa sources: as expected'; "
pre+="else echo '### sdpa sources DIFFER from the expected shas'; [ \"\${M1_REQUIRE_SOURCES:-0}\" = 1 ] && exit 97; fi; "
pre+="exec python3 -B /bench/sdpa_prefill_bench.py --out /results/m1-$stamp.json \"\$@\""

# shellcheck disable=SC2206
extra=(${M1_ARGS:-})
argv=(docker run --rm --name "$name" --network none
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g
  --device "$node"
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G
  --mount "type=bind,src=$S/sdpa_prefill_bench.py,dst=/bench/sdpa_prefill_bench.py,readonly"
  ${reader_mount[@]+"${reader_mount[@]}"}
  ${graft_mounts[@]+"${graft_mounts[@]}"}
  ${watcher[@]+"${watcher[@]}"}
  ${pf_test[@]+"${pf_test[@]}"}
  --mount "type=bind,src=$R,dst=/results"
  --mount "type=bind,src=$R/kcache-m1-$stamp,dst=/kcache"
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  -e "M1_REQUIRE_SOURCES=$REQUIRE_SOURCES"
  --entrypoint sh "$IMAGE" -c "$pre" m1 "${extra[@]}")
echo "### M1 $stamp node=$node card=$QUAL_CARD ($QUAL_TAG) image=${IMAGE:7:12} reader=${M1_READER:-served} graft=${KOPGRAFT_PF:-none} watcher=${WATCHER:-0}"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/m1-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status; report $R/m1-$stamp.json; log $R/m1-$stamp.log"
grep -F 'M1 VERDICT:' "$R/m1-$stamp.log" || true
grep -F 'M1 Q2:' "$R/m1-$stamp.log" || true
# A hang: the watchdog's exit 3, the container cap (124 / 137), or the bench's faulthandler backstop
# (exit 1 with 'Timeout (' when a blocking ttnn call held the GIL and the watchdog thread never ran).
hung=0
case "$status" in
  3|124|137) hung=1 ;;
  1) grep -qF 'Timeout (' "$R/m1-$stamp.log" && hung=1 ;;
esac
if [ "$hung" = 1 ]; then
  # The reset hint resolves the target's node and PCI address again and derives its tt-smi board
  # index from the PCI address (never the /dev/tenstorrent number); it resets nothing itself.
  echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
  qual_reset_hint >&2
fi
exit "$status"
