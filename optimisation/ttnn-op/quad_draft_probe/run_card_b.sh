#!/usr/bin/env bash
# Q0 for Q4, one four-user 64-row draft pass (QWEN_FAST_QUAD_DRAFT; quad-draft-plan.md section 5): run
# probe_quad_draft_card_b.py on the qualification card (QUAL_CARD, default card B), in the gate's image (P6, the
# v218-v224 image). The served graft is mounted exactly as lever_n_m3native_run_arm.sh mounts it for the v220 argv
# (KOPGRAFT64 K64i): _ttnn.so, _ttnncpp.so (both paths), attn_prep, nlp_concat_heads_decode, sdpa_decode and sdpa.
# The matmul factory is in that _ttnncpp.so and the kernels JIT from the image's tree (R1: the probe hashes both).
# The served modules the cases call and the probe's candidates (quad_candidates.py and quad_conv_io.cpp, the
# probe-local 64-row conv I/O kernel, E1/E1b) are this checkout's, mounted at /bench beside the harness (never over
# /experiment-scripts/ci), so the probe needs no new image. No build, no weights. NOT a CI workflow: run it by hand.
#
#   bash run_card_b.sh                # the cores pass (TT_METAL_DEVICE_PROFILER=1, its own process), then Q0a-Q0e
#   WATCHER=1 bash run_card_b.sh      # THE FIRST HARDWARE PASS (per_core_M=2, 64-row norms and heads, the NQH 64 /
#                                     # NKH 16 fold, E1/E1b and the 64-row concats were never run): TT_METAL_WATCHER=5,
#                                     # seed 0, two regimes per step, one weight shard, Q0a-Q0d, no timing, no cores
#                                     # pass, a per-call watchdog (WATCHDOG_S, default 120 s), a 2,700 s container
#                                     # timeout; the watcher's own log lands in $RESULTS/watcher-<stamp>/
#   QUAD_DRY_RUN=1 bash run_card_b.sh # print the launch argv and exit: no node resolution, no holder check, no
#                                     # docker, nothing launched (the graft is checked when it exists)
#
# Before anything is launched the graft must verify against its MANIFEST.sha256 and its _ttnncpp.so must be
# EXPECT_TTNNCPP_SHA256 (default K64i, cf54d716; empty skips). The harness checks the mapped binary again inside the
# container, refuses an SDPA source tree that is not the T16 admission's, and records the op-source trees.
#
# Env: KOPGRAFT64 (default ~/opgraft-K64i; 'none' runs the image's own binary and op tree, and then expects no
# binary unless EXPECT_TTNNCPP_SHA256 is set), IMAGE (default image P6), RESULTS (~/kwork64/quaddraft/<card tag>),
# CARD_B_ARGS (extra harness args, appended last, e.g. "--steps Q0a --ops matmul-q"), QUAD_CORES (1: the cores pass
# first, the default except under WATCHER=1), EXPECT_TTNNCPP_SHA256, WATCHER=1, WATCHDOG_S, QUAD_DRY_RUN=1, QUAL_CARD
# (the target board id under /dev/tenstorrent/by-id; default card B, blackhole-F36F768B9A5CAFA0; card M or card A,
# the serving pair, is refused unless ALLOW_SERVING_CARD=1, which prints a loud warning).
# QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is set as the arm sets it. Every run gets a fresh kernel cache.
#
# ON A HANG (the WATCHDOG line and exit 3; exit 1 with 'Timeout (' in the log, the faulthandler backstop when a
# blocking call holds the GIL; or the timeout's 124 / 137): the EXIT trap removes the container. Then reset THE
# TARGET CARD ONLY with the hint printed below. This script never resets anything itself.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ci=$(cd "$here/../../../scripts/ci" 2>/dev/null && pwd || echo "$here/../../../scripts/ci")
G=${KOPGRAFT64:-$HOME/opgraft-K64i}
IMAGE=${IMAGE:-sha256:c9a585ef3eebb775c8de1883b0e0032ec3b16305feb1e23968e0661f401363e9}
K64I_SHA256=cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4
if [ "$G" = none ]; then
  EXPECT=${EXPECT_TTNNCPP_SHA256-}
else
  EXPECT=${EXPECT_TTNNCPP_SHA256-$K64I_SHA256}
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
R=${RESULTS:-$HOME/kwork64/quaddraft/$QUAL_TAG}
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
DRY=${QUAD_DRY_RUN:-0}
name=qwen-quaddraft-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=5400   # the cores pass, Q0a (22 ops x 3 seeds x 4 regimes, 2 shards per matmul), Q0b-Q0d, then Q0e

# The target's node, resolved by board id now; never shared: refuse while a container or a host
# process can reach it (a container on another board - a CI gate on the serving pair - does not block).
if [ "$DRY" = 1 ]; then
  node=$QUAL_BYID
else
  qual_card_resolve
  node=$QUAL_NODE
  qual_refuse_holders
fi

HARNESS=("$here/probe_quad_draft_card_b.py" "$here/quad_candidates.py" "$here/quad_conv_io.cpp"
         "$ci/pair_row_exact.py" "$ci/draft_attention.py" "$ci/dflash_batched_mask.py" "$ci/draft_shared_head.py"
         "$ci/draft_head_preparation.py" "$ci/dflash_t16_native_attention.py" "$ci/draft_mlp.py"
         "$ci/draft_convolution_fused_io.cpp" "$ci/draft_convolution_fused_compute.cpp")
for file in "${HARNESS[@]}"; do
  test -s "$file" || { echo "refusing: $file missing (run from a checkout: the harness mounts scripts/ci's served modules)" >&2; exit 1; }
done

# The served graft, checked before anything is launched (in a dry run, only when it exists here).
KM=()
if [ "$G" = none ]; then
  echo "### KOPGRAFT64=none: the image's own _ttnncpp.so and op tree (not what v218-v224 serve)"
else
  if [ "$DRY" = 1 ] && [ ! -e "$G" ]; then
    echo "### dry run: $G does not exist here; the graft was not checked"
  else
    for part in _ttnn.so _ttnncpp.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
      test -e "$G/$part" || { echo "refusing: $G/$part missing (the probe needs the served graft, build_k64i.sh)" >&2; exit 1; }
    done
    (cd "$G" && sha256sum -c --quiet MANIFEST.sha256 >&2) \
      || { echo "refusing: $G/MANIFEST.sha256 does not verify (not the graft its build script made)" >&2; exit 1; }
    so_sha=$(sha256sum "$G/_ttnncpp.so" | cut -c1-64)
    if [ -n "$EXPECT" ] && [ "$so_sha" != "$EXPECT" ]; then
      echo "refusing: $G/_ttnncpp.so is ${so_sha:0:16}, not ${EXPECT:0:16} (EXPECT_TTNNCPP_SHA256= skips)" >&2
      exit 1
    fi
    echo "### graft $G: _ttnncpp.so ${so_sha:0:16}, manifest verified"
  fi
  KM=(--mount "type=bind,src=$G/_ttnn.so,dst=/opt/tt-metal/ttnn/ttnn/_ttnn.so,readonly"
      --mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so,readonly"
      --mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly")
  for op in attn_prep:transformer/attn_prep nlp_concat_heads_decode:experimental/transformer/nlp_concat_heads_decode \
            sdpa_decode:transformer/sdpa_decode sdpa:transformer/sdpa; do
    KM+=(--mount "type=bind,src=$G/${op%%:*},dst=$OPS/${op#*:},readonly")
  done
fi

if [ "$DRY" != 1 ]; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-$stamp"
  chmod 0777 "$R" "$R/kcache-$stamp"
fi

args=(--out "/results/probe-$stamp.json" --expect-binary-sha256 "$EXPECT")
BM=()
for file in "${HARNESS[@]}"; do
  BM+=(--mount "type=bind,src=$file,dst=/bench/$(basename "$file"),readonly")
done
WM=()
CORES=${QUAD_CORES:-1}
if [ "${WATCHER:-}" = "1" ]; then
  # One pass under the NoC sanitiser over every new program; timing and the profiler are meaningless here.
  # CARD_B_ARGS, appended last, can widen it.
  timeout_s=2700
  CORES=${QUAD_CORES:-0}
  args+=(--steps Q0a,Q0b,Q0c,Q0d --seeds 0 --regimes normal,row0x100 --b-regimes normal,partner100 --shards 1
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
CM=()
if [ "$CORES" = 1 ]; then
  # The cores pass: its own python process under the device profiler, its raw log kept under $R.
  CM=(-e "QUAD_CORES_OUT=/results/cores-$stamp.json" -e "QUAD_PROFILER_DIR=/results/profiler-$stamp")
  if [ "$DRY" != 1 ]; then
    mkdir -p "$R/profiler-$stamp"
    chmod 0777 "$R/profiler-$stamp"
  fi
fi
# shellcheck disable=SC2206
extra=(${CARD_B_ARGS:-})

# The container's script: record the binaries and kernels it runs, the cores pass (when asked; a hang there
# stops the run with its exit 3), then the probe with the cores report folded in.
# One line (printf %q of a newline is $'...', which the runner test's shlex cannot read).
inner='sha256sum /opt/tt-metal/build_Release/lib/_ttnncpp.so /opt/tt-metal/build_Release/ttnn/_ttnncpp.so '
inner+="$OPS/transformer/sdpa/device/kernels/compute/sdpa.cpp "
inner+="$OPS/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp 2>&1; "
inner+='if [ -n "${QUAD_CORES_OUT:-}" ]; then '
inner+='TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$QUAD_PROFILER_DIR" '
inner+='python3 -B /bench/probe_quad_draft_card_b.py "$@" --steps cores --out "$QUAD_CORES_OUT"; '
inner+='cores=$?; echo "### cores pass exit $cores"; if [ "$cores" = 3 ]; then exit 3; fi; '
inner+='set -- "$@" --cores-report "$QUAD_CORES_OUT"; fi; '
inner+='exec python3 -B /bench/probe_quad_draft_card_b.py "$@"'

argv=(docker run --rm --name "$name" --network none
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g
  --device "$node"
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G
  "${BM[@]}"
  --mount "type=bind,src=$R,dst=/results"
  --mount "type=bind,src=$R/kcache-$stamp,dst=/kcache"
  ${KM[@]+"${KM[@]}"}
  ${WM[@]+"${WM[@]}"}
  ${CM[@]+"${CM[@]}"}
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
  --entrypoint sh "$IMAGE" -c "$inner"
  probe "${args[@]}" ${extra[@]+"${extra[@]}"})
echo "### quaddraft $stamp card=$QUAL_CARD ($QUAL_TAG) node=$node image=${IMAGE:7:12} graft=$G watcher=${WATCHER:-0} cores=$CORES"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/probe-$stamp.log"
status=${PIPESTATUS[0]}
log=$R/probe-$stamp.log
echo "### exit $status; report $R/probe-$stamp.json"
if [ "${WATCHER:-}" = "1" ]; then
  wlog=$R/watcher-$stamp/watcher.log
  if [ -s "$wlog" ]; then
    echo "### watcher log $wlog: $(grep -ciE 'error|assert|tripped|sanitiz' "$wlog" || true) error/assert lines"
    grep -iE 'error|assert|tripped|sanitiz' "$wlog" | head -20 || true
  else
    echo "### no watcher log at $wlog"
  fi
fi
grep -E '^QUAD_PROBE step=' "$log" | sed 's/^/### /' || echo '### no QUAD_PROBE step line'
echo "### summary: $(grep -E '^\{"' "$log" | tail -1 || echo 'no JSON summary line')"
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
