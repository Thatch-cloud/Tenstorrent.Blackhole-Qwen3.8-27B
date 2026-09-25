# The CI gate's serving pair, for the device-holder check and the board reset of
# .github/workflows/qwen-lever-n-m3native-gate.yml, which sources this file (it sources qual_card.sh).
#
# Card M and card A only, by board id: QUAL_SERVING_CARDS, the two ids lever_n_m3native_run_arm.sh
# mounts (its M3NATIVE_CARDS default; test_qual_card.py keeps the two lists equal). Card B (PCIe only, no
# Ethernet cable) is the qualification card and may be mid-session: nothing here names it, and the gate
# passes tt-smi -r only the two nodes resolved here. tt-smi itself still enumerates every board when it
# lists them (-ls) and when it re-initialises after a reset, as the rig's telemetry exporter does every
# 30 s; a hung card B can therefore fail that re-init, and the gate's reset step with it - the gate
# never resets card B.
#
# Both functions return 1 (the reason on stderr, "refusing: ...") instead of guessing, leaving the two
# arrays empty, and take an optional wait in seconds: a reset returns while the switch cards (A, and B)
# are still hot-cycling, so their by-id links - and their sysfs entries - are missing for a few seconds;
# the functions poll every two seconds for up to that long. A board id on the wrong PCI address, or both
# ids on one node, is refused at once.
#
# serving_pair_nodes [wait]    sets SERVING_PAIR_NODES, the two /dev/tenstorrent/N nodes (readlink -f of
#                              the board ids, at the moment it is called: nodes renumber across a reset).
#                              The holder check needs nothing more.
# serving_pair_resolve [wait]  also sets SERVING_PAIR_PCI, their PCI addresses from sysfs, which must be
#                              0000:d1:00.0 (M) and 0000:f3:00.0 (A), so a reset never follows a board id
#                              whose mapping changed. The gate resets by these nodes (tt-smi -r takes a
#                              /dev/tenstorrent path; a bare number would be tt-smi's own board index,
#                              which renumbers across resets), resolving them again before every call.
# serving_pair_heal [wait] [last] the reset step's self-heal right after the pair reset: a serving card
#                              whose by-id link is still missing is re-probed at the driver, never
#                              reset again (the telemetry race; see below); one whose PCI device is
#                              absent is first rescanned at its own upstream port (a link that did not
#                              train; see below).
serving_pair_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$serving_pair_dir/qual_card.sh"
SERVING_PAIR_EXPECTED_PCI='0000:d1:00.0 0000:f3:00.0'
# Each serving card's own upstream port, index-aligned with the addresses above: card M's AMD root port,
# and card A's PCIe switch downstream port (its secondary bus, f3, holds card A alone).
SERVING_PAIR_UPSTREAM_PORTS='0000:d0:01.1 0000:f2:00.0'

# One attempt (nodes|pci). Returns 1 for what a wait may cure (a missing node, an unreadable PCI
# address), 2 for what it cannot; the reason on stderr.
serving_pair_try() {
  local cards expected i node pci
  read -r -a cards <<< "$QUAL_SERVING_CARDS"
  read -r -a expected <<< "$SERVING_PAIR_EXPECTED_PCI"
  SERVING_PAIR_NODES=()
  SERVING_PAIR_PCI=()
  for i in 0 1; do
    node=$(readlink -f -- "$QUAL_BYID_ROOT/${cards[$i]}" 2>/dev/null || true)
    if [ -z "$node" ] || ! qual_is_char "$node"; then
      echo "serving card ${cards[$i]} ($(qual_card_label "${cards[$i]}")) has no device node under $QUAL_BYID_ROOT" >&2
      return 1
    fi
    if [ "$i" = 1 ] && [ "$node" = "${SERVING_PAIR_NODES[0]}" ]; then
      echo "both serving cards resolve to $node" >&2
      return 2
    fi
    SERVING_PAIR_NODES+=("$node")
    [ "$1" = pci ] || continue
    pci=$(qual_pci_of "$node")
    if [ "$pci" != "${expected[$i]}" ]; then
      echo "${cards[$i]} is $node at PCI ${pci:-unknown}, not ${expected[$i]}; the pair's mapping changed" >&2
      if [ -z "$pci" ]; then return 1; fi
      return 2
    fi
    SERVING_PAIR_PCI+=("$pci")
  done
}

serving_pair_wait() {  # nodes|pci [seconds]
  local waited=0 limit=${2:-0} i cards st
  while :; do
    st=0
    serving_pair_try "$1" 2>/dev/null || st=$?
    if [ "$st" = 0 ]; then
      break
    fi
    if [ "$st" != 1 ] || [ "$waited" -ge "$limit" ]; then
      echo -n "refusing: " >&2
      serving_pair_try "$1" || true
      SERVING_PAIR_NODES=()
      SERVING_PAIR_PCI=()
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  read -r -a cards <<< "$QUAL_SERVING_CARDS"
  for i in 0 1; do
    echo "serving pair: ${cards[$i]} ($(qual_card_label "${cards[$i]}")) -> ${SERVING_PAIR_NODES[$i]}${SERVING_PAIR_PCI[$i]:+, PCI ${SERVING_PAIR_PCI[$i]}}"
  done
  if [ "$waited" != 0 ]; then
    echo "serving pair: both board ids resolved after ${waited} s"
  fi
}

serving_pair_nodes() {
  serving_pair_wait nodes "${1:-0}"
}

serving_pair_resolve() {
  serving_pair_wait pci "${1:-0}"
}

# The self-heal after the pair reset: the telemetry race, and a link that did not train.
#
# After a tt-smi reset a board behind the PCIe switch (card A, 0000:f3:00.0) can re-enumerate before its
# ARC firmware is ready. The driver logs 'tenstorrent 0000:f3:00.0: Telemetry not available'; the board's
# node exists and tt-smi -ls lists it, but udev never creates its by-id link, so every later lookup by
# board id refuses (run v190 lost card A: the arm waited 120 s for the link and refused). Re-probing the
# driver for that one PCI device - unbind, 3 s, bind - brings the link back within seconds, without
# another reset (proven on the qualification card, 2026-09-24 06:09 UTC).
#
# Or the board does not re-enumerate at all (run v217): card A's switch port logged 'pciehp 0000:f2:00.0:
# Slot(1): Cannot train link' after the reset, and 0000:f3:00.0 was gone - no PCI device, no node, no
# by-id link - so there was nothing to re-probe. A rescan of that card's own upstream port (1 into
# /sys/bus/pci/devices/0000:f2:00.0/rescan) brought it back within ~5 s: enumerated, bound to the driver,
# telemetry fine, its by-id link made by udev (verified by hand on the rig after v217). A rescan only
# adds what the kernel does not have: no device already enumerated is removed, reset or re-probed.
#
# serving_pair_heal [wait]  waits up to [wait] s (default 60) for both serving cards' by-id links. A
#   serving card whose link is still missing and whose PCI device is absent is rescanned first: only its
#   own upstream port (0000:d0:01.1 for card M, 0000:f2:00.0 for card A; serving_pair_rescan_write
#   refuses any other), only while that port is in sysfs, at most SERVING_PAIR_RESCANS times, each
#   followed by a wait of up to SERVING_PAIR_RESCAN_WAIT s for the device to be present and bound; then
#   up to SERVING_PAIR_REPROBE_WAIT s for its link. A device still absent after the last rescan refuses
#   (a power-cycle of the PCIe switch or a host reboot is a human's call); one back without its link
#   goes on to the driver re-probe, with every check below.
#   For a serving card whose link is still missing it re-probes that card's driver and waits up to
#   SERVING_PAIR_REPROBE_WAIT s for the link, at most SERVING_PAIR_REPROBES times per card. It re-probes
#   only card M's or card A's own PCI address (0000:d1:00.0, 0000:f3:00.0: the addresses the reset step
#   checked the pair against right before its reset; serving_pair_driver_write refuses any other), and
#   only while - checked again before every attempt - that PCI device exists, it is bound to the
#   tenstorrent driver, exactly one node is at that address, that node is no board's by-id target, and
#   nothing holds it (fuser with sudo -n, as the gate's holder step; that step already passed, this is
#   the re-check). Anything else, or a link still missing after the last re-probe, refuses: return 1,
#   "refusing: ..." on stderr, nothing guessed. Every step is logged on stdout with a [reset] prefix.
# serving_pair_heal [wait] [last]  with [last], the caller's $SECONDS (the step shell's clock, which the
#   pipeline subshell inherits) after which no rescan and no unbind starts: a rescan or re-probe that
#   would begin later is refused, never started, so the runner's step timeout cannot kill the heal between
#   an unbind and its bind and leave the card unbound. Empty: no limit (by hand). Anything but digits
#   refuses.
SERVING_PAIR_DRIVER=tenstorrent
SERVING_PAIR_REPROBES=2
SERVING_PAIR_REPROBE_WAIT=60
SERVING_PAIR_RESCANS=2
SERVING_PAIR_RESCAN_WAIT=30

# "card A (blackhole-..., PCI 0000:f3:00.0)"
serving_pair_card_name() {  # board-id pci
  local label
  label=$(qual_card_label "$1")
  echo "${label%%,*} ($1, PCI $2)"
}

# Whether a board id's by-id link resolves to a device node now.
serving_pair_linked() {  # board-id
  local node
  node=$(readlink -f -- "$QUAL_BYID_ROOT/$1" 2>/dev/null || true)
  [ -n "$node" ] && qual_is_char "$node"
}

# The device nodes whose PCI address (sysfs, by device numbers) is $1, one per line.
serving_pair_nodes_at() {  # pci
  local node
  for node in "$QUAL_TT_ROOT"/[0-9]*; do
    if qual_is_char "$node" && [ "$(qual_pci_of "$node")" = "$1" ]; then
      echo "$node"
    fi
  done
}

# sudo -n unless root, as the gate's steps run fuser and tt-smi.
serving_pair_sudo() {
  if [ "$(id -u)" = 0 ]; then "$@"; else sudo -n "$@"; fi
}

# Nothing holds the node: fuser -v, up to five tries two seconds apart (the rig's telemetry exporter
# opens every board for a moment every 30 s). Otherwise returns 1 with what fuser said on stdout: a
# holder, a failed fuser (sudo refused, say) or no fuser at all is never taken for "nothing holds it".
serving_pair_unheld() {  # node
  local try st out
  if ! command -v fuser > /dev/null 2>&1; then
    echo 'fuser is not installed'
    return 1
  fi
  for try in 1 2 3 4 5; do
    st=0
    out=$(serving_pair_sudo fuser -v "$1" 2>&1) || st=$?
    if [ "$st" != 0 ] && [ -z "$out" ]; then
      return 0
    fi
    [ "$try" = 5 ] || sleep 2
  done
  echo "$out"
  return 1
}

# Writes a PCI address to the tenstorrent driver's unbind or bind file: one of the two writes this file
# makes (the other is serving_pair_rescan_write). It refuses (2), touching nothing, any address that is
# not card M's or card A's.
serving_pair_driver_write() {  # pci unbind|bind
  local known ok=0
  for known in $SERVING_PAIR_EXPECTED_PCI; do
    if [ "$1" = "$known" ]; then ok=1; fi
  done
  if [ "$ok" != 1 ]; then
    echo "refusing: '$1' is not card M's or card A's PCI address ($SERVING_PAIR_EXPECTED_PCI); no driver write" >&2
    return 2
  fi
  case $2 in
    unbind|bind) ;;
    *) echo "refusing: '$2' is not unbind or bind" >&2; return 2 ;;
  esac
  echo "$1" | serving_pair_sudo tee "$QUAL_SYS_ROOT/bus/pci/drivers/$SERVING_PAIR_DRIVER/$2" > /dev/null
}

# Writes 1 to an upstream port's rescan file: the other write this file makes. It refuses (2), touching
# nothing, any port that is not card M's or card A's own upstream port.
serving_pair_rescan_write() {  # port
  local known ok=0
  for known in $SERVING_PAIR_UPSTREAM_PORTS; do
    if [ "$1" = "$known" ]; then ok=1; fi
  done
  if [ "$ok" != 1 ]; then
    echo "refusing: '$1' is not card M's or card A's upstream port ($SERVING_PAIR_UPSTREAM_PORTS); no rescan" >&2
    return 2
  fi
  echo 1 | serving_pair_sudo tee "$QUAL_SYS_ROOT/bus/pci/devices/$1/rescan" > /dev/null
}

# The upstream port of card M's or card A's PCI address; nothing for any other address.
serving_pair_upstream_of() {  # pci
  local expected ports i
  read -r -a expected <<< "$SERVING_PAIR_EXPECTED_PCI"
  read -r -a ports <<< "$SERVING_PAIR_UPSTREAM_PORTS"
  for i in "${!expected[@]}"; do
    if [ "$1" = "${expected[$i]}" ]; then echo "${ports[$i]:-}"; fi
  done
}

serving_pair_present() {  # pci
  [ -e "$QUAL_SYS_ROOT/bus/pci/devices/$1" ]
}

serving_pair_bound() {  # pci
  [ -e "$QUAL_SYS_ROOT/bus/pci/drivers/$SERVING_PAIR_DRIVER/$1" ]
}

# May the card's PCI device be re-probed now? Sets serving_pair_heal_node (the node at that address) or
# serving_pair_heal_why (why not) and returns 1. Never called in a subshell: it sets globals.
serving_pair_reprobe_check() {  # board-id pci
  local pci=$2 node link target holders
  local found=()
  serving_pair_heal_node=
  serving_pair_heal_why=
  if [ ! -e "$QUAL_SYS_ROOT/bus/pci/devices/$pci" ]; then
    serving_pair_heal_why="PCI device $pci is absent (the board did not re-enumerate; not the telemetry race)"
    return 1
  fi
  if ! serving_pair_bound "$pci"; then
    serving_pair_heal_why="PCI device $pci is not bound to the $SERVING_PAIR_DRIVER driver"
    return 1
  fi
  while IFS= read -r node; do
    if [ -n "$node" ]; then found+=("$node"); fi
  done < <(serving_pair_nodes_at "$pci")
  if [ "${#found[@]}" != 1 ]; then
    serving_pair_heal_why="${#found[@]} device nodes are at PCI $pci${found[0]+ (${found[*]})}, not one"
    return 1
  fi
  node=${found[0]}
  for link in "$QUAL_BYID_ROOT"/*; do
    target=$(readlink -f -- "$link" 2>/dev/null || true)
    if [ -n "$target" ] && [ "$target" = "$node" ]; then
      serving_pair_heal_why="$node, at PCI $pci, is the by-id target of ${link##*/} ($(qual_card_label "${link##*/}"))"
      return 1
    fi
  done
  if ! holders=$(serving_pair_unheld "$node"); then
    serving_pair_heal_why="$node is held (or its holders could not be checked): ${holders//$'\n'/; }"
    return 1
  fi
  serving_pair_heal_node=$node
}

# One card whose PCI device is absent: at most SERVING_PAIR_RESCANS rescans of its upstream port, each
# checked first, each followed by a wait for the device (present and bound); then a wait for its by-id
# link. Returns 0 when the link is back, 3 when the device is back without it (the caller goes on to the
# driver re-probe, which checks everything again), 1 when it refuses.
serving_pair_rescan_card() {  # board-id pci [last]
  local card=$1 pci=$2 last=${3:-} name port rescan=1 waited
  name=$(serving_pair_card_name "$card" "$pci")
  port=$(serving_pair_upstream_of "$pci")
  while ! serving_pair_present "$pci"; do
    if [ "$rescan" -gt "$SERVING_PAIR_RESCANS" ]; then
      echo "refusing: $name PCI device $pci still absent after $SERVING_PAIR_RESCANS rescans of its upstream port $port; no driver re-probe." >&2
      echo "  Its link did not train (dmesg: pciehp 'Cannot train link'): a power-cycle of the PCIe switch or a host reboot" >&2
      echo "  brings it back - a human action, never this step's." >&2
      return 1
    fi
    if [ -z "$port" ] || [ ! -e "$QUAL_SYS_ROOT/bus/pci/devices/$port" ]; then
      echo "refusing: $name PCI device $pci absent after reset, and its upstream port ${port:-(none known)} is not in sysfs; no rescan" >&2
      return 1
    fi
    if [ -n "$last" ] && [ "$SECONDS" -gt "$last" ]; then
      echo "refusing: $name PCI device $pci absent after reset, but the step is ${SECONDS} s in and no rescan may start after ${last} s;" >&2
      echo "  no rescan $rescan/$SERVING_PAIR_RESCANS" >&2
      return 1
    fi
    echo "[reset] $name PCI device absent after reset (link did not train); rescan of its upstream port $port $rescan/$SERVING_PAIR_RESCANS"
    if ! serving_pair_rescan_write "$port"; then
      echo "refusing: the rescan of $port failed; $name PCI device $pci is still absent" >&2
      return 1
    fi
    waited=0
    while ! { serving_pair_present "$pci" && serving_pair_bound "$pci"; } && [ "$waited" -lt "$SERVING_PAIR_RESCAN_WAIT" ]; do
      sleep 2
      waited=$((waited + 2))
    done
    if serving_pair_present "$pci"; then
      if ! serving_pair_bound "$pci"; then
        echo "[reset]   $pci is back ${waited} s after rescan $rescan/$SERVING_PAIR_RESCANS, but not bound to $SERVING_PAIR_DRIVER"
        return 3
      fi
      echo "[reset]   $pci is back ${waited} s after rescan $rescan/$SERVING_PAIR_RESCANS, bound to $SERVING_PAIR_DRIVER"
      break
    fi
    echo "[reset]   $pci still absent ${waited} s after rescan $rescan/$SERVING_PAIR_RESCANS"
    rescan=$((rescan + 1))
  done
  waited=0
  while ! serving_pair_linked "$card" && [ "$waited" -lt "$SERVING_PAIR_REPROBE_WAIT" ]; do
    sleep 2
    waited=$((waited + 2))
  done
  if serving_pair_linked "$card"; then
    echo "[reset] $name by-id link back ${waited} s after rescan $rescan/$SERVING_PAIR_RESCANS -> $(readlink -f -- "$QUAL_BYID_ROOT/$card")"
    return 0
  fi
  echo "[reset] $name by-id link still missing ${waited} s after rescan $rescan/$SERVING_PAIR_RESCANS; on to the driver re-probe"
  return 3
}

# One card: if its PCI device is absent, the rescans above first; then at most SERVING_PAIR_REPROBES
# driver re-probes, each checked first, each followed by a wait for the by-id link.
serving_pair_heal_card() {  # board-id pci [last]
  local card=$1 pci=$2 last=${3:-} name attempt=1 waited st
  name=$(serving_pair_card_name "$card" "$pci")
  if ! serving_pair_present "$pci"; then
    st=0
    serving_pair_rescan_card "$card" "$pci" "$last" || st=$?
    if [ "$st" != 3 ]; then return "$st"; fi
  fi
  while [ "$attempt" -le "$SERVING_PAIR_REPROBES" ]; do
    if serving_pair_linked "$card"; then
      echo "[reset] $name by-id link is present now; no re-probe"
      return 0
    fi
    if ! serving_pair_reprobe_check "$card" "$pci"; then
      echo "refusing: $name by-id missing after reset, and $serving_pair_heal_why; no driver re-probe" >&2
      return 1
    fi
    if [ -n "$last" ] && [ "$SECONDS" -gt "$last" ]; then
      echo "refusing: $name by-id missing after reset, but the step is ${SECONDS} s in and no unbind may start after ${last} s" >&2
      echo "  (its timeout could kill it before the bind); no driver re-probe $attempt/$SERVING_PAIR_REPROBES" >&2
      return 1
    fi
    echo "[reset] $name by-id missing after reset (telemetry race); driver re-probe $attempt/$SERVING_PAIR_REPROBES"
    echo "[reset]   $serving_pair_heal_node is at PCI $pci, bound to $SERVING_PAIR_DRIVER, no holder: unbind $pci"
    if ! serving_pair_driver_write "$pci" unbind; then
      # A refused sudo writes nothing, and the kernel fails an unbind only for a device it had not bound;
      # either way, say what the card is now rather than assume it.
      if serving_pair_bound "$pci"; then
        echo "refusing: unbinding $pci failed; $name is still bound, as it was" >&2
      else
        echo "refusing: unbinding $pci failed and $pci is UNBOUND now; $name is out of service until it is bound:" >&2
        echo "  echo $pci | sudo -n tee /sys/bus/pci/drivers/$SERVING_PAIR_DRIVER/bind" >&2
      fi
      return 1
    fi
    sleep 3
    echo "[reset]   bind $pci"
    serving_pair_driver_write "$pci" bind || true
    if ! serving_pair_bound "$pci"; then
      echo "[reset]   $pci is not bound after the bind; binding again in 3 s"
      sleep 3
      serving_pair_driver_write "$pci" bind || true
      if ! serving_pair_bound "$pci"; then
        echo "refusing: $pci is UNBOUND after driver re-probe $attempt/$SERVING_PAIR_REPROBES; $name is out of service until it is bound:" >&2
        echo "  echo $pci | sudo -n tee /sys/bus/pci/drivers/$SERVING_PAIR_DRIVER/bind" >&2
        return 1
      fi
    fi
    waited=0
    while ! serving_pair_linked "$card" && [ "$waited" -lt "$SERVING_PAIR_REPROBE_WAIT" ]; do
      sleep 2
      waited=$((waited + 2))
    done
    if serving_pair_linked "$card"; then
      echo "[reset] $name by-id link back ${waited} s after driver re-probe $attempt/$SERVING_PAIR_REPROBES -> $(readlink -f -- "$QUAL_BYID_ROOT/$card")"
      return 0
    fi
    echo "[reset] $name by-id link still missing ${waited} s after driver re-probe $attempt/$SERVING_PAIR_REPROBES"
    attempt=$((attempt + 1))
  done
  echo "refusing: $name by-id link still missing after $SERVING_PAIR_REPROBES driver re-probes; not guessing." >&2
  echo "  Check dmesg for 'Telemetry not available', then reset card M and card A together." >&2
  return 1
}

serving_pair_heal() {  # [seconds] [last]
  local limit=${1:-60} last=${2:-} waited=0 cards expected i missing
  # A non-numeric wait would make the -ge below an error, which is false: the wait would never end.
  case $limit in ''|*[!0-9]*) echo "refusing: serving_pair_heal wait '$limit' is not a number of seconds" >&2; return 1 ;; esac
  case $last in *[!0-9]*) echo "refusing: serving_pair_heal last '$last' is not a number of seconds" >&2; return 1 ;; esac
  read -r -a cards <<< "$QUAL_SERVING_CARDS"
  read -r -a expected <<< "$SERVING_PAIR_EXPECTED_PCI"
  while :; do
    missing=()
    for i in 0 1; do
      serving_pair_linked "${cards[$i]}" || missing+=("$i")
    done
    if [ "${#missing[@]}" = 0 ] || [ "$waited" -ge "$limit" ]; then
      break
    fi
    sleep 2
    waited=$((waited + 2))
  done
  if [ "${#missing[@]}" = 0 ]; then
    echo "[reset] both serving cards' by-id links present ${waited} s after the reset; no re-probe"
    return 0
  fi
  for i in "${missing[@]}"; do
    echo "[reset] $(serving_pair_card_name "${cards[$i]}" "${expected[$i]}") by-id link still missing ${waited} s after the reset"
    serving_pair_heal_card "${cards[$i]}" "${expected[$i]}" "$last" || return 1
  done
  echo "[reset] both serving cards' by-id links present after the self-heal"
}
