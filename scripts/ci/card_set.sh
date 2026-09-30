# The four-card set (TP4, C2_CARDS=quad in qwen-c2-serving.yml): EVERY Blackhole board present under
# /dev/tenstorrent/by-id, resolved when it is used. Nothing here names a board: the set is whatever by-id links
# exist (blackhole-*), and it must be exactly CARD_SET_EXPECT of them (4 by default) - more or fewer refuses,
# never guesses. /dev/tenstorrent/N numbers, tt-smi indices and UMD chip ids all renumber on every reset, so
# each is read at the moment of use.
#
# Why all four at once: Ethernet links train only at board init, and the four-card fabric is a full mesh (every
# board is cabled to every other), so a reset of any subset leaves links to the rest untrained. The four-card
# reset step resets every board of the set in one tt-smi call, then heals the two traps the rig has shown after
# resets (memory tt-rig-hardware-topology): a board behind the PCIe switch comes back late (the functions poll),
# comes back without its by-id link (the telemetry race: driver unbind/bind), or does not re-enumerate at all (a
# link that did not train: a rescan of that board's OWN upstream port, read from sysfs before the reset).
#
# Every function returns non-zero with the reason on stderr ("refusing: ...") instead of guessing, and every
# write it makes is to an address or port of the snapshot taken before the reset (card_set_resolve), never any
# other.
#
# card_set_nodes [wait]        sets CARD_SET_IDS and CARD_SET_NODES (readlink -f of each by-id link, now).
# card_set_resolve [wait]      also CARD_SET_PCI (sysfs) and CARD_SET_PORTS (each board's upstream port, the
#                              parent of its PCI device in sysfs), and keeps that as the heal's snapshot
#                              (CARD_SET_SNAPSHOT_IDS / _PCI / _PORTS).
# card_set_unheld OUT          nothing but the telemetry exporter's tt-smi holds any node of the set (six tries two
#                              seconds apart, fuser -v into OUT); 1 when held.
# card_set_heal [wait] [last]  after the reset: waits for every snapshot board's by-id link, rescans the own upstream
#                              port of a board whose PCI device is absent, re-probes the driver of one that is back
#                              without its link; [last] is the step's $SECONDS after which no rescan or unbind starts.
# card_set_devices             prints the set's nodes, one per line (the docker --device list).
card_set_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$card_set_dir/qual_card.sh"
CARD_SET_EXPECT=${CARD_SET_EXPECT:-4}
CARD_SET_DRIVER=tenstorrent
CARD_SET_REPROBES=2
CARD_SET_REPROBE_WAIT=60
CARD_SET_RESCANS=2
CARD_SET_RESCAN_WAIT=30
CARD_SET_IDS=()
CARD_SET_NODES=()
CARD_SET_PCI=()
CARD_SET_PORTS=()
CARD_SET_SNAPSHOT_IDS=()
CARD_SET_SNAPSHOT_PCI=()
CARD_SET_SNAPSHOT_PORTS=()

card_set_sudo() {
  if [ "$(id -u)" = 0 ]; then "$@"; else sudo -n "$@"; fi
}

card_set_is_pci() {
  case $1 in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f]:[0-9a-f][0-9a-f]:[0-9a-f][0-9a-f].[0-7]) return 0 ;;
  esac
  return 1
}

# The board ids present now: every by-id name that starts blackhole- and holds only board-id characters, sorted.
card_set_present() {
  local link name
  for link in "$QUAL_BYID_ROOT"/blackhole-*; do
    [ -e "$link" ] || [ -L "$link" ] || continue
    name=${link##*/}
    case $name in
      *[!A-Za-z0-9._-]*) continue ;;
    esac
    echo "$name"
  done | LC_ALL=C sort
}

# A PCI device's upstream port: the parent of its sysfs device directory (a root port or a switch downstream
# port), or nothing.
card_set_upstream_of() {
  local dev parent
  dev=$(readlink -e -- "$QUAL_SYS_ROOT/bus/pci/devices/$1" 2>/dev/null) || return 0
  parent=${dev%/*}
  parent=${parent##*/}
  if card_set_is_pci "$parent" && [ -e "$QUAL_SYS_ROOT/bus/pci/devices/$parent" ]; then
    echo "$parent"
  fi
}

# One attempt (nodes|pci): 1 for what a wait may cure (a board short, a node or PCI address not there yet), 2 for
# what it cannot (more boards than expected, two ids on one node, a board with no upstream port).
card_set_try() {
  local ids id node pci port i
  CARD_SET_IDS=()
  CARD_SET_NODES=()
  CARD_SET_PCI=()
  CARD_SET_PORTS=()
  mapfile -t ids < <(card_set_present)
  if [ "${#ids[@]}" -gt "$CARD_SET_EXPECT" ]; then
    echo "${#ids[@]} Blackhole boards are present (${ids[*]}), not $CARD_SET_EXPECT" >&2
    return 2
  fi
  for id in "${ids[@]}"; do
    node=$(readlink -f -- "$QUAL_BYID_ROOT/$id" 2>/dev/null || true)
    if [ -z "$node" ] || ! qual_is_char "$node"; then
      echo "board $id has no device node under $QUAL_TT_ROOT" >&2
      return 1
    fi
    for i in "${!CARD_SET_NODES[@]}"; do
      if [ "${CARD_SET_NODES[$i]}" = "$node" ]; then
        echo "boards ${CARD_SET_IDS[$i]} and $id both resolve to $node" >&2
        return 2
      fi
    done
    CARD_SET_IDS+=("$id")
    CARD_SET_NODES+=("$node")
    [ "$1" = pci ] || continue
    pci=$(qual_pci_of "$node")
    if [ -z "$pci" ]; then
      echo "board $id ($node) has no PCI address in sysfs" >&2
      return 1
    fi
    port=$(card_set_upstream_of "$pci")
    if [ -z "$port" ]; then
      echo "board $id (PCI $pci) has no upstream port in sysfs" >&2
      return 2
    fi
    CARD_SET_PCI+=("$pci")
    CARD_SET_PORTS+=("$port")
  done
  if [ "${#CARD_SET_IDS[@]}" != "$CARD_SET_EXPECT" ]; then
    echo "${#CARD_SET_IDS[@]} Blackhole boards are present, not $CARD_SET_EXPECT" >&2
    return 1
  fi
}

card_set_wait() {  # nodes|pci [seconds]
  local waited=0 limit=${2:-0} st i
  case $limit in ''|*[!0-9]*) echo "refusing: card set wait '$limit' is not a number of seconds" >&2; return 1 ;; esac
  while :; do
    st=0
    card_set_try "$1" 2>/dev/null || st=$?
    if [ "$st" = 0 ]; then
      break
    fi
    if [ "$st" != 1 ] || [ "$waited" -ge "$limit" ]; then
      echo -n "refusing: " >&2
      card_set_try "$1" || true
      CARD_SET_IDS=()
      CARD_SET_NODES=()
      CARD_SET_PCI=()
      CARD_SET_PORTS=()
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  for i in "${!CARD_SET_IDS[@]}"; do
    echo "card set: ${CARD_SET_IDS[$i]} -> ${CARD_SET_NODES[$i]}${CARD_SET_PCI[$i]:+, PCI ${CARD_SET_PCI[$i]}}${CARD_SET_PORTS[$i]:+, upstream ${CARD_SET_PORTS[$i]}}"
  done
  if [ "$waited" != 0 ]; then
    echo "card set: all $CARD_SET_EXPECT boards resolved after ${waited} s"
  fi
}

card_set_nodes() {
  card_set_wait nodes "${1:-0}"
}

card_set_resolve() {
  card_set_wait pci "${1:-0}" || return 1
  CARD_SET_SNAPSHOT_IDS=("${CARD_SET_IDS[@]}")
  CARD_SET_SNAPSHOT_PCI=("${CARD_SET_PCI[@]}")
  CARD_SET_SNAPSHOT_PORTS=("${CARD_SET_PORTS[@]}")
}

card_set_devices() {
  local node
  for node in "${CARD_SET_NODES[@]}"; do echo "$node"; done
}

# Nothing but tt-smi (the rig's telemetry exporter opens every board for a moment every 30 s) holds a node of the
# set: fuser -v on all of them, six tries two seconds apart, its output in $1.
card_set_unheld() {  # out
  local out=$1 attempt status
  if [ "${#CARD_SET_NODES[@]}" = 0 ]; then
    echo 'refusing: the card set is not resolved (card_set_nodes first)' >&2
    return 1
  fi
  for attempt in 1 2 3 4 5 6; do
    status=0
    card_set_sudo fuser -v "${CARD_SET_NODES[@]}" > "$out" 2>&1 || status=$?
    if [ "$status" = 1 ] && [ ! -s "$out" ]; then return 0; fi
    if [ "$status" = 0 ] && ! grep -v -E '^\s*USER|tt-smi|^/dev/tenstorrent/[0-9]+:\s*$' "$out" | grep -q .; then return 0; fi
    [ "$attempt" = 6 ] || sleep 2
  done
  return 1
}

card_set_in_snapshot() {  # value array-name
  local -n list=$2
  local item
  for item in "${list[@]}"; do
    [ "$item" = "$1" ] && return 0
  done
  return 1
}

# The heal's two writes. Each refuses (2), touching nothing, any address or port the pre-reset snapshot does not hold.
card_set_driver_write() {  # pci unbind|bind
  if ! card_set_is_pci "$1" || ! card_set_in_snapshot "$1" CARD_SET_SNAPSHOT_PCI; then
    echo "refusing: '$1' is not the PCI address of a board of the card set (${CARD_SET_SNAPSHOT_PCI[*]}); no driver write" >&2
    return 2
  fi
  case $2 in
    unbind|bind) ;;
    *) echo "refusing: '$2' is not unbind or bind" >&2; return 2 ;;
  esac
  echo "$1" | card_set_sudo tee "$QUAL_SYS_ROOT/bus/pci/drivers/$CARD_SET_DRIVER/$2" > /dev/null
}

card_set_rescan_write() {  # port
  if ! card_set_is_pci "$1" || ! card_set_in_snapshot "$1" CARD_SET_SNAPSHOT_PORTS; then
    echo "refusing: '$1' is not the upstream port of a board of the card set (${CARD_SET_SNAPSHOT_PORTS[*]}); no rescan" >&2
    return 2
  fi
  echo 1 | card_set_sudo tee "$QUAL_SYS_ROOT/bus/pci/devices/$1/rescan" > /dev/null
}

card_set_linked() {  # board-id
  local node
  node=$(readlink -f -- "$QUAL_BYID_ROOT/$1" 2>/dev/null || true)
  [ -n "$node" ] && qual_is_char "$node"
}

card_set_present_pci() {  # pci
  [ -e "$QUAL_SYS_ROOT/bus/pci/devices/$1" ]
}

card_set_bound() {  # pci
  [ -e "$QUAL_SYS_ROOT/bus/pci/drivers/$CARD_SET_DRIVER/$1" ]
}

# May the board at PCI $2 be re-probed now? Sets card_set_heal_why and returns 1 when not.
card_set_reprobe_check() {  # board-id pci
  local pci=$2 node link target found=() holders st try
  card_set_heal_why=
  if ! card_set_present_pci "$pci"; then card_set_heal_why="PCI device $pci is absent"; return 1; fi
  if ! card_set_bound "$pci"; then card_set_heal_why="PCI device $pci is not bound to $CARD_SET_DRIVER"; return 1; fi
  for node in "$QUAL_TT_ROOT"/[0-9]*; do
    if qual_is_char "$node" && [ "$(qual_pci_of "$node")" = "$pci" ]; then found+=("$node"); fi
  done
  if [ "${#found[@]}" != 1 ]; then
    card_set_heal_why="${#found[@]} device nodes are at PCI $pci, not one"
    return 1
  fi
  for link in "$QUAL_BYID_ROOT"/*; do
    target=$(readlink -f -- "$link" 2>/dev/null || true)
    if [ -n "$target" ] && [ "$target" = "${found[0]}" ]; then
      card_set_heal_why="${found[0]}, at PCI $pci, is already the by-id target of ${link##*/}"
      return 1
    fi
  done
  for try in 1 2 3 4 5; do
    st=0
    holders=$(card_set_sudo fuser -v "${found[0]}" 2>&1) || st=$?
    if [ "$st" != 0 ] && [ -z "$holders" ]; then return 0; fi
    [ "$try" = 5 ] || sleep 2
  done
  card_set_heal_why="${found[0]} is held (or its holders could not be checked): ${holders//$'\n'/; }"
  return 1
}

card_set_heal_card() {  # board-id pci port [last]
  local card=$1 pci=$2 port=$3 last=${4:-} rescan=1 attempt=1 waited
  while ! card_set_present_pci "$pci"; do
    if [ "$rescan" -gt "$CARD_SET_RESCANS" ]; then
      echo "refusing: board $card PCI device $pci still absent after $CARD_SET_RESCANS rescans of its upstream port $port;" >&2
      echo "  its link did not train: a power-cycle of the PCIe switch or a host reboot is a human's call" >&2
      return 1
    fi
    if [ ! -e "$QUAL_SYS_ROOT/bus/pci/devices/$port" ]; then
      echo "refusing: board $card PCI device $pci absent, and its upstream port $port is not in sysfs; no rescan" >&2
      return 1
    fi
    if [ -n "$last" ] && [ "$SECONDS" -gt "$last" ]; then
      echo "refusing: board $card PCI device $pci absent, but no rescan may start after ${last} s (now ${SECONDS} s)" >&2
      return 1
    fi
    echo "[reset] board $card PCI device $pci absent after the reset; rescan of its upstream port $port $rescan/$CARD_SET_RESCANS"
    card_set_rescan_write "$port" || return 1
    waited=0
    while ! { card_set_present_pci "$pci" && card_set_bound "$pci"; } && [ "$waited" -lt "$CARD_SET_RESCAN_WAIT" ]; do
      sleep 2
      waited=$((waited + 2))
    done
    rescan=$((rescan + 1))
  done
  waited=0
  while ! card_set_linked "$card" && [ "$waited" -lt "$CARD_SET_REPROBE_WAIT" ]; do
    sleep 2
    waited=$((waited + 2))
  done
  while ! card_set_linked "$card"; do
    if [ "$attempt" -gt "$CARD_SET_REPROBES" ]; then
      echo "refusing: board $card by-id link still missing after $CARD_SET_REPROBES driver re-probes; not guessing" >&2
      return 1
    fi
    if ! card_set_reprobe_check "$card" "$pci"; then
      echo "refusing: board $card by-id missing after the reset, and $card_set_heal_why; no driver re-probe" >&2
      return 1
    fi
    if [ -n "$last" ] && [ "$SECONDS" -gt "$last" ]; then
      echo "refusing: board $card by-id missing, but no unbind may start after ${last} s (now ${SECONDS} s)" >&2
      return 1
    fi
    echo "[reset] board $card by-id missing after the reset (telemetry race); driver re-probe $attempt/$CARD_SET_REPROBES of $pci"
    if ! card_set_driver_write "$pci" unbind; then
      echo "refusing: unbinding $pci failed" >&2
      return 1
    fi
    sleep 3
    card_set_driver_write "$pci" bind || true
    if ! card_set_bound "$pci"; then
      sleep 3
      card_set_driver_write "$pci" bind || true
      if ! card_set_bound "$pci"; then
        echo "refusing: $pci is UNBOUND after re-probe $attempt; bind it: echo $pci | sudo -n tee /sys/bus/pci/drivers/$CARD_SET_DRIVER/bind" >&2
        return 1
      fi
    fi
    waited=0
    while ! card_set_linked "$card" && [ "$waited" -lt "$CARD_SET_REPROBE_WAIT" ]; do
      sleep 2
      waited=$((waited + 2))
    done
    attempt=$((attempt + 1))
  done
  echo "[reset] board $card by-id link present -> $(readlink -f -- "$QUAL_BYID_ROOT/$card")"
}

card_set_heal() {  # [seconds] [last]
  local limit=${1:-60} last=${2:-} waited=0 missing i
  case $limit in ''|*[!0-9]*) echo "refusing: card_set_heal wait '$limit' is not a number of seconds" >&2; return 1 ;; esac
  case $last in *[!0-9]*) echo "refusing: card_set_heal last '$last' is not a number of seconds" >&2; return 1 ;; esac
  if [ "${#CARD_SET_SNAPSHOT_IDS[@]}" = 0 ]; then
    echo 'refusing: no pre-reset snapshot (card_set_resolve before the reset)' >&2
    return 1
  fi
  while :; do
    missing=()
    for i in "${!CARD_SET_SNAPSHOT_IDS[@]}"; do
      card_set_linked "${CARD_SET_SNAPSHOT_IDS[$i]}" || missing+=("$i")
    done
    if [ "${#missing[@]}" = 0 ] || [ "$waited" -ge "$limit" ]; then
      break
    fi
    sleep 2
    waited=$((waited + 2))
  done
  if [ "${#missing[@]}" = 0 ]; then
    echo "[reset] all ${#CARD_SET_SNAPSHOT_IDS[@]} boards' by-id links present ${waited} s after the reset; no heal"
    return 0
  fi
  for i in "${missing[@]}"; do
    echo "[reset] board ${CARD_SET_SNAPSHOT_IDS[$i]} by-id link still missing ${waited} s after the reset"
    card_set_heal_card "${CARD_SET_SNAPSHOT_IDS[$i]}" "${CARD_SET_SNAPSHOT_PCI[$i]}" "${CARD_SET_SNAPSHOT_PORTS[$i]}" \
      "$last" || return 1
  done
  echo "[reset] all ${#CARD_SET_SNAPSHOT_IDS[@]} boards' by-id links present after the heal"
}
