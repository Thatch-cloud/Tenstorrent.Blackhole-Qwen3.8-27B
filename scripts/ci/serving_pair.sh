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
serving_pair_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$serving_pair_dir/qual_card.sh"
SERVING_PAIR_EXPECTED_PCI='0000:d1:00.0 0000:f3:00.0'

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
