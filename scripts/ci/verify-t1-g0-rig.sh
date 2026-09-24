#!/usr/bin/env bash
# Manual rig runner for G0 of verify-trace T1 (scripts/ci/verify_t1_device_compare.py): the
# #11 matmul-config and #8a per-shard-argmax byte compares that must pass before any token
# gate. NOT a CI workflow - invoke by hand on the rig; no allowlisting, no runner group.
#
# Needs no weights and no fixtures. The harness imports lever_n_m3native_patch.py (never in
# the image) and verify_trace_t1.py (in the image only from the wave-2 build on), so this
# checkout's scripts/ci is mounted read-only at /checkout/scripts/ci and the harness runs from
# there: its own directory is first on sys.path, and the image's PYTHONPATH (the model tree,
# ttnn) follows. The explicit hardware allocation the harness demands is passed with -e, so it
# is set INSIDE the container, not only on the host.
#
# One card by default (--mesh 1x1: the two vocab shards run one after the other on it): the
# qualification card, QUAL_CARD (a board id; default card B, blackhole-F36F768B9A5CAFA0), via
# scripts/ci/qual_card.sh. Card M and card A are the serving pair (tt-rig-hardware-topology
# memory): QUAL_CARD may name one only with ALLOW_SERVING_CARD=1. Pass `--mesh 1x2` to open the
# pair with the fabric, as the model does - the only run that also proves the chip-to-vocab-offset
# order the host combine assumes (chip_order_proven). That IS the serving pair (card B has no
# Ethernet cable), so it needs ALLOW_SERVING_CARD=1 and ignores QUAL_CARD. Device nodes renumber
# across a board reset, so the cards are resolved by board id right before `docker run`; the run
# is refused while a container or a host process can reach a card it opens. It does no reset.
#
# Usage:
#   scripts/ci/verify-t1-g0-rig.sh <host-output-dir> <image-sha> [extra verify_t1_device_compare.py args...]
#
# Example:
#   scripts/ci/verify-t1-g0-rig.sh ~/verify-t1-g0 sha256:c0dad9... --part matmul --iterations 40
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <host-output-dir> <image-sha> [extra verify_t1_device_compare.py args...]" >&2
  exit 2
fi
outdir=$1
image=$2
shift 2
extra_args=("$@")

pair=0
previous=''
for arg in "${extra_args[@]}"; do
  if [ "$arg" = --mesh=1x2 ] || { [ "$previous" = --mesh ] && [ "$arg" = 1x2 ]; }; then
    pair=1
  fi
  previous=$arg
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/qual_card.sh"
devices=()
opened=()   # "<board id> <node>" of every card the container is given, for the recheck before docker run
if [ "$pair" = 1 ]; then
  # G0: refuse while anything can reach either card (serving targets: any container on any card).
  for card in $QUAL_SERVING_CARDS; do
    QUAL_CARD=$card
    qual_card_select
    qual_card_resolve
    qual_refuse_holders
    devices+=(--device "$QUAL_NODE")
    opened+=("$QUAL_CARD $QUAL_NODE")
  done
else
  qual_card_select
  qual_card_resolve
  qual_refuse_holders
  devices+=(--device "$QUAL_NODE")
  opened+=("$QUAL_CARD $QUAL_NODE")
fi

for name in verify_t1_device_compare.py verify_trace_t1.py lever_n_m3native_patch.py gdn_prefill_conv_exact.py qual_card.sh; do
  test -f "$here/$name"
done

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
chmod 0777 "$outdir"

name="verify-t1-g0-$(date -u +%Y%m%d%H%M%S)-$$"
for card in "${opened[@]}"; do
  qual_card_recheck "${card%% *}" "${card#* }"   # still on the node its holder check cleared
done
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

timeout -k 30 1200 docker run --rm --name "$name" --network none \
  --hostname verify-t1-g0 --add-host verify-t1-g0:127.0.0.1 \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 2048 --memory 32g --cpus 8 --shm-size 4g \
  "${devices[@]}" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$here,dst=/checkout/scripts/ci,readonly" \
  --mount "type=bind,src=$outdir,dst=/results" \
  -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 \
  -e TT_METAL_HOME=/opt/tt-metal -e OMP_NUM_THREADS=8 \
  -e TT_METAL_CACHE=/tmp/verify-t1-g0-kernel-cache \
  --workdir /opt/tt-metal \
  --entrypoint python3 "$image" -B /checkout/scripts/ci/verify_t1_device_compare.py \
  --output /results/verify-t1-g0.json "${extra_args[@]}" \
  > "$outdir/verify-t1-g0-console.log" 2>&1 || true

timeout -k 10 60 docker run --rm --network none \
  --mount "type=bind,src=$outdir,dst=/results" \
  --entrypoint sh "$image" -c 'chmod -R a+rwX /results' \
  > /dev/null 2>&1 || true

test -s "$outdir/verify-t1-g0.json"
echo "results: $outdir/verify-t1-g0.json"
echo "console: $outdir/verify-t1-g0-console.log"
python3 - "$outdir/verify-t1-g0.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print('passed:', report.get('passed'), 'mesh:', report.get('mesh'), 'chip_order_proven:', report.get('chip_order_proven'))
if report.get('error'):
    print('error:', report['error'])
for part in ('matmul', 'argmax'):
    for entry in report.get(part) or []:
        if not entry.get('exact'):
            print('NOT EXACT', part, json.dumps(entry))
PY
