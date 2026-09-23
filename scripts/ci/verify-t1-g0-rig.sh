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
# Card M is half of the serving pair (tt-rig-hardware-topology memory): this refuses while any
# running container can reach a card, and does no reset. Device nodes renumber across a board
# reset, so the cards are resolved by by-id right before `docker run`.
#
# One card by default (--mesh 1x1: the two vocab shards run one after the other on card M).
# Pass `--mesh 1x2` to open the pair with the fabric, as the model does - the only run that
# also proves the chip-to-vocab-offset order the host combine assumes (chip_order_proven).
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

running=$(docker ps -q)
if [ -n "$running" ]; then
  docker inspect $running | python3 -c '
import json, sys
for container in json.load(sys.stdin):
    config = container["HostConfig"]
    mapped = [item.get("PathOnHost", "") for item in config.get("Devices") or []]
    mounts = [item.get("Source", "") for item in container.get("Mounts", [])]
    if config.get("Privileged") or any(path == "/dev" or path.startswith("/dev/tenstorrent") for path in mapped + mounts):
        raise SystemExit("Refusing G0: running container can reach a card: " + container["Name"])
'
fi

card_m=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
test -e "$card_m"
devices=(--device "$card_m")
if [ "$pair" = 1 ]; then
  card_a=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)
  test -e "$card_a"
  devices+=(--device "$card_a")
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for name in verify_t1_device_compare.py verify_trace_t1.py lever_n_m3native_patch.py gdn_prefill_conv_exact.py; do
  test -f "$here/$name"
done

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
chmod 0777 "$outdir"

name="verify-t1-g0-$(date -u +%Y%m%d%H%M%S)-$$"
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
