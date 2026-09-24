#!/usr/bin/env bash
# Manual rig runner for the prefill-style-vs-decode-fake-head SDPA comparison
# (scripts/ci/sdpa_verify_prefill_style_bench.py). NOT a CI workflow - invoke by hand on the
# rig; no allowlisting, no runner group (see hardware-ci-runner-allowlist memory).
#
# Needs no weights or fixtures beyond the bench script itself: ONE device and hugepages.
# The bench builds every Q/K/V/mask/page-table tensor from synthetic random data on device
# (see the script's docstring), so unlike matmul64_sweep_rig.sh there is no HF cache mount and
# no offline-env requirement - --network none is still set because nothing here should ever
# need the network, not because anything would otherwise reach for it.
#
# Device: the qualification card, QUAL_CARD (a board id; default card B,
# blackhole-F36F768B9A5CAFA0; card M or card A, the serving pair, only with
# ALLOW_SERVING_CARD=1), via scripts/ci/qual_card.sh. It used to be /dev/tenstorrent/2, a node
# number - card M after the 2026-09-23 renumbering, half of the serving pair. Device numbers
# renumber across a board reset and a switch power-cycle, so the card is resolved by board id
# at run time and checked again right before `docker run`; the run is refused while a
# container or a host process can reach that card. This script does no reset itself.
#
# Usage:
#   scripts/ci/sdpa_bench_rig.sh <host-output-dir> <image-sha> [extra bench.py args...]
#
# Example:
#   scripts/ci/sdpa_bench_rig.sh ~/sdpa-bench-results sha256:c0dad9... --users 1,4 --iters 20
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <host-output-dir> <image-sha> [extra sdpa_verify_prefill_style_bench.py args...]" >&2
  exit 2
fi
outdir=$1
image=$2
shift 2
extra_args=("$@")

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/qual_card.sh"
qual_card_select
qual_card_resolve
qual_refuse_holders
device=$QUAL_NODE

script_src="$here/sdpa_verify_prefill_style_bench.py"
test -f "$script_src"

name="sdpa-bench-$(date -u +%Y%m%d%H%M%S)-$$"
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

chmod 0777 "$outdir"
timeout -k 30 900 docker run --rm --name "$name" --network none \
  --hostname sdpa-bench --add-host sdpa-bench:127.0.0.1 \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 2048 --memory 32g --cpus 8 --shm-size 4g \
  --device "$device" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$script_src,dst=/bench/sdpa_verify_prefill_style_bench.py,readonly" \
  --mount "type=bind,src=$outdir,dst=/results" \
  -e TT_METAL_HOME=/opt/tt-metal -e OMP_NUM_THREADS=8 \
  -e TT_METAL_CACHE=/tmp/sdpa-bench-kernel-cache \
  --workdir /opt/tt-metal \
  --entrypoint python3 "$image" -B /bench/sdpa_verify_prefill_style_bench.py \
  --out /results/sdpa-verify-prefill-style-bench.json --device-id 0 "${extra_args[@]}" \
  > "$outdir/sdpa-bench-console.log" 2>&1 || true
docker run --rm --network none --mount "type=bind,src=$outdir,dst=/p" --entrypoint sh "$image" -c "chmod -R a+rwX /p" > /dev/null 2>&1 || true

test -s "$outdir/sdpa-verify-prefill-style-bench.json"
echo "results: $outdir/sdpa-verify-prefill-style-bench.json"
echo "console: $outdir/sdpa-bench-console.log"
