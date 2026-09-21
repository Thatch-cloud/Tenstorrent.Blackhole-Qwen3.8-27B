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
# Device: /dev/tenstorrent/2 as given in the task brief. Per tt-rig-hardware-topology memory,
# device numbers can renumber across a board reset; this script does no reset itself and runs
# right before the docker invocation, so the ordering is safe, but if the rig was reset since
# the device was last confirmed, re-check with `ls -la /dev/tenstorrent/by-id/` before trusting
# device 2 unchanged.
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

device=/dev/tenstorrent/2
test -e "$device"

script_src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sdpa_verify_prefill_style_bench.py"
test -f "$script_src"

name="sdpa-bench-$(date -u +%Y%m%d%H%M%S)-$$"
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
