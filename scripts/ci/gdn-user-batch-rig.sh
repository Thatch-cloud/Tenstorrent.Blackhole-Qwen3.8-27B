#!/usr/bin/env bash
# Manual rig runner for the batched-GDN equality and timing test
# (scripts/ci/gdn_user_batch_device_test.py). NOT a CI workflow - invoke by hand on the
# rig; no allowlisting, no runner group (see hardware-ci-runner-allowlist memory).
#
# Needs no weights and no fixtures: the test builds every input from synthetic random
# data on device, runs the four per-user recurrence/norm launches and the one batched
# launch, and compares every output byte. --network none is set because nothing here
# should ever reach the network.
#
# ONE card. The test opens a 1x1 mesh by default, which gdn_user_batch accepts for this
# comparison only (the serving mesh is 1x2). Pass --chips 2 in the extra arguments, and
# give the container a second --device, to add the real single-user gdn_multitoken.execute
# as a third arm - that is the launch the serving stack makes today.
#
# GDN_USER_BATCH_VERIFY_T1=1 runs both arms with QWEN_FAST_VERIFY_T1=1: the batched program is
# then built from rectangle core ranges with one descriptor per role over all 96 cores
# (verify_trace_t1 cut #12), and the byte compare against the per-user launches (and, with
# --chips 2, the native single-user launch) is that cut's device check. gdn_user_batch.py
# imports verify_trace_t1.py, so it is mounted beside it whether or not the flag is set - an
# image built before verify-trace T1 does not carry it.
#
# Device: /dev/tenstorrent/2 as given in the task brief. Per tt-rig-hardware-topology
# memory, device numbers can renumber across a board reset; this script does no reset
# itself, but if the rig was reset since the device was last confirmed, re-check with
# `ls -la /dev/tenstorrent/by-id/` before trusting device 2 unchanged.
#
# Usage:
#   scripts/ci/gdn-user-batch-rig.sh <host-output-dir> <image-sha> [extra device-test args...]
#
# Example:
#   scripts/ci/gdn-user-batch-rig.sh ~/gdn-user-batch sha256:c0dad9... --iterations 40
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <host-output-dir> <image-sha> [extra gdn_user_batch_device_test.py args...]" >&2
  exit 2
fi
outdir=$1
image=$2
shift 2
extra_args=("$@")

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)

# Device nodes RENUMBER across a board reset, and tt-smi indices do not track them: on
# 2026-09-21 after a reset, tt-smi index 1 (PCI 0000:f3:00) was /dev/tenstorrent/0 while
# tt-smi index 0 (PCI 0000:d1:00) was /dev/tenstorrent/2. Always map before trusting a
# number: `ls -la /dev/tenstorrent/by-id/` and
# `cat /sys/class/tenstorrent/tenstorrent!<n>/device/uevent | grep PCI_SLOT_NAME`
# against `tt-smi -ls`. Override with GDN_USER_BATCH_DEVICE when the mapping has moved.
device=${GDN_USER_BATCH_DEVICE:-/dev/tenstorrent/2}
test -e "$device"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mounts=()
for name in gdn_user_batch_device_test.py gdn_user_batch.py gdn_user_batch_conv.py gdn_multitoken.py verify_trace_t1.py; do
  test -f "$here/$name"
  mounts+=(--mount "type=bind,src=$here/$name,dst=/bench/$name,readonly")
done

name="gdn-user-batch-$(date -u +%Y%m%d%H%M%S)-$$"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

chmod 0777 "$outdir"
timeout -k 30 900 docker run --rm --name "$name" --network none \
  --hostname gdn-user-batch --add-host gdn-user-batch:127.0.0.1 \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 2048 --memory 32g --cpus 8 --shm-size 4g \
  --device "$device" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  "${mounts[@]}" \
  --mount "type=bind,src=$outdir,dst=/results" \
  -e TT_METAL_HOME=/opt/tt-metal -e OMP_NUM_THREADS=8 \
  -e TT_METAL_CACHE=/tmp/gdn-user-batch-kernel-cache \
  ${GDN_USER_BATCH_VERIFY_T1:+-e QWEN_FAST_VERIFY_T1=1} \
  --workdir /opt/tt-metal \
  --entrypoint python3 "$image" -B /bench/gdn_user_batch_device_test.py \
  --out /results/gdn-user-batch.json "${extra_args[@]}" \
  > "$outdir/gdn-user-batch-console.log" 2>&1 || true
docker run --rm --network none --mount "type=bind,src=$outdir,dst=/p" --entrypoint sh "$image" -c "chmod -R a+rwX /p" > /dev/null 2>&1 || true

test -s "$outdir/gdn-user-batch.json"
echo "results: $outdir/gdn-user-batch.json"
echo "console: $outdir/gdn-user-batch-console.log"
python3 - "$outdir/gdn-user-batch.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print('result:', report.get('result'), 'verify_t1:', report.get('verify_t1'))
print('bit_exact:', report.get('bit_exact'), 'initial_state_immutable:', report.get('initial_state_immutable'))
if report.get('timings'):
    print('timings:', json.dumps(report['timings'], indent=2))
if report.get('mismatches'):
    print('mismatches:', json.dumps(report['mismatches'][:4], indent=2))
PY
