#!/usr/bin/env bash
# Manual rig runner for the K5-A card-B probe (scripts/ci/gdn_seq_block_device_test.py): the served
# batched GDN launch (arm C) against K5-A (A), its bisection build (A0) and its negative control (N),
# byte for byte over regimes R1-R4, then trace-timed. NOT a CI workflow - invoke by hand on the rig;
# no allowlisting, no runner group (see hardware-ci-runner-allowlist memory). Cloned from
# gdn-user-batch-rig.sh.
#
# Needs no weights and no fixtures: every input is synthetic. --network none is set because nothing
# here should ever reach the network.
#
# Image: P5, pinned by id (sha256:0fd9ad1f..., the image of gate arms v206-v217). Its /opt/tt-metal
# holds the hash-pinned native GDN kernels the control and the K5-A compute prefix are generated from,
# and its /experiment-scripts/ci supplies gdn_multitoken.py. Only the NEW files and the two the plan
# names are bind-mounted, one file each, at /bench (never over /experiment-scripts/ci):
# gdn_seq_block.py and its three .cpp sources, the harness, gdn_user_batch.py and verify_trace_t1.py.
# The harness's own directory is first on sys.path, so these shadow the image's copies.
# GDN_SEQ_BLOCK_IMAGE overrides the image (loudly); the id is checked either way.
#
# QWEN_FAST_VERIFY_T1=1 is always set: the control is then the coalesced served build (one
# descriptor per role over rectangle ranges, verify_trace_t1 cut #12), exactly what the arms serve,
# and K5-A is built with the same geometry. TT_METAL_CACHE is a fresh per-run directory inside the
# container, so no stale JIT binary can stand in for a changed source (the kernels also carry
# SRC_TAG, a compile arg from each generated source's sha256).
#
# Device: card B and nothing else. ALLOW_SERVING_CARD is forced to 0 whatever the caller's
# environment holds, and after qual_card_resolve the run is refused unless the target is
# $QUAL_CARD_B and not a serving card (a leftover QUAL_CARD or ALLOW_SERVING_CARD=1 cannot move it
# to card M or card A). The board is resolved by board id right before the run - never a
# /dev/tenstorrent number, which renumbers across a board reset (tt-rig-hardware-topology memory).
# This script never resets anything; on a timeout (exit 124/137) or any exit but 0 (pass) and 1
# (a completed fail or partial pass) it prints qual_reset_hint's recovery lines. Run it only while
# the serving pair's runner group is idle (qual_card.sh prints the reminder).
#
# Usage:
#   scripts/ci/gdn-seq-block-rig.sh <host-output-dir> [extra gdn_seq_block_device_test.py args...]
#
# Examples:
#   scripts/ci/gdn-seq-block-rig.sh ~/gdn-seq-block
#   scripts/ci/gdn-seq-block-rig.sh ~/gdn-seq-block-p0 --skip-timing
#   scripts/ci/gdn-seq-block-rig.sh ~/gdn-seq-block-p1 --regimes R1 --seeds 17 --diag nosnap,passthrough
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <host-output-dir> [extra gdn_seq_block_device_test.py args...]" >&2
  exit 2
fi
outdir=$1
shift 1
extra_args=("$@")

P5=sha256:0fd9ad1f14a4e5d3d4464be55465cb6bb8d52e1b533c430d1c2f7f219df2b0e8
image=${GDN_SEQ_BLOCK_IMAGE:-$P5}
if [ "$image" != "$P5" ]; then
  echo "### WARNING: GDN_SEQ_BLOCK_IMAGE=$image is not image P5 ($P5); the plan's probe runs on P5" >&2
fi
image_id=$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null || true)
if [ -z "$image_id" ]; then
  echo "refusing: image $image is not present on this host" >&2
  exit 2
fi
echo "### image: $image -> $image_id"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mounts=()
for name in gdn_seq_block.py gdn_seq_block_compute.cpp gdn_seq_block_reader.cpp gdn_seq_block_writer.cpp \
    gdn_seq_block_device_test.py gdn_user_batch.py verify_trace_t1.py; do
  test -f "$here/$name"
  if grep -q $'\r' "$here/$name"; then
    echo "refusing: $name has CR line endings; its sha256 is its identity" >&2
    exit 2
  fi
  mounts+=(--mount "type=bind,src=$here/$name,dst=/bench/$name,readonly")
done
sha256sum "$here"/gdn_seq_block* "$here/gdn_user_batch.py" "$here/verify_trace_t1.py"

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)

. "$here/qual_card.sh"
ALLOW_SERVING_CARD=0   # never card M or card A, whatever the caller's environment says
qual_card_select
qual_card_resolve
if [ "$QUAL_CARD" != "$QUAL_CARD_B" ] || [ "$QUAL_SERVING" != 0 ]; then
  echo "refusing: this probe runs on card B ($QUAL_CARD_B) only, not $QUAL_CARD ($(qual_card_label))" >&2
  exit 2
fi
qual_refuse_holders
device=$QUAL_NODE

name="gdn-seq-block-$(date -u +%Y%m%d%H%M%S)-$$"
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

chmod 0777 "$outdir"
status=0
timeout -k 30 2700 docker run --rm --name "$name" --network none \
  --hostname gdn-seq-block --add-host gdn-seq-block:127.0.0.1 \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 2048 --memory 32g --cpus 8 --shm-size 4g \
  --device "$device" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  "${mounts[@]}" \
  --mount "type=bind,src=$outdir,dst=/results" \
  -e TT_METAL_HOME=/opt/tt-metal -e OMP_NUM_THREADS=8 \
  -e "TT_METAL_CACHE=/tmp/$name-kernel-cache" \
  -e QWEN_FAST_VERIFY_T1=1 \
  --workdir /opt/tt-metal \
  --entrypoint python3 "$image" -B /bench/gdn_seq_block_device_test.py \
  --out /results/gdn-seq-block.json "${extra_args[@]}" \
  > "$outdir/gdn-seq-block-console.log" 2>&1 || status=$?
docker run --rm --network none --mount "type=bind,src=$outdir,dst=/p" --entrypoint sh "$image" -c "chmod -R a+rwX /p" > /dev/null 2>&1 || true

echo "### exit $status (0 pass; 1 fail or partial pass; 2 error; 124/137 timeout)"
echo "console: $outdir/gdn-seq-block-console.log"
case "$status" in
  0|1) ;;
  124|137)
    echo "HANG SUSPECTED (exit $status: the 2700 s timeout fired): the container is removed on exit." >&2
    qual_reset_hint >&2 ;;
  *)
    echo "exit $status: if the console log shows a device timeout or a dispatch error, card B may be wedged:" >&2
    qual_reset_hint >&2 ;;
esac
if [ ! -s "$outdir/gdn-seq-block.json" ]; then
  echo "no report at $outdir/gdn-seq-block.json" >&2
  exit "$(( status == 0 ? 2 : status ))"
fi
echo "results: $outdir/gdn-seq-block.json"
python3 - "$outdir/gdn-seq-block.json" "$outdir/gdn-seq-block-console.log" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print('result:', report.get('result'), 'verify_t1:', report.get('verify_t1'), 'control_is_served:',
      report.get('control_is_served'))
print('p0:', json.dumps({key: value for key, value in (report.get('p0') or {}).items() if key != 'arms'}))
print('traced:', json.dumps({key: value for key, value in (report.get('traced') or {}).items() if key != 'arms'}))
print('p1:', json.dumps(report.get('p1')))
if report.get('timings'):
    print('per-launch us:', json.dumps((report['timings'] or {}).get('per_launch'), indent=2))
if report.get('error'):
    print('error:', report['error'])
summary = [line for line in open(sys.argv[2], errors='replace') if line.startswith('{"kind": "gdn-seq-block-probe"')]
print(summary[-1].rstrip() if summary else '{"kind": "gdn-seq-block-probe", "result": "no-summary-line"}')
PY
exit "$status"
