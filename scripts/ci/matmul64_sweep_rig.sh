#!/usr/bin/env bash
# Manual rig runner for the M=64 decode matmul grid sweep (scripts/ci/matmul64_sweep.py).
# NOT a CI workflow - invoke by hand on the rig; no allowlisting, no runner group.
#
# Needs no weights or fixtures: ONE device, hugepages, TT_METAL_HOME, and the sweep
# script itself bind-mounted at /bench/matmul64_sweep.py. The sweep still calls
# Qwen36ModelArgs(mesh_device=None) once (load_model_dims in matmul64_sweep.py) to read
# the model's real dim/head-count/hidden_dim numbers instead of guessing them, so the
# existing HF cache directory other rig runners (e.g. lever_n_m3native_run_arm.sh's
# `target` mount) already reference is bind-mounted read-only here too - a bind mount
# costs nothing to prepare (no copy, no new fixture) and --network none plus the
# offline env vars below mean it is the ONLY way that call can resolve without network
# access.
#
# Device: /dev/tenstorrent/2, card M (by-id blackhole-CEF5729692C19E6D, the wide-PCIe
# card - see tt-rig-hardware-topology memory). Resolved by by-id at run time, not
# hardcoded, since device numbers renumber across a board reset (docs/gotchas.md) - this
# script does no reset itself, so resolving right before `docker run` is the safe
# ordering that memory describes.
#
# Usage:
#   scripts/ci/matmul64_sweep_rig.sh <host-output-dir> <image-sha> [extra matmul64_sweep.py args...]
#
# Example:
#   scripts/ci/matmul64_sweep_rig.sh ~/matmul64-results sha256:c0dad9... --full --shapes mlp_w1,mlp_w2
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <host-output-dir> <image-sha> [extra matmul64_sweep.py args...]" >&2
  exit 2
fi
outdir=$1
image=$2
shift 2
extra_args=("$@")

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
# The main container below runs as root with --cap-drop ALL. Writing into a host
# directory it does not own fails EACCES unless the directory is already
# world-writable (run against image v65 hit exactly this: matmul64-sweep.json never
# got written, and the four already-completed shapes' results were lost with it -
# see the post-run fixup below and matmul64_sweep.py's own per-shape partial saves,
# which now save after every shape rather than only in one final `finally`).
chmod 0777 "$outdir"

device=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
test -e "$device"

hf_target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
script_src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/matmul64_sweep.py"
test -f "$script_src"

name="matmul64-sweep-$(date -u +%Y%m%d%H%M%S)-$$"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

timeout -k 30 900 docker run --rm --name "$name" --network none \
  --hostname matmul64-sweep --add-host matmul64-sweep:127.0.0.1 \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 2048 --memory 32g --cpus 8 --shm-size 4g \
  --device "$device" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$hf_target,dst=/models/hub/models--Qwen--Qwen3.8-27B,readonly" \
  --mount "type=bind,src=$script_src,dst=/bench/matmul64_sweep.py,readonly" \
  --mount "type=bind,src=$outdir,dst=/results" \
  -e HF_MODEL=/models/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TT_METAL_HOME=/opt/tt-metal -e OMP_NUM_THREADS=8 \
  -e TT_METAL_CACHE=/tmp/matmul64-kernel-cache \
  --workdir /opt/tt-metal \
  --entrypoint python3 "$image" -B /bench/matmul64_sweep.py \
  --out /results/matmul64-sweep.json --device-id 0 "${extra_args[@]}" \
  > "$outdir/matmul64-sweep-console.log" 2>&1 || true

# The main container's own writes into /results come out root-owned even with the
# chmod 0777 above (that widened the DIRECTORY, not files root subsequently creates in
# it), so the runner user could not read or delete them afterwards. A SEPARATE
# default-capability container (no --cap-drop) hands the tree back, mirroring
# lever_n_m3native_run_arm.sh's own tracy .logs fixup. Best-effort: a failure here must
# not hide the sweep's own exit state, which the `test -s` below still checks.
timeout -k 10 60 docker run --rm --network none \
  --mount "type=bind,src=$outdir,dst=/results" \
  --entrypoint sh "$image" -c 'chmod -R a+rwX /results' \
  > /dev/null 2>&1 || true

test -s "$outdir/matmul64-sweep.json"
echo "results: $outdir/matmul64-sweep.json"
echo "console: $outdir/matmul64-sweep-console.log"
