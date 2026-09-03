#!/bin/bash
# Per-module device profile. These tests drive one layer eagerly, so the op count is
# small enough for the 12k-marker device buffer that the whole-model run overflows.
# Kernel durations are valid in eager mode (only the op-to-op gaps are host-inflated),
# which is exactly the half we need: the traced gap comes out as 54.21ms - sum(kernel).
#   modprof.sh <tag> <test-file> <node-id>
set -u
TAG=$1; FILE=$2; NODE=$3
IMG=zot.thatch.local:5000/tt-serving:v0.77.0-rc1-prstack
Q=/opt/tt-metal/models/demos/blackhole/qwen36
DESC=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
F3=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)

echo "### $TAG  $NODE  load=[$(cut -d' ' -f1-3 /proc/loadavg)]  start=$(date -Is)"
docker rm -f "$TAG" >/dev/null 2>&1 || true
timeout 3600 docker run --name "$TAG" \
  --device "$D1" --device "$F3" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/ttcache:/ttcache" \
  -e HF_MODEL=Qwen/Qwen3.8-27B -e MESH_DEVICE=P300 \
  -e TT_METAL_CACHE=/ttcache \
  -e TT_MESH_GRAPH_DESC_PATH="$DESC" -e TT_METAL_HOME=/opt/tt-metal \
  -e TT_METAL_DEVICE_PROFILER=1 \
  --workdir /opt/tt-metal --entrypoint python3 "$IMG" \
  -m tracy -r -m pytest "$Q/tests/$FILE::$NODE" --timeout=3000 \
  > "$HOME/$TAG.log" 2>&1
echo "  rc=$?  end=$(date -Is)"
echo "  result   : $(grep -oE '[0-9]+ (passed|failed)' "$HOME/$TAG.log" | tail -1)"
echo "  overflow : $(grep -c 'markers were dropped' "$HOME/$TAG.log") warnings"

mkdir -p "$HOME/$TAG-out"
for p in /opt/tt-metal/generated/profiler/reports /opt/tt-metal/generated/profiler/.logs; do
  docker cp "$TAG:$p/." "$HOME/$TAG-out/" 2>/dev/null && echo "  copied $(basename $p)"
done
docker rm -f "$TAG" >/dev/null 2>&1 || true
rm -f "$HOME/$TAG-out/tracy_profile_log_host.tracy" "$HOME/$TAG-out/profile_log_device.csv"
echo "=== artefacts ==="
find "$HOME/$TAG-out" -name "*.csv" -printf "%10s  %p\n" 2>/dev/null | sort -rn | head -5
echo "=== DONE $TAG ==="
