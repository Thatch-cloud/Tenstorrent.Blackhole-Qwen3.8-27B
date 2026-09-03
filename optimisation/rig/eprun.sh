#!/bin/bash
# Endpoint A/B for lever A, ITL-based. tt-vllm:qwen38-fused-decode already contains the
# fused op and the patched module, so the arms differ only by the env var.
#   eprun.sh <tag> <flag 0|1>
set -u
TAG=$1; FLAG=$2
IMG=zot.thatch.local:5000/tt-vllm:qwen38-fused-decode
DESC=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
TTCFG='{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}'
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
F3=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)
L="$HOME/ep-$TAG.log"
echo "### $TAG flag=$FLAG load=[$(cut -d' ' -f1-3 /proc/loadavg)] start=$(date -Is)"
docker rm -f epserve >/dev/null 2>&1 || true
docker run -d --name epserve -p 8001:8000 -w /opt/vllm-tt-plugin \
  --device "$D1" --device "$F3" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/ttcache:/ttcache" \
  -e QWEN_GDN_FUSED_DECODE="$FLAG" \
  -e HF_MODEL=Qwen/Qwen3.8-27B -e MESH_DEVICE=P300 \
  -e TT_METAL_CACHE=/ttcache \
  -e TT_MESH_GRAPH_DESC_PATH=$DESC \
  -e VLLM_RPC_TIMEOUT=100000 -e VLLM_PLUGINS=tt,tt_model_registry \
  -e QWEN36_BATCHED_DECODE_MODE=host \
  "$IMG" python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.8-27B --served-model-name qwen3.8-27b \
    --max_model_len 4096 --max-num-seqs 8 --no-enable-prefix-caching \
    --block-size 64 --reasoning-parser qwen3 --port 8000 --host 0.0.0.0 \
    --additional-config "$TTCFG" >/dev/null
ready=0
for i in $(seq 1 80); do
  curl -sf http://localhost:8001/v1/models >/dev/null 2>&1 && { ready=$((i*15)); break; }
  docker ps --filter name=epserve --format '{{.Names}}' | grep -q epserve || break
  sleep 15
done
if [ "$ready" = "0" ]; then
  echo "  NOT READY"; docker logs --tail 20 epserve > "$L" 2>&1
  docker rm -f epserve >/dev/null 2>&1; echo "=== DONE $TAG ==="; exit 0
fi
echo "  ready after ${ready}s"
python3 "$HOME/bench_itl.py" 2>&1 | grep -E "^BENCH_"
docker logs epserve > "$L" 2>&1
echo "  ENGAGED : $(grep -ohE 'QWEN_GDN_FUSED_DECODE engaged[^)]*\)' "$L" | head -1)"
docker rm -f epserve >/dev/null 2>&1
echo "=== DONE $TAG ==="
