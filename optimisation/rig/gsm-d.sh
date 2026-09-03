#!/bin/bash
# Lever D precision gate: the README's GSM8K run (60 items, 8 concurrent, max_tokens 2048,
# greedy, canonical test set) on the 4k serve config with lever A on, KV dtype as arg.
#   gsm-d.sh <tag> <sdpa_bf8 0|1>
set -u
TAG=$1; BF8=$2
IMG=zot.thatch.local:5000/tt-vllm:qwen38-fused-decode
DESC=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
TTCFG='{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}'
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
F3=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)
echo "### $TAG sdpa_bf8=$BF8 load=[$(cut -d' ' -f1-3 /proc/loadavg)] start=$(date -Is)"
docker rm -f epserve >/dev/null 2>&1 || true
docker run -d --name epserve -p 8001:8000 -w /opt/vllm-tt-plugin \
  --device "$D1" --device "$F3" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/hf-cache:/root/.cache/huggingface" -v "$HOME/ttcache:/ttcache" \
  -e QWEN_GDN_FUSED_DECODE=1 -e QWEN_SDPA_BF8="$BF8" \
  -e HF_MODEL=Qwen/Qwen3.8-27B -e MESH_DEVICE=P300 -e TT_MESH_GRAPH_DESC_PATH=$DESC \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/ttcache \
  -e VLLM_RPC_TIMEOUT=100000 -e VLLM_PLUGINS=tt,tt_model_registry -e QWEN36_BATCHED_DECODE_MODE=host \
  "$IMG" python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.8-27B --served-model-name qwen3.8-27b \
    --max_model_len 4096 --max-num-seqs 8 --no-enable-prefix-caching \
    --block-size 64 --reasoning-parser qwen3 --port 8000 --host 0.0.0.0 \
    --additional-config "$TTCFG" >/dev/null
ready=0
for i in $(seq 1 80); do curl -sf http://localhost:8001/v1/models >/dev/null 2>&1 && { ready=$((i*15)); break; }
  docker ps --filter name=epserve --format '{{.Names}}' | grep -q epserve || break; sleep 15; done
[ "$ready" = "0" ] && { echo "  NOT READY"; docker logs --tail 15 epserve 2>&1 | tail -5; docker rm -f epserve >/dev/null; echo "=== DONE $TAG ==="; exit 0; }
echo "  ready after ${ready}s"
python3 "$HOME/gsm_client.py" 2>&1 | sed 's/^/  /'
docker rm -f epserve >/dev/null 2>&1
echo "=== DONE $TAG ==="
