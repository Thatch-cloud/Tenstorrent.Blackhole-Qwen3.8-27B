#!/bin/bash
# Row 3d take 2: shard-greedy on the endpoint through a greedy-only sampler object.
#   run3e.sh <tag> <shard 0|1>     mounts only wrap-3e/model.py; A on in both arms.
set -u
TAG=$1; SHARD=$2
IMG=zot.thatch.local:5000/tt-vllm:qwen38-fused-decode
Q=/opt/tt-metal/models/demos/blackhole/qwen36
DESC=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
F3=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)
if [ "$SHARD" = "1" ]; then
  TTCFG='{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824, "sample_on_device_mode": "decode_only"}}'
else
  TTCFG='{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}'
fi
L="$HOME/r3e-$TAG.log"
echo "### $TAG shard=$SHARD load=[$(cut -d' ' -f1-3 /proc/loadavg)] start=$(date -Is)"
docker rm -f epserve >/dev/null 2>&1 || true
docker run -d --name epserve -p 8001:8000 -w /opt/vllm-tt-plugin \
  --device "$D1" --device "$F3" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/ttcache:/ttcache" -e TT_METAL_CACHE=/ttcache \
  -v "$HOME/wrap-3e/model.py:$Q/tt/model.py:ro" \
  -e QWEN_GDN_FUSED_DECODE=1 -e QWEN36_SHARD_GREEDY="$SHARD" \
  -e HF_MODEL=Qwen/Qwen3.8-27B -e MESH_DEVICE=P300 \
  -e TT_MESH_GRAPH_DESC_PATH=$DESC \
  -e VLLM_RPC_TIMEOUT=100000 -e VLLM_PLUGINS=tt,tt_model_registry \
  "$IMG" python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.8-27B --served-model-name qwen3.8-27b \
    --max_model_len 4096 --max-num-seqs 8 --no-enable-prefix-caching \
    --block-size 64 --reasoning-parser qwen3 --port 8000 --host 0.0.0.0 \
    --additional-config "$TTCFG" >/dev/null
ready=0
for i in $(seq 1 60); do
  curl -sf http://localhost:8001/v1/models >/dev/null 2>&1 && { ready=$((i*15)); break; }
  docker ps --filter name=epserve --format '{{.Names}}' | grep -q epserve || break
  sleep 15
done
docker logs epserve > "$L" 2>&1
if [ "$ready" = "0" ]; then
  echo "  NOT READY -- failure:"
  grep -oE "Error[^\"]{0,140}|Traceback|assert[^\"]{0,100}|RuntimeError[^\"]{0,120}|TT_FATAL[^)]{0,90}|AttributeError[^\"]{0,120}" "$L" | sort -u | head -8
  docker rm -f epserve >/dev/null 2>&1; echo "=== DONE $TAG ==="; exit 0
fi
echo "  ready after ${ready}s"
curl -s http://localhost:8001/v1/completions -H "Content-Type: application/json" -d '{
  "model":"qwen3.8-27b","prompt":"The capital of New Zealand is","max_tokens":24,"temperature":0}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('  TEXT:', repr(d['choices'][0]['text']) if 'choices' in d else d)" 2>&1 | head -2
docker logs epserve > "$L" 2>&1
echo "  HOOK: $(grep -ohE 'QWEN36_SHARD_GREEDY[^\"]{0,90}' "$L" | sort -u | head -3 | tr '\n' '|')"
grep -oE "Traceback|Error[^\"]{0,120}" "$L" | sort -u | head -4 | sed 's/^/  ERR: /'
python3 "$HOME/bench_itl.py" 2>&1 | grep -E "^BENCH_(MAIN|TEXT_MAIN)"
docker logs epserve > "$L" 2>&1
docker rm -f epserve >/dev/null 2>&1
echo "=== DONE $TAG ==="
