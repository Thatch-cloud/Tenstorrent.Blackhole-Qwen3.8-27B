set -u
cat > "$HOME/gsm-k.sh" <<'SCRIPT_EOF'
#!/bin/bash
# GSM8K drift gate for the K stack on the endpoint: A + C + D (the gated ship stack) plus the
# in-place state write and K's two kernels (conv+gates, packed q/k/v), all mounted from the K
# graft. Same 60-item / 8-concurrent / 2048-token greedy run as gsm-c.sh; compare with 57/60.
#   gsm-k.sh <tag> <k 0|1>
set -u
TAG=$1; KON=$2
IMG=zot.thatch.local:5000/tt-vllm:qwen38-fused-decode
Q=/opt/tt-metal/models/demos/blackhole/qwen36
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
DRO=/opt/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py
DESC=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
TTCFG='{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}'
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
F3=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)
O=$HOME/opgraft-K
KM=""; KE=""
if [ "$KON" = "1" ]; then
  KM="-v $O/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro -v $O/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro"
  KM="$KM -v $O/gdn_decay:$OPS/gdn_decay:ro -v $O/decode_gated_delta_rule:$OPS/decode_gated_delta_rule:ro -v $O/gdn_conv_gates:$OPS/gdn_conv_gates:ro"
  KM="$KM -v $HOME/wrap-K/tp.py:$Q/tt/gdn/tp.py:ro -v $HOME/wrap-K/ttnn_delta_rule_ops.py:$DRO:ro"
  KE="-e QWEN_GDN_CONV_GATES=1 -e QWEN_GDN_PACKED_QKV=1 -e QWEN_GDN_FUSED_INPLACE=1"
fi
echo "### $TAG k=$KON load=[$(cut -d' ' -f1-3 /proc/loadavg)] start=$(date -Is)"
docker rm -f epserve >/dev/null 2>&1 || true
docker run -d --name epserve -p 8001:8000 -w /opt/vllm-tt-plugin \
  --device "$D1" --device "$F3" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/hf-cache:/root/.cache/huggingface" -v "$HOME/ttcache:/ttcache" $KM \
  -e QWEN_GDN_FUSED_DECODE=1 -e QWEN_SDPA_BF8=1 -e QWEN35_GDN_DECODE_BF16=1 -e QWEN35_GDN_STATE_BF16=1 $KE \
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
L="$HOME/gsmk-$TAG.log"; docker logs epserve > "$L" 2>&1
[ "$ready" = "0" ] && { echo "  NOT READY"; grep -oE "Error[^\"]{0,140}|TT_FATAL[^)]{0,90}" "$L" | sort -u | head -4; docker rm -f epserve >/dev/null; echo "=== DONE $TAG ==="; exit 0; }
echo "  ready after ${ready}s"
echo "  KFLAGS: $(grep -ohE 'QWEN_GDN_CONV_GATES engaged|QWEN_GDN_PACKED_QKV engaged|in-place happened=[A-Za-z]+' "$L" | sort -u | tr '\n' '|')"
python3 "$HOME/gsm_client.py" 2>&1 | sed 's/^/  /'
docker logs epserve > "$L" 2>&1
echo "  KFLAGS(end): $(grep -ohE 'QWEN_GDN_CONV_GATES engaged|QWEN_GDN_PACKED_QKV engaged|in-place happened=[A-Za-z]+' "$L" | sort -u | tr '\n' '|')"
docker rm -f epserve >/dev/null 2>&1
echo "=== DONE $TAG ==="
SCRIPT_EOF
chmod +x "$HOME/gsm-k.sh"; bash -n "$HOME/gsm-k.sh" && echo "gsm-k.sh syntax OK"
nohup bash -c 'while ! grep -q "KSHIP COMPLETE" ~/kship.out 2>/dev/null; do sleep 20; done; ~/gsm-k.sh gsm-k 1; echo "=== GSMK COMPLETE ==="' > "$HOME/gsmk.out" 2>&1 &
echo "GSM8K K-stack gate queued behind the endpoint A/B (pid=$!)"
