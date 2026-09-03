#!/bin/bash
# Lever A at B=8. Same graft/flag structure as arun.sh; adds the two env vars
# batched2.sh established are required for short-context B=8
# (QWEN36_BATCHED_DECODE_MODE=host, QWEN_BATCHED_GROUPED=0 -- without the latter,
# short prompts hit the BH<=ncores batched-GDN prefill limit). Batched runs emit a
# different perf line: per-user-decode + aggregate.
#   arun8.sh <tag> <graft 0|1|2> <flag 0|1>
set -u
TAG=$1; GRAFT=$2; FLAG=$3; MODE=${4:-host}
IMG=zot.thatch.local:5000/tt-serving:v0.77.0-rc1-prstack
Q=/opt/tt-metal/models/demos/blackhole/qwen36
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
DRO=/opt/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py
DESC=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
NODE="demo/text_demo.py::test_demo_text[blackhole-batched_128_b8-device_params0-mesh_device0]"
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
F3=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)
G=""; O=$HOME/opgraft-53587
[ "$GRAFT" != "0" ] && WRAPSRC=$O/ttnn_delta_rule_ops.py
  [ "${WRAP3C:-0}" = "1" ] && WRAPSRC=$HOME/wrap-3c/ttnn_delta_rule_ops.py   # 3c: patched wrapper instead of the graft one
  G="-v $WRAPSRC:$DRO:ro"
if [ "$GRAFT" = "1" ]; then
  G="$G -v $O/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro"
  G="$G -v $O/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro"
  G="$G -v $O/gdn_decay:$OPS/gdn_decay:ro"
  G="$G -v $O/decode_gated_delta_rule:$OPS/decode_gated_delta_rule:ro"
fi
if [ "${WRAPL:-0}" = "1" ]; then
  G="$G -v $HOME/wrap-3c/model_config.py:$Q/tt/model_config.py:ro"
fi
if [ "${WRAP3C:-0}" = "1" ]; then
  # docker rejects duplicate mount targets, so the wrapper is chosen above; only tp.py is added here
  G="$G -v $HOME/wrap-3c/tp.py:$Q/tt/gdn/tp.py:ro"
fi
L="$HOME/a8-$TAG.log"
echo "### $TAG graft=$GRAFT flag=$FLAG mode=$MODE load=[$(cut -d' ' -f1-3 /proc/loadavg)] start=$(date -Is)"
docker rm -f "a8-$TAG" >/dev/null 2>&1 || true
timeout 5400 docker run --rm --name "a8-$TAG" \
  --device "$D1" --device "$F3" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/hf-cache:/root/.cache/huggingface" -v "$HOME/ttcache:/ttcache" \
  \
  -v "$HOME/wrap-shard/model.py:$Q/tt/model.py:ro" \
  $G \
  -e QWEN_GDN_FUSED_DECODE="$FLAG" -e QWEN_GDN_FUSED_INPLACE="${INPLACE:-0}" -e QWEN_PROJ_1D="${PROJ1D:-1}" -e QWEN35_GDN_DECODE_BF16="${GDNBF16:-0}" -e QWEN35_GDN_STATE_BF16="${GDNSTATE:-0}" \
  -e QWEN36_BATCHED_DECODE_MODE="$MODE" -e QWEN_BATCHED_GROUPED=0 \
  -e HF_MODEL=Qwen/Qwen3.8-27B -e MESH_DEVICE=P300 -e QWEN_SDPA_BF8=0 \
  \
  -e QWEN36_DEBUG_DECODE_TIMING=1 \
  -e TT_MESH_GRAPH_DESC_PATH="$DESC" -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/ttcache \
  "$IMG" pytest "$Q/$NODE" -v -s --timeout=5000 > "$L" 2>&1
echo "  rc=$?  end=$(date -Is)"
echo "  PROJ1D   : $(grep -ohE '\[PROJ_1D\][^\"]{0,60}' "$L" | head -1)"
echo "  INPLACE  : $(grep -ohE 'in-place happened=[A-Za-z]+' "$L" | head -1)"
echo "  result   : $(grep -oE '[0-9]+ (passed|failed)' "$L" | tail -1)"
echo "  ENGAGED  : $(grep -ohE 'engaged: ttnn[^ ]*[^)]*\)' "$L" | sort -u | head -1)"
echo "  perf     : $(grep -oE 'B=[0-9]+\] ttft=[0-9.]+s per-user-decode=[0-9.]+ tok/s aggregate=[0-9.]+ tok/s' "$L" | tail -1)"
echo "  phases   : $(grep -oE '\[DEBUG_DECODE_TIMING\] [a-z_]+: avg=[0-9.]+ms' "$L" | tr '\n' ' ')"
echo "  error    : $(grep -oE 'TT_FATAL[^)]{0,70}|Out of Memory|per_core_M[^ ]*|bank_manager[^ ]*' "$L" | head -1)"
echo "  shardhook: $(grep -ohE 'QWEN36_SHARD_NO_SAMPLER engaged[^\"]{0,70}' "$L" | head -1)"
echo "  gen[0]   : $(grep -oE 'user 0[^|]{0,90}' "$L" | tail -1)"
echo "  toks     : $(grep -oE '\[GEN\][^\"]{0,80}' "$L" | tail -1)"
echo "=== DONE $TAG ==="
