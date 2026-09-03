set -u
# End-to-end speculative generation on the traced serving path, with the fused GDN kernel
# grafted into the vLLM image.
#
# Same graft as tt-run-verify-traced-grafted.sh (the op lives in tt-gdn-decay:test, built from
# tt-serving; the traced Generator path only exists in tt-vllm; both carry tt-metal 9f9cd4fd and
# byte-identical libtt_metal.so, so mounting the rebuilt _ttnn.so plus the op's JIT kernel
# sources is sufficient). Adds two mounts the timing benchmark did not need:
#
#   model.py           -- the hidden-retention patch (K3), so the MTP head has something to draft
#                         from. Copied out of the image and patched here, so the mounted file is
#                         provably image + patch and nothing else.
#   weight_mapping.py  -- keeps the `mtp.` prefixed tensors, which the stock mapping drops.
IMG=zot.thatch.local:5000/tt-vllm:qwen38-fused-decode   # the plugin image is gone; this one has both ops
DESC=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
TDIR=/opt/tt-metal/models/demos/blackhole/qwen36/tests
QDIR=/opt/tt-metal/models/demos/blackhole/qwen36/tt
OPD=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/gdn_decay
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
F3=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)
mkdir -p ~/ttcache

# Refresh the patched model.py from the image every run, so it can never drift.
docker rm -f mxsd >/dev/null 2>&1 || true
docker create --name mxsd "$IMG" >/dev/null
docker cp mxsd:$QDIR/model.py ~/wrap/model.py >/dev/null
docker rm -f mxsd >/dev/null
python3 ~/wrap/patch_hid.py ~/wrap/model.py

# SD_GRAFT=0 drops the three graft mounts: the negative control for the whole measurement.
# With the stock library SD_IMPL=fused must die on
#   AttributeError: module 'ttnn._ttnn.operations.transformer' has no attribute
#   'gdn_recurrent_step'
# Which op build to graft. opgraft = fp32 inputs; opgraft-bf16 = bf16 q/k/v
# (10ak item 1), which REQUIRES MD_BF16_IN=1 -- the validator rejects fp32.
OPGRAFT=${OPGRAFT:-$HOME/opgraft-53587}   # gdn_decay (verify) + decode_gated_delta_rule (baseline A)
GRAFT=""
if [ "${SD_GRAFT:-1}" = "1" ]; then
  GRAFT="-v $OPGRAFT/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro"
  GRAFT="$GRAFT -v $OPGRAFT/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro"
  GRAFT="$GRAFT -v $OPGRAFT/gdn_decay:$OPD:ro"
  # #53587: the fused T=1 decode op plus its integration, so the BASELINE is fair
  if [ -d "$OPGRAFT/decode_gated_delta_rule" ]; then
    GRAFT="$GRAFT -v $OPGRAFT/decode_gated_delta_rule:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule:ro"
    GRAFT="$GRAFT -v $OPGRAFT/ttnn_delta_rule_ops.py:/opt/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py:ro"
  fi
fi
TAG=${TAG:-sd}
echo "START $TAG K=${SD_K:-2} impl=${SD_IMPL:-fused} ngen=${SD_NGEN:-64} \
reject=${SD_FORCE_REJECT:-0} safe=${SD_SAFE_STAGE:-1} graft=${SD_GRAFT:-1} $(date -Is)"
docker rm -f sdrun >/dev/null 2>&1 || true
timeout 5400 docker run --rm --name sdrun \
  --device "$D1" --device "$F3" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/hf-cache:/root/.cache/huggingface" -v "$HOME/wrap:/wrap" -v "$HOME/ttcache:/ttcache" \
  -v "$HOME/wrap/test_gdn_decode_multi.py:$TDIR/test_gdn_decode_multi.py" \
  -v "$HOME/wrap/model.py:$QDIR/model.py" \
  -v "$HOME/wrap/weight_mapping.py:$QDIR/weight_mapping.py" \
  $GRAFT \
  -e SD_K="${SD_K:-2}" -e SD_NGEN="${SD_NGEN:-64}" -e SD_IMPL="${SD_IMPL:-fused}" \
  -e SD_CHECK="${SD_CHECK:-1}" -e SD_FORCE_REJECT="${SD_FORCE_REJECT:-0}" \
  -e SD_ARMS="${SD_ARMS:-1}" -e SD_GATES="${SD_GATES:-3}" -e SD_SAFE_STAGE="${SD_SAFE_STAGE:-1}" -e SD_SPEC="${SD_SPEC:-1}" \
  -e SD_PROMPT="${SD_PROMPT:-}" \
  -e SD_DRAFT_PROFILE="${SD_DRAFT_PROFILE:-0}" \
  -e SD_DRAFT_FAST_READ="${SD_DRAFT_FAST_READ:-0}" \
  -e SD_DRAFT_RM_READ="${SD_DRAFT_RM_READ:-0}" \
  -e SD_TRACE_MTP="${SD_TRACE_MTP:-0}" -e SD_TRACE_MTP_REPS="${SD_TRACE_MTP_REPS:-30}" \
  -e SD_DRAFT_MODE="${SD_DRAFT_MODE:-mtp}" \
  -e SD_DRAFT_SHORTLIST="${SD_DRAFT_SHORTLIST:-0}" \
  -e SD_SHORTLIST_AUDIT="${SD_SHORTLIST_AUDIT:-0}" \
  -e MD_BF16_IN="${MD_BF16_IN:-0}" \
  -e MD_GQA_KERNEL="${MD_GQA_KERNEL:-0}" \
  -e SD_CHECK_TRACED="${SD_CHECK_TRACED:-0}" \
  -e SD_FAST_CARRY="${SD_FAST_CARRY:-0}" -e SPEC_FAST_CARRY="${SD_FAST_CARRY:-0}" \
  -e SD_DRAFT_DEV_ARGMAX="${SD_DRAFT_DEV_ARGMAX:-0}" \
  -e QWEN36_LOAD_MTP=1 \
  -e QWEN_GDN_FUSED_DECODE="${QWEN_GDN_FUSED_DECODE:-1}" \
  -e HF_MODEL=Qwen/Qwen3.8-27B -e MESH_DEVICE=P300 \
  -e TT_MESH_GRAPH_DESC_PATH=$DESC -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/ttcache \
  --workdir /opt/tt-metal --entrypoint python3 "$IMG" /wrap/spec_generate.py > ~/$TAG.log 2>&1
echo "rc=$?  $(date -Is)"
grep -E "^SD " ~/$TAG.log
echo "--- errors ---"
grep -E "TT_FATAL|RuntimeError|AssertionError|Error:|ModuleNotFound|TypeError|ImportError|Traceback" ~/$TAG.log | head -8 | cut -c1-260
echo "=== DONE $TAG ==="
