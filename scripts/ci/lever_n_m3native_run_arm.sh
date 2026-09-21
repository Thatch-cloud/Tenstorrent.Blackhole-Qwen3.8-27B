#!/usr/bin/env bash
# Serve the Lever N M3native gate: mount the native 64-row decode graft (model_config.py,
# attention/tp.py, gdn/tp.py, mlp.py) read-only over the pinned qwen36/tt sources and run
# the four-user packed gate against it.
#
# One arm only, unlike M1/M2's baseline/resumable split: native_m3 is a hasattr-gated
# overlay switch (model_batch.two_tile_bindings), not a scheduler edit with a stock
# counterpart worth serving separately. The same image with the graft mounted IS the
# thing under test; its own [PINDIAG] native_m3 marker and the retired binders' zero
# call counts (lever_n_m3native_gate.py) are the positive controls that the overlay
# actually engaged rather than silently falling back to the two-call path a stock
# image (no _64 attrs) would take unnoticed.
#
# $1 is the image sha, defaulting to the pinned qwen-fp2u-image.yml fast-serving image so
# the graft rides the exact image the four-user runs measure against.
#
# QWEN_FAST_FOUR_AS_TWO=0 is required: serving_runtime.py defaults four scheduler
# requests to TWO 32-row M1 blocks (QWEN_FAST_FOUR_AS_TWO, default ON at count 4), and
# under that default the single 64-row M3 block this gate exists to exercise is never
# built - two_tile_bindings(32) returns () and the [PINDIAG] native_m3 marker can never
# fire (gate 1, run 35556533480: a false negative - no decode round ever ran, even
# though the graft was mounted correctly). Setting it to 0 keeps the single M3 block.
set -euo pipefail
image="${1:-sha256:11ef6dd3a6dac3ae01ec6a01b64425a2150716b7e114f0bf023b0d28b1abae34}"
target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
cache=/home/thatch/.cache/qwen-experiments
revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
root=/opt/tt-metal/models/demos/blackhole/qwen36/tt

mkdir -p experiment-results draft-config
if [ ! -s draft-config/config.json ]; then
  curl -fsSL --max-time 30 \
    "https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/$revision/config.json" \
    > draft-config/config.json
fi

mounts=()
for component in attention convolution mlp projection selector; do
  mounts+=(--mount "type=bind,src=$cache/dflash2-$component-$revision,dst=/experiment-dflash-fixture/$component,readonly")
done
for layer in 1 2 3 4; do
  mounts+=(--mount "type=bind,src=$cache/dflash2-stack-$revision/layer-$layer,dst=/experiment-dflash-fixture/layer-$layer,readonly")
done
# The tracked references (scripts/ci/references/packed-gate) are what the runner has;
# a local runner-evidence.local copy, when present, is preferred (freshest local run).
refs="$PWD/scripts/ci/references/packed-gate"
if [ -d "$PWD/runner-evidence.local/packed-gate" ]; then refs="$PWD/runner-evidence.local/packed-gate"; fi
mounts+=(--mount "type=bind,src=$refs,dst=/bench/packed-gate-reference,readonly")
mapfile -t nodes < <(ls /dev/tenstorrent | grep -E '^[0-9]+$' | sort)
devices=()
for node in "${nodes[@]}"; do devices+=(--device "/dev/tenstorrent/$node"); done

# Optional K64 kernel graft (~/opgraft-K64: the batch-64 attn_decode_prep and
# nlp_concat_heads_decode C++ ops, both proved bit-exact on device). Unlike native_m3
# this patches no Python source, only the two ops and their bindings, so it is opt-in
# by env (KOPGRAFT64=<dir>) rather than hasattr-detected; default empty mounts nothing
# and passes no QWEN_FAST_NATIVE_ATTN, today's exact two-call behaviour
# (two_tile_decode.native_attn_enabled). Mirrors ~/kwork64/test-k64.sh on the rig.
#
# The grafted _ttnncpp.so also replaces the combined-runtime binary that
# dflash_combined_sim_runtime.BINARY_SHA256 pins, so its hash is passed through as
# QWEN_FAST_RUNTIME_BINARY_SHA256 for runtime_binary_override to admit at attach.
# Run 35558196471 mounted an earlier K64 graft binary and was refused at attach by
# that pin: it had been built over the ORIGINAL SDPA factory, not the COMBINED one
# the pin requires, so admitting it would have changed the qualified draft fp32 SDPA
# intermediates unnoticed. ~/opgraft-K64c on the rig is the same kernel graft rebuilt
# over the combined SDPA factory fd8c0676 on 2026-09-21 03:47 UTC, so it passes both
# runtime_binary_override checks (every pinned path matches, and the factory hash is
# the combined one) and is admitted for measurement without touching the pin itself.
KM=""
graft_binary_sha=""
if [ -n "${KOPGRAFT64:-}" ]; then
  KM="$KM -v $KOPGRAFT64/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro"
  KM="$KM -v $KOPGRAFT64/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro"
  KM="$KM -v $KOPGRAFT64/_ttnncpp.so:/opt/tt-metal/build_Release/lib/_ttnncpp.so:ro"
  KM="$KM -v $KOPGRAFT64/attn_prep:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/attn_prep:ro"
  KM="$KM -v $KOPGRAFT64/nlp_concat_heads_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode:ro"
  graft_binary_sha=$(sha256sum "$KOPGRAFT64/_ttnncpp.so" | cut -c1-64)
fi

name="qwen-m3native-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
timeout -k 30 2200 docker run --rm --name "$name" --network none \
  --hostname qwen-m3native --add-host qwen-m3native:127.0.0.1 \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 4096 --memory 96g --cpus 16 --shm-size 8g \
  "${devices[@]}" \
  --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$target,dst=/models/hub/models--Qwen--Qwen3.8-27B,readonly" \
  --mount "type=bind,src=$PWD/draft-config,dst=/draft-config,readonly" \
  --mount "type=bind,src=$PWD/graft/model_config.py,dst=$root/model_config.py,readonly" \
  --mount "type=bind,src=$PWD/graft/attention/tp.py,dst=$root/attention/tp.py,readonly" \
  --mount "type=bind,src=$PWD/graft/gdn/tp.py,dst=$root/gdn/tp.py,readonly" \
  --mount "type=bind,src=$PWD/graft/mlp.py,dst=$root/mlp.py,readonly" \
  --mount "type=bind,src=$PWD/scripts/ci/lever_n_m3native_gate.py,dst=/bench/lever_n_m3native_gate.py,readonly" \
  --mount "type=bind,src=$PWD/scripts/ci/longctx_cycle_bench.py,dst=/bench/longctx_cycle_bench.py,readonly" \
  --mount type=volume,src=qwen-experiments-f1e9b1a64b4f,dst=/experiment-cache \
  "${mounts[@]}" \
  $KM \
  ${KOPGRAFT64:+-e QWEN_FAST_NATIVE_ATTN=1} \
  ${KOPGRAFT64:+-e QWEN_FAST_RUNTIME_BINARY_SHA256=$graft_binary_sha} \
  -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 \
  -e QWEN_FAST_FOUR_AS_TWO=0 \
  -e QWEN_FABRIC_LINK_PROBE=1 -e QWEN_FROZEN_COMBINED_RUNTIME=1 -e QWEN_DSPARK_REQUEST_CONTEXT=32768 \
  -e QWEN_FAST_EAGER_PROPOSAL=1 -e QWEN_FAST_SHARD_CHECK=0 -e QWEN_FAST_PHASE_LOG=1 -e QWEN_FAST_CARRY_LOG=1 \
  -e QWEN_FAST_SHARED_CCL=1 -e QWEN_FAST_PACKED_STEP=1 -e QWEN_FAST_PACKED_AUDIT=1 -e QWEN_FAST_FAULTHANDLER=1 \
  -e TT_METAL_WATCHER=20 -e TT_METAL_WATCHER_APPEND=1 -e TT_METAL_WATCHER_DISABLE_ASSERT=1 \
  -e QWEN_GDN_DIRECT_WINDOW=1 -e QWEN_GDN_SHARED_QK_EXPERIMENT=1 \
  -e QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1 -e QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1 \
  -e QWEN_SDPA_BF8=1 -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 \
  -e QWEN_SKIP_UNUSED_SINGLETON_POSITIONS=1 -e QWEN_FAST_PHASE_TIMING=1 \
  -e QWEN_GDN_GROUPED_GATHER_ABBA=0 -e QWEN_GDN_GATE_EXP_ABBA=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e TT_METAL_HOME=/opt/tt-metal -e MESH_DEVICE=P300 -e OMP_NUM_THREADS=8 \
  -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels \
  -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
  --entrypoint python3 "$image" -B /bench/lever_n_m3native_gate.py \
  --users 4 --context 33024 --prompt-tokens 32768 --max-tokens 256 --stream-timeout 600 \
  --prompt-base 1000 --prompt-user-offset 1 --stagger 0 \
  --references /bench/packed-gate-reference \
  > experiment-results/m3native-gate-stdout.log 2>&1 || true

sed -n '/M3NATIVE_GATE_JSON_BEGIN/,/M3NATIVE_GATE_JSON_END/p' experiment-results/m3native-gate-stdout.log \
  | sed '1d;$d' > experiment-results/m3native-gate.json || true
sed -n '/M3NATIVE_GATE_LOG_BEGIN/,/M3NATIVE_GATE_LOG_END/p' experiment-results/m3native-gate-stdout.log \
  | sed '1d;$d' > experiment-results/m3native-server-tail.log || true
test -s experiment-results/m3native-gate.json
