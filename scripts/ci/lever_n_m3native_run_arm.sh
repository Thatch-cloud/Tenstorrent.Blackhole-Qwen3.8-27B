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
image="${1:-sha256:8310fdfe6257a546a47822372deb7646c4380874815141f2f2f6c76f5e21fe80}"
target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
cache=/home/thatch/.cache/qwen-experiments
revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
root=/opt/tt-metal/models/demos/blackhole/qwen36/tt

# Arm geometry: M3NATIVE_USERS/CONTEXT/PROMPT_TOKENS default to today's four-user 32768
# recipe, so an unset environment reproduces the exact prior invocation. A wider arm (e.g.
# the 131k one-user attach probe) overrides these three; QWEN_DSPARK_REQUEST_CONTEXT tracks
# prompt_tokens below because that is what the frozen recipe's request-context selector
# reads. Setting it past 32768 does NOT admit a wider T16 fast path today - the staged
# combined-runtime recipe is CLI-locked to context 32768 (frozen_recipe_context.py:196-198)
# and every downstream literal it emits is a hard 32768, not derived from this value; see
# docs/lever-n-131k-attach-arm.md for the exact refusal chain. This arm exists to measure
# attach and report, not to reach a decode round.
users="${M3NATIVE_USERS:-4}"
context="${M3NATIVE_CONTEXT:-33024}"
prompt_tokens="${M3NATIVE_PROMPT_TOKENS:-32768}"
dspark_context="$prompt_tokens"
# QWEN_FAST_MAX_POSITION raises dflash_prefill_window's absolute prefill-position ceiling
# (default 65504) so a wider-than-default context can reach prefill at all; only set when
# non-default, so the default arm's docker invocation stays byte-identical to before.
max_position=""
if [ "$context" != "33024" ]; then
  max_position="$context"
fi
# M3NATIVE_ALLOW_MISSING_REFERENCES=1 threads --allow-missing-references to the gate for
# arms with no single-stream reference yet (e.g. 131k): comparisons still record
# reference_present=false and the dram/packed_phase lines still print, but gate_passed no
# longer requires every user to have a reference. Unset by default (empty, no flag added).
allow_missing_references=""
if [ "${M3NATIVE_ALLOW_MISSING_REFERENCES:-}" = "1" ]; then
  allow_missing_references="--allow-missing-references"
fi

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

# Proposals: the fp2u lane runs each request's draft proposal EAGERLY
# (QWEN_FAST_EAGER_PROPOSAL=1) because per-request proposal traces clobbered each
# other on one mesh (run 35477522469, serving_request_factory.prepare_proposal). At
# four users that is four sequential ~213 ms eager proposals per round (run
# 35559199392), more than half the round. M3NATIVE_TRACED_PROPOSAL=1 replaces that env
# with an inert marker so each request captures its own proposal trace again; the
# gate's byte-exact check against the single-stream references catches a clobbered trace.
if [ "${M3NATIVE_TRACED_PROPOSAL:-}" = "1" ]; then
  M3NATIVE_TRACED_PROPOSAL="-e M3NATIVE_TRACED_PROPOSAL=1"
else
  M3NATIVE_TRACED_PROPOSAL=""
fi

# Optional device-op profiling (M3NATIVE_PROFILE=1), off by default and byte-identical to
# today's invocation when unset. Mirrors dflash-request-profile.sh's own tracy wrapper
# exactly (the same five profiler env vars, the same `python3 -m tracy -p
# --check-exit-code --disable-device-data-dump-to-files --disable-device-data-push-to-tracy
# --dump-device-data-mid-run --op-support-count 20000 -o <dir>` invocation), so the packed
# round this gate serves is attributed with the same method already qualified for the T8
# verifier profile (docs/current-verifier-profile-2026-09-09.md), not a new one. The
# mid-run dumps are large, so profiling caps the run at --max-tokens 48 (three or four
# packed rounds after the four prefills) instead of the usual 256; every other arg, mount
# and env var stays the same. The existing 2200 s outer timeout is kept as-is: profiling
# fewer rounds is less work than the unprofiled 256-token run it replaces, and the
# workflow's own 40-minute step / 45-minute job ceilings leave no room to grow it to
# dflash-request-profile.sh's 4200 s (that budget covers a much larger, unrelated run).
#
# Profile mode also sets QWEN_FAST_PROFILED_BLOCK_STREAM=1 so serving_runtime's
# profiled_block_stream_override admits TT_METAL_DEVICE_PROFILER into the mandatory
# block-stream recipe for this attribution measurement only (run 35561945903 refused it).
# M3NATIVE_MAX_TOKENS overrides the 256-token run so users finish at DIFFERENT rounds (the
# early-finish transition, runs 35564623068 / 35567165791); the gate compares a shorter
# stream as a prefix of its reference.
max_tokens="${M3NATIVE_MAX_TOKENS:-256}"
trace_region_bytes=1073741824
entry_args=(-B /bench/lever_n_m3native_gate.py)
if [ "${M3NATIVE_PROFILE:-}" = "1" ]; then
  # The container runs as root with every capability dropped (no CAP_DAC_OVERRIDE), so
  # it cannot create tracy's .logs inside a host directory owned by the runner user
  # (run 35561589158: 'rm -rf /experiment-results-profile/.logs; mkdir -p ...' exit 1).
  mkdir -p experiment-results/profile
  chmod 0777 experiment-results/profile
  mounts+=(--mount "type=bind,src=$PWD/experiment-results/profile,dst=/experiment-results-profile")
  max_tokens=48
  trace_region_bytes=268435456
  entry_args=(-m tracy -p --check-exit-code --disable-device-data-dump-to-files
              --disable-device-data-push-to-tracy --dump-device-data-mid-run
              --op-support-count 20000 -o /experiment-results-profile
              /bench/lever_n_m3native_gate.py)
fi
# Traced proposals cost DRAM: each request uploads its own proposal-trace inputs (the
# (1,1,context+32,5120) history and per-layer cached K/V), 0.80 GB per engine against
# 0.58 GB eager, and the fourth user's admission ran out of memory with 716 MB largest
# free (run 35565478581). The traced arm halves the recipe's 1 GiB trace region for the
# headroom; decode traces are command streams and this repo's probes run at 256 MiB.
if [ -n "${M3NATIVE_TRACED_PROPOSAL:-}" ]; then
  trace_region_bytes=536870912
fi
# The tt-metal watcher (TT_METAL_WATCHER=20, inherited from the fp2u lane's hang diagnosis)
# compiles NoC sanitisation and waypoints into every kernel. The same folded SDPA decode
# call costs 0.46 ms on card M without it and 2.05 ms in the gate's device profile with
# it (run 35567165791 vs sdpa_verify_prefill_style_bench.py), so every packed-round
# trace_ms measured so far (1453, 1118, 1100 ms) carries watcher overhead the
# single-stream baselines never had. Off by default; M3NATIVE_WATCHER=1 re-enables it.
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
  ${M3NATIVE_PIPELINED_COMMITS:+-e QWEN_FAST_PIPELINED_COMMITS=1} \
  ${M3NATIVE_PIPELINED_PROPOSALS:+-e QWEN_FAST_PIPELINED_PROPOSALS=1} \
  ${M3NATIVE_FAST_COMMIT:+-e QWEN_FAST_FAST_COMMIT=1} \
  ${M3NATIVE_GDN_USER_BATCH:+-e QWEN_FAST_GDN_USER_BATCH=1} \
  ${M3NATIVE_PROFILE:+-e TTNN_OP_PROFILER=1} \
  ${M3NATIVE_PROFILE:+-e TT_METAL_DEVICE_PROFILER=1} \
  ${M3NATIVE_PROFILE:+-e TT_METAL_PROFILER_TRACE_TRACKING=1} \
  ${M3NATIVE_PROFILE:+-e TT_METAL_PROFILER_CPP_POST_PROCESS=1} \
  ${M3NATIVE_PROFILE:+-e TT_METAL_PROFILER_MID_RUN_DUMP=1} \
  ${M3NATIVE_PROFILE:+-e QWEN_FAST_PROFILE_DUMP_ROUND=4} \
  ${M3NATIVE_PROFILE:+-e QWEN_FAST_PROFILED_BLOCK_STREAM=1} \
  -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 \
  -e QWEN_FAST_FOUR_AS_TWO=0 \
  -e QWEN_FABRIC_LINK_PROBE=1 -e QWEN_FROZEN_COMBINED_RUNTIME=1 -e QWEN_DSPARK_REQUEST_CONTEXT=$dspark_context \
  ${max_position:+-e QWEN_FAST_MAX_POSITION=$max_position} \
  ${M3NATIVE_TRACED_PROPOSAL:--e QWEN_FAST_EAGER_PROPOSAL=1} -e QWEN_FAST_SHARD_CHECK=0 -e QWEN_FAST_PHASE_LOG=1 -e QWEN_FAST_CARRY_LOG=1 \
  -e QWEN_FAST_SHARED_CCL=1 -e QWEN_FAST_PACKED_STEP=1 -e QWEN_FAST_PACKED_AUDIT=1 -e QWEN_FAST_FAULTHANDLER=1 \
  ${M3NATIVE_WATCHER:+-e TT_METAL_WATCHER=20} ${M3NATIVE_WATCHER:+-e TT_METAL_WATCHER_APPEND=1} ${M3NATIVE_WATCHER:+-e TT_METAL_WATCHER_DISABLE_ASSERT=1} \
  -e QWEN_GDN_DIRECT_WINDOW=1 -e QWEN_GDN_SHARED_QK_EXPERIMENT=1 \
  -e QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1 -e QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1 \
  -e QWEN_SDPA_BF8=1 -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 \
  -e QWEN_SKIP_UNUSED_SINGLETON_POSITIONS=1 -e QWEN_FAST_PHASE_TIMING=1 \
  -e QWEN_GDN_GROUPED_GATHER_ABBA=0 -e QWEN_GDN_GATE_EXP_ABBA=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e TT_METAL_HOME=/opt/tt-metal -e MESH_DEVICE=P300 -e OMP_NUM_THREADS=8 \
  -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels \
  -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
  --entrypoint python3 "$image" "${entry_args[@]}" \
  --users "$users" --context "$context" --prompt-tokens "$prompt_tokens" --max-tokens "$max_tokens" --stream-timeout 600 --trace-region-bytes "$trace_region_bytes" \
  --prompt-base 1000 --prompt-user-offset 1 --stagger 0 \
  --references /bench/packed-gate-reference $allow_missing_references \
  > experiment-results/m3native-gate-stdout.log 2>&1 || true

if [ "${M3NATIVE_PROFILE:-}" = "1" ]; then
  # tracy wrote its .logs as root inside the bind mount; the runner user could not
  # delete them and the next checkout died (run 35562881085, EACCES unlink). A container
  # with its default capabilities hands the tree back before anything else runs.
  timeout -k 10 120 docker run --rm --network none     --mount "type=bind,src=$PWD/experiment-results/profile,dst=/p"     --entrypoint sh "$image" -c 'chmod -R a+rwX /p' > /dev/null 2>&1 || true
fi
sed -n '/M3NATIVE_GATE_JSON_BEGIN/,/M3NATIVE_GATE_JSON_END/p' experiment-results/m3native-gate-stdout.log \
  | sed '1d;$d' > experiment-results/m3native-gate.json || true
sed -n '/M3NATIVE_GATE_LOG_BEGIN/,/M3NATIVE_GATE_LOG_END/p' experiment-results/m3native-gate-stdout.log \
  | sed '1d;$d' > experiment-results/m3native-server-tail.log || true
# Every '[PINDIAG] dram after' line, straight to the workflow log: an arm that never
# reaches a packed round (e.g. the 131k attach probe) still has this to read without
# downloading the artifact.
grep -F '[PINDIAG] dram after' experiment-results/m3native-server-tail.log || true
test -s experiment-results/m3native-gate.json
