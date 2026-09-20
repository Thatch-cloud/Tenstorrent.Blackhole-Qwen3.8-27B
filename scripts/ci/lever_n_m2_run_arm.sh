#!/usr/bin/env bash
# Serve one M2 gate arm. One invocation per CI job: each server start opens the mesh,
# and two opens in a single job wedged card 2 in run 35414199329 with
# "Read 0xffffffff over PCIe ID 2". $1 is m2|baseline, $2 an optional peer report
# from the other arm's job.
#
# Both arms mount the M1 three-file graft (model.py, qwen36_vllm.py, platform.py) -
# chunked prefill has to actually run for this gate to mean anything. Only the "m2"
# arm additionally mounts the two M2 scheduler files (scheduler.py, lane_scheduler.py);
# "baseline" leaves those two at the image's stock, unpatched content, which is what
# demonstrates the stall M2 fixes.
set -euo pipefail
arm="$1"
peer="${2:-}"
image=sha256:bd878710e15c574e53451976122a9ebb0d5f146fdd37eeb92a9138f773be33fa
target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
root=/opt/tt-metal/models/demos/blackhole/qwen36/tt
plugin=/opt/qwen-fast-plugin/src/vllm_tt_plugin

mkdir -p experiment-results

mounts=(
  --mount "type=bind,src=$PWD/graft/model.py,dst=$root/model.py,readonly"
  --mount "type=bind,src=$PWD/graft/qwen36_vllm.py,dst=$root/qwen36_vllm.py,readonly"
  --mount "type=bind,src=$PWD/graft/platform.py,dst=$plugin/platform.py,readonly"
)
if [ "$arm" = "m2" ]; then
  mounts+=(
    --mount "type=bind,src=$PWD/graft/scheduler.py,dst=$plugin/scheduler.py,readonly"
    --mount "type=bind,src=$PWD/graft/lane_scheduler.py,dst=$plugin/lane_scheduler.py,readonly"
  )
elif [ "$arm" != "baseline" ]; then
  echo "unknown arm: $arm" >&2
  exit 1
fi

peer_args=()
if [ -n "$peer" ] && [ -s "$peer" ]; then
  mounts+=(--mount "type=bind,src=$PWD/$peer,dst=/bench/peer.json,readonly")
  peer_args=(--peer /bench/peer.json)
fi
mapfile -t nodes < <(ls /dev/tenstorrent | grep -E '^[0-9]+$' | sort)
devices=()
for node in "${nodes[@]}"; do devices+=(--device "/dev/tenstorrent/$node"); done

name="qwen-m2-$arm-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
timeout -k 30 1800 docker run --rm --name "$name" --network none \
  --hostname qwen-m2 --add-host qwen-m2:127.0.0.1 \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 4096 --memory 96g --cpus 16 --shm-size 8g \
  "${devices[@]}" \
  --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$target,dst=/models/hub/models--Qwen--Qwen3.8-27B,readonly" \
  --mount "type=bind,src=$PWD/scripts/ci/lever_n_m2_gate.py,dst=/bench/lever_n_m2_gate.py,readonly" \
  --mount type=volume,src=qwen-experiments-f1e9b1a64b4f,dst=/experiment-cache \
  "${mounts[@]}" \
  -e TT_M1_FORCE_CHUNKED_PREFILL=1 \
  -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 \
  -e QWEN_FABRIC_LINK_PROBE=1 -e QWEN_FROZEN_COMBINED_RUNTIME=1 \
  -e QWEN_GDN_DIRECT_WINDOW=1 -e QWEN_GDN_SHARED_QK_EXPERIMENT=1 \
  -e QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1 -e QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1 \
  -e QWEN_SDPA_BF8=1 -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 \
  -e QWEN_SKIP_UNUSED_SINGLETON_POSITIONS=1 -e QWEN_FAST_PHASE_TIMING=1 \
  -e QWEN_GDN_GROUPED_GATHER_ABBA=0 -e QWEN_GDN_GATE_EXP_ABBA=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e TT_METAL_HOME=/opt/tt-metal -e MESH_DEVICE=P300 -e OMP_NUM_THREADS=8 \
  -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels \
  -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
  --entrypoint python3 "$image" -B /bench/lever_n_m2_gate.py \
  --arm "$arm" "${peer_args[@]}" \
  --seqs 4 --decode-tokens 96 --decode-warmup-tokens 8 \
  --prefill-target-tokens 6000 --prefill-max-tokens 8 \
  > "experiment-results/m2-$arm-stdout.log" 2>&1 || true

sed -n '/M2_GATE_JSON_BEGIN/,/M2_GATE_JSON_END/p' "experiment-results/m2-$arm-stdout.log" \
  | sed '1d;$d' > "experiment-results/m2-$arm.json" || true
sed -n '/M2_GATE_LOG_BEGIN/,/M2_GATE_LOG_END/p' "experiment-results/m2-$arm-stdout.log" \
  | sed '1d;$d' > "experiment-results/server-tails-$arm.log" || true
test -s "experiment-results/m2-$arm.json"
