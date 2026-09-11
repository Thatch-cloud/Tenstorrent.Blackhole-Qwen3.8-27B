#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
SIM_ROOT=${SIM_ROOT:-/opt/ttsim}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH="$SIM_ROOT/venv/bin:/usr/lib/llvm-20/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONDONTWRITEBYTECODE=1
export TT_METAL_HOME="$SIM_ROOT/tt-metal"
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/lib"
export TT_METAL_SIMULATOR="$SIM_ROOT/simulator/libttsim_bh_x2.so"
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$SIM_ROOT/simulator/blackhole_P300_both_mmio.yaml"
export TT_METAL_CACHE="$SIM_ROOT/kernel-cache"
if [ "${QWEN_SIM_PACKER_ZERO_GRAFT:-0}" = 1 ]; then
    packer="$TT_METAL_HOME/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h"
    [[ "$(sha256sum "$packer" | cut -d ' ' -f 1)" = 8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7 ]]
    export TT_METAL_CACHE="$SIM_ROOT/kernel-cache-packer-zero-a00d91e"
fi
unset TT_METAL_SLOW_DISPATCH_MODE TT_MESH_GRAPH_DESC_PATH QWEN_HARDWARE_TESTS QWEN_CARDS_ALLOCATED
test -f "$TT_METAL_SIMULATOR"
test -f "$TT_METAL_MOCK_CLUSTER_DESC_PATH"
cp "$TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "$SIM_ROOT/simulator/soc_descriptor.yaml"
if [ "${QWEN_SIM_SHARED_BDF:-0}" = 1 ]; then
    shared="$SIM_ROOT/simulator/shared-bdf"
    mkdir -p "$shared"
    cp "$TT_METAL_SIMULATOR" "$shared/libttsim_bh_x2.so"
    cp "$SIM_ROOT/simulator/soc_descriptor.yaml" "$shared/soc_descriptor.yaml"
    cp "$TT_METAL_MOCK_CLUSTER_DESC_PATH" "$shared/cluster_descriptor.yaml"
    export TT_METAL_SIMULATOR="$shared/libttsim_bh_x2.so"
fi
mkdir -p "$SIM_ROOT/results" "$TT_METAL_CACHE"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
launcher=()
case "${QWEN_SIM_BOUNDED_MEMORY:-0}" in
    0) ;;
    1) launcher=(systemd-run --scope --quiet --collect --unit="qwen-ttsim-$RUN_ID"
        -p MemoryMax=3221225472 -p MemorySwapMax=4294967296 --) ;;
    *) printf 'QWEN_SIM_BOUNDED_MEMORY must be 0 or 1\n' >&2; exit 64 ;;
esac
PROBE=${QWEN_SIM_LAYER_PROBE:-dspark-native-cached-layer-probe}
case "$PROBE" in
    dspark-native-cached-layer-probe|gdn-norm-scatter-probe|target-t16-attention-probe|dspark-approx-fixed-attention-probe|dram-sharded-projection-probe|dram-mlp-probe|dram-mlp-sharded-probe|dram-mlp-down-probe|dspark-score-layout-probe|dspark-markov-score-layout-probe|gdn-batched-publication-probe|tensor-bit-compare-probe|dspark-banked-trace-probe|gdn-native-slot-probe) ;;
    *) exit 64 ;;
esac
{
REPORT="$SIM_ROOT/results/$RUN_ID-$PROBE.json"
printf 'report=%s\n' "$REPORT"
cd "$SIM_ROOT"
status=0
"${launcher[@]}" timeout -k 10 "${KERNEL_TIMEOUT:-300}" "$SIM_ROOT/venv/bin/python" "$SCRIPT_DIR/../../scripts/ci/$PROBE.py" \
    --output "$REPORT" "$@" 2>&1 | tee "$SIM_ROOT/results/$RUN_ID-$PROBE.log" || status=$?
printf '%s\n' "$status" > "$SIM_ROOT/results/$RUN_ID-$PROBE.exit-status"
exit "$status"
}
