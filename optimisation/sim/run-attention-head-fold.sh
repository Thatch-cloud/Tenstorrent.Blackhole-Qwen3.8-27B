#!/usr/bin/env bash
set -euo pipefail
SIM_ROOT=${SIM_ROOT:-/opt/ttsim}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH="$SIM_ROOT/venv/bin:/usr/lib/llvm-20/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONDONTWRITEBYTECODE=1
export TT_METAL_HOME="$SIM_ROOT/tt-metal"
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/lib"
export TT_METAL_SIMULATOR="$SIM_ROOT/simulator/libttsim_bh_x2.so"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$SIM_ROOT/simulator/blackhole_P300_both_mmio.yaml"
export TT_METAL_CACHE="$SIM_ROOT/kernel-cache"
unset TT_MESH_GRAPH_DESC_PATH QWEN_HARDWARE_TESTS QWEN_CARDS_ALLOCATED
test -f "$TT_METAL_SIMULATOR"
test -f "$TT_METAL_HOME/ttnn/ttnn/_ttnn.so"
test -f "$TT_METAL_MOCK_CLUSTER_DESC_PATH"
cp "$TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "$SIM_ROOT/simulator/soc_descriptor.yaml"
mkdir -p "$SIM_ROOT/results" "$TT_METAL_CACHE"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
PROBE=${QWEN_SIM_PROBE:-attention-head-fold}
[[ "$PROBE" = attention-head-fold || "$PROBE" = attention-mask-replay ]]
export QWEN_SIM_REPORT="$SIM_ROOT/results/$RUN_ID-$PROBE.json"
printf 'report=%s\n' "$QWEN_SIM_REPORT"
cd "$SIM_ROOT"
status=0
timeout -k 10 "${KERNEL_TIMEOUT:-300}" "$SIM_ROOT/venv/bin/python" "$SCRIPT_DIR/$PROBE.py" "$@" 2>&1 | sed -E '/\| +info +\| +Metal +\| (DFB size:|Writing DFB config)/d' | tee "$SIM_ROOT/results/$RUN_ID-$PROBE.log" || status=$?
printf '%s\n' "$status" > "$SIM_ROOT/results/$RUN_ID-$PROBE.exit-status"
exit "$status"
