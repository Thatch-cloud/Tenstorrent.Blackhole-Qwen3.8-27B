#!/usr/bin/env bash
set -euo pipefail
SIM_ROOT=/opt/ttsim
METAL_ROOT="$SIM_ROOT/tt-metal"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH="$SIM_ROOT/venv/bin:/usr/lib/llvm-20/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
test -f "$METAL_ROOT/build_Release/build.ninja"
cd "$METAL_ROOT"
mode=${1:-stock}
case "$mode" in
stock)
printf '%s\n' \
    '61b4c71959eb6100db2cb3ac45a51c29045db17d88630b08fb12f8632327f8fe  tt_metal/impl/context/metal_env_impl.hpp' \
    'b7382e2a89dee1c1e0e818a659662946c3d8d57d20fa65ff27a199a40c32430c  tt_metal/llrt/core_descriptor.cpp' | sha256sum -c -
patch="$SCRIPT_DIR/eth-dispatch-harvesting.patch"
git apply --check "$patch"
;;
harvesting)
printf '%s\n' 'e4f514dca6fd3886cf0e767a5a0a7473099fc6488c38e8d09f48ebd5a7c3dee0  tt_metal/llrt/core_descriptor.cpp' | sha256sum -c -
patch="$SCRIPT_DIR/eth-dispatch-harvesting.patch"
git apply --reverse --check "$patch"
;;
*) printf 'Expected stock or harvesting source state\n' >&2; exit 2 ;;
esac
common_patch="$SCRIPT_DIR/eth-dispatch-common-pool.patch"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
BACKUP="$SIM_ROOT/runtime-baselines/eth-dispatch-$RUN_ID"
mkdir -p "$BACKUP" "$SIM_ROOT/results"
cp build_Release/lib/libtt_metal.so "$BACKUP/libtt_metal.so"
cp build_Release/lib/_ttnncpp.so "$BACKUP/lib-ttnncpp.so"
sha256sum "$patch" "$common_patch" "$BACKUP/"*.so > "$SIM_ROOT/results/$RUN_ID-eth-dispatch-build.sha256"
if [ "$mode" = stock ]; then
    git apply "$patch"
fi
git apply --check "$common_patch"
git apply "$common_patch"
status=0
timeout -k 30 1800 ninja -C build_Release -j 2 ttnncpp 2>&1 | tee "$SIM_ROOT/results/$RUN_ID-eth-dispatch-build.log" || status=$?
printf '%s\n' "$status" > "$SIM_ROOT/results/$RUN_ID-eth-dispatch-build.exit-status"
test "$status" = 0
metal_source=build_Release/tt_metal/libtt_metal.so
metal_destination=build_Release/lib/libtt_metal.so
if [ "$(readlink -f "$metal_source")" != "$(readlink -f "$metal_destination")" ]; then
    cp "$metal_source" "$metal_destination"
fi
source=build_Release/ttnn/_ttnncpp.so
destination=build_Release/lib/_ttnncpp.so
if [ "$(readlink -f "$source")" != "$(readlink -f "$destination")" ]; then
    cp "$source" "$destination"
fi
export TT_METAL_HOME="$METAL_ROOT"
export PYTHONPATH="$METAL_ROOT/ttnn:$METAL_ROOT"
export LD_LIBRARY_PATH="$METAL_ROOT/build_Release/lib"
export PYTHONDONTWRITEBYTECODE=1
python -c 'import ttnn; names = ("attn_decode_prep", "gdn_decode_norm_gate", "gdn_decode_conv_gates"); assert all(callable(getattr(ttnn.transformer, name)) for name in names); print("Patched simulator runtime imports; no device opened")'
sha256sum build_Release/lib/libtt_metal.so "$destination" tt_metal/llrt/core_descriptor.cpp \
    tt_metal/impl/context/metal_env_impl.hpp >> "$SIM_ROOT/results/$RUN_ID-eth-dispatch-build.sha256"
