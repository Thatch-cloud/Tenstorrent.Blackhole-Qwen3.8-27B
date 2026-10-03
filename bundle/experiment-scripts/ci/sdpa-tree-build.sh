#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_HARDWARE_TESTS:-0}" = 1
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${TT_METAL_HOME:-}" = /opt/tt-metal
python3 /experiment-scripts/ci/sdpa-tree-audit.py
PYTHONPATH=/experiment-scripts/ci python3 -c 'from sdpa_tree_scratch import audit; audit("/opt/tt-metal")'
command -v clang++-20
command -v ninja
test -f /opt/tt-metal/build_Release/build.ninja
python3 /experiment-scripts/ci/sdpa_graft_build.py
grafts=/experiment-optimisation/sim/sdpa-graft-registration.patch
git -C /opt/tt-metal apply --check "$grafts"
git -C /opt/tt-metal apply "$grafts"
patch=/experiment-optimisation/sim/sdpa-tree-scratch.patch
git -C /opt/tt-metal apply --check "$patch"
git -C /opt/tt-metal apply "$patch"
PYTHONPATH=/experiment-scripts/ci python3 -c 'from sdpa_tree_scratch import audit; audit("/opt/tt-metal", patched=True)'
timeout -k 30 1800 ninja -C /opt/tt-metal/build_Release -j 2 ttnncpp
source=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so
destination=/opt/tt-metal/build_Release/lib/_ttnncpp.so
test -f "$source"
test -f "$destination"
if [ "$(readlink -f "$source")" != "$(readlink -f "$destination")" ]; then
    cp "$source" "$destination"
fi
sha256sum "$destination" "$patch" > /experiment/results/sdpa-tree-build.sha256
sha256sum "$grafts" >> /experiment/results/sdpa-tree-build.sha256
python3 -c 'import ttnn; names = ("attn_decode_prep", "gdn_decode_norm_gate", "gdn_decode_conv_gates", "decode_gated_delta_rule_packed"); assert all(callable(getattr(ttnn.transformer, name)) for name in names); print("Existing transformer graft ABI imports before any device opens")'
ldd /opt/tt-metal/ttnn/ttnn/_ttnn.so > /experiment/results/sdpa-tree-loaded-libraries.txt
