#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CCL_LAZY_BUILD:-0}" = 1
test "${TT_METAL_HOME:-}" = /opt/tt-metal
test -f /opt/tt-metal/build_Release/build.ninja
python3 /experiment-scripts/ci/sdpa_graft_build.py
git -C /opt/tt-metal apply --check /tmp/ccl-graft-registration.patch
git -C /opt/tt-metal apply /tmp/ccl-graft-registration.patch
python3 /experiment-scripts/ci/lazy_ccl_links.py --root /opt/tt-metal \
    --output /experiment/results/ccl-links-source.json
timeout -k 30 1800 ninja -C /opt/tt-metal/build_Release -j 2 ttnncpp
source=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so
destination=/opt/tt-metal/build_Release/lib/_ttnncpp.so
test -f "$source"
test -f "$destination"
if [ "$(readlink -f "$source")" != "$(readlink -f "$destination")" ]; then
    cp "$source" "$destination"
fi
sha256sum "$destination" > /experiment/results/ccl-links-build.sha256
python3 -c 'import ttnn; names = ("attn_decode_prep", "gdn_decode_norm_gate", "gdn_decode_conv_gates", "decode_gated_delta_rule_packed"); assert all(callable(getattr(ttnn.transformer, name)) for name in names)'
