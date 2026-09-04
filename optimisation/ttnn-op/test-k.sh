#!/bin/bash
# Run a K unit test in the serving image with the opgraft-K libraries and every grafted op's
# kernel sources mounted (same graft shape as arunk.sh), on card M only.
#   test-k.sh [test-file]
set -u
T=${1:-$HOME/kwork/test_gdn_conv_gates.py}
IMG=zot.thatch.local:5000/tt-serving:v0.77.0-rc1-prstack
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
G=$HOME/opgraft-K
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
echo "### test start $(date -Is) load=[$(cut -d' ' -f1-3 /proc/loadavg)]"
timeout 900 docker run --rm --name ktest \
  --device "$D1" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/ttcache:/ttcache" -e TT_METAL_CACHE=/ttcache \
  -v "$G/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro" \
  -v "$G/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro" \
  -v "$G/gdn_conv_gates:$OPS/gdn_conv_gates:ro" \
  -v "$G/gdn_norm_gate:$OPS/gdn_norm_gate:ro" \
  -v "$G/attn_prep:$OPS/attn_prep:ro" \
  -v "$G/decode_gated_delta_rule:$OPS/decode_gated_delta_rule:ro" \
  -v "$G/gdn_decay:$OPS/gdn_decay:ro" \
  -v "$T:/tmp/test_k.py:ro" \
  -w /opt/tt-metal \
  "$IMG" python3 /tmp/test_k.py 2>&1 | grep -vE "^\s*$" | tail -60
echo "### test end $(date -Is)"
