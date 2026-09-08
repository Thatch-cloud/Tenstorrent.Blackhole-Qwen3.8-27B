#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
test "${QWEN_SIM_ONLY:-0}" = 1
test ! -e /dev/tenstorrent
unset QWEN_HARDWARE_TESTS QWEN_CARDS_ALLOCATED TT_METAL_SLOW_DISPATCH_MODE TT_MESH_GRAPH_DESC_PATH
mkdir -p /tmp/ttsim /experiment/results
cp /simulator-assets/libttsim_bh_x2.so /tmp/ttsim/
cp /simulator-assets/cluster_descriptor.yaml /tmp/ttsim/
cp /opt/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml /tmp/ttsim/soc_descriptor.yaml
export TT_METAL_HOME=/opt/tt-metal
export PYTHONPATH=/opt/tt-metal:/opt/tt-metal/ttnn:/experiment-scripts/ci
export TT_METAL_SIMULATOR=/tmp/ttsim/libttsim_bh_x2.so
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/tmp/ttsim/cluster_descriptor.yaml
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_CACHE=/tmp/ttsim-kernel-cache
export QWEN_SIM_SHARED_BDF=1
export MESH_DEVICE=P300
git -C /opt/tt-metal rev-parse HEAD > /experiment/results/simulator-runtime.txt
test "$(cat /experiment/results/simulator-runtime.txt)" = 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
cd /opt/tt-metal
if [ "${QWEN_SIM_CASE:-stack}" = shortlist ]; then
    for width in 32768 65536; do
        timeout -k 15 3200 python3 -u /experiment-scripts/ci/draft-shortlist-probe.py \
            --width "$width" --output "/experiment/results/draft-shortlist-$width.json"
    done
    exit 0
fi
test "${QWEN_SIM_CASE:-stack}" = stack
timeout -k 15 6600 python3 -u /experiment-scripts/ci/learned-attention-probe.py \
    --fixture /fixture-attention --convolution-fixture /fixture-convolution --mlp-fixture /fixture-mlp \
    --stack-fixtures /fixture-stack --stack-layers 5 --selector-fixture /fixture-selector \
    --fp32-rope --explicit-softmax --fused-row-sum --fused-dots --cache-dot-tiles --captured-stack \
    --output /experiment/results/learned-attention-simulator.json
