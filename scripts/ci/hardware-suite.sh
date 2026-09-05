#!/usr/bin/env bash
set -euo pipefail
mkdir -p /experiment/results /experiment/optimisation /experiment/scripts
cp -R /experiment-scripts/. /experiment/scripts/
cp -R /experiment-optimisation/. /experiment/optimisation/
unset TT_METAL_SIMULATOR TT_METAL_SLOW_DISPATCH_MODE TT_METAL_MOCK_CLUSTER_DESC_PATH TT_MESH_GRAPH_DESC_PATH
export PYTHONPATH=/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=2
python3 -m pip freeze > /experiment/results/packages.txt
timeout 120 python3 /experiment/scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
for device in 0 1; do
    timeout 600 python3 /experiment/scripts/ci/hardware-correctness.py --suite kernels --device-index "$device" \
        --output "/experiment/results/kernels-device-$device.json"
    for length in 64 65 79 96; do
        for seed in 0 1; do
            export PREFILL_RESULT_PATH="/experiment/results/prefill-device-$device-length-$length-seed-$seed.json"
            timeout 180 python3 /experiment/optimisation/sim/prefill-state.py --hardware \
                --device-index "$device" --length "$length" --seed "$seed"
        done
    done
done
export TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
timeout 180 python3 /experiment/scripts/ci/hardware-correctness.py --suite fabric --output /experiment/results/fabric.json
