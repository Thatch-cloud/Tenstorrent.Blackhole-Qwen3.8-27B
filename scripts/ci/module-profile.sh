#!/usr/bin/env bash
set -euo pipefail
cd /opt/tt-metal
export TT_METAL_DEVICE_PROFILER=1
root=models/demos/blackhole/qwen36/tests
mkdir -p /experiment/results/profiles
python3 /experiment-scripts/ci/stage-profile.py /opt/tt-metal > /experiment/results/profiles/fixture-stage.json
for module in gdn mlp attention; do
    output="/experiment/results/profiles/$module"
    mkdir -p "$output"
    case "$module" in
        gdn) selection=("$root/test_gdn_tp.py::test_gdn_tp" -k B1) ;;
        mlp) selection=("$root/test_mlp_tp.py::test_mlp_tp") ;;
        attention) selection=("$root/test_attention_tp.py::test_attention_tp_paged") ;;
    esac
    timeout 900 python3 -m tracy -o "$output" -r -m pytest "${selection[@]}" \
        --timeout=800 -x -v -s --junitxml="$output/tests.xml" 2>&1 | tee "$output/console.log"
    python3 /experiment-scripts/ci/check-profile.py "$output"
done
