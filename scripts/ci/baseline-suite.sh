#!/usr/bin/env bash
set -euo pipefail
mkdir -p /experiment/results
unset TT_METAL_SIMULATOR TT_METAL_SLOW_DISPATCH_MODE TT_METAL_MOCK_CLUSTER_DESC_PATH
export PYTHONPATH=/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
revision=$(cat /models/hub/models--Qwen--Qwen3.8-27B/refs/main)
[[ "$revision" =~ ^[a-f0-9]{40}$ ]]
export MODEL_WEIGHTS_DIR="/models/hub/models--Qwen--Qwen3.8-27B/snapshots/$revision"
export HF_MODEL="$MODEL_WEIGHTS_DIR"
python3 - <<'PY' > /experiment/results/model-manifest.json
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ['MODEL_WEIGHTS_DIR'])
index = json.loads((root / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    if not (root / name).is_file():
        raise RuntimeError(f'Missing weight shard: {name}')
files = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json']
print(json.dumps(dict(snapshot=root.name, files={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in files}, config=json.loads((root / 'config.json').read_text()),
    flags={name: value for name, value in os.environ.items() if name.startswith(('QWEN', 'TT_', 'MESH_DEVICE'))}), indent=2))
PY
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.8-27B --revision "$revision" --served-model-name qwen3.8-27b \
    --max-model-len 65536 --max-num-seqs 8 --no-enable-prefix-caching --block-size 64 \
    --reasoning-parser qwen3 --port 8000 --host 127.0.0.1 \
    --additional-config '{"tt":{"l1_small_size":24576,"fabric_config":"FABRIC_1D","trace_region_size":1073741824}}' \
    > /experiment/results/server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
ready=0
for attempt in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8000/v1/models > /experiment/results/models.json; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -80 /experiment/results/server.log; exit 1; fi
    sleep 5
done
if [ "$ready" != 1 ]; then tail -80 /experiment/results/server.log; exit 1; fi
printf 'Endpoint ready; starting warmed baseline matrix\n'
timeout 4800 python3 /experiment-scripts/ci/baseline-client.py --tokenizer "$MODEL_WEIGHTS_DIR" \
    --output /experiment/results --context 65536 --tokens 1024 --repeats 3
