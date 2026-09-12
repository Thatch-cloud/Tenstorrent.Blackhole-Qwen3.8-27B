#!/usr/bin/env bash
set -euo pipefail
mkdir -p experiment-results
exec > >(tee experiment-results/model-inventory.log) 2>&1
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
for container in $(docker ps -aq); do
    docker inspect --format '{{.Name}} {{range .Mounts}}{{.Type}}:{{.Source}} -> {{.Destination}}; {{end}}' "$container" || continue
done
probe_id=$(docker create -i --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --pids-limit 128 --memory 2g --cpus 2 \
    --mount type=bind,src=/home,dst=/host-home,readonly \
    --entrypoint /bin/bash "$image" -s)
trap 'docker rm -f "$probe_id" >/dev/null' EXIT
docker start -ai "$probe_id" <<'PROBE'
set -euo pipefail
for directory in /host-home/*/hf-cache/hub/models--Qwen--Qwen3.8-27B /host-home/*/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B; do
    if [ -d "$directory" ]; then
        printf 'model_cache=%s\n' "$directory"
        find "$directory/snapshots" -mindepth 1 -maxdepth 1 -type d -printf 'snapshot=%f\n'
    fi
done
for source in /opt/tt-metal /opt/vllm-tt-plugin; do
    git -C "$source" rev-parse HEAD
done
python3 - <<'PY'
import importlib.metadata
import hashlib
from pathlib import Path
for name in ('vllm', 'vllm-tt-plugin', 'transformers', 'torch'):
    print(name, importlib.metadata.version(name))
root = Path('/opt/tt-metal/models/demos/blackhole/qwen36/tt')
for name in ('mlp.py', 'model_config.py'):
    source = root / name
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 524288:
        raise RuntimeError(f'Expected bounded native source file: {source}')
    content = source.read_bytes()
    print(f'QWEN_NATIVE_SOURCE_BEGIN {name} sha256={hashlib.sha256(content).hexdigest()}', flush=True)
    for line, value in enumerate(content.decode('utf-8').splitlines(), 1):
        print(f'{line}: {value}')
    print(f'QWEN_NATIVE_SOURCE_END {name}', flush=True)
PY
find /opt/vllm-tt-plugin/src/vllm_tt_plugin -maxdepth 3 -type f \( -name '*runner*' -o -name '*scheduler*' -o -name '*platform*' \) -print
grep -R -n -E 'sample_on_device_mode|force_argmax|kv_cache_usage_perc|get_num_available_blocks_tt' /opt/vllm-tt-plugin/src/vllm_tt_plugin --include='*.py' | head -80 || true
PROBE
test "$(docker inspect --format '{{.State.ExitCode}}' "$probe_id")" = 0
