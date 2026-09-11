#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_SIM_ONLY:-0}" = 1
test "${QWEN_T32_FP32_BUILD:-0}" = 1
test "${QWEN_HARDWARE_TESTS:-0}" != 1
test "${QWEN_CARDS_ALLOCATED:-0}" != 1
test ! -e /dev/tenstorrent
test "${TT_METAL_HOME:-}" = /opt/tt-metal
test -f /opt/tt-metal/build_Release/build.ninja
test ! -e /experiment/results/t32-fp32-build.json
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
from t32_ci_runtime import PINS
from t32_attention_fp32_patch import SOURCE_PATH, patched_bytes
root = Path('/opt/tt-metal')
binaries = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in PINS if name.endswith('.so')}
assert binaries == {name: value for name, value in PINS.items() if name.endswith('.so')}
source = root / SOURCE_PATH
original = source.read_bytes()
candidate = patched_bytes(original)
report = dict(stage='prepared', base_binaries=binaries,
              original_factory=hashlib.sha256(original).hexdigest(),
              candidate_factory=hashlib.sha256(candidate).hexdigest())
Path('/experiment/results/t32-fp32-build.json').write_text(json.dumps(report, indent=2))
source.write_bytes(candidate)
PY
python3 /experiment-scripts/ci/sdpa_graft_build.py
grafts=/simulator-support/sdpa-graft-registration.patch
git -C /opt/tt-metal apply --check "$grafts"
git -C /opt/tt-metal apply "$grafts"
timeout -k 30 1800 ninja -C /opt/tt-metal/build_Release -j 4 ttnncpp
source=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so
destination=/opt/tt-metal/build_Release/lib/_ttnncpp.so
test -f "$source"
test -f "$destination"
if [ "$(readlink -f "$source")" != "$(readlink -f "$destination")" ]; then
    cp "$source" "$destination"
fi
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
from t32_attention_fp32_patch import SOURCE_PATH
root = Path('/opt/tt-metal')
path = Path('/experiment/results/t32-fp32-build.json')
report = json.loads(path.read_text())
assert hashlib.sha256((root / SOURCE_PATH).read_bytes()).hexdigest() == report['candidate_factory']
report['binaries'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in report['base_binaries']}
assert len(set(report['binaries'].values())) == 1
assert all(report['binaries'][name] != value for name, value in report['base_binaries'].items())
report['registration_patch'] = hashlib.sha256(Path('/simulator-support/sdpa-graft-registration.patch').read_bytes()).hexdigest()
report['stage'] = 'built'
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
PY
