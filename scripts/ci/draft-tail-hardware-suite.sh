#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export PYTHONPATH=/experiment-scripts/ci:/opt/tt-metal/ttnn:/opt/tt-metal
test "$(git -C /opt/tt-metal rev-parse HEAD)" = 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
test -z "${TT_METAL_SIMULATOR:-}"
export TT_METAL_DEVICE_PROFILER=0 TTNN_OP_PROFILER=0
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
verify_sources() {
    python3 - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('/experiment-scripts/ci')
sources = json.loads((root / 'draft-tail-hardware-sources.json').read_text())
for name, expected in sources.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise ValueError('Staged source changed: ' + name)
PY
}
verify_sources
status=0
timeout -k 10 210 python3 -u /experiment-scripts/ci/draft-tail-hardware-probe.py \
    --output /experiment/results/draft-tail-hardware.json 2>&1 | tee /experiment/results/probe.log || status=$?
printf '%s\n' "$status" > /experiment/results/draft-tail-hardware.exit-status
test "$status" = 0
verify_sources
python3 - <<'PY'
import json
from pathlib import Path
from frozen_draft_tail_hardware import validate_hardware
root = Path('/experiment/results')
report = json.loads((root / 'draft-tail-hardware.json').read_text())
sources = json.loads(Path('/experiment-scripts/ci/draft-tail-hardware-sources.json').read_text())
if report['harness_sha256'] != sources.pop('draft-tail-hardware-probe.py') or report['sources'] != sources:
    raise ValueError('Hardware report must bind every staged source')
summary = validate_hardware(report)
(root / 'draft-tail-component-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
PY
