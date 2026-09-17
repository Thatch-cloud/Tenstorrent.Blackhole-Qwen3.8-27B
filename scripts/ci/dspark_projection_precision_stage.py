"""Stage a bounded real-weight query-projection simulator screen."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once


CASE = 'dspark-projection-hifi2'
MOUNT = '''if [ "${QWEN_SIM_CASE:-stack}" = dspark-projection-hifi2 ]; then
    kinds=''
    checkpoint="$cache/dspark-b9a5dbdf03bc999c6c73c426b19c2d9041cea393/model.safetensors"
    test -f "$checkpoint"
    mounts+=(--mount "type=bind,src=$checkpoint,dst=/dspark-model.safetensors,readonly")
fi
'''
BRANCH = '''if [ "${QWEN_SIM_CASE:-stack}" = dspark-projection-hifi2 ]; then
    status=0
    timeout -k 15 510 python3 -u /experiment-scripts/ci/dspark-projection-precision-probe.py \\
        --checkpoint /dspark-model.safetensors --projection self_attn.q_proj.weight \\
        --output /experiment/results/dspark-projection-hifi2.json || status=$?
    printf '%s\\n' "$status" > /experiment/results/dspark-projection-hifi2.exit-status
    exit "$status"
fi
'''


def adapt(runner, suite):
    if CASE in runner or CASE in suite:
        raise ValueError('Fresh precision route required')
    anchor = 'case "${QWEN_SIM_CASE:-stack}" in '
    runner = replace_once(runner, anchor, anchor + CASE + '|')
    runner = replace_once(runner, 'mounts=()\n', 'mounts=()\n' + MOUNT)
    suite = replace_once(suite, 'cd /opt/tt-metal\n', 'cd /opt/tt-metal\n' + BRANCH)
    return runner, suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh precision-screen staging required')
    scripts = options.checkout / 'scripts/ci'
    names = ('run-simulator.sh', 'simulator-suite.sh')
    originals = [(scripts / name).read_text() for name in names]
    payloads = dict(zip(names, adapt(*originals), strict=True))
    for name in ('dspark-projection-precision-probe.py', 'dspark_projection_precision.py'):
        payloads[name] = Path(__file__).with_name(name).read_text()
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        component_execution_only=True, simulator_qualified=False, target_correctness_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
