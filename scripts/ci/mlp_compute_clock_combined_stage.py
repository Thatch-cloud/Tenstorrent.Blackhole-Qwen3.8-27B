"""Add bounded diagnostic ownership to a fresh frozen combined ladder checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from mlp_compute_clock_experiment import HELPERS, candidate_class


def adapt(sources):
    result = dict(sources)
    result['dspark_request_experiment.py'] = replace_once(result['dspark_request_experiment.py'],
        "    schedule = (('publication', True), ('publication', False), ('publication', False))",
        "    schedule = (('publication', True),)")
    result['dspark-target-hardware.py'] = replace_once(result['dspark-target-hardware.py'],
        '            from dspark_request_experiment import run_loaded_requests',
        '            from mlp_compute_clock_experiment import run_loaded_requests')
    result['run-dspark-hardware.sh'] = replace_once(result['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_MLP_COMPUTE_CLOCK_COMBINED=${QWEN_MLP_COMPUTE_CLOCK_COMBINED:-0}" \\\n'
        '    -e "QWEN_DSPARK_MODE=$mode"')
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    if options.manifest.exists() or (scripts / 'compute-clock-evidence').exists():
        raise ValueError('Fresh combined clock staging required')
    names = ('dspark_request_experiment.py', 'dspark-target-hardware.py', 'run-dspark-hardware.sh')
    original = {name: (scripts / name).read_bytes() for name in names}
    result = adapt({name: payload.decode() for name, payload in original.items()})
    for name in (*HELPERS, 'mlp_compute_clock_experiment.py', 'mlp_compute_clock_combined.py', 'mlp_compute_clock_hardware.py', 'mlp_compute_clock_report.py', 'mlp_clock_report.py'):
        result[name] = Path(__file__).with_name(name).read_text()
    for name, source in result.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(options.evidence, scripts / 'compute-clock-evidence')
    candidate_class(scripts, scripts / 'compute-clock-evidence')
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(payload).hexdigest() for name, payload in original.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        combined_qualified=False, performance_qualified=False, diagnostic_only=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
