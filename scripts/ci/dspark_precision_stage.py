"""Stage drafter-only precision comparison on the frozen winning combined recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from dspark_precision_gate import qualify


FILES = ('dspark_precision_experiment.py', 'dspark_precision_comparison.py',
    'dspark_precision_device.py', 'dspark_precision_gate.py', 'dspark_layer_precision.py',
    'dspark_projection_precision.py', 'dspark_projection_precision_report.py',
    'dspark_projection_precision_stage.py', 'dspark-projection-precision-probe.py')


def adapt(sources):
    result = dict(sources)
    result['dspark_request_experiment.py'] = replace_once(result['dspark_request_experiment.py'],
        "    schedule = (('publication', True), ('publication', False), ('publication', False))",
        "    schedule = (('publication', True), ('publication', True), ('publication', False),\n"
        "        ('publication', False), ('publication', False), ('publication', False))")
    result['dspark-target-hardware.py'] = replace_once(result['dspark-target-hardware.py'],
        '            from dspark_request_experiment import run_loaded_requests',
        '            from dspark_precision_experiment import run_loaded_requests')
    result['run-dspark-hardware.sh'] = replace_once(result['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_DSPARK_PROJECTION_HIFI2=${QWEN_DSPARK_PROJECTION_HIFI2:-0}" \\\n'
        '    -e "QWEN_DSPARK_MODE=$mode"')
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--reviewed', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    if (options.manifest.exists() or (scripts / 'precision-evidence').exists()
            or (scripts / 'dspark-precision-reviewed.json').exists()):
        raise ValueError('Fresh frozen-recipe precision staging required')
    names = ('dspark_request_experiment.py', 'dspark-target-hardware.py', 'run-dspark-hardware.sh')
    original = {name: (scripts / name).read_bytes() for name in names}
    result = adapt({name: raw.decode() for name, raw in original.items()})
    for name in FILES:
        result[name] = Path(__file__).with_name(name).read_bytes().decode()
    reviewed = options.reviewed.read_bytes()
    for name, source in result.items():
        (scripts / name).write_bytes(source.encode())
    (scripts / 'dspark-precision-reviewed.json').write_bytes(reviewed)
    shutil.copytree(options.evidence, scripts / 'precision-evidence')
    admission = qualify(scripts, scripts / 'precision-evidence', reviewed_reports=json.loads(reviewed))
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(raw).hexdigest() for name, raw in original.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        reviewed_sha256=hashlib.sha256(reviewed).hexdigest(), simulator=admission,
        hardware_qualified=False, performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
