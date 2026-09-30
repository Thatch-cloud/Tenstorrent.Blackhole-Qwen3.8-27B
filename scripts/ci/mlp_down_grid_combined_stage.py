"""Stage complete matched mlp-down-grid requests on the frozen winning T16 recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from mlp_down_grid_gate import validate_report, REPORT_SHA256, LOCAL_SOURCES


def adapt(sources):
    result = dict(sources)
    result['dspark_request_experiment.py'] = replace_once(result['dspark_request_experiment.py'],
        "    schedule = (('publication', True), ('publication', False), ('publication', False))",
        "    schedule = (('publication', True), ('publication', True), ('publication', False),\n"
        "        ('publication', False), ('publication', False), ('publication', False))")
    result['dspark-target-hardware.py'] = replace_once(result['dspark-target-hardware.py'],
        '            from dspark_request_experiment import run_loaded_requests',
        '            from mlp_down_grid_experiment import run_loaded_requests')
    result['run-dspark-hardware.sh'] = replace_once(result['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_MLP_DOWN_GRID=${QWEN_MLP_DOWN_GRID:-0}" \\\n'
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
    if options.manifest.exists() or (scripts / 'mlp-down-grid-evidence').exists():
        raise ValueError('Fresh combined MLP-down staging required')
    raw = (options.evidence / 'gdn-output-grid.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact reviewed MLP-down report required')
    report = json.loads(raw)
    validate_report(report)
    names = ('dspark_request_experiment.py', 'dspark-target-hardware.py', 'run-dspark-hardware.sh')
    original = {name: (scripts / name).read_bytes() for name in names}
    result = adapt({name: data.decode() for name, data in original.items()})
    for name in ('mlp_down_grid_experiment.py', 'mlp_down_grid_comparison.py', 'mlp_down_grid_gate.py',
            'mlp_down_grid_scope.py', 'mlp_down_grid.py', 'mlp-down-grid-probe.py'):
        result[name] = Path(__file__).with_name(name).read_bytes().decode()
    for measured, name in LOCAL_SOURCES.items():
        source = result[name].encode() if name in result else (scripts / name).read_bytes()
        if hashlib.sha256(source).hexdigest() != report['sources'].get('/experiment-scripts/ci/' + measured):
            raise ValueError('Staged projection dependency differs from simulation: ' + name)
    for name, source in result.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(options.evidence, scripts / 'mlp-down-grid-evidence')
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(data).hexdigest() for name, data in original.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        simulator_report_sha256=REPORT_SHA256, runtime_source_admission_required=True,
        hardware_qualified=False, performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
