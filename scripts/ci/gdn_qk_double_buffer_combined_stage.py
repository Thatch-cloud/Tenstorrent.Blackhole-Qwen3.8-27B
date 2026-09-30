"""Stage complete matched qk-double-buffer requests on the frozen winning T16 recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from gdn_qk_double_buffer_gate import validate_report, REPORT_SHA256, DEPENDENCIES


def adapt(sources):
    result = dict(sources)
    result['dspark_request_experiment.py'] = replace_once(result['dspark_request_experiment.py'],
        "    schedule = (('publication', True), ('publication', False), ('publication', False))",
        "    schedule = (('publication', True), ('publication', True), ('publication', False),\n"
        "        ('publication', False), ('publication', False), ('publication', False))")
    result['dspark-target-hardware.py'] = replace_once(result['dspark-target-hardware.py'],
        '            from dspark_request_experiment import run_loaded_requests',
        '            from gdn_qk_double_buffer_experiment import run_loaded_requests')
    result['run-dspark-hardware.sh'] = replace_once(result['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_GDN_QK_DOUBLE_BUFFER=${QWEN_GDN_QK_DOUBLE_BUFFER:-0}" \\\n'
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
    if options.manifest.exists() or (scripts / 'qk-double-buffer-evidence').exists():
        raise ValueError('Fresh combined Q/K-buffer staging required')
    raw = (options.evidence / 'gdn-shared-recurrence.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact reviewed recurrence report required')
    report = json.loads(raw)
    validate_report(report)
    names = ('dspark_request_experiment.py', 'dspark-target-hardware.py', 'run-dspark-hardware.sh')
    original = {name: (scripts / name).read_bytes() for name in names}
    result = adapt({name: data.decode() for name, data in original.items()})
    for name in ('gdn_qk_double_buffer_experiment.py', 'gdn_qk_double_buffer_comparison.py', 'gdn_qk_double_buffer_gate.py',
            'gdn_qk_double_buffer_scope.py', 'gdn_qk_double_buffer.py'):
        result[name] = Path(__file__).with_name(name).read_bytes().decode()
    for name in DEPENDENCIES:
        source = result[name].encode() if name in result else (scripts / name).read_bytes()
        if hashlib.sha256(source).hexdigest() != report['sources'].get('/experiment-scripts/ci/' + name):
            raise ValueError('Staged recurrence dependency differs from simulation: ' + name)
    for name, source in result.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(options.evidence, scripts / 'qk-double-buffer-evidence')
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(data).hexdigest() for name, data in original.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        simulator_report_sha256=REPORT_SHA256, runtime_source_admission_required=True,
        hardware_qualified=False, performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
