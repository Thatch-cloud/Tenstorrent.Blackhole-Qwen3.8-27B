"""Stage norm-reader ABBA requests on a fresh winning frozen ladder."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once
from mlp_read_order_stage import adapt as adapt_schedule
from shared_qk_norm_scatter_gate import REPORT_SHA256, DEPENDENCIES, validate_report


def adapt(sources):
    scope = sources['frozen_draft_tail_scope.py']
    result = adapt_schedule({name: value for name, value in sources.items() if name != 'frozen_draft_tail_scope.py'})
    result['dspark-target-hardware.py'] = replace_once(result['dspark-target-hardware.py'],
        'from mlp_read_order_experiment import run_loaded_requests',
        'from shared_qk_norm_experiment import run_loaded_requests')
    result['run-dspark-hardware.sh'] = result['run-dspark-hardware.sh'].replace(
        'QWEN_MLP_READ_ORDER', 'QWEN_SHARED_QK_NORM_COMPARISON')
    result['frozen_draft_tail_scope.py'] = replace_once(scope,
        'from frozen_gdn_norm_scope import runtime_scope as norm_scope',
        'from shared_qk_norm_comparison_scope import runtime_scope as norm_scope')
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
    if options.manifest.exists() or (scripts / 'shared-qk-norm-scatter.json').exists():
        raise ValueError('Fresh norm comparison staging required')
    raw = (options.evidence / 'gdn-shared-recurrence.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained scatter simulation required')
    report = json.loads(raw)
    validate_report(report)
    if (options.evidence / 'gdn-shared-recurrence.exit-status').read_text().strip() != '0':
        raise ValueError('Clean simulator exit required')
    cleanup = json.loads((options.evidence / 'container-cleanup.json').read_text())
    if any(cleanup.get(name) != 0 for name in ('stop_exit', 'logs_exit', 'copy_exit', 'remove_exit')):
        raise ValueError('Clean simulator container teardown required')
    names = ('dspark_request_experiment.py', 'dspark-target-hardware.py',
        'run-dspark-hardware.sh', 'frozen_draft_tail_scope.py')
    original = {name: (scripts / name).read_bytes() for name in names}
    result = adapt({name: payload.decode() for name, payload in original.items()})
    for name in ('shared_qk_norm_experiment.py', 'shared_qk_norm_comparison.py',
            'shared_qk_norm_comparison_scope.py', 'shared_qk_norm_scatter_gate.py',
            'shared_qk_norm_scatter.py', 'gdn_norm_scatter.py'):
        result[name] = Path(__file__).with_name(name).read_text()
    for name in DEPENDENCIES:
        payload = result[name].encode() if name in result else (scripts / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != report['sources'].get('/experiment-scripts/ci/' + name):
            raise ValueError('Scatter simulator dependency drift: ' + name)
    for name, source in result.items():
        (scripts / name).write_bytes(source.encode())
    (scripts / 'shared-qk-norm-scatter.json').write_bytes(raw)
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(payload).hexdigest() for name, payload in original.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        simulator_report_sha256=REPORT_SHA256, hardware_qualified=False,
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
