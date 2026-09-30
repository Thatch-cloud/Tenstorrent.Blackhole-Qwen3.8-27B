"""Stage two audited arms and complete ABBA requests on the frozen T16 recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from gdn_window_write_gate import qualify
from gdn_window_write_stage import stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    if options.manifest.exists() or (scripts / 'gdn-window-write-evidence').exists():
        raise ValueError('Fresh combined window-write staging required')
    simulation = stage(options.checkout)
    evidence = qualify(scripts, options.evidence)
    changes = {
        'dspark_request_experiment.py': (
            "    schedule = (('publication', True), ('publication', False), ('publication', False))",
            "    schedule = (('publication', True), ('publication', True), ('publication', False),\n"
            "        ('publication', False), ('publication', False), ('publication', False))"),
        'dspark-target-hardware.py': (
            '            from dspark_request_experiment import run_loaded_requests',
            '            from gdn_window_write_experiment import run_loaded_requests'),
        'run-dspark-hardware.sh': (
            '    -e "QWEN_DSPARK_MODE=$mode"',
            '    -e "QWEN_GDN_WINDOW_WRITE=${QWEN_GDN_WINDOW_WRITE:-0}" \\\n'
            '    -e "QWEN_DSPARK_MODE=$mode"'),
    }
    originals = {name: (scripts / name).read_bytes() for name in changes}
    payloads = {name: replace_once(originals[name].decode(), *change) for name, change in changes.items()}
    for name in ('gdn_window_write_experiment.py', 'gdn_window_write_comparison.py',
            'gdn_window_write_gate.py', 'gdn_window_write_scope.py', 'gdn_window_write_overlap.py',
            'gdn-window-write-probe.py'):
        payloads[name] = Path(__file__).with_name(name).read_text()
    for name, source in payloads.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(options.evidence, scripts / 'gdn-window-write-evidence')
    if qualify(scripts, scripts / 'gdn-window-write-evidence') != evidence:
        raise ValueError('Qualification changed after staging')
    options.manifest.write_text(json.dumps(dict(simulation=simulation, admission=evidence,
        before={name: hashlib.sha256(payload).hexdigest() for name, payload in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        hardware_qualified=False, performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
