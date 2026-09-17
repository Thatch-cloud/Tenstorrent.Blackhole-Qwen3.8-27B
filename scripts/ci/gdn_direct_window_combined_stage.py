"""Stage qualified direct-window convolution on the unchanged combined T16 recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from gdn_direct_window_gate import qualify
from gdn_direct_window_hardware_sources import payloads as hardware_payloads


def stage(checkout, evidence, native_root, manifest):
    scripts = Path(checkout) / 'scripts/ci'
    directory = Path(__file__).parent
    manifest = Path(manifest)
    if manifest.exists() or (scripts / 'gdn-direct-window-evidence').exists():
        raise ValueError('Fresh direct-window combined staging required')
    qualify(directory, evidence, native_root)
    for name in ('gdn_batched_conv.py', 'gdn_conv_windows.py', 'gdn_conv_windows.cpp',
                 'attention_batch.py', 'gdn_multitoken_conv.py'):
        if (scripts / name).read_bytes() != (directory / name).read_bytes():
            raise ValueError('Frozen runtime control differs: ' + name)
    names = ('gdn_direct_window.py', 'gdn_direct_window_device.py', 'gdn-direct-window-probe.py',
             'gdn_direct_window_report.py', 'gdn_direct_window_gate.py', 'gdn_direct_window_hardware_sources.py',
             'gdn_direct_window_scope.py', 'gdn_direct_window_experiment.py', 'gdn_direct_window_comparison.py')
    payloads = {name: (directory / name).read_text() for name in names}
    payloads.update(hardware_payloads(directory))
    changes = {
        'dspark_request_experiment.py': (
            "    schedule = (('publication', True), ('publication', False), ('publication', False))",
            "    schedule = (('publication', True), ('publication', True), ('publication', False),\n"
            "        ('publication', False), ('publication', False), ('publication', False))"),
        'dspark-target-hardware.py': (
            '            from dspark_request_experiment import run_loaded_requests',
            '            from gdn_direct_window_experiment import run_loaded_requests'),
        'run-dspark-hardware.sh': (
            '    -e "QWEN_DSPARK_MODE=$mode"',
            '    -e "QWEN_GDN_DIRECT_WINDOW=${QWEN_GDN_DIRECT_WINDOW:-0}" \\\n'
            '    -e "QWEN_DSPARK_MODE=$mode"'),
    }
    originals = {name: (scripts / name).read_bytes() for name in changes}
    for name, change in changes.items():
        payloads[name] = replace_once(originals[name].decode(), *change)
    for name, source in payloads.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(evidence, scripts / 'gdn-direct-window-evidence')
    admission = qualify(scripts, scripts / 'gdn-direct-window-evidence', native_root)
    result = dict(admission=admission,
        before={name: hashlib.sha256(source).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        hardware_qualified=False, performance_qualified=False)
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.evidence, options.native_root, options.manifest)
