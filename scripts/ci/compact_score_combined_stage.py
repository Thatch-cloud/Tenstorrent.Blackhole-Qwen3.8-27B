"""Stage compact draft selection on the unchanged matched T16 request recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from compact_score_gate import qualify
from compact_score_hardware_sources import payloads as hardware_payloads
from frozen_recipe_context import replace_once


def stage(checkout, evidence, manifest):
    scripts = Path(checkout) / 'scripts/ci'
    manifest = Path(manifest)
    if manifest.exists() or (scripts / 'compact-score-evidence').exists():
        raise ValueError('Fresh compact combined staging required')
    directory = Path(__file__).parent
    qualify(directory, evidence)
    names = ('compact_score_gate.py', 'compact_score_report.py', 'compact_score_hardware_sources.py',
             'compact_score_scope.py', 'compact_score_experiment.py', 'compact_score_comparison.py',
             'compact_score_device.py', 'compact_markov.py', 'compact-score-probe.py',
             'compact_score_io.cpp', 'compact_score_compute.cpp', 'compact_score_reduce.cpp')
    payloads = {name: (directory / name).read_text() for name in names}
    for name in ('dspark_markov_device.py', 'dspark_markov_score_layout.py', 'dspark_score_layout.py',
                 'dspark_score_layout_io.cpp', 'dspark_score_layout_compute.cpp',
                 'attention_batch.py', 'gdn_multitoken_conv.py'):
        if (scripts / name).read_bytes() != (directory / name).read_bytes():
            raise ValueError('Frozen control differs from qualified source: ' + name)
    payloads.update(hardware_payloads(directory))
    changes = {
        'dspark_request_experiment.py': (
            "    schedule = (('publication', True), ('publication', False), ('publication', False))",
            "    schedule = (('publication', True), ('publication', True), ('publication', False),\n"
            "        ('publication', False), ('publication', False), ('publication', False))"),
        'dspark-target-hardware.py': (
            '            from dspark_request_experiment import run_loaded_requests',
            '            from compact_score_experiment import run_loaded_requests'),
        'run-dspark-hardware.sh': (
            '    -e "QWEN_DSPARK_MODE=$mode"',
            '    -e "QWEN_COMPACT_SCORE_HARDWARE=${QWEN_COMPACT_SCORE_HARDWARE:-0}" \\\n'
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
    shutil.copytree(evidence, scripts / 'compact-score-evidence')
    admission = qualify(scripts, scripts / 'compact-score-evidence')
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
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.evidence, options.manifest)
