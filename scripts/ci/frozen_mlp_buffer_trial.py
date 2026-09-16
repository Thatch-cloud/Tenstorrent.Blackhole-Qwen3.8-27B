"""Unqualified four-block fused-MLP buffering candidate; no serving integration."""

import hashlib
import argparse
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import replace_once
from frozen_recipe_context import REVISION


def transform(source):
    result = replace_once(source,
        'cb(0, ttnn.bfloat16, 2048, 16, all_cores)',
        'cb(0, ttnn.bfloat16, 2048, 32, all_cores)')
    result = replace_once(result,
        'cb(1, ttnn.bfloat4_b, 576, 32 * pairs_per_worker, workers)',
        'cb(1, ttnn.bfloat4_b, 576, 64 * pairs_per_worker, workers)')
    result = replace_once(result,
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,',
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,\n'
        '                             input_buffer_blocks=4, weight_buffer_blocks=4,')
    compile(result, 'fused_1d.py', 'exec')
    return result


def manifest(source):
    candidate = transform(source)
    return dict(control_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        token_rows=16, pairs_per_worker=3, input_buffer_blocks=4, weight_buffer_blocks=4,
        extra_input_bytes_per_core=16 * 2048,
        extra_weight_bytes_per_worker=32 * 3 * 576,
        compute_changed=False, readers_changed=False,
        simulator_qualified=False, hardware_qualified=False, performance_qualified=False)


def adapt_probe(source):
    changes = (
        ('    trace_rows = (1, 8, 16, 32) if options.trace_t16 else (1, 8, 32)',
            "    if options.hardware or options.timing or not (options.target_math and options.trace_t16\n"
            "            and options.trace_replay and options.device_weight_check):\n"
            "        parser.error('T16 simulator-only buffering qualification requires all correctness gates')\n"
            '    trace_rows = (16,)'),
        ('for rows in (1, 2, 4, 8, 16, 32):', 'for rows in (16,):'),
        ("len(report['checks']) != 12", "len(report['checks']) != 2"),
        ('All six widths and both chips required', 'T16 and both chips required'),
        ("    report['passed'] = True", "    report['buffer_candidate_sha256'] = hashlib.sha256(\n"
            "        Path(__file__).with_name('fused_1d.py').read_bytes()).hexdigest()\n"
            "    report['qualification_scope'] = 'T16 buffering only; other row widths and performance unqualified'\n"
            "    report['passed'] = True"))
    for before, after in changes:
        source = replace_once(source, before, after)
    compile(source, 'fused-batch-probe.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    sources = {}
    for name in ('fused_1d.py', 'fused-batch-probe.py'):
        original = subprocess.check_output(['git', '-C', str(options.checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode()
        if (scripts / name).read_text() != original.replace('\r\n', '\n'):
            raise ValueError('Exact historical source required: ' + name)
        sources[name] = original.replace('\r\n', '\n')
    if options.manifest.exists():
        raise ValueError('Fresh candidate manifest required')
    candidate = transform(sources['fused_1d.py'])
    probe = adapt_probe(sources['fused-batch-probe.py'])
    suite = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    evidence = manifest(sources['fused_1d.py'])
    for name, source in (('fused_1d.py', candidate), ('fused-batch-probe.py', probe), ('simulator-suite.sh', suite)):
        (scripts / name).write_bytes(source.encode())
    evidence['probe_sha256'] = hashlib.sha256(probe.encode()).hexdigest()
    options.manifest.write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
