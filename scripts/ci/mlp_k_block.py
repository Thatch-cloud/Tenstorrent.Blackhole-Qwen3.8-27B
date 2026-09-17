"""Stage a T16 K32 fused MLP candidate; numerical qualification required."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import REVISION, replace_once
from frozen_mlp_buffer_trial import adapt_probe


CHANGES = {
    'fused_1d.py': (
        ('k_block=8,', 'k_block=32,'),
        ('cb(0, ttnn.bfloat16, 2048, 16, all_cores)', 'cb(0, ttnn.bfloat16, 2048, 64, all_cores)'),
        ('cb(1, ttnn.bfloat4_b, 576, 32 * pairs_per_worker, workers)',
         'cb(1, ttnn.bfloat4_b, 576, 128 * pairs_per_worker, workers)'),
        ('compile_time_args=[8, 1, 8, 8, pairs_per_worker, 16 * pairs_per_worker, 2 * pairs_per_worker,',
         'compile_time_args=[32, 1, 32, 32, pairs_per_worker, 64 * pairs_per_worker, 2 * pairs_per_worker,'),
        ('20, 1, 1, 1, 2, 2, 1, 2 * pairs_per_worker, 0, 0, 0],',
         '5, 1, 1, 1, 2, 2, 1, 2 * pairs_per_worker, 0, 0, 0],')),
    'fused_1d_input.cpp': (
        ('block < 20', 'block < 5'),
        ('cb_reserve_back(0, 8)', 'cb_reserve_back(0, 32)'),
        ('tile < 8', 'tile < 32'),
        ('block * 8 + tile', 'block * 32 + tile'),
        ('target, 8 * 2048', 'target, 32 * 2048'),
        ('cb_push_back(0, 8)', 'cb_push_back(0, 32)'),
        ('cb_wait_front(0, 8)', 'cb_wait_front(0, 32)'),
        ('cb_pop_front(0, 8)', 'cb_pop_front(0, 32)')),
    'fused_1d_weights.cpp': (
        ('block < 20', 'block < 5'),
        ('cb_reserve_back(1, 16 * pairs_per_worker)', 'cb_reserve_back(1, 64 * pairs_per_worker)'),
        ('inner < 8', 'inner < 32'),
        ('(block * 8 + inner) * 544', '(block * 32 + inner) * 544'),
        ('cb_push_back(1, 16 * pairs_per_worker)', 'cb_push_back(1, 64 * pairs_per_worker)')),
}


def transform(name, source):
    for before, after in CHANGES[name]:
        source = replace_once(source, before, after)
    if name.endswith('.py'):
        compile(source, name, 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh K-block staging required')
    scripts = options.checkout / 'scripts/ci'
    originals = {}
    for name in (*CHANGES, 'fused-batch-probe.py'):
        source = subprocess.check_output(['git', '-C', str(options.checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != source:
            raise ValueError('Exact frozen source required: ' + name)
        originals[name] = source
    payloads = {name: transform(name, originals[name]) for name in CHANGES}
    payloads['fused-batch-probe.py'] = replace_once(adapt_probe(originals['fused-batch-probe.py']),
        'T16 buffering only; other row widths and performance unqualified',
        'T16 K32 reduction blocks; no performance qualification')
    payloads['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        k_block=32, reduction_blocks=5, buffer_blocks=2,
        extra_input_bytes_per_core=48 * 2048, extra_weight_bytes_per_worker=96 * 3 * 576,
        compute_source_changed=False, accumulation_grouping_changed=True,
        weight_layout_changed=False, simulator_qualified=False, hardware_qualified=False,
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
