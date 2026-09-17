"""Unqualified bank-staggered issue order; no weight layout or arithmetic changes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_mlp_buffer_trial import adapt_probe
from frozen_recipe_context import REVISION, replace_once


ANCHOR = '            for (uint32_t column = 0; column < 2 * pairs_per_worker; ++column) {\n'
LOOP = '''            for (uint32_t issue = 0; issue < 2 * pairs_per_worker; ++issue) {
                const uint32_t shifted = issue + read_phase;
                const uint32_t column = shifted >= 2 * pairs_per_worker ? shifted - 2 * pairs_per_worker : shifted;
'''
SETUP = '''    static_assert(pairs_per_worker == 3);
    const uint32_t read_phase = ((first_pair / pairs_per_worker) / 4) % (2 * pairs_per_worker);
'''


def transform(source):
    result = replace_once(source, ANCHOR, LOOP)
    result = replace_once(result, '    for (uint32_t block = 0; block < 20; ++block) {\n',
        SETUP + '    for (uint32_t block = 0; block < 20; ++block) {\n')
    if remove(result) != source:
        raise ValueError('Read-order change modified unrelated operations')
    return result


def remove(source):
    return replace_once(replace_once(source, SETUP, ''), LOOP, ANCHOR)


def accesses(worker, block, staggered):
    if (type(worker) is not int or not 0 <= worker < 91 or type(block) is not int
            or not 0 <= block < 20 or type(staggered) is not bool):
        raise ValueError('Explicit T16 91-worker K-block geometry required')
    first_pair = 3 * worker
    valid_pairs = min(3, 272 - first_pair)
    phase = (worker // 4) % 6 if staggered else 0
    return [dict(page=(block * 8 + inner) * 544 + first_pair * 2 + column
            if column < 2 * valid_pairs else None, destination_tile=inner * 6 + column)
        for inner in range(8) for issue in range(6) for column in ((issue + phase) % 6,)]


def issue_model(staggered):
    accesses_by_worker = [accesses(worker, 0, staggered)[:6] for worker in range(91)]
    return [dict(Counter(worker[issue]['page'] % 8 for worker in accesses_by_worker
        if worker[issue]['page'] is not None)) for issue in range(6)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh read-order manifest required')
    scripts = options.checkout / 'scripts/ci'
    sources = {}
    for name in ('fused_1d_weights.cpp', 'fused-batch-probe.py'):
        source = subprocess.check_output(['git', '-C', str(options.checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != source:
            raise ValueError('Exact frozen source required: ' + name)
        sources[name] = source
    candidate = transform(sources['fused_1d_weights.cpp'])
    probe = adapt_probe(sources['fused-batch-probe.py'])
    probe = replace_once(probe, "'T16 buffering only; other row widths and performance unqualified'",
        "'T16 staggered weight-read order only; no performance qualification'")
    suite = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    payloads = {'fused_1d_weights.cpp': candidate, 'fused-batch-probe.py': probe, 'simulator-suite.sh': suite}
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        synchronized_eight_bank_model=dict(baseline=issue_model(False), candidate=issue_model(True)),
        workers=91, pairs_per_worker=3, phase_group=4, compute_changed=False, buffers_changed=False,
        weight_layout_changed=False, simulator_qualified=False, hardware_qualified=False,
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
