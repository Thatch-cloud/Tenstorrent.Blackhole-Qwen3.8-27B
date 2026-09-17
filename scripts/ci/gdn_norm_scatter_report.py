"""Validate isolated norm evidence without claiming request qualification."""

import hashlib
from pathlib import Path


def validate(report, directory):
    if not (report.get('passed') is True and report.get('closed') is True
            and report.get('backend') == 'simulator' and report.get('output_poisoned') is True):
        raise ValueError('Complete poisoned simulator evidence required')
    names = {'gdn-norm-scatter-probe.py', 'gdn_norm_scatter.py', 'gdn_vsplit_norm_batch.py',
             'gdn_vsplit.py', 'attention_batch.py'}
    before = report.get('before', {})
    if set(before) != names or before != report.get('after'):
        raise ValueError('Source provenance incomplete')
    if any(hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != digest
           for name, digest in before.items()):
        raise ValueError('Current source differs from simulator evidence')
    for field, replays in (('checks', (None,)), ('replay_checks', (0, 1))):
        records = report.get(field, [])
        expected = {(rows, case, chip, replay) for rows in (1, 16, 32)
                    for case in range(2) for chip in range(2) for replay in replays}
        actual = [(entry.get('rows'), entry.get('case'), entry.get('chip'), entry.get('replay'))
                  for entry in records]
        flags = ('exact', 'unchanged', 'padding_zero', 'written') + (
            ('finite',) if field == 'checks' else ('stale_detected',))
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Missing or duplicated shape/chip/replay evidence')
        if any(entry.get(flag) is not True for entry in records for flag in flags):
            raise ValueError('Numerical or storage control failed')
    poison = report.get('poison_checks', [])
    if len(poison) != 30 or any(entry.get('verified') is not True for entry in poison):
        raise ValueError('Output poisoning evidence incomplete')
    for rows in (1, 16, 32):
        for case in range(2):
            if sum(entry.get('rows') == rows and entry.get('case') == case for entry in poison) != 5:
                raise ValueError('Output poison coverage incomplete')
    return {'eager_checks': 12, 'replay_checks': 24, 'poison_checks': 30,
            'scope': 'isolated simulator norm only; no hardware speed qualification'}
