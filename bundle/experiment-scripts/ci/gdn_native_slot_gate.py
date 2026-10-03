"""Independent admission for synthetic direct-state math and zero-publication evidence."""

import hashlib
import json
from pathlib import Path


SOURCES = {
    'projected': ('gdn-native-projected-probe.py', 'gdn_native_slot_projected.py', 'gdn_native_slot_recurrence.py',
        'gdn_native_slot_windows.py', 'gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'gdn_batched_conv.py',
        'gdn_vsplit.py', 'gdn_vsplit_norm_batch.py', 'gdn_vsplit_prefetch.py', 'gdn_multitoken.py',
        'gdn_multitoken_conv.py', 'attention_batch.py', 'feature_projection.py'),
    'zero': ('gdn-native-zero-probe.py', 'gdn_native_slot_publication.py', 'gdn_commit_dma.py',
        'gdn_commit_dma.cpp', 'gdn_state_copy.py', 'gdn_state_copy.cpp', 'gdn_publication_fixture.py',
        'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py'),
}


def matrix(records, fields, expected):
    actual = {tuple(record.get(field) for field in fields) for record in records}
    if len(records) != len(expected) or actual != expected or any(record.get('exact') is not True for record in records):
        raise ValueError('Every declared native-slot comparison must be present exactly once and pass')


def qualify(directory):
    directory = Path(directory)
    result = {}
    for name, sources in SOURCES.items():
        path = directory / f'gdn-native-{name}-simulator.json'
        raw = path.read_bytes()
        report = json.loads(raw)
        if (path.with_suffix('.exit-status').read_text().strip() != '0'
                or report.get('passed') is not True or report.get('closed_cleanly') is not True
                or report.get('stage') != 'complete' or report.get('backend') != 'simulator'):
            raise ValueError('Complete clean native-slot simulator report required')
        fingerprints = {source: hashlib.sha256((directory / source).read_bytes()).hexdigest() for source in sources}
        if report.get('sources') != fingerprints or report.get('sources_after') != fingerprints:
            raise ValueError('Native-slot simulator source changed or missing')
        if name == 'projected':
            modes = ((0, 'eager'), *((seed, mode) for seed in range(3) for mode in ('control_replay', 'native_replay')))
            matrix(report.get('checks', []), ('seed', 'mode', 'operand', 'chip'),
                {(seed, mode, operand, chip) for seed, mode in modes for operand in range(6) for chip in range(2)})
            matrix(report.get('immutable_checks', []), ('seed', 'mode', 'operand', 'chip'),
                {(seed, mode, operand, chip) for seed, mode in ((0, 'eager'), (0, 'replay'), (1, 'replay'), (2, 'replay'))
                    for operand in range(18) for chip in range(2)})
        else:
            if report.get('prefix') != 0 or report.get('layers') != 1:
                raise ValueError('Declared zero-publication fixture required')
            matrix(report.get('checks', []), ('pattern', 'mode', 'operand', 'chip'),
                {(pattern, mode, operand, chip) for pattern, mode in ((0, 'eager'), (1, 'replay'), (0, 'replay'))
                    for operand in range(20) for chip in range(2)})
        result[name] = hashlib.sha256(raw).hexdigest()
    return dict(reports=result, scope='Synthetic T16 composition and single-layer zero publication; learned hardware audit still required')
