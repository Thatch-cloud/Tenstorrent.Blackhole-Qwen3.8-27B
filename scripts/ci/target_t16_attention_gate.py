"""Isolated T16 replay qualification; not full-request or coding-quality proof."""

import hashlib
import json
from pathlib import Path


SOURCES = {'attention_replay.py', 'attention_mask_replay.py', 'attention_mask_replay.cpp',
           'attention_parallel.py', 'attention_fold_dma.py', 'attention_fold_dma.cpp',
           'target-t16-attention-probe.py'}


def validate_request_option(enabled, *, rows, position, remaining, replay, norm_batch,
                            native_sampling, group_rows, short_context):
    if type(enabled) is not bool:
        raise ValueError('Explicit T16 attention selection required')
    if not enabled:
        return
    from dspark_context_selection import request_context
    if request_context() == 8192:
        from target_t16_attention_8k_gate import validate_request_option as validate_8k
        return validate_8k(enabled, rows=rows, position=position, remaining=remaining,
            replay=replay, norm_batch=norm_batch, native_sampling=native_sampling,
            group_rows=group_rows, short_context=short_context)
    if (any(type(value) is not int for value in (rows, position, remaining, group_rows))
            or rows != 16 or position != 4096 or not 1 <= remaining <= 256
            or replay is not True or norm_batch is not True or native_sampling is not True
            or group_rows != 4 or short_context is not False):
        raise ValueError('Qualified T16 experiment requires a bounded 4096..4352 request and native sampling')


def validate(report, directory):
    if report.get('passed') is not True or report.get('closed') is not True or report.get('backend') != 'simulator':
        raise ValueError('Complete closed simulator result required')
    hashes = report.get('sources', {})
    if set(hashes) != SOURCES or hashes != report.get('sources_after'):
        raise ValueError('Complete stable source fingerprints required')
    if any(hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != digest
           for name, digest in hashes.items()):
        raise ValueError('Qualified attention source changed')
    expected = {(4352, start, ticket, chip) for ticket, start in enumerate((4096, 4113, 4336, 4096))
                for chip in range(2)}
    for field, copies in (('checks', 1), ('mask_checks', 2)):
        records = report.get(field, [])
        keys = [(entry.get('capacity'), entry.get('start'), entry.get('ticket'), entry.get('chip'))
                for entry in records]
        if (len(keys) != len(expected) * copies or set(keys) != expected
                or any(keys.count(key) != copies for key in expected)
                or any(entry.get('exact') is not True for entry in records)):
            raise ValueError('Complete exact replay/mask coverage required')
    source_checks = report.get('source_checks', [])
    if (len(source_checks) != 4 or any(entry.get('capacity') != 4352 or entry.get('exact') is not True
            for entry in source_checks) or sorted(entry.get('chip') for entry in source_checks) != [0, 0, 1, 1]):
        raise ValueError('Both KV tensors must remain unchanged on both chips')
    plain = report.get('unpoisoned_replay', [])
    if (len(plain) != 2 or {entry.get('chip') for entry in plain} != {0, 1}
            or any(entry.get('exact') is not True or entry.get('nonfinite') != 0
                   or entry.get('mismatches') != 0 for entry in plain)):
        raise ValueError('Unpoisoned replay control incomplete')
    if report.get('stale_controls') != 2 or report.get('mask_poison_controls') != 8:
        raise ValueError('Stale-input and mask-poison controls incomplete')
    return dict(scope='T16 attention component only', replay_checks=8, mask_checks=16,
                kv_checks=4, full_request_qualified=False)


def qualify(directory):
    from dspark_context_selection import request_context
    if request_context() == 8192:
        from target_t16_attention_8k_gate import qualify as qualify_8k
        return qualify_8k(directory)
    directory = Path(directory)
    path = directory / 'target-t16-attention-simulator.json'
    report = json.loads(path.read_text())
    result = validate(report, directory)
    result['report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
