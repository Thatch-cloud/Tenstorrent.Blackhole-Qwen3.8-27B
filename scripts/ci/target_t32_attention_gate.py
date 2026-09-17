"""Isolated T32 replay qualification; not full-request or coding-quality proof."""

import hashlib
import json
import os
from pathlib import Path


SOURCES = {'attention_replay.py', 'attention_mask_replay.py', 'attention_mask_replay.cpp',
           'attention_parallel.py', 'attention_fold_dma.py', 'attention_fold_dma.cpp',
           'target-t32-attention-probe.py'}

RETAINED_REPORT = '774a53bf54fcbb7fe5c02fb1f358be63c9fcc0254679f19543232d4079429b1b'
MASK_DELTA = ('7841495a15ee090aae7b78edc118ba0de2967bb3ad72b843d53251a091435749',
              '3e431742e35a2b94b4a02a60fa334a93a44a471eaefcacd25e52fbafdf03361f')


def qualify_request(path, *, position, remaining, directory=None, hardware_mask_compatibility=False):
    if (type(position) is not int or position != 4096 or type(remaining) is not int
            or not 1 <= remaining <= 224):
        raise ValueError('T32 replay request must remain within its 4096-to-4320 simulator window')
    path = Path(path)
    if path.with_suffix('.exit-status').read_text().strip() != '0':
        raise ValueError('Clean T32 attention simulator exit required')
    raw = path.read_bytes()
    if type(hardware_mask_compatibility) is not bool:
        raise ValueError('Explicit mask compatibility policy required')
    report = json.loads(raw)
    delta = None
    if hardware_mask_compatibility:
        if (hashlib.sha256(raw).hexdigest() != RETAINED_REPORT
                or any(os.environ.get(name) != '1' for name in
                    ('QWEN_T32_FUSED_SCORE_HARDWARE', 'QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))
                or any(os.environ.get(name) == '1' for name in ('QWEN_CONTEXT_LADDER_SIM', 'QWEN_SIM_ONLY'))
                or any(os.environ.get(name) for name in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH'))
                or report.get('sources', {}).get('attention_mask_replay.py') != MASK_DELTA[0]
                or report.get('sources') != report.get('sources_after')):
            raise ValueError('Pinned mask evidence and disabled simulator branch on allocated hardware required')
        delta = dict(source='attention_mask_replay.py', recorded=MASK_DELTA[0], current=MASK_DELTA[1],
            reason='Only added simulator-context branch; disabled in this hardware request')
        report['sources']['attention_mask_replay.py'] = MASK_DELTA[1]
        report['sources_after']['attention_mask_replay.py'] = MASK_DELTA[1]
    evidence = validate(report, Path(__file__).parent if directory is None else directory)
    if delta is not None:
        evidence['reviewed_inactive_source_delta'] = delta
    return dict(evidence, report_sha256=hashlib.sha256(raw).hexdigest(), position=position,
        remaining=remaining, capacity=4352)


def validate(report, directory):
    if report.get('rows') != 32 or report.get('passed') is not True or report.get('closed') is not True or report.get('backend') != 'simulator':
        raise ValueError('Complete closed simulator result required')
    hashes = report.get('sources', {})
    if set(hashes) != SOURCES or hashes != report.get('sources_after'):
        raise ValueError('Complete stable source fingerprints required')
    if any(hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != digest
           for name, digest in hashes.items()):
        raise ValueError('Qualified attention source changed')
    expected = {(4352, start, ticket, chip) for ticket, start in enumerate((4096, 4113, 4320, 4096))
                for chip in range(2)}
    for field, copies in (('checks', 1), ('mask_checks', 3)):
        records = report.get(field, [])
        keys = [(entry.get('capacity'), entry.get('start'), entry.get('ticket'), entry.get('chip'))
                for entry in records]
        if (len(keys) != len(expected) * copies or set(keys) != expected
                or any(keys.count(key) != copies for key in expected)
                or any(entry.get('exact') is not True for entry in records)):
            raise ValueError('Complete exact replay/mask coverage required')
    source_checks = report.get('source_checks', [])
    mask_keys = [(entry.get('capacity'), entry.get('start'), entry.get('ticket'), entry.get('chip'),
                  entry.get('bundle')) for entry in report['mask_checks']]
    if set(mask_keys) != {(*key, bundle) for key in expected for bundle in range(3)}:
        raise ValueError('Every distinct T32 mask bundle must be checked on each chip')
    if (len(source_checks) != 4 or any(entry.get('capacity') != 4352 or entry.get('exact') is not True
            for entry in source_checks) or sorted(entry.get('chip') for entry in source_checks) != [0, 0, 1, 1]):
        raise ValueError('Both KV tensors must remain unchanged on both chips')
    plain = report.get('unpoisoned_replay', [])
    if (len(plain) != 2 or {entry.get('chip') for entry in plain} != {0, 1}
            or any(entry.get('exact') is not True or entry.get('nonfinite') != 0
                   or entry.get('mismatches') != 0 for entry in plain)):
        raise ValueError('Unpoisoned replay control incomplete')
    if report.get('stale_controls') != 2 or report.get('mask_poison_controls') != 12:
        raise ValueError('Stale-input and mask-poison controls incomplete')
    return dict(scope='T32 attention component only', replay_checks=8, mask_checks=24,
                kv_checks=4, full_request_qualified=False)
