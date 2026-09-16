"""Component admission for the explicitly selected offline 32K runtime candidate."""

import hashlib
import os
from pathlib import Path

from frozen_combined_gate import qualify as qualify_components, REPORTS
from frozen_context_geometry import selected_geometry


REPORT_SHA256 = REPORTS['draft-numerical.json']


def validate_target_option(enabled, *, rows, position, remaining, replay, norm_batch,
                           native_sampling, group_rows, short_context):
    if type(enabled) is not bool:
        raise ValueError('Explicit T16 attention selection required')
    if not enabled:
        return
    if (any(type(value) is not int for value in (rows, position, remaining, group_rows))
            or rows != 16 or position != 32768 or not 1 <= remaining <= 256
            or replay is not True or norm_batch is not True or native_sampling is not True
            or group_rows != 4 or short_context is not False):
        raise ValueError('Qualified 32K T16 replay with four-row groups and native sampling required')


def qualify_target(directory):
    evidence = qualify(directory)
    return dict(evidence['target'], report_sha256=REPORTS['target-replay.json'])


def qualify(directory):
    if (os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or selected_geometry()['context'] != 32768
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Explicit allocated offline 32K combined candidate required')
    directory = Path(directory)
    evidence = directory / 'frozen-evidence'
    draft = evidence / 'draft/scripts/ci'
    target = evidence / 'target/scripts/ci'
    result = qualify_components(evidence, draft_sources=draft, target_sources=target, context=32768)
    from frozen_combined_gate import load_reports
    reports = load_reports(evidence)
    excluded = {'frozen_probe_evidence.py', 'dspark_attention_8k_gate.py',
        'dspark_attention_value_diagnostics.py'}
    expected = {name: checksum for name, checksum in reports['draft-numerical.json']['sources'].items()
        if not name.endswith('-probe.py') and not name.startswith('../') and name not in excluded}
    for name, checksum in reports['target-replay.json']['sources'].items():
        if name.endswith('-probe.py'):
            continue
        if name in expected and expected[name] != checksum:
            raise ValueError('Component evidence disagrees on shared source: ' + name)
        expected[name] = checksum
    for name, checksum in expected.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Combined runtime component source differs: ' + name)
    result.update(report_sha256=REPORTS['draft-numerical.json'], runtime_component_sources=expected)
    return result
