"""Retained 256-key synthetic attention qualification, not full-request or speed proof."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = '2baa47c721607fac512b72413024c15a510ef475954cab976ee6d59222ab61af'


def require_matrix(records, fields, expected, flag):
    keys = [tuple(record.get(field) for field in fields) for record in records]
    if len(keys) != len(expected) or set(keys) != expected or any(record.get(flag) is not True for record in records):
        raise ValueError('Complete unique passing numerical/ownership/control matrix required')


def qualify(directory, report_path=None):
    directory = Path(directory)
    path = Path(report_path) if report_path is not None else directory / 'dspark-native-8k-attention.json'
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Retained successful 256-key simulator report required')
    report = json.loads(payload)
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('capacity') != 8448
            or report.get('positions') != [8192, 8433] or report.get('proposal_rows') != 15
            or report.get('key_chunk_size') != 256 or report.get('native_padded_keys') != 8704
            or report.get('numerical_tolerances') != dict(rtol=.01, atol=.01)):
        raise ValueError('Complete exact-scope simulator qualification required')
    sources = report.get('sources', {})
    if not sources or sources != report.get('sources_after') or report.get('native_sources') != report.get('native_sources_after'):
        raise ValueError('Stable source/runtime evidence required')
    for name, checksum in sources.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Qualified attention dependency changed: ' + name)
    build = report.get('factory_build', {})
    if (build.get('passed') is not True or build.get('import_passed') is not True
            or build.get('factory_enabled') is not True or build.get('precision_variant') != 'stats-only'):
        raise ValueError('Qualified FP32-statistics factory build required')
    for name, checksum in build.get('builders', {}).items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Qualified factory builder changed: ' + name)
    coordinates = set()
    for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
        expected = {(ordinal, case, chip) for ordinal, case in enumerate(cases) for chip in range(2)}
        records = report.get(mode + '_checks', [])
        require_matrix(records, ('ordinal', 'case', 'chip'), expected, 'passed')
        if any(record.get('failed_elements') != 0 or record.get('numerical_close') is not True
                or mode == 'replay' and record.get('replay_exact') is not True for record in records):
            raise ValueError('Every numerical comparison and changed-input replay must pass')
        coordinates.update((mode, *coordinate) for coordinate in expected)
    names = ('query', 'history_key', 'history_value', 'query_key', 'query_value', 'mask')
    require_matrix(report.get('input_checks', []), ('mode', 'ordinal', 'case', 'chip', 'name'),
        {(*coordinate, name) for coordinate in coordinates for name in names}, 'exact')
    require_matrix(report.get('layout_checks', []), ('mode', 'ordinal', 'case', 'chip', 'name'),
        {(*coordinate, name) for coordinate in coordinates for name in ('key', 'value')}, 'passed')
    if any(record['sha256'] != record['expected_sha256'] for record in report['layout_checks']):
        raise ValueError('Exact preserved physical key/value layout required')
    require_matrix(report.get('fixture_controls', []), ('name', 'case', 'chip'),
        {(name, 1 if name == 'frontier_update' else 0, chip)
         for name in ('oldest', 'last_proposal', 'gap_poison', 'frontier_update') for chip in range(2)}, 'detected')
    require_matrix(report.get('stale_controls', []), ('chip',), {(0,), (1,)}, 'detected')
    return dict(report_sha256=REPORT_SHA256, source_count=len(sources), eager_checks=4,
        replay_checks=4, input_checks=48, layout_checks=16, fixture_controls=8, stale_controls=2,
        capacity=8448, key_chunk_size=256, native_padded_keys=8704,
        full_request_qualified=False, performance_qualified=False)
