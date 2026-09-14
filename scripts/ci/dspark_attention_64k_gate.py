"""Pinned hardware component evidence; does not admit a full-model request."""

import hashlib
import json
from pathlib import Path

from dspark_attention_8k_gate import require_matrix


REPORT_SHA256 = 'bcf40ffe3834298526dee40daefe1dee88bdd8d05f76c5c3f2fab8885d8a7c20'
RUN_ID = 34797353681


def qualify(directory, report_path):
    directory = Path(directory)
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained 64K hardware report required')
    report = json.loads(payload)
    expected_scope = dict(passed=True, closed_cleanly=True, backend='hardware',
        context=65536, capacity=66560, positions=[65536, 66545], proposal_rows=15,
        key_chunk_size=1024, native_padded_keys=67584,
        normalization_mode='sfpu-column-scratch', numerical_tolerances=dict(rtol=.01, atol=.01))
    if any(report.get(name) != value for name, value in expected_scope.items()):
        raise ValueError('Complete exact-scope hardware qualification required')
    sources = report.get('sources', {})
    if not sources or sources != report.get('sources_after'):
        raise ValueError('Stable qualified script dependencies required')
    if not report.get('native_sources') or report['native_sources'] != report.get('native_sources_after'):
        raise ValueError('Stable native runtime evidence required')
    build = report.get('factory_build', {})
    if any(build.get(name) is not True for name in ('passed', 'import_passed', 'factory_enabled')):
        raise ValueError('Successful native factory build required')
    if build.get('backend') != 'hardware' or build.get('experiment') != 'full-context-ladder':
        raise ValueError('Hardware ladder build required')
    for dependencies in (sources, build.get('builders', {}), build.get('ladder_builders', {})):
        if not dependencies:
            raise ValueError('Complete build and script dependency fingerprints required')
        for name, checksum in dependencies.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Qualified 64K dependency changed: ' + name)
    coordinates = set()
    for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
        expected = {(ordinal, case, chip) for ordinal, case in enumerate(cases) for chip in range(2)}
        records = report.get(mode + '_checks', [])
        require_matrix(records, ('ordinal', 'case', 'chip'), expected, 'passed')
        if any(record.get('failed_elements') != 0 or record.get('numerical_close') is not True
                or mode == 'replay' and record.get('replay_exact') is not True for record in records):
            raise ValueError('Every numerical and replay comparison must pass')
        coordinates.update((mode, *coordinate) for coordinate in expected)
    names = ('query', 'history_key', 'history_value', 'query_key', 'query_value', 'mask')
    require_matrix(report.get('input_checks', []), ('mode', 'ordinal', 'case', 'chip', 'name'),
        {(*coordinate, name) for coordinate in coordinates for name in names}, 'exact')
    require_matrix(report.get('layout_checks', []), ('mode', 'ordinal', 'case', 'chip', 'name'),
        {(*coordinate, name) for coordinate in coordinates for name in ('key', 'value')}, 'passed')
    if any(record['sha256'] != record['expected_sha256'] for record in report['layout_checks']):
        raise ValueError('Exact physical layout preservation required')
    require_matrix(report.get('fixture_controls', []), ('name', 'case', 'chip'),
        {(name, 1 if name == 'frontier_update' else 0, chip)
         for name in ('oldest', 'last_proposal', 'gap_poison', 'frontier_update') for chip in range(2)}, 'detected')
    require_matrix(report.get('stale_controls', []), ('chip',), {(0,), (1,)}, 'detected')
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256, context=65536, capacity=66560,
        key_chunk_size=1024, native_padded_keys=67584, source_count=len(sources),
        component_qualified=True, runtime_admitted=False, full_request_qualified=False,
        performance_qualified=False)
