"""Pinned full-history split-K component evidence; not request or TG acceptance."""

import hashlib
import json
from pathlib import Path

from dspark_attention_8k_gate import require_matrix
from dspark_splitk_sim_gate import qualify as qualify_simulator, verify_sources


RUN_ID = 34918350712
REPORT_SHA256 = 'a25f7e3e905f74e1f1e671c4a5976a7a3c866219829ea52835045d98232b897c'


def validate_matrix(report):
    coordinates = set()
    for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
        expected = {(ordinal, case, chip) for ordinal, case in enumerate(cases) for chip in range(2)}
        records = report.get(mode + '_checks', [])
        require_matrix(records, ('ordinal', 'case', 'chip'), expected, 'passed')
        if any(record.get('failed_elements') != 0 or record.get('numerical_close') is not True
                or mode == 'replay' and record.get('replay_exact') is not True for record in records):
            raise ValueError('Every split-K numerical and replay check must pass')
        coordinates.update((mode, *coordinate) for coordinate in expected)
    names = ('query', 'history_key', 'history_value', 'query_key', 'query_value', 'mask')
    require_matrix(report.get('input_checks', []), ('mode', 'ordinal', 'case', 'chip', 'name'),
        {(*coordinate, name) for coordinate in coordinates for name in names}, 'exact')
    require_matrix(report.get('layout_checks', []), ('mode', 'ordinal', 'case', 'chip', 'name'),
        {(*coordinate, name) for coordinate in coordinates for name in ('key', 'value')}, 'passed')
    if any(record['sha256'] != record['expected_sha256'] for record in report['layout_checks']):
        raise ValueError('Exact split-K physical layout required')
    require_matrix(report.get('fixture_controls', []), ('name', 'case', 'chip'),
        {(name, 1 if name == 'frontier_update' else 0, chip)
         for name in ('oldest', 'last_proposal', 'gap_poison', 'frontier_update') for chip in range(2)}, 'detected')
    require_matrix(report.get('stale_controls', []), ('chip',), {(0,), (1,)}, 'detected')


def qualify(directory, report_path, simulator_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained split-K hardware report required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='hardware',
        capacity=66560, positions=[65536, 66545], proposal_rows=15,
        key_chunk_size=256, max_cores_per_head=8, native_padded_keys=67584,
        numerical_tolerances=dict(rtol=.01, atol=.01), splitk_execution_calls=10,
        expected_execution_calls=10, local_denominator_arithmetic='sfpu-fp32-multiply-add-and-copy',
        tree_denominator_arithmetic='sfpu-fp32-two-products-add-and-transport',
        local_numerator_add='native-fpu', correction_factor_rounding='explicit-fp32-to-bf16-rne')
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError('Complete qualified split-K hardware scope required')
    for name in ('sources', 'native_sources'):
        if not report.get(name) or report[name] != report.get(name + '_after'):
            raise ValueError('Stable split-K source and native runtime required')
    simulator = qualify_simulator(directory, simulator_path)
    build = report['splitk_factory']
    if (build.get('passed') is not True or build.get('import_passed') is not True
            or build.get('backend') != 'hardware'
            or build.get('simulator_report_sha256') != simulator['report_sha256']
            or build.get('source_after') != simulator['factory_source_after']):
        raise ValueError('Matching simulator-qualified hardware build required')
    binaries = build.get('binaries', {})
    if (set(binaries) != {'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'}
            or len(set(binaries.values())) != 1
            or any(report['native_sources'].get(name) != checksum for name, checksum in binaries.items())):
        raise ValueError('Both qualified hardware libraries must match')
    for manifest in (report['sources'], report['wrapper_sources'], build['builders']):
        verify_sources(directory, manifest)
    validate_matrix(report)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256,
        simulator_report_sha256=simulator['report_sha256'], context=65536, capacity=66560,
        key_chunk_size=256, max_cores_per_head=8, native_padded_keys=67584,
        kernel=report['splitk_kernel'], factory_source_before=simulator['factory_source_before'],
        factory_source_after=simulator['factory_source_after'],
        component_qualified=True, runtime_admitted=False, full_request_qualified=False,
        performance_qualified=False, serving_qualified=False)
