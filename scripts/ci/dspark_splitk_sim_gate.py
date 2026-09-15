"""Pinned parallel split-K simulator evidence; not hardware or serving acceptance."""

import hashlib
import json
from pathlib import Path

from dspark_attention_8k_gate import require_matrix


REPORT_SHA256 = '2690e9bba19aa6a80319edd1a96fb605736bfa8f01dbcd8ec74ddf344addb756'


def verify_sources(directory, sources):
    if not sources:
        raise ValueError('Nonempty qualified source manifest required')
    for name, expected in sources.items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Qualified split-K source changed: ' + name)


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Pinned successful recurrence simulator report required')
    report = json.loads(payload)
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('capacity') != 384
            or report.get('positions') != [128, 369] or report.get('proposal_rows') != 15
            or report.get('numerical_tolerances') != dict(rtol=.01, atol=.01)
            or report.get('splitk_execution_calls') != 3):
        raise ValueError('Complete retained split-K simulator scope required')
    configuration = report['diagnostic_override']
    if (configuration.get('key_chunk_size') != 32 or configuration.get('max_cores_per_head') != 8
            or configuration.get('local_denominator_arithmetic') != 'sfpu-fp32-multiply-add-and-copy'
            or configuration.get('local_numerator_add') != 'sfpu-fp32-broadcast-multiply-add-single-pack'
            or configuration.get('stripe_keys') is not False
            or configuration.get('local_denominator_storage') != 'float32'
            or configuration.get('transfer_statistics_storage') != 'float32'
            or configuration.get('tree_scratch_allocation') != 'one-payload-per-reduction-round'):
        raise ValueError('Qualified parallel precision and scratch configuration required')
    if (report['sources'] != report['sources_after']
            or report['native_sources'] != report['native_sources_after']):
        raise ValueError('Unchanged simulator sources required')
    target = report['target_attention']
    if target.get('passed') is not True or target.get('closed') is not True or target['sources'] != target['sources_after']:
        raise ValueError('Independent native target gate required')
    factory = report['splitk_factory']
    if factory.get('passed') is not True or factory.get('device_access') is not False:
        raise ValueError('Completed simulator factory build required')
    manifests = (report['sources'], report['candidate_sources'], factory['builders'], target['sources'])
    for sources in manifests:
        verify_sources(directory, sources)
    for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
        records = report[mode + '_checks']
        require_matrix(records, ('ordinal', 'case', 'chip'),
            {(ordinal, case, chip) for ordinal, case in enumerate(cases) for chip in range(2)}, 'passed')
        if any(record.get('failed_elements') != 0 or record.get('numerical_close') is not True
                or mode == 'replay' and record.get('replay_exact') is not True for record in records):
            raise ValueError('Exact replay and retained numerical tolerance required')
    return dict(report_sha256=REPORT_SHA256, simulator_qualified=True,
        hardware_qualified=False, full_request_qualified=False, performance_qualified=False,
        serving_qualified=False, configuration=configuration,
        factory_source_before=factory['source_before'], factory_source_after=factory['source_after'])
