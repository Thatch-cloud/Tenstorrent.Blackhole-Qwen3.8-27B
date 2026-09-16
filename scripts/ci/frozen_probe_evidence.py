"""Join complete same-context probe shards; never qualify model throughput."""

from frozen_context_geometry import geometry
from dspark_attention_8k_gate import require_matrix


def validate_pair(numerical, diagnostics, context):
    shape = geometry(context)
    if numerical.get('passed') is not True or diagnostics.get('diagnostics_complete') is not True:
        raise ValueError('Completed numerical and diagnostic shards required')
    for report, part in ((numerical, 'numerical'), (diagnostics, 'diagnostics')):
        if (report.get('probe_part') != part or report.get('closed_cleanly') is not True
                or report.get('backend') != 'simulator' or report.get('error') or report.get('cleanup_error')
                or report.get('capacity') != shape['capacity']
                or report.get('positions') != list(shape['positions'])
                or report.get('proposal_rows') != 15 or report.get('key_chunk_size') != 256
                or report.get('native_padded_keys') != shape['padded_keys']
                or report.get('numerical_tolerances') != dict(rtol=.01, atol=.01)):
            raise ValueError('Closed exact-context simulator evidence required')
        for field in ('sources', 'native_sources'):
            if not report.get(field) or report[field] != report.get(field + '_after'):
                raise ValueError('Stable source and native identities required')
        build = report.get('factory_build', {})
        if (build.get('passed') is not True or build.get('import_passed') is not True
                or build.get('factory_enabled') is not True or build.get('precision_variant') != 'stats-only'):
            raise ValueError('Successful precise statistics factory audit required')
    for field in ('sources', 'native_sources', 'factory_build'):
        if numerical[field] != diagnostics[field]:
            raise ValueError('Shards must use the same source and factory identities')
    coordinates = set()
    for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
        expected = {(ordinal, case, chip) for ordinal, case in enumerate(cases) for chip in range(2)}
        records = numerical.get(mode + '_checks', [])
        require_matrix(records, ('ordinal', 'case', 'chip'), expected, 'passed')
        if any(record.get('failed_elements') != 0 or record.get('numerical_close') is not True
                or mode == 'replay' and record.get('replay_exact') is not True for record in records):
            raise ValueError('Exact replay and numerical tolerance required')
        coordinates.update((mode, *coordinate) for coordinate in expected)
    require_matrix(numerical.get('input_checks', []), ('mode', 'ordinal', 'case', 'chip', 'name'),
        {(*coordinate, name) for coordinate in coordinates
            for name in ('query', 'history_key', 'history_value', 'query_key', 'query_value', 'mask')}, 'exact')
    require_matrix(numerical.get('layout_checks', []), ('mode', 'ordinal', 'case', 'chip', 'name'),
        {(*coordinate, name) for coordinate in coordinates for name in ('key', 'value')}, 'passed')
    if any(not record.get('sha256') or record['sha256'] != record.get('expected_sha256')
            for record in numerical['layout_checks']):
        raise ValueError('Exact retained physical layout required')
    require_matrix(numerical.get('fixture_controls', []), ('name', 'case', 'chip'),
        {(name, 1 if name == 'frontier_update' else 0, chip)
            for name in ('oldest', 'last_proposal', 'gap_poison', 'frontier_update') for chip in range(2)}, 'detected')
    require_matrix(numerical.get('stale_controls', []), ('chip',), {(0,), (1,)}, 'detected')
    records = diagnostics.get('value_diagnostics', [])
    require_matrix(records, ('kind', 'chip'),
        {(kind, chip) for kind in ('constant', 'oldest', 'last_proposal') for chip in range(2)}, 'finite')
    if any(record.get('failed_elements') != 0 for record in records):
        raise ValueError('Value diagnostic numerical discrepancies remain')
    return dict(context=context, capacity=shape['capacity'], complete_probe_coverage=True,
        source_files_verified=False, artifact_provenance_verified=False,
        full_request_qualified=False, performance_qualified=False)
