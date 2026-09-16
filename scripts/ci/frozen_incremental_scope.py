"""Matched frozen 32K requests changing only full-bank versus incremental publication."""

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
from unittest.mock import patch

from history_append_hardware_gate import qualify, REPORT_SHA256
from incremental_history_scope import incremental_history


def validate_records(records, enabled):
    if not enabled:
        if records:
            raise ValueError('Control must not use incremental publication')
        return
    if len(records) != 1:
        raise ValueError('One complete incremental history session required')
    record = records[0]
    if (record.get('initial_position') != 32768 or record.get('capacity') != 33024
            or record.get('failed') is not False or record.get('restored') is not True
            or not 0 < record.get('committed', 0) <= record.get('prepared', 0)
            or record['prepared'] != record['committed'] + record['discarded']
            or not 32 <= record.get('max_touched_rows', 0) <= 96):
        raise ValueError('Complete bounded and restored incremental transaction session required')


@contextmanager
def runtime_scope(directory):
    if (os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Allocated admitted 32K hardware required')
    import full_dspark_request
    import dspark_stable_history
    import dspark_publication_scope
    import gdn_shared_qk_variants

    directory = Path(directory)
    qualify(directory, directory / 'history-append-hardware.json')
    measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route

    def measured(*args, **kwargs):
        enabled = kwargs.get('gdn_shared_qk', False)
        if type(enabled) is not bool or not all(kwargs.get(name) is True for name in (
                'captured_publication', 'fused_t16_mlp', 'target_attention_t16', 'score_layout')):
            raise ValueError('Complete matched fused runtime required')
        kwargs['gdn_shared_qk'] = True
        records = []
        with (incremental_history(dspark_stable_history.StableHistoryKV,
                dspark_publication_scope, records) if enabled else nullcontext()):
            result = measure(*args, **kwargs)
        validate_records(records, enabled)
        result['incremental_history'] = dict(enabled=enabled, records=records,
            report_sha256=REPORT_SHA256 if enabled else None)
        return result

    def route(value, arm):
        audit = value.get('incremental_history', {})
        enabled = arm == 'publication'
        if arm not in ('control', 'publication') or audit.get('enabled') is not enabled:
            raise ValueError('Matched publication arm identity required')
        if enabled and audit.get('report_sha256') != REPORT_SHA256:
            raise ValueError('Qualified writer report identity required')
        validate_records(audit.get('records', []), enabled)
        original_route(value, 'publication')

    with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
            patch.object(gdn_shared_qk_variants, 'validate_route', route):
        yield
