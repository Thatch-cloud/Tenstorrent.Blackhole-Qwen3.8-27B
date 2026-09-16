"""Matched tail-first assembly on the unchanged qualified norm/incremental runtime."""

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
from unittest.mock import patch

from frozen_draft_tail import append_queries
from frozen_draft_tail_gate import qualify_hardware, HARDWARE_SHA256
from frozen_gdn_norm_scope import runtime_scope as norm_scope


@contextmanager
def runtime_scope(directory):
    with norm_scope(directory):
        with comparison_scope(directory) as evidence:
            yield evidence


@contextmanager
def comparison_scope(directory):
    if os.environ.get('QWEN_COMBINED_TRACE_PROFILE', '0') != '0':
        raise ValueError('Unprofiled matched combined requests required')
    import full_dspark_request
    import dspark_native_cached_layer
    import gdn_shared_qk_variants

    directory = Path(directory)
    report_path = directory / 'frozen-draft-tail-hardware.json'
    evidence = qualify_hardware(directory, report_path)
    measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route

    def measured(*args, **kwargs):
        enabled = kwargs.get('gdn_shared_qk', False)
        if type(enabled) is not bool:
            raise ValueError('Explicit draft-tail comparison arm required')
        kwargs['gdn_shared_qk'] = True
        calls = []

        def append(*operands, **options):
            if options.get('position') != 33024 or options.get('proposals') != 15:
                raise ValueError('Only the admitted 32K fifteen-query history geometry may use this hook')
            result = append_queries(*operands, **options)
            calls.append(True)
            return result

        with patch.object(dspark_native_cached_layer, 'append_queries', append) if enabled else nullcontext():
            result = measure(*args, **kwargs)
        if enabled and (not calls or len(calls) % 10):
            raise ValueError('Every K/V assembly across all five draft layers must use the candidate')
        result['draft_tail'] = dict(enabled=enabled, calls=len(calls),
            report_sha256=HARDWARE_SHA256 if enabled else None)
        return result

    def route(value, arm):
        enabled = arm == 'publication'
        audit = value.get('draft_tail', {})
        if (arm not in ('control', 'publication') or audit.get('enabled') is not enabled
                or audit.get('report_sha256') != (HARDWARE_SHA256 if enabled else None)
                or type(audit.get('calls')) is not int
                or (enabled and (audit['calls'] <= 0 or audit['calls'] % 10))
                or (not enabled and audit['calls'] != 0)):
            raise ValueError('Matched executed draft-tail arm identity required')
        original_route(value, 'publication')

    with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
            patch.object(gdn_shared_qk_variants, 'validate_route', route):
        try:
            yield evidence
        finally:
            if qualify_hardware(directory, report_path) != evidence:
                raise ValueError('Qualified draft-tail sources changed during the request')
