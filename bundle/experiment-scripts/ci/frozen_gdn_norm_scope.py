"""Matched norm prefetch with qualified incremental publication in both arms."""

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
from unittest.mock import patch

from frozen_gdn_norm_gate import REPORT_SHA256, qualify
from frozen_gdn_norm_prefetch import build
from frozen_incremental_scope import runtime_scope as incremental_scope


@contextmanager
def runtime_scope(directory):
    with incremental_scope(directory):
        with comparison_scope(directory) as evidence:
            yield evidence


@contextmanager
def comparison_scope(directory):
    import full_dspark_request
    import gdn_shared_qk_scope
    import gdn_shared_qk_gate
    import gdn_shared_qk_variants

    directory = Path(directory)
    evidence = qualify(directory / 'frozen-gdn-norm.json', directory, os.environ['TT_METAL_HOME'])
    measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route

    def measured(*args, **kwargs):
        enabled = kwargs.get('gdn_shared_qk', False)
        if type(enabled) is not bool:
            raise ValueError('Explicit norm-prefetch arm required')
        kwargs['gdn_shared_qk'] = True
        builds = []

        def prefetched_build(*operands, **options):
            result = build(*operands, **options)
            builds.append(len(result))
            return result

        with (patch.object(gdn_shared_qk_scope, 'build', prefetched_build) if enabled else nullcontext()), \
                (patch.object(gdn_shared_qk_gate, 'qualify', lambda *args: evidence) if enabled else nullcontext()):
            result = measure(*args, **kwargs)
        if enabled and (not builds or len(builds) % 48 or any(count != 3 for count in builds)):
            raise ValueError('Every target GDN layer must use the qualified three-stage pipeline')
        result['gdn_norm_prefetch'] = dict(enabled=enabled, builds=len(builds),
            report_sha256=REPORT_SHA256 if enabled else None)
        return result

    def route(value, arm):
        enabled = arm == 'publication'
        audit = value.get('gdn_norm_prefetch', {})
        if arm not in ('control', 'publication') or audit.get('enabled') is not enabled:
            raise ValueError('Matched norm-prefetch arm identity required')
        if enabled and (audit.get('report_sha256') != REPORT_SHA256
                or type(audit.get('builds')) is not int or audit['builds'] <= 0 or audit['builds'] % 48):
            raise ValueError('Executed qualified norm-prefetch required')
        with (patch.object(gdn_shared_qk_variants, 'REPORT_SHA256', REPORT_SHA256) if enabled else nullcontext()):
            original_route(value, 'publication')

    with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
            patch.object(gdn_shared_qk_variants, 'validate_route', route):
        yield evidence
