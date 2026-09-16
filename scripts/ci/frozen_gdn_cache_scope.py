"""Matched complete requests: shared-Q/K in both arms, input cache only in candidate."""

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
from unittest.mock import patch

from frozen_gdn_cache_gate import REPORT_SHA256, qualify
from frozen_gdn_input_cache import build


@contextmanager
def runtime_scope(directory):
    if (os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Allocated admitted 32K hardware required')
    import full_dspark_request
    import gdn_shared_qk_scope
    import gdn_shared_qk_gate
    import gdn_shared_qk_variants

    directory = Path(directory)
    evidence = qualify(directory / 'frozen-gdn-cache.json', directory, os.environ['TT_METAL_HOME'])
    measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route

    def measured(*args, **kwargs):
        enabled = kwargs.get('gdn_shared_qk', False)
        if type(enabled) is not bool or not all(kwargs.get(name) is True for name in (
                'captured_publication', 'fused_t16_mlp', 'target_attention_t16', 'score_layout')):
            raise ValueError('Complete matched fused runtime required')
        kwargs['gdn_shared_qk'] = True
        builds = []

        def cached_build(*operands, **options):
            result = build(*operands, **options)
            builds.append(len(result))
            return result

        with (patch.object(gdn_shared_qk_scope, 'build', cached_build) if enabled else nullcontext()), \
                (patch.object(gdn_shared_qk_gate, 'qualify', lambda *args: evidence) if enabled else nullcontext()):
            result = measure(*args, **kwargs)
        if enabled and (not builds or len(builds) % 48 or any(count != 3 for count in builds)):
            raise ValueError('All target recurrence layers must use the cached three-stage pipeline')
        result['gdn_input_cache'] = dict(enabled=enabled, builds=len(builds),
            report_sha256=REPORT_SHA256 if enabled else None)
        return result

    def route(value, arm):
        enabled = arm == 'publication'
        audit = value.get('gdn_input_cache', {})
        if arm not in ('control', 'publication') or audit.get('enabled') is not enabled:
            raise ValueError('Matched cache arm identity required')
        if enabled and (audit.get('report_sha256') != REPORT_SHA256 or not audit.get('builds')):
            raise ValueError('Executed qualified cache required')
        with (patch.object(gdn_shared_qk_variants, 'REPORT_SHA256', REPORT_SHA256) if enabled else nullcontext()):
            original_route(value, 'publication')

    with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
            patch.object(gdn_shared_qk_variants, 'validate_route', route):
        yield evidence
