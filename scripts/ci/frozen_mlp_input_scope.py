"""Matched MLP input prefetch on the qualified incremental/norm-prefetch runtime."""

from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
from unittest.mock import patch

from frozen_gdn_norm_scope import runtime_scope as norm_scope
from frozen_mlp_input_gate import REPORT_SHA256, qualify


@contextmanager
def runtime_scope(directory):
    with norm_scope(directory):
        with comparison_scope(directory) as evidence:
            yield evidence


@contextmanager
def comparison_scope(directory):
    import full_dspark_request
    import fused_t16_scope
    import dspark_fusion_variants
    import gdn_shared_qk_variants

    directory = Path(directory)
    evidence = qualify(directory, directory / 'frozen-mlp-input.json')
    spec = importlib.util.spec_from_file_location('frozen_mlp_input_candidate',
        directory / 'frozen-mlp-input-candidate' / 'fused_1d.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route

    def measured(*args, **kwargs):
        enabled = kwargs.get('gdn_shared_qk', False)
        if type(enabled) is not bool:
            raise ValueError('Explicit activation-prefetch arm required')
        kwargs['gdn_shared_qk'] = True
        with (patch.object(fused_t16_scope, 'FusedProjection', candidate.FusedProjection) if enabled else nullcontext()), \
                (patch.object(fused_t16_scope, 'qualify_simulator', lambda: evidence) if enabled else nullcontext()):
            result = measure(*args, **kwargs)
        result['mlp_input_prefetch'] = dict(enabled=enabled,
            report_sha256=REPORT_SHA256 if enabled else None)
        return result

    def route(value, arm):
        enabled = arm == 'publication'
        expected = dict(enabled=enabled, report_sha256=REPORT_SHA256 if enabled else None)
        if arm not in ('control', 'publication') or value.get('mlp_input_prefetch') != expected:
            raise ValueError('Matched activation-prefetch arm identity required')
        with (patch.object(dspark_fusion_variants, 'REPORT_SHA256', REPORT_SHA256) if enabled else nullcontext()):
            original_route(value, 'publication')

    with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
            patch.object(gdn_shared_qk_variants, 'validate_route', route):
        yield evidence
