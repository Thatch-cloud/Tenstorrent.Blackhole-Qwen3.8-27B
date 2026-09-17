"""Diagnostic-only MLP scope installed inside qualified norm/incremental scopes."""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

from frozen_wait_zone_gate import qualify, SIMULATOR_SHA256, HARDWARE_SHA256


def validate_marker_audit(value):
    if (value.get('mlp_wait_zones') != dict(enabled=True, simulator_report_sha256=SIMULATOR_SHA256,
            hardware_marker_sha256=HARDWARE_SHA256)
            or value.get('instrumented_timing') is not True
            or value.get('committed_tokens_per_second') is not None):
        raise ValueError('Instrumented admitted marker request without TG required')


@contextmanager
def runtime_scope(directory):
    if (os.environ.get('QWEN_COMBINED_TRACE_PROFILE') != '1'
            or os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Explicit allocated 32K combined diagnostic required')
    import full_dspark_request
    import fused_t16_scope
    import dspark_fusion_variants
    import gdn_shared_qk_variants

    directory = Path(directory)
    evidence = qualify(directory)
    spec = importlib.util.spec_from_file_location('frozen_wait_zone_candidate',
        directory / 'frozen-wait-zone-candidate/fused_1d.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route

    def measured(*args, **kwargs):
        if any(kwargs.get(name) is not True for name in
                ('combined_profile', 'gdn_shared_qk', 'fused_t16_mlp', 'audit_features')):
            raise ValueError('Only the audited complete candidate may use diagnostic kernels')
        with patch.object(fused_t16_scope, 'FusedProjection', candidate.FusedProjection), \
                patch.object(fused_t16_scope, 'qualify_simulator', lambda: evidence):
            result = measure(*args, **kwargs)
        result['mlp_wait_zones'] = dict(enabled=True, simulator_report_sha256=SIMULATOR_SHA256,
            hardware_marker_sha256=HARDWARE_SHA256)
        validate_marker_audit(result)
        return result

    def route(value, arm):
        if arm != 'publication':
            raise ValueError('One audited combined diagnostic arm required')
        validate_marker_audit(value)
        with patch.object(dspark_fusion_variants, 'REPORT_SHA256', SIMULATOR_SHA256):
            original_route(value, arm)

    with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
            patch.object(gdn_shared_qk_variants, 'validate_route', route):
        try:
            yield evidence
        finally:
            if qualify(directory) != evidence:
                raise ValueError('Diagnostic sources changed during the combined request')


def validate_route(value, arm):
    from frozen_incremental_scope import validate_records
    from frozen_gdn_norm_gate import REPORT_SHA256 as NORM_SHA256
    from history_append_hardware_gate import REPORT_SHA256 as HISTORY_SHA256
    import dspark_fusion_variants
    import gdn_shared_qk_variants

    validate_marker_audit(value)
    if arm != 'publication':
        raise ValueError('One diagnostic arm required')
    norm = value.get('gdn_norm_prefetch', {})
    history = value.get('incremental_history', {})
    if (norm.get('enabled') is not True or norm.get('report_sha256') != NORM_SHA256
            or type(norm.get('builds')) is not int or norm['builds'] <= 0 or norm['builds'] % 48
            or history.get('enabled') is not True or history.get('report_sha256') != HISTORY_SHA256):
        raise ValueError('Qualified norm prefetch and incremental publication must remain enabled')
    validate_records(history.get('records', []), True)
    with patch.object(dspark_fusion_variants, 'REPORT_SHA256', SIMULATOR_SHA256), \
            patch.object(gdn_shared_qk_variants, 'REPORT_SHA256', NORM_SHA256):
        gdn_shared_qk_variants.validate_route(value, arm)
