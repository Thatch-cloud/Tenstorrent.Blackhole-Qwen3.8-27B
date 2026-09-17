"""Matched norm readers with shared-Q/K and incremental publication in both arms."""

from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch

from frozen_gdn_norm_gate import qualify as qualify_prefetch, REPORT_SHA256 as PREFETCH_SHA256
from frozen_gdn_norm_prefetch import build as prefetch_build
from frozen_incremental_scope import runtime_scope as incremental_scope
from shared_qk_norm_scatter_gate import qualify as qualify_scatter, REPORT_SHA256 as SCATTER_SHA256
from shared_qk_norm_scatter import build as scatter_build
from shared_qk_norm_comparison import SCHEDULE


def validate_identity(value):
    record = value.get('norm_reader', {})
    enabled = record.get('scatter')
    expected = SCATTER_SHA256 if enabled else PREFETCH_SHA256
    if (type(enabled) is not bool or record.get('simulator_report_sha256') != expected
            or type(record.get('builds')) is not int or record['builds'] < 48 or record['builds'] % 48
            or record.get('restored') is not True):
        raise ValueError('Executed source-bound norm reader identity required')
    return expected


@contextmanager
def runtime_scope(directory):
    import full_dspark_request
    import gdn_shared_qk_scope
    import gdn_shared_qk_gate
    import gdn_shared_qk_variants

    if (os.environ.get('QWEN_SHARED_QK_NORM_COMPARISON') != '1'
            or os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '4096'
            or os.environ.get('TT_METAL_DEVICE_PROFILER')):
        raise ValueError('Explicit unprofiled 4K norm comparison required')
    directory = Path(directory)
    evidence = {
        False: qualify_prefetch(directory / 'frozen-gdn-norm.json', directory, os.environ['TT_METAL_HOME']),
        True: qualify_scatter(directory / 'shared-qk-norm-scatter.json', directory, os.environ['TT_METAL_HOME']),
    }
    calls = []
    with incremental_scope(directory):
        measure = full_dspark_request.measure_dspark_request
        original_route = gdn_shared_qk_variants.validate_route

        def measured(*args, **kwargs):
            if len(calls) >= len(SCHEDULE):
                raise ValueError('Unexpected extra comparison request')
            enabled, audit = SCHEDULE[len(calls)]
            if kwargs.get('audit_features') is not audit or any(kwargs.get(name) is not True for name in (
                    'gdn_shared_qk', 'fused_t16_mlp', 'captured_publication', 'target_attention_t16',
                    'commit_only_gdn', 'proposal_trace', 'native_attention', 'score_layout')):
                raise ValueError('Complete winning configuration required in both arms')
            calls.append((enabled, audit))
            builds = []
            implementation = scatter_build if enabled else prefetch_build

            def selected_build(*operands, **options):
                result = implementation(*operands, **options)
                builds.append(len(result))
                return result

            with patch.object(gdn_shared_qk_scope, 'build', selected_build), \
                    patch.object(gdn_shared_qk_gate, 'qualify', lambda *args: evidence[enabled]):
                result = measure(*args, **kwargs)
            if not builds or any(count != 3 for count in builds):
                raise ValueError('Every layer must execute the complete three-stage GDN')
            result['norm_reader'] = dict(scatter=enabled, builds=len(builds), restored=True,
                simulator_report_sha256=SCATTER_SHA256 if enabled else PREFETCH_SHA256)
            validate_identity(result)
            return result

        def route(value, arm):
            if arm != 'publication':
                raise ValueError('Winning incremental publication required in both arms')
            expected = validate_identity(value)
            with patch.object(gdn_shared_qk_variants, 'REPORT_SHA256', expected):
                original_route(value, arm)

        with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
                patch.object(gdn_shared_qk_variants, 'validate_route', route):
            yield evidence
        if calls != list(SCHEDULE):
            raise ValueError('Two audits and four complete ABBA requests required')
    qualify_prefetch(directory / 'frozen-gdn-norm.json', directory, os.environ['TT_METAL_HOME'])
    qualify_scatter(directory / 'shared-qk-norm-scatter.json', directory, os.environ['TT_METAL_HOME'])
