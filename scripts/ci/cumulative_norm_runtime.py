"""Explicit request-owned norm selection with matching simulator admission."""

from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
from unittest.mock import patch

from frozen_gdn_norm_gate import qualify as qualify_prefetch, REPORT_SHA256 as PREFETCH_SHA256
from frozen_gdn_norm_prefetch import build as prefetch_build
from frozen_incremental_scope import runtime_scope as incremental_scope
from shared_qk_norm_scatter_gate import qualify as qualify_scatter, REPORT_SHA256 as SCATTER_SHA256
from shared_qk_norm_scatter import build as scatter_build


POLICIES = {'prefetch': PREFETCH_SHA256, 'scatter': SCATTER_SHA256}
_selected = ContextVar('cumulative_norm_policy', default=None)
_active = ContextVar('cumulative_norm_active', default=False)


def require_active():
    if not _active.get():
        raise ValueError('Cumulative normalization wrapper must be installed before requests')


@contextmanager
def selected_runtime_scope(directory):
    flag = os.environ.get('QWEN_CUMULATIVE_NORM', '0')
    if flag not in ('0', '1'):
        raise ValueError('Explicit zero/one cumulative normalization policy required')
    if flag == '0':
        from frozen_gdn_norm_scope import runtime_scope as original_scope
        with original_scope(directory) as evidence:
            yield evidence
    else:
        with runtime_scope(directory) as evidence:
            yield evidence


@contextmanager
def select_norm(policy):
    if policy not in POLICIES or _selected.get() is not None:
        raise ValueError('Known non-nested cumulative normalization policy required')
    token = _selected.set(policy)
    try:
        yield
    finally:
        _selected.reset(token)


def validate_identity(request):
    identity = request.get('norm_reader', {})
    policy, builds = identity.get('policy'), identity.get('builds')
    expected = POLICIES.get(policy)
    if (expected is None or identity.get('report_sha256') != expected
            or identity.get('restored') is not True or type(builds) is not int or builds < 48 or builds % 48
            or request.get('gdn_shared_qk', {}).get('admission', {}).get('report_sha256') != expected):
        raise ValueError('Executed normalization and shared-Q/K admission must agree')
    prefetch = request.get('gdn_norm_prefetch', {})
    if prefetch != dict(enabled=policy == 'prefetch', builds=builds if policy == 'prefetch' else 0,
                        report_sha256=PREFETCH_SHA256 if policy == 'prefetch' else None):
        raise ValueError('Scatter must not be reported as executed prefetch')
    return expected


@contextmanager
def runtime_scope(directory):
    with incremental_scope(directory):
        with normalization_scope(directory) as evidence:
            yield evidence


@contextmanager
def normalization_scope(directory):
    import full_dspark_request
    import gdn_shared_qk_scope
    import gdn_shared_qk_gate
    import gdn_shared_qk_variants

    if os.environ.get('QWEN_CUMULATIVE_NORM') != '1':
        raise ValueError('Explicit cumulative normalization experiment required')
    directory = Path(directory)

    def qualify():
        return dict(prefetch=qualify_prefetch(directory / 'frozen-gdn-norm.json', directory, os.environ['TT_METAL_HOME']),
                    scatter=qualify_scatter(directory / 'shared-qk-norm-scatter.json', directory, os.environ['TT_METAL_HOME']))

    evidence = qualify()
    measure, original_route = full_dspark_request.measure_dspark_request, gdn_shared_qk_variants.validate_route
    native_build, native_gate = gdn_shared_qk_scope.build, gdn_shared_qk_gate.qualify

    def measured(*args, **kwargs):
        if kwargs.get('gdn_shared_qk') is not True:
            raise ValueError('Cumulative requests must retain shared Q/K in both arms')
        policy = _selected.get() or 'prefetch'
        implementation = prefetch_build if policy == 'prefetch' else scatter_build
        builds = []

        def build(*operands, **options):
            result = implementation(*operands, **options)
            builds.append(len(result))
            return result

        with patch.object(gdn_shared_qk_scope, 'build', build), \
                patch.object(gdn_shared_qk_gate, 'qualify', lambda *args: evidence[policy]):
            result = measure(*args, **kwargs)
        if not builds or any(count != 3 for count in builds):
            raise ValueError('Complete three-stage normalization pipeline required')
        restored = gdn_shared_qk_scope.build is native_build and gdn_shared_qk_gate.qualify is native_gate
        result['norm_reader'] = dict(policy=policy, builds=len(builds), restored=restored,
                                     report_sha256=POLICIES[policy])
        result['gdn_norm_prefetch'] = dict(enabled=policy == 'prefetch',
            builds=len(builds) if policy == 'prefetch' else 0,
            report_sha256=PREFETCH_SHA256 if policy == 'prefetch' else None)
        validate_identity(result)
        return result

    def route(request, arm):
        if arm != 'publication':
            raise ValueError('Cumulative normalization requires the retained publication recipe')
        expected = validate_identity(request)
        with patch.object(gdn_shared_qk_variants, 'REPORT_SHA256', expected):
            original_route(request, arm)

    if _active.get():
        raise ValueError('Nested cumulative normalization wrappers are unsupported')
    token = _active.set(True)
    try:
        with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
                patch.object(gdn_shared_qk_variants, 'validate_route', route):
            yield evidence
    finally:
        _active.reset(token)
        if qualify() != evidence:
            raise ValueError('Normalization source admission changed during cumulative requests')
