"""Annotate executed epilogues before the combined request route is validated."""

from contextlib import contextmanager
from unittest.mock import patch

from cumulative_fusion_validation import validate_fusion_policy
from cumulative_register_scope import scoped_register_epilogue


@contextmanager
def runtime_scope(directory, *, runtime_root):
    import full_dspark_request
    import gdn_shared_qk_variants
    import dspark_fusion_variants

    original_measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route
    active = []

    @contextmanager
    def candidate():
        if active:
            raise ValueError('Nested cumulative register requests are unsupported')
        results = []
        with scoped_register_epilogue(directory, runtime_root=runtime_root) as audit:
            active.append(results)
            try:
                yield audit
            finally:
                active.pop()
        if len(results) != 1:
            raise ValueError('Exactly one complete request per register scope required')
        results[0]['register_epilogue'] = dict(audit, register_resident=True)
        validate_fusion_policy(results[0], 'register')

    def measure(*args, **kwargs):
        result = original_measure(*args, **kwargs)
        if active:
            active[-1].append(result)
        else:
            result['register_epilogue'] = dict(register_resident=False, report_sha256=None,
                constructions=0, calls=0, restored=True)
        return result

    def route(request, arm):
        identity = request.get('register_epilogue', {})
        enabled = identity.get('register_resident')
        if type(enabled) is not bool:
            raise ValueError('Explicit request-owned register selection required')
        expected = validate_fusion_policy(request, 'register' if enabled else 'baseline')
        with patch.object(dspark_fusion_variants, 'REPORT_SHA256', expected):
            original_route(request, arm)

    with patch.object(full_dspark_request, 'measure_dspark_request', measure), \
            patch.object(gdn_shared_qk_variants, 'validate_route', route):
        yield candidate
