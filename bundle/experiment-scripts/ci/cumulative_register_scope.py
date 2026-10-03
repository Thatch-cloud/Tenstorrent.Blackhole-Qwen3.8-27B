"""Request-owned register epilogue selection without replacing the MLP forward path."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
from unittest.mock import patch

from mlp_register_epilogue_gate import qualify, REPORT_SHA256


@contextmanager
def scoped_register_epilogue(directory, *, runtime_root):
    import fused_t16_scope

    directory = Path(directory)
    evidence_path = directory / 'register-epilogue-evidence'
    evidence = qualify(directory, evidence_path, runtime_root=runtime_root)
    spec = importlib.util.spec_from_file_location('cumulative_register_candidate',
        directory / 'mlp-register-epilogue-candidate/fused_1d.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    original_projection = fused_t16_scope.FusedProjection
    original_qualification = fused_t16_scope.qualify_simulator
    if getattr(original_projection, '_cumulative_register', False):
        raise ValueError('Nested cumulative register epilogues are unsupported')
    audit = dict(report_sha256=REPORT_SHA256, constructions=0, calls=0, restored=False)

    class Projection:
        _cumulative_register = True

        def __init__(self, *args, **kwargs):
            if kwargs.get('token_rows') != 16:
                raise ValueError('Register epilogue is qualified only for T16')
            self.implementation = candidate.FusedProjection(*args, **kwargs)
            audit['constructions'] += 1

        @property
        def manifest(self):
            return self.implementation.manifest

        def __call__(self, value):
            result = self.implementation(value)
            audit['calls'] += 1
            return result

    try:
        with patch.object(fused_t16_scope, 'FusedProjection', Projection), \
                patch.object(fused_t16_scope, 'qualify_simulator', lambda: evidence):
            yield audit
    finally:
        audit['restored'] = (fused_t16_scope.FusedProjection is original_projection
            and fused_t16_scope.qualify_simulator is original_qualification)
        if not audit['restored'] or qualify(directory, evidence_path, runtime_root=runtime_root) != evidence:
            raise ValueError('Cumulative register epilogue bindings or admission changed')


def validate_request(request, audit):
    fusion = request.get('fused_t16_mlp', {})
    hits = fusion.get('hits', [])
    if (audit.get('report_sha256') != REPORT_SHA256 or audit.get('restored') is not True
            or type(audit.get('constructions')) is not int or audit['constructions'] != 64
            or type(audit.get('calls')) is not int or audit['calls'] <= 0
            or len(hits) != 64 or any(type(count) is not int or count <= 0 for count in hits)
            or audit['calls'] != sum(hits) or fusion.get('passed_simulator') != REPORT_SHA256):
        raise ValueError('All 64 register epilogues must execute under matching admission')
