"""Opt-in physical experiment scope; no serving integration or default changes."""

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from mlp_block_stream import BLOCK_BYTES, geometry
from mlp_block_stream_gate import qualify, REPORT_SHA256, CANDIDATE_SHA256
from mlp_block_stream_pool import bindings
from mlp_block_stream_projection import adapt_projection
from mlp_register_epilogue_gate import qualify as qualify_register


def require_hardware(environment):
    if (any(environment.get(name) != '1' for name in
            ('QWEN_MLP_BLOCK_STREAM_EXPERIMENT', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS'))
            or environment.get('TT_METAL_SIMULATOR') or environment.get('QWEN_SIM_ONLY') == '1'
            or environment.get('TT_METAL_DEVICE_PROFILER')):
        raise ValueError('Explicit allocated unprofiled block-stream hardware experiment required')


def validate_stream(operations, stream):
    if (stream.dtype != operations.uint32 or stream.layout != operations.ROW_MAJOR_LAYOUT
            or stream.memory_config() != operations.DRAM_MEMORY_CONFIG
            or tuple(stream.shape) != (1, 1, geometry()['stream_pages'], BLOCK_BYTES // 4)):
        raise ValueError('Complete interleaved raw stream required')


@contextmanager
def scoped_block_stream(directory, evidence, *, runtime_root, operations, weights, streams):
    import fused_t16_scope

    require_hardware(os.environ)
    if getattr(fused_t16_scope.FusedProjection, '_block_stream_experiment', False):
        raise ValueError('Nested block-stream scopes are unsupported')
    directory = Path(directory)
    weights, streams = tuple(weights), tuple(streams)
    if len(weights) != 64 or len(streams) != 64:
        raise ValueError('Complete caller-owned target stream pool required')
    native_bindings, stream_bindings = bindings(operations, weights), bindings(operations, streams)
    for chip in range(2):
        if set(pair[chip] for pair in native_bindings) & set(pair[chip] for pair in stream_bindings):
            raise ValueError('Streams cannot alias any native layer weights')
    for stream in streams:
        validate_stream(operations, stream)
    register = qualify_register(directory, directory / 'register-epilogue-evidence', runtime_root=runtime_root)
    admission = qualify(directory, evidence, register)
    baseline, = register['kernels']
    expected, = admission['kernels']
    source_path = directory / 'mlp-register-epilogue-candidate/fused_1d.py'
    source = adapt_projection(source_path.read_text())
    if hashlib.sha256(source.encode()).hexdigest() != CANDIDATE_SHA256:
        raise ValueError('Executed projection source differs from simulator candidate')
    candidate = ModuleType('block_stream_hardware_candidate')
    candidate.__file__ = str(source_path)
    exec(compile(source, str(source_path), 'exec'), candidate.__dict__)
    mapping = dict(zip(native_bindings, streams, strict=True))
    audit = dict(report_sha256=REPORT_SHA256, constructions=0, calls=0, restored=False,
        constructed_layers=[], stream_allocations=64, serving_defaults_changed=False)
    constructed = set()
    active = True

    def validate(projection, current_operations):
        require_hardware(os.environ)
        if not active or current_operations is not operations:
            raise ValueError('Projection escaped its hardware scope')
        stream, native_address, stream_address = projection._hardware_stream_binding
        if (bindings(operations, (projection.weights,))[0] != native_address
                or bindings(operations, (stream,))[0] != stream_address
                or hashlib.sha256(projection.compute.encode()).hexdigest() != expected['fused_compute_sha256']
                or projection.manifest != expected):
            raise ValueError('Physical stream binding or arithmetic changed')
        validate_stream(operations, stream)
        return stream

    candidate.validate_binding = validate

    class Projection:
        _block_stream_experiment = True

        def __init__(self, *args, **kwargs):
            self.implementation = candidate.FusedProjection(*args, **kwargs)
            implementation = self.implementation
            if implementation.manifest != baseline:
                raise ValueError('Unchanged register projection required before transport binding')
            key = bindings(operations, (implementation.weights,))[0]
            if key not in mapping or key in constructed:
                raise ValueError('Exactly one projection per admitted native layer required')
            stream = mapping[key]
            implementation._hardware_stream_binding = (stream, key, bindings(operations, (stream,))[0])
            implementation.manifest = copy.deepcopy(expected)
            validate(implementation, operations)
            constructed.add(key)
            audit['constructions'] += 1
            audit['constructed_layers'].append(native_bindings.index(key))

        @property
        def manifest(self):
            return self.implementation.manifest

        def __call__(self, value):
            result = self.implementation(value)
            audit['calls'] += 1
            return result

    original_projection, original_qualification = fused_t16_scope.FusedProjection, fused_t16_scope.qualify_simulator
    try:
        with patch.object(fused_t16_scope, 'FusedProjection', Projection), \
                patch.object(fused_t16_scope, 'qualify_simulator', lambda: copy.deepcopy(admission)):
            yield audit
            if sorted(audit['constructed_layers']) != list(range(64)) or audit['calls'] < 64:
                raise ValueError('Every target layer must construct and execute the candidate')
    finally:
        active = False
        audit['restored'] = (fused_t16_scope.FusedProjection is original_projection
            and fused_t16_scope.qualify_simulator is original_qualification)
        if (not audit['restored'] or bindings(operations, weights) != native_bindings
                or bindings(operations, streams) != stream_bindings
                or qualify(directory, evidence, register) != admission
                or qualify_register(directory, directory / 'register-epilogue-evidence', runtime_root=runtime_root) != register):
            raise ValueError('Hardware scope did not restore bindings or admission changed')
