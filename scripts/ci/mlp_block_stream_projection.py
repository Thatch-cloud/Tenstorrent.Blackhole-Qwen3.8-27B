"""Simulator-only adapter: retain fused arithmetic, replace weight transport only."""

import hashlib
import os

from frozen_recipe_context import replace_once
from mlp_block_stream import BLOCK_BYTES, geometry


def adapt_projection(source):
    if 'mlp_block_stream_projection' in source:
        raise ValueError('Block-stream projection already adapted')
    result = replace_once(source, 'import hashlib\n',
        'import hashlib\nfrom mlp_block_stream import reader_source\n'
        'from mlp_block_stream_projection import validate_binding\n')
    result = replace_once(result, '    def __call__(self, value):\n        import ttnn\n',
        '    def __call__(self, value):\n        import ttnn\n'
        '        stream_weights = validate_binding(self, ttnn)\n')
    result = replace_once(result, 'ttnn.get_device_tensors(self.weights),',
        'ttnn.get_device_tensors(stream_weights),')
    result = replace_once(result,
        'kernel_source=str(Path(__file__).with_name("fused_1d_weights.cpp")), core_ranges=workers,',
        'kernel_source=reader_source(Path(__file__).with_name("fused_1d_weights.cpp").read_text()),\n'
        '                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=workers,')
    result = replace_once(result, 'ttnn.generic_op([value, self.weights, output], mesh_program)',
        'ttnn.generic_op([value, stream_weights, output], mesh_program)')
    compile(result, 'block_stream_fused_1d.py', 'exec')
    return result


def identity(projection, stream, operations):
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_HARDWARE_TESTS') or os.environ.get('QWEN_CARDS_ALLOCATED')):
        raise ValueError('Block-stream MLP remains simulator-only until numerical admission')
    if (projection.token_rows != 16 or projection.pairs_per_worker != 3 or projection.intermediates is not False
            or projection.math_approx_mode is not True):
        raise ValueError('Exact T16 three-pair fused target configuration required')
    shape = geometry()
    if (stream.dtype != operations.uint32 or stream.layout != operations.ROW_MAJOR_LAYOUT
            or stream.memory_config() != operations.DRAM_MEMORY_CONFIG
            or tuple(stream.shape) != (1, 1, shape['stream_pages'], BLOCK_BYTES // 4)):
        raise ValueError('Complete owned block-major stream required')
    source = projection.weights
    if (source.dtype != operations.bfloat4_b or source.layout != operations.TILE_LAYOUT
            or source.memory_config() != operations.DRAM_MEMORY_CONFIG
            or tuple(source.shape) not in ((5120, 17408), (1, 1, 5120, 17408))):
        raise ValueError('Original native BF4 weights must remain available and unchanged')
    source_shards, stream_shards = (operations.get_device_tensors(tensor) for tensor in (source, stream))
    if len(source_shards) != 2 or len(stream_shards) != 2:
        raise ValueError('Both chips required for stream binding')
    addresses = tuple((before.buffer_address(), after.buffer_address())
        for before, after in zip(source_shards, stream_shards, strict=True))
    if any(before == after for before, after in addresses):
        raise ValueError('Stream cannot alias native weights')
    return dict(addresses=addresses, compute_sha256=hashlib.sha256(projection.compute.encode()).hexdigest())


def bind_stream(projection, stream, operations):
    if hasattr(projection, '_block_stream_binding'):
        raise ValueError('A projection cannot be rebound to another stream')
    binding = identity(projection, stream, operations)
    projection._block_stream_binding = (stream, binding)
    return dict(binding, extra_weight_bytes_per_chip=geometry()['stream_bytes'],
        borrowed_source=True, stream_owned_by_caller=True, arithmetic_changed=False,
        simulator_qualified=False, hardware_qualified=False)


def validate_binding(projection, operations):
    binding = getattr(projection, '_block_stream_binding', None)
    if binding is None:
        raise ValueError('Explicit caller-owned stream binding required before execution')
    stream, expected = binding
    if identity(projection, stream, operations) != expected:
        raise ValueError('Weight buffers or fused arithmetic changed after stream binding')
    return stream
