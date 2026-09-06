"""Recurrence-only T-token prototype derived from hash-pinned native kernels."""

import hashlib
from pathlib import Path
import struct


KERNEL_ROOT = 'ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule/device/kernels'
HASHES = {
    'compute/decode_gated_delta_rule.cpp': 'b59314e0acaea06b574feffe91a256d3022e819fb5257154e2e72eeb7978b928',
    'dataflow/reader_decode_gated_delta_rule.cpp': 'fb2d44f415567ef2bb1e6210f30ac731018443bbc372f3d3a4b54ca44476b6fe',
    'dataflow/writer_decode_gated_delta_rule.cpp': '3dd507f77f9932c6cf3f171530ff4fba48a437436aa1b2ab3408d79e242d798a',
}


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Native kernel anchor changed')
    return source.replace(before, after, 1)


def transform(kind, source):
    if kind == 'compute':
        for before, after in (
            ('WAIT(cb_state, kv);', 'WAIT(it == 0 ? cb_state : 30, kv);'),
            ('copy_tiles(cb_state, cb_sf, kv);', 'copy_tiles(it == 0 ? cb_state : 30, cb_sf, kv);'),
            ('POP(cb_state, kv);', 'POP(it == 0 ? cb_state : 30, kv);'),
            ('copy_tiles(cb_snew, cb_sout, kv);',
             'copy_tiles(cb_snew, cb_sout, kv);\n        if (it + 1 < n_inst) { copy_tiles(cb_snew, 30, kv); }'),
        ):
            source = replace_once(source, before, after)
    elif kind in ('reader', 'writer'):
        source = replace_once(source,
            'for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {',
            'for (uint32_t token = 0; token < n_inst; ++token) {\n        const uint32_t bh = bh_start + token * H;')
        if kind == 'reader':
            source = replace_once(source, 'CircularBuffer cbs(cb_state);', 'if (token == 0) {\n        CircularBuffer cbs(cb_state);')
            source = replace_once(source, 'cbs.push_back(kv);', 'cbs.push_back(kv);\n        }')
    else:
        raise ValueError('Unknown kernel role')
    return source


def load_kernels(root=Path('/opt/tt-metal')):
    result = {}
    for relative, expected in HASHES.items():
        data = (root / KERNEL_ROOT / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Native kernel hash changed: {relative}')
        kind = 'compute' if relative.startswith('compute/') else 'reader' if '/reader_' in relative else 'writer'
        result[kind] = transform(kind, data.decode())
    return result


def validate_geometry(qkv, beta, gate, initial):
    rows = qkv[1] if len(qkv) == 3 else 0
    if rows not in (1, 2, 4, 8, 16) or tuple(qkv) != (1, rows, 5120):
        raise ValueError('Expected packed TP2 QKV [1,T,5120]')
    if tuple(beta) != (1, rows, 24) or tuple(gate) != (1, rows, 24) or tuple(initial) != (1, 24, 128, 128):
        raise ValueError('Expected T sequential rows sharing one 24-head initial state')
    return rows


def cb_plan():
    io = {0: 4, 1: 4, 2: 4, 3: 1, 4: 1, 5: 16, 18: 16, 19: 8, 27: 2, 30: 16}
    fp32 = {6: 1, 7: 4, 8: 4, 9: 1, 10: 4, 11: 4, 12: 4, 13: 1, 14: 16,
            15: 4, 16: 8, 17: 16, 20: 4, 21: 4, 22: 4, 23: 1, 24: 1, 25: 16,
            26: 16, 28: 1, 29: 1}
    return io, fp32


def execute(mesh, qkv, beta, gate, initial, kernels):
    import ttnn

    inputs = [qkv, beta, gate, initial]
    rows = validate_geometry(*(tuple(value.shape) for value in inputs))
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() != ttnn.DRAM_MEMORY_CONFIG for value in inputs):
        raise ValueError('First prototype requires interleaved DRAM BF16 TILE inputs')
    output = ttnn.empty((rows, 1, 24, 128), device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    states = ttnn.empty((rows, 24, 128, 128), device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tensors = inputs + [output, states]
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(value) != 2 for value in shards):
        raise ValueError('Both chips required')
    grid = mesh.compute_with_storage_grid_size()
    coordinates = [(head // grid.y, head % grid.y) for head in range(24)]
    if any(horizontal >= grid.x for horizontal, vertical in coordinates):
        raise ValueError('At least 24 cores required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*point), ttnn.CoreCoord(*point)) for point in coordinates])
    buffers = []
    for counts, dtype, page in ((cb_plan()[0], ttnn.bfloat16, 2048), (cb_plan()[1], ttnn.float32, 4096)):
        for index, count in counts.items():
            buffers.append(ttnn.CBDescriptor(total_size=count * page, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32]))) ]))
    bits = lambda value: struct.unpack('<I', struct.pack('<f', value))[0]
    compute_args = [4, 4, 1, bits(1e-6), bits(128 ** -0.5), 0, 0, 0]
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [value[chip] for value in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != len(addresses):
            raise ValueError('Inputs, initial state and prefix outputs must not alias')
        reader_args = [4, 4, 1, bits(1e-6), bits(128 ** -0.5), 24, 1, 160, 0, 32, 64, 3, 1, 0, 0, 0]
        for index in (0, 0, 0, 1, 2, 3, 3, 3):
            reader_args.extend(ttnn.TensorAccessorArgs(local[index]).get_compile_time_args())
        writer_args = [4, 4, 0, 1, 24, rows]
        for index in (4, 5):
            writer_args.extend(ttnn.TensorAccessorArgs(local[index]).get_compile_time_args())
        descriptors = []
        for role, args, config in (
            ('reader', reader_args, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                                    noc=ttnn.NOC.RISCV_1_default)),
            ('writer', writer_args, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                    noc=ttnn.NOC.RISCV_0_default)),
            ('compute', compute_args, ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                                  fp32_dest_acc_en=True, math_approx_mode=False)),
        ):
            runtime = ttnn.RuntimeArgs()
            for head, (horizontal, vertical) in enumerate(coordinates):
                runtime[horizontal][vertical] = ([head, rows, addresses[0], addresses[0], addresses[0],
                                                  addresses[1], addresses[2], addresses[3]] if role == 'reader' else
                                                 [head, rows, addresses[4], 256, addresses[5]] if role == 'writer' else [rows])
            descriptor = ttnn.KernelDescriptor(kernel_source=kernels[role],
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=args, config=config)
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=descriptors, cbs=buffers)
    ttnn.generic_op(tensors, program)
    return output, states
