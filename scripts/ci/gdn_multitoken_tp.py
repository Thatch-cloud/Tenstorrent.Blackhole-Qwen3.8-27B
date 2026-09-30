"""gdn_multitoken.execute (the native T-token recurrence, one chip program per device) at any served width.

gdn_multitoken.py is hash-pinned (gdn_vsplit.HELPER_HASH, the TP2 evidence) and carries the pair's geometry as
literals: 24 value heads, 5120 QKV columns, the reader's conv-page and tile-offset arguments 160 / 32 / 64 / 96, a
3072-column fused output, and a two-chip check. This twin is call for call the pinned validate_geometry and execute with
those numbers read from tp_shapes (12 heads, 2560 columns, 80 / 16 / 32 / 48 pages and tile offsets, 1536 columns at
four cards) and the chip loop over the mesh's devices. The kernels are the pinned ones (gdn_multitoken.load_kernels /
transform, imported, never copied): they take every one of these numbers as a compile-time argument.

tp_addresses.install() rebinds gdn_multitoken.execute / validate_geometry to these at QWEN_FAST_TP=4 only, so the callers
that imported them by name (gdn_multitoken_conv, gdn_batched_conv) reach the twin. At the pair nothing is rebound.
"""

import os
from pathlib import Path
import struct

import gdn_multitoken as pinned
import tp_shapes

TILE = tp_shapes.TILE


def validate_geometry(qkv, beta, gate, initial):
    found = tp_shapes.active()
    rows = qkv[1] if len(qkv) == 3 else 0
    if rows not in (1, 2, 4, 8, 16, 32) or tuple(qkv) != (1, rows, found.gdn_qkv):
        raise ValueError('Expected packed TP%d QKV [1,T,%d]' % (found.tp, found.gdn_qkv))
    if (tuple(beta) != (1, rows, found.gdn_nv) or tuple(gate) != (1, rows, found.gdn_nv)
            or tuple(initial) != (1, found.gdn_nv, 128, 128)):
        raise ValueError('Expected T sequential rows sharing one %d-head initial state' % found.gdn_nv)
    return rows


def reader_geometry(found=None):
    """(H, Ct, KOT, VOT, WTZ): the value-head count, conv pages, and the K / V / z tile offsets the reader takes as
    compile-time arguments (pair: 24, 160, 32, 64, 96)."""
    found = found or tp_shapes.active()
    key_tiles = found.gdn_key // TILE
    return found.gdn_nv, found.gdn_conv_pages, key_tiles, 2 * key_tiles, found.gdn_value // TILE


def execute(mesh, qkv, beta, gate, initial, kernels, *, z=None, norm_w=None):
    import ttnn

    found = tp_shapes.active()
    heads, conv_pages, key_offset, value_offset, z_offset = reader_geometry(found)
    inputs = [qkv, beta, gate, initial]
    rows = validate_geometry(*(tuple(value.shape) for value in inputs))
    fuse_norm_gate = z is not None
    if fuse_norm_gate != (norm_w is not None):
        raise ValueError('Both z and norm_w required')
    if fuse_norm_gate:
        pinned.validate_handoff_runtime(Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal')))
        if tuple(z.shape) != (1, rows, found.gdn_z) or tuple(norm_w.shape) != (1, 1, 128):
            raise ValueError('Expected z [1,T,%d] and norm_w [1,1,128]' % found.gdn_z)
        inputs += [z, norm_w]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() != ttnn.DRAM_MEMORY_CONFIG for value in inputs):
        raise ValueError('First prototype requires interleaved DRAM BF16 TILE inputs')
    output = ttnn.empty((1, rows, found.gdn_z) if fuse_norm_gate else (rows, 1, heads, 128), device=mesh,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT if fuse_norm_gate else ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.L1_MEMORY_CONFIG if fuse_norm_gate else ttnn.DRAM_MEMORY_CONFIG)
    states = ttnn.empty((rows, heads, 128, 128), device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tensors = inputs[:4] + [output, states] + inputs[4:]
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    chips = tp_shapes.chip_count()
    if any(len(value) != chips for value in shards):
        raise ValueError('%s chips required' % tp_shapes.all_chips())
    grid = mesh.compute_with_storage_grid_size()
    coordinates = [(head // grid.y, head % grid.y) for head in range(heads)]
    if any(horizontal >= grid.x for horizontal, vertical in coordinates):
        raise ValueError('At least %d cores required' % heads)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*point), ttnn.CoreCoord(*point)) for point in coordinates])
    buffers = []
    plan = pinned.cb_plan(fuse_norm_gate)
    for counts, dtype, page in ((plan[0], ttnn.bfloat16, 2048), (plan[1], ttnn.float32, 4096)):
        for index, count in counts.items():
            buffers.append(ttnn.CBDescriptor(total_size=count * page, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))]))
    bits = lambda value: struct.unpack('<I', struct.pack('<f', value))[0]
    compute_args = [4, 4, 1, bits(1e-6), bits(128 ** -0.5), int(fuse_norm_gate),
                    bits(128e-6) if fuse_norm_gate else 0, bits(128 ** 0.5) if fuse_norm_gate else 0]
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local = [value[chip] for value in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != len(addresses):
            raise ValueError('Inputs, initial state and prefix outputs must not alias')
        reader_args = [4, 4, 1, bits(1e-6), bits(128 ** -0.5), heads, 1, conv_pages, 0, key_offset, value_offset,
                       3, 1, 0, 0, 0]
        if fuse_norm_gate:
            reader_args[13:16] = [1, z_offset, 0]
        for index in (0, 0, 0, 1, 2, 3, 6 if fuse_norm_gate else 3, 7 if fuse_norm_gate else 3):
            reader_args.extend(ttnn.TensorAccessorArgs(local[index]).get_compile_time_args())
        writer_args = [4, 4, int(fuse_norm_gate), 1, heads, rows]
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
                                                  addresses[1], addresses[2], addresses[3]] + (addresses[6:8] if fuse_norm_gate else []) if role == 'reader' else
                                                 [head, rows, addresses[4], 2048 if fuse_norm_gate else 256, addresses[5]] if role == 'writer' else [rows])
            descriptor = ttnn.KernelDescriptor(kernel_source=kernels[role],
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=args, config=config)
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=descriptors, cbs=buffers)
    ttnn.generic_op(tensors, program)
    return output, states
