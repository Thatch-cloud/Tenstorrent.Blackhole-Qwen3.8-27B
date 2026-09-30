"""Simulator-only direct-window descriptor with immutable entry history and native math."""

import hashlib
import os
from pathlib import Path

from gdn_direct_window import reader


DIRECTORY = 'ttnn/cpp/ttnn/operations/transformer/gdn_conv_gates/device'
HASHES = {
    'gdn_conv_gates_program_factory.cpp': '0d02d3422ede6f814d9663ee4d739f16399b5bdf543c66f754db5f20127429f5',
    'kernels/compute/gdn_conv_gates.cpp': '9bdc2e38ef8415c2d4c9b4c7211972d7d61e39c6defed34acf19d7b9549391da',
    'kernels/dataflow/reader_gdn_conv_gates.cpp': '6662bbbefd64c13341af3839bb8543c38520fe0b7ed91c0b5f6a9fe5a55c418f',
    'kernels/dataflow/writer_gdn_conv_gates.cpp': '0e4193a0aa4eff3aaae2ce9b6fcf96eae282dffb3a65a59cf1eeb81882b9dce3',
}
IO_PAGES = {0: 8, 1: 8, 4: 2, 5: 8, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 4}
FP32_PAGES = {2: 4, 3: 2}


def sources(root):
    result = {}
    for name, expected in HASHES.items():
        payload = (Path(root) / DIRECTORY / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Pinned convolution source changed: ' + name)
        result[name] = payload.decode()
    return dict(reader=reader(result['kernels/dataflow/reader_gdn_conv_gates.cpp']),
                writer=result['kernels/dataflow/writer_gdn_conv_gates.cpp'],
                compute=result['kernels/compute/gdn_conv_gates.cpp'])


def work(grid_x, grid_y):
    if type(grid_x) is not int or type(grid_y) is not int or grid_x < 9 or grid_y < 9:
        raise ValueError('Audited Blackhole-sized worker grid required')
    count = grid_x * grid_y
    per_core = (160 + count - 1) // count
    convolution = (160 + per_core - 1) // per_core
    workers = convolution + (convolution < count)
    return [(index // grid_y, index % grid_y,
             index * per_core if index < convolution else 0,
             min(per_core, 160 - index * per_core) if index < convolution else 0,
             1 if index == workers - 1 else 0) for index in range(workers)]


def execute(operations, mesh, projected, history, taps, dt_bias, neg_exp_A, retain, *, root='/opt/tt-metal'):
    if os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR'):
        raise ValueError('Direct convolution windows are simulator-only')
    if list(mesh.shape) != [1, 2] or not callable(retain) or len(history) != 4 or len(taps) != 4:
        raise ValueError('Two-chip owned convolution with four history slots and taps required')
    inputs = [projected, *history, *taps, dt_bias, neg_exp_A]
    shapes = [(1, 16, 8240)] + [(1, 1, 5120)] * 8 + [(1, 1, 24)] * 2
    for value, shape in zip(inputs, shapes, strict=True):
        if (tuple(value.shape) != shape or value.dtype != operations.bfloat16
                or value.layout != operations.TILE_LAYOUT
                or value.memory_config() not in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG)):
            raise ValueError('Exact native BF16 tiled T16 projection/history/gate geometry required')
    kernels = sources(root)
    grid = mesh.compute_with_storage_grid_size()
    workers = work(grid.x, grid.y)
    cores = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(horizontal, vertical),
        operations.CoreCoord(horizontal, vertical)) for horizontal, vertical, *_ in workers])
    outputs = [retain(operations.empty(shape, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
        device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG))
        for shape in [(1, 16, 5120), (1, 16, 24), (1, 16, 24)] + [(1, 16, 5120)] * 4]
    tensors = inputs + outputs
    shards = [operations.get_device_tensors(value) for value in tensors]
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Every operand must cover both cards')
    buffers = []
    for counts, dtype, page in ((IO_PAGES, operations.bfloat16, 2048), (FP32_PAGES, operations.float32, 4096)):
        for index, count in counts.items():
            buffers.append(operations.CBDescriptor(total_size=count * page, core_ranges=cores,
                format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=operations.TileDescriptor(operations.Tile([32, 32])))]))
    program = operations.MeshProgramDescriptor()
    for chip, local in enumerate(zip(*shards, strict=True)):
        addresses = [value.buffer_address() for value in local]
        if len(addresses) != len(set(addresses)):
            raise ValueError('Immutable inputs and checkpoint outputs must not alias')
        read_tensors = list(local[:9]) + [local[0], local[0], local[9], local[10]]
        write_tensors = [local[11], *local[14:18], local[12], local[13]]
        compile_args = dict(reader=[4, 160, 1, 16, 1, 258, 1, 258, 8192, 258, 8216, 24],
                            writer=[4, 160, 1], compute=[4, 1, 0x3f800000, 0x41a00000])
        for role, values in (('reader', read_tensors), ('writer', write_tensors)):
            for value in values:
                compile_args[role].extend(operations.TensorAccessorArgs(value).get_compile_time_args())
        configs = dict(reader=operations.DataMovementConfigDescriptor(
                processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default),
            writer=operations.DataMovementConfigDescriptor(
                processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default),
            compute=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4,
                fp32_dest_acc_en=True, math_approx_mode=False))
        descriptors = []
        for role in ('reader', 'writer', 'compute'):
            runtime = operations.RuntimeArgs()
            for horizontal, vertical, start, count, gates in workers:
                runtime[horizontal][vertical] = ([count, gates] if role == 'compute' else
                    [start, count, gates] + [value.buffer_address() for value in
                        (read_tensors if role == 'reader' else write_tensors)])
            descriptors.append(operations.KernelDescriptor(kernel_source=kernels[role],
                source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=compile_args[role], runtime_args=runtime, config=configs[role]))
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=descriptors, cbs=buffers)
    operations.generic_op(tensors, program)
    return outputs
