"""Opt-in T8 QK experiment: exact live rows, defined zero padding, no serving integration."""

from pathlib import Path

from draft_dot import dot_core_layout, dot_geometry


def live_qk_geometry(left_shape, right_shape):
    if tuple(left_shape) != (1, 16, 32, 128):
        raise ValueError('Only T8 draft QK with padded32 and width128 is supported')
    return dot_geometry(left_shape, right_shape)


def validate_live_qk_mask(mask):
    import torch

    if (not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu'
            or mask.dtype != torch.bfloat16 or mask.ndim != 4
            or tuple(mask.shape[:3]) != (1, 1, 32) or mask.shape[3] % 32
            or not 32 <= mask.shape[3] <= 2080):
        raise ValueError('Host BF16 T8 attention mask required before capture')
    if not ((mask == 0) | torch.isneginf(mask)).all() or not (mask[..., :8, :] == 0).any(-1).all():
        raise ValueError('Each live query needs a nonempty zero/negative-infinity mask')
    if not (mask[..., 8:, :] == 0).sum(-1).eq(1).all():
        raise ValueError('Each padded query must allow exactly one zero-bias key')


def live_qk(mesh, left, right, owned):
    import ttnn

    workers, key_tiles, width_tiles = live_qk_geometry(tuple(left.shape), tuple(right.shape))
    columns, full_rows, remainder = dot_core_layout(workers)
    grid = mesh.compute_with_storage_grid_size()
    if remainder or grid.x < columns or grid.y < full_rows or width_tiles != 4:
        raise ValueError('Bounded full-row QK worker grid required')
    for tensor in (left, right):
        if tensor.dtype != ttnn.float32 or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError('Interleaved FP32 QK operands required')
    output = ttnn.empty((1, 16, 32, right.shape[2]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    owned.append(output)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, full_rows - 1))])
    buffers = [ttnn.CBDescriptor(total_size=4096 * count, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32,
            page_size=4096, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        for index, count in ((0, 4), (1, 2), (2, 5), (16, 1))]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode.extend(modes)
    program = ttnn.MeshProgramDescriptor()
    device_tensors = [ttnn.get_device_tensors(tensor) for tensor in (left, right, output)]
    if any(len(shards) != 2 for shards in device_tensors):
        raise ValueError('Exactly two QK shards required')
    for chip, shards in enumerate(zip(*device_tensors, strict=True)):
        runtime = ttnn.RuntimeArgs()
        for worker in range(workers):
            runtime[worker % columns][worker // columns] = [tensor.buffer_address() for tensor in shards] + [worker, workers, key_tiles]
        reader = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_live_qk_io.cpp')), core_ranges=cores,
            compile_time_args=[argument for tensor in shards for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
            runtime_args=runtime, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        compute = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_live_qk_compute.cpp')),
            core_ranges=cores, runtime_args=runtime, config=config)
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[reader, compute], cbs=buffers)
    ttnn.generic_op([left, right, output], program)
    return output
