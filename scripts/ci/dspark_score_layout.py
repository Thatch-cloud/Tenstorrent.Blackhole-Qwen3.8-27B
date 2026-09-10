"""Fuse one FP32 Markov base row plus FP32 bias into compact scores; dot product unchanged."""

from pathlib import Path


def geometry(base_shape, bias_shape, step, worker_limit=110):
    if (len(base_shape) != 4 or tuple(base_shape[:2]) != (1, 1)
            or base_shape[2] not in (1, 3, 7, 15) or base_shape[3] not in (64, 248320)
            or tuple(bias_shape) != (1, 1, 1, base_shape[3])
            or type(step) is not int or not 0 <= step < base_shape[2]
            or type(worker_limit) is not int or worker_limit not in (1, 64, 110)):
        raise ValueError('Complete supported FP32 score rows, one bias row and bounded workers required')
    tiles = base_shape[3] // 32
    return min(tiles, worker_limit), tiles


def execute(operations, mesh, base, bias, step, retain, *, worker_limit=110):
    workers, tiles = geometry(tuple(base.shape), tuple(bias.shape), step, worker_limit)
    if list(mesh.shape) != [1, 2] or not callable(retain):
        raise ValueError('Two-chip mesh and explicit output ownership required')
    for value in (base, bias):
        if (value.dtype != operations.float32 or value.layout != operations.TILE_LAYOUT
                or value.memory_config() != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('Original tiled FP32 interleaved DRAM base and bias required')
    columns = min(11, workers)
    rows, remainder = divmod(workers, columns)
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < columns or grid.y < rows + bool(remainder):
        raise ValueError('Score workers exceed the compute grid')
    ranges = []
    if rows:
        ranges.append(operations.CoreRange(operations.CoreCoord(0, 0), operations.CoreCoord(columns - 1, rows - 1)))
    if remainder:
        ranges.append(operations.CoreRange(operations.CoreCoord(0, rows), operations.CoreCoord(remainder - 1, rows)))
    cores = operations.CoreRangeSet(ranges)
    output = retain(operations.empty(tuple(bias.shape), dtype=operations.float32, layout=operations.ROW_MAJOR_LAYOUT,
        device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG))
    buffers = [operations.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.float32,
            page_size=4096, tile=operations.TileDescriptor(operations.Tile([32, 32])))]) for index in (0, 1, 16)]
    config = operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [operations.UnpackToDestMode.Default] * 64
    for index in (0, 1):
        modes[index] = operations.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode.extend(modes)
    program = operations.MeshProgramDescriptor()
    shards = [operations.get_device_tensors(value) for value in (base, bias, output)]
    if any(len(values) != 2 for values in shards):
        raise ValueError('Both score replicas required')
    for chip, tensors in enumerate(zip(*shards, strict=True)):
        runtime = operations.RuntimeArgs()
        for worker in range(workers):
            runtime[worker % columns][worker // columns] = [value.buffer_address() for value in tensors] + [
                worker, workers, tiles, step]
        reader = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('dspark_score_layout_io.cpp')),
            core_ranges=cores, compile_time_args=[item for value in tensors
                for item in operations.TensorAccessorArgs(value).get_compile_time_args()], runtime_args=runtime,
            config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                noc=operations.NOC.RISCV_0_default))
        compute = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('dspark_score_layout_compute.cpp')),
            core_ranges=cores, runtime_args=runtime, config=config)
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=[reader, compute], cbs=buffers)
    operations.generic_op([base, bias, output], program)
    return output
