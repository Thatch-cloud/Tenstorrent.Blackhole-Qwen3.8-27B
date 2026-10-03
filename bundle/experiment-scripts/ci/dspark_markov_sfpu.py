"""Unqualified single-row Markov dot/add prototype using SFPU arithmetic instead of grouped FPU products."""

from pathlib import Path


def geometry(latent_shape, weight_shape, base_shape, step, worker_limit=110):
    if (type(worker_limit) is not int or worker_limit not in (1, 64, 80, 110)
            or tuple(latent_shape) != (1, 1, 1, 256) or len(base_shape) != 4
            or tuple(base_shape[:2]) != (1, 1) or base_shape[2] not in (1, 3, 7, 15)
            or base_shape[3] not in (64, 248320) or tuple(weight_shape) != (1, 1, base_shape[3], 256)
            or type(step) is not int or not 0 <= step < base_shape[2]):
        raise ValueError('One rank256 latent, transposed complete codebook and valid full-vocabulary score row required')
    tiles = base_shape[3] // 32
    return min(worker_limit, tiles), tiles, base_shape[3]


def execute(operations, mesh, latent, weight, base, step, owned, *, worker_limit=110):
    workers, tiles, vocabulary = geometry(tuple(latent.shape), tuple(weight.shape), tuple(base.shape), step, worker_limit)
    if list(mesh.shape) != [1, 2]:
        raise ValueError('Two independently scored replicas required')
    for tensor, dtype in ((latent, operations.bfloat16), (weight, operations.bfloat16), (base, operations.float32)):
        if (tensor.dtype != dtype or tensor.layout != operations.TILE_LAYOUT
                or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('Explicit interleaved BF16 latent/codebook and FP32 base scores required')
    columns = 11 if workers > 80 else 8
    rows, remainder = divmod(workers, columns)
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < min(columns, workers) or grid.y < rows + bool(remainder):
        raise ValueError('Markov score workers exceed the available compute grid')
    ranges = []
    if rows:
        ranges.append(operations.CoreRange(operations.CoreCoord(0, 0), operations.CoreCoord(columns - 1, rows - 1)))
    if remainder:
        ranges.append(operations.CoreRange(operations.CoreCoord(0, rows), operations.CoreCoord(remainder - 1, rows)))
    cores = operations.CoreRangeSet(ranges)
    output = operations.empty((1, 1, 1, vocabulary), dtype=operations.float32, layout=operations.ROW_MAJOR_LAYOUT,
        device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    owned.append(output)
    buffers = []
    for index, count, dtype, size in ((0, 2, operations.bfloat16, 2048), (1, 8, operations.bfloat16, 2048),
            (2, 1, operations.float32, 4096), (3, 1, operations.float32, 4096), (16, 1, operations.float32, 4096)):
        buffers.append(operations.CBDescriptor(total_size=count * size, core_ranges=cores,
            format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                page_size=size, tile=operations.TileDescriptor(operations.Tile([32, 32])))]))
    config = operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [operations.UnpackToDestMode.Default] * 64
    modes[2] = operations.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode.extend(modes)
    program = operations.MeshProgramDescriptor()
    shards = [operations.get_device_tensors(value) for value in (latent, weight, base, output)]
    if any(len(values) != 2 for values in shards):
        raise ValueError('Both Markov operand and output replicas required')
    for chip, tensors in enumerate(zip(*shards, strict=True)):
        runtime = operations.RuntimeArgs()
        for worker in range(workers):
            runtime[worker % columns][worker // columns] = [value.buffer_address() for value in tensors] + [worker, workers, tiles, step]
        reader = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('dspark_markov_sfpu_io.cpp')),
            core_ranges=cores, compile_time_args=[item for value in tensors
                for item in operations.TensorAccessorArgs(value).get_compile_time_args()], runtime_args=runtime,
            config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                noc=operations.NOC.RISCV_0_default))
        compute = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('dspark_markov_sfpu_compute.cpp')),
            core_ranges=cores, runtime_args=runtime, config=config)
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(kernels=[reader, compute], cbs=buffers)
    operations.generic_op([latent, weight, base, output], program)
    return output
