"""Build an eight-head normalization program with caller-owned persistent tensors."""

from gdn_shared_qk_compute import load_compute
from gdn_shared_qk_dataflow import buffer_plan, kernels as dataflow_kernels
from gdn_vsplit import bits, core_coordinates


def build(operations, mesh, packed, query, key, *, root, serial=False):
    if type(serial) is not bool or tuple(mesh.shape) != (1, 2):
        raise ValueError('Explicit policy and a two-chip mesh required')
    tensors = [packed, query, key]
    for tensor, shape, dtype in ((packed, (1, 16, 5120), operations.bfloat16),
            (query, (1, 16, 1024), operations.float32), (key, (1, 16, 1024), operations.float32)):
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.layout != operations.TILE_LAYOUT
                or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('Expected T16 packed BF16 input and distinct FP32 TILE DRAM outputs')
    shards = [operations.get_device_tensors(tensor) for tensor in tensors]
    if any(len(values) != 2 for values in shards):
        raise ValueError('Every tensor must cover both chips')
    grid = mesh.compute_with_storage_grid_size()
    coordinates = core_coordinates(grid.x, grid.y, 8)
    cores = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(*point),
        operations.CoreCoord(*point)) for point in coordinates])
    buffers = []
    io, fp32 = buffer_plan(serial=serial)
    for counts, dtype, page in ((io, operations.bfloat16, 2048), (fp32, operations.float32, 4096)):
        for index, count in counts.items():
            buffers.append(operations.CBDescriptor(total_size=count * page, core_ranges=cores,
                format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=operations.TileDescriptor(operations.Tile([32, 32])))]))
    sources = dict(dataflow_kernels(), compute=load_compute(root, serial=serial))
    program = operations.MeshProgramDescriptor()
    for chip in range(2):
        local = [values[chip] for values in shards]
        addresses = [tensor.buffer_address() for tensor in local]
        if len(set(addresses)) != 3:
            raise ValueError('Input and normalized outputs must not alias')
        reader_args = [int(serial)] + operations.TensorAccessorArgs(local[0]).get_compile_time_args()
        writer_args = [int(serial)]
        for tensor in local[1:]:
            writer_args.extend(operations.TensorAccessorArgs(tensor).get_compile_time_args())
        configs = dict(reader=operations.DataMovementConfigDescriptor(
                processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default),
            writer=operations.DataMovementConfigDescriptor(
                processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default),
            compute=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4,
                fp32_dest_acc_en=True, math_approx_mode=False))
        descriptors = []
        for role, arguments in (('reader', reader_args), ('writer', writer_args),
                ('compute', [bits(1e-6), bits(128 ** -0.5)])):
            runtime = operations.RuntimeArgs()
            for head, (horizontal, vertical) in enumerate(coordinates):
                runtime[horizontal][vertical] = ([head, 16, addresses[0]] if role == 'reader' else
                    [head, 16, *addresses[1:]] if role == 'writer' else [16])
            descriptor = operations.KernelDescriptor(kernel_source=sources[role],
                source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=arguments, config=configs[role])
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=descriptors, cbs=buffers)
    return program
