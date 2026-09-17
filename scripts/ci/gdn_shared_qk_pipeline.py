"""Three-stage shared-Q/K GDN programs on caller-owned persistent tensors."""

from gdn_shared_qk_program import build as build_normalization
from gdn_shared_qk_recurrence import load_kernels, recurrence_spec
from gdn_vsplit import cb_plan, core_coordinates, runtime_args, stage_spec, build_program
from gdn_vsplit_norm_batch import validate_runtime


def build_recurrence(ttnn, mesh, shards, kernels):
    stage, rows, prefetch_inputs = 'recurrence', 16, False
    spec = recurrence_spec(stage_spec(stage, rows))
    grid = mesh.compute_with_storage_grid_size()
    coordinates = core_coordinates(grid.x, grid.y, spec['workers'])
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*point), ttnn.CoreCoord(*point))
                               for point in coordinates])
    buffers = []
    io, fp32 = cb_plan(stage, prefetch_inputs=prefetch_inputs)
    for counts, dtype, page in ((io, ttnn.bfloat16, 2048), (fp32, ttnn.float32, 4096)):
        for index, count in counts.items():
            buffers.append(ttnn.CBDescriptor(total_size=count * page, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))]))
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [value[chip] for value in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != len(addresses):
            raise ValueError('Prototype inputs and outputs must not alias')
        descriptors = []
        configs = dict(
            reader=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                     noc=ttnn.NOC.RISCV_1_default),
            writer=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default),
            compute=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, math_approx_mode=False))
        for role in ('reader', 'writer', 'compute'):
            args = list(spec[role])
            for index in spec.get(role + '_accessors', []):
                args.extend(ttnn.TensorAccessorArgs(local[index]).get_compile_time_args())
            runtime = ttnn.RuntimeArgs()
            for worker, (horizontal, vertical) in enumerate(coordinates):
                runtime[horizontal][vertical] = runtime_args(spec, role, worker, rows, addresses)
            descriptor = ttnn.KernelDescriptor(kernel_source=kernels[stage][role],
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=args, config=configs[role])
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
            kernels=descriptors, cbs=buffers)
    return program



def build(operations, mesh, tensors, *, root):
    if len(tensors) != 11:
        raise ValueError("Eleven distinct persistent recurrence/normalization tensors required")
    validate_runtime(root)
    packed, beta, gate, initial, bridge, states, z, weights, output, query, key = tensors
    expected = ((1, 16, 5120), (1, 16, 24), (1, 16, 24), (1, 24, 128, 128),
        (16, 1, 96, 32), (16, 24, 128, 128), (1, 16, 3072), (1, 1, 128),
        (1, 16, 3072), (1, 16, 1024), (1, 16, 1024))
    for index, (tensor, shape) in enumerate(zip(tensors, expected, strict=True)):
        dtype = operations.float32 if index in (4, 9, 10) else operations.bfloat16
        layout = operations.ROW_MAJOR_LAYOUT if index == 4 else operations.TILE_LAYOUT
        memory = operations.L1_MEMORY_CONFIG if index == 8 else operations.DRAM_MEMORY_CONFIG
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.layout != layout or tensor.memory_config() != memory:
            raise ValueError("Unexpected persistent operand geometry, dtype, layout or placement")
    normalization = build_normalization(operations, mesh, packed, query, key, root=root)
    shards = [operations.get_device_tensors(tensor) for tensor in tensors]
    if any(len(values) != 2 for values in shards):
        raise ValueError("Every persistent tensor must cover both chips")
    kernels = load_kernels(root)
    recurrence = build_recurrence(operations, mesh, shards, kernels)
    norm_gate = build_program(operations, mesh, shards, kernels, "norm_gate", 16)
    return (([packed, query, key], normalization), (tensors, recurrence), (tensors, norm_gate))
