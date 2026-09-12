"""Single-dispatch grouped causal convolution with explicit BF16 rounding boundaries."""

from pathlib import Path

from draft_convolution import validate_shapes


def checked_convolution(operations, mesh, hidden, dynamic, base, *, fp32_intermediates=False,
                        retain_temporaries=None, audit=False, checks=None, context=None):
    if fp32_intermediates is not True or not callable(retain_temporaries) or type(audit) is not bool:
        raise ValueError('Fused request convolution requires exact FP32 arithmetic and an explicit lifetime owner')
    if audit and (not isinstance(checks, list) or not isinstance(context, dict)):
        raise ValueError('Audited convolution requires an owned check log and call context')
    output = fused_convolution(operations, mesh, hidden, dynamic, base)
    retain_temporaries(output)
    if audit:
        import torch
        from draft_convolution import grouped_causal_convolution

        control = grouped_causal_convolution(operations, mesh, hidden, dynamic, base, fp32_intermediates=True)
        try:
            for chip, (candidate, reference) in enumerate(zip(operations.get_device_tensors(output),
                    operations.get_device_tensors(control), strict=True)):
                exact = torch.equal(operations.to_torch(candidate).view(torch.int16),
                    operations.to_torch(reference).view(torch.int16))
                checks.append(dict(**context, chip=chip, rows=hidden.shape[2], exact=exact))
                if not exact:
                    raise AssertionError('Fused learned convolution differs from the composed BF16-rounding path')
        finally:
            operations.deallocate(control)
    return output


def fused_convolution(operations, mesh, hidden, dynamic, base):
    rows = validate_shapes(hidden, dynamic, base)
    tensors = [hidden, *dynamic, *base]
    if list(mesh.shape) != [1, 2] or any(value.dtype != operations.bfloat16
            or value.layout != operations.TILE_LAYOUT or value.memory_config() != operations.DRAM_MEMORY_CONFIG
            for value in tensors):
        raise ValueError('Two-chip interleaved DRAM BF16 convolution operands required')
    parts = [operations.get_device_tensors(value) for value in tensors]
    if any(len(shards) != 2 for shards in parts):
        raise ValueError('Both operand shards required')
    output = operations.empty(tuple(hidden.shape), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
        device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    tensors.append(output)
    parts.append(operations.get_device_tensors(output))
    cores = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0), operations.CoreCoord(7, 9))])
    buffers = [operations.CBDescriptor(total_size=2048 * count, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
            page_size=2048, tile=operations.TileDescriptor(operations.Tile([32, 32])))])
        for index, count in ((0, 7), (1, 2), (16, 1))]
    compute = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_convolution_fused_compute.cpp')),
        core_ranges=cores, compile_time_args=[2], config=operations.ComputeConfigDescriptor(
            math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False))
    program = operations.MeshProgramDescriptor()
    try:
        for chip in range(2):
            local = [shards[chip] for shards in parts]
            if local[-1].buffer_address() in {value.buffer_address() for value in local[:-1]}:
                raise ValueError('Convolution output must not alias borrowed inputs')
            runtime = operations.RuntimeArgs()
            for worker in range(80):
                runtime[worker % 8][worker // 8] = [value.buffer_address() for value in local] + [rows, worker]
            reader = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_convolution_fused_io.cpp')),
                core_ranges=cores, compile_time_args=[argument for value in local
                    for argument in operations.TensorAccessorArgs(value).get_compile_time_args()], runtime_args=runtime,
                config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                    noc=operations.NOC.RISCV_0_default))
            coordinate = operations.MeshCoordinate(0, chip)
            program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[reader, compute], cbs=buffers)
        operations.generic_op(tensors, program)
    except BaseException:
        operations.deallocate(output)
        raise
    return output
