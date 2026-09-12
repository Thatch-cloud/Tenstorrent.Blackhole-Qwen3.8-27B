"""Device-resident learned layer-zero MLP branch and independent stage checks."""

from draft_convolution import grouped_causal_convolution, convolution_reference
from draft_mlp import split_mlp_weights, swiglu_device, swiglu_reference
from feature_collective import gather_add_projection
from feature_normalization import bf16_ulp_distance, rms_reference
from projection_rounding import grouped_projection_reference


def prepare_mlp_branch(operations, mesh, weights, convolution, retain):
    import torch

    shards = split_mlp_weights(*(weights[f'layers.0.mlp.{name}_proj.weight'] for name in ('gate', 'up', 'down')))
    norm_weight = convolution['layers.0.post_attention_layernorm.weight']
    conv_weight = convolution['layers.0.mlp_conv.kernel_projection.weight'].T.contiguous()
    base_weight = convolution['layers.0.mlp_conv.base_kernel']

    def upload(value, sharded=False, layout=None):
        return retain(operations.from_torch(value, device=mesh, dtype=operations.bfloat16,
            layout=operations.TILE_LAYOUT if layout is None else layout, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ShardTensorToMesh(mesh, dim=0) if sharded else operations.ReplicateTensorToMesh(mesh)))

    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    return dict(operations=operations, mesh=mesh, source_weights=weights, source_convolution=convolution,
        shards=shards, norm_weight=norm_weight, conv_weight=conv_weight, base_weight=base_weight, kernel=kernel,
        device_norm=upload(norm_weight.reshape(1, 1, 160, 32), layout=operations.ROW_MAJOR_LAYOUT),
        device_conv=upload(conv_weight),
        bases=[upload(base_weight[phase, offset].reshape(1, 1, 1, 5120)) for phase in range(2) for offset in range(2)],
        device_projections=[upload(torch.cat([rank[index] for rank in shards], dim=0), True) for index in range(3)])


def execute_mlp_branch(operations, mesh, collectives, hidden, weights, convolution, retain, *, parameters=None,
                       trace_safe=False, convolution_operation=None):
    convolve = convolution_operation or grouped_causal_convolution
    if tuple(hidden.shape) != (1, 1, 32, 5120) or hidden.dtype != operations.bfloat16:
        raise ValueError('A padded32-row BF16 hidden block is required')
    if parameters is None:
        parameters = prepare_mlp_branch(operations, mesh, weights, convolution, retain)
    if (parameters['operations'] is not operations or parameters['mesh'] is not mesh
            or parameters['source_weights'] is not weights or parameters['source_convolution'] is not convolution):
        raise ValueError('Prepared MLP parameters belong to a different mesh or learned layer')
    kernel = parameters['kernel']
    ownership = dict(retain_temporaries=retain) if trace_safe else {}

    def project(value, weight, grid, columns):
        program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid,
            in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=columns,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        return retain(operations.matmul(value, weight, dtype=operations.float32, program_config=program,
            compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))

    normalized = retain(operations.rms_norm(hidden, epsilon=1e-6, weight=parameters['device_norm'],
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    projected = project(normalized, parameters['device_conv'], (8, 5), 1)
    rounded = retain(operations.typecast(projected, operations.bfloat16))
    dynamic = [retain(operations.slice(rounded, (0, 0, 0, offset * 320), (1, 1, 32, (offset + 1) * 320)))
        for offset in range(4)]
    bases = parameters['bases']
    prepared = retain(convolve(operations, mesh, normalized, dynamic[:2], bases[:2], fp32_intermediates=True, **ownership))
    projections = [project(prepared, parameters['device_projections'][index], (8, 10), 4)
        for index in range(2)]
    activation = swiglu_device(operations, *projections, retain)
    partial = project(activation, parameters['device_projections'][2], (8, 10), 2)
    reduced = retain(gather_add_projection(operations, mesh, collectives, partial, **ownership))
    rounded_output = retain(operations.typecast(reduced, operations.bfloat16))
    finished = retain(convolve(operations, mesh, rounded_output, dynamic[2:], bases[2:], fp32_intermediates=True, **ownership))
    wide_finished = retain(operations.typecast(finished, operations.float32))
    wide_hidden = retain(operations.typecast(hidden, operations.float32))
    summed = retain(operations.add(wide_finished, wide_hidden, dtype=operations.float32))
    output = retain(operations.typecast(summed, operations.bfloat16))
    return dict(hidden=hidden, normalized=normalized, conv_projection=projected, dynamic=dynamic, prepared=prepared,
        projections=projections, activation=activation, partial=partial, reduced=reduced, finished=finished,
        output=output, shards=parameters['shards'], norm_weight=parameters['norm_weight'],
        conv_weight=parameters['conv_weight'], base_weight=parameters['base_weight'])


def validate_projection(actual, inputs, weight, *, stage, failure_capture=None, checkpoint=None, layer=0, chip=0):
    import torch

    expected = grouped_projection_reference(inputs, weight, destination_rounding=True, fidelity_span=32)
    close = torch.isclose(actual.double(), expected, rtol=1e-4, atol=1e-4)
    if failure_capture is not None and not close.all():
        torch.save(dict(activation=inputs.contiguous().clone(), weight=weight.contiguous().clone(),
            actual=actual.double().clone(), reference=expected.clone(),
            ordinary_reference=inputs.double() @ weight.double(),
            mismatches=(~close).nonzero(), stage=stage,
            chip=chip, checkpoint=checkpoint, layer=layer, fidelity_span=32), failure_capture)
        print(f'Connected MLP {stage} failure capture: {failure_capture}', flush=True)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-4,
        msg=lambda message: f'Layer {layer} chip {chip} {stage}: {message}')
    return float((actual.double() - expected).abs().max())


def validate_mlp_branch(state, host, chip, *, failure_capture=None, checkpoint=None, layer=0):
    import torch

    def exact(actual, expected, stage):
        if not torch.equal(actual, expected):
            raise AssertionError(f'Connected MLP {stage} must be exact')

    def projection(actual, inputs, weight, stage):
        return validate_projection(actual, inputs, weight, stage=stage, failure_capture=failure_capture,
            checkpoint=checkpoint, layer=layer, chip=chip)

    hidden = host(state['hidden'], chip)
    normalized = host(state['normalized'], chip)
    norm_ulps = int(bf16_ulp_distance(normalized, rms_reference(hidden, state['norm_weight'])).max())
    if norm_ulps > 2:
        raise AssertionError('Connected MLP normalization exceeds two BF16 ULPs')
    conv_projection = host(state['conv_projection'], chip)
    conv_error = projection(conv_projection, normalized, state['conv_weight'], 'convolution')
    dynamic = conv_projection.bfloat16().split(320, dim=-1)
    for actual, expected in zip(state['dynamic'], dynamic, strict=True):
        exact(host(actual, chip), expected, 'dynamic kernel')
    bases = state['base_weight']
    prepared = host(state['prepared'], chip)
    exact(prepared, convolution_reference(normalized, dynamic[:2],
        [bases[0, offset].reshape(1, 1, 1, 5120) for offset in range(2)]), 'prepare convolution')
    projections = [host(value, chip) for value in state['projections']]
    projection_errors = [projection(value[..., :8, :], prepared[..., :8, :], state['shards'][chip][index], ('gate', 'up')[index])
        for index, value in enumerate(projections)]
    activation = host(state['activation'], chip)
    activation_ulps = int(bf16_ulp_distance(activation, swiglu_reference(*projections)).max())
    if activation_ulps > 2:
        raise AssertionError('Connected MLP activation exceeds two BF16 ULPs')
    down_error = projection(host(state['partial'], chip)[..., :8, :], activation[..., :8, :], state['shards'][chip][2], 'down')
    reduced = host(state['reduced'], chip)
    exact(reduced, host(state['partial'], 0) + host(state['partial'], 1), 'fabric sum')
    finished = host(state['finished'], chip)
    exact(finished, convolution_reference(reduced.bfloat16(), dynamic[2:],
        [bases[1, offset].reshape(1, 1, 1, 5120) for offset in range(2)]), 'finish convolution')
    exact(host(state['output'], chip), (finished.float() + hidden.float()).bfloat16(), 'residual')
    return dict(chip=chip, stage='connected_mlp', norm_ulps=norm_ulps, activation_ulps=activation_ulps,
        conv_projection_error=conv_error, gate_up_errors=projection_errors, down_error=down_error,
        prepare_exact=True, finish_exact=True, residual_exact=True, sum_exact=True)
