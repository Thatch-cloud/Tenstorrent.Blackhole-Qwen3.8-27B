"""Prepared learned attention branch for device-resident draft-stack execution."""

from draft_attention import composed_draft_attention, draft_sdpa
from draft_convolution import grouped_causal_convolution
from feature_collective import gather_add_projection


def prepare_attention_branch(operations, mesh, weights, convolution, retain, *, precise_native=False):
    import torch

    native_kernel = None
    if precise_native:
        import os
        from native_draft_sdpa import audit_active_kernel

        native_kernel = audit_active_kernel(os.environ['TT_METAL_HOME'])

    def upload(value, *, sharded=False, row_major=False):
        return retain(operations.from_torch(value, device=mesh, dtype=operations.bfloat16,
            layout=operations.ROW_MAJOR_LAYOUT if row_major else operations.TILE_LAYOUT,
            memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ShardTensorToMesh(mesh, dim=0) if sharded else operations.ReplicateTensorToMesh(mesh)))

    def projection(name, dimension):
        parts = [part.T.contiguous() for part in weights[f'layers.0.self_attn.{name}_proj.weight'].chunk(2, dim=dimension)]
        return upload(torch.cat(parts, dim=0), sharded=True)

    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    base = convolution['layers.0.attention_conv.base_kernel']
    return dict(operations=operations, mesh=mesh, source_weights=weights, source_convolution=convolution,
        kernel=kernel, native_kernel=native_kernel,
        norm=upload(convolution['layers.0.input_layernorm.weight'].reshape(1, 1, 160, 32), row_major=True),
        convolution=upload(convolution['layers.0.attention_conv.kernel_projection.weight'].T.contiguous()),
        bases=[upload(base[phase, offset].reshape(1, 1, 1, 5120)) for phase in range(2) for offset in range(2)],
        projections={name: projection(name, 0) for name in ('q', 'k', 'v')},
        head_norms={name: upload(weights[f'layers.0.self_attn.{name}_norm.weight'].reshape(1, 1, 4, 32), row_major=True)
                    for name in ('q', 'k')}, output_projection=projection('o', 1))


def execute_attention_branch(operations, mesh, collectives, hidden, history, mask, rope, retain, *,
                             parameters, context, wide_dot_placement=False):
    if parameters['operations'] is not operations or parameters['mesh'] is not mesh:
        raise ValueError('Prepared attention parameters belong to another mesh or runtime')
    if parameters.get('native_kernel') and wide_dot_placement:
        raise ValueError('Native SDPA replaces the composed dot placement control')
    if type(context) is not int or context < 1 or context > 2048:
        raise ValueError('Explicit bounded committed feature context required')
    key_rows = ((context + 8 + 31) // 32) * 32
    if (tuple(hidden.shape) != (1, 1, 32, 5120) or tuple(history.shape) != (1, 1, key_rows, 5120)
            or hidden.dtype != operations.bfloat16 or history.dtype != operations.bfloat16
            or tuple(mask.shape) != (1, 1, 32, key_rows) or mask.dtype != operations.bfloat16):
        raise ValueError('Padded BF16 proposal, context and mask geometry required')
    if set(rope) != {'q', 'k'} or any(len(rope[name]) != 2 or any(
            tuple(table.shape) != (1, 1, 32 if name == 'q' else key_rows, 128)
            or table.dtype != operations.bfloat16 for table in rope[name]) for name in ('q', 'k')):
        raise ValueError('Caller-owned BF16 position tables required for query and keys')
    kernel = parameters['kernel']

    def project(value, weight, grid, rows, columns):
        program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid,
            in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32,
            per_core_N=columns, fuse_batch=True, fused_activation=None, mcast_in0=True)
        return retain(operations.matmul(value, weight, dtype=operations.float32,
            compute_kernel_config=kernel, program_config=program, memory_config=operations.DRAM_MEMORY_CONFIG))

    normalized = retain(operations.rms_norm(hidden, epsilon=1e-6, weight=parameters['norm'],
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    projected = project(normalized, parameters['convolution'], (8, 5), 32, 1)
    rounded = retain(operations.typecast(projected, operations.bfloat16))
    dynamic = [retain(operations.slice(rounded, (0, 0, 0, offset * 320), (1, 1, 32, (offset + 1) * 320)))
        for offset in range(4)]
    prepared = retain(grouped_causal_convolution(operations, mesh, normalized, dynamic[:2], parameters['bases'][:2],
        fp32_intermediates=True, retain_temporaries=retain))
    context_input = retain(operations.slice(history, (0, 0, 0, 0), (1, 1, context, 5120)))
    proposal_input = retain(operations.slice(prepared, (0, 0, 0, 0), (1, 1, 8, 5120)))
    parts = [context_input, proposal_input]
    if key_rows > context + 8:
        zeros = retain(operations.zeros_like(history))
        parts.append(retain(operations.slice(zeros, (0, 0, 0, 0), (1, 1, key_rows - context - 8, 5120))))
    keys = retain(operations.concat(parts, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
    heads = {}
    for name, count in (('q', 16), ('k', 4), ('v', 4)):
        rows = 32 if name == 'q' else key_rows
        projection = project(prepared if name == 'q' else keys, parameters['projections'][name], (8, 8), rows, 1)
        rounded = retain(operations.typecast(projection, operations.bfloat16))
        reshaped = retain(operations.reshape(rounded, (1, rows, count, 128)))
        heads[name] = retain(operations.transpose(reshaped, 1, 2))
        if name != 'v':
            norm = retain(operations.rms_norm(heads[name], epsilon=1e-6, weight=parameters['head_norms'][name],
                compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
            wide = [retain(operations.typecast(value, operations.float32)) for value in (norm, *rope[name])]
            rotated = retain(operations.experimental.rotary_embedding_hf(*wide, is_decode_mode=False,
                compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
            heads[name] = retain(operations.typecast(rotated, operations.bfloat16))
    attention_owned = []
    try:
        if parameters.get('native_kernel'):
            attention = draft_sdpa(operations, heads['q'], heads['k'], heads['v'], mask)
            attention_owned.append(attention)
        else:
            attention = composed_draft_attention(operations, mesh, heads['q'], heads['k'], heads['v'], mask,
                explicit_softmax=True, fused_row_sum=True, fused_dots=True, cache_dot_tiles=True,
                wide_dot_placement=wide_dot_placement, trace_owned=attention_owned)
    finally:
        for value in attention_owned:
            retain(value)
    rounded = retain(operations.typecast(attention, operations.bfloat16))
    transposed = retain(operations.transpose(rounded, 1, 2))
    merged = retain(operations.reshape(transposed, (1, 1, 32, 2048)))
    partial = project(merged, parameters['output_projection'], (8, 10), 32, 2)
    reduced = retain(gather_add_projection(operations, mesh, collectives, partial, retain_temporaries=retain))
    rounded = retain(operations.typecast(reduced, operations.bfloat16))
    finished = retain(grouped_causal_convolution(operations, mesh, rounded, dynamic[2:], parameters['bases'][2:],
        fp32_intermediates=True, retain_temporaries=retain))
    wide = [retain(operations.typecast(value, operations.float32)) for value in (finished, hidden)]
    summed = retain(operations.add(*wide, dtype=operations.float32))
    return retain(operations.typecast(summed, operations.bfloat16))
