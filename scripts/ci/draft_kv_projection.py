"""Learned BF16 K/V row projection for an opt-in historical draft cache."""

from draft_head_layout import split_projected_heads


def project_key_value(operations, inputs, query, cosine_sine, retain, *, parameters):
    rows = inputs.shape[2] if len(inputs.shape) == 4 else 0
    if (parameters.get('operations') is not operations or parameters.get('native_head_layout') is not True
            or rows not in range(32, 2081, 32) or tuple(inputs.shape) != (1, 1, rows, 5120)
            or tuple(query.shape) != (1, 1, 32, 2048) or len(cosine_sine) != 2
            or any(tuple(table.shape) != (1, 1, rows, 128) for table in cosine_sine)
            or any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT
                or value.memory_config() != operations.DRAM_MEMORY_CONFIG for value in (inputs, query, *cosine_sine))):
        raise ValueError('Owned BF16 tiled K/V rows, explicit absolute rotary tables and native head parameters required')
    kernel = parameters['kernel']
    program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 8),
        in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32,
        per_core_N=1, fuse_batch=True, fused_activation=None, mcast_in0=True)
    flat = {}
    for name in ('k', 'v'):
        projected = retain(operations.matmul(inputs, parameters['projections'][name], dtype=operations.float32,
            compute_kernel_config=kernel, program_config=program, memory_config=operations.DRAM_MEMORY_CONFIG))
        flat[name] = retain(operations.typecast(projected, operations.bfloat16))
    heads = split_projected_heads(operations, query, flat['k'], flat['v'], retain)
    normalized = retain(operations.rms_norm(heads['k'], epsilon=1e-6, weight=parameters['head_norms']['k'],
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    wide = [retain(operations.typecast(value, operations.float32)) for value in (normalized, *cosine_sine)]
    rotated = retain(operations.experimental.rotary_embedding_hf(*wide, is_decode_mode=False,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    return dict(q=heads['q'], k=retain(operations.typecast(rotated, operations.bfloat16)), v=heads['v'])
