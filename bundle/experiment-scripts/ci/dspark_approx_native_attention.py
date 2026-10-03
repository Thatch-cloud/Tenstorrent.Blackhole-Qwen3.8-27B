"""Unqualified approximate-exp proposal attention; never target attention."""

from dspark_full_attention import validate_inputs


def execute(operations, mesh, query, key, value, mask, owned, *, context_rows,
        proposals, mask_validated=False):
    if mesh is None or not isinstance(owned, list):
        raise ValueError('Explicit mesh and caller-owned attention output required')
    validate_inputs(operations, query, key, value, mask, context_rows, proposals, mask_validated)
    if key.shape[2] % 64:
        raise ValueError('Aligned 64-key proposal chunks required')
    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    program = operations.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8),
        q_chunk_size=32, k_chunk_size=64, exp_approx_mode=True)
    output = operations.transformer.scaled_dot_product_attention(query, key, value,
        attn_mask=mask, is_causal=False, scale=128 ** -0.5, program_config=program,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG)
    owned.append(output)
    return output
