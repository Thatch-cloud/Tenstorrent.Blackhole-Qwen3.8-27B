"""Synthetic-only 256-key draft trial; preserve all live operands and poison new padding."""

from dspark_full_attention import validate_inputs


KEY_CHUNK = 256
PADDED_KEYS = 8704


def execute(operations, mesh, query, key, value, mask, owned, *, context_rows,
        proposals, mask_validated=False):
    if mesh is None or not isinstance(owned, list) or context_rows != 8448 or proposals != 15:
        raise ValueError('Explicit synthetic 8K mesh and owned outputs required')
    validate_inputs(operations, query, key, value, mask, context_rows, proposals, mask_validated)
    if key.shape[2] != 8512:
        raise ValueError('Original fixed-storage operand shape required')

    def retain(tensor):
        owned.append(tensor)
        return tensor

    key_padding = [(0, 0), (0, 0), (0, PADDED_KEYS - 8512), (0, 0)]
    key = retain(operations.pad(key, key_padding, 8192.))
    value = retain(operations.pad(value, key_padding, -8192.))
    mask = retain(operations.pad(mask, [(0, 0), (0, 0), (0, 0), (0, PADDED_KEYS - 8512)], float('-inf')))
    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    program = operations.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8),
        q_chunk_size=32, k_chunk_size=KEY_CHUNK, exp_approx_mode=False)
    return retain(operations.transformer.scaled_dot_product_attention(query, key, value,
        attn_mask=mask, is_causal=False, scale=128 ** -.5, program_config=program,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
