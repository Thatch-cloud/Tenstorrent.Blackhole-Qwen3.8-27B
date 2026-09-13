"""Synthetic ladder SDPA adapter; no production admission or history truncation."""

from dspark_full_attention import validate_inputs
from dspark_ladder_geometry import geometry


def adapter(context, output_tokens=1024):
    fixture = geometry(context, output_tokens)

    def execute(operations, mesh, query, key, value, mask, owned, *, context_rows,
            proposals, mask_validated=False):
        if (mesh is None or not isinstance(owned, list) or context_rows != fixture['capacity']
                or proposals != fixture['proposals']):
            raise ValueError('Exact ladder capacity and fifteen-query fixture required')
        validate_inputs(operations, query, key, value, mask, context_rows, proposals, mask_validated)
        if key.shape[2] != fixture['storage_keys']:
            raise ValueError('Exact full-history storage keys required')

        def retain(tensor):
            owned.append(tensor)
            return tensor

        padding = fixture['extra_masked_keys']
        if padding:
            key_padding = [(0, 0), (0, 0), (0, padding), (0, 0)]
            key = retain(operations.pad(key, key_padding, 8192.))
            value = retain(operations.pad(value, key_padding, -8192.))
            mask = retain(operations.pad(mask, [(0, 0), (0, 0), (0, 0), (0, padding)], float('-inf')))
        kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        program = operations.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8),
            q_chunk_size=32, k_chunk_size=fixture['key_chunk'], exp_approx_mode=False)
        return retain(operations.transformer.scaled_dot_product_attention(query, key, value,
            attn_mask=mask, is_causal=False, scale=128 ** -.5, program_config=program,
            compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))

    return execute
