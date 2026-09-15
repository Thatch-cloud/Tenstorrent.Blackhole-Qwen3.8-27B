"""Unqualified ladder adapter around the unchanged simulator-admitted split-K kernel."""

from dspark_full_attention import validate_inputs
from dspark_splitk_attention import execute_folded
from matched_context_geometry import geometry


def adapter(context):
    plan = geometry(context)

    def execute(operations, mesh, query, key, value, mask, owned):
        if mesh is None or not isinstance(owned, list):
            raise ValueError('Explicit mesh and owned intermediates required')
        validate_inputs(operations, query, key, value, mask, plan['capacity'], 15, True)
        if key.shape[2] != plan['storage_keys'] or value.shape[2] != plan['storage_keys']:
            raise ValueError('Complete context-specific history and proposal storage required')
        padding = plan['extra_masked_keys']
        if padding:
            key = operations.pad(key, [(0, 0), (0, 0), (0, padding), (0, 0)], 8192.)
            owned.append(key)
            value = operations.pad(value, [(0, 0), (0, 0), (0, padding), (0, 0)], -8192.)
            owned.append(value)
            mask = operations.pad(mask, [(0, 0), (0, 0), (0, 0), (0, padding)], float('-inf'))
            owned.append(mask)
        return execute_folded(operations, query, key, value, mask, owned,
            key_chunk_size=256, max_cores_per_head=8, stripe_keys=False, fp32_dest_acc=True)

    return execute
