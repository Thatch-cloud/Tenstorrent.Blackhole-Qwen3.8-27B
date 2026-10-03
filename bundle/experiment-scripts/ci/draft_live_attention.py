"""T8 attention integration of the simulator- and hardware-qualified live-query kernel."""

from draft_attention import validate_attention
from draft_dot import fused_dot
from draft_live_qk import live_qk
from draft_row_sum import row_sum


def live_attention(operations, mesh, query, key, value, mask, *, trace_owned, mask_validated=False):
    if mask_validated is not True or not isinstance(trace_owned, list):
        raise ValueError('Validated T8 single-key padding mask and explicit trace ownership required')
    validate_attention(operations, query, key, value, mask)
    owned = []

    def retain(tensor):
        owned.append(tensor)
        return tensor

    try:
        keys = retain(operations.repeat_interleave(key, 4, dim=1, memory_config=operations.DRAM_MEMORY_CONFIG))
        values = retain(operations.repeat_interleave(value, 4, dim=1, memory_config=operations.DRAM_MEMORY_CONFIG))
        query, keys, values = [retain(operations.typecast(tensor, operations.float32)) for tensor in (query, keys, values)]
        scores = live_qk(mesh, query, keys, owned)
        scaled = retain(operations.multiply(scores, 128 ** -.5, dtype=operations.float32))
        wide_mask = retain(operations.typecast(mask, operations.float32))
        masked = retain(operations.add(scaled, wide_mask, dtype=operations.float32))
        maximum = retain(operations.max(masked, dim=-1, keepdim=True))
        centered = retain(operations.subtract(masked, maximum, dtype=operations.float32))
        exponentials = retain(operations.exp(centered, fast_and_approximate_mode=False))
        total = row_sum(mesh, exponentials, owned)
        inverse = retain(operations.reciprocal(total))
        probabilities = retain(operations.multiply(exponentials, inverse, dtype=operations.float32))
        transposed = retain(operations.transpose(values, -1, -2))
        return fused_dot(mesh, probabilities, transposed, owned, cache_tiles=True)
    finally:
        trace_owned.extend(owned)
