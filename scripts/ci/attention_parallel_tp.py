"""attention_parallel.execute (device-packed parallel groups for one stream) at any served width.

attention_parallel.py is frozen-recipe evidence and slices each bundle entry's result as `count * 12` folded rows.
This twin is the same function with the head rows per token from tp_shapes (6 at four cards) and the fold launches
of attention_fold_dma_tp. tp_addresses.install() rebinds attention_parallel.execute to it at QWEN_FAST_TP=4 only.
"""

from attention_fold_dma_tp import device_layout_dma
import tp_shapes


def execute(mesh, operations, query, keys, values, metadata, owned, *, scale, memory_config):
    head_rows = tp_shapes.active().attn_fold_rows
    chunks = []
    for bundle, pages, mask, config in metadata:
        if not 1 <= len(bundle) <= 3 or not 1 <= bundle[0]['rows'] <= 8:
            raise ValueError('At most three groups of up to eight queries required')
        count = bundle[0]['rows']
        if any(group['rows'] != count or group['signature'] != bundle[0]['signature'] for group in bundle):
            raise ValueError('Parallel groups must share shape and native chunk workload')
        packed = [device_layout_dma(mesh, query, count, owned, offset=group['offset']) for group in bundle]
        stacked = operations.concat(packed, dim=1, memory_config=operations.DRAM_MEMORY_CONFIG) if len(bundle) > 1 else packed[0]
        owned.append(stacked)
        result = operations.transformer.paged_scaled_dot_product_attention_decode(stacked, keys, values,
            page_table_tensor=pages, is_causal=False, attn_mask=mask, scale=scale,
            program_config=config, memory_config=memory_config)
        owned.append(result)
        for index in range(len(bundle)):
            selected = operations.slice(result, (0, index, 0, 0), (1, index + 1, count * head_rows, 256),
                memory_config=operations.DRAM_MEMORY_CONFIG) if len(bundle) > 1 else result
            owned.append(selected)
            chunks.append(device_layout_dma(mesh, selected, count, owned, inverse=True))
    if not chunks:
        raise ValueError('Complete nonempty group metadata required')
    output = operations.concat(chunks, dim=1, memory_config=memory_config)
    owned.append(output)
    return output
