"""Device-packed parallel groups for one stream; caller owns all temporaries."""

from attention_fold_dma import device_layout_dma


def execute(mesh, operations, query, keys, values, metadata, owned, *, scale, memory_config):
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
            selected = operations.slice(result, (0, index, 0, 0), (1, index + 1, count * 12, 256),
                memory_config=operations.DRAM_MEMORY_CONFIG) if len(bundle) > 1 else result
            owned.append(selected)
            chunks.append(device_layout_dma(mesh, selected, count, owned, inverse=True))
    if not chunks:
        raise ValueError('Complete nonempty group metadata required')
    output = operations.concat(chunks, dim=1, memory_config=memory_config)
    owned.append(output)
    return output
