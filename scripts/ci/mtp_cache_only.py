"""Native MTP prefix through paged KV publication, without unused decoder outputs."""


def update_cache(operations, mtp, embedding, hidden, positions, cosine, sine, pages, owned):
    attention = mtp.attention
    if (tuple(embedding.shape) != (1, 1, 1, 5120) or tuple(hidden.shape) != tuple(embedding.shape)
            or tuple(positions.shape) != (1,) or not attention.use_paged or not attention._fused_qkv
            or not callable(getattr(attention, '_qkv_raw_decode', None)) or pages is None):
        raise ValueError('Native fused-prep single-row MTP with paged KV required')
    normalized_embedding = operations.rms_norm(embedding, weight=mtp._nw['mtp.pre_fc_norm_embedding.weight'], epsilon=mtp.eps)
    normalized_hidden = operations.rms_norm(hidden, weight=mtp._nw['mtp.pre_fc_norm_hidden.weight'], epsilon=mtp.eps)
    joined = operations.concat([normalized_embedding, normalized_hidden], dim=-1)
    fused = operations.matmul(joined, mtp.w_fc, memory_config=operations.DRAM_MEMORY_CONFIG)
    attention_input = operations.rms_norm(fused, weight=mtp._nw['mtp.layers.0.input_layernorm.weight'], epsilon=mtp.eps)
    owned.extend((normalized_embedding, normalized_hidden, joined, fused, attention_input))
    projected = attention._qkv_raw_decode(attention_input)
    owned.append(projected)
    query, gate, keys, values = operations.transformer.attn_decode_prep(
        projected, cosine, sine, attention.tw['q_norm'], attention.tw['k_norm'],
        attention.NH, attention.NKV, attention.HD, attention.rope_dim, attention._kv_shard_cfg(1),
        batch=1, memory_config=operations.DRAM_MEMORY_CONFIG)
    owned.extend((query, gate, keys, values))
    operations.experimental.paged_update_cache(attention.paged_k, keys, update_idxs_tensor=positions, page_table=pages)
    operations.experimental.paged_update_cache(attention.paged_v, values, update_idxs_tensor=positions, page_table=pages)


def cache_digest(operations, mtp, valid_rows):
    import hashlib
    import torch

    caches = (mtp.attention.paged_k, mtp.attention.paged_v)
    if type(valid_rows) is not int or not 0 < valid_rows <= caches[0].shape[0] * caches[0].shape[2]:
        raise ValueError('Nonempty valid MTP cache prefix required')
    result = dict(valid_rows=valid_rows, keys=[], values=[])
    for name, cache in zip(('keys', 'values'), caches, strict=True):
        parts = operations.get_device_tensors(cache)
        if len(parts) != 2:
            raise ValueError('Both physical chips required for MTP cache equality')
        for part in parts:
            tensor = operations.to_torch(part)
            page_count = (valid_rows + tensor.shape[2] - 1) // tensor.shape[2]
            prefix = tensor[:page_count].permute(0, 2, 1, 3).reshape(-1, tensor.shape[1], tensor.shape[3])[:valid_rows]
            result[name].append(hashlib.sha256(prefix.contiguous().view(torch.uint8).numpy()).hexdigest())
    return result
