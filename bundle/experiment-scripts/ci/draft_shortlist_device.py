"""Experimental replicated MTP draft head; requires simulator qualification."""

from draft_vocabulary import materialize_head


def prepare_head(operations, mesh, weight, token_ids, owned):
    token_ids = tuple(token_ids)
    if (mesh.get_num_devices() != 2 or tuple(weight.shape) != (248320, 5120)
            or len(token_ids) not in (32768, 65536)):
        raise ValueError('Pinned TP2 target head and 32K or 64K draft shortlist required')
    packed, mapping = materialize_head(weight, token_ids)
    head = operations.from_torch(
        packed.T.contiguous().reshape(1, 1, 5120, len(token_ids)),
        device=mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
        memory_config=operations.DRAM_MEMORY_CONFIG,
        mesh_mapper=operations.ReplicateTensorToMesh(mesh))
    owned.append(head)
    identifiers = operations.from_torch(
        mapping.reshape(1, 1, 1, -1), device=mesh, dtype=operations.uint32,
        layout=operations.ROW_MAJOR_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
        mesh_mapper=operations.ReplicateTensorToMesh(mesh))
    owned.append(identifiers)
    return head, identifiers


def select_token(operations, hidden, head, identifiers, owned):
    width = head.shape[-1]
    if (tuple(hidden.shape) != (1, 1, 1, 5120)
            or tuple(head.shape) != (1, 1, 5120, width) or width not in (32768, 65536)
            or tuple(identifiers.shape) != (1, 1, 1, width)
            or any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT
                   for value in (hidden, head))
            or identifiers.dtype != operations.uint32 or identifiers.layout != operations.ROW_MAJOR_LAYOUT
            or any(value.memory_config() != operations.DRAM_MEMORY_CONFIG
                   for value in (hidden, head, identifiers))):
        raise ValueError('Replicated single-row MTP hidden, prepared head and global-ID map required')
    logits = operations.linear(hidden, head, memory_config=operations.DRAM_MEMORY_CONFIG)
    owned.append(logits)
    row_logits = operations.to_layout(logits, operations.ROW_MAJOR_LAYOUT)
    if row_logits is not logits:
        owned.append(row_logits)
    indices = operations.argmax(row_logits, dim=-1, keepdim=True)
    owned.append(indices)
    tokens = operations.gather(identifiers, dim=-1, index=indices)
    owned.append(tokens)
    return tokens
