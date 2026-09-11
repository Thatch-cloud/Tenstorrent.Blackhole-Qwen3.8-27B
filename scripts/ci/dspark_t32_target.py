"""Experimental 31-query target boundaries; complete vocabulary and row-zero proposal semantics."""

from dspark_t32_inputs import geometry
from dspark_inputs import MASK_TOKEN, VOCABULARY
from dspark_projection import require_tensor
from projection_link_policy import projection_links


def query_inputs(anchor, position, proposals):
    import torch

    geometry(position, proposals)
    if type(anchor) is not int or not 0 <= anchor < VOCABULARY:
        raise ValueError('Global target anchor required')
    identifiers = torch.full((1, proposals), MASK_TOKEN, dtype=torch.int64)
    identifiers[0, 0] = anchor
    return identifiers


def gather(operations, mesh, collectives, value, retain):
    if list(mesh.shape) != [1, 2] or not callable(retain):
        raise ValueError('Explicit two-chip mesh and tensor ownership required')
    return retain(operations.experimental.all_gather_async(value, persistent_output_buffer=None, dim=3,
        multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=projection_links(),
        memory_config=operations.DRAM_MEMORY_CONFIG, topology=operations.Topology.Linear,
        chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2))


def noise_embeddings(operations, target, mesh, collectives, identifiers, retain, *, proposals):
    geometry(1, proposals)
    if (list(mesh.shape) != [1, 2] or target.mesh_device is not mesh or target.num_devices != 2
            or target.vocab_size != VOCABULARY or not callable(retain)
            or tuple(identifiers.shape) != (1, proposals) or identifiers.dtype != operations.uint32
            or identifiers.layout != operations.ROW_MAJOR_LAYOUT
            or identifiers.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Borrowed TP2 target and validated anchor/query identifiers required')
    local = retain(target.embd(identifiers, memory_config=operations.DRAM_MEMORY_CONFIG))
    local = retain(operations.reshape(local, (1, 1, proposals, 2560)))
    require_tensor(operations, local, (1, 1, proposals, 2560), operations.bfloat16)
    hidden = gather(operations, mesh, collectives, local, retain)
    require_tensor(operations, hidden, (1, 1, proposals, 5120), operations.bfloat16)
    output = retain(operations.pad(hidden, [(0, 0), (0, 0), (0, 32 - proposals), (0, 0)], 0.0))
    require_tensor(operations, output, (1, 1, 32, 5120), operations.bfloat16)
    return output


def gather_logits(operations, mesh, collectives, local, retain, *, proposals):
    geometry(1, proposals)
    require_tensor(operations, local, (1, 1, 32, VOCABULARY // 2), operations.bfloat16)
    complete = gather(operations, mesh, collectives, local, retain)
    require_tensor(operations, complete, (1, 1, 32, VOCABULARY), operations.bfloat16)
    queries = retain(operations.slice(complete, (0, 0, 0, 0), (1, 1, proposals, VOCABULARY)))
    output = retain(operations.typecast(queries, operations.float32))
    require_tensor(operations, output, (1, 1, proposals, VOCABULARY), operations.float32)
    return output


def shared_head_logits(operations, target, mesh, collectives, hidden, retain, *, proposals):
    if (target.mesh_device is not mesh or target.num_devices != 2 or target.vocab_size != VOCABULARY
            or getattr(target, '_lmhead_vocab_sharded', False) is not True):
        raise ValueError('Borrowed vocabulary-sharded target head on the exact mesh required')
    require_tensor(operations, hidden, (1, 1, 32, 5120), operations.bfloat16)
    local = retain(operations.linear(hidden, target.lm_head_weight))
    return gather_logits(operations, mesh, collectives, local, retain, proposals=proposals)


def pack_tokens(operations, records, retain):
    if len(records) != 31 or not callable(retain):
        raise ValueError('Every Markov proposal row, including row zero, must be retained')
    values = []
    for record in records:
        token = record['token']
        if (token.dtype != operations.uint32 or token.layout != operations.ROW_MAJOR_LAYOUT
                or token.memory_config() != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('One native uint32 row-major Markov token per query required')
        value = retain(operations.reshape(token, (1, 1, 1, 1)))
        values.append(value)
    while len(values) > 1:
        values = [values[start] if len(values[start:start + 8]) == 1 else
            retain(operations.concat(values[start:start + 8], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
            for start in range(0, len(values), 8)]
    return values[0]
