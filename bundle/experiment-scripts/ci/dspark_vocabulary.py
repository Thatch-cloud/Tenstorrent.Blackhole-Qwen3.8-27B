"""Unqualified DSpark target-head bridge: complete TP2 vocabulary, seven rows from row zero, no shortlist."""

from dspark_inputs import PROPOSALS, VOCABULARY
from dspark_projection import require_tensor
from projection_link_policy import projection_links


def gather_logits(operations, mesh, collectives, local_logits, retain):
    if list(mesh.shape)!=[1,2] or not callable(retain):
        raise ValueError('Explicit TP2 mesh and caller-owned logits required')
    require_tensor(operations,local_logits,(1,1,32,VOCABULARY//2),operations.bfloat16)
    gathered = retain(operations.experimental.all_gather_async(local_logits,persistent_output_buffer=None,dim=3,
        multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(),num_links=projection_links(),
        memory_config=operations.DRAM_MEMORY_CONFIG,topology=operations.Topology.Linear,
        chunks_per_sync=10,num_workers_per_link=2,num_buffers_per_channel=2))
    require_tensor(operations,gathered,(1,1,32,VOCABULARY),operations.bfloat16)
    queries = retain(operations.slice(gathered,(0,0,0,0),(1,1,PROPOSALS,VOCABULARY)))
    require_tensor(operations,queries,(1,1,PROPOSALS,VOCABULARY),operations.bfloat16)
    wide = retain(operations.typecast(queries,operations.float32))
    require_tensor(operations,wide,(1,1,PROPOSALS,VOCABULARY),operations.float32)
    return dict(full_logits=gathered,base_logits=wide)


def shared_head_logits(operations, model, mesh, collectives, normalized, retain):
    if (model.num_devices!=2 or model.vocab_size!=VOCABULARY or model._lmhead_vocab_sharded is not True
            or model.mesh_device is not mesh or list(mesh.shape)!=[1,2] or not callable(retain)):
        raise ValueError('Borrowed target TP2 vocabulary head on the exact caller mesh required')
    require_tensor(operations,normalized,(1,1,32,5120),operations.bfloat16)
    local = retain(operations.linear(normalized,model.lm_head_weight))
    result = gather_logits(operations,mesh,collectives,local,retain)
    return dict(local_logits=local,**result)
