"""Unqualified complete DSpark proposal composition using borrowed target embeddings and LM head."""

from dspark_inputs import PROPOSALS, VOCABULARY
from dspark_markov_device import execute as markov
from dspark_pipeline import INPUTS, execute as backbone
from dspark_projection import require_tensor
from dspark_vocabulary import shared_head_logits
from projection_link_policy import projection_links


def noise_embeddings(operations, target, mesh, collectives, identifiers, retain):
    if (list(mesh.shape)!=[1,2] or target.mesh_device is not mesh or target.num_devices!=2
            or target.vocab_size!=VOCABULARY or not callable(retain)
            or tuple(identifiers.shape)!=(1,PROPOSALS) or identifiers.dtype!=operations.uint32
            or identifiers.layout!=operations.ROW_MAJOR_LAYOUT
            or identifiers.memory_config()!=operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Borrowed TP2 target and seven validated device query identifiers required')
    local = retain(target.embd(identifiers,memory_config=operations.DRAM_MEMORY_CONFIG))
    local = retain(operations.reshape(local,(1,1,PROPOSALS,2560)))
    require_tensor(operations,local,(1,1,PROPOSALS,2560),operations.bfloat16)
    hidden = retain(operations.experimental.all_gather_async(local,persistent_output_buffer=None,dim=3,
        multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(),num_links=projection_links(),
        memory_config=operations.DRAM_MEMORY_CONFIG,topology=operations.Topology.Linear,
        chunks_per_sync=10,num_workers_per_link=2,num_buffers_per_channel=2))
    require_tensor(operations,hidden,(1,1,PROPOSALS,5120),operations.bfloat16)
    padded = retain(operations.pad(hidden,[(0,0),(0,0),(0,32-PROPOSALS),(0,0)],0.0))
    require_tensor(operations,padded,(1,1,32,5120),operations.bfloat16)
    return padded


def propose(operations, target, mesh, collectives, identifiers, anchor, inputs, parameters,
        layer_weights, predecessor, successor, owned, *, inputs_validated=False):
    if (inputs_validated is not True or set(inputs)!=set(INPUTS)-{'noise'} or not isinstance(owned,list)
            or getattr(target,'_lmhead_vocab_sharded',False) is not True):
        raise ValueError('Validated anchor/query/mask contract, complete feature history and caller-owned proposal tensors required')

    def retain(value):
        owned.append(value)
        return value

    noise = noise_embeddings(operations,target,mesh,collectives,identifiers,retain)
    learned = backbone(operations,mesh,collectives,dict(inputs,noise=noise),parameters,layer_weights,retain,mask_validated=True)
    logits = shared_head_logits(operations,target,mesh,collectives,learned['backbone']['final_norm'],retain)
    records = markov(operations,anchor,logits['base_logits'],predecessor,successor,owned)
    if len(records)!=PROPOSALS:
        raise ValueError('All seven DSpark query rows must produce proposals, including row zero')
    return dict(noise=noise,learned=learned,logits=logits,records=records)
