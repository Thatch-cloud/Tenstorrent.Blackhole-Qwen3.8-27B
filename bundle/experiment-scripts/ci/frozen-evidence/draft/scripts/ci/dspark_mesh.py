"""Device-only TP2 handoffs for complete padded DSpark layer partials; no host arithmetic."""

from dspark_projection import require_tensor
from projection_link_policy import projection_links


def gather_partials(operations, mesh, collectives, partial, retain):
    if list(mesh.shape)!=[1,2] or not callable(retain):
        raise ValueError('Explicit two-chip mesh and caller-owned collective tensors required')
    require_tensor(operations,partial,(1,1,32,5120),operations.float32)
    gathered = retain(operations.experimental.all_gather_async(partial,persistent_output_buffer=None,dim=0,
        multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(),num_links=projection_links(),
        memory_config=operations.DRAM_MEMORY_CONFIG,topology=operations.Topology.Linear,
        chunks_per_sync=10,num_workers_per_link=2,num_buffers_per_channel=2))
    require_tensor(operations,gathered,(2,1,32,5120),operations.float32)
    return tuple(retain(operations.slice(gathered,(chip,0,0,0),(chip+1,1,32,5120))) for chip in range(2))
