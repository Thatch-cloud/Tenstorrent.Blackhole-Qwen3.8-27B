"""Pinned native TP2 reduce-scatter without the wrapper's forced deallocation of borrowed MLP output."""


def reduce_partial(operations, mesh, collectives, partial, topology):
    if (list(mesh.shape) != [1, 2] or tuple(partial.shape) != (1, 1, 8, 5120)
            or partial.dtype != operations.bfloat16 or partial.layout != operations.TILE_LAYOUT
            or partial.memory_config() != operations.L1_MEMORY_CONFIG or partial.is_sharded()
            or topology != operations.Topology.Linear or collectives.get_num_links(0) != 4):
        raise ValueError('Pinned interleaved T8 partial and four-link linear TP2 collective required')
    return operations.experimental.reduce_scatter_minimal_async(
        partial, persistent_output_buffers=None, dim=3,
        multi_device_global_semaphore=collectives.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=4,
        memory_config=operations.DRAM_MEMORY_CONFIG, intermediate_memory_config=operations.DRAM_MEMORY_CONFIG,
        topology=topology, chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2, subdevice_id=None)
