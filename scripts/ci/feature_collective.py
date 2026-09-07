"""Explicit FP32 TP2 reduce-scatter/all-gather; no host tensor reduction."""


def reduce_projection(operations, mesh, collectives, value):
    if list(mesh.shape) != [1, 2] or tuple(value.shape) != (1, 1, 1, 5120) or value.dtype != operations.float32:
        raise ValueError('Single-row full-width FP32 TP2 projection required')
    reduced = output = None
    try:
        reduced = operations.experimental.reduce_scatter_minimal_async(value,
            persistent_output_buffers=None, dim=3,
            multi_device_global_semaphore=collectives.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=1,
            memory_config=operations.DRAM_MEMORY_CONFIG, intermediate_memory_config=operations.DRAM_MEMORY_CONFIG,
            topology=operations.Topology.Linear, chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
        output = operations.experimental.all_gather_async(reduced,
            persistent_output_buffer=None, dim=3,
            multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=1,
            memory_config=operations.DRAM_MEMORY_CONFIG, topology=operations.Topology.Linear,
            chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
        operations.synchronize_device(mesh)
        return output
    except BaseException:
        if output is not None:
            operations.deallocate(output)
        raise
    finally:
        if reduced is not None:
            operations.deallocate(reduced)


def gather_add_projection(operations, mesh, collectives, value):
    if list(mesh.shape) != [1, 2] or tuple(value.shape) != (1, 1, 1, 5120) or value.dtype != operations.float32:
        raise ValueError('Single-row full-width FP32 TP2 projection required')
    temporaries = []
    output = None
    try:
        gathered = operations.experimental.all_gather_async(value,
            persistent_output_buffer=None, dim=0,
            multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=1,
            memory_config=operations.DRAM_MEMORY_CONFIG, topology=operations.Topology.Linear,
            chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
        temporaries.append(gathered)
        for chip in range(2):
            temporaries.append(operations.slice(gathered, (chip, 0, 0, 0), (chip + 1, 1, 1, 5120)))
        output = operations.add(temporaries[1], temporaries[2], dtype=operations.float32,
            memory_config=operations.DRAM_MEMORY_CONFIG)
        operations.synchronize_device(mesh)
        return output
    except BaseException:
        if output is not None:
            operations.deallocate(output)
        raise
    finally:
        for tensor in reversed(temporaries):
            operations.deallocate(tensor)
