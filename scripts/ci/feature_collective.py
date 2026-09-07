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


def gather_add_projection(operations, mesh, collectives, value, *, retain_temporaries=None):
    shape = tuple(value.shape)
    if (list(mesh.shape) != [1, 2] or len(shape) != 4 or shape[:2] != (1, 1)
            or shape[2] not in (1, 8, 32) or shape[3] != 5120 or value.dtype != operations.float32):
        raise ValueError('One, eight or 32 rows of full-width FP32 TP2 projection required')
    rows = shape[2]
    temporaries = []
    output = None
    def retain(value):
        temporaries.append(value)
        if retain_temporaries is not None:
            retain_temporaries(value)
        return value
    try:
        gathered = operations.experimental.all_gather_async(value,
            persistent_output_buffer=None, dim=0,
            multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=1,
            memory_config=operations.DRAM_MEMORY_CONFIG, topology=operations.Topology.Linear,
            chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
        retain(gathered)
        for chip in range(2):
            retain(operations.slice(gathered, (chip, 0, 0, 0), (chip + 1, 1, rows, 5120)))
        output = operations.add(temporaries[1], temporaries[2], dtype=operations.float32,
            memory_config=operations.DRAM_MEMORY_CONFIG)
        if retain_temporaries is None:
            operations.synchronize_device(mesh)
        return output
    except BaseException:
        if output is not None:
            operations.deallocate(output)
        raise
    finally:
        if retain_temporaries is None:
            for tensor in reversed(temporaries):
                operations.deallocate(tensor)
