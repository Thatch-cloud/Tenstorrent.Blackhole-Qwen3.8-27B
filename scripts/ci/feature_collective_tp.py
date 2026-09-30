"""feature_collective at any served width.

The pair's module is pinned (recorded evidence hashes its bytes; test_tp2_pins) and stays as it was;
tp_addresses rebinds the names below to these at four cards only. Each is the pair's function with its literal chip and
head counts read from tp_shapes; at two chips it would be call for call the pinned one."""

import os
from mesh_link_policy import fast_ccl_topology, projection_links
import tp_shapes


def reduce_projection(operations, mesh, collectives, value):
    links = projection_links()
    width = tp_shapes.mesh_width(mesh)
    if width is None or tuple(value.shape) != (1, 1, 1, 5120) or value.dtype != operations.float32:
        raise ValueError('Single-row full-width FP32 TP%d projection required' % tp_shapes.requested_tp(os.environ))
    topology = fast_ccl_topology(operations)
    reduced = output = None
    try:
        reduced = operations.experimental.reduce_scatter_minimal_async(value,
            persistent_output_buffers=None, dim=3,
            multi_device_global_semaphore=collectives.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=links,
            memory_config=operations.DRAM_MEMORY_CONFIG, intermediate_memory_config=operations.DRAM_MEMORY_CONFIG,
            topology=topology, chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
        output = operations.experimental.all_gather_async(reduced,
            persistent_output_buffer=None, dim=3,
            multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=links,
            memory_config=operations.DRAM_MEMORY_CONFIG, topology=topology,
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


def gather_add_projection(operations, mesh, collectives, value, *, retain_temporaries=None, observe=None):
    links = projection_links()
    shape = tuple(value.shape)
    width = tp_shapes.mesh_width(mesh)
    if (width is None or len(shape) != 4 or shape[:2] != (1, 1)
            or shape[2] not in (1, 8, 32) or shape[3] != 5120 or value.dtype != operations.float32):
        raise ValueError('One, eight or 32 rows of full-width FP32 TP%d projection required'
                         % tp_shapes.requested_tp(os.environ))
    topology = fast_ccl_topology(operations)
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
            barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=links,
            memory_config=operations.DRAM_MEMORY_CONFIG, topology=topology,
            chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
        retain(gathered)
        if observe is not None:
            # QWEN_FAST_PROPOSAL_AUDIT (dflash_device.ProposalAudit): both partials as
            # this chip received them, before the add - replicated when the gather is.
            observe('gathered', gathered)
        for chip in range(width):
            retain(operations.slice(gathered, (chip, 0, 0, 0), (chip + 1, 1, rows, 5120)))
        output = operations.add(temporaries[1], temporaries[2], dtype=operations.float32,
            memory_config=operations.DRAM_MEMORY_CONFIG)
        for chip in range(2, width):
            partial = output
            output = operations.add(partial, temporaries[chip + 1], dtype=operations.float32,
                memory_config=operations.DRAM_MEMORY_CONFIG)
            retain(partial)
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
