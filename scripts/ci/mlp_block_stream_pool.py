"""Explicit experiment-owned stream allocation; native model weights stay borrowed."""

from contextlib import contextmanager
import time

from mlp_block_stream import BLOCK_BYTES, geometry, pack


def memory_requirement(num_banks, *, layers=64, reserve_bytes=1024 ** 3):
    if (type(num_banks) is not int or num_banks <= 0 or type(layers) is not int
            or not 1 <= layers <= 64 or type(reserve_bytes) is not int or reserve_bytes < 0):
        raise ValueError('Positive bank count, bounded layer count and explicit reserve required')
    shape = geometry()
    per_bank = ((shape['stream_pages'] + num_banks - 1) // num_banks) * BLOCK_BYTES
    reserve_per_bank = (reserve_bytes + num_banks - 1) // num_banks
    return dict(num_banks=num_banks, layers=layers, stream_bytes_per_bank=per_bank,
        required_free_per_bank=layers * per_bank + reserve_per_bank,
        reserve_bytes_per_chip=reserve_bytes,
        allocated_stream_bytes_per_chip=layers * per_bank * num_banks,
        logical_stream_bytes_per_chip=layers * shape['stream_bytes'])


def admit_memory(views, *, layers=64, reserve_bytes=1024 ** 3):
    if len(views) != 2:
        raise ValueError('Both physical chips require memory admission')
    admitted = []
    for chip, view in enumerate(views):
        requirement = memory_requirement(view['num_banks'], layers=layers, reserve_bytes=reserve_bytes)
        free = view['total_bytes_free_per_bank']
        contiguous = view['largest_contiguous_bytes_free_per_bank']
        if (type(free) is not int or type(contiguous) is not int
                or free < requirement['required_free_per_bank']
                or contiguous < requirement['stream_bytes_per_bank']):
            raise ValueError(f'Insufficient free or contiguous DRAM on chip {chip}')
        admitted.append(dict(chip=chip, **requirement, free_per_bank=free, contiguous_per_bank=contiguous))
    return admitted


def bindings(operations, weights):
    result = []
    for weight in weights:
        shards = operations.get_device_tensors(weight)
        if len(shards) != 2:
            raise ValueError('Two-chip native weights required')
        result.append(tuple(shard.buffer_address() for shard in shards))
    if len(set(result)) != len(result):
        raise ValueError('Distinct native layer weights required')
    return result


@contextmanager
def owned_streams(operations, mesh, weights, *, reserve_bytes=1024 ** 3, packer=pack):
    weights = tuple(weights)
    if len(weights) != 64:
        raise ValueError('Complete 64-layer target required')
    native = bindings(operations, weights)
    devices = [shard.device() for shard in operations.get_device_tensors(weights[0])]

    def views():
        result = []
        for device in devices:
            view = operations.get_memory_view(device, operations.BufferType.DRAM)
            result.append({name: getattr(view, name) for name in ('num_banks',
                'total_bytes_free_per_bank', 'largest_contiguous_bytes_free_per_bank')})
        return result

    operations.synchronize_device(mesh)
    before = views()
    audit = dict(admission=admit_memory(before, reserve_bytes=reserve_bytes), memory_before=before,
        allocated_layers=0, released=False, native_bindings_unchanged=False, serving_defaults_changed=False)
    streams = []
    started = time.perf_counter()
    try:
        for index, weight in enumerate(weights):
            admit_memory(views(), layers=64 - index, reserve_bytes=reserve_bytes)
            stream = packer(mesh, weight)
            streams.append(stream)
            audit['allocated_layers'] = len(streams)
        operations.synchronize_device(mesh)
        audit.update(setup_ms=(time.perf_counter() - started) * 1000, memory_after=views())
        if bindings(operations, weights) != native:
            raise ValueError('Native weight bindings changed during packing')
        yield tuple(streams), audit
    finally:
        operations.synchronize_device(mesh)
        for stream in reversed(streams):
            operations.deallocate(stream)
        audit['released'] = True
        audit['native_bindings_unchanged'] = bindings(operations, weights) == native
        audit['memory_released'] = views()
        if not audit['native_bindings_unchanged']:
            raise ValueError('Native weight bindings changed during experiment')
