"""Changed-input trace qualification for the DRAM projection experiment."""

from attention_batch import capture_operation
from dram_sharded_projection import execute
from gdn_multitoken_conv import addresses, release_owned
from tensix_projection_raw import copy_raw, raw_shape


def check(operations, torch, mesh, source, weight, original_weight, config, compute, control,
          host_patterns, mapper, retain):
    payloads = [operations.from_torch(pattern, dtype=operations.bfloat16,
        layout=operations.TILE_LAYOUT, mesh_mapper=mapper) for pattern in host_patterns]
    def allocate_raw(shape, tile_bytes=2048):
        return retain(operations.from_torch(torch.zeros(raw_shape(shape, tile_bytes), dtype=torch.int32),
            dtype=operations.uint32, layout=operations.ROW_MAJOR_LAYOUT, device=mesh,
            memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ReplicateTensorToMesh(mesh)))
    raw_input = allocate_raw(tuple(source.padded_shape))
    raw_output = allocate_raw((1, 1, 32, config['plan']['width']))
    def read(value, destination, tile_bytes=2048):
        copy_raw(operations, value, destination, tile_bytes)
        operations.synchronize_device(mesh)
        return [operations.to_torch(shard).clone() for shard in operations.get_device_tensors(destination)]
    def upload(pattern):
        operations.copy_host_to_device_tensor(payloads[pattern], source)
        operations.synchronize_device(mesh)
    references, input_words = [], []
    temporary = []
    def keep_temporary(value):
        temporary.append(value)
        return value
    def clear_temporary():
        operations.synchronize_device(mesh)
        release_owned(operations, temporary)
        temporary.clear()
    tile_bytes = 576 if config['plan']['dtype'] == 'bfloat4_b' else 1088
    raw_weight = allocate_raw(tuple(original_weight.padded_shape), tile_bytes)
    expected_weight = read(original_weight, raw_weight, tile_bytes)
    def check_weight():
        restored = retain(operations.to_memory_config(weight, operations.DRAM_MEMORY_CONFIG))
        actual_weight = read(restored, raw_weight, tile_bytes)
        if any(not torch.equal(actual, expected)
               for actual, expected in zip(actual_weight, expected_weight, strict=True)):
            raise AssertionError('DRAM-sharded packed weight words changed')
    check_weight()
    for pattern in range(2):
        upload(pattern)
        input_words.append(read(source, raw_input))
        try:
            references.append(read(control(keep_temporary), raw_output))
        finally:
            clear_temporary()
    if any(torch.equal(references[0][chip], references[1][chip]) for chip in range(2)):
        raise AssertionError('Replay patterns must distinguish stale input on each chip')
    if any(torch.equal(reference[0], reference[1]) for reference in references):
        raise AssertionError('Replay must distinguish independent chip outputs')
    upload(0)
    try:
        execute(operations, source, weight, config, compute, keep_temporary, preserve_partials=True)
    finally:
        clear_temporary()
    trace = None
    checks = []
    poisoned = operations.from_torch(torch.full((1, 1, 16, config['plan']['width']),
        float('nan'), dtype=torch.bfloat16), dtype=operations.bfloat16,
        layout=operations.TILE_LAYOUT, mesh_mapper=operations.ReplicateTensorToMesh(mesh))
    try:
        trace, output = capture_operation(operations, mesh, lambda: execute(
            operations, source, weight, config, compute, retain, preserve_partials=True))
        tracked = (source, weight, output, raw_input, raw_output)
        bindings = [addresses(operations, value) for value in tracked]
        if any(len({binding[chip] for binding in bindings}) != len(tracked) for chip in range(2)):
            raise AssertionError('Replay buffers must be disjoint')
        for repetition, pattern in enumerate((0, 1, 0)):
            upload(pattern)
            operations.copy_host_to_device_tensor(poisoned, output)
            if not all(torch.isnan(operations.to_torch(shard)).all()
                       for shard in operations.get_device_tensors(output)):
                raise AssertionError('Output poison did not reach both chips')
            operations.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = read(output, raw_output)
            unchanged = read(source, raw_input)
            if bindings != [addresses(operations, value) for value in tracked]:
                raise AssertionError('Replay bindings moved')
            for chip in range(2):
                if not torch.equal(actual[chip], references[pattern][chip]):
                    raise AssertionError('Replay differs from native physical output words')
                if not torch.equal(unchanged[chip], input_words[pattern][chip]):
                    raise AssertionError('Replay mutated physical input words')
                checks.append(dict(repetition=repetition, pattern=pattern, chip=chip,
                    exact_all_32_rows=True, input_unchanged=True, bindings_stable=True))
    finally:
        if trace is not None:
            operations.release_trace(mesh, trace)
    check_weight()
    return dict(checks=checks, stale_input_discriminating=True,
        packed_weight_integrity_qualified=True, output_poison_qualified=True,
        poisoned_logical_rows=16, compared_physical_rows=32)
