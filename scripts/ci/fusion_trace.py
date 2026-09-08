"""Exact changing-input native and fused projection replay, with caller-owned parameters."""

from attention_batch import capture_operation
from gdn_multitoken_conv import addresses, release_owned


def validate_replays(operations, mesh, inputs, hidden, references, native, fused, parameters):
    import torch

    rows = hidden.shape[-2]
    other = torch.randn(hidden.shape, generator=torch.Generator().manual_seed(7391 + rows)).bfloat16()
    values = (hidden, other)
    payloads = [operations.from_torch(value, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
        mesh_mapper=operations.ReplicateTensorToMesh(mesh)) for value in values]
    protected = (inputs, *parameters)
    original_addresses = [addresses(operations, value) for value in protected]
    traces, outputs, owned = {}, {}, []
    report = dict(rows=rows, passed=False, checks=[], negative_controls=[])

    def host(value, chip):
        return operations.to_torch(operations.get_device_tensors(value)[chip]).clone()

    try:
        operations.copy_host_to_device_tensor(payloads[1], inputs)
        temporary = []
        try:
            result = native(temporary)
            alternate = [host(result, chip) for chip in range(2)]
            if any(torch.equal(first, second) for first, second in zip(references, alternate, strict=True)):
                raise AssertionError('Distinct inputs must produce distinct native references')
            operations.synchronize_device(mesh)
        finally:
            release_owned(operations, temporary)
        expected = (references, alternate)
        operations.copy_host_to_device_tensor(payloads[0], inputs)
        operations.synchronize_device(mesh)
        for name, operation in (('control', native), ('fused', fused)):
            trace, output = capture_operation(operations, mesh, lambda: operation(owned))
            traces[name], outputs[name] = trace, output
        output_addresses = {name: addresses(operations, value) for name, value in outputs.items()}
        for repetition, pattern in enumerate((0, 1, 0)):
            operations.copy_host_to_device_tensor(payloads[pattern], inputs)
            operations.synchronize_device(mesh)
            for name, trace in traces.items():
                operations.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if ([addresses(operations, value) for value in protected] != original_addresses
                        or addresses(operations, outputs[name]) != output_addresses[name]):
                    raise AssertionError('Captured buffers moved')
                for chip in range(2):
                    if not torch.equal(host(outputs[name], chip), expected[pattern][chip]):
                        raise AssertionError(f'Captured {name} differs from native reference at pattern{pattern}/chip{chip}')
                    if not torch.equal(host(inputs, chip), values[pattern]):
                        raise AssertionError('Projection modified caller input')
                    report['checks'].append(dict(arm=name, repetition=repetition, pattern=pattern, chip=chip, exact=True))
                if repetition == 0:
                    operations.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for chip in range(2):
                        observed = host(outputs[name], chip)
                        if not torch.equal(observed, expected[0][chip]) or torch.equal(observed, expected[1][chip]):
                            raise AssertionError('Missing-input-update negative control failed')
                        report['negative_controls'].append(dict(arm=name, chip=chip, stale_input_detected=True))
    finally:
        for trace in traces.values():
            operations.release_trace(mesh, trace)
        release_owned(operations, owned)
    report['passed'] = len(report['checks']) == 12 and len(report['negative_controls']) == 4
    return report
