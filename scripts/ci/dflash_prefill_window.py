"""Bounded owned draft features at an absolute target prefill frontier."""

from gdn_multitoken_conv import addresses


def prefill_window(position):
    if type(position) is not int or not 1 <= position <= 65504:
        raise ValueError('Absolute prefill position within the target allocation and verification headroom required')
    return dict(start=max(0, position - 2048), end=position, rows=min(position, 2048))


def snapshot_prefill_tail(operations, value, position, *, checks=None):
    window = prefill_window(position)
    if (len(value.shape) != 4 or tuple(value.shape)[:2] != (1, 1) or value.shape[2] < position
            or value.shape[3] != 2560 or value.dtype != operations.bfloat16):
        raise ValueError('Complete TP2 BF16 prefill feature output required; chunks cannot stand in for the full context')
    sliced = output = None
    try:
        sliced = operations.slice(value, (0, 0, window['start'], 0), (1, 1, position, 2560))
        output = operations.clone(sliced, memory_config=operations.DRAM_MEMORY_CONFIG)
        if checks is not None:
            import torch

            originals = operations.get_device_tensors(value)
            snapshots = operations.get_device_tensors(output)
            if len(originals) != 2 or len(snapshots) != 2:
                raise AssertionError('Both prefill feature shards required')
            for chip, (original, snapshot) in enumerate(zip(originals, snapshots, strict=True)):
                expected = operations.to_torch(original)[..., window['start']:position, :].contiguous()
                actual = operations.to_torch(snapshot).contiguous()
                if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                    raise AssertionError('Prefill tail snapshot changed, shifted or included padded feature rows')
                checks.append(dict(chip=chip, **window, exact=True))
        return output
    except BaseException:
        if output is not None:
            operations.deallocate(output)
        raise
    finally:
        if sliced is not None and addresses(operations, sliced) != addresses(operations, value):
            operations.deallocate(sliced)
