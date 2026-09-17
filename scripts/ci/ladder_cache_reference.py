"""Reconstruct a zero-initialized native oracle; candidate comparisons remain full-cache."""


def snapshot(operations, cache, initial, physical_pages):
    import torch

    pages = sorted(set(physical_pages))
    if (not pages or any(type(page) is not int or not 0 <= page < initial.shape[0] for page in pages)
            or tuple(cache.shape) != tuple(initial.shape) or initial.device.type != 'cpu'
            or initial.dtype != torch.bfloat16 or bool(torch.count_nonzero(initial))):
        raise ValueError('Zero-initialized cache and complete written physical page set required')
    expected = [initial.clone(), initial.clone()]
    ranges = []
    for page in pages:
        if ranges and ranges[-1][1] == page:
            ranges[-1][1] = page + 1
        else:
            ranges.append([page, page + 1])
    original_addresses = tuple(part.buffer_address() for part in operations.get_device_tensors(cache))
    for start, end in ranges:
        region = operations.slice(cache, (start, 0, 0, 0), (end, *cache.shape[1:]))
        try:
            parts = operations.get_device_tensors(region)
            if len(parts) != 2:
                raise ValueError('Both native reference chips required')
            for chip, part in enumerate(parts):
                host = operations.to_torch(part)
                if tuple(host.shape) != (end - start, *initial.shape[1:]):
                    raise ValueError('Exact physical reference page shape required')
                expected[chip][start:end] = host
        finally:
            if tuple(part.buffer_address() for part in operations.get_device_tensors(region)) != original_addresses:
                operations.deallocate(region)
    return expected
