"""Bounded target KV readback preserving the legacy 64-page digest stream."""


def digest_prefix(operations, caches, valid, *, digest, addresses, read_pages=256, evidence=None):
    if type(valid) is not int or valid < 1:
        raise ValueError('Positive integer valid prefix required')
    if type(read_pages) is not int or read_pages not in (64, 128, 256):
        raise ValueError('Bounded whole digest-group readback required')
    pages = (valid + 63) // 64
    if not caches or any(len(value.shape) != 4 or value.shape[0] < pages
            or value.shape[2] != 64 for value in caches):
        raise ValueError('Every cache must cover the complete paged prefix')
    result = []
    statistics = dict(slice_groups=0, shard_readbacks=0, peak_host_tensor_bytes=0,
        digest_groups=0, read_pages=read_pages, valid=valid)
    for value in caches:
        for start in range(0, pages, read_pages):
            end = min(start + read_pages, pages)
            sliced = operations.slice(value, (start, 0, 0, 0),
                (end, value.shape[1], 64, value.shape[3]))
            try:
                statistics['slice_groups'] += 1
                shards = operations.get_device_tensors(sliced)
                if len(shards) != 2:
                    raise AssertionError('Both physical target KV shards required')
                grouped = [[] for offset in range(0, end - start, 64)]
                for shard in shards:
                    host = operations.to_torch(shard)
                    expected = (end - start, value.shape[1], 64, value.shape[3])
                    if tuple(host.shape) != expected:
                        raise AssertionError('Unexpected target KV shard shape')
                    statistics['shard_readbacks'] += 1
                    statistics['peak_host_tensor_bytes'] = max(
                        statistics['peak_host_tensor_bytes'], host.numel() * host.element_size())
                    for group, offset in enumerate(range(0, end - start, 64)):
                        block = host[offset:offset + 64]
                        logical = block.permute(1, 0, 2, 3).reshape(block.shape[1], -1, block.shape[3])
                        count = min(block.shape[0] * 64, valid - (start + offset) * 64)
                        grouped[group].append(digest(logical[:, :count]))
                        del logical, block
                    del host
                for group in grouped:
                    result.extend(group)
                statistics['digest_groups'] += len(grouped)
            finally:
                if addresses(operations, sliced) != addresses(operations, value):
                    operations.deallocate(sliced)
    if evidence is not None:
        evidence.update(statistics)
    return result
