"""Preallocate feature prefixes before capture, then publish into those owned buffers."""

from gdn_multitoken_conv import addresses, release_owned


def allocate_prefix_pool(operations, allocate, *, prefixes=(1, 2, 16, 32), tap_count=5):
    prefixes = tuple(prefixes)
    if (not prefixes or len(set(prefixes)) != len(prefixes)
            or any(type(prefix) is not int or prefix not in (1, 2, 4, 8, 16, 17, 32) for prefix in prefixes)
            or type(tap_count) is not int or tap_count < 1):
        raise ValueError('Explicit bounded prefix pool geometry required')
    pool = {(epoch, 0): () for epoch in range(2)}
    owned, protected = [], []
    try:
        for epoch in range(2):
            for prefix in prefixes:
                group = []
                for tap in range(tap_count):
                    value = allocate(prefix)
                    current = addresses(operations, value)
                    if any(any(left == right for left, right in zip(current, other, strict=True)) for other in protected):
                        raise ValueError('Prefix pool allocations must own independent chip storage')
                    owned.append(value)
                    protected.append(current)
                    if tuple(value.shape) != (1, 1, prefix, 2560):
                        raise ValueError('Prefix pool must use chip-local [1,1,prefix,2560] geometry')
                    group.append(value)
                pool[epoch, prefix] = tuple(group)
        return pool
    except BaseException:
        release_owned(operations, owned)
        raise


def copy_prefix(operations, features, prefix):
    features = tuple(features)
    if not features:
        raise ValueError('Captured feature tensors required')
    shape = tuple(features[0].shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[2] not in (1, 2, 4, 8, 16, 32) or shape[3] != 2560:
        raise ValueError('Expected TP2 feature tensors with local logical shape [1,1,T,2560]')
    if any(tuple(value.shape) != shape for value in features):
        raise ValueError('All feature taps must have the same geometry')
    if type(prefix) is not int or not 0 <= prefix <= shape[2]:
        raise ValueError('Committed feature prefix must lie within the captured block')
    protected = [addresses(operations, value) for value in features]
    owned = []
    try:
        if prefix:
            for feature, source_addresses in zip(features, protected, strict=True):
                sliced = operations.slice(feature, (0, 0, 0, 0), (1, 1, prefix, shape[3]),
                    memory_config=operations.DRAM_MEMORY_CONFIG)
                temporary_addresses = addresses(operations, sliced)
                temporary_owned = all(before != after for before, after in zip(source_addresses, temporary_addresses, strict=True))
                try:
                    result = operations.clone(sliced, memory_config=operations.DRAM_MEMORY_CONFIG)
                    result_addresses = addresses(operations, result)
                    if any(any(left == right for left, right in zip(result_addresses, other, strict=True))
                           for other in [*protected, temporary_addresses, *[addresses(operations, value) for value in owned]]):
                        raise ValueError('Published feature prefix must own independent chip storage')
                    owned.append(result)
                    if tuple(result.shape) != (1, 1, prefix, shape[3]):
                        raise ValueError('Published feature prefix geometry changed')
                finally:
                    if temporary_owned:
                        operations.deallocate(sliced)
        return tuple(owned)
    except BaseException:
        release_owned(operations, owned)
        raise


def allocate_prefixes(operations, features, prefixes):
    groups = []
    try:
        for prefix in prefixes:
            groups.append(copy_prefix(operations, features, prefix))
        return tuple(groups)
    except BaseException:
        release_owned(operations, [value for group in groups for value in group])
        raise


def publish_prefix(operations, features, destinations, prefix):
    features, destinations = tuple(features), tuple(destinations)
    if not features or type(prefix) is not int or not 0 <= prefix <= features[0].shape[2]:
        raise ValueError('Valid feature prefix required')
    shape = tuple(features[0].shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != 2560 or any(tuple(value.shape) != shape for value in features):
        raise ValueError('Matching captured feature geometry required')
    if len(destinations) != (len(features) if prefix else 0):
        raise ValueError('Preallocated destinations required for every committed tap')
    if not prefix:
        return
    source_addresses = [addresses(operations, value) for value in features]
    destination_addresses = [addresses(operations, value) for value in destinations]
    for index, target in enumerate(destinations):
        if tuple(target.shape) != (1, 1, prefix, 2560):
            raise ValueError('Preallocated prefix geometry differs')
        if any(any(left == right for left, right in zip(destination_addresses[index], other, strict=True))
               for other in [*source_addresses, *destination_addresses[:index]]):
            raise ValueError('Prefix destinations must be independently owned')
    for source, target, protected in zip(features, destinations, source_addresses, strict=True):
        sliced = operations.slice(source, (0, 0, 0, 0), (1, 1, prefix, 2560),
            memory_config=operations.DRAM_MEMORY_CONFIG)
        temporary_owned = all(left != right for left, right in zip(addresses(operations, sliced), protected, strict=True))
        try:
            operations.copy(sliced, target)
        finally:
            if temporary_owned:
                operations.deallocate(sliced)
