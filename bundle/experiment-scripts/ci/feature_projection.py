"""Pack a tap-major learned projection for TP2 hidden-sharded target features."""


def require_projection_environment(environment, hardware):
    if environment.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast dispatch required')
    if hardware:
        if (environment.get('QWEN_HARDWARE_TESTS') != '1' or environment.get('QWEN_CARDS_ALLOCATED') != '1'
                or environment.get('TT_METAL_SIMULATOR') or environment.get('TT_METAL_MOCK_CLUSTER_DESC_PATH')):
            raise RuntimeError('Explicit allocation and non-simulated hardware required')
    elif not environment.get('TT_METAL_SIMULATOR'):
        raise RuntimeError('Simulator required unless --hardware is explicitly selected')


def sparse_input_permutation(width, active_terms, stride):
    if (any(type(value) is not int or value < 1 for value in (width, active_terms, stride))
            or (active_terms - 1) * stride >= width):
        raise ValueError('Positive dimensions and in-bounds sparse terms required')
    destinations = set(range(0, active_terms * stride, stride))
    remaining = iter(range(active_terms, width))
    return tuple(index // stride if index in destinations else next(remaining) for index in range(width))


def projection_shards(weight, *, tap_count=5, hidden_size=5120):
    if (type(tap_count) is not int or tap_count < 1 or type(hidden_size) is not int
            or hidden_size < 2 or hidden_size % 2 or weight.ndim != 2
            or weight.shape[0] < 1 or weight.shape[1] != tap_count * hidden_size):
        raise ValueError('Output-by-tap-major-hidden projection and even TP2 hidden size required')
    outputs = weight.shape[0]
    grouped = weight.reshape(outputs, tap_count, hidden_size)
    width = hidden_size // 2
    return tuple(grouped[:, :, chip * width:(chip + 1) * width].reshape(outputs, tap_count * width)
        .transpose(0, 1).contiguous() for chip in range(2))


def concatenate_local_features(operations, features, *, tap_count=5, hidden_size=5120):
    features = tuple(features)
    if (type(tap_count) is not int or tap_count < 1 or type(hidden_size) is not int
            or hidden_size < 2 or hidden_size % 2 or len(features) != tap_count):
        raise ValueError('Complete ordered feature taps and even TP2 hidden size required')
    shape = tuple(features[0].shape)
    if (len(shape) != 4 or shape[:2] != (1, 1) or shape[2] < 1 or shape[3] != hidden_size // 2
            or any(tuple(value.shape) != shape for value in features)):
        raise ValueError('Matching chip-local [1,1,T,hidden/2] features required')
    return operations.concat(features, dim=-1, memory_config=operations.DRAM_MEMORY_CONFIG)
