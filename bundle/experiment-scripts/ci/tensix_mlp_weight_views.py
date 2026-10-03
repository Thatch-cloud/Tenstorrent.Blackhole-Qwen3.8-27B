"""Borrow canonical tiled weight views without changing native MLP tensors or their storage."""


WEIGHTS = dict(gate=(5120, 8704, 'bfloat4_b'), up=(5120, 8704, 'bfloat4_b'),
    down=(8704, 5120, 'bfloat8_b'))


def tensor_spec(operations, tensor):
    def describe(value):
        dtype = next((name for name in ('bfloat4_b', 'bfloat8_b')
            if value.dtype == getattr(operations, name)), str(value.dtype))
        return dict(shape=list(value.shape), padded_shape=list(value.padded_shape), dtype=dtype,
            layout='tile' if value.layout == operations.TILE_LAYOUT else str(value.layout),
            memory='interleaved_dram' if value.memory_config() == operations.DRAM_MEMORY_CONFIG
                else str(value.memory_config()))
    shards = operations.get_device_tensors(tensor)
    if len(shards) != 2:
        raise ValueError('Both chip-local weight tensors required')
    return dict(**describe(tensor), shards=[dict(**describe(shard), address=shard.buffer_address())
        for shard in shards])


def validate_spec(spec, name, canonical=False):
    inner, width, dtype = WEIGHTS[name]
    expected = dict(dtype=dtype, layout='tile', memory='interleaved_dram')
    shapes = [[1, 1, inner, width]] if canonical else [[inner, width], [1, 1, inner, width]]
    if (not isinstance(spec, dict) or spec.get('shape') not in shapes
            or not isinstance(spec.get('shards'), list) or len(spec['shards']) != 2):
        raise ValueError('Only exact native 2D or canonical 4D TP2 MLP weight shapes are supported')
    for item in [spec, *spec['shards']]:
        if (item.get('shape') != spec['shape'] or item.get('padded_shape') != spec['shape']
                or any(item.get(key) != value for key, value in expected.items())):
            raise ValueError('Unpadded BF4/BF4/BF8 tiled interleaved DRAM weights required on both chips')
    if any(type(shard.get('address')) is not int or shard['address'] <= 0 for shard in spec['shards']):
        raise ValueError('Allocated chip-local weight buffers required')


def qualify_views(checks):
    if not isinstance(checks, dict) or set(checks) != set(WEIGHTS):
        raise ValueError('All three native MLP weight-view audits required')
    for name, check in checks.items():
        native, candidate = check.get('native'), check.get('candidate')
        validate_spec(native, name)
        validate_spec(candidate, name, canonical=True)
        if (check.get('native_after') != native
                or [shard['address'] for shard in native['shards']]
                    != [shard['address'] for shard in candidate['shards']]):
            raise ValueError('Weight views must preserve both physical buffers and original tensor metadata')
    return dict(passed=True, weights=3, chip_aliases=6, native_metadata_unchanged=True)


def weight_views(operations, weights):
    if set(weights) != set(WEIGHTS):
        raise ValueError('All three native MLP weights required')
    native = {name: tensor_spec(operations, tensor) for name, tensor in weights.items()}
    for name, spec in native.items():
        validate_spec(spec, name)
    views, checks = {}, {}
    for name, tensor in weights.items():
        inner, width, unused_dtype = WEIGHTS[name]
        views[name] = (tensor if len(tensor.shape) == 4
            else operations.experimental.view(tensor, (1, 1, inner, width)))
        checks[name] = dict(native=native[name], candidate=tensor_spec(operations, views[name]),
            native_after=tensor_spec(operations, tensor))
    qualify_views(checks)
    return views, checks
