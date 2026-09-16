"""Geometry-only source adapter for the historical native draft attention probe."""


REVISION = '8c102b20df22329106955b4006bf4d650bb94e40'


def geometry(context):
    if type(context) is not int or context not in (8192, 32768, 65536):
        raise ValueError('Explicit frozen-recipe 8K, 32K or 64K context required')
    capacity = context + 256
    return dict(context=context, capacity=capacity, proposals=15,
        positions=(context, capacity - 15), storage_keys=capacity + 64,
        padded_keys=((capacity + 64 + 255) // 256) * 256, key_chunk=256)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Historical geometry anchor missing or ambiguous: ' + before)
    return source.replace(before, after, 1)


def adapt_probe_sources(sources, context):
    shape = geometry(context)
    result = dict(sources)
    changes = {
        'dspark_attention_chunk_trial.py': (
            ('PADDED_KEYS = 8704', f"PADDED_KEYS = {shape['padded_keys']}"),
            ('context_rows != 8448', f"context_rows != {shape['capacity']}"),
            ('key.shape[2] != 8512', f"key.shape[2] != {shape['storage_keys']}"),
            ('PADDED_KEYS - 8512), (0, 0)', f"PADDED_KEYS - {shape['storage_keys']}), (0, 0)"),
            ('PADDED_KEYS - 8512)]', f"PADDED_KEYS - {shape['storage_keys']})]")),
        'dspark-native-8k-attention-probe.py': (
            ('CAPACITY, PROPOSALS = 8448, 15', f"CAPACITY, PROPOSALS = {shape['capacity']}, 15"),
            ('POSITIONS = (8192, 8433)', f"POSITIONS = {shape['positions']}"),
            ('native_padded_keys=8704', f"native_padded_keys={shape['padded_keys']}")),
        'dspark_stats_pack.py': (
            ('get_compile_time_arg_val(3) == 272',
                f"get_compile_time_arg_val(3) == {shape['padded_keys'] // 32}"),),
    }
    for name, replacements in changes.items():
        for before, after in replacements:
            result[name] = replace_once(result[name], before, after)
    return result
