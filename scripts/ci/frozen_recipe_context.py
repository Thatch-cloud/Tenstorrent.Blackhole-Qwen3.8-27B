"""Geometry-only source adapter for the historical native draft attention probe."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(32768, 65536), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    checkout = options.checkout.resolve(strict=True)

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(checkout), *arguments])

    if git('rev-parse', 'HEAD').decode().strip() != REVISION:
        raise ValueError('Exact historical recipe checkout required')
    if git('status', '--porcelain', '--untracked-files=no').strip() or options.manifest.exists():
        raise ValueError('Clean tracked checkout and fresh manifest required')
    names = ('dspark_attention_chunk_trial.py', 'dspark-native-8k-attention-probe.py',
        'dspark_stats_pack.py')
    sources = {}
    for name in names:
        original = git('show', f'{REVISION}:scripts/ci/{name}').decode()
        actual = (checkout / 'scripts/ci' / name).read_text()
        if original.replace('\r\n', '\n') != actual:
            raise ValueError('Historical source differs: ' + name)
        sources[name] = actual
    adapted = adapt_probe_sources(sources, options.context)
    for name, source in adapted.items():
        compile(source, name, 'exec')
    for name, source in adapted.items():
        (checkout / 'scripts/ci' / name).write_text(source)
    checksum = lambda source: hashlib.sha256(source.encode()).hexdigest()
    options.manifest.write_text(json.dumps(dict(revision=REVISION,
        geometry=geometry(options.context), before={name: checksum(source) for name, source in sources.items()},
        after={name: checksum(source) for name, source in adapted.items()},
        scope='Geometry-only draft probe; no numerical or runtime admission',
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
