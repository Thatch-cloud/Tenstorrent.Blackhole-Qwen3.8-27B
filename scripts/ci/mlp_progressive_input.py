"""Simulator-only progressive activation multicast over the serial BF4 stream."""

import hashlib
import os
from pathlib import Path

from frozen_recipe_context import replace_once


CONTROL_SHA256 = '3e907b91884d5b07c915a91c677da8cf299fab2a4832247fe69bcc1a593112bf'


def require_simulator():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_HARDWARE_TESTS') or os.environ.get('QWEN_CARDS_ALLOCATED')):
        raise ValueError('Progressive activation delivery remains simulator-only')


def reader(original):
    if hashlib.sha256(original.encode()).hexdigest() != CONTROL_SHA256:
        raise ValueError('Exact original activation reader required')
    prefix = original[:original.index('    for (uint32_t block = 0; block < 20; ++block) {')]
    return prefix + '''    cb_reserve_back(0, 160);
    noc_semaphore_set(received, 0);
    if (worker == 0) {
        noc_semaphore_wait(ready, receivers);
        noc_semaphore_set(ready, 0);
    } else {
        noc_semaphore_inc(get_noc_addr(first_x, first_y, get_semaphore(0)), 1);
    }
    for (uint32_t block = 0; block < 20; ++block) {
        cb_reserve_back(0, 8);
        const uint32_t destination = get_write_ptr(0);
        if (worker == 0) {
            for (uint32_t tile = 0; tile < 8; ++tile) {
                noc_async_read_tile(block * 8 + tile, input, destination + tile * 2048);
            }
            noc_async_read_barrier();
            const uint64_t target = get_noc_multicast_addr(last_x, last_y, first_x, first_y, destination);
            noc_async_write_multicast(destination, target, 8 * 2048, receivers);
            noc_async_write_barrier();
            noc_semaphore_set(received, block + 1);
            const uint64_t signal = get_noc_multicast_addr(last_x, last_y, first_x, first_y, get_semaphore(1));
            noc_semaphore_set_multicast(get_semaphore(1), signal, receivers);
            noc_async_write_barrier();
        } else {
            noc_semaphore_wait_min(received, block + 1);
        }
        cb_push_back(0, 8);
        if (worker >= workers) {
            cb_wait_front(0, 8);
            cb_pop_front(0, 8);
        }
    }
    // The same exit drain the original carries (commit 4d890d6a). This transform
    // replaces everything from the block loop onwards with its own body, so
    // without repeating the drain here the progressive variant would exit with a
    // multicast or a semaphore increment still in flight - exactly the fault that
    // commit fixed, silently reintroduced in this arm only.
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
'''


def projection(original):
    source = replace_once(original, 'cb(0, ttnn.bfloat16, 2048, 16, all_cores)',
        'cb(0, ttnn.bfloat16, 2048, 160, all_cores)')
    source = replace_once(source,
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,',
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,\n'
        '                             progressive_input=True, input_buffer_tiles=160,')
    source = replace_once(source, '    def __call__(self, value):\n        import ttnn\n',
        '    def __call__(self, value):\n        import ttnn\n'
        '        from mlp_progressive_input import require_simulator\n        require_simulator()\n')
    compile(source, 'progressive_fused_1d.py', 'exec')
    return source


def stage(checkout, manifest):
    import json

    scripts, manifest = Path(checkout) / 'scripts/ci', Path(manifest)
    if manifest.exists():
        raise ValueError('Fresh progressive staging manifest required')
    original_projection = (scripts / 'fused_1d.py').read_text()
    if 'stream_weights = validate_binding(self, ttnn)' not in original_projection:
        raise ValueError('Stage the admitted serial block stream first')
    originals = {name: (scripts / name).read_text() for name in ('fused_1d.py', 'fused_1d_input.cpp')}
    payloads = {'fused_1d.py': projection(originals['fused_1d.py']),
        'fused_1d_input.cpp': reader(originals['fused_1d_input.cpp']),
        'mlp_progressive_input.py': Path(__file__).read_text()}
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    manifest.write_text(json.dumps(dict(before={name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in originals.items()}, after={name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in payloads.items()}, extra_l1_bytes_per_multicast_core=144 * 2048,
        arithmetic_changed=False, simulator_qualified=False, hardware_qualified=False,
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.manifest)
