"""Unqualified full-K activation staging for the fused T16 MLP."""

from frozen_recipe_context import replace_once


def reader(source):
    source = replace_once(source, '    for (uint32_t block = 0; block < 20; ++block) {',
        '    {')
    source = replace_once(source, '            for (uint32_t tile = 0; tile < 8; ++tile) {',
        '            for (uint32_t tile = 0; tile < 160; ++tile) {')
    source = replace_once(source, 'noc_async_read_tile(block * 8 + tile, input,',
        'noc_async_read_tile(tile, input,')
    for operation in ('cb_reserve_back', 'cb_push_back', 'cb_wait_front', 'cb_pop_front'):
        source = replace_once(source, f'{operation}(0, 8);', f'{operation}(0, 160);')
    source = replace_once(source,
        '            const uint64_t target = get_noc_multicast_addr(last_x, last_y, first_x, first_y, destination);\n'
        '            noc_async_write_multicast(destination, target, 8 * 2048, receivers);',
        '            for (uint32_t block = 0; block < 20; ++block) {\n'
        '                const uint32_t chunk = destination + block * 8 * 2048;\n'
        '                const uint64_t target = get_noc_multicast_addr(last_x, last_y, first_x, first_y, chunk);\n'
        '                noc_async_write_multicast(chunk, target, 8 * 2048, receivers);\n'
        '            }')
    return source


def projection(source):
    source = replace_once(source, 'cb(0, ttnn.bfloat16, 2048, 16, all_cores)',
        'cb(0, ttnn.bfloat16, 2048, 160, all_cores)')
    source = replace_once(source,
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,',
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,\n'
        '                             full_k_input_prefetch=True, input_buffer_tiles=160,')
    compile(source, 'fused_1d.py', 'exec')
    return source


def memory_budget(pairs_per_worker=3):
    if pairs_per_worker != 3:
        raise ValueError('Only current three-pair T16 worker mapping is considered')
    return dict(input_bytes_per_core=160 * 2048,
        extra_input_bytes_per_core=(160 - 16) * 2048,
        worker_cb_bytes=160 * 2048 + 32 * 3 * 576 + 3 * 2048 + 6 * 4096 + 6 * 2048,
        scope='Declared CB bytes only; not device L1 admission or a performance prediction')
