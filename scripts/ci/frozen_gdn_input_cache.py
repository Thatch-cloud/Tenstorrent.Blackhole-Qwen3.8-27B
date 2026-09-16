"""Unqualified T16 V/beta/gate cache for the existing shared-Q/K recurrence."""

from gdn_multitoken import replace_once
from gdn_shared_qk_recurrence import INPUTS
from gdn_vsplit_prefetch import CACHE_AND_GATHER


LOOP = '    for (uint32_t token = 0; token < n_inst; ++token) {'


def cache_source():
    source = CACHE_AND_GATHER
    start = source.index('    for (uint32_t tile = 0; tile < 4; ++tile) {')
    end = source.index('    noc.async_read(v_acc, cache, tb_io,')
    source = source[:start] + source[end:]
    for before, after in (
            ('cache.reserve_back(11)', 'cache.reserve_back(3)'),
            ('cache.push_back(11)', 'cache.push_back(3)'),
            ('cache.wait_front(11)', 'cache.wait_front(3)'),
            ('8 * tb_io', '0 * tb_io'), ('9 * tb_io', '1 * tb_io'), ('10 * tb_io', '2 * tb_io')):
        source = replace_once(source, before, after)
    return source


def reader(source):
    source = replace_once(source, LOOP, cache_source() + LOOP)
    source = replace_once(source, INPUTS,
        '        gather_normalized(20, 10, token);\n'
        '        gather_normalized(21, 11, token);\n'
        '        gather_cached_row(cb_v, 1, 0, token);\n'
        '        gather_cached_scalar(cb_beta, 1, token);\n'
        '        gather_cached_scalar(cb_g, 2, token);\n\n')
    return replace_once(source, '    query_cache.pop_front(4);',
        '    cache.pop_front(3);\n    query_cache.pop_front(4);')


def buffers(io, fp32):
    if 31 in io or 31 in fp32:
        raise ValueError('Recurrence cache CB31 must be unused')
    return dict(io) | {31: 3}, dict(fp32)
