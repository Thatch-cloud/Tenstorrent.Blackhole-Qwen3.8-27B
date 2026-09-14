"""Finite ladder-only SDPA precision selectors; unqualified until device tests pass."""

from contextlib import contextmanager
from unittest.mock import patch

from dspark_fp32_intermediates import transform as eight_k_transform
from dspark_ladder_geometry import ladder
import dspark_stats_pack


KEY_TILES = tuple(sorted({272, *(row['native_keys'] // 32 for row in ladder())}))


def output_precision(source):
    before = '    tt::DataFormat im_df = tt::DataFormat::Float16_b;'
    after = ('    tt::DataFormat im_df = qwen_draft_fp32_intermediates && Skt == 2112\n'
        '        ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;')
    if source.count(before) != 1:
        raise ValueError('Unique output-intermediate format selector required')
    return source.replace(before, after)


def predicate(expression):
    return '(' + ' || '.join(f'{expression} == {tiles}' for tiles in KEY_TILES) + ')'


def geometry_predicate(keys, chunk):
    pairs = sorted({(272, 8), *((row['native_keys'] // 32, row['key_chunk'] // 32) for row in ladder())})
    return '(' + ' || '.join(f'({keys} == {key_tiles} && {chunk} == {chunk_tiles})'
        for key_tiles, chunk_tiles in pairs) + ')'


def transform(source):
    candidate = eight_k_transform(source)
    before = b'Skt == 272 && Sq_chunk_t == 1 && Sk_chunk_t == 8 &&'
    if candidate.count(before) != 1:
        raise ValueError('Unique qualified baseline factory selector required')
    after = (geometry_predicate('Skt', 'Sk_chunk_t') + ' && Sq_chunk_t == 1 &&').encode()
    return output_precision(candidate.replace(before, after).decode()).encode()


def selector_assert():
    return ('static_assert(!QWEN_DRAFT_EXP_APPROX,\n'
        '    "Ladder probe requires precise draft arithmetic");\n'
        'static_assert(' + geometry_predicate('get_compile_time_arg_val(3)', 'get_compile_time_arg_val(8)') + ',\n'
        '    "Ladder probe requires an explicit padded history geometry");\n')


@contextmanager
def scoped_stats_pack():
    with patch.object(dspark_stats_pack, 'SELECTOR_ASSERT', selector_assert()), dspark_stats_pack.scoped_stats_pack():
        yield
