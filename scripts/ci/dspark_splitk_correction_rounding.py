"""Explicit BF16 rounding of local and tree softmax correction factors."""

from dspark_splitk_merge_exp import HELPER


FORMAT = '<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>'


def transform(source):
    if source.count(HELPER) != 1:
        raise ValueError('Exact shared split-K correction helper required')
    candidate = HELPER.replace('    binop_with_scalar_tile_init();',
        '    typecast_tile_init' + FORMAT + '();\n    binop_with_scalar_tile_init();').replace(
        '        exp_tile<false>(0);',
        '        exp_tile<false>(0);\n        typecast_tile' + FORMAT + '(0);')
    return source.replace(HELPER, candidate)
