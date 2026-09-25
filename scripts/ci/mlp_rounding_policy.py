"""Unqualified nearest-away policy for the register epilogue's BF16 boundary."""

import hashlib
from pathlib import Path
import re

from gdn_multitoken import replace_once


HEADER = 'tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h'
PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
INSTRUCTIONS = '''TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG0, 5);
TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);
TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);'''
OVERRIDE = '''                                MATH((sfpi::vConstIntPrgm0 = 0));
                                MATH((sfpi::vConstIntPrgm1 = 0x8000));
'''


def validate_header(source):
    name = 'inline void calculate_typecast_fp32_to_fp16b() {'
    if source.count(name) != 1:
        raise ValueError('Unique native BF16 cast implementation required')
    body = source.split(name, 1)[1].split('\n}', 1)[0]
    instructions = re.findall(r'TTI_[A-Z_]+\([^;]+;', body)
    compact = lambda value: re.sub(r'\s+', '', value)
    if [compact(value) for value in instructions] != [compact(value) for value in INSTRUCTIONS.splitlines()]:
        raise ValueError('Native BF16 rounding instruction chain changed')
    for anchor in ('sfpi::vConstIntPrgm0 = 1;', 'sfpi::vConstIntPrgm1 = 0x7fff;',
            'sfpi::vConstIntPrgm2 = 0xffff0000;'):
        if source.count(anchor) != 1:
            raise ValueError('Native BF16 rounding constants changed')


def runtime(root):
    root = Path(root)
    raw = (root / HEADER).read_bytes()
    try:
        validate_header(raw.decode())
    except ValueError as error:
        source = raw.decode()
        begin = source.find('inline void calculate_typecast_fp32_to_fp16b() {')
        excerpt = source[begin:].split('\n}', 1)[0] if begin >= 0 else 'function absent'
        raise ValueError(f'{error}; header SHA256={hashlib.sha256(raw).hexdigest()}; native body={excerpt}') from error
    return dict(typecast_header_sha256=hashlib.sha256(raw).hexdigest(),
        packer_header_sha256=hashlib.sha256((root / PACKER).read_bytes()).hexdigest(),
        rounding_policy='nearest-away hypothesis', hardware_qualified=False)


def transform(source):
    from mlp_register_epilogue import CAST, transform as register_epilogue

    candidate = register_epilogue(source)
    anchor = f'                                typecast_tile_init<{CAST}>();\n'
    return replace_once(candidate, anchor, anchor + OVERRIDE)


def round_word(word, *, ties_away):
    if type(word) is not int or not 0 <= word <= 0xffffffff or type(ties_away) is not bool:
        raise ValueError('FP32 storage word and explicit tie policy required')
    if word & 0x7f800000 == 0x7f800000:
        raise ValueError('Finite fixture words required')
    bias = 0x8000 if ties_away else 0x7fff + ((word >> 16) & 1)
    return ((word + bias) & 0xffffffff) & 0xffff0000
