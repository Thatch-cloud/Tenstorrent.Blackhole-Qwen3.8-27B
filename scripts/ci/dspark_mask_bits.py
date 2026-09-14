"""Unqualified exact negative-infinity mask fast path; arbitrary masks retain arithmetic."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_ladder_score_center


BEFORE = '''                    converted.bits = mask_bits;
                    scores[offset] = scores[offset] + converted.value;'''
AFTER = '''                    converted.bits = mask_bits;
                    union { float value; uint32_t bits; } score_bits;
                    score_bits.value = scores[offset];
                    if (mask_bits == 0xff800000U &&
                        ((score_bits.bits & 0x7f800000U) != 0x7f800000U || score_bits.bits == 0xff800000U)) {
                        scores[offset] = converted.value;
                    } else {
                        scores[offset] = scores[offset] + converted.value;
                    }'''


def transform(source):
    if source.count(BEFORE) != 1:
        raise ValueError('Unique scalar mask-addition anchor required')
    return source.replace(BEFORE, AFTER)


@contextmanager
def mask_scope():
    with patch.object(dspark_ladder_score_center, 'HELPER', transform(dspark_ladder_score_center.HELPER)):
        yield
