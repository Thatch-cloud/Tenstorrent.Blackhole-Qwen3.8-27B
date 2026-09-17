"""Unqualified removal of the normalization reciprocal's scalar staging copy."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_ladder_normalization


START = '    CircularBuffer(reciprocal_cb).wait_front(1);'
END = '    CircularBuffer(numerator_cb).wait_front(columns);'
DECLARATION = 'void qwen_stage_score_tile(uint32_t source_cb, uint32_t scratch_cb, bool restore_relu);\n'


def transform(source):
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('Unique normalization staging boundaries required')
    begin = source.index(START)
    end = source.index(END)
    if begin >= end or 'index < 1024' not in source[begin:end]:
        raise ValueError('Original scalar reciprocal staging required')
    return DECLARATION + source[:begin] + '    qwen_stage_score_tile(reciprocal_cb, scratch_cb, false);\n' + source[end:]


@contextmanager
def normalization_stage_scope():
    with patch.object(dspark_ladder_normalization, 'HELPER', transform(dspark_ladder_normalization.HELPER)):
        yield
