"""Unqualified exact infinity-check candidate; not enabled in runtime defaults."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_ladder_score_center
import native_draft_sdpa


def transform(source):
    before = 'if (scores[offset] == -__builtin_inff() && maxima[first] == -__builtin_inff()) continue;'
    after = '''union { float value; uint32_t bits; } score_bits, maximum_bits;
                    score_bits.value = scores[offset];
                    maximum_bits.value = maxima[first];
                    if (score_bits.bits == 0xff800000U && maximum_bits.bits == 0xff800000U) continue;'''
    if source.count(before) != 1:
        raise ValueError('Unique scalar infinity comparison required')
    return source.replace(before, after)


@contextmanager
def bitwise_infinity_checks():
    with patch.object(dspark_ladder_score_center, 'HELPER', transform(dspark_ladder_score_center.HELPER)):
        yield


@contextmanager
def candidate_entrypoint(script):
    original = native_draft_sdpa.run_precise_probe

    def run(requested):
        return original(script)

    with patch.object(native_draft_sdpa, 'run_precise_probe', run):
        yield
