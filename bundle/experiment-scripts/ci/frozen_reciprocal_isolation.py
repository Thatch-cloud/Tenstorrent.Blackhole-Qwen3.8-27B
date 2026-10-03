"""Keep the qualified draft reciprocal out of shared-header target compilation."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa
from dspark_ladder_scalar_reciprocal import BEFORE, BODY


def isolate(substitutions):
    result = dict(substitutions)
    entries = list(result['compute_common.hpp'])
    expected = (BEFORE, BEFORE + BODY)
    if entries.count(expected) != 1:
        raise ValueError('Exactly one qualified scalar reciprocal substitution required')
    entries[entries.index(expected)] = (BEFORE,
        BEFORE + '\n#if defined(QWEN_DRAFT_EXP_APPROX)\n' + BODY + '#endif\n')
    result['compute_common.hpp'] = tuple(entries)
    return result


@contextmanager
def isolated_reciprocal():
    original = native_draft_sdpa.replacements
    with patch.object(native_draft_sdpa, 'replacements', lambda: isolate(original())):
        yield
