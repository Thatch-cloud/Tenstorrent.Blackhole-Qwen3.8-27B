"""Value-only diagnostics; never substitutes for full attention correctness."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_attention_value_diagnostics as baseline


COEFFICIENTS = {'negative_last': (0, -1), 'scaled_last': (0, 64),
    'contrast_unit': (1, -1), 'contrast_scaled': (64, -64)}
KINDS = (*baseline.KINDS, *COEFFICIENTS)


def fixture(original, kind, position, proposals, *, fallback):
    if kind not in COEFFICIENTS:
        return fallback(original, kind, position, proposals)
    values = fallback(original, 'oldest', position, proposals)
    oldest, last = COEFFICIENTS[kind]
    values['history_value'][:, :, 0] = oldest
    values['query_value'][:, :, proposals - 1] = last
    return values


@contextmanager
def diagnostic_scope():
    original = baseline.diagnostic_fixture
    with patch.object(baseline, 'KINDS', KINDS), patch.object(baseline, 'diagnostic_fixture',
            lambda *args: fixture(*args, fallback=original)):
        yield


def summarize(records):
    indexed = {(value['kind'], value['chip']): value for value in records}
    summaries = []
    for chip in range(2):
        for kind, (oldest, last) in COEFFICIENTS.items():
            required = ((kind, chip), ('oldest', chip), ('last_proposal', chip))
            if not all(key in indexed for key in required):
                continue
            actual, first, final = (indexed[key]['actual_by_head_row'] for key in required)
            residuals = [(abs(value - oldest * first[head][row] - last * final[head][row]), head, row)
                for head, values in enumerate(actual) for row, value in enumerate(values)]
            maximum, head, row = max(residuals)
            summaries.append(dict(kind=kind, chip=chip, max_linearity_residual=maximum,
                head=head, row=row, qualification=False))
    return summaries
