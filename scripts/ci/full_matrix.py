"""Exact coverage checks for the static long-context verifier experiment."""

from itertools import product


def validate_static_matrix(report, max_rows):
    if type(max_rows) is not int or max_rows not in (16, 32) or report.get('rows') != max_rows:
        raise ValueError('Explicit supported static verifier width required')
    lengths = (4095, 16383)
    widths = (1, 2, 4, 8, 16, 32) if max_rows == 32 else (1, 2, 4, 8, 16)
    prefixes = (0, 1, max_rows // 2, max_rows)
    cases = (
        ('mode_checks', ('length', 'trace'), product(lengths, (False, True)), ('logits_exact',)),
        ('batch_checks', ('length', 'rows', 'trace'), product(lengths, widths, (False, True)),
         ('logits_exact', 'all_gdn_states_exact', 'valid_kv_exact')),
        ('checks', ('length', 'prefix', 'trace'), product(lengths, prefixes, (False, True)),
         ('logits_exact', 'all_gdn_states_exact', 'valid_kv_exact')),
        ('negative_controls', ('length', 'trace'), product(lengths, (False, True)),
         ('stale_gdn_detected', 'wrong_page_detected')),
        ('timings', ('length', 'rows'), product(lengths, widths), ('exact',)),
    )
    for name, keys, expected_cases, flags in cases:
        expected = set(expected_cases)
        records = report.get(name, [])
        actual = [tuple(record.get(key) for key in keys) for record in records]
        if len(actual) != len(expected) or set(actual) != expected:
            raise AssertionError(f'Incomplete or duplicated static matrix: {name}')
        if any(record.get(flag) is not True for record in records for flag in flags):
            raise AssertionError(f'Non-exact static matrix: {name}')
    if any(record.get('correction_steps') != 2 for record in report['checks']):
        raise AssertionError('Two corrected continuation steps required')
