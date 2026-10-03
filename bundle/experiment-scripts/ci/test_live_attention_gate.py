from copy import deepcopy
import unittest

from live_attention_gate import FIXTURE_SHA256, qualify_integration


class LiveAttentionGateTests(unittest.TestCase):
    def fixture(self, context=31):
        return dict(passed=True, closed_cleanly=True, backend='simulator', context=context,
            sources={'integration': 'hash'}, native_sources={'native': 'hash'},
            fixture_sha256=FIXTURE_SHA256 if context == 31 else None,
            eager_checks=[dict(pattern=pattern, chip=chip, exact_all_rows=True)
                for pattern in range(2) for chip in range(2)],
            replay_checks=[dict(repetition=repetition, pattern=pattern, arm=arm, chip=chip,
                exact_all_rows=True, inputs_unchanged=True, bindings_stable=True)
                for repetition, pattern in enumerate((0, 1, 0)) for arm in range(2) for chip in range(2)],
            negative_controls=[dict(arm=arm, chip=chip, stale_detected=True) for arm in range(2) for chip in range(2)])

    def qualify(self, report, context=31):
        qualify_integration(report, context, {'integration': 'hash'}, {'native': 'hash'})

    def test_short_learned_and_long_synthetic_are_distinct(self):
        for context in (31, 2048):
            self.qualify(self.fixture(context), context)
        with self.assertRaises(ValueError):
            self.qualify(self.fixture(2048))

    def test_completion_backend_source_and_fixture_are_required(self):
        for key, value in (('passed', 1), ('closed_cleanly', False), ('backend', 'hardware'),
                ('sources', {}), ('native_sources', {}), ('fixture_sha256', 'other'), ('context', True)):
            report = self.fixture()
            report[key] = value
            with self.assertRaises(ValueError):
                self.qualify(report)

    def test_incomplete_duplicate_inexact_and_boolean_rows_are_rejected(self):
        for field, flag in (('eager_checks', 'exact_all_rows'), ('replay_checks', 'exact_all_rows'),
                ('replay_checks', 'inputs_unchanged'), ('replay_checks', 'bindings_stable'),
                ('negative_controls', 'stale_detected')):
            for mutation in ('missing', 'duplicate', 'inexact', 'boolean'):
                report = self.fixture()
                if mutation == 'missing':
                    report[field].pop()
                elif mutation == 'duplicate':
                    report[field][-1] = deepcopy(report[field][0])
                elif mutation == 'boolean':
                    report[field][0]['chip'] = False
                else:
                    report[field][0][flag] = False
                with self.assertRaises(ValueError):
                    self.qualify(report)
