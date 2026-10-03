from copy import deepcopy
import unittest

from live_qk_gate import NATIVE_SOURCES, qualify_simulator, timing_summary


class LiveQKGateTests(unittest.TestCase):
    def fixture(self):
        return dict(passed=True, closed_cleanly=True, context=31, attention=True, backend='simulator',
            sources={'source': 'hash'}, native_sources=dict.fromkeys(NATIVE_SOURCES, 'hash'),
            eager_checks=[dict(pattern=pattern, chip=chip, component=name,
                **(dict(exact_live=True, padding_zero=True) if name == 'scores' else dict(exact_all_rows=True)))
                for pattern in range(2) for chip in range(2) for name in ('scores', 'probabilities', 'output')],
            replay_checks=[dict(repetition=repetition, pattern=pattern, arm=arm, chip=chip, component=name, exact=True)
                for repetition, pattern in enumerate((0, 1, 0)) for arm in range(2) for chip in range(2)
                for name in ('scores', 'probabilities', 'output')],
            negative_controls=[dict(arm=arm, chip=chip, stale_detected=True) for arm in range(2) for chip in range(2)])

    def qualify(self, report):
        qualify_simulator(report, 31, {'source': 'hash'}, dict.fromkeys(NATIVE_SOURCES, 'hash'))

    def test_complete_correctness_gate(self):
        self.qualify(self.fixture())

    def test_source_context_backend_and_completion_are_required(self):
        for field, value in (('sources', {}), ('native_sources', {}), ('context', 2048),
                ('backend', 'hardware'), ('closed_cleanly', False), ('passed', False), ('attention', False)):
            report = self.fixture()
            report[field] = value
            with self.assertRaises(ValueError):
                self.qualify(report)

    def test_missing_duplicated_or_inexact_checks_are_rejected(self):
        for field in ('eager_checks', 'replay_checks', 'negative_controls'):
            for mode in ('missing', 'duplicate', 'inexact', 'boolean_index'):
                report = self.fixture()
                if mode == 'missing':
                    report[field].pop()
                elif mode == 'duplicate':
                    report[field][-1] = deepcopy(report[field][0])
                elif mode == 'boolean_index':
                    report[field][0]['chip'] = False
                else:
                    flag = {'eager_checks': 'exact_live', 'replay_checks': 'exact', 'negative_controls': 'stale_detected'}[field]
                    report[field][0][flag] = False
                with self.assertRaises(ValueError):
                    self.qualify(report)

    def samples(self):
        return [dict(pattern=pattern, block=block, order=order, arm=arm, replays=50,
            ms=1. if arm else 2., outputs_exact=True, inputs_unchanged=True, bindings_stable=True)
            for pattern in range(2) for block in range(3) for order, arm in enumerate((0, 1, 1, 0))]

    def test_each_block_must_win(self):
        samples = self.samples()
        summary = timing_summary(samples)
        self.assertEqual(summary['control_ms'], 2.)
        self.assertEqual(summary['candidate_ms'], 1.)
        self.assertTrue(summary['eligible_for_learned_integration'])
        samples[1]['ms'] = samples[2]['ms'] = 2.
        self.assertFalse(timing_summary(samples)['eligible_for_learned_integration'])

    def test_bad_timing_cannot_pass(self):
        for field, value in (('ms', 0), ('ms', float('inf')), ('ms', float('nan')), ('ms', True),
                ('replays', 49), ('outputs_exact', False), ('inputs_unchanged', False), ('bindings_stable', False)):
            samples = self.samples()
            samples[0][field] = value
            with self.assertRaises(ValueError):
                timing_summary(samples)
        with self.assertRaises(ValueError):
            timing_summary(self.samples()[:-1])
        with self.assertRaises(ValueError):
            timing_summary(None)
