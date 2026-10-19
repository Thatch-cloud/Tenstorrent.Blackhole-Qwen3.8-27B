"""sched_report.py (the SCH pack's READ): the figures it takes from a container log, on synthetic lines made with the producers' own format strings, and the B-over-A comparison.

    nice python3 -B -m unittest test_sched_report      (from scripts/ci)
"""

import io
import unittest

import levern_policy as policy
import sched_report as report
import serving_prefill_admission as admission
import serving_request_factory as factory

PACED = policy.STEP_LINE_MERGED + policy.STEP_SCHED_SUFFIX


def prefill(n, req, seats, prev_kind, prev_ms, short=1, guard=0, final=1, cap=1500, owed='970'):
    return PACED.format(n, 'prefill', seats, req, 0, 4088, 4088, 4088, final, 'paid', prev_kind, '%.1f' % prev_ms, owed, 0, 0.5, 0.5, 0, short, cap, guard)


def decode(n, seats, prev_kind, prev_ms, cap=1500, owed='500'):
    return PACED.format(n, 'decode', seats, '-', '-', seats, '-', '-', '-', 'owed', prev_kind, '%.1f' % prev_ms, owed, 0, 0.5, 0.5, 0, 0, cap, 0)


def shared_log():
    """Two returning turns beside six decoders: each step 1.8 s, then six rounds (the share) after the first, one after the second; one rebind each."""
    lines = [prefill(1, 'r0', 6, 'decode', 140.0)]
    lines += [decode(2 + index, 6, 'prefill' if index == 0 else 'decode', 1800.0 if index == 0 else 140.0) for index in range(6)]
    lines += [prefill(8, 'r1', 6, 'decode', 140.0)]
    lines += [decode(9, 6, 'prefill', 1800.0)]
    lines += [prefill(10, 'r2', 6, 'decode', 140.0, guard=1)]
    lines += [factory.LEAN_SPLIT_MARKER + 'request=r0 a_ms=210.0 b_ms=90.0 c_ms=190.0 total_ms=490.0',
              factory.LEAN_SPLIT_MARKER + 'request=r1 a_ms=70.0 b_ms=85.0 c_ms=185.0 total_ms=340.0',
              '[PINDIAG] admission lean rebind phases request=r0 slot=3 fence_ms=3.0 rezero_ms=9.0 project_ms=120.0 copy_ms=15.0 history_ms=12.0 engine_ms=26.0',
              factory.LEAN_ADOPT_MARKER + 'request=r0 slot=3 readback=1 ms=160.0', factory.LEAN_ADOPT_MARKER + 'request=r1 slot=3 readback=0 ms=12.0',
              admission.LEAN_REUSED_LINE.format('r1', 3000.0, 4800, 4700, 1000, 512),
              '[PINDIAG] sticky engine built req=r0 ms=541.0 frontier=0 prompt=4088 kind=rebind', '[PINDIAG] sticky engine built req=r1 ms=330.0 frontier=0 prompt=4088 kind=rebind',
              '[PINDIAG] sticky engine built req=r9 ms=2400.0 frontier=0 prompt=4088 kind=build',
              '[PINDIAG] dram after engine r0: chip0 allocated=28.77GB',
              '[MEMLEDGER] before op=rebind point=req=r0 slot=3 chip0 largest_free=4437.6MB free=4.597GB estimate=100.0MB',
              '[MEMLEDGER] before op=rebind point=req=r0 slot=3 chip1 largest_free=4437.6MB free=4.597GB estimate=100.0MB',
              '[PINDIAG] prefix: host tier spill complete on cap_blocks=512 hard_max_blocks=4096 batch_blocks=512 warn_ms=1000',
              'host tier spill blocks=1 bytes=2228224 ms=4.3 held=1 batches=1 lifted=0', 'host tier spill blocks=32 bytes=71303168 ms=57.0 held=2 batches=1 lifted=0',
              'host tier spill blocks=256 bytes=570425344 ms=439.0 held=3 batches=1 lifted=0', 'host tier spill blocks=3905 bytes=8701214720 ms=6700.0 held=4 batches=8 lifted=3393',
              'host tier spill slow blocks=3905 ms=6700.0 warn_ms=1000 batches=8']
    return lines


class ReadTests(unittest.TestCase):
    def setUp(self):
        self.values = report.figures(report.read(shared_log()))

    def test_the_admission(self):
        values = self.values
        self.assertEqual((values['rebinds'], values['builds']), (2, 1))
        self.assertEqual(values['rebind_p50'], 541.0)
        self.assertEqual(values['build_p50'], 2400.0)
        self.assertEqual((values['splits'], values['split_a_p50'], values['split_c_p50']), (2, 210.0, 190.0))
        self.assertEqual((values['phase_project_p50'], values['phase_engine_p50'], values['phase_rezero_p50']), (120.0, 26.0, 9.0))
        self.assertEqual((values['adopts'], values['adopts_skipped']), (2, 1))
        self.assertEqual((values['backstop_reused'], values['dram_lines'], values['rebind_brackets']), (1, 1, 1), 'one bracket a rebind: chip 0 carries it')

    def test_the_scheduler_sees_the_rounds_after_each_returning_turn_and_the_guard(self):
        values = self.values
        self.assertEqual(values['shorts_beside_decoders'], 2, 'the first two returning turns are followed by a prefill step; the third is still open')
        self.assertEqual((values['rounds_after_short_max'], values['rounds_after_short_p50']), (6, 6))
        self.assertEqual(values['guard_steps'], 1)
        self.assertEqual(values['cap_ms'], 1500.0)
        self.assertGreater(values['pending_prefill_share'], 0.5)
        self.assertEqual(values['longest_stretch_ms'], 1800.0)

    def test_the_tier(self):
        values = self.values
        self.assertEqual(values['spills'], 4)
        self.assertEqual((values['spill_max_blocks'], values['spill_max_ms']), (3905, 6700.0))
        self.assertAlmostEqual(values['spill_ms_per_block'], 1.7, delta=0.05)
        self.assertGreater(values['spill_fixed_ms'], 0.0)
        self.assertEqual((values['spill_lifted'], values['spill_slow'], values['spill_complete']), (3393, 1, True))

    def test_a_figure_with_no_line_behind_it_is_none_not_zero(self):
        values = report.figures(report.read(['nothing to see']))
        for name in ('rebind_p50', 'split_a_p50', 'phase_project_p50', 'pending_prefill_share', 'rounds_after_short_p50', 'spill_ms_per_block', 'spill_lifted', 'guard_steps', 'cap_ms', 'gap_ms_p90'):
            self.assertIsNone(values[name], name)
        self.assertEqual((values['rebinds'], values['spills'], values['step_lines']), (0, 0, 0))

    def test_a_log_without_the_pacing_fields_reads_clean(self):
        plain = policy.STEP_LINE_MERGED.format(1, 'prefill', 6, 'r0', 0, 4088, 4088, 4088, 1, 'paid', 'decode', '140.0', '0', 0, 0.5, 0.5, 0)
        values = report.figures(report.read([plain, plain.replace('n=1 ', 'n=2 ')]))
        self.assertEqual(values['step_lines'], 2)
        self.assertIsNone(values['guard_steps'])
        self.assertIsNone(values['rounds_after_short_p50'])
        self.assertEqual(values['shorts_beside_decoders'], 0)


class PrintTests(unittest.TestCase):
    def test_one_log_prints_a_row_per_figure(self):
        out = io.StringIO()
        report.report(report.read(shared_log()), out)
        text = out.getvalue()
        for phrase in ('engine builds of returning turns', 'lean split C: the rebind', 'rounds after a returning turn beside decoders', 'blocks complete mode spilled beyond the cap'):
            self.assertIn(phrase, text)
        self.assertIn('541.0', text)

    def test_two_logs_print_b_over_a(self):
        slow = [line.replace('ms=541.0', 'ms=700.0') for line in shared_log()]
        out = io.StringIO()
        report.compare(report.read(slow), report.read(shared_log()), out)
        row = [line for line in out.getvalue().splitlines() if line.startswith('engine builds of returning turns')][0]
        self.assertIn('0.77', row, '541 over 700')


if __name__ == '__main__':
    unittest.main()
