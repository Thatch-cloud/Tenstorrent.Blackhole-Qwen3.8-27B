"""The ship-5 admission chain's timing model (sched2_model.py) and its log reader (sched2_report.py), on the CPU.

The model is the bookkeeping of a measured card run, so its tests hold what makes it believable: the eager replay reproduces the measured burst, each lever removes exactly the
segment it names and nothing else, the levers compose as the arithmetic says, and the report reads the lines the serving path writes.

    nice python3 -B -m unittest test_sched2_model      (from scripts/ci)
"""

import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sched2_model as model  # noqa: E402
import sched2_report as report  # noqa: E402
import admission_pipeline as pipeline  # noqa: E402


class CalibrationTests(unittest.TestCase):
    def test_the_eager_replay_reproduces_every_measured_first_token_within_two_percent(self):
        self.assertEqual(model.check(), [])
        for row, got in zip(model.arrivals(), model.replay()):
            self.assertAlmostEqual(got, row['ttft'], delta=0.02 * row['ttft'])

    def test_the_last_first_token_is_the_measured_one(self):
        self.assertAlmostEqual(model.replay()[-1], model.MEASURED_LAST_S, delta=0.05)

    def test_the_segments_are_consistent_with_themselves(self):
        for row in model.arrivals():
            self.assertLessEqual(row['bracket'] + row['readback'], row['bind'], 'the first-use costs are part of the bind')
            self.assertGreater(row['bind'], 0)
        self.assertEqual(len(model.BURST), 8)
        self.assertEqual(model.BURST[0][5], 0, 'the first arrival of the burst took native slot 0: nothing to adopt, no readback')

    def test_a_warm_turns_bind_is_its_three_parts(self):
        self.assertAlmostEqual(model.WARM['a'] + model.WARM['b'] + model.WARM['c'], model.WARM['bind'], delta=20)
        self.assertAlmostEqual(model.WARM['rezero'] + model.WARM['project'] + model.WARM['history'] + model.WARM['engine'], model.WARM['c'], delta=20)


class LeverTests(unittest.TestCase):
    def test_no_lever_is_the_eager_replay_and_an_unknown_one_is_refused(self):
        self.assertEqual(model.replay(()), model.replay())
        with self.assertRaisesRegex(ValueError, 'unknown'):
            model.replay(('speed',))

    def test_ledger_trim_takes_exactly_the_brackets_out_of_the_last_first_token(self):
        taken = model.replay()[-1] - model.replay(('ledger',))[-1]
        self.assertAlmostEqual(taken, sum(row['bracket'] for row in model.arrivals()) / 1000.0, places=6)

    def test_seed_first_takes_exactly_the_last_arrivals_own_bind_out_of_the_last_first_token(self):
        taken = model.replay()[-1] - model.replay(('seed',))[-1]
        self.assertAlmostEqual(taken, model.arrivals()[-1]['bind'] / 1000.0, places=6)

    def test_seed_first_moves_every_first_token_earlier_by_that_arrivals_own_bind_alone(self):
        eager, seed = model.replay(), model.replay(('seed',))
        for index, row in enumerate(model.arrivals()):
            self.assertAlmostEqual(eager[index] - seed[index], row['bind'] / 1000.0, places=6, msg=index)

    def test_hold_reuse_takes_the_hold_out_of_the_arrivals_that_reuse(self):
        taken = model.replay()[-1] - model.replay(('hold',))[-1]
        reused = [row['hold'] for index, row in enumerate(model.arrivals()) if model.hold_hit(index)]
        self.assertAlmostEqual(taken, sum(reused) / 1000.0, places=6)
        self.assertEqual([model.hold_hit(index) for index in range(8)], [False, True, True, True, False, True, True, True])
        self.assertEqual([model.hold_hit(index, reuses=1) for index in range(4)], [False, True, False, True])

    def test_the_levers_compose_as_the_arithmetic_says(self):
        both = model.replay(('hold', 'ledger'))[-1]
        self.assertAlmostEqual(model.replay()[-1] - both,
                               (model.replay()[-1] - model.replay(('hold',))[-1]) + (model.replay()[-1] - model.replay(('ledger',))[-1]), places=6)
        everything = model.replay(('hold', 'ledger', 'seed'))[-1]
        self.assertAlmostEqual(both - everything, (model.arrivals()[-1]['bind'] - model.arrivals()[-1]['bracket']) / 1000.0, places=6, msg='seed first takes the TRIMMED bind')
        self.assertLess(everything, both)

    def test_no_lever_makes_any_first_token_later(self):
        eager = model.replay()
        for levers in (('hold',), ('ledger',), ('seed',), ('hold', 'ledger'), ('hold', 'ledger', 'seed')):
            for before, after in zip(eager, model.replay(levers)):
                self.assertLessEqual(after, before + 1e-9, levers)

    def test_the_uncapped_bound_is_at_least_as_good_as_the_capped_replay(self):
        for levers in (('hold',), ('ledger',), ('seed',), ('hold', 'ledger', 'seed')):
            self.assertLessEqual(model.amplified(levers)[-1], model.replay(levers)[-1] + 1e-9, levers)
        self.assertAlmostEqual(model.amplified(())[-1], model.replay(())[-1], places=6)

    def test_a_warm_turn_gains_nothing_from_the_ledger_trim_and_the_bind_from_seed_first(self):
        self.assertEqual(model.returning(('ledger',)), model.WARM_TTFT_P50_MS)
        self.assertAlmostEqual(model.WARM_TTFT_P50_MS - model.returning(('seed',)), model.WARM['bind'])
        self.assertAlmostEqual(model.WARM_TTFT_P50_MS - model.returning(('hold',)), model.WARM['hold'] * 3 / 4)
        self.assertLess(model.returning(('hold', 'seed')), model.returning(('seed',)))

    def test_the_tables_print_and_name_every_lever(self):
        text = '\n'.join(model.tables())
        for word in ('ESTIMATES', 'ledger trim', 'hold reuse', 'seed first', 'all three', 'WHERE THE 8-USER BURST GOES'):
            self.assertIn(word, text)

    def test_the_command_line(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(model.main([]), 0)
            self.assertEqual(model.main(['--check']), 0)
        self.assertIn('LEVERS', out.getvalue())


def line(at, text, module='serving_prefill_admission:_log:367'):
    return '(EngineCore pid=68) 2026-10-19 12:51:%06.3f | INFO     | %s - %s' % (at, module, text)


def admission(request, base, *, start=0, prompt=3353, hold=0.120, prefill=0.700, drain=0.002, a=60.0, b=40.0, c=180.0, bracket=0.0, readback=False, seed=False, reused=0, kind='rebind',
              phases=(0.1, 30.0, 95.0, 0.2, 15.0, 30.0)):
    """The lines of one admission, in the order the serving path writes them, timestamps in seconds from `base`."""
    t = base
    out = [line(t, '[PINDIAG] kv reservation fit request=%s reserved=67 running=71 pool=16499 decodes=1' % request)]
    t += hold
    out.append(line(t, '[PINDIAG] dram admission fit prompt=%d largest_free=4690.8MB need=968.4MB request=%s decodes=1 free=4906.7MB trace_largest_free=261.0MB' % (prompt, request)))
    if reused:
        out.insert(1, line(t - 0.001, '[PINDIAG] admission hold reading reused prompt=%d age_ms=3000 reuses=%d free=1 largest_free=1' % (prompt, reused), 'admission_pipeline:_log:1'))
    out.append(line(t, '[PINDIAG] lever N step n=3 kind=prefill seats=1 req=%s start=%d tokens=%d end=%d prompt=%d final=1 reason=paid prev=prefill:1ms owed_ms=0 owed_rounds=0 f_eff=0.500 need=0.500 gap_ms=1'
                    % (request, start, prompt - start, prompt, prompt)))
    t += prefill
    out.append(line(t, '[PINDIAG] lever N route req=%s start=%d end=%d prompt=%d final=1 wrote_slot=1 ms=%.1f programs=1->1 window=4 source=COLD captured=-' % (request, start, prompt, prompt, prefill * 1000), 'dflash_device:pindiag:46'))
    t += drain
    out.append(line(t, '[PINDIAG] admission lean backstop reused the admission reading request=%s age_ms=700 free=1 largest_free=1 need=1 allowance=1: no allocator read after the prefill' % request, 'serving_request_factory:_log:369'))
    t += a / 1000.0
    if readback:
        out.append(line(t, '[PINDIAG] admission lean adopt request=%s slot=1 readback=1 ms=%.1f' % (request, a), 'serving_request_factory:_log:369'))
    else:
        out.append(line(t, '[PINDIAG] admission lean adopt request=%s slot=1 readback=0 ms=%.1f' % (request, a), 'serving_request_factory:_log:369'))
    t += (b + c) / 1000.0
    out.append(line(t, '[PINDIAG] admission lean rebind phases request=%s slot=1 fence_ms=%.1f rezero_ms=%.1f project_ms=%.1f copy_ms=%.1f history_ms=%.1f engine_ms=%.1f' % ((request,) + phases), 'dflash_device:pindiag:46'))
    out.append(line(t, '[PINDIAG] admission lean split request=%s a_ms=%.1f b_ms=%.1f c_ms=%.1f total_ms=%.1f' % (request, a, b, c + bracket, a + b + c + bracket), 'serving_request_factory:_log:369'))
    built = a + b + c + bracket
    out.append(line(t, '[PINDIAG] sticky engine built req=%s ms=%.1f frontier=%d prompt=%d kind=%s' % (request, built, start, prompt, kind), 'dflash_device:pindiag:46'))
    if seed:
        out.append(line(t, '[PINDIAG] admission seed first deferred request=%s slot=1' % request, 'admission_pipeline:_log:1'))
        out.append(line(t, '[PINDIAG] admission seed first built request=%s ms=%.1f after_seed_ms=3.0' % (request, built), 'admission_pipeline:_log:1'))
    return out, t


class ReportTests(unittest.TestCase):
    def log(self, lines):
        handle = tempfile.NamedTemporaryFile('w', suffix='.log', delete=False, encoding='utf-8')
        self.addCleanup(os.unlink, handle.name)
        handle.write('\n'.join(lines) + '\n')
        handle.close()
        return handle.name

    def test_one_admissions_segments_are_read_back(self):
        lines, _ = admission('r1', 10.0, prefill=0.690, b=11.0, c=181.0)
        rows, facts = report.parse(lines)
        self.assertEqual(list(rows), ['r1'])
        segment = report.derived(rows['r1'])
        self.assertAlmostEqual(segment['hold'], 120.0, delta=1)
        self.assertAlmostEqual(segment['prefill'], 690.0, delta=1)
        self.assertAlmostEqual(segment['drain'], 2.0, delta=1)
        self.assertEqual((segment['a'], segment['b'], segment['c']), (60.0, 11.0, 181.0))
        self.assertAlmostEqual(segment['built'], 252.0)
        self.assertAlmostEqual(segment['serial'], 120.0 + 2.0 + 252.0, delta=1)
        self.assertIsNone(segment['readback'])
        self.assertAlmostEqual(segment['rezero'], 30.0)
        self.assertAlmostEqual(segment['bracket'], 181.0 - sum((0.1, 30.0, 95.0, 0.2, 15.0, 30.0)), delta=0.1)

    def test_a_slots_first_rebind_shows_its_bracket_and_a_native_slots_first_adoption_its_readback(self):
        lines, _ = admission('r1', 10.0, bracket=250.0, readback=True, a=200.0)
        rows, _ = report.parse(lines)
        segment = report.derived(rows['r1'])
        self.assertAlmostEqual(segment['bracket'], 250.0 + 180.0 - 170.3, delta=0.1)
        self.assertEqual(segment['readback'], 200.0)

    def test_the_seed_first_lines_are_counted_and_the_gain_read(self):
        lines, _ = admission('r1', 10.0, seed=True)
        lines += [line(20.0, '[PINDIAG] admission seed first eager request=r2 reason=no parked engines', 'admission_pipeline:_log:1')]
        rows, facts = report.parse(lines)
        self.assertTrue(rows['r1']['seed_first'])
        self.assertEqual(report.derived(rows['r1'])['seed_gain'], 3.0)
        self.assertEqual(facts['seed_eager'], {'no parked engines': 1})
        text = '\n'.join(report.facts_lines(facts, list(rows.values())))
        self.assertIn('1 deferred, 1 bound in the sampler', text)

    def test_hold_reuse_lines_and_the_installed_line_are_read(self):
        lines, _ = admission('r1', 10.0, reused=2)
        lines.append(line(1.0, '[PINDIAG] admission pipeline installed: seed_first=1 hold_reuse=1 ledger_trim=0 max_age_ms=10000 max_reuses=3 drift_mb=256', 'dflash_device:pindiag:46'))
        lines.append(line(2.0, '[PINDIAG] admission ledger trim: the memory ledger ended after the attach', 'dflash_device:pindiag:46'))
        rows, facts = report.parse(lines)
        self.assertEqual(facts['installed'], (1, 1, 0))
        self.assertEqual(facts['hold_reused'], 1)
        self.assertTrue(facts['ledger_trimmed'])
        text = '\n'.join(report.facts_lines(facts, list(rows.values())))
        self.assertIn('hold reuse: 1 holds reused', text)
        self.assertIn('seed_first=1 hold_reuse=1 ledger_trim=0', text)

    def test_an_incomplete_admission_is_left_out(self):
        lines, _ = admission('r1', 10.0)
        lines = [entry for entry in lines if 'sticky engine built' not in entry]
        rows, _ = report.parse(lines)
        self.assertEqual(rows, {})

    def test_the_table_and_the_burst_view_and_the_second_log(self):
        lines, t = [], 10.0
        for index in range(3):
            more, end = admission('req-%d' % index, t, start=0 if index < 2 else 2048, bracket=200.0 if index == 0 else 0.0, readback=index == 1, a=190.0 if index == 1 else 60.0)
            lines += more
            t = end + 0.5
        path = self.log(lines)
        text = report.report(path)
        self.assertIn('all admissions', text)
        self.assertIn('returning turns (start > 0)', text)
        self.assertIn('fresh prompts (start = 0)', text)
        self.assertIn('the bind (A + B + C)', text)
        burst = report.report(path, bursts=True)
        self.assertIn('gap after', burst)
        self.assertEqual(len([entry for entry in burst.split('\n') if entry.strip()[:1].isdigit()]), 2, 'the two fresh prompts, one line each')
        both = report.report(path, second=path)
        self.assertIn('== second log', both)

    def test_the_command_line(self):
        lines, _ = admission('r1', 10.0)
        path = self.log(lines)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(report.main([path]), 0)
        self.assertIn('hold + drain + bind', out.getvalue())

    def test_every_line_the_report_reads_is_one_the_serving_path_writes(self):
        """The module's own markers and the report's patterns match."""
        samples = {report.SEED_DEFERRED: pipeline.SEED_DEFERRED_LINE.format('rid', 3), report.SEED_BUILT: pipeline.SEED_BUILT_LINE.format('rid', 12.5, 3.0),
                   report.SEED_EAGER: pipeline.SEED_EAGER_LINE.format('rid', 'why'), report.HOLD_REUSED: pipeline.HOLD_REUSED_LINE.format(3353, 1200.0, 2, 1, 1),
                   report.INSTALLED: pipeline.INSTALLED_LINE.format(1, 1, 0, 10000, 3, 256), report.LEDGER_TRIM: pipeline.LEDGER_TRIM_LINE}
        for pattern, sample in samples.items():
            self.assertIsNotNone(pattern.search(sample), sample)


if __name__ == '__main__':
    unittest.main()
