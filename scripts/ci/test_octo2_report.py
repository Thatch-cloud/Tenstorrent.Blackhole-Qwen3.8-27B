"""The tp4/octo-2 paired read (octo2_report): the early draft joined to the shape it drafted for, the per-shape rates, the verdict between a control boot and an arm boot -
on logs written by the PRODUCERS (serving_octo.OctoState writes the round lines, early_draft the draft lines, octo_draft_tp the octo draft's lines)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import early_draft  # noqa: E402
import octo2_report as report  # noqa: E402
import octo_draft_tp  # noqa: E402
import test_octo_judge as judged  # noqa: E402


def boot_text(pairs, *, octo_step=94.7, m3_step=112.9, gap=45.0, octo_draft_ms=30.0, m3_draft_ms=30.0, octo_committed=29, m3_committed=32, live=8, draft_lines=None):
    """An alternate boot: octo, m3, octo, m3 ... each round followed by the early draft for the next, as the hook writes them."""
    boot = judged.Boot(programs=lambda: 5000)
    rounds = 0
    for _ in range(pairs):
        for shape, step, committed in (('octo', octo_step, octo_committed), ('m3', m3_step, m3_committed)):
            boot.round(shape, live=live, committed=committed, step=step, gap=gap)
            rounds += 1
            following = 'm3' if shape == 'octo' else 'octo'
            boot.lines.append('%s round=%d path=reuse live=%d draft_ms=%.2f reason=-' % (early_draft.MARKER, rounds, live, octo_draft_ms if following == 'octo' else m3_draft_ms))
            if draft_lines is not None and following == 'octo':
                boot.lines.append(draft_lines(rounds))
    return boot.text()


class JoinTests(unittest.TestCase):
    def test_a_draft_belongs_to_the_shape_of_the_round_that_follows_it(self):
        text = boot_text(20, octo_draft_ms=18.0, m3_draft_ms=31.0)
        drafts = report.drafts_by_planned_shape(report.timeline(text))
        self.assertEqual(set(drafts['octo']), {18.0})
        self.assertEqual(set(drafts['m3']), {31.0})
        self.assertEqual((len(drafts['octo']), len(drafts['m3'])), (19, 20), 'the last draft has no round after it; the first octo round has no draft before it')

    def test_the_summary_reads_the_rate_and_tau_of_each_shape(self):
        summary = report.boot_summary(boot_text(20), 8)
        octo, m3 = summary['shapes']['octo'], summary['shapes']['m3']
        self.assertEqual((octo['rounds'], m3['rounds']), (19, 20), 'the first octo round of a boot has no gap read and is left out')
        self.assertAlmostEqual(octo['tokens_per_seat_round'], 29 / 8.0, places=3)
        self.assertAlmostEqual(octo['rate_aggregate'], 29 / 8.0 / ((94.7 + 45.0) / 1000.0), places=1)
        self.assertAlmostEqual(m3['rate_aggregate'], 32 / 8.0 / ((112.9 + 45.0) / 1000.0), places=1)
        self.assertEqual(summary['early_draft_ms']['octo']['median'], 30.0)
        self.assertIsNone(summary['octo_draft'])

    def test_the_octo_drafts_own_lines_are_read(self):
        lines = lambda number: octo_draft_tp.ROUND_LINE.format(round=number, built=int(number == 2), ms='6.2' if number > 2 else '450.1')
        summary = report.boot_summary(boot_text(20, draft_lines=lines), 8)
        self.assertEqual(summary['octo_draft'], dict(rounds=20, builds=1, fallbacks=0, enqueue_ms_median=6.2))


class VerdictTests(unittest.TestCase):
    def test_a_faster_octo_round_with_the_m3_rounds_where_they_were_is_a_go(self):
        control = boot_text(30)
        arm = boot_text(30, octo_step=80.0, octo_draft_ms=18.0)
        result = report.compare(control, arm)
        self.assertTrue(result['go'], result['reason'])
        self.assertGreater(result['octo_gain_aggregate'], 0.05)
        self.assertEqual(result['m3_drift'], 0.0)
        self.assertEqual(result['draft_ms_change'], -12.0)
        self.assertEqual(result['tau_change'], 0.0)

    def test_a_gain_below_the_bar_is_a_no_go(self):
        result = report.compare(boot_text(30), boot_text(30, octo_step=93.0))
        self.assertFalse(result['go'])

    def test_a_drifting_m3_rate_makes_the_boots_not_comparable(self):
        result = report.compare(boot_text(30), boot_text(30, octo_step=70.0, m3_step=140.0))
        self.assertIsNone(result['go'])
        self.assertIn('not comparable', result['reason'])

    def test_a_tau_loss_eats_a_time_gain(self):
        control = boot_text(30)
        arm = boot_text(30, octo_step=80.0, octo_committed=24)
        result = report.compare(control, arm)
        self.assertFalse(result['go'], result['reason'])
        self.assertLess(result['tau_change'], 0)

    def test_too_few_octo_rounds_is_unread(self):
        result = report.compare(boot_text(30), boot_text(5))
        self.assertIsNone(result['go'])

    def test_the_command_line_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, text in (('control', boot_text(30)), ('fast', boot_text(30, octo_step=80.0)), ('same', boot_text(30)), ('few', boot_text(3))):
                paths[name] = os.path.join(directory, name + '.log')
                Path(paths[name]).write_text(text, encoding='utf-8')
            run = lambda arm: subprocess.run([sys.executable, '-B', str(HERE / 'octo2_report.py'), '--control', paths['control'], '--arm', paths[arm]],
                                             capture_output=True, text=True, timeout=60)
            go, no_go, unread = run('fast'), run('same'), run('few')
            self.assertEqual((go.returncode, no_go.returncode, unread.returncode), (0, 1, 2), (go.stdout, no_go.stdout, unread.stdout))
            self.assertIn('OCTO2_VERDICT GO', go.stdout)
            payload = json.loads(go.stdout.splitlines()[0].split(' ', 1)[1])
            self.assertEqual(payload['arm']['live'], 8)
            missing = subprocess.run([sys.executable, '-B', str(HERE / 'octo2_report.py'), '--control', '/nonexistent', '--arm', paths['fast']], capture_output=True, text=True, timeout=60)
            self.assertEqual(missing.returncode, 2)

    def test_the_module_is_stdlib_only_python_37(self):
        import ast

        tree = ast.parse((HERE / 'octo2_report.py').read_text(encoding='utf-8'), feature_version=(3, 7))
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {'argparse', 'json', 're', 'sys', 'octo_markers', 'octo_judge'})
        self.assertNotIn(b'\r', (HERE / 'octo2_report.py').read_bytes())


if __name__ == '__main__':
    unittest.main()
