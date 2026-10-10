"""draft_wp6_smoke: what a gated arm must see in the server log from the four drafter fusion levers, and nothing from a lever the profile did not ask for.

    py -3.11 -B -m unittest test_draft_wp6_smoke      (from scripts/ci)
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import draft_fusion_tp as fusion  # noqa: E402
import draft_wp6_smoke as smoke  # noqa: E402

ALL = {flag: '1' for flag in fusion.LEVERS}
ALL_AUDITED = dict(ALL, **{audit: '1' for audit in fusion.AUDITS})


def log(*, engaged=True, audited=False, fell=False, mismatch=False, levers=fusion.TABLE):
    lines = ['[server] boot']
    for flag, audit_flag, e, f, a, m, what in levers:
        if engaged:
            lines.append('%s site=mlp rows=64' % e)
        if audited:
            lines.append('%s 3 exact=True site=mlp rows=64' % a)
        if fell:
            lines.append('%s site=mlp reason=shape' % f)
        if mismatch:
            lines.append('%s site=mlp rows=64 chips=[1]' % m)
    return '\n'.join(lines)


class TableTests(unittest.TestCase):
    def test_the_rig_hosts_copy_of_the_table_is_the_levers_table(self):
        self.assertEqual(smoke.TABLE, fusion.TABLE)

    def test_the_markers_are_the_names_the_manifest_generates(self):
        for flag, audit_flag, engaged, fell_back, audit, mismatch, what in fusion.TABLE:
            base = engaged[:-len(' engaged')]
            self.assertEqual((fell_back, audit, mismatch), (base + ' fell back', base + ' audit', base + ' audit mismatch'))
            self.assertTrue(engaged.startswith('[PINDIAG] tp4 draft '))
            self.assertEqual(audit_flag, flag + '_AUDIT')


class ProblemTests(unittest.TestCase):
    def test_a_clean_run_of_every_lever_audited_or_not(self):
        self.assertEqual(smoke.problems(ALL, log()), [])
        self.assertEqual(smoke.problems(ALL_AUDITED, log(audited=True)), [])

    def test_a_lever_that_never_engaged_fails(self):
        found = smoke.problems(ALL, log(engaged=False))
        self.assertEqual(len(found), 4)
        self.assertTrue(all('no engaged line' in item for item in found))

    def test_a_fall_back_or_a_mismatch_fails_even_with_the_engaged_line(self):
        self.assertTrue(any('fell back' in item for item in smoke.problems(ALL, log(fell=True))))
        self.assertTrue(any('audit found a difference' in item for item in smoke.problems(ALL_AUDITED, log(audited=True, mismatch=True))))

    def test_an_audited_arm_that_audited_nothing_fails(self):
        found = smoke.problems(ALL_AUDITED, log(audited=False))
        self.assertEqual(len([item for item in found if 'never audited' in item]), 4)

    def test_an_audit_line_without_exact_true_does_not_count(self):
        text = log().replace('[PINDIAG] tp4 draft reduce engaged', '[PINDIAG] tp4 draft reduce audit 1 exact=False\n[PINDIAG] tp4 draft reduce engaged')
        found = smoke.problems(dict(ALL, QWEN_FAST_DRAFT_REDUCE_AUDIT='1'), text)
        self.assertEqual(len([item for item in found if 'never audited' in item]), 1)

    def test_an_audit_flag_without_its_lever_fails(self):
        found = smoke.problems({'QWEN_FAST_DRAFT_TAIL_AUDIT': '1'}, '')
        self.assertEqual(len(found), 1)
        self.assertIn('compare nothing', found[0])

    def test_a_lever_that_ran_where_nobody_asked_for_it_is_a_leak(self):
        self.assertEqual(smoke.problems({}, log(levers=fusion.TABLE[:1])).__len__(), 1)
        self.assertIn('nobody asked', smoke.problems({}, log(levers=fusion.TABLE[:1]))[0])
        self.assertEqual(smoke.problems({}, log(engaged=False)), [])
        self.assertEqual(smoke.problems({}, ''), [])
        only_reduce = {fusion.REDUCE: '1'}
        leaked = smoke.problems(only_reduce, log(levers=fusion.TABLE[:2]))
        self.assertEqual(len(leaked), 1)
        self.assertIn('QWEN_FAST_DRAFT_TAIL is not set', leaked[0])


class ChainsTests(unittest.TestCase):
    def test_chains_reads_the_largest_milestone_and_expects_the_count_the_hooks_give(self):
        text = '[PINDIAG] tp4 draft reduce engaged chains=1 site=mlp\n[PINDIAG] tp4 draft reduce engaged chains=5 site=mlp'
        self.assertEqual(smoke.chains_run(text), 5)
        self.assertEqual(smoke.chains_run(''), 0)
        self.assertEqual(smoke.chains_problems({fusion.REDUCE: '1'}, text, 5), [])
        self.assertEqual(len(smoke.chains_problems({fusion.REDUCE: '1'}, text, 10)), 1)
        self.assertEqual(smoke.chains_problems({}, text, 10), [])
        self.assertEqual(smoke.chains_problems({fusion.REDUCE: '1'}, text, None), [])


class MainTests(unittest.TestCase):
    def test_the_command_line_exit_code_follows_the_problems(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'container.log'
            path.write_text(log())
            flags = [item for flag in fusion.LEVERS for item in ('--env', '%s=1' % flag)]
            with patch('builtins.print') as printed:
                self.assertEqual(smoke.main(['--log', str(path)] + flags), 0)
                self.assertEqual(smoke.main(['--log', str(path)] + flags + ['--expect-chains', '10']), 1)
                self.assertEqual(smoke.main(['--log', str(path), '--env', 'QWEN_FAST_DRAFT_MM_GRID_AUDIT=1']), 1)
            self.assertTrue(printed.called)


if __name__ == '__main__':
    unittest.main()
