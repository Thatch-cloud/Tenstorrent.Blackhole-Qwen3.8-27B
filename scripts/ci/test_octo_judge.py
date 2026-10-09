"""The octo-T8 smoke rules (octo_judge, called from c2_smoke_check.check): the positive control, the rounds per shape under alternate, the refusal of a program
compiled on a shape switch, the flag-off leak rule, the lone-user rules and the paired timing - on logs rendered by the PRODUCERS (serving_octo.octo_admission and
serving_octo.OctoState write the lines; octo_markers parses them), plus the same rules through c2_smoke_check.check and its command line."""

import io
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import octo_judge
import octo_markers
import serving_octo as octo
import c2_smoke_check

ALL_PIECES = frozenset(key for key, text in octo.DEVICE_PIECES)
M3_TWO = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')


def env_of(mode='alternate', min_live=6, **extra):
    environ = {name: wanted for name, wanted, why in octo.REQUIRED_ENV}
    environ.update({octo.GATE_ENV: '1', octo.GATE_PROFILE_ENV: '1', octo.OCTO_FLAG: mode, octo.MIN_LIVE_FLAG: str(min_live)})
    environ.update(extra)
    return environ


class Boot:
    """A container log written by the real producers: the admission, then rounds through OctoState."""

    def __init__(self, mode='alternate', min_live=6, programs=None, admit=True):
        self.lines, self.now = [], [1000.0]
        self.mode, self.min_live = mode, min_live
        self.state = octo.OctoState(mode, min_live, clock=lambda: self.now[0], programs=programs, log=self.lines.append)
        if admit:
            with patch.multiple(octo, BUILT=set(ALL_PIECES), gdn_batch_users=Mock(return_value=8)):
                octo.octo_admission(M3_TWO, env_of(mode, min_live), log=lambda template, *values: self.lines.append(template.format(*values)))

    def round(self, ran, *, eligible=True, live=8, rows=None, committed=None, step=None, gap=20.0):
        rows = rows or (8 if ran == 'octo' else 16)
        committed = committed if committed is not None else live * (4 if ran == 'octo' else 5)
        step = step if step is not None else (50.0 if ran == 'octo' else 96.0)
        self.state.plan(eligible)
        self.now[0] += gap / 1000.0
        self.state.begin()
        self.now[0] += step / 1000.0
        self.state.finish(ran, live=live, rows=rows, committed=committed)

    def alternate(self, pairs, **kwargs):
        for _ in range(pairs):
            self.round('octo', **kwargs)
            self.round('m3', **kwargs)

    def text(self):
        return '\n'.join(self.lines) + '\n'


class JudgeTests(unittest.TestCase):
    def test_a_clean_alternate_boot_passes_with_both_shapes_counted(self):
        boot = Boot(programs=lambda: 5000)
        boot.alternate(20)
        problems, facts = octo_judge.judge(env_of(), boot.text())
        self.assertEqual(problems, [])
        self.assertEqual((facts['rounds'], facts['counted_per_shape'], facts['fallbacks'], facts['programs_compiled']),
                         (dict(octo=20, m3=20, seq=0), dict(octo=20, m3=20), 0, 0))

    def test_a_clean_live_boot_passes(self):
        boot = Boot('live', programs=lambda: 5000)
        for _ in range(20):
            boot.round('octo')
        self.assertEqual(octo_judge.judge(env_of('live'), boot.text())[0], [])

    def test_the_flag_off_arm_passes_with_no_line_and_fails_on_any(self):
        self.assertEqual(octo_judge.judge({}, 'plain log\n'), ([], {}))
        self.assertEqual(octo_judge.judge({octo.OCTO_FLAG: 'off'}, ''), ([], {}))
        boot = Boot()
        boot.alternate(2)
        problems, facts = octo_judge.judge({}, boot.text())
        self.assertEqual(len(problems), 1)
        self.assertIn('line(s) on a profile without QWEN_FAST_OCTO', problems[0])

    def test_a_flag_set_and_the_shape_never_executed_is_the_mounted_not_executed_case(self):
        boot = Boot(programs=lambda: 5000)
        for _ in range(40):
            boot.round('m3', eligible=False, live=3)             # every round on the M3 blocks: the texts would equal the control's
        problems, facts = octo_judge.judge(env_of(), boot.text())
        self.assertTrue(any('only 0 round(s) ran on the octo block' in problem for problem in problems), problems)
        self.assertTrue(any('only 0 counted octo round(s) under alternate' in problem for problem in problems), problems)
        # no admission at all
        problems, facts = octo_judge.judge(env_of(), '')
        self.assertTrue(any('0 [OCTO] admitted lines' in problem for problem in problems), problems)

    def test_an_octo_round_must_be_eight_rows_with_enough_live_seats(self):
        boot = Boot('live', programs=lambda: 5000)
        for _ in range(20):
            boot.round('octo')
        boot.round('octo', live=5)
        problems, facts = octo_judge.judge(env_of('live'), boot.text())
        self.assertTrue(any('ran as octo at rows=8 live=5' in problem for problem in problems), problems)

    def test_the_counted_rounds_must_alternate(self):
        boot = Boot(programs=lambda: 5000)
        boot.alternate(10)
        boot.state.next_shape = 'octo'                            # a broken alternation: octo twice in a row
        boot.round('octo')
        boot.state.next_shape = 'octo'
        boot.round('octo')
        boot.alternate(10)
        problems, facts = octo_judge.judge(env_of(), boot.text())
        self.assertTrue(any('the counted rounds do not alternate: counted round' in problem for problem in problems), problems)

    def test_a_counted_m3_round_under_live_is_refused(self):
        boot = Boot('live', programs=lambda: 5000)
        for _ in range(20):
            boot.round('octo')
        boot.state.plan_eligible = True
        boot.state.planned = 'm3'
        boot.now[0] += 0.02
        boot.state.begin()
        boot.state.finish('m3', live=8, rows=16, committed=40)
        problems, facts = octo_judge.judge(env_of('live'), boot.text())
        self.assertTrue(any('counted m3 round(s) under live' in problem for problem in problems), problems)

    def test_too_many_fall_backs_fail_the_arm(self):
        boot = Boot('live', programs=lambda: 5000)
        for _ in range(20):
            boot.round('octo')
        for _ in range(4):
            boot.state.plan(True)
            boot.now[0] += 0.02
            boot.state.begin()
            boot.state.finish('seq', live=5, rows=4, committed=10)
        problems, facts = octo_judge.judge(env_of('live'), boot.text())
        self.assertEqual(facts['fallbacks'], 4)
        self.assertTrue(any('4 of 24 eligible rounds ran as something else than planned' in problem for problem in problems), problems)
        # one in twenty-five stays inside the share
        boot = Boot('live', programs=lambda: 5000)
        for _ in range(24):
            boot.round('octo')
        boot.state.plan(True)
        boot.now[0] += 0.02
        boot.state.begin()
        boot.state.finish('seq', live=5, rows=4, committed=10)
        self.assertEqual(octo_judge.judge(env_of('live'), boot.text())[0], [])

    def test_a_program_compiled_on_a_shape_switch_fails_the_arm_and_so_does_a_first_round_that_compiled(self):
        readings = iter([5000, 5000] + [5000, 5003] + [5003] * 400)         # the first round: clean; the second round (the switch to m3): +3
        boot = Boot(programs=lambda: next(readings))
        boot.alternate(20)
        problems, facts = octo_judge.judge(env_of(), boot.text())
        self.assertEqual(facts['programs_compiled'], 3)
        self.assertTrue(any('3 program(s) compiled on the first m3 round after a switch' in problem and '5000 -> 5003' in problem for problem in problems), problems)
        readings = iter([5000, 5004] + [5004] * 400)
        boot = Boot(programs=lambda: next(readings))
        boot.alternate(20)
        problems, facts = octo_judge.judge(env_of(), boot.text())
        self.assertTrue(any('compiled on the first octo round after the process start' in problem for problem in problems), problems)

    def test_an_unreadable_program_counter_means_the_tripwire_did_not_run(self):
        boot = Boot(programs=None)
        boot.alternate(20)
        problems, facts = octo_judge.judge(env_of(), boot.text())
        self.assertTrue(any('the program counter was unreadable' in problem for problem in problems), problems)

    def test_dropped_round_lines_are_seen_by_the_counters_on_the_lines_themselves(self):
        boot = Boot(programs=lambda: 5000)
        boot.alternate(20)
        lines = [line for line in boot.lines if not (line.startswith('[OCTO] round=7 ') or line.startswith('[OCTO] round=8 '))]
        problems, facts = octo_judge.judge(env_of(), '\n'.join(lines))
        self.assertTrue(any('lines are missing' in problem for problem in problems), problems)

    def test_the_admission_must_match_the_profile_and_a_refusal_line_fails_the_arm(self):
        boot = Boot('live', programs=lambda: 5000)
        for _ in range(20):
            boot.round('octo')
        problems, facts = octo_judge.judge(env_of('alternate'), boot.text())
        self.assertTrue(any('the admission says mode=live min_live=6 and the profile asks alternate 6' in problem for problem in problems), problems)
        problems, facts = octo_judge.judge(env_of('live'), boot.text() + '[OCTO] refused: something\n')
        self.assertTrue(any('1 [OCTO] refused line(s)' in problem for problem in problems), problems)

    def test_a_malformed_mode_is_a_problem_not_a_pass(self):
        problems, facts = octo_judge.judge({octo.OCTO_FLAG: 'on'}, '')
        self.assertEqual(len(problems), 1)
        self.assertIn("QWEN_FAST_OCTO='on' is neither off, live nor alternate", problems[0])


class SoloPackedTests(unittest.TestCase):
    ENV = {octo.SOLO_PACKED_FLAG: '1'}
    RAN = '[PINDIAG] packed padded round live=1 round=3 segments=0 idle=1,2,3 padded=1'
    SKIPPED = '[PINDIAG] packed padded skipped live=1 eligible=1 reason=entries=1_block_users=4'

    def test_the_lone_user_round_must_have_run_and_no_eligible_lone_round_may_have_gone_to_the_engines(self):
        self.assertEqual(octo_judge.judge(self.ENV, self.RAN + '\n'), ([], dict(solo_packed_rounds=1, solo_packed_skipped_eligible=0)))
        problems, facts = octo_judge.judge(self.ENV, 'nothing\n')
        self.assertTrue(any('not one padded round of a lone live user ran' in problem for problem in problems), problems)
        problems, facts = octo_judge.judge(self.ENV, self.RAN + '\n' + self.SKIPPED + '\n')
        self.assertTrue(any('1 eligible lone-user round(s) went to the per-request engines' in problem for problem in problems), problems)

    def test_an_ineligible_lone_round_is_not_a_reason_and_the_flag_off_reads_none_of_it(self):
        self.assertEqual(octo_judge.judge(self.ENV, self.RAN + '\n[PINDIAG] packed padded skipped live=1 eligible=0 reason=x\n')[0], [])
        self.assertEqual(octo_judge.judge({}, self.SKIPPED + '\n'), ([], {}))

    def test_the_two_patterns_are_the_producers_own_lines(self):
        import packed_verifier

        self.assertTrue(octo_judge.PADDED_LIVE1.search('%s live=1 round=1 segments=0 idle=1,2,3 padded=1' % packed_verifier.PADDED_ROUND_MARKER))
        self.assertTrue(octo_judge.PADDED_SKIPPED_LIVE1.search('%s live=1 eligible=1 reason=x' % packed_verifier.PADDED_SKIPPED_MARKER))
        self.assertFalse(octo_judge.PADDED_LIVE1.search('%s live=2 round=1' % packed_verifier.PADDED_ROUND_MARKER))


class PairedTimingTests(unittest.TestCase):
    def boot(self, octo_committed, m3_committed, *, pairs=21, octo_step=50.0, m3_step=96.0, live=8):
        boot = Boot(programs=lambda: 5000)
        for _ in range(pairs):
            boot.round('octo', committed=octo_committed, step=octo_step, live=live)
            boot.round('m3', committed=m3_committed, step=m3_step, live=live)
        return boot

    def test_the_verdict_is_go_at_ten_percent_more_committed_tokens_per_second_per_seat(self):
        # octo: 32 tokens in 70 ms (50 step + 20 gap) = 57.1 tok/s per seat at 8 live; m3: 42 in 116 ms = 45.3: x1.26
        verdict = octo_judge.pair_verdict(self.boot(32, 42).text())
        self.assertTrue(verdict['go'], verdict)
        self.assertEqual(verdict['pairs'], 20, 'the first round of a boot has no gap: its pair is left out')
        self.assertAlmostEqual(verdict['octo_rate'], 32 / 8.0 / 0.070, places=1)
        self.assertAlmostEqual(verdict['m3_rate'], 42 / 8.0 / 0.116, places=1)
        self.assertGreater(verdict['ratio'], 1.25)
        self.assertAlmostEqual(verdict['median_pair_ratio'], verdict['ratio'], places=3)

    def test_a_gain_under_ten_percent_is_no_go(self):
        verdict = octo_judge.pair_verdict(self.boot(32, 42, octo_step=64.0).text())     # 32 in 84 ms = 47.6 against 45.3: x1.05
        self.assertFalse(verdict['go'], verdict)
        self.assertAlmostEqual(verdict['ratio'], 1.05, places=2)

    def test_the_median_and_the_aggregate_must_both_clear_the_bar(self):
        # a few enormous octo rounds lift the aggregate, not the median: NO-GO although the aggregate clears the bar
        boot = Boot(programs=lambda: 5000)
        for number in range(21):
            boot.round('octo', committed=26 if number < 14 else 3200)            # fourteen pairs lose (26 in 70 ms against 42 in 116), seven win hugely
            boot.round('m3', committed=42)
        verdict = octo_judge.pair_verdict(boot.text())
        self.assertGreater(verdict['ratio'], octo_judge.MIN_GAIN)
        self.assertLess(verdict['median_pair_ratio'], octo_judge.MIN_GAIN)
        self.assertFalse(verdict['go'], verdict)
        # and the reverse: the median clears, the aggregate does not
        boot = Boot(programs=lambda: 5000)
        for number in range(21):
            boot.round('octo', committed=32 if number < 14 else 0)               # fourteen pairs win x1.26, seven commit nothing
            boot.round('m3', committed=42)
        verdict = octo_judge.pair_verdict(boot.text())
        self.assertGreaterEqual(verdict['median_pair_ratio'], octo_judge.MIN_GAIN)
        self.assertLess(verdict['ratio'], octo_judge.MIN_GAIN)
        self.assertFalse(verdict['go'], verdict)

    def test_fewer_than_sixteen_pairs_at_eight_live_are_unjudged_not_go(self):
        verdict = octo_judge.pair_verdict(self.boot(32, 42, pairs=16).text())
        self.assertIsNone(verdict['go'], '16 pairs less the one without a gap')
        self.assertTrue(octo_judge.pair_verdict(self.boot(32, 42, pairs=17).text())['go'])
        verdict = octo_judge.pair_verdict(self.boot(32, 42, live=7).text())
        self.assertIsNone(verdict['go'], 'only rounds at eight live seats are paired')
        self.assertEqual(octo_judge.pair_verdict('')['pairs'], 0)

    def test_a_seat_rate_is_committed_per_live_seat_over_the_cycle(self):
        item = dict(step_ms=50.0, gap_ms=20.0, committed=32, live=8)
        self.assertAlmostEqual(octo_judge.seat_rate(item), 57.142857, places=3)
        self.assertIsNone(octo_judge.seat_rate(dict(step_ms=0.0, gap_ms=0.0, committed=1, live=8)))


class SmokeCheckTests(unittest.TestCase):
    def container(self, boot):
        return boot.text()

    def test_check_runs_the_octo_rules_once_and_reports_their_facts(self):
        boot = Boot(programs=lambda: 5000)
        boot.alternate(20)
        with patch.object(c2_smoke_check, 'octo_container_problems', wraps=c2_smoke_check.octo_container_problems) as ran:
            problems, facts = c2_smoke_check.check('', boot.text(), False, env=env_of())
        self.assertEqual(ran.call_count, 1)
        self.assertEqual([problem for problem in problems if 'octo' in problem.lower() or '[OCTO]' in problem], [])
        self.assertEqual(facts['octo']['counted_per_shape'], dict(octo=20, m3=20))

    def test_check_fails_an_arm_whose_shape_never_ran(self):
        boot = Boot(programs=lambda: 5000)
        for _ in range(30):
            boot.round('m3', eligible=False, live=3)
        problems, facts = c2_smoke_check.check('', boot.text(), False, env=env_of())
        self.assertTrue(any('ran on the octo block' in problem for problem in problems), problems)

    def test_an_ordinary_profile_and_log_add_no_problem_and_no_fact(self):
        problems, facts = c2_smoke_check.check('', 'plain\n', False, env={'QWEN_FAST_TP': '4'})
        self.assertFalse([problem for problem in problems if 'OCTO' in problem or 'octo' in problem])
        self.assertNotIn('octo', facts)

    def test_the_eight_seat_hostgap_rule_counts_a_third_block_when_octo_is_on(self):
        # the engaged line is written once per block the pre-stage engaged over: three with the octo block
        engaged = c2_smoke_check.HOSTGAP_ENGAGED
        env = dict(env_of(), **{c2_smoke_check.HOSTGAP_TWO_BLOCK_FLAG: '1', 'QWEN_FAST_M3_BLOCKS': '2'})
        for count, octo_on, clean in ((3, True, True), (2, True, False), (2, False, True), (3, False, False)):
            with self.subTest(count=count, octo=octo_on):
                this = dict(env) if octo_on else {key: value for key, value in env.items() if key != octo.OCTO_FLAG}
                log = '\n'.join('%s block=%s users=4 mode=first' % (engaged, label) for label in 'ABC'[:count])
                problems, facts = c2_smoke_check.hostgap_problems(this, log, False)
                complaint = [problem for problem in problems if 'not once per M3 block' in problem]
                self.assertEqual(not complaint, clean, problems)

    def test_the_command_line_prints_the_facts_the_problems_and_the_verdict(self):
        boot = Boot(programs=lambda: 5000)
        boot.alternate(20)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'container.log')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(boot.text())
            arguments = ['--container-log', path] + sum((['--env', '%s=%s' % item] for item in env_of().items()), [])
            with patch('sys.stdout', new_callable=io.StringIO) as out:
                self.assertEqual(octo_judge.main(arguments + ['--verdict']), 0)
            printed = out.getvalue()
            self.assertIn('OCTO_JUDGE {', printed)
            self.assertIn('OCTO_VERDICT GO:', printed)
            with patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(octo_judge.main(['--container-log', os.path.join(folder, 'missing.log')]), 2)
            slow = Boot(programs=lambda: 5000)
            for _ in range(21):
                slow.round('octo', committed=32, step=80.0)
                slow.round('m3', committed=40, step=96.0)
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(slow.text())
            with patch('sys.stdout', new_callable=io.StringIO) as out:
                self.assertEqual(octo_judge.main(arguments + ['--verdict']), 1)
            self.assertIn('OCTO_VERDICT NO-GO', out.getvalue())


def smoke_log(users_by_test):
    results = {}
    for test, users in users_by_test.items():
        results[test] = dict(users=[dict(content_sha256='c%d' % index if hash_ is None else hash_, reasoning_sha256='r%d' % index, tokens=800, finish='length',
                                         decode_tok_s=rate) for index, (hash_, rate) in enumerate(users)])
    return 'noise\nSMOKE_JSON ' + json.dumps(results) + '\n'


class CompareTests(unittest.TestCase):
    """octo_compare: an octo arm against its flag-off control, answer for answer."""

    def logs(self, control_rate=25.0, arm_rate=33.0, arm_hash=None):
        control = smoke_log({'concurrent8_code_equal': [(None, control_rate)] * 8, 'concurrent8_code': [(None, control_rate)] * 8})
        users = [(None, arm_rate)] * 8
        if arm_hash is not None:
            users[3] = (arm_hash, arm_rate)
        return control, smoke_log({'concurrent8_code_equal': users, 'concurrent8_code': [(None, arm_rate)] * 8})

    def test_identical_answers_pass_and_the_rates_are_reported_never_gated(self):
        import octo_compare

        control, arm = self.logs()
        problems, compared, lines = octo_compare.compare(control, arm)
        self.assertEqual(problems, [])
        self.assertEqual(compared, 16)
        rate = [line for line in lines if line.startswith('OCTO_COMPARE rate concurrent8_code_equal')][0]
        self.assertEqual(rate, 'OCTO_COMPARE rate concurrent8_code_equal control=25.0 octo=33.0 ratio=1.32')
        slower = octo_compare.compare(*self.logs(arm_rate=20.0))
        self.assertEqual(slower[0], [], 'a slower arm is a number, not a mismatch')

    def test_one_differing_hash_is_a_mismatch_naming_the_user(self):
        import octo_compare

        control, arm = self.logs(arm_hash='deadbeef')
        problems, compared, lines = octo_compare.compare(control, arm)
        self.assertEqual(len(problems), 1)
        self.assertIn('concurrent8_code_equal user 3: content hash differs (control \'c3\', interleaved \'deadbeef\')', problems[0])

    def test_the_arms_container_log_is_judged_too(self):
        import octo_compare

        control, arm = self.logs()
        boot = Boot(programs=lambda: 5000)
        boot.alternate(20)
        self.assertEqual(octo_compare.compare(control, arm, boot.text(), env_of())[0], [])
        problems, compared, lines = octo_compare.compare(control, arm, 'a log with no octo line at all\n', env_of())
        self.assertTrue(any(problem.startswith('octo arm: ') and 'octo block' in problem for problem in problems), problems)

    def test_the_command_line_exits_zero_one_and_two(self):
        import octo_compare

        control, arm = self.logs()
        _, bad = self.logs(arm_hash='deadbeef')
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for name, text in (('control', control), ('arm', arm), ('bad', bad), ('empty', 'no smoke json here\n')):
                paths[name] = os.path.join(folder, name + '.log')
                with open(paths[name], 'w', encoding='utf-8') as handle:
                    handle.write(text)
            with patch('sys.stdout', new_callable=io.StringIO) as out:
                self.assertEqual(octo_compare.main(['--control', paths['control'], '--octo', paths['arm']]), 0)
            self.assertIn('OCTO_COMPARE {"ok": true, "compared": 16, "problems": 0}', out.getvalue())
            with patch('sys.stdout', new_callable=io.StringIO) as out:
                self.assertEqual(octo_compare.main(['--control', paths['control'], '--octo', paths['bad']]), 1)
            self.assertIn('OCTO_COMPARE MISMATCH', out.getvalue())
            with patch('sys.stdout', new_callable=io.StringIO), patch('sys.stderr', new_callable=io.StringIO):
                self.assertEqual(octo_compare.main(['--control', paths['control'], '--octo', paths['empty']]), 1, 'an arm with no smoke json is a mismatch')
                self.assertEqual(octo_compare.main(['--control', paths['control'], '--octo', os.path.join(folder, 'missing.log')]), 2)


class MarkersTests(unittest.TestCase):
    def test_the_producers_lines_parse_and_a_foreign_line_does_not(self):
        boot = Boot(programs=lambda: 5000)
        boot.alternate(3)
        found = octo_markers.scan(boot.text() + 'noise [OCTO] round=zero\n')
        self.assertEqual(len(found['rounds']), 6)
        self.assertEqual(len(found['programs']), 6, 'every round of an alternate boot is the first of its shape after a switch')
        self.assertEqual(len(found['admitted']), 1)
        self.assertGreaterEqual(found['unqualified'], 1)
        self.assertEqual(found['refused'], 0)
        self.assertEqual([item['shape'] for item in found['rounds']], ['octo', 'm3'] * 3)
        self.assertEqual(found['rounds'][-1]['octo_rounds'], 3)
        self.assertTrue(all(len(line) < 180 for line in boot.lines))

    def test_the_markers_module_is_stdlib_only_python_37(self):
        import ast

        with open(octo_markers.__file__.replace('.pyc', '.py'), encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), feature_version=(3, 7))
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {'re'})
        with open(octo_judge.__file__.replace('.pyc', '.py'), encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), feature_version=(3, 7))
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {'argparse', 'json', 'os', 'octo_markers', 're', 'sys'})


if __name__ == '__main__':
    unittest.main()
