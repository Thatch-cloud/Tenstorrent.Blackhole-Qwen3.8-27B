"""c2_smoke_check on a LADDER-ONLY smoke (warmup and prefill_ladder_solo: a 16-token answer per rung).

Run OQ-T1 (v791, run 38095202391; the control ladder on the combined arm) failed its smoke check only because the lever-engagement rules asked for lines the speculative decode
rounds log: the drafter's attention branch (permute, q|k|v, 64-row head), the host-side per-round levers (lean write_packed, batched read-backs, pre-stage diff) and their site
rules, the eight-seat quad blocks, the two-block pre-stage and the ramp commit time. A run that never decodes past 16 tokens per rung cannot show them, so they are 'not asked'
(facts['not_asked'], one 'SMOKE_CHECK not asked' line each). These tests hold both halves: not asked in a ladder-only run, every rule exactly as before wherever decode runs
(any other test in the smoke, none readable), and never waived: fall-back lines, audit mismatches, and the levers that act in prefill.

Run from scripts/ci:  python -B -m unittest test_smoke_ladder_only
"""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as smoke_check  # noqa: E402

PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
FX = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-all'
FX_AUDIT = FX + '-audit'
FX_ONEQ = FX + '-oneq'
FX_ONEQ_AUDIT = FX + '-oneq-audit'

RUNG = {'prompt_tokens': 4062, 'ttft_s': 1.8, 'tokens': 16, 'finish': 'length', 'text': 'We need answer user request. Need analyze', 'wall_s': 2.3}
LADDER = {'warmup': {'value': 200, 'wall_s': 0.7}, 'prefill_ladder_solo': {'rungs': {'4096': dict(RUNG)}}}
DECODE_ONLY_WHAT = ('F-F2 drafter permutation kernels', 'F-F3c fused drafter q|k|v projection', 'F-F4 one 64-row drafter head', 'WPH-2 lean write_packed',
                    'WPH-3 batched read-backs (asynchronous copies)')


def smoke_text(extra=None, tests=None):
    results = dict(LADDER if tests is None else tests)
    results.update(extra or {})
    return 'SMOKE_JSON ' + json.dumps(results) + '\n'


def env_of(name):
    return dict(PROFILES[name]['env'])


def engaged_lines(env, only=None, skip_decode_only=False):
    """The engaged (and, for the audit flags the profile sets, passing audit) lines of the table levers the profile asks for."""
    lines = []
    for flag, value, engaged, _fell, _what in smoke_check.FUSION_LEVERS:
        if env.get(flag) == value and (only is None or flag in only) and not (skip_decode_only and smoke_check.decode_only_flag(flag)):
            lines.append(engaged + ' (test)')
    for flag, marker, _what in smoke_check.FUSION_AUDITS:
        if env.get(flag) == '1' and (only is None or flag in only) and not (skip_decode_only and smoke_check.decode_only_flag(flag)):
            lines.append(marker + ' n=8 exact=True')
    return lines


def container(env, **kwargs):
    return '\n'.join(engaged_lines(env, **kwargs)) + '\n'


def check(env, text, smoke):
    problems, facts = smoke_check.check(smoke, text, slide=True, env=env, entry=None)
    return problems, facts


def mentions(problems, *needles):
    return [problem for problem in problems if all(needle in problem for needle in needles)]


STEADY = {'concurrent8_steady': {'users': []}}
# Words that identify a problem as one of the waived families (a decode-only lever's name, its flag, the packages' site rules, the steady-mix conditions, the ramp commit).
WAIVED_WORDS = tuple(DECODE_ONLY_WHAT) + tuple(smoke_check.DECODE_ONLY_FLAGS) + (
    'draft permute engaged site', 'lean writer', 'batched reader', 'PACKED-PRESTAGE-DIFF', 'blocks were not judged', 'lever was not judged', 'no [PACKED-PUBLISH] round with a commit',
    '[PINDIAG] tp4 draft', 'drafter')


def waived(env, text):
    """(problems of the log under a ladder-only smoke, under a decode smoke (concurrent8_steady added), and the ones the ladder-only smoke is NOT asked relative to the old
    behaviour: the same ladder-only smoke read with the waiver switched off)."""
    from unittest import mock

    ladder, _ = check(env, text, smoke_text())
    decode, _ = check(env, text, smoke_text(STEADY))
    with mock.patch.object(smoke_check, 'ladder_only', lambda smoke: False):
        strict, _ = check(env, text, smoke_text())
    return ladder, decode, [problem for problem in strict if problem not in ladder]


class LadderOnlyPredicateTests(unittest.TestCase):
    def test_only_warmup_and_the_solo_ladder_are_ladder_only(self):
        self.assertTrue(smoke_check.ladder_only(LADDER))
        self.assertTrue(smoke_check.ladder_only({'prefill_ladder_solo': {}}))
        self.assertTrue(smoke_check.ladder_only({'warmup': {}}))

    def test_anything_else_in_the_smoke_is_a_decode_run(self):
        for extra in ('concurrent8_steady', 'concurrent4_steady', 'prefill_ladder_busy', 'prefill_few_decoders', 'coding', 'concurrent8_code', 'some_future_test'):
            with self.subTest(test=extra):
                self.assertFalse(smoke_check.ladder_only(dict(LADDER, **{extra: {}})))

    def test_no_smoke_or_an_unreadable_one_is_strict(self):
        for value in (None, {}, [], 'warmup'):
            with self.subTest(value=value):
                self.assertFalse(smoke_check.ladder_only(value))

    def test_the_decode_only_set_is_exactly_the_drafter_and_the_per_round_host_levers(self):
        self.assertEqual(smoke_check.DECODE_ONLY_FLAGS, frozenset((
            'QWEN_FAST_DRAFT_REDUCE', 'QWEN_FAST_DRAFT_TAIL', 'QWEN_FAST_DRAFT_GATEUP1', 'QWEN_FAST_DRAFT_MM_GRID', 'QWEN_FAST_DRAFT_PERMUTE', 'QWEN_FAST_DRAFT_QKV1',
            'QWEN_FAST_DRAFT_HEAD64', 'QWEN_FAST_PRESTAGE_DIFF', 'QWEN_FAST_WRITE_PACKED_LEAN', 'QWEN_FAST_BATCHED_READS', 'QWEN_FAST_TP4_DRAFT_CONV',
            'QWEN_FAST_TP4_DRAFT_HEADS')))
        for flag in ('QWEN_FAST_SDPA_PF_ONEQ', 'QWEN_FAST_GRID13', 'QWEN_FAST_CCL_OPTIONS', 'QWEN_FAST_KV_PAGE_WRITER', 'QWEN_FAST_MLP_CFG', 'QWEN_FAST_MLP_GATEUP',
                     'QWEN_FAST_TP4_SHARD_ARGMAX', 'QWEN_FAST_DEVICE_ZEROS', 'QWEN_FAST_LAZY_SHARD_W'):
            self.assertFalse(smoke_check.decode_only_flag(flag), flag)
            self.assertFalse(smoke_check.decode_only_flag(flag + '_AUDIT'), flag)
        self.assertTrue(smoke_check.decode_only_flag('QWEN_FAST_DRAFT_PERMUTE_AUDIT'))

    def test_every_decode_only_flag_is_a_lever_of_a_table_and_every_rule_is_a_fusion_rule(self):
        known = {flag for flag, *_ in smoke_check.FUSION_LEVERS} | {flag for flag, *_ in smoke_check.SAMPDRAFT_LEVERS}
        self.assertEqual(sorted(smoke_check.DECODE_ONLY_FLAGS - known), [])
        self.assertEqual(sorted(set(smoke_check.DECODE_ONLY_RULES) - set(smoke_check.FUSION_RULES)), [])


class LadderOnlyRunTests(unittest.TestCase):
    """The combined arm's ladder, as run OQ-T1 logged it: the prefill levers' lines are in the log, the decode-only levers' are not."""

    def setUp(self):
        self.env = env_of(FX)
        self.log = container(self.env, skip_decode_only=True)

    def test_the_ladder_only_run_is_not_asked_for_the_decode_only_levers(self):
        ladder, decode, dropped = waived(self.env, self.log)
        for what in DECODE_ONLY_WHAT:
            self.assertEqual(mentions(ladder, what), [], what)
            self.assertTrue(mentions(decode, what, 'never ran'), what)
        for text in ('draft permute engaged site=kv', 'draft permute engaged site=fold', 'draft permute engaged site=unfold', 'lean writer', 'batched reader', 'PACKED-PRESTAGE-DIFF'):
            self.assertEqual(mentions(ladder, text), [], text)
            self.assertTrue(mentions(decode, text), text)
        for text in ('blocks were not judged', 'lever was not judged', 'no [PACKED-PUBLISH] round with a commit'):
            self.assertEqual(mentions(ladder, text), [], text)
        _, facts = check(self.env, self.log, smoke_text())
        text = ' | '.join(facts['not_asked'])
        for what in DECODE_ONLY_WHAT:
            self.assertIn(what, text)
        for needle in ('draft_permute_smoke', 'concurrent8_steady', 'ramp commit'):
            self.assertIn(needle, text)

    def test_a_ladder_only_run_drops_nothing_but_the_waived_families_and_adds_nothing(self):
        ladder, decode, dropped = waived(self.env, self.log)
        self.assertTrue(dropped)
        self.assertEqual(sorted(dropped), sorted(set(dropped)), 'each dropped problem once')
        for problem in dropped:
            self.assertTrue(any(word in problem for word in WAIVED_WORDS), problem)
        ladder_problems, _, _ = waived(self.env, self.log)
        self.assertEqual([problem for problem in ladder_problems if problem not in decode], [], 'a ladder-only run raises nothing a decode run does not')
        kept = [problem for problem in ladder if problem in decode]
        self.assertTrue(mentions(kept, 'QWEN_FAST_TP4_SHARD_ARGMAX', 'engaged'), 'the sampler lever is judged in a ladder-only run')

    def test_the_same_log_with_a_decode_test_fails_every_one_of_them(self):
        problems, facts = check(self.env, self.log, smoke_text(STEADY))
        self.assertNotIn('not_asked', facts)
        for what in DECODE_ONLY_WHAT:
            self.assertTrue(mentions(problems, what, 'never ran'), what)
        self.assertTrue(mentions(problems, 'draft permute engaged site=kv'), problems)
        self.assertTrue(mentions(problems, 'lean writer'), problems)
        self.assertTrue(mentions(problems, 'batched reader'), problems)
        self.assertTrue(mentions(problems, 'no [PACKED-PUBLISH] round with a commit'), problems)

    def test_the_strict_messages_are_the_old_ones_for_no_smoke_and_for_a_busy_ladder(self):
        for smoke in ('', smoke_text({'prefill_ladder_busy': {}})):
            with self.subTest(smoke=smoke[:40]):
                problems, facts = check(self.env, self.log, smoke)
                self.assertNotIn('not_asked', facts)
                for what in DECODE_ONLY_WHAT:
                    self.assertTrue(mentions(problems, what, 'never ran'), what)

    def test_the_blocks_and_pre_stage_conditions_are_not_asked_without_steady_and_asked_otherwise(self):
        ladder, _ = check(self.env, self.log, smoke_text())
        self.assertEqual(mentions(ladder, 'concurrent8_steady'), [], ladder)
        busy, _ = check(self.env, self.log, smoke_text({'prefill_ladder_busy': {}}))
        self.assertTrue(mentions(busy, 'concurrent8_steady', 'not judged'), busy)

    def test_the_prefill_levers_are_judged_in_a_ladder_only_run(self):
        env = env_of(FX_ONEQ)
        log = container(env, skip_decode_only=True)
        problems, facts = check(env, log, smoke_text())
        self.assertEqual(mentions(problems, 'S1 prefill SDPA one q chunk per core'), [], problems)
        self.assertEqual(mentions(problems, 'QWEN_FAST_SDPA_PF_ONEQ'), [], problems)
        for flag in ('QWEN_FAST_SDPA_PF_ONEQ', 'QWEN_FAST_CCL_OPTIONS', 'QWEN_FAST_KV_PAGE_WRITER', 'QWEN_FAST_MLP_CFG'):
            engaged = next(line for f, v, line, _fell, _what in smoke_check.FUSION_LEVERS if f == flag and env.get(f) == v)
            missing = log.replace(engaged + ' (test)', '')
            self.assertNotEqual(missing, log, flag)
            problems, _ = check(env, missing, smoke_text())
            self.assertTrue([problem for problem in problems if flag in problem or engaged in problem], (flag, problems))

    def test_a_fall_back_of_a_decode_only_lever_still_fails_a_ladder_only_run(self):
        for _flag, _value, _engaged, fell_back, what in smoke_check.FUSION_LEVERS:
            if what in DECODE_ONLY_WHAT:
                problems, _ = check(self.env, self.log + fell_back + ' reason=x\n', smoke_text())
                self.assertTrue(mentions(problems, what, 'fell back'), what)

    def test_an_audit_mismatch_still_fails_a_ladder_only_run(self):
        problems, _ = check(self.env, self.log + '[PINDIAG] tp4 draft permute audit mismatch layer=3\n', smoke_text())
        self.assertTrue(mentions(problems, 'audit mismatch'), problems)

    def test_the_audited_arm_is_not_asked_for_decode_audits_but_is_asked_for_the_prefill_ones(self):
        env = env_of(FX_ONEQ_AUDIT)
        log = container(env, skip_decode_only=True)
        ladder, decode, _ = waived(env, log)
        for flag in ('QWEN_FAST_DRAFT_PERMUTE_AUDIT', 'QWEN_FAST_DRAFT_QKV1_AUDIT', 'QWEN_FAST_WRITE_PACKED_LEAN_AUDIT', 'QWEN_FAST_BATCHED_READS_AUDIT'):
            self.assertEqual(mentions(ladder, flag, 'never audited'), [], flag)
            self.assertTrue(mentions(decode, flag, 'never audited'), flag)
        _, facts = check(env, log, smoke_text())
        self.assertIn('no audit line (exact=True) required of the decode-only levers', ' | '.join(facts['not_asked']))
        oneq_audit = '[PINDIAG] sdpa prefill oneq audit n=8 exact=True'
        self.assertIn(oneq_audit, log)
        problems, _ = check(env, log.replace(oneq_audit, ''), smoke_text())
        self.assertTrue(mentions(problems, 'QWEN_FAST_SDPA_PF_ONEQ_AUDIT', 'never audited'), problems)

    def test_the_gates_that_share_the_rules_stay_strict(self):
        """lever_engagement_problems (the prefix and serving gates) takes no smoke-derived waiver."""
        problems = smoke_check.lever_engagement_problems(self.env, self.log)
        for what in DECODE_ONLY_WHAT:
            self.assertTrue(mentions(problems, what, 'never ran'), what)


class MainPrintsTheNotAskedLinesTests(unittest.TestCase):
    def run_main(self, smoke, log, profile=FX):
        with tempfile.TemporaryDirectory() as directory:
            smoke_path, log_path = Path(directory, 'smoke.log'), Path(directory, 'container.log')
            smoke_path.write_text(smoke, encoding='utf-8')
            log_path.write_text(log, encoding='utf-8')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = smoke_check.main(['--smoke-log', str(smoke_path), '--container-log', str(log_path), '--profile', profile])
        return code, out.getvalue()

    def test_a_ladder_only_run_prints_what_was_not_asked_and_the_decode_run_prints_the_failures(self):
        env = env_of(FX)
        log = container(env, skip_decode_only=True)
        _code, text = self.run_main(smoke_text(), log)
        lines = [line for line in text.splitlines() if line.startswith('SMOKE_CHECK not asked: ladder-only run')]
        self.assertTrue(lines, text)
        self.assertTrue(all('warmup, prefill_ladder_solo' in line for line in lines))
        failed = [line for line in text.splitlines() if line.startswith('SMOKE_CHECK FAILED')]
        for what in DECODE_ONLY_WHAT:
            self.assertEqual(mentions(failed, what), [], what)
        code, text = self.run_main(smoke_text({'concurrent8_steady': {'users': []}}), log)
        self.assertEqual(code, 1)
        self.assertNotIn('SMOKE_CHECK not asked', text)
        self.assertIn('SMOKE_CHECK FAILED: QWEN_FAST_DRAFT_PERMUTE=1 is set and no engaged line', text)


if __name__ == '__main__':
    unittest.main()
