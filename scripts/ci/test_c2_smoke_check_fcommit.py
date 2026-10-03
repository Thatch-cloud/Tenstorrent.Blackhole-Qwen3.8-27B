"""c2_smoke_check's fused-commit stop conditions (the tp4/fcommit window): what the audited and timed smokes must have logged.

The lines are built from the fused commit's own formats: the publication line by the module's `note`, the audit line by the pinned
audit's text, the engaged line by the four-card twin's engaged_line() template (test_fused_commit_tp4 holds that against a real
capture). The check reads them with the gate's own regexes (lever_n_m3native_gate.h1b_report), so a change of either format fails here.

    py -3.11 -B -m unittest test_c2_smoke_check_fcommit      (from scripts/ci)
"""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import c2_smoke_check as check
import fused_commit as pinned
import fused_commit_tp as twin
import lever_n_m3native_gate as gate
import test_fused_commit as base
import test_speed_window_compare as sw

HERE = Path(__file__).resolve().parent
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']


NL = chr(10)


def env_of(profile):
    return PROFILES[profile]['env']


def engaged(traces=68, inplace=1, live=1, audit=1, tp=4):
    return ('[PINDIAG] fused commit engaged users=4 rows=16 inplace=%d live_banks=%d audit=%d kernel=direct traces=%d '
            'layout=1x80 tp=%d workers=8 tproj_ms=1.2 slide_ms=0.4' % (inplace, live, audit, traces, tp))


def publications(rounds, *, reasons=None, fused_users=4):
    """The pinned module's own `note` lines: `rounds` rounds of four users, every publication fused except the ones `reasons`
    ({(round, segment): reason}) sends to today's path."""
    reasons = reasons or {}
    lines, loguru = base.logged()
    holder = object.__new__(pinned.FusedCommit)
    holder.counts, holder.refusals = dict(fused=0, today=0, window=0, late=0), {}
    holder.block = SimpleNamespace(rounds=0)
    with loguru:
        for number in range(1, rounds + 1):
            holder.block.rounds = number
            for segment in range(fused_users):
                reason = reasons.get((number, segment))
                if reason is None:
                    holder.note(segment, 9, 'fused', None, 'window')
                else:
                    holder.note(segment, 9, 'today', reason)
    return lines


def audits(lines_for, checked=80, mismatches=0, mode='inplace'):
    return ['[PACKED-FUSED-AUDIT] round=%d segment=%d prefix=9 mode=%s checked=%d mismatches=%d' % (
        number, segment, mode, checked, mismatches) for number, segment in lines_for]


LIVE_MARKER = '[PINDIAG] pair live banks engaged pair=[0, 1, 2, 3] context=(2048, 2048, 2048, 2048)'


def clean_log(rounds=6, *, checked=80, audit=True, live=True, traces=68, extra=()):
    fused = publications(rounds)
    body = [engaged(traces=traces)] + ([LIVE_MARKER] if live else []) + fused
    if audit:
        body += audits([(number, segment) for number in range(1, rounds + 1) for segment in range(4)], checked=checked)
    return '\n'.join(body + list(extra))


class FormatTests(unittest.TestCase):
    def test_the_lines_are_the_modules_and_the_gates_regexes_read_them(self):
        lines = publications(2, reasons={(1, 2): 'parity'})
        self.assertEqual(lines[2], '[PACKED-FUSED] round=1 segment=2 prefix=9 path=today reason=parity tables=-')
        summary = gate.h1b_summary('\n'.join([engaged()] + lines))
        self.assertEqual((summary['publications'], summary['fused'], summary['today'], summary['today_reasons']),
                         (8, 7, 1, {'parity': 1}))
        self.assertEqual(summary['engaged']['traces'], 68)

    def test_the_twins_engaged_line_template_is_the_one_the_gate_reads(self):
        import test_fused_commit_tp4 as tp4

        fixture = tp4.CaptureTests('test_the_timing_replays_measure_the_widest_trace_of_every_segment_and_are_reported')
        fixture.setUp()
        try:
            fused = fixture.build()
            fused.capture(fixture.capture)
            line = fused.engaged_line()
        finally:
            fixture.doCleanups()
        self.assertEqual(gate.FUSED_ENGAGED_LINE.search(line).groups(), ('4', '16', '1', '0', '0', line.split('kernel=')[1].split()[0], '68'))
        self.assertEqual(line.split(' traces=')[0].split('fused commit engaged ')[1],
                         engaged().split(' traces=')[0].split('fused commit engaged ')[1].replace('live_banks=1', 'live_banks=0')
                         .replace('audit=1', 'audit=0').replace('kernel=direct', 'kernel=' + line.split('kernel=')[1].split()[0]))


class FusedProblemTests(unittest.TestCase):
    ENV = env_of('c2-packed-tp4-gate-fcommit-quad')

    def problems(self, log, env=None, steady=True):
        return check.fused_problems(self.ENV if env is None else env, log, steady)

    def test_a_clean_audited_arm_passes(self):
        self.assertEqual(self.problems(clean_log()), [])

    def test_the_expected_refusals_pass_and_any_other_reason_fails(self):
        for reason in ('ramp', 'parity'):
            log = '\n'.join([engaged(), LIVE_MARKER] + publications(5, reasons={(1, 0): reason, (1, 1): reason})
                            + audits([(n, s) for n in range(2, 6) for s in range(4)]
                                     + [(1, 2), (1, 3)]))
            self.assertEqual(self.problems(log), [], reason)
        log = '\n'.join([engaged(), LIVE_MARKER] + publications(5, reasons={(2, 3): 'scope'})
                        + audits([(n, s) for n in range(1, 6) for s in range(4) if (n, s) != (2, 3)]))
        found = self.problems(log)
        self.assertTrue(any("outside ('ramp', 'parity')" in text and 'scope' in text for text in found), found)

    def test_a_refused_build_an_unexpected_engagement_and_a_trace_count_fail(self):
        refused = '[PINDIAG] fused commit refused users=4 reason=the_four-card_slide_(QWEN_FAST_TP_KV_SLIDE=1)_is_not_live'
        found = self.problems('\n'.join([refused] + publications(2)))
        self.assertTrue(any('refused the fused commit' in text for text in found), found)
        self.assertTrue(any('engaged line' in text and '0 times' in text for text in found), found)
        found = self.problems(clean_log(traces=4))
        self.assertTrue(any('captured 4 traces, not 68' in text for text in found), found)
        found = self.problems(clean_log() + '\n' + engaged())
        self.assertTrue(any('appears 2 times' in text for text in found), found)
        found = self.problems(clean_log().replace('inplace=1', 'inplace=0'))
        self.assertTrue(any('engaged inplace=0' in text for text in found), found)

    def test_the_audit_must_cover_every_fused_publication_check_the_width_s_items_and_find_no_mismatch(self):
        found = self.problems(clean_log(checked=40))
        self.assertTrue(any('checked [40] items, not 80' in text for text in found), found)
        found = self.problems(clean_log(audit=False))
        self.assertTrue(any('audits for' in text for text in found), found)
        self.assertTrue(any('no [PACKED-FUSED-AUDIT] line was logged' in text for text in found), found)
        bad = clean_log(extra=audits([(7, 0)], mismatches=4) + ['[PINDIAG] fused commit audit mismatch round=7 segment=0 prefix=9 at=bank3v.3'])
        found = self.problems(bad)
        self.assertTrue(any('no mismatch' in text for text in found), found)

    def test_a_discard_after_an_in_place_slide_fails(self):
        found = self.problems(clean_log(extra=['[PINDIAG] fused commit discard after in-place slide segment=1 position=5000 prefix=4']))
        self.assertTrue(any('discarded after the slide moved the live banks' in text for text in found), found)

    def test_the_steady_mix_needs_a_round_of_four_fused_users_and_the_live_bank_marker(self):
        one_at_a_time = '\n'.join([engaged(), LIVE_MARKER] + publications(4, reasons={(n, n - 1): 'parity' for n in range(1, 5)})
                                  + audits([(n, s) for n in range(1, 5) for s in range(4) if (n, s) != (n, n - 1)]))
        found = self.problems(one_at_a_time)
        self.assertTrue(any('no round had all four users on the fused path' in text for text in found), found)
        self.assertEqual([text for text in self.problems(one_at_a_time, steady=False) if 'all four users' in text], [])
        found = self.problems(clean_log(live=False))
        self.assertTrue(any('live-bank marker' in text for text in found), found)
        self.assertEqual([text for text in self.problems(clean_log(live=False), steady=False) if 'live-bank' in text], [])

    def test_a_profile_without_the_flag_logs_no_fused_line_and_the_out_of_place_fallback_counts_four_traces(self):
        plain = env_of('c2-packed-tp4-speed-quad')
        self.assertEqual(check.fused_problems(plain, 'nothing here', True), [])
        found = check.fused_problems(plain, clean_log(), True)
        self.assertTrue(any('fused commit ran on a profile without it' in text for text in found), found)
        oop = env_of('c2-packed-tp4-speed-fcommit-oop')
        log = '\n'.join([engaged(traces=4, inplace=0, live=0, audit=0)] + publications(3))
        self.assertEqual(check.fused_problems(oop, log, True), [])
        found = check.fused_problems(oop, '\n'.join([engaged(traces=68, inplace=0, live=0, audit=0)] + publications(3)), True)
        self.assertTrue(any('captured 68 traces, not 4' in text for text in found), found)

    def test_the_timed_profiles_need_no_audit_and_the_pairs_audit_is_forty_per_out_of_place_publication(self):
        timed = env_of('c2-packed-tp4-speed-fcommit-quad')
        log = '\n'.join([engaged(audit=0), LIVE_MARKER] + publications(4))
        self.assertEqual(check.fused_problems(timed, log, True), [])
        oop_audited = dict(env_of('c2-packed-tp4-gate-fcommit'), QWEN_FAST_FUSED_COMMIT_INPLACE='0')
        log = '\n'.join([engaged(traces=4, inplace=0, live=0)] + publications(2)
                        + audits([(n, s) for n in (1, 2) for s in range(4)], checked=40, mode='oop'))
        self.assertEqual(check.fused_problems(oop_audited, log, False), [])


class TwoBlockTests(unittest.TestCase):
    """QWEN_FAST_M3_BLOCKS=2 (eight seats): each 64-row block builds its own fused commit and logs its own engaged line, and both
    blocks count their own rounds, so one round of eight users logs eight publication lines under one round number."""
    EIGHT = env_of('c2-packed-tp4-8-best')

    def log(self, engaged_lines=2, rounds=4, both=True):
        body = [engaged() for _ in range(engaged_lines)] + [LIVE_MARKER]
        body += publications(rounds)
        if both:
            body += publications(rounds)
        return NL.join(body)

    def problems(self, log, env=None, steady=False):
        return check.fused_problems(self.EIGHT if env is None else env, log, steady)

    def test_the_profile_is_two_blocks_and_one_block_is_the_default(self):
        self.assertEqual(self.EIGHT.get('QWEN_FAST_M3_BLOCKS'), '2')
        self.assertEqual(check.m3_blocks(self.EIGHT), 2)
        self.assertEqual(check.m3_blocks({}), 1)
        self.assertEqual(check.m3_blocks({'QWEN_FAST_M3_BLOCKS': '1'}), 1)

    def test_two_engaged_lines_one_per_block_pass_on_the_eight_seat_profile(self):
        self.assertEqual([text for text in self.problems(self.log()) if 'engaged line' in text], [])

    def test_one_or_three_engaged_lines_fail_by_name_at_two_blocks(self):
        for lines in (1, 3, 0):
            with self.subTest(lines=lines):
                found = [text for text in self.problems(self.log(engaged_lines=lines)) if 'engaged line' in text]
                self.assertEqual(len(found), 1, found)
                self.assertIn('appears %d times, not once per M3 block (2 blocks)' % lines, found[0])

    def test_one_block_still_wants_exactly_one_engaged_line(self):
        four = env_of('c2-packed-tp4-best-gate')
        one = NL.join([engaged(), LIVE_MARKER] + publications(2))
        self.assertEqual([text for text in check.fused_problems(four, one, False) if 'engaged line' in text], [])
        found = check.fused_problems(four, NL.join([engaged(), engaged(), LIVE_MARKER] + publications(2)), False)
        self.assertTrue(any('appears 2 times, not once' in text for text in found), found)

    def test_every_block_captures_its_own_trace_count(self):
        first_bad = self.log().replace(engaged(traces=68), engaged(traces=60), 1)
        found = self.problems(first_bad)
        self.assertTrue(any('captured 60 traces' in text for text in found), found)
        second_bad = NL.join([engaged(traces=68), engaged(traces=60), LIVE_MARKER] + publications(2))
        found = self.problems(second_bad)
        self.assertTrue(any('a further fused-commit block captured 60 traces, not 68' in text for text in found), found)

    def test_the_steady_round_of_both_blocks_counts_as_a_fused_round(self):
        self.assertEqual([text for text in self.problems(self.log(rounds=4, both=True), steady=True) if 'all four users' in text], [])
        no_round = NL.join([engaged(), engaged(), LIVE_MARKER] + publications(4, reasons={(n, n - 1): 'parity' for n in range(1, 5)}))
        found = self.problems(no_round, steady=True)
        self.assertTrue(any('no round had all four users on the fused path' in text for text in found), found)


class CheckIntegrationTests(unittest.TestCase):
    def test_check_runs_the_fused_rules_for_a_profile_env_and_prints_the_facts(self):
        env = env_of('c2-packed-tp4-gate-fcommit-quad')
        quad_round = '[PACKED-SELECT] round=%d pairs=[[0, 1, 2, 3]] users=4 calls=1' % 1
        log = '\n'.join([sw.QUAD_LOG, clean_log()])
        problems, facts = check.check(sw.smoke_text(sw.STEADY_OK), log, False, env=env)
        self.assertEqual([text for text in problems if 'fused' in text.lower()], [], problems)
        self.assertEqual(facts['fused']['fused'], 24)
        self.assertEqual(facts['fused']['audit_checked'], 24 * 80)
        json.dumps(facts)
        broken = log.replace('checked=80', 'checked=40')
        problems, _ = check.check(sw.smoke_text(sw.STEADY_OK), broken, False, env=env)
        self.assertTrue(any('checked [40] items' in text for text in problems), problems)
        del quad_round

    def test_the_fused_commit_flags_of_every_profile_pass_the_gates_flag_rules(self):
        for name, profile in PROFILES.items():
            env = profile['env']
            if env.get('QWEN_FAST_TP') != '4' or 'fcommit' not in name:
                continue
            with self.subTest(profile=name):
                report = gate.h1b_report(env, clean_log(audit=env.get('QWEN_FAST_FUSED_COMMIT_AUDIT') == '1',
                                                        live=env.get('QWEN_FAST_FUSED_COMMIT_LIVE_BANKS') == '1',
                                                        traces=68 if env.get('QWEN_FAST_FUSED_COMMIT_INPLACE') == '1' else 4))
                self.assertIsNotNone(report)
                self.assertEqual([text for text in report['problems'] if 'does nothing' in text], [])


if __name__ == '__main__':
    unittest.main()
