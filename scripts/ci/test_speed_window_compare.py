"""speed_window_compare and c2_smoke_check: the speed window's offline exactness reading and its enforced stop conditions."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_smoke_check as check  # noqa: E402
import speed_window_compare as swc  # noqa: E402

TEXTS = ['alpha ' * 20, 'beta ' * 20]
# '<P|S><position>:<prefix>:<emitted>:<rows>:<cap>' per round (acceptance_report.encode_paths)
SOLO_PATHS = ['S100:3:-:-:- S103:5:-:-:- S108:2:-:-:-', 'S50:1:-:-:- S51:4:-:-:-']


def report(texts=TEXTS, paths=SOLO_PATHS):
    streams = [dict(text=text) for text in texts]
    body = dict(streams=streams)
    if paths is not None:
        body['s2'] = dict(paths=dict(users={str(i): dict(encoded=encoded) for i, encoded in enumerate(paths)}))
    return body


def tree(root, solo, concurrent):
    for arm, body in (('matrix-solo', solo), ('matrix-concurrent', concurrent)):
        folder = Path(root) / 'results' / arm
        folder.mkdir(parents=True)
        (folder / 'm3native-gate.json').write_text(json.dumps(body), encoding='utf-8')
    return Path(root)


class CompareTests(unittest.TestCase):
    def run_pair(self, first, second, **options):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            return swc.compare_trees(tree(one, *first), tree(two, *second), **options)

    def test_identical_trees_are_identical(self):
        result, code = self.run_pair((report(), report()), (report(), report()))
        self.assertEqual(code, 0, result)
        self.assertEqual(result['problems'], [])

    def test_a_text_difference_is_a_difference_with_its_first_character(self):
        other = report(texts=[TEXTS[0], TEXTS[1][:10] + 'X' + TEXTS[1][11:]])
        result, code = self.run_pair((report(), report()), (other, report()))
        self.assertEqual(code, 1)
        self.assertIn('matrix-solo user 1: text differs (first at character 10)', result['problems'])

    def test_equal_texts_with_a_different_solo_prefix_sequence_are_a_wrong_slide(self):
        # the greedy verifier emits the target's own argmax whatever the draft holds: same text, different acceptance
        wrong = report(paths=['S100:3:-:-:- S103:4:-:-:- S107:2:-:-:-', SOLO_PATHS[1]])
        result, code = self.run_pair((report(), report()), (wrong, report()))
        self.assertEqual(code, 1)
        self.assertTrue(result['arms']['matrix-solo'][0]['text_equal'])
        self.assertEqual(result['arms']['matrix-solo'][0]['prefix_first_divergence'], 1)
        self.assertIn('accepted-prefix sequence differs (first at round 1)', result['problems'][0])

    def test_a_concurrent_prefix_difference_is_reported_but_only_strict_judges_it(self):
        wrong = report(paths=['P100:3:-:-:- P103:4:-:-:-', SOLO_PATHS[1]])
        base = report(paths=['P100:3:-:-:- P103:5:-:-:-', SOLO_PATHS[1]])
        self.assertEqual(self.run_pair((report(), base), (report(), wrong))[1], 0)
        result, code = self.run_pair((report(), base), (report(), wrong), strict_concurrent=True)
        self.assertEqual(code, 1, result)

    def test_missing_path_records_are_incomparable_unless_texts_only(self):
        bare = report(paths=None)
        self.assertEqual(self.run_pair((bare, report()), (bare, report()))[1], 2)
        self.assertEqual(self.run_pair((bare, report()), (bare, report()), texts_only=True)[1], 0)

    def test_a_missing_arm_is_incomparable(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            first = tree(one, report(), report())
            Path(two, 'empty').mkdir()
            result, code = swc.compare_trees(first, Path(two))
        self.assertEqual(code, 2)
        self.assertTrue(result['incomparable'])


def paths(*rounds):
    """'<P|S><position>:<prefix>:<emitted>:<rows>:<cap>' records from (kind, position, prefix, rows) tuples."""
    return ' '.join('%s%d:%d:-:%s:-' % (kind, position, prefix, '-' if rows is None else rows)
                    for kind, position, prefix, rows in rounds)


# One user's 12 full-width rounds, positions 4096, 4103, ...; the solo arm logs sequential rounds (rows 16), the concurrent
# arm packed ones (no rows field).
SOLO_ROUNDS = [('S', 4096 + 7 * step, 3 + step % 5, 16) for step in range(12)]
PACKED_ROUNDS = [('P', 4096 + 7 * step, 3 + step % 5, None) for step in range(12)]


class PositionKeyedTests(unittest.TestCase):
    def texts_and_paths(self, concurrent_rounds, solo_rounds, users=2):
        streams = ['answer ' * 20] * users
        solo = report(texts=streams, paths=[paths(*solo_rounds)] * users)
        concurrent = report(texts=streams, paths=[paths(*concurrent_rounds)] * users)
        return solo, concurrent

    def within(self, concurrent_rounds, solo_rounds, **options):
        solo, concurrent = self.texts_and_paths(concurrent_rounds, solo_rounds)
        with tempfile.TemporaryDirectory() as root:
            return swc.compare_batched_vs_singles([tree(root, solo, concurrent)], **options)

    def test_a_batched_draft_equal_to_the_singles_is_identical(self):
        result, code = self.within(PACKED_ROUNDS, SOLO_ROUNDS)
        self.assertEqual(code, 0, result)
        (rows,) = result['within'].values()
        self.assertEqual([(row['prefix_equal'], row['positions']['common'], row['positions']['coverage']) for row in rows],
                         [(True, 12, 1.0)] * 2)

    def test_a_different_accepted_prefix_at_a_shared_position_is_a_different_draft(self):
        wrong = list(PACKED_ROUNDS)
        wrong[7] = ('P', wrong[7][1], wrong[7][2] + 1, None)
        result, code = self.within(wrong, SOLO_ROUNDS)
        self.assertEqual(code, 1)
        self.assertIn('differs from the solo one at 1 of 12 shared positions (first at position %d): the batched draft is not '
                      'the single-user draft' % (4096 + 49), result['problems'][0])

    def test_a_scheduling_difference_does_not_move_a_positions_draft(self):
        """The concurrent arm took an extra short sequential round in the middle (rows 8: not a full block, not compared) and
        one round boundary moved: the round-indexed comparison would call every later round different."""
        extra = PACKED_ROUNDS[:5] + [('S', 4131 + 2, 1, 8)] + PACKED_ROUNDS[5:]
        result, code = self.within(extra, SOLO_ROUNDS)
        self.assertEqual(code, 0, result)
        by_round, _ = self.run_trees((self.texts_and_paths(extra, SOLO_ROUNDS)), (self.texts_and_paths(PACKED_ROUNDS, SOLO_ROUNDS)),
                                     strict_concurrent=True)
        self.assertEqual(by_round['problems'][0][:37], 'matrix-concurrent user 0: accepted-pr')

    def run_trees(self, first, second, **options):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            return swc.compare_trees(tree(one, *first), tree(two, *second), **options)

    def test_too_few_shared_positions_is_incomparable_and_the_coverage_is_the_callers_to_set(self):
        disjoint = [('P', position + 3, prefix, None) for _, position, prefix, _ in PACKED_ROUNDS]
        result, code = self.within(disjoint, SOLO_ROUNDS)
        self.assertEqual(code, 2, 'no shared position: nothing was compared')
        half = PACKED_ROUNDS[:6] + [('P', position + 3, prefix, None) for _, position, prefix, _ in PACKED_ROUNDS[6:]]
        self.assertEqual(self.within(half, SOLO_ROUNDS, min_coverage=0.5)[1], 0)
        self.assertEqual(self.within(half, SOLO_ROUNDS, min_coverage=0.6)[1], 2)

    def test_texts_and_missing_records_are_judged_too(self):
        solo, concurrent = self.texts_and_paths(PACKED_ROUNDS, SOLO_ROUNDS)
        concurrent['streams'][1] = dict(text='another ' * 20)
        with tempfile.TemporaryDirectory() as root:
            result, code = swc.compare_batched_vs_singles([tree(root, solo, concurrent)])
        self.assertEqual(code, 1)
        self.assertIn('user 1: the concurrent text differs from the solo text', result['problems'][0])
        bare = report(paths=None)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(swc.compare_batched_vs_singles([tree(root, bare, bare)])[1], 2)
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'empty').mkdir()
            self.assertEqual(swc.compare_batched_vs_singles([Path(root)])[1], 2)

    def test_a_tail_round_is_not_a_full_block_and_is_never_compared(self):
        solo = SOLO_ROUNDS + [('S', 4096 + 7 * 12, 1, 4)]
        packed = PACKED_ROUNDS + [('S', 4096 + 7 * 12, 3, 4)]
        self.assertEqual(self.within(packed, solo)[1], 0)

    def test_the_two_tree_comparison_by_position_judges_the_concurrent_arm_only_when_strict(self):
        base_paths = self.texts_and_paths(PACKED_ROUNDS, SOLO_ROUNDS)
        moved = list(PACKED_ROUNDS[:6]) + [('P', 4144, 6, None)] + list(PACKED_ROUNDS[6:])
        other = self.texts_and_paths(moved, SOLO_ROUNDS)
        self.assertEqual(self.run_trees(base_paths, other, position_keyed=True)[1], 0, 'reported, not judged')
        result, code = self.run_trees(base_paths, other, position_keyed=True, strict_concurrent=True)
        self.assertEqual(code, 0, 'an extra round at a new position is a scheduling difference: the shared positions agree')
        wrong = list(PACKED_ROUNDS)
        wrong[3] = ('P', wrong[3][1], 7, None)
        result, code = self.run_trees(base_paths, self.texts_and_paths(wrong, SOLO_ROUNDS), position_keyed=True,
                                      strict_concurrent=True)
        self.assertEqual(code, 1)
        self.assertIn('accepted prefix differs at 1 of 12 shared positions (first at position 4117)', result['problems'][0])
        self.assertIn('DIFFERS at position 4117', swc.render(result, code))

    def test_users_restricts_the_judgement_to_the_prompts_two_trees_share(self):
        """The real-text corpus fixes a prompt by (user index, length): a tree that served another length to user 0 is compared
        on the users it served alike."""
        first = self.texts_and_paths(PACKED_ROUNDS, SOLO_ROUNDS)
        other_texts = [dict(text='another prompt ' * 20), dict(text='answer ' * 20)]
        second = (dict(first[0], streams=other_texts), dict(first[1], streams=other_texts))
        self.assertEqual(self.run_trees(first, second)[1], 1, 'user 0 served another prompt')
        result, code = self.run_trees(first, second, users={1})
        self.assertEqual(code, 0, result)
        self.assertEqual([row['user'] for row in result['arms']['matrix-solo']], [1])
        with tempfile.TemporaryDirectory() as root:
            solo, concurrent = self.texts_and_paths(PACKED_ROUNDS, SOLO_ROUNDS)
            wrong = list(PACKED_ROUNDS)
            wrong[2] = ('P', wrong[2][1], wrong[2][2] + 1, None)
            mixed = dict(concurrent, s2=dict(paths=dict(users={'0': dict(encoded=paths(*wrong)),
                                                               '1': dict(encoded=paths(*PACKED_ROUNDS))})))
            found = swc.compare_batched_vs_singles([tree(root, solo, mixed)], users={1})
            self.assertEqual(found[1], 0, 'user 0 is not judged')
            self.assertEqual(swc.compare_batched_vs_singles([Path(root)])[1], 1)

    def test_the_command_line(self):
        solo, concurrent = self.texts_and_paths(PACKED_ROUNDS, SOLO_ROUNDS)
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            first = tree(root, solo, concurrent)
            saved = Path(out) / 'result.json'
            self.assertEqual(swc.main([str(first), '--batched-vs-singles', '--json', str(saved)]), 0)
            self.assertIn('within', json.loads(saved.read_text(encoding='utf-8')))
            self.assertEqual(swc.main([str(first), str(first), '--batched-vs-singles', '--position-keyed',
                                       '--strict-concurrent-prefixes']), 0)
            with self.assertRaises(SystemExit):
                swc.main([str(first)])
            with self.assertRaises(SystemExit):
                swc.main([str(first), '--batched-vs-singles', '--min-coverage', '0'])
            self.assertEqual(swc.main([str(first), '--batched-vs-singles', '--users', '1']), 0)
            for bad in ('x', '', '-1', ',,'):
                with self.assertRaises(SystemExit):
                    swc.main([str(first), '--batched-vs-singles', '--users', bad])


SMOKE_OK = dict(warmup=dict(value=200, wall_s=1.0),
                warm_lifecycle=dict(cold=[dict(chars=600, tokens=8, ttft=1.0)], warm=[dict(chars=600, tokens=8, ttft=0.5)]),
                coding=dict(tokens=1500, finish='length', decode_tok_s=30.0, text='def main():\n    import argparse\n' * 5),
                concurrent4=dict(users=[dict(tokens=800, finish='length', text='Here is the answer to the question. ' * 5)] * 4))


def smoke_text(results):
    return 'noise\nSMOKE_JSON ' + json.dumps(results) + '\n'


def publish(*rounds):
    return '\n'.join('(EngineCore pid=66) 2026 | INFO | x - [PACKED-PUBLISH] round=%d stages={features: [0.08,0.04], '
                     'prepare_history: [%s], publish_target: [0.85,0.11]}' % (index, ','.join('%.2f' % v for v in values))
                     for index, values in enumerate(rounds, 1))


class SmokeCheckTests(unittest.TestCase):
    def test_a_clean_smoke_with_single_digit_ramp_commits_passes_on_the_slide_profile(self):
        problems, facts = check.check(smoke_text(SMOKE_OK), publish((900.0, 8.0), (6.0, 9.0), (5.0, 8.5), (5.5, 9.0)), True)
        self.assertEqual(problems, [])
        self.assertEqual(facts['publish_rounds'], 4)

    def test_the_v140_ramp_commits_fail_it_and_the_flag_off_profile_is_not_held_to_it(self):
        log = publish(*[(310.0, 9.0)] * 6)
        problems, _ = check.check(smoke_text(SMOKE_OK), log, True)
        self.assertTrue(any('ramp commit' in text for text in problems), problems)
        self.assertEqual(check.check(smoke_text(SMOKE_OK), log, False)[0], [])

    def test_an_errored_test_an_empty_stream_and_garbage_text_fail(self):
        bad = dict(SMOKE_OK, coding=dict(error='ReadTimeout()'), concurrent4=dict(users=[
            dict(tokens=0, finish=None, text=''), dict(tokens=800, finish='length', text='aaaa' * 60),
            dict(tokens=800, finish='length', text='fine words here ' * 8), dict(tokens=800, finish='length', text='ok ' * 40)]))
        problems, _ = check.check(smoke_text(bad), publish((5.0, 5.0)), True)
        joined = ' | '.join(problems)
        self.assertIn('coding: ReadTimeout()', joined)
        self.assertIn('concurrent4 user 0: no tokens', joined)
        self.assertIn('concurrent4 user 1: garbage text (a repetition of 1 characters)', joined)

    def test_an_audit_mismatch_line_fails_and_a_missing_smoke_json_fails(self):
        log = publish((5.0, 5.0)) + '\n[PINDIAG] verify t1 audit mismatch round=3 rows=[5] shard=[7] sampler=[9]\n'
        problems, facts = check.check(smoke_text(SMOKE_OK), log, True)
        self.assertEqual(facts['audit_mismatches'], 1)
        self.assertTrue(any('audit mismatch' in text for text in problems))
        self.assertEqual(check.check('no json here', publish((5.0, 5.0)), False)[0], ['no SMOKE_JSON line in the smoke log'])

    def test_the_speed_profiles_carry_the_flag_the_check_reads(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen_c2_profiles.json')
        self.assertTrue(check.slide_on('c2-packed-tp4-speed', path))
        self.assertFalse(check.slide_on('c2-packed-tp4-speed-noslide', path))

    def test_the_quad_smoke_step_runs_the_check_after_the_smoke(self):
        workflow = (Path(__file__).resolve().parents[2] / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        quad = workflow[workflow.index("steps.job.outputs.cards == 'quad'\n        working-directory: c2\n        timeout-minutes: 210"):]
        self.assertLess(quad.index('c2_serving_smoke.py'), quad.index('c2_smoke_check.py'))
        self.assertIn('--profile "$PROFILE"', quad)


STEADY_OK = dict(SMOKE_OK, concurrent4_steady=dict(users=[dict(tokens=800, finish='length',
                                                              text='Here is the explanation of the code. ' * 5)] * 4))
SELECT = '(EngineCore pid=66) 2026 | INFO | x - [PACKED-SELECT] round=%d pairs=%s users=4 calls=1 collect_ms=1.0 select_ms=0.1'
QUAD_LOG = '\n'.join(['[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=32/8 rows=64 sdpa=fold conv=110']
                     + [SELECT % (n, '[[0, 1, 2, 3]]') for n in range(1, 6)]
                     + ['[QUAD-DRAFT] round=%d built=%d ms=7.5' % (n, n == 1) for n in range(1, 6)])
PAIR_LOG = '\n'.join(SELECT % (n, '[[0, 1], [2, 3]]') for n in range(1, 6))
QUAD_ENV = dict(QWEN_FAST_QUAD_DRAFT='1')
OFF_ENV = dict(QWEN_FAST_QUAD_DRAFT='0')


class DraftCheckTests(unittest.TestCase):
    def problems(self, log, env, smoke=STEADY_OK):
        return check.check(smoke_text(smoke), publish((5.0, 5.0)) + '\n' + log, False, env=env)[0]

    def test_a_quad_that_engaged_once_and_served_rounds_passes_and_a_pairs_arm_that_paired_passes(self):
        self.assertEqual(self.problems(QUAD_LOG, QUAD_ENV), [])
        self.assertEqual(self.problems(PAIR_LOG, OFF_ENV), [])

    def test_a_quad_arm_that_never_served_or_engaged_twice_or_fell_back_fails(self):
        self.assertTrue(any('no round was served by the quad' in text for text in self.problems(PAIR_LOG, QUAD_ENV)))
        self.assertTrue(any('appears 0 times' in text for text in self.problems(PAIR_LOG, QUAD_ENV)))
        twice = QUAD_LOG + '\n[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=32/8 rows=64 sdpa=fold conv=110'
        self.assertTrue(any('appears 2 times' in text for text in self.problems(twice, QUAD_ENV)))
        fell = QUAD_LOG + '\n[QUAD-DRAFT] fallback round=9 reason=RuntimeError:x'
        self.assertTrue(any('fell back 1 times' in text for text in self.problems(fell, QUAD_ENV)))
        disabled = QUAD_LOG + '\n[PINDIAG] quad draft disabled round=2 failures=2 reason=x'
        self.assertTrue(any('disabled 1 times' in text for text in self.problems(disabled, QUAD_ENV)))

    def test_a_flag_off_arm_that_ran_the_quad_or_never_paired_fails(self):
        self.assertTrue(any('the quad ran' in text for text in self.problems(QUAD_LOG, OFF_ENV)))
        self.assertTrue(any('never batched a draft' in text for text in self.problems('', OFF_ENV)))

    def test_only_a_smoke_that_ran_the_steady_mix_is_held_to_it(self):
        self.assertEqual(self.problems('', QUAD_ENV, SMOKE_OK), [])
        self.assertEqual(self.problems('', OFF_ENV, SMOKE_OK), [])
        bad = dict(STEADY_OK, concurrent4_steady=dict(error='ReadTimeout()'))
        self.assertTrue(any('concurrent4_steady: ReadTimeout()' in text for text in self.problems(QUAD_LOG, QUAD_ENV, bad)))
        empty = dict(STEADY_OK, concurrent4_steady=dict(users=[dict(tokens=0, finish=None, text='')] * 4))
        self.assertTrue(any('concurrent4_steady user 0: no tokens' in text for text in self.problems(QUAD_LOG, QUAD_ENV, empty)))

    def test_an_unequal_audit_line_fails_whatever_ran(self):
        for line, fragment in (('[QUAD-AUDIT] round=4 equal=0 stage=values:chunk0:chip1:pair0 users=4 checks=9', 'QUAD-AUDIT'),
                               ('[DRAFT-SINGLES-AUDIT] round=4 group=[0, 1, 2, 3] equal=0 stage=raw:values:chunk0:chip1:u2 '
                                'checks=9', 'DRAFT-SINGLES-AUDIT')):
            with self.subTest(line=line):
                found = self.problems(QUAD_LOG + '\n' + line, QUAD_ENV, SMOKE_OK)
                self.assertEqual(len(found), 1)
                self.assertIn(fragment, found[0])
        equal = '[DRAFT-SINGLES-AUDIT] round=4 group=[0, 1, 2, 3] equal=1 stage=all checks=96'
        self.assertEqual(self.problems(QUAD_LOG + '\n' + equal, QUAD_ENV), [])
        audited = dict(QUAD_ENV, QWEN_FAST_DRAFT_SINGLES_AUDIT='all')
        self.assertTrue(any('no [DRAFT-SINGLES-AUDIT] line was logged' in text for text in self.problems(QUAD_LOG, audited)))
        self.assertEqual(self.problems(QUAD_LOG + '\n' + equal, audited), [])

    def test_the_facts_count_what_the_log_says(self):
        facts = check.check(smoke_text(STEADY_OK), publish((5.0, 5.0)) + '\n' + QUAD_LOG, False, env=QUAD_ENV)[1]['draft']
        self.assertEqual((facts['quad_rounds'], facts['pair_rounds'], facts['quad_markers'], facts['quad_lines']), (5, 0, 1, 5))

    def test_the_marker_and_line_formats_are_the_modules(self):
        import draft_singles_audit
        import quad_draft
        import dflash_packed_proposal_coordinator as coordinator

        self.assertEqual(check.QUAD_MARKER, quad_draft.MARKER)
        self.assertEqual(check.QUAD_DISABLED, quad_draft.DISABLED_MARKER)
        self.assertTrue(check.QUAD_FALLBACK in quad_draft.FALLBACK_LINE)
        self.assertTrue(check.QUAD_LINE.search(quad_draft.ROUND_LINE.format(round=3, built=1, ms='1.0')))
        self.assertTrue(check.QUAD_AUDIT.search(quad_draft.AUDIT_LINE % (3, 1, 'all', 4, 70)))
        line = draft_singles_audit.AUDIT_LINE % (3, [0, 1, 2, 3], 1, 'all', 96)
        self.assertTrue(check.SINGLES_AUDIT_LINE.search(line), line)
        select = coordinator.SELECT_LINE.format(round=3, pairs=[[0, 1, 2, 3]], users=4, calls=1, collect_ms='1', select_ms='2')
        self.assertTrue(check.QUAD_ROUND.search(select) and not check.PAIR_ROUND.search(select))
        select = coordinator.SELECT_LINE.format(round=3, pairs=[[0, 1], [2, 3]], users=4, calls=1, collect_ms='1', select_ms='2')
        self.assertTrue(check.PAIR_ROUND.search(select) and not check.QUAD_ROUND.search(select))

    def test_the_steady_smoke_test_exists_is_opt_in_and_runs_four_prompts_past_the_ramp(self):
        smoke = (Path(__file__).resolve().parent / 'c2_serving_smoke.py').read_text(encoding='utf-8')
        self.assertIn("if ONLY and 'concurrent4_steady' in ONLY:\n    record('concurrent4_steady', concurrent4_steady)", smoke)
        self.assertIn('span = 14000', smoke)
        compile(smoke, 'c2_serving_smoke.py', 'exec')   # the rig host's python 3.7 takes the same syntax


SMOKE_SOURCE = (Path(__file__).resolve().parent / 'c2_serving_smoke.py').read_text(encoding='utf-8')


def smoke_functions(*names):
    """The named top-level functions of c2_serving_smoke.py, executed alone (the script itself talks to a server on import)."""
    class Request:
        def __init__(self, url, data=None, method=None, headers=None):
            self.data = data

    class Response:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter(self.lines)

    namespace = dict(json=json, time=__import__('time'), hashlib=__import__('hashlib'), BASE='http://x', MODEL='m',
                     sys=sys, os=os, urllib=type('urllib', (), dict(request=type('request', (), dict(
                         Request=Request, urlopen=lambda request, timeout=None: namespace['SSE'])))))
    namespace['Response'] = Response
    for name in names:
        start = SMOKE_SOURCE.index('def %s(' % name)
        end = SMOKE_SOURCE.index('\n\n\n', start)
        exec(SMOKE_SOURCE[start:end], namespace)
    return namespace


def sse(deltas, completion_tokens):
    """A streamed body: one chunk per (field, text) delta, then the usage chunk and [DONE]."""
    lines = [b'data: ' + json.dumps(dict(choices=[dict(delta={field: text}, finish_reason=None)])).encode() + b'\n'
             for field, text in deltas]
    lines.append(b'data: ' + json.dumps(dict(choices=[dict(delta={}, finish_reason='length')])).encode() + b'\n')
    lines.append(b'data: ' + json.dumps(dict(choices=[], usage=dict(completion_tokens=completion_tokens, prompt_tokens=9))).encode() + b'\n')
    lines.append(b'data: [DONE]\n')
    return lines


class SmokeStreamTests(unittest.TestCase):
    def read(self, deltas, tokens=20):
        namespace = smoke_functions('stream')
        namespace['SSE'] = namespace['Response'](sse(deltas, tokens))
        return namespace['stream']([dict(role='user', content='hi')], 20)

    def test_the_text_stays_300_characters_and_the_full_answer_is_two_hashes_and_two_lengths(self):
        thinking = [('reasoning_content', 'think %d ' % n) for n in range(30)]
        answer = [('content', 'answer word %d. ' % n) for n in range(30)]
        got = self.read(thinking + answer)
        self.assertEqual(len(got['text']), 300)
        self.assertEqual(got['completion_tokens'], 20)
        self.assertEqual(got['tokens'], 20)
        self.assertEqual(got['content_chars'], sum(len(text) for _, text in answer))
        self.assertEqual(got['reasoning_chars'], sum(len(text) for _, text in thinking))
        self.assertNotEqual(got['content_sha256'], got['reasoning_sha256'])
        self.assertNotIn('delta_stamps', got)

    def test_the_same_tokens_split_differently_between_the_fields_give_the_same_two_hashes(self):
        # </think> falling in one 16-row delta or across two 4-row ones: the fields' contents are the same text either way
        whole = [('reasoning_content', 'let me think'), ('reasoning_content', ' about it'), ('content', 'The answer'),
                 ('content', ' is 4.')]
        split = [('reasoning_content', 'let me '), ('reasoning_content', 'think about'), ('reasoning_content', ' it'),
                 ('content', 'The '), ('content', 'answer is'), ('content', ' 4.')]
        one, other = self.read(whole), self.read(split)
        self.assertEqual((one['content_sha256'], one['reasoning_sha256']), (other['content_sha256'], other['reasoning_sha256']))
        self.assertEqual(one['text'], other['text'])
        different = self.read(whole[:3] + [('content', ' is 5.')])
        self.assertNotEqual(one['content_sha256'], different['content_sha256'])
        self.assertEqual(one['reasoning_sha256'], different['reasoning_sha256'])

    def test_the_delta_stamps_are_kept_only_when_asked(self):
        namespace = smoke_functions('stream')
        namespace['SSE'] = namespace['Response'](sse([('content', 'a'), ('content', 'b')], 2))
        got = namespace['stream']([dict(role='user', content='hi')], 2, keep_stamps=True)
        self.assertEqual(len(got['delta_stamps']), 2)


class SmokeLiveReportTests(unittest.TestCase):
    def user(self, start, first, end, tokens, stamps):
        return dict(started_at=start, first_at=first, ended_at=end, tokens=tokens, decode_tok_s=10.0, ttft=first - start,
                    delta_stamps=stamps)

    def test_the_rate_is_over_the_window_all_four_are_live_in(self):
        report = smoke_functions('live_report')['live_report']
        # user 3's first token is at 10 s and user 0 ends at 20 s: the window is 10 s; every user emits one delta per second
        users = [self.user(0, 1 + n * 3, 20 + n, 20, [float(t) for t in range(1 + n * 3, 21 + n)]) for n in range(4)]
        for index, user in enumerate(users):
            user['tokens'] = len(user['delta_stamps'])
        aggregate = report(users)
        self.assertEqual(aggregate['live4_window_s'], 10.0)
        self.assertEqual(aggregate['last_first_token_s'], 10.0)
        self.assertEqual([user['live4_tok_s'] for user in users], [1.0] * 4)
        self.assertEqual(aggregate['live4_agg_tok_s'], 4.0)
        self.assertTrue(all('delta_stamps' not in user for user in users))

    def test_tokens_are_scaled_from_deltas_and_a_user_that_never_streamed_leaves_no_window(self):
        report = smoke_functions('live_report')['live_report']
        users = [self.user(0, 1, 11, 20, [1 + t * 0.5 for t in range(20)])] * 3 + [dict(error='ReadTimeout()')]
        users = [dict(user) for user in users]
        aggregate = report(users)
        self.assertIsNone(aggregate['live4_window_s'])
        self.assertIsNone(aggregate['live4_agg_tok_s'])
        self.assertNotIn('delta_stamps', users[0])


class SmokeNewTestsTests(unittest.TestCase):
    def test_the_new_tests_sit_where_the_plan_puts_them(self):
        order = [SMOKE_SOURCE.index(text) for text in (
            "record('concurrent4', concurrent)", "record('concurrent4_v164order', concurrent4_v164order)",
            "record('alive_after_drop', alive)", "record('concurrent4_solo', concurrent4_solo)",
            "record('replay_concurrent4', replay_concurrent4)")]
        self.assertEqual(order, sorted(order))
        for name in ('concurrent4_code', 'concurrent4_code_equal'):
            self.assertGreater(SMOKE_SOURCE.index("record('%s', %s)" % (name, name)), order[-1],
                               name + ' runs after the like-for-like part, replay_concurrent4 included')
        between = SMOKE_SOURCE[order[0]:order[1]]
        self.assertNotIn("record('", between[len("record('concurrent4', concurrent)"):])
        for name in ('concurrent4_v164order', 'concurrent4_solo', 'replay_concurrent4', 'concurrent4_code', 'concurrent4_code_equal'):
            self.assertIn("if ONLY and '%s' in ONLY:" % name, SMOKE_SOURCE)
        self.assertIn("record('concurrent4', concurrent)", SMOKE_SOURCE)
        self.assertIn("return run_users('concurrent4', concurrent_prompts())", SMOKE_SOURCE)

    def test_the_v164_order_is_the_admission_order_of_the_prompts_by_length(self):
        # v164 admitted 8,376, 7,423, 452 and 533 tokens: the coding, explain, LRU and Rust prompts (indices 0, 2, 1, 3)
        self.assertIn("order=(0, 2, 1, 3), stagger=0.25, timeout=300", SMOKE_SOURCE)
        prompts = SMOKE_SOURCE[SMOKE_SOURCE.index('def concurrent_prompts'):SMOKE_SOURCE.index('def concurrent():')]
        self.assertLess(prompts.index('coding'), prompts.index('LRU'))
        self.assertLess(prompts.index('LRU'), prompts.index('Explain this code'))
        self.assertLess(prompts.index('Explain this code'), prompts.index('Rust'))

    def test_replay_concurrent4_is_the_platform_replays_traffic_shape(self):
        replay = (Path(__file__).resolve().parent / 'c2_platform_replay.py').read_text(encoding='utf-8')
        # The smoke's parser list is c2_platform_replay's four, then four more for the eight-user replay (replay_concurrent8).
        self.assertIn("'Write a unit test for a %s parser in Python.'", replay)
        self.assertIn("'Write a unit test for a %s parser in Python.'", SMOKE_SOURCE)
        self.assertIn("('CSV', 'JSON', 'INI', 'TOML')[index]", replay)
        self.assertIn("REPLAY_PARSERS = ('CSV', 'JSON', 'INI', 'TOML',", SMOKE_SOURCE)
        self.assertIn("% REPLAY_PARSERS[index]", SMOKE_SOURCE)
        self.assertIn("in Python.'\n                                       % ('CSV', 'JSON', 'INI', 'TOML')[index], 300)", replay)
        self.assertIn('max_tokens=300', SMOKE_SOURCE)
        self.assertNotIn('stream=True', SMOKE_SOURCE[SMOKE_SOURCE.index('def replay_concurrent4'):])

    def test_the_smoke_still_compiles_for_the_rig_hosts_python(self):
        compile(SMOKE_SOURCE, 'c2_serving_smoke.py', 'exec')


class SmokeCodePromptsTests(unittest.TestCase):
    def prompts(self, targets, root):
        namespace = smoke_functions('code_corpus', 'code_prompts')
        namespace.update(CODE_CHARS_PER_TOKEN=3.6, __file__=str(Path(__file__).resolve().parent / 'c2_serving_smoke.py'),
                         sysconfig=__import__('sysconfig'))
        os.environ['SMOKE_CODE_ROOT'] = str(root)
        try:
            return namespace['code_prompts'](targets)
        finally:
            del os.environ['SMOKE_CODE_ROOT']

    def tree(self, root):
        package = Path(root) / 'pkg'
        package.mkdir()
        for number in range(40):
            (package / ('module%02d.py' % number)).write_text(
                ''.join('def function_%d_%d(value):\n    return value * %d + %d\n\n' % (number, line, line, number)
                        for line in range(400)), encoding='utf-8')
        return package

    def test_four_users_read_disjoint_code_windows_of_about_their_target_length(self):
        with tempfile.TemporaryDirectory() as root:
            prompts, info = self.prompts((4096, 8192, 16384, 24576), self.tree(root))
        self.assertEqual(len(prompts), 4)
        self.assertEqual(info['files'], 40)
        lengths = [len(prompt) for prompt in prompts]
        self.assertEqual(lengths, sorted(lengths))
        for target, length in zip((4096, 8192, 16384, 24576), lengths):
            self.assertGreater(length, target * 3.6)
            self.assertLess(length, target * 3.6 + 1500)
        windows = [prompt[prompt.index('\n') + 1:prompt.index('\n</repository_context>')] for prompt in prompts]
        self.assertEqual(len(set(windows)), 4, 'each user reads a different part of the code')
        for prompt in prompts:
            self.assertTrue(prompt.endswith('Python code block.') or 'repository context above' in prompt)

    def test_the_equal_shape_is_four_prompts_of_one_length_with_four_different_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            prompts, _ = self.prompts((4096,) * 4, self.tree(root))
        self.assertEqual(len({len(prompt) - len(prompt.split('</repository_context>')[1]) for prompt in prompts}), 1)
        self.assertEqual(len({prompt.split('</repository_context>')[1] for prompt in prompts}), 4)

    def test_both_code_tests_ask_for_800_tokens_and_the_four_prompt_shapes(self):
        self.assertIn("code_prompts((4096, 8192, 16384, 24576))", SMOKE_SOURCE)
        self.assertIn("code_prompts((4096,) * 4)", SMOKE_SOURCE)
        self.assertIn("def run_users(label, prompts, max_tokens=800", SMOKE_SOURCE)


def user_with(**fields):
    return dict(dict(tokens=800, completion_tokens=800, finish='length', prompt_tokens=100, content_sha256='c', reasoning_sha256='r',
                     text='Here is the answer to the question. ' * 5), **fields)


def packed_log(*requests):
    """[PACKED] audit lines: (request id, segment, first position) then a later round of each."""
    lines = []
    for request, segment, position in requests:
        lines += ['(EngineCore pid=66) | INFO | x - [PACKED] request=%s segment=%d position=%d prefix=3 emitted=4 predictions=[1]'
                  % (request, segment, position + step) for step in (0, 4)]
    return '\n'.join(lines)


class SoloComparisonTests(unittest.TestCase):
    PROMPTS = (8376, 452, 7423, 533)

    def results(self, concurrent, solo=None, **more):
        body = dict(SMOKE_OK, concurrent4=dict(users=concurrent), **more)
        if solo is not None:
            body['concurrent4_solo'] = dict(users=solo)
        return body

    def users(self, changes=None):
        changes = changes or {}
        return [user_with(prompt_tokens=prompt, **changes.get(index, {})) for index, prompt in enumerate(self.PROMPTS)]

    LOG = None

    def log(self):
        return packed_log(('cmpl-a', 0, 8376), ('cmpl-b', 1, 452), ('cmpl-c', 2, 7423), ('cmpl-d', 3, 533))

    def test_equal_hashes_tokens_and_finish_are_no_problem(self):
        problems, _ = check.check(smoke_text(self.results(self.users(), self.users())), self.log(), False)
        self.assertEqual(problems, [])

    def test_a_differing_hash_names_the_user_and_its_packed_segment(self):
        for field, label in (('content_sha256', 'content'), ('reasoning_sha256', 'reasoning'), ('completion_tokens', 'tokens'),
                             ('finish', 'finish')):
            concurrent = self.users({3: {field: 'other' if field != 'completion_tokens' else 612}})
            problems, _ = check.check(smoke_text(self.results(concurrent, self.users())), self.log(), False)
            self.assertEqual(len(problems), 1, (field, problems))
            self.assertIn('concurrent4 user 3: diverged from solo (segment 3)', problems[0])
            self.assertIn(label, problems[0])

    def test_a_user_without_a_packed_line_is_a_problem_with_its_segment_unread(self):
        concurrent = self.users({1: dict(content_sha256='other')})
        problems, _ = check.check(smoke_text(self.results(concurrent, self.users())), '', False)
        self.assertEqual(problems, ['concurrent4 user 1: diverged from solo (segment unread): content differ '
                                    '(800 tokens against 800 solo)'])

    def test_the_v164_order_users_are_held_to_the_same_solo_references(self):
        results = self.results(self.users(), self.users(), concurrent4_v164order=dict(users=self.users({0: dict(content_sha256='x')})))
        problems, _ = check.check(smoke_text(results), self.log(), False)
        self.assertEqual(len(problems), 1)
        self.assertIn('concurrent4_v164order user 0: diverged from solo (segment 0)', problems[0])

    def test_without_concurrent4_solo_or_its_hashes_the_comparison_is_skipped_and_the_core_rules_stand(self):
        plain = smoke_text(self.results(self.users({3: dict(content_sha256='other')})))
        self.assertEqual(check.check(plain, self.log(), False)[0], [])
        old = [dict(tokens=800, finish='length', text='fine words here ' * 8)] * 4
        self.assertEqual(check.check(smoke_text(self.results(self.users(), old)), self.log(), False)[0], [])
        self.assertEqual(check.check(smoke_text(SMOKE_OK), '', False)[0], [])
        bad = self.users({2: dict(tokens=0, finish=None, text='')})
        problems, _ = check.check(smoke_text(self.results(bad)), self.log(), False)
        self.assertIn('concurrent4 user 2: no tokens', ' | '.join(problems))

    def test_stream_rules_cover_the_new_concurrent_tests_and_the_solo_run(self):
        empty = dict(tokens=0, finish=None, text='')
        results = dict(SMOKE_OK, concurrent4_v164order=dict(users=[empty] * 4), concurrent4_solo=dict(users=[empty] * 4),
                       concurrent4_code=dict(users=[dict(user_with(), tokens=5, finish='stop')] * 4),
                       concurrent4_code_equal=dict(error='ReadTimeout()'),
                       replay_concurrent4=dict(users=[dict(status=200, tokens=300, finish='length'), dict(status=500, tokens=0),
                                                      dict(error='boom'), dict(status=200, tokens=0, finish='stop')]))
        joined = ' | '.join(check.smoke_problems(results))
        for text in ('concurrent4_v164order user 0: no tokens', 'concurrent4_solo user 3: no tokens',
                     'concurrent4_code user 1: finished with stop after 5 token(s)', 'concurrent4_code_equal: ReadTimeout()',
                     'replay_concurrent4 user 1: status 500', 'replay_concurrent4 user 2: boom', 'replay_concurrent4 user 3: no tokens'):
            self.assertIn(text, joined)
        self.assertNotIn('replay_concurrent4 user 0', joined)

    def test_the_summary_line_matches_the_packed_audits_format(self):
        import serving_packed_step
        line = serving_packed_step.AUDIT_LINE.format(request='cmpl-a', segment=2, position=452, prefix=3, emitted=4, predictions=[1])
        self.assertEqual(check.packed_segments(line), {'cmpl-a': (2, 452)})


if __name__ == '__main__':
    unittest.main()
