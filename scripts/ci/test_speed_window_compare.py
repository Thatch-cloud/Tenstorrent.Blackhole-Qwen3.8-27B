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


if __name__ == '__main__':
    unittest.main()
