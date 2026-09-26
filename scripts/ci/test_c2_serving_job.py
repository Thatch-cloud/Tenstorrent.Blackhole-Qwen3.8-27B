"""c2_serving_job: the qwen-c2-serving.yml job file parse, held on CPU (stdlib only)."""

import io
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_platform_replay as replay  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
JOB_FILE = os.path.join(ROOT, '.github', 'c2-serving-job.env')
PROFILES = ['coding', 'exact', 'general']


def read(**values):
    base = dict(C2_IMAGE_TAG='v6-teardown')
    base.update(values)
    return job.read_job(base, PROFILES)


class ParseTests(unittest.TestCase):
    def test_comments_blanks_and_whitespace(self):
        text = '# a comment\n\nC2_ACTIONS = status build\n C2_IMAGE_TAG=v7 \nC2_SMOKE_TESTS=\n'
        self.assertEqual(job.parse_env(text), dict(C2_ACTIONS='status build', C2_IMAGE_TAG='v7', C2_SMOKE_TESTS=''))

    def test_defaults(self):
        outputs = read()
        self.assertEqual((outputs['actions'], outputs['profile'], outputs['gate_plan']), ('status', 'general', 'bringup'))
        self.assertEqual(outputs['gate_lengths'], '', 'empty: the gate runs its ladder, top rung fitted to the '
                                                      'image\'s profile (an explicit list is never changed)')
        self.assertEqual((outputs['gate_max_tokens'], outputs['gate_memory_prompt']), ('4096', ''))
        self.assertEqual((outputs['platform_image'], outputs['replay_profile'], outputs['tests']), ('', '', ''))
        self.assertEqual(outputs['replay_served_model'], '', 'empty: the replay\'s own default, the :tt name')

    def test_the_gate_keys(self):
        outputs = read(C2_ACTIONS='reset gate', C2_PROFILE='exact', C2_GATE_PLAN='bringup, matrix memory',
                       C2_GATE_LENGTHS='60, 2048 120000', C2_GATE_MAX_TOKENS='3000', C2_GATE_MEMORY_PROMPT='123136',
                       C2_REPLAY_PROFILE='general')
        self.assertEqual(outputs['actions'], 'reset gate')
        self.assertEqual(outputs['gate_plan'], 'bringup,matrix,memory')
        self.assertEqual(outputs['gate_lengths'], '60,2048,120000')
        self.assertEqual((outputs['gate_max_tokens'], outputs['gate_memory_prompt']), ('3000', '123136'))
        self.assertEqual(outputs['replay_profile'], 'general')

    def test_the_replay_served_model(self):
        """Reviewer defect 3: CI must be able to replay an image without the alias, which advertises the
        checkpoint, as well as one with it."""
        for model in ('Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-27B:tt', 'org_1/name.v2-x:b2'):
            with self.subTest(model=model):
                self.assertEqual(read(C2_REPLAY_SERVED_MODEL=model)['replay_served_model'], model)

    def test_what_is_refused(self):
        for values in (dict(C2_ACTIONS='status deploy'), dict(C2_IMAGE_TAG='V7!'), dict(C2_IMAGE_TAG=''),
                       dict(C2_PROFILE='c2'), dict(C2_GATE_PLAN='bringup soak'), dict(C2_GATE_LENGTHS='60,x'),
                       dict(C2_GATE_LENGTHS='60,0'), dict(C2_GATE_MAX_TOKENS='-1'), dict(C2_GATE_MEMORY_PROMPT='big'),
                       dict(C2_PLATFORM_IMAGE='x'), dict(C2_REPLAY_PROFILE='nope'),
                       dict(C2_REPLAY_SERVED_MODEL='Qwen3.8-27B'), dict(C2_REPLAY_SERVED_MODEL='Qwen/Qwen3.8-27B:'),
                       dict(C2_REPLAY_SERVED_MODEL=':tt'), dict(C2_REPLAY_SERVED_MODEL='Qwen/Qwen3.8-27B:tt:x'),
                       dict(C2_REPLAY_SERVED_MODEL='Qwen/a/b'), dict(C2_REPLAY_SERVED_MODEL='Qwen/Qwen3.8 27B'),
                       dict(C2_REPLAY_SERVED_MODEL='Qwen/$(id)'), dict(C2_REPLAY_SERVED_MODEL='Qwen/x;true')):
            with self.subTest(values=values), self.assertRaises(job.JobError):
                read(**values)

    def test_a_profile_is_any_the_profiles_file_defines(self):
        """The c2 profile lands in qwen_c2_profiles.json on another track: the parse reads the names
        from the file, so the job may name it the moment it exists and not before."""
        outputs = job.read_job(dict(C2_IMAGE_TAG='v7-c2', C2_PROFILE='c2'), PROFILES + ['c2'])
        self.assertEqual(outputs['profile'], 'c2')
        self.assertIn('exact', job.profile_names())
        self.assertIn('general', job.profile_names())

    def test_render_is_sorted_and_refuses_newlines(self):
        self.assertEqual(job.render(dict(b='2', a='1')), 'a=1\nb=2\n')
        with self.assertRaises(job.JobError):
            job.render(dict(a='x\ny'))


class FileTests(unittest.TestCase):
    def test_the_tracked_job_file_parses(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(job.main([JOB_FILE]), 0)
        lines = dict(line.split('=', 1) for line in out.getvalue().splitlines())
        self.assertIn('actions', lines)
        self.assertIn('replay_served_model', lines)

    def test_main_refuses_with_the_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'job.env')
            with open(path, 'w') as handle:
                handle.write('C2_ACTIONS=deploy\nC2_IMAGE_TAG=v7\n')
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                self.assertEqual(job.main([path]), 1)
            self.assertEqual(out.getvalue(), '')
            self.assertIn('unknown deploy', err.getvalue())
        with redirect_stderr(io.StringIO()):
            self.assertEqual(job.main([]), 2)

    def test_the_workflow_reads_only_outputs_the_parse_writes_and_runs_the_parse(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        used = set(re.findall(r'steps\.job\.outputs\.([a-z_]+)', text))
        written = set(read())
        self.assertTrue(used, 'the workflow reads no job outputs - the regex has drifted')
        self.assertEqual(sorted(used - written), [])
        self.assertIn('python3 scripts/ci/c2_serving_job.py .github/c2-serving-job.env', text)
        for action in job.ACTIONS:
            if action != 'status':
                self.assertIn("contains(steps.job.outputs.actions, '%s')" % action, text, action)

    def test_the_replay_step_passes_the_served_model_only_when_set(self):
        """Reviewer defect 3: empty leaves c2_platform_replay.py's default (the :tt name); set, it is
        --served-model, e.g. Qwen/Qwen3.8-27B for an image without the alias."""
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        step = text[text.index('- name: Replay'):text.index('- name: Push')]
        self.assertIn('REPLAY_SERVED_MODEL: ${{ steps.job.outputs.replay_served_model }}', step)
        invocation = step[step.index('python3 scripts/ci/c2_platform_replay.py'):]
        self.assertIn('${REPLAY_SERVED_MODEL:+--served-model "$REPLAY_SERVED_MODEL"}', invocation)
        self.assertEqual(invocation.count('--served-model'), 1, 'never unconditionally')
        with open(replay.__file__, encoding='utf-8') as handle:
            parser_text = handle.read()
        self.assertIn("parser.add_argument('--served-model', default=SERVED_MODEL", parser_text)
        self.assertEqual(replay.SERVED_MODEL, 'Qwen/Qwen3.8-27B:tt')
        self.assertTrue(job.MODEL_ID.fullmatch(replay.SERVED_MODEL) and job.MODEL_ID.fullmatch(replay.CHECKPOINT))

    def test_the_workflow_steps_run_in_the_parses_order(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        positions = [text.index("contains(steps.job.outputs.actions, '%s')" % action) for action in job.ACTIONS]
        self.assertEqual(positions, sorted(positions))


if __name__ == '__main__':
    unittest.main()
