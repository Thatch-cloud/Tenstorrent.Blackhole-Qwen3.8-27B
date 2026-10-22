"""The seeded sampled arm of the test clients (docs/tp4-sampled-verify.md section 5), pure CPU, stdlib plus the checkers.

  - sampled_client: greedy (the default) is every client as it always was; model-default adds the model's own temperature / top_p / top_k and a FIXED seed that is a function of the base seed
    and the PROMPT alone, so a re-sent prompt re-draws the same text, a longer budget begins with the shorter one's, and two profiles get the same seed for the same request;
  - the serving smoke (c2_serving_smoke.py) sends the arm on every completion request that does not name its own terms and reports which arm ran in SMOKE_JSON's `sampling` entry;
  - the smoke check judges a sampled profile's run: the attach line, the engaged lines, the keyed first-token and round lines, the audit line;
  - the tau lab sends the arm and serves a gate-only sampled profile only for it; the job reader validates C2_SMOKE_SAMPLING / C2_TAULAB_SAMPLING against the profile.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check as check  # noqa: E402
import c2_tau_lab as lab  # noqa: E402
import make_sampled_profiles  # noqa: E402
import sampled_accept as sa  # noqa: E402
import sampled_client as client  # noqa: E402
import test_c2_serving_smoke_262k as smoke_support  # noqa: E402

PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
NAMES = sorted(PROFILES['profiles'])
SAMPLED, AUDITED = make_sampled_profiles.TIMED, make_sampled_profiles.AUDIT
SHIPPED = make_sampled_profiles.SHIP


class ArmTests(unittest.TestCase):
    def test_the_modes_are_strict_and_greedy_is_the_default(self):
        self.assertEqual(client.mode_of(None), 'greedy')
        self.assertEqual(client.mode_of(''), 'greedy')
        self.assertEqual(client.mode_of('model-default'), 'model-default')
        for bad in ('sampled', 'Greedy', 'model_default', ' greedy', '1'):
            with self.assertRaises(client.SamplingError):
                client.mode_of(bad)

    def test_the_base_seed_is_a_decimal_below_2_to_the_63(self):
        self.assertEqual(client.base_seed_of(None), client.DEFAULT_BASE_SEED)
        self.assertEqual(client.base_seed_of('0'), 0)
        self.assertEqual(client.base_seed_of(str(2 ** 63 - 1)), 2 ** 63 - 1)
        for bad in ('-1', str(2 ** 63), 'x', '1.5', '0x10'):
            with self.assertRaises(client.SamplingError):
                client.base_seed_of(bad)

    def test_the_models_defaults_are_the_samplers_and_the_contracts(self):
        import serving_c2_contract as contract

        self.assertEqual(client.MODEL_DEFAULTS, sa.MODEL_DEFAULTS)
        self.assertEqual(client.MODEL_DEFAULTS, contract.SAMPLED_MODEL_DEFAULTS)

    def test_greedy_adds_nothing_and_returns_the_body_itself(self):
        arm = client.Arm('greedy')
        body = dict(model='m', messages=[{'role': 'user', 'content': 'hi'}], max_tokens=5)
        self.assertIs(arm.apply(body, '/v1/chat/completions'), body)
        self.assertEqual(arm.terms(body), {})
        self.assertEqual(arm.report(), dict(mode='greedy', base_seed=client.DEFAULT_BASE_SEED, requests=0))

    def test_model_default_adds_the_models_terms_and_a_seed_and_keeps_the_clients_own(self):
        arm = client.Arm('model-default', 7)
        body = dict(model='m', messages=[{'role': 'user', 'content': 'hi'}], max_tokens=5)
        sent = arm.apply(body, '/v1/chat/completions')
        self.assertEqual({key: sent[key] for key in ('temperature', 'top_p', 'top_k')}, dict(temperature=1.0, top_p=0.95, top_k=20))
        self.assertIsInstance(sent['seed'], int)
        self.assertTrue(0 <= sent['seed'] < 2 ** 63)
        self.assertNotIn('temperature', body, 'the caller\'s body is not mutated')
        own = arm.apply(dict(body, temperature=0.0, seed=5), '/v1/chat/completions')
        self.assertEqual((own['temperature'], own['seed'], own['top_k']), (0.0, 5, 20))
        self.assertIs(arm.apply(body, '/tokenize'), body, 'only completions carry the arm')
        self.assertIs(arm.apply(body, '/v1/models'), body)
        self.assertEqual(arm.report()['requests'], 2)
        self.assertEqual(arm.report()['terms'], dict(temperature=1.0, top_p=0.95, top_k=20))

    def test_the_seed_is_a_function_of_the_base_and_the_prompt_alone(self):
        arm = client.Arm('model-default', 11)
        chat = dict(messages=[{'role': 'user', 'content': 'same prompt'}], max_tokens=10, stream=True)
        again = dict(model='other', messages=[{'role': 'user', 'content': 'same prompt'}], max_tokens=4000, stream=False, n=1)
        other = dict(messages=[{'role': 'user', 'content': 'another prompt'}], max_tokens=10)
        completions = dict(prompt=[1, 2, 3], max_tokens=8)
        seeds = [arm.terms(body)['seed'] for body in (chat, again, other, completions)]
        self.assertEqual(seeds[0], seeds[1], 'the budget, the stream flag and the model name do not move the seed')
        self.assertEqual(len(set(seeds)), 3)
        self.assertNotEqual(client.Arm('model-default', 12).terms(chat)['seed'], seeds[0], 'the base seed does')
        self.assertEqual(client.Arm('model-default', 11).terms(chat)['seed'], seeds[0], 'and nothing else varies between runs')
        self.assertEqual(client.seed_for(11, completions), client.seed_for(11, dict(prompt=[1, 2, 3], max_tokens=1)))

    def test_the_tools_are_part_of_the_prompt(self):
        arm = client.Arm('model-default')
        plain = dict(messages=[{'role': 'user', 'content': 'x'}])
        tooled = dict(plain, tools=[{'type': 'function', 'function': {'name': 'f'}}])
        self.assertNotEqual(arm.terms(plain)['seed'], arm.terms(tooled)['seed'])

    def test_the_arm_reads_its_environment(self):
        arm = client.Arm.from_environ({'C2_SMOKE_SAMPLING': 'model-default', 'C2_SMOKE_SEED': '99'})
        self.assertEqual((arm.mode, arm.base_seed), ('model-default', 99))
        self.assertEqual(client.Arm.from_environ({}).mode, 'greedy')
        with self.assertRaises(client.SamplingError):
            client.Arm.from_environ({'C2_SMOKE_SAMPLING': 'yes'})


class SmokeRunTests(smoke_support.SmokeRuns):
    """The real smoke script against the fake server (test_c2_serving_smoke_262k's), with and without the arm."""

    def smoke(self, **environ):
        with smoke_support.TokenFake() as fake:
            env = dict(os.environ, SMOKE_CODE_ROOT=self.code_root, **environ)
            done = subprocess.run([sys.executable, '-B', str(smoke_support.SMOKE), fake.base, smoke_support.MODEL, 'warmup,coding'], capture_output=True, text=True,
                                  timeout=300, env=env, cwd=str(HERE))
            requests = list(fake.requests)
        return done, requests

    def results(self, done):
        line = [text for text in done.stdout.splitlines() if text.startswith('SMOKE_JSON ')][-1]
        return json.loads(line[len('SMOKE_JSON '):])

    def test_the_default_is_greedy_and_the_requests_are_what_they_were(self):
        done, requests = self.smoke()
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertEqual(self.results(done)['sampling'], dict(mode='greedy', base_seed=client.DEFAULT_BASE_SEED, requests=0))
        self.assertGreaterEqual(len(requests), 2)
        for request in requests:
            for name in ('temperature', 'top_p', 'top_k', 'seed'):
                self.assertNotIn(name, request['body'])

    def test_model_default_sends_the_terms_and_a_fixed_seed_and_says_so_in_smoke_json(self):
        done, requests = self.smoke(C2_SMOKE_SAMPLING='model-default', C2_SMOKE_SEED='4242')
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        report = self.results(done)['sampling']
        self.assertEqual((report['mode'], report['base_seed']), ('model-default', 4242))
        self.assertEqual(report['requests'], len(requests))
        self.assertEqual(report['terms'], dict(temperature=1.0, top_p=0.95, top_k=20))
        for request in requests:
            body = request['body']
            self.assertEqual((body['temperature'], body['top_p'], body['top_k']), (1.0, 0.95, 20))
            self.assertEqual(body['seed'], client.seed_for(4242, body))
        self.assertEqual(len({request['body']['seed'] for request in requests}), len({json.dumps(request['body'].get('messages')) for request in requests}),
                         'one seed a prompt')
        again_done, again = self.smoke(C2_SMOKE_SAMPLING='model-default', C2_SMOKE_SEED='4242')
        self.assertEqual([request['body']['seed'] for request in again], [request['body']['seed'] for request in requests], 'repeatable across runs')
        __, other = self.smoke(C2_SMOKE_SAMPLING='model-default', C2_SMOKE_SEED='4243')
        self.assertNotEqual([request['body']['seed'] for request in other], [request['body']['seed'] for request in requests])

    def test_a_malformed_arm_stops_the_smoke_before_it_sends_anything(self):
        done, requests = self.smoke(C2_SMOKE_SAMPLING='sampled')
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(requests, [])
        self.assertIn('sampling must be one of', done.stderr)

    def test_the_smoke_check_still_reads_the_results_with_the_new_entry(self):
        done, _ = self.smoke(C2_SMOKE_SAMPLING='model-default')
        self.assertEqual(check.smoke_problems(self.results(done)), [])


class SmokeCheckTests(unittest.TestCase):
    ENV = {'QWEN_FAST_SAMPLED': '1'}
    AUDIT_ENV = {'QWEN_FAST_SAMPLED': '1', 'QWEN_FAST_SAMPLED_AUDIT': '1'}
    ARM = {'sampling': {'mode': 'model-default', 'base_seed': 1, 'requests': 9}}
    GOOD = '\n'.join(['[PINDIAG] sampled verify engaged site=packed_verify rows=64 users=4 k=64 launches=4 tensors=8 chunk=32768 audit=0'] * 4
                     + ['[PINDIAG] sampled verify attach blocks=4 k=64 audit=0', '[PINDIAG] sampled first token request=r position=100 token=7',
                        '[PINDIAG] sampled verify round site=packed_verify n=1 entries=2 rows=32 read_ms=1.00 merge_ms=1.00 draw_ms=1.00',
                        '[PINDIAG] sampled verify audit site=packed_verify round=1 rows=32 slice=exact draws=exact tie_rows=0'])

    def test_the_markers_are_the_modules(self):
        import sampled_verify as sv

        self.assertEqual(check.SAMPLED_ATTACH, sv.ATTACH_MARKER)
        self.assertEqual(check.SAMPLED_ENGAGED, sv.ENGAGED_MARKER)
        self.assertEqual(check.SAMPLED_REFUSED, sv.REFUSED_MARKER)
        self.assertEqual(check.SAMPLED_ROUND, sv.ROUND_MARKER)
        self.assertEqual(check.SAMPLED_FIRST_TOKEN, sv.FIRST_TOKEN_MARKER)
        self.assertEqual(check.SAMPLED_AUDIT, sv.AUDIT_MARKER + ' ')
        self.assertTrue(check.MISMATCH.search(sv.AUDIT_MISMATCH), 'a sampled audit mismatch is an audit mismatch to the smoke check')
        self.assertEqual((check.SAMPLED_FLAG, check.SAMPLED_AUDIT_FLAG), (sv.FLAG, sv.AUDIT_FLAG))

    def problems(self, env, smoke, text, decode=True):
        return check.sampled_problems(env, smoke, text, decode)

    def test_a_profile_without_the_switch_is_not_asked(self):
        self.assertEqual(self.problems({}, None, ''), [])
        self.assertEqual(self.problems(None, None, ''), [])
        self.assertEqual(self.problems({'QWEN_FAST_SAMPLED': '0'}, self.ARM, ''), [])

    def test_a_good_run_passes_audited_or_not(self):
        self.assertEqual(self.problems(self.ENV, self.ARM, self.GOOD), [])
        self.assertEqual(self.problems(self.AUDIT_ENV, self.ARM, self.GOOD), [])

    def test_each_missing_line_is_a_problem_of_its_own(self):
        lines = self.GOOD.splitlines()
        for needle, expect in (('attach', 'attach lines'), ('engaged', 'no engaged line'), ('first token', 'first-token'), ('verify round', 'sampled round')):
            text = '\n'.join(line for line in lines if needle not in line)
            problems = self.problems(self.ENV, self.ARM, text)
            self.assertTrue(any(expect in problem for problem in problems), (needle, problems))
        text = '\n'.join(line for line in lines if 'audit site' not in line)
        self.assertEqual(self.problems(self.ENV, self.ARM, text), [])
        self.assertTrue(any('passing audit line' in problem for problem in self.problems(self.AUDIT_ENV, self.ARM, text)))

    def test_a_refusal_or_a_block_count_mismatch_fails(self):
        refused = self.GOOD + '\n[PINDIAG] sampled verify refused: this ttnn build has no topk'
        self.assertTrue(any('refused the sampled readback' in problem for problem in self.problems(self.ENV, self.ARM, refused)))
        three = '\n'.join(self.GOOD.splitlines()[1:])
        self.assertTrue(any('block(s) and 3 engaged' in problem for problem in self.problems(self.ENV, self.ARM, three)))

    def test_a_smoke_that_sent_only_greedy_requests_checked_nothing_the_profile_is_for(self):
        for smoke in (None, {}, {'sampling': {'mode': 'greedy'}}):
            problems = self.problems(self.ENV, smoke, self.GOOD)
            self.assertTrue(any('C2_SMOKE_SAMPLING=model-default' in problem for problem in problems), (smoke, problems))
        self.assertTrue(any('no request carried' in problem for problem in self.problems(self.ENV, {'sampling': {'mode': 'model-default', 'requests': 0}}, self.GOOD)))
        self.assertEqual(self.problems(self.ENV, {}, self.GOOD.split('\n[PINDIAG] sampled first')[0], decode=False), [], 'a ladder-only smoke is not asked what only decode rounds show')

    def test_the_whole_check_runs_the_rule(self):
        text = 'SMOKE_JSON ' + json.dumps(dict(self.ARM, warmup=dict(value=200)))
        problems, _ = check.check(text, '', False, env=dict(self.ENV))
        self.assertTrue(any('attach lines' in problem for problem in problems), problems)
        problems, _ = check.check(text, self.GOOD, False, env=dict(self.ENV))
        self.assertFalse(any('sampled' in problem for problem in problems), problems)
        problems, _ = check.check(text, self.GOOD + '\n[PINDIAG] sampled verify audit mismatch site=packed_verify round=2 row=3 values differ', False, env=dict(self.AUDIT_ENV))
        self.assertTrue(any('audit mismatch in the container log' in problem for problem in problems), problems)
        problems, _ = check.check(text, '', False, env={})
        self.assertFalse(any('sampled' in problem for problem in problems))


class TauLabTests(unittest.TestCase):
    def profiles(self):
        return json.loads(json.dumps(PROFILES))

    def test_the_request_body_is_greedy_unless_the_arm_is_sampled(self):
        job_record = dict(id='x', prompt_ids=[1, 2, 3])
        _, greedy = lab.request_body(job_record, True, 10, 'm')
        self.assertEqual((greedy['temperature'], greedy['top_p']), (0.0, 1.0))
        self.assertNotIn('seed', greedy)
        _, still = lab.request_body(job_record, True, 10, 'm', client.Arm('greedy'))
        self.assertEqual(still, greedy)
        path, sampled = lab.request_body(job_record, True, 10, 'm', client.Arm('model-default', 5))
        self.assertEqual(path, '/v1/completions')
        self.assertEqual((sampled['temperature'], sampled['top_p'], sampled['top_k']), (1.0, 0.95, 20))
        self.assertEqual(sampled['seed'], client.seed_for(5, sampled))
        self.assertEqual({key: value for key, value in sampled.items() if key not in ('temperature', 'top_p', 'top_k', 'seed')},
                         {key: value for key, value in greedy.items() if key not in ('temperature', 'top_p')})
        chat = dict(id='c', messages=[{'role': 'user', 'content': 'hi'}], tools=[{'type': 'function', 'function': {'name': 'f'}}])
        path, body = lab.request_body(chat, True, 10, 'm', client.Arm('model-default', 5))
        self.assertEqual((path, body['tool_choice'], body['seed']), ('/v1/chat/completions', 'auto', client.seed_for(5, body)))

    def test_the_same_turn_gets_the_same_seed_on_any_profile_and_run(self):
        record = dict(id='x', prompt_ids=[9, 8, 7])
        first = lab.request_body(record, True, 10, 'm', client.Arm('model-default', 5))[1]['seed']
        second = lab.request_body(dict(id='y', prompt_ids=[9, 8, 7]), False, 4000, 'm2', client.Arm('model-default', 5))[1]['seed']
        self.assertEqual(first, second)

    def test_a_gate_only_sampled_base_is_served_only_for_the_sampled_arm(self):
        profiles = self.profiles()
        name, document = lab.derive_profile(profiles, SAMPLED, sampling=True)
        self.assertEqual(name, SAMPLED + '+taulab')
        self.assertEqual(document['profiles'][name]['env']['QWEN_FAST_SAMPLED'], '1')
        with self.assertRaisesRegex(lab.LabError, 'serves it only for the sampled arm'):
            lab.derive_profile(profiles, SAMPLED)
        with self.assertRaisesRegex(lab.LabError, 'gate-only'):
            lab.derive_profile(profiles, 'c2-packed-tp4-8-gate')
        with self.assertRaisesRegex(lab.LabError, 'does not turn sampled verification on'):
            lab.derive_profile(profiles, 'c2-packed-tp4', sampling=True)
        lab.derive_profile(profiles, 'c2-packed-tp4')

    def test_the_options_exist_and_default_to_the_lab_as_it_was(self):
        options = lab.build_parser().parse_args(['--data', 'd', '--results', 'r'])
        self.assertEqual((options.sampling, options.sampling_seed), ('greedy', client.DEFAULT_BASE_SEED))
        options = lab.build_parser().parse_args(['--data', 'd', '--results', 'r', '--sampling', 'model-default', '--sampling-seed', '9'])
        self.assertEqual((options.sampling, options.sampling_seed), ('model-default', 9))
        with self.assertRaises(SystemExit):
            lab.build_parser().parse_args(['--data', 'd', '--results', 'r', '--sampling', 'hot'])


class JobReaderTests(unittest.TestCase):
    def read(self, **values):
        base = dict(C2_IMAGE_TAG='tp4-test', C2_CARDS='quad', C2_ACTIONS='smoke', C2_PROFILE=SAMPLED, C2_SMOKE_TESTS='concurrent8_steady')
        base.update(values)
        return job.read_job(base, NAMES)

    def test_the_default_renders_empty(self):
        outputs = self.read()
        self.assertEqual((outputs['smoke_sampling'], outputs['smoke_seed'], outputs['taulab_sampling'], outputs['taulab_sampling_seed']), ('', '', '', ''))

    def test_model_default_passes_for_a_sampled_profile(self):
        outputs = self.read(C2_SMOKE_SAMPLING='model-default', C2_SMOKE_SEED='77')
        self.assertEqual((outputs['smoke_sampling'], outputs['smoke_seed']), ('model-default', '77'))
        self.assertEqual(self.read(C2_PROFILE=AUDITED, C2_SMOKE_SAMPLING='model-default')['smoke_sampling'], 'model-default')
        self.assertEqual(self.read(C2_SMOKE_SAMPLING='greedy')['smoke_sampling'], 'greedy')

    def test_it_is_refused_where_the_server_would_serve_it_greedy(self):
        with self.assertRaisesRegex(job.JobError, 'turns QWEN_FAST_SAMPLED on'):
            self.read(C2_PROFILE='c2-packed-tp4', C2_SMOKE_SAMPLING='model-default')
        with self.assertRaisesRegex(job.JobError, 'C2_ACTIONS has no smoke'):
            self.read(C2_ACTIONS='status', C2_SMOKE_SAMPLING='model-default')
        with self.assertRaisesRegex(job.JobError, 'must be one of'):
            self.read(C2_SMOKE_SAMPLING='sampled')
        with self.assertRaisesRegex(job.JobError, 'decimal integer'):
            self.read(C2_SMOKE_SAMPLING='model-default', C2_SMOKE_SEED='-3')
        with self.assertRaisesRegex(job.JobError, 'set C2_SMOKE_SAMPLING=model-default'):
            self.read(C2_SMOKE_SEED='5')

    def test_the_tau_lab_options_follow_the_same_rules(self):
        outputs = self.read(C2_ACTIONS='taulab', C2_TAULAB_PROFILE=SAMPLED, C2_TAULAB_SAMPLING='model-default', C2_TAULAB_SAMPLING_SEED='5')
        self.assertEqual((outputs['taulab_sampling'], outputs['taulab_sampling_seed']), ('model-default', '5'))
        with self.assertRaisesRegex(job.JobError, 'turns QWEN_FAST_SAMPLED on'):
            self.read(C2_ACTIONS='taulab', C2_TAULAB_SAMPLING='model-default')
        with self.assertRaisesRegex(job.JobError, 'set C2_TAULAB_SAMPLING=model-default'):
            self.read(C2_ACTIONS='taulab', C2_TAULAB_SAMPLING_SEED='5')

    def test_the_workflow_passes_the_arm_to_the_quad_smoke_and_the_lab_and_names_no_switch(self):
        text = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        for needle in ('SMOKE_SAMPLING: ${{ steps.job.outputs.smoke_sampling }}', 'SMOKE_SEED: ${{ steps.job.outputs.smoke_seed }}',
                       'C2_SMOKE_SAMPLING="$SMOKE_SAMPLING" C2_SMOKE_SEED="$SMOKE_SEED"', 'TAULAB_SAMPLING: ${{ steps.job.outputs.taulab_sampling }}',
                       'TAULAB_SAMPLING_SEED: ${{ steps.job.outputs.taulab_sampling_seed }}', '--sampling "$TAULAB_SAMPLING"', '--sampling-seed "$TAULAB_SAMPLING_SEED"'):
            self.assertEqual(text.count(needle), 1, needle)
        self.assertNotIn('QWEN_FAST_SAMPLED', text)

    def test_the_job_doc_names_the_four_options(self):
        for name in ('C2_SMOKE_SAMPLING', 'C2_SMOKE_SEED', 'C2_TAULAB_SAMPLING', 'C2_TAULAB_SAMPLING_SEED'):
            self.assertIn(name, job.__doc__)


if __name__ == '__main__':
    unittest.main()
