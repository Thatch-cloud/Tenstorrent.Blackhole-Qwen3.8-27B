"""c2_serving_gate: the C2 serving image's gates in the node agent's container shape, held on CPU.

Nothing here opens a device or runs docker: the arms' `docker run` argv, the harness options each
plan passes (parsed by the harness's own parse_options), what each plan refuses before any container,
the verdicts, and the whole driver with an injected executor that writes the gate's stdout the way the
container would."""

import ast
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as driver  # noqa: E402
import c2_serving_job  # noqa: E402
import lever_n_m3native_gate as gate  # noqa: E402
import real_text_prompts  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
DOCKERFILE = os.path.join(ROOT, 'docker', 'qwen-c2-serving.Dockerfile')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
with open(PROFILES_PATH, encoding='utf-8') as _handle:
    CHECKOUT_PROFILES = json.load(_handle)
with open(driver.REFERENCE, encoding='utf-8') as _handle:
    V235 = json.load(_handle)


def profiles_with_c2():
    """The checkout's profiles, c2 and c2-gate included (s1/core: 131,328 geometry, 16,384 ceiling; c2
    keeps 8,192 of answer room, so a 123,136-token prompt cap; c2-gate keeps 256, so v235's 131,072)."""
    return json.loads(json.dumps(CHECKOUT_PROFILES))


def narrow_profiles(cap=114944):
    """The checkout's profiles plus c2-narrow: c2 with its prompt cap lowered to `cap`, for the ladder
    fitting a profile below the G4 ladder's top rung."""
    profiles = profiles_with_c2()
    narrow = json.loads(json.dumps(profiles['profiles']['c2']))
    narrow['max_prompt_tokens'] = cap
    profiles['profiles']['c2-narrow'] = narrow
    return profiles


def image_environment():
    """The serving image's own ENV: every ENV instruction of its Dockerfile, continuations joined."""
    with open(DOCKERFILE, encoding='utf-8') as handle:
        text = handle.read().replace(chr(13) + chr(10), chr(10)).replace(chr(92) + chr(10), ' ')
    environ = {}
    for line in text.split(chr(10)):
        if line.startswith('ENV '):
            for token in line[4:].split():
                name, _, value = token.partition('=')
                environ[name] = value
    return environ


def load_profile(name, profiles=None):
    profiles = profiles or CHECKOUT_PROFILES
    return dict(profiles['profiles'][name], name=name)


def served_configuration(name='exact', profiles=None):
    """What the harness records inside the image under `name`: the image's ENV, the agent's -e set, and
    the contract's profile environment, which the image's .pth boot hook applies in EVERY python
    process of the image - the harness's too (so QWEN_FAST_OUTPUT_BUDGET is set, QWEN36_BATCHED_DECODE_MODE
    is gone)."""
    environ = image_environment()
    environ.update(dict(item.split('=', 1) for item in driver.AGENT_ENV))
    environ.update(QWEN_C2_SERVING='1', QWEN_C2_PROFILE=name)
    contract.apply_environment(load_profile(name, profiles), environ)
    return gate.qwen_configuration(environ)


def served_argv(name='exact', profiles=None, platform=None):
    """The contract's launched argv (sys.argv[1:] of its log line) for the platform argv under `name`."""
    profile = load_profile(name, profiles)
    platform = platform or gate.platform_argv(8000)
    return contract.rewrite_argv(['/opt/venv/lib/python3.10/site-packages/vllm/entrypoints/openai/api_server.py']
                                 + platform[3:], profile, profile['snapshots'][0])[1:]


def parse_harness(args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return gate.parse_options(list(args))


def smoke_run_flags():
    """The smoke step's docker run, as (flag, value) pairs."""
    with open(WORKFLOW, encoding='utf-8') as handle:
        text = handle.read()
    smoke = text[text.index('- name: Smoke on cards M+A'):text.index('- name: Run the gate')]
    block = smoke[smoke.index('docker run -d'):smoke.index('--entrypoint python3')]
    tokens = block.replace(chr(92) + chr(10), ' ').split()
    pairs = set()
    for index, token in enumerate(tokens):
        if token in ('--tmpfs', '--shm-size', '--memory', '--cpus', '--cap-add', '-e', '-v'):
            pairs.add((token, tokens[index + 1]))
        if token == '--read-only':
            pairs.add((token, None))
    return pairs


AGENT = os.path.join(HERE, 'references', 'c2-serving', 'agent-container-36104200953.json')


class ShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(AGENT, encoding='utf-8') as handle:
            cls.agent = json.load(handle)['argv']
        cls.mine = driver.agent_shape('img', 'n', 'exact', ['/dev/tenstorrent/3', '/dev/tenstorrent/1'])

    def values(self, arguments, flag):
        return [arguments[i + 1] for i, token in enumerate(arguments) if token == flag]

    def test_the_tmpfs_set_and_resources_are_the_agents_own(self):
        """The recorded agent container (replay v11's copy of thatch-inference-Qwen-Qwen3.8-27B)."""
        self.assertEqual(sorted(self.values(self.mine, '--tmpfs')), sorted(self.values(self.agent, '--tmpfs')))
        self.assertIn('--read-only', self.agent)
        self.assertIn('--read-only', self.mine)
        self.assertEqual(self.values(self.agent, '--shm-size'), [str(4 * 1024 ** 3)])
        self.assertEqual(self.values(self.mine, '--shm-size'), ['4g'])
        self.assertEqual(self.values(self.agent, '--memory'), [str(80 * 1024 ** 3)])
        self.assertEqual(self.values(self.mine, '--memory'), ['80g'])
        self.assertEqual(self.values(self.agent, '--cpus'), self.values(self.mine, '--cpus'))
        self.assertEqual([cap.replace('CAP_', '') for cap in self.values(self.agent, '--cap-add')],
                         self.values(self.mine, '--cap-add'))
        agent_binds = set(self.values(self.agent, '-v'))
        for bind in self.values(self.mine, '-v'):
            self.assertIn(bind, agent_binds)
        self.assertEqual(len(self.values(self.mine, '--device')), len(self.values(self.agent, '--device')))

    def test_the_environment_the_platform_adds_is_the_agents(self):
        agent_env = set(self.values(self.agent, '-e'))
        for variable in driver.AGENT_ENV:
            self.assertIn(variable, agent_env)
        self.assertIn('QWEN_C2_SERVING=1', agent_env)
        self.assertEqual(self.values(self.mine, '-e')[-2:], ['QWEN_C2_SERVING=1', 'QWEN_C2_PROFILE=exact'])

    def test_the_smoke_shape_is_a_subset_of_the_agents(self):
        """The smoke step's container named four tmpfs mounts of the agent's eight; the gate keeps all eight."""
        smoke = smoke_run_flags()
        smoke_paths = {value.split(':')[0] for flag, value in smoke if flag == '--tmpfs'}
        mine_paths = {value.split(':')[0] for value in self.values(self.mine, '--tmpfs')}
        self.assertTrue(smoke_paths < mine_paths)
        for pair in smoke:
            if pair[0] == '-e' and not pair[1].startswith('QWEN_C2_PROFILE='):
                self.assertIn(pair[1], self.values(self.mine, '-e'))

    def test_the_cards_go_in_the_order_given(self):
        arguments = driver.agent_shape('img', 'n', 'exact', ['/dev/tenstorrent/3', '/dev/tenstorrent/1'])
        devices = [arguments[i + 1] for i, token in enumerate(arguments) if token == '--device']
        self.assertEqual(devices, ['/dev/tenstorrent/3', '/dev/tenstorrent/1'])

    def test_the_gate_run_mounts_the_harness_read_only_and_runs_it(self):
        arguments = driver.gate_run('img', 'n', 'exact', ['/dev/a', '/dev/b'], '/ckout', '/res/arm', ['--users', '4'])
        mounts = [arguments[i + 1] for i, token in enumerate(arguments) if token == '--mount']
        for script in driver.BENCH_SCRIPTS:
            self.assertIn('type=bind,src=%s,dst=/bench/%s,readonly' % (
                os.path.join('/ckout', 'scripts', 'ci', script), script), mounts)
        self.assertIn('type=bind,src=/res/arm,dst=/gate-results', mounts)
        self.assertEqual(arguments[-7:], ['--entrypoint', 'python3', 'img', '-B', '/bench/lever_n_m3native_gate.py',
                                          '--users', '4'])
        self.assertNotIn('-p', arguments)

    def test_every_module_the_harness_imports_from_scripts_ci_is_mounted(self):
        """The image's /experiment-scripts/ci holds the bundle's copies (77d6995a); a module the harness
        imports and /bench does not carry would load that stale copy or none (serving-image-bundle-
        provenance). Walked from the harness through every mounted script's own imports."""
        seen, queue = set(), ['lever_n_m3native_gate']
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            with open(os.path.join(HERE, name + '.py'), encoding='utf-8') as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
                for module in names:
                    if os.path.isfile(os.path.join(HERE, module.split('.')[0] + '.py')):
                        queue.append(module.split('.')[0])
        self.assertEqual(sorted(name + '.py' for name in seen), sorted(driver.BENCH_SCRIPTS))


class PlanTests(unittest.TestCase):
    def test_the_bringup_is_v235s_shape_through_the_platform_argv(self):
        (arm, args, timeout), = driver.plan_arms('bringup', 'exact', CHECKOUT_PROFILES)
        options = parse_harness(args)
        self.assertEqual(arm, 'bringup-concurrent')
        self.assertEqual((options.users, options.prompt_tokens, options.max_tokens, options.context),
                         (V235['users'], V235['prompt_tokens'], V235['max_tokens'], V235['context']))
        self.assertEqual((options.prompt_source, options.eos, options.stagger), ('real-text', 'stop', V235['stagger']))
        self.assertEqual((options.server_argv, options.expect_profile, options.snapshot),
                         ('platform', 'exact', driver.SNAPSHOT))
        self.assertIsNone(options.prompt_lengths, 'the uniform build: the prompts v235 built')
        self.assertEqual(options.results, gate.Path('/gate-results'))
        self.assertGreater(timeout, options.readiness_seconds + options.stream_timeout)
        self.assertEqual(driver.asked(args), dict(streams=4, lengths=[131072] * 4, max_tokens=256, budgets=[256] * 4))

    def test_the_snapshot_is_the_profiles_first_under_the_agents_mount(self):
        self.assertEqual(driver.SNAPSHOT, CHECKOUT_PROFILES['profiles']['exact']['snapshots'][0])

    def test_the_default_ladder_is_fitted_to_the_profile_with_a_note(self):
        """A profile admitting less than the ladder's 120,000 would answer it with a 400, a hardware FAIL
        that reads as the fast path's (review finding 3). Lowered, and said so. c2 itself (123,136)
        takes the whole ladder."""
        notes = []
        (_, c_args, _), _ = driver.plan_arms('matrix', 'c2', profiles_with_c2(), notes=notes)
        self.assertEqual((parse_harness(c_args).prompt_lengths, notes), (list(c2_serving_job.LADDER), []))
        arms = driver.plan_arms('matrix', 'c2-narrow', narrow_profiles(), notes=notes)
        (c_arm, c_args, _), (s_arm, s_args, _) = arms
        concurrent, solo = parse_harness(c_args), parse_harness(s_args)
        fitted = list(c2_serving_job.LADDER[:-1]) + [114944]
        self.assertEqual((c_arm, s_arm), ('matrix-concurrent', 'matrix-solo'))
        self.assertEqual((concurrent.prompt_lengths, solo.prompt_lengths), (fitted, fitted))
        self.assertEqual(len(notes), 1)
        self.assertIn('[120000] lowered to 114944', notes[0])
        self.assertEqual((concurrent.users, concurrent.sequential_users), (9, 0))
        self.assertEqual((solo.users, solo.sequential_users), (1, 9))
        self.assertEqual((concurrent.max_tokens, solo.max_tokens, concurrent.context), (4096, 4096, 131328))
        self.assertEqual([a for a in c_args if a not in s_args], ['--stagger', '0.25'])

    def test_what_a_profile_cannot_serve_is_refused_before_any_container(self):
        c2 = profiles_with_c2()
        for plan, profile, profiles, kwargs, words in (
                ('matrix', 'c2', c2, dict(lengths=[60, 124000]), 'exceed 123136'),         # past the edge
                ('matrix', 'c2-gate', c2, dict(lengths=[131073]), 'exceed 131072'),
                ('matrix', 'c2', c2, dict(lengths=[60, 120000], max_tokens=16384),         # the answer room
                 'leaves 11328 tokens'),
                ('matrix', 'c2', c2, dict(lengths=[44, 2048]), 'below 45 tokens'),         # the compact template
                ('matrix', 'exact', c2, dict(lengths=[60, 2048]), 'output ceiling 256'),   # 4096 > 256: cut silently
                ('matrix', 'general', c2, dict(lengths=[60], max_tokens=256), None),       # fits
                ('memory', 'c2', c2, dict(memory_prompt=123137), 'exceed 123136'),
                ('lifecycle', 'exact', c2, {}, 'output ceiling 256'),
                ('lifecycle', 'general', c2, {}, 'output ceiling 256')):
            with self.subTest(plan=plan, profile=profile, kwargs=kwargs):
                if words is None:
                    driver.plan_arms(plan, profile, profiles, **kwargs)
                    continue
                with self.assertRaises(driver.PlanError) as caught:
                    driver.plan_arms(plan, profile, profiles, **kwargs)
                self.assertIn(words, str(caught.exception))

    def test_profile_limits_is_the_contracts_own_edge(self):
        """profile_limits mirrors serving_c2_contract.enforce_request: the largest prompt it admits, the
        next one refused, and a max_tokens past the ceiling cut to it. A contract change that moves the
        limit (the plan's 123,136 cap) fails here until profile_limits follows."""
        class Params(object):
            n, logprobs, prompt_logprobs, structured_outputs, stop, min_tokens = 1, None, None, None, None, 0
            logit_bias = allowed_token_ids = bad_words = stop_token_ids = max_tokens = None

        profiles = profiles_with_c2()
        self.assertTrue({'exact', 'c2', 'c2-gate'} <= set(profiles['profiles']))
        for name, profile in sorted(profiles['profiles'].items()):
            context, ceiling, room = driver.profile_limits(profiles, name)
            # Every limit the contract's boot reads from the profile (request_limits), as it installs them.
            kwargs = dict(max_model_len=context, eos_ids=frozenset(profile['eos_ids']),
                          **contract.request_limits(profile))
            with self.subTest(profile=name):
                params = Params()
                params.max_tokens = ceiling + 1000
                contract.enforce_request(params, prompt_tokens=min(room, context - ceiling), **kwargs)
                self.assertEqual(params.max_tokens, ceiling, 'a budget past the ceiling is cut without a word')
                contract.enforce_request(Params(), prompt_tokens=room, **kwargs)
                with self.assertRaises(contract.ContractError):
                    contract.enforce_request(Params(), prompt_tokens=room + 1, **kwargs)

    def test_the_c2_profiles_limits(self):
        """c2 keeps 8,192 tokens of answer room (a 123,136 cap), c2-gate 256 (v235's 131,072 admitted)."""
        self.assertEqual(driver.profile_limits(profiles_with_c2(), 'c2'), (131328, 16384, 123136))
        self.assertEqual(driver.profile_limits(profiles_with_c2(), 'c2-gate'), (131328, 16384, 131072))
        self.assertEqual(driver.profile_limits(profiles_with_c2(), 'exact'), (131328, 256, 131072))
        self.assertEqual([name for name in sorted(CHECKOUT_PROFILES['profiles'])
                          if driver.any_request_profile(CHECKOUT_PROFILES, name)], ['c2', 'c2-gate'])

    def test_the_memory_arms_are_the_largest_prompts_and_on_c2_any_the_shortest(self):
        """G5: four of the largest admitted prompts at what the contract leaves them (c2: 123,136 + 8,192;
        a larger max_tokens would be clamped and read as a budget cut) and, where C2-any is on, four
        60-token prompts at the whole 16,384 ceiling - every proposal bucket per drafter (R3)."""
        (arm, args, _), (short_arm, short_args, _) = driver.plan_arms('memory', 'c2', profiles_with_c2())
        options, short = parse_harness(args), parse_harness(short_args)
        self.assertEqual((arm, short_arm), ('memory-concurrent', 'memory-short'))
        self.assertEqual((options.prompt_lengths, options.max_tokens), ([123136] * 4, 8192))
        self.assertEqual((short.prompt_lengths, short.max_tokens, short.users), ([60] * 4, 16384, 4))
        (_, args, _), _ = driver.plan_arms('memory', 'c2', profiles_with_c2(), memory_prompt=100000)
        self.assertEqual((parse_harness(args).prompt_lengths, parse_harness(args).max_tokens), ([100000] * 4, 16384))
        (_, args, _), = driver.plan_arms('memory', 'exact', CHECKOUT_PROFILES)
        self.assertEqual((parse_harness(args).prompt_lengths, parse_harness(args).max_tokens), ([131072] * 4, 256))

    def test_a_profile_whose_edge_refuses_v235s_prompts_is_warned_about(self):
        self.assertIsNone(driver.bringup_warning(CHECKOUT_PROFILES, 'exact'))
        self.assertIsNone(driver.bringup_warning(CHECKOUT_PROFILES, 'c2-gate'))
        warning = driver.bringup_warning(profiles_with_c2(), 'c2')
        self.assertIn('admits prompts up to 123136 tokens', warning)
        self.assertIn('131072-token prompts', warning)
        self.assertIsNotNone(driver.bringup_warning(CHECKOUT_PROFILES, 'general'))

    def test_an_unknown_plan_is_refused(self):
        with self.assertRaises(ValueError):
            driver.plan_arms('soak', 'exact', CHECKOUT_PROFILES)

    def test_the_worst_case_counts_every_rerun_and_fits_the_workflows_step_one_plan_at_a_time(self):
        """Review finding 2: matrix plus its re-run was 4 x 7,200 s against a 380-minute step."""
        profiles = profiles_with_c2()
        step = WorkflowTests.budget_literals()['step']
        arms = dict((plan, driver.plan_arms(plan, 'c2', profiles)) for plan in c2_serving_job.GATE_PLANS)
        for plan in c2_serving_job.GATE_PLANS:
            with self.subTest(plan=plan):
                self.assertLessEqual(driver.worst_case_seconds([plan], arms), step)
        self.assertEqual(driver.worst_case_seconds(['matrix'], arms), 4 * (5400 + driver.ARM_OVERHEAD_SECONDS))
        self.assertEqual(driver.worst_case_seconds(['lifecycle'], arms), 6 * (3300 + driver.ARM_OVERHEAD_SECONDS))
        self.assertGreater(driver.worst_case_seconds(['matrix', 'lifecycle'], arms), step)


def served_report(texts=None, gate_passed=True, profile='exact', problems=None, users=4, engines=4, error=None,
                  configuration=None, finish='length', completion=256, max_tokens=256, argv=None, lengths=None,
                  acceptance=None):
    streams = V235['streams']
    texts = texts or [s['text'] for s in streams][:users]
    entries = [dict(text=t, finish_reason=finish, completion_tokens=completion, prompt_tokens=131072) for t in texts]
    if error is not None:
        entries[error] = dict(error='HTTP 400: prompt too long')
    dram = [dict(event='engine', request='cmpl-%d' % i, chips=[dict(chip=0, free_gb=3.0 - i, largest_free_mb=100.0)],
                 line='[PINDIAG] dram after engine cmpl-%d: chip0 ...' % i) for i in range(engines)]
    real_text = dict(users=[dict(user=i, prompt_sha256=streams[i % 4]['prompt_sha256']) for i in range(users)])
    if lengths is not None:
        real_text['prompt_lengths'] = list(lengths)
    report = dict(
        users=users, gate_passed=gate_passed, max_tokens=max_tokens,
        qwen_configuration=configuration or served_configuration('exact'), configuration_scope='qwen-tt',
        streams=entries, real_text=real_text,
        comparisons=[dict(user=i, max_tokens=max_tokens) for i in range(users)],
        platform=dict(served_profile=profile, served_argv=argv if argv is not None else served_argv('exact'),
                      problems=problems or []),
        dram=dict(events=dram, engines=engines, min_free_gb=3.0 - max(engines - 1, 0)),
        real_text_stream_problems=['user %d: stream error' % error] if error is not None else [],
        flag_markers=dict(missing=[] if gate_passed else ['QWEN_FAST_GDN_SEQ_BLOCK: x']))
    if acceptance is not None:
        report['acceptance'] = dict(users=[dict(user=i, full_draft=dict(rounds=200, mean_emitted=mean))
                                           for i, mean in enumerate(acceptance)])
    return report


def matrix_report(texts, **kwargs):
    """A matrix arm's report: long answers that ended at EOS inside the 4096-token budget."""
    kwargs.setdefault('finish', 'stop')
    kwargs.setdefault('completion', 300)
    kwargs.setdefault('max_tokens', 4096)
    return served_report(texts=texts, profile='c2', users=len(texts), **kwargs)


class VerdictTests(unittest.TestCase):
    def test_the_fixture_is_the_images_own_configuration(self):
        """Built from the Dockerfile's ENV, the agent's -e set and the exact profile, as the contract's
        boot hook leaves the harness's environment - not from v235's record (review finding 4)."""
        configuration = served_configuration('exact')
        self.assertEqual(configuration['QWEN_FAST_OUTPUT_BUDGET'], '256', 'the contract exports the profile budget')
        self.assertNotIn('QWEN_FAST_OUTPUT_BUDGET', V235['qwen_configuration'], 'v235 ran without it')
        self.assertNotIn('QWEN36_BATCHED_DECODE_MODE', configuration, 'the contract drops what the agent sets')
        self.assertEqual(configuration['QWEN_C2_PROFILE'], 'exact')
        self.assertIn('TT_METAL_CACHE', configuration)

    def test_bringup_passes_on_identical_texts_the_harness_verdict_and_v235s_argv(self):
        result = driver.bringup_verdict(served_report(), V235, 'exact', driver.asked(
            driver.plan_arms('bringup', 'exact', CHECKOUT_PROFILES)[0][1]))
        self.assertEqual((result['verdict'], result['texts']), ('PASS', 'IDENTICAL'))
        self.assertEqual([u['verdict'] for u in result['users']], ['IDENTICAL'] * 4)
        self.assertEqual(result['arithmetic_diff'], {}, 'QWEN_FAST_OUTPUT_BUDGET=256 is v235\'s default')
        self.assertEqual(result['argv_diff'], {})

    def test_a_real_divergence_under_the_images_configuration_is_diverged_not_not_comparable(self):
        """Review finding 4: v235 carries no QWEN_FAST_OUTPUT_BUDGET and the image carries 256, its default;
        a one-character divergence must fail the bring-up, not send people after a new reference."""
        texts = [s['text'] for s in V235['streams']]
        texts[1] = texts[1][:10] + '?' + texts[1][11:]
        result = driver.bringup_verdict(served_report(texts=texts), V235, 'exact')
        self.assertEqual((result['verdict'], result['texts'], result['users'][1]['verdict']), ('FAIL', 'FAIL', 'DIVERGED'))
        self.assertIn('user 1: DIVERGED at character 10', '\n'.join(result['lines']))
        # A budget that really differs is still arithmetic.
        other = dict(served_configuration('exact'), QWEN_FAST_OUTPUT_BUDGET='16384')
        result = driver.bringup_verdict(served_report(texts=texts, configuration=other), V235, 'exact')
        self.assertEqual(result['users'][1]['verdict'], 'NOT_COMPARABLE')

    def test_bringup_fails_on_a_missing_marker_a_contract_problem_or_a_cut_budget(self):
        markers = driver.bringup_verdict(served_report(gate_passed=False), V235, 'exact')
        self.assertEqual((markers['verdict'], markers['texts']), ('FAIL', 'IDENTICAL'))
        self.assertEqual(markers['missing_markers'], ['QWEN_FAST_GDN_SEQ_BLOCK: x'])
        contract_problem = driver.bringup_verdict(served_report(problems=['served profile general']), V235, 'exact')
        self.assertEqual(contract_problem['verdict'], 'FAIL')
        self.assertEqual(driver.bringup_verdict(None, V235)['verdict'], 'FAIL')
        want = driver.asked(driver.plan_arms('bringup', 'exact', CHECKOUT_PROFILES)[0][1])
        cut = driver.bringup_verdict(served_report(completion=200), V235, 'exact', want)
        self.assertEqual(cut['verdict'], 'FAIL')

    def test_the_launched_argv_must_be_v235s_engine_on_exact_and_is_reported_elsewhere(self):
        """Review finding 9: the bring-up diffs the contract's launched argv against v235's command."""
        wrong = served_argv('exact')
        wrong[wrong.index('--max-num-seqs') + 1] = '2'
        result = driver.bringup_verdict(served_report(argv=wrong), V235, 'exact')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['argv_diff'], {'--max-num-seqs': ['2', '4']})
        self.assertIn('the launched argv is not v235\'s engine', '\n'.join(result['lines']))
        other = driver.bringup_verdict(served_report(argv=wrong, profile='c2'), V235, 'c2')
        self.assertEqual(other['verdict'], 'PASS', 'on another profile the argv difference is reported only')
        self.assertIn('launched argv differs from v235\'s engine in: --max-num-seqs', other['lines'])
        gate_profile = driver.bringup_verdict(served_report(argv=wrong, profile='c2-gate'), V235, 'c2-gate')
        self.assertEqual(gate_profile['verdict'], 'FAIL', 'c2-gate serves exact' + "'" + 's engine with c2' + "'" + 's environment')

    def test_memory_records_and_needs_every_engines_line(self):
        want = driver.asked(driver.plan_arms('memory', 'exact', CHECKOUT_PROFILES)[0][1])
        result = driver.memory_verdict(served_report(), want=want)
        self.assertEqual((result['verdict'], result['engines'], len(result['lines'])), ('PASS', 4, 4))
        self.assertEqual(driver.memory_verdict(served_report(engines=3))['verdict'], 'FAIL')
        self.assertEqual(driver.memory_verdict(served_report(error=2))['verdict'], 'FAIL')

    def test_the_launched_argv_line_is_the_contracts_own(self):
        line = driver.launched_argv_line(served_report(argv=['--max-model-len', '131328']))
        self.assertEqual(line, '[QWEN-C2] profile exact: vLLM argv ["--max-model-len", "131328"]')
        self.assertTrue(gate.QWEN_C2_ARGV.search(line))
        self.assertIsNone(driver.launched_argv_line(dict(platform=dict(problems=['x']))))
        self.assertIsNone(driver.launched_argv_line(None))


class MatrixVerdictTests(unittest.TestCase):
    """Review finding 1 (blocker): the matrix verdict read only the platform's problems."""

    TEXTS = ['answer %d ' % i * 50 for i in range(4)]
    WANT = dict(concurrent=dict(streams=4, lengths=[60, 2048, 4096, 90], max_tokens=4096, budgets=[4096] * 4),
                solo=dict(streams=4, lengths=[60, 2048, 4096, 90], max_tokens=4096, budgets=[4096] * 4))

    def verdict(self, concurrent, solo):
        return driver.matrix_verdict(concurrent, solo, wants=self.WANT)

    def test_identical_complete_arms_pass(self):
        result = self.verdict(matrix_report(self.TEXTS), matrix_report(self.TEXTS))
        self.assertEqual((result['verdict'], result['platform_problems']), ('PASS', []))

    def test_aborted_truncated_or_fatal_arms_fail_even_on_identical_texts(self):
        broken = dict(finish='abort', completion=17)
        concurrent, solo = matrix_report(self.TEXTS, **broken), matrix_report(self.TEXTS, **broken)
        for report in (concurrent, solo):
            report['real_text_stream_problems'] = ['user 0: finish_reason \'abort\', not stop or length',
                                                   'user 1: the server reports usage.prompt_tokens 17 for a 2048-token prompt']
            report['fatal'] = 'RuntimeError: engine died'
        result = self.verdict(concurrent, solo)
        self.assertEqual(result['policy']['verdict'], 'PASS', 'the texts alone agree')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('fatal: RuntimeError: engine died' in p for p in result['platform_problems']))
        self.assertTrue(any('usage.prompt_tokens 17' in p for p in result['platform_problems']))

    def test_a_budget_cut_before_the_engine_fails_both_identical_arms(self):
        """Both arms cut to 256 of the 4096 asked end 'length' identically: that is not a pass."""
        cut = dict(finish='length', completion=256, max_tokens=4096)
        result = self.verdict(matrix_report(self.TEXTS, **cut), matrix_report(self.TEXTS, **cut))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('concurrent: user 0: ended "length" after 256 of its 4096 tokens', result['platform_problems'][0])
        ran = self.verdict(matrix_report(self.TEXTS, max_tokens=256), matrix_report(self.TEXTS))
        self.assertTrue(any('ran max_tokens 256, asked 4096' in p for p in ran['platform_problems']))
        over = self.verdict(matrix_report(self.TEXTS, completion=5000), matrix_report(self.TEXTS))
        self.assertTrue(any('past its 4096 budget' in p for p in over['platform_problems']))

    def test_prompts_built_to_other_lengths_or_missing_streams_fail(self):
        result = self.verdict(matrix_report(self.TEXTS, lengths=[60, 2048, 4096, 91]),
                              matrix_report(self.TEXTS, lengths=[60, 2048, 4096, 91]))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('prompts built at [60, 2048, 4096, 91] tokens, asked [60, 2048, 4096, 90]',
                      result['platform_problems'][0])
        short = self.verdict(matrix_report(self.TEXTS[:3]), matrix_report(self.TEXTS[:3]))
        self.assertTrue(any('3 streams, asked 4' in p for p in short['platform_problems']))

    def test_a_collapsed_acceptance_fails_a_corruption_both_arms_share(self):
        """Review finding 15: concurrent and solo run one engine per length, so a length-specific
        corruption matches itself; a target that lost its arithmetic stops accepting drafts."""
        result = self.verdict(matrix_report(self.TEXTS, acceptance=(3.1, 2.9, 1.1, 3.4)),
                              matrix_report(self.TEXTS, acceptance=(3.0, 2.8, 1.1, 3.3)))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('user 2: full-draft rounds emit 1.10 tokens', result['platform_problems'][0])
        few = matrix_report(self.TEXTS)
        few['acceptance'] = dict(users=[dict(user=0, full_draft=dict(rounds=10, mean_emitted=1.0))])
        self.assertEqual(self.verdict(few, matrix_report(self.TEXTS))['verdict'], 'PASS', 'too few rounds to judge')

    def test_an_arm_with_problems_is_not_rerun(self):
        bad = list(self.TEXTS)
        bad[2] = 'answer X ' * 50
        broken = matrix_report(bad)
        broken['fatal'] = 'RuntimeError: engine died'
        self.assertEqual(self.verdict(broken, matrix_report(self.TEXTS))['verdict'], 'FAIL')
        self.assertEqual(self.verdict(matrix_report(bad), matrix_report(self.TEXTS))['verdict'], 'RERUN')


class FakeDocker(object):
    """Writes each arm's gate stdout the way the container would, from a per-arm report factory, and
    optionally a server.log (a wedge, for one)."""

    def __init__(self, reports, server_logs=None, default_log=None):
        self.reports, self.calls, self.server_logs = reports, [], server_logs or {}
        self.default_log = default_log

    def __call__(self, arguments, stdout_path, timeout, name):
        arm = name[len(driver.CONTAINER_PREFIX):]
        self.calls.append(dict(arm=arm, arguments=arguments, timeout=timeout))
        report = self.reports[arm](len([c for c in self.calls if c['arm'] == arm]))
        with open(stdout_path, 'w') as handle:
            if report is not None:
                handle.write('[REALTEXT] building\n%s\n%s\n%s\n' % (gate.BEGIN, json.dumps(report), gate.END))
            else:
                handle.write('docker: Error response from daemon\n')
        log_text = self.server_logs.get(arm, self.default_log)
        if log_text is not None:
            with open(os.path.join(os.path.dirname(stdout_path), 'server.log'), 'w') as handle:
                handle.write(log_text)
        return 0 if report is not None else 125


def any_request_log(requests=4, prompt=60, ladder='[256, 512, 1024, 2048]'):
    """What an engine under QWEN_FAST_ANY_REQUEST=1 logs: the D2 consumer's live line on its first step,
    and one build line per request (serving_request_factory)."""
    lines = ['INFO [PINDIAG] request quarantine installed on vllm_tt_plugin.scheduler.TTScheduler',
             'INFO ' + driver.QUARANTINE_LIVE]
    lines += ['INFO %scmpl-%d: captures <= 4 rows, replay attention and the T16 gate off, budget 16384 of '
              'max_tokens 16384 at position %d, proposal ladder %s' % (driver.ANY_REQUEST_ENGINE, i, prompt, ladder)
              for i in range(requests)]
    return chr(10).join(lines) + chr(10)


class DriverTests(unittest.TestCase):
    def run_driver(self, argv, reports, profiles=None, server_logs=None, containers=None, corpus=None,
                   default_log='any-request'):
        profiles = profiles or profiles_with_c2()
        if default_log == 'any-request':
            # An engine under a C2-any profile logs the consumer's live line; any other logs none of it.
            name = argv[argv.index('--profile') + 1] if '--profile' in argv else None
            default_log = any_request_log() if name in profiles['profiles'] and driver.any_request_profile(
                profiles, name) else None
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(profiles, handle)
            results = os.path.join(directory, 'results')
            docker, lines = FakeDocker(reports, server_logs, default_log), []
            code = driver.main(argv + ['--image', 'zot/img:c2', '--results', results, '--profiles', path],
                               execute=docker, devices=['/dev/tenstorrent/3', '/dev/tenstorrent/1'], log=lines.append,
                               containers=containers or (lambda: []),
                               corpus=corpus or (lambda: dict(V235['real_text']['corpus'])))
            summary_path = os.path.join(results, 'c2-gate-summary.json')
            summary = None
            if os.path.isfile(summary_path):
                with open(summary_path, encoding='utf-8') as handle:
                    summary = json.load(handle)
            files = sorted(os.listdir(results)) if os.path.isdir(results) else []
            arm_files = {arm: sorted(os.listdir(os.path.join(results, arm))) for arm in files if arm != 'c2-gate-summary.json'}
        return code, summary, docker.calls, lines, arm_files

    def test_a_bringup_that_reproduces_v235(self):
        code, summary, calls, lines, files = self.run_driver(
            ['--profile', 'exact', '--plan', 'bringup'], {'bringup-concurrent': lambda n: served_report()},
            profiles=CHECKOUT_PROFILES)
        self.assertEqual(code, 0, lines)
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['results']['bringup']['verdict'], 'PASS')
        self.assertEqual([c['arm'] for c in calls], ['bringup-concurrent'])
        self.assertIn('[C2-GATE] arm bringup-concurrent: exit 0 after', lines[1])
        self.assertIn('launched: [QWEN-C2] profile exact: vLLM argv', lines[1])
        self.assertIn('[C2-GATE] bringup user 0: IDENTICAL', lines)
        self.assertEqual(lines[-1], 'C2_GATE profile=exact plans=bringup passed=True')
        self.assertEqual(files['bringup-concurrent'], ['docker-run.json', 'gate-stdout.log', 'm3native-gate.json'])

    def test_a_bringup_on_another_corpus_is_not_comparable_before_any_card_is_opened(self):
        other = dict(V235['real_text']['corpus'], sha256='0' * 64, files=1970)
        code, summary, calls, lines, _ = self.run_driver(
            ['--profile', 'exact', '--plan', 'bringup'], {'bringup-concurrent': lambda n: served_report()},
            profiles=CHECKOUT_PROFILES, corpus=lambda: other)
        self.assertEqual((code, calls), (1, []))
        self.assertEqual(summary['results']['bringup']['verdict'], 'NOT_COMPARABLE')
        self.assertIn("its prompts are not the reference's", summary['results']['bringup']['reason'])

        def unreadable():
            raise RuntimeError('corpus read exited 125')
        code, summary, calls, lines, _ = self.run_driver(
            ['--profile', 'exact', '--plan', 'bringup'], {'bringup-concurrent': lambda n: served_report()},
            profiles=CHECKOUT_PROFILES, corpus=unreadable)
        self.assertEqual((code, [c['arm'] for c in calls]), (0, ['bringup-concurrent']), 'a failed check runs the arm')
        self.assertTrue(any('could not be read first (corpus read exited 125)' in line for line in lines))

    def test_a_matrix_divergence_reruns_both_arms_once_and_fails_when_it_reproduces(self):
        ladder = ','.join(str(length) for length in (60, 2048, 114944, 4096))
        texts = ['answer %d ' % i * 50 for i in range(4)]
        bad = list(texts)
        bad[2] = 'answer X ' * 50
        reports = {'matrix-concurrent': lambda n: matrix_report(bad),
                   'matrix-solo': lambda n: matrix_report(texts),
                   'matrix-concurrent-rerun': lambda n: matrix_report(bad),
                   'matrix-solo-rerun': lambda n: matrix_report(texts)}
        code, summary, calls, lines, _ = self.run_driver(['--profile', 'c2', '--plan', 'matrix', '--lengths', ladder],
                                                         reports)
        self.assertEqual(code, 1)
        self.assertEqual([c['arm'] for c in calls], ['matrix-concurrent', 'matrix-solo', 'matrix-concurrent-rerun',
                                                     'matrix-solo-rerun'])
        policy = summary['results']['matrix']['policy']
        self.assertEqual((summary['results']['matrix']['verdict'], policy['users'][2]['verdict']), ('FAIL', 'DIVERGED'))
        self.assertIn('[C2-GATE] matrix: a first divergence - re-running both arms once (the exactness policy)', lines)
        self.assertEqual(calls[0]['arguments'][calls[0]['arguments'].index('--prompt-lengths') + 1], ladder)

    def test_a_matrix_that_matches_needs_no_rerun_and_a_flaky_one_is_unstable(self):
        texts = ['answer %d ' % i * 50 for i in range(4)]
        same = {'matrix-concurrent': lambda n: matrix_report(texts), 'matrix-solo': lambda n: matrix_report(texts)}
        argv = ['--profile', 'c2', '--plan', 'matrix', '--lengths', '60,2048,4096,90']
        code, summary, calls, _, _ = self.run_driver(argv, same)
        self.assertEqual((code, summary['results']['matrix']['verdict'], len(calls)), (0, 'PASS', 2))
        bad = list(texts)
        bad[0] = 'other ' * 50
        flaky = dict(same, **{'matrix-concurrent': lambda n: matrix_report(bad),
                              'matrix-concurrent-rerun': lambda n: matrix_report(texts),
                              'matrix-solo-rerun': lambda n: matrix_report(texts)})
        code, summary, calls, _, _ = self.run_driver(argv, flaky)
        self.assertEqual((code, summary['results']['matrix']['verdict']), (1, 'UNSTABLE'))

    def test_the_default_ladder_runs_fitted_and_says_so(self):
        texts = ['answer %d ' % i * 50 for i in range(9)]
        same = {'matrix-concurrent': lambda n: matrix_report(texts), 'matrix-solo': lambda n: matrix_report(texts)}
        code, summary, calls, lines, _ = self.run_driver(['--profile', 'c2', '--plan', 'matrix'], same)
        self.assertEqual(code, 0, lines)
        self.assertFalse(any('[C2-GATE] note:' in line for line in lines), 'c2 takes the whole ladder')
        self.assertTrue(calls[0]['arguments'][calls[0]['arguments'].index('--prompt-lengths') + 1].endswith(',120000'))
        code, summary, calls, lines, _ = self.run_driver(['--profile', 'c2-narrow', '--plan', 'matrix'], same,
                                                         profiles=narrow_profiles())
        self.assertEqual(code, 0, lines)
        self.assertTrue(any('[C2-GATE] note: matrix: the default ladder\'s [120000] lowered to 114944' in line
                            for line in lines))
        self.assertTrue(calls[0]['arguments'][calls[0]['arguments'].index('--prompt-lengths') + 1].endswith(',114944'))

    def test_plans_the_profile_or_the_budget_cannot_take_are_refused_before_any_container(self):
        for argv, words in ((['--plan', 'matrix', '--lengths', '60,124000'], 'exceed 123136'),
                            (['--plan', 'bringup,matrix', '--lengths', '60,2048', '--max-tokens', '20000'],
                             'exceeds profile c2\'s output ceiling 16384'),
                            (['--plan', 'matrix,lifecycle', '--budget-seconds', '22200'], 'past the 22200 s'),
                            (['--plan', 'bringup', '--budget-seconds', '3000'], 'past the 3000 s')):
            with self.subTest(argv=argv):
                code, summary, calls, lines, _ = self.run_driver(['--profile', 'c2'] + argv, {})
                self.assertEqual((code, calls, summary), (2, [], None))
                self.assertIn(words, lines[-1])

    def test_an_arm_with_no_report_fails_its_plan_and_the_others_still_run(self):
        reports = {'bringup-concurrent': lambda n: None,
                   'memory-concurrent': lambda n: served_report(profile='c2', finish='stop', completion=500,
                                                                max_tokens=8192),
                   'memory-short': lambda n: served_report(profile='c2', finish='stop', completion=500,
                                                           max_tokens=16384)}
        code, summary, calls, lines, _ = self.run_driver(['--profile', 'c2', '--plan', 'bringup,memory'], reports)
        self.assertEqual(code, 1)
        self.assertEqual(summary['results']['bringup']['verdict'], 'FAIL')
        self.assertEqual(summary['results']['memory']['verdict'], 'PASS', summary['results']['memory'])
        self.assertEqual(sorted(summary['results']['memory']['arms']), ['memory-concurrent', 'memory-short'])
        ladders = summary['results']['memory']['arms']['memory-short']['any_request_engines']
        self.assertEqual(len(ladders), 4)
        self.assertTrue(all(line.endswith('proposal ladder [256, 512, 1024, 2048]') for line in ladders), ladders)
        self.assertTrue(any('NO [QWEN-C2] argv line' in line for line in lines))
        self.assertTrue(any('[PINDIAG] dram after engine cmpl-3' in line for line in lines))

    def test_a_platform_container_on_the_pair_stops_the_gate_as_infra(self):
        """Plan 2.1 item 1, checked before every arm: a placement can reload onto M+A mid-gate."""
        present = iter([[], ['thatch-inference-Qwen-Qwen3.8-27B']])
        texts = ['answer %d ' % i * 50 for i in range(4)]
        code, summary, calls, lines, _ = self.run_driver(
            ['--profile', 'c2', '--plan', 'matrix', '--lengths', '60,2048,4096,90'],
            {'matrix-concurrent': lambda n: matrix_report(texts), 'matrix-solo': lambda n: matrix_report(texts)},
            containers=lambda: next(present))
        self.assertEqual([c['arm'] for c in calls], ['matrix-concurrent'], 'the solo arm never started')
        self.assertEqual((code, summary['results']['matrix']['verdict']), (1, 'INFRA'))
        self.assertIn('thatch-inference-Qwen-Qwen3.8-27B', summary['infra'])
        self.assertEqual(summary['arms']['matrix-solo']['exit'], 'skipped')

    def test_the_ethernet_core_wedge_is_infra_and_skips_what_follows(self):
        """Review finding 11: the m3native gate resets before every run for llrt.cpp:594; an arm that
        hits it here is not a fast-path FAIL."""
        wedge = 'x\nTT_THROW @ llrt.cpp:594: Timed out while waiting for active ethernet core 1\n'
        code, summary, calls, lines, _ = self.run_driver(
            ['--profile', 'c2', '--plan', 'bringup,memory'],
            {'bringup-concurrent': lambda n: None}, server_logs={'bringup-concurrent': wedge})
        self.assertEqual([c['arm'] for c in calls], ['bringup-concurrent'])
        self.assertEqual((summary['results']['bringup']['verdict'], summary['results']['memory']['verdict']),
                         ('INFRA', 'INFRA'))
        self.assertIn('reset M+A', summary['infra'])
        self.assertTrue(lines[-1].startswith('C2_GATE profile=c2 plans=bringup,memory passed=False infra='))

    def test_c2_any_arms_need_the_quarantine_consumers_live_line(self):
        """Memory graft-mounted-is-not-graft-executed: a consumer wrapped onto a scheduler class the
        engine never builds is silent until a refusal strands a request. Under QWEN_FAST_ANY_REQUEST=1
        every arm that served must show it; its absence fails the arm even on identical texts."""
        texts = ['answer %d ' % i * 50 for i in range(4)]
        same = {'matrix-concurrent': lambda n: matrix_report(texts), 'matrix-solo': lambda n: matrix_report(texts)}
        argv = ['--profile', 'c2', '--plan', 'matrix', '--lengths', '60,2048,4096,90']
        code, summary, _, lines, files = self.run_driver(argv, same)
        self.assertEqual((code, summary['results']['matrix']['verdict']), (0, 'PASS'), lines)
        self.assertIn('server.log', files['matrix-concurrent'])
        silent = 'INFO [PINDIAG] request quarantine installed on vllm_tt_plugin.scheduler.TTScheduler' + chr(10)
        code, summary, _, lines, _ = self.run_driver(argv, same, server_logs={'matrix-solo': silent})
        self.assertEqual((code, summary['results']['matrix']['verdict']), (1, 'FAIL'))
        self.assertTrue(any('solo: QWEN_FAST_ANY_REQUEST=1 but the server log has no' in p
                            for p in summary['results']['matrix']['platform_problems']), summary['results']['matrix'])
        code, summary, _, _, _ = self.run_driver(argv, same, default_log=None)
        self.assertEqual((code, summary['results']['matrix']['verdict']), (1, 'FAIL'), 'no server.log: unproven')

    def test_no_c2_any_marker_may_appear_off_the_switch(self):
        """exact must run v235's path: an any-request marker in its log means the switch leaked."""
        code, summary, _, _, _ = self.run_driver(
            ['--profile', 'exact', '--plan', 'bringup'], {'bringup-concurrent': lambda n: served_report()},
            profiles=CHECKOUT_PROFILES, server_logs={'bringup-concurrent': any_request_log()})
        self.assertEqual((code, summary['results']['bringup']['verdict']), (1, 'FAIL'))
        self.assertTrue(any('carries C2-any markers' in p for p in summary['results']['bringup']['platform_problems']))
        self.assertEqual(driver.any_request_check('INFO nothing of it' + chr(10), False), ([], [], []))
        self.assertEqual(driver.any_request_check(None, False), ([], [], []))

    def test_the_consumer_counts_in_any_class_and_its_class_is_recorded(self):
        """A subclass of the wrapped TTScheduler still runs the inherited wrapper: live, and noted."""
        problems, _, consumers = driver.any_request_check(
            'INFO ' + driver.QUARANTINE_LIVE_PREFIX + 'OneInFlightScheduler' + chr(10), True)
        self.assertEqual((problems, consumers), ([], ['OneInFlightScheduler']))
        texts = ['answer %d ' % i * 50 for i in range(4)]
        same = {'matrix-concurrent': lambda n: matrix_report(texts), 'matrix-solo': lambda n: matrix_report(texts)}
        code, summary, _, lines, _ = self.run_driver(
            ['--profile', 'c2', '--plan', 'matrix', '--lengths', '60,2048,4096,90'], same,
            default_log='INFO ' + driver.QUARANTINE_LIVE_PREFIX + 'OneInFlightScheduler' + chr(10))
        self.assertEqual(code, 0, lines)
        self.assertEqual(summary['arms']['matrix-solo']['quarantine_consumers'], ['OneInFlightScheduler'])
        self.assertTrue(any('the D2 consumer ran in OneInFlightScheduler, not TTScheduler' in line for line in lines))

    def test_dry_run_and_bad_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(profiles_with_c2(), handle)
            lines = []
            base = ['--image', 'img', '--results', os.path.join(directory, 'r'), '--profiles', path]
            self.assertEqual(driver.main(base + ['--profile', 'exact', '--plan', 'bringup,memory', '--dry-run'],
                                         log=lines.append), 0)
            self.assertEqual(json.loads(lines[0]), dict(worst_case_seconds=2 * 120 + 3600 + 5400, budget_seconds=None))
            self.assertEqual([json.loads(line)['arm'] for line in lines[1:]], ['bringup-concurrent', 'memory-concurrent'])
            self.assertEqual(driver.main(base + ['--profile', 'nope'], log=lines.append), 2)
            self.assertEqual(driver.main(base + ['--profile', 'exact', '--plan', 'soak'], log=lines.append), 2)

    def test_serving_pair_resolves_by_board_id_and_refuses_a_missing_card(self):
        with tempfile.TemporaryDirectory() as directory:
            by_id = os.path.join(directory, 'by-id')
            os.makedirs(by_id)
            for name in ('3', '1'):
                open(os.path.join(directory, name), 'w').close()
            with self.assertRaisesRegex(RuntimeError, driver.CARD_M):
                driver.serving_pair(by_id)
            if not hasattr(os, 'symlink'):
                return
            try:
                os.symlink(os.path.join(directory, '3'), os.path.join(by_id, driver.CARD_M))
                os.symlink(os.path.join(directory, '1'), os.path.join(by_id, driver.CARD_A))
            except (OSError, NotImplementedError):
                self.skipTest('no symlinks here')
            self.assertEqual(driver.serving_pair(by_id), [os.path.realpath(os.path.join(directory, '3')),
                                                          os.path.realpath(os.path.join(directory, '1'))])


class ContainerLifetimeTests(unittest.TestCase):
    """Review finding 2: the arm's container must not outlive the step that ran it."""

    def run_execute(self, outcome):
        calls = []

        def run(arguments, **kwargs):
            calls.append(list(arguments))
            if arguments[:2] == ['docker', 'run']:
                if isinstance(outcome, BaseException):
                    raise outcome
                return subprocess.CompletedProcess(arguments, outcome)
            return subprocess.CompletedProcess(arguments, 0)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(driver.subprocess, 'run', side_effect=run):
            try:
                status = driver.Runner._execute(['docker', 'run', 'img'], os.path.join(directory, 'out.log'), 60, 'n')
            except BaseException as error:
                status = error
        return status, calls

    def test_the_container_is_removed_before_and_after_every_way_the_run_ends(self):
        for outcome in (0, subprocess.TimeoutExpired('docker', 60), SystemExit(143), KeyboardInterrupt()):
            with self.subTest(outcome=type(outcome).__name__):
                status, calls = self.run_execute(outcome)
                self.assertEqual(calls[0], ['docker', 'rm', '-f', 'n'], 'a leftover of the same name goes first')
                self.assertEqual(calls[-1], ['docker', 'rm', '-f', 'n'])
                self.assertEqual(calls[1], ['docker', 'run', 'img'])
                if isinstance(outcome, subprocess.TimeoutExpired):
                    self.assertEqual(status, 'timeout')
                elif isinstance(outcome, BaseException):
                    self.assertIs(status, outcome, 'the signal still ends the driver')

    def test_sigterm_becomes_an_exit_so_the_removal_runs(self):
        with self.assertRaises(SystemExit) as caught:
            driver._terminate(signal.SIGTERM, None)
        self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
        with open(os.path.join(HERE, 'c2_serving_gate.py'), encoding='utf-8') as handle:
            self.assertIn('signal.signal(signal.SIGTERM, _terminate)', handle.read())


def lifecycle_report(events, alive=True, texts=None, profile='c2', drift=0.0, phases=None, barriers=None):
    """A lifecycle arm's report: six users, the solo texts cut or extended the way `events` says, each
    drop recorded by the watch in the phase its kind must hit (or `phases`[user])."""
    texts = texts or ['lifecycle answer %d ' % i * 40 for i in range(6)]
    shas = ['%064x' % (i + 7) for i in range(6)]
    streams, comparisons, drops, watched = [], [], {}, {}
    phases = phases or {}
    for index, text in enumerate(texts):
        kind = events.get(index)
        if kind == 'drop':
            stream = dict(text=text[:30], dropped='after 5 chunks')
            drops[str(index)] = 'chunks 5'
            watched[str(index)] = dict(kind='chunks', spec='chunks 5', fired=True, fired_s=10.0 + index,
                                       reason='after 5 chunks', live=4, phase=phases.get(index, 'decode'))
        elif kind == 'cancel':
            stream = dict(text='', dropped='5 s into its prefill')
            drops[str(index)] = 'prefill+5'
            watched[str(index)] = dict(kind='prefill', spec='prefill+5', fired=True, fired_s=6.0,
                                       reason='5 s into its prefill', live=0, phase=phases.get(index, 'prefill'))
        elif kind == 'one':
            stream = dict(text=text[:8], finish_reason='length', completion_tokens=1)
        elif kind == 'ignore':
            stream = dict(text=text + ' and on past EOS', finish_reason='length', completion_tokens=2048)
        else:
            stream = dict(text=text, finish_reason='stop', completion_tokens=300, prompt_tokens=100)
        streams.append(stream)
        comparisons.append(dict(user=index, prompt_sha256=shas[index], max_tokens=1 if kind == 'one' else 2048,
                                ignore_eos=kind == 'ignore'))
    seats = 4 if events else 1
    return dict(streams=streams, comparisons=comparisons, max_tokens=2048, qwen_configuration=dict(QWEN_C2_PROFILE=profile),
                real_text=dict(users=[dict(user=i, prompt_sha256=s) for i, s in enumerate(shas)]),
                platform=dict(served_profile=profile, served_argv=['--x'], problems=[]),
                real_text_stream_problems=[], alive=alive,
                alive_after=[dict(text='OK')] * seats if alive else [dict(text='OK')] * (seats - 1) + [
                    dict(error='no answer within 600 s (a seat not given back?)')],
                flag_markers=dict(missing=[]), user_events=dict(drops=drops),
                lifecycle=dict(events=watched, barriers=barriers or []),
                ledger=dict(residual_status='passed', idle_drift_gb={'0': drift, '1': drift}))


class LifecycleTests(unittest.TestCase):
    def test_the_arms_parse_and_share_one_prompt_set(self):
        arms = driver.plan_arms('lifecycle', 'c2', profiles_with_c2())
        self.assertEqual([arm for arm, _, _ in arms], ['lifecycle-drops', 'lifecycle-edges', 'lifecycle-solo'])
        options = [parse_harness(args) for _, args, _ in arms]
        self.assertEqual({tuple(o.prompt_lengths) for o in options}, {driver.LIFECYCLE_LENGTHS})
        self.assertEqual({o.max_tokens for o in options}, {driver.LIFECYCLE_MAX_TOKENS})
        drops, edges, solo = options
        self.assertEqual(drops.events['drops'], {0: ('build', 0), 1: ('live', 4), 2: ('live', 3), 3: ('live', 2)})
        self.assertEqual(drops.events['ignore_eos'], [5])
        barrier = ('barrier', (60, (1, 2, 3)))
        self.assertEqual(edges.events, dict(drops={0: ('prefill', 5.0), 1: barrier, 2: barrier, 3: barrier},
                                            max_tokens={4: 1}, ignore_eos=[]))
        self.assertEqual((drops.alive_check, edges.alive_check), (4, 4), 'every seat of the profile, at once')
        self.assertEqual((solo.users, solo.sequential_users, any(solo.events.values()), solo.alive_check),
                         (1, 6, False, 1))
        self.assertGreater(len(driver.LIFECYCLE_LENGTHS), 4, 'more users than seats: the 5th request takes a freed slot')
        self.assertGreaterEqual(driver.LIFECYCLE_LENGTHS[0], 60000, 'the cancelled first arrival has a long prefill')
        self.assertEqual(len(set(driver.LIFECYCLE_LENGTHS)), len(driver.LIFECYCLE_LENGTHS),
                         'the server log names a user by its prompt length')

    def test_consistent_events_a_live_engine_and_a_flat_ledger_pass(self):
        solo = lifecycle_report({})
        drops = lifecycle_report({0: 'drop', 1: 'drop', 2: 'drop', 3: 'drop', 5: 'ignore'})
        edges = lifecycle_report({0: 'cancel', 1: 'drop', 2: 'drop', 3: 'drop', 4: 'one'})
        self.assertEqual(driver.lifecycle_verdict(drops, solo)['verdict'], 'PASS')
        result = driver.lifecycle_verdict(edges, solo)
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        self.assertTrue(any(line.startswith('ledger idle drift') and 'P7 residual passed' in line
                            for line in result['lines']))
        self.assertIn('user 0: prefill+5 -> 5 s into its prefill at 6.0 s during prefill, 0 live', result['lines'])

    def test_an_event_that_missed_its_phase_is_not_exercised(self):
        solo = lifecycle_report({})
        late = lifecycle_report({0: 'cancel', 4: 'one'}, phases={0: 'decode'})
        result = driver.lifecycle_verdict(late, solo)
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertIn('user 0: prefill+5 fired during decode, not prefill', result['shortfalls'])
        unfired = lifecycle_report({1: 'drop'})
        unfired['lifecycle']['events']['1'] = dict(kind='chunks', spec='chunks 5', fired=False, missed='never due')
        self.assertEqual(driver.lifecycle_verdict(unfired, solo)['verdict'], 'NOT_EXERCISED')
        unwatched = lifecycle_report({1: 'drop'})
        del unwatched['lifecycle']
        self.assertEqual(driver.lifecycle_verdict(unwatched, solo)['verdict'], 'NOT_EXERCISED')
        undelivered = lifecycle_report({0: 'cancel'})
        undelivered['lifecycle']['events']['0'].update(delivered=False, undelivered='no socket under the response')
        result = driver.lifecycle_verdict(undelivered, solo)
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertIn('its cancel never reached the socket', result['shortfalls'][0])

    def test_a_multiple_drop_must_close_together(self):
        solo = lifecycle_report({})
        events = {1: 'drop', 2: 'drop', 3: 'drop'}
        together = [dict(members=[1, 2, 3], chunks=60, complete=True, released_s=13.0)]
        report = lifecycle_report(events, barriers=together)
        for user in ('1', '2', '3'):
            report['lifecycle']['events'][user]['fired_s'] = 13.0 + 0.01 * int(user)
        self.assertEqual(driver.lifecycle_verdict(report, solo)['verdict'], 'PASS')
        report['lifecycle']['events']['3']['fired_s'] = 15.0
        self.assertEqual(driver.lifecycle_verdict(report, solo)['verdict'], 'NOT_EXERCISED')
        broken = lifecycle_report(events, barriers=[dict(together[0], complete=False)])
        result = driver.lifecycle_verdict(broken, solo)
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertIn('released before all had 60 chunks', result['shortfalls'][0])

    def test_the_ledger_must_stay_flat_against_the_solo_arm(self):
        """Review finding 7: the row passes on "the ledger stays flat", now judged, not only reported."""
        solo = lifecycle_report({}, drift=0.2)
        grown = driver.lifecycle_verdict(lifecycle_report({1: 'drop'}, drift=0.5), solo)
        self.assertEqual(grown['verdict'], 'FAIL')
        self.assertIn('the ledger did not stay flat', grown['problems'][0])
        self.assertEqual(driver.lifecycle_verdict(lifecycle_report({1: 'drop'}, drift=0.25), solo)['verdict'], 'PASS')
        missing = lifecycle_report({1: 'drop'})
        missing['ledger'] = dict(residual_status=None, idle_drift_gb=None)
        result = driver.lifecycle_verdict(missing, solo)
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertIn('flatness not judged', result['shortfalls'][0])

    def test_a_seat_not_given_back_or_a_failed_survivor_fails(self):
        solo = lifecycle_report({})
        dead = driver.lifecycle_verdict(lifecycle_report({1: 'drop'}, alive=False), solo)
        self.assertEqual(dead['verdict'], 'FAIL')
        self.assertIn('did not answer every seat after the streams (no answer within 600 s', dead['problems'][0])
        broken = lifecycle_report({1: 'drop'})
        broken['real_text_stream_problems'] = ['user 4: stream error HTTP 500']
        self.assertEqual(driver.lifecycle_verdict(broken, solo)['verdict'], 'FAIL')
        self.assertEqual(driver.lifecycle_verdict(None, solo)['verdict'], 'FAIL')

    def test_ignore_eos_that_stopped_at_eos_is_a_divergence(self):
        """Review finding 5: an engine that ignores ignore_eos stops where the solo run did; that was IDENTICAL."""
        solo = lifecycle_report({})
        ignored = lifecycle_report({5: 'ignore'})
        ignored['streams'][5] = dict(solo['streams'][5])
        result = driver.lifecycle_verdict(ignored, solo)
        self.assertEqual(result['verdict'], 'RERUN')
        self.assertEqual(result['policy']['first']['users'][5]['detail'], 'ignore_eos ignored: stopped at the solo EOS')

    def test_the_plan_reruns_a_divergent_arm_with_the_solo_once(self):
        texts = ['lifecycle answer %d ' % i * 40 for i in range(6)]
        bad = list(texts)
        bad[5] = 'something else entirely ' * 30
        reports = {'lifecycle-drops': lambda n: lifecycle_report({1: 'drop'}, texts=bad),
                   'lifecycle-edges': lambda n: lifecycle_report({4: 'one'}),
                   'lifecycle-solo': lambda n: lifecycle_report({}),
                   'lifecycle-drops-rerun': lambda n: lifecycle_report({1: 'drop'}),
                   'lifecycle-solo-rerun': lambda n: lifecycle_report({})}
        code, summary, calls, lines, _ = DriverTests().run_driver(['--profile', 'c2', '--plan', 'lifecycle'], reports)
        self.assertEqual([c['arm'] for c in calls], ['lifecycle-drops', 'lifecycle-edges', 'lifecycle-solo',
                                                     'lifecycle-solo-rerun', 'lifecycle-drops-rerun'])
        result = summary['results']['lifecycle']
        self.assertEqual((result['arms']['lifecycle-drops']['verdict'], result['arms']['lifecycle-edges']['verdict']),
                         ('UNSTABLE', 'PASS'))
        self.assertEqual((result['verdict'], code), ('UNSTABLE', 1))
        self.assertTrue(any('re-running lifecycle-drops and the solo arm once' in line for line in lines))


class WorkflowTests(unittest.TestCase):
    @staticmethod
    def text():
        with open(WORKFLOW, encoding='utf-8') as handle:
            return handle.read()

    @classmethod
    def budget_literals(cls):
        """The gate step's budget arithmetic: (step seconds left, job minutes) as the step computes them."""
        text = cls.text()
        step = text[text.index('- name: Run the gate'):text.index('- name: Replay')]
        step_left = re.search(r'step_left=\$\(\( ([0-9]+) \* 60 - ([0-9]+) \)\)', step)
        job_left = re.search(r'job_left=\$\(\( ([0-9]+) \* 60 ', step)
        return dict(step=int(step_left.group(1)) * 60 - int(step_left.group(2)), step_minutes=int(step_left.group(1)),
                    job_minutes=int(job_left.group(1)))

    def test_the_gate_step_runs_the_driver_with_the_jobs_choices(self):
        text = self.text()
        step = text[text.index('- name: Run the gate'):text.index('- name: Replay')]
        self.assertIn("if: contains(steps.job.outputs.actions, 'gate')", step)
        self.assertIn('python3 scripts/ci/c2_serving_gate.py', step)
        for option, output in (('--profile', 'profile'), ('--plan', 'gate_plan'), ('--lengths', 'gate_lengths'),
                               ('--max-tokens', 'gate_max_tokens')):
            self.assertRegex(step, re.escape(output.upper()) + r': \$\{\{ steps\.job\.outputs\.' + output + r' \}\}')
            self.assertIn(option, step)
        self.assertIn('${GATE_LENGTHS:+--lengths "$GATE_LENGTHS"}', step, 'empty: the driver\'s fitted ladder')
        self.assertIn('--budget-seconds "$budget"', step)
        self.assertIn("grep -F '[QWEN-C2]'", step)

    def test_the_budget_is_the_steps_and_the_jobs_own_timeouts(self):
        text = self.text()
        literals = self.budget_literals()
        job = re.search(r'\n  c2-serving:\n(?:    [^\n]*\n)*?    timeout-minutes: ([0-9]+)\n', text)
        step = text[text.index('- name: Run the gate'):text.index('- name: Replay')]
        self.assertEqual(int(job.group(1)), literals['job_minutes'])
        self.assertIn('timeout-minutes: %d' % literals['step_minutes'], step)
        self.assertIn('echo "C2_JOB_STARTED=$(date +%s)" >> "$GITHUB_ENV"', text)
        self.assertIn('${C2_JOB_STARTED:?}', step)

    def test_every_gate_container_goes_with_the_step(self):
        text = self.text()
        step = text[text.index('- name: Run the gate'):text.index('- name: Replay')]
        self.assertIn("grep '^%s' | xargs -r docker rm -f" % driver.CONTAINER_PREFIX, step)
        self.assertIn('trap remove_gate_containers EXIT', step)
        self.assertLess(step.index('trap remove_gate_containers EXIT'), step.index('python3 scripts/ci/c2_serving_gate.py'))
        self.assertLess(step.index('remove_gate_containers\n          fi'), step.index('sudo -n fuser'),
                        'a stale gate container is removed before the cards are checked for holders')


if __name__ == '__main__':
    unittest.main()
