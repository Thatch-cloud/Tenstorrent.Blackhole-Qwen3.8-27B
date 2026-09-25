"""c2_serving_gate: the C2 serving image's gates in the node agent's container shape, held on CPU.

Nothing here opens a device or runs docker: the arms' `docker run` argv, the harness options each
plan passes (parsed by the harness's own parse_options), the verdicts, and the whole driver with an
injected executor that writes the gate's stdout the way the container would."""

import ast
import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as driver  # noqa: E402
import c2_serving_job  # noqa: E402
import lever_n_m3native_gate as gate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    CHECKOUT_PROFILES = json.load(_handle)
with open(driver.REFERENCE, encoding='utf-8') as _handle:
    V235 = json.load(_handle)


def profiles_with_c2():
    """The checkout's profiles plus a c2 shaped like the plan's (2.2: 131,328 geometry, 16,384 ceiling,
    a ~123k prompt cap) - the real one lands in qwen_c2_profiles.json on another track."""
    profiles = json.loads(json.dumps(CHECKOUT_PROFILES))
    c2 = json.loads(json.dumps(profiles['profiles']['exact']))
    c2['env']['QWEN_FAST_OUTPUT_BUDGET'] = '16384'
    c2['max_prompt_tokens'] = 123136
    profiles['profiles']['c2'] = c2
    return profiles


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


def pairs_of(arguments):
    pairs = set()
    for index, token in enumerate(arguments):
        if token in ('--tmpfs', '--shm-size', '--memory', '--cpus', '--cap-add', '-e', '-v'):
            pairs.add((token, arguments[index + 1]))
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

    def test_the_snapshot_is_the_profiles_first_under_the_agents_mount(self):
        self.assertEqual(driver.SNAPSHOT, CHECKOUT_PROFILES['profiles']['exact']['snapshots'][0])

    def test_the_matrix_runs_the_ladder_concurrent_then_solo_on_the_same_prompts(self):
        arms = driver.plan_arms('matrix', 'c2', profiles_with_c2())
        (c_arm, c_args, _), (s_arm, s_args, _) = arms
        concurrent, solo = parse_harness(c_args), parse_harness(s_args)
        self.assertEqual((c_arm, s_arm), ('matrix-concurrent', 'matrix-solo'))
        self.assertEqual(concurrent.prompt_lengths, list(c2_serving_job.LADDER))
        self.assertEqual(solo.prompt_lengths, list(c2_serving_job.LADDER))
        self.assertEqual((concurrent.users, concurrent.sequential_users), (9, 0))
        self.assertEqual((solo.users, solo.sequential_users), (1, 9))
        self.assertEqual(concurrent.max_tokens, solo.max_tokens)
        self.assertEqual(concurrent.max_tokens, 4096)
        self.assertEqual(concurrent.context, 131328)
        self.assertEqual([a for a in c_args if a not in s_args], ['--stagger', '0.25'])

    def test_the_memory_arm_is_four_largest_prompts_at_the_ceiling(self):
        (arm, args, _), = driver.plan_arms('memory', 'c2', profiles_with_c2())
        options = parse_harness(args)
        self.assertEqual(driver.profile_limits(profiles_with_c2(), 'c2'), (131328, 16384, 114944))
        self.assertEqual(options.prompt_lengths, [114944] * 4, 'the contract today: context less the ceiling')
        self.assertEqual(options.max_tokens, 16384)
        (_, args, _), = driver.plan_arms('memory', 'c2', profiles_with_c2(), memory_prompt=123136)
        self.assertEqual(parse_harness(args).prompt_lengths, [123136] * 4)
        (_, args, _), = driver.plan_arms('memory', 'exact', CHECKOUT_PROFILES)
        self.assertEqual((parse_harness(args).prompt_lengths, parse_harness(args).max_tokens), ([131072] * 4, 256))

    def test_an_unknown_plan_is_refused(self):
        with self.assertRaises(ValueError):
            driver.plan_arms('soak', 'exact', CHECKOUT_PROFILES)


def served_report(texts=None, gate_passed=True, profile='exact', problems=None, users=4, engines=4, error=None,
                  configuration=None):
    streams = V235['streams']
    texts = texts or [s['text'] for s in streams][:users]
    entries = [dict(text=t, finish_reason='length', completion_tokens=256, prompt_tokens=131072) for t in texts]
    if error is not None:
        entries[error] = dict(error='HTTP 400: prompt too long')
    dram = [dict(event='engine', request='cmpl-%d' % i, chips=[dict(chip=0, free_gb=3.0 - i, largest_free_mb=100.0)],
                 line='[PINDIAG] dram after engine cmpl-%d: chip0 ...' % i) for i in range(engines)]
    return dict(
        users=users, gate_passed=gate_passed, max_tokens=256,
        qwen_configuration=configuration or dict(V235['qwen_configuration'], QWEN_C2_SERVING='1',
                                                 QWEN_C2_PROFILE=profile),
        streams=entries,
        real_text=dict(users=[dict(user=i, prompt_sha256=streams[i % 4]['prompt_sha256']) for i in range(users)]),
        comparisons=[dict(user=i, max_tokens=256) for i in range(users)],
        platform=dict(served_profile=profile, served_argv=['--max-model-len', '131328'], problems=problems or []),
        dram=dict(events=dram, engines=engines, min_free_gb=3.0 - max(engines - 1, 0)),
        real_text_stream_problems=['user %d: stream error' % error] if error is not None else [],
        flag_markers=dict(missing=[] if gate_passed else ['QWEN_FAST_GDN_SEQ_BLOCK: x']))


class VerdictTests(unittest.TestCase):
    def test_bringup_passes_on_identical_texts_and_the_harness_verdict(self):
        result = driver.bringup_verdict(served_report(), V235)
        self.assertEqual((result['verdict'], result['texts']), ('PASS', 'IDENTICAL'))
        self.assertEqual([u['verdict'] for u in result['users']], ['IDENTICAL'] * 4)
        self.assertEqual(result['arithmetic_diff'], {})

    def test_bringup_fails_on_a_divergence_a_missing_marker_or_a_contract_problem(self):
        texts = [s['text'] for s in V235['streams']]
        texts[1] = texts[1][:10] + '?' + texts[1][11:]
        result = driver.bringup_verdict(served_report(texts=texts), V235)
        self.assertEqual((result['verdict'], result['users'][1]['verdict']), ('FAIL', 'DIVERGED'))
        self.assertIn('user 1: DIVERGED at character 10', '\n'.join(result['lines']))
        markers = driver.bringup_verdict(served_report(gate_passed=False), V235)
        self.assertEqual((markers['verdict'], markers['texts']), ('FAIL', 'IDENTICAL'))
        self.assertEqual(markers['missing_markers'], ['QWEN_FAST_GDN_SEQ_BLOCK: x'])
        contract = driver.bringup_verdict(served_report(problems=['served profile general']), V235)
        self.assertEqual(contract['verdict'], 'FAIL')
        self.assertEqual(driver.bringup_verdict(None, V235)['verdict'], 'FAIL')

    def test_memory_records_and_needs_every_engines_line(self):
        result = driver.memory_verdict(served_report())
        self.assertEqual((result['verdict'], result['engines'], len(result['lines'])), ('PASS', 4, 4))
        self.assertEqual(driver.memory_verdict(served_report(engines=3))['verdict'], 'FAIL')
        self.assertEqual(driver.memory_verdict(served_report(error=2))['verdict'], 'FAIL')

    def test_the_launched_argv_line_is_the_contracts_own(self):
        line = driver.launched_argv_line(served_report())
        self.assertEqual(line, '[QWEN-C2] profile exact: vLLM argv ["--max-model-len", "131328"]')
        self.assertTrue(gate.QWEN_C2_ARGV.search(line))
        self.assertIsNone(driver.launched_argv_line(dict(platform=dict(problems=['x']))))
        self.assertIsNone(driver.launched_argv_line(None))


class FakeDocker(object):
    """Writes each arm's gate stdout the way the container would, from a per-arm report factory."""

    def __init__(self, reports):
        self.reports, self.calls = reports, []

    def __call__(self, arguments, stdout_path, timeout, name):
        arm = name[len('qwen-c2-gate-'):]
        self.calls.append(dict(arm=arm, arguments=arguments, timeout=timeout))
        report = self.reports[arm](len([c for c in self.calls if c['arm'] == arm]))
        with open(stdout_path, 'w') as handle:
            if report is not None:
                handle.write('[REALTEXT] building\n%s\n%s\n%s\n' % (gate.BEGIN, json.dumps(report), gate.END))
            else:
                handle.write('docker: Error response from daemon\n')
        return 0 if report is not None else 125


class DriverTests(unittest.TestCase):
    def run_driver(self, argv, reports, profiles=None):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(profiles or profiles_with_c2(), handle)
            results = os.path.join(directory, 'results')
            docker, lines = FakeDocker(reports), []
            code = driver.main(argv + ['--image', 'zot/img:c2', '--results', results, '--profiles', path],
                               execute=docker, devices=['/dev/tenstorrent/3', '/dev/tenstorrent/1'], log=lines.append)
            with open(os.path.join(results, 'c2-gate-summary.json'), encoding='utf-8') as handle:
                summary = json.load(handle)
            files = sorted(os.listdir(results))
            arm_files = {arm: sorted(os.listdir(os.path.join(results, arm))) for arm in files if arm != 'c2-gate-summary.json'}
        return code, summary, docker.calls, lines, arm_files

    def test_a_bringup_that_reproduces_v235(self):
        code, summary, calls, lines, files = self.run_driver(
            ['--profile', 'exact', '--plan', 'bringup'], dict(**{'bringup-concurrent': lambda n: served_report()}))
        self.assertEqual(code, 0)
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['results']['bringup']['verdict'], 'PASS')
        self.assertEqual([c['arm'] for c in calls], ['bringup-concurrent'])
        self.assertIn('[C2-GATE] arm bringup-concurrent: exit 0 after', lines[1])
        self.assertIn('launched: [QWEN-C2] profile exact: vLLM argv', lines[1])
        self.assertIn('[C2-GATE] bringup user 0: IDENTICAL', lines)
        self.assertEqual(lines[-1], 'C2_GATE profile=exact plans=bringup passed=True')
        self.assertEqual(files['bringup-concurrent'], ['docker-run.json', 'gate-stdout.log', 'm3native-gate.json'])
        self.assertEqual(summary['arms']['bringup-concurrent']['launched'],
                         '[QWEN-C2] profile exact: vLLM argv ["--max-model-len", "131328"]')

    def test_a_matrix_divergence_reruns_both_arms_once_and_fails_when_it_reproduces(self):
        ladder = ','.join(str(length) for length in (60, 2048, 120000, 4096))
        texts = ['answer %d ' % i * 50 for i in range(4)]
        bad = list(texts)
        bad[2] = 'answer X ' * 50
        reports = {'matrix-concurrent': lambda n: served_report(texts=bad, profile='c2'),
                   'matrix-solo': lambda n: served_report(texts=texts, profile='c2'),
                   'matrix-concurrent-rerun': lambda n: served_report(texts=bad, profile='c2'),
                   'matrix-solo-rerun': lambda n: served_report(texts=texts, profile='c2')}
        code, summary, calls, lines, _ = self.run_driver(['--profile', 'c2', '--plan', 'matrix', '--lengths', ladder],
                                                         reports)
        self.assertEqual(code, 1)
        self.assertEqual([c['arm'] for c in calls], ['matrix-concurrent', 'matrix-solo', 'matrix-concurrent-rerun',
                                                     'matrix-solo-rerun'])
        policy = summary['results']['matrix']['policy']
        self.assertEqual((summary['results']['matrix']['verdict'], policy['users'][2]['verdict']), ('FAIL', 'DIVERGED'))
        self.assertIn('[C2-GATE] matrix: a first divergence - re-running both arms once (the exactness policy)', lines)
        self.assertIn('--prompt-lengths', calls[0]['arguments'])
        self.assertEqual(calls[0]['arguments'][calls[0]['arguments'].index('--prompt-lengths') + 1], ladder)

    def test_a_matrix_that_matches_needs_no_rerun_and_a_flaky_one_is_unstable(self):
        texts = ['answer %d ' % i * 50 for i in range(4)]
        same = {'matrix-concurrent': lambda n: served_report(texts=texts, profile='c2'),
                'matrix-solo': lambda n: served_report(texts=texts, profile='c2')}
        code, summary, calls, _, _ = self.run_driver(['--profile', 'c2', '--plan', 'matrix', '--lengths', '60,2048,4096,9'],
                                                     same)
        self.assertEqual((code, summary['results']['matrix']['verdict'], len(calls)), (0, 'PASS', 2))
        bad = list(texts)
        bad[0] = 'other ' * 50
        flaky = dict(same, **{'matrix-concurrent': lambda n: served_report(texts=bad, profile='c2'),
                              'matrix-concurrent-rerun': lambda n: served_report(texts=texts, profile='c2'),
                              'matrix-solo-rerun': lambda n: served_report(texts=texts, profile='c2')})
        code, summary, calls, _, _ = self.run_driver(['--profile', 'c2', '--plan', 'matrix', '--lengths', '60,2048,4096,9'],
                                                     flaky)
        self.assertEqual((code, summary['results']['matrix']['verdict']), (1, 'UNSTABLE'))

    def test_an_arm_with_no_report_fails_its_plan_and_the_others_still_run(self):
        reports = {'bringup-concurrent': lambda n: None, 'memory-concurrent': lambda n: served_report(profile='c2')}
        code, summary, calls, lines, _ = self.run_driver(['--profile', 'c2', '--plan', 'bringup,memory'], reports)
        self.assertEqual(code, 1)
        self.assertEqual(summary['results']['bringup']['verdict'], 'FAIL')
        self.assertEqual(summary['results']['memory']['verdict'], 'PASS')
        self.assertTrue(any('NO [QWEN-C2] argv line' in line for line in lines))
        self.assertTrue(any('[PINDIAG] dram after engine cmpl-3' in line for line in lines))

    def test_dry_run_and_bad_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(profiles_with_c2(), handle)
            lines = []
            base = ['--image', 'img', '--results', os.path.join(directory, 'r'), '--profiles', path]
            self.assertEqual(driver.main(base + ['--profile', 'exact', '--plan', 'bringup,memory', '--dry-run'],
                                         log=lines.append), 0)
            self.assertEqual([json.loads(line)['arm'] for line in lines], ['bringup-concurrent', 'memory-concurrent'])
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


class WorkflowTests(unittest.TestCase):
    def test_the_gate_step_runs_the_driver_with_the_jobs_choices(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        step = text[text.index('- name: Run the gate'):text.index('- name: Replay')]
        self.assertIn("if: contains(steps.job.outputs.actions, 'gate')", step)
        self.assertIn('python3 scripts/ci/c2_serving_gate.py', step)
        for option, output in (('--profile', 'profile'), ('--plan', 'gate_plan'), ('--lengths', 'gate_lengths'),
                               ('--max-tokens', 'gate_max_tokens')):
            self.assertRegex(step, re.escape(output.upper()) + r': \$\{\{ steps\.job\.outputs\.' + output + r' \}\}')
            self.assertIn(option, step)
        self.assertIn("grep -F '[QWEN-C2]'", step)


if __name__ == '__main__':
    unittest.main()
