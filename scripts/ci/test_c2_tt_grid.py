"""C2_TT_GRID: the compute-grid clamp (TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE) on every path of qwen-c2-serving.yml that opens a card, held on CPU.

When the cards' firmware stops soft-harvesting two Tensix columns, tt-metal selects the unharvested core descriptor and exposes a 13x10 compute grid
instead of 11x10. TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="10,9" clamps it back and is valid on both card kinds, so the job file may ask for it
(C2_TT_GRID: exactly 10,9, 11,9 or 12,9, or unset: the variable is not passed at all, today's behaviour) and every container a card job starts must
carry it. Held here:

  the parse         the three values pass, anything else is refused, unset is '' and passes nothing (c2_serving_job);
  the drivers       the gate, the prefix gate, the tau lab and the platform replay put the pair into the argv they launch (--tt-grid), and
                    launch exactly what they did before without it;
  the cardm harness the step hands QUAL_TT_GRID to the harness; a harness that does not put it into its container is refused with the key
                    set, and the seven run_card_b.sh harnesses do (their dry runs show it);
  the workflow      the eight steps that open a card are the steps that carry it - a new launch site that forgets the clamp fails here - and
                    their launch commands, run in bash with the variable set and unset, hold the pair only when it is set.

Stdlib only; the shell checks skip without bash."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_platform_replay as replay  # noqa: E402
import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import c2_tau_lab as lab  # noqa: E402
import test_c2_prefix_gate as prefix_tests  # noqa: E402
import test_tau_lab as tau_tests  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
CPU_WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml')
OPS = os.path.join(ROOT, 'optimisation', 'ttnn-op')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
NL = chr(10)
ENV = 'TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE'
PAIR = '-e'
BASH = shutil.which('bash')

with open(os.path.join(HERE, 'qual_card.sh'), encoding='utf-8') as _handle:
    # The harnesses refuse a board id they do not know: the library's own first serving card (no board id is written in this file).
    CARD_M = re.search(r"^QUAL_SERVING_CARDS='(blackhole-[0-9A-F]+)", _handle.read(), flags=re.M).group(1)
PROFILE_NAMES = ['coding', 'exact', 'general']


def read(**values):
    base = dict(C2_IMAGE_TAG='v6-teardown')
    base.update(values)
    return job.read_job(base, PROFILE_NAMES)


def pairs_in(argv):
    """Every ('-e', 'NAME=value') of an argv whose NAME is the clamp's."""
    return [(argv[i], argv[i + 1]) for i in range(len(argv) - 1) if argv[i] == PAIR and argv[i + 1].split('=', 1)[0] == ENV]


def clamp_count(argv):
    return len([token for token in argv if token.startswith(ENV + '=')])


class ParseTests(unittest.TestCase):
    def test_unset_is_empty_and_passes_nothing(self):
        self.assertEqual(read()['tt_grid'], '')
        self.assertEqual(job.read_tt_grid({}), '')
        self.assertEqual(job.read_tt_grid(dict(C2_TT_GRID='')), '')
        self.assertEqual(job.tt_grid_arguments(None), [])
        self.assertEqual(job.tt_grid_arguments(''), [])

    def test_the_three_exact_values_pass_through(self):
        for value in ('10,9', '11,9', '12,9'):
            with self.subTest(value=value):
                self.assertEqual(read(C2_TT_GRID=value)['tt_grid'], value)
                self.assertEqual(job.tt_grid_arguments(value), ['-e', '%s=%s' % (ENV, value)])
        self.assertEqual(job.TT_GRIDS, ('10,9', '11,9', '12,9'))
        self.assertEqual(job.TT_GRID_ENV, ENV)

    def test_everything_else_is_refused(self):
        for value in ('13,9', '9,9', '10,10', '10, 9', ' 10,9', '10,9 ', '10;9', '10x9', '110', '10', '0', 'true', 'none', '10,9,',
                      '$(id)', '10,9' + NL, '010,9', '10,09', '+10,9', '11x10', '10.9'):
            with self.subTest(value=value):
                with self.assertRaises(job.JobError):
                    read(C2_TT_GRID=value)
                with self.assertRaises(job.JobError):
                    job.tt_grid_arguments(value)

    def test_the_file_parse_strips_the_line_but_not_the_value(self):
        self.assertEqual(job.parse_env('C2_TT_GRID = 10,9 ' + NL)['C2_TT_GRID'], '10,9')
        with self.assertRaises(job.JobError):
            read(**job.parse_env('C2_TT_GRID=10, 9'))

    def test_any_job_may_carry_it_and_it_never_changes_another_output(self):
        base = read()
        for actions in ('status', 'status reset', 'status build', 'status smoke', 'status gate'):
            with self.subTest(actions=actions):
                clamped = read(C2_ACTIONS=actions, C2_TT_GRID='10,9')
                unclamped = read(C2_ACTIONS=actions)
                self.assertEqual(dict(clamped, tt_grid=''), unclamped)
        self.assertEqual(dict(read(C2_TT_GRID='12,9'), tt_grid=''), base)

    def test_main_prints_the_output_for_a_sample_job_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'job.env')
            for text, expected in (('C2_ACTIONS=status' + NL + 'C2_IMAGE_TAG=v7-grid' + NL + 'C2_TT_GRID=10,9' + NL, 'tt_grid=10,9'),
                                   ('C2_ACTIONS=status' + NL + 'C2_IMAGE_TAG=v7-grid' + NL, 'tt_grid='),
                                   ('C2_ACTIONS=status' + NL + 'C2_IMAGE_TAG=v7-grid' + NL + 'C2_TT_GRID=' + NL, 'tt_grid=')):
                with self.subTest(expected=expected):
                    with open(path, 'w') as handle:
                        handle.write(text)
                    with redirect_stdout(io.StringIO()) as out:
                        self.assertEqual(job.main([path]), 0)
                    self.assertIn(expected + NL, out.getvalue())
                    self.assertEqual(len([line for line in out.getvalue().splitlines() if line.startswith('tt_grid=')]), 1)
            with open(path, 'w') as handle:
                handle.write('C2_ACTIONS=status' + NL + 'C2_IMAGE_TAG=v7-grid' + NL + 'C2_TT_GRID=13,9' + NL)
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                self.assertEqual(job.main([path]), 1)
            self.assertEqual(out.getvalue(), '')
            self.assertIn('C2_TT_GRID must be one of 10,9, 11,9, 12,9 or unset', err.getvalue())

    def test_the_workflows_own_invocation_prints_it(self):
        """python3 -s scripts/ci/c2_serving_job.py <job file> >> $GITHUB_OUTPUT, on a sample job file."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'job.env')
            with open(path, 'w') as handle:
                handle.write('C2_ACTIONS=status' + NL + 'C2_IMAGE_TAG=v7-grid' + NL + 'C2_TT_GRID=11,9' + NL)
            result = subprocess.run([sys.executable, '-s', '-B', os.path.join(HERE, 'c2_serving_job.py'), path], capture_output=True,
                                    text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('tt_grid=11,9' + NL, result.stdout)


class DriverArgvTests(unittest.TestCase):
    """What the drivers launch: the pair when asked, byte for byte what they launched before when not."""

    def shape(self, **more):
        return gate.agent_shape('img', 'name', 'general', ['/dev/tenstorrent/1', '/dev/tenstorrent/0'], **more)

    def test_the_agent_shape_adds_the_pair_after_the_profile_and_nothing_without_it(self):
        plain = self.shape()
        self.assertEqual(clamp_count(plain), 0)
        self.assertEqual(self.shape(tt_grid=None), plain)
        self.assertEqual(self.shape(tt_grid=''), plain)
        for value in job.TT_GRIDS:
            with self.subTest(value=value):
                clamped = self.shape(tt_grid=value)
                self.assertEqual(pairs_in(clamped), [(PAIR, '%s=%s' % (ENV, value))])
                self.assertEqual(clamp_count(clamped), 1)
                at = clamped.index('%s=%s' % (ENV, value))
                self.assertEqual(clamped[at - 1], PAIR)
                self.assertEqual(clamped[:at - 1] + clamped[at + 1:], plain, 'the pair is the only difference')
                self.assertGreater(at, clamped.index('QWEN_C2_PROFILE=general'))
        with self.assertRaises(job.JobError):
            self.shape(tt_grid='13,9')

    def test_the_pair_comes_with_a_gate_only_profile_and_an_arms_own_environment_too(self):
        env = (('QWEN_FAST_EXTENT_AUDIT', '1'),)
        plain = self.shape(env=env, gate_only=True)
        clamped = self.shape(env=env, gate_only=True, tt_grid='10,9')
        self.assertIn('QWEN_C2_GATE=1', clamped)
        self.assertEqual(clamp_count(clamped), 1)
        at = clamped.index('%s=10,9' % ENV)
        self.assertEqual(clamped[:at - 1] + clamped[at + 1:], plain)

    def test_the_gate_run_and_the_prefix_server_run_carry_it(self):
        devices = ['/dev/tenstorrent/1', '/dev/tenstorrent/0']
        plain = gate.gate_run('img', 'name', 'general', devices, ROOT, '/tmp/arm', ['--x'])
        clamped = gate.gate_run('img', 'name', 'general', devices, ROOT, '/tmp/arm', ['--x'], tt_grid='10,9')
        self.assertEqual(clamp_count(plain), 0)
        self.assertEqual(pairs_in(clamped), [(PAIR, ENV + '=10,9')])
        self.assertLess(clamped.index(ENV + '=10,9'), clamped.index('--entrypoint'), 'before the image: docker would hand it to the entrypoint')
        plain = prefix_gate.server_run('img', 'name', 'general', devices)
        clamped = prefix_gate.server_run('img', 'name', 'general', devices, tt_grid='12,9')
        self.assertEqual(clamp_count(plain), 0)
        self.assertEqual(pairs_in(clamped), [(PAIR, ENV + '=12,9')])
        self.assertLess(clamped.index(ENV + '=12,9'), clamped.index('img'))
        self.assertEqual(prefix_gate.server_run('img', 'name', 'general', devices, tt_grid=None), plain)

    def test_the_gate_dry_run_shows_the_pair_in_every_arm_and_only_with_the_flag(self):
        results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, results, True)

        def dockers(*extra):
            lines = []
            code = gate.main(['--image', 'img', '--profiles', PROFILES_PATH, '--results', os.path.join(results, 'r'),
                              '--profile', 'general', '--plan', 'bringup,memory', '--dry-run'] + list(extra),
                             devices=['/dev/tenstorrent/1', '/dev/tenstorrent/0'], log=lines.append)
            self.assertEqual(code, 0, lines)
            return [json.loads(line)['docker'] for line in lines if line.startswith('{') and '"docker"' in line]
        plain = dockers()
        self.assertEqual(len(plain), 2)
        self.assertEqual([clamp_count(argv) for argv in plain], [0, 0])
        clamped = dockers('--tt-grid', '10,9')
        self.assertEqual([pairs_in(argv) for argv in clamped], [[(PAIR, ENV + '=10,9')]] * 2)
        for before, after in zip(plain, clamped):
            at = after.index(ENV + '=10,9')
            self.assertEqual(after[:at - 1] + after[at + 1:], before)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            dockers('--tt-grid', '13,9')

    def test_the_gate_runner_launches_it_and_the_summary_says_so(self):
        calls = []

        def execute(arguments, stdout_path, timeout, name):
            calls.append(arguments)
            with open(stdout_path, 'w') as handle:
                handle.write('')
            return 1

        with open(gate.REFERENCE, encoding='utf-8') as handle:
            corpus = json.load(handle)['real_text']['corpus']

        def run(*extra):
            del calls[:]
            with tempfile.TemporaryDirectory() as directory:
                lines = []
                results = os.path.join(directory, 'r')
                gate.main(['--image', 'img', '--profiles', PROFILES_PATH, '--results', results, '--profile', 'general',
                           '--plan', 'bringup'] + list(extra), execute=execute, devices=['/dev/tenstorrent/1', '/dev/tenstorrent/0'],
                          log=lines.append, containers=lambda: [], corpus=lambda: dict(corpus))
                with open(os.path.join(results, 'c2-gate-summary.json'), encoding='utf-8') as handle:
                    summary = json.load(handle)
                with open(os.path.join(results, 'bringup-concurrent', 'docker-run.json'), encoding='utf-8') as handle:
                    recorded = json.load(handle)
            return summary, lines, recorded
        summary, lines, recorded = run()
        self.assertTrue(calls)
        self.assertEqual([clamp_count(argv) for argv in calls], [0] * len(calls))
        self.assertNotIn('tt_grid', summary)
        self.assertEqual(clamp_count(recorded), 0)
        summary, lines, recorded = run('--tt-grid', '10,9')
        self.assertTrue(calls)
        self.assertEqual([pairs_in(argv) for argv in calls], [[(PAIR, ENV + '=10,9')]] * len(calls))
        self.assertEqual(summary['tt_grid'], '10,9')
        self.assertEqual(pairs_in(recorded), [(PAIR, ENV + '=10,9')], 'docker-run.json is the launched argv')
        self.assertTrue(any('compute-grid clamp: %s=10,9' % ENV in line for line in lines), lines)

    def test_the_prefix_gate_dry_run_and_its_runner_carry_it(self):
        def dockers(*extra):
            with tempfile.TemporaryDirectory() as directory:
                lines = []
                with open(os.path.join(directory, 'profiles.json'), 'w', encoding='utf-8') as handle:
                    json.dump(prefix_tests.profiles(), handle)
                code = prefix_gate.main(['--image', 'img', '--profiles', os.path.join(directory, 'profiles.json'),
                                         '--results', os.path.join(directory, 'out'), '--plan', 'bringup,exactness-eager',
                                         '--dry-run'] + list(extra), devices=['/a', '/b'], log=lines.append)
            self.assertEqual(code, 0, lines)
            return [json.loads(line)['docker'] for line in lines if line.startswith('{') and '"docker"' in line]
        plain = dockers()
        self.assertTrue(plain)
        self.assertEqual([clamp_count(argv) for argv in plain], [0] * len(plain))
        clamped = dockers('--tt-grid', '10,9')
        self.assertEqual(len(clamped), len(plain))
        self.assertEqual([pairs_in(argv) for argv in clamped], [[(PAIR, ENV + '=10,9')]] * len(plain))
        # The Runner itself: the arms' `docker run -d` as it launches them.
        for grid, expected in (('10,9', [(PAIR, ENV + '=10,9')]), (None, [])):
            with self.subTest(grid=grid), tempfile.TemporaryDirectory() as directory:
                harness = prefix_tests.Harness()
                runner = harness.runner(directory)
                self.assertIsNone(runner.tt_grid, 'unset by default')
                runner.tt_grid = grid
                arms = prefix_gate.plan_arms('bringup', 'general-prefix', 'general', prefix_tests.profiles())
                prefix_gate.run_plan('bringup', arms, runner, prefix_tests.GOOD_ANCHOR)
                launched = [call for call in harness.calls if call[:3] == ['docker', 'run', '-d']]
                self.assertTrue(launched)
                self.assertEqual([pairs_in(call) for call in launched], [expected] * len(launched))

    def test_the_prefix_main_hands_its_runner_the_clamp(self):
        seen = []

        def factory(*args, **kwargs):
            runner = prefix_tests.Harness().runner(tempfile.mkdtemp())
            seen.append(runner)
            return runner
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, 'profiles.json'), 'w', encoding='utf-8') as handle:
                json.dump(prefix_tests.profiles(), handle)
            for extra, expected in ((['--tt-grid', '11,9'], '11,9'), ([], None)):
                lines = []
                prefix_gate.main(['--image', 'img', '--profiles', os.path.join(directory, 'profiles.json'),
                                  '--results', os.path.join(directory, 'out%s' % len(seen)), '--plan', 'bringup'] + extra,
                                 devices=['/a', '/b'], log=lines.append, runner_factory=factory, anchor=prefix_tests.GOOD_ANCHOR)
                self.assertEqual(seen[-1].tt_grid, expected)
                with open(os.path.join(directory, 'out%s' % (len(seen) - 1), 'c2-prefix-summary.json'), encoding='utf-8') as handle:
                    summary = json.load(handle)
                self.assertEqual(summary.get('tt_grid'), expected)
                self.assertEqual(any('compute-grid clamp' in line for line in lines), expected is not None)
        for runner in seen:
            shutil.rmtree(runner.results, True)

    def test_the_tau_lab_dry_run_carries_it(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'data')
        os.makedirs(data)
        tau_tests.make_data(data)

        def argv(*extra):
            out = []
            code = lab.main(['--image', 'x/y:tp4-serve-2', '--data', data, '--results', os.path.join(root, 'r'), '--dry-run',
                             '--profiles', PROFILES_PATH] + list(extra), say=out.append, corpus=tau_tests.SmallCorpus.make())
            self.assertEqual(code, 0, out)
            return json.loads(out[-1]), out
        plain, out = argv()
        self.assertEqual(clamp_count(plain), 0)
        self.assertFalse(any('compute-grid clamp' in line for line in out))
        clamped, out = argv('--tt-grid', '10,9')
        self.assertEqual(pairs_in(clamped), [(PAIR, ENV + '=10,9')])
        self.assertEqual(clamped.index(ENV + '=10,9') > clamped.index('QWEN_C2_PROFILE=c2-packed-tp4+taulab'), True)
        at = clamped.index(ENV + '=10,9')
        self.assertEqual(clamped[:at - 1] + clamped[at + 1:], plain)
        self.assertTrue(any('compute-grid clamp: %s=10,9' % ENV in line for line in out))

    def test_every_driver_takes_the_flag_with_the_same_three_values(self):
        for parser in (gate.build_parser(), prefix_gate.build_parser(), lab.build_parser()):
            action = [a for a in parser._actions if a.dest == 'tt_grid']
            self.assertEqual(len(action), 1)
            self.assertEqual(tuple(action[0].choices), job.TT_GRIDS)
            self.assertIsNone(action[0].default)
        result = subprocess.run([sys.executable, '-s', '-B', os.path.join(HERE, 'c2_platform_replay.py'), '--help'], capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--tt-grid {10,9,11,9,12,9}', result.stdout)


class ReplayTests(unittest.TestCase):
    INFO = dict(Config=dict(Env=['THATCH_SERVING_SESSION_CAP=8', ENV + '=12,9', 'QWEN_C2_PROFILE=general']),
                HostConfig=dict(Tmpfs={}, Binds=[], CapAdd=[], Devices=[]), Mounts=[])

    def arguments(self, extra_env):
        return replay.run_arguments(self.INFO, 'img', 'copy', 8011, profile='general', devices=['/dev/tenstorrent/1'], extra_env=extra_env)

    def test_unset_leaves_the_recorded_environment_alone(self):
        self.assertEqual(replay.tt_grid_env(replay.DEFAULT_ENV, None), replay.DEFAULT_ENV)
        self.assertEqual(replay.tt_grid_env(replay.DEFAULT_ENV, ''), replay.DEFAULT_ENV)
        self.assertEqual(replay.tt_grid_env(('A=1', 'B=2'), None), ('A=1', 'B=2'))
        self.assertEqual(pairs_in(self.arguments(replay.DEFAULT_ENV)), [(PAIR, ENV + '=12,9')], 'the recorded value is the agent\'s, as before')

    def test_the_clamp_goes_last_and_replaces_the_recorded_and_any_given_value(self):
        env = replay.tt_grid_env(replay.DEFAULT_ENV, '10,9')
        self.assertEqual(env, replay.DEFAULT_ENV + (ENV + '=10,9',))
        self.assertEqual(pairs_in(self.arguments(env)), [(PAIR, ENV + '=10,9')], 'one pair, the clamp: the recorded 12,9 is dropped')
        env = replay.tt_grid_env(('A=1', ENV + '=11,9', 'B=2'), '10,9')
        self.assertEqual(env, ('A=1', 'B=2', ENV + '=10,9'))
        self.assertEqual(clamp_count(self.arguments(env)), 1)
        with self.assertRaises(job.JobError):
            replay.tt_grid_env(replay.DEFAULT_ENV, '13,9')

    def test_the_constants_are_the_job_parsers(self):
        self.assertIs(replay.c2_serving_job, job)


class CardMHarnessTests(unittest.TestCase):
    FORWARDING = ('c1e_gateup', 'draft_slide_inplace', 'k64j_probe', 'k64j', 'pair_row_probe', 'quad_draft_probe', 'sdpa_decode_slice')
    # The harnesses that print their docker argv on a dry run, and the variable that selects the dry run.
    DRY = (('sdpa_decode_slice', 'K64I_DRY_RUN'), ('pair_row_probe', 'PAIR_ROW_DRY_RUN'), ('k64j_probe', 'K64J_DRY_RUN'),
           ('k64j', 'K64J_CARD_DRY_RUN'), ('quad_draft_probe', 'QUAD_DRY_RUN'))

    def harness(self, directory):
        return 'optimisation/ttnn-op/%s/run_card_b.sh' % directory

    def text(self, relative):
        with open(os.path.join(ROOT, *relative.split('/')), encoding='utf-8') as handle:
            return handle.read()

    def harnesses_with_the_board_block(self):
        found = []
        for base, _, names in os.walk(OPS):
            for name in names:
                if name.endswith('.sh'):
                    path = os.path.join(base, name)
                    with open(path, encoding='utf-8') as handle:
                        if job.QUAL_CARD_BEGIN in handle.read():
                            found.append(os.path.relpath(path, ROOT).replace(os.sep, '/'))
        return sorted(found)

    def test_a_harness_is_accepted_with_the_key_only_if_it_forwards_the_step_variable(self):
        found = self.harnesses_with_the_board_block()
        self.assertGreater(len(found), 10)
        forwarding, refused = [], []
        for relative in found:
            values = dict(C2_CARDM_HARNESS=relative)
            if not job.CARDM_HARNESS.fullmatch(relative):
                continue
            self.assertEqual(job.read_cardm(values, True)[0], relative, 'without the key it is accepted as before')
            if job.CARDM_TT_GRID_VARIABLE in self.text(relative):
                forwarding.append(relative)
                self.assertEqual(job.read_cardm(values, True, tt_grid='10,9')[0], relative)
            else:
                refused.append(relative)
                with self.assertRaisesRegex(job.JobError, 'does not forward QUAL_TT_GRID into its container'):
                    job.read_cardm(values, True, tt_grid='10,9')
        self.assertEqual(sorted(forwarding), sorted(self.harness(d) for d in self.FORWARDING))
        self.assertTrue(refused, 'the harnesses that were not changed are refused, not silently run unclamped')

    def test_the_job_refuses_such_a_harness_with_the_key_set_and_accepts_it_without(self):
        unchanged = self.harness('v5split').replace('run_card_b.sh', 'run_card_m.sh')
        values = dict(C2_ACTIONS='cardm', C2_CARDM_HARNESS=unchanged)
        self.assertEqual(read(**values)['cardm_harness'], unchanged)
        with self.assertRaisesRegex(job.JobError, 'C2_TT_GRID is set but C2_CARDM_HARNESS'):
            read(C2_TT_GRID='10,9', **values)
        values = dict(C2_ACTIONS='cardm', C2_CARDM_HARNESS=self.harness('k64j'), C2_CARDM_ARGS='--sections K2')
        self.assertEqual(read(C2_TT_GRID='10,9', **values)['tt_grid'], '10,9')

    def test_the_step_owns_the_variable(self):
        for pair in ('QUAL_TT_GRID=10,9', 'QUAL_TT_GRID='):
            with self.assertRaises(job.JobError):
                read(C2_ACTIONS='cardm', C2_CARDM_HARNESS=self.harness('k64j'), C2_CARDM_ENV=pair)

    def passthrough_lines(self, relative):
        lines = [line for line in self.text(relative).split(NL) if '${QUAL_TT_GRID:+' in line]
        self.assertEqual(len(lines), 1, relative)
        return lines[0]

    def expand(self, line, grid):
        """The words a harness line gives its docker argv under bash -u, with QUAL_TT_GRID set to `grid` or unset."""
        words = re.sub(r' \\$', '', line.strip())
        script = 'set -u' + NL + ('QUAL_TT_GRID=%s' % shlex.quote(grid) + NL if grid is not None else '') + \
                 "show() { printf '%s\\n' \"$@\"; }" + NL + 'show x ' + words + ' y' + NL
        env = dict(os.environ)
        env.pop('QUAL_TT_GRID', None)
        result = subprocess.run([BASH, '--noprofile', '--norc', '-c', script], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.split(NL)[:-1]

    @unittest.skipIf(BASH is None, 'no bash')
    def test_every_forwarding_harness_gives_docker_the_pair_and_nothing_when_unset(self):
        for directory in self.FORWARDING:
            relative = self.harness(directory)
            with self.subTest(harness=relative):
                line = self.passthrough_lines(relative)
                self.assertEqual(self.expand(line, None), ['x', 'y'])
                self.assertEqual(self.expand(line, ''), ['x', 'y'])
                self.assertEqual(self.expand(line, '10,9'), ['x', '-e', ENV + '=10,9', 'y'])
                self.assertIn('docker run --rm', ' '.join(self.text(relative).split()))

    def invoke(self, directory, name, dry, grid, results):
        env = {'PATH': os.environ['PATH'], 'HOME': results, 'QUAL_CARD': CARD_M, 'ALLOW_SERVING_CARD': '1', 'RESULTS': results, dry: '1'}
        if grid is not None:
            env['QUAL_TT_GRID'] = grid
        return subprocess.run([BASH, os.path.join(OPS, name, 'run_card_b.sh')], env=env, capture_output=True, text=True,
                              timeout=120)

    @unittest.skipIf(BASH is None, 'no bash')
    def test_the_dry_runs_show_the_pair_in_the_launched_argv_and_nothing_else_changes(self):
        for name, dry in self.DRY:
            with self.subTest(harness=name), tempfile.TemporaryDirectory() as results:
                argvs = {}
                for grid in (None, '', '10,9'):
                    result = self.invoke(name, name, dry, grid, results)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    lines = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
                    self.assertEqual(len(lines), 1, result.stdout)
                    # The kernel-cache directory carries a time stamp: runs of six digits or more are masked.
                    argvs[grid] = [re.sub(r'\d{6,}', 'N', word) for word in shlex.split(lines[0][len('### argv: '):])]
                self.assertEqual(clamp_count(argvs[None]), 0)
                self.assertEqual(argvs[''], argvs[None])
                self.assertEqual(pairs_in(argvs['10,9']), [(PAIR, ENV + '=10,9')])
                at = argvs['10,9'].index(ENV + '=10,9')
                self.assertEqual(argvs['10,9'][:at - 1] + argvs['10,9'][at + 1:], argvs[None], 'the pair is the only difference')
                self.assertLess(at, argvs['10,9'].index('--entrypoint'))


def workflow_text():
    with open(WORKFLOW, encoding='utf-8') as handle:
        return handle.read()


def steps():
    """{step name: its text}, from each '      - name:' line to the next."""
    text = workflow_text()
    starts = [m.start() for m in re.finditer(r'^      - name: ', text, flags=re.M)] + [len(text)]
    found = {}
    for begin, end in zip(starts, starts[1:]):
        block = text[begin:end]
        found[block.split(NL, 1)[0][len('      - name: '):]] = block
    return found


def script_of(block):
    marker = '        run: |' + NL
    body = block[block.index(marker) + len(marker):]
    return NL.join(line[10:] if line.startswith(' ' * 10) else line for line in body.split(NL))


# The steps that start tt-metal on a card, and how each gives its container the clamp.
OPENING = {
    'Four-card fabric probe (all four cards, inside the image)': 'docker',
    'Run a qualification harness on card M': 'harness',
    'Smoke on cards M+A': 'docker',
    "Run the gate in the agent's container shape": 'driver',
    "Prefix-reuse gates in the agent's container shape": 'driver',
    'Tau lab (W-T1) on the four-card set': 'driver',
    'Smoke on the four-card set': 'docker',
    "Replay the node agent's serving sequence": 'driver',
}
DRIVER_SCRIPTS = ('c2_serving_gate.py', 'c2_prefix_gate.py', 'c2_tau_lab.py', 'c2_platform_replay.py')


def command_at(script, opener):
    """The shell command whose first line contains `opener`, continuation lines joined, up to the end of its last line."""
    lines = script.split(NL)
    for index, line in enumerate(lines):
        if opener in line and not line.lstrip().startswith('#'):
            collected = [line]
            while collected[-1].rstrip().endswith('\\'):
                index += 1
                collected.append(lines[index])
            return NL.join(collected)
    raise AssertionError('no line with %r' % opener)


class WorkflowTests(unittest.TestCase):
    def test_the_steps_that_open_a_card_are_the_steps_that_carry_the_clamp(self):
        found = steps()
        for name in OPENING:
            self.assertIn(name, found)
        wired = sorted(name for name, block in found.items() if 'TT_GRID: ${{ steps.job.outputs.tt_grid }}' in block)
        self.assertEqual(wired, sorted(OPENING))
        # No other step starts a container on a card or runs a driver that does: a new launch site that forgets the clamp lands here.
        for name, block in found.items():
            if name in OPENING or 'run: |' not in block:
                continue
            script = script_of(block)
            self.assertNotIn('"${devices[@]}"', script, name)
            self.assertNotRegex(script, r'docker run[^\n]*--device', name)
            for driver in DRIVER_SCRIPTS:
                self.assertNotIn(driver, script, name)
            self.assertNotIn('bash "$HARNESS"', script, name)

    def test_each_opening_step_hands_its_launch_the_clamp_by_its_own_means(self):
        found = steps()
        for name, how in OPENING.items():
            script = script_of(found[name])
            with self.subTest(step=name):
                if how == 'docker':
                    self.assertIn('${TT_GRID:+-e "%s=$TT_GRID"}' % ENV, script)
                    self.assertEqual(script.count('${TT_GRID:+-e "%s=$TT_GRID"}' % ENV), 1)
                elif how == 'harness':
                    self.assertIn('${TT_GRID:+QUAL_TT_GRID="$TT_GRID"} bash "$HARNESS"', script)
                elif name.startswith('Tau lab'):
                    self.assertIn('if [ -n "$TT_GRID" ]; then args+=(--tt-grid "$TT_GRID"); fi', script)
                else:
                    self.assertIn('${TT_GRID:+--tt-grid "$TT_GRID"}', script)
                    self.assertEqual(script.count('--tt-grid'), 1)

    def test_every_driver_launch_in_the_workflow_is_a_step_above(self):
        found = steps()
        for driver in DRIVER_SCRIPTS:
            users = [name for name, block in found.items() if 'scripts/ci/' + driver in block and 'run: |' in block]
            self.assertEqual(len(users), 1, (driver, users))
            self.assertEqual(OPENING[users[0]], 'driver')

    def run_bash(self, script, grid):
        env = {'PATH': os.environ['PATH']}
        if grid is not None:
            env['TT_GRID'] = grid
        result = subprocess.run([BASH, '--noprofile', '--norc', '-c', script], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.split(NL)[:-1]

    @unittest.skipIf(BASH is None, 'no bash')
    def test_the_docker_launches_hold_the_pair_only_when_the_variable_is_set(self):
        found = steps()
        for name, opener in (('Four-card fabric probe (all four cards, inside the image)', 'docker run --rm --name "$name" --read-only'),
                             ('Smoke on cards M+A', 'docker run -d --name "$name" --read-only'),
                             ('Smoke on the four-card set', 'docker run -d --name "$name" --read-only')):
            command = command_at(script_of(found[name]), opener)
            command = re.sub(r' 2>&1 \| tee.*$', '', command, flags=re.S)
            with self.subTest(step=name):
                script = ("docker() { printf '%s\\n' \"$@\"; }" + NL + 'timeout() { shift 3; "$@"; }' + NL +
                          'name=n; image=img; devices=(--device /dev/tenstorrent/1); kvmount=(); results=/r; PROFILE=p; FABRIC=FABRIC_1D' + NL +
                          'script=s.py; report=r.json; limit=1; m=/dev/tenstorrent/1; a=/dev/tenstorrent/0' + NL + command + NL)
                unset = self.run_bash(script, None)
                empty = self.run_bash(script, '')
                clamped = self.run_bash(script, '10,9')
                self.assertEqual(clamp_count(unset), 0)
                self.assertEqual(empty, unset, 'an empty output passes nothing')
                self.assertEqual(pairs_in(clamped), [(PAIR, ENV + '=10,9')])
                at = clamped.index(ENV + '=10,9')
                self.assertEqual(clamped[:at - 1] + clamped[at + 1:], unset, 'the pair is the only difference')
                self.assertLess(at, clamped.index('img'), 'before the image')

    @unittest.skipIf(BASH is None, 'no bash')
    def test_the_driver_launches_hold_the_flag_only_when_the_variable_is_set(self):
        found = steps()
        for name, opener in (("Run the gate in the agent's container shape", 'python3 scripts/ci/c2_serving_gate.py'),
                             ("Prefix-reuse gates in the agent's container shape", 'python3 scripts/ci/c2_prefix_gate.py'),
                             ("Replay the node agent's serving sequence", 'python3 scripts/ci/c2_platform_replay.py')):
            command = command_at(script_of(found[name]), opener)
            command = re.sub(r' 2>&1 \\\n? *\| tee.*$| \|\| replay_status=\$\?.*$', '', command, flags=re.S)
            command = command.replace('${replay_wrap[@]+"${replay_wrap[@]}"} ', '')
            with self.subTest(step=name):
                script = ("python3() { printf '%s\\n' \"$@\"; }" + NL + 'image=img; PROFILE=p; GATE_PLAN=bringup; GATE_MAX_TOKENS=1; budget=1' + NL +
                          'results=/r; CARDS=quad; PREFIX_PROFILE=pp; PREFIX_PLAN=bringup; PREFIX_BASELINE=none; PREFIX_AGENTS=1' + NL +
                          'source=s; IMAGE=img; box_left=' + NL + command + NL)
                unset = self.run_bash(script, None)
                empty = self.run_bash(script, '')
                clamped = self.run_bash(script, '10,9')
                self.assertEqual(unset.count('--tt-grid'), 0)
                self.assertEqual(empty, unset)
                self.assertEqual(clamped.count('--tt-grid'), 1)
                at = clamped.index('--tt-grid')
                self.assertEqual(clamped[at + 1], '10,9')
                self.assertEqual(clamped[:at] + clamped[at + 2:], unset, 'the flag is the only difference')

    @unittest.skipIf(BASH is None, 'no bash')
    def test_the_cardm_step_hands_the_harness_its_own_variable(self):
        script = script_of(steps()['Run a qualification harness on card M'])
        command = command_at(script, 'env ${extra[@]+"${extra[@]}"} QUAL_CARD=')
        command = re.sub(r' 2>&1 \| tee.*$', '', command, flags=re.S)
        with tempfile.TemporaryDirectory() as directory:
            fake = os.path.join(directory, 'harness.sh')
            with open(fake, 'w') as handle:
                handle.write('echo "QUAL_TT_GRID=${QUAL_TT_GRID-unset}"' + NL + 'echo "QUAL_CARD=$QUAL_CARD"' + NL)
            wrapper = ('extra=(WATCHER=1); results=%s; HARNESS=%s; HARNESS_ARGS=--sections; ' % (shlex.quote(directory), shlex.quote(fake)) + NL +
                       command + NL)
            for grid, expected in ((None, 'QUAL_TT_GRID=unset'), ('', 'QUAL_TT_GRID=unset'), ('10,9', 'QUAL_TT_GRID=10,9')):
                with self.subTest(grid=grid):
                    out = self.run_bash(wrapper, grid)
                    self.assertEqual(out[0], expected)
                    self.assertRegex(out[1], r'^QUAL_CARD=blackhole-[0-9A-F]+$')

    def test_the_header_documents_the_key(self):
        text = workflow_text()
        self.assertIn('C2_TT_GRID', text.split('on:' + NL, 1)[0])
        with open(os.path.join(HERE, 'c2_serving_job.py'), encoding='utf-8') as handle:
            self.assertIn('C2_TT_GRID', handle.read().split('Stdlib only', 1)[0])


class AllowlistTests(unittest.TestCase):
    def test_this_module_is_named_in_the_cpu_workflow(self):
        with open(CPU_WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        modules = set(token for line in re.findall(r'python -B -m unittest ([^\n]+)', text) for token in line.split())
        self.assertIn('test_c2_tt_grid', modules)
        self.assertIn('test_c2_serving_job', modules)


if __name__ == '__main__':
    unittest.main()
