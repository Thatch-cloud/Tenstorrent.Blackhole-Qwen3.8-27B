"""c2_serving_job: the qwen-c2-serving.yml job file parse, held on CPU (stdlib only).

The cardm action's two steps are also run here as the runner would run them (bash; skipped without it),
against stand-ins for docker, sudo, fuser, readlink and a harness that records what it was given."""

import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_platform_replay as replay  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
JOB_FILE = os.path.join(ROOT, '.github', 'c2-serving-job.env')
K64J_RUNNER = os.path.join(ROOT, 'optimisation', 'ttnn-op', 'k64j', 'run_card_b.sh')
PROFILES = ['coding', 'exact', 'general']
CARD_M, CARD_A, CARD_B = 'blackhole-CEF5729692C19E6D', 'blackhole-3707293C249A5E67', 'blackhole-F36F768B9A5CAFA0'
K64J_SHA = '152951c1c0de5c9dfad2d62c295393a43b2ecf353965c55c709da7e539b975b7'
K64J_HARNESS = 'optimisation/ttnn-op/k64j/run_card_b.sh'
NL = chr(10)


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
        self.assertEqual((outputs['cardm_harness'], outputs['cardm_args'], outputs['cardm_env']), ('', '', ''))

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
        for key in ('cardm_harness', 'cardm_args', 'cardm_env'):
            self.assertIn(key, lines)

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


def workflow_text():
    with open(WORKFLOW, encoding='utf-8') as handle:
        return handle.read()


def step_text(name):
    """One step of the workflow, from its '- name:' line to the next step's."""
    text = workflow_text()
    start = text.index('      - name: ' + name + NL)
    end = text.find(NL + '      - ', start)
    return text[start:end + 1] if end >= 0 else text[start:]


def step_script(name):
    """The step's run block as the runner writes it to a file (its last key in every step here)."""
    step = step_text(name)
    marker = '        run: |' + NL
    return textwrap.dedent(step[step.index(marker) + len(marker):])


def holder_check(script):
    """The M+A holder check of a step: from its 'clear=0' through its refusal line."""
    start = script.index('clear=0' + NL)
    end = script.index(NL, script.index('if [ "$clear" != 1 ]', start)) + 1
    return script[start:end]


RUN_STEP = 'Run a qualification harness on card M'
COLLECT_STEP = 'Collect the card-M results'
CB2A_WATCHER = dict(C2_ACTIONS='cardm', C2_CARDM_HARNESS=K64J_HARNESS, C2_CARDM_ARGS='--sections K2,X7,Z',
                    C2_CARDM_ENV='WATCHER=1 KOPGRAFT64=/home/thatch/opgraft-K64j EXPECT_TTNNCPP_SHA256=' + K64J_SHA)
CB2A_FULL = dict(C2_ACTIONS='cardm', C2_CARDM_HARNESS=K64J_HARNESS,
                 C2_CARDM_ARGS='--sections K2,X7,Z --seeds 0,1,2,3,4 --variants normal,peaky --no-timing',
                 C2_CARDM_ENV='KOPGRAFT64=/home/thatch/opgraft-K64j EXPECT_TTNNCPP_SHA256=' + K64J_SHA)


class CardMParseTests(unittest.TestCase):
    """The cardm action's job keys: a qual_card harness under optimisation/ttnn-op, plain arguments, and an
    environment that can never pick the board, the results directory or the shell."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(job.QUAL_CARD_LIBRARY, encoding='utf-8') as handle:
            self.library = handle.read()

    def tearDown(self):
        self.tmp.cleanup()

    def harness(self, relative, text):
        path = os.path.join(self.root, *relative.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8', newline=NL) as handle:
            handle.write(text)
        return relative

    def fake(self, **values):
        return job.read_cardm(values, 'cardm' in values.get('C2_ACTIONS', ''), root=self.root)

    def test_the_cb2a_passes_parse_to_exactly_their_values(self):
        watcher, full = read(**CB2A_WATCHER), read(**CB2A_FULL)
        self.assertEqual((watcher['actions'], watcher['cardm_harness']), ('cardm', K64J_HARNESS))
        self.assertEqual(watcher['cardm_args'], '--sections K2,X7,Z')
        self.assertEqual(watcher['cardm_env'],
                         'WATCHER=1 KOPGRAFT64=/home/thatch/opgraft-K64j EXPECT_TTNNCPP_SHA256=' + K64J_SHA)
        self.assertEqual(full['cardm_args'], '--sections K2,X7,Z --seeds 0,1,2,3,4 --variants normal,peaky --no-timing')
        self.assertEqual(full['cardm_env'], 'KOPGRAFT64=/home/thatch/opgraft-K64j EXPECT_TTNNCPP_SHA256=' + K64J_SHA)
        self.assertEqual(read(C2_ACTIONS='reset cardm', C2_CARDM_HARNESS=K64J_HARNESS,
                              C2_CARDM_ARGS='  --sections   K2,X7,Z ')['cardm_args'], '--sections K2,X7,Z')

    def test_the_tracked_job_file_carries_the_watcher_pass_inertly(self):
        with open(JOB_FILE, encoding='utf-8') as handle:
            values = job.parse_env(handle.read())
        self.assertNotIn('cardm', job.split_list(values['C2_ACTIONS']), 'queue it by C2_ACTIONS, not by default')
        self.assertEqual({key: values[key] for key in CB2A_WATCHER if key != 'C2_ACTIONS'},
                         {key: value for key, value in CB2A_WATCHER.items() if key != 'C2_ACTIONS'})

    def test_cardm_needs_a_harness_and_the_keys_are_checked_whenever_set(self):
        with self.assertRaisesRegex(job.JobError, 'C2_CARDM_HARNESS must name'):
            read(C2_ACTIONS='status cardm')
        with self.assertRaises(job.JobError):
            read(C2_CARDM_HARNESS='scripts/ci/qual_card.sh')        # set without cardm: still refused
        with self.assertRaises(job.JobError):
            read(C2_CARDM_ENV='QUAL_CARD=' + CARD_B)
        self.assertEqual(read(C2_CARDM_HARNESS=K64J_HARNESS)['cardm_harness'], K64J_HARNESS)

    def test_real_qual_card_harnesses_are_accepted(self):
        for harness in (K64J_HARNESS, 'optimisation/ttnn-op/k64j_probe/run_card_b.sh',
                        'optimisation/ttnn-op/sdpa_decode_qwen/run_card_m.sh'):
            with self.subTest(harness=harness):
                self.assertEqual(read(C2_ACTIONS='cardm', C2_CARDM_HARNESS=harness)['cardm_harness'], harness)

    def test_a_harness_outside_optimisation_ttnn_op_or_not_a_script_is_refused(self):
        for harness in ('scripts/ci/verify-t1-g0-rig.sh', 'optimisation/ttnn-op/run.sh',
                        'optimisation/ttnn-op/../../scripts/ci/verify-t1-g0-rig.sh', 'optimisation/ttnn-op/k64j/../x.sh',
                        'optimisation/ttnn-op/.k64j/run_card_b.sh', 'optimisation/ttnn-op/k64j/k64j_card_b.py',
                        '/' + K64J_HARNESS, './' + K64J_HARNESS, K64J_HARNESS + ' x', 'optimisation/ttnn-op/k64j/$(id).sh',
                        'optimisation/ttnn-op/k64j/missing.sh', 'optimisation\\ttnn-op\\k64j\\run_card_b.sh'):
            with self.subTest(harness=harness), self.assertRaises(job.JobError):
                read(C2_ACTIONS='cardm', C2_CARDM_HARNESS=harness)

    def test_a_harness_must_select_its_board_by_the_canonical_block(self):
        good = '#!/usr/bin/env bash' + NL + self.library + 'qual_card_select' + NL + 'echo run' + NL
        self.assertEqual(self.fake(C2_CARDM_HARNESS=self.harness('optimisation/ttnn-op/a/good.sh', good))[0],
                         'optimisation/ttnn-op/a/good.sh')
        drifted = self.library.replace("QUAL_CARD_B=blackhole-F36F768B9A5CAFA0", "QUAL_CARD_B=blackhole-CEF5729692C19E6D")
        self.assertNotEqual(drifted, self.library)
        cases = {
            'none.sh': '#!/usr/bin/env bash' + NL + 'docker run --device /dev/tenstorrent/0 x' + NL,
            'named.sh': '# uses qual_card.sh' + NL + '. scripts/ci/qual_card.sh' + NL + 'qual_card_select' + NL,
            'drifted.sh': drifted + 'qual_card_select' + NL,
            'unselected.sh': self.library + 'QUAL_CARD=' + CARD_B + NL + 'qual_card_select' + NL,
            'twice.sh': self.library + 'qual_card_select' + NL + self.library + 'qual_card_select' + NL,
            'truncated.sh': self.library.rsplit(job.QUAL_CARD_END, 1)[0],
        }
        for name, text in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(job.JobError, 'canonical qual_card.sh block'):
                self.fake(C2_CARDM_HARNESS=self.harness('optimisation/ttnn-op/a/' + name, text))

    def test_a_harness_that_leaves_optimisation_ttnn_op_by_a_link_is_refused(self):
        good = self.library + 'qual_card_select' + NL
        outside = os.path.join(self.root, 'elsewhere')
        os.makedirs(outside)
        with open(os.path.join(outside, 'run.sh'), 'w', encoding='utf-8', newline=NL) as handle:
            handle.write(good)
        os.makedirs(os.path.join(self.root, 'optimisation', 'ttnn-op'))
        try:
            os.symlink(outside, os.path.join(self.root, 'optimisation', 'ttnn-op', 'linked'), target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest('no symlinks here')
        with self.assertRaisesRegex(job.JobError, 'not a file of this checkout'):
            self.fake(C2_CARDM_HARNESS='optimisation/ttnn-op/linked/run.sh')

    def test_the_environment_never_picks_the_board_the_results_or_the_shell(self):
        for pair in ('QUAL_CARD=' + CARD_B, 'QUAL_CARD=' + CARD_A, 'ALLOW_SERVING_CARD=0', 'RESULTS=/tmp/x',
                     'CARD_B_ARGS=--seeds', 'QUAL_BYID_ROOT=/tmp', 'QUAL_SERVING_CARDS=x', 'QUAL_TT_ROOT=/x',
                     'PATH=/tmp', 'HOME=/tmp', 'BASH_ENV=/tmp/x', 'BASHOPTS=x', 'ENV=/tmp/x', 'SHELLOPTS=xtrace',
                     'IFS=x', 'LD_PRELOAD=/tmp/x.so', 'LD_LIBRARY_PATH=/tmp', 'DOCKER_HOST=tcp://x:2375',
                     'DOCKER_CONFIG=/tmp', 'SUDO_ASKPASS=/tmp/x', 'PS4=x', 'CDPATH=/', 'GLOBIGNORE=x'):
            with self.subTest(pair=pair), self.assertRaisesRegex(job.JobError, 'may not set'):
                read(C2_ACTIONS='cardm', C2_CARDM_HARNESS=K64J_HARNESS, C2_CARDM_ENV='WATCHER=1 ' + pair)

    def test_the_environment_is_name_value_with_plain_values(self):
        for env in ('WATCHER', '=1', 'watcher=1', '1WATCHER=1', 'WATCHER-S=1', 'WATCHER=$(id)', 'WATCHER=`id`',
                    'WATCHER=1;id', 'IMAGE=a|b', "WATCHER='1'", 'WATCHER="1"', 'KOPGRAFT64=/tmp/*', 'WATCHER=1 WATCHER=1',
                    'X=a&b', 'X=a>b', 'X=~'):
            with self.subTest(env=env), self.assertRaises(job.JobError):
                read(C2_ACTIONS='cardm', C2_CARDM_HARNESS=K64J_HARNESS, C2_CARDM_ENV=env)
        outputs = read(C2_ACTIONS='cardm', C2_CARDM_HARNESS=K64J_HARNESS,
                       C2_CARDM_ENV='WATCHER=1  WATCHDOG_S=120 IMAGE=sha256:57cb K64J_CARD_DRY_RUN=1 EMPTY=')
        self.assertEqual(outputs['cardm_env'], 'WATCHER=1 WATCHDOG_S=120 IMAGE=sha256:57cb K64J_CARD_DRY_RUN=1 EMPTY=')

    def test_the_arguments_are_plain_words(self):
        for args in ('--sections K2 *', "--sections 'K2'", '--seeds $(id)', '--seeds 0;id', '--seeds 0|x',
                     '--x `id`', '--x a"b', '--x ?', '--x [ab]', '--x a&', '--x {a,b}', '--x ~'):
            with self.subTest(args=args), self.assertRaisesRegex(job.JobError, 'plain'):
                read(C2_ACTIONS='cardm', C2_CARDM_HARNESS=K64J_HARNESS, C2_CARDM_ARGS=args)
        self.assertEqual(read(C2_ACTIONS='cardm', C2_CARDM_HARNESS=K64J_HARNESS,
                              C2_CARDM_ARGS='--k2-sweep 232:263 --cb2-extents 2304,131328 --out=/results/x.json')
                         ['cardm_args'], '--k2-sweep 232:263 --cb2-extents 2304,131328 --out=/results/x.json')


class CardMWorkflowTests(unittest.TestCase):
    """The cardm steps as text: card M alone, after the gate's refusals, results uploaded."""

    def test_the_run_step_reads_the_parses_cardm_outputs(self):
        step = step_text(RUN_STEP)
        self.assertIn("if: contains(steps.job.outputs.actions, 'cardm')", step)
        for variable, key in (('HARNESS', 'cardm_harness'), ('HARNESS_ARGS', 'cardm_args'), ('HARNESS_ENV', 'cardm_env')):
            self.assertIn('%s: ${{ steps.job.outputs.%s }}' % (variable, key), step)

    def test_card_m_is_the_steps_own_choice_made_after_the_job_environment(self):
        script = step_script(RUN_STEP)
        launch = [line for line in script.splitlines() if 'bash "$HARNESS"' in line or line.startswith('env ')]
        self.assertEqual(len([line for line in script.splitlines() if line.startswith('env ')]), 1)
        env_line = script[script.index('env ${extra[@]+"${extra[@]}"}'):script.index('bash "$HARNESS"')]
        self.assertTrue(launch)
        for fixed in ('QUAL_CARD=%s' % CARD_M, 'ALLOW_SERVING_CARD=1', 'RESULTS="$results"', 'CARD_B_ARGS="$HARNESS_ARGS"'):
            self.assertIn(fixed, env_line)
            self.assertGreater(env_line.index(fixed), env_line.index('${extra[@]+"${extra[@]}"}'), 'the step\'s own wins')
        self.assertEqual(set(job.CARDM_STEP_ENV), {'QUAL_CARD', 'ALLOW_SERVING_CARD', 'RESULTS', 'CARD_B_ARGS'})

    def test_the_gates_refusals_come_before_the_harness(self):
        script = step_script(RUN_STEP)
        launch = script.index('bash "$HARNESS"')
        platform = script.index("grep -q '^thatch-inference-'")
        holders = script.index(holder_check(script))
        before = script.index('docker ps -q | sort > "$RUNNER_TEMP/cardm-containers-before.txt"')
        self.assertLess(platform, holders)
        self.assertLess(holders, before)
        self.assertLess(before, launch)
        self.assertIn('refusing to open card M under it" >&2; exit 1', script[platform:holders])

    def test_the_holder_check_is_the_gates(self):
        gate = holder_check(step_script('Run the gate in the agent\'s container shape'))
        mine = holder_check(step_script(RUN_STEP))
        self.assertEqual(mine, gate.replace('gate-holders.txt', 'cardm-holders.txt')
                         .replace('refusing the gate', 'refusing the card-M run'))
        self.assertIn('sudo -n fuser -v "$m" "$a"', mine)
        script = step_script(RUN_STEP)
        self.assertIn('m=$(readlink -f /dev/tenstorrent/by-id/%s)' % CARD_M, script)
        self.assertIn('a=$(readlink -f /dev/tenstorrent/by-id/%s)' % CARD_A, script)

    def test_card_a_and_card_b_are_never_opened_reset_or_mapped(self):
        steps = step_text(RUN_STEP) + step_text(COLLECT_STEP)
        self.assertNotIn(CARD_B, steps)
        self.assertNotIn(CARD_B.split('-')[1], steps)
        self.assertEqual(steps.count(CARD_A), 1, 'card A only in the holder check')
        for word in ('tt-smi -r', 'serving_pair', '--device', 'docker run', '/dev/tenstorrent/0', '/dev/tenstorrent/1'):
            self.assertNotIn(word, steps)

    def test_the_step_outlasts_the_harness_timeouts(self):
        with open(K64J_RUNNER, encoding='utf-8') as handle:
            runner = handle.read()
        worst = max(int(value) for value in re.findall(r'^\s*timeout_s=([0-9]+)', runner, flags=re.M))
        self.assertEqual(worst, 5400)
        minutes = int(re.search(r'timeout-minutes: ([0-9]+)', step_text(RUN_STEP)).group(1))
        self.assertGreaterEqual(minutes * 60, worst + 30 + 20 + 300, 'container timeout, kill, removal, checks')
        job_minutes = int(re.search(r'^    timeout-minutes: ([0-9]+)', workflow_text(), flags=re.M).group(1))
        self.assertLess(minutes, job_minutes)

    def test_the_results_are_collected_and_uploaded(self):
        step = step_text(COLLECT_STEP)
        self.assertIn("if: always() && contains(steps.job.outputs.actions, 'cardm')", step)
        self.assertIn('out="$RUNNER_TEMP/c2-results/cardm"', step)
        self.assertIn('results="/tmp/c2-cardm-results/$GITHUB_RUN_ID"', step)
        self.assertIn('results="/tmp/c2-cardm-results/$GITHUB_RUN_ID"', step_script(RUN_STEP))
        self.assertIn('kcache-*) continue', step)
        upload = step_text('Upload results')
        self.assertIn('if: always()', upload)
        self.assertIn('path: ${{ runner.temp }}/c2-results', upload)
        text = workflow_text()
        self.assertLess(text.index('- name: ' + RUN_STEP), text.index('- name: ' + COLLECT_STEP))
        self.assertLess(text.index('- name: ' + COLLECT_STEP), text.index('- name: G1 base drift'))

    def test_the_pair_concurrency_group_holds_the_whole_job(self):
        text = workflow_text()
        self.assertIn('concurrency:' + NL + '  group: qwen-two-p150a-exclusive' + NL + '  cancel-in-progress: false', text)

    def test_the_job_file_and_the_workflow_document_cardm(self):
        with open(JOB_FILE, encoding='utf-8') as handle:
            header = handle.read()
        for word in ('reset cardm drift', 'C2_CARDM_HARNESS', 'C2_CARDM_ARGS', 'C2_CARDM_ENV', CARD_M,
                     'ALLOW_SERVING_CARD=1', 'qwen-two-p150a-exclusive', '--sections K2,X7,Z --seeds 0,1,2,3,4'):
            self.assertIn(word, header)
        self.assertIn('#   cardm  - ', workflow_text())


try:
    from test_qual_card import find_bash  # noqa: E402
except ImportError:  # pragma: no cover - the module is beside this one
    find_bash = lambda: None  # noqa: E731
BASH = find_bash()

# Stand-ins for the rig: docker (names, ids, inspect, rm), readlink (by-id links are files naming a node),
# sudo (runs the command), fuser (a log; holders from FAKE_DIR/held) and sleep (none).
FAKES = NL.join([
    'docker() {',
    '  case "$1 $2" in',
    '    "ps -a") cat "$FAKE_DIR/names" ;;',
    '    "ps -q") cat "$FAKE_DIR/ids" ;;',
    '    "inspect -f") case $3 in *Devices*) cat "$FAKE_DIR/devices-$4" 2>/dev/null ;; *) cat "$FAKE_DIR/name-$4" ;; esac ;;',
    '    "rm -f") echo "$3" >> "$FAKE_DIR/removed" ;;',
    '    *) echo "unexpected docker $*" >&2; return 99 ;;',
    '  esac',
    '}',
    'readlink() { local p=${@: -1}; case $p in /dev/tenstorrent/by-id/*) cat "$FAKE_DIR/byid-${p##*/}" ;; '
    '*) printf "%s\\n" "$p" ;; esac; }',
    'sudo() { [ "$1" = -n ] && shift; "$@"; }',
    'fuser() { echo "$*" >> "$FAKE_DIR/fuser.log"; if [ -s "$FAKE_DIR/held" ]; then cat "$FAKE_DIR/held" >&2; return 0; fi; '
    'return 1; }',
    'sleep() { :; }',
    '',
])
# A harness that records what it was given, writes what run_card_b.sh writes, and may "start" containers.
HARNESS = NL.join([
    '#!/usr/bin/env bash',
    'set -euo pipefail',
    'for name in QUAL_CARD ALLOW_SERVING_CARD RESULTS CARD_B_ARGS WATCHER KOPGRAFT64; do',
    '  echo "$name=${!name-unset}"',
    'done > "$FAKE_DIR/harness-env"',
    'mkdir -p "$RESULTS/kcache-1" "$RESULTS/watcher-1"',
    'echo "{}" > "$RESULTS/card-1.json"',
    'echo "K64J_CARD k2_verdict=REDUCED-PASS" > "$RESULTS/card-1.log"',
    'echo cache > "$RESULTS/kcache-1/blob"',
    'echo clean > "$RESULTS/watcher-1/watcher.log"',
    'if [ -s "$FAKE_DIR/started" ]; then cat "$FAKE_DIR/started" >> "$FAKE_DIR/ids"; fi',
    'echo "harness ran"',
    'exit "${FAKE_EXIT:-0}"',
    '',
])
TT_SMI_ONLY = ('                     USER        PID ACCESS COMMAND' + NL
               + '/dev/tenstorrent/2:  root         99 F.... tt-smi' + NL)
PYTHON_HOLDER = '/dev/tenstorrent/1:  thatch     4242 F.... python3' + NL


@unittest.skipUnless(BASH, 'bash not found')
class CardMStepRunTests(unittest.TestCase):
    """The two cardm steps run as the runner runs them (bash -eo pipefail), on a fake rig: card B is node 0,
    card A node 1, card M node 2."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.run_id = 'test-cardm-%s' % uuid.uuid4().hex[:12]
        self.runner_temp = os.path.join(self.dir, 'runner-temp')
        os.makedirs(os.path.join(self.runner_temp, 'c2-results'))
        for card, node in ((CARD_B, '0'), (CARD_A, '1'), (CARD_M, '2')):
            self.write('byid-' + card, '/dev/tenstorrent/%s' % node + NL)
        self.write('names', 'qwen-something-else' + NL)
        self.write('ids', 'c0' + NL)
        self.write('name-c0', '/older' + NL)
        self.write('devices-c0', '/dev/tenstorrent/2' + NL)
        self.write('harness.sh', HARNESS)

    def tearDown(self):
        subprocess.run([BASH, '--noprofile', '--norc', '-c', 'rm -rf "/tmp/c2-cardm-results/$0"', self.run_id],
                       capture_output=True, timeout=60)
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.dir, name).replace(os.sep, '/')

    def write(self, name, text):
        with open(os.path.join(self.dir, name), 'w', encoding='utf-8', newline=NL) as handle:
            handle.write(text)

    def read(self, name):
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            return ''
        with open(path, encoding='utf-8') as handle:
            return handle.read()

    def run_step(self, name, after='', **env):
        self.write('step.sh', FAKES + step_script(name) + after)
        full = dict(os.environ)
        for variable in ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'RESULTS', 'CARD_B_ARGS', 'WATCHER', 'KOPGRAFT64', 'FAKE_EXIT'):
            full.pop(variable, None)
        full.update(FAKE_DIR=self.path(''), RUNNER_TEMP=self.runner_temp.replace(os.sep, '/'), GITHUB_RUN_ID=self.run_id,
                    HARNESS=self.path('harness.sh'), HARNESS_ARGS='--sections K2,X7,Z',
                    HARNESS_ENV='WATCHER=1 KOPGRAFT64=/home/thatch/opgraft-K64j')
        full.update(env)
        return subprocess.run([BASH, '--noprofile', '--norc', '-eo', 'pipefail', self.path('step.sh')], env=full,
                              cwd=self.dir, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)

    def test_a_platform_serving_container_refuses_before_the_cards_are_looked_at(self):
        self.write('names', 'qwen-something-else' + NL + 'thatch-inference-Qwen-Qwen3.8-27B' + NL)
        result = self.run_step(RUN_STEP)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('a platform serving container exists; refusing to open card M under it', result.stderr)
        self.assertEqual((self.read('harness-env'), self.read('fuser.log')), ('', ''))

    def test_anything_but_tt_smi_holding_m_or_a_refuses(self):
        self.write('held', TT_SMI_ONLY + PYTHON_HOLDER)
        result = self.run_step(RUN_STEP)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('M or A is held; refusing the card-M run', result.stderr)
        self.assertIn('python3', result.stderr)
        self.assertEqual(self.read('harness-env'), '')
        self.assertEqual(self.read('fuser.log').splitlines(), ['-v /dev/tenstorrent/2 /dev/tenstorrent/1'] * 6)

    def test_the_harness_runs_on_card_m_with_the_steps_own_card_and_results(self):
        self.write('held', TT_SMI_ONLY)                    # the telemetry exporter: ignored, as the gate ignores it
        # Past the parse (which refuses these), the step's own values still win.
        result = self.run_step(RUN_STEP, HARNESS_ENV='WATCHER=1 KOPGRAFT64=/g QUAL_CARD=%s ALLOW_SERVING_CARD=0 '
                               'RESULTS=/tmp/elsewhere CARD_B_ARGS=--seeds' % CARD_B)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = dict(line.split('=', 1) for line in self.read('harness-env').splitlines())
        self.assertEqual(seen, dict(QUAL_CARD=CARD_M, ALLOW_SERVING_CARD='1', RESULTS='/tmp/c2-cardm-results/' + self.run_id,
                                    CARD_B_ARGS='--sections K2,X7,Z', WATCHER='1', KOPGRAFT64='/g'))
        self.assertIn('harness ran', result.stdout)
        self.assertIn('card M /dev/tenstorrent/2, card A /dev/tenstorrent/1', result.stdout)
        self.assertEqual(self.read(os.path.join('runner-temp', 'cardm-containers-before.txt')), 'c0' + NL)

    def test_an_empty_environment_and_the_harness_status(self):
        result = self.run_step(RUN_STEP, HARNESS_ENV='', HARNESS_ARGS='')
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = dict(line.split('=', 1) for line in self.read('harness-env').splitlines())
        self.assertEqual((seen['WATCHER'], seen['CARD_B_ARGS'], seen['QUAL_CARD']), ('unset', '', CARD_M))
        result = self.run_step(RUN_STEP, FAKE_EXIT='3')
        self.assertEqual(result.returncode, 3, 'a hang (exit 3) fails the step')
        self.assertEqual(self.run_step(RUN_STEP, HARNESS='').returncode, 1, 'no harness, nothing runs')

    def test_the_collection_removes_only_new_containers_on_card_m_and_uploads_all_but_the_cache(self):
        # The harness "starts" c1 on card M (left behind), c2 on card B (a card-B job's), c3 a platform container
        # on the pair, c4 with no device; c0 was there before (on card M).
        for cid, name, devices in (('c1', '/qwen-k64j-card-card-m', '/dev/tenstorrent/2'),
                                   ('c2', '/qwen-card-b-other', '/dev/tenstorrent/0'),
                                   ('c3', '/thatch-inference-Qwen-Qwen3.8-27B', '/dev/tenstorrent/2' + NL + '/dev/tenstorrent/1'),
                                   ('c4', '/nodevice', '')):
            self.write('name-' + cid, name + NL)
            self.write('devices-' + cid, devices + NL if devices else '')
        self.write('started', 'c1' + NL + 'c2' + NL + 'c3' + NL + 'c4' + NL)
        self.assertEqual(self.run_step(RUN_STEP).returncode, 0)
        result = self.run_step(COLLECT_STEP, after=NL + 'test -e "$results" && echo LEFT=yes || echo LEFT=no' + NL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read('removed').split(), ['c1'])
        out = os.path.join(self.runner_temp, 'c2-results', 'cardm')
        self.assertEqual(sorted(os.listdir(out)), ['card-1.json', 'card-1.log', 'harness-console.log', 'watcher-1'])
        self.assertTrue(os.path.isfile(os.path.join(out, 'watcher-1', 'watcher.log')))
        with open(os.path.join(out, 'harness-console.log'), encoding='utf-8') as handle:
            self.assertIn('harness ran', handle.read())
        self.assertIn('LEFT=no', result.stdout)
        self.assertIn('K64J_CARD k2_verdict=REDUCED-PASS', result.stdout)

    def test_the_collection_prints_the_extent_readers_verdict_line_too(self):
        """run_card_b.sh K64J_HARNESS=extent_reader (CB2b) logs K64J_READER, not K64J_CARD: the step shows either."""
        self.write('harness.sh', HARNESS.replace(
            'echo cache', 'echo "K64J_READER verdict=PASS scope=full chips=1of2" > "$RESULTS/reader-1.log"' + NL
            + 'echo cache'))
        self.assertEqual(self.run_step(RUN_STEP).returncode, 0)
        result = self.run_step(COLLECT_STEP)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(NL + 'K64J_READER verdict=PASS scope=full chips=1of2' + NL, result.stdout)
        self.assertIn(NL + 'K64J_CARD k2_verdict=REDUCED-PASS' + NL, result.stdout)

    def test_the_collection_without_a_run_touches_nothing(self):
        result = self.run_step(COLLECT_STEP)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read('removed'), '')
        self.assertFalse(os.path.exists(os.path.join(self.runner_temp, 'c2-results', 'cardm')))


if __name__ == '__main__':
    unittest.main()
