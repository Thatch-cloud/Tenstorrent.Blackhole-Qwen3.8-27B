"""The C2 serving image's gates, run in the node agent's container shape (qwen-c2-serving.yml 'gate').

    python3 scripts/ci/c2_serving_gate.py --image zot.thatch.local:5000/tt-vllm:qwen38-c2-<tag> \
        --profile exact --plan bringup --results "$RUNNER_TEMP/c2-results/gate"

The fast path has never completed a request inside the serving image (c2-serve-for-real-plan 2.0 item
7): every C2 rate was measured in the m3native gate's own container (16 CPUs, bind mounts, the gate's
argv). This runs that gate's harness (lever_n_m3native_gate.py, mounted at /bench as
lever_n_m3native_run_arm.sh mounts it) INSIDE the serving image, shaped like the agent's container
(read-only root, the agent's tmpfs set, 8 CPUs, 80g, 4g shm, cards M then A by board id, /models =
~/hf-cache/hub, QWEN_C2_SERVING=1 and the job's profile, a p300 descriptor and
QWEN36_BATCHED_DECODE_MODE=host the contract must undo - held against the agent's own recorded
container, references/c2-serving/agent-container-36104200953.json), with the harness launching vLLM from the
platform's argv (--server-argv platform), so what serves is the contract's argv, not the gate's. The
one addition to the agent's shape is a bind mount for the gate's results (the agent's tmpfs cannot be
copied out). Each arm is one container; the image's engine skips tt-metal's teardown at exit, so the
next arm reopens the pair (the platform replay's docker stop/start proved it).

PLANS (C2_GATE_PLAN, run in order):
  bringup  4 users x 131072-token real-text prompts, 256 out, EOS on, 0.25 s apart (v235's shape),
           against the tracked v235 texts: IDENTICAL/DIVERGED per user (real_text_compare.
           reference_verdicts). Passes when all four are IDENTICAL and the harness's own verdict (every
           lever's marker, the packed rounds) holds. Gate table row Bring-up.
  matrix   real text at one prompt length per user (C2_GATE_LENGTHS, default the G4 ladder) with long
           answers (C2_GATE_MAX_TOKENS), EOS on: a concurrent arm (every user at once, 0.25 s apart;
           more users than seats queue) and a solo arm (the same prompts one at a time, the reference),
           under the exactness policy - a first divergence re-runs both arms once (real_text_compare.
           exactness_policy). Gate table row G4 part 1.
  memory   4 users x the profile's largest admitted prompt x its output ceiling, concurrent: every
           '[PINDIAG] dram after engine' line recorded, and the floor printed. Gate table row G5.
Every arm prints the contract's launched-argv line (read-the-launched-argv) and its DRAM lines.
Writes <results>/<arm>/ (the gate's stdout, report, server.log, prompts, docker argv) and
<results>/c2-gate-summary.json; exits 0 only when every plan passed.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_serving_job  # noqa: E402
import real_text_compare  # noqa: E402

CARD_M = 'blackhole-CEF5729692C19E6D'
CARD_A = 'blackhole-3707293C249A5E67'
HUB = '/home/thatch/hf-cache/hub'
SNAPSHOT = '/models/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
P300 = '/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto'
IMAGE_PROFILES = '/opt/qwen-c2/profiles.json'
SERVED_NAME = 'Qwen/Qwen3.8-27B'
RESULTS_IN_CONTAINER = '/gate-results'
# What the harness imports from scripts/ci, mounted at /bench over the image's (bundle) copies.
BENCH_SCRIPTS = ('lever_n_m3native_gate.py', 'longctx_cycle_bench.py', 'm3native_ttft_profile.py',
                 'acceptance_report.py', 'real_text_prompts.py')
REFERENCE = os.path.join(HERE, 'references', 'c2-serving', 'v235-real-text-4x131072.json')
BRINGUP_USERS, BRINGUP_PROMPT, BRINGUP_MAX_TOKENS = 4, 131072, 256
MEMORY_USERS = 4
STAGGER = 0.25            # v235's M3NATIVE_STAGGER: admission in user order
READINESS_SECONDS = 1800
ARM_SECONDS = dict(bringup=3600, matrix=7200, memory=5400)
STREAM_SECONDS = dict(bringup=900, matrix=3600, memory=3600)


# The agent's own container (references/c2-serving/agent-container-36104200953.json, the replay's copy of
# thatch-inference-Qwen-Qwen3.8-27B): its tmpfs set - wider than the smoke step's - and the environment
# the platform adds that is not the image's, with the p300 descriptor and the batched decode mode the
# contract must undo. The Thatch layer's own variables (THATCH_*, PYTHONPATH=/opt/thatch/py,
# VLLM_PLUGINS) belong to its runtime, which the base image does not carry.
AGENT_TMPFS = ('/opt/tt-metal/generated:rw,exec,nosuid,size=1g', '/root/.cache/flashinfer:rw,exec,nosuid,size=512m',
               '/root/.cache/huggingface/hub:rw,nosuid,size=64m', '/root/.cache/tt-metal-cache:rw,exec,nosuid,size=8g',
               '/root/.cache/ttnn:rw,exec,nosuid,size=1g', '/root/.cache/vllm:rw,nosuid,size=256m',
               '/root/.triton:rw,exec,nosuid,size=512m', '/tmp:rw,exec,nosuid,size=512m')
AGENT_ENV = ('HF_HOME=/models', 'HF_HUB_CACHE=/models', 'MESH_DEVICE=P300', 'QWEN36_BATCHED_DECODE_MODE=host',
             'VLLM_RPC_TIMEOUT=100000', 'TT_MESH_GRAPH_DESC_PATH=%s' % P300, 'TRITON_CACHE_DIR=/root/.triton',
             'DO_NOT_TRACK=1', 'VLLM_NO_USAGE_STATS=1', 'PYTHONDONTWRITEBYTECODE=1')


def agent_shape(image, name, profile, devices, hub=HUB):
    """`docker run` of the node agent's container, up to the image: read-only root, the agent's tmpfs
    set, 8 CPUs, 80g, 4g shm, the two cards in the order given, hugepages, SYS_NICE, the hub at
    /models, the environment the platform adds (AGENT_ENV), and QWEN_C2_SERVING=1 with the profile."""
    arguments = ['docker', 'run', '--rm', '--name', name, '--read-only']
    for tmpfs in AGENT_TMPFS:
        arguments += ['--tmpfs', tmpfs]
    arguments += ['--shm-size', '4g', '--memory', '80g', '--cpus', '8']
    for device in devices:
        arguments += ['--device', device]
    arguments += ['-v', '/dev/hugepages-1G:/dev/hugepages-1G', '--cap-add', 'SYS_NICE', '-v', '%s:/models' % hub]
    for variable in AGENT_ENV + ('QWEN_C2_SERVING=1', 'QWEN_C2_PROFILE=%s' % profile):
        arguments += ['-e', variable]
    return arguments


def gate_run(image, name, profile, devices, checkout, arm_dir, gate_args, hub=HUB):
    """The whole `docker run` of one arm: the agent's shape, the harness mounted read-only at /bench,
    the arm's results directory, and the harness as the entrypoint."""
    arguments = agent_shape(image, name, profile, devices, hub)
    for script in BENCH_SCRIPTS:
        arguments += ['--mount', 'type=bind,src=%s,dst=/bench/%s,readonly' % (
            os.path.join(checkout, 'scripts', 'ci', script), script)]
    arguments += ['--mount', 'type=bind,src=%s,dst=%s' % (arm_dir, RESULTS_IN_CONTAINER)]
    return arguments + ['--entrypoint', 'python3', image, '-B', '/bench/lever_n_m3native_gate.py'] + list(gate_args)


def profile_limits(profiles, name):
    """(max-model-len, output ceiling, largest admitted prompt) of a profile, as the contract computes
    the prompt limit today (serving_c2_contract.enforce_request: context less budget, or the profile's
    max_prompt_tokens when that is lower)."""
    profile = profiles['profiles'][name]
    context = int(profile['engine']['max-model-len'])
    ceiling = int(profile['env']['QWEN_FAST_OUTPUT_BUDGET'])
    room = context - ceiling
    if profile.get('max_prompt_tokens'):
        room = min(room, int(profile['max_prompt_tokens']))
    return context, ceiling, room


def common_args(profile, context, stream_seconds, readiness=READINESS_SECONDS):
    return ['--server-argv', 'platform', '--expect-profile', profile, '--served-model-name', SERVED_NAME,
            '--snapshot', SNAPSHOT, '--context', str(context), '--readiness-seconds', str(readiness),
            '--stream-timeout', str(stream_seconds), '--prompt-source', 'real-text', '--allow-missing-references',
            '--eos', 'stop', '--results', RESULTS_IN_CONTAINER]


def plan_arms(plan, profile, profiles, lengths=c2_serving_job.LADDER, max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS,
              memory_prompt=None):
    """The arms one plan runs, in order: [(arm name, harness arguments, docker timeout seconds)]."""
    context, ceiling, room = profile_limits(profiles, profile)
    if plan == 'bringup':
        args = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--users', str(BRINGUP_USERS), '--prompt-tokens', str(BRINGUP_PROMPT),
            '--max-tokens', str(BRINGUP_MAX_TOKENS), '--stagger', str(STAGGER)]
        return [('bringup-concurrent', args, ARM_SECONDS[plan])]
    if plan == 'matrix':
        text = ','.join(str(length) for length in lengths)
        base = common_args(profile, context, STREAM_SECONDS[plan]) + ['--prompt-lengths', text,
                                                                       '--max-tokens', str(max_tokens)]
        return [('matrix-concurrent', base + ['--users', str(len(lengths)), '--stagger', str(STAGGER)],
                 ARM_SECONDS[plan]),
                ('matrix-solo', base + ['--users', '1', '--sequential-users', str(len(lengths))], ARM_SECONDS[plan])]
    if plan == 'memory':
        prompt = memory_prompt or room
        args = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--users', str(MEMORY_USERS), '--prompt-lengths', ','.join([str(prompt)] * MEMORY_USERS),
            '--max-tokens', str(ceiling), '--stagger', str(STAGGER)]
        return [('memory-concurrent', args, ARM_SECONDS[plan])]
    raise ValueError('unknown plan %r' % plan)


def launched_argv_line(report):
    """The contract's launched argv as its own log line spells it, or None."""
    platform = (report or {}).get('platform') or {}
    if platform.get('served_argv') is None:
        return None
    return '[QWEN-C2] profile %s: vLLM argv %s' % (platform.get('served_profile'), json.dumps(platform['served_argv']))


def dram_lines(report):
    return [event.get('line') for event in ((report or {}).get('dram') or {}).get('events') or []]


def stream_problems(report):
    return list((report or {}).get('real_text_stream_problems') or [])


def bringup_verdict(report, reference):
    """IDENTICAL/DIVERGED per user against the reference, and PASS only when all four are IDENTICAL,
    the contract served the expected profile and the harness's own verdict held."""
    if report is None:
        return dict(verdict='FAIL', reason='no gate report', users=[])
    result = real_text_compare.reference_verdicts(report, reference)
    problems = list(((report.get('platform') or {}).get('problems')) or [])
    if report.get('fatal'):
        problems.append('fatal: %s' % report['fatal'])
    missing = ((report.get('flag_markers') or {}).get('missing')) or []
    passed = result['verdict'] == 'IDENTICAL' and not problems and bool(report.get('gate_passed'))
    return dict(verdict='PASS' if passed else 'FAIL', texts=result['verdict'], users=result['users'],
                arithmetic_diff=result['arithmetic_diff'], configuration_diff=result['configuration_diff'],
                gate_passed=report.get('gate_passed'), platform_problems=problems, missing_markers=missing,
                lines=real_text_compare.render_reference(result).split('\n'))


def memory_verdict(report, users=MEMORY_USERS):
    """G5 records, it does not judge a threshold: PASS when every stream completed and every user's
    engine logged its DRAM line."""
    if report is None:
        return dict(verdict='FAIL', reason='no gate report')
    dram = report.get('dram') or {}
    problems = stream_problems(report) + list(((report.get('platform') or {}).get('problems')) or [])
    if report.get('fatal'):
        problems.append('fatal: %s' % report['fatal'])
    if (dram.get('engines') or 0) < users:
        problems.append('%s dram-after-engine lines for %d users' % (dram.get('engines') or 0, users))
    return dict(verdict='FAIL' if problems else 'PASS', problems=problems, engines=dram.get('engines'),
                min_free_gb=dram.get('min_free_gb'), min_largest_free_mb=dram.get('min_largest_free_mb'),
                lines=dram_lines(report))


def matrix_verdict(concurrent, solo, rerun=None):
    """The exactness policy over the concurrent arm and its solo reference (and their re-run)."""
    if concurrent is None or solo is None:
        return dict(verdict='FAIL', reason='an arm left no gate report')
    result = real_text_compare.exactness_policy(concurrent, solo, rerun)
    problems = []
    for label, report in (('concurrent', concurrent), ('solo', solo)) + (
            (('concurrent re-run', rerun[0]), ('solo re-run', rerun[1])) if rerun else ()):
        problems += ['%s: %s' % (label, p) for p in ((report.get('platform') or {}).get('problems') or [])]
    verdict = result['verdict'] if not problems or result['verdict'] in ('FAIL', 'RERUN') else 'FAIL'
    return dict(verdict=verdict, policy=result, platform_problems=problems,
                lines=real_text_compare.render_policy(result).split('\n'))


def extract(stdout_text):
    try:
        return real_text_compare.extract_report(stdout_text)
    except ValueError:
        return None


class Runner(object):
    """Runs arms with docker and keeps their files; `execute` is injectable for the CPU tests."""

    def __init__(self, image, profile, results, checkout, devices, hub=HUB, execute=None, log=print):
        self.image, self.profile, self.results, self.checkout = image, profile, results, checkout
        self.devices, self.hub, self.log = devices, hub, log
        self.execute = execute or self._execute
        self.arms = {}

    @staticmethod
    def _execute(arguments, stdout_path, timeout, name):
        with open(stdout_path, 'w') as handle:
            try:
                return subprocess.run(arguments, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return 'timeout'

    def run(self, arm, gate_args, timeout):
        arm_dir = os.path.join(self.results, arm)
        os.makedirs(arm_dir, exist_ok=True)
        os.chmod(arm_dir, 0o777)
        name = 'qwen-c2-gate-%s' % arm
        arguments = gate_run(self.image, name, self.profile, self.devices, self.checkout, arm_dir, gate_args, self.hub)
        with open(os.path.join(arm_dir, 'docker-run.json'), 'w') as handle:
            json.dump(arguments, handle, indent=1)
        self.log('[C2-GATE] arm %s: %s' % (arm, ' '.join(gate_args)))
        started = time.time()
        status = self.execute(arguments, os.path.join(arm_dir, 'gate-stdout.log'), timeout, name)
        seconds = round(time.time() - started, 1)
        with open(os.path.join(arm_dir, 'gate-stdout.log'), errors='replace') as handle:
            report = extract(handle.read())
        if report is not None:
            with open(os.path.join(arm_dir, 'm3native-gate.json'), 'w') as handle:
                json.dump(report, handle, indent=2)
        line = launched_argv_line(report)
        self.log('[C2-GATE] arm %s: exit %s after %s s; launched: %s' % (arm, status, seconds, line or
                 'NO [QWEN-C2] argv line (%s)' % '; '.join(((report or {}).get('platform') or {}).get('problems')
                                                             or ['no report'])))
        for text in dram_lines(report):
            self.log('[C2-GATE] arm %s: %s' % (arm, text))
        for problem in stream_problems(report):
            self.log('[C2-GATE] arm %s: stream problem: %s' % (arm, problem))
        if report and report.get('fatal'):
            self.log('[C2-GATE] arm %s: fatal: %s' % (arm, report['fatal']))
        self.arms[arm] = dict(exit=status, seconds=seconds, launched=line, gate_passed=(report or {}).get('gate_passed'),
                              fatal=(report or {}).get('fatal'))
        return report


def run_plan(plan, runner, profiles, reference=None, lengths=c2_serving_job.LADDER,
             max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS, memory_prompt=None):
    arms = plan_arms(plan, runner.profile, profiles, lengths, max_tokens, memory_prompt)
    if plan == 'bringup':
        (arm, args, timeout), = arms
        result = bringup_verdict(runner.run(arm, args, timeout), reference)
    elif plan == 'memory':
        (arm, args, timeout), = arms
        result = memory_verdict(runner.run(arm, args, timeout))
    else:
        (c_arm, c_args, c_timeout), (s_arm, s_args, s_timeout) = arms
        concurrent, solo = runner.run(c_arm, c_args, c_timeout), runner.run(s_arm, s_args, s_timeout)
        result = matrix_verdict(concurrent, solo)
        if result['verdict'] == 'RERUN':
            runner.log('[C2-GATE] matrix: a first divergence - re-running both arms once (the exactness policy)')
            rerun = (runner.run(c_arm + '-rerun', c_args, c_timeout), runner.run(s_arm + '-rerun', s_args, s_timeout))
            result = matrix_verdict(concurrent, solo, rerun if None not in rerun else None)
            if None in rerun:
                result.update(verdict='FAIL', reason='a re-run arm left no gate report')
    for line in result.get('lines') or ():
        runner.log('[C2-GATE] %s %s' % (plan, line))
    runner.log('[C2-GATE] %s %s' % (plan, result['verdict']))
    return result


def serving_pair(root='/dev/tenstorrent/by-id'):
    """Cards M then A, resolved by board id now (minors renumber on every reset)."""
    devices = []
    for board in (CARD_M, CARD_A):
        path = os.path.realpath(os.path.join(root, board))
        if not os.path.exists(path) or path == os.path.join(root, board):
            raise RuntimeError('card %s has no device node under %s' % (board, root))
        devices.append(path)
    return devices


def image_profiles(image):
    """The profiles the image itself carries (not the checkout's: the image may predate it)."""
    output = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'cat', image,
                             IMAGE_PROFILES], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if output.returncode:
        raise RuntimeError('cannot read %s from %s: %s' % (IMAGE_PROFILES, image, output.stderr[-400:]))
    return json.loads(output.stdout.decode('utf-8'))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--plan', default='bringup', help='comma-separated: %s' % ', '.join(c2_serving_job.GATE_PLANS))
    parser.add_argument('--results', required=True)
    parser.add_argument('--lengths', default=None, help='matrix prompt lengths (default: the G4 ladder)')
    parser.add_argument('--max-tokens', type=int, default=c2_serving_job.DEFAULT_MAX_TOKENS)
    parser.add_argument('--memory-prompt', type=int, default=None)
    parser.add_argument('--reference', default=REFERENCE)
    parser.add_argument('--checkout', default=os.path.dirname(os.path.dirname(HERE)))
    parser.add_argument('--profiles', default=None, help='a profiles JSON instead of the image\'s own')
    parser.add_argument('--hub', default=HUB)
    parser.add_argument('--dry-run', action='store_true', help='print every arm\'s docker argv and run nothing')
    return parser


def main(argv=None, execute=None, devices=None, log=print):
    options = build_parser().parse_args(argv)
    plans = c2_serving_job.split_list(options.plan)
    unknown = sorted(set(plans) - set(c2_serving_job.GATE_PLANS))
    if not plans or unknown:
        log('unknown plan(s): %s' % ', '.join(unknown or ['(none)']))
        return 2
    lengths = [c2_serving_job.positive_int('--lengths', part) for part in c2_serving_job.split_list(options.lengths)] \
        if options.lengths else list(c2_serving_job.LADDER)
    if options.profiles:
        with open(options.profiles, encoding='utf-8') as handle:
            profiles = json.load(handle)
    else:
        profiles = image_profiles(options.image)
    if options.profile not in profiles['profiles']:
        log('profile %r is not in the image\'s profiles (%s)' % (options.profile, ', '.join(sorted(profiles['profiles']))))
        return 2
    with open(options.reference, encoding='utf-8') as handle:
        reference = json.load(handle)
    os.makedirs(options.results, exist_ok=True)
    if options.dry_run:
        for plan in plans:
            for arm, args, timeout in plan_arms(plan, options.profile, profiles, lengths, options.max_tokens,
                                                options.memory_prompt):
                log(json.dumps(dict(arm=arm, timeout=timeout, docker=gate_run(
                    options.image, 'qwen-c2-gate-' + arm, options.profile, devices or ['<M>', '<A>'],
                    options.checkout, os.path.join(options.results, arm), args, options.hub))))
        return 0
    runner = Runner(options.image, options.profile, options.results, options.checkout,
                    devices if devices is not None else serving_pair(), options.hub, execute, log)
    context, ceiling, room = profile_limits(profiles, options.profile)
    summary = dict(image=options.image, profile=options.profile, plans=plans, context=context,
                   output_ceiling=ceiling, largest_prompt=room, results={}, arms=runner.arms)
    for plan in plans:
        summary['results'][plan] = run_plan(plan, runner, profiles, reference, lengths, options.max_tokens,
                                            options.memory_prompt)
    summary['passed'] = all(result['verdict'] == 'PASS' for result in summary['results'].values())
    with open(os.path.join(options.results, 'c2-gate-summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
    log('C2_GATE profile=%s plans=%s passed=%s' % (options.profile, ','.join(plans), summary['passed']))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
