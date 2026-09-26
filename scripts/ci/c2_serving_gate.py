"""The C2 serving image's gates, run in the node agent's container shape (qwen-c2-serving.yml 'gate').

    python3 scripts/ci/c2_serving_gate.py --image zot.thatch.local:5000/tt-vllm:qwen38-c2-<tag> \
        --profile exact --plan bringup --results "$RUNNER_TEMP/c2-results/gate" [--budget-seconds S]

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

PLANS (C2_GATE_PLAN, run in order). Every plan is checked against the IMAGE's profile before any
container starts (plan_arms): a prompt past what the contract admits (profile_limits, the contract's
own prompt_room: context less the output ceiling or less min_answer_tokens, lowered by
max_prompt_tokens), an answer budget past the ceiling or past what a prompt leaves of the context (the
contract would cut it silently), or a length below the compact framing's template
(real_text_prompts.COMPACT_MIN_TARGET) is refused - except the default ladder's top rung, which is
lowered to the profile's largest admitted prompt with a logged note.
  bringup  4 users x 131072-token real-text prompts, 256 out, EOS on, 0.25 s apart (v235's shape),
           against the tracked v235 texts: IDENTICAL/DIVERGED per user (real_text_compare.
           reference_verdicts). Passes when all four are IDENTICAL, the harness's own verdict (every
           lever's marker, the packed rounds) holds, every stream spent exactly what it asked, and -
           on the profiles v235's engine is (REFERENCE_PROFILES) - the launched argv is v235's engine
           (real_text_compare.served_engine_diff). First, in a throwaway container without devices,
           the image's real-text corpus (its vLLM source) is checked against v235's: another corpus
           builds other prompts, so the plan is NOT_COMPARABLE without spending M+A on it. Gate table
           row Bring-up.
  matrix   real text at one prompt length per user (C2_GATE_LENGTHS, default the G4 ladder) with long
           answers (C2_GATE_MAX_TOKENS), EOS on: a concurrent arm (every user at once, 0.25 s apart;
           more users than seats queue) and a solo arm (the same prompts one at a time, the reference),
           under the exactness policy - a first divergence re-runs both arms once (real_text_compare.
           exactness_policy). Any arm's stream problem, fatal error, a prompt built to another length
           than asked, a 'length' finish short of its budget, or a user whose full-draft acceptance
           collapsed (both arms share one engine, so a corruption common to both would otherwise
           match) FAILS it. Gate table row G4 part 1.
  memory   4 users x the profile's largest admitted prompt x what the contract leaves it (the output
           ceiling, or context less the prompt), concurrent; on a C2-any profile also memory-short, 4
           users x MEMORY_SHORT_PROMPT x the whole ceiling (every proposal bucket per drafter, R3):
           every '[PINDIAG] dram after engine' line and every per-request engine's proposal ladder
           recorded, and the floor printed. Gate table row G5.
  lifecycle  six users on four seats, two event arms (LIFECYCLE_EVENTS) and a solo reference:
           lifecycle-drops  user 0 (the first arrival) leaves while its engine builds, users 1-3 at 4, 3
                            and 2 live streams, user 4 takes user 0's freed seat, user 5 (ignore_eos)
                            user 1's;
           lifecycle-edges  user 0 (60,000 tokens, the first arrival) is cancelled 5 s into its
                            prefill, users 1-3 leave together once each has 60 chunks (a triple drop
                            in one round), user 4 asks max_tokens=1, user 5 takes a freed seat.
           The harness's StreamWatch times every drop against the server log's prefill/build markers
           and records the phase it hit; an event that missed its phase makes the arm NOT_EXERCISED.
           What each event leaves must agree with the solo text (real_text_compare.lifecycle_pass,
           under the policy). Then, per arm, as many requests at once as the profile has seats must all
           answer (slots were released), and the ledger's idle allocation after every stream, less the
           one before the first, must match the solo arm's to LEDGER_FLAT_GB per chip (the ledger stays
           flat). Gate table row Lifecycle.
Every arm prints the contract's launched-argv line (read-the-launched-argv) and its DRAM lines. Under
a profile that turns C2-any on (QWEN_FAST_ANY_REQUEST=1) every arm that served must show the D2
quarantine consumer's live line (QUARANTINE_LIVE) in its server log, and under any other profile none
of the C2-any markers may appear (ANY_REQUEST_MARKERS): either is a problem that fails the arm.

SAFETY. Before each arm: no thatch-inference-* container may exist (a placement reloading onto M+A
mid-gate), and a leftover gate container of the same name is removed; after each arm, however it
ended (a SIGTERM from a cancelled or timed-out step included), its container is removed. An arm
whose server log shows tt-metal's ethernet-core wedge (llrt.cpp:594, the reason the m3native gate
resets before every run) makes its plan INFRA - not a fast-path FAIL - and the arms after it are
skipped. --budget-seconds (the workflow passes what is left of its step and job) refuses a plan list
whose worst case (every arm to its limit, every re-run) does not fit, before anything runs.

Writes <results>/<arm>/ (the gate's stdout, report, server.log, prompts, docker argv) and
<results>/c2-gate-summary.json; exits 0 only when every plan passed, 2 when refused up front.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_serving_job  # noqa: E402
import real_text_compare  # noqa: E402
import real_text_prompts  # noqa: E402
import serving_c2_contract  # noqa: E402  (stdlib only at import: json, os, sys)

CARD_M = 'blackhole-CEF5729692C19E6D'
CARD_A = 'blackhole-3707293C249A5E67'
HUB = '/home/thatch/hf-cache/hub'
SNAPSHOT = '/models/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
P300 = '/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto'
IMAGE_PROFILES = '/opt/qwen-c2/profiles.json'
SERVED_NAME = 'Qwen/Qwen3.8-27B'
RESULTS_IN_CONTAINER = '/gate-results'
CONTAINER_PREFIX = 'qwen-c2-gate-'
PLATFORM_PREFIX = 'thatch-inference-'
# What the harness imports from scripts/ci, mounted at /bench over the image's (bundle) copies.
BENCH_SCRIPTS = ('lever_n_m3native_gate.py', 'longctx_cycle_bench.py', 'm3native_ttft_profile.py',
                 'acceptance_report.py', 'real_text_prompts.py')
REFERENCE = os.path.join(HERE, 'references', 'c2-serving', 'v235-real-text-4x131072.json')
# v235's served engine is what these profiles rewrite the platform argv to (test_real_text_gate; c2-gate
# is exact's engine with c2's environment, test_serving_c2_contract): on them a launched argv that is
# not v235's engine fails the bring-up; on any other it is reported.
REFERENCE_PROFILES = ('exact', 'c2-gate')
BRINGUP_USERS, BRINGUP_PROMPT, BRINGUP_MAX_TOKENS = 4, 131072, 256
MEMORY_USERS = 4
# G5's short-prompt arm, on a profile that takes any request (ANY_REQUEST_FLAG): four users with prompts
# this short and the profile's whole output ceiling. Each drafter then captures every proposal bucket
# (256..2048: one per rung from min(position, 2048) to min(2048, position + budget), serving_request_factory),
# the per-request worst case the large-prompt arm never reaches (R3).
MEMORY_SHORT_PROMPT = 60
# C2-any (QWEN_FAST_ANY_REQUEST=1, the c2 profiles). The D2 quarantine's consumer logs
# QUARANTINE_LIVE_PREFIX + the running scheduler's class from inside the wrapper, on the engine's first
# step (QUARANTINE_LIVE on the TT platform): a consumer on a class the engine never builds is silent
# until a refusal strands a request (serving_request_quarantine, memory graft-mounted-is-not-graft-
# executed), so every arm under the switch must show it. A subclass of the wrapped class still runs the
# inherited wrapper, so any class counts; the class is recorded and one other than TTScheduler is noted.
# Every per-request
# engine logs its build with its proposal ladder (ANY_REQUEST_ENGINE; R3, G5), recorded per arm. None of
# ANY_REQUEST_MARKERS may appear under a profile that leaves the switch off: exact runs v235's path.
ANY_REQUEST_FLAG = 'QWEN_FAST_ANY_REQUEST'
QUARANTINE_LIVE_PREFIX = '[PINDIAG] request quarantine consumer live in '
QUARANTINE_LIVE = QUARANTINE_LIVE_PREFIX + 'TTScheduler'
ANY_REQUEST_ENGINE = '[PINDIAG] any-request engine for '
ANY_REQUEST_MARKERS = ('[PINDIAG] request quarantine', ANY_REQUEST_ENGINE, '[PINDIAG] attach source check')
ANY_REQUEST_LINES_KEPT = 32
STAGGER = 0.25            # v235's M3NATIVE_STAGGER: admission in user order
READINESS_SECONDS = 1800
ARM_SECONDS = dict(bringup=3600, matrix=5400, memory=5400, lifecycle=3300)
STREAM_SECONDS = dict(bringup=900, matrix=3600, memory=3600, lifecycle=1800)
ARM_OVERHEAD_SECONDS = 120   # docker start and removal, the profile read, per arm
RERUN_PLANS = ('matrix', 'lifecycle')   # a first divergence runs every arm of these once more
# Lifecycle (gate table row Lifecycle): six users on four seats, so users 4 and 5 are the 5th and 6th
# requests and take seats others free. Two event arms and one solo reference build the same prompts;
# user 0 is the first arrival and long enough that 5 s into its prefill is well inside it (a 60k-token
# prefill takes tens of seconds). --drops grammar: lever_n_m3native_gate.DROP_GRAMMAR.
LIFECYCLE_LENGTHS = (60000, 4096, 8192, 2049, 16384, 255)
LIFECYCLE_MAX_TOKENS = 2048
LIFECYCLE_EVENTS = (
    ('lifecycle-drops', ['--drops', '0:build,1:live=4,2:live=3,3:live=2', '--user-ignore-eos', '5']),
    ('lifecycle-edges', ['--drops', '0:prefill+5,1+2+3:60', '--user-max-tokens', '4:1']),
)
# The phase each drop kind must hit for its event to count (the harness's report['lifecycle']).
EVENT_PHASES = dict(chunks=('decode',), live=('decode',), barrier=('decode',), prefill=('prefill',),
                    build=('build',), seconds=('queued', 'prefill', 'prefill-or-build', 'build'))
BARRIER_SPREAD_SECONDS = 0.5     # a multiple drop's closes, first to last
LEDGER_FLAT_GB = 0.064           # idle drift beyond the solo arm's, per chip (the ledger's own cap, 64 MiB)
# A user whose full-draft rounds emit this few tokens on average has lost its target: real coding text
# accepts 2.76-4.0 per step (c2-serve-for-real-plan 2.2), a corrupt target about one (the bonus token).
MIN_ACCEPTANCE_MEAN, MIN_ACCEPTANCE_ROUNDS = 1.5, 50
WEDGE = 'Timed out while waiting for active ethernet core'
VERDICT_ORDER = ('FAIL', 'INFRA', 'RERUN', 'NOT_COMPARABLE', 'UNSTABLE', 'NOT_EXERCISED')


class PlanError(ValueError):
    """A plan this profile cannot serve as asked; refused before any container starts."""


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
    """(max-model-len, output ceiling, largest admitted prompt) of a profile, the prompt limit computed
    by the contract's own functions (serving_c2_contract.request_limits and prompt_room, which
    enforce_request uses): context less the output ceiling - or less min_answer_tokens where the profile
    names it (c2: 123,136; c2-gate: 131,072) - lowered by max_prompt_tokens. test_c2_serving_gate holds
    this against enforce_request itself on every profile of the checkout."""
    profile = profiles['profiles'][name]
    context = int(profile['engine']['max-model-len'])
    limits = serving_c2_contract.request_limits(profile)
    room = serving_c2_contract.prompt_room(context, limits['budget'], limits['max_prompt_tokens'],
                                           limits['min_answer_tokens'])
    return context, limits['budget'], room


def profile_seats(profiles, name):
    return int(profiles['profiles'][name]['engine'].get('max-num-seqs') or 1)


def any_request_profile(profiles, name):
    """Whether the profile turns C2-any on (serving_fast_policy.any_request_enabled's switch)."""
    return str((profiles['profiles'][name].get('env') or {}).get(ANY_REQUEST_FLAG, '0')) == '1'


def answer_room(context, lengths, max_tokens):
    """The largest max_tokens every prompt in `lengths` can be served in full: the contract clamps a
    request to context less its prompt, and a 'length' finish short of what the harness asked reads as
    a budget cut (request_problems)."""
    return min([max_tokens] + [context - length for length in lengths])


def common_args(profile, context, stream_seconds, readiness=READINESS_SECONDS):
    return ['--server-argv', 'platform', '--expect-profile', profile, '--served-model-name', SERVED_NAME,
            '--snapshot', SNAPSHOT, '--context', str(context), '--readiness-seconds', str(readiness),
            '--stream-timeout', str(stream_seconds), '--prompt-source', 'real-text', '--allow-missing-references',
            '--eos', 'stop', '--results', RESULTS_IN_CONTAINER]


def check_lengths(profile, lengths, room, what):
    short = [length for length in lengths if length < real_text_prompts.COMPACT_MIN_TARGET]
    if short:
        raise PlanError('%s %s: below %d tokens the compact framing\'s template leaves no room for code '
                        '(real_text_prompts.COMPACT_TEMPLATE_TOKENS)' % (what, short, real_text_prompts.COMPACT_MIN_TARGET))
    long_ = [length for length in lengths if length > room]
    if long_:
        raise PlanError('%s %s exceed %d, the largest prompt profile %s admits today (profile_limits): its edge '
                        'would answer 400' % (what, long_, room, profile))


def check_budget(profile, max_tokens, ceiling, what):
    if max_tokens > ceiling:
        raise PlanError('%s %d exceeds profile %s\'s output ceiling %d: the contract would cut every answer to %d '
                        'without a word' % (what, max_tokens, profile, ceiling, ceiling))


def plan_arms(plan, profile, profiles, lengths=None, max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS,
              memory_prompt=None, notes=None):
    """The arms one plan runs, in order: [(arm name, harness arguments, docker timeout seconds)].
    Refuses (PlanError) what the profile cannot serve as asked; `lengths` None is the default ladder,
    whose rungs past the profile's largest admitted prompt are lowered to it (noted in `notes`)."""
    context, ceiling, room = profile_limits(profiles, profile)
    notes = notes if notes is not None else []
    if plan == 'bringup':
        args = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--users', str(BRINGUP_USERS), '--prompt-tokens', str(BRINGUP_PROMPT),
            '--max-tokens', str(BRINGUP_MAX_TOKENS), '--stagger', str(STAGGER)]
        return [('bringup-concurrent', args, ARM_SECONDS[plan])]
    if plan == 'matrix':
        if lengths is None:
            lengths = [min(length, room) for length in c2_serving_job.LADDER]
            lowered = [length for length in c2_serving_job.LADDER if length > room]
            if lowered:
                notes.append('matrix: the default ladder\'s %s lowered to %d, the largest prompt profile %s admits'
                             % (lowered, room, profile))
        lengths = list(lengths)
        check_lengths(profile, lengths, room, 'matrix prompt lengths')
        check_budget(profile, max_tokens, ceiling, 'matrix --max-tokens')
        fits = answer_room(context, lengths, max_tokens)
        if fits < max_tokens:
            raise PlanError('matrix --max-tokens %d: a %d-token prompt leaves %d tokens of profile %s\'s context %d, '
                            'so the contract would cut that answer to %d and its \'length\' finish would read as a '
                            'budget cut' % (max_tokens, context - fits, fits, profile, context, fits))
        text = ','.join(str(length) for length in lengths)
        base = common_args(profile, context, STREAM_SECONDS[plan]) + ['--prompt-lengths', text,
                                                                       '--max-tokens', str(max_tokens)]
        return [('matrix-concurrent', base + ['--users', str(len(lengths)), '--stagger', str(STAGGER)],
                 ARM_SECONDS[plan]),
                ('matrix-solo', base + ['--users', '1', '--sequential-users', str(len(lengths))], ARM_SECONDS[plan])]
    if plan == 'memory':
        # The largest prompt gets what the contract leaves it: the ceiling, or context less the prompt
        # when that is smaller (c2: 123,136 + 8,192), so every stream can run to its budget.
        prompt = memory_prompt or room
        check_lengths(profile, [prompt], room, 'memory prompt')
        args = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--users', str(MEMORY_USERS), '--prompt-lengths', ','.join([str(prompt)] * MEMORY_USERS),
            '--max-tokens', str(answer_room(context, [prompt], ceiling)), '--stagger', str(STAGGER)]
        arms = [('memory-concurrent', args, ARM_SECONDS[plan])]
        if any_request_profile(profiles, profile):
            short = [MEMORY_SHORT_PROMPT] * MEMORY_USERS
            check_lengths(profile, short, room, 'memory short prompts')
            arms.append(('memory-short', common_args(profile, context, STREAM_SECONDS[plan]) + [
                '--users', str(MEMORY_USERS), '--prompt-lengths', ','.join(str(length) for length in short),
                '--max-tokens', str(answer_room(context, short, ceiling)), '--stagger', str(STAGGER)],
                ARM_SECONDS[plan]))
        return arms
    if plan == 'lifecycle':
        check_lengths(profile, LIFECYCLE_LENGTHS, room, 'lifecycle prompt lengths')
        check_budget(profile, LIFECYCLE_MAX_TOKENS, ceiling, 'lifecycle --max-tokens')
        base = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--prompt-lengths', ','.join(str(length) for length in LIFECYCLE_LENGTHS),
            '--max-tokens', str(LIFECYCLE_MAX_TOKENS)]
        users = str(len(LIFECYCLE_LENGTHS))
        seats = str(profile_seats(profiles, profile))
        arms = [(arm, base + ['--users', users, '--stagger', str(STAGGER), '--alive-check', seats] + events,
                 ARM_SECONDS[plan]) for arm, events in LIFECYCLE_EVENTS]
        return arms + [('lifecycle-solo', base + ['--users', '1', '--sequential-users', users, '--alive-check', '1'],
                        ARM_SECONDS[plan])]
    raise ValueError('unknown plan %r' % plan)


def worst_case_seconds(plans, arms_of):
    """Every arm to its docker limit, every re-run a first divergence can ask for, plus per-arm overhead."""
    total = 0
    for plan in plans:
        for _, _, timeout in arms_of[plan]:
            runs = 2 if plan in RERUN_PLANS else 1
            total += runs * (timeout + ARM_OVERHEAD_SECONDS)
    return total


def asked(args):
    """What one arm's harness arguments ask for: streams, a prompt length and a budget per user."""
    def value(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default
    streams = int(value('--sequential-users', 0) or 0) or int(value('--users', 4))
    if '--prompt-lengths' in args:
        lengths = [int(part) for part in value('--prompt-lengths').split(',')]
    else:
        lengths = [int(value('--prompt-tokens', 32768))] * streams
    max_tokens = int(value('--max-tokens', 256))
    budgets = [max_tokens] * streams
    for part in (value('--user-max-tokens') or '').split(','):
        if part.strip():
            user, _, budget = part.partition(':')
            budgets[int(user)] = int(budget)
    return dict(streams=streams, lengths=lengths, max_tokens=max_tokens, budgets=budgets)


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


def request_problems(report, want):
    """What the arm served against what it asked (`want`, from asked()): the prompt lengths built, the
    stream count, the answer budget, and every stream that neither errored nor was dropped: its
    completion_tokens reported, never past its budget, and exactly its budget when it ended 'length' -
    a shorter 'length' is a budget cut somewhere between the request and the engine."""
    problems = []
    built = ((report or {}).get('real_text') or {}).get('prompt_lengths')
    if built is not None and list(built) != list(want['lengths']):
        problems.append('prompts built at %s tokens, asked %s' % (built, want['lengths']))
    if report.get('max_tokens') is not None and report['max_tokens'] != want['max_tokens']:
        problems.append('the harness ran max_tokens %s, asked %d' % (report['max_tokens'], want['max_tokens']))
    streams = report.get('streams') or []
    if len(streams) != want['streams']:
        problems.append('%d streams, asked %d' % (len(streams), want['streams']))
    for index, stream in enumerate(streams):
        stream = stream or {}
        if stream.get('error') or stream.get('dropped'):
            continue
        budget = want['budgets'][index] if index < len(want['budgets']) else want['max_tokens']
        finish, count = stream.get('finish_reason'), stream.get('completion_tokens')
        if count is None:
            problems.append('user %d: no completion_tokens reported' % index)
        elif count > budget:
            problems.append('user %d: %d completion tokens, past its %d budget' % (index, count, budget))
        elif finish == 'length' and count != budget:
            problems.append('user %d: ended "length" after %d of its %d tokens: its budget was cut on the way to '
                            'the engine' % (index, count, budget))
    return problems


def acceptance_problems(report):
    """Users whose full-draft rounds collapsed (MIN_ACCEPTANCE_MEAN over at least MIN_ACCEPTANCE_ROUNDS)."""
    problems = []
    for entry in ((report or {}).get('acceptance') or {}).get('users') or []:
        full = entry.get('full_draft') or {}
        if (full.get('rounds') or 0) >= MIN_ACCEPTANCE_ROUNDS and full.get('mean_emitted') is not None \
                and full['mean_emitted'] < MIN_ACCEPTANCE_MEAN:
            problems.append('user %s: full-draft rounds emit %.2f tokens on average over %d rounds (floor %.1f): '
                            'a target that lost its arithmetic' % (entry.get('user'), full['mean_emitted'],
                                                                   full['rounds'], MIN_ACCEPTANCE_MEAN))
    return problems


def arm_problems(label, report, want=None, acceptance=False):
    """Everything besides the texts that fails an arm: the contract's problems, stream problems, a fatal
    error, what it served against what it asked, and (the matrix) a collapsed acceptance."""
    problems = ['%s: %s' % (label, p) for p in ((report.get('platform') or {}).get('problems') or [])]
    problems += ['%s: %s' % (label, p) for p in report.get('c2_gate_problems') or []]
    problems += ['%s: %s' % (label, p) for p in stream_problems(report)]
    if report.get('fatal'):
        problems.append('%s: fatal: %s' % (label, report['fatal']))
    if want is not None:
        problems += ['%s: %s' % (label, p) for p in request_problems(report, want)]
    if acceptance:
        problems += ['%s: %s' % (label, p) for p in acceptance_problems(report)]
    return problems


def bringup_verdict(report, reference, profile=None, want=None):
    """IDENTICAL/DIVERGED per user against the reference, and PASS only when all four are IDENTICAL,
    the contract served the expected profile - with v235's engine argv, on REFERENCE_PROFILES - every
    stream spent what it asked, and the harness's own verdict held."""
    if report is None:
        return dict(verdict='FAIL', reason='no gate report', users=[])
    result = real_text_compare.reference_verdicts(report, reference)
    problems = arm_problems('bring-up', report, want)
    served_argv = (report.get('platform') or {}).get('served_argv')
    argv_diff = real_text_compare.served_engine_diff(served_argv, reference.get('command'))
    if argv_diff and profile in REFERENCE_PROFILES:
        problems.append('bring-up: the launched argv is not v235\'s engine (served | v235): %s' % ', '.join(
            '%s=%s|%s' % (flag, json.dumps(values[0]), json.dumps(values[1])) for flag, values in sorted(argv_diff.items())))
    missing = ((report.get('flag_markers') or {}).get('missing')) or []
    passed = result['verdict'] == 'IDENTICAL' and not problems and bool(report.get('gate_passed'))
    lines = real_text_compare.render_reference(result).split('\n') + ['problem: %s' % p for p in problems]
    if argv_diff and profile not in REFERENCE_PROFILES:
        lines.append('launched argv differs from v235\'s engine in: %s' % ', '.join(sorted(argv_diff)))
    return dict(verdict='PASS' if passed else 'FAIL', texts=result['verdict'], users=result['users'],
                arithmetic_diff=result['arithmetic_diff'], configuration_diff=result['configuration_diff'],
                gate_passed=report.get('gate_passed'), platform_problems=problems, argv_diff=argv_diff,
                missing_markers=missing, lines=lines)


def memory_verdict(report, users=MEMORY_USERS, want=None):
    """G5 records, it does not judge a threshold: PASS when every stream completed as asked and every
    user's engine logged its DRAM line."""
    if report is None:
        return dict(verdict='FAIL', reason='no gate report')
    dram = report.get('dram') or {}
    problems = arm_problems('memory', report, want)
    if (dram.get('engines') or 0) < users:
        problems.append('%s dram-after-engine lines for %d users' % (dram.get('engines') or 0, users))
    return dict(verdict='FAIL' if problems else 'PASS', problems=problems, engines=dram.get('engines'),
                min_free_gb=dram.get('min_free_gb'), min_largest_free_mb=dram.get('min_largest_free_mb'),
                lines=dram_lines(report) + ['problem: %s' % p for p in problems])


def matrix_verdict(concurrent, solo, rerun=None, wants=None):
    """The exactness policy over the concurrent arm and its solo reference (and their re-run), and
    every arm's own problems (arm_problems, acceptance included): any problem FAILS the plan outright
    (no re-run can clear it)."""
    if concurrent is None or solo is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    wants = wants or {}
    result = real_text_compare.exactness_policy(concurrent, solo, rerun)
    problems = []
    for label, report in (('concurrent', concurrent), ('solo', solo)) + (
            (('concurrent re-run', rerun[0]), ('solo re-run', rerun[1])) if rerun else ()):
        problems += arm_problems(label, report, wants.get(label.replace(' re-run', '')), acceptance=True)
    verdict = 'FAIL' if problems else result['verdict']
    return dict(verdict=verdict, policy=result, platform_problems=problems,
                lines=real_text_compare.render_policy(result).split('\n') + ['problem: %s' % p for p in problems])


def event_shortfalls(report):
    """Every lifecycle drop that did not land where it was meant to (EVENT_PHASES; a live=K drop at K
    live streams; a multiple drop complete and within BARRIER_SPREAD_SECONDS), as text."""
    asked_drops = (report.get('user_events') or {}).get('drops') or {}
    if not asked_drops:
        return []
    lifecycle = report.get('lifecycle')
    if not lifecycle:
        return ['no lifecycle record: the drops were never watched']
    shortfalls = []
    events = lifecycle.get('events') or {}
    for user, event in sorted(events.items(), key=lambda item: int(item[0])):
        kind, spec = event.get('kind'), event.get('spec')
        if not event.get('fired'):
            shortfalls.append('user %s: %s never fired (%s)' % (user, spec, event.get('missed')))
            continue
        if event.get('delivered') is False:
            shortfalls.append('user %s: %s fired but its cancel never reached the socket (%s): the stream ran on' % (
                user, spec, event.get('undelivered')))
        if event.get('phase') not in EVENT_PHASES.get(kind, ()):
            shortfalls.append('user %s: %s fired during %s, not %s' % (
                user, spec, event.get('phase'), ' or '.join(EVENT_PHASES.get(kind, ('?',)))))
        if kind == 'live' and ('live=%s' % event.get('live')) != spec:
            shortfalls.append('user %s: %s fired at %s live streams' % (user, spec, event.get('live')))
    for barrier in lifecycle.get('barriers') or []:
        members = [str(member) for member in barrier.get('members') or []]
        fired = [events.get(member, {}).get('fired_s') for member in members]
        if not barrier.get('complete'):
            shortfalls.append('users %s: the multiple drop released before all had %s chunks' % (
                '+'.join(members), barrier.get('chunks')))
        elif None in fired or max(fired) - min(fired) > BARRIER_SPREAD_SECONDS:
            shortfalls.append('users %s: the multiple drop closed over %s s, not one moment' % (
                '+'.join(members), None if None in fired else round(max(fired) - min(fired), 3)))
    return shortfalls


def ledger_drift(report):
    return ((report or {}).get('ledger') or {}).get('idle_drift_gb')


def lifecycle_verdict(event, solo, rerun=None, want=None, solo_want=None):
    """One lifecycle event arm against the solo reference, under the exactness policy (a dropped or
    budget-cut user must be a prefix of its solo text, an ignore_eos user must run past it, every other
    user identical), plus what the row asks beyond the texts: the engine answered every seat's request
    afterwards, every stream that was not dropped completed as asked, the contract served the expected
    profile, the ledger's idle drift matches the solo arm's (LEDGER_FLAT_GB), and every drop hit its
    phase (else NOT_EXERCISED). The P7 residual status (attach time) is reported, not judged."""
    if event is None or solo is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    policy = real_text_compare.exactness_policy(event, solo, rerun, real_text_compare.lifecycle_pass)
    problems = []
    for label, report, report_want in (('event', event, want), ('solo', solo, solo_want)) + (
            (('event re-run', rerun[0], want), ('solo re-run', rerun[1], solo_want)) if rerun else ()):
        problems += arm_problems(label, report, report_want)
        if not report.get('alive'):
            entries = report.get('alive_after') or []
            entries = entries if isinstance(entries, list) else [entries]
            errors = [str((entry or {}).get('error')) for entry in entries if (entry or {}).get('error')]
            problems.append('%s: the engine did not answer every seat after the streams (%s)' % (
                label, '; '.join(errors) or 'no alive check'))
    shortfalls = event_shortfalls(event) + (['re-run: %s' % s for s in event_shortfalls(rerun[0])] if rerun else [])
    mine, theirs = ledger_drift(event), ledger_drift(solo)
    if mine is None or theirs is None:
        shortfalls.append('the ledger\'s idle readings are missing (event %s, solo %s): flatness not judged' % (
            mine is not None, theirs is not None))
    else:
        grown = {chip: round(mine[chip] - theirs.get(chip, 0.0), 3) for chip in sorted(mine)
                 if abs(mine[chip] - theirs.get(chip, 0.0)) > LEDGER_FLAT_GB}
        if grown:
            problems.append('the ledger did not stay flat: idle allocation after the events differs from the solo '
                            'arm\'s by %s GB per chip (limit %s)' % (grown, LEDGER_FLAT_GB))
    residual = (event.get('ledger') or {}).get('residual_status')
    if problems:
        verdict = 'FAIL'
    elif policy['verdict'] == 'PASS' and shortfalls:
        verdict = 'NOT_EXERCISED'
    else:
        verdict = policy['verdict']
    lines = real_text_compare.render_policy(policy).split('\n') + ['problem: %s' % p for p in problems] + [
        'not exercised: %s' % s for s in shortfalls] + [
        'ledger idle drift (GB per chip) event %s, solo %s; P7 residual %s (attach time, reported only)' % (
            mine, theirs, residual)]
    for user, entry in sorted(((event.get('lifecycle') or {}).get('events') or {}).items(), key=lambda i: int(i[0])):
        lines.append('user %s: %s -> %s at %s s during %s, %s live' % (
            user, entry.get('spec'), entry.get('reason') or 'not fired', entry.get('fired_s'), entry.get('phase'),
            entry.get('live')))
    return dict(verdict=verdict, policy=policy, problems=problems, shortfalls=shortfalls, ledger_residual=residual,
                ledger_drift=dict(event=mine, solo=theirs), events=(event.get('lifecycle') or {}).get('events'),
                lines=lines)


def worst(verdicts):
    for verdict in VERDICT_ORDER:
        if verdict in verdicts:
            return verdict
    return 'PASS' if verdicts else 'FAIL'


def extract(stdout_text):
    try:
        return real_text_compare.extract_report(stdout_text)
    except ValueError:
        return None


def remove_container(name):
    try:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass


def platform_containers():
    """Names of the platform's serving containers present (running or not), or [] if docker cannot say."""
    try:
        output = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return [name for name in output.stdout.decode('utf-8', 'replace').split() if name.startswith(PLATFORM_PREFIX)]


def server_log(arm_dir):
    """The arm's server log (the harness writes it beside its report), or None."""
    try:
        with open(os.path.join(arm_dir, 'server.log'), errors='replace') as handle:
            return handle.read()
    except OSError:
        return None


def any_request_check(log_text, any_request):
    """(problems, per-request engine lines, consumer classes) of one served arm's server log. Under a
    C2-any profile the D2 consumer must have logged its live line (QUARANTINE_LIVE_PREFIX, any class);
    under any other no C2-any marker may appear."""
    if log_text is None:
        return (['no server.log: the D2 quarantine consumer\'s live line (%s) cannot be checked' % QUARANTINE_LIVE]
                if any_request else []), [], []
    lines = log_text.splitlines()
    engines = [line.strip() for line in lines if ANY_REQUEST_ENGINE in line]
    consumers = sorted(set(line.split(QUARANTINE_LIVE_PREFIX, 1)[1].strip().split(' ')[0]
                           for line in lines if QUARANTINE_LIVE_PREFIX in line))
    if any_request:
        if not consumers:
            return (['%s=1 but the server log has no "%s": the quarantine consumer never ran in the scheduler '
                     'this engine built, so a refusal would strand its request (serving_request_quarantine)' % (
                         ANY_REQUEST_FLAG, QUARANTINE_LIVE)], engines, consumers)
        return [], engines, consumers
    leaked = sorted(marker for marker in ANY_REQUEST_MARKERS if marker in log_text)
    if leaked:
        return (['the profile leaves %s off, but the server log carries C2-any markers (%s): this is not the '
                 'profile\'s path' % (ANY_REQUEST_FLAG, '; '.join(leaked))], engines, consumers)
    return [], engines, consumers


def wedged(arm_dir):
    """Whether the arm's server log (or the gate's stdout, which quotes its tail) shows the wedge."""
    for name in ('server.log', 'gate-stdout.log'):
        path = os.path.join(arm_dir, name)
        try:
            with open(path, errors='replace') as handle:
                if WEDGE in handle.read():
                    return True
        except OSError:
            continue
    return False


class Runner(object):
    """Runs arms with docker and keeps their files; `execute` and `containers` are injectable for the
    CPU tests. Once an arm hits infrastructure trouble (`infra`: a platform container on M+A, the
    ethernet-core wedge) every later arm is skipped."""

    def __init__(self, image, profile, results, checkout, devices, hub=HUB, execute=None, log=print,
                 containers=None, corpus=None, any_request=False):
        self.image, self.profile, self.results, self.checkout = image, profile, results, checkout
        self.devices, self.hub, self.log = devices, hub, log
        self.execute = execute or self._execute
        self.containers = containers or platform_containers
        self.corpus = corpus or (lambda: image_corpus(image, checkout))
        self.any_request = any_request   # the profile turns C2-any on (any_request_profile)
        self.arms = {}
        self.infra = None

    @staticmethod
    def _execute(arguments, stdout_path, timeout, name):
        """docker run, the container removed before (a leftover of the same name makes docker exit 125)
        and after, however the run ended - a timeout, or the SIGTERM/SIGINT of a cancelled or timed-out
        step (main turns SIGTERM into SystemExit, so this finally runs)."""
        remove_container(name)
        try:
            with open(stdout_path, 'w') as handle:
                try:
                    return subprocess.run(arguments, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout).returncode
                except subprocess.TimeoutExpired:
                    return 'timeout'
        finally:
            remove_container(name)

    def run(self, arm, gate_args, timeout):
        arm_dir = os.path.join(self.results, arm)
        os.makedirs(arm_dir, exist_ok=True)
        os.chmod(arm_dir, 0o777)
        if self.infra is None:
            held = self.containers()
            if held:
                self.infra = 'a platform serving container exists (%s): M+A may be reloaded under the gate' % (
                    ', '.join(held))
        if self.infra is not None:
            self.log('[C2-GATE] arm %s: skipped - %s' % (arm, self.infra))
            self.arms[arm] = dict(exit='skipped', infra=self.infra)
            return None
        name = CONTAINER_PREFIX + arm
        arguments = gate_run(self.image, name, self.profile, self.devices, self.checkout, arm_dir, gate_args, self.hub)
        with open(os.path.join(arm_dir, 'docker-run.json'), 'w') as handle:
            json.dump(arguments, handle, indent=1)
        self.log('[C2-GATE] arm %s: %s' % (arm, ' '.join(gate_args)))
        started = time.time()
        status = self.execute(arguments, os.path.join(arm_dir, 'gate-stdout.log'), timeout, name)
        seconds = round(time.time() - started, 1)
        with open(os.path.join(arm_dir, 'gate-stdout.log'), errors='replace') as handle:
            report = extract(handle.read())
        engines, consumers = [], []
        if report is not None:
            # What the server log says about C2-any, judged with the arm's own problems (arm_problems).
            problems, engines, consumers = any_request_check(server_log(arm_dir), self.any_request)
            if problems:
                report['c2_gate_problems'] = list(report.get('c2_gate_problems') or []) + problems
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
        for problem in (report or {}).get('c2_gate_problems') or []:
            self.log('[C2-GATE] arm %s: problem: %s' % (arm, problem))
        for text in engines[:8]:
            self.log('[C2-GATE] arm %s: %s' % (arm, text))
        if consumers and consumers != ['TTScheduler']:
            self.log('[C2-GATE] arm %s: note: the D2 consumer ran in %s, not TTScheduler (the scheduler class v235 logged)'
                     % (arm, ', '.join(consumers)))
        infra = None
        if wedged(arm_dir):
            infra = self.infra = ('arm %s hit tt-metal\'s ethernet-core wedge ("%s", llrt.cpp:594): reset M+A '
                                  'before the next run; nothing after it ran' % (arm, WEDGE))
            self.log('[C2-GATE] arm %s: INFRA - %s' % (arm, infra))
        self.arms[arm] = dict(exit=status, seconds=seconds, launched=line, gate_passed=(report or {}).get('gate_passed'),
                              fatal=(report or {}).get('fatal'), infra=infra,
                              any_request_engines=engines[:ANY_REQUEST_LINES_KEPT], quarantine_consumers=consumers)
        return report


def bringup_warning(profiles, profile):
    """Why v235's shape may be refused at this profile's edge, or None. The contract caps a prompt at
    context less the output ceiling, or less min_answer_tokens (profile_limits): c2 admits at most
    123,136, so v235's 131,072-token prompts come back as 400s; c2-gate (min_answer_tokens 256) admits
    them, which is why the c2 bring-up runs on c2-gate."""
    context, ceiling, room = profile_limits(profiles, profile)
    if room >= BRINGUP_PROMPT and context >= BRINGUP_PROMPT + BRINGUP_MAX_TOKENS:
        return None
    return ('profile %s admits prompts up to %d tokens (context %d, output ceiling %d; profile_limits); the '
            'bring-up sends %d-token prompts, which its edge will refuse' % (
                profile, room, context, ceiling, BRINGUP_PROMPT))


def run_plan(plan, runner, profiles, reference=None, lengths=None, max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS,
             memory_prompt=None):
    notes = []
    arms = plan_arms(plan, runner.profile, profiles, lengths, max_tokens, memory_prompt, notes)
    for note in notes:
        runner.log('[C2-GATE] note: %s' % note)
    if plan == 'bringup':
        warning = bringup_warning(profiles, runner.profile)
        if warning:
            runner.log('[C2-GATE] bringup WARNING: %s' % warning)
        (arm, args, timeout), = arms
        try:
            mismatch = corpus_mismatch(runner.corpus(), reference)
        except Exception as error:
            mismatch = None
            runner.log('[C2-GATE] bringup WARNING: the image\'s corpus could not be read first (%s)' % error)
        if mismatch:
            result = dict(verdict='NOT_COMPARABLE', reason=mismatch, lines=['corpus: %s' % mismatch])
        else:
            result = bringup_verdict(runner.run(arm, args, timeout), reference, runner.profile, asked(args))
        if warning:
            result['warning'] = warning
    elif plan == 'memory':
        results = []
        for arm, args, timeout in arms:
            one = memory_verdict(runner.run(arm, args, timeout), want=asked(args))
            one['any_request_engines'] = (runner.arms.get(arm) or {}).get('any_request_engines') or []
            results.append((arm, one))
        if len(results) == 1:
            result = results[0][1]
        else:
            # memory-short on a C2-any profile: each arm judged alone, the plan the worst of them; every
            # per-request engine's proposal ladder recorded with its arm (R3).
            result = dict(verdict=worst([one['verdict'] for _, one in results]), arms=dict(results),
                          lines=['%s: %s' % (arm, line) for arm, one in results
                                 for line in (one.get('lines') or []) + one['any_request_engines']] +
                          ['%s %s' % (arm, one['verdict']) for arm, one in results])
    elif plan == 'lifecycle':
        reports = [(arm, args, timeout, runner.run(arm, args, timeout)) for arm, args, timeout in arms]
        (s_arm, s_args, s_timeout, solo) = reports[-1]
        solo_want = asked(s_args)
        arms_results = dict((arm, lifecycle_verdict(report, solo, want=asked(args), solo_want=solo_want))
                            for arm, args, _, report in reports[:-1])
        again = [(arm, args, timeout) for arm, args, timeout, _ in reports[:-1] if arms_results[arm]['verdict'] == 'RERUN']
        if again:
            runner.log('[C2-GATE] lifecycle: a first divergence - re-running %s and the solo arm once (the exactness '
                       'policy)' % ', '.join(arm for arm, _, _ in again))
            solo_again = runner.run(s_arm + '-rerun', s_args, s_timeout)
            first = dict((arm, report) for arm, _, _, report in reports)
            for arm, args, timeout in again:
                event_again = runner.run(arm + '-rerun', args, timeout)
                rerun = (event_again, solo_again) if event_again is not None and solo_again is not None else None
                arms_results[arm] = lifecycle_verdict(first[arm], solo, rerun, want=asked(args), solo_want=solo_want)
                if rerun is None:
                    arms_results[arm].update(verdict='FAIL', reason='a re-run arm left no gate report')
        result = dict(verdict=worst([r['verdict'] for r in arms_results.values()]), arms=arms_results,
                      lines=['%s: %s' % (arm, line) for arm, r in arms_results.items() for line in r['lines']] +
                      ['%s %s' % (arm, r['verdict']) for arm, r in arms_results.items()])
    else:
        (c_arm, c_args, c_timeout), (s_arm, s_args, s_timeout) = arms
        wants = dict(concurrent=asked(c_args), solo=asked(s_args))
        concurrent, solo = runner.run(c_arm, c_args, c_timeout), runner.run(s_arm, s_args, s_timeout)
        result = matrix_verdict(concurrent, solo, wants=wants)
        if result['verdict'] == 'RERUN':
            runner.log('[C2-GATE] matrix: a first divergence - re-running both arms once (the exactness policy)')
            rerun = (runner.run(c_arm + '-rerun', c_args, c_timeout), runner.run(s_arm + '-rerun', s_args, s_timeout))
            result = matrix_verdict(concurrent, solo, rerun if None not in rerun else None, wants=wants)
            if None in rerun:
                result.update(verdict='FAIL', reason='a re-run arm left no gate report')
    if runner.infra is not None:
        result.update(verdict='INFRA', reason=runner.infra)
    if notes:
        result['notes'] = notes
    for line in result.get('lines') or ():
        runner.log('[C2-GATE] %s %s' % (plan, line))
    runner.log('[C2-GATE] %s %s%s' % (plan, result['verdict'], ' (%s)' % result['reason'] if result.get('reason') else ''))
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


def image_corpus(image, checkout):
    """The real-text corpus the image's harness would build (real_text_prompts.build_corpus over the
    image's installed vLLM), read in a throwaway container - no network, no devices, the contract's
    boot off: {root, files, characters, sha256, ...}. v235's prompts came from corpus sha 6cf533fb; an
    image whose vLLM source differs builds other prompts, and every bring-up user reads NOT_COMPARABLE."""
    script = ('import json, sys; sys.path.insert(0, "/bench"); import real_text_prompts as r; '
              'corpus, info = r.build_corpus(r.package_root()); print(json.dumps(info))')
    output = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-e', 'QWEN_C2_SERVING=0', '--mount',
                             'type=bind,src=%s,dst=/bench/real_text_prompts.py,readonly' % os.path.join(
                                 checkout, 'scripts', 'ci', 'real_text_prompts.py'),
                             '--entrypoint', 'python3', image, '-B', '-c', script],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if output.returncode:
        raise RuntimeError('corpus read exited %d: %s' % (output.returncode, output.stderr[-400:]))
    return json.loads(output.stdout.decode('utf-8').strip().splitlines()[-1])


def corpus_mismatch(corpus, reference):
    """Why the image's corpus is not the reference's, or None."""
    theirs = ((reference or {}).get('real_text') or {}).get('corpus') or {}
    if not theirs.get('sha256') or corpus.get('sha256') == theirs['sha256']:
        return None
    return ('the image builds its real-text prompts from corpus %s (%s files, %s characters), the reference from %s '
            '(%s files): its prompts are not the reference\'s' % (
                str(corpus.get('sha256'))[:16], corpus.get('files'), corpus.get('characters'),
                theirs['sha256'][:16], theirs.get('files')))


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
    parser.add_argument('--lengths', default=None, help='matrix prompt lengths (default: the G4 ladder, its top '
                                                        'rung lowered to what the profile admits)')
    parser.add_argument('--max-tokens', type=int, default=c2_serving_job.DEFAULT_MAX_TOKENS)
    parser.add_argument('--memory-prompt', type=int, default=None)
    parser.add_argument('--budget-seconds', type=int, default=None,
                        help='refuse a plan list whose worst case (every arm to its limit, every re-run) is longer')
    parser.add_argument('--reference', default=REFERENCE)
    parser.add_argument('--checkout', default=os.path.dirname(os.path.dirname(HERE)))
    parser.add_argument('--profiles', default=None, help='a profiles JSON instead of the image\'s own')
    parser.add_argument('--hub', default=HUB)
    parser.add_argument('--dry-run', action='store_true', help='print every arm\'s docker argv and run nothing')
    return parser


def main(argv=None, execute=None, devices=None, log=print, containers=None, corpus=None):
    options = build_parser().parse_args(argv)
    plans = c2_serving_job.split_list(options.plan)
    unknown = sorted(set(plans) - set(c2_serving_job.GATE_PLANS))
    if not plans or unknown:
        log('unknown plan(s): %s' % ', '.join(unknown or ['(none)']))
        return 2
    lengths = [c2_serving_job.positive_int('--lengths', part) for part in c2_serving_job.split_list(options.lengths)] \
        if options.lengths else None
    if options.profiles:
        with open(options.profiles, encoding='utf-8') as handle:
            profiles = json.load(handle)
    else:
        profiles = image_profiles(options.image)
    if options.profile not in profiles['profiles']:
        log('profile %r is not in the image\'s profiles (%s)' % (options.profile, ', '.join(sorted(profiles['profiles']))))
        return 2
    # Every plan is checked against the image's profile before any container starts.
    arms_of = {}
    for plan in plans:
        try:
            arms_of[plan] = plan_arms(plan, options.profile, profiles, lengths, options.max_tokens, options.memory_prompt)
        except PlanError as error:
            log('refused: %s' % error)
            return 2
    worst_case = worst_case_seconds(plans, arms_of)
    if options.budget_seconds is not None and worst_case > options.budget_seconds:
        log('refused: plans %s may take %d s (every arm to its limit, every re-run), past the %d s this step and job '
            'leave: run fewer plans per tag' % (','.join(plans), worst_case, options.budget_seconds))
        return 2
    with open(options.reference, encoding='utf-8') as handle:
        reference = json.load(handle)
    os.makedirs(options.results, exist_ok=True)
    if options.dry_run:
        log(json.dumps(dict(worst_case_seconds=worst_case, budget_seconds=options.budget_seconds)))
        for plan in plans:
            for arm, args, timeout in arms_of[plan]:
                log(json.dumps(dict(arm=arm, timeout=timeout, docker=gate_run(
                    options.image, CONTAINER_PREFIX + arm, options.profile, devices or ['<M>', '<A>'],
                    options.checkout, os.path.join(options.results, arm), args, options.hub))))
        return 0
    runner = Runner(options.image, options.profile, options.results, options.checkout,
                    devices if devices is not None else serving_pair(), options.hub, execute, log, containers, corpus,
                    any_request=any_request_profile(profiles, options.profile))
    context, ceiling, room = profile_limits(profiles, options.profile)
    summary = dict(image=options.image, profile=options.profile, plans=plans, context=context,
                   output_ceiling=ceiling, largest_prompt=room, worst_case_seconds=worst_case,
                   budget_seconds=options.budget_seconds, results={}, arms=runner.arms)
    try:
        for plan in plans:
            summary['results'][plan] = run_plan(plan, runner, profiles, reference, lengths, options.max_tokens,
                                                options.memory_prompt)
    finally:
        summary['passed'] = bool(summary['results']) and len(summary['results']) == len(plans) and all(
            result['verdict'] == 'PASS' for result in summary['results'].values())
        summary['infra'] = runner.infra
        with open(os.path.join(options.results, 'c2-gate-summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2)
    log('C2_GATE profile=%s plans=%s passed=%s%s' % (options.profile, ','.join(plans), summary['passed'],
                                                     ' infra=%s' % runner.infra if runner.infra else ''))
    return 0 if summary['passed'] else 1


def _terminate(signum, frame):
    """SIGTERM (a cancelled or timed-out step) as SystemExit, so every finally - the arm's container
    removal first - runs before the runner's SIGKILL."""
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main())
