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
                            and 2 live streams, user 4 takes user 0's freed seat (max_tokens 256, so it
                            leaves while user 3 still streams), user 5 (ignore_eos)
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

S2 PLANS (C2-packed-any, s2-design.md W11 and 6.3; the S2 image, graft K64j, with its profiles c2-packed
(traffic: c2 plus the block and QWEN_FAST_EXTENT_REPLAY=1) and c2-packed-gate (c2-gate plus the flag)).
Each arm may serve its own profile and add gate-only environment (Arm: -e NAME=value, never a profile's key):
QWEN_FAST_EXTENT_AUDIT=1 on every arm whose profile has the flag, the capture-position knob on G3b's, the
forced cap on M4's. Each knob must reach the harness's own process (its qwen_configuration) or the arm fails
(read-the-launched-argv). The exactness policy is strict (CB2a K2 passed bitwise): a reproduced divergence
FAILS, and a flag the card evidence claims exact never makes one NOT_COMPARABLE (real_text_compare.
S2_EXACT_CLAIMS); --policy dc-i (the user's decision D-c(i), with --policy-decision) is the only relaxation.
  warm, warm-off  M1: solo on c2-packed over every later length (WARM_LENGTHS), short prompts decoding
           past 2048 of history, then 4 x 131072, 4 x 16384 and 4 x 4096 on c2-packed-gate (warm) and on
           c2-gate (warm-off), the two shorter with the capture knob. Completes; kernel-cache entries recorded.
  control  M3-M4, G3: --pairs A/B pairs (default 2: ABAB) of c2-gate (A) and c2-packed-gate (B) at v235's
           shape. Every arm IDENTICAL to v235; B's median four-live packed round, net of its audit's ms, at
           most CONTROL_TOLERANCE x A's in every pair; B shows the S2 lines and A none of them.
  forced-cap  M4, Q5: c2-packed-gate with QWEN_FAST_GATE_FORCE_CAP=8 against without, at v235's shape: both
           IDENTICAL to v235 and to each other, and the capped arm's [PACKED] caps all at most 8.
  control-below  M5, G3b: per family F (--families, default 16640 and 4352) and pair, c2-gate against
           c2-packed-gate, both with the capture knob at F - 256, 4 x (F - 256) prompts, 224 out (every ticket
           stays in F). Every user IDENTICAL across the arms; the flag-off arm must serve packed rounds inside
           F, else NO_DECISION (Q22).
  lifecycle-arrival  M6's added event: the arrival that makes the live count two (user 1) leaves in its build.
           (lifecycle itself on an S2 profile also needs one 'survivor narrowed' line across its arms, else
           NOT_EXERCISED, and every refuse_round request ended FINISHED_ABORTED.)
  mixed, short, boundaries, staggered  M7-M10, G4 (--lengths replaces a plan's default set): a concurrent and
           a solo arm under the policy with one re-run, plus: every arm's extent audit clean and complete,
           zero idle commits, refuse_round rounds, cap refusals and other audit mismatches, the four-live
           per-user rate (net of the audit) above GENERAL_RATE; mixed needs a packed round of two families,
           short the one-bucket ladder, no hold or refusal and the before-point floor (FLOOR_GB), boundaries
           (ignore_eos, 8192 out) cap events and BOUNDARY_MIN_CROSSINGS 256-key crossings per user, staggered
           (STAGGER_SECONDS apart) padded rounds at two or three live and a padded-probe arm.
  churn    M11, G5: CHURN_LENGTHS over four seats, each user's seat freed and retaken: no engine death, no
           hold or refusal, the floor, and dead proposal traces released at departures.
  permuted  built, required only under D-c(i): two concurrent arms, the second admitted in reverse order;
           at least two SAME_PATH users (else NOT_EXERCISED), and no same-path user may differ.
On an S2 profile the four S1 plans add their S2 checks too (memory: no hold or refusal and the floor).
Every arm of an S2 plan or on an S2 profile (--jit auto) must leave the kernel cache as it found it: a judged
arm that compiled is a problem (M1 warms it first). A non-identical user of an S2 arm gets a divergence record
(t*, character, round, positions, E, paths), and any NOT_COMPARABLE or UNSTABLE one is listed in the summary's
s2_exit_blockers with it (<results>/<plan>-divergence-records.json).

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
import lever_n_m3native_gate as harness  # noqa: E402  (stdlib only; real_text_compare imports it too)
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
REFERENCE_PROFILES = ('exact', 'c2-gate', 'c2-packed-gate')
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
# The one-fresh-prefill cap (serving_prefill_admission) logs ADMISSION_LIVE_PREFIX + the class from its
# first prefill step (the platform warmup). Without it the TTScheduler batches simultaneous arrivals into
# one step and the lifecycle kills the engine (run 36211578069), so every arm under the switch must show it.
ANY_REQUEST_FLAG = 'QWEN_FAST_ANY_REQUEST'
QUARANTINE_LIVE_PREFIX = '[PINDIAG] request quarantine consumer live in '
QUARANTINE_LIVE = QUARANTINE_LIVE_PREFIX + 'TTScheduler'
ANY_REQUEST_ENGINE = '[PINDIAG] any-request engine for '
ADMISSION_LIVE_PREFIX = '[PINDIAG] one fresh prefill per step live in '
ANY_REQUEST_MARKERS = ('[PINDIAG] request quarantine', ANY_REQUEST_ENGINE, '[PINDIAG] attach source check',
                       '[PINDIAG] one fresh prefill per step')
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
    # User 4 answers at most 256 tokens: run 36222651529 left user 3's live=2 drop NOT_EXERCISED because
    # user 4's natural answer kept three streams live until user 3 reached EOS on its own.
    ('lifecycle-drops', ['--drops', '0:build,1:live=4,2:live=3,3:live=2', '--user-ignore-eos', '5',
                         '--user-max-tokens', '4:256']),
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
VERDICT_ORDER = ('FAIL', 'INFRA', 'RERUN', 'NOT_COMPARABLE', 'UNSTABLE', 'NOT_EXERCISED', 'NO_DECISION')

# S2 (C2-packed-any; the module docstring's S2 PLANS). The profiles are W8's; the knobs are gate-only.
EXTENT_FLAG = harness.EXTENT_REPLAY_FLAG
AUDIT_ENV = ((harness.EXTENT_AUDIT_FLAG, '1'),)
# C2_GATE_AUDITS=all adds these three on the G4 arms (each already in the image under its parent flag; their
# mismatch lines FAIL an S2 arm whenever they appear).
G4_ALL_AUDITS = (('QWEN_FAST_PRESTAGE_AUDIT', '1'), ('QWEN_FAST_PAIR_MASK_AUDIT', '1'),
                 ('QWEN_FAST_FUSED_COMMIT_AUDIT', '1'))
PADDED_PROBE_ENV = (('QWEN_FAST_PADDED_PROBE', '1'),)
# What an arm may add with -e: the gate-only knobs, the padded probe and the G4 audits - never a profile key.
ARM_ENV_NAMES = frozenset(harness.S2_GATE_KNOBS + ('QWEN_FAST_PADDED_PROBE',) + tuple(name for name, _ in G4_ALL_AUDITS))
S2_OFF_PROFILE, S2_GATE_PROFILE, S2_TRAFFIC_PROFILE = 'c2-gate', 'c2-packed-gate', 'c2-packed'
S2_PLANS = c2_serving_job.S2_GATE_PLANS
WARM_PLANS = ('warm', 'warm-off')
DEFAULT_PAIRS = 2
BELOW_FAMILIES = (16640, 4352)
BELOW_MAX_TOKENS = 224           # F - 256 + 223 + 16 <= F: every ticket of a 224-token answer stays in F (B1)
CONTROL_TOLERANCE = 1.02         # flag-on median four-live round against flag-off, per pair (M3-M4; Q16)
FORCE_CAP = 8
GENERAL_RATE, TARGET_RATE = 13.0, 27.0   # tok/s per user at four live: general's (MEM goal-c2-serving.md:11), S2's aim
FLOOR_GB = 0.25                  # before-point floor, GB per chip (s2-design 3.3, M8)
MIXED_LENGTHS = (1536, 20000, 60000, 120000)       # M7's first set; the second, 5000,40000,90000,123136, by --lengths
SHORT_LENGTHS = (60, 255, 2047, 120000)
BOUNDARY_LENGTHS = (8192, 30000, 70000, 110000)
BOUNDARY_MAX_TOKENS = 8192
BOUNDARY_MIN_CROSSINGS = 16
STAGGER_SECONDS = 30.0
CHURN_LENGTHS = (110000, 120000, 123136, 110000, 120000, 123136, 110000, 120000, 1536)
CHURN_MAX_TOKENS = (768, 1024, 1280, 1536, 768, 1024, 1280, 1536, 2304)
# M1: every length a later arm serves (G4's sets, the lifecycle's, G3b's), solo; below 2048 decoding past 2048 of
# history (WARM_SHORT_TOKENS, ignore_eos) so the drafter's ramp shapes are compiled too.
WARM_LENGTHS = (60, 255, 1536, 2047, 2049, 4096, 5000, 8192, 16384, 20000, 30000, 40000, 60000, 70000, 90000,
                110000, 120000, 123136)
WARM_TOKENS, WARM_SHORT_TOKENS = 256, 2304
ARM_SECONDS.update({'warm': 5400, 'warm-off': 3600, 'control': 3600, 'forced-cap': 3600, 'control-below': 2400,
                    'lifecycle-arrival': 3300, 'mixed': 5400, 'short': 5400, 'boundaries': 5400, 'staggered': 4200,
                    'churn': 9000, 'permuted': 5400})
STREAM_SECONDS.update({'warm': 1800, 'warm-off': 900, 'control': 900, 'forced-cap': 900, 'control-below': 900,
                       'lifecycle-arrival': 1800, 'mixed': 3600, 'short': 3600, 'boundaries': 3600,
                       'staggered': 3600, 'churn': 7200, 'permuted': 3600})
WARM_SHORTER_SECONDS = 2400      # the 4 x 16384 and 4 x 4096 warm arms
# The kernel cache (B6): the image's TT_METAL_CACHE, under /experiment-cache (a link to /models/.qwen-c2, the
# hub's .qwen-c2 on the host: Dockerfile) or /models (the hub).
KERNEL_CACHE_ENV = 'TT_METAL_CACHE'
CACHE_MOUNTS = (('/experiment-cache/', '.qwen-c2'), ('/models/', ''))
JIT_MODES = c2_serving_job.JIT_MODES


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


def agent_shape(image, name, profile, devices, hub=HUB, env=()):
    """`docker run` of the node agent's container, up to the image: read-only root, the agent's tmpfs
    set, 8 CPUs, 80g, 4g shm, the two cards in the order given, hugepages, SYS_NICE, the hub at
    /models, the environment the platform adds (AGENT_ENV), and QWEN_C2_SERVING=1 with the profile.
    `env` (an S2 arm's gate-only knobs, ARM_ENV_NAMES) follows as more -e pairs; empty, the argv is
    exactly the agent's."""
    arguments = ['docker', 'run', '--rm', '--name', name, '--read-only']
    for tmpfs in AGENT_TMPFS:
        arguments += ['--tmpfs', tmpfs]
    arguments += ['--shm-size', '4g', '--memory', '80g', '--cpus', '8']
    for device in devices:
        arguments += ['--device', device]
    arguments += ['-v', '/dev/hugepages-1G:/dev/hugepages-1G', '--cap-add', 'SYS_NICE', '-v', '%s:/models' % hub]
    for variable in AGENT_ENV + ('QWEN_C2_SERVING=1', 'QWEN_C2_PROFILE=%s' % profile):
        arguments += ['-e', variable]
    for name_, value in env:
        if name_ not in ARM_ENV_NAMES:
            raise PlanError('%s is not an environment an arm may add (%s): a profile\'s keys are the profile\'s'
                            % (name_, ', '.join(sorted(ARM_ENV_NAMES))))
        arguments += ['-e', '%s=%s' % (name_, value)]
    return arguments


def gate_run(image, name, profile, devices, checkout, arm_dir, gate_args, hub=HUB, env=()):
    """The whole `docker run` of one arm: the agent's shape, the harness mounted read-only at /bench,
    the arm's results directory, and the harness as the entrypoint."""
    arguments = agent_shape(image, name, profile, devices, hub, env)
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


def s2_profile(profiles, name):
    """Whether the profile builds the S2 extent readers (QWEN_FAST_EXTENT_REPLAY=1: c2-packed, c2-packed-gate)."""
    return str((profiles['profiles'][name].get('env') or {}).get(EXTENT_FLAG, '0')) == '1'


class Arm(tuple):
    """One arm: (name, harness arguments, docker timeout) - a plain 3-tuple to everything that unpacks or
    compares it - carrying what an S2 plan adds: the profile it serves when not the gate's --profile, the
    environment it adds (gate-only knobs, ARM_ENV_NAMES), whether a first divergence re-runs it (None: its
    plan's rule, RERUN_PLANS), whether its kernel-cache growth may be judged, and its role in the plan."""

    def __new__(cls, name, args, timeout, profile=None, env=(), rerun=None, judged=True, role=None, **extra):
        arm = tuple.__new__(cls, (name, args, timeout))
        arm.profile, arm.env, arm.rerun, arm.judged, arm.role = profile, tuple(env), rerun, judged, role
        arm.extra = extra
        return arm


def arm_runs(plan, arm):
    """How often an arm may run in the worst case: twice when a first divergence re-runs it."""
    rerun = getattr(arm, 'rerun', None)
    return 2 if (plan in RERUN_PLANS if rerun is None else rerun) else 1


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
              memory_prompt=None, notes=None, s2=None):
    """The arms one plan runs, in order: [(arm name, harness arguments, docker timeout seconds)].
    Refuses (PlanError) what the profile cannot serve as asked; `lengths` None is the default ladder,
    whose rungs past the profile's largest admitted prompt are lowered to it (noted in `notes`). An S2
    plan's arms, and the S1 plans' on an S2 profile, are Arms (s2: pairs, families, audits); on any
    other profile the S1 plans' arms are exactly what they were."""
    if plan in S2_PLANS:
        return s2_plan_arms(plan, profile, profiles, lengths, max_tokens, notes, s2 or {})
    arms = base_plan_arms(plan, profile, profiles, lengths, max_tokens, memory_prompt, notes)
    if profile in profiles['profiles'] and s2_profile(profiles, profile):
        env = s2_env(profiles, profile, (s2 or {}).get('audits'), g4=plan == 'matrix')
        arms = [Arm(name, args, timeout, env=env) for name, args, timeout in arms]
    return arms


def base_plan_arms(plan, profile, profiles, lengths=None, max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS,
                   memory_prompt=None, notes=None):
    """plan_arms for the four S1 plans."""
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


G4_PLANS = ('mixed', 'short', 'boundaries', 'staggered')


def s2_env(profiles, profile, audits=None, g4=False):
    """What an arm on `profile` adds: the extent audit where the profile has the flag (every packed round of
    every S2 gate arm is audited, s2-design 6.1), and on a G4 arm under audits 'all' the other three audits."""
    if profile not in profiles['profiles'] or not s2_profile(profiles, profile):
        return ()
    return AUDIT_ENV + (G4_ALL_AUDITS if g4 and audits == 'all' else ())


def need_profile(profiles, name, s2, plan):
    """Refuse a plan whose fixed profile the image lacks, or has without (with) the extent flag."""
    if name not in profiles['profiles']:
        raise PlanError('%s serves profile %s, which the image\'s profiles do not carry (s2-design W8 adds c2-packed '
                        'and c2-packed-gate)' % (plan, name))
    if s2_profile(profiles, name) != s2:
        raise PlanError('%s needs profile %s %s %s=1' % (plan, name, 'with' if s2 else 'without', EXTENT_FLAG))


def v235_args(profiles, profile, plan):
    """v235's shape (the bring-up's arguments) on `profile`, or PlanError where its edge refuses it."""
    warning = bringup_warning(profiles, profile)
    if warning:
        raise PlanError('%s: %s' % (plan, warning))
    context = profile_limits(profiles, profile)[0]
    return common_args(profile, context, STREAM_SECONDS[plan]) + [
        '--users', str(BRINGUP_USERS), '--prompt-tokens', str(BRINGUP_PROMPT), '--max-tokens', str(BRINGUP_MAX_TOKENS),
        '--stagger', str(STAGGER)]


def four_user_args(profiles, profile, plan, prompt, max_tokens=BELOW_MAX_TOKENS):
    """Four users at one prompt length (G3b's and M1's 4 x 16384 and 4 x 4096)."""
    context, ceiling, room = profile_limits(profiles, profile)
    check_lengths(profile, [prompt], room, '%s prompt' % plan)
    check_budget(profile, max_tokens, ceiling, '%s --max-tokens' % plan)
    if prompt + max_tokens > context:
        raise PlanError('%s: a %d-token prompt leaves %d of profile %s\'s context, not %d' % (
            plan, prompt, context - prompt, profile, max_tokens))
    return common_args(profile, context, STREAM_SECONDS[plan]) + [
        '--users', '4', '--prompt-tokens', str(prompt), '--max-tokens', str(max_tokens), '--stagger', str(STAGGER)]


def below_families(families):
    """G3b's families, each a multiple of 256 the flag-off block serves (c2_serving_job.BELOW_FAMILY_RANGE)."""
    low, high = c2_serving_job.BELOW_FAMILY_RANGE
    for family in families:
        if family % 256 or not low <= family <= high:
            raise PlanError('control-below: family %d is not one the flag-off block serves (a multiple of 256 in %d..%d)'
                            % (family, low, high))
    return list(families)


def sized_arm_args(profile, profiles, plan, lengths, budget):
    """(common arguments, lengths text, context) for a per-user-length plan, refused as the matrix is."""
    context, ceiling, room = profile_limits(profiles, profile)
    check_lengths(profile, lengths, room, '%s prompt lengths' % plan)
    check_budget(profile, budget, ceiling, '%s --max-tokens' % plan)
    fits = answer_room(context, lengths, budget)
    if fits < budget:
        raise PlanError('%s --max-tokens %d: a %d-token prompt leaves %d tokens of profile %s\'s context %d' % (
            plan, budget, context - fits, fits, profile, context))
    return common_args(profile, context, STREAM_SECONDS[plan]), ','.join(str(length) for length in lengths), context


def s2_plan_arms(plan, profile, profiles, lengths, max_tokens, notes, s2):
    """plan_arms for S2_PLANS (the module docstring's S2 PLANS). control, forced-cap, control-below and the
    warm plans serve their own fixed profiles (c2-gate, c2-packed-gate, c2-packed) whatever --profile is;
    every other S2 plan serves --profile, which must have the extent flag."""
    notes = notes if notes is not None else []
    pairs = int(s2.get('pairs') or DEFAULT_PAIRS)
    seconds = ARM_SECONDS[plan]
    if plan in ('control', 'forced-cap'):
        need_profile(profiles, S2_GATE_PROFILE, True, plan)
        on_env = s2_env(profiles, S2_GATE_PROFILE)
        on_args = v235_args(profiles, S2_GATE_PROFILE, plan)
        if plan == 'forced-cap':
            return [Arm('forced-cap-off', on_args, seconds, profile=S2_GATE_PROFILE, env=on_env, role='off'),
                    Arm('forced-cap-on', on_args, seconds, profile=S2_GATE_PROFILE,
                        env=on_env + ((harness.FORCE_CAP_FLAG, str(FORCE_CAP)),), role='on')]
        need_profile(profiles, S2_OFF_PROFILE, False, plan)
        off_args = v235_args(profiles, S2_OFF_PROFILE, plan)
        arms = []
        for index in range(1, pairs + 1):
            arms.append(Arm('control-off-%d' % index, off_args, seconds, profile=S2_OFF_PROFILE, role='off', pair=index))
            arms.append(Arm('control-on-%d' % index, on_args, seconds, profile=S2_GATE_PROFILE, env=on_env, role='on',
                            pair=index))
        return arms
    if plan == 'control-below':
        need_profile(profiles, S2_OFF_PROFILE, False, plan)
        need_profile(profiles, S2_GATE_PROFILE, True, plan)
        arms = []
        for family in below_families(s2.get('families') or BELOW_FAMILIES):
            prompt = family - 256
            knob = ((harness.CAPTURE_POSITION_FLAG, str(prompt)),)
            for index in range(1, pairs + 1):
                for role, name in (('off', S2_OFF_PROFILE), ('on', S2_GATE_PROFILE)):
                    arms.append(Arm('below-%d-%s-%d' % (family, role, index), four_user_args(profiles, name, plan, prompt),
                                    seconds, profile=name, env=knob + s2_env(profiles, name), role=role, pair=index,
                                    family=family))
        return arms
    if plan in WARM_PLANS:
        gate = S2_GATE_PROFILE if plan == 'warm' else S2_OFF_PROFILE
        need_profile(profiles, gate, plan == 'warm', plan)
        arms = []
        if plan == 'warm':
            need_profile(profiles, S2_TRAFFIC_PROFILE, True, plan)
            warm = list(lengths or WARM_LENGTHS)
            short = [index for index, length in enumerate(warm) if length < 2048]
            common, text, _ = sized_arm_args(S2_TRAFFIC_PROFILE, profiles, plan, warm, WARM_SHORT_TOKENS if short
                                             else WARM_TOKENS)
            args = common + ['--prompt-lengths', text, '--max-tokens', str(WARM_TOKENS), '--users', '1',
                             '--sequential-users', str(len(warm))]
            if short:
                args += ['--user-max-tokens', ','.join('%d:%d' % (index, WARM_SHORT_TOKENS) for index in short),
                         '--user-ignore-eos', ','.join(str(index) for index in short)]
            arms.append(Arm('warm-solo', args, ARM_SECONDS['warm'], profile=S2_TRAFFIC_PROFILE,
                            env=s2_env(profiles, S2_TRAFFIC_PROFILE), judged=False, role='solo'))
        arms.append(Arm('%s-4x%d' % (plan, BRINGUP_PROMPT), v235_args(profiles, gate, plan), ARM_SECONDS['warm-off'],
                        profile=gate, env=s2_env(profiles, gate), judged=False, role='four'))
        for prompt in (BELOW_FAMILIES[0] - 256, BELOW_FAMILIES[1] - 256):
            arms.append(Arm('%s-4x%d' % (plan, prompt), four_user_args(profiles, gate, plan, prompt), WARM_SHORTER_SECONDS,
                            profile=gate, env=((harness.CAPTURE_POSITION_FLAG, str(prompt)),) + s2_env(profiles, gate),
                            judged=False, role='four'))
        return arms
    if profile not in profiles['profiles'] or not s2_profile(profiles, profile):
        raise PlanError('%s is an S2 plan: run it on c2-packed (a profile with %s=1), not %s' % (plan, EXTENT_FLAG, profile))
    context, ceiling, room = profile_limits(profiles, profile)
    env = s2_env(profiles, profile, s2.get('audits'), g4=plan in G4_PLANS)
    seats = str(profile_seats(profiles, profile))
    if plan == 'lifecycle-arrival':
        check_lengths(profile, LIFECYCLE_LENGTHS, room, 'lifecycle prompt lengths')
        check_budget(profile, LIFECYCLE_MAX_TOKENS, ceiling, 'lifecycle --max-tokens')
        base = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--prompt-lengths', ','.join(str(length) for length in LIFECYCLE_LENGTHS),
            '--max-tokens', str(LIFECYCLE_MAX_TOKENS)]
        users = str(len(LIFECYCLE_LENGTHS))
        return [Arm('lifecycle-arrival', base + ['--users', users, '--stagger', str(STAGGER), '--alive-check', seats,
                                                 '--drops', '1:build'], seconds, env=env, rerun=True, role='event'),
                Arm('lifecycle-arrival-solo', base + ['--users', '1', '--sequential-users', users, '--alive-check', '1'],
                    seconds, env=env, rerun=True, role='solo')]
    if plan == 'churn':
        churn = list(lengths or CHURN_LENGTHS)
        budgets = list(CHURN_MAX_TOKENS) if lengths is None else [max_tokens] * len(churn)
        common, text, _ = sized_arm_args(profile, profiles, plan, churn, max(budgets))
        args = common + ['--prompt-lengths', text, '--max-tokens', str(max(budgets)), '--users', str(len(churn)),
                         '--user-max-tokens', ','.join('%d:%d' % item for item in enumerate(budgets)),
                         '--user-ignore-eos', ','.join(str(index) for index in range(len(churn))),
                         '--stagger', str(STAGGER), '--alive-check', seats]
        if len(churn) < 2 * int(seats):
            notes.append('churn: %d users over %s seats is fewer than two replacements per seat' % (len(churn), seats))
        return [Arm('churn', args, seconds, env=env, role='churn')]
    if plan == 'permuted':
        chosen = list(lengths or MIXED_LENGTHS)
        common, text, _ = sized_arm_args(profile, profiles, plan, chosen, max_tokens)
        base = common + ['--prompt-lengths', text, '--max-tokens', str(max_tokens), '--users', str(len(chosen)),
                         '--stagger', str(STAGGER)]
        reverse = ','.join(str(index) for index in reversed(range(len(chosen))))
        return [Arm('permuted-forward', base, seconds, env=env, role='forward'),
                Arm('permuted-reverse', base + ['--start-order', reverse], seconds, env=env, role='reverse')]
    defaults = dict(mixed=MIXED_LENGTHS, short=SHORT_LENGTHS, boundaries=BOUNDARY_LENGTHS, staggered=MIXED_LENGTHS)
    chosen = list(lengths or defaults[plan])
    budget = BOUNDARY_MAX_TOKENS if plan == 'boundaries' else max_tokens
    if plan == 'boundaries' and max_tokens != BOUNDARY_MAX_TOKENS:
        notes.append('boundaries: every user answers %d tokens with ignore_eos (--max-tokens %d not used), so each '
                     'crosses at least %d 256-key boundaries' % (BOUNDARY_MAX_TOKENS, max_tokens, BOUNDARY_MIN_CROSSINGS))
    common, text, _ = sized_arm_args(profile, profiles, plan, chosen, budget)
    base = common + ['--prompt-lengths', text, '--max-tokens', str(budget)]
    if plan == 'boundaries':
        base += ['--user-ignore-eos', ','.join(str(index) for index in range(len(chosen)))]
    stagger = STAGGER_SECONDS if plan == 'staggered' else STAGGER
    users = str(len(chosen))
    arms = [Arm('%s-concurrent' % plan, base + ['--users', users, '--stagger', str(stagger)], seconds, env=env, rerun=True,
                role='concurrent'),
            Arm('%s-solo' % plan, base + ['--users', '1', '--sequential-users', users], seconds, env=env, rerun=True,
                role='solo')]
    if plan == 'staggered':
        arms.append(Arm('staggered-probe', base + ['--users', users, '--stagger', str(stagger)], seconds,
                        env=env + PADDED_PROBE_ENV, rerun=False, role='probe'))
    return arms


def worst_case_seconds(plans, arms_of):
    """Every arm to its docker limit, every re-run a first divergence can ask for, plus per-arm overhead."""
    total = 0
    for plan in plans:
        for arm in arms_of[plan]:
            total += arm_runs(plan, arm) * (arm[2] + ARM_OVERHEAD_SECONDS)
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


def matrix_verdict(concurrent, solo, rerun=None, wants=None, relaxation=None):
    """The exactness policy over the concurrent arm and its solo reference (and their re-run), and
    every arm's own problems (arm_problems, acceptance included): any problem FAILS the plan outright
    (no re-run can clear it). `relaxation` is the user's D-c decision (Runner.relaxation), never a default."""
    if concurrent is None or solo is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    wants = wants or {}
    result = real_text_compare.exactness_policy(concurrent, solo, rerun, relaxation=relaxation)
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


def lifecycle_verdict(event, solo, rerun=None, want=None, solo_want=None, relaxation=None):
    """One lifecycle event arm against the solo reference, under the exactness policy (a dropped or
    budget-cut user must be a prefix of its solo text, an ignore_eos user must run past it, every other
    user identical), plus what the row asks beyond the texts: the engine answered every seat's request
    afterwards, every stream that was not dropped completed as asked, the contract served the expected
    profile, the ledger's idle drift matches the solo arm's (LEDGER_FLAT_GB), and every drop hit its
    phase (else NOT_EXERCISED). The P7 residual status (attach time) is reported, not judged."""
    if event is None or solo is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    policy = real_text_compare.exactness_policy(event, solo, rerun, real_text_compare.lifecycle_pass,
                                                relaxation=relaxation)
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
    C2-any profile the D2 consumer and the one-fresh-prefill cap must both have logged their live lines
    (QUARANTINE_LIVE_PREFIX and ADMISSION_LIVE_PREFIX, any class); under any other no C2-any marker may
    appear."""
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
        if not any(ADMISSION_LIVE_PREFIX in line for line in lines):
            return (['%s=1 but the server log has no "%s<class>": the one-fresh-prefill cap never ran, so '
                     'prompts arriving together share a prefill step and fail the engine '
                     '(serving_prefill_admission, run 36211578069)' % (ANY_REQUEST_FLAG, ADMISSION_LIVE_PREFIX)],
                    engines, consumers)
        return [], engines, consumers
    leaked = sorted(marker for marker in ANY_REQUEST_MARKERS if marker in log_text)
    if leaked:
        return (['the profile leaves %s off, but the server log carries C2-any markers (%s): this is not the '
                 'profile\'s path' % (ANY_REQUEST_FLAG, '; '.join(leaked))], engines, consumers)
    return [], engines, consumers


def s2_log_check(log_text, s2_on, env, report):
    """An arm's S2 problems as the host sees them: the harness's own (report['s2']['problems']), or without
    that record an S2 arm's missing record and a flag-off arm's S2 lines; and every -e knob the arm added
    must be in the harness process's configuration with its value (read-the-launched-argv: three runs were
    lost to variables that never crossed into the container)."""
    problems = []
    configuration = (report or {}).get('qwen_configuration')
    for name, value in env:
        if configuration is None:
            problems.append('-e %s=%s: the report records no configuration, so the knob cannot be shown to have '
                            'reached the container' % (name, value))
        elif configuration.get(name) != value:
            problems.append('-e %s=%s was passed, but the harness process carries %s=%r: the knob never reached the '
                            'container' % (name, value, name, configuration.get(name)))
    s2 = (report or {}).get('s2')
    if s2 is not None:
        return problems + ['S2: %s' % problem for problem in s2.get('problems') or []]
    if s2_on:
        return problems + ['%s=1 but the report carries no S2 record (report[\'s2\']): the harness mounted is not '
                           'W11\'s, or it died before reading the log' % EXTENT_FLAG]
    if log_text is not None:
        leaked = sorted(marker for marker in harness.S2_MARKERS if marker in log_text)
        if leaked:
            problems.append('the profile leaves %s off, but the server log carries S2 markers (%s): this is not the '
                            'profile\'s path' % (EXTENT_FLAG, '; '.join(leaked)))
    return problems


def host_live_rate(log_text):
    """acceptance_report.live_rate over the whole server log (both arms of a G3 pair read the same way), or
    an error record; never raises."""
    try:
        import acceptance_report
        return acceptance_report.live_rate(log_text, 4)
    except Exception as error:
        return dict(error='%s: %s' % (type(error).__name__, error))


def host_cache_dir(container_path, hub=HUB):
    """Where the image's kernel cache (its TT_METAL_CACHE) lives on the host: the image links /experiment-cache
    to /models/.qwen-c2, and /models is the hub (CACHE_MOUNTS); None for any other path."""
    for prefix, below in CACHE_MOUNTS:
        if container_path and container_path.startswith(prefix):
            rest = container_path[len(prefix):].strip('/')
            return os.path.join(*([hub] + ([below] if below else []) + ([rest] if rest else [])))
    return None


def count_entries(path):
    """The files under a kernel-cache directory (every JIT build adds some), or None when it cannot be read:
    as this user first, then with sudo -n (the container writes it as root)."""
    if not path:
        return None
    if os.path.isdir(path):
        total, failed = 0, []
        for _root, _dirs, files in os.walk(path, onerror=failed.append):
            total += len(files)
        if not failed:
            return total
    try:
        output = subprocess.run(['sudo', '-n', 'find', path, '-type', 'f'], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if output.returncode:
        return None
    return len([line for line in output.stdout.decode('utf-8', 'replace').splitlines() if line.strip()])


def image_kernel_cache(image, hub=HUB):
    """The image's kernel cache directory on the host (its ENV TT_METAL_CACHE through host_cache_dir), or None."""
    try:
        output = subprocess.run(['docker', 'image', 'inspect', '--format', '{{json .Config.Env}}', image],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
        environ = dict(item.split('=', 1) for item in json.loads(output.stdout.decode('utf-8') or 'null') or []
                       if '=' in item)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return host_cache_dir(environ.get(KERNEL_CACHE_ENV), hub)


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
                 containers=None, corpus=None, any_request=False, profiles=None, cache_entries=None, jit='auto',
                 policy='strict', decision=None):
        self.image, self.profile, self.results, self.checkout = image, profile, results, checkout
        self.devices, self.hub, self.log = devices, hub, log
        self.execute = execute or self._execute
        self.containers = containers or platform_containers
        self.corpus = corpus or (lambda: image_corpus(image, checkout))
        self.any_request = any_request   # the profile turns C2-any on (any_request_profile)
        # S2: the profiles (an arm may serve another than --profile), the kernel-cache entry counter (None:
        # not counted), whose growth is judged (JIT_MODES) and the exactness policy (strict, or dc-i with the
        # user's recorded decision).
        self.profiles, self.cache_entries, self.jit = profiles, cache_entries, jit
        self.policy, self.decision = policy, decision
        self.arms = {}
        self.infra = None

    def any_request_for(self, profile):
        if self.profiles is not None and profile in self.profiles['profiles']:
            return any_request_profile(self.profiles, profile)
        return self.any_request

    def s2_for(self, profile):
        return bool(self.profiles is not None and profile in self.profiles['profiles']
                    and s2_profile(self.profiles, profile))

    def judges(self, plan, profile, judged=True):
        """Whether an arm's kernel-cache growth fails it: never a warm arm, and under --jit auto only an S2
        plan's or an S2 profile's (M1 warms exactly those); judge takes every other arm too (M2), record none."""
        if plan in WARM_PLANS or not judged or self.jit == 'record':
            return False
        return self.jit == 'judge' or plan in S2_PLANS or self.s2_for(profile)

    def relaxation(self, profile):
        """The relaxation a verdict on `profile` applies: dc-i only on an S2 profile, only with the decision."""
        return 'dc-i' if self.policy == 'dc-i' and self.decision and self.s2_for(profile) else None

    def count_cache(self):
        if self.cache_entries is None:
            return None
        try:
            return self.cache_entries()
        except Exception:
            return None

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

    def run(self, arm, gate_args, timeout, profile=None, env=(), judged=False, measure=False):
        """One arm on --profile, or (an S2 arm) on `profile` with `env` added; `judged`: its kernel-cache
        growth is a problem (judges()); `measure`: the host reads its four-live rounds from the whole server
        log (report['c2_gate_live4']: both arms of a G3 pair read the same way)."""
        profile = profile or self.profile
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
        arguments = gate_run(self.image, name, profile, self.devices, self.checkout, arm_dir, gate_args, self.hub,
                             env=env)
        with open(os.path.join(arm_dir, 'docker-run.json'), 'w') as handle:
            json.dump(arguments, handle, indent=1)
        self.log('[C2-GATE] arm %s: %s%s%s' % (arm, ' '.join(gate_args),
                                                ' [profile %s]' % profile if profile != self.profile else '',
                                                ''.join(' [-e %s=%s]' % pair for pair in env)))
        cache_before = self.count_cache()
        started = time.time()
        status = self.execute(arguments, os.path.join(arm_dir, 'gate-stdout.log'), timeout, name)
        seconds = round(time.time() - started, 1)
        cache_after = self.count_cache()
        cache = dict(before=cache_before, after=cache_after, judged=judged,
                     added=None if cache_before is None or cache_after is None else cache_after - cache_before)
        with open(os.path.join(arm_dir, 'gate-stdout.log'), errors='replace') as handle:
            report = extract(handle.read())
        engines, consumers = [], []
        if report is not None:
            # What the server log says about C2-any, judged with the arm's own problems (arm_problems).
            log_text = server_log(arm_dir)
            problems, engines, consumers = any_request_check(log_text, self.any_request_for(profile))
            # S2: the harness's S2 problems, the S2 lines off the flag, the knobs reaching the container, the
            # kernel cache, and the four-live rate from the whole server log.
            problems += s2_log_check(log_text, self.s2_for(profile), env, report)
            if judged and cache['added']:
                problems.append('the kernel cache grew by %d entries during this arm: it compiled what M1 (warm) did '
                                'not (s2-design B6)' % cache['added'])
            if log_text is not None and (measure or self.s2_for(profile) or env):
                report['c2_gate_live4'] = host_live_rate(log_text)
            if cache['before'] is not None or self.s2_for(profile) or env:
                report['c2_gate_kernel_cache'] = cache
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
        if profile != self.profile or env or cache['before'] is not None:
            # S2 arms (and any counted cache): what served and what the kernel cache did.
            self.arms[arm].update(profile=profile, env=['%s=%s' % pair for pair in env], kernel_cache=cache)
            if cache['added'] is not None:
                self.log('[C2-GATE] arm %s: kernel cache %s -> %s entries (%+d%s)' % (
                    arm, cache['before'], cache['after'], cache['added'], ', judged' if judged else ''))
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


def spec_profile(runner, spec):
    """The profile an arm serves: its own (an S2 Arm's), or the gate's --profile."""
    return getattr(spec, 'profile', None) or runner.profile


def run_arm(runner, plan, spec, suffix=''):
    """Run one arm - a plain (name, args, timeout) or an Arm - with its profile, environment and kernel-cache
    judgement (Runner.judges); `suffix` names a re-run ('-rerun')."""
    name, args, timeout = spec
    profile = spec_profile(runner, spec)
    return runner.run(name + suffix, args, timeout, profile=getattr(spec, 'profile', None), env=getattr(spec, 'env', ()),
                      judged=runner.judges(plan, profile, getattr(spec, 'judged', True)), measure=plan in S2_PLANS)


def s2_of(report):
    return (report or {}).get('s2') or {}


def s2_g4_problems(label, report):
    """What fails an S2 serving arm beyond the harness's problems (s2-design M7): an idle commit, a round that
    went to refuse_round, and any prestage, pair-mask or fused-commit audit mismatch."""
    s2 = s2_of(report)
    if not s2:
        return []
    problems = []
    if s2.get('idle_commits'):
        problems.append('%s: %d "%s" lines' % (label, s2['idle_commits'], harness.PADDED_IDLE_COMMIT_MARKER))
    refused = s2.get('refused') or {}
    if refused.get('refused'):
        problems.append('%s: %d rounds went to refuse_round (%d requests aborted): a serving arm must serve every '
                        'round' % (label, refused['refused'], refused.get('aborted_lines') or 0))
    for name, count in sorted((s2.get('other_audits') or {}).items()):
        if count:
            problems.append('%s: %d %s audit mismatches' % (label, count, name))
    return problems


def memory_s2_problems(label, report):
    """G5 on an S2 profile (s2-design B4, B8; M8, M11): no DRAM hold and no refused request at four users or
    fewer, and the before-point floor at least FLOOR_GB per chip."""
    s2 = s2_of(report)
    if not s2:
        return []
    problems = []
    if s2.get('dram_holds'):
        problems.append('%s: %d "%s" lines at no more than four users: the block and four engines did not fit (%s)'
                        % (label, s2['dram_holds'], harness.DRAM_HOLD_MARKER, '; '.join(s2.get('dram_hold_lines') or [])))
    if s2.get('quarantined'):
        problems.append('%s: %d requests refused (RequestRefused, quarantined) at no more than four users (%s)' % (
            label, s2['quarantined'], '; '.join(s2.get('quarantined_lines') or [])))
    before = s2.get('before') or {}
    if before.get('floor_gb') is None:
        problems.append('%s: no ledger before-point carries an estimate: the floor cannot be judged' % label)
    elif before['floor_gb'] < FLOOR_GB:
        problems.append('%s: before-point floor %.3f GB per chip, below %.2f (%s)' % (
            label, before['floor_gb'], FLOOR_GB, (before.get('floor_point') or {}).get('detail')))
    return problems


def live4_of(report):
    """The four-live packed rounds of an arm: the host's reading of its whole log, else the harness's."""
    live = (report or {}).get('c2_gate_live4') or s2_of(report).get('live4')
    return live if live and 'error' not in live else None


def g4_checks(plan, concurrent, solo, rerun=None, probe=None):
    """(problems, shortfalls, facts) of a G4 plan beyond the policy and matrix_verdict (module docstring)."""
    problems, shortfalls = [], []
    arms = [('concurrent', concurrent), ('solo', solo)]
    if rerun is not None and None not in rerun:
        arms += [('concurrent re-run', rerun[0]), ('solo re-run', rerun[1])]
    for label, report in arms:
        if report is None:
            continue
        problems += s2_g4_problems(label, report)
        if plan == 'short':
            problems += memory_s2_problems(label, report)
            ladders = s2_of(report).get('ladders') or []
            wrong = sorted(set(ladder for ladder in ladders if ladder.replace(' ', '') != '[2048]'))
            if wrong or not ladders:
                problems.append('%s: proposal ladders %s, not the one 2048 bucket every engine builds under %s (W6a)'
                                % (label, wrong or 'unlogged', EXTENT_FLAG))
    s2 = s2_of(concurrent)
    live = live4_of(concurrent)
    facts = dict(live4=live, rounds=s2.get('rounds'))
    rate = (live or {}).get('net_per_user_tok_s')
    if rate is None:
        shortfalls.append('concurrent: no timed four-live packed round: the per-user rate is unmeasured')
    elif rate <= GENERAL_RATE:
        problems.append('concurrent: %.1f tok/s per user at four live (net of the audit), not above general\'s %.1f '
                        '(target %.0f)' % (rate, GENERAL_RATE, TARGET_RATE))
    rounds = s2.get('rounds') or {}
    if plan == 'mixed' and not rounds.get('multi_family_rounds'):
        shortfalls.append('concurrent: no packed round of two or more live families: mixed positions not exercised')
    if plan == 'boundaries':
        users = (s2.get('paths') or {}).get('users') or {}
        if not users:
            shortfalls.append('concurrent: no path records: cap events and boundary crossings unseen')
        for user, entry in sorted(users.items(), key=lambda item: int(item[0])):
            if not entry.get('cap_fields'):
                shortfalls.append('user %s: its [PACKED] lines carry no cap= (W4): cap events unseen' % user)
            elif not entry.get('cap_events'):
                shortfalls.append('user %s: no commit capped at a boundary' % user)
            if (entry.get('boundaries') or 0) < BOUNDARY_MIN_CROSSINGS:
                shortfalls.append('user %s crossed %s 256-key boundaries, fewer than %d' % (
                    user, entry.get('boundaries'), BOUNDARY_MIN_CROSSINGS))
    if plan == 'staggered':
        by_live = rounds.get('by_live') or {}
        if not (by_live.get('2') or by_live.get('3')):
            shortfalls.append('concurrent: no packed round at two or three live: the padded path not exercised')
        if probe is None:
            problems.append('probe: the arm left no gate report')
        else:
            problems += arm_problems('probe', probe)
            problems += s2_g4_problems('probe', probe)
            markers = probe.get('flag_markers') or {}
            problems += ['probe: %s' % line for line in markers.get('missing') or []
                         if line.startswith('QWEN_FAST_PADDED_PROBE')]
            if not markers.get('padded_probe'):
                shortfalls.append('probe: no padded probe line')
    return problems, shortfalls, facts


def with_checks(result, problems, shortfalls, facts=None):
    """A verdict with a plan's own problems (FAIL) and shortfalls (NOT_EXERCISED where it would pass)."""
    if problems:
        result['verdict'] = 'FAIL'
    elif shortfalls and result.get('verdict') == 'PASS':
        result['verdict'] = 'NOT_EXERCISED'
    result['s2_problems'], result['shortfalls'] = problems, shortfalls
    if facts:
        result['facts'] = facts
    result['lines'] = list(result.get('lines') or []) + ['problem: %s' % p for p in problems] + [
        'not exercised: %s' % s for s in shortfalls]
    return result


def run_matrix(plan, runner, arms, checks=None):
    """A concurrent arm against its solo reference under the policy, one re-run on a first divergence (matrix,
    and the G4 plans with `checks`)."""
    by_role = dict((getattr(spec, 'role', None) or ('concurrent', 'solo')[index], spec)
                   for index, spec in enumerate(arms[:2]))
    by_role.update((spec.role, spec) for spec in arms[2:])
    c_spec, s_spec = by_role['concurrent'], by_role['solo']
    wants = dict(concurrent=asked(c_spec[1]), solo=asked(s_spec[1]))
    relaxation = runner.relaxation(spec_profile(runner, c_spec))
    concurrent, solo = run_arm(runner, plan, c_spec), run_arm(runner, plan, s_spec)
    result = matrix_verdict(concurrent, solo, wants=wants, relaxation=relaxation)
    rerun = None
    if result['verdict'] == 'RERUN':
        runner.log('[C2-GATE] %s: a first divergence - re-running both arms once (the exactness policy)' % plan)
        rerun = (run_arm(runner, plan, c_spec, '-rerun'), run_arm(runner, plan, s_spec, '-rerun'))
        result = matrix_verdict(concurrent, solo, rerun if None not in rerun else None, wants=wants,
                                relaxation=relaxation)
        if None in rerun:
            result.update(verdict='FAIL', reason='a re-run arm left no gate report')
    if checks is None and runner.s2_for(spec_profile(runner, c_spec)):
        checks = 'matrix'
    if checks is not None:
        probe = run_arm(runner, plan, by_role['probe']) if 'probe' in by_role else None
        if concurrent is not None and solo is not None:
            problems, shortfalls, facts = g4_checks(checks, concurrent, solo, rerun, probe)
            result = with_checks(result, problems, shortfalls, facts)
    return result


def run_lifecycle(plan, runner, arms):
    """The event arms against the last (solo) arm, each divergent one re-run with the solo once; on an S2
    profile also every refuse_round request ended FINISHED_ABORTED (W5b) and, for lifecycle itself, at least
    one narrowed survivor across the event arms (D1 exercised, M6)."""
    reports = [(spec, run_arm(runner, plan, spec)) for spec in arms]
    s_spec, solo = reports[-1]
    solo_want = asked(s_spec[1])
    relaxation = runner.relaxation(spec_profile(runner, s_spec))
    arms_results = dict((spec[0], lifecycle_verdict(report, solo, want=asked(spec[1]), solo_want=solo_want,
                                                    relaxation=relaxation))
                        for spec, report in reports[:-1])
    again = [spec for spec, _ in reports[:-1] if arms_results[spec[0]]['verdict'] == 'RERUN']
    if again:
        runner.log('[C2-GATE] %s: a first divergence - re-running %s and the solo arm once (the exactness '
                   'policy)' % (plan, ', '.join(spec[0] for spec in again)))
        solo_again = run_arm(runner, plan, s_spec, '-rerun')
        first = dict((spec[0], report) for spec, report in reports)
        for spec in again:
            event_again = run_arm(runner, plan, spec, '-rerun')
            rerun = (event_again, solo_again) if event_again is not None and solo_again is not None else None
            arms_results[spec[0]] = lifecycle_verdict(first[spec[0]], solo, rerun, want=asked(spec[1]),
                                                      solo_want=solo_want, relaxation=relaxation)
            if rerun is None:
                arms_results[spec[0]].update(verdict='FAIL', reason='a re-run arm left no gate report')
    result = dict(verdict=worst([r['verdict'] for r in arms_results.values()]), arms=arms_results,
                  lines=['%s: %s' % (arm, line) for arm, r in arms_results.items() for line in r['lines']] +
                  ['%s %s' % (arm, r['verdict']) for arm, r in arms_results.items()])
    if runner.s2_for(spec_profile(runner, s_spec)):
        problems, shortfalls = [], []
        for spec, report in reports[:-1]:
            refused = s2_of(report).get('refused') or {}
            if refused and not refused.get('accounted', True):
                problems.append('%s: %d refuse_round rounds, but %d complete FINISHED_ABORTED groups: a refused '
                                'request was left to kill the engine (W5b)' % (spec[0], refused.get('refused'),
                                                                               refused.get('complete_groups')))
        narrowed = sum(s2_of(report).get('narrowed') or 0 for _, report in reports[:-1])
        if plan == 'lifecycle' and not narrowed:
            shortfalls.append('no "%s" line across the event arms: D1 was not exercised (M6: re-run)'
                              % harness.NARROWED_MARKER)
        result = with_checks(result, problems, shortfalls, dict(narrowed=narrowed))
    return result


def run_memory(plan, runner, arms):
    """G5: each arm judged alone (memory_verdict), the plan the worst of them; on an S2 profile with
    memory_s2_problems."""
    results = []
    for spec in arms:
        report = run_arm(runner, plan, spec)
        one = memory_verdict(report, want=asked(spec[1]))
        if report is not None and runner.s2_for(spec_profile(runner, spec)):
            one = with_checks(one, memory_s2_problems(spec[0], report), [],
                              dict(before=s2_of(report).get('before'), dram_holds=s2_of(report).get('dram_holds')))
        one['any_request_engines'] = (runner.arms.get(spec[0]) or {}).get('any_request_engines') or []
        results.append((spec[0], one))
    if len(results) == 1:
        return results[0][1]
    # memory-short on a C2-any profile: each arm judged alone, the plan the worst of them; every per-request
    # engine's proposal ladder recorded with its arm (R3).
    return dict(verdict=worst([one['verdict'] for _, one in results]), arms=dict(results),
                lines=['%s: %s' % (arm, line) for arm, one in results
                       for line in (one.get('lines') or []) + one['any_request_engines']] +
                ['%s %s' % (arm, one['verdict']) for arm, one in results])


def corpus_first(plan, runner, reference):
    """The image's corpus against the reference's before any card is opened: the mismatch, or None."""
    try:
        return corpus_mismatch(runner.corpus(), reference)
    except Exception as error:
        runner.log('[C2-GATE] %s WARNING: the image\'s corpus could not be read first (%s)' % (plan, error))
        return None


def run_reference_pairs(plan, runner, reference, arms):
    """control (G3) and forced-cap (Q5): every arm against v235 (bringup_verdict); control's flag-on arms timed
    against the flag-off arm of their pair; forced-cap's arms against each other and its caps."""
    mismatch = corpus_first(plan, runner, reference)
    if mismatch:
        return dict(verdict='NOT_COMPARABLE', reason=mismatch, lines=['corpus: %s' % mismatch])
    reports, results = {}, {}
    for spec in arms:
        report = reports[spec[0]] = run_arm(runner, plan, spec)
        verdict = bringup_verdict(report, reference, spec.profile, asked(spec[1]))
        if spec.role == 'on' and report is not None and not (s2_of(report).get('rounds') or {}).get('count'):
            verdict.update(verdict='FAIL')
            verdict['lines'] = list(verdict.get('lines') or []) + ['problem: the flag-on arm served no packed extent round']
        results[spec[0]] = verdict
    problems, shortfalls, pairs = [], [], []
    if plan == 'control':
        for index in sorted(set(spec.extra['pair'] for spec in arms)):
            off, on = live4_of(reports.get('control-off-%d' % index)), live4_of(reports.get('control-on-%d' % index))
            if off is None or on is None:
                shortfalls.append('pair %d: no timed four-live packed round in the %s arm: the round is not compared' % (
                    index, 'flag-off' if off is None else 'flag-on'))
                continue
            ratio = on['net_median_round_ms'] / off['median_round_ms']
            pairs.append(dict(pair=index, off_ms=off['median_round_ms'], on_ms=on['median_round_ms'],
                              on_net_ms=on['net_median_round_ms'], on_audit_ms=on.get('median_audit_ms'),
                              ratio=round(ratio, 4), raw_ratio=round(on['median_round_ms'] / off['median_round_ms'], 4)))
            if ratio > CONTROL_TOLERANCE:
                problems.append('pair %d: the flag-on median four-live round, %.2f ms net of its audit, is %.4f x the '
                                'flag-off %.2f ms, past %.2f' % (index, on['net_median_round_ms'], ratio,
                                                                 off['median_round_ms'], CONTROL_TOLERANCE))
    else:
        off, on = reports.get('forced-cap-off'), reports.get('forced-cap-on')
        if off is not None and on is not None:
            compared = real_text_compare.policy_pass(on, off)
            for user in compared['users']:
                if user['verdict'] != 'IDENTICAL':
                    problems.append('user %s: the capped arm is %s against the uncapped at character %s: commit '
                                    'granularity changed the text (Q5)' % (user['user'], user['verdict'],
                                                                          user.get('first_divergence')))
            caps = (s2_of(on).get('paths') or {}).get('users') or {}
            seen = [entry for entry in caps.values() if entry.get('cap_fields')]
            if not seen:
                shortfalls.append('the capped arm\'s [PACKED] lines carry no cap= (W4): the forced cap is unseen')
            for user, entry in sorted(caps.items(), key=lambda item: int(item[0])):
                if entry.get('max_cap') is not None and entry['max_cap'] > FORCE_CAP:
                    problems.append('user %s: a packed commit capped at %d under %s=%d' % (
                        user, entry['max_cap'], harness.FORCE_CAP_FLAG, FORCE_CAP))
    verdicts = [result['verdict'] for result in results.values()]
    result = dict(verdict=worst(verdicts), arms=results, pairs=pairs,
                  lines=['%s: %s' % (arm, line) for arm, one in results.items() for line in one.get('lines') or []] +
                  ['%s %s' % (arm, one['verdict']) for arm, one in results.items()] +
                  ['pair %(pair)d: flag-off %(off_ms)s ms, flag-on %(on_ms)s ms (net %(on_net_ms)s, audit '
                   '%(on_audit_ms)s): %(ratio)s x' % entry for entry in pairs])
    return with_checks(result, problems, shortfalls)


def below_pair(family, off_spec, off, on_spec, on):
    """One G3b pair at family F: both arms served as asked, the flag-off block served packed rounds inside F
    (else NO_DECISION, Q22), the flag-on arm served packed extent rounds, and every user IDENTICAL across them."""
    problems, shortfalls = [], []
    if off is None or on is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    problems += arm_problems('flag-off', off, asked(off_spec[1])) + arm_problems('flag-on', on, asked(on_spec[1]))
    low, high = family - 256, family - 16
    packed, outside = 0, []
    for user, entry in sorted(((s2_of(off).get('paths') or {}).get('users') or {}).items()):
        for record in real_text_compare.decode_paths(entry.get('encoded')):
            if record['path'] == 'P':
                packed += 1
                if record['position'] is not None and not low <= record['position'] <= high:
                    outside.append('user %s round %d at %d' % (user, record['round'], record['position']))
    if outside:
        problems.append('flag-off: packed rounds outside family %d (%s)' % (family, '; '.join(outside[:3])))
    if not (s2_of(on).get('rounds') or {}).get('count'):
        shortfalls.append('flag-on: no packed extent round')
    compared = real_text_compare.policy_pass(on, off)
    texts = [user['verdict'] for user in compared['users']]
    for user in compared['users']:
        if user['verdict'] != 'IDENTICAL':
            problems.append('user %s: flag-on is %s against flag-off at character %s (%s)' % (
                user['user'], user['verdict'], user.get('first_divergence'), user.get('reason') or user.get('detail')))
    if problems:
        verdict = 'FAIL'
    elif not packed:
        verdict = 'NO_DECISION'
        shortfalls.append('flag-off: the block served no packed round at family %d (Q22): G3b decides nothing here; '
                          'fall back to CB2b R2 and the audit' % family)
    elif shortfalls:
        verdict = 'NOT_EXERCISED'
    else:
        verdict = 'PASS'
    return dict(verdict=verdict, family=family, users=texts, flag_off_packed_rounds=packed, problems=problems,
                shortfalls=shortfalls, lines=['problem: %s' % p for p in problems] + [
                    'not exercised: %s' % s for s in shortfalls] + ['users %s; flag-off packed rounds %d' % (texts, packed)])


def run_below(plan, runner, arms):
    """control-below (G3b): each family's pairs in order, each pair judged alone (below_pair)."""
    reports = dict((spec[0], (spec, run_arm(runner, plan, spec))) for spec in arms)
    pairs = {}
    for spec in arms:
        if spec.role != 'off':
            continue
        family, index = spec.extra['family'], spec.extra['pair']
        off_spec, off = reports['below-%d-off-%d' % (family, index)]
        on_spec, on = reports['below-%d-on-%d' % (family, index)]
        pairs['%d-%d' % (family, index)] = below_pair(family, off_spec, off, on_spec, on)
    return dict(verdict=worst([pair['verdict'] for pair in pairs.values()]), pairs=pairs,
                lines=['family %s: %s' % (key, line) for key, pair in pairs.items() for line in pair['lines']] +
                ['family %s %s' % (key, pair['verdict']) for key, pair in pairs.items()])


def run_warm(plan, runner, arms):
    """M1: every arm must complete as asked; the kernel cache's growth is recorded, never judged."""
    results = {}
    for spec in arms:
        report = run_arm(runner, plan, spec)
        problems = arm_problems(spec[0], report, asked(spec[1])) if report is not None else ['%s: no gate report' % spec[0]]
        results[spec[0]] = dict(verdict='FAIL' if problems else 'PASS', problems=problems,
                                kernel_cache=(runner.arms.get(spec[0]) or {}).get('kernel_cache'))
    return dict(verdict=worst([one['verdict'] for one in results.values()]), arms=results,
                lines=['%s: problem: %s' % (arm, p) for arm, one in results.items() for p in one['problems']] +
                ['%s %s kernel cache %s' % (arm, one['verdict'], one['kernel_cache']) for arm, one in results.items()])


def run_churn(plan, runner, profiles, arms):
    """M11 (G5 churn): no engine death, no hold or refusal, the floor, and dead proposal traces released."""
    spec, = arms
    report = run_arm(runner, plan, spec)
    if report is None:
        return dict(verdict='FAIL', reason='the arm left no gate report', lines=[])
    problems = arm_problems('churn', report, asked(spec[1])) + memory_s2_problems('churn', report)
    if not report.get('alive'):
        problems.append('churn: the engine did not answer every seat after the streams')
    shortfalls = []
    s2 = s2_of(report)
    releases = s2.get('releases') or {}
    users = asked(spec[1])['streams']
    replacements = users - profile_seats(profiles, spec_profile(runner, spec))
    if not s2.get('quads_built'):
        shortfalls.append('no quad was built: no departure met a formed quad')
    elif not releases.get('lines') or not releases.get('quad'):
        problems.append('%d "[PACKED-PROPOSE] released" lines (%d with a quad) over %d replacements: dead proposal '
                        'traces were not released at detach (W6c)' % (releases.get('lines') or 0,
                                                                        releases.get('quad') or 0, replacements))
    facts = dict(releases=releases, replacements=replacements, quads_built=s2.get('quads_built'),
                 before=s2.get('before'), trace_region_lines=s2.get('trace_region_lines'))
    result = dict(verdict='PASS', lines=['releases %s over %d replacements; trace region %s' % (
        releases, replacements, s2.get('trace_region_lines') or 'not logged (Q18)')])
    return with_checks(result, problems, shortfalls, facts)


def run_permuted(plan, runner, arms):
    """The permuted arm (A4; required only under D-c(i)): two concurrent arms, the second admitted in reverse;
    the users whose every round took the same path in both (SAME_PATH) must match, and there must be two."""
    forward_spec, reverse_spec = arms
    forward, reverse = run_arm(runner, plan, forward_spec), run_arm(runner, plan, reverse_spec)
    if forward is None or reverse is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    problems = arm_problems('forward', forward, asked(forward_spec[1])) + arm_problems('reverse', reverse,
                                                                                       asked(reverse_spec[1]))
    shortfalls = []
    same = real_text_compare.same_path_users(forward, reverse)
    if same is None:
        shortfalls.append('an arm carries no path records: SAME_PATH users cannot be counted')
        same = []
    elif len(same) < 2:
        shortfalls.append('%d SAME_PATH users, fewer than two: NOT_EXERCISED (A4)' % len(same))
    streams_f, streams_r = forward.get('streams') or [], reverse.get('streams') or []
    for user in same:
        compared = real_text_compare.compare_user(streams_f[user] if user < len(streams_f) else None,
                                                  streams_r[user] if user < len(streams_r) else None)
        if compared['verdict'] != 'exact':
            problems.append('user %d took the same path in both arms but is %s at character %s: a same-path '
                            'divergence FAILS under any policy' % (user, compared['verdict'],
                                                                   compared.get('first_divergence')))
    result = dict(verdict='PASS', same_path_users=same,
                  lines=['SAME_PATH users %s (required only under decision D-c(i))' % same])
    return with_checks(result, problems, shortfalls)


def run_s2_plan(plan, runner, profiles, reference, arms):
    """The S2 plans (module docstring's S2 PLANS)."""
    if plan in ('control', 'forced-cap'):
        return run_reference_pairs(plan, runner, reference, arms)
    if plan == 'control-below':
        return run_below(plan, runner, arms)
    if plan in WARM_PLANS:
        return run_warm(plan, runner, arms)
    if plan == 'lifecycle-arrival':
        return run_lifecycle(plan, runner, arms)
    if plan == 'churn':
        return run_churn(plan, runner, profiles, arms)
    if plan == 'permuted':
        return run_permuted(plan, runner, arms)
    return run_matrix(plan, runner, arms, checks=plan)


def divergence_records(result):
    """Every divergence record a plan's policies carry (exactness_policy's 'records'), as (where, record)."""
    found = []
    policy = result.get('policy') or {}
    found += [('', record) for record in policy.get('records') or []]
    for name, one in sorted((result.get('arms') or {}).items()):
        found += [(name, record) for record in ((one or {}).get('policy') or {}).get('records') or []]
    return found


def run_plan(plan, runner, profiles, reference=None, lengths=None, max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS,
             memory_prompt=None, s2=None):
    notes = []
    arms = plan_arms(plan, runner.profile, profiles, lengths, max_tokens, memory_prompt, notes, s2)
    for note in notes:
        runner.log('[C2-GATE] note: %s' % note)
    if plan in S2_PLANS:
        result = run_s2_plan(plan, runner, profiles, reference, arms)
    elif plan == 'bringup':
        warning = bringup_warning(profiles, runner.profile)
        if warning:
            runner.log('[C2-GATE] bringup WARNING: %s' % warning)
        spec, = arms
        try:
            mismatch = corpus_mismatch(runner.corpus(), reference)
        except Exception as error:
            mismatch = None
            runner.log('[C2-GATE] bringup WARNING: the image\'s corpus could not be read first (%s)' % error)
        if mismatch:
            result = dict(verdict='NOT_COMPARABLE', reason=mismatch, lines=['corpus: %s' % mismatch])
        else:
            result = bringup_verdict(run_arm(runner, plan, spec), reference, runner.profile, asked(spec[1]))
        if warning:
            result['warning'] = warning
    elif plan == 'memory':
        result = run_memory(plan, runner, arms)
    elif plan == 'lifecycle':
        result = run_lifecycle(plan, runner, arms)
    else:
        result = run_matrix(plan, runner, arms)
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
    parser.add_argument('--plan', default='bringup', help='comma-separated: %s' % ', '.join(c2_serving_job.ALL_GATE_PLANS))
    parser.add_argument('--results', required=True)
    parser.add_argument('--lengths', default=None, help='matrix prompt lengths (default: the G4 ladder, its top '
                                                        'rung lowered to what the profile admits); an S2 plan\'s '
                                                        'set in place of its default')
    parser.add_argument('--max-tokens', type=int, default=c2_serving_job.DEFAULT_MAX_TOKENS)
    parser.add_argument('--memory-prompt', type=int, default=None)
    parser.add_argument('--budget-seconds', type=int, default=None,
                        help='refuse a plan list whose worst case (every arm to its limit, every re-run) is longer')
    parser.add_argument('--reference', default=REFERENCE)
    parser.add_argument('--checkout', default=os.path.dirname(os.path.dirname(HERE)))
    parser.add_argument('--profiles', default=None, help='a profiles JSON instead of the image\'s own')
    parser.add_argument('--hub', default=HUB)
    parser.add_argument('--dry-run', action='store_true', help='print every arm\'s docker argv and run nothing')
    # S2 (the module docstring's S2 PLANS; c2_serving_job's C2_GATE_* keys).
    parser.add_argument('--pairs', type=int, default=None, help='control and control-below A/B pairs (default %d)'
                                                                % DEFAULT_PAIRS)
    parser.add_argument('--families', default=None, help='control-below families (default %s)'
                                                          % ','.join(str(family) for family in BELOW_FAMILIES))
    parser.add_argument('--jit', choices=JIT_MODES, default='auto',
                        help='whose kernel-cache growth fails an arm: auto (S2 arms and plans), judge (every arm but '
                             'warm), record (none)')
    parser.add_argument('--policy', choices=c2_serving_job.POLICIES, default='strict',
                        help='strict (default), or dc-i: the user\'s decision D-c(i) (needs --policy-decision)')
    parser.add_argument('--policy-decision', default=None, help='where the user\'s D-c decision is recorded')
    parser.add_argument('--audits', choices=c2_serving_job.AUDIT_SETS, default='extent',
                        help='extent (the extent audit on every S2 arm) or all (also prestage, pair-mask and '
                             'fused-commit on the G4 arms)')
    return parser


def s2_options(options, log):
    """The S2 options as plan_arms takes them, or None (logged) when they are refused."""
    try:
        if options.pairs is not None and not 1 <= options.pairs <= c2_serving_job.MAX_PAIRS:
            raise c2_serving_job.JobError('--pairs must be 1..%d, got %d' % (c2_serving_job.MAX_PAIRS, options.pairs))
        families = [c2_serving_job.below_family('--families', part)
                    for part in c2_serving_job.split_list(options.families)] if options.families else None
    except c2_serving_job.JobError as error:
        log('refused: %s' % error)
        return None
    if options.policy == 'dc-i' and not (options.policy_decision or '').strip():
        log('refused: --policy dc-i needs --policy-decision: only the user\'s D-c decision relaxes the strict policy')
        return None
    return dict(pairs=options.pairs, families=families, audits=options.audits)


def write_records(results, plan, result):
    """Every divergence record of a plan to <results>/<plan>-divergence-records.json, and the S2 exit blockers
    among them: a NOT_COMPARABLE or UNSTABLE user blocks S2 exit until its record is read (s2-design W11)."""
    found = divergence_records(result)
    if not found:
        return []
    path = os.path.join(results, '%s-divergence-records.json' % plan)
    with open(path, 'w') as handle:
        json.dump([dict(record, arm=where) for where, record in found], handle, indent=2)
    return [dict(plan=plan, arm=where, user=record.get('user'), verdict=record.get('verdict'), record=path)
            for where, record in found if record.get('verdict') in ('NOT_COMPARABLE', 'UNSTABLE')]


def main(argv=None, execute=None, devices=None, log=print, containers=None, corpus=None, cache_entries=None):
    options = build_parser().parse_args(argv)
    plans = c2_serving_job.split_list(options.plan)
    unknown = sorted(set(plans) - set(c2_serving_job.ALL_GATE_PLANS))
    if not plans or unknown:
        log('unknown plan(s): %s' % ', '.join(unknown or ['(none)']))
        return 2
    s2 = s2_options(options, log)
    if s2 is None:
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
            arms_of[plan] = plan_arms(plan, options.profile, profiles, lengths, options.max_tokens, options.memory_prompt,
                                      s2=s2)
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
            for spec in arms_of[plan]:
                arm, args, timeout = spec
                log(json.dumps(dict(arm=arm, timeout=timeout, docker=gate_run(
                    options.image, CONTAINER_PREFIX + arm, getattr(spec, 'profile', None) or options.profile,
                    devices or ['<M>', '<A>'], options.checkout, os.path.join(options.results, arm), args, options.hub,
                    env=getattr(spec, 'env', ())))))
        return 0
    cache_dir = None
    if cache_entries is None and execute is None:
        # The kernel cache the image's arms read and write (B6), counted before and after every arm.
        cache_dir = image_kernel_cache(options.image, options.hub)
        if cache_dir is not None:
            cache_entries = lambda: count_entries(cache_dir)
        else:
            log('[C2-GATE] note: the image\'s kernel cache (%s) is not on the hub: kernel-cache growth is not counted'
                % KERNEL_CACHE_ENV)
    runner = Runner(options.image, options.profile, options.results, options.checkout,
                    devices if devices is not None else serving_pair(), options.hub, execute, log, containers, corpus,
                    any_request=any_request_profile(profiles, options.profile), profiles=profiles,
                    cache_entries=cache_entries, jit=options.jit, policy=options.policy,
                    decision=options.policy_decision)
    context, ceiling, room = profile_limits(profiles, options.profile)
    summary = dict(image=options.image, profile=options.profile, plans=plans, context=context,
                   output_ceiling=ceiling, largest_prompt=room, worst_case_seconds=worst_case,
                   budget_seconds=options.budget_seconds, results={}, arms=runner.arms)
    s2_run = any(plan in S2_PLANS for plan in plans) or s2_profile(profiles, options.profile)
    if s2_run:
        summary.update(policy=options.policy, policy_decision=options.policy_decision, jit=options.jit,
                       kernel_cache=cache_dir, s2_exit_blockers=[])
    try:
        for plan in plans:
            summary['results'][plan] = run_plan(plan, runner, profiles, reference, lengths, options.max_tokens,
                                                options.memory_prompt, s2=s2)
            blockers = write_records(options.results, plan, summary['results'][plan])
            if blockers:
                summary.setdefault('s2_exit_blockers', []).extend(blockers)
                for blocker in blockers:
                    log('[C2-GATE] S2 EXIT BLOCKED: %(plan)s %(arm)s user %(user)s is %(verdict)s - its record is '
                        '%(record)s' % blocker)
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
