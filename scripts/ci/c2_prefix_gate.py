"""The prefix-reuse (general-prefix, G1) gates, run in the node agent's container shape (qwen-c2-serving.yml 'prefix').

    python3 scripts/ci/c2_prefix_gate.py --image zot.thatch.local:5000/tt-vllm:qwen38-c2-<tag> \
        --profile general-prefix --plan bringup --results "$RUNNER_TEMP/c2-results/prefix" [--budget-seconds S]

TT prefix-reuse design, section 2.2 (G1): the gate table's Bring-up, Exactness (traced and eager),
Lifecycle and Timing rows. Each ARM is one serving container started exactly as the S1 gate shapes
it (c2_serving_gate.agent_shape: read-only root, the agent's tmpfs set, 8 CPUs, 80g, 4g shm, cards M
then A by board id, /models = ~/hf-cache/hub, the platform's env, QWEN_C2_SERVING=1), detached with
its API on 127.0.0.1:PORT and launched from the PLATFORM's vLLM argv (--no-enable-prefix-caching
included, which the contract must drop), so what serves is the image's contract argv. The driver
(prefix_replay) then talks to it over the OpenAI API from this host, with real-text multi-turn
coding-agent conversations (prefix_agent_corpus), and reads the engine's own markers from the
container log (prefix_markers). An arm that needs a knob the image's profile does not set runs a
DERIVED profile - the image's own profile with that one change, written to the arm's directory and
mounted read-only at DERIVED_MOUNT with QWEN_C2_PROFILES pointing at it (serving_c2_contract.boot
reads that variable) - and the knob's effect is checked in the log, never assumed.

PLANS (--plan, comma-separated, run in order; each exactness and lifecycle arm is also a plan of its own,
named as the arm - c2_serving_job.PREFIX_ARM_PLANS - that runs only it, judged exactly as inside its plan:
--plan exactness-eager; an arm never runs twice in one invocation):
  bringup    bringup-reference on the BASELINE profile (general): a three-turn real-text conversation;
             bringup-prefix on the prefix profile: each turn UNSALTED (fail-closed tenancy leaves reuse
             off inside a reuse engine: byte-identical to the reference, no grant, and vLLM's
             prefix_cache_hits unchanged across it - nothing was published), then the same messages
             under a FRESH salt (a first request that captures at floor2048(L) mid-prefill: it must
             equal the reference byte for byte too, show captured=[floor2048(L)], match the unsalted
             row's slot and logits digests, and - the first one - compile nothing after the unsalted
             row of the same prompt), then salted cold/hit pairs: the first hit. PASS also needs the
             launched argv with the prefix cache on and async scheduling off at block size 64, the
             platform's 'Automatic prefix caching is enabled' and 'Chunked prefill is not supported
             ... disabling it', '[PINDIAG] prefix: install' (TTScheduler, block_size=64,
             QWEN_SDPA_BF8=1), '[TP chunk-replay]', no program compiled by any hit after its cold twin
             (F3; every arm checks it), and the anchor probe: every file the image's graft.sha256
             pins at its pin, model.py and qwen36_vllm.py at the prefix stage's pins (the bytes
             qwen_prefix_model_patch stages, PATCHED_SHA256: graft.sha256 pins neither), and a
             '[PREFIX]' marker in one of them or a pinned file. G2's DRAM reading - the model graft's
             '[PINDIAG] dram after registry' with per-chip figures, and 'dram after first capture'
             once the arm stored a checkpoint - missing or 'unavailable' makes it NOT_EXERCISED.
  exactness  exactness-traced: a chained real-text conversation 2.6k -> 60k (hits near 4k, 9k, 16k,
             24k, 33k, 42k, 51k, 60k), every turn cold (fresh salt) vs hit compared in full, output
             tokens and the rows' slot and logits digests; a changed suffix and an early divergence
             forked at turn 3; previous prompts SERVED at exactly 2047/2048/2049 tokens with tail-only
             hits; one tenant's three conversations sharing the full system block, each with its own
             2.6k-token task: the second captures the gap boundary and the third restores exactly it;
             every Q, h and capture plan against the oracle (prefix_judge.Oracle).
             exactness-audit: the short chain and the boundaries with QWEN_PREFIX_AUDIT=1 (derived):
             each hit's KV [0,L) and GDN slot digests equal its cold twin's. exactness-eager: the same
             with trace_mode decode_only (derived): the eager loop C1 reuses, rows path=eager, no
             '[TP chunk-replay]', and the model graft's '[PINDIAG] prefix: eager prefill warmed' line
             (the eager prefill compiled before the decode trace is parked: G1 v47, run 36246961161,
             ran without it, compiled 133 programs in its first row and hung the device on the next
             prefill). On every exactness arm a [PREFIX] row whose program cache grew is a FAIL (F3:
             a compile after the traces are parked, the second-request hang #48536).
  lifecycle  lifecycle-evict (VLLM_SERVER_DEV_MODE=1, derived): four arrivals at once (two hits, two
             same-tenant first turns: the same-step rule), an abort while waiting for a seat, an abort
             during a hit's prefill, a KV flood that evicts conversations built to ~56k (eviction
             coupling), reset_prefix_cache, the kill switch file (grants off, latched, and nothing
             published under it: the next turn's raw vLLM hit is bounded by what was published
             before), an in-place restart (docker stop/start: the first turn after it misses, the next
             hits). lifecycle-store (QWEN_PREFIX_STORE_GIB=0.5, derived): checkpoint LRU eviction.
             lifecycle-tiny (QWEN36_MAX_TOKENS_ALL_USERS sized for a 1280-block pool, derived; the TT
             worker overwrites num-gpu-blocks-override, plugin worker.py:388-390), sized by /tokenize
             in blocks: an allocation failure after a grant (a cached conversation's next turn cannot
             fit beside a filler, waits with a staged Q > 0 grant, then restores Q > 0 once the filler
             ends), then preemption (three ignore_eos answers outgrow the pool); it sends nothing
             unless vLLM logs exactly that pool. Every hit against a cold twin; a request vLLM preempted and resumed
             (a second [PREFIX] row) may diverge after it resumes and is reported by name, not failed;
             concurrent hits run a batch control before any divergence counts. Each arm requires the
             registry's stats export (prefix_markers.REQUIRED_STATS) for what only it can show.
  timing     timing-prefix and (unless --baseline none) timing-baseline: busy agents in the metering
             shape (~2k-token tool results, exponential 15 s gaps, full system block, compaction past
             60k) at --agents 1,4,5,6, one phase each; TTFT and turn time p50/p90, hit rate from the
             [PREFIX] rows' L-Q (never vllm:prefix_cache_hits), trim loss, restore/capture ms, RSS and
             the CI pod count per phase (prefix_report). It records; it fails only on failed turns
             or missing markers.

Verdicts per plan: PASS, FAIL, INFRA (a platform container on M+A, tt-metal's ethernet-core wedge),
NOT_COMPARABLE (a preempted request, or an engine whose bytes depend on the batch), UNSTABLE (two
cold runs of one prompt disagree), NOT_EXERCISED (a check whose event did not happen, or whose
evidence was not logged). A hit that diverges from two agreeing cold runs is a FAIL.

SAFETY: as c2_serving_gate - no thatch-inference-* container may exist, a leftover container of the
same name is removed first, every container is removed however the arm ends (SIGTERM included), a
wedge stops the plan list as INFRA. The kill-switch file is written through the container into the
host's ~/hf-cache/hub/.qwen-c2 and removed only if its content is KILL_SWITCH_OWNER (the workflow's
trap does the same from the host). --budget-seconds refuses a plan list whose worst case does not
fit, and no request or readiness wait runs past its arm's deadline.

Writes <results>/<arm>/ (server.log, records.jsonl, pairs.json, events.json, arm.json, docker-run.json,
profiles.json for a derived arm) and <results>/c2-prefix-summary.json; exits 0 only when every plan
passed, 2 when refused up front.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import binascii
import copy
import json
import os
import signal
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_serving_gate as gate  # noqa: E402
import c2_serving_job  # noqa: E402
import prefix_agent_corpus as corpus_module  # noqa: E402
import prefix_judge as judge  # noqa: E402
import prefix_markers as markers  # noqa: E402
import prefix_replay as replay  # noqa: E402
import prefix_report as report  # noqa: E402
import qwen_prefix_model_patch as model_patch  # noqa: E402

PLANS = c2_serving_job.PREFIX_PLANS + tuple(arm for arm, _ in c2_serving_job.PREFIX_ARM_PLANS)
PORT = 8021
CONTAINER_PREFIX = 'qwen-c2-prefix-'
DERIVED_MOUNT = '/prefix-gate/profiles.json'
READINESS_SECONDS = 1800
ARM_OVERHEAD_SECONDS = 180
SETTLE_SECONDS = 3.0
MAX_LISTED = 16
# 1280 blocks: above one 65,536-token request, small enough to build a failed allocation and a preemption.
TINY_BLOCKS = replay.TINY_POOL_TOKENS // judge.BLOCK
SMALL_STORE_GIB = 0.5         # three checkpoints of CHECKPOINT_NBYTES
MODEL_ROOT = '/opt/tt-metal/models/demos/blackhole/qwen36/tt'
GRAFT_PINS = '/opt/qwen-c2/graft.sha256'
ANCHOR_FILES = ('model.py', 'qwen36_vllm.py')
# The anchor files are the prefix stage's, not the C2 graft's: the image build runs qwen_prefix_model_patch
# (qwen_prefix_stage STAGES), which writes nothing but these bytes, and graft.sha256 pins only the files
# the C2 graft lays. A served anchor file off its stage pin is an image that predates this checkout's
# model graft, or a tree changed after the stage.
STAGE_PINS = dict(model_patch.PATCHED_SHA256)
KILL_SWITCH_OWNER = replay.KILL_SWITCH_OWNER
# What the platform hands vLLM (the smoke step's argv; TS injects --no-enable-prefix-caching for TT).
PLATFORM_ARGS = ['--model', 'Qwen/Qwen3.8-27B', '--served-model-name', replay.SERVED_NAME, '--host', '0.0.0.0',
                 '--port', '8000', '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_xml',
                 '--enable-auto-tool-choice', '--max-model-len', '65536', '--max-num-seqs', '2', '--block-size', '64',
                 '--no-enable-prefix-caching', '--additional-config',
                 json.dumps({'tt': {'l1_small_size': 24576, 'fabric_config': 'FABRIC_1D',
                                    'trace_region_size': 1073741824}})]


def tiny_pool_env(profile, pool=replay.TINY_POOL_TOKENS):
    """QWEN36_MAX_TOKENS_ALL_USERS for a pool of `pool` tokens: the TT worker adds one block per
    sequence of padding (plugin worker.py:538-539) and rounds up to blocks (:573), and the profile's
    engine flags win over the platform's (the contract owns max-num-seqs and block-size)."""
    engine = profile.get('engine') or {}
    return pool - int(engine.get('block-size', judge.BLOCK)) * int(engine.get('max-num-seqs', 1))


# kind -> the profile's changes: env, engine flags, additional-config tt keys (or a function of the profile).
DERIVED = dict(
    audit=dict(env=dict(QWEN_PREFIX_AUDIT='1')),
    eager=dict(tt=dict(trace_mode='decode_only')),
    dev=dict(env=dict(VLLM_SERVER_DEV_MODE='1')),
    store=dict(env=dict(QWEN_PREFIX_STORE_GIB=str(SMALL_STORE_GIB))),
    tiny=lambda profile: dict(env=dict(QWEN36_MAX_TOKENS_ALL_USERS=str(tiny_pool_env(profile)),
                                       VLLM_SERVER_DEV_MODE='1')),
)
# (arm, scenario, which profile, derived changes, docker timeout seconds, strict oracle)
PLAN_ARMS = dict(
    bringup=(('bringup-reference', 'bringup_reference', 'baseline', None, 2400, True),
             ('bringup-prefix', 'bringup_prefix', 'prefix', None, 3600, True)),
    exactness=(('exactness-traced', 'exactness_traced', 'prefix', None, 9000, True),
               ('exactness-audit', 'exactness_audit', 'prefix', 'audit', 5400, True),
               ('exactness-eager', 'exactness_eager', 'prefix', 'eager', 5400, True)),
    lifecycle=(('lifecycle-evict', 'lifecycle_evict', 'prefix', 'dev', 9000, False),
               ('lifecycle-store', 'lifecycle_store', 'prefix', 'store', 3600, False),
               ('lifecycle-tiny', 'lifecycle_tiny', 'prefix', 'tiny', 4800, False)),
    timing=(('timing-prefix', 'timing', 'prefix', None, 9000, False),
            ('timing-baseline', 'timing', 'baseline', None, 9000, False)),
)
# The single-arm plans (c2_serving_job.PREFIX_ARM_PLANS): the arm exactly as its plan runs it.
PLAN_ARMS.update([(arm, tuple(entry for entry in PLAN_ARMS[plan] if entry[0] == arm))
                  for arm, plan in c2_serving_job.PREFIX_ARM_PLANS])


class PlanError(ValueError):
    """A plan the image's profiles cannot run as asked; refused before any container starts."""


def is_prefix_profile(profile):
    return (str((profile.get('env') or {}).get('QWEN_PREFIX_REUSE', '0')) == '1'
            and (profile.get('engine') or {}).get('enable-prefix-caching') is True)


def derive(profiles, base, kind):
    """(derived name, a profiles document holding only it): the image's `base` profile with the
    DERIVED[kind] changes (profile env, engine flags, additional-config tt keys)."""
    profile = copy.deepcopy(profiles['profiles'][base])
    changes = DERIVED[kind]
    if callable(changes):
        changes = changes(profile)
    profile.setdefault('env', {}).update(changes.get('env') or {})
    profile.setdefault('engine', {}).update(changes.get('engine') or {})
    if changes.get('tt'):
        additional = profile['engine'].setdefault('additional-config', {})
        additional.setdefault('tt', {}).update(changes['tt'])
    profile['description'] = '%s, derived by c2_prefix_gate for the %s arm: %s' % (base, kind, json.dumps(changes))
    name = '%s+%s' % (base, kind)
    return name, dict(default=name, profiles={name: profile})


def plan_arms(plan, profile, baseline, profiles):
    """[dict(arm, scenario, served, derived, timeout, strict)] for one plan, or PlanError."""
    if plan not in PLAN_ARMS:
        raise ValueError('unknown plan %r' % plan)
    names = profiles['profiles']
    if profile not in names:
        raise PlanError('profile %r is not in the image\'s profiles (%s)' % (profile, ', '.join(sorted(names))))
    if not is_prefix_profile(names[profile]):
        raise PlanError('profile %r is not a prefix-reuse profile: it needs env QWEN_PREFIX_REUSE=1 and engine '
                        'enable-prefix-caching true' % profile)
    arms = []
    for arm, scenario, which, kind, timeout, strict in PLAN_ARMS[plan]:
        if which == 'baseline':
            if not baseline or baseline == 'none':
                if plan == 'bringup':
                    raise PlanError('the bring-up compares against a baseline profile; --baseline none leaves it '
                                    'nothing to compare')
                continue
            if baseline not in names:
                raise PlanError('baseline profile %r is not in the image\'s profiles' % baseline)
            if is_prefix_profile(names[baseline]):
                raise PlanError('baseline profile %r turns prefix reuse on: it cannot be the no-reuse reference'
                                % baseline)
            served, derived = baseline, None
        elif kind:
            served, derived = derive(profiles, profile, kind)
        else:
            served, derived = profile, None
        arms.append(dict(arm=arm, scenario=scenario, served=served, derived=derived, timeout=timeout, strict=strict,
                         prefix=which == 'prefix', kind=kind))
    return arms


def worst_case_seconds(arms_of):
    return sum(arm['timeout'] + ARM_OVERHEAD_SECONDS for arms in arms_of.values() for arm in arms)


# The model graft's per-row slot and logits digests (qwen_prefix_model_patch): a gate instrument, not a
# profile knob, so the image's own profile serves; on every prefix arm but timing, which measures TTFT.
DIGESTS_ENV = 'QWEN_PREFIX_DIGESTS=1'


# The registry's counters (qwen_prefix_registry.StatsExport) go out at most every QWEN_PREFIX_STATS_S (30 s by
# default) and only from a schedule() call, so an engine that falls idle right after an event would leave it
# unexported: every prefix arm exports on every step instead (a tmpfs file write, a log line only on change).
STATS_NOW_ENV = 'QWEN_PREFIX_STATS_S=0'


def wants_digests(arm):
    return bool(arm.get('prefix')) and arm.get('scenario') != 'timing'


# The salt key a prefix arm mounts (serving_c2_contract.SALT_KEY_ENV names it in the container): the image
# keeps a cache_salt only when it verifies against its key, so the gate mints every salt under a key of
# its own (prefix_replay.mint_salt) - never the operator's /models/.qwen-c2/prefix-salt.key.
SALT_KEY_MOUNT = '/prefix-gate/salt.key'
SALT_KEY_ENV = 'QWEN_PREFIX_SALT_KEY_FILE'


def write_salt_key(path, urandom=os.urandom):
    """A fresh 64-character key at `path` (the contract reads it stripped, at least 32 bytes). -> the key bytes."""
    key = binascii.hexlify(urandom(32))
    with open(path, 'wb') as handle:
        handle.write(key + b'\n')
    return key


def server_run(image, name, served, devices, port=PORT, hub=gate.HUB, derived_path=None, digests=False,
               salt_key_path=None, stats_now=False):
    """`docker run -d` of one serving container: the S1 gate's agent shape (its --rm dropped: the
    reload drill stops and starts the same container), the API on 127.0.0.1:port, a derived
    profiles file when the arm has one, the row digests when asked, the arm's salt key when it has
    one, and the platform's vLLM argv."""
    arguments = [token for token in gate.agent_shape(image, name, served, devices, hub) if token != '--rm']
    arguments[2:2] = ['-d']
    arguments += ['-p', '127.0.0.1:%d:8000' % port]
    if digests:
        arguments += ['-e', DIGESTS_ENV]
    if stats_now:
        arguments += ['-e', STATS_NOW_ENV]
    if salt_key_path:
        arguments += ['--mount', 'type=bind,src=%s,dst=%s,readonly' % (salt_key_path, SALT_KEY_MOUNT),
                      '-e', '%s=%s' % (SALT_KEY_ENV, SALT_KEY_MOUNT)]
    if derived_path:
        arguments += ['--mount', 'type=bind,src=%s,dst=%s,readonly' % (derived_path, DERIVED_MOUNT),
                      '-e', 'QWEN_C2_PROFILES=%s' % DERIVED_MOUNT]
    return arguments + ['--entrypoint', 'python3', image, '-m', 'vllm.entrypoints.openai.api_server'] + PLATFORM_ARGS


def anchor_script(root=MODEL_ROOT, pins=GRAFT_PINS, files=ANCHOR_FILES):
    """The shell the anchor probe runs in the image: the sha256 of model.py and qwen36_vllm.py, the
    image's graft pins, the sha256 of every file they pin (as installed under the model root: the
    Dockerfile checks the same, qwen-c2-serving.Dockerfile:45) and the files carrying '[PREFIX]'."""
    listed = "awk '{p=$2; sub(/^[*]/, \"\", p); if (p ~ /^graft[/]/ && p !~ /[.]orig$/) print substr(p, 7)}' " + pins
    return ('cd ' + root + ' && sha256sum ' + ' '.join(files) + '; echo ==pins; cat ' + pins + '; echo ==pinned; '
            + listed + ' | while read -r p; do sha256sum "$p" 2>/dev/null || echo "missing  $p"; done; '
            'echo ==markers; grep -rlF "[PREFIX]" . || true')


def anchor_probe(image, run=subprocess.run):
    """The image's model tree, read in a throwaway container (no devices, no network): see
    anchor_script and parse_anchor."""
    try:
        result = run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'sh', image, '-c', anchor_script()],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        text = (result.stdout or b'').decode('utf-8', 'replace')
    except (OSError, subprocess.SubprocessError) as error:
        return dict(error=repr(error)[:300])
    return parse_anchor(text)


def parse_anchor(text):
    """-> dict(files: model.py/qwen36_vllm.py sha, pins: every non-.orig graft.sha256 entry (path
    under the model root -> pinned sha), actual: that path's sha as installed (None: missing),
    mismatched, stage_pins: each anchor file graft.sha256 does not pin -> its STAGE_PINS sha,
    stage_mismatched: those whose served sha is not it, marker_files, prefix_marker_in: marker files
    the probe can vouch for (the two anchor files or a pinned one), unpinned: anchor files pinned by
    neither)."""
    files, pins, actual, marked, section = {}, {}, {}, [], 'files'
    for line in text.splitlines():
        if line.startswith('==pins'):
            section = 'pins'
            continue
        if line.startswith('==pinned'):
            section = 'pinned'
            continue
        if line.startswith('==markers'):
            section = 'markers'
            continue
        if section == 'markers':
            path = line.strip()
            if path:
                marked.append(path[2:] if path.startswith('./') else path)
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        path = parts[1].lstrip('*')
        if section == 'pinned' and parts[0] == 'missing':
            actual[path] = None
        elif len(parts[0]) != 64:
            continue
        elif section == 'files':
            files[path] = parts[0]
        elif section == 'pins':
            if path.startswith('graft/') and not path.endswith('.orig'):
                pins[path[len('graft/'):]] = parts[0]
        elif section == 'pinned':
            actual[path] = parts[0]
    mismatched = sorted(path for path in pins if actual.get(path) != pins[path])
    stage_pins = dict((path, STAGE_PINS[path]) for path in ANCHOR_FILES if path not in pins and path in STAGE_PINS)
    stage_mismatched = sorted(path for path in stage_pins if files.get(path) != stage_pins[path])
    vouched = set(ANCHOR_FILES) | set(pins)
    return dict(files=files, pins=pins, actual=actual, mismatched=mismatched, stage_pins=stage_pins,
                stage_mismatched=stage_mismatched, marker_files=marked,
                prefix_marker_in=sorted(path for path in marked if path in vouched),
                unpinned=sorted(path for path in ANCHOR_FILES if path not in pins and path not in stage_pins))


# -- judging one arm -----------------------------------------------------------------------------

def quiet_windows(events):
    """Log windows whose failure lines are expected (the restart drill's stop and boot)."""
    return [tuple(event['log_window']) for event in (events or {}).values()
            if isinstance(event, dict) and event.get('log_window')]


def generic_problems(arm, scanned, records, error, expect_profile, store_gib=judge.DEFAULT_STORE_GIB, windows=()):
    """Everything an arm fails on whatever its plan: the scenario's own error, failure lines in the
    log, the launched argv, and - on a prefix arm - the platform's prefix lines, the install marker,
    stale grants, markers against the oracle, vLLM's raw hit where publishing is the claim, and the
    loop path. -> (problems, notes, not_exercised)."""
    problems, notes, missing = [], [], []
    if error:
        problems.append('the scenario stopped: %s' % error)
    # A traceback alone is recorded (the lifecycle drops streams on purpose, and a server may log
    # one for a closed socket), and so is any failure line inside a quiet window (the restart
    # drill's docker stop); every other fatal signature - an engine death, a refused install, an
    # assertion (the model graft's grant check), the wedge - fails the arm, wherever it is.
    recorded, fatal = 0, []
    for entry in scanned.get('failures') or ():
        quiet = any(start <= entry['index'] <= end for start, end in windows)
        if entry['signature'] == markers.TRACEBACK or quiet:
            recorded += 1
            if recorded <= MAX_LISTED:
                notes.append('server log %s at line %s: %s' % ('traceback' if not quiet else 'failure line during the '
                                                               'restart drill', entry['index'], entry['line'][:200]))
        else:
            fatal.append(entry)
    if recorded > MAX_LISTED:
        notes.append('%d more tracebacks or restart-drill failure lines (server.log)' % (recorded - MAX_LISTED))
    for entry in fatal[:MAX_LISTED]:
        problems.append('server log line %s: %s' % (entry['index'], entry['line'][:300]))
    if len(fatal) > MAX_LISTED:
        problems.append('%d more fatal server log lines (server.log)' % (len(fatal) - MAX_LISTED))
    launches = scanned.get('launches') or []
    if not launches:
        problems.append('no "[QWEN-C2] profile <name>: vLLM argv" line: the serving contract did not launch vLLM')
    else:
        launched = launches[-1]
        if launched['profile'] != expect_profile:
            problems.append('served profile %r, expected %r' % (launched['profile'], expect_profile))
        notes.append('launched argv (%s): %s' % (launched['profile'], json.dumps(launched['argv'])[:1500]))
    failed = [r['tag'] for r in records if not r.get('ok') and not r.get('aborted')]
    if failed:
        problems.append('%d requests failed: %s' % (len(failed), ', '.join(failed[:8])))
    if not arm['prefix']:
        return problems, notes, missing
    if launches:
        problems += markers.prefix_argv_problems(launches[-1]['argv'])
    if 'enabled' not in (scanned.get('apc') or []):
        problems.append('no "Automatic prefix caching is enabled" line from the TT platform')
    if not scanned.get('chunking_off') and 'no-enable-chunked-prefill' not in markers.argv_flags(
            (launches[-1]['argv'] if launches else [])):
        problems.append('no "Chunked prefill is not supported ... disabling it" line: chunked prefill may still be '
                        'on (Lever N\'s second meaning of start_pos, F5)')
    installs = scanned.get('installs') or []
    if not installs:
        problems.append('no "[PINDIAG] prefix: install" line: the scheduler graft never ran in the engine\'s '
                        'scheduler (memory graft-mounted-is-not-graft-executed)')
    else:
        install = installs[-1]
        if 'TTScheduler' not in str(install.get('scheduler')):
            problems.append('the graft installed on %s, not TTScheduler' % install.get('scheduler'))
        if install.get('block_size') != 64:
            problems.append('KV block size %s, not 64 (F7)' % install.get('block_size'))
        if str(install.get('QWEN_SDPA_BF8')) != '1':
            problems.append('QWEN_SDPA_BF8=%s: the served KV is not bf8 (F4)' % install.get('QWEN_SDPA_BF8'))
        if install.get('store_gib') is not None and abs(float(install['store_gib']) - float(store_gib)) > 0.051:
            problems.append('checkpoint store %s GiB, the arm asked %s' % (install['store_gib'], store_gib))
        notes.append('install: %s' % json.dumps(dict((k, v) for k, v in install.items() if k not in ('index', 'time'))))
    for entry in scanned.get('refused') or ():
        problems.append('a stale grant was refused at commit (%s start_pos=%s Q=%s, F2)' % (
            entry['tag'], entry['start_pos'], entry['q']))
    for entry in scanned.get('capture_skipped') or ():
        notes.append('capture skipped for %s at %s: %s' % (entry['tag'], entry['pos'], entry['reason']))
    for record in records:
        if record.get('role') in ('seat', 'flood') or record.get('aborted'):
            continue
        for severity, text in judge.reuse_problems(record, sequential=arm['strict']):
            if severity == 'FAIL':
                problems.append(text)
            elif severity == 'LOST' and arm['strict']:
                missing.append(text)
            else:
                notes.append(text)
        if record.get('expected_raw_h') is not None and record.get('ok'):
            raw, attempts = judge.raw_hit_per_attempt(record)
            if raw is None:
                missing.append('%s: no vllm:prefix_cache_hits reading around it: what it found published is not '
                               'measured' % record['tag'])
            elif raw > record['expected_raw_h']:
                problems.append('%s: vLLM found %d cached tokens (%d attempt(s)) where only %d were published for it: '
                                'a request published blocks the design forbids (an unsalted one, or one under the '
                                'kill switch)' % (record['tag'], raw, attempts, record['expected_raw_h']))
            else:
                notes.append('%s: vLLM found %d cached tokens (published for it: %d)' % (
                    record['tag'], raw, record['expected_raw_h']))
    path = 'eager' if arm.get('kind') == 'eager' else 'traced'
    if path == 'eager' and installs:
        warm = scanned.get('eager_warm') or []
        if not warm:
            problems.append('no "%s" line: the model graft did not compile the eager prefill before the decode trace '
                            'was parked, so the first request compiles it after the park and a later prefill can hang '
                            'the device (G1 v47, run 36246961161): an image built before the eager warm' % markers.EAGER_WARM)
        else:
            notes.append('eager warm: %s' % json.dumps(dict((k, v) for k, v in warm[-1].items() if k not in ('index', 'time'))))
    other = 'traced' if path == 'eager' else 'eager'
    paths = sorted(set(str(row.get('path')) for r in records for row in (r.get('markers') or {}).get('rows') or ()
                       if row.get('path')))
    if other in paths:
        problems.append('[PREFIX] rows report path %s, the arm serves %s' % (', '.join(paths), path))
    elif paths and path not in paths:
        notes.append('[PREFIX] rows report path %s only (no %s row)' % (', '.join(paths), path))
    if path == 'traced' and not scanned.get('chunk_replay') and any(r.get('ok') for r in records):
        problems.append('no "[TP chunk-replay]" line: the traced chunk loop never ran')
    if path == 'eager' and scanned.get('chunk_replay'):
        problems.append('%d "[TP chunk-replay]" lines on the eager arm: trace_mode decode_only did not reach the '
                        'model' % scanned['chunk_replay'])
    return problems, notes, missing


def row_growth_problems(rows):
    """F3 on every [PREFIX] row of an exactness arm: a row whose program cache grew compiled after the traces
    were parked, the second-request hang (#48536). G1 v47's eager arm (run 36246961161) logged one such row
    (115 -> 248) and its next prefill hung the device; its traced and audit arms' rows compiled nothing.
    -> problems."""
    grown = []
    for row in rows:
        before, after = row.get('programs_before'), row.get('programs')
        if isinstance(before, int) and isinstance(after, int) and after > before:
            grown.append('%s (path %s, Q=%s L=%s) compiled %d programs inside its prefill row: a compile after the '
                         'traces were parked (F3, the second-request hang #48536)' % (
                             row.get('tag') or row.get('req'), row.get('path'), row.get('q'), row.get('l'),
                             after - before))
    if len(grown) > MAX_LISTED:
        grown = grown[:MAX_LISTED] + ['%d more rows compiled programs (server.log)' % (len(grown) - MAX_LISTED)]
    return grown


def pair_problems(pairs):
    """Pair verdicts (settled) as (problems, unstable, not_comparable, rerun)."""
    problems, unstable, not_comparable, rerun = [], [], [], []
    for pair in pairs:
        text = '%s %s turn %s (L=%s, %s vs %s): %s%s' % (pair['case'], pair['conv'], pair['turn'], pair['prompt_tokens'],
                                                       pair['cold'], pair['hit'], pair['verdict'],
                                                       ' - %s' % pair['detail'] if pair.get('detail') else '')
        if pair['verdict'] in ('DIVERGED', 'ERROR'):
            problems.append(text)
        elif pair['verdict'] == 'UNSTABLE':
            unstable.append(text)
        elif pair['verdict'] == 'NOT_COMPARABLE':
            not_comparable.append(text)
        elif pair['verdict'] == 'RERUN':
            rerun.append(text)
    return problems, unstable, not_comparable, rerun


def by_tag(records):
    return dict((record['tag'], record) for record in records)


def hits(records, case):
    """The hit records of one case, in order (a re-run's case is '<case>:rerun', so it is left out)."""
    return [r for r in records if r.get('role') == 'hit' and r.get('case') == case]


def q_of(record):
    return (record.get('markers') or {}).get('q')


def captured_of(record):
    return sorted(((record.get('markers') or {}).get('row') or {}).get('captured') or ())


def digest_findings(arm, driver):
    """Every compared pair's slot and logits digests (and, on the bring-up, each fresh-salt capture
    turn's against its unsalted twin: the capture must not disturb the prefill). A digest missing on
    every row is one finding, not one per pair. -> (problems, not exercised, notes)."""
    index = by_tag(driver.records)
    couples = []
    for pair in driver.pairs:
        cold, hit = index.get(pair['cold']), index.get(pair['hit'])
        if pair['verdict'] != 'ERROR' and cold and hit and cold.get('ok') and hit.get('ok'):
            couples.append((cold, hit))
    unsalted = dict((r.get('prompt_sha'), r) for r in driver.records if r.get('role') == 'unsalted' and r.get('ok'))
    for record in driver.records:
        if record.get('role') == 'capture' and record.get('ok') and record.get('prompt_sha') in unsalted:
            couples.append((unsalted[record['prompt_sha']], record))
    problems, absent = [], []
    for first, second in couples:
        for severity, text in judge.digest_problems(first, second):
            (problems if severity == 'FAIL' else absent).append(text)
    missing, notes = [], []
    if absent:
        target = missing if arm['strict'] else notes
        if len(absent) >= 2 * len(couples):
            target.append('no [PREFIX] row carries slot_sha and logits_sha: the GDN state and logits after prefill '
                          'are not compared (%d pairs)' % len(couples))
        else:
            target.extend(absent[:MAX_LISTED])
    return problems, missing, notes


def exercised_exactness(arm, records, events):
    """The exactness cases that must have happened, by the markers: -> (not exercised, problems, lines)."""
    missing, problems, lines = [], [], []
    chain = hits(records, 'chain')
    reached = [(r.get('prompt_tokens'), q_of(r)) for r in chain]
    lines.append('chain hits (L, Q): %s' % reached)
    if not any(q for _, q in reached[1:] if q):
        missing.append('no chain turn after the first got Q > 0')
    for target in replay.BOUNDARY_PROMPTS:
        name = 'boundary-%d' % target
        turns = hits(records, name)
        event = events.get(name) or {}
        if not event.get('fitted'):
            missing.append('%s: the first prompt could not be fitted to %d tokens (%s)' % (name, target, event.get('error')))
            continue
        served = event.get('served')
        if served != target:
            missing.append('%s: /tokenize fitted %s tokens but the chat endpoint served %s: the case run is not the one '
                           'named' % (name, event.get('tokens'), served))
            continue
        if len(turns) < 2:
            missing.append('%s: fewer than two turns ran' % name)
            continue
        second = turns[1]
        q, length = q_of(second), second.get('prompt_tokens') or 0
        want = judge.floor_chunk(served)
        lines.append('%s: turn 1 served L=%s, turn 2 L=%s Q=%s (want %s)' % (name, served, length, q, want))
        if q is not None and q != want:
            (problems if q > want else missing).append('%s: turn 2 restored Q=%s, the design gives %d' % (name, q, want))
        if want and q == want and judge.floor_chunk(length) == q:
            lines.append('%s: a tail-only hit (no full chunk after Q)' % name)
    tail_only = [r for r in records if r.get('role') == 'hit' and q_of(r) and judge.floor_chunk(r.get('prompt_tokens') or 0) == q_of(r)]
    if not tail_only:
        missing.append('no tail-only hit (Q = floor2048(L)) ran')
    if arm['arm'] == 'exactness-traced':
        for case in ('changed-suffix', 'early-divergence'):
            found = hits(records, case)
            lines.append('%s (L, Q): %s' % (case, [(r.get('prompt_tokens'), q_of(r)) for r in found]))
            if not any(q_of(r) for r in found):
                missing.append('%s: no hit with Q > 0' % case)
        suffix, early = hits(records, 'changed-suffix'), hits(records, 'early-divergence')
        if suffix and early and q_of(suffix[0]) and q_of(early[0]) is not None and q_of(early[0]) >= q_of(suffix[0]):
            missing.append('the early divergence restored Q=%s, not below the changed suffix\'s %s: it did not fall '
                           'back to an older checkpoint' % (q_of(early[0]), q_of(suffix[0])))
        more_missing, more_problems, more_lines = shared_gap(records)
        missing += more_missing
        problems += more_problems
        lines += more_lines
    return missing, problems, lines


def shared_gap(records):
    """The gap capture (design 2.0.1 item 2a.5): the tenant's second conversation misses inside the
    shared block and captures its gap boundary G = floor2048(h) below its own prompt boundary; the
    third restores exactly G. -> (not exercised, problems, lines)."""
    missing, problems, lines = [], [], []
    shared = hits(records, 'shared-system')
    lines.append('shared-system (L, Q, captured): %s' % [(r.get('prompt_tokens'), q_of(r), captured_of(r))
                                                        for r in shared])
    if len(shared) < 3:
        missing.append('shared-system: %d of 3 conversations ran' % len(shared))
        return missing, problems, lines
    second, third = shared[1], shared[2]
    own = judge.floor_chunk(second.get('prompt_tokens') or 0)
    gaps = [position for position in captured_of(second) if position < own]
    if not gaps:
        missing.append('shared-system: the second conversation captured no gap boundary (captured %s, L=%s, Q=%s): '
                       'its own first request\'s capture already covered the shared block, or the gap capture did not '
                       'run' % (captured_of(second), second.get('prompt_tokens'), q_of(second)))
        return missing, problems, lines
    gap = max(gaps)
    lines.append('shared-system: gap boundary %d captured by %s, restored by %s at Q=%s' % (
        gap, second['tag'], third['tag'], q_of(third)))
    if q_of(third) is None or q_of(third) < gap:
        missing.append('shared-system: the third conversation restored Q=%s, not the gap boundary %d' % (q_of(third), gap))
    elif q_of(third) > gap:
        problems.append('shared-system: the third conversation restored Q=%s past the shared block\'s gap boundary %d'
                        % (q_of(third), gap))
    return missing, problems, lines


def audit_findings(records, pairs):
    index = by_tag(records)
    problems, missing = [], []
    for pair in pairs:
        cold, hit = index.get(pair['cold']), index.get(pair['hit'])
        if not cold or not hit or not hit.get('ok'):
            continue
        for severity, text in judge.audit_problems(cold, hit):
            (problems if severity == 'FAIL' else missing).append(text)
    return problems, missing


# counter -> (what it shows, what a zero means)
STAT_REASONS = dict(
    same_step_rejects=('the same-step rule', 'no same-step reject: the four arrivals never shared a scheduler step'),
    evicted_coupled=('eviction coupling (F8)', 'no checkpoint left with its evicted block (eviction coupling, F8)'),
    evicted_lru=('the checkpoint LRU', 'no checkpoint was pushed out of the small store by the LRU'),
    dropped_hits=('an allocation failure after a grant (F2)',
                  'no staged grant with Q > 0 was dropped at commit (an allocation failure after a grant, F2)'),
)


def required_stats(stats, names):
    """-> not-exercised lines for each counter of `names` that the stats export lacks or left at zero
    (nothing when there is no export at all: lifecycle_findings says that once)."""
    out = []
    if stats is None:
        return out
    for name in names:
        what, zero = STAT_REASONS[name]
        if name not in stats:
            out.append('the registry stats export has no %s counter: %s is not observable' % (name, what))
        elif not stats[name]:
            out.append(zero)
    return out


def tiny_findings(records, events, stats):
    """The tiny pool's two phases (prefix_replay.tiny_dropped_grant, tiny_preemption).
    -> (problems, not exercised, lines)."""
    problems, missing, lines = [], [], []
    index = by_tag(records)
    grant = events.get('tiny-grant') or {}
    tiny = events.get('tiny') or {}
    if not grant and tiny.get('skipped') and tiny.get('expected_pool'):
        missing.append('the tiny pool did not run: %s' % tiny.get('reason'))
        return problems, missing, ['tiny pool: %s' % json.dumps(tiny)]
    lines.append('allocation failure after a grant: %s' % json.dumps(grant))
    waited = index.get(grant.get('tag')) or {}
    if not grant:
        missing.append('the dropped-grant phase did not run')
    elif not grant.get('ok'):
        problems.append('a request of the dropped-grant phase failed')
    elif not grant.get('filler_running') or not grant.get('waited_for_filler'):
        missing.append('the next turn did not wait for the filler (its first token came before the filler ended): no '
                       'allocation failed after a grant (%s)' % json.dumps(grant))
    elif not q_of(waited):
        missing.append('the turn that waited was admitted at Q=%s: no grant with Q > 0 was staged while it waited'
                       % q_of(waited))
    else:
        lines.append('%s waited for the filler (its first token at %s s, the filler ended at %s s) with a staged grant '
                     'on its cached prefix, then restored Q=%s' % (grant['tag'], grant.get('first_token_s'),
                                                                     grant.get('filler_end_s'), q_of(waited)))
    direct = bool(grant.get('ok') and grant.get('waited_for_filler') and q_of(waited))
    if stats is not None and 'dropped_hits' in stats:
        if not stats['dropped_hits'] and direct:
            problems.append('the registry counted no dropped grant with Q > 0 though %s waited with one staged: the '
                            'dropped_hits counter (or the drop) is wrong' % grant.get('tag'))
        elif not stats['dropped_hits']:
            missing.append(STAT_REASONS['dropped_hits'][1])
    elif stats is not None and not direct:
        missing += required_stats(stats, ('dropped_hits',))
    lines.append('preemption: %s' % json.dumps(tiny))
    if not tiny or tiny.get('skipped'):
        missing.append('the preemption phase did not run: %s' % (tiny.get('reason') or 'no tiny event'))
        return problems, missing, lines
    if not tiny.get('ok'):
        problems.append('a request of the preemption phase failed')
    resumed = sorted('%s (%d admissions)' % (r['tag'], judge.admissions(r)) for r in records if judge.admissions(r) > 1)
    lines.append('preempted and resumed: %s' % (', '.join(resumed) or 'none'))
    if not (tiny.get('preemptions') or 0) > 0:
        missing.append('no preemption happened on the tiny pool')
    elif not resumed:
        missing.append('vLLM counted %s preemptions but no request printed a second [PREFIX] row: the resumed prefill '
                       'is not measured' % tiny.get('preemptions'))
    return problems, missing, lines


def lifecycle_findings(arm, records, events, scanned, stats):
    """The lifecycle events against what each must leave. -> (problems, not exercised, lines)."""
    problems, missing, lines = [], [], []
    index = by_tag(records)
    if arm['arm'] == 'lifecycle-evict':
        arrivals = events.get('arrivals') or {}
        lines.append('arrivals: %s' % json.dumps(arrivals))
        if not arrivals.get('ok'):
            problems.append('a request of the four simultaneous arrivals failed')
        waiting = events.get('abort-waiting') or {}
        lines.append('abort while waiting: %s' % json.dumps(waiting))
        aborted = index.get(waiting.get('tag')) or {}
        if waiting.get('phase') != 'waiting' or not waiting.get('aborted') or waiting.get('first_token'):
            missing.append('the abort did not land while the request waited (%s)' % json.dumps(waiting))
        elif (aborted.get('markers') or {}).get('grants') or (aborted.get('markers') or {}).get('rows'):
            missing.append('the request aborted "while waiting" was admitted after all (it has a grant or a row)')
        prefill = events.get('abort-prefill') or {}
        lines.append('abort during a hit prefill: %s' % json.dumps(prefill))
        target = index.get(prefill.get('tag')) or {}
        if not prefill.get('aborted') or prefill.get('first_token_before_abort'):
            missing.append('the abort did not land before the hit\'s first token (%s)' % json.dumps(prefill))
        elif not ((target.get('markers') or {}).get('grant') or {}).get('q'):
            missing.append('the request aborted "during its prefill" has no grant with Q > 0: it was not a hit\'s '
                           'prefill (%s)' % ((target.get('markers') or {}).get('grant'),))
        flood = [r for r in hits(records, 'after-flood')]
        lost = [r for r in flood if q_of(r) is not None and q_of(r) < (r.get('expected') or {}).get('q', 0)]
        lines.append('after the flood (L, Q, oracle Q): %s' % [(r.get('prompt_tokens'), q_of(r),
                                                               (r.get('expected') or {}).get('q')) for r in flood])
        if not lost:
            missing.append('the KV flood evicted no conversation\'s hit (%s)' % json.dumps(events.get('flood')))
        reset = events.get('reset-prefix-cache') or {}
        after = hits(records, 'after-reset')
        if reset.get('status') != 200:
            missing.append('POST /reset_prefix_cache answered %s (VLLM_SERVER_DEV_MODE=1 is needed)' % reset.get('status'))
        elif after and q_of(after[0]):
            problems.append('a hit after reset_prefix_cache restored Q=%s: the reset left the prefix cache or the '
                            'checkpoint registry in place (vLLM answers 200 without checking, dev/cache/api_router.py)'
                            % q_of(after[0]))
        kill = events.get('kill-switch') or {}
        lines.append('kill switch: %s' % json.dumps(kill))
        if not kill.get('written'):
            missing.append('the kill switch file could not be written')
        else:
            if not scanned.get('kill_switch'):
                problems.append('the kill switch file was present but no "[PINDIAG] prefix: kill switch" line')
            for case in ('kill-on', 'kill-latched'):
                for record in hits(records, case):
                    if q_of(record):
                        problems.append('%s: Q=%s with the kill switch engaged' % (record['tag'], q_of(record)))
            latched = index.get(kill.get('latched_tag')) or {}
            bound = latched.get('expected_raw_h')
            if bound is None:
                missing.append('the kill-latched turn has no published-token bound: publishing under the kill switch '
                               'is not measured')
            elif judge.floor_chunk(kill.get('on_prompt') or 0) <= bound:
                missing.append('the kill-on turn (L=%s) crossed no chunk boundary past the %d tokens published before '
                               'the kill: publishing under the kill switch is not observable' % (kill.get('on_prompt'),
                                                                                                bound))
            if not kill.get('removed'):
                problems.append('the kill switch file this gate wrote could not be removed')
        reload_first, reload_second = hits(records, 'after-reload'), hits(records, 'after-reload-2')
        if not events.get('reload'):
            missing.append('the in-place restart did not run')
        else:
            if reload_first and q_of(reload_first[0]):
                problems.append('the first turn after the restart restored Q=%s: state outlived the engine' % q_of(reload_first[0]))
            if reload_second and not q_of(reload_second[0]):
                problems.append('the second turn after the restart missed: reuse did not come back')
        # The restart drill starts a new registry: its counters so far were read just before it.
        before = (events.get('stats-before-restart') or {}).get('stats')
        if before is not None:
            lines.append('registry stats before the restart: %s' % json.dumps(before, sort_keys=True))
            if before.get('pins'):
                problems.append('%s checkpoint pins held before the restart, every request ended' % before['pins'])
            if before.get('commit_mismatch'):
                problems.append('%s grants refused at commit before the restart (start_pos != Q)'
                                % before['commit_mismatch'])
        missing += required_stats(before if before is not None else stats,
                                  ('same_step_rejects',) + (('evicted_coupled',) if lost else ()))
    elif arm['arm'] == 'lifecycle-store':
        after = hits(records, 'store-after')
        lines.append('after the store filled (L, Q, oracle Q): %s' % [(r.get('prompt_tokens'), q_of(r),
                                                                      (r.get('expected') or {}).get('q')) for r in after])
        if not any(q_of(r) is not None and q_of(r) < (r.get('expected') or {}).get('q', 0) for r in after):
            missing.append('no checkpoint was pushed out of the small store (the next turn hit as if unbounded)')
        missing += required_stats(stats, ('evicted_lru',))
    elif arm['arm'] == 'lifecycle-tiny':
        more_problems, more_missing, more_lines = tiny_findings(records, events, stats)
        problems += more_problems
        missing += more_missing
        lines += more_lines
    if stats is None:
        missing.append('no registry stats export (%s or a "[PINDIAG] prefix: stats" line): held pins, commit '
                       'mismatches and the counters this arm needs are not checked' % markers.STATS_FILE)
    else:
        lines.append('registry stats: %s' % json.dumps(stats, sort_keys=True))
        if stats.get('pins'):
            problems.append('%s checkpoint pins held after every request ended' % stats['pins'])
        if stats.get('commit_mismatch'):
            problems.append('%s grants refused at commit (start_pos != Q)' % stats['commit_mismatch'])
    return problems, missing, lines


def judge_arm(arm, driver, scanned, stats, error):
    """One arm's verdict dict (the plan's cross-arm checks come after, in judge_plan)."""
    records = driver.records
    index = by_tag(records)
    for pair in driver.pairs:
        judge.settle(pair, index)
    problems, notes, missing = generic_problems(arm, scanned, records, error, arm['served'],
                                                SMALL_STORE_GIB if arm.get('kind') == 'store' else judge.DEFAULT_STORE_GIB,
                                                quiet_windows(driver.events))
    lines = []
    if arm['prefix']:
        bringup = arm['scenario'] == 'bringup_prefix'
        program, detail, unmeasured = judge.program_cache_problems(scanned.get('rows') or [], driver.pairs,
                                                                   first_capture=bringup, require_hit=bringup)
        lines.append('program cache across hits and the first capture: %s' % json.dumps(detail, sort_keys=True))
        problems += program
        (missing if bringup else notes).extend(unmeasured)
        digest_fail, digest_missing, digest_notes = digest_findings(arm, driver)
        problems += digest_fail
        missing += digest_missing
        notes += digest_notes
        if bringup and stats is not None and stats.get('unsalted_denied'):
            problems.append('the registry denied %s unsalted requests a hit: an unsalted request published blocks'
                            % stats['unsalted_denied'])
    if arm['scenario'].startswith('exactness'):
        problems += row_growth_problems(scanned.get('rows') or [])
        more_missing, more_problems, more_lines = exercised_exactness(arm, records, driver.events)
        missing += more_missing
        problems += more_problems
        lines += more_lines
        if arm.get('kind') == 'audit':
            audit_fail, audit_missing = audit_findings(records, driver.pairs)
            problems += audit_fail
            missing += audit_missing
    elif arm['scenario'].startswith('lifecycle'):
        more_problems, more_missing, more_lines = lifecycle_findings(arm, records, driver.events, scanned, stats)
        problems += more_problems
        missing += more_missing
        lines += more_lines
    diverged, unstable, not_comparable, rerun = pair_problems(driver.pairs)
    problems += diverged
    if scanned.get('failures') and any(markers.WEDGE in (entry.get('line') or '') for entry in scanned['failures']):
        verdict = 'INFRA'
    elif problems:
        verdict = 'FAIL'
    elif rerun:
        verdict = 'RERUN'
    elif not_comparable:
        verdict = 'NOT_COMPARABLE'
    elif unstable:
        verdict = 'UNSTABLE'
    elif missing:
        verdict = 'NOT_EXERCISED'
    else:
        verdict = 'PASS'
    identical = sum(1 for pair in driver.pairs if pair['verdict'] == 'IDENTICAL')
    lines = ['%d pairs identical of %d' % (identical, len(driver.pairs))] + lines
    lines += ['problem: %s' % text for text in problems] + ['unstable: %s' % text for text in unstable]
    lines += ['not comparable: %s' % text for text in not_comparable] + ['not exercised: %s' % text for text in missing]
    return dict(verdict=verdict, problems=problems, unstable=unstable, not_comparable=not_comparable, rerun=rerun,
                not_exercised=missing, notes=notes, lines=lines, pairs=len(driver.pairs), identical=identical,
                dram=scanned.get('dram')[:16] if scanned.get('dram') else [],
                dram_readings=(scanned.get('dram_readings') or [])[:16], kv_tokens=scanned.get('kv_tokens'))


def bringup_cross(reference, prefix, anchor):
    """The bring-up's cross-arm checks: the prefix arm's unsalted turns AND its fresh-salt capture
    turns against the baseline's reference turns byte for byte, each capture turn's row showing
    captured=[floor2048(L)], the anchor probe. -> (problems, not exercised, lines)."""
    problems, missing, lines = [], [], []
    refs = [r for r in reference.records if r.get('role') == 'reference']
    unsalted = [r for r in prefix.records if r.get('role') == 'unsalted']
    captures = [r for r in prefix.records if r.get('role') == 'capture']
    if not refs or len(refs) != len(unsalted) or len(unsalted) != len(captures):
        problems.append('the reference ran %d turns, the prefix arm %d unsalted and %d capture turns' % (
            len(refs), len(unsalted), len(captures)))
    for ref, mine, capture in zip(refs, unsalted, captures):
        for label, other in (('grants disabled (unsalted)', mine), ('capturing (fresh salt)', capture)):
            result = judge.compare(ref, other)
            lines.append('%s vs baseline, turn %s (L=%s): %s%s' % (
                label, other.get('turn'), other.get('prompt_tokens'), result['verdict'],
                ' - %s' % result['detail'] if result.get('detail') else ''))
            if result['verdict'] != 'IDENTICAL':
                problems.append('turn %s %s is not byte-identical to the baseline (%s): %s' % (
                    other.get('turn'), label, result['verdict'], result.get('detail')))
        length = capture.get('prompt_tokens') or 0
        want = [judge.floor_chunk(length)] if length >= judge.CHUNK else []
        if capture.get('ok') and captured_of(capture) != want:
            problems.append('turn %s under a fresh salt (L=%s) captured %s, not %s: the capture path the salted '
                            'traffic takes was not exercised' % (capture.get('turn'), length, captured_of(capture), want))
    if anchor.get('error'):
        problems.append('the anchor probe could not read the image: %s' % anchor['error'])
    else:
        lines.append('anchor: %s' % json.dumps(anchor, sort_keys=True))
        if not anchor.get('pins'):
            problems.append('the anchor probe read no pins from the image\'s %s' % GRAFT_PINS)
        if anchor.get('mismatched'):
            problems.append('the served model tree is not the image\'s pinned graft: %s' % ', '.join(anchor['mismatched']))
        if not anchor.get('prefix_marker_in'):
            problems.append('no "[PREFIX]" marker in %s or a file graft.sha256 pins (marker files: %s): the prefix model '
                            'graft is not in the served tree' % (' or '.join(ANCHOR_FILES), anchor.get('marker_files')))
        files, stage_pins = anchor.get('files') or {}, anchor.get('stage_pins') or {}
        stale = anchor.get('stage_mismatched') or []
        if stale:
            problems.append('the served %s not the prefix stage\'s graft (qwen_prefix_model_patch.PATCHED_SHA256): %s: '
                            'the image predates this checkout\'s model graft, or its tree changed after the stage' % (
                                ' and '.join(stale) + (' is' if len(stale) == 1 else ' are'),
                                '; '.join('%s sha256 %s, pinned %s' % (path, files.get(path), stage_pins.get(path))
                                          for path in stale)))
        for path, pin in sorted(stage_pins.items()):
            if path not in stale:
                lines.append('anchor: %s at the prefix stage\'s pin %s (qwen_prefix_model_patch.PATCHED_SHA256)' % (
                    path, pin))
        for path in anchor.get('unpinned') or ():
            problems.append('anchor: %s (sha256 %s) is pinned by neither graft.sha256 nor the prefix stage: the probe '
                            'cannot vouch for the served model tree' % (path, files.get(path)))
    return problems, missing, lines


def dram_findings(result):
    """G2's DRAM reading on the bring-up's prefix arm: the model graft's '[PINDIAG] dram after registry'
    (the model warm, the registry created) and, once the arm stored a checkpoint (the registry's
    captures counter), 'dram after first capture' - each with per-chip figures, not 'unavailable'.
    -> (not exercised, lines: every reading the arm logged and the KV pool beside it)."""
    readings = result.get('dram_readings') or []
    lines = ['dram after %s: %s' % (reading.get('point'), reading.get('text')) for reading in readings]
    if not readings:
        lines.append('dram: none logged')
    if result.get('kv_tokens'):
        lines.append('dram: beside a KV pool of %d tokens (vLLM\'s GPU KV cache size)' % result['kv_tokens'])
    stored = int((result.get('stats') or {}).get('captures') or 0)
    wanted = [(markers.DRAM_REGISTRY, 'the model warm and the registry created')]
    if stored:
        wanted.append((markers.DRAM_FIRST_CAPTURE, 'the arm stored %d checkpoint(s)' % stored))
    missing = []
    for point, why in wanted:
        mine = [reading for reading in readings if reading.get('point') == point]
        if not any(reading.get('chips') for reading in mine):
            missing.append('no "[PINDIAG] dram after %s" reading with per-chip figures on the prefix arm (%s%s): G2 has '
                           'no DRAM reading' % (point, why, '; logged: %s' % mine[0].get('text') if mine else ''))
    return missing, lines


def timing_summary(driver):
    phases = {}
    for name, phase in sorted(driver.phases.items()):
        records = [r for r in driver.records if r.get('case') == name]
        phases[name] = report.phase_summary(records, phase.get('seconds'), phase.get('rss_gb_max'), phase.get('ci_pods'))
    return phases


# -- running arms --------------------------------------------------------------------------------

class Runner(object):
    """Runs arms: a container per arm, the scenario against it, its files kept. `docker`, `make_client`
    and `containers` are injectable for the CPU tests."""

    def __init__(self, image, results, checkout, devices, hub=gate.HUB, port=PORT, docker=None, make_client=None,
                 make_log=None, make_container=None, containers=None, log=print, clock=time.time, sleep=time.sleep,
                 corpus=None, seed=0, agents=replay.TIMING_AGENTS, turns=replay.TIMING_TURNS):
        self.image, self.results, self.checkout, self.devices = image, results, checkout, devices
        self.hub, self.port, self.log, self.clock, self.sleep = hub, port, log, clock, sleep
        self.docker = docker or self._docker
        self.make_client = make_client or (lambda: replay.Client(port, clock=clock))
        self.make_log = make_log or (lambda name, path: replay.LogFollower(name, path))
        self.make_container = make_container or (lambda name: replay.Container(name))
        self.containers = containers or gate.platform_containers
        self.corpus = corpus
        self.seed, self.agents, self.turns = seed, agents, turns
        self.infra = None
        self.arms = {}
        self.drivers = {}

    @staticmethod
    def _docker(arguments, timeout=600):
        try:
            result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            return result.returncode, (result.stdout or b'').decode('utf-8', 'replace')
        except (OSError, subprocess.SubprocessError) as error:
            return None, repr(error)[:300]

    def get_corpus(self):
        if self.corpus is None:
            self.corpus = corpus_module.Corpus(corpus_module.load_sources(self.checkout))
        return self.corpus

    def wait_ready(self, client, container, seconds=READINESS_SECONDS):
        deadline = self.clock() + seconds
        while self.clock() < deadline:
            if not container.running():
                return 'the container exited before its API answered'
            if client.ready():
                return None
            self.sleep(10)
        return 'no /v1/models answer in %d s' % seconds

    def run(self, arm, profiles_doc=None):
        name = CONTAINER_PREFIX + arm['arm']
        arm_dir = os.path.join(self.results, arm['arm'])
        os.makedirs(arm_dir, exist_ok=True)
        if self.infra is None:
            held = self.containers()
            if held:
                self.infra = 'a platform serving container exists (%s): M+A may be reloaded under the gate' % ', '.join(held)
        if self.infra is not None:
            self.log('[PREFIX-GATE] arm %s: skipped - %s' % (arm['arm'], self.infra))
            result = dict(verdict='INFRA', reason=self.infra, lines=[])
            self.arms[arm['arm']] = result
            return None, result
        derived_path = None
        if arm.get('derived'):
            derived_path = os.path.join(arm_dir, 'profiles.json')
            with open(derived_path, 'w', encoding='utf-8') as handle:
                json.dump(arm['derived'], handle, indent=1, sort_keys=True)
        salt_key = salt_key_path = None
        if arm.get('prefix'):
            salt_key_path = os.path.join(arm_dir, 'salt.key')
            salt_key = write_salt_key(salt_key_path)
        arguments = server_run(self.image, name, arm['served'], self.devices, self.port, self.hub, derived_path,
                               digests=wants_digests(arm), salt_key_path=salt_key_path,
                               stats_now=bool(arm.get('prefix')))
        with open(os.path.join(arm_dir, 'docker-run.json'), 'w') as handle:
            json.dump(arguments, handle, indent=1)
        started = self.clock()
        deadline = started + arm['timeout'] - 60
        self.log('[PREFIX-GATE] arm %s: profile %s, scenario %s, %d s' % (arm['arm'], arm['served'], arm['scenario'],
                                                                         arm['timeout']))
        self.docker(['docker', 'rm', '-f', name], 120)
        client = self.make_client()
        container = self.make_container(name)
        follower = self.make_log(name, os.path.join(arm_dir, 'server.log'))
        # The oracle's store is the registry's default (8 GiB) on every arm: on the small-store arm
        # the evictions it does not model are what the arm looks for (a LOST hit).
        driver = replay.Driver(client, arm['arm'], self.get_corpus(), follower, container, seed=self.seed,
                               deadline=deadline, clock=self.clock, sleep=self.sleep, say=self.log,
                               strict=arm['strict'], salt_key=salt_key)
        error, stats, metrics, output = None, None, {}, ''
        try:
            code, output = self.docker(arguments, 300)
            if code != 0:
                raise replay.EngineDead('docker run exited %s: %s' % (code, output[-400:]))
            follower.start()
            problem = self.wait_ready(client, container, max(60, min(READINESS_SECONDS, deadline - self.clock())))
            if problem:
                raise replay.EngineDead(problem)
            self.log('[PREFIX-GATE] arm %s: ready after %.0f s' % (arm['arm'], self.clock() - started))
            kwargs = {}
            if arm['scenario'] in ('lifecycle_evict', 'lifecycle_tiny'):
                kwargs['pool_tokens'] = markers.scan(follower.lines()).get('kv_tokens')
            if arm['scenario'] == 'lifecycle_evict':
                kwargs['restart'] = lambda: self.restart(container, client, follower, driver)
            if arm['scenario'] == 'timing':
                kwargs.update(agents=self.agents, turns=self.turns)
            replay.SCENARIOS[arm['scenario']](driver, **kwargs)
        except Exception as failure:   # noqa: BLE001 - any failure ends the arm, recorded and judged
            error = '%s: %s' % (type(failure).__name__, failure)
            if not isinstance(failure, (replay.EngineDead, replay.OutOfTime)):
                error += ' | ' + traceback.format_exc()[-1500:].replace('\n', ' | ')
            self.log('[PREFIX-GATE] arm %s: %s' % (arm['arm'], error))
        finally:
            try:
                if arm['scenario'] == 'lifecycle_evict':
                    replay.kill_switch_off(container)
                self.sleep(SETTLE_SECONDS)
                text = container.read_file(markers.STATS_FILE)
                if text:
                    try:
                        stats = json.loads(text)
                    except ValueError:
                        stats = None
                metrics = client.metrics()
            finally:
                follower.stop()
                code, output = self.docker(['docker', 'logs', '--timestamps', name], 600)
                with open(os.path.join(arm_dir, 'server-final.log'), 'w', encoding='utf-8') as handle:
                    handle.write(output or '')
                self.docker(['docker', 'rm', '-f', name], 120)
        # The follower's lines: the records' log windows index them. docker logs' own copy only when
        # the follower caught nothing.
        lines = follower.lines() or (output or '').splitlines()
        scanned = markers.scan(lines)
        stats = stats if stats is not None else scanned.get('stats')
        judge.resolve(driver.records, scanned)
        result = judge_arm(arm, driver, scanned, stats, error)
        result.update(seconds=round(self.clock() - started, 1), metrics=dict(
            (key, value) for key, value in metrics.items() if key in (
                'vllm:num_preemptions', 'vllm:prefix_cache_queries', 'vllm:prefix_cache_hits',
                'vllm:prompt_tokens', 'vllm:prompt_tokens_cached')), stats=stats, error=error)
        if arm['scenario'] == 'timing':
            result['phases'] = timing_summary(driver)
            for phase, summary in sorted(result['phases'].items()):
                result['lines'].append(report.render_phase(phase, summary))
        if any(markers.WEDGE in (entry.get('line') or '') for entry in scanned.get('failures') or ()):
            self.infra = ('arm %s hit tt-metal\'s ethernet-core wedge (llrt.cpp:594): reset M+A before the next run'
                          % arm['arm'])
        self.write(arm_dir, driver, scanned, result)
        self.arms[arm['arm']] = result
        self.drivers[arm['arm']] = driver
        for line in result['lines']:
            self.log('[PREFIX-GATE] %s %s' % (arm['arm'], line))
        self.log('[PREFIX-GATE] arm %s: %s after %s s' % (arm['arm'], result['verdict'], result['seconds']))
        return driver, result

    def restart(self, container, client, follower, driver=None):
        """The reload drill: docker stop (graceful: the image's engine skips tt-metal's teardown),
        docker start, wait for the API (no longer than the arm has left), follow the new log.
        -> dict(seconds to ready, log_window: the follower's lines from the stop to ready)."""
        left = driver.remaining() if driver is not None else None
        if left is not None and left <= 60:
            raise replay.OutOfTime('no time left for the in-place restart')
        first = follower.mark()
        since = follower.last_time()
        started = self.clock()
        container.stop()
        follower.stop()
        container.start()
        follower.start(since=since)
        seconds = READINESS_SECONDS if left is None else max(60, min(READINESS_SECONDS, left - 60))
        problem = self.wait_ready(client, container, seconds)
        if problem:
            raise replay.EngineDead('after the in-place restart: %s' % problem)
        return dict(seconds=round(self.clock() - started, 1), log_window=[first, follower.mark()])

    @staticmethod
    def write(arm_dir, driver, scanned, result):
        with open(os.path.join(arm_dir, 'records.jsonl'), 'w', encoding='utf-8') as handle:
            handle.write(replay.records_jsonl(driver.records))
        with open(os.path.join(arm_dir, 'pairs.json'), 'w', encoding='utf-8') as handle:
            json.dump(driver.pairs, handle, indent=1)
        with open(os.path.join(arm_dir, 'events.json'), 'w', encoding='utf-8') as handle:
            json.dump(dict(events=driver.events, phases=driver.phases), handle, indent=1, default=str)
        summary = dict((key, scanned[key]) for key in ('installs', 'refused', 'capture_skipped', 'kill_switch', 'stats',
                                                        'launches', 'apc', 'chunking_off', 'chunk_replay', 'kv_tokens',
                                                        'failures', 'eager_warm'))
        summary.update(grants=len(scanned['grants']), rows=len(scanned['rows']), audits=len(scanned['audits']),
                       dram=scanned['dram'][:32], dram_readings=(scanned.get('dram_readings') or [])[:32])
        with open(os.path.join(arm_dir, 'arm.json'), 'w', encoding='utf-8') as handle:
            json.dump(dict(result=result, markers=summary), handle, indent=1, default=str)


def run_plan(plan, arms, runner, anchor=None):
    results = {}
    for arm in arms:
        driver, result = runner.run(arm)
        results[arm['arm']] = (driver, result)
    verdicts = [result['verdict'] for _, result in results.values()]
    lines = ['%s %s' % (name, result['verdict']) for name, (_, result) in results.items()]
    extra = {}
    if plan == 'bringup' and len(results) == 2 and all(driver is not None for driver, _ in results.values()):
        reference, prefix = results['bringup-reference'][0], results['bringup-prefix'][0]
        problems, missing, cross = bringup_cross(reference, prefix, anchor or {})
        dram_missing, dram_lines = dram_findings(results['bringup-prefix'][1])
        cross += dram_lines
        missing += dram_missing
        lines += cross + ['problem: %s' % text for text in problems] + ['not exercised: %s' % text for text in missing]
        verdicts.append('FAIL' if problems else ('NOT_EXERCISED' if missing else 'PASS'))
        extra.update(cross_problems=problems, cross_not_exercised=missing)
    if plan == 'timing':
        prefix = (results.get('timing-prefix') or (None, {}))[1].get('phases') or {}
        base = (results.get('timing-baseline') or (None, {}))[1].get('phases') or {}
        comparison = report.compare_phases(prefix, base)
        extra['comparison'] = comparison
        for phase, values in sorted(comparison.items()):
            lines.append('%s with reuse vs without: continuation TTFT p50 %s vs %s s, p90 %s vs %s s, turns/h %s vs %s'
                         % (phase, values['ttft_p50'][0], values['ttft_p50'][1], values['ttft_p90'][0],
                            values['ttft_p90'][1], values['turns_per_hour'][0], values['turns_per_hour'][1]))
    verdict = judge.worst(verdicts)
    if runner.infra is not None:
        verdict = 'INFRA'
    for line in lines:
        runner.log('[PREFIX-GATE] %s %s' % (plan, line))
    runner.log('[PREFIX-GATE] %s %s' % (plan, verdict))
    out = dict(verdict=verdict, arms=dict((name, result) for name, (_, result) in results.items()), lines=lines)
    out.update(extra)
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True)
    parser.add_argument('--profile', default='general-prefix')
    parser.add_argument('--baseline', default='general', help='the no-reuse profile, or none (timing only)')
    parser.add_argument('--plan', default='bringup', help='comma-separated: %s' % ', '.join(PLANS))
    parser.add_argument('--agents', default=','.join(str(count) for count in replay.TIMING_AGENTS))
    parser.add_argument('--turns', type=int, default=replay.TIMING_TURNS)
    parser.add_argument('--results', required=True)
    parser.add_argument('--budget-seconds', type=int, default=None)
    parser.add_argument('--checkout', default=os.path.dirname(os.path.dirname(HERE)))
    parser.add_argument('--profiles', default=None, help='a profiles JSON instead of the image\'s own')
    parser.add_argument('--hub', default=gate.HUB)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='print every arm\'s docker argv and run nothing')
    return parser


def main(argv=None, devices=None, log=print, runner_factory=None, anchor=None):
    options = build_parser().parse_args(argv)
    plans = c2_serving_job.split_list(options.plan)
    unknown = sorted(set(plans) - set(PLANS))
    if not plans or unknown:
        log('unknown plan(s): %s' % ', '.join(unknown or ['(none)']))
        return 2
    try:
        agents = [c2_serving_job.positive_int('--agents', part) for part in c2_serving_job.split_list(options.agents)]
    except c2_serving_job.JobError as error:
        log('refused: %s' % error)
        return 2
    if options.profiles:
        with open(options.profiles, encoding='utf-8') as handle:
            profiles = json.load(handle)
    else:
        profiles = gate.image_profiles(options.image)
    arms_of, ran = {}, {}
    for plan in plans:
        try:
            arms_of[plan] = plan_arms(plan, options.profile, options.baseline, profiles)
        except PlanError as error:
            log('refused: %s' % error)
            return 2
        for arm in arms_of[plan]:
            if arm['arm'] in ran:
                log('refused: arm %s would run twice (plans %s and %s) into one results directory'
                    % (arm['arm'], ran[arm['arm']], plan))
                return 2
            ran[arm['arm']] = plan
    worst_case = worst_case_seconds(arms_of)
    if options.budget_seconds is not None and worst_case > options.budget_seconds:
        log('refused: plans %s may take %d s (every arm to its limit), past the %d s this step and job leave: run '
            'fewer plans per tag' % (','.join(plans), worst_case, options.budget_seconds))
        return 2
    os.makedirs(options.results, exist_ok=True)
    if options.dry_run:
        log(json.dumps(dict(worst_case_seconds=worst_case, budget_seconds=options.budget_seconds)))
        for plan in plans:
            for arm in arms_of[plan]:
                log(json.dumps(dict(plan=plan, arm=arm['arm'], served=arm['served'], timeout=arm['timeout'],
                                    docker=server_run(options.image, CONTAINER_PREFIX + arm['arm'], arm['served'],
                                                      devices or ['<M>', '<A>'], options.port, options.hub,
                                                      '<results>/%s/profiles.json' % arm['arm'] if arm['derived']
                                                      else None, digests=wants_digests(arm),
                                                      salt_key_path='<results>/%s/salt.key' % arm['arm']
                                                      if arm.get('prefix') else None,
                                                      stats_now=bool(arm.get('prefix'))))))
        return 0
    runner = (runner_factory or Runner)(options.image, options.results, options.checkout,
                                        devices if devices is not None else gate.serving_pair(), options.hub,
                                        options.port, log=log, seed=options.seed, agents=agents, turns=options.turns)
    summary = dict(image=options.image, profile=options.profile, baseline=options.baseline, plans=plans,
                   worst_case_seconds=worst_case, budget_seconds=options.budget_seconds, results={})
    try:
        if 'bringup' in plans and anchor is None:
            anchor = anchor_probe(options.image)
        summary['anchor'] = anchor
        for plan in plans:
            summary['results'][plan] = run_plan(plan, arms_of[plan], runner, anchor)
    finally:
        summary['passed'] = bool(summary['results']) and len(summary['results']) == len(plans) and all(
            result['verdict'] == 'PASS' for result in summary['results'].values())
        summary['infra'] = runner.infra
        with open(os.path.join(options.results, 'c2-prefix-summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, default=str)
    log('C2_PREFIX profile=%s plans=%s passed=%s%s' % (options.profile, ','.join(plans), summary['passed'],
                                                       ' infra=%s' % runner.infra if runner.infra else ''))
    return 0 if summary['passed'] else 1


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main())
