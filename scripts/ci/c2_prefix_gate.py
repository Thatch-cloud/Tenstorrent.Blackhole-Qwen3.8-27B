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

PLANS (--plan, comma-separated, run in order):
  bringup    bringup-reference on the BASELINE profile (general): a three-turn real-text conversation;
             bringup-prefix on the prefix profile: the same conversation UNSALTED (fail-closed tenancy
             leaves reuse off inside a reuse engine: it must equal the reference byte for byte, with
             no grant and Q=0 throughout), then salted as cold/hit pairs - the first hit. PASS needs:
             the launched argv with the prefix cache on and async scheduling off at block size 64, the
             platform's 'Automatic prefix caching is enabled' and 'Chunked prefill is not supported
             ... disabling it', '[PINDIAG] prefix: install' (TTScheduler, block_size=64,
             QWEN_SDPA_BF8=1), '[TP chunk-replay]', the program cache unchanged across the first hit
             (F3: [PREFIX] programs=), the image's model tree: the five grafted files at the image's
             own graft.sha256 and a '[PREFIX]' marker in it (the anchor probe, recorded with every
             sha). A missing '[PINDIAG] dram' reading (for G2) makes it NOT_EXERCISED.
  exactness  exactness-traced: a chained real-text conversation 2.6k -> 60k (hits near 4k, 9k, 16k,
             24k, 33k, 42k, 51k, 60k), every turn cold (fresh salt) vs hit compared in full, a
             changed suffix and an early divergence forked at turn 3, previous prompts of exactly
             2047/2048/2049 tokens with tail-only hits, one tenant's three conversations sharing the
             full system block (gap capture); every Q against the oracle (prefix_judge.Oracle).
             exactness-audit: the short chain and the boundaries with QWEN_PREFIX_AUDIT=1 (derived):
             each hit's KV [0,L) and GDN slot digests equal its cold twin's. exactness-eager: the same
             with trace_mode decode_only (derived): the eager loop C1 reuses, rows path=eager, no
             '[TP chunk-replay]'.
  lifecycle  lifecycle-evict (VLLM_SERVER_DEV_MODE=1, derived): four arrivals at once (two hits, two
             same-tenant first turns), an abort while waiting for a seat, an abort during a hit's
             prefill, a KV flood that evicts conversations built to ~56k, reset_prefix_cache, the
             kill switch file (grants off, latched), an in-place restart (docker stop/start: the
             first turn after it misses, the next hits). lifecycle-store (QWEN_PREFIX_STORE_GIB=0.5,
             derived): checkpoint LRU eviction. lifecycle-tiny (num-gpu-blocks-override=TINY_BLOCKS,
             derived): allocation failure after a grant and preemption, ignore_eos answers. Every
             hit against a cold twin; preempted requests may diverge after they resume (recompute
             re-prefills their own output), which is reported, not failed, when preemptions happened.
  timing     timing-prefix and (unless --baseline none) timing-baseline: busy agents in the metering
             shape (~2k-token tool results, exponential 15 s gaps, full system block, compaction past
             60k) at --agents 1,4,5,6, one phase each; TTFT and turn time p50/p90, hit rate from the
             [PREFIX] rows' L-Q (never vllm:prefix_cache_hits), trim loss, restore/capture ms, RSS and
             the CI pod count per phase (prefix_report). It records; it fails only on failed turns
             or missing markers.

Verdicts per plan: PASS, FAIL, INFRA (a platform container on M+A, tt-metal's ethernet-core wedge),
NOT_COMPARABLE, UNSTABLE (the policy: a divergence that did not reproduce, or cold runs that
disagree), NOT_EXERCISED (a check whose event did not happen, or whose evidence was not logged).

SAFETY: as c2_serving_gate - no thatch-inference-* container may exist, a leftover container of the
same name is removed first, every container is removed however the arm ends (SIGTERM included), a
wedge stops the plan list as INFRA. The kill-switch file is written through the container into the
host's ~/hf-cache/hub/.qwen-c2 and removed only if its content is KILL_SWITCH_OWNER (the workflow's
trap does the same from the host). --budget-seconds refuses a plan list whose worst case does not fit.

Writes <results>/<arm>/ (server.log, records.jsonl, pairs.json, events.json, arm.json, docker-run.json,
profiles.json for a derived arm) and <results>/c2-prefix-summary.json; exits 0 only when every plan
passed, 2 when refused up front.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
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

PLANS = c2_serving_job.PREFIX_PLANS
PORT = 8021
CONTAINER_PREFIX = 'qwen-c2-prefix-'
DERIVED_MOUNT = '/prefix-gate/profiles.json'
READINESS_SECONDS = 1800
ARM_OVERHEAD_SECONDS = 180
SETTLE_SECONDS = 3.0
TINY_BLOCKS = 1280            # 81,920 tokens: above one 65,536-token request, below three ~25k conversations
SMALL_STORE_GIB = 0.5         # three checkpoints of CHECKPOINT_NBYTES
MODEL_ROOT = '/opt/tt-metal/models/demos/blackhole/qwen36/tt'
ANCHOR_FILES = ('model.py', 'qwen36_vllm.py', 'model_config.py', 'attention/tp.py', 'gdn/tp.py', 'mlp.py', 'layer.py')
GRAFTED = ('model_config.py', 'attention/tp.py', 'gdn/tp.py', 'mlp.py', 'layer.py')
KILL_SWITCH_OWNER = replay.KILL_SWITCH_OWNER
# What the platform hands vLLM (the smoke step's argv; TS injects --no-enable-prefix-caching for TT).
PLATFORM_ARGS = ['--model', 'Qwen/Qwen3.8-27B', '--served-model-name', replay.SERVED_NAME, '--host', '0.0.0.0',
                 '--port', '8000', '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_xml',
                 '--enable-auto-tool-choice', '--max-model-len', '65536', '--max-num-seqs', '2', '--block-size', '64',
                 '--no-enable-prefix-caching', '--additional-config',
                 json.dumps({'tt': {'l1_small_size': 24576, 'fabric_config': 'FABRIC_1D',
                                    'trace_region_size': 1073741824}})]
# (arm, scenario, which profile, derived changes, docker timeout seconds, strict oracle)
DERIVED = dict(
    audit=dict(env=dict(QWEN_PREFIX_AUDIT='1')),
    eager=dict(tt=dict(trace_mode='decode_only')),
    dev=dict(env=dict(VLLM_SERVER_DEV_MODE='1')),
    store=dict(env=dict(QWEN_PREFIX_STORE_GIB=str(SMALL_STORE_GIB))),
    tiny=dict(engine={'num-gpu-blocks-override': TINY_BLOCKS}, env=dict(VLLM_SERVER_DEV_MODE='1')),
)
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


class PlanError(ValueError):
    """A plan the image's profiles cannot run as asked; refused before any container starts."""


def is_prefix_profile(profile):
    return (str((profile.get('env') or {}).get('QWEN_PREFIX_REUSE', '0')) == '1'
            and (profile.get('engine') or {}).get('enable-prefix-caching') is True)


def derive(profiles, base, kind):
    """(derived name, a profiles document holding only it): the image's `base` profile with the
    DERIVED[kind] changes (profile env, engine flags, additional-config tt keys)."""
    changes = DERIVED[kind]
    profile = copy.deepcopy(profiles['profiles'][base])
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


def server_run(image, name, served, devices, port=PORT, hub=gate.HUB, derived_path=None):
    """`docker run -d` of one serving container: the S1 gate's agent shape (its --rm dropped: the
    reload drill stops and starts the same container), the API on 127.0.0.1:port, a derived
    profiles file when the arm has one, and the platform's vLLM argv."""
    arguments = [token for token in gate.agent_shape(image, name, served, devices, hub) if token != '--rm']
    arguments[2:2] = ['-d']
    arguments += ['-p', '127.0.0.1:%d:8000' % port]
    if derived_path:
        arguments += ['--mount', 'type=bind,src=%s,dst=%s,readonly' % (derived_path, DERIVED_MOUNT),
                      '-e', 'QWEN_C2_PROFILES=%s' % DERIVED_MOUNT]
    return arguments + ['--entrypoint', 'python3', image, '-m', 'vllm.entrypoints.openai.api_server'] + PLATFORM_ARGS


def anchor_probe(image, run=subprocess.run):
    """The image's model tree, read in a throwaway container (no devices, no network): the sha256 of
    model.py, qwen36_vllm.py and the five grafted files, the image's own graft.sha256 pins, and the
    files under the tree that carry a '[PREFIX]' marker."""
    script = ('cd %s && sha256sum %s; echo ==pins; cat /opt/qwen-c2/graft.sha256; echo ==markers; '
              'grep -rlF "[PREFIX]" . || true' % (MODEL_ROOT, ' '.join(ANCHOR_FILES)))
    try:
        result = run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'sh', image, '-c', script],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        text = (result.stdout or b'').decode('utf-8', 'replace')
    except (OSError, subprocess.SubprocessError) as error:
        return dict(error=repr(error)[:300])
    return parse_anchor(text)


def parse_anchor(text):
    files, pins, marked, section = {}, {}, [], 'files'
    for line in text.splitlines():
        if line.startswith('==pins'):
            section = 'pins'
            continue
        if line.startswith('==markers'):
            section = 'markers'
            continue
        if section == 'markers':
            if line.strip():
                marked.append(line.strip().lstrip('./'))
            continue
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            path = parts[1].lstrip('*')
            if section == 'files':
                files[path] = parts[0]
            elif path.startswith('graft/') and not path.endswith('.orig'):
                pins[path[len('graft/'):]] = parts[0]
    mismatched = sorted(path for path in GRAFTED if path not in files or files[path] != pins.get(path))
    return dict(files=files, pins=dict((path, pins.get(path)) for path in GRAFTED), mismatched=mismatched,
                marker_files=marked)


# -- judging one arm -----------------------------------------------------------------------------

def generic_problems(arm, scanned, records, error, expect_profile, store_gib=judge.DEFAULT_STORE_GIB):
    """Everything an arm fails on whatever its plan: the scenario's own error, failure lines in the
    log, the launched argv, and - on a prefix arm - the platform's prefix lines, the install marker,
    stale grants, markers against the oracle and the loop path. -> (problems, notes, not_exercised)."""
    problems, notes, missing = [], [], []
    if error:
        problems.append('the scenario stopped: %s' % error)
    # A traceback alone is recorded (the lifecycle drops streams on purpose, and a server may log
    # one for a closed socket); the fatal signatures - an engine death, a refused install, an
    # assertion (the model graft's grant check), the wedge - fail the arm.
    for entry in (scanned.get('failures') or [])[:16]:
        if entry['signature'] == markers.TRACEBACK:
            notes.append('server log traceback at line %s: %s' % (entry['index'], entry['line'][:200]))
        else:
            problems.append('server log: %s' % entry['line'][:300])
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
    path = 'eager' if arm.get('kind') == 'eager' else 'traced'
    other = 'traced' if path == 'eager' else 'eager'
    paths = sorted(set(str(row.get('path')) for r in records for row in (r.get('markers') or {}).get('rows') or ()
                       if row.get('path')))
    if other in paths:
        problems.append('[PREFIX] rows report path %s, the arm serves %s' % (', '.join(paths), path))
    elif paths and path not in paths:
        notes.append('[PREFIX] rows report path %s only (no %s row)' % (', '.join(paths), path))
    if path == 'traced' and not scanned.get('chunk_replay'):
        problems.append('no "[TP chunk-replay]" line: the traced chunk loop never ran')
    if path == 'eager' and scanned.get('chunk_replay'):
        problems.append('%d "[TP chunk-replay]" lines on the eager arm: trace_mode decode_only did not reach the '
                        'model' % scanned['chunk_replay'])
    return problems, notes, missing


def pair_problems(pairs, allow_divergence_reason=None):
    """Pair verdicts as (problems, unstable, not_comparable, rerun)."""
    problems, unstable, not_comparable, rerun = [], [], [], []
    for pair in pairs:
        text = '%s %s turn %s (L=%s, %s vs %s): %s%s' % (pair['case'], pair['conv'], pair['turn'], pair['prompt_tokens'],
                                                       pair['cold'], pair['hit'], pair['verdict'],
                                                       ' - %s' % pair['detail'] if pair.get('detail') else '')
        if pair['verdict'] in ('DIVERGED', 'ERROR'):
            if allow_divergence_reason and pair['verdict'] == 'DIVERGED':
                not_comparable.append(text + ' (%s)' % allow_divergence_reason)
            else:
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
        if len(turns) < 2:
            missing.append('%s: fewer than two turns ran' % name)
            continue
        second = turns[1]
        q, length = q_of(second), second.get('prompt_tokens') or 0
        want = judge.floor_chunk(target)
        lines.append('%s: turn 2 L=%s Q=%s (want %s)' % (name, length, q, want))
        if q is not None and q != want:
            (problems if q > want else missing).append('%s: turn 2 restored Q=%s, the design gives %d' % (name, q, want))
        if want and q == want and judge.floor_chunk(length) == q:
            lines.append('%s: a tail-only hit (no full chunk after Q)' % name)
    tail_only = [r for r in records if r.get('role') == 'hit' and q_of(r) and judge.floor_chunk(r.get('prompt_tokens') or 0) == q_of(r)]
    if not tail_only:
        missing.append('no tail-only hit (Q = floor2048(L)) ran')
    if arm['arm'] == 'exactness-traced':
        for case in ('changed-suffix', 'early-divergence', 'shared-system'):
            found = hits(records, case)
            lines.append('%s (L, Q): %s' % (case, [(r.get('prompt_tokens'), q_of(r)) for r in found]))
            if not any(q_of(r) for r in found):
                missing.append('%s: no hit with Q > 0' % case)
        suffix, early = hits(records, 'changed-suffix'), hits(records, 'early-divergence')
        if suffix and early and q_of(suffix[0]) and q_of(early[0]) is not None and q_of(early[0]) >= q_of(suffix[0]):
            missing.append('the early divergence restored Q=%s, not below the changed suffix\'s %s: it did not fall '
                           'back to an older checkpoint' % (q_of(early[0]), q_of(suffix[0])))
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


def lifecycle_findings(arm, records, events, scanned, stats):
    """The lifecycle events against what each must leave. -> (problems, not exercised, lines, reason
    a divergence is allowed or None)."""
    problems, missing, lines, allow = [], [], [], None
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
        elif (aborted.get('markers') or {}).get('grant') or (aborted.get('markers') or {}).get('rows'):
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
    elif arm['arm'] == 'lifecycle-store':
        after = hits(records, 'store-after')
        lines.append('after the store filled (L, Q, oracle Q): %s' % [(r.get('prompt_tokens'), q_of(r),
                                                                      (r.get('expected') or {}).get('q')) for r in after])
        if not any(q_of(r) is not None and q_of(r) < (r.get('expected') or {}).get('q', 0) for r in after):
            missing.append('no checkpoint was pushed out of the small store (the next turn hit as if unbounded)')
        if stats and not stats.get('evicted_lru'):
            missing.append('the registry counted no LRU eviction')
    elif arm['arm'] == 'lifecycle-tiny':
        tiny = events.get('tiny') or {}
        lines.append('tiny pool: %s' % json.dumps(tiny))
        if not tiny.get('ok'):
            problems.append('a request on the tiny pool failed')
        if (tiny.get('preemptions') or 0) > 0:
            allow = 'a preempted request re-prefills its own output on resume, not a cold equivalent'
        else:
            missing.append('no preemption happened on the tiny pool')
        if stats is None:
            missing.append('no registry stats export: an allocation failure after a grant is not observable')
        elif not stats.get('dropped_attempts'):
            missing.append('the registry counted no dropped admission attempt (no allocation failure after a grant)')
    if stats is not None:
        lines.append('registry stats: %s' % json.dumps(stats, sort_keys=True))
        if stats.get('pins'):
            problems.append('%s checkpoint pins held after every request ended' % stats['pins'])
        if stats.get('commit_mismatch'):
            problems.append('%s grants refused at commit (start_pos != Q)' % stats['commit_mismatch'])
    return problems, missing, lines, allow


def judge_arm(arm, driver, scanned, stats, error):
    """One arm's verdict dict (the plan's cross-arm checks come after, in judge_plan)."""
    records = driver.records
    problems, notes, missing = generic_problems(arm, scanned, records, error, arm['served'],
                                                SMALL_STORE_GIB if arm.get('kind') == 'store' else judge.DEFAULT_STORE_GIB)
    lines = []
    allow = None
    if arm['scenario'].startswith('exactness'):
        more_missing, more_problems, more_lines = exercised_exactness(arm, records, driver.events)
        missing += more_missing
        problems += more_problems
        lines += more_lines
        if arm.get('kind') == 'audit':
            audit_fail, audit_missing = audit_findings(records, driver.pairs)
            problems += audit_fail
            missing += audit_missing
    elif arm['scenario'].startswith('lifecycle'):
        more_problems, more_missing, more_lines, allow = lifecycle_findings(arm, records, driver.events, scanned, stats)
        problems += more_problems
        missing += more_missing
        lines += more_lines
    diverged, unstable, not_comparable, rerun = pair_problems(driver.pairs, allow)
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
                dram=scanned.get('dram')[:16] if scanned.get('dram') else [], kv_tokens=scanned.get('kv_tokens'))


def bringup_cross(reference, prefix, anchor):
    """The bring-up's cross-arm checks: the prefix arm's unsalted turns against the baseline's
    reference turns byte for byte, the first hit's program cache, the anchor probe, the DRAM line.
    -> (problems, not exercised, lines)."""
    problems, missing, lines = [], [], []
    refs = [r for r in reference.records if r.get('role') == 'reference']
    unsalted = [r for r in prefix.records if r.get('role') == 'unsalted']
    if not refs or len(refs) != len(unsalted):
        problems.append('the reference ran %d turns and the grants-disabled arm %d' % (len(refs), len(unsalted)))
    for ref, mine in zip(refs, unsalted):
        result = judge.compare(ref, mine)
        lines.append('grants disabled vs %s, turn %s (L=%s): %s%s' % (
            'baseline', mine.get('turn'), mine.get('prompt_tokens'), result['verdict'],
            ' - %s' % result['detail'] if result.get('detail') else ''))
        if result['verdict'] != 'IDENTICAL':
            problems.append('with QWEN_PREFIX_REUSE=1 and no grant, turn %s is not byte-identical to the baseline '
                            '(%s): %s' % (mine.get('turn'), result['verdict'], result.get('detail')))
    program, detail = judge.program_cache_problems(prefix.records)
    lines.append('program cache across the first hit: %s' % json.dumps(detail))
    problems += program
    if anchor.get('error'):
        problems.append('the anchor probe could not read the image: %s' % anchor['error'])
    else:
        lines.append('anchor: %s' % json.dumps(anchor, sort_keys=True))
        if anchor.get('mismatched'):
            problems.append('the served model tree is not the image\'s pinned graft: %s' % ', '.join(anchor['mismatched']))
        if not anchor.get('marker_files'):
            problems.append('no file under %s carries a "[PREFIX]" marker: the prefix model graft is not in the image'
                            % MODEL_ROOT)
    return problems, missing, lines


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
        arguments = server_run(self.image, name, arm['served'], self.devices, self.port, self.hub, derived_path)
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
                               strict=arm['strict'])
        error, stats, metrics, output = None, None, {}, ''
        try:
            code, output = self.docker(arguments, 300)
            if code != 0:
                raise replay.EngineDead('docker run exited %s: %s' % (code, output[-400:]))
            follower.start()
            problem = self.wait_ready(client, container)
            if problem:
                raise replay.EngineDead(problem)
            self.log('[PREFIX-GATE] arm %s: ready after %.0f s' % (arm['arm'], self.clock() - started))
            kwargs = {}
            if arm['scenario'] in ('lifecycle_evict', 'lifecycle_tiny'):
                kwargs['pool_tokens'] = markers.scan(follower.lines()).get('kv_tokens')
            if arm['scenario'] == 'lifecycle_evict':
                kwargs['restart'] = lambda: self.restart(container, client, follower)
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

    def restart(self, container, client, follower):
        """The reload drill: docker stop (graceful: the image's engine skips tt-metal's teardown),
        docker start, wait for the API, follow the new log. -> seconds to ready."""
        since = follower.last_time()
        started = self.clock()
        container.stop()
        follower.stop()
        container.start()
        follower.start(since=since)
        problem = self.wait_ready(client, container)
        if problem:
            raise replay.EngineDead('after the in-place restart: %s' % problem)
        return round(self.clock() - started, 1)

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
                                                        'failures'))
        summary.update(grants=len(scanned['grants']), rows=len(scanned['rows']), audits=len(scanned['audits']),
                       dram=scanned['dram'][:32])
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
        dram = results['bringup-prefix'][1].get('dram')
        cross.append('dram: %s' % (dram or 'none logged'))
        if not dram:
            missing.append('no "[PINDIAG] dram" line on the prefix arm: G2 has no DRAM reading')
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
    arms_of = {}
    for plan in plans:
        try:
            arms_of[plan] = plan_arms(plan, options.profile, options.baseline, profiles)
        except PlanError as error:
            log('refused: %s' % error)
            return 2
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
                                                      else None))))
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
