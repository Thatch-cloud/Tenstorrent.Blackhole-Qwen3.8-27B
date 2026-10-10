"""The C2 gate's TP4 op-level profile plans (c2_serving_job.OPS_GATE_PLANS): ops-twin and ops-trace, the arms around
them, and what runs after each: handing the profiler's root-owned output back, compressing the CPP device report and
analysing it (tp4_profile_report). Attribution only: neither arm is a timing result for a placement, and neither is
judged against v235 (docs/tp4-profile.md).

    python3 scripts/ci/ops_profile_plan.py handback --results <gate results> --image <image>

handback is the workflow's always() step: it hands every ops arm's profile tree back to the runner user and prunes the
files the gate did not get to - the root-owned-logs trap (a cancelled profile run left root-owned logs and the next
checkout died with EACCES) - even when the gate step was cancelled.

PLANS, one arm each, run after every judged plan of the job (c2_serving_job refuses any other order), the twin first:
  ops-twin   the four-card TIMED profile (c2-packed-tp4-speed: the verify's T1/T2 audits off, the drafter's K/V slide on),
             served bytes, no profiler: four real-text users at 4,096 tokens each, the production shape (v170 had 4k / 8k / 16k / 24k, so the attention slope is v170's; the
             exactness divergence at 32k and above is unresolved), user 0 asked for 256 tokens and users 1-3 for 128 (enough 4-live rounds at ~6 accepted tokens each), all
             with ignore_eos. The unperturbed round times and the texts the profiled arm is compared with.
  ops-trace  the same, profiled with v138's tracy recipe (the qualified op-level recipe): tracy -p, trace tracking, the
             CPP post-process only (--disable-device-data-dump-to-files: no raw profile_log_device.csv, which is tens
             of GB), op-support 20000 (200000 segfaulted the dispatch thread three times; anything else is refused),
             and a read-back after every second verify replay of ANY kind (QWEN_FAST_PROFILE_DUMP_EVERY, packed_verifier),
             so no window overflows the per-core buffer. No prefill flush (never run at 20000).
One container per arm yields 4-live rounds, then 3- and 2-live padded rounds as users 1-3 finish, then user 0 alone on the
1/2/4-row sequential engines (a lone user has no 16-row step on this profile: tp4_profile_report composes it).

PREFILL PLANS (docs/tp4-profile.md, 'Prefill profile'): ops-prefill-twin and ops-prefill-trace, the same pair around ONE user and ONE
real-text coding prompt of PREFILL_TOKENS = 131,072 tokens (64 prefill chunks of 2,048: chunk 0 is the short context, chunk 63 the
long one, so one prompt gives every context) and an 8-token answer. The twin is unprofiled (the text reference and the unperturbed
prefill wall time); the trace arm adds QWEN_PREFILL_PROFILE_FLUSH=1 to the profiler environment (the layer.py hook in the image's
graft drains the device profiler every 16 decoder layers, so no chunk overflows the per-core buffers) and its CPP report is
analysed by tp4_prefill_profile_report (per-op ms per chunk, matmul efficiency, attention against context, the GDN chunked
prefill, glue copies, collectives) into prefill-profile-report.{json,md} beside the compressed report. op-support stays 20000.
The flush hook has NEVER run at TP4 or at op-support 20000: these two plans are DATA, not a gate (a job that names them runs them
soft). NEVER cancel an ops-* job once its gate step has started: the profiler writes as root and a cancelled run leaves root-owned
logs (the next checkout dies with EACCES); the workflow's always() hand-back step covers the prefill arm too.

Verdicts: PASS, FAIL (a harness failure, a stream without text, garbage, or ops-trace texts differing from ops-twin's -
a difference is reported as a FAIL to look at: profiling shifts scheduling, and the exactness divergence at 32k and above is unresolved), NOT_EXERCISED (the disk guard stopped the arm, or no twin to
compare with), INFRA (the gate's own). A report that finds too few complete sessions says so in the plan's lines and
does not fail the plan: the product is data.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import gzip
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading

import c2_serving_job

PLANS = c2_serving_job.OPS_GATE_PLANS
TWIN = 'ops-twin'
TRACE = 'ops-trace'
PREFILL_TWIN = 'ops-prefill-twin'
PREFILL_TRACE = 'ops-prefill-trace'
PREFILL_PLANS = (PREFILL_TWIN, PREFILL_TRACE)
PLAN_KINDS = {TWIN: 'twin', TRACE: 'ops', PREFILL_TWIN: 'twin', PREFILL_TRACE: 'ops'}
# Each profiled plan's twin (the text reference and the perturbation reference): its arm directory is the twin plan's name.
PLAN_TWIN = {TRACE: TWIN, PREFILL_TRACE: PREFILL_TWIN}
# The shapes are fixed here, not in the job file, so a job cannot drift from what the analysis expects.
USERS = 4
LENGTHS = (4096, 4096, 4096, 4096)     # the production shape (4 x 4k coding); v170 had 4k / 8k / 16k / 24k
MAX_TOKENS = 128
USER_MAX_TOKENS = ((0, 256),)
# An eight-seat profile (engine max-num-seqs 8: two 64-row blocks) is profiled with eight of the same 4k users: the shape per user is unchanged,
# only the count follows the seats (tp4/262k8). Every four-seat profile keeps the four users it always had, byte for byte.
SEAT_USERS = 8
# The prefill plans' shape: one user, one prompt, a short answer (fixed here for the same reason: the analysis expects it).
PREFILL_USERS = 1
PREFILL_TOKENS = 131072
MAX_TOKENS_PREFILL = 8
PREFILL_CHUNK = 2048
FLUSH_FLAG = 'QWEN_PREFILL_PROFILE_FLUSH'
FLUSH_MARKER = '[PINDIAG] prefill profile flush'
# The hook logs one line per chosen chunk's begin (chunks 0, 1, 31, 32, 62, 63) and one for its first flush.
MIN_FLUSH_MARKERS = 7
# Docker limits (seconds): readiness (a profiled arm compiles every profiler-define kernel cold into its tmpfs) plus the
# stream timeout plus the close (the last read-back, tracy's post-process).
READINESS_SECONDS = 1800
STREAM_SECONDS = {TWIN: 1800, TRACE: 3600, PREFILL_TWIN: 3600, PREFILL_TRACE: 5400}
CLOSE_SECONDS = {TWIN: 300, TRACE: 900, PREFILL_TWIN: 900, PREFILL_TRACE: 900}
ARM_SECONDS = dict((plan, READINESS_SECONDS + STREAM_SECONDS[plan] + CLOSE_SECONDS[plan]) for plan in PLANS)
OP_SUPPORT = 20000
OP_SUPPORT_LIMIT = 20000
DUMP_EVERY = 2
PROFILE_DIR = '/opt/tt-metal/generated/profiler'
PROFILE_SUBDIR = 'ops-profile'
SCRATCH_CACHE = '/root/.cache/tt-metal-cache'
DUMP_FLAG = 'QWEN_FAST_PROFILE_DUMP_EVERY'
PROFILER_ENV = (('TT_METAL_DEVICE_PROFILER', '1'), ('TT_METAL_PROFILER_TRACE_TRACKING', '1'),
                ('TT_METAL_PROFILER_MID_RUN_DUMP', '1'), ('TT_METAL_PROFILER_CPP_POST_PROCESS', '1'),
                ('TTNN_OP_PROFILER', '1'), ('TT_METAL_PROFILER_DIR', PROFILE_DIR),
                ('QWEN_FAST_PROFILED_BLOCK_STREAM', '1'), ('TT_METAL_CACHE', SCRATCH_CACHE),
                (DUMP_FLAG, str(DUMP_EVERY)))
ARM_ENV_NAMES = frozenset(name for name, _ in PROFILER_ENV)
# The prefill trace arm alone adds the flush hook's switch (arm_env); a decode plan's environment never carries it.
PREFILL_ONLY_ENV = (FLUSH_FLAG,)
# What the harness's qwen_configuration records (QWEN*_ and TT_ names): the rest cannot be shown to arrive.
CONFIGURATION_PREFIX = re.compile(r'(?:QWEN[0-9]*_|TT_)')
PATH_ENV = frozenset(['TT_METAL_PROFILER_DIR', 'TT_METAL_CACHE'])
# What the timed profile must carry (the twin's whole point: a verify trace without the audits inside it).
REQUIRED_CONFIGURATION = (('QWEN_FAST_TP', '4'), ('QWEN_FAST_VERIFY_T1_AUDIT', '0'), ('QWEN_FAST_VERIFY_T2_AUDIT', '0'))
AUDIT_MARKERS = ('[PINDIAG] verify t1 audit', '[PINDIAG] verify t2 audit')
READBACK_MARKER = '[PINDIAG] device profiler read back after replay'
MIN_READBACKS = 20
# The disk (DiskGuard): the CPP report is about 0.2 GB over four chips; the raw device log is suppressed.
GB = 2 ** 30
PROFILE_CAP = 4 * GB
DISK_STOP_USED = 0.85
GUARD_SECONDS = 20
KEEP_BYTES = 8 * 2 ** 20            # a profile file larger than this leaves the results tree after it is compressed
OPS_SUBDIR = 'ops'                  # <results>/ops/: what a reader downloads
CSV_NAME = 'cpp_device_perf_report.csv'
CSV_GZ = CSV_NAME + '.gz'
REPORT_JSON = 'tp4-profile-report.json'
REPORT_MD = 'tp4-profile-report.md'
# The prefill trace arm keeps its own copy and report beside the decode arm's (a job may name both pairs).
PREFILL_CSV_GZ = 'cpp_device_perf_report.prefill.csv.gz'
PREFILL_REPORT_JSON = 'prefill-profile-report.json'
PREFILL_REPORT_MD = 'prefill-profile-report.md'
PRUNED = 'ops-pruned.json'
# What no served profile may carry: every profiler variable.
PROFILE_REFUSED = re.compile(r'(?:TT_METAL_DEVICE_PROFILER|TT_METAL_PROFILE|TT_METAL_PROFILER_|TTNN_OP_PROFILER|QWEN_FAST_PROFILE_|QWEN_PREFILL_PROFILE_)')
REFUSED_TRACY = ('--device-trace-profiler', '--profile-dispatch-cores', '--enable-sum-profiling',
                 '--profiler-capture-perf-counters')


class OpsPlanError(ValueError):
    """An ops plan the gate refuses before any container starts."""


# ---- plans and arms ----

def tracy_args():
    """The tracy wrapper's options for the profiled arm: v138's recipe."""
    args = ['-p', '--check-exit-code', '--disable-device-data-dump-to-files', '--disable-device-data-push-to-tracy',
            '--dump-device-data-mid-run', '--op-support-count', str(OP_SUPPORT), '-o', PROFILE_DIR]
    return check_tracy(args)


def check_tracy(args):
    """Refuse a tracy argument list the pass must never run."""
    for refused in REFUSED_TRACY:
        if refused in args:
            raise OpsPlanError('tracy %s is refused on an ops arm (not the qualified recipe)' % refused)
    if '--disable-device-data-dump-to-files' not in args:
        raise OpsPlanError('an ops arm passes --disable-device-data-dump-to-files (the raw device log is tens of GB; '
                           'the CPP report is still written)')
    if '--op-support-count' not in args:
        raise OpsPlanError('an ops arm states its op-support count (%d)' % OP_SUPPORT)
    count = args[args.index('--op-support-count') + 1]
    if not count.isdigit() or not 0 < int(count) <= OP_SUPPORT_LIMIT:
        raise OpsPlanError('op-support count %s refused: at most %d (200000 segfaulted the dispatch thread)'
                           % (count, OP_SUPPORT_LIMIT))
    output = args[args.index('-o') + 1] if '-o' in args else None
    if output is None or posixpath.normpath(output) != PROFILE_DIR:
        raise OpsPlanError('tracy output must be %s (the arm\'s bind-mounted %s, and its TT_METAL_PROFILER_DIR), got %r'
                           % (PROFILE_DIR, PROFILE_SUBDIR, output))
    return args


def arm_env(kind, plan=None):
    """The arm's added environment: none for a twin; the profiler's for a profiled arm, and for the prefill trace arm alone also the
    flush hook's switch."""
    if kind == 'twin':
        return ()
    return PROFILER_ENV + (((FLUSH_FLAG, '1'),) if plan == PREFILL_TRACE else ())


def check_env(env):
    allowed = ARM_ENV_NAMES | frozenset(PREFILL_ONLY_ENV)
    for name, value in env:
        if name not in allowed:
            raise OpsPlanError('%s is not an environment an ops arm may add (%s)' % (name, ', '.join(sorted(allowed))))
        if name == DUMP_FLAG and not (value.isdigit() and int(value) >= 1):
            raise OpsPlanError('%s must be a positive replay count, got %r' % (DUMP_FLAG, value))
        if name == FLUSH_FLAG and value != '1':
            raise OpsPlanError('%s must be 1 (the hook is on or absent), got %r' % (FLUSH_FLAG, value))
    return env


def check_profiles(profiles):
    """No profile of the image may carry a profiler variable (a served container is never profiled)."""
    for name, profile in sorted(profiles['profiles'].items()):
        carried = sorted(key for key in (profile.get('env') or {}) if PROFILE_REFUSED.match(key))
        if carried:
            raise OpsPlanError('profile %s carries %s: a served profile never runs the profiler' % (name, ', '.join(carried)))


def check_timed_profile(profiles, profile):
    """The plans profile the four-card TIMED verify: a profile at QWEN_FAST_TP=4 with both verify audits off."""
    env = (profiles['profiles'][profile].get('env') or {})
    for name, want in REQUIRED_CONFIGURATION:
        if str(env.get(name, '')) != want:
            raise OpsPlanError('profile %s has %s=%r: the ops plans profile the four-card timed verify (%s)' % (
                profile, name, env.get(name), ', '.join('%s=%s' % pair for pair in REQUIRED_CONFIGURATION)))


def plan_users(profiles, profile):
    """How many 4k users the plan runs on `profile`: the eight seats of an eight-seat profile, else USERS."""
    seats = ((profiles['profiles'].get(profile) or {}).get('engine') or {}).get('max-num-seqs')
    return SEAT_USERS if seats == SEAT_USERS else USERS


def plan_arms(plan, profile, profiles, gate):
    """[gate.Arm] of one ops-* plan on `profile`, refused (gate.PlanError) where the profile cannot serve it."""
    if plan not in PLAN_KINDS:
        raise gate.PlanError('unknown ops plan %r' % plan)
    try:
        check_profiles(profiles)
        if profile not in profiles['profiles']:
            raise OpsPlanError('%s: profile %s is not in the image' % (plan, profile))
        check_timed_profile(profiles, profile)
        kind = PLAN_KINDS[plan]
        ops = dict(plan=plan, kind=kind, env=check_env(arm_env(kind, plan)), tracy=tracy_args() if kind == 'ops' else None)
    except OpsPlanError as error:
        raise gate.PlanError(str(error))
    context, ceiling, room = gate.profile_limits(profiles, profile)
    if plan in PREFILL_PLANS:
        # One user, one 131,072-token prompt, an 8-token answer: every chunk of the prefill under one profile (chunk 0 short, chunk 63 long).
        ops['users'] = PREFILL_USERS
        ops['shape'] = 'prefill'
        gate.check_lengths(profile, [PREFILL_TOKENS], room, '%s prompt lengths' % plan)
        gate.check_budget(profile, MAX_TOKENS_PREFILL, ceiling, '%s --max-tokens' % plan)
        args = gate.common_args(profile, context, STREAM_SECONDS[plan], readiness=READINESS_SECONDS) + [
            '--users', str(PREFILL_USERS), '--prompt-lengths', str(PREFILL_TOKENS),
            '--max-tokens', str(MAX_TOKENS_PREFILL), '--user-ignore-eos', '0', '--stagger', str(gate.STAGGER)]
        return [gate.Arm(plan, args, ARM_SECONDS[plan], rerun=False, judged=False, role='ops', ops=ops)]
    users = plan_users(profiles, profile)
    lengths = LENGTHS if users == USERS else (LENGTHS[0],) * users
    if users != USERS:
        ops['users'] = users
    gate.check_lengths(profile, list(lengths), room, '%s prompt lengths' % plan)
    gate.check_budget(profile, max(MAX_TOKENS, max(tokens for _, tokens in USER_MAX_TOKENS)), ceiling,
                      '%s --max-tokens' % plan)
    args = gate.common_args(profile, context, STREAM_SECONDS[plan], readiness=READINESS_SECONDS) + [
        '--users', str(users), '--prompt-lengths', ','.join(str(length) for length in lengths),
        '--max-tokens', str(MAX_TOKENS),
        '--user-max-tokens', ','.join('%d:%d' % pair for pair in USER_MAX_TOKENS),
        '--user-ignore-eos', ','.join(str(user) for user in range(users)), '--stagger', str(gate.STAGGER)]
    return [gate.Arm(plan, args, ARM_SECONDS[plan], rerun=False, judged=False, role='ops', ops=ops)]


def planned(arm_dir, ops):
    """What prepare_arm will add to the arm's docker run, without touching the disk (--dry-run)."""
    mounts = []
    if ops['kind'] != 'twin':
        mounts = ['--mount', 'type=bind,src=%s,dst=%s' % (os.path.join(arm_dir, PROFILE_SUBDIR), PROFILE_DIR)]
    return dict(ops, mounts=mounts, env=check_env(ops['env']))


def prepare_arm(arm_dir, ops):
    """The arm's profile directory, writable whatever user the container runs as (a harmless precaution), and the additions
    gate_run makes."""
    prepared = planned(arm_dir, ops)
    if ops['kind'] != 'twin':
        path = os.path.join(arm_dir, PROFILE_SUBDIR)
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o777)
    return prepared


def docker_additions(prepared):
    """(env pairs, mounts, entry args) a prepared arm adds to gate_run; entry args replace '-B'."""
    env = check_env(prepared.get('env') or ())
    entry = None
    if prepared.get('tracy'):
        entry = ['-B', '-m', 'tracy'] + list(check_tracy(list(prepared['tracy'])))
    return env, list(prepared.get('mounts') or []), entry


# ---- the disk ----

def tree_bytes(path):
    total = 0
    for directory, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(directory, name))
            except OSError:
                continue
    return total


def docker_stop(name, timeout=120):
    return subprocess.run(['docker', 'stop', '-t', '30', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=timeout).returncode


class DiskGuard(object):
    """While the profiled arm runs: stop its container (stop()) once its profile tree passes `cap` bytes or the disk
    passes DISK_STOP_USED. tripped holds why, or None."""

    def __init__(self, path, cap, stop, usage=shutil.disk_usage, size=tree_bytes, interval=GUARD_SECONDS, log=print):
        self.path, self.cap, self.stop, self.usage, self.size = path, cap, stop, usage, size
        self.interval, self.log = interval, log
        self.tripped = None
        self._done = threading.Event()
        self._thread = None

    def check(self):
        written = self.size(self.path)
        if written > self.cap:
            return 'the profile tree reached %.1f GB (cap %.0f GB)' % (written / float(GB), self.cap / float(GB))
        total, used, _ = self.usage(self.path)
        if used / float(total) > DISK_STOP_USED:
            return 'the disk reached %.1f%% used (the guard stops at %.0f%%)' % (100.0 * used / total, 100 * DISK_STOP_USED)
        return None

    def _run(self):
        while not self._done.wait(self.interval):
            try:
                reason = self.check()
            except Exception as error:   # the guard never takes the arm down by itself failing
                self.log('[C2-GATE] disk guard check failed: %r' % (error,))
                continue
            if reason:
                self.tripped = reason
                self.log('[C2-GATE] disk guard: %s; stopping the arm' % reason)
                try:
                    self.stop()
                except Exception as error:
                    self.log('[C2-GATE] disk guard: the stop failed: %r' % (error,))
                return

    def start(self):
        self._thread = threading.Thread(target=self._run, name='ops-disk-guard')
        self._thread.daemon = True
        self._thread.start()
        return self

    def finish(self):
        self._done.set()
        if self._thread is not None:
            self._thread.join(self.interval + 5)
        return self.tripped


def disk_problem(path, need, usage=shutil.disk_usage, max_used=0.80):
    """Why a profiled arm may not start (the results disk would pass the rig's no-build-above-80% rule with `need` bytes
    written), or None."""
    total, used, _ = usage(path)
    if (used + need) / float(total) > max_used:
        return 'the results disk is %.1f%% used and the arm writes up to %.1f GB: past %.0f%%' % (
            100.0 * used / total, need / float(GB), 100 * max_used)
    return None


# ---- what runs around the arm ----

def docker_handback(image, directory, timeout=300):
    """Hand a tree the container's root wrote back to this user (the EACCES trap): a throwaway container with the tree
    mounted, no network, no devices."""
    return subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--mount', 'type=bind,src=%s,dst=/p' % directory,
                           '--entrypoint', 'sh', image, '-c', 'chown -R %d:%d /p; chmod -R a+rwX /p'
                           % (os.getuid(), os.getgid())],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout).returncode


def handback_arm(image, arm_dir, prepared, handback):
    """The arm's profile tree handed back to this user (called in a finally: however the arm ended). Never raises: a
    failed hand-back is reported, and the workflow's always() step tries again."""
    if prepared['kind'] == 'twin':
        return None
    profile = os.path.join(arm_dir, PROFILE_SUBDIR)
    if not os.path.isdir(profile):
        return None
    try:
        return handback(image, profile)
    except Exception as error:
        return 'failed: %s' % error


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prune(profile, keep_bytes=None):
    """Every file above keep_bytes out of the results tree, each recorded with its size and sha256 first."""
    keep_bytes = KEEP_BYTES if keep_bytes is None else keep_bytes
    pruned = []
    for directory, _, names in os.walk(profile):
        for name in names:
            path = os.path.join(directory, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > keep_bytes:
                pruned.append(dict(path=os.path.relpath(path, profile), bytes=size, sha256=file_sha256(path)))
                os.remove(path)
    return pruned


def find_csv(profile):
    """The CPP device report under a profile tree (tracy writes it under <-o>/.logs), or None."""
    for directory, _, names in os.walk(profile):
        if CSV_NAME in names:
            return os.path.join(directory, CSV_NAME)
    return None


def gzip_file(source, target):
    with open(source, 'rb') as reading, gzip.open(target, 'wb', compresslevel=6) as writing:
        shutil.copyfileobj(reading, writing, 1 << 20)
    return os.path.getsize(target)


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=1, sort_keys=True)


def artifact_names(plan):
    """(compressed CPP report, report json, report md) a profiled plan keeps in <results>/ops/: the decode arm's names unchanged, the prefill
    arm's its own."""
    if plan == PREFILL_TRACE:
        return PREFILL_CSV_GZ, PREFILL_REPORT_JSON, PREFILL_REPORT_MD
    return CSV_GZ, REPORT_JSON, REPORT_MD


def finish_arm(arm_dir, results, twin_arm_dir=None, log=print, analyse=None, plan=None):
    """After the profiled arm (and handback_arm): compress the CPP report into <results>/ops/, run the analysis over it
    with both arms' server logs and gate reports, and prune the raw tree. Returns the arm's summary (never raises).
    `plan` is the profiled plan (None: the decode trace): the prefill trace plan keeps its own names and runs the prefill analysis."""
    summary = dict(kind='ops')
    profile = os.path.join(arm_dir, PROFILE_SUBDIR)
    out = os.path.join(results, OPS_SUBDIR)
    os.makedirs(out, exist_ok=True)
    csv_name, json_name, md_name = artifact_names(plan)
    csv_path = find_csv(profile) if os.path.isdir(profile) else None
    if csv_path is None:
        summary['problem'] = 'no %s under %s: tracy wrote no CPP report (the arm ended before its first read-back?)' % (
            CSV_NAME, PROFILE_SUBDIR)
    else:
        try:
            summary['csv_bytes'] = os.path.getsize(csv_path)
            summary['csv_gz_bytes'] = gzip_file(csv_path, os.path.join(out, csv_name))
            if analyse is None:
                if plan == PREFILL_TRACE:
                    import tp4_prefill_profile_report as analyse
                else:
                    import tp4_profile_report as analyse
            report = analyse.analyse_files(
                os.path.join(out, csv_name), server_log=os.path.join(arm_dir, 'server.log'),
                gate_json=os.path.join(arm_dir, 'm3native-gate.json'),
                twin_log=os.path.join(twin_arm_dir, 'server.log') if twin_arm_dir else None,
                twin_json=os.path.join(twin_arm_dir, 'm3native-gate.json') if twin_arm_dir else None)
            write_json(os.path.join(out, json_name), report)
            with open(os.path.join(out, md_name), 'w', encoding='utf-8') as handle:
                handle.write(analyse.render_markdown(report))
            summary['report'] = '%s/%s' % (OPS_SUBDIR, json_name)
            summary['validity'] = report.get('validity')
            for line in analyse.render_markdown(report).splitlines()[:40]:
                log('[C2-GATE] %s: %s' % (plan or TRACE, line))
        except Exception as error:
            summary['problem'] = 'analysis failed: %r' % (error,)
    if os.path.isdir(profile):
        summary['pruned'] = prune(profile)
        write_json(os.path.join(out, PRUNED), summary['pruned'])
    return summary


# ---- verdicts ----

def stream_texts(report):
    return [(stream or {}).get('text') for stream in (report or {}).get('streams') or []]


def text_problems(report, whose):
    """A stream without text cannot be compared: refused, never compared as equal Nones."""
    problems = []
    for index, stream in enumerate((report or {}).get('streams') or []):
        stream = stream or {}
        text = stream.get('text')
        if not isinstance(text, str) or not (text or stream.get('completion_tokens') or stream.get('tokens')):
            problems.append('%s stream %d carries no text' % (whose, index))
    return problems


def arrival_problems(report, env):
    """-e variables the harness process does not carry (read-the-launched-argv), within what it records; and the timed
    profile's own flags (REQUIRED_CONFIGURATION)."""
    configuration = (report or {}).get('qwen_configuration')
    problems = []
    if configuration is None:
        return ['the report records no configuration: neither the profile\'s flags nor the profiler variables can be '
                'shown to have arrived']
    for name, want in REQUIRED_CONFIGURATION:
        if str(configuration.get(name)) != want:
            problems.append('the launched configuration has %s=%r, not %s: this is not the timed four-card profile'
                            % (name, configuration.get(name), want))
    for name, value in env:
        if not CONFIGURATION_PREFIX.match(name):
            continue
        carried = configuration.get(name)
        if name in PATH_ENV and isinstance(carried, str) and carried:
            carried_path, value_path = posixpath.normpath(carried), posixpath.normpath(value)
            same = carried_path == value_path or carried_path.startswith(value_path + '/')
        else:
            same = carried == value
        if not same:
            problems.append('-e %s=%s never reached the container (it carries %r)' % (name, value, carried))
    return problems


def log_problems(log_text, kind, plan=None):
    """What the server log must and must not say: no verify audit line on either arm, and on the profiled arm the
    read-back lines (at least MIN_READBACKS); on the PREFILL profiled arm the flush hook's marker lines (at least
    MIN_FLUSH_MARKERS) instead - a note when they are missing, never a failure (the hook has not run at TP4 before)."""
    if log_text is None:
        return [], ['no server log to check the audit and read-back lines against']
    problems, notes = [], []
    for marker in AUDIT_MARKERS:
        if marker in log_text:
            problems.append('"%s" is in the server log: the verify trace carries an audit, so it is not the timed trace'
                            % marker)
    if kind == 'ops' and plan == PREFILL_TRACE:
        flushes = log_text.count(FLUSH_MARKER)
        if flushes < MIN_FLUSH_MARKERS:
            notes.append('%d prefill flush marker lines (%s), fewer than the %d a %d-token prompt logs: the flush hook did not '
                         'run (or the image lacks it), so the per-core buffers may have overflowed inside a chunk and the '
                         'report may be missing rows' % (flushes, FLUSH_MARKER, MIN_FLUSH_MARKERS, PREFILL_TOKENS))
    elif kind == 'ops':
        readbacks = log_text.count(READBACK_MARKER)
        if readbacks < MIN_READBACKS:
            notes.append('%d read-back lines (%s), fewer than the %d expected: the run may have been short, or the '
                         'cadence hook did not run' % (readbacks, READBACK_MARKER, MIN_READBACKS))
    return problems, notes


def verdict(plan, report, arm, twin_report, log_text=None, analysis=None, users=USERS):
    """The plan's result dict from the arm's harness report, its runner record and the twin's report."""
    kind = PLAN_KINDS[plan]
    ops = (arm or {}).get('ops') or {}
    lines = []
    if ops.get('disk_guard'):
        return dict(verdict='NOT_EXERCISED', reason='the disk guard stopped the arm: %s' % ops['disk_guard'],
                    lines=lines, ops=ops)
    if report is None:
        return dict(verdict='FAIL', reason='no harness report (exit %s)' % (arm or {}).get('exit'), lines=lines, ops=ops)
    problems = []
    if report.get('fatal'):
        problems.append('fatal: %s' % report['fatal'])
    problems += ['stream: %s' % problem for problem in report.get('real_text_stream_problems') or []]
    if len(report.get('streams') or []) != users:
        problems.append('%d streams, asked %d' % (len(report.get('streams') or []), users))
    problems += arrival_problems(report, (arm or {}).get('ops_env') or ())
    problems += text_problems(report, 'this arm\'s')
    found, notes = log_problems(log_text, kind, plan)
    problems += found
    lines += ['note: %s' % note for note in notes]
    digests = [hashlib.sha256(text.encode('utf-8')).hexdigest() if isinstance(text, str) else None
               for text in stream_texts(report)]
    if kind == 'twin':
        lines.append('texts %s' % ','.join(str(digest)[:12] for digest in digests))
        return dict(verdict='FAIL' if problems else 'PASS', reason='; '.join(problems) or None, lines=lines,
                    texts=digests, packed_phase=report.get('packed_phase'))
    shortfalls = []
    twin = PLAN_TWIN.get(plan, TWIN)
    if twin_report is None:
        shortfalls.append('texts not compared: no %s report' % twin)
    elif text_problems(twin_report, twin):
        shortfalls.append('texts not compared: %s' % '; '.join(text_problems(twin_report, twin)))
    elif not problems and stream_texts(twin_report) != stream_texts(report):
        problems.append('texts differ from %s: the texts diverged (profiling shifts scheduling; the exactness divergence at 32k and above is unresolved, so compare against a twin-vs-twin first)' % twin)
    if problems:
        return dict(verdict='FAIL', reason='; '.join(problems), lines=lines, ops=ops)
    if ops.get('problem'):
        lines.append('note: %s' % ops['problem'])
    validity = (ops.get('validity') or analysis or {})
    for problem in validity.get('problems') or []:
        lines.append('note: validity: %s' % problem)
    lines += ['note: %s' % problem for problem in report.get('c2_gate_problems') or []]
    if shortfalls:
        return dict(verdict='NOT_EXERCISED', reason='; '.join(shortfalls), lines=lines, ops=ops)
    return dict(verdict='PASS', reason=None, lines=lines, ops=ops)


def twin_report_of(results, plan):
    """The twin arm's harness report for the profiled plan (or None)."""
    if PLAN_KINDS[plan] == 'twin':
        return None
    path = os.path.join(results, PLAN_TWIN[plan], 'm3native-gate.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def handback_results(results, image, handback=docker_handback, log=print):
    """Every arm's ops-profile tree under a gate results directory handed back and pruned; {arm: pruned}."""
    done = {}
    if not os.path.isdir(results):
        return done
    for arm in sorted(os.listdir(results)):
        profile = os.path.join(results, arm, PROFILE_SUBDIR)
        if not os.path.isdir(profile):
            continue
        try:
            handback(image, profile)
        except Exception as error:
            log('[OPS] %s: hand-back failed: %s' % (arm, error))
        kept = os.path.join(results, OPS_SUBDIR, artifact_names(arm)[0])
        csv_path = find_csv(profile)
        if csv_path is not None and not os.path.exists(kept):
            # The gate never got to compress the report (cancelled, timed out): keep it before the prune drops it.
            os.makedirs(os.path.dirname(kept), exist_ok=True)
            try:
                gzip_file(csv_path, kept)
                log('[OPS] %s: kept the CPP report as %s/%s' % (arm, OPS_SUBDIR, CSV_GZ))
            except OSError as error:
                log('[OPS] %s: could not keep the CPP report: %s' % (arm, error))
        done[arm] = prune(profile)
        log('[OPS] %s: handed back; %d large files pruned' % (arm, len(done[arm])))
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command')
    handback_parser = commands.add_parser('handback')
    handback_parser.add_argument('--results', required=True)
    handback_parser.add_argument('--image', required=True)
    options = parser.parse_args(argv)
    if options.command == 'handback':
        handback_results(options.results, options.image)
        return 0
    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
