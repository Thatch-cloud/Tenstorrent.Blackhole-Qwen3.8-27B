"""The C2 gate's LLK profiling plans (c2_serving_job.LLK_GATE_PLANS), their arms, and what runs around each
arm: reading the image's kernel sources, building the instrumented overlay, handing the profiler's root-owned
output back, exporting and analysing it. Attribution only: no llk-* arm is a timing result, none is judged for
exactness against v235, and none blocks a placement (docs/llk-profiling-harness.md).

    python3 scripts/ci/llk_profile_plan.py handback --results <gate results> --image <image>
    python3 scripts/ci/llk_profile_plan.py preflight --image <image> --checkout . --out <dir>

handback is the workflow's always() step: it hands every arm's llk-profile tree back to the runner user and
prunes a raw log the gate did not get to - the root-owned-logs trap - even when the gate step was cancelled.
preflight is the build job's llkcheck action (no card): every instrumentation the arms will make, made now
from the image's own bytes, so an anchor that drifted fails the build job, not the M+A window.

PLANS, one arm each, run after every judged plan of the job (c2_serving_job refuses any other order):
  llk-decode-twin      4 users x 32768 real-text tokens, 48 out, served bytes, no profiler: the tokens and the
                       packed-round trace time the profiled decode arms are compared with.
  llk-decode-zones     the same, profiled: stage zones and wait sums (QWEN_LLK_ZONES=stages; tracy
                       --enable-sum-profiling), K5-A instrumented in-process (llk_zone_override), the file
                       kernels by read-only bind mounts over their paths (llk_kernels).
  llk-decode-counters  the same, profiled with hardware counters (FPU, PACK, UNPACK, L1_0, INSTRN: mask 47,
                       one L1 bank) and envelope zones only (QWEN_LLK_ZONES=tag) to name each counted kernel.
  llk-prefill-twin     1 user x 32768, max_tokens 1, served bytes.
  llk-prefill-zones    as llk-decode-zones for that prefill.
  llk-prefill-counters as llk-decode-counters for that prefill.
Every profiled arm: op-support count 20000 (200000 segfaulted the dispatch thread, v129/v131/v135; anything
else is refused), the profiler's output in <arm>/llk-profile bind-mounted at tt-metal's default artifacts
directory (outside the checkout and out of the 1 GB /opt/tt-metal/generated tmpfs) and named twice, as tracy's
-o and as TT_METAL_PROFILER_DIR (tracy hands its -o to the child that way, so the two must agree or the arrival
check fails), the JIT cache in the agent shape's per-container 8 GB tmpfs (TT_METAL_CACHE=
/root/.cache/tt-metal-cache: instrumented and profiler-define compiles never touch the image's shared kernel
cache, and every profiled arm compiles cold), and NOT tracy's --disable-device-data-dump-to-files: the
qualified m3native recipe passes it, and in v0.77.0 it suppresses profile_log_device.csv
(DeviceProfiler::writeDeviceResultsToFiles), the only file that carries the zones.

The profiler's buffer (marker_budget): each RISC's DRAM buffer holds OP_SUPPORT programs' guaranteed markers and
nothing is read back until a drain; once full, every later marker is lost. v138 (the same 4 x 32768 shape, no
zones, no prefill drain) filled every core before its round-4 read-back: the four prefills alone are ~19,600
programs per core. So every profiled arm drains the prefill (QWEN_PREFILL_PROFILE_FLUSH: a read-back every 16
layers, positive-controlled by its marker), a decode arm reads back again after packed round DUMP_ROUND, the
decode arms carry no matmul zones, and the budget of the window between two drains is checked before any
container starts. A decode arm is read over complete trace replay sessions only (llk_profile_report 'traced'),
so an overflow later in the run - the close's read-back, which is not guaranteed anyway - costs sessions, not
the arm.

The disk (disk_problem, DiskGuard): the raw device log of a decode arm is tens of GB (e). Before a profiled arm
the results disk must stay under DISK_MAX_USED with the arm's DISK_NEED written; during it a guard stops the
container if the profile tree passes its PROFILE_CAP or the disk passes DISK_STOP_USED. The raw log leaves the
results tree as soon as it is exported.

Verdicts: PASS, FAIL (a harness failure, a stream without text, or a profiled arm whose text differs from its
twin's - the copies are byte-reversible, so that would be a real finding), NOT_EXERCISED (no twin text to compare
with, a required kernel without zones on every compute thread of both chips or without MIN_SESSIONS complete
sessions, a wait sum missing on a zones arm, a counters arm without counter rows for a required kernel on both
chips, dropped markers in the analysed window, the prefill drain not engaged, the marker budget or the disk
refusing the arm, the disk guard stopping it, an earlier profiled arm that never became ready, no counter rows at
all = UNSUPPORTED, or a capability the image lacks), INFRA (the gate's own).

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import base64
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
import llk_kernels
import llk_profile_report
import llk_zones

PLANS = c2_serving_job.LLK_GATE_PLANS
PLAN_SHAPES = {
    'llk-decode-twin': ('decode', 'twin'), 'llk-decode-zones': ('decode', 'zones'),
    'llk-decode-counters': ('decode', 'counters'), 'llk-prefill-twin': ('prefill', 'twin'),
    'llk-prefill-zones': ('prefill', 'zones'), 'llk-prefill-counters': ('prefill', 'counters'),
}
TWIN = dict(decode='llk-decode-twin', prefill='llk-prefill-twin')
SHAPES = dict(decode=dict(users=4, prompt=32768, max_tokens=48), prefill=dict(users=1, prompt=32768, max_tokens=1))
# The registry's required kernels per phase: K5A and SDPA_DEC in decode, SDPA_PF in prefill.
REQUIRED = dict((phase, tuple(entry['key'] for entry in llk_kernels.KERNELS
                              if entry.get('required') and llk_kernels.in_phase(entry, phase)))
                for phase in llk_kernels.PHASES)
LEVEL = dict(twin=None, zones='stages', counters='tag')
# Which rows each phase is read over (llk_profile_report windows): decode's packed rounds are trace replays.
WINDOW = dict(decode='traced', prefill='untraced')
MIN_SESSIONS = llk_profile_report.MIN_SESSIONS
# Docker limits (seconds): the harness's readiness allowance (every profiled arm compiles cold into its tmpfs; the
# C2 cold compile is unmeasured, so the gate's own 1800 s holds) plus the stream timeout plus the close (the last
# read-back, the report). Each limit is at least readiness + stream, so a slow arm ends in the harness's own report.
READINESS_SECONDS = 1800
STREAM_SECONDS = dict(decode=1800, prefill=900)
CLOSE_SECONDS = 300
ARM_SECONDS = dict((plan, READINESS_SECONDS + STREAM_SECONDS[phase] + CLOSE_SECONDS)
                   for plan, (phase, _) in PLAN_SHAPES.items())
OP_SUPPORT = 20000
OP_SUPPORT_LIMIT = 20000
COUNTER_GROUPS = ('fpu', 'pack', 'unpack', 'l1_0', 'instrn')   # TT_METAL_PROFILE_PERF_COUNTERS 1+2+4+8+32 = 47
COUNTER_MASK = 47
PROFILE_DIR = '/opt/tt-metal/generated/profiler'
SCRATCH_CACHE = '/root/.cache/tt-metal-cache'
FLUSH_FLAG = 'QWEN_PREFILL_PROFILE_FLUSH'
DUMP_ROUND = 2
PROFILER_ENV = (('TT_METAL_DEVICE_PROFILER', '1'), ('TT_METAL_PROFILER_TRACE_TRACKING', '1'),
                ('TT_METAL_PROFILER_MID_RUN_DUMP', '1'), ('TTNN_OP_PROFILER', '1'),
                ('TT_METAL_PROFILER_DIR', PROFILE_DIR), ('QWEN_FAST_PROFILE_DUMP_ROUND', str(DUMP_ROUND)),
                (FLUSH_FLAG, '1'), ('QWEN_FAST_PROFILED_BLOCK_STREAM', '1'))
ARM_ENV_NAMES = frozenset([name for name, _ in PROFILER_ENV] + ['QWEN_LLK_ZONES', 'TT_METAL_CACHE'])
# What the harness's qwen_configuration records (QWEN*_ and TT_ names): the rest cannot be shown to arrive.
CONFIGURATION_PREFIX = re.compile(r'(?:QWEN[0-9]*_|TT_)')
# Path-valued names, compared normalised (tracy writes its -o back as TT_METAL_PROFILER_DIR).
PATH_ENV = frozenset(['TT_METAL_PROFILER_DIR', 'TT_METAL_CACHE'])

# ---- the profiler buffer model (marker_budget); (e) = estimate, from v138 (run of 2026-09-23, 4 x 32768) ----
# Per RISC the DRAM buffer is bufferEndIndex = 240000 uint32 at op-support 20000 (v138 server.log): 12 uint32, six
# two-word markers, per program - its id and the guaranteed markers. A zone costs two optional markers per
# execution and a used sum slot one per program, taken from the same buffer.
GUARANTEED_MARKERS = 6
CAPACITY_MARKERS = OP_SUPPORT * GUARANTEED_MARKERS
BUDGET_FRACTION = 0.8
# (e) programs one worker core runs: a packed round is 3605 verify + 2 x 528 draft ops per chip (one verify replay
# per round: 34 rounds, 20 verify sessions recorded), and a core in every op runs them all; a prefill layer is
# 30326 untraced ops per chip over 4 users x 16 chunks x 64 layers, 7.4 per layer.
ROUND_PROGRAMS = 4661
LAYER_PROGRAMS = 8
FLUSH_LAYERS = 16
# (e) invocations per packed round (decode) or per prefill layer on one core: 48 GDN and 16 attention layers.
INVOCATIONS = dict(decode=dict(K5A=48, K5A_RD=48, K5A_WR=48, SDPA_DEC=16, SDPA_DEC_RD=16, ATTN_PREP=16,
                               CONV_GATES=48),
                   prefill=dict(SDPA_PF=0.25, MM=6, AGMM=1))

# ---- the disk (disk_problem, DiskGuard) ----
GB = 2 ** 30
# (e) a profiled arm's raw device log: a full per-core buffer is ~150M rows over both chips; v138's untraced
# (prefill) work alone is ~141M rows, a packed round ~17M, at ~150 bytes a row: decode ~40 GB, prefill ~6 GB.
DISK_NEED = dict(decode=48 * GB, prefill=10 * GB)
PROFILE_CAP = dict(decode=64 * GB, prefill=16 * GB)
DISK_MAX_USED = 0.80        # with the arm's DISK_NEED written (the rig's no-build-above-80% rule)
DISK_STOP_USED = 0.85       # the guard stops the container past this (the rig has hit DiskPressure before)
GUARD_SECONDS = 20
# What no served profile may carry: every profiler and LLK variable.
PROFILE_REFUSED = re.compile(r'(?:QWEN_LLK_|TT_METAL_DEVICE_PROFILER|TT_METAL_PROFILE|TT_METAL_PROFILER_|TTNN_OP_PROFILER)')
REFUSED_TRACY = ('--disable-device-data-dump-to-files', '--device-trace-profiler', '--profile-dispatch-cores')
PROBES = ('tools/tracy/__main__.py', 'tools/tracy/common.py', 'tt_metal/tools/profiler/kernel_profiler.hpp',
          'tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h', 'tt_metal/tools/profiler/perf_counters.hpp')
TRACY_OPTIONS = ('--op-support-count', '--dump-device-data-mid-run', '--disable-device-data-push-to-tracy',
                 '--check-exit-code', '--enable-sum-profiling', '--profiler-capture-perf-counters')
KEEP_BYTES = 8 * 2 ** 20          # a profile file larger than this leaves the results tree after export
EXPORT_LIMIT_BYTES = 512 * 2 ** 20
RECORD_LINE = re.compile(r'\[LLK\] record (\{.*\})\s*$')
MANIFEST = 'llk-manifest.json'
REPORT = 'llk-report.json'
EXPORT = 'llk-export.csv.gz'
PRUNED = 'llk-pruned.json'
PROFILE_SUBDIR = 'llk-profile'
OVERLAY_SUBDIR = 'llk-overlay'


class LlkPlanError(ValueError):
    """An LLK plan the gate refuses before any container starts."""


# ---- plans and arms ----

def tracy_args(kind, sums=True):
    """The tracy wrapper's options for a profiled arm (None for a twin)."""
    if kind == 'twin':
        return None
    # -o is the bind mount itself, as v138's recipe ran it (-o at its mount point, the logs under <-o>/.logs), and
    # the same path as the arm's TT_METAL_PROFILER_DIR: tracy passes its -o to the child through that variable.
    args = ['-p', '--check-exit-code', '--disable-device-data-push-to-tracy', '--dump-device-data-mid-run',
            '--op-support-count', str(OP_SUPPORT), '-o', PROFILE_DIR]
    if kind == 'zones' and sums:
        args.append('--enable-sum-profiling')
    if kind == 'counters':
        args += ['--profiler-capture-perf-counters', ','.join(COUNTER_GROUPS)]
    check_tracy(args)
    return args


def check_tracy(args):
    """Refuse a tracy argument list the pass must never run."""
    for refused in REFUSED_TRACY:
        if refused in args:
            raise LlkPlanError('tracy %s is refused on an LLK arm (%s)' % (
                refused, 'it suppresses profile_log_device.csv' if 'dump-to-files' in refused else 'not qualified'))
    if '--op-support-count' not in args:
        raise LlkPlanError('an LLK arm states its op-support count (%d)' % OP_SUPPORT)
    count = args[args.index('--op-support-count') + 1]
    if not count.isdigit() or not 0 < int(count) <= OP_SUPPORT_LIMIT:
        raise LlkPlanError('op-support count %s refused: at most %d (200000 segfaulted the dispatch thread)'
                           % (count, OP_SUPPORT_LIMIT))
    output = args[args.index('-o') + 1] if '-o' in args else None
    if output is None or posixpath.normpath(output) != PROFILE_DIR:
        raise LlkPlanError('tracy output must be %s (the arm\'s bind-mounted llk-profile, and its '
                           'TT_METAL_PROFILER_DIR), got %r' % (PROFILE_DIR, output))
    return args


def arm_env(kind, level):
    if kind == 'twin':
        return ()
    return PROFILER_ENV + (('QWEN_LLK_ZONES', level), ('TT_METAL_CACHE', SCRATCH_CACHE))


def check_profiles(profiles):
    """No profile of the image may carry a profiler or LLK variable (a served container is never profiled)."""
    for name, profile in sorted(profiles['profiles'].items()):
        carried = sorted(key for key in (profile.get('env') or {}) if PROFILE_REFUSED.match(key))
        if carried:
            raise LlkPlanError('profile %s carries %s: a served profile never runs the profiler or LLK zones'
                               % (name, ', '.join(carried)))


def plan_arms(plan, profile, profiles, gate):
    """[gate.Arm] of one llk-* plan on `profile`, refused (gate.PlanError) where the profile cannot serve it."""
    if plan not in PLAN_SHAPES:
        raise gate.PlanError('unknown LLK plan %r' % plan)
    try:
        check_profiles(profiles)
    except LlkPlanError as error:
        raise gate.PlanError(str(error))
    if profile not in profiles['profiles']:
        raise gate.PlanError('%s: profile %s is not in the image' % (plan, profile))
    phase, kind = PLAN_SHAPES[plan]
    shape = SHAPES[phase]
    context, ceiling, room = gate.profile_limits(profiles, profile)
    gate.check_lengths(profile, [shape['prompt']], room, '%s prompt' % plan)
    gate.check_budget(profile, shape['max_tokens'], ceiling, '%s --max-tokens' % plan)
    args = gate.common_args(profile, context, STREAM_SECONDS[phase], readiness=READINESS_SECONDS) + [
        '--users', str(shape['users']), '--prompt-tokens', str(shape['prompt']),
        '--max-tokens', str(shape['max_tokens']), '--stagger', str(gate.STAGGER)]
    level = LEVEL[kind]
    llk = dict(plan=plan, phase=phase, kind=kind, level=level, env=arm_env(kind, level), tracy=tracy_args(kind))
    return [gate.Arm(plan, args, ARM_SECONDS[plan], rerun=False, judged=False, role='llk', llk=llk)]


def check_env(env):
    for name, _ in env:
        if name not in ARM_ENV_NAMES:
            raise LlkPlanError('%s is not an environment an LLK arm may add (%s)' % (name, ', '.join(sorted(ARM_ENV_NAMES))))
    return env


def planned(arm_dir, llk):
    """What prepare_arm will add to the arm's docker run, without reading the image (--dry-run)."""
    mounts = []
    if llk['kind'] != 'twin':
        mounts = ['--mount', 'type=bind,src=%s,dst=%s' % (os.path.join(arm_dir, PROFILE_SUBDIR), PROFILE_DIR),
                  '--mount', 'type=bind,src=<%s/...>,dst=<each instrumented kernel path>,readonly' % os.path.join(
                      arm_dir, OVERLAY_SUBDIR)]
    return dict(llk, mounts=mounts, env=check_env(llk['env']))


def docker_additions(prepared):
    """(env pairs, mounts, entry args) a prepared arm adds to gate_run; entry args replace '-B <harness>'."""
    env = check_env(prepared.get('env') or ())
    entry = None
    if prepared.get('tracy'):
        entry = ['-B', '-m', 'tracy'] + list(check_tracy(list(prepared['tracy'])))
    return env, list(prepared.get('mounts') or []), entry


# ---- the profiler buffer budget ----

def static_markers(entry, level, sums):
    """Optional markers per RISC per program of a generated entry (its records exist only in the worker)."""
    stages = level == 'stages'
    regions = [(stage, multiplicity) for stage, _, _, multiplicity in entry.get('stages') or ()] if stages else []
    return llk_zones.marker_count(regions, envelope=True, sums=bool(stages and sums))


def mover(entry):
    return entry.get('part') in ('reader', 'writer')


def marker_budget(phase, level, sums, records=()):
    """The markers one RISC writes between two drains, against its DRAM buffer (CAPACITY_MARKERS). The window: a
    decode arm's DUMP_ROUND packed rounds after the last prefill drain, plus that drain's leftover (FLUSH_LAYERS
    prefill layers); a prefill arm's FLUSH_LAYERS layers. Each kernel counts its optional markers per program (its
    record's, or static_markers for the generated K5-A) times its INVOCATIONS in the window; compute and
    data-movement kernels fill different RISCs, so the fuller class counts. An estimate (e): ok is the lint's
    answer, and the traced window is what tolerates it being wrong."""
    if phase == 'decode':
        programs = DUMP_ROUND * ROUND_PROGRAMS + FLUSH_LAYERS * LAYER_PROGRAMS
        scale = DUMP_ROUND
    else:
        programs = FLUSH_LAYERS * LAYER_PROGRAMS
        scale = FLUSH_LAYERS
    markers = {}
    for entry in llk_kernels.KERNELS:
        if entry['route'] == 'generated' and llk_kernels.in_phase(entry, phase):
            markers[entry['key']] = (entry, static_markers(entry, level, sums))
    for record in records or ():
        entry = llk_kernels.BY_KEY.get(record.get('kernel'))
        if entry is None or 'refused' in record or not record.get('markers') or entry.get('header'):
            continue   # a header's sums share the including kernel's slots
        markers[record.get('key') or entry['key']] = (entry, record['markers']['per_risc_per_program'])
    kernels, classes = {}, dict(compute=0, mover=0)
    for key, (entry, per_program) in sorted(markers.items()):
        invocations = scale * INVOCATIONS[phase].get(entry['key'], 0)
        kernels[key] = dict(markers_per_program=per_program, invocations=invocations)
        classes['mover' if mover(entry) else 'compute'] += int(round(per_program * invocations))
    estimate = programs * GUARANTEED_MARKERS + max(classes.values())
    share = round(estimate / float(CAPACITY_MARKERS), 3)
    return dict(phase=phase, level=level, sums=bool(sums), window_programs=programs, capacity=CAPACITY_MARKERS,
                optional=classes, estimate=estimate, fraction=share, limit=BUDGET_FRACTION,
                ok=share <= BUDGET_FRACTION, kernels=kernels)


def budget_problem(budget):
    if budget['ok']:
        return None
    return ('marker budget: ~%d markers per RISC between two drains, %.0f%% of the %d the profiler buffer holds '
            '(limit %.0f%%): the arm would overflow and lose its window' % (
                budget['estimate'], 100 * budget['fraction'], budget['capacity'], 100 * budget['limit']))


# ---- the disk ----

def disk_problem(path, need, usage=shutil.disk_usage):
    """Why the results disk cannot take a profiled arm's raw device log (need bytes), or None."""
    total, used, free = usage(path)
    after = (used + need) / float(total)
    if need > free or after > DISK_MAX_USED:
        return ('disk: %.0f GB free of %.0f GB (%.0f%% used); the arm may write %.0f GB, leaving the disk %.0f%% '
                'used (at most %.0f%%): free space first' % (free / float(GB), total / float(GB),
                                                             100.0 * used / total, need / float(GB), 100 * after,
                                                             100 * DISK_MAX_USED))
    return None


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
    """While a profiled arm runs: stop its container (stop()) once its profile tree passes `cap` bytes or the disk
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
            return 'the disk reached %.1f%% used (the guard stops at %.0f%%)' % (100.0 * used / total,
                                                                                100 * DISK_STOP_USED)
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
        self._thread = threading.Thread(target=self._run, name='llk-disk-guard')
        self._thread.daemon = True
        self._thread.start()
        return self

    def finish(self):
        self._done.set()
        if self._thread is not None:
            self._thread.join(self.interval + 5)
        return self.tripped


# ---- reading the image ----

READER = r'''
import base64, glob, importlib.util, json, os, sys
root = "/opt/tt-metal"
request = json.loads(sys.argv[1])
out = dict(files={}, globs={}, probes={}, tracy_module=None)
def read(rel, limit=4 << 20):
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read(limit)).decode()
for rel in request["paths"]:
    out["files"][rel] = read(rel)
for pattern in request["patterns"]:
    hits = sorted(os.path.relpath(hit, root) for hit in glob.glob(os.path.join(root, pattern), recursive=True)
                  if os.path.isfile(hit))[:request["limit"]]
    out["globs"][pattern] = hits
    for rel in hits:
        out["files"][rel] = read(rel)
for rel in request["probes"]:
    out["probes"][rel] = read(rel)
try:
    spec = importlib.util.find_spec("tracy")
    out["tracy_module"] = spec.origin if spec else None
except Exception as error:
    out["tracy_module"] = None
print(json.dumps(out))
'''


def read_image(image, paths, patterns, probes=PROBES, timeout=600):
    """{files: path -> text or None, globs: pattern -> [paths], probes: path -> text or None, tracy_module}
    read from the image in a throwaway container: no devices, no network, the contract's boot off."""
    request = json.dumps(dict(paths=list(paths), patterns=list(patterns), probes=list(probes),
                              limit=llk_kernels.DISCOVER_LIMIT))
    output = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-e', 'QWEN_C2_SERVING=0', '--entrypoint',
                             'python3', image, '-B', '-c', READER, request],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if output.returncode:
        raise RuntimeError('reading the image\'s kernel sources exited %d: %s' % (
            output.returncode, output.stderr.decode('utf-8', 'replace')[-400:]))
    return decode_reading(json.loads(output.stdout.decode('utf-8').strip().splitlines()[-1]))


def decode_reading(raw):
    def text(value):
        return None if value is None else llk_zones.decode(base64.b64decode(value))
    return dict(files=dict((path, text(value)) for path, value in (raw.get('files') or {}).items()),
                globs=dict(raw.get('globs') or {}),
                probes=dict((path, text(value)) for path, value in (raw.get('probes') or {}).items()),
                tracy_module=raw.get('tracy_module'))


def empty_reader(image, paths, patterns, probes=PROBES):
    """The reader of a run with a fake executor (tests): nothing in the image, nothing to mount."""
    return dict(files={}, globs={}, probes={}, tracy_module=None)


def capabilities(reading):
    """What the image's profiler offers, from its own sources (task L0-02's probe): unknown is None with a
    reason, never False."""
    probes = reading.get('probes') or {}
    unknown = []

    def has(path, *needles):
        text = probes.get(path)
        if text is None:
            unknown.append('%s not read' % path)
            return None
        return all(needle in text for needle in needles)

    def number(path, pattern):
        text = probes.get(path)
        match = re.search(pattern, text or '')
        return int(match.group(1)) if match else None

    tracy = probes.get('tools/tracy/__main__.py')
    options = dict((option, None if tracy is None else ('"%s"' % option) in tracy) for option in TRACY_OPTIONS)
    if tracy is None:
        unknown.append('tools/tracy/__main__.py not read')
    result = dict(
        tracy_module=reading.get('tracy_module'),
        tracy_options=options,
        dump_to_files_flag=None if tracy is None else '"--disable-device-data-dump-to-files"' in tracy,
        profiler_dir_env=has('tools/tracy/common.py', 'TT_METAL_PROFILER_DIR'),
        zone_macro=has('tt_metal/tools/profiler/kernel_profiler.hpp', 'define DeviceZoneScopedN'),
        sum_zones=has('tt_metal/tools/profiler/kernel_profiler.hpp', 'define DeviceZoneScopedSumN1',
                      'define DeviceZoneScopedSumN2'),
        perf_counters=has('tt_metal/tools/profiler/perf_counters.hpp', 'PERF_COUNTER_PROFILER_ID'),
        optional_markers=number('tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h',
                                r'PROFILER_L1_OPTIONAL_MARKER_COUNT\s*=\s*(\d+)'),
        sum_count=number('tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h', r'SUM_COUNT\s*=\s*(\d+)'),
        unknown=unknown)
    return result


def unsupported(kind, caps):
    """Why the image cannot run an arm of `kind`, or None."""
    if kind == 'twin':
        return None
    needed = ['--op-support-count', '--dump-device-data-mid-run', '--disable-device-data-push-to-tracy',
              '--check-exit-code']
    if kind == 'counters':
        needed.append('--profiler-capture-perf-counters')
    missing = [option for option in needed if caps['tracy_options'].get(option) is not True]
    if missing:
        return 'the image\'s tracy lacks %s' % ', '.join(missing)
    if not caps.get('tracy_module'):
        return 'python3 -m tracy is not importable in the image'
    if caps.get('zone_macro') is not True:
        return 'the image\'s kernel_profiler.hpp defines no DeviceZoneScopedN'
    if kind == 'counters' and caps.get('perf_counters') is not True:
        return 'UNSUPPORTED: the image has no perf_counters.hpp counter readout'
    return None


# ---- one arm: prepare, finish ----

def prepare_arm(image, arm_dir, llk, reader, log=print):
    """The arm's additions (env, tracy, mounts) and its manifest, or {'skip': reason}. Reads the image's kernel
    sources (reader: read_image, or a fake), writes the instrumented copies under <arm>/llk-overlay and the
    manifest, and makes <arm>/llk-profile writable for the container's root."""
    prepared = dict(llk)
    if llk['kind'] == 'twin':
        prepared.update(mounts=[], manifest=None)
        return prepared
    paths, patterns = llk_kernels.file_requests(llk['phase'])
    reading = reader(image, paths, patterns, PROBES)
    caps = capabilities(reading)
    reason = unsupported(llk['kind'], caps)
    manifest = dict(schema='qwen-llk-manifest/1', plan=llk['plan'], phase=llk['phase'], kind=llk['kind'],
                    level=llk['level'], capabilities=caps, files=[], generated=[], mounts=[])
    if reason:
        manifest['skipped'] = reason
        write_json(os.path.join(arm_dir, MANIFEST), manifest)
        return dict(prepared, skip=reason, skip_scope='image', manifest=manifest)
    sums = bool(caps.get('sum_zones')) and llk['kind'] == 'zones'
    if llk['kind'] == 'zones' and not sums:
        prepared['tracy'] = tracy_args('zones', sums=False)
    budget = caps.get('optional_markers') or llk_zones.MARKER_BUDGET
    overlay = os.path.join(arm_dir, OVERLAY_SUBDIR)
    profile = os.path.join(arm_dir, PROFILE_SUBDIR)
    for directory in (overlay, profile):
        os.makedirs(directory, exist_ok=True)
    os.chmod(profile, 0o777)
    mounts = ['--mount', 'type=bind,src=%s,dst=%s' % (profile, PROFILE_DIR)]
    for path, text, record in llk_kernels.plan_files(llk['phase'], reading['files'], reading['globs'], llk['level'],
                                                     sums_supported=sums, budget=budget):
        manifest['files'].append(record)
        if text is None:
            log('[C2-GATE] %s: %s not instrumented: %s' % (llk['plan'], record.get('key') or record.get('kernel'),
                                                          record.get('refused')))
            continue
        destination = llk_kernels.check_destination(path)
        host = os.path.join(overlay, *path.split('/'))
        os.makedirs(os.path.dirname(host), exist_ok=True)
        with open(host, 'wb') as handle:
            handle.write(llk_zones.encode(text))
        mounts += ['--mount', 'type=bind,src=%s,dst=%s,readonly' % (host, destination)]
        manifest['mounts'].append(dict(path=path, destination=destination, sha256=record['instrumented_sha256']))
    manifest['budget'] = marker_budget(llk['phase'], llk['level'], sums, manifest['files'])
    refused = budget_problem(manifest['budget'])
    if refused:
        manifest['skipped'] = refused
        write_json(os.path.join(arm_dir, MANIFEST), manifest)
        return dict(prepared, skip=refused, skip_scope='arm', manifest=manifest)
    write_json(os.path.join(arm_dir, MANIFEST), manifest)
    prepared.update(mounts=mounts, manifest=manifest, sums=sums)
    return prepared


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=1, sort_keys=True)


def docker_handback(image, directory, timeout=300):
    """Hand a tree the container's root wrote back to this user (the EACCES trap): a throwaway container with
    the tree mounted, no network, no devices."""
    return subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--mount', 'type=bind,src=%s,dst=/p'
                           % directory, '--entrypoint', 'sh', image, '-c', 'chown -R %d:%d /p; chmod -R a+rwX /p'
                           % (os.getuid(), os.getgid())],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout).returncode


def generated_records(console):
    """The override's own records ('[LLK] record {...}' lines): its positive control."""
    records = []
    for line in (console or '').splitlines():
        match = RECORD_LINE.search(line)
        if match:
            try:
                records.append(json.loads(match.group(1)))
            except ValueError:
                continue
    return records


def device_logs(profile):
    found = []
    for directory, _, names in os.walk(profile):
        for name in names:
            if name == 'profile_log_device.csv':
                found.append(os.path.join(directory, name))
    return sorted(found)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prune(profile, keep_bytes=None, known=None):
    """Every file above keep_bytes (KEEP_BYTES) out of the results tree (the raw log is gigabytes), each recorded
    with its size and sha256 first; `known` ({relative path: (bytes, sha256)}) is what the export already hashed,
    so a tens-of-GB log is not read a second time."""
    keep_bytes = KEEP_BYTES if keep_bytes is None else keep_bytes
    known = known or {}
    pruned = []
    for directory, _, names in os.walk(profile):
        for name in names:
            path = os.path.join(directory, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > keep_bytes:
                relative = os.path.relpath(path, profile)
                hashed = known.get(relative)
                digest = hashed[1] if hashed and hashed[0] == size else file_sha256(path)
                pruned.append(dict(path=relative, bytes=size, sha256=digest))
                os.remove(path)
    return pruned


def handback_arm(image, arm_dir, prepared, handback):
    """The arm's profile tree handed back to this user (called in a finally: however the arm ended). Never
    raises: a failed hand-back is reported, and the workflow's always() step tries again."""
    if prepared['kind'] == 'twin' or prepared.get('skip'):
        return None
    profile = os.path.join(arm_dir, PROFILE_SUBDIR)
    if not os.path.isdir(profile):
        return None
    try:
        return handback(image, profile)
    except Exception as error:
        return 'failed: %s' % error


def finish_arm(image, arm_dir, prepared, console='', twin=None, profiled=None, log=print):
    """After the arm (and handback_arm): export and analyse the device log, prune the raw files. Returns the
    arm's LLK summary (also <arm>/llk-report.json when a log was found)."""
    summary = dict(kind=prepared['kind'], level=prepared.get('level'))
    if prepared['kind'] == 'twin' or prepared.get('skip'):
        return summary
    profile = os.path.join(arm_dir, PROFILE_SUBDIR)
    manifest = dict(prepared.get('manifest') or {})
    manifest['generated'] = generated_records(console)
    write_json(os.path.join(arm_dir, MANIFEST), manifest)
    logs = device_logs(profile)
    summary['device_logs'] = [os.path.relpath(path, arm_dir) for path in logs]
    records = [record for record in manifest.get('files', []) + manifest['generated'] if record.get('zones')]
    known = {}
    if len(logs) > 1:
        summary['note'] = 'more than one profile_log_device.csv; the first was analysed'
    if not logs:
        summary['problem'] = 'no profile_log_device.csv under %s' % PROFILE_SUBDIR
    elif not records:
        summary['problem'] = 'no instrumented kernel in the manifest'
    else:
        try:
            exported = llk_profile_report.export_filtered(logs[0], os.path.join(arm_dir, EXPORT))
            summary['export'] = exported
            known[os.path.relpath(logs[0], profile)] = (exported['source']['bytes'], exported['source']['sha256'])
            if os.path.getsize(os.path.join(arm_dir, EXPORT)) > EXPORT_LIMIT_BYTES:
                raise llk_profile_report.ReportError('the filtered export is past %d bytes' % EXPORT_LIMIT_BYTES)
            preamble, rows = llk_profile_report.read_rows(os.path.join(arm_dir, EXPORT))
            report = llk_profile_report.analyse(rows, records, console, twin, profiled, preamble,
                                                window=WINDOW[prepared['phase']])
            report['source'] = exported
            write_json(os.path.join(arm_dir, REPORT), report)
            summary['report'] = REPORT
            summary['complete'] = report['complete']
            summary['drops'] = report['drops']
            summary['selection'] = report['selection']
            summary['counters_seen'] = report['counters_seen']
            summary['coverage'] = llk_profile_report.coverage(report, REQUIRED[prepared['phase']],
                                                              kind=prepared['kind'], min_sessions=MIN_SESSIONS)
            summary['kernels'] = [kernel['kernel'] for kernel in report['kernels']]
            summary['ranking'] = report['ranking']
            for line in llk_profile_report.render(report).splitlines():
                log('[C2-GATE] %s: %s' % (prepared['plan'], line))
        except (llk_profile_report.ReportError, llk_zones.ZoneError, OSError, ValueError) as error:
            summary['problem'] = 'analysis refused: %s' % error
    summary['pruned'] = prune(profile, known=known)
    write_json(os.path.join(arm_dir, PRUNED), summary['pruned'])
    return summary


# ---- verdicts ----

def stream_texts(report):
    """Each stream's generated text (the harness report's streams[i]['text']), None where a stream has none."""
    return [(stream or {}).get('text') for stream in (report or {}).get('streams') or []]


def text_digest(text):
    return hashlib.sha256(llk_zones.encode(text)).hexdigest() if isinstance(text, str) else None


def text_problems(report, whose):
    """A stream without text cannot be compared: refused, never compared as equal Nones. No text is a missing or
    non-string 'text', or an empty one from a stream that reports no completion token; an empty text after one
    or more tokens (a prefill arm's single token can detokenise to nothing) is a value and is compared."""
    problems = []
    for index, stream in enumerate((report or {}).get('streams') or []):
        stream = stream or {}
        text = stream.get('text')
        if not isinstance(text, str) or not (text or stream.get('completion_tokens') or stream.get('tokens')):
            problems.append('%s stream %d carries no text' % (whose, index))
    return problems


def arrival_problems(report, env):
    """-e variables the harness process does not carry (read-the-launched-argv), within what it records."""
    configuration = (report or {}).get('qwen_configuration')
    problems = []
    for name, value in env:
        if not CONFIGURATION_PREFIX.match(name):
            continue
        if configuration is None:
            problems.append('the report records no configuration: -e %s=%s is unshown' % (name, value))
            continue
        carried = configuration.get(name)
        if name in PATH_ENV and isinstance(carried, str) and carried:
            # The same directory, or one inside it (tracy may hand the child a folder under its -o): either way
            # the output stays in the arm's mount.
            carried_path, value_path = posixpath.normpath(carried), posixpath.normpath(value)
            same = carried_path == value_path or carried_path.startswith(value_path + '/')
        else:
            same = carried == value
        if not same:
            problems.append('-e %s=%s never reached the container (it carries %r)' % (name, value, carried))
    return problems


def verdict(plan, report, arm, twin_report):
    """The plan's result dict from the arm's harness report, its runner record and the twin's report."""
    phase, kind = PLAN_SHAPES[plan]
    llk = (arm or {}).get('llk') or {}
    lines = []
    if llk.get('disk_guard'):
        return dict(verdict='NOT_EXERCISED', reason='the disk guard stopped the arm: %s' % llk['disk_guard'],
                    lines=lines, llk=llk)
    if report is None:
        return dict(verdict='FAIL', reason='no harness report (exit %s)' % (arm or {}).get('exit'), lines=lines, llk=llk)
    problems = []
    if report.get('fatal'):
        problems.append('fatal: %s' % report['fatal'])
    problems += ['stream: %s' % problem for problem in report.get('real_text_stream_problems') or []]
    streams = report.get('streams') or []
    if len(streams) != SHAPES[phase]['users']:
        problems.append('%d streams, asked %d' % (len(streams), SHAPES[phase]['users']))
    problems += arrival_problems(report, (arm or {}).get('llk_env') or ())
    problems += text_problems(report, 'this arm\'s')
    digests = [text_digest(text) for text in stream_texts(report)]
    if kind == 'twin':
        lines.append('tokens %s' % ','.join(str(digest)[:12] for digest in digests))
        return dict(verdict='FAIL' if problems else 'PASS', reason='; '.join(problems) or None, lines=lines,
                    tokens=digests, packed_phase=report.get('packed_phase'))
    shortfalls = []
    if twin_report is None:
        shortfalls.append('tokens not compared: no %s report' % TWIN[phase])
    elif text_problems(twin_report, TWIN[phase]):
        shortfalls.append('tokens not compared: %s' % '; '.join(text_problems(twin_report, TWIN[phase])))
    elif not problems and stream_texts(twin_report) != stream_texts(report):
        problems.append('tokens differ from %s: %s against %s (the instrumented copies are byte-reversible, so this '
                        'is a finding)' % (TWIN[phase], [str(digest)[:12] for digest in digests],
                                           [str(text_digest(text))[:12] for text in stream_texts(twin_report)]))
    if problems:
        return dict(verdict='FAIL', reason='; '.join(problems), lines=lines, llk=llk)
    if llk.get('problem'):
        shortfalls.append(llk['problem'])
    shortfalls += llk.get('coverage') or []
    if llk.get('complete') is False:
        shortfalls.append('dropped markers in the analysed window: %s' % llk.get('drops'))
    if kind == 'counters' and not llk.get('counters_seen'):
        shortfalls.append('UNSUPPORTED: no counter rows (event %d) in the device log' % llk_profile_report.COUNTER_ID)
    missing = [str(marker) for marker in (report.get('flag_markers') or {}).get('missing') or []]
    shortfalls += ['the prefill profiler drain was not seen: %s' % marker for marker in missing
                   if marker.startswith(FLUSH_FLAG)]
    for rank in llk.get('ranking') or []:
        lines.append('#%(rank)d %(kernel)s %(bound)s: %(reason)s' % rank)
    # The judged arms' own S2 checks are notes here: a profiling arm is attribution, not an S2 exit gate.
    lines += ['note: %s' % problem for problem in report.get('c2_gate_problems') or []]
    lines += ['note: flag marker missing: %s' % marker for marker in missing if not marker.startswith(FLUSH_FLAG)]
    if shortfalls:
        return dict(verdict='NOT_EXERCISED', reason='; '.join(shortfalls), lines=lines, llk=llk)
    return dict(verdict='PASS', reason=None, lines=lines, llk=llk)


READINESS_TIMEOUT = 'readiness exceeded'   # lever_n_m3native_gate.start_server's TimeoutError


def never_ready(report):
    """Why a profiled arm's server never came up in time (its cold compile outlasted the readiness allowance), or
    None. Only the harness's readiness timeout counts: a server that crashed before readiness failed fast, so the
    next profiled arm (another kernel set, another level) still runs."""
    if report is None or report.get('ready') is not False:
        return None
    fatal = str(report.get('fatal') or '')
    if READINESS_TIMEOUT not in fatal:
        return None
    return 'a profiled arm\'s server never became ready within %d s (%s): every profiled arm compiles cold' % (
        READINESS_SECONDS, fatal)


def twin_report_of(results, plan):
    """The twin arm's harness report for a profiled plan (or None)."""
    phase, kind = PLAN_SHAPES[plan]
    if kind == 'twin':
        return None
    path = os.path.join(results, TWIN[phase], 'm3native-gate.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


# ---- the no-card preflight (the build job's llkcheck action) ----

PREFLIGHT = r'''
import json, sys
sys.path[:0] = ["/llk", "/experiment-scripts/ci"]
out = {}
try:
    import gdn_seq_block, llk_kernels
    build = gdn_seq_block.load_kernels("/opt/tt-metal", int(sys.argv[1]))
    out["qualified"] = bool(build.qualified)
    for level in ("stages", "tag"):
        texts, records = llk_kernels.instrument_generated(dict(build), level, sums_supported=True)
        out[level] = records
except Exception as error:
    out["error"] = "%s: %s" % (type(error).__name__, error)
print(json.dumps(out))
'''
PREFLIGHT_MODULES = ('llk_zones.py', 'llk_kernels.py')


def generated_preflight(image, checkout, level=0, timeout=600):
    """The K5-A build the image serves, instrumented at both levels by this checkout's transforms, in a
    throwaway container (no devices, no network): {qualified, stages: [record], tag: [record]} or {error}."""
    mounts = []
    for name in PREFLIGHT_MODULES:
        mounts += ['--mount', 'type=bind,src=%s,dst=/llk/%s,readonly' % (os.path.join(checkout, 'scripts', 'ci', name), name)]
    output = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-e', 'QWEN_C2_SERVING=0'] + mounts +
                            ['--entrypoint', 'python3', image, '-B', '-c', PREFLIGHT, str(level)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if output.returncode:
        return dict(error='exit %d: %s' % (output.returncode, output.stderr.decode('utf-8', 'replace')[-400:]))
    return json.loads(output.stdout.decode('utf-8').strip().splitlines()[-1])


def preflight(image, checkout, out_dir, reader=read_image, generated=generated_preflight):
    """Everything the llk-* arms will do to the image's sources, done now with no card: every file kernel of
    both phases instrumented at both levels from the image's bytes, the image's K5-A build instrumented in the
    image, and the profiler capabilities read. Returns {phases, generated, problems}; a problem is a required
    kernel that cannot be instrumented, a capability the zones arms need, or a window past the marker budget."""
    result = dict(schema='qwen-llk-preflight/1', phases={}, generated=None, problems=[])
    for phase in llk_kernels.PHASES:
        paths, patterns = llk_kernels.file_requests(phase)
        reading = reader(image, paths, patterns, PROBES)
        caps = capabilities(reading)
        entry = dict(capabilities=caps, unsupported=dict((kind, unsupported(kind, caps)) for kind in ('zones', 'counters')),
                     levels={}, budgets={})
        for level in llk_zones.LEVELS:
            sums = bool(caps.get('sum_zones')) and level == 'stages'
            planned = llk_kernels.plan_files(phase, reading['files'], reading['globs'], level,
                                             sums_supported=bool(caps.get('sum_zones')),
                                             budget=caps.get('optional_markers') or llk_zones.MARKER_BUDGET)
            entry['levels'][level] = [record for _, _, record in planned]
            entry['budgets'][level] = marker_budget(phase, level, sums, entry['levels'][level])
            refused = budget_problem(entry['budgets'][level])
            if refused:
                result['problems'].append('%s at level %s: %s' % (phase, level, refused))
            for key in REQUIRED[phase]:
                if llk_kernels.BY_KEY[key]['route'] != 'file':
                    continue
                mine = [record for record in entry['levels'][level] if record.get('kernel') == key]
                if not mine or any('refused' in record for record in mine):
                    result['problems'].append('%s %s at level %s: %s' % (phase, key, level, '; '.join(
                        record.get('refused', '') for record in mine) or 'no record'))
        if entry['unsupported']['zones']:
            result['problems'].append('%s zones arm: %s' % (phase, entry['unsupported']['zones']))
        result['phases'][phase] = entry
    gen = generated(image, checkout)
    result['generated'] = gen
    if gen.get('error'):
        result['problems'].append('K5A: the image\'s build could not be instrumented: %s' % gen['error'])
    else:
        for level in llk_zones.LEVELS:
            for record in gen.get(level) or []:
                if 'refused' in record and llk_kernels.BY_KEY[record['kernel']].get('required'):
                    result['problems'].append('%s at level %s: %s' % (record['kernel'], level, record['refused']))
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, 'llk-preflight.json'), result)
    return result


# ---- the workflow's always() step ----

def handback_results(results, image, handback=docker_handback, log=print):
    """Every arm's llk-profile tree under a gate results directory handed back and pruned; {arm: pruned}."""
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
            log('[LLK] %s: hand-back failed: %s' % (arm, error))
        done[arm] = prune(profile)
        log('[LLK] %s: handed back; %d large files pruned' % (arm, len(done[arm])))
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command')
    handback_parser = commands.add_parser('handback')
    handback_parser.add_argument('--results', required=True)
    handback_parser.add_argument('--image', required=True)
    preflight_parser = commands.add_parser('preflight')
    preflight_parser.add_argument('--image', required=True)
    preflight_parser.add_argument('--checkout', required=True)
    preflight_parser.add_argument('--out', required=True)
    options = parser.parse_args(argv)
    if options.command == 'handback':
        handback_results(options.results, options.image)
        return 0
    if options.command == 'preflight':
        result = preflight(options.image, options.checkout, options.out)
        for problem in result['problems']:
            print('[LLK] preflight problem: %s' % problem)
        print('[LLK] preflight %s: %d problems' % ('FAILED' if result['problems'] else 'passed', len(result['problems'])))
        return 1 if result['problems'] else 0
    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
