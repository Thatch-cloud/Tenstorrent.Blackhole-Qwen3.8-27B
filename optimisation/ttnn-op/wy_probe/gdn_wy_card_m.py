#!/usr/bin/env python3
"""Window-WY probe on ONE card, ONE layer: two 16-row windows of the WY / UT form against K5-A run twice. NON-EXACT, RESEARCH ONLY.

The window form (scripts/ci/gdn_wy_model.py, docs/gdn-wy-probe.md) is a different arithmetic from the served recurrence, so this harness never
compares it with K5-A for equality and its verdict is never a licence to serve. It decides one thing: whether the kernel is worth the
programme's next step (the owner's call), by three kill rules measured on a single p150a, a 1x1 mesh, four users x 12 value heads at the
four-card geometry, no model, no collective, run with QWEN_FAST_TP=4 and QWEN_FAST_VERIFY_T1=1 (the K5-A control is the coalesced build the model runs):

  timing     per layer, in one trace of 48 distinct-input launches per arm, 25 serpentine rounds: A (one K5-A launch, 16 rows), A2 (two K5-A
             launches, the sequential baseline for a 32-row block), W1 (one window), W2 (two windows in one launch). KILL when W2 > 90 us
             a layer (the sequential chain is ~193 us a window, ~387 us for two).
  causality  rows 0..n-1 of the output and the committed state after n rows, as RAW BYTES, with every later row of the block set to the real
             draft data, to zeros, and to other random data (and, informationally, NaN: a window form multiplies masked zeros by later rows, so
             poison leaks); n from 1 to 31 across both windows. Raw bytes because ttnn's to_torch hides -0 and denormals. KILL on any mismatch.
             Controls: K5-A under the same variants (it is causal: any difference is a harness fault, NO-DECISION), and a changed EARLIER row
             must differ for both arms (a blind compare is NO-DECISION).
  packing    the four-user launch against four one-user launches, byte for byte, outputs and committed states. KILL on any mismatch.
  sram       the kernel's circular-buffer bytes a core against the CPU plan (gdn_wy_model.sram_plan) and the core's L1 size; KILL over L1.
  accuracy   a CORRECTNESS GATE: the W arm's outputs and committed states for BOTH windows, in every regime and seed, against the fp64 sequential
             reference of the same host inputs (gdn_wy_model), with K5-A (run twice, chained through its own committed state) as the yardstick:
             the W error against fp64 must stay within 1.5x K5-A's error against fp64 for the output rows and each committed state. KILL above
             it, or on non-finite output. A reference that fails (non-finite, or K5-A itself far from fp64) is NO-DECISION. The differing
             fraction against K5-A, max abs and the ulp histogram of window one are reported beside it, informational.
  inputs     after every launch the inputs are read back and compared with what was uploaded; a changed input page is a KILL.

THE KERNEL DOES NOT EXIST YET. The W arms need scripts/ci/gdn_wy_block.py (the planned interface below). Without it this harness runs the
baseline half only (selftest, the K5-A controls, timing of A and A2), prints the verdict NO-DECISION (exit 4), and says why. The baseline
half and every W path are CPU-tested on a fake runtime (test_gdn_wy_card_m.py, with a fake kernel that computes with gdn_wy_model); neither
has run on a card.

Planned kernel interface (gdn_wy_block):
  load_kernels(root, unqualified=True)                -> a build object with .cb_bytes(windows) (bytes of CBs a core), .dram_bytes(windows)
                                                         (the reader and writer byte totals a core moves, as the generator emits them), .sha256()
  execute(device, groups, operations, output_memory, kernels, windows, commit_rows=None)
        groups: one tuple per user (qkv, beta, gate, initial, z, norm_w) with 16 * windows rows (qkv [1, R, 2560], beta and gate [1, R, 12],
        z [1, R, 1536]); returns [(output, states)] per user, output [1, R, 1536], states [windows, 12, 128, 128]: states[w] is the bf16
        state after commit_rows[w] rows of window w (commit_rows defaults to all rows; 0 leaves states[w] unwritten and the harness does
        not read it). Window w + 1 starts from window w's committed state, in the launch, in L1.

Exit codes: 0 CONTINUE (every kill rule held at full scope: the owner decides the next step); 10 KILL (not 1: exit 1 is what the launcher's
refusals and an import-time crash return); 2 usage error; 3 the per-call watchdog fired (the partial report is written first; reset the
target card only); 4 NO-DECISION. The last stdout line is one JSON object.

  QWEN_FAST_TP=4 QWEN_FAST_VERIFY_T1=1 python3 gdn_wy_card_m.py --out report.json
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import gdn_v5_card_m as base
import gdn_seq_block as seq
import gdn_seq_block_device_test as dev
import tp_shapes
import verify_trace_t1

try:
    import gdn_wy_model as model
except ImportError:  # pragma: no cover - the model is mounted next to the other scripts
    model = None

VERDICT = 'GDN_WY'
KIND = 'gdn-wy-probe'
SECTIONS = ('selftest', 'sram', 'causality', 'packing', 'accuracy', 'timing')
REGIMES = ('R1', 'R2')
ROWS = 16
WINDOWS = 2
HEADS_PER_CHIP = 12
PLAN_USERS = 4
PLAN_SEEDS = (17, 23)
PLAN_TIMING_LAUNCHES = 48
PLAN_TIMING_ROUNDS = 25
PLAN_COMMITS = (1, 4, 8, 15, 16, 17, 24, 31)   # committed rows n of a 32-row block (n = 32 has no later row)
KILL_TWO_WINDOWS_US = 90.0                     # two windows in one launch, per layer
ACCURACY_FACTOR = 1.5                          # W error vs fp64 may be at most this times K5-A's error vs fp64 (E1 measured ratios <= 1.01)
A_SANITY_REL = 0.25                            # K5-A's own error vs fp64 above this: the reference or the layout is wrong, not the kernel
EXIT_KILL = 10
SEQUENTIAL_WINDOW_US = 193.4                   # K5-A, measured (anchor only; the run measures its own)
L1_BYTES = 1572864                             # Blackhole Tensix L1
TIMING_ARMS = ('A', 'A2', 'W1', 'W2')
WY_ARMS = ('W1', 'W2')
MODULES = ('gdn_seq_block.py', 'gdn_seq_block_compute.cpp', 'gdn_seq_block_reader.cpp', 'gdn_seq_block_writer.cpp',
           'gdn_seq_block_device_test.py', 'gdn_multitoken.py', 'tp_shapes.py', 'verify_trace_t1.py', 'gdn_v5_card_m.py',
           'gdn_wy_model.py', 'gdn_wy_card_m.py', 'gdn_wy_block.py')
ENV_READ = ('QWEN_FAST_TP', 'QWEN_FAST_VERIFY_T1', 'TT_METAL_HOME', 'TT_METAL_CACHE', 'TT_METAL_WATCHER')


# ---- pure helpers (no ttnn: held by test_gdn_wy_card_m on CPU) ----

def commit_plan(n):
    """commit_rows per window for a block of two windows committing n rows: (min(16, n), max(0, n - 16))."""
    return (min(ROWS, n), max(0, n - ROWS))


def window_inputs(torch, regime, seed, users, found, windows=WINDOWS):
    """(norm_w, [dict(qkv, beta, gate, initial, z) per user], poison): `windows` 16-row windows of base.host_inputs (seed + 977 w), rows
    concatenated; the initial state is the first window's."""
    parts = [base.host_inputs(torch, regime, seed + 977 * w, users, found) for w in range(windows)]
    groups = []
    for user in range(users):
        merged = {name: torch.cat([part[1][user][name] for part in parts], dim=1) for name in ('qkv', 'beta', 'gate', 'z')}
        merged['initial'] = parts[0][1][user]['initial']
        groups.append(merged)
    return parts[0][0], groups, parts[0][2]


def vary_rows(torch, users_host, n, variant, other_users=None):
    """The inputs with every row from n on replaced (variant 'real' keeps them): 'zeros', 'other' (another draw's rows),
    'nan' (informational). 'earlier' replaces row 0 with another draw's row (the negative control: rows 0..n-1 MUST change)."""
    names = ('qkv', 'beta', 'gate', 'z')
    changed = []
    for user, values in enumerate(users_host):
        new = {name: values[name].clone() for name in names}
        new['initial'] = values['initial']
        for name in names:
            if variant == 'zeros':
                new[name][:, n:] = 0
            elif variant == 'other':
                new[name][:, n:] = other_users[user][name][:, n:]
            elif variant == 'nan':
                new[name][:, n:] = float('nan')
            elif variant == 'earlier':
                new[name][:, 0:1] = other_users[user][name][:, 0:1]
            elif variant != 'real':
                raise ValueError(variant)
        changed.append(new)
    return changed


def prefix_bytes_equal(torch, expected, actual, n, kind, commit_rows):
    """Raw-byte equality of rows 0..n-1 of the output image and of the committed states, for one user.
    expected / actual: (output_image [1, R_padded, 1536], states_image). kind 'A': states hold the 16 snapshots and the commit is snapshot
    n - 1; kind 'W': states[w] is window w's committed state for every w with commit_rows[w] > 0."""
    out_e, out_a = expected[0][0, :n], actual[0][0, :n]
    parts = dict(output=bool(torch.equal(dev.bits16(torch, out_e), dev.bits16(torch, out_a))))
    if kind == 'A':
        parts['states'] = bool(torch.equal(dev.bits16(torch, expected[1][n - 1]), dev.bits16(torch, actual[1][n - 1])))
    else:
        for window, rows in enumerate(commit_rows):
            if rows > 0:
                parts['states%d' % window] = bool(torch.equal(dev.bits16(torch, expected[1][window]),
                                                              dev.bits16(torch, actual[1][window])))
    return dict(exact=all(parts.values()), parts=parts)


def timing_verdict(a_us, a2_us, w1_us, w2_us):
    """The timing rule from per-unit medians (a unit is one launch set: A one window, A2 two sequential launches, W1 one window,
    W2 two windows in one launch). 'kill' when W2 > 90 us a layer, 'continue' otherwise, 'not-run' when W2 was not measured."""
    out = dict(kill_threshold_us=KILL_TWO_WINDOWS_US, a_us=a_us, a2_us=a2_us, w1_us=w1_us, w2_us=w2_us)
    if w2_us is None:
        return dict(out, label='not-run', speedup_vs_a2=None, per_layer_saving_us=None, verify_ms_saving_est=None)
    saving = None if a2_us is None else a2_us - w2_us
    return dict(out, label='kill' if w2_us > KILL_TWO_WINDOWS_US else 'continue',
                speedup_vs_a2=None if not a2_us else a2_us / w2_us, per_layer_saving_us=saving,
                verify_ms_saving_est=None if saving is None else 48 * saving / 1000)


def sram_verdict(cb_bytes_by_windows, l1_bytes=L1_BYTES):
    """KILL when the two-window build's circular buffers exceed L1; the one-window figure and the plan are reported beside it."""
    two = cb_bytes_by_windows.get(2)
    if two is None:
        return dict(label='not-run', l1_bytes=l1_bytes)
    return dict(label='kill' if two > l1_bytes else 'ok', l1_bytes=l1_bytes, cb_bytes=dict(cb_bytes_by_windows),
                l1_fraction_two_windows=two / l1_bytes)


def scope_missing(arguments):
    """What a run lacks of the full probe: empty means scope=full."""
    missing = ['section %s' % name for name in SECTIONS if name not in arguments.sections]
    missing += ['regime %s' % name for name in REGIMES if name not in arguments.regimes]
    missing += ['seed %d' % seed for seed in PLAN_SEEDS if seed not in arguments.seeds]
    missing += ['commit n=%d' % n for n in PLAN_COMMITS if n not in arguments.commits]
    if 'timing' in arguments.sections:
        missing += ['timing arm %s' % arm for arm in TIMING_ARMS if arm not in arguments.timing_arms]
    if arguments.users != PLAN_USERS:
        missing.append('users %d of %d' % (arguments.users, PLAN_USERS))
    if 'timing' in arguments.sections and (arguments.timing_launches < PLAN_TIMING_LAUNCHES
                                           or arguments.timing_rounds < PLAN_TIMING_ROUNDS):
        missing.append('timing %d launches x %d rounds (plan %d x %d)' % (
            arguments.timing_launches, arguments.timing_rounds, PLAN_TIMING_LAUNCHES, PLAN_TIMING_ROUNDS))
    return missing


def decide(report):
    """(verdict, problems): KILL on any hard miss of the W arm (outranks everything), NO-DECISION when the kernel is absent, a section
    raised, a harness control failed or the scope is reduced, CONTINUE only when every kill rule held at full scope."""
    kill, undecided = [], []
    sections = report.get('sections', {})
    for name, section in sections.items():
        if section.get('error'):
            undecided.append('section %s raised: %s' % (name, section['error']))
    if not report.get('a_qualified', False):
        undecided.append('control A is not the qualified K5-A build')
    if report.get('kernel_import_error'):
        undecided.append('the window kernel module (gdn_wy_block) is present but raised on import: %s' % report['kernel_import_error'])
    elif not report.get('kernel_present', False):
        undecided.append('the window kernel (gdn_wy_block) is not built: baseline only, no W arm ran')
    causality = sections.get('causality', {})
    if causality and not causality.get('error'):
        if causality.get('control_a_causal') is False:
            undecided.append('control: K5-A is not causal under the harness (a harness fault, not a finding)')
        if causality.get('control_blind') is True:
            undecided.append('control: a changed earlier row did not change the compared bytes (a blind compare)')
        for failure in causality.get('w_failures', [])[:3]:
            kill.append('causality: %s' % failure)
    packing = sections.get('packing', {})
    if packing and not packing.get('error') and packing.get('w_exact') is False:
        kill.append('packing: the four-user launch differs from the one-user launches (%s)' % packing.get('first_failure'))
    timing = sections.get('timing', {})
    timing_scope_full = not any(item.startswith('timing ') for item in report.get('missing_scope', []))
    if timing and not timing.get('error') and timing.get('label') == 'kill':
        if timing_scope_full:
            kill.append('timing: two windows take %.1f us a layer (> %.0f)' % (timing['w2_us'], KILL_TWO_WINDOWS_US))
        else:
            undecided.append('timing: two windows take %.1f us a layer (> %.0f) at a reduced timing scope, where the per-launch overhead is '
                             'not amortised: not a KILL until %d launches x %d rounds have run' % (
                                 timing['w2_us'], KILL_TWO_WINDOWS_US, PLAN_TIMING_LAUNCHES, PLAN_TIMING_ROUNDS))
    if timing and not timing.get('error') and report.get('kernel_present') and timing.get('label') == 'not-run':
        undecided.append('timing: the W2 arm was not measured, so the main kill rule was not tested')
    sram = sections.get('sram', {})
    if sram and not sram.get('error') and sram.get('label') == 'kill':
        kill.append('sram: two windows need %d B of circular buffers, L1 is %d' % (sram['cb_bytes'][2], sram['l1_bytes']))
    if report.get('unwritten'):
        kill.append('pages never written: %s' % report['unwritten'][:3])
    if report.get('inputs_moved'):
        kill.append('an input moved: %s' % report['inputs_moved'][:3])
    accuracy = sections.get('accuracy', {})
    if accuracy and not accuracy.get('error') and accuracy.get('nonfinite'):
        kill.append('accuracy: non-finite values in the window arm\'s outputs')
    if accuracy and not accuracy.get('error'):
        for failure in accuracy.get('correctness_failures', [])[:3]:
            kill.append('correctness: %s' % failure)
        if accuracy.get('reference_failed'):
            undecided.append('control: the fp64 reference or K5-A\'s own error against it is not usable (%s)' % accuracy['reference_failed'])
    if report.get('missing_scope'):
        undecided.append('reduced scope: %s' % '; '.join(report['missing_scope'][:6]))
    if kill:
        return 'KILL', kill + undecided
    if undecided:
        return 'NO-DECISION', undecided
    return 'CONTINUE', []


def exit_code(verdict):
    return {'CONTINUE': 0, 'KILL': EXIT_KILL, 'NO-DECISION': 4}[verdict]


def verdict_line(report):
    timing = report.get('sections', {}).get('timing', {})
    causality = report.get('sections', {}).get('causality', {})
    packing = report.get('sections', {}).get('packing', {})
    accuracy = report.get('sections', {}).get('accuracy', {})
    return '%s verdict=%s scope=%s kernel=%s causality=%s packing=%s correctness=%s timing=%s a_us=%s a2_us=%s w2_us=%s NON-EXACT research probe, never a serving licence' % (
        VERDICT, report['verdict'], 'reduced' if report.get('missing_scope') else 'full',
        'present' if report.get('kernel_present') else 'absent',
        'not-run' if not causality or causality.get('error') else ('clean' if not causality.get('w_failures') else 'MISMATCH')
        if report.get('kernel_present') else 'control-only',
        'not-run' if not packing or packing.get('error') or packing.get('w_exact') is None else
        ('exact' if packing['w_exact'] else 'DIFFERS'),
        'not-run' if not accuracy or accuracy.get('error') or accuracy.get('correct') is None else ('within-bound' if accuracy['correct'] else 'WRONG'),
        timing.get('label', 'not-run'), _us(timing.get('a_us')), _us(timing.get('a2_us')), _us(timing.get('w2_us')))


def _us(value):
    return 'n/a' if value is None else '%.1f' % value


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sections', default=','.join(SECTIONS))
    parser.add_argument('--regimes', default=','.join(REGIMES))
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in PLAN_SEEDS))
    parser.add_argument('--commits', default=','.join(str(n) for n in PLAN_COMMITS), help='committed rows n of a 32-row block')
    parser.add_argument('--users', type=int, default=PLAN_USERS, choices=(1, 2, 3, 4))
    parser.add_argument('--timing-launches', type=int, default=PLAN_TIMING_LAUNCHES)
    parser.add_argument('--timing-rounds', type=int, default=PLAN_TIMING_ROUNDS)
    parser.add_argument('--timing-arms', default=','.join(TIMING_ARMS))
    parser.add_argument('--call-timeout', type=float, default=180.0, help='per-call watchdog, seconds (0 = off)')
    parser.add_argument('--output-memory', choices=('l1', 'dram'), default='l1')
    parser.add_argument('--trace-region', type=int, default=134217728)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal')))
    arguments = parser.parse_args(argv)
    for name in ('sections', 'regimes', 'timing_arms'):
        setattr(arguments, name, [item for item in getattr(arguments, name).split(',') if item])
    arguments.seeds = [int(item) for item in arguments.seeds.split(',') if item]
    arguments.commits = [int(item) for item in arguments.commits.split(',') if item]
    if any(name not in SECTIONS for name in arguments.sections):
        parser.error('--sections names only %s' % (SECTIONS,))
    if any(name not in REGIMES for name in arguments.regimes):
        parser.error('--regimes names only %s' % (REGIMES,))
    if any(name not in TIMING_ARMS for name in arguments.timing_arms) or 'A' not in arguments.timing_arms:
        parser.error('--timing-arms names only %s and includes A' % (TIMING_ARMS,))
    if any(not 1 <= n < ROWS * WINDOWS for n in arguments.commits):
        parser.error('--commits are 1..31 (n = 32 has no later row)')
    if arguments.timing_launches < 1 or arguments.timing_rounds < 1:
        parser.error('--timing-launches and --timing-rounds must be positive')
    return arguments


def load_window_kernel():
    """The planned kernel module, or None while it is not built. Only the absence of gdn_wy_block ITSELF reads as 'not built': any other
    import error (a module the kernel needs, a syntax error, a failing import-time check) propagates so it is recorded as an error."""
    try:
        import gdn_wy_block
    except ModuleNotFoundError as error:
        if error.name == 'gdn_wy_block':
            return None
        raise
    return gdn_wy_block


def json_safe(value):
    """NaN and Infinity are not JSON: map every non-finite float to null so a strict consumer (jq, JSON.parse) reads the report."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dump_report(report):
    return json.dumps(json_safe(report), indent=2, default=str, allow_nan=False)


def to_raw(torch, qkv, beta, gate, z, rows):
    """One user's host inputs as the model's raw dict (q, k, v, z, beta, g), batch 1, at the four-card shard (4 key heads, 12 value heads)."""
    key = model.NK * model.DK
    flat = qkv.float()
    return dict(q=flat[:, :rows, 0:key].reshape(1, rows, model.NK, model.DK), k=flat[:, :rows, key:2 * key].reshape(1, rows, model.NK, model.DK),
                v=flat[:, :rows, 2 * key:].reshape(1, rows, model.NV, model.DV), z=z.float()[:, :rows].reshape(1, rows, model.NV, model.DV),
                beta=beta.float()[:, :rows], g=gate.float()[:, :rows])


def host_reference(torch, user, norm_w, rows):
    """The fp64 sequential reference of one user's `rows` rows from the bf16 host inputs, in the harness' layouts: (output [1, rows, 1536]
    gated and unrounded, snapshots [rows, 12, 128, 128]) and the bf16 window model's committed states and output for the chained windows."""
    raw = to_raw(torch, user['qkv'], user['beta'], user['gate'], user['z'], rows)
    weight = norm_w[0, 0].float()
    initial = user['initial'].float()
    p64 = model.prep(raw, weight, model.F64)
    o64, snaps, _ = model.seq_round(initial.double().clone(), p64, rows, model.MODELS['fp64'])
    layout = lambda y: y.transpose(1, 2).reshape(1, rows, model.NV * model.DV)
    reference = dict(output=layout(model.gated(o64, p64, rows, round_bf16=False)), states=torch.stack([snap[0] for snap in snaps]))
    bf = model.MODELS['bf16']
    pbf = model.prep(raw, weight, bf.dtype)
    out_bf, state_bf, _ = model.wy_verify(initial.to(bf.dtype), pbf, rows, bf)
    out_16, state_16, _ = model.wy_verify(initial.to(bf.dtype), pbf, ROWS, bf)
    reference['model_output'] = layout(model.gated(out_bf, pbf, rows))
    reference['model_states'] = [state_16[0], state_bf[0]]
    return reference


def accuracy_case(torch, user, norm_w, a_output, a_states, w_output, w_states, rows=ROWS * WINDOWS, factor=ACCURACY_FACTOR):
    """The correctness gate for one user. a_output / w_output: [1, rows, 1536] images (K5-A's two windows joined, W's two windows); a_states
    / w_states: the committed states after 16 and after `rows` rows ([12, 128, 128] each). Error is max abs over the fp64 reference's largest
    magnitude (gdn_wy_model.cmp rel_to_max). ok: W's error within `factor` x K5-A's on the output rows and on each committed state."""
    truth = host_reference(torch, user, norm_w, rows)
    items = (('output', a_output, w_output, truth['output'], truth['model_output']),
             ('state_after_%d' % ROWS, a_states[0], w_states[0], truth['states'][ROWS - 1], truth['model_states'][0]),
             ('state_after_%d' % rows, a_states[1], w_states[1], truth['states'][rows - 1], truth['model_states'][1]))
    metrics, ok, reference_failed = {}, True, None
    for name, a_value, w_value, expected, modelled in items:
        a_error = model.cmp(a_value.float(), expected)
        w_error = model.cmp(w_value.float(), expected)
        a_rel, w_rel = a_error['rel_to_max'], w_error['rel_to_max']
        if not bool(torch.isfinite(expected).all()):
            reference_failed = 'non-finite fp64 reference (%s)' % name
        elif not (math.isfinite(a_rel) and a_rel <= A_SANITY_REL):
            reference_failed = 'K5-A %s error against fp64 is %s (bound %.2f)' % (name, a_rel, A_SANITY_REL)
        within = bool(w_error['finite']) and w_rel <= factor * a_rel
        ok = ok and within
        metrics[name] = dict(a_vs_fp64=a_rel, w_vs_fp64=w_rel, limit=factor * a_rel, ok=within, w_finite=bool(w_error['finite']),
                             w_vs_bf16_model=model.cmp(w_value.float(), modelled.float())['rel_to_max'],
                             bf16_model_vs_fp64=model.cmp(modelled.float(), expected)['rel_to_max'])
    return dict(ok=ok, reference_failed=reference_failed, metrics=metrics)


# ---- the device part ----

def main(argv=None):
    arguments = parse(argv)
    here = Path(__file__).resolve().parent
    ci = Path(dev.__file__).resolve().parent
    kernel_import_error = None
    try:
        wy = load_window_kernel()
    except Exception as error:  # noqa: BLE001 - a present but broken kernel is a recorded error, never 'not built'
        wy, kernel_import_error = None, '%s: %s' % (type(error).__name__, error)
    report = dict(scope='window WY (gdn_wy_block) against K5-A on one card, four-card geometry; NON-EXACT research probe; no model, no '
                        'collective', argv=sys.argv[1:], users=arguments.users, rows=ROWS, windows=WINDOWS, sections={},
                  stages=[], unwritten=[], inputs_moved=[], a_qualified=False, kernel_present=wy is not None,
                  kernel_import_error=kernel_import_error, launches=dict(run_arm=0, input_checks=0), verdict='NO-DECISION', missing_scope=scope_missing(arguments),
                  env_read={name: os.environ.get(name) for name in ENV_READ},
                  module_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name in MODULES for path in (here / name, ci / name) if path.exists()})
    summary = dict(kind=KIND, verdict='NO-DECISION')
    mesh = [None]

    def write(extra=None):
        report.update(extra or {})
        arguments.out.write_text(dump_report(report))

    watchdog = base.Watchdog(lambda label: write(dict(error='watchdog: %r' % (label,), watchdog=label)))
    watchdog.seconds = arguments.call_timeout

    def stage(label, **details):
        report['stages'].append(dict(stage=label, **details))
        write()
        print(json.dumps(report['stages'][-1], default=str), flush=True)

    code = 4
    try:
        stage('environment', **report['env_read'])
        import torch
        import ttnn
        report['ttnn_path'] = ttnn.__file__
        if os.environ.get('QWEN_FAST_TP') != '4' or tp_shapes.chip_count() != 4:
            raise AssertionError('QWEN_FAST_TP=4 is required (the launched environment says %r)' % os.environ.get('QWEN_FAST_TP'))
        if not verify_trace_t1.cut('coalesce'):
            raise AssertionError('QWEN_FAST_VERIFY_T1=1 with the coalesce cut is required: K5-A must run as the coalesced build the model runs')
        found = tp_shapes.geometry(4)
        if (found.gdn_nv, found.gdn_qkv, found.gdn_z, found.gdn_value) != (HEADS_PER_CHIP, 2560, 1536, 1536):
            raise AssertionError('four-card geometry drifted: %r' % (found,))

        stage('load-kernels', root=str(arguments.root), kernel_present=wy is not None)
        builds = dict(A=seq.load_kernels(arguments.root, 0, unqualified=True))
        report['a_qualified'] = bool(builds['A'].qualified)
        report['generated_sha256'] = dict(A=seq.sha256(builds['A']))
        if wy is not None:
            builds['W'] = wy.load_kernels(arguments.root, unqualified=True)
            report['generated_sha256']['W'] = builds['W'].sha256()
        report['cb_bytes_per_core'] = dict(A=seq.cb_bytes())

        stage('mesh-open')
        mesh[0] = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=arguments.trace_region)
        device = mesh[0]
        grid = device.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        output_memory = ttnn.L1_MEMORY_CONFIG if arguments.output_memory == 'l1' else ttnn.DRAM_MEMORY_CONFIG

        # ---- the plumbing of gdn_v5_card_m.main (raw page copies, exact uploads, sentinel fill), same as there ----
        def raw_copy(source, destination, pages):
            workers = min(dev.RAW_WORKERS, pages)
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(workers - 1, 0))])
            scratch = ttnn.CBDescriptor(total_size=2048, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048,
                                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
            program = ttnn.MeshProgramDescriptor()
            for chip, (left, right) in enumerate(zip(ttnn.get_device_tensors(source), ttnn.get_device_tensors(destination), strict=True)):
                args = []
                for value in (left, right):
                    args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
                runtime = ttnn.RuntimeArgs()
                for worker in range(workers):
                    runtime[worker][0] = [left.buffer_address(), right.buffer_address(), pages, worker, workers]
                kernel = ttnn.KernelDescriptor(kernel_source=dev.RAW_COPY,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=args, config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))
                kernel.runtime_args = runtime
                coordinate = ttnn.MeshCoordinate(0, chip)
                program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[scratch])
            with watchdog.guard('raw_copy'):
                ttnn.generic_op([source, destination], program)

        def words_tensor(words):
            return ttnn.from_torch(words.reshape(1, 1, -1, 512), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(device))

        def sync(label):
            with watchdog.guard(label):
                ttnn.synchronize_device(device)

        sentinel = words_tensor(torch.full((dev.MAX_PAGES, 512), dev.SENTINEL_WORD, dtype=torch.int32))

        def fill_sentinel(value):
            pages = dev.page_count(value.shape)
            if pages > dev.MAX_PAGES:
                raise ValueError('A launch allocated %d pages; the sentinel holds %d' % (pages, dev.MAX_PAGES))
            raw_copy(sentinel, value, pages)

        def upload_exact(logical, image=None):
            image = dev.pad_image(torch, logical) if image is None else image
            target = ttnn.empty(tuple(logical.shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            words = dev.tile_image(torch, image)
            source = words_tensor(words)
            try:
                raw_copy(source, target, words.shape[0])
                sync('upload_exact')
            finally:
                ttnn.deallocate(source)
            return target

        def upload(value):
            return ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(device))

        def raw_host(value):
            shape = dev.padded_shape(value.shape)
            pages = dev.page_count(value.shape)
            sink = words_tensor(torch.zeros(pages, 512, dtype=torch.int32))
            try:
                raw_copy(value, sink, pages)
                sync('raw_host')
                images = [dev.untile_image(torch, dev.words_of(torch, ttnn.to_torch(shard)).reshape(pages, 512), shape)
                          for shard in ttnn.get_device_tensors(sink)]
                raw_copy(sentinel, sink, pages)
                return images
            finally:
                ttnn.deallocate(sink)

        def release(produced):
            for output, states in produced or ():
                ttnn.deallocate(output)
                ttnn.deallocate(states)

        sentinel_operations = dev.SentinelOperations(ttnn, fill_sentinel)

        def launch(arm, groups, operations=None, commit_rows=None):
            operations = ttnn if operations is None else operations
            with watchdog.guard('launch %s' % arm):
                if arm == 'A':
                    return seq.execute(device, groups, operations, output_memory=output_memory, kernels=builds['A'])
                windows = 1 if arm == 'W1' else 2
                return wy.execute(device, groups, operations, output_memory=output_memory, kernels=builds['W'], windows=windows,
                                  commit_rows=commit_rows)

        def coalesced_once(arm, where):
            counts = verify_trace_t1.take()
            where.setdefault('verify_t1_counts', {})[arm] = counts
            if arm == 'A' and counts != {'coalesced': 1}:
                raise AssertionError('Arm A did not run as one coalesced launch: verify_trace_t1 counts %s' % counts)

        class Case:
            """One host input set on the device: groups, the inputs' byte checks, what to free."""

            def __init__(self, host):
                norm_w, users_host, poison = host
                self.owned, self.checks, self.groups = [], [], []
                self.norm_w = upload_exact(norm_w)
                self.owned.append(self.norm_w)
                self.checks.append((dict(user=None, tensor='norm_w'), self.norm_w, dev.pad_image(torch, norm_w)))
                for user, values in enumerate(users_host):
                    tensors = []
                    for name in ('qkv', 'beta', 'gate', 'initial', 'z'):
                        fill = 0.0 if poison[user] is None or name == 'initial' else poison[user]
                        image = dev.pad_image(torch, values[name], fill)
                        tensors.append(upload_exact(values[name], image))
                        self.owned.append(tensors[-1])
                        self.checks.append((dict(user=user, tensor=name), tensors[-1], image))
                    self.groups.append(tuple(tensors) + (self.norm_w,))
                moved = self.moved()
                if moved:
                    raise AssertionError('Inputs did not land byte for byte: %s' % moved)

            def moved(self):
                return [dict(label) for label, tensor, image in self.checks
                        if not torch.equal(dev.bits16(torch, raw_host(tensor)[0]), dev.bits16(torch, image))]

            def free(self):
                for value in self.owned:
                    ttnn.deallocate(value)

        def split_users(host, first, count):
            return host[0], host[1][first:first + count], host[2][first:first + count]

        def run_arm(arm, case, label, where, commit_rows=None):
            """(per-user [(output_image, states_image)]) read as raw bytes; unwritten pages recorded (an uncommitted state window excepted)."""
            report['launches']['run_arm'] += 1
            produced = launch(arm, case.groups, sentinel_operations, commit_rows)
            try:
                sync('synchronize %s' % arm)
                coalesced_once(arm, where)
                images = []
                for user, (output, states) in enumerate(produced):
                    out_image, states_image = raw_host(output)[0], raw_host(states)[0]
                    skip = uncommitted(arm, commit_rows)
                    counts = dict(output=dev.unwritten_pages(torch, out_image))
                    if arm in WY_ARMS:
                        for window in range(states_image.shape[0]):
                            if window not in skip:
                                counts['states%d' % window] = dev.unwritten_pages(torch, states_image[window])
                    else:
                        counts['states'] = dev.unwritten_pages(torch, states_image)
                    for name, count in counts.items():
                        if count:
                            report['unwritten'].append(dict(arm=arm, case=label, user=user, tensor=name, pages=count))
                    images.append((out_image, states_image))
                # a kernel that writes into its own inputs must not pass: read them back after the launch (V5's gate does the same)
                report['launches']['input_checks'] += 1
                for item in case.moved():
                    report['inputs_moved'].append(dict(item, arm=arm, case=label))
                return images
            finally:
                release(produced)

        def uncommitted(arm, commit_rows):
            return tuple(w for w, rows in enumerate(commit_rows or ()) if rows == 0) if arm in WY_ARMS else ()

        def section(name, body):
            if name not in arguments.sections:
                return
            entry = report['sections'].setdefault(name, dict(error=None))
            stage('section', name=name)
            try:
                body(entry)
            except BaseException as error:  # noqa: BLE001
                entry['error'] = repr(error)
                entry['traceback'] = traceback.format_exc()
                print(entry['traceback'], flush=True)
            stage('section-done', name=name, error=entry['error'])

        verify_trace_t1.take()
        regimes_seeds = [(regime, seed) for regime in arguments.regimes for seed in (arguments.seeds if regime == 'R1' else arguments.seeds[:1])]

        # ---------------- selftest ----------------
        def selftest(entry):
            known = torch.arange(1024 * 512, dtype=torch.int64).remainder(2 ** 31).to(torch.int32).reshape(1024, 512)
            source, sink = words_tensor(known), words_tensor(torch.zeros(1024, 512, dtype=torch.int32))
            try:
                raw_copy(source, sink, 1024)
                sync('selftest')
                back = dev.words_of(torch, ttnn.to_torch(ttnn.get_device_tensors(sink)[0])).reshape(1024, 512)
            finally:
                ttnn.deallocate(source)
                ttnn.deallocate(sink)
            entry['raw_round_trip_exact'] = bool(torch.equal(back, known))
            entry['env'] = report['env_read']
            if not entry['raw_round_trip_exact']:
                raise AssertionError('the raw page copy did not round-trip known bytes')

        # ---------------- sram ----------------
        def sram(entry):
            plan = model.sram_plan(windows=2) if model else None
            entry['plan'] = plan
            entry['dram_plan'] = model.dram_bytes_per_core(windows=2) if model else None
            entry['dram_plan_note'] = 'PLAN (gdn_wy_model.dram_bytes_per_core), not a measurement; the kernel own dram_bytes(windows) is reported as dram_kernel when it has one'
            entry['a_cb_bytes'] = seq.cb_bytes()
            if wy is None:
                entry.update(label='not-run', l1_bytes=L1_BYTES)
                return
            entry.update(sram_verdict({1: builds['W'].cb_bytes(1), 2: builds['W'].cb_bytes(2)}))
            if plan:
                entry['plan_delta_two_windows'] = entry['cb_bytes'][2] - plan['chained_total_bytes']
            if hasattr(builds['W'], 'dram_bytes'):
                entry['dram_kernel'] = {1: builds['W'].dram_bytes(1), 2: builds['W'].dram_bytes(2)}

        # ---------------- causality ----------------
        def causality(entry):
            entry.update(cases=[], w_failures=[], control_a_causal=True, control_blind=False, nan_leaks=[])
            for regime, seed in regimes_seeds:
                # window-sized host sets: A takes 16 rows (windows 1 of the same draw), W takes 32
                host32 = window_inputs(torch, regime, seed, arguments.users, found, 2)
                other32 = window_inputs(torch, regime, seed + 5000, arguments.users, found, 2)
                host16 = (host32[0], [{k: (v[:, :ROWS] if k != 'initial' else v) for k, v in u.items()} for u in host32[1]], host32[2])
                other16 = (other32[0], [{k: (v[:, :ROWS] if k != 'initial' else v) for k, v in u.items()} for u in other32[1]], other32[2])
                for n in sorted(set(arguments.commits)):
                    commit = commit_plan(n)
                    arms = [('A', host16, other16, n)] if n <= ROWS else []
                    if wy is not None:
                        arms.append(('W2', host32, other32, n))
                    for arm, host, other, rows in arms:
                        kind = 'A' if arm == 'A' else 'W'
                        reference = None
                        for variant in ('real', 'zeros', 'other', 'nan', 'earlier'):
                            label = 'causality/%s/%s/seed%d/n%d/%s' % (arm, regime, seed, n, variant)
                            users_host = vary_rows(torch, host[1], rows, variant, other[1])
                            case = Case((host[0], users_host, host[2]))
                            try:
                                images = run_arm(arm, case, label, entry, commit if kind == 'W' else None)
                            finally:
                                case.free()
                            if variant == 'real':
                                reference = images
                                continue
                            results = [prefix_bytes_equal(torch, reference[u], images[u], rows, kind, commit) for u in range(arguments.users)]
                            equal = all(r['exact'] for r in results)
                            row = dict(case=label, arm=arm, n=n, variant=variant, equal=equal)
                            entry['cases'].append(row)
                            if variant == 'earlier':
                                if equal:
                                    entry['control_blind'] = True
                            elif variant == 'nan':
                                if not equal:
                                    entry['nan_leaks'].append(label)
                            elif not equal:
                                if arm == 'A':
                                    entry['control_a_causal'] = False
                                else:
                                    entry['w_failures'].append('%s differs on rows 0..%d or the committed state' % (label, rows - 1))

        # ---------------- packing ----------------
        def packing(entry):
            if wy is None:
                entry.update(w_exact=None, note='no window kernel: nothing to pack')
                return
            entry.update(cases=[], w_exact=True, first_failure=None)
            for regime, seed in regimes_seeds:
                host = window_inputs(torch, regime, seed, arguments.users, found, 2)
                commit = commit_plan(ROWS * WINDOWS)
                case = Case(host)
                try:
                    packed = run_arm('W2', case, 'packing/packed', entry, commit)
                finally:
                    case.free()
                for user in range(arguments.users):
                    solo_case = Case(split_users(host, user, 1))
                    try:
                        solo = run_arm('W2', solo_case, 'packing/solo%d' % user, entry, commit)
                    finally:
                        solo_case.free()
                    same = prefix_bytes_equal(torch, packed[user], solo[0], ROWS * WINDOWS, 'W', commit)
                    entry['cases'].append(dict(regime=regime, seed=seed, user=user, exact=same['exact'], parts=same['parts']))
                    if not same['exact'] and entry['w_exact']:
                        entry['w_exact'] = False
                        entry['first_failure'] = dict(regime=regime, seed=seed, user=user, parts=same['parts'])

        # ---------------- accuracy: the correctness gate ----------------
        def accuracy(entry):
            if wy is None:
                entry.update(note='no window kernel', nonfinite=False, correct=None, correctness_failures=[], reference_failed=None)
                return
            entry.update(cases=[], nonfinite=False, correct=True, correctness_failures=[], reference_failed=None,
                         factor=ACCURACY_FACTOR, rule='W error vs fp64 <= %.1f x K5-A error vs fp64 (output rows and each committed state, both windows)' % ACCURACY_FACTOR)
            for regime, seed in regimes_seeds:
                host32 = window_inputs(torch, regime, seed, arguments.users, found, 2)
                host16 = (host32[0], [{k: (v[:, :ROWS] if k != 'initial' else v) for k, v in u.items()} for u in host32[1]], host32[2])
                a_case, w_case = Case(host16), Case(host32)
                try:
                    a = run_arm('A', a_case, 'accuracy/A1/%s/%d' % (regime, seed), entry)
                    w = run_arm('W2', w_case, 'accuracy/W/%s/%d' % (regime, seed), entry, commit_plan(ROWS * WINDOWS))
                finally:
                    a_case.free()
                    w_case.free()
                # K5-A's second window starts from its OWN committed state (snapshot 15), the chain the sequential baseline A2 runs
                second = (host32[0], [{k: (v[:, ROWS:] if k != 'initial' else a[u][1][ROWS - 1].reshape(1, HEADS_PER_CHIP, 128, 128))
                                       for k, v in user.items()} for u, user in enumerate(host32[1])], host32[2])
                a2_case = Case(second)
                try:
                    a2 = run_arm('A', a2_case, 'accuracy/A2/%s/%d' % (regime, seed), entry)
                finally:
                    a2_case.free()
                for user in range(arguments.users):
                    result = dev.compare(torch, a[user][0][0, :ROWS], w[user][0][0, :ROWS], base.locate_output)
                    finite = bool(torch.isfinite(w[user][0].float()).all())
                    entry['nonfinite'] = entry['nonfinite'] or not finite
                    a_output = torch.cat([a[user][0][:, :ROWS], a2[user][0][:, :ROWS]], dim=1)
                    verdict = accuracy_case(torch, host32[1][user], host32[0], a_output, [a[user][1][ROWS - 1], a2[user][1][ROWS - 1]],
                                            w[user][0][:, :ROWS * WINDOWS], [w[user][1][0], w[user][1][1]])
                    label = '%s seed %d user %d' % (regime, seed, user)
                    if not verdict['ok']:
                        entry['correct'] = False
                        entry['correctness_failures'].append('%s: %s' % (label, '; '.join(
                            '%s W %.3g > %.1f x K5-A %.3g' % (name, m['w_vs_fp64'], ACCURACY_FACTOR, m['a_vs_fp64'])
                            for name, m in verdict['metrics'].items() if not m['ok'])))
                    if verdict['reference_failed'] and not entry['reference_failed']:
                        entry['reference_failed'] = '%s: %s' % (label, verdict['reference_failed'])
                    entry['cases'].append(dict(regime=regime, seed=seed, user=user, window_one_rows_vs_k5a=dict(
                        exact=result['exact'], differing=result.get('differing'), of=result.get('of'), max_abs=result.get('max_abs'),
                        ulp_histogram=result.get('ulp_histogram')), finite=finite, gate=verdict))
            if entry['reference_failed']:
                entry['correct'] = None

        # ---------------- timing ----------------
        def timing(entry):
            arms = [arm for arm in arguments.timing_arms if arm in ('A', 'A2') or wy is not None]
            orders = dev.serpentine(arms, arguments.timing_rounds)
            sets, traces, owned = [], {}, []
            entry.update(arms=arms, launches=arguments.timing_launches, rounds=arguments.timing_rounds,
                         orders=[','.join(order) for order in orders], per_launch={}, verify_t1_counts={},
                         sequential_window_anchor_us=SEQUENTIAL_WINDOW_US, unit='one launch set: A one window, A2 two windows, W1 one, W2 two')

            checks = {}   # set index -> [(label, device tensor, host value)], compared on the LOGICAL region only (padding is ttnn's)

            def upload_host(host, index, tag):
                norm_w_device = upload(host[0])
                owned.append(norm_w_device)
                kept = checks.setdefault(index, [])
                kept.append((dict(user=None, tensor='norm_w', set=index, uploaded_for=tag), norm_w_device, host[0]))
                groups = []
                for user, values in enumerate(host[1]):
                    tensors = tuple(upload(values[name]) for name in ('qkv', 'beta', 'gate', 'initial', 'z'))
                    owned.extend(tensors)
                    kept.extend((dict(user=user, tensor=name, set=index, uploaded_for=tag), tensor, values[name])
                                for name, tensor in zip(('qkv', 'beta', 'gate', 'initial', 'z'), tensors))
                    groups.append(tensors + (norm_w_device,))
                return groups

            def inputs_after_replays():
                """A launch that writes into its inputs would pass every other check: read a sample of the sets back (every sixth and the last)."""
                moved = []
                for index in sorted(set(range(0, len(sets), 6)) | {len(sets) - 1}):
                    for label, tensor, value in checks.get(index, ()):
                        shown = dev.bits16(torch, raw_host(tensor)[0])[tuple(slice(0, size) for size in value.shape)]
                        if not torch.equal(shown, dev.bits16(torch, value.bfloat16())):
                            moved.append(label)
                report['launches']['input_checks'] += 1
                return moved

            def run_unit(arm, item):
                if arm == 'A':
                    release(launch('A', item['a'][0]))
                elif arm == 'A2':
                    release(launch('A', item['a'][0]))
                    release(launch('A', item['a'][1]))
                else:
                    release(launch(arm, item['w1'] if arm == 'W1' else item['w2'], None, None))

            try:
                for index in range(arguments.timing_launches):
                    host32 = window_inputs(torch, 'R1', 3000 + index, arguments.users, found, 2)
                    second = window_inputs(torch, 'R1', 3000 + index + 977, arguments.users, found, 1)
                    first16 = (host32[0], [{k: (v[:, :ROWS] if k != 'initial' else v) for k, v in u.items()} for u in host32[1]], host32[2])
                    second16 = (second[0], second[1], second[2])
                    item = dict(a=[upload_host(first16, index, 'A first window'), upload_host(second16, index, 'A second window')])
                    if wy is not None:
                        item['w2'] = upload_host(host32, index, 'W2')
                        item['w1'] = upload_host(first16, index, 'W1')
                    sets.append(item)
                for arm in arms:
                    stage('timing-capture', arm=arm)
                    run_unit(arm, sets[0])
                    sync('timing warm')
                    verify_trace_t1.take()
                    handle = ttnn.begin_trace_capture(device, cq_id=0)
                    try:
                        for item in sets:
                            run_unit(arm, item)
                    finally:
                        ttnn.end_trace_capture(device, handle, cq_id=0)
                    sync('timing capture')
                    traces[arm] = handle
                    entry['verify_t1_counts'][arm] = verify_trace_t1.take()
                for arm in arms:
                    for unused in range(2):
                        ttnn.execute_trace(device, traces[arm], cq_id=0, blocking=False)
                    sync('timing warm replay')
                samples = {arm: [] for arm in arms}
                stage('timing-replay')
                for order in orders:
                    for arm in order:
                        begin = time.perf_counter()
                        with watchdog.guard('timing replay %s' % arm, max(arguments.call_timeout, 600.0)):
                            ttnn.execute_trace(device, traces[arm], cq_id=0, blocking=False)
                            ttnn.synchronize_device(device)
                        samples[arm].append((time.perf_counter() - begin) * 1e6 / arguments.timing_launches)
                for arm in arms:
                    entry['per_launch'][arm] = dev.summarize_timing(samples[arm])
                entry['samples_us'] = samples
                for label in inputs_after_replays():
                    report['inputs_moved'].append(dict(label, arm='timing replays', case='timing'))
                medians = {arm: entry['per_launch'][arm]['median_us'] for arm in arms}
                entry.update(timing_verdict(medians.get('A'), medians.get('A2'), medians.get('W1'), medians.get('W2')))
            finally:
                for handle in traces.values():
                    ttnn.release_trace(device, handle)
                for value in owned:
                    ttnn.deallocate(value)
                sync('timing teardown')

        section('selftest', selftest)
        section('sram', sram)
        section('causality', causality)
        section('packing', packing)
        section('accuracy', accuracy)
        section('timing', timing)

        verdict, problems = decide(report)
        report.update(verdict=verdict, problems=problems)
        report['verdict_line'] = verdict_line(report)
        code = exit_code(verdict)
        summary.update(verdict=verdict, scope='reduced' if report['missing_scope'] else 'full', problems=problems,
                       kernel_present=report['kernel_present'], a_qualified=report['a_qualified'],
                       timing={key: report['sections'].get('timing', {}).get(key)
                               for key in ('label', 'a_us', 'a2_us', 'w1_us', 'w2_us', 'speedup_vs_a2')},
                       report=str(arguments.out))
        print(report['verdict_line'], flush=True)
    except BaseException as error:  # noqa: BLE001
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        report['verdict'] = 'NO-DECISION'
        summary.update(verdict='NO-DECISION', error=repr(error)[:400])
        print(report['traceback'], flush=True)
        code = 4
    finally:
        write()
        if mesh[0] is not None:
            try:
                import ttnn
                ttnn.close_mesh_device(mesh[0])
            except BaseException:  # noqa: BLE001
                pass
        summary['kind'] = KIND
        print(json.dumps(summary, default=str), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
