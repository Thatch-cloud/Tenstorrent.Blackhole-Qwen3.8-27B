"""Attribute the Lever N M3native packed-round device profile to ttnn ops and to
inferred layer-type buckets (full attention / GDN / MLP / norms / collectives /
sampling), from the pinned tracy device profiler's own exported CSVs.

Reuses the CSV parsing and column names request_verifier_profile_report.py already
established for this pinned tracy/tt-metal build (OP NAME, CORE COUNT, DEVICE ID,
DEVICE KERNEL DURATION [ns], METAL TRACE ID, METAL TRACE REPLAY SESSION ID) and the
same trace-tracking idea it uses: a captured trace is attached once, then replayed
once per round. packed_verifier.py's own comment on
`self.fixture.retained.replay(operation)` says exactly this - "a captured trace's
Python call graph runs once at attach, not per round". The M3native gate serves
decode under trace_mode='decode_only' (lever_n_m3native_gate.py's start_server), so
only decode/packed-round device ops are trace-tracked; prefill runs untraced and
its rows carry no METAL TRACE ID. This script relies on that distinction to isolate
packed-round device activity from prefill, without any wall-clock correlation
between the Python log and the device CSV's own cycle-based timestamps - there is
no shared clock to convert between them here.

WHAT THIS DOES NOT CLAIM: OP NAME in the pinned C++ device report is frequently a
generic primitive name. docs/current-verifier-profile-2026-09-09.md's own
core-count-based matmul groups still "mix MLP down and attention/GDN output
projections" - the same ambiguity applies here. The layer-type buckets below match
op names by keyword and are therefore best-effort: an op whose name carries no
distinguishing token (a bare "Matmul", say) reports as 'other', not as a guess.
Nothing here converts a kernel duration sum into a critical path, a TG figure or a
promotion claim; summed kernel durations across many cores/RISCs can and typically
do exceed a single host-side blocking-call wall-clock duration, because those cores
run in parallel while a wall clock does not.

Attribution method, in order of preference (see select_round_rowsets):
  - per_round_trace_replay: exactly one METAL TRACE ID is trace-tracked, and every
    chip reports the same number of distinct METAL TRACE REPLAY SESSION ID values
    as there are [PACKED-PHASE] round lines in the log; each replay session maps
    1:1, in ascending order, to a round.
  - per_round_trace_replay_capture_excluded: as above but with exactly one extra
    replay session per chip, assumed to be the trace's own capture pass (not a
    real round) and excluded, keeping the round mapping 1:1.
  - averaged_over_decode_phase: trace-tracked rows exist but cannot be split into
    one set per round (session counts do not match the round count, sessions are
    not consistent across chips, or more than one distinct trace id was
    trace-tracked); every op's total across ALL trace-tracked rows is divided
    evenly by the number of packed rounds instead.
  - averaged_over_all_rows: no METAL TRACE ID / METAL TRACE REPLAY SESSION ID data
    is available at all (columns absent, or empty on every row); every op's total
    across the ENTIRE CSV is divided evenly by the number of packed rounds. This
    cannot exclude prefill or any other untraced device activity and is the
    weakest of the four methods.
Every report states which method applied and why. A missing or empty CSV column is
reported as 'not exported by the pinned profiler', never silently treated as zero.

CPU-only, stdlib only.
"""

import argparse
import copy
import csv
import json
import re
import statistics
import sys
from pathlib import Path

PACKED_PHASE_RE = re.compile(
    r'\[PACKED-PHASE\]\s+round=(?P<round>\d+)\s+users=(?P<users>\d+)\s+'
    r'bind_ms=(?P<bind_ms>[0-9.]+)\s+input_ms=(?P<input_ms>[0-9.]+)\s+'
    r'trace_ms=(?P<trace_ms>[0-9.]+)\s+sync_ms=(?P<sync_ms>[0-9.]+)\s+'
    r'readback_ms=(?P<readback_ms>[0-9.]+)')
PHASE_BEGIN_RE = re.compile(r'\[PHASE\] packed_verify .* begin')
PHASE_END_RE = re.compile(r'\[PHASE\] packed_verify .* end ([0-9.]+) ms')

OP_NAME_COLUMN = 'OP NAME'
CORE_COUNT_COLUMN = 'CORE COUNT'
DEVICE_ID_COLUMN = 'DEVICE ID'
DURATION_COLUMN = 'DEVICE KERNEL DURATION [ns]'
TRACE_ID_COLUMN = 'METAL TRACE ID'
REPLAY_SESSION_COLUMN = 'METAL TRACE REPLAY SESSION ID'

REQUIRED_COLUMNS = (OP_NAME_COLUMN, DEVICE_ID_COLUMN, DURATION_COLUMN)
NOT_EXPORTED = 'not exported by the pinned profiler'

# Priority-ordered: the first bucket whose keyword matches (as a substring of the
# op name, with case and separators normalised away) wins. Keywords are drawn from
# this repo's own graft module/op names - qkvzab, gdn_norm_gate,
# decode_gated_delta_rule, gate_up/mlp_down, force_argmax, attn_prep,
# nlp_concat_heads_decode are all grep-confirmed in scripts/ci and optimisation -
# plus tt-metal's own primitive op names (sdpa, rms_norm, all_gather,
# reduce_scatter, all_reduce). There is deliberately no keyword for a bare
# "matmul", "wo" or generic "out_proj"/"output_projection" token: this repo's own
# test names use "output_projection" for both attention and GDN, so a wrong guess
# there is worse than reporting 'other'.
BUCKETS = (
    ('full_attention', ('attndecodeprep', 'attnprep', 'sdpa',
                         'scaleddotproductattention', 'pagedattention',
                         'nlpconcatheadsdecode')),
    ('gdn', ('gateddeltarule', 'decodegateddeltarule', 'gdnnormgate', 'gdndecay',
             'qkvzab', 'gdnconv', 'conv1d', 'gdnrecurrence')),
    ('mlp', ('gateup', 'mlpdown', 'mlpw1', 'mlpw2', 'mlpw3', 'w1matmul',
             'w2matmul', 'w3matmul')),
    ('norm', ('rmsnorm', 'layernorm', 'distributednorm')),
    ('collective', ('allgatherminimalmatmulasync', 'allgather', 'reducescatter',
                     'allreduce', 'fabricallgather')),
    ('sampling_lm_head', ('lmhead', 'forceargmax', 'argmax', 'embedding', 'sampl')),
)


def _normalize(text):
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def classify_op(op_name):
    """Best-effort layer-type bucket for one OP NAME; 'other' when no keyword
    matches (see the module docstring: this is not a last-resort guess)."""
    normalized = _normalize(op_name)
    for bucket, keywords in BUCKETS:
        if any(keyword in normalized for keyword in keywords):
            return bucket
    return 'other'


def find_csv(profile_dir, name):
    """Mirrors request_verifier_profile_report.py's own search order: directly in
    profile_dir, or the preserved copy a wrapper like dflash-request-profile.sh
    makes (metadata/), or the raw tracy output directory (.logs/)."""
    profile_dir = Path(profile_dir)
    for location in (None, 'metadata', '.logs'):
        candidate = profile_dir / name if location is None else profile_dir / location / name
        if candidate.is_file():
            return candidate
    return None


def load_rows(path):
    with Path(path).open(newline='', encoding='utf-8', errors='replace') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return rows, columns


def parse_packed_rounds(log_text):
    """Every '[PACKED-PHASE] round=N ...' line packed_verifier.py logs under
    QWEN_FAST_PACKED_AUDIT=1, in the order they appear."""
    rounds = []
    for match in PACKED_PHASE_RE.finditer(log_text):
        rounds.append(dict(
            round=int(match['round']), users=int(match['users']),
            bind_ms=float(match['bind_ms']), input_ms=float(match['input_ms']),
            trace_ms=float(match['trace_ms']), sync_ms=float(match['sync_ms']),
            readback_ms=float(match['readback_ms'])))
    return rounds


def parse_phase_markers(log_text):
    """A coarse corroborating count of serving_worker_hook.py's own
    '[PHASE] packed_verify <id> begin/end <ms> ms' lines (QWEN_FAST_PHASE_LOG=1) -
    not used for attribution, only reported alongside the [PACKED-PHASE] round
    count as a cross-check that the same number of rounds were observed both ways."""
    return dict(packed_verify_begin=len(PHASE_BEGIN_RE.findall(log_text)),
                packed_verify_end_ms=[float(value) for value in PHASE_END_RE.findall(log_text)])


def _row_duration_ns(row):
    value = row.get(DURATION_COLUMN)
    if value in (None, ''):
        return None
    try:
        duration = float(value)
    except ValueError:
        return None
    return duration if duration >= 0 else None


def attribute_rows(rows, *, has_core_count):
    """Sum DEVICE KERNEL DURATION [ns] over `rows`, grouped by (OP NAME, CORE
    COUNT) - CORE COUNT omitted from the key when the column is absent - both per
    chip (DEVICE ID) and in total. A row with a missing or unusable duration is
    skipped and counted, never treated as zero-cost."""
    per_op = {}
    per_chip_total_ns = {}
    grand_total_ns = 0.0
    skipped = 0
    for row in rows:
        duration = _row_duration_ns(row)
        if duration is None:
            skipped += 1
            continue
        op_name = row.get(OP_NAME_COLUMN) or '<missing OP NAME>'
        cores = row.get(CORE_COUNT_COLUMN) if has_core_count else None
        device = row.get(DEVICE_ID_COLUMN, '<missing DEVICE ID>')
        entry = per_op.setdefault((op_name, cores), dict(op=op_name, cores=cores,
            calls=0, total_ns=0.0, chips={}))
        entry['calls'] += 1
        entry['total_ns'] += duration
        chip_entry = entry['chips'].setdefault(device, dict(calls=0, total_ns=0.0))
        chip_entry['calls'] += 1
        chip_entry['total_ns'] += duration
        per_chip_total_ns[device] = per_chip_total_ns.get(device, 0.0) + duration
        grand_total_ns += duration
    ops = []
    for entry in per_op.values():
        ops.append(dict(
            op=entry['op'], cores=entry['cores'], calls=entry['calls'],
            total_ms=entry['total_ns'] / 1e6,
            mean_ms=(entry['total_ns'] / entry['calls']) / 1e6,
            per_chip_ms={chip: values['total_ns'] / 1e6 for chip, values in entry['chips'].items()},
            per_chip_calls={chip: values['calls'] for chip, values in entry['chips'].items()}))
    ops.sort(key=lambda item: -item['total_ms'])
    return dict(ops=ops, grand_total_ms=grand_total_ns / 1e6,
        per_chip_total_ms={chip: value / 1e6 for chip, value in per_chip_total_ns.items()},
        rows_used=len(rows) - skipped, rows_skipped_invalid_duration=skipped)


def bucket_attribution(op_attribution):
    """Roll attribute_rows' per-op table up into the layer-type buckets, each with
    its total and its share of the ATTRIBUTED total (the sum of every op reported
    here, which is not the same thing as the log's own trace_ms - see the sanity
    section of build_report)."""
    totals, calls = {}, {}
    for entry in op_attribution['ops']:
        bucket = classify_op(entry['op'])
        totals[bucket] = totals.get(bucket, 0.0) + entry['total_ms']
        calls[bucket] = calls.get(bucket, 0) + entry['calls']
    grand_total = op_attribution['grand_total_ms']
    buckets = [dict(bucket=bucket, total_ms=total, calls=calls[bucket],
        share_of_attributed_total=(total / grand_total) if grand_total else None)
        for bucket, total in totals.items()]
    buckets.sort(key=lambda item: -item['total_ms'])
    return buckets


def select_round_rowsets(rows, columns, num_rounds):
    """Choose an attribution method (see the module docstring) and return
    (method, rowsets, caveats). `rowsets` has exactly `num_rounds` entries, one
    per round in round order, only for the two per_round_trace_replay* methods;
    every averaged method returns a single combined rowset instead, meant to be
    divided evenly by num_rounds by the caller."""
    caveats = []
    missing = [name for name in (TRACE_ID_COLUMN, REPLAY_SESSION_COLUMN) if name not in columns]
    if missing:
        caveats.append('%s %s; packed rounds could not be isolated from prefill or '
            'from each other. Attribution is averaged over every row in the CSV.'
            % (' and '.join(missing), NOT_EXPORTED))
        return 'averaged_over_all_rows', [rows], caveats

    tracked = [row for row in rows if row.get(TRACE_ID_COLUMN) not in (None, '')
               and row.get(REPLAY_SESSION_COLUMN) not in (None, '')]
    if not tracked:
        caveats.append('No row carried a METAL TRACE ID / METAL TRACE REPLAY SESSION '
            'ID value; packed rounds could not be isolated from prefill or from each '
            'other. Attribution is averaged over every row in the CSV.')
        return 'averaged_over_all_rows', [rows], caveats

    trace_ids = sorted({row[TRACE_ID_COLUMN] for row in tracked})
    if len(trace_ids) != 1:
        caveats.append('%d distinct METAL TRACE ID values were trace-tracked (expected '
            'one captured decode trace); packed rounds could not be told apart. '
            'Attribution is averaged over every trace-tracked row.' % len(trace_ids))
        return 'averaged_over_decode_phase', [tracked], caveats

    sessions_by_device = {}
    for row in tracked:
        sessions_by_device.setdefault(row[DEVICE_ID_COLUMN], set()).add(int(row[REPLAY_SESSION_COLUMN]))
    distinct_session_sets = {frozenset(sessions) for sessions in sessions_by_device.values()}
    if len(distinct_session_sets) != 1:
        caveats.append('Chips reported different trace replay sessions for the same '
            'trace; packed rounds could not be mapped 1:1 across chips. Attribution '
            'is averaged over every trace-tracked row.')
        return 'averaged_over_decode_phase', [tracked], caveats

    ordered_sessions = sorted(next(iter(distinct_session_sets)))
    replay_count = len(ordered_sessions)
    if num_rounds and replay_count == num_rounds:
        drop_first = False
        method = 'per_round_trace_replay'
    elif num_rounds and replay_count == num_rounds + 1:
        drop_first = True
        method = 'per_round_trace_replay_capture_excluded'
        caveats.append('One extra trace replay session per chip beyond the observed '
            'round count was assumed to be the trace capture pass itself (not a real '
            'round) and excluded.')
    else:
        caveats.append('%d trace replay session(s) per chip did not match the %s '
            'packed round(s) observed in the log; attribution is averaged over every '
            'trace-tracked row instead of split per round.'
            % (replay_count, num_rounds if num_rounds else 'zero'))
        return 'averaged_over_decode_phase', [tracked], caveats

    if drop_first:
        ordered_sessions = ordered_sessions[1:]
    rowsets = [[row for row in tracked if int(row[REPLAY_SESSION_COLUMN]) == session]
               for session in ordered_sessions]
    return method, rowsets, caveats


def build_report(profile_dir, log_path):
    profile_dir = Path(profile_dir)
    log_path = Path(log_path)
    log_text = log_path.read_text(encoding='utf-8', errors='replace') if log_path.is_file() else ''
    packed_rounds = parse_packed_rounds(log_text)
    num_rounds = len(packed_rounds)

    report = dict(
        scope='M3native packed-round device attribution; not a TG or critical-path claim',
        profile_dir=str(profile_dir), log_path=str(log_path),
        log_found=log_path.is_file(), packed_rounds=packed_rounds,
        num_packed_rounds_observed=num_rounds, phase_markers=parse_phase_markers(log_text),
        caveats=[])

    tracy_path = find_csv(profile_dir, 'tracy_ops_data.csv')
    device_path = find_csv(profile_dir, 'cpp_device_perf_report.csv')
    report['tracy_ops_data_csv'] = str(tracy_path) if tracy_path else None
    report['cpp_device_perf_report_csv'] = str(device_path) if device_path else None
    if tracy_path is None:
        report['caveats'].append('tracy_ops_data.csv %s; op-name disambiguation from '
            'kernel source paths was not attempted, only the OP NAME column.' % NOT_EXPORTED)

    report['attribution_method'] = None
    report['per_op'] = []
    report['per_bucket'] = []
    report['sanity'] = None

    if device_path is None:
        report['error'] = 'cpp_device_perf_report.csv %s; no device attribution is possible.' % NOT_EXPORTED
        return report

    rows, columns = load_rows(device_path)
    report['device_csv_columns'] = columns
    missing_required = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing_required:
        report['error'] = ('%s %s; no device attribution is possible.'
            % (' and '.join(missing_required), NOT_EXPORTED))
        return report

    has_core_count = CORE_COUNT_COLUMN in columns
    if not has_core_count:
        report['caveats'].append('%s %s; ops are grouped by OP NAME only.' % (CORE_COUNT_COLUMN, NOT_EXPORTED))

    method, rowsets, method_caveats = select_round_rowsets(rows, columns, num_rounds)
    report['attribution_method'] = method
    report['caveats'].extend(method_caveats)

    per_round_capable = method in ('per_round_trace_replay', 'per_round_trace_replay_capture_excluded')
    combined_rows = [row for rowset in rowsets for row in rowset] if per_round_capable else rowsets[0]
    combined = attribute_rows(combined_rows, has_core_count=has_core_count)

    top_ops = combined['ops'][:40]
    report['per_op'] = top_ops
    report['per_op_total_ops_observed'] = len(combined['ops'])
    report['per_bucket'] = bucket_attribution(combined)
    report['grand_total_attributed_ms'] = combined['grand_total_ms']
    report['per_chip_total_attributed_ms'] = combined['per_chip_total_ms']
    report['rows_used'] = combined['rows_used']
    report['rows_skipped_invalid_duration'] = combined['rows_skipped_invalid_duration']

    sanity = dict(method=method)
    if num_rounds:
        scale = 1.0 / num_rounds
        mean_per_round = copy.deepcopy(dict(per_op=top_ops, per_bucket=report['per_bucket'],
            grand_total_ms=combined['grand_total_ms']))
        for entry in mean_per_round['per_op']:
            entry['total_ms'] *= scale
            entry['mean_ms'] *= scale
            entry['per_chip_ms'] = {chip: value * scale for chip, value in entry['per_chip_ms'].items()}
        for entry in mean_per_round['per_bucket']:
            entry['total_ms'] *= scale
        mean_per_round['grand_total_ms'] *= scale
        report['mean_per_round'] = mean_per_round

        reported_trace_ms = [entry['trace_ms'] for entry in packed_rounds]
        mean_trace_ms = statistics.fmean(reported_trace_ms)
        sanity['mean_reported_trace_ms'] = mean_trace_ms
        sanity['mean_attributed_device_ms_per_round'] = mean_per_round['grand_total_ms']
        sanity['delta_ms'] = mean_per_round['grand_total_ms'] - mean_trace_ms
        sanity['ratio_attributed_to_reported'] = (
            mean_per_round['grand_total_ms'] / mean_trace_ms) if mean_trace_ms else None

        if per_round_capable:
            per_round_sanity = []
            for round_info, rowset in zip(packed_rounds, rowsets):
                attributed_ms = sum(_row_duration_ns(row) or 0.0 for row in rowset) / 1e6
                per_round_sanity.append(dict(
                    round=round_info['round'], reported_trace_ms=round_info['trace_ms'],
                    attributed_device_ms=attributed_ms,
                    delta_ms=attributed_ms - round_info['trace_ms'],
                    ratio_attributed_to_reported=(
                        attributed_ms / round_info['trace_ms']) if round_info['trace_ms'] else None))
            sanity['per_round'] = per_round_sanity
        else:
            sanity['note'] = ('mean_per_round above is the whole attributed scope (method=%s) '
                'divided evenly by %d observed round(s); it is an average, not a measured '
                'per-round breakdown.' % (method, num_rounds))
    else:
        sanity['note'] = 'No [PACKED-PHASE] round lines were found in the log; per-round figures cannot be computed.'
    report['sanity'] = sanity

    report['caveats'].extend([
        'Kernel-duration sums across many cores/RISCs are not directly comparable to a '
        'single host-side blocking-call wall-clock duration (cores run in parallel); a '
        'higher attributed sum than the reported trace_ms is expected, not an error.',
        'Layer-type buckets are matched by OP NAME keyword only (see module docstring); '
        'an op with no distinguishing name reports as "other", not as a guess.',
    ])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile-dir', required=True, type=Path,
        help='directory holding tracy_ops_data.csv and cpp_device_perf_report.csv '
             '(or their metadata/ or .logs/ subdirectories)')
    parser.add_argument('--log', required=True, type=Path,
        help='the m3native-gate-stdout.log to read [PACKED-PHASE]/[PHASE] lines from')
    parser.add_argument('--out', type=Path, help='also write the JSON report here')
    options = parser.parse_args()

    report = build_report(options.profile_dir, options.log)
    text = json.dumps(report, indent=2)
    print(text)
    if options.out:
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(text, encoding='utf-8', newline='\n')
    return 1 if report.get('error') else 0


if __name__ == '__main__':
    sys.exit(main())
