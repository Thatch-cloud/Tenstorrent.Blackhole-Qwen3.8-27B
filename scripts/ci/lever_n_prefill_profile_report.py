"""Group a prefill device profile by op and by prefill chunk (prefill ranking, M2).

The M2 arm profiles one 131,072-token prompt on the served image with the layer.py flush hook
(QWEN_PREFILL_PROFILE_FLUSH=1, lever_n_m3native_patch section G): the device profiler is read
every 16 decoder layers, so a chunk can no longer overflow the per-core buffers, and chunks 0,
1, 31, 32, 62 and 63 of each prompt are bracketed by tracy signposts
qwen_prefill_p<prompt>_chunk_<n>_begin / _end.

The arm's tracy invocation (-p --disable-device-data-dump-to-files ... with no ops report)
writes cpp_device_perf_report.csv but NO tracy_ops_data.csv (the gate12b capture has none),
and the device CSV carries no signpost rows. So the per-chunk split this report relies on is
DEVICE-ONLY (by_sdpa); the signpost join runs only if a tracy_ops_data.csv is ever exported.

This report reads the pinned profiler's exported CSVs (the same files and column names
lever_n_m3native_profile_report.py reads) and produces:

  per_op       every untraced device op (prefill runs untraced; decode rows carry a METAL
               TRACE ID and are excluded), per chip: calls, total and mean kernel ms.
  by_sdpa      THE per-chunk split. SDPAOperation (the chunked prefill SDPA; decode's op has
               another name) runs exactly once per full-attention layer per chunk per chip
               (16 of 64 layers), and nothing else runs it. Per chip, in GLOBAL CALL COUNT
               order, its calls are grouped by 16; the LAST `chunks` groups are the profiled
               prompt (MAX_TOKENS=1: nothing prefills after it), so warm-up or probe prefills
               earlier in the process are counted as warm-up groups and cannot shift the
               index. A chunk's rows start at layer 0's attention norm (the 7th
               LayerNormPreAllGather before the group's first SDPA: 2 per GDN layer 0-2 plus
               layer 3's attention norm) and end before the (2 x layers + 1)-th such norm, so
               a final norm or lm_head between chunks falls in no chunk (gap_rows).
  coverage     the ranking's 100%-capture check, over the PROMPT's rows only (by_sdpa's span):
               TernaryDeviceOperation should appear 3 x GDN layers x chunks times per chip,
               LayerNormPreAllGather 2 x layers x chunks times. A ratio below 1 means rows were
               lost. The all-untraced counts are kept alongside (warm-up rows inflate them).
  by_norm      the cross-check: per chip, a new chunk at every (2 x layers)-th
               LayerNormPreAllGather from the first, re-indexed by by_sdpa's warm-up groups.
               Anything else that runs that op (a per-chunk final norm, a draft model) drifts
               it; `agreement` lists, per wanted chunk, whether both splits start at the same
               GLOBAL CALL COUNT.
  signposted   only when a tracy_ops_data.csv exists: the last prompt's signposted chunks
               joined to the device CSV by GLOBAL CALL COUNT.
  log          the flush hook's own [PINDIAG] markers and any dropped-marker / buffer-full
               lines, which mean the capture is not complete whatever the counts say.

WHAT THIS DOES NOT CLAIM: kernel-duration sums across cores are not a critical path; the busy
union is per chip from the device's own cycle counters (converted with the frequency each row
implies), not host wall time. The wall-vs-busy comparison the ranking asks for needs the
second, unprofiled arm's per-chunk wall; this report gives the device side of it.

CPU-only, stdlib only.
"""

import argparse
import csv
import io
import json
import re
import statistics
import sys
from pathlib import Path

from lever_n_m3native_profile_report import find_csv, load_rows, NOT_EXPORTED

OP_NAME = 'OP NAME'
DEVICE_ID = 'DEVICE ID'
DURATION = 'DEVICE KERNEL DURATION [ns]'
TRACE_ID = 'METAL TRACE ID'
CALL_COUNT = 'GLOBAL CALL COUNT'
FW_START, FW_END, FW_DURATION = 'DEVICE FW START CYCLE', 'DEVICE FW END CYCLE', 'DEVICE FW DURATION [ns]'

SIGNPOST_RE = re.compile(r'^qwen_prefill_p(\d+)_chunk_(\d+)_(begin|end)$')
FLUSH_MARKER = '[PINDIAG] prefill profile flush'
DROPPED_RE = re.compile(r'(?i)(markers? (?:were |was )?dropped|buffers? (?:is |are |were |was )?full)')
TERNARY = 'TernaryDeviceOperation'
SDPA_OP = 'SDPAOperation'
NORM = 'LayerNormPreAllGather'
SDPA_TOKENS = ('SDPA', 'ScaledDotProductAttention')


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def untraced(rows):
    """Prefill rows: no METAL TRACE ID (the gate serves decode under trace, prefill untraced)."""
    return [row for row in rows if not (row.get(TRACE_ID) or '').strip()]


def per_op(rows):
    """{op: {calls, total_ms, mean_ms, per_chip: {chip: {calls, total_ms}}}}, heaviest first."""
    table = {}
    for row in rows:
        duration = _number(row.get(DURATION))
        if duration is None:
            continue
        op = (row.get(OP_NAME) or '<missing OP NAME>').strip()
        chip = (row.get(DEVICE_ID) or '?').strip()
        entry = table.setdefault(op, dict(op=op, calls=0, total_ms=0.0, per_chip={}))
        entry['calls'] += 1
        entry['total_ms'] += duration / 1e6
        cell = entry['per_chip'].setdefault(chip, dict(calls=0, total_ms=0.0))
        cell['calls'] += 1
        cell['total_ms'] += duration / 1e6
    ops = sorted(table.values(), key=lambda item: -item['total_ms'])
    for entry in ops:
        entry['mean_ms'] = entry['total_ms'] / entry['calls']
    return ops


def _count(rows, token):
    counts = {}
    for row in rows:
        if token in (row.get(OP_NAME) or ''):
            chip = (row.get(DEVICE_ID) or '?').strip()
            counts[chip] = counts.get(chip, 0) + 1
    return counts


def coverage(rows, *, chunks, layers, gdn_layers, all_rows=None):
    """The ranking's capture check, per chip: observed / expected for the two anchor ops over
    `rows` (the prompt's span), with the all-untraced counts alongside when given."""
    expected = {TERNARY: 3 * gdn_layers * chunks, NORM: 2 * layers * chunks}
    result = {}
    for token, want in expected.items():
        observed = _count(rows, token)
        result[token] = dict(expected_per_chip=want, observed_per_chip=observed,
                             ratio_per_chip={chip: count / want for chip, count in observed.items()} if want else {})
        if all_rows is not None:
            result[token]['observed_all_untraced_per_chip'] = _count(all_rows, token)
    return result


def busy(rows):
    """Per chip: union of [FW start, FW end] cycle intervals, and the span, in ms.

    The cycle-to-ns rate is the median (end - start) / FW DURATION [ns] over the rows that
    carry all three, so the report never assumes a clock."""
    result = {}
    by_chip = {}
    for row in rows:
        start, end = _number(row.get(FW_START)), _number(row.get(FW_END))
        if start is None or end is None or end < start:
            continue
        by_chip.setdefault((row.get(DEVICE_ID) or '?').strip(), []).append((start, end, _number(row.get(FW_DURATION))))
    for chip, intervals in by_chip.items():
        rates = [(end - start) / ns for start, end, ns in intervals if ns and end > start]
        if not rates:
            result[chip] = dict(error='no row carries both FW cycles and FW DURATION [ns]')
            continue
        rate = statistics.median(rates)  # cycles per ns
        union, current = 0.0, None
        for start, end, _ in sorted(intervals):
            if current is None or start > current[1]:
                if current is not None:
                    union += current[1] - current[0]
                current = [start, end]
            else:
                current[1] = max(current[1], end)
        union += current[1] - current[0]
        first = min(start for start, _, _ in intervals)
        last = max(end for _, end, _ in intervals)
        result[chip] = dict(busy_union_ms=union / rate / 1e6, span_ms=(last - first) / rate / 1e6,
                            cycles_per_ns=rate, ops=len(intervals))
    return result


def sdpa_calls(rows):
    """Per chip: calls and per-call mean/median kernel ms of the chunked SDPA op."""
    result = {}
    for row in rows:
        name = row.get(OP_NAME) or ''
        duration = _number(row.get(DURATION))
        if duration is None or not any(token in name for token in SDPA_TOKENS) or 'Decode' in name:
            continue
        result.setdefault((row.get(DEVICE_ID) or '?').strip(), []).append(duration / 1e6)
    return {chip: dict(calls=len(values), mean_ms=statistics.fmean(values), median_ms=statistics.median(values))
            for chip, values in result.items()}


def summarise(rows, top=25):
    return dict(ops=per_op(rows)[:top], busy=busy(rows), sdpa=sdpa_calls(rows), rows=len(rows))


def parse_tracy_ops(text):
    """(time_ns, kind, value) events from tracy_ops_data.csv, in time order.

    The pinned process_ops_logs reads this file as ';'-delimited with '`' quoting and the
    columns MessageName / total_ns. A signpost is 'TT_SIGNPOST: <label>'; a device op either
    carries its JSON (with global_call_count) after ' ->' or, when cached, is
    'TT_DNN_DEVICE_OP: <name>, <hash>, <device>, <op id>, ...'."""
    lines = text.splitlines()
    if not lines:
        return []
    delimiter = ';' if ';' in lines[0] else ','
    events = []
    for row in csv.DictReader(io.StringIO(text), delimiter=delimiter, quotechar='`'):
        message = row.get('MessageName') or row.get('message') or ''
        when = _number(row.get('total_ns'))
        if when is None:
            continue
        if 'TT_SIGNPOST' in message:
            label = message.split('TT_SIGNPOST:', 1)[-1].strip().strip('`').splitlines()[0].strip()
            events.append((when, 'signpost', label))
        elif 'TT_DNN_DEVICE_OP' in message:
            call = None
            if '->' in message:
                body = message.split('->', 1)[1].strip().strip('`')
                try:
                    call = int(json.loads(body)['global_call_count'])
                except (ValueError, KeyError, TypeError):
                    call = None
            else:
                parts = message.split(':', 1)[-1].split(',')
                if len(parts) > 3:
                    try:
                        call = int(parts[3].strip())
                    except ValueError:
                        call = None
            if call is not None:
                events.append((when, 'op', call))
    events.sort(key=lambda event: event[0])
    return events


def chunk_of_calls(events):
    """GLOBAL CALL COUNT -> signposted chunk index of the LAST prompt, for ops between a chunk's
    begin and its end (or the next begin, if an end is missing). Earlier prompts' signposts
    (warm-up prefills) close the current chunk but are not mapped."""
    prompts = [int(match.group(1)) for match in (SIGNPOST_RE.match(value) for _, kind, value in events
                                                  if kind == 'signpost') if match]
    last = max(prompts) if prompts else None
    mapping, current = {}, None
    for _, kind, value in events:
        if kind == 'signpost':
            match = SIGNPOST_RE.match(value)
            if match:
                current = (int(match.group(2)) if match.group(3) == 'begin' and int(match.group(1)) == last
                           else None)
        elif current is not None:
            mapping[value] = current
    return mapping


def group_by_signpost(rows, mapping):
    groups = {}
    for row in rows:
        call = _number(row.get(CALL_COUNT))
        chunk = mapping.get(int(call)) if call is not None else None
        if chunk is not None:
            groups.setdefault(chunk, []).append(row)
    return {str(chunk): summarise(group) for chunk, group in sorted(groups.items())}


def _by_chip(rows):
    """Per chip, the rows that carry a GLOBAL CALL COUNT, in call order."""
    by_chip = {}
    for row in rows:
        if _number(row.get(CALL_COUNT)) is not None:
            by_chip.setdefault((row.get(DEVICE_ID) or '?').strip(), []).append(row)
    for chip_rows in by_chip.values():
        chip_rows.sort(key=lambda row: _number(row.get(CALL_COUNT)))
    return by_chip


def _is_norm(row):
    return NORM in (row.get(OP_NAME) or '')


def _is_sdpa(row):
    return (row.get(OP_NAME) or '').strip() == SDPA_OP


def split_by_sdpa(rows, *, chunks, layers, gdn_layers, first_full_layer=3):
    """Per chip: {'rows': call-ordered rows, 'bounds': {prompt chunk: (start, end) positions},
    'consistency': {...}}. See the module docstring (by_sdpa)."""
    per_chunk = layers - gdn_layers
    norms_before = 2 * first_full_layer + 1
    norms_per_chunk = 2 * layers
    result = {}
    for chip, chip_rows in _by_chip(rows).items():
        sdpa = [index for index, row in enumerate(chip_rows) if _is_sdpa(row)]
        groups = len(sdpa) // per_chunk if per_chunk > 0 else 0
        remainder = len(sdpa) - groups * per_chunk
        consistency = dict(sdpa_calls=len(sdpa), sdpa_per_chunk=per_chunk, groups=groups, remainder=remainder,
                           warmup_groups=max(groups - chunks, 0),
                           complete=bool(per_chunk) and groups >= chunks and remainder == 0)
        bounds, starts = {}, []
        if groups:
            # Whole groups counted from the END (a partial group, from lost rows, is dropped at the
            # front): the last `chunks` groups are the prompt. With too few groups the earliest
            # prompt chunks are the ones missing.
            anchored = sdpa[remainder:]
            for group in range(max(groups - chunks, 0), groups):
                anchor = anchored[group * per_chunk]
                floor = anchored[group * per_chunk - 1] + 1 if group else 0
                start, seen = anchor, 0
                for index in range(anchor - 1, floor - 1, -1):
                    if _is_norm(chip_rows[index]):
                        seen += 1
                        start = index
                        if seen == norms_before:
                            break
                starts.append((group - (groups - chunks), start))
            for position, (chunk, start) in enumerate(starts):
                limit = starts[position + 1][1] if position + 1 < len(starts) else len(chip_rows)
                end, seen = limit, 0
                for index in range(start, limit):
                    if _is_norm(chip_rows[index]):
                        seen += 1
                        if seen == norms_per_chunk + 1:
                            end = index
                            break
                bounds[chunk] = (start, end)
            consistency['gap_rows'] = sum(starts[i + 1][1] - bounds[starts[i][0]][1] for i in range(len(starts) - 1))
        result[chip] = dict(rows=chip_rows, bounds=bounds, consistency=consistency)
    return result


def prompt_rows(split):
    """Every chip's rows inside the prompt's chunks (for coverage)."""
    kept = []
    for entry in split.values():
        for start, end in entry['bounds'].values():
            kept.extend(entry['rows'][start:end])
    return kept


def group_by_sdpa(split, *, wanted):
    chunks = {}
    for entry in split.values():
        for chunk, (start, end) in entry['bounds'].items():
            if chunk in wanted:
                chunks.setdefault(chunk, []).extend(entry['rows'][start:end])
    return dict(consistency={chip: entry['consistency'] for chip, entry in split.items()},
                chunks={str(chunk): summarise(group) for chunk, group in sorted(chunks.items())})


def group_by_norm(rows, *, layers, wanted, split=None):
    """Cross-check split: per chip, in GLOBAL CALL COUNT order, every (2 x layers)-th
    LayerNormPreAllGather opens a chunk, re-indexed by the SDPA split's warm-up groups; and,
    per wanted chunk, whether it starts at the same GLOBAL CALL COUNT as the SDPA split."""
    per_chunk = 2 * layers
    chunks, consistency, agreement = {}, {}, {}
    for chip, chip_rows in _by_chip(rows).items():
        warmup = split[chip]['consistency']['warmup_groups'] if split and chip in split else 0
        norms, chunk, starts = 0, -1, {}
        for index, row in enumerate(chip_rows):
            if _is_norm(row):
                if norms % per_chunk == 0:
                    chunk += 1
                    starts[chunk - warmup] = index
                norms += 1
            if chunk - warmup in wanted:
                chunks.setdefault(chunk - warmup, []).append(row)
        consistency[chip] = dict(norm_calls=norms, chunks_inferred=chunk + 1, warmup_groups=warmup,
                                 divides_evenly=(norms % per_chunk == 0))
        if split and chip in split:
            bounds = split[chip]['bounds']

            def call(index, chip_rows=chip_rows):
                return int(_number(chip_rows[index].get(CALL_COUNT)))

            agreement[chip] = {str(c): dict(sdpa_start_call=call(bounds[c][0]) if c in bounds else None,
                                            norm_start_call=call(starts[c]) if c in starts else None,
                                            agree=(c in bounds and c in starts and bounds[c][0] == starts[c]))
                               for c in sorted(wanted)}
    return dict(norms_per_chunk=per_chunk, consistency=consistency, agreement=agreement,
                chunks={str(chunk): summarise(group) for chunk, group in sorted(chunks.items())})


def scan_log(text):
    lines = text.splitlines()
    return dict(flush_markers=[line.strip() for line in lines if FLUSH_MARKER in line][:20],
                flush_marker_count=sum(1 for line in lines if FLUSH_MARKER in line),
                dropped_or_full=[line.strip() for line in lines if DROPPED_RE.search(line)][:20])


def build_report(profile_dir, log_path=None, *, chunks=64, layers=64, gdn_layers=48,
                 wanted=(0, 1, 31, 32, 62, 63), first_full_layer=3):
    report = dict(scope='Prefill device attribution by op and by chunk; not a critical path or TTFT claim',
                  profile_dir=str(profile_dir), caveats=[])
    if log_path is not None and Path(log_path).is_file():
        report['log'] = scan_log(Path(log_path).read_text(encoding='utf-8', errors='replace'))
        if report['log']['flush_marker_count'] == 0:
            report['caveats'].append('No [PINDIAG] prefill profile flush marker in the log: the flush hook '
                                     'did not run, so per-core buffers may have overflowed mid-chunk.')
        if report['log']['dropped_or_full']:
            report['caveats'].append('The log reports dropped markers or full profiler buffers; the capture '
                                     'is incomplete regardless of the coverage ratios.')
    device_path = find_csv(profile_dir, 'cpp_device_perf_report.csv')
    if device_path is None:
        report['error'] = 'cpp_device_perf_report.csv %s' % NOT_EXPORTED
        return report
    rows, columns = load_rows(device_path)
    missing = [name for name in (OP_NAME, DEVICE_ID, DURATION) if name not in columns]
    if missing:
        report['error'] = '%s %s' % (' and '.join(missing), NOT_EXPORTED)
        return report
    prefill = untraced(rows) if TRACE_ID in columns else rows
    if TRACE_ID not in columns:
        report['caveats'].append('%s %s; decode rows could not be excluded.' % (TRACE_ID, NOT_EXPORTED))
    report.update(device_csv=str(device_path), rows_total=len(rows), rows_prefill=len(prefill),
                  per_op=per_op(prefill), busy=busy(prefill), sdpa=sdpa_calls(prefill))
    split = None
    if CALL_COUNT in columns:
        split = split_by_sdpa(prefill, chunks=chunks, layers=layers, gdn_layers=gdn_layers,
                              first_full_layer=first_full_layer)
        report['by_sdpa'] = group_by_sdpa(split, wanted=set(wanted))
        for chip, entry in sorted(split.items()):
            if not entry['consistency']['complete']:
                report['caveats'].append('chip %s: %d SDPAOperation calls do not make %d whole chunks of %d; '
                                         'the per-chunk split is not trustworthy.'
                                         % (chip, entry['consistency']['sdpa_calls'], chunks, layers - gdn_layers))
    else:
        report['by_sdpa'] = None
        report['caveats'].append('%s %s; no per-chunk split.' % (CALL_COUNT, NOT_EXPORTED))
    in_prompt = prompt_rows(split) if split and any(entry['bounds'] for entry in split.values()) else None
    if in_prompt is None:
        report['caveats'].append('coverage counted over every untraced row (no SDPA-anchored prompt span): '
                                 'warm-up prefills inflate it.')
    report['coverage'] = coverage(in_prompt if in_prompt is not None else prefill, chunks=chunks, layers=layers,
                                  gdn_layers=gdn_layers, all_rows=prefill)
    report['coverage_scope'] = 'prompt span (by_sdpa)' if in_prompt is not None else 'all untraced rows'
    tracy_path = find_csv(profile_dir, 'tracy_ops_data.csv')
    report['tracy_ops_data_csv'] = str(tracy_path) if tracy_path else None
    if tracy_path is None:
        report['signposted'] = None
        report['caveats'].append('tracy_ops_data.csv %s (expected: the arm exports no ops data); the per-chunk '
                                 'split is by_sdpa.' % NOT_EXPORTED)
    elif CALL_COUNT not in columns:
        report['signposted'] = None
        report['caveats'].append('%s %s; signposts cannot be joined to device rows.' % (CALL_COUNT, NOT_EXPORTED))
    else:
        events = parse_tracy_ops(tracy_path.read_text(encoding='utf-8', errors='replace'))
        labels = [value for _, kind, value in events if kind == 'signpost' and SIGNPOST_RE.match(value)]
        report['signposts_found'] = labels
        report['signposted'] = group_by_signpost(prefill, chunk_of_calls(events)) if labels else None
        if not labels:
            report['caveats'].append('tracy_ops_data.csv carries no qwen_prefill_chunk signpost.')
    report['by_norm'] = (group_by_norm(prefill, layers=layers, wanted=set(wanted), split=split)
                         if CALL_COUNT in columns else None)
    if report['by_norm']:
        disagree = sorted('%s:%s' % (chip, chunk) for chip, table in report['by_norm']['agreement'].items()
                          for chunk, cell in table.items() if not cell['agree'])
        if disagree:
            report['caveats'].append('by_sdpa and the LayerNormPreAllGather cross-check start these chunks at '
                                     'different calls (chip:chunk): %s' % ', '.join(disagree))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile-dir', required=True, type=Path)
    parser.add_argument('--log', type=Path, help='m3native-gate-stdout.log (flush markers, dropped lines)')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--chunks', type=int, default=64, help='prefill chunks expected (131072 / 2048)')
    parser.add_argument('--layers', type=int, default=64)
    parser.add_argument('--gdn-layers', type=int, default=48)
    options = parser.parse_args(argv)
    report = build_report(options.profile_dir, options.log, chunks=options.chunks, layers=options.layers,
                          gdn_layers=options.gdn_layers)
    text = json.dumps(report, indent=2)
    if options.out:
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(text, encoding='utf-8', newline='\n')
    summary = dict(rows_prefill=report.get('rows_prefill'), error=report.get('error'),
                   coverage={op: entry['ratio_per_chip'] for op, entry in (report.get('coverage') or {}).items()},
                   coverage_scope=report.get('coverage_scope'),
                   sdpa_chunks=sorted((report.get('by_sdpa') or {}).get('chunks', {}), key=int),
                   signposts=len(report.get('signposts_found') or []), caveats=report['caveats'])
    print('PREFILL_PROFILE ' + json.dumps(summary))
    return 1 if report.get('error') else 0


if __name__ == '__main__':
    sys.exit(main())
