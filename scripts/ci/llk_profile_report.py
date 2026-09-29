"""Read tt-metal's device profile (profile_log_device.csv) for the LLK zones and counters, and say, per kernel
and per thread, where the time went: LLK-bound or dataflow-bound, ranked. Attribution, never a throughput claim.

    python3 llk_profile_report.py report --csv <profile_log_device.csv[.gz]> --manifest <llk-manifest.json> \
        [--console <server.log>] [--twin <twin m3native-gate.json>] [--profiled <m3native-gate.json>] --out <json>
    python3 llk_profile_report.py export --csv <profile_log_device.csv> --out <export.csv.gz>

The CSV (tt-metal v0.77.0 impl/profiler/profiler.cpp writeCSVHeader / dumpDeviceResultsToCSV): line 1
'ARCH: blackhole, CHIP_FREQ[MHz]: N, Max Compute Cores: M', line 2 the column names, then one row per marker:
chip, core, RISC (BRISC, NCRISC, TRISC_0/1/2), timer id, cycle, data, run host id, trace id, trace id counter,
zone name, type (ZONE_START, ZONE_END, ZONE_TOTAL, TS_DATA, TS_EVENT, TS_DATA_16B), source line and file,
and meta data (JSON with ',' written as ';'). Only rows of zones named QWEN_LLK_* and counter rows (timer id
9090, perf_counters.hpp PERF_COUNTER_PROFILER_ID) are read; a multi-GB log streams through a line filter.

One kernel invocation on one thread is one execution key: (chip, core, RISC, run host id, trace id, trace id
counter). Zones pair START/END per key and name in time order; an END without a START, or a START left open,
is a DROP and is counted, never hidden. The wait sums (ZONE_TOTAL of QWEN_LLK_WAIT_IN / _OUT) belong to the
envelope zone of the same key - the kernel that ran there - whatever file they were compiled in.

Per thread (llk_zones' sync table): busy = envelope - WAIT_IN - WAIT_OUT. TRISC_0's WAIT_IN is the unpacker
starved by the reader; TRISC_2's WAIT_IN is the packer waiting on math (math->pack), its WAIT_OUT the writer's
back-pressure; TRISC_1's WAIT_OUT is math waiting on the packer to free dest. Math waiting on unpack
(unpack->math) is a hardware stall on srcA/srcB valid: it comes from the counter pass
(WAITING_FOR_SRCA_VALID/SRCB_VALID over the reference count) or it is left undetermined; the stage lockstep
(the unpack/math duration correlation the MLP diagnostic used) is shown beside it as evidence, not a verdict.

Stdlib only, Python 3.7 syntax.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys

import llk_kernels
import llk_zones

SCHEMA = 'qwen-llk-profile-report/1'
COLUMNS = ('PCIe slot', 'core_x', 'core_y', 'RISC processor type', 'timer_id', 'time[cycles since reset]', 'data',
           'run host ID', 'trace id', 'trace id counter', 'zone name', 'type', 'source line', 'source file',
           'meta data')
COUNTER_ID = 9090
COMPUTE_THREADS = ('TRISC_0', 'TRISC_1', 'TRISC_2')
MOVER_THREADS = ('BRISC', 'NCRISC')
RISCS = MOVER_THREADS + COMPUTE_THREADS
THREAD_ROLE = dict(TRISC_0='unpack', TRISC_1='math', TRISC_2='pack', BRISC='data movement 0',
                   NCRISC='data movement 1')
DROP_LINE = 'markers were dropped'
DEFAULT_MAX_BYTES = 64 * 2 ** 30
# Classification thresholds (heuristic, stated in every report).
DATAFLOW_FRACTION = 0.5
LLK_WAIT_FRACTION = 0.3
SRC_WAIT_FRACTION = 0.3
LOCKSTEP_R = 0.99
RECONFIG_HEAVY = 4
CAVEATS = (
    'Zone intervals on different threads overlap in time: per-thread fractions are not additive, and no stage '
    'time is a critical-path share.',
    'A wait sum includes the synchronisation call\'s own few cycles; a thread with no sum row waited zero cycles '
    '(tt-metal writes only non-zero sums).',
    'Math waiting on unpack is a hardware stall that no software zone sees: it is taken from the counter pass, '
    'or reported undetermined.',
    'Counters are per kernel invocation (started and stopped around the compute kernel), not per stage.',
    'Reconfiguration counts are static (same-file source, calls followed one level into same-file helpers) times '
    'the stage instances seen; helpers in included headers are not counted.',
    'Profiled rounds are perturbed by the markers and the mid-run dumps; the perturbation line compares with the '
    'unprofiled twin arm. Nothing here is a timing result.',
    'Classification thresholds: dataflow at %.0f%% of a thread\'s envelope waiting on dataflow, LLK waits at %.0f%%, '
    'srcA/srcB-valid stalls at %.0f%%, lockstep at r >= %.2f, reconfiguration-heavy at %d static calls per stage '
    'instance.' % (100 * DATAFLOW_FRACTION, 100 * LLK_WAIT_FRACTION, 100 * SRC_WAIT_FRACTION, LOCKSTEP_R,
                   RECONFIG_HEAVY),
)


class ReportError(ValueError):
    """A device log or manifest this report refuses, with the reason."""


# ---- reading ----

def _open(path):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace', newline='')
    return open(path, 'r', encoding='utf-8', errors='replace', newline='')


def read_preamble(stream):
    """{arch, chip_freq_mhz, max_compute_cores} from line 1, the column names checked from line 2."""
    first = stream.readline()
    if not first.startswith('ARCH: '):
        raise ReportError('not a tt-metal device log: line 1 must start "ARCH: ", got %r' % first[:60])
    fields = {}
    for part in first.strip().split(','):
        name, _, value = part.partition(':')
        fields[name.strip()] = value.strip()
    arch = fields.get('ARCH')
    if arch != 'blackhole':
        raise ReportError('a Blackhole device log is required, got ARCH %r' % arch)
    try:
        freq = int(fields.get('CHIP_FREQ[MHz]', ''))
    except ValueError:
        raise ReportError('line 1 carries no CHIP_FREQ[MHz]')
    header = [name.strip() for name in stream.readline().rstrip('\r\n').split(',')]
    if tuple(header) != COLUMNS:
        raise ReportError('the column names are not v0.77.0\'s (%s), got %s' % (', '.join(COLUMNS), header))
    cores = fields.get('Max Compute Cores')
    return dict(arch=arch, chip_freq_mhz=freq, max_compute_cores=int(cores) if cores and cores.isdigit() else None)


def _wanted(line):
    return 'QWEN_LLK_' in line or ',9090,' in line


def _int(text, what):
    try:
        return int(text)
    except ValueError:
        raise ReportError('%s is not an integer: %r' % (what, text))


def parse_row(line):
    """One marker row as a dict, or ReportError."""
    fields = next(csv.reader([line]))
    if len(fields) != len(COLUMNS):
        raise ReportError('a row has %d fields, not %d: %r' % (len(fields), len(COLUMNS), line[:120]))
    risc = fields[3].strip()
    row = dict(chip=_int(fields[0], 'PCIe slot'), core=(_int(fields[1], 'core_x'), _int(fields[2], 'core_y')),
               risc=risc, timer=_int(fields[4], 'timer_id'), cycle=_int(fields[5], 'time'),
               data=_int(fields[6], 'data'), host=_int(fields[7], 'run host ID'),
               trace=_int(fields[8], 'trace id') if fields[8].strip() else None,
               replay=_int(fields[9], 'trace id counter') if fields[9].strip() else None,
               zone=fields[10].strip(), phase=fields[11].strip(), meta=fields[14].strip())
    return row


def check_size(path, max_bytes):
    size = os.path.getsize(path)
    if max_bytes is not None and size > max_bytes:
        raise ReportError('%s is %d bytes, past the %d this report reads: refused before parsing (export a filtered '
                          'copy on the rig first)' % (path, size, max_bytes))
    return size


def read_rows(path, max_bytes=DEFAULT_MAX_BYTES):
    """(preamble, [row]) of the QWEN_LLK_* and counter rows of a device log or of its filtered export."""
    check_size(path, max_bytes)
    rows = []
    with _open(path) as stream:
        preamble = read_preamble(stream)
        for line in stream:
            if not _wanted(line):
                continue
            row = parse_row(line.rstrip('\r\n'))
            if row['zone'].startswith(llk_zones.PREFIX) or row['timer'] == COUNTER_ID:
                if row['risc'] not in RISCS:
                    raise ReportError('RISC %r is not a Tensix worker thread (%s)' % (row['risc'], ', '.join(RISCS)))
                rows.append(row)
    return preamble, rows


def export_filtered(path, out_path, max_bytes=DEFAULT_MAX_BYTES):
    """Write the two header lines and every QWEN_LLK_* and counter row of `path` to `out_path` (gzip), streaming;
    {bytes, sha256} of the whole log and {rows, sha256} of the export."""
    check_size(path, max_bytes)
    digest = hashlib.sha256()
    kept = 0
    size = 0
    with open(path, 'rb') as source, gzip.open(out_path, 'wb') as sink:
        for number, raw in enumerate(source):
            digest.update(raw)
            size += len(raw)
            if number < 2:
                if number == 0 and not raw.startswith(b'ARCH: '):
                    raise ReportError('not a tt-metal device log: %s' % path)
                sink.write(raw)
                continue
            if b'QWEN_LLK_' in raw or b',9090,' in raw:
                sink.write(raw)
                kept += 1
    with open(out_path, 'rb') as handle:
        exported = hashlib.sha256(handle.read()).hexdigest()
    return dict(source=dict(path=os.path.basename(path), bytes=size, sha256=digest.hexdigest()),
                export=dict(path=os.path.basename(out_path), rows=kept, sha256=exported))


# ---- pairing ----

def execution(row):
    return (row['chip'], row['core'][0], row['core'][1], row['risc'], row['host'], row['trace'], row['replay'])


def pair(rows):
    """(intervals, totals, counters, drops). intervals: [dict(key, zone, start, end)]; totals:
    {(key, zone): cycles}; counters: [dict(key5, risc, type, value, ref)]."""
    events = {}
    totals = {}
    counters = []
    for row in rows:
        if row['timer'] == COUNTER_ID and not row['zone'].startswith(llk_zones.PREFIX):
            counters.append(counter_row(row))
            continue
        key = execution(row)
        if row['phase'] in ('ZONE_START', 'ZONE_END'):
            events.setdefault((key, row['zone']), []).append((row['cycle'], 0 if row['phase'] == 'ZONE_START' else 1))
        elif row['phase'] == 'ZONE_TOTAL':
            totals[(key, row['zone'])] = totals.get((key, row['zone']), 0) + row['data']
        else:
            raise ReportError('zone %s carries a %s row: QWEN_LLK zones are scopes and sums only'
                              % (row['zone'], row['phase']))
    intervals = []
    drops = dict(unmatched_end=0, unmatched_start=0)
    for (key, zone), marks in sorted(events.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        stack = []
        for cycle, end in sorted(marks):
            if not end:
                stack.append(cycle)
            elif stack:
                start = stack.pop()
                intervals.append(dict(key=key, zone=zone, start=start, end=cycle))
            else:
                drops['unmatched_end'] += 1
        drops['unmatched_start'] += len(stack)
    return intervals, totals, counters, drops


def counter_row(row):
    meta = row['meta']
    try:
        fields = json.loads(meta.replace(';', ',')) if meta else {}
    except ValueError:
        raise ReportError('a counter row\'s meta data is not JSON: %r' % meta[:120])
    kind = fields.get('counter type')
    if not isinstance(kind, str) or not isinstance(fields.get('value'), int) or not isinstance(fields.get('ref cnt'), int):
        raise ReportError('a counter row needs "counter type", "value" and "ref cnt": %r' % meta[:120])
    key = execution(row)
    return dict(key5=key[:3] + key[4:], risc=row['risc'], type=kind, value=fields['value'], ref=fields['ref cnt'])


# ---- statistics (3.7: no statistics.fmean / quantiles) ----

def summary(values):
    if not values:
        return None
    ordered = sorted(values)

    def rank(fraction):
        return ordered[min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))]
    return dict(count=len(values), total=sum(values), mean=round(sum(values) / float(len(values)), 1),
                p50=rank(0.5), p90=rank(0.9), max=ordered[-1])


def pearson(xs, ys):
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = sum(xs) / float(len(xs)), sum(ys) / float(len(ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 6)


def fraction(part, whole):
    if part is None or not whole:
        return None
    return round(float(part) / whole, 4)


# ---- analysis ----

def manifest_records(manifest):
    """Every instrumentation record of an llk-manifest.json (the gate's) - or a bare list of records."""
    if isinstance(manifest, list):
        return manifest
    records = list(manifest.get('generated') or []) + list(manifest.get('files') or [])
    if not records:
        raise ReportError('the manifest names no instrumented kernel')
    return records


def counter_metrics(rows):
    """Ratios (sum of value / sum of reference count) per counter type, and the derived metrics."""
    sums = {}
    banks = set()
    for row in rows:
        value, ref = sums.get(row['type'], (0, 0))
        sums[row['type']] = (value + row['value'], ref + row['ref'])
        if row['type'].startswith('L1_') and row['type'][3:4].isdigit():
            banks.add(row['type'][:4])
    if len(banks) > 1:
        raise ReportError('counters of L1 banks %s in one pass: tt-metal counts one L1 bank per run'
                          % ', '.join(sorted(banks)))
    ratios = dict((kind, round(value / float(ref), 4) if ref else None) for kind, (value, ref) in sums.items())

    def ratio(*names):
        found = [ratios[name] for name in names if ratios.get(name) is not None]
        return max(found) if found else None
    derived = dict(fpu_util=ratio('FPU_COUNTER'), sfpu_util=ratio('SFPU_COUNTER'), math_util=ratio('MATH_COUNTER'),
                   unpack_busy=ratio('UNPACK0_BUSY_THREAD0', 'UNPACK1_BUSY_THREAD0'),
                   pack_busy=ratio('PACKER_BUSY'),
                   math_waiting_on_unpack=ratio('WAITING_FOR_SRCA_VALID', 'WAITING_FOR_SRCB_VALID'))
    return dict(rows=len(rows), ratios=ratios, derived=derived, l1_bank=sorted(banks)[0] if banks else None)


def classify(threads, counters, stages):
    """(bound, llk_bound, reason) for one kernel from its per-thread fractions (and counters)."""
    compute = [risc for risc in COMPUTE_THREADS if risc in threads]
    derived = (counters or {}).get('derived') or {}
    if not compute:
        movers = [threads[risc] for risc in MOVER_THREADS if risc in threads]
        worst_in = max([t['wait_in_fraction'] or 0 for t in movers] or [0])
        worst_out = max([t['wait_out_fraction'] or 0 for t in movers] or [0])
        if worst_in >= DATAFLOW_FRACTION:
            return 'dataflow-read', False, 'a data-movement thread waits on reads or its producer for %.0f%%' % (100 * worst_in)
        if worst_out >= DATAFLOW_FRACTION:
            return 'dataflow-write', False, 'a data-movement thread waits on writes or ring space for %.0f%%' % (100 * worst_out)
        return 'data-movement', False, 'a data-movement kernel: no LLK thread'
    unpack, maths, pack = (threads.get(risc) or {} for risc in COMPUTE_THREADS)
    starved = unpack.get('wait_in_fraction')
    backpressure = pack.get('wait_out_fraction')
    pack_on_math = pack.get('wait_in_fraction')
    math_on_pack = maths.get('wait_out_fraction')
    src_wait = derived.get('math_waiting_on_unpack')
    if None in (starved, backpressure, pack_on_math, math_on_pack):
        if src_wait is not None and src_wait >= SRC_WAIT_FRACTION:
            return 'llk-unpack', True, 'no wait sums; the counters show math stalled on srcA/srcB valid %.0f%%' % (100 * src_wait)
        return 'undetermined', False, 'no wait sums for every compute thread (level tag, sums unsupported or dropped)'
    if starved >= DATAFLOW_FRACTION:
        return 'dataflow-in', False, 'unpack waits on the reader for %.0f%% of its envelope' % (100 * starved)
    if backpressure >= DATAFLOW_FRACTION:
        return 'dataflow-out', False, 'pack waits on the writer for %.0f%% of its envelope' % (100 * backpressure)
    if math_on_pack >= LLK_WAIT_FRACTION and math_on_pack >= pack_on_math:
        return 'llk-pack', True, 'math waits on the packer to free dest for %.0f%%' % (100 * math_on_pack)
    if pack_on_math >= LLK_WAIT_FRACTION:
        if src_wait is not None and src_wait >= SRC_WAIT_FRACTION:
            return 'llk-unpack', True, ('pack waits on math %.0f%%, and math stalls on srcA/srcB valid %.0f%% (counters)'
                                        % (100 * pack_on_math, 100 * src_wait))
        if src_wait is not None:
            sfpu, fpu = derived.get('sfpu_util') or 0, derived.get('fpu_util') or 0
            return ('llk-math-sfpu' if sfpu > fpu else 'llk-math'), True, (
                'pack waits on math %.0f%%; math is not starved of operands (%.0f%% srcA/srcB-valid stalls)'
                % (100 * pack_on_math, 100 * src_wait))
        lockstep = [stage['name'] for stage in stages if (stage.get('lockstep') or {}).get('lockstep')]
        return 'llk-unpack-or-math', True, (
            'pack waits on math %.0f%%; whether math itself waits on unpack needs the counter pass%s'
            % (100 * pack_on_math, ' (unpack/math lockstep in %s)' % ', '.join(lockstep) if lockstep else ''))
    return 'llk-balanced', True, 'no thread waits on another or on dataflow for %.0f%% or more' % (100 * LLK_WAIT_FRACTION)


def lockstep(instances):
    """Unpack/math evidence over one stage's paired instances: [(d0, e0, d1, e1)]."""
    if len(instances) < 3:
        return None
    unpack = [d0 for d0, _, _, _ in instances]
    maths = [d1 for _, _, d1, _ in instances]
    r = pearson(unpack, maths)
    lag = sum(e1 - e0 for _, e0, _, e1 in instances) / float(len(instances))
    mean_math = sum(maths) / float(len(maths))
    close = mean_math > 0 and abs(sum(d1 - d0 for d0, _, d1, _ in instances) / float(len(instances))) <= 0.05 * mean_math
    return dict(r_unpack_math=r, mean_math_end_after_unpack_end=round(lag, 1),
                lockstep=bool(r is not None and r >= LOCKSTEP_R and close))


def analyse(rows, records, console='', twin=None, profiled=None, preamble=None):
    """The report dict for paired rows of one arm."""
    index = llk_kernels.zone_index(records)
    intervals, totals, counters, drops = pair(rows)
    drops['console'] = DROP_LINE in (console or '').lower()
    unknown = sorted(set(item['zone'] for item in intervals) - set(index))
    envelopes = {}
    stage_rows = {}
    for item in intervals:
        meta = index.get(item['zone'])
        if meta is None:
            continue
        duration = item['end'] - item['start']
        if meta['kind'] == 'envelope':
            envelopes.setdefault(meta['kernel'], {})[item['key']] = (item['start'], item['end'])
        else:
            stage_rows.setdefault((meta['kernel'], meta['stage'], item['zone']), []).append(dict(item, duration=duration))
    owner = {}
    for kernel, keys in envelopes.items():
        for key in keys:
            owner[key] = kernel
    sums_enabled = dict((record.get('key') or record.get('kernel'), bool((record.get('sync') or {}).get('enabled')))
                        for record in records)
    kernels = []
    unattributed = dict(sums=0, counters=0)
    per_kernel_totals = {}
    for (key, zone), cycles in totals.items():
        kernel = owner.get(key)
        if kernel is None:
            unattributed['sums'] += 1
            continue
        slot = 'wait_in' if zone == llk_zones.WAIT_IN else 'wait_out' if zone == llk_zones.WAIT_OUT else None
        if slot is None:
            continue
        per_kernel_totals.setdefault(kernel, {}).setdefault(key, {})[slot] = cycles
    counter_rows = {}
    owner5 = dict((key[:3] + key[4:], kernel) for key, kernel in owner.items())
    for row in counters:
        kernel = owner5.get(row['key5'])
        if kernel is None:
            unattributed['counters'] += 1
            continue
        counter_rows.setdefault(kernel, []).append(row)
    for kernel in sorted(envelopes):
        keys = envelopes[kernel]
        waits = per_kernel_totals.get(kernel, {})
        record_sums = sums_enabled.get(kernel, False) or bool(waits)
        threads = {}
        for risc in RISCS:
            mine = [(key, end - start) for key, (start, end) in keys.items() if key[3] == risc]
            if not mine:
                continue
            envelope_total = sum(duration for _, duration in mine)
            if record_sums:
                wait_in = sum((waits.get(key) or {}).get('wait_in', 0) for key, _ in mine)
                wait_out = sum((waits.get(key) or {}).get('wait_out', 0) for key, _ in mine)
            else:
                wait_in = wait_out = None
            busy = None if wait_in is None else max(0, envelope_total - wait_in - wait_out)
            threads[risc] = dict(role=THREAD_ROLE[risc], envelope=summary([duration for _, duration in mine]),
                                 wait_in=wait_in, wait_out=wait_out, busy=busy,
                                 wait_in_fraction=fraction(wait_in, envelope_total),
                                 wait_out_fraction=fraction(wait_out, envelope_total),
                                 busy_fraction=fraction(busy, envelope_total))
        stages = []
        for (owner_kernel, stage, zone), items in sorted(stage_rows.items()):
            if owner_kernel != kernel:
                continue
            meta = index[zone]
            by_risc = {}
            for item in items:
                by_risc.setdefault(item['key'][3], []).append(item)
            per_thread = dict((risc, summary([item['duration'] for item in by_risc[risc]])) for risc in sorted(by_risc))
            paired = []
            ordinal = {}
            for risc in ('TRISC_0', 'TRISC_1'):
                for item in sorted(by_risc.get(risc) or [], key=lambda entry: (entry['key'], entry['start'])):
                    base = item['key'][:3] + item['key'][4:]
                    count = ordinal.get((risc, base), 0)
                    ordinal[(risc, base)] = count + 1
                    item['ordinal'] = (base, count)
            unpack_items = dict((item['ordinal'], item) for item in by_risc.get('TRISC_0') or [])
            for item in by_risc.get('TRISC_1') or []:
                match = unpack_items.get(item['ordinal'])
                if match is not None:
                    paired.append((match['duration'], match['end'], item['duration'], item['end']))
            instances = max([len(value) for value in by_risc.values()] or [0])
            static = meta['reconfig_static']
            stages.append(dict(name=stage, instances_per_thread=dict((risc, len(value)) for risc, value in by_risc.items()),
                               threads=per_thread, lockstep=lockstep(paired), reconfig_static=static,
                               reconfig_total=None if static is None else static * instances,
                               reconfig_heavy=bool(static is not None and static >= RECONFIG_HEAVY)))
        metrics = counter_metrics(counter_rows[kernel]) if kernel in counter_rows else None
        bound, llk_bound, reason = classify(threads, metrics, stages)
        critical = {}
        for key, (start, end) in keys.items():
            base = key[:3] + key[4:]
            critical[base] = max(critical.get(base, 0), end - start)
        chips = sorted(set(key[0] for key in keys))
        attribution = dict(
            unpack_starved_by_reader=(threads.get('TRISC_0') or {}).get('wait_in_fraction'),
            math_blocked_by_pack=(threads.get('TRISC_1') or {}).get('wait_out_fraction'),
            pack_waiting_on_math=(threads.get('TRISC_2') or {}).get('wait_in_fraction'),
            pack_blocked_by_writer=(threads.get('TRISC_2') or {}).get('wait_out_fraction'),
            math_waiting_on_unpack=((metrics or {}).get('derived') or {}).get('math_waiting_on_unpack'),
            math_waiting_on_unpack_source='counters' if metrics and metrics['derived'].get('math_waiting_on_unpack')
            is not None else 'undetermined')
        kernels.append(dict(kernel=kernel, chips=chips, cores=len(set(key[1:3] + (key[0],) for key in keys)),
                            invocations=len(critical), critical_cycles=sum(critical.values()),
                            threads=threads, stages=stages, attribution=attribution, counters=metrics,
                            bound=bound, llk_bound=llk_bound, reason=reason,
                            reconfig_heavy_stages=[stage['name'] for stage in stages if stage['reconfig_heavy']]))
    whole = sum(kernel['critical_cycles'] for kernel in kernels) or 1
    for kernel in kernels:
        kernel['share_of_instrumented'] = round(kernel['critical_cycles'] / float(whole), 4)
    ranked = sorted(kernels, key=lambda kernel: -kernel['critical_cycles'])
    ranking = [dict(rank=rank + 1, kernel=kernel['kernel'], share_of_instrumented=kernel['share_of_instrumented'],
                    bound=kernel['bound'], llk_bound=kernel['llk_bound'], reason=kernel['reason'])
               for rank, kernel in enumerate(ranked)]
    report = dict(schema=SCHEMA, device=preamble, kernels=ranked, ranking=ranking, drops=drops,
                  unknown_zones=unknown, unattributed=unattributed, counters_seen=len(counters),
                  caveats=list(CAVEATS), perturbation=perturbation(twin, profiled),
                  complete=not (drops['unmatched_end'] or drops['unmatched_start'] or drops['console']))
    return report


def perturbation(twin, profiled):
    """The profiled arm's packed round trace time (and first-token time) against its unprofiled twin's."""
    if not twin or not profiled:
        return None

    def trace(report):
        return ((report or {}).get('packed_phase') or {}).get('trace_ms_mean')

    def ttft(report):
        values = [stream.get('ttft_s') for stream in (report or {}).get('streams') or []
                  if isinstance((stream or {}).get('ttft_s'), (int, float))]
        return max(values) if values else None
    result = dict(twin_trace_ms_mean=trace(twin), profiled_trace_ms_mean=trace(profiled),
                  twin_ttft_s=ttft(twin), profiled_ttft_s=ttft(profiled))
    if result['twin_trace_ms_mean'] and result['profiled_trace_ms_mean']:
        result['trace_ratio'] = round(result['profiled_trace_ms_mean'] / result['twin_trace_ms_mean'], 4)
    if result['twin_ttft_s'] and result['profiled_ttft_s']:
        result['ttft_ratio'] = round(result['profiled_ttft_s'] / result['twin_ttft_s'], 4)
    return result


def coverage(report, required, chips=(0, 1)):
    """What a required kernel lacks in a report: [problem]. A compute kernel needs its envelope on all three
    TRISCs on every chip; the others on some thread of every chip."""
    problems = []
    by_kernel = dict((kernel['kernel'], kernel) for kernel in report.get('kernels') or [])
    for key in required:
        kernel = by_kernel.get(key)
        if kernel is None:
            problems.append('%s: no QWEN_LLK_%s zone at all (not executed, not instrumented, or its markers were '
                            'dropped)' % (key, key))
            continue
        missing_chips = sorted(set(chips) - set(kernel['chips']))
        if missing_chips:
            problems.append('%s: no zone on chip %s' % (key, ', '.join(str(chip) for chip in missing_chips)))
        entry = llk_kernels.BY_KEY.get(key) or {}
        if entry.get('part', 'compute') == 'compute' and not entry.get('header'):
            missing = [risc for risc in COMPUTE_THREADS if risc not in kernel['threads']]
            if missing:
                problems.append('%s: no envelope on %s' % (key, ', '.join(missing)))
    return problems


def render(report):
    """A short human summary, one line per ranked kernel."""
    lines = ['LLK profile: %d kernels, complete=%s, drops %s' % (len(report['kernels']), report['complete'],
                                                                 report['drops'])]
    for rank in report['ranking']:
        lines.append('  #%(rank)d %(kernel)s %(share_of_instrumented).1f%% %(bound)s - %(reason)s' % dict(
            rank, share_of_instrumented=100 * rank['share_of_instrumented']))
    if report.get('perturbation'):
        lines.append('  perturbation vs twin: %s' % json.dumps(report['perturbation'], sort_keys=True))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command')
    report_parser = commands.add_parser('report')
    report_parser.add_argument('--csv', required=True)
    report_parser.add_argument('--manifest', required=True)
    report_parser.add_argument('--console', default=None)
    report_parser.add_argument('--twin', default=None)
    report_parser.add_argument('--profiled', default=None)
    report_parser.add_argument('--out', required=True)
    report_parser.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES)
    export_parser = commands.add_parser('export')
    export_parser.add_argument('--csv', required=True)
    export_parser.add_argument('--out', required=True)
    export_parser.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES)
    options = parser.parse_args(argv)
    try:
        if options.command == 'export':
            print(json.dumps(export_filtered(options.csv, options.out, options.max_bytes), indent=1))
            return 0
        if options.command != 'report':
            parser.print_help()
            return 2
        with open(options.manifest, encoding='utf-8') as handle:
            records = manifest_records(json.load(handle))
        console = ''
        if options.console:
            with open(options.console, encoding='utf-8', errors='replace') as handle:
                console = handle.read()
        loaded = []
        for path in (options.twin, options.profiled):
            if path:
                with open(path, encoding='utf-8') as handle:
                    loaded.append(json.load(handle))
            else:
                loaded.append(None)
        preamble, rows = read_rows(options.csv, options.max_bytes)
        report = analyse(rows, records, console, loaded[0], loaded[1], preamble)
        with open(options.out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
        print(render(report))
    except (ReportError, llk_zones.ZoneError, OSError, ValueError) as error:
        sys.stderr.write('llk profile report refused: %s\n' % error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
