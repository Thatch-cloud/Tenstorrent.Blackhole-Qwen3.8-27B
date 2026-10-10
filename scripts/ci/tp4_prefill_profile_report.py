"""The offline analysis of a TP4 PREFILL op-level device profile (ops_profile_plan's ops-prefill-trace arm).

    python3 scripts/ci/tp4_prefill_profile_report.py --results <the downloaded gate/ directory> [--out <dir>]
    python3 scripts/ci/tp4_prefill_profile_report.py <cpp_device_perf_report[.prefill].csv[.gz]> [--server-log L]
            [--prompt-tokens 131072] [--chips 4] [--peak-tflops 774] [--out <dir>]

--results reads the artifact's layout: ops/cpp_device_perf_report.prefill.csv.gz (the gate compresses tracy's CPP report there),
ops-prefill-trace/server.log and m3native-gate.json (and the same two files under ops-prefill-twin/). It writes
prefill-profile-report.json and prefill-profile-report.md beside the CSV (or into --out) and prints the markdown. The exit status is
1 when the validity block has a PROBLEM (notes do not count). docs/tp4-profile.md ('Prefill profile') reads it.

WHAT IT READS. The arm prefilled ONE real-text prompt (131,072 tokens by default: 64 chunks of the model's own 2,048-token outer
chunk) with QWEN_PREFILL_PROFILE_FLUSH=1, which drains the device profiler every 16 decoder layers, so a chunk cannot overflow the
per-core buffers. Prefill runs UNTRACED (decode replays carry a METAL TRACE ID and are dropped here). Per chip, in GLOBAL CALL COUNT
order (else kernel start cycle, else file order), the report finds the chunks like lever_n_prefill_profile_report does: the SDPA op
runs once per attention layer per chunk (16 of 64 layers; the layer pattern is three GDN layers then one attention layer), whole groups of
16 counted from the END are the profiled prompt (earlier groups are warm-up prefills), and a chunk starts at the 7th
LayerNormPreAllGather before its first SDPA (two per GDN layer 0-2 plus layer 3's attention norm). Inside a chunk the two
LayerNormPreAllGather per layer cut the 64 layers into a mixer half and an MLP half, a layer is attention when its mixer holds an SDPA
op and GDN otherwise, and what follows the last MLP (a final norm, lm_head, eager decode glue) is NOT part of the chunk.

WHAT IT REPORTS (every figure measured from the CSV except the peak constant, which is an ESTIMATE):
  * per-op ms per prefill chunk, for chunk 0 (the shortest context), chunk 1, the chunks at about 1/4, 1/2 and 3/4 and the last one:
    calls and ms per op, the median over the chips and the maximum over the chips;
  * a category table for every chunk against its context: weight matmuls, attention SDPA prefill, the GDN chunked prefill (conv
    calls and scan ops), glue/layout copies (tilize, untilize, typecast, transpose, permute, reshape, slice, concat, copy, pad,
    reshard), norms, elementwise, collectives and other (with its top ops, so nothing is hidden), and the unclassified share;
  * attention prefill against context: the SDPA ms per chunk fitted to a + b x (context at the chunk's start, in 1,000 tokens), per
    attention layer too (a is the fixed cost of a chunk at context 0, b the slope per 1,000 tokens of context);
  * the GDN chunked prefill: conv calls per chunk and per GDN layer, scan ops per chunk, and their ms;
  * collectives per chunk: calls, ms per call (median chip), the minimum over the chips (the intrinsic time) and the skew (what the
    slowest chip made the others wait), per kind;
  * compute efficiency of the weight matmuls: achieved TFLOP/s per chip and the percentage of the stated peak, per weight (in/out
    projection of GDN and attention layers, gate, up, down). FLOPs are 2 x M x K x N with M = 2,048 rows and (K, N) per chip from
    the CSV's input-shape columns when they exist, else from tp4_profile_report.weight_table (the model's geometry divided by the chip
    count: an ASSUMPTION, printed as such). The peak is 774 TFLOP/s at LoFi (140 Tensix cores x 1.35 GHz x 2 x 2,048 MAC per cycle), divided by
    1, 2, 3 or 4 for LoFi/HiFi2/HiFi3/HiFi4 as the MATH FIDELITY column says (HiFi2 assumed where the column is missing): an
    ESTIMATE from the vendor sheet's arithmetic, not a measurement, set with --peak-tflops. A fused all-gather + matmul kernel's time includes
    its gather;
  * a validity block: chips found, chunks found against the prompt's chunk count, per-chip agreement on every chunk's op count, the
    layer structure (64 layers, 48 GDN, 16 attention), the flush markers and dropped-marker lines of the server log, the share of
    kernel time no rule classified.

It has NEVER seen a real TP4 prefill CSV: the classification is by substring rules, every absent column or op is a NOTE (never a
traceback), and a structure it cannot find is a validity PROBLEM that says what was missing. Kernel-duration sums are not a critical
path; the span (first kernel start to last kernel end, in the clock the CSV implies) is reported beside the sum.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import bisect
import collections
import json
import os
import re
import statistics
import sys

import tp4_profile_report as base

CHUNK = 2048
LAYERS = 64
GDN_LAYERS = 48
ATTN_LAYERS = 16
FIRST_ATTN_LAYER = 3                 # layers 0-2 are GDN, 3 is attention, and so on: attention at layer % 4 == 3
DEFAULT_PROMPT_TOKENS = 131072
DEFAULT_CHIPS = 4
NORM_PRE = 'LayerNormPreAllGather'
FLUSH_MARKER = '[PINDIAG] prefill profile flush'
MIN_FLUSH_MARKERS = 7
TOP_OPS = 25
UNCLASSIFIED_NOTE_SHARE = 0.10
ROW_AGREEMENT_PROBLEM = 0.01        # chips whose op count for a chunk differ by more than this fraction: a problem
LOFI_PEAK_TFLOPS = 774.0            # 140 cores x 1.35 GHz x 2 x 2,048 MAC/cycle: the vendor sheet's FP8 figure, an ESTIMATE
FIDELITY_DIVISOR = collections.OrderedDict([('LoFi', 1), ('HiFi2', 2), ('HiFi3', 3), ('HiFi4', 4)])
ASSUMED_FIDELITY = 'HiFi2'

OP_NAME = 'OP NAME'
DEVICE_ID = 'DEVICE ID'
DURATION = 'DEVICE KERNEL DURATION [ns]'
TRACE_ID = 'METAL TRACE ID'
CALL_COUNT = 'GLOBAL CALL COUNT'
KERNEL_START = 'DEVICE KERNEL START CYCLE'
KERNEL_END = 'DEVICE KERNEL END CYCLE'
CORE_COUNT = 'CORE COUNT'
FIDELITY = 'MATH FIDELITY'
REQUIRED = (OP_NAME, DEVICE_ID, DURATION)
INPUT_DIM = re.compile(r'^INPUT_([01])_([WZYX])(?:_PAD\[LOGICAL\])?$')

CATEGORIES = ('matmul', 'attn.sdpa', 'gdn.conv', 'gdn.scan', 'glue', 'norm', 'eltwise', 'collective', 'embedding', 'other')
CATEGORY_TITLES = collections.OrderedDict([
    ('matmul', 'weight matmuls'), ('attn.sdpa', 'attention SDPA'), ('gdn.conv', 'GDN conv'), ('gdn.scan', 'GDN scan/other'),
    ('glue', 'glue/layout copies'), ('norm', 'norms'), ('eltwise', 'elementwise'), ('collective', 'collectives'),
    ('embedding', 'embedding'), ('other', 'other (unclassified)')])
SDPA_TOKENS = ('SDPA', 'ScaledDotProductAttention', 'Sdpa')
GDN_TOKENS = ('Gdn', 'GatedDelta', 'DeltaRule', 'Recurr', 'Scan', 'LinearAttention')
GLUE_TOKENS = ('Tilize', 'Untilize', 'Typecast', 'Transpose', 'Permute', 'Reshape', 'Slice', 'Concat', 'Copy', 'Pad', 'Reshard',
               'InterleavedToSharded', 'ShardedToInterleaved', 'Clone', 'Move', 'Repeat', 'Fill', 'View', 'Gather', 'Scatter')
ELTWISE_TOKENS = ('Binary', 'Unary', 'Ternary', 'Where', 'Eltwise', 'Silu', 'Gelu', 'Sigmoid', 'Softmax', 'Rotary', 'Mul', 'Add')
# The weight matmul keys (position in the layer) and the tp4_profile_report.weight_table rows they take their (K, N) from.
TABLE_KEYS = {'gdn_in': 'mm.gdn_in', 'gdn_out': 'mm.gdn_out', 'attn_in': 'mm.attn_in', 'attn_out': 'mm.attn_out',
              'mlp_gate': 'mm.mlp.gate', 'mlp_up': 'mm.mlp.up', 'mlp_down': 'mm.mlp.down'}


class ReportError(ValueError):
    """An input the analysis cannot read."""


Row = collections.namedtuple('Row', 'dev op cat ns order cores flops fid start end')


def classify_op(name):
    """The category of one device op by its name (substring rules; the name is the CSV's, e.g. 'MatmulDeviceOperation')."""
    if name.startswith('LayerNorm') or ('Norm' in name and 'Gdn' not in name and 'Conv' not in name):
        return 'norm'
    if 'Matmul' in name:
        return 'matmul'
    if any(token in name for token in SDPA_TOKENS) and 'Decode' not in name:
        return 'attn.sdpa'
    if 'Conv' in name:
        return 'gdn.conv'
    if any(token in name for token in GDN_TOKENS):
        return 'gdn.scan'
    if base.is_coll(name):
        return 'collective'
    if any(token in name for token in GLUE_TOKENS):
        return 'glue'
    if 'Embedding' in name:
        return 'embedding'
    if any(token in name for token in ELTWISE_TOKENS):
        return 'eltwise'
    return 'other'


def _dim(value):
    """A shape dimension from 'N' or 'N[logical]' (the logical size when the padded one is shown first), or None."""
    text = (value or '').strip()
    if not text:
        return None
    match = re.search(r'\[(\d+)\]', text)
    text = match.group(1) if match else text
    try:
        return int(float(text))
    except ValueError:
        return None


def shape_columns(columns):
    """{(input, axis): column name} of the CSV's input-shape columns (INPUT_0_W_PAD[LOGICAL] ... INPUT_1_X_PAD[LOGICAL])."""
    found = {}
    for name in columns:
        match = INPUT_DIM.match(name)
        if match:
            found[(int(match.group(1)), match.group(2))] = name
    return found


def matmul_flops(row, shapes):
    """2 x M x K x N of a matmul row from its input shapes (M = W x Z x Y of input 0, K = its X, N = input 1's X, input 1's Y = K), or None."""
    if not shapes:
        return None
    dims = {}
    for (index, axis), column in shapes.items():
        dims[(index, axis)] = _dim(row.get(column))
    m = 1
    for axis in 'WZY':
        value = dims.get((0, axis))
        if value is None:
            return None
        m *= value
    k, n, k1 = dims.get((0, 'X')), dims.get((1, 'X')), dims.get((1, 'Y'))
    if not k or not n or k1 != k:
        return None
    return 2.0 * m * k * n


def load(path):
    """(rows by chip, facts): every untraced row as a Row, ordered per chip by GLOBAL CALL COUNT (else kernel start cycle, else file
    order). facts: the columns seen, traced and untraced row counts, rows without a duration."""
    import csv
    by_chip = collections.defaultdict(list)
    facts = dict(columns=[], rows=0, traced=0, no_duration=0, shapes=False, order_by=None, fidelity=False)
    with base.open_text(path) as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        facts['columns'] = columns
        missing = [name for name in REQUIRED if name not in columns]
        if missing:
            raise ReportError('%s lacks the column(s) %s: not a cpp_device_perf_report.csv' % (path, ', '.join(missing)))
        shapes = shape_columns(columns)
        complete = all((index, axis) in shapes for index in (0, 1) for axis in 'WZYX')
        facts['shapes'] = complete
        facts['fidelity'] = FIDELITY in columns
        facts['order_by'] = CALL_COUNT if CALL_COUNT in columns else (KERNEL_START if KERNEL_START in columns else 'file order')
        for position, row in enumerate(reader):
            facts['rows'] += 1
            if TRACE_ID in columns and (row.get(TRACE_ID) or '').strip():
                facts['traced'] += 1
                continue
            duration = base.number(row, DURATION)
            if duration is None:
                facts['no_duration'] += 1
                duration = 0.0
            name = (row.get(OP_NAME) or '<missing OP NAME>').strip()
            order = base.number(row, CALL_COUNT) if CALL_COUNT in columns else None
            if order is None and KERNEL_START in columns:
                order = base.number(row, KERNEL_START)
            if order is None:
                order = float(position)
            category = classify_op(name)
            flops = matmul_flops(row, shapes) if complete and category == 'matmul' else None
            cores = base.number(row, CORE_COUNT) if CORE_COUNT in columns else None
            start = base.number(row, KERNEL_START) if KERNEL_START in columns else None
            end = base.number(row, KERNEL_END) if KERNEL_END in columns else None
            by_chip[(row.get(DEVICE_ID) or '?').strip()].append(
                Row(dev=(row.get(DEVICE_ID) or '?').strip(), op=name, cat=category, ns=duration, order=order, cores=cores,
                    flops=flops, fid=(row.get(FIDELITY) or '').strip() or None, start=start, end=end))
    for rows in by_chip.values():
        rows.sort(key=lambda item: item.order)
    return dict(by_chip), facts


# ---- splitting one chip's rows into chunks and layers ----

def split_chunks(rows, expected):
    """Where the prompt's chunks are in one chip's call-ordered rows: dict(spans [(start, end)] of the LAST `expected` groups (fewer when
    rows are missing), warmup groups before them, remainder (SDPA rows that make no whole group), method, notes)."""
    out = dict(spans=[], warmup=0, remainder=0, method='sdpa', notes=[], groups=0)
    sdpa = [i for i, row in enumerate(rows) if row.cat == 'attn.sdpa']
    norms = [i for i, row in enumerate(rows) if NORM_PRE in row.op]
    if not norms:
        out['notes'].append('no %s op: the layers cannot be told apart' % NORM_PRE)
        return out
    if not sdpa:
        out['method'] = 'norm'
        out['notes'].append('no SDPA op found: chunks are cut every %d %s ops from the end instead' % (2 * LAYERS, NORM_PRE))
        per = 2 * LAYERS
        groups = len(norms) // per
        remainder = len(norms) - groups * per
        out.update(groups=groups, remainder=remainder, warmup=max(groups - expected, 0))
        anchored = norms[remainder:]
        starts = [anchored[group * per] for group in range(max(groups - expected, 0), groups)]
    else:
        per = ATTN_LAYERS
        groups = len(sdpa) // per
        remainder = len(sdpa) - groups * per
        out.update(groups=groups, remainder=remainder, warmup=max(groups - expected, 0))
        anchored = sdpa[remainder:]
        before = 2 * FIRST_ATTN_LAYER + 1
        starts = []
        previous_last = -1
        for group in range(max(groups - expected, 0), groups):
            anchor = anchored[group * per]
            position = bisect.bisect_left(norms, anchor)
            start = norms[position - before] if position >= before else None
            if start is None or start <= previous_last:
                out['notes'].append('group %d: the %d norm ops before its first SDPA are not all inside it' % (group, before))
                start = None
            starts.append(start)
            previous_last = anchored[group * per + per - 1]
        if any(start is None for start in starts):
            starts = [start for start in starts if start is not None]
    for index, start in enumerate(starts):
        limit = starts[index + 1] if index + 1 < len(starts) else len(rows)
        out['spans'].append((start, limit))
    return out


def layer_split(rows, span):
    """([layer dict(index, type, mixer (start, end), mlp (start, end))], end of the chunk, problem or None) of one chunk's span. The chunk ends at
    the last MLP half's usual length (the median of the other layers'), so a final norm, lm_head or eager decode glue after it is not counted."""
    start, limit = span
    marks = [i for i in range(start, limit) if NORM_PRE in rows[i].op]
    if len(marks) < 2 * LAYERS:
        return [], limit, '%d %s ops in the chunk, %d needed for %d layers' % (len(marks), NORM_PRE, 2 * LAYERS, LAYERS)
    layers = []
    for layer in range(LAYERS):
        mixer_start, mlp_start = marks[2 * layer], marks[2 * layer + 1]
        mlp_end = marks[2 * layer + 2] if layer + 1 < LAYERS else None
        layers.append(dict(index=layer, mixer=(mixer_start, mlp_start), mlp=(mlp_start, mlp_end)))
    lengths = [layer['mlp'][1] - layer['mlp'][0] for layer in layers[:-1]]
    usual = int(statistics.median(lengths)) if lengths else 0
    last = layers[-1]['mlp'][0]
    end = min(limit, last + usual) if usual else limit
    layers[-1]['mlp'] = (last, end)
    for layer in layers:
        mixer = rows[layer['mixer'][0]:layer['mixer'][1]]
        layer['type'] = 'attn' if any(row.cat == 'attn.sdpa' for row in mixer) else 'gdn'
    return layers, end, None


def summarise_chunk(rows, span):
    """dict(layers, types, ms, cat_ms, cat_calls, ops {op: [calls, ms]}, ops_count, span_ms or None, gap_rows, problem) of one chunk on one chip."""
    layers, end, problem = layer_split(rows, span)
    summary = dict(layers=len(layers), problem=problem, start=span[0], end=end, gap_rows=max(span[1] - end, 0) if span[1] != len(rows) else None)
    if problem:
        return summary
    chunk = rows[span[0]:end]
    cat_ms, cat_calls, ops, type_ms = collections.defaultdict(float), collections.defaultdict(int), {}, collections.defaultdict(float)
    for layer in layers:
        for half in ('mixer', 'mlp'):
            for row in rows[layer[half][0]:layer[half][1]]:
                type_ms[layer['type']] += row.ns / 1e6
    for row in chunk:
        cat_ms[row.cat] += row.ns / 1e6
        cat_calls[row.cat] += 1
        cell = ops.setdefault(row.op, [0, 0.0])
        cell[0] += 1
        cell[1] += row.ns / 1e6
    starts = [row.start for row in chunk if row.start is not None]
    ends = [row.end for row in chunk if row.end is not None]
    span_ms = None
    ratios = [(row.end - row.start) / row.ns for row in chunk if row.start is not None and row.end is not None and row.ns > 0 and row.end > row.start]
    if starts and ends and ratios:
        span_ms = (max(ends) - min(starts)) / statistics.median(ratios) / 1e6
    summary.update(types=[layer['type'] for layer in layers], ms=sum(cat_ms.values()), cat_ms=dict(cat_ms), cat_calls=dict(cat_calls),
                   ops=ops, ops_count=len(chunk), span_ms=span_ms, type_ms=dict(type_ms))
    return summary


# ---- statistics ----

def med(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def top(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


def fit_line(xs, ys):
    """(a, b, r2) of the least-squares line y = a + b x, or None with fewer than two distinct x."""
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    mean_x, mean_y = sum(xs) / float(len(xs)), sum(ys) / float(len(ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return intercept, slope, (1.0 - residual / total) if total else 1.0


def sample_chunks(count):
    """Chunk indices the per-op tables cover: 0, 1, about 1/4, 1/2, 3/4 and the last."""
    if count <= 0:
        return []
    picks = {0, min(1, count - 1), count // 4, count // 2, (3 * count) // 4, count - 1}
    return sorted(picks)


# ---- the sections ----

def per_op_tables(chunk_summaries, wanted):
    """{chunk: [dict(op, category, calls, ms_median, ms_max)]} (top ops, heaviest first) over the chips that have the chunk."""
    tables = {}
    for chunk in wanted:
        per_chip = [summaries[chunk]['ops'] for summaries in chunk_summaries.values() if chunk < len(summaries) and 'ops' in summaries[chunk]]
        names = sorted(set(name for ops in per_chip for name in ops))
        table = []
        for name in names:
            calls = [ops.get(name, [0, 0.0])[0] for ops in per_chip]
            millis = [ops.get(name, [0, 0.0])[1] for ops in per_chip]
            table.append(dict(op=name, category=classify_op(name), calls=med(calls), ms_median=med(millis), ms_max=top(millis)))
        table.sort(key=lambda item: -(item['ms_median'] or 0.0))
        tables[chunk] = table
    return tables


def category_rows(chunk_summaries):
    """One row per chunk: context (tokens already in the KV cache at its start), total ms and ms per category (median over chips), conv calls."""
    chunks = min(len(summaries) for summaries in chunk_summaries.values()) if chunk_summaries else 0
    rows = []
    for chunk in range(chunks):
        per_chip = [summaries[chunk] for summaries in chunk_summaries.values() if 'cat_ms' in summaries[chunk]]
        if not per_chip:
            continue
        row = dict(chunk=chunk, context=chunk * CHUNK, ms=med([item['ms'] for item in per_chip]),
                   span_ms=med([item.get('span_ms') for item in per_chip]), ops=med([item['ops_count'] for item in per_chip]))
        for category in CATEGORIES:
            row[category] = med([item['cat_ms'].get(category, 0.0) for item in per_chip])
            row[category + '.calls'] = med([item['cat_calls'].get(category, 0) for item in per_chip])
        row['gdn_ms'] = med([item['type_ms'].get('gdn', 0.0) for item in per_chip])
        row['attn_ms'] = med([item['type_ms'].get('attn', 0.0) for item in per_chip])
        rows.append(row)
    return rows


def sdpa_fit(rows):
    """The SDPA ms per chunk against the chunk's context in 1,000 tokens: a + b x, per chunk and per attention layer."""
    xs = [row['context'] / 1000.0 for row in rows if row.get('attn.sdpa') is not None]
    ys = [row['attn.sdpa'] for row in rows if row.get('attn.sdpa') is not None]
    fit = fit_line(xs, ys)
    if fit is None:
        return None
    a, b, r2 = fit
    return dict(fixed_ms_per_chunk=a, slope_ms_per_1k_per_chunk=b, r2=r2, fixed_ms_per_attention_layer=a / ATTN_LAYERS,
                slope_ms_per_1k_per_attention_layer=b / ATTN_LAYERS, points=len(xs), at_last_chunk_ms=a + b * xs[-1])


def gdn_section(rows):
    chunks = [row for row in rows if row.get('gdn.conv.calls') is not None]
    if not chunks:
        return None
    pick = chunks[len(chunks) // 2]
    return dict(chunk=pick['chunk'], conv_calls_per_chunk=pick['gdn.conv.calls'], conv_calls_per_gdn_layer=(pick['gdn.conv.calls'] or 0) / float(GDN_LAYERS),
                conv_ms_per_chunk=pick['gdn.conv'], scan_ops_per_chunk=pick['gdn.scan.calls'], scan_ms_per_chunk=pick['gdn.scan'],
                gdn_layers_ms=pick['gdn_ms'], attention_layers_ms=pick['attn_ms'],
                conv_calls_constant=len(set(row['gdn.conv.calls'] for row in chunks)) == 1)


def chunk_rows(by_chip, splits, summaries, chunk):
    """{chip: (rows, layers)} of one chunk on the chips that have it."""
    found = {}
    for chip, rows in by_chip.items():
        if chunk < len(summaries[chip]) and 'types' in summaries[chip][chunk]:
            summary = summaries[chip][chunk]
            layers, _, _ = layer_split(rows, splits[chip]['spans'][chunk])
            found[chip] = (rows, layers, summary)
    return found


def collectives_section(by_chip, splits, summaries, chunk):
    """Per kind: calls per chunk, ms per call (median chip), the minimum over the chips (intrinsic) and the skew (max - min), at one chunk."""
    found = chunk_rows(by_chip, splits, summaries, chunk)
    lists = {}
    for chip, (rows, layers, summary) in found.items():
        lists[chip] = [row for row in rows[summary['start']:summary['end']] if row.cat == 'collective' or (row.cat == 'matmul' and base.is_coll(row.op))]
    out = dict(chunk=chunk, kinds=[], aligned=False, fused_matmul_gather=sorted(set(row.op for rows in lists.values() for row in rows if row.cat == 'matmul')))
    if not lists:
        return out
    lengths = set(len(rows) for rows in lists.values())
    out['aligned'] = len(lengths) == 1
    kinds = collections.OrderedDict()
    if out['aligned']:
        for position in range(lengths.pop()):
            calls = [rows[position] for rows in lists.values()]
            name = calls[0].op
            cell = kinds.setdefault(name, dict(op=name, calls=0, median_ms=[], min_ms=[], skew_ms=[], name_agrees=True))
            cell['calls'] += 1
            values = [row.ns / 1e6 for row in calls]
            cell['median_ms'].append(statistics.median(values))
            cell['min_ms'].append(min(values))
            cell['skew_ms'].append(max(values) - min(values))
            cell['name_agrees'] = cell['name_agrees'] and all(row.op == name for row in calls)
    else:
        for chip, rows in lists.items():
            for row in rows:
                cell = kinds.setdefault(row.op, dict(op=row.op, calls=0, median_ms=[], min_ms=[], skew_ms=[], name_agrees=False))
                cell['calls'] += 1
                cell['median_ms'].append(row.ns / 1e6)
        for cell in kinds.values():
            cell['calls'] = int(round(cell['calls'] / float(len(lists))))
            cell['min_ms'], cell['skew_ms'] = [], []
    for cell in kinds.values():
        out['kinds'].append(dict(op=cell['op'], calls_per_chunk=cell['calls'], ms_per_call=med(cell['median_ms']),
                                 min_ms_per_call=med(cell['min_ms']), skew_ms_per_call=med(cell['skew_ms']),
                                 total_ms=sum(cell['median_ms']) if out['aligned'] else sum(cell['median_ms']) / max(len(lists), 1),
                                 per_call_aligned=cell['name_agrees'] and out['aligned']))
    out['kinds'].sort(key=lambda item: -(item['total_ms'] or 0.0))
    return out


def fidelity_divisor(name):
    """How many times slower than LoFi a MATH FIDELITY string runs (1, 2, 3 or 4); an unrecognised string counts as the assumed fidelity."""
    lowered = (name or '').lower()
    for key, divisor in FIDELITY_DIVISOR.items():
        if key.lower() in lowered:
            return divisor
    return FIDELITY_DIVISOR[ASSUMED_FIDELITY]


def weight_key(layer_type, half, position, count):
    """The weight a matmul at `position` of `count` in a layer half takes: in/out projections of the mixer, gate/up/down of the MLP."""
    if half == 'mixer':
        if position == 0:
            return '%s_in' % layer_type
        if position == count - 1:
            return '%s_out' % layer_type
        return '%s_mid%d' % (layer_type, position)
    if count == 3:
        return ('mlp_gate', 'mlp_up', 'mlp_down')[position]
    if count == 2:
        return ('mlp_gateup', 'mlp_down')[position]
    return 'mlp_%d_of_%d' % (position, count)


def matmul_section(by_chip, splits, summaries, chunk, peak_tflops, tp, shapes_in_csv):
    """Compute efficiency of the weight matmuls at one chunk, per weight: ms, calls, achieved TFLOP/s per chip and % of the stated peak."""
    table = base.weight_table(tp)
    found = chunk_rows(by_chip, splits, summaries, chunk)
    cells = collections.OrderedDict()
    for chip, (rows, layers, summary) in found.items():
        for layer in layers:
            for half in ('mixer', 'mlp'):
                begin, finish = layer[half]
                mms = [row for row in rows[begin:finish] if row.cat == 'matmul']
                for position, row in enumerate(mms):
                    key = weight_key(layer['type'], half, position, len(mms))
                    flops, source = row.flops, 'csv shapes'
                    if flops is None:
                        if key == 'mlp_gateup' and 'mm.mlp.gate' in table:
                            k, n, _ = table['mm.mlp.gate']
                            flops, source = 2.0 * CHUNK * k * 2 * n, 'weight table (assumed)'
                        elif key in TABLE_KEYS and TABLE_KEYS[key] in table:
                            k, n, _ = table[TABLE_KEYS[key]]
                            flops, source = 2.0 * CHUNK * k * n, 'weight table (assumed)'
                        else:
                            source = 'unknown shape'
                    cell = cells.setdefault(key, dict(key=key, ns=[], flops=flops, source=source, cores=[], fidelity=set(), per_chip_calls=collections.Counter(),
                                                      fused=set()))
                    cell['ns'].append(row.ns)
                    if row.cores:
                        cell['cores'].append(row.cores)
                    cell['fidelity'].add(row.fid or ASSUMED_FIDELITY + ' (assumed)')
                    cell['per_chip_calls'][chip] += 1
                    if base.is_coll(row.op):
                        cell['fused'].add(row.op)
    out = []
    total_flops = total_ns = 0.0
    for cell in cells.values():
        ns = med(cell['ns'])
        fidelities = sorted(cell['fidelity'])
        divisor = max(fidelity_divisor(name) for name in fidelities) if fidelities else fidelity_divisor(ASSUMED_FIDELITY)
        peak = peak_tflops / divisor
        tflops = cell['flops'] / ns / 1e3 if cell['flops'] and ns else None
        calls = med(list(cell['per_chip_calls'].values()))
        out.append(dict(weight=cell['key'], calls_per_chunk=calls, ms_per_call=ns / 1e6 if ns is not None else None, flops_per_call=cell['flops'],
                        flops_source=cell['source'], achieved_tflops=tflops, peak_tflops=peak, pct_of_peak=100.0 * tflops / peak if tflops else None,
                        cores=med(cell['cores']), fidelity=fidelities, fused_with_gather=sorted(cell['fused'])))
        if cell['flops'] and ns and calls:
            total_flops += cell['flops'] * calls
            total_ns += ns * calls
    overall = None
    if total_ns:
        tflops = total_flops / total_ns / 1e3
        overall = dict(achieved_tflops=tflops, matmul_ms_per_chunk=total_ns / 1e6, flops_per_chunk=total_flops,
                       pct_of_stated_peak=100.0 * tflops / peak_tflops, note='all weight matmuls with a known shape, against the stated peak at LoFi (a HiFi2 weight tops out at half of it)')
    return dict(chunk=chunk, weights=out, overall=overall, peak_tflops_lofi=peak_tflops, shapes=('csv input-shape columns' if shapes_in_csv
                else 'weight table (an assumption: the CSV carries no input-shape columns)'))


def log_facts(text):
    if text is None:
        return None
    lines = text.splitlines()
    return dict(flush_marker_count=sum(1 for line in lines if FLUSH_MARKER in line), flush_markers=[line.strip() for line in lines if FLUSH_MARKER in line][:12],
                dropped=[line.strip()[:200] for line in lines if base.DROP_LINE.search(line)][:12])


def prompt_tokens_of(gate_json, default):
    data = base.read_json(gate_json) if gate_json else None
    for stream in (data or {}).get('streams') or []:
        value = (stream or {}).get('prompt_tokens')
        if isinstance(value, int) and value > 0:
            return value
    return default


def timing_of(gate_json):
    """The stream's first-token time and answer length from a harness report, when it records them (the profiled arm's wall time is not a timing result)."""
    data = base.read_json(gate_json) if gate_json else None
    for stream in (data or {}).get('streams') or []:
        for key in ('ttft', 'ttft_s', 'first_token_s'):
            if isinstance((stream or {}).get(key), (int, float)):
                return dict(key=key, seconds=stream[key])
    return None


def analyse_rows(by_chip, facts, prompt_tokens=DEFAULT_PROMPT_TOKENS, chips=DEFAULT_CHIPS, server_log_text=None, peak_tflops=LOFI_PEAK_TFLOPS, tp=4):
    expected = max(1, -(-prompt_tokens // CHUNK))
    problems, notes = [], []
    report = dict(scope='Prefill device attribution by op and by 2,048-token chunk; kernel-duration sums, not a critical path or a TTFT claim',
                  prompt_tokens=prompt_tokens, expected_chunks=expected, chips_expected=chips, columns_seen=len(facts['columns']),
                  rows=facts['rows'], rows_traced_dropped=facts['traced'], rows_untraced=sum(len(rows) for rows in by_chip.values()),
                  order_by=facts['order_by'])
    if len(by_chip) != chips:
        problems.append('%d chip(s) in the report (%s), %d expected' % (len(by_chip), ', '.join(sorted(by_chip)) or 'none', chips))
    if facts['no_duration']:
        notes.append('%d row(s) had no kernel duration (counted as 0)' % facts['no_duration'])
    if facts['order_by'] != CALL_COUNT:
        notes.append('no %s column: rows are ordered by %s' % (CALL_COUNT, facts['order_by']))
    if not facts['shapes']:
        notes.append('no INPUT_0/1 shape columns: matmul FLOPs come from the model geometry (an assumption)')
    if not facts['fidelity']:
        notes.append('no %s column: %s assumed for the peak' % (FIDELITY, ASSUMED_FIDELITY))
    if not by_chip:
        problems.append('no untraced row: the arm profiled no prefill (or every row carries a trace id)')
        report['validity'] = dict(ok=False, problems=problems, notes=notes)
        return report
    splits, summaries = {}, {}
    for chip, rows in sorted(by_chip.items()):
        splits[chip] = split_chunks(rows, expected)
        summaries[chip] = [summarise_chunk(rows, span) for span in splits[chip]['spans']]
        for text in splits[chip]['notes']:
            notes.append('chip %s: %s' % (chip, text))
    counts = dict((chip, len(split['spans'])) for chip, split in splits.items())
    found = min(counts.values()) if counts else 0
    report['chunks_found_per_chip'] = counts
    report['warmup_groups_per_chip'] = dict((chip, split['warmup']) for chip, split in splits.items())
    report['split_method'] = sorted(set(split['method'] for split in splits.values()))
    if found < expected:
        problems.append('%d prefill chunk(s) found on the shortest chip, %d expected for a %d-token prompt (rows were lost, or the prompt was shorter): %s'
                        % (found, expected, prompt_tokens, ', '.join('%s:%d' % item for item in sorted(counts.items()))))
    if len(set(counts.values())) > 1:
        problems.append('the chips found different chunk counts: %s' % ', '.join('%s:%d' % item for item in sorted(counts.items())))
    for chip, split in sorted(splits.items()):
        if split['remainder']:
            notes.append('chip %s: %d SDPA/norm row(s) make no whole chunk group (dropped from the front)' % (chip, split['remainder']))
    broken = sorted(set(item['problem'] for items in summaries.values() for item in items if item.get('problem')))
    for text in broken[:3]:
        problems.append('chunk structure: %s' % text)
    good = dict((chip, [item for item in items]) for chip, items in summaries.items())
    usable = min((len(items) for items in good.values()), default=0)
    complete = [chunk for chunk in range(usable) if all('types' in good[chip][chunk] for chip in good)]
    shape_bad = []
    for chunk in complete:
        for chip in good:
            types = good[chip][chunk]['types']
            if types.count('gdn') != GDN_LAYERS or types.count('attn') != ATTN_LAYERS or any(
                    (kind == 'attn') != (index % 4 == FIRST_ATTN_LAYER) for index, kind in enumerate(types)):
                shape_bad.append((chip, chunk))
    if shape_bad:
        problems.append('layer structure: %d chunk(s) do not have %d GDN and %d attention layers in the 3:1 pattern (first: chip %s chunk %d)'
                        % (len(shape_bad), GDN_LAYERS, ATTN_LAYERS, shape_bad[0][0], shape_bad[0][1]))
    disagreeing = []
    for chunk in complete:
        counts_at = [good[chip][chunk]['ops_count'] for chip in good]
        if max(counts_at) - min(counts_at) > ROW_AGREEMENT_PROBLEM * max(counts_at):
            disagreeing.append((chunk, min(counts_at), max(counts_at)))
        elif max(counts_at) != min(counts_at):
            notes.append('chunk %d: the chips differ by %d op(s) (%d..%d)' % (chunk, max(counts_at) - min(counts_at), min(counts_at), max(counts_at)))
    if disagreeing:
        problems.append('per-chip op counts disagree in %d chunk(s) (first: chunk %d, %d..%d ops): rows were lost on some chip'
                        % (len(disagreeing), disagreeing[0][0], disagreeing[0][1], disagreeing[0][2]))
    logs = log_facts(server_log_text)
    report['server_log'] = logs
    if logs is None:
        notes.append('no server log: the flush markers and dropped-marker lines were not checked')
    else:
        if logs['flush_marker_count'] < MIN_FLUSH_MARKERS:
            notes.append('%d flush marker line(s) in the server log, %d expected: the flush hook may not have run, so per-core buffers may have overflowed inside a chunk'
                         % (logs['flush_marker_count'], MIN_FLUSH_MARKERS))
        if logs['dropped']:
            problems.append('the server log reports dropped or lost profiler markers (%d line(s), first: %s)' % (len(logs['dropped']), logs['dropped'][0]))
    by_chip_complete = dict((chip, good[chip]) for chip in good)
    rows = category_rows(by_chip_complete) if complete else []
    report['by_chunk'] = rows
    wanted = [rows[index]['chunk'] for index in sample_chunks(len(rows))]
    report['sampled_chunks'] = wanted
    report['per_op'] = dict((str(chunk), table[:TOP_OPS]) for chunk, table in per_op_tables(by_chip_complete, wanted).items()) if rows else {}
    report['sdpa_fit'] = sdpa_fit(rows)
    report['gdn'] = gdn_section(rows)
    if rows:
        total = sum(row['ms'] or 0.0 for row in rows)
        other = sum(row['other'] or 0.0 for row in rows)
        report['unclassified_share'] = other / total if total else 0.0
        report['category_totals_ms'] = dict((category, sum(row[category] or 0.0 for row in rows)) for category in CATEGORIES)
        report['total_kernel_ms'] = total
        if total and other / total > UNCLASSIFIED_NOTE_SHARE:
            names = sorted(set(op['op'] for table in report['per_op'].values() for op in table if op['category'] == 'other'))
            notes.append('%.1f%% of kernel time is in the "other" category (no rule matched): %s' % (100.0 * other / total, ', '.join(names[:8]) or 'see by_chunk'))
        middle = wanted[len(wanted) // 2] if wanted else 0
        report['collectives'] = collectives_section(by_chip, splits, summaries, middle)
        if not report['collectives']['aligned']:
            notes.append('the chips list different collective sequences at chunk %d: per-call minimum and skew are not computed' % middle)
        report['matmul_efficiency'] = dict(
            (str(chunk), matmul_section(by_chip, splits, summaries, chunk, peak_tflops, tp, facts['shapes'])) for chunk in sorted(set([wanted[0], middle, wanted[-1]])))
    report['validity'] = dict(ok=not problems, problems=problems, notes=notes)
    return report


def analyse_files(csv_path, server_log=None, gate_json=None, twin_log=None, twin_json=None, chips=DEFAULT_CHIPS, prompt_tokens=None,
                  peak_tflops=LOFI_PEAK_TFLOPS):
    """The report dict of one cpp_device_perf_report CSV (the signature ops_profile_plan.finish_arm calls; the twin log and json only
    contribute the twin's first-token time, if recorded)."""
    by_chip, facts = load(csv_path)
    tokens = prompt_tokens or prompt_tokens_of(gate_json, DEFAULT_PROMPT_TOKENS)
    report = analyse_rows(by_chip, facts, prompt_tokens=tokens, chips=chips, server_log_text=base.read_text(server_log), peak_tflops=peak_tflops)
    report['trace_arm_timing'] = timing_of(gate_json)
    report['twin_arm_timing'] = timing_of(twin_json)
    return report


# ---- the markdown ----

def fmt(value, digits=2):
    if value is None:
        return '-'
    if isinstance(value, float):
        return ('%.' + str(digits) + 'f') % value
    return str(value)


def table(header, body):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(row) + ' |' for row in body]
    return lines


def render_markdown(report):
    lines = ['# TP4 prefill op profile', '', '%s' % report['scope'], '',
             'Prompt %s tokens (%s chunks of %d expected); %s rows, %s traced (decode) rows dropped, %s prefill rows on %s chip(s); ordered by %s.'
             % (report['prompt_tokens'], report['expected_chunks'], CHUNK, report['rows'], report['rows_traced_dropped'], report['rows_untraced'],
                len(report.get('chunks_found_per_chip') or {}) or 0, report['order_by']), '']
    validity = report['validity']
    lines += ['## Validity', '', 'Verdict: %s' % ('OK' if validity['ok'] else 'PROBLEMS')]
    lines += ['- PROBLEM: %s' % text for text in validity['problems']] + ['- note: %s' % text for text in validity['notes']] + ['']
    rows = report.get('by_chunk') or []
    if rows:
        lines += ['## Kernel ms per chunk by category (median over chips)', '']
        step = max(1, len(rows) // 12)
        picked = [row for index, row in enumerate(rows) if index % step == 0 or index == len(rows) - 1]
        header = ['chunk', 'context'] + [CATEGORY_TITLES[c] for c in CATEGORIES] + ['total', 'span']
        body = [[str(row['chunk']), str(row['context'])] + [fmt(row[c]) for c in CATEGORIES] + [fmt(row['ms']), fmt(row.get('span_ms'))] for row in picked]
        lines += table(header, body) + ['']
        totals = report.get('category_totals_ms') or {}
        if totals:
            lines += ['Whole prompt (all %d chunks, median chip): %s; total %s ms; unclassified %.1f%%.' % (
                len(rows), ', '.join('%s %s ms' % (CATEGORY_TITLES[c], fmt(totals[c], 1)) for c in CATEGORIES), fmt(report.get('total_kernel_ms'), 1),
                100.0 * report.get('unclassified_share', 0.0)), '']
    fit = report.get('sdpa_fit')
    if fit:
        lines += ['## Attention prefill against context', '',
                  'SDPA ms per chunk = %s + %s x (context in 1,000 tokens); R^2 %s over %d chunks. Per attention layer: %s + %s x context(1k). At the last chunk: %s ms.'
                  % (fmt(fit['fixed_ms_per_chunk'], 3), fmt(fit['slope_ms_per_1k_per_chunk'], 4), fmt(fit['r2'], 4), fit['points'],
                     fmt(fit['fixed_ms_per_attention_layer'], 3), fmt(fit['slope_ms_per_1k_per_attention_layer'], 4), fmt(fit['at_last_chunk_ms'])), '']
    gdn = report.get('gdn')
    if gdn:
        lines += ['## GDN chunked prefill (chunk %d)' % gdn['chunk'], '',
                  'Conv calls per chunk %s (%s per GDN layer), %s ms; scan/other GDN ops per chunk %s, %s ms; GDN layers %s ms, attention layers %s ms. Conv calls per chunk %s across chunks.'
                  % (fmt(gdn['conv_calls_per_chunk'], 0), fmt(gdn['conv_calls_per_gdn_layer']), fmt(gdn['conv_ms_per_chunk']), fmt(gdn['scan_ops_per_chunk'], 0),
                     fmt(gdn['scan_ms_per_chunk']), fmt(gdn['gdn_layers_ms']), fmt(gdn['attention_layers_ms']),
                     'constant' if gdn['conv_calls_constant'] else 'VARY'), '']
    collectives = report.get('collectives')
    if collectives and collectives['kinds']:
        lines += ['## Collectives (chunk %d)' % collectives['chunk'], '']
        lines += table(['op', 'calls/chunk', 'ms/call', 'min over chips', 'skew', 'total ms'],
                       [[item['op'], fmt(item['calls_per_chunk'], 0), fmt(item['ms_per_call'], 3), fmt(item['min_ms_per_call'], 3), fmt(item['skew_ms_per_call'], 3),
                         fmt(item['total_ms'])] for item in collectives['kinds']]) + ['']
        if collectives['fused_matmul_gather']:
            lines += ['Fused gather+matmul kernels (counted under weight matmuls): %s.' % ', '.join(collectives['fused_matmul_gather']), '']
    for chunk, section in sorted((report.get('matmul_efficiency') or {}).items(), key=lambda item: int(item[0])):
        lines += ['## Weight matmul efficiency (chunk %s)' % chunk, '',
                  'Peak: %s TFLOP/s at LoFi, divided by the MATH FIDELITY factor: an ESTIMATE from the vendor sheet, not a measurement. FLOPs from the %s.'
                  % (fmt(section['peak_tflops_lofi'], 0), section['shapes']), '']
        lines += table(['weight', 'calls/chunk', 'ms/call', 'GFLOP/call', 'TFLOP/s', 'peak', '% of peak', 'cores', 'fidelity', 'shape from'],
                       [[item['weight'], fmt(item['calls_per_chunk'], 0), fmt(item['ms_per_call'], 3),
                         fmt(item['flops_per_call'] / 1e9 if item['flops_per_call'] else None, 1), fmt(item['achieved_tflops'], 1), fmt(item['peak_tflops'], 0),
                         fmt(item['pct_of_peak'], 1), fmt(item['cores'], 0), ','.join(item['fidelity']), item['flops_source']] for item in section['weights']]) + ['']
        if section['overall']:
            lines += ['All weight matmuls with a known shape: %s TFLOP/s achieved, %s ms per chunk, %s%% of the LoFi peak (%s).' % (
                fmt(section['overall']['achieved_tflops'], 1), fmt(section['overall']['matmul_ms_per_chunk']), fmt(section['overall']['pct_of_stated_peak'], 1),
                section['overall']['note']), '']
    for chunk, ops in sorted((report.get('per_op') or {}).items(), key=lambda item: int(item[0])):
        context = int(chunk) * CHUNK
        lines += ['## Per-op ms, chunk %s (context %d)' % (chunk, context), '']
        lines += table(['op', 'category', 'calls', 'ms (median chip)', 'ms (max chip)'],
                       [[item['op'], item['category'], fmt(item['calls'], 0), fmt(item['ms_median'], 3), fmt(item['ms_max'], 3)] for item in ops]) + ['']
    timing = [('trace arm', report.get('trace_arm_timing')), ('twin arm', report.get('twin_arm_timing'))]
    if any(item for _, item in timing):
        lines += ['## First-token time recorded by the harness (the profiled arm is perturbed: not a timing result)', '']
        lines += ['- %s: %s s (%s)' % (name, fmt(item['seconds']), item['key']) for name, item in timing if item] + ['']
    return '\n'.join(lines) + '\n'


# ---- the command line ----

def artifact_paths(results):
    """(csv, trace server log, trace gate json, twin server log, twin gate json) under a downloaded gate/ directory."""
    ops_dir = os.path.join(results, 'ops')
    csv_path = os.path.join(ops_dir, 'cpp_device_perf_report.prefill.csv.gz')
    paths = [csv_path]
    for arm in ('ops-prefill-trace', 'ops-prefill-twin'):
        paths += [os.path.join(results, arm, 'server.log'), os.path.join(results, arm, 'm3native-gate.json')]
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('csv', nargs='?', help='cpp_device_perf_report.csv[.gz]')
    parser.add_argument('--results', help='the downloaded gate/ directory')
    parser.add_argument('--server-log')
    parser.add_argument('--gate-json')
    parser.add_argument('--twin-log')
    parser.add_argument('--twin-json')
    parser.add_argument('--chips', type=int, default=DEFAULT_CHIPS)
    parser.add_argument('--prompt-tokens', type=int)
    parser.add_argument('--peak-tflops', type=float, default=LOFI_PEAK_TFLOPS)
    parser.add_argument('--out')
    options = parser.parse_args(argv)
    if bool(options.csv) == bool(options.results):
        parser.error('name a CSV or --results')
    csv_path, server_log, gate_json, twin_log, twin_json = options.csv, options.server_log, options.gate_json, options.twin_log, options.twin_json
    if options.results:
        csv_path, server_log, gate_json, twin_log, twin_json = artifact_paths(options.results)
    if not os.path.isfile(csv_path):
        sys.stderr.write('no such CSV: %s\n' % csv_path)
        return 2
    try:
        report = analyse_files(csv_path, server_log=server_log, gate_json=gate_json, twin_log=twin_log, twin_json=twin_json, chips=options.chips,
                               prompt_tokens=options.prompt_tokens, peak_tflops=options.peak_tflops)
    except ReportError as error:
        sys.stderr.write('%s\n' % error)
        return 2
    out = options.out or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out, exist_ok=True)
    markdown = render_markdown(report)
    with open(os.path.join(out, 'prefill-profile-report.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    with open(os.path.join(out, 'prefill-profile-report.md'), 'w', encoding='utf-8') as handle:
        handle.write(markdown)
    sys.stdout.write(markdown)
    return 0 if report['validity']['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
