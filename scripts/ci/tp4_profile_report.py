"""The offline analysis of a TP4 op-level device profile (ops_profile_plan's ops-trace arm).

    python3 scripts/ci/tp4_profile_report.py --results <the downloaded gate/ directory> [--out <dir>]
    python3 scripts/ci/tp4_profile_report.py <cpp_device_perf_report.csv[.gz]> [--server-log L] [--gate-json J]
            [--twin-log L] [--twin-json J] [--chips 4] [--out <dir>]

--results reads the artifact's layout: ops/cpp_device_perf_report.csv.gz (the gate compresses tracy's CPP report there),
ops-trace/server.log and m3native-gate.json, and the same two files under ops-twin/. It writes tp4-profile-report.json
and tp4-profile-report.md beside the CSV (or into --out) and prints the markdown. docs/tp4-profile.md reads it.

What it does, from tt-metal's cpp_device_perf_report.csv (TT_METAL_PROFILER_CPP_POST_PROCESS=1, trace tracking on):
  * keeps only trace replays that are COMPLETE on every chip (the profiler's per-core buffers fill at different points, so
    late sessions are truncated), and maps each verify replay to its packed round by session id ([PACKED-PHASE] round=);
  * identifies traces by what they contain, not by id: a trace of 64 layers is a verify; its attention layers hold one
    SDPA launch per user in the block (four: the packed 64-row block; one: a lone user's 1/2/4-row step);
  * classifies every op by its ROLE in the layer, never by core count (TP4 core counts differ from TP2's): a layer is GDN
    if its mixer holds GdnConvGates, else attention; matmuls by position (first = in-projection, last = out-projection;
    gate, up, down in the MLP half); the recurrence is the longest generic op after the last conv-gates launch; the
    attention core is the SDPA op, or per user the longest generic between AttnPrep and the heads concat;
  * reports time per category and layer type, weight throughput and ns per tile per core per matmul, collectives per call
    with the skew between chips, the cross-chip critical path, the in-trace dispatch gaps, the per-user and per-live-count
    cost, an SDPA-against-context fit, the projected lone 16-row verify, the device time between verify replays,
    profiling overhead against the twin, and TP2 and the research projection against this measurement.
The FW envelopes are not used: they overlap their neighbours (dispatch pre-launches the next program on idle cores);
kernel-to-kernel gaps are the in-trace dispatch cost.

Marks: every figure is measured from the CSV except those in PROJECTION_TP4_4U_8K and TP2_C2_4X32K, which are the
research file's estimates (e) and derived numbers (d), printed beside the measurement to show which terms moved.

Stdlib only, Python 3.7 syntax.
"""
import argparse
import collections
import csv
import gzip
import json
import math
import os
import re
import statistics
import sys

CLK_GHZ = 1.35          # Blackhole AICLK; checked against kernel ns / cycles per file (clock_check)
LAYERS = 64
COLLECTIVES = ('AllGather', 'ReduceScatter', 'AllReduce', 'AllBroadcast', 'AllToAll')
MIN_SESSIONS = 8        # complete verify-64 (and 4-row) sessions a profile needs to be called complete
MIN_READBACKS = 20
PERTURBATION_LIMIT = 0.03
ROUND_MAP_TOLERANCE = 0.25   # median |device span - host trace_ms| / trace_ms past which the session -> round map is doubted
READBACK_MARKER = '[PINDIAG] device profiler read back after replay'
AUDIT_MARKERS = ('[PINDIAG] verify t1 audit', '[PINDIAG] verify t2 audit')
DROP_LINE = re.compile(r'(?i)(?:dropp?ed|lost|overflow)[^\n]*marker|marker[^\n]*(?:dropp?ed|lost|overflow)')
PACKED_PHASE = re.compile(r'\[PACKED-PHASE\] round=([0-9]+) users=([0-9]+)([^\n]*)')
PACKED_LINE = re.compile(r'\[PACKED\] request=(\S+) segment=([0-9]+) position=([0-9]+) prefix=([0-9]+) emitted=([0-9]+)')
LIVE_FIELD = re.compile(r'\blive=([0-9]+)')
IDLE_FIELD = re.compile(r'\bidle=(\S+)')
TRACE_MS_FIELD = re.compile(r'\btrace_ms=([0-9.]+)')

# Categories that scale with the users in the block (one launch or one piece per user) and those shared by the block.
PER_USER = ('attn.sdpa', 'gdn.conv_gates', 'attn.glue', 'gdn.glue')
GROUPS = collections.OrderedDict([
    ('weight matmuls', ('mm.',)), ('gdn.recurrence', ('gdn.recurrence',)), ('gdn.glue', ('gdn.glue',)),
    ('gdn.conv_gates', ('gdn.conv_gates',)), ('attn.sdpa', ('attn.sdpa',)), ('attn.glue', ('attn.glue',)),
    ('collectives', ('collective', 'sampler.collective')), ('sampler', ('sampler.argmax',)),
    ('norms, adds, input', ('norm', 'residual_add', 'mlp.eltwise', 'mlp.glue', 'input/embedding'))])
# The research file's numbers, ms, printed beside the measurement (project_tp4.out.txt; TP2 C2 is derived, v138 plus the
# measured levers; the TP4 column is an estimate).
TP2_C2_4X32K = collections.OrderedDict([
    ('weight matmuls', 33.7), ('gdn.recurrence', 10.3), ('gdn.glue', 9.9), ('gdn.conv_gates', 6.8), ('attn.sdpa', 14.7),
    ('attn.glue', 6.4), ('collectives', 4.7), ('sampler', 0.4), ('norms, adds, input', 2.2)])
PROJECTION_TP4_4U_8K = collections.OrderedDict([
    ('weight matmuls', 17.7), ('gdn.recurrence', 10.3), ('gdn.glue', 6.9), ('gdn.conv_gates', 3.8), ('attn.sdpa', 6.5),
    ('attn.glue', 4.3), ('collectives', 6.8), ('sampler', 0.4), ('norms, adds, input', 2.2)])
PROJECTION_TP4_TRACE_MS = 61.7
PROJECTION_TP4_1U_16ROW_MS = 43.9
TP2_C2_TRACE_MS = 91.9
# Bytes per weight element (the tile formats: bfp4_b 576 B, bfp8_b 1088 B, per 32x32 tile), and the tile bytes.
TILE_BYTES = {'bf4': 576, 'bf8': 1088, 'bf16': 2048}
DRAM_GBPS = 405.0       # the p150a's measured streaming peak the research uses; 472 is the DRAM-sharded best


class ReportError(ValueError):
    """An input the analysis cannot read."""


def is_coll(op):
    return any(name in op for name in COLLECTIVES)


def is_norm(op):
    return op.startswith('LayerNorm') or ('Norm' in op and 'Gdn' not in op)


def clean_op(name):
    return name.replace('DeviceOperation', '').replace('Operation', '')


# ---- reading the CSV ----

Op = collections.namedtuple('Op', 'op cores k fs ks ke spread')


def open_text(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', newline='', encoding='utf-8', errors='replace')
    return open(path, newline='', encoding='utf-8', errors='replace')


def number(row, key):
    value = row.get(key) or ''
    try:
        return float(value)
    except ValueError:
        return None


def load(path):
    """({(trace, session, device): [Op sorted by time]}, {device: [(kernel start cycle, kernel ns, trace id)] of every
    row, replayed or eager}, the columns seen)."""
    sessions = collections.defaultdict(list)
    every = collections.defaultdict(list)
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        need = ('DEVICE ID', 'OP NAME', 'DEVICE KERNEL DURATION [ns]')
        missing = [name for name in need if name not in (reader.fieldnames or [])]
        if missing:
            raise ReportError('%s lacks the column(s) %s: not a cpp_device_perf_report.csv' % (path, ', '.join(missing)))
        columns = list(reader.fieldnames)
        for row in reader:
            device = row['DEVICE ID']
            duration = number(row, 'DEVICE KERNEL DURATION [ns]') or 0.0
            start = number(row, 'DEVICE KERNEL START CYCLE')
            trace = row.get('METAL TRACE ID') or ''
            every[device].append((start if start is not None else 0.0, duration, trace))
            session = row.get('METAL TRACE REPLAY SESSION ID') or ''
            if not trace or not session:
                continue
            fw_start = number(row, 'DEVICE FW START CYCLE')
            end = number(row, 'DEVICE KERNEL END CYCLE')
            sessions[(trace, session, device)].append(Op(
                clean_op(row.get('OP NAME') or ''), int(float(row.get('CORE COUNT') or 0)), duration,
                fw_start if fw_start is not None else (start or 0.0), start or 0.0, end or 0.0,
                number(row, 'DEVICE KERNEL FIRST TO LAST START [ns]') or 0.0))
    for ops in sessions.values():
        ops.sort(key=lambda op: op.fs)
    return sessions, every, columns


def complete_sessions(sessions, chips=None):
    """{trace: dict(devices, ops, sessions_total, complete={session: {device: [Op]}}, truncated)} - only sessions with the
    trace's full op count on every device it appears on (and, when `chips` is given, on that many)."""
    by_trace = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (trace, session, device), ops in sessions.items():
        by_trace[trace][session][device] = ops
    out = {}
    for trace, sids in by_trace.items():
        devices = sorted(set(device for per in sids.values() for device in per))
        full = max(len(ops) for per in sids.values() for ops in per.values())
        keep = dict((sid, per) for sid, per in sids.items()
                    if sorted(per) == devices and all(len(ops) == full for ops in per.values()))
        if chips is not None and len(devices) != chips:
            keep = {}
        out[trace] = dict(devices=devices, ops=full, sessions_total=len(sids), complete=keep,
                          truncated=len(sids) - len(keep))
    return out


def span_ns(ops):
    return (max(op.ke for op in ops) - min(op.ks for op in ops)) / CLK_GHZ


def gaps(ops):
    """(idle gap sum, overlap sum) in ns over one chip's replay: kernel end of op i to kernel start of op i+1."""
    gap = overlap = 0.0
    for a, b in zip(ops, ops[1:]):
        delta = (b.ks - a.ke) / CLK_GHZ
        if delta >= 0:
            gap += delta
        else:
            overlap += -delta
    return gap, overlap


def clock_check(ops):
    """The AICLK the file implies (cycles per ns from the ops' kernel start and end cycles), median over ops."""
    ratios = [(op.ke - op.ks) / op.k for op in ops if op.k > 0 and op.ke > op.ks]
    return statistics.median(ratios) if ratios else None


# ---- classifying one replay ----

def segment(ops):
    """(layers, final norm index): each layer is a ((mixer start, mixer end), (mlp start, mlp end)) pair cut at the norm ops."""
    norms = [i for i, op in enumerate(ops) if is_norm(op.op)]
    if len(norms) < 3:
        return [], (norms[-1] if norms else len(ops))
    halves, final = norms[:-1], norms[-1]
    layers = []
    for h in range(0, len(halves) - 1, 2):
        mixer = (halves[h], halves[h + 1])
        mlp = (halves[h + 1], halves[h + 2] if h + 2 < len(halves) else final)
        layers.append((mixer, mlp))
    return layers, final


def is_matmul(op):
    return 'Matmul' in op


def classify(ops):
    """(roles, layer count, layer types): roles[i] = (layer index or None, layer type, category, detail)."""
    layers, final = segment(ops)
    roles = [(None, 'pre', 'input/embedding', None)] * len(ops)
    types = []
    for index, ((m0, m1), (p0, p1)) in enumerate(layers):
        mixer, mlp = list(range(m0, m1)), list(range(p0, p1))
        names = [ops[i].op for i in mixer]
        ltype = 'gdn' if any(name.startswith('GdnConvGates') for name in names) else 'attn'
        types.append(ltype)
        sdpa = []
        if ltype == 'attn':
            named = [i for i in mixer if ops[i].op.startswith('SdpaDecode') or ops[i].op.startswith('SDPA')
                     or ops[i].op.startswith('Sdpa')]
            if named:
                sdpa = named
            else:
                prep = [i for i in mixer if ops[i].op.startswith('AttnPrep')]
                heads = [i for i in mixer if 'ConcatHeads' in ops[i].op]
                if prep and heads:
                    inner = [i for i in range(prep[0] + 1, heads[0]) if ops[i].op.startswith('GenericOp')]
                    if inner:
                        top = max(ops[i].k for i in inner)
                        sdpa = [i for i in inner if ops[i].k >= 0.5 * top]
        mm = [i for i in mixer if is_matmul(ops[i].op)]
        generic = [i for i in mixer if ops[i].op.startswith('GenericOp')]
        recurrence = None
        if ltype == 'gdn' and generic:
            conv = [i for i in mixer if ops[i].op.startswith('GdnConvGates')]
            after = [i for i in generic if not conv or i > max(conv)]
            recurrence = max(after or generic, key=lambda i: ops[i].k)
        for i in mixer:
            name = ops[i].op
            detail = None
            if is_coll(name):
                cat = 'collective'
            elif is_norm(name):
                cat = 'norm'
            elif mm and i == mm[0]:
                cat = 'mm.%s_in' % ltype
            elif mm and i == mm[-1] and len(mm) > 1:
                cat = 'mm.%s_out' % ltype
            elif i == recurrence:
                cat = 'gdn.recurrence'
            elif ltype == 'gdn' and name.startswith('GdnConvGates'):
                cat = 'gdn.conv_gates'
            elif i in sdpa:
                cat, detail = 'attn.sdpa', sdpa.index(i)
            elif name.startswith('BinaryNg') and i > (mm[-1] if mm else m1):
                cat = 'residual_add'
            else:
                cat = '%s.glue' % ltype
            roles[i] = (index, ltype, cat, detail)
        mlp_mm = [i for i in mlp if is_matmul(ops[i].op)]
        for i in mlp:
            name = ops[i].op
            detail = None
            if is_coll(name):
                cat = 'collective'
            elif is_norm(name):
                cat = 'norm'
            elif is_matmul(name):
                cat = 'mm.mlp'
                detail = ('gate', 'up', 'down')[mlp_mm.index(i)] if len(mlp_mm) == 3 else mlp_mm.index(i)
            elif name.startswith('BinaryNg'):
                cat = 'mlp.eltwise' if mlp_mm and i < mlp_mm[-1] else 'residual_add'
            else:
                cat = 'mlp.glue'
            roles[i] = (index, ltype, cat, detail)
    for i in range(final, len(ops)):
        name = ops[i].op
        if is_matmul(name):
            cat = 'mm.lm_head'
        elif is_coll(name):
            cat = 'sampler.collective'
        elif is_norm(name):
            cat = 'norm'
        else:
            cat = 'sampler.argmax/glue'
        roles[i] = (None, 'tail', cat, None)
    for i in range(0, layers[0][0][0] if layers else 0):
        roles[i] = (None, 'pre', 'collective' if is_coll(ops[i].op) else 'input/embedding', None)
    return roles, len(layers), types


def sdpa_per_attention_layer(ops, roles):
    """The SDPA launches in one attention layer of this replay (the users in the block), or None without an attention layer."""
    counts = collections.Counter(role[0] for role in roles if role[2] == 'attn.sdpa')
    return statistics.median(counts.values()) if counts else None


def trace_signature(ops):
    """What one replay contains: (kind, detail) - 'verify-packed' (users = SDPA launches per attention layer >= 2),
    'verify-single' (one), 'drafter', 'small' or 'other'."""
    norms = sum(1 for op in ops if is_norm(op.op))
    if norms >= 2 * LAYERS:
        roles, layers, _ = classify(ops)
        users = sdpa_per_attention_layer(ops, roles)
        if users is not None and users >= 2:
            return 'verify-packed', dict(layers=layers, users=int(users))
        return 'verify-single', dict(layers=layers, users=1)
    if any(op.op.startswith('TopK') or 'ArgMax' in op.op for op in ops) and any('Sdpa' in op.op or 'SDPA' in op.op
                                                                              for op in ops):
        return 'drafter', dict(ops=len(ops))
    if len(ops) <= 20:
        return 'small', dict(ops=len(ops))
    return 'other', dict(ops=len(ops))


def median(values):
    return statistics.median(values) if values else 0.0


# ---- the host log ----

def parse_log(text):
    """{rounds: {round: dict(users, live, idle, trace_ms, positions {segment: position})}, readbacks, audits, drops}."""
    rounds = {}
    order = []
    for match in PACKED_PHASE.finditer(text):
        rest = match.group(3)
        live = LIVE_FIELD.search(rest)
        idle = IDLE_FIELD.search(rest)
        trace_ms = TRACE_MS_FIELD.search(rest)
        number_ = int(match.group(1))
        rounds[number_] = dict(users=int(match.group(2)), live=int(live.group(1)) if live else None,
                               idle=[int(x) for x in re.findall(r'[0-9]+', idle.group(1))] if idle else [],
                               trace_ms=float(trace_ms.group(1)) if trace_ms else None, positions={}, at=match.start())
        order.append(number_)
    # [PACKED] audit lines: attach each to the phase line after it or before it, whichever leaves the counts consistent.
    packed = [(m.start(), int(m.group(2)), int(m.group(3))) for m in PACKED_LINE.finditer(text)]
    for mode in ('after', 'before'):
        attached = dict((number_, {}) for number_ in order)
        for at, segment_, position in packed:
            if mode == 'after':
                owners = [n for n in order if rounds[n]['at'] <= at]
                owner = owners[-1] if owners else None
            else:
                owners = [n for n in order if rounds[n]['at'] >= at]
                owner = owners[0] if owners else None
            if owner is not None:
                attached[owner][segment_] = position
        consistent = sum(1 for n in order if rounds[n]['live'] is not None and len(attached[n]) == rounds[n]['live'])
        if order and consistent >= 0.5 * len(order):
            for n in order:
                rounds[n]['positions'] = attached[n]
            break
    return dict(rounds=rounds, readbacks=text.count(READBACK_MARKER),
                audits=[marker for marker in AUDIT_MARKERS if marker in text],
                drops=len(DROP_LINE.findall(text)),
                phase_trace_ms=[r['trace_ms'] for r in rounds.values() if r['trace_ms'] is not None])


def read_text(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8', errors='replace') as handle:
        return handle.read()


def read_json(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


# ---- weights ----

def weight_table(tp=4):
    """{detail key: (K, N per chip, dtype)} of the weight matmuls, from tp_shapes' geometry (the model's totals divided by
    the chip count); the dtypes are the families' (gate and up bf4, the rest bf8): assumptions, printed as such."""
    try:
        import tp_shapes
        geo = tp_shapes.geometry(tp)
        hidden = tp_shapes.HIDDEN
        head_dim = tp_shapes.ATTENTION_HEAD_DIM
    except Exception:
        return {}
    qkv = geo.attn_heads * head_dim * 2 + geo.attn_kv_heads * head_dim * 2
    return {'mm.mlp.gate': (hidden, geo.mlp, 'bf4'), 'mm.mlp.up': (hidden, geo.mlp, 'bf4'),
            'mm.mlp.down': (geo.mlp, hidden, 'bf8'), 'mm.gdn_in': (hidden, geo.gdn_qkvzab_padded, 'bf8'),
            'mm.gdn_out': (geo.gdn_value, hidden, 'bf8'), 'mm.attn_in': (hidden, qkv, 'bf8'),
            'mm.attn_out': (geo.attn_out, hidden, 'bf8'), 'mm.lm_head': (hidden, geo.vocab, 'bf8')}


# ---- analysing one verify trace ----

def analyse_trace(info, table=None):
    """The breakdown of one verify trace over its complete sessions."""
    devices = info['devices']
    per_cat = dict((d, collections.defaultdict(list)) for d in devices)
    per_type = dict((d, collections.defaultdict(list)) for d in devices)
    totals = dict((d, []) for d in devices)
    spans = dict((d, []) for d in devices)
    gap_sums = dict((d, []) for d in devices)
    critical, skew = [], []
    ccl = collections.defaultdict(lambda: collections.defaultdict(list))   # (kind, position class) -> per call min/skew
    mm_ops = collections.defaultdict(list)                                  # weight key -> [(ns, cores)]
    per_session = {}
    layer_types, layers = [], 0
    sdpa_by_user = collections.defaultdict(list)      # user index (SDPA order in the layer) -> µs
    conv_counts = []
    first = next(iter(info['complete'].values()))
    ref_ops = first[devices[0]]
    for sid, per in info['complete'].items():
        chip_cat = {}
        for d in devices:
            ops = per[d]
            roles, layers, layer_types = classify(ops)
            cat, ty = collections.defaultdict(float), collections.defaultdict(float)
            for op, (li, lt, name, detail) in zip(ops, roles):
                cat[name] += op.k
                ty[lt] += op.k
            for name, value in cat.items():
                per_cat[d][name].append(value / 1e6)
            for name, value in ty.items():
                per_type[d][name].append(value / 1e6)
            totals[d].append(sum(op.k for op in ops) / 1e6)
            spans[d].append(span_ns(ops) / 1e6)
            gap_sums[d].append(gaps(ops)[0] / 1e6)
            chip_cat[d] = cat
            if d == devices[0]:
                per_session[sid] = dict(kernel_ms=sum(op.k for op in ops) / 1e6, span_ms=span_ns(ops) / 1e6,
                                        first_cycle=min(op.ks for op in ops), last_cycle=max(op.ke for op in ops),
                                        sdpa_us=[], categories=dict((k, v / 1e6) for k, v in cat.items()))
                # per-user SDPA in layer order (the k-th SDPA launch of each attention layer is user k's segment)
                users = collections.defaultdict(list)
                for op, (li, lt, name, detail) in zip(ops, roles):
                    if name == 'attn.sdpa':
                        users[detail].append(op.k / 1e3)
                per_session[sid]['sdpa_us'] = [median(users[u]) for u in sorted(users)]
                for u in users:
                    sdpa_by_user[u].append(median(users[u]))
                conv_counts.append(collections.Counter(role[0] for op, role in zip(ops, roles)
                                                       if role[2] == 'gdn.conv_gates'))
                for op, (li, lt, name, detail) in zip(ops, roles):
                    if name.startswith('mm.') and name != 'mm.lm_head':
                        key = name if detail is None else '%s.%s' % (name, detail)
                        mm_ops[key].append((op.k, op.cores))
                    elif name == 'mm.lm_head':
                        mm_ops[name].append((op.k, op.cores))
        n = len(per[devices[0]])
        path = skews = 0.0
        for i in range(n):
            durations = [per[d][i].k for d in devices]
            path += max(durations)
            if is_coll(per[devices[0]][i].op):
                skews += max(durations) - min(durations)
                ccl[per[devices[0]][i].op]['min'].append(min(durations))
                ccl[per[devices[0]][i].op]['skew'].append(max(durations) - min(durations))
        critical.append(path / 1e6)
        skew.append(skews / 1e6)
    cats = sorted(set(k for d in devices for k in per_cat[d]), key=lambda k: -median(per_cat[devices[0]][k]))
    total0 = median(totals[devices[0]])
    categories = collections.OrderedDict(
        (k, dict(ms=[round(median(per_cat[d][k]), 4) for d in devices],
                 share=round(100.0 * median(per_cat[devices[0]][k]) / total0, 2) if total0 else 0.0)) for k in cats)
    # per layer means (chip 0, the first complete session)
    per_layer = collections.defaultdict(lambda: collections.defaultdict(float))
    roles0, _, types0 = classify(ref_ops)
    for op, (li, lt, name, detail) in zip(ref_ops, roles0):
        if li is not None:
            per_layer[li][name] += op.k / 1e3
    by_layer_type = {}
    for lt in ('gdn', 'attn'):
        members = [li for li, t in enumerate(types0) if t == lt]
        if not members:
            continue
        keys = sorted(set(k for li in members for k in per_layer[li]))
        by_layer_type[lt] = dict(layers=len(members), us=dict(
            (k, round(statistics.mean(per_layer[li][k] for li in members), 2)) for k in keys),
            total_us=round(statistics.mean(sum(per_layer[li].values()) for li in members), 2))
    # weights
    weights = {}
    table = table or {}
    for key, samples in sorted(mm_ops.items()):
        ns = median([k for k, _ in samples])
        cores = int(median([c for _, c in samples]))
        entry = dict(us=round(ns / 1e3, 2), cores=cores, count=len(samples) // max(1, len(info['complete'])))
        spec = table.get(key)
        if spec:
            k_dim, n_dim, dtype = spec
            tiles = (k_dim // 32) * (n_dim // 32)
            nbytes = tiles * TILE_BYTES[dtype]
            entry.update(dtype=dtype, mbytes=round(nbytes / 1e6, 2), gbps=round(nbytes / ns, 1) if ns else None,
                         ns_per_tile_per_core=round(ns / (tiles / float(cores)), 1) if cores else None,
                         pct_of_dram=round(100.0 * nbytes / ns / DRAM_GBPS, 1) if ns else None)
        weights[key] = entry
    collectives = {}
    for name, series in sorted(ccl.items()):
        collectives[name] = dict(calls_per_replay=len(series['min']) // max(1, len(info['complete'])),
                                 us_min_over_chips=round(median(series['min']) / 1e3, 2),
                                 skew_us=round(median(series['skew']) / 1e3, 2))
    conv = [median(list(c.values())) for c in conv_counts if c]
    return dict(
        ops=info['ops'], devices=devices, layers=layers, gdn_layers=types0.count('gdn'), attn_layers=types0.count('attn'),
        complete_sessions=len(info['complete']), sessions_total=info['sessions_total'],
        kernel_sum_ms=[round(median(totals[d]), 3) for d in devices], span_ms=[round(median(spans[d]), 3) for d in devices],
        gap_ms=[round(median(gap_sums[d]), 3) for d in devices],
        critical_path_ms=round(median(critical), 3), collective_skew_ms=round(median(skew), 3),
        categories=categories, by_layer_type=by_layer_type, weights=weights, collectives=collectives,
        conv_gates_per_gdn_layer=median(conv) if conv else None,
        sdpa_us_by_user=dict((str(u), round(median(v), 2)) for u, v in sorted(sdpa_by_user.items())),
        per_session=per_session, clock_ghz_implied=round(clock_check(ref_ops) or 0.0, 4) or None)


def group_of(category):
    for group, prefixes in GROUPS.items():
        if any(category == prefix or category.startswith(prefix) for prefix in prefixes):
            return group
    return 'norms, adds, input'


def grouped(analysis):
    """{group: ms} of one trace analysis (chip 0)."""
    out = collections.OrderedDict((group, 0.0) for group in GROUPS)
    for name, entry in analysis['categories'].items():
        out[group_of(name)] += entry['ms'][0]
    out['in-trace gaps'] = analysis['gap_ms'][0]
    return collections.OrderedDict((k, round(v, 3)) for k, v in out.items())


# ---- what the round and the log add ----

def sdpa_fit(analysis, rounds):
    """Per-user SDPA µs against that user's context (the segment's position at its round), least squares over every
    live (session, segment) point: intercept µs and slope µs per 1k tokens, or None without positions."""
    points = []
    for sid, entry in analysis['per_session'].items():
        try:
            round_ = rounds.get(int(sid))
        except ValueError:
            round_ = None
        if not round_ or not round_['positions']:
            continue
        for segment_, us in enumerate(entry['sdpa_us']):
            if segment_ in round_['idle'] or segment_ not in round_['positions']:
                continue
            points.append((round_['positions'][segment_] / 1024.0, us))
    if len(points) < 4 or len(set(x for x, _ in points)) < 2:
        return None
    mean_x = statistics.mean(x for x, _ in points)
    mean_y = statistics.mean(y for _, y in points)
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    return dict(points=len(points), intercept_us=round(mean_y - slope * mean_x, 2), slope_us_per_1k=round(slope, 3),
                contexts_k=sorted(set(round(x, 1) for x, _ in points))[:12])


def by_live(analysis, rounds):
    """Median kernel ms of the verify-packed sessions by the live users of their round, and the marginal cost per user."""
    groups = collections.defaultdict(list)
    for sid, entry in analysis['per_session'].items():
        try:
            round_ = rounds.get(int(sid))
        except ValueError:
            round_ = None
        if round_ and round_['live'] is not None:
            groups[round_['live']].append(entry['kernel_ms'])
    table = dict((str(live), dict(sessions=len(v), kernel_ms=round(median(v), 3))) for live, v in sorted(groups.items()))
    lives = sorted(groups)
    marginal = None
    if len(lives) >= 2:
        marginal = round((median(groups[lives[-1]]) - median(groups[lives[0]])) / float(lives[-1] - lives[0]), 3)
    return dict(table=table, marginal_ms_per_live_user=marginal)


def round_map_check(analysis, rounds):
    """How well the session id = round assumption holds: median |device span - host trace_ms| / trace_ms over the sessions."""
    errors = []
    for sid, entry in analysis['per_session'].items():
        try:
            round_ = rounds.get(int(sid))
        except ValueError:
            round_ = None
        if round_ and round_['trace_ms']:
            errors.append(abs(entry['span_ms'] - round_['trace_ms']) / round_['trace_ms'])
    if not errors:
        return dict(compared=0, median_error=None, ok=None)
    err = median(errors)
    return dict(compared=len(errors), median_error=round(err, 4), ok=err <= ROUND_MAP_TOLERANCE)


def round_timeline(analysis, every, trace_id, devices, rounds):
    """Device time between consecutive verify replays of chip 0: the interval from one replay's first kernel start to the
    next's, less the verify's own span, split into the other device work that started inside it (eager and other traced
    kernels) and the idle rest (host fences, dispatch). Per pair of session ids k, k+1; median over the pairs."""
    device = devices[0]
    sessions = sorted((int(sid), e) for sid, e in analysis['per_session'].items() if sid.isdigit())
    rows = every.get(device) or []
    out = []
    for (a, ea), (b, eb) in zip(sessions, sessions[1:]):
        if b != a + 1:
            continue
        start, stop = ea['first_cycle'], eb['first_cycle']
        interval = (stop - start) / CLK_GHZ / 1e6
        other = sum(k for ks, k, trace in rows if start < ks < stop and (trace != trace_id)) / 1e6
        out.append(dict(round=a, interval_ms=interval, verify_ms=ea['span_ms'], other_busy_ms=other,
                        idle_ms=max(0.0, interval - ea['span_ms'] - other),
                        live=(rounds.get(a) or {}).get('live')))
    if not out:
        return dict(pairs=0)
    return dict(pairs=len(out), interval_ms=round(median([o['interval_ms'] for o in out]), 2),
                verify_ms=round(median([o['verify_ms'] for o in out]), 2),
                other_busy_ms=round(median([o['other_busy_ms'] for o in out]), 2),
                idle_ms=round(median([o['idle_ms'] for o in out]), 2),
                best_idle_ms=round(min(o['idle_ms'] for o in out), 2),
                worst_idle_ms=round(max(o['idle_ms'] for o in out), 2))


def project_16_row(single, packed):
    """The lone user's 16-row verify (a T16 draft block) composed from the lone 4-row step and the 64-row block: shared
    categories are interpolated between 4 and 64 rows, per-user categories are one user's (the lone step's own)."""
    if not single or not packed:
        return None
    s, b = single['categories'], packed['categories']
    table = collections.OrderedDict()
    for name in sorted(set(s) | set(b)):
        s_ms = s.get(name, {'ms': [0.0]})['ms'][0]
        b_ms = b.get(name, {'ms': [0.0]})['ms'][0]
        if name in PER_USER:
            table[name] = round(s_ms, 3)
        else:
            table[name] = round(s_ms + (b_ms - s_ms) * (16.0 - 4.0) / (64.0 - 4.0), 3)
    gap = single['gap_ms'][0] + (packed['gap_ms'][0] - single['gap_ms'][0]) * (16.0 - 4.0) / (64.0 - 4.0)
    total = sum(table.values()) + gap
    groups = collections.OrderedDict((g, 0.0) for g in GROUPS)
    for name, ms in table.items():
        groups[group_of(name)] += ms
    return dict(rows=16, method='per-user categories from the 4-row lone step; shared categories interpolated 4 -> 64 rows',
                total_ms=round(total, 2), categories=table,
                groups=collections.OrderedDict((k, round(v, 3)) for k, v in groups.items()),
                projection_ms=PROJECTION_TP4_1U_16ROW_MS)


def compare_with_projection(groups):
    """{group: dict(measured, projected, tp2_c2, tp4_over_tp2)} for the packed 64-row verify."""
    out = collections.OrderedDict()
    for group in GROUPS:
        measured = groups.get(group, 0.0)
        tp2 = TP2_C2_4X32K.get(group)
        out[group] = dict(measured_ms=measured, projected_ms=PROJECTION_TP4_4U_8K.get(group), tp2_c2_ms=tp2,
                          tp4_over_tp2=round(measured / tp2, 2) if tp2 else None)
    return out


def perturbation(gate_json, twin_json, twin_log, log, analysis):
    """The twin's median [PACKED-PHASE] trace_ms against the profiled arm's and the device span: flagged above 3%."""
    profiled = median((log or {}).get('phase_trace_ms') or [])
    twin = median((twin_log or {}).get('phase_trace_ms') or [])
    span = median([e['span_ms'] for e in analysis['per_session'].values()]) if analysis else 0.0
    out = dict(twin_trace_ms=round(twin, 3) if twin else None, profiled_trace_ms=round(profiled, 3) if profiled else None,
               device_span_ms=round(span, 3) if span else None, flagged=None)
    if twin and profiled:
        out['profiled_over_twin'] = round(profiled / twin, 4)
        out['flagged'] = abs(profiled / twin - 1.0) > PERTURBATION_LIMIT
    return out


def texts_of(gate_json):
    return [(stream or {}).get('text') for stream in (gate_json or {}).get('streams') or []]


# ---- the whole analysis ----

def analyse_sessions(sessions, every, columns=(), chips=4, log_text=None, gate_json=None, twin_log=None, twin_json=None,
                     table=None):
    table = weight_table(chips) if table is None else table
    traces = complete_sessions(sessions, chips=None)
    devices_all = sorted(set(device for (_, _, device) in sessions))
    log = parse_log(log_text) if log_text else None
    twin = parse_log(twin_log) if twin_log else None
    problems, notes = [], []
    if len(devices_all) != chips:
        problems.append('%d chips in the CSV (%s), expected %d' % (len(devices_all), ','.join(devices_all), chips))
    listing = []
    signatures = {}
    for trace, info in sorted(traces.items(), key=lambda item: -item[1]['ops']):
        row = dict(trace=trace, ops=info['ops'], sessions=info['sessions_total'], complete=len(info['complete']),
                   truncated=info['truncated'], kind='other')
        sample = next(iter(info['complete'].values()), None)
        if sample is not None:
            kind, detail = trace_signature(sample[info['devices'][0]])
            row.update(kind=kind, detail=detail)
            row['kernel_sum_ms'] = round(sum(op.k for op in sample[info['devices'][0]]) / 1e6, 3)
            signatures[trace] = kind
        listing.append(row)
    packed = [t for t, k in signatures.items() if k == 'verify-packed']
    single = [t for t, k in signatures.items() if k == 'verify-single']
    pick = lambda ids: max(ids, key=lambda t: (len(traces[t]['complete']), traces[t]['ops'])) if ids else None
    packed_id = pick(packed)
    # The lone lane's widths (1, 2, 4 rows) are told apart by their kernel sums (a wider verify takes longer).
    single_sorted = sorted(single, key=lambda t: next(r['kernel_sum_ms'] for r in listing if r['trace'] == t))
    labels = dict((t, w) for t, w in zip(single_sorted, ('w1', 'w2', 'w4'))) if len(single_sorted) == 3 else {}
    for row in listing:
        if row['trace'] in labels:
            row['label'] = 'verify-%s' % labels[row['trace']]
    lone_complete = [t for t in single_sorted if traces[t]['complete']]
    lone_id = lone_complete[-1] if lone_complete else None
    result = dict(traces=listing, chips=devices_all)
    packed_analysis = single_analysis = None
    if packed_id and traces[packed_id]['complete']:
        packed_analysis = analyse_trace(dict(traces[packed_id]), table)
    else:
        problems.append('no verify-64 (packed block) replay is complete on every chip: read the trace list')
    if lone_id and traces[lone_id]['complete']:
        single_analysis = analyse_trace(dict(traces[lone_id]), table)
    rounds = (log or {}).get('rounds') or {}
    if packed_analysis:
        packed_analysis['trace'] = packed_id
        packed_analysis['groups'] = grouped(packed_analysis)
        packed_analysis['sdpa_fit'] = sdpa_fit(packed_analysis, rounds)
        packed_analysis['by_live'] = by_live(packed_analysis, rounds)
        packed_analysis['round_map'] = round_map_check(packed_analysis, rounds)
        packed_analysis['round_timeline'] = round_timeline(packed_analysis, every, packed_id, packed_analysis['devices'],
                                                           rounds)
        packed_analysis['vs_projection'] = compare_with_projection(packed_analysis['groups'])
        packed_analysis['projection_ms'] = PROJECTION_TP4_TRACE_MS
        packed_analysis['tp2_c2_ms'] = TP2_C2_TRACE_MS
        four_live = [sid for sid, entry in packed_analysis['per_session'].items()
                     if (rounds.get(int(sid) if sid.isdigit() else -1) or {}).get('live') == 4]
        packed_analysis['four_live_sessions'] = len(four_live) if rounds else None
        enough = len(four_live) if rounds and any(r['live'] is not None for r in rounds.values()) \
            else packed_analysis['complete_sessions']
        if enough < MIN_SESSIONS:
            notes.append('%d complete 4-live verify-64 sessions on all chips, fewer than the %d asked for: the figures '
                         'below are medians over few samples' % (enough, MIN_SESSIONS))
        if packed_analysis['round_map']['ok'] is False:
            notes.append('the session id = round assumption is doubtful (median device-span error %.0f%% against the '
                         'host trace_ms): per-round attachments (live count, contexts) may be off' %
                         (100 * packed_analysis['round_map']['median_error']))
    if single_analysis:
        single_analysis['trace'] = lone_id
        single_analysis['groups'] = grouped(single_analysis)
        if single_analysis['complete_sessions'] < MIN_SESSIONS:
            notes.append('%d complete 4-row lone-step sessions, fewer than the %d asked for' % (
                single_analysis['complete_sessions'], MIN_SESSIONS))
    else:
        notes.append('no complete lone-user (single-SDPA) verify replay: no 4-row step and no 16-row projection')
    result['verify_packed'] = packed_analysis
    result['verify_lone'] = single_analysis
    result['single_user_labels'] = dict((t, 'w%s' % w[1:]) for t, w in labels.items())
    result['lane_16_row'] = project_16_row(single_analysis, packed_analysis)
    # what the log says
    if log is None:
        notes.append('no server log: no round mapping, no read-back count, no audit check')
    else:
        if log['audits']:
            problems.append('the server log carries verify audit lines (%s): the trace is the audited one, not the timed '
                            'one' % '; '.join(log['audits']))
        if log['readbacks'] < MIN_READBACKS:
            notes.append('%d read-back lines (%s), fewer than %d' % (log['readbacks'], READBACK_MARKER, MIN_READBACKS))
        if log['drops']:
            notes.append('%d log lines about dropped or lost profiler markers: some windows are truncated' % log['drops'])
    configuration = (gate_json or {}).get('qwen_configuration')
    if gate_json is not None:
        if configuration is None:
            notes.append('the gate report records no configuration: the launched argv is unshown')
        else:
            for name, want in (('QWEN_FAST_TP', '4'), ('QWEN_FAST_VERIFY_T1_AUDIT', '0'), ('QWEN_FAST_VERIFY_T2_AUDIT', '0')):
                if str(configuration.get(name)) != want:
                    problems.append('the launched configuration has %s=%r, not %s' % (name, configuration.get(name), want))
    if twin_json is not None and gate_json is not None:
        a, b = texts_of(gate_json), texts_of(twin_json)
        result['texts_identical'] = (a == b) if a and b else None
        if result['texts_identical'] is False:
            problems.append('ops-trace texts differ from ops-twin texts: profiling changed the arithmetic (a finding)')
    result['perturbation'] = perturbation(gate_json, twin_json, twin, log, packed_analysis)
    if result['perturbation'].get('flagged'):
        notes.append('profiling perturbed the round: profiled trace_ms is %.1f%% of the twin\'s' %
                     (100 * result['perturbation']['profiled_over_twin']))
    result['validity'] = dict(ok=not problems, problems=problems, notes=notes,
                              readbacks=(log or {}).get('readbacks'), drops=(log or {}).get('drops'),
                              packed_sessions=packed_analysis['complete_sessions'] if packed_analysis else 0,
                              lone_sessions=single_analysis['complete_sessions'] if single_analysis else 0)
    return result


def analyse_files(csv_path, server_log=None, gate_json=None, twin_log=None, twin_json=None, chips=4):
    sessions, every, columns = load(csv_path)
    return analyse_sessions(sessions, every, columns, chips=chips, log_text=read_text(server_log),
                            gate_json=read_json(gate_json), twin_log=read_text(twin_log), twin_json=read_json(twin_json))


# ---- output ----

def fmt(value, digits=2):
    return '-' if value is None else ('%.*f' % (digits, value) if isinstance(value, float) else str(value))


def render_markdown(report):
    lines = ['# TP4 op profile: the timed verify', '']
    validity = report.get('validity') or {}
    lines.append('Validity: %s. %d complete verify-64 sessions and %d complete lone-step sessions on %d chips.' % (
        'OK' if validity.get('ok') else 'PROBLEMS', validity.get('packed_sessions', 0), validity.get('lone_sessions', 0),
        len(report.get('chips') or [])))
    for problem in validity.get('problems') or []:
        lines.append('- PROBLEM: %s' % problem)
    for note in validity.get('notes') or []:
        lines.append('- note: %s' % note)
    lines += ['', '## Traces (complete sessions only)', '',
              '| trace | kind | ops | sessions | complete | kernel sum ms |', '|---|---|---:|---:|---:|---:|']
    for row in report.get('traces') or []:
        if row['ops'] < 20 and row['kind'] in ('small', 'other'):
            continue
        lines.append('| %s | %s | %d | %d | %d | %s |' % (row['trace'], row.get('label') or row['kind'], row['ops'],
                                                          row['sessions'], row['complete'], fmt(row.get('kernel_sum_ms'))))
    packed = report.get('verify_packed')
    if packed:
        lines += ['', '## The packed 64-row verify (trace %s, %d layers: %d GDN, %d attention)' % (
            packed['trace'], packed['layers'], packed['gdn_layers'], packed['attn_layers']), '',
            'Kernel sum %s ms, device span %s ms, in-trace gaps %s ms (per chip); cross-chip critical path %s ms, '
            'collective skew %s ms. Projection %.1f ms, TP2 C2 %.1f ms.' % (
                '/'.join(fmt(v) for v in packed['kernel_sum_ms']), '/'.join(fmt(v) for v in packed['span_ms']),
                '/'.join(fmt(v) for v in packed['gap_ms']), fmt(packed['critical_path_ms']),
                fmt(packed['collective_skew_ms']), packed['projection_ms'], packed['tp2_c2_ms']), '',
            '| group | measured ms | projected | TP2 C2 | TP4 / TP2 |', '|---|---:|---:|---:|---:|']
        for group, entry in packed['vs_projection'].items():
            lines.append('| %s | %s | %s | %s | %s |' % (group, fmt(entry['measured_ms']), fmt(entry['projected_ms']),
                                                         fmt(entry['tp2_c2_ms']), fmt(entry['tp4_over_tp2'])))
        lines.append('| in-trace gaps | %s | 2.8 | 2.8 | - |' % fmt(packed['groups'].get('in-trace gaps')))
        lines += ['', '| category | ms per chip | share |', '|---|---|---:|']
        for name, entry in packed['categories'].items():
            lines.append('| %s | %s | %s%% |' % (name, ' / '.join(fmt(v) for v in entry['ms']), fmt(entry['share'], 1)))
        for lt, entry in packed['by_layer_type'].items():
            lines += ['', 'Per %s layer (%d layers, %s us): %s' % (lt, entry['layers'], fmt(entry['total_us'], 1),
                                                                   ', '.join('%s %s' % (k, fmt(v, 1)) for k, v in sorted(
                                                                       entry['us'].items(), key=lambda kv: -kv[1])))]
        lines += ['', '### Weight matmuls (assumed dtypes; ns per tile per core; DRAM %.0f GB/s)' % DRAM_GBPS, '',
                  '| matmul | us | cores | dtype | MB | GB/s | % of DRAM | ns/tile/core |', '|---|---:|---:|---|---:|---:|---:|---:|']
        for key, entry in packed['weights'].items():
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
                key, fmt(entry['us']), entry['cores'], entry.get('dtype', '-'), fmt(entry.get('mbytes')),
                fmt(entry.get('gbps'), 1), fmt(entry.get('pct_of_dram'), 1), fmt(entry.get('ns_per_tile_per_core'), 1)))
        lines += ['', '### Collectives (per call: minimum over chips = intrinsic, skew = waiting on the slowest chip)', '',
                  '| op | calls per replay | us (min over chips) | skew us |', '|---|---:|---:|---:|']
        for name, entry in packed['collectives'].items():
            lines.append('| %s | %d | %s | %s |' % (name, entry['calls_per_replay'], fmt(entry['us_min_over_chips']),
                                                    fmt(entry['skew_us'])))
        if packed.get('sdpa_fit'):
            fit = packed['sdpa_fit']
            lines += ['', 'SDPA against context (%d points, contexts %s k): %s us fixed + %s us per 1k tokens.' % (
                fit['points'], fit['contexts_k'], fmt(fit['intercept_us']), fmt(fit['slope_us_per_1k'], 3))]
        lines += ['', 'SDPA us by user in the block (segment order): %s.' % ', '.join(
            '%s: %s' % (u, fmt(v)) for u, v in packed['sdpa_us_by_user'].items())]
        live = packed['by_live']
        if live['table']:
            lines += ['', 'Kernel ms by live users: %s; marginal %s ms per live user.' % (
                ', '.join('%s live: %s (%d)' % (k, fmt(v['kernel_ms']), v['sessions']) for k, v in live['table'].items()),
                fmt(live['marginal_ms_per_live_user']))]
        timeline = packed['round_timeline']
        if timeline.get('pairs'):
            lines += ['', 'Device time between verify replays (%d consecutive pairs): interval %s ms = verify %s + other '
                      'device work %s + idle %s (best %s, worst %s; an interval that holds a profiler read-back is inflated, so read the best).' % (
                          timeline['pairs'], fmt(timeline['interval_ms']), fmt(timeline['verify_ms']),
                          fmt(timeline['other_busy_ms']), fmt(timeline['idle_ms']), fmt(timeline['best_idle_ms']),
                          fmt(timeline['worst_idle_ms']))]
    lone = report.get('verify_lone')
    if lone:
        lines += ['', '## The lone-user verify (trace %s; the 4-row step)' % lone['trace'], '',
                  'Kernel sum %s ms, span %s ms, gaps %s ms.' % ('/'.join(fmt(v) for v in lone['kernel_sum_ms']),
                                                                  '/'.join(fmt(v) for v in lone['span_ms']),
                                                                  '/'.join(fmt(v) for v in lone['gap_ms']))]
    lane = report.get('lane_16_row')
    if lane:
        lines += ['', '## Projected lone 16-row verify: %s ms (research estimate %.1f)' % (fmt(lane['total_ms']),
                                                                                             lane['projection_ms']), '',
                  lane['method'] + '.', '', '| group | ms |', '|---|---:|']
        for group, ms in lane['groups'].items():
            lines.append('| %s | %s |' % (group, fmt(ms)))
    perturb = report.get('perturbation') or {}
    lines += ['', '## Profiling overhead', '',
              'Twin trace_ms %s, profiled trace_ms %s, device span %s; texts identical: %s.' % (
                  fmt(perturb.get('twin_trace_ms')), fmt(perturb.get('profiled_trace_ms')), fmt(perturb.get('device_span_ms')),
                  report.get('texts_identical'))]
    return '\n'.join(lines) + '\n'


def artifact_paths(results):
    """The files of a downloaded gate/ directory: (csv, trace log, trace json, twin log, twin json)."""
    def first(*names):
        for name in names:
            path = os.path.join(results, *name.split('/'))
            if os.path.isfile(path):
                return path
        return None
    return dict(csv_path=first('ops/cpp_device_perf_report.csv.gz', 'ops/cpp_device_perf_report.csv'),
                server_log=first('ops-trace/server.log'), gate_json=first('ops-trace/m3native-gate.json'),
                twin_log=first('ops-twin/server.log'), twin_json=first('ops-twin/m3native-gate.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='?', help='cpp_device_perf_report.csv or .csv.gz')
    parser.add_argument('--results', help='a downloaded gate/ directory (the artifact layout)')
    parser.add_argument('--server-log')
    parser.add_argument('--gate-json')
    parser.add_argument('--twin-log')
    parser.add_argument('--twin-json')
    parser.add_argument('--chips', type=int, default=4)
    parser.add_argument('--out', help='directory for tp4-profile-report.json and .md (default: beside the CSV)')
    options = parser.parse_args(argv)
    paths = artifact_paths(options.results) if options.results else {}
    csv_path = options.csv or paths.get('csv_path')
    if not csv_path:
        parser.error('give the CSV, or --results with ops/cpp_device_perf_report.csv.gz under it')
    report = analyse_files(csv_path, server_log=options.server_log or paths.get('server_log'),
                           gate_json=options.gate_json or paths.get('gate_json'),
                           twin_log=options.twin_log or paths.get('twin_log'),
                           twin_json=options.twin_json or paths.get('twin_json'), chips=options.chips)
    out = options.out or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'tp4-profile-report.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
    text = render_markdown(report)
    with open(os.path.join(out, 'tp4-profile-report.md'), 'w', encoding='utf-8') as handle:
        handle.write(text)
    sys.stdout.write(text)
    return 0 if report['validity']['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
