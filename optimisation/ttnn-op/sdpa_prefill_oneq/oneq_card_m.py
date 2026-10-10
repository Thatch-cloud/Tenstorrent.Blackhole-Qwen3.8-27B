"""Card-M proof and timing of the [QWEN-SDPA-PF] oneq lever (apply_factory_ps.py; graft K64j-OQ from build_k64j_oq.sh).

The served causal chunked paged SDPA at the TP4 shape (the model's forward_prefill_paged call: 6 local Q heads, 1 KV head, head dim
256, q/k chunk 128, bf8 Q/K/V, HiFi2 with fp32 dest, the flexible path with a device chunk_start tensor, a 4096-block page table =
the 8 x 262k serving pool) in four programs, on ONE card of the mesh:

    stock      no program word                          the stock factory path (what the chain was qualified against)
    served     0x5EFA0003                               the production chain: 48 busy cores x 2 q chunks, 8 chains of 6
    oneq       0x5EFA000B                               ONE q chunk per core: 96 busy cores, 16 chains of 6   (the lever)
    oneq_noc   0x5EFA000F                               oneq with the chain ordered by NoC distance (a diagnostic arm)

BYTES (the proof): every arm's output, sha256 of the int16 view of ttnn.to_torch, must equal the stock path's at every
(rows 2048/1024/512) x (Q memory dram/l1) x (seed) x (Q variant normal/peaky) x (chunk_start 0 ... 251,904, i.e. contexts 2k to 254k)
case. Also: TP2's head counts (12 Q heads, 2 KV heads) at 1024 and 512 rows (96 and 48 q chunks fit the grid), the refusals (TT_FATAL,
never a hang: oneq at TP2's 2048 rows = 192 chunks on 130 cores; the bare 0x8 word; a legacy int chunk_start; bf16 K/V; an odd chunk
count), the program cache (the oneq word keys its own program; a chunk_start change adds none), 400 alternating served / oneq calls
against the stock sha, and an 11 x 10 grid pass on a 13 x 10 device (the same bytes on the grid the stack ran before the unlock).

TIME: per (arm, chunk_start) the median of interleaved single calls bracketed by ttnn.synchronize_device (the pattern of
sdpa_prefill_bench.py), one call = one attention layer; oneq_report.py turns the medians into the estimated-vs-measured table.

The factory log is checked against the planner: one '[QWEN-SDPA-PF] flags=... chains=.. members=..' line per chain program and, for
oneq programs, one '[QWEN-SDPA-PF] oneq=1 q_chunks=.. cores=.. chunks_per_core=1 ...' line (16 chains of 96 members at TP4 2048 rows).

Everything the C++ side prints goes to <out>.native.log. The helpers above run() import no ttnn and are unit-tested on CPU
(scripts/ci/test_sdpa_oneq_card.py, with a fake ttnn).

    python3 -B oneq_card_m.py --out /results/oneq.json [--expect-binary-sha256 <sha>]
"""

import argparse
import collections
import contextlib  # noqa: F401 - used by callers that redirect stdout in tests
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oneq_planner as planner  # noqa: E402
import oneq_report as report_lib  # noqa: E402

GEOMETRIES = {'tp4': dict(nqh=6, nkh=1), 'tp2': dict(nqh=12, nkh=2)}
HD, BLOCK, Q_CHUNK = 256, 64, 128
POOL_BLOCKS = 4096                      # 262,144 tokens: the 8 x 262k serving page table
TP2_POOL_BLOCKS = 1152                  # up to 73,728 tokens: the TP2 geometry's rows 1024 / 512 checks
PF_TAG = 0x5EFA0000
ARMS = collections.OrderedDict([('stock', None), ('served', 0x3), ('oneq', 0xB), ('oneq_noc', 0xF)])
ROWS = (2048, 1024, 512)
STARTS = (0, 128, 1920, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 129024, 163840, 196608, 229376, 251904)
TIME_STARTS = (0, 2048, 8192, 32768, 65536, 129024, 196608, 251904)
TP2_ROWS = (1024, 512)
TP2_STARTS = (0, 2048, 65536 - 2048)
SEEDS = (0, 1)
VARIANTS = ('normal', 'peaky')
Q_MEMORY = ('dram', 'l1')
GRID_STARTS = (0, 2048, 65536, 251904)
SERVED_WORD = PF_TAG | 0x3
ONEQ_MARKER = b'[QWEN-SDPA-PF] oneq needs one q chunk per core'
CHAIN_MARKER = b'[QWEN-SDPA-PF] flags='
FLAGS_LINE = re.compile(r'\[QWEN-SDPA-PF\] flags=(0x[0-9a-f]+) kv_chain=1 chains=([0-9]+) members=([0-9]+) order=([a-z]+)')
ONEQ_LINE = re.compile(r'\[QWEN-SDPA-PF\] oneq=1 q_chunks=([0-9]+) cores=([0-9]+) chunks_per_core=1 chains=([0-9]+) members=([0-9]+)')
Q1_NAME = 'test_sdpa_prefill_chain_card_m.py'
WIN_RATIO, PARTIAL_RATIO, NOC_GAIN = 0.75, 0.95, 0.02


def load_q1():
    """The Q1 harness module (its Watchdog, NativeLog and digest helpers), from beside this file, the chain directory, or the container's mounts."""
    for directory in (HERE, HERE.parent / 'sdpa_prefill_chain', Path('/bench_pf'), Path('/bench')):
        path = directory / Q1_NAME
        if path.is_file():
            spec = importlib.util.spec_from_file_location('pf_card_q1', str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError('%s not found beside %s, in sdpa_prefill_chain or under /bench_pf' % (Q1_NAME, HERE))


Q1 = load_q1()
WATCHDOG = Q1.Watchdog(0)
COMPILE_GRACE_S = Q1.COMPILE_GRACE_S


# ---------------------------------------------------------------------------------------------
# Host helpers (no ttnn): the matrix, words, labels, the factory-log check, verdicts.
# ---------------------------------------------------------------------------------------------

def word(flags):
    return None if flags is None else PF_TAG | flags


def blocks_needed(start, rows, block=BLOCK):
    return -(-(start + rows) // block)


def validate_starts(starts, rows, pool_blocks=POOL_BLOCKS):
    for start in starts:
        if start < 0 or start % Q_CHUNK or start % BLOCK:
            raise ValueError('chunk_start %d must be a non-negative multiple of %d' % (start, Q_CHUNK))
        if blocks_needed(start, rows) > pool_blocks:
            raise ValueError('chunk_start %d + rows %d exceeds the %d-block pool' % (start, rows, pool_blocks))


def case_label(rows, qmem, seed, variant, start, geometry='tp4', grid=None):
    return '%s-r%d-%s-s%d-%s%s@%d' % (geometry, rows, qmem, seed, variant, '' if grid is None else '-g%dx%d' % grid, start)


def oneq_refusal(geometry, rows, grid):
    """The planner's reading of the factory: the TT_FATAL text a oneq call must raise, or None when it runs; the envelope for odd chunk counts."""
    nqh, nkh = GEOMETRIES[geometry]['nqh'], GEOMETRIES[geometry]['nkh']
    if (rows // Q_CHUNK) % 2:
        return 'kv_chain outside its qualified envelope'
    return planner.plan(nqh, nkh, rows, Q_CHUNK, grid, oneq=True)['refusal']


def expected_programs(requested):
    """Counters of the log lines a set of built chain programs must have produced.

    requested: {(geometry, rows, width, qmem, flags, grid)}. -> (Counter of (flags, chains, members, order) for the F4 line, Counter of
    (q_chunks, cores, chains, members) for the oneq line). One program, one factory run, one line each."""
    flags_lines, oneq_lines = collections.Counter(), collections.Counter()
    for geometry, rows, _width, _qmem, flags, grid in requested:
        nqh, nkh = GEOMETRIES[geometry]['nqh'], GEOMETRIES[geometry]['nkh']
        oneq = bool(flags & planner.FLAG_ONEQ)
        the_plan = planner.plan(nqh, nkh, rows, Q_CHUNK, grid, oneq=oneq, noc_order=bool(flags & planner.FLAG_NOC_ORDER))
        flags_lines[(flags, the_plan['chain_count'], the_plan['member_count'], 'noc' if flags & planner.FLAG_NOC_ORDER else 'raster')] += 1
        if oneq:
            oneq_lines[(the_plan['total'], the_plan['num_cores'], the_plan['chain_count'], the_plan['member_count'])] += 1
    return flags_lines, oneq_lines


def factory_lines(text):
    flags = [(int(m.group(1), 16), int(m.group(2)), int(m.group(3)), m.group(4)) for m in FLAGS_LINE.finditer(text)]
    oneq = [tuple(int(value) for value in m.groups()) for m in ONEQ_LINE.finditer(text)]
    return flags, oneq


def check_factory_lines(text, requested):
    """[problem] comparing the native log's factory lines with the planner's reading of every chain program the run built."""
    flags, oneq = factory_lines(text)
    want_flags, want_oneq = expected_programs(requested)
    problems = []
    for name, got, want in (('flags line', collections.Counter(flags), want_flags), ('oneq line', collections.Counter(oneq), want_oneq)):
        if got != want:
            extra, missing = got - want, want - got
            problems.append('%s: unexpected %s, missing %s' % (name, dict(extra) or 'none', dict(missing) or 'none'))
    return problems


def loaded_binary(maps='/proc/self/maps'):
    """(path, sha256, has the chain factory, has the oneq edits) of the one _ttnncpp.so this process mapped."""
    import hashlib

    paths = sorted({line.split()[-1] for line in Path(maps).read_text().splitlines() if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    data = Path(paths[0]).read_bytes()
    return dict(path=paths[0], sha256=hashlib.sha256(data).hexdigest(), chain=CHAIN_MARKER in data, oneq=ONEQ_MARKER in data)


def timing_verdict(timing, rows=2048):
    """The timing rule, per Q memory: OQ-WIN when the median oneq / served ratio over the starts of 32k and beyond is at most 0.75,
    OQ-PARTIAL up to 0.95, OQ-NO-WIN above (or no data). noc: 'oneq_noc' replaces 'oneq' only when it is at least 2 percent faster."""
    out = {}
    for qmem, arms in (timing.get(str(rows)) or {}).items():
        served = {int(s): v['median_ms'] for s, v in (arms.get('served') or {}).items() if v.get('median_ms') is not None}
        entry = dict(ratio=None, verdict='OQ-NO-WIN', best='oneq')
        for name in ('oneq', 'oneq_noc'):
            mine = {int(s): v['median_ms'] for s, v in (arms.get(name) or {}).items() if v.get('median_ms') is not None}
            ratios = [mine[s] / served[s] for s in sorted(set(mine) & set(served)) if s >= 32768]
            if ratios:
                entry[name] = statistics.median(ratios)
        if 'oneq' in entry:
            ratio = entry['oneq']
            if 'oneq_noc' in entry and entry['oneq_noc'] < entry['oneq'] * (1 - NOC_GAIN):
                ratio, entry['best'] = entry['oneq_noc'], 'oneq_noc'
            entry['ratio'] = ratio
            entry['verdict'] = 'OQ-WIN' if ratio <= WIN_RATIO else 'OQ-PARTIAL' if ratio <= PARTIAL_RATIO else 'OQ-NO-WIN'
        out[qmem] = entry
    return out


def schedule(arms, starts, rounds):
    """[(arm, start)] in interleaved rounds, the arm order rotated per round and per start (card drift spreads over every arm alike)."""
    out = []
    for round_ in range(rounds):
        for index, start in enumerate(starts):
            shift = (round_ + index) % len(arms)
            out.extend((arm, start) for arm in arms[shift:] + arms[:shift])
    return out


# ---------------------------------------------------------------------------------------------
# Device part: the qualification card only.
# ---------------------------------------------------------------------------------------------

class Bench:
    """One device: the K/V pool per seed, page tables, queries per (rows, seed, variant, qmem), chunk_start tensors; and the call itself."""

    def __init__(self, ttnn, torch, device, geometry='tp4', pool_blocks=POOL_BLOCKS, requested=None):
        self.ttnn, self.torch, self.device = ttnn, torch, device
        self.geometry, self.pool_blocks = geometry, pool_blocks
        self.nqh, self.nkh = GEOMETRIES[geometry]['nqh'], GEOMETRIES[geometry]['nkh']
        self.requested = requested if requested is not None else set()
        self.pools, self.tables, self.queries, self.starts = {}, {}, {}, {}
        size = device.compute_with_storage_grid_size()
        self.native_grid = (size.x, size.y)

    def upload(self, host, dtype, memory='dram'):
        ttnn = self.ttnn
        config = ttnn.L1_MEMORY_CONFIG if memory == 'l1' else ttnn.DRAM_MEMORY_CONFIG
        layout = ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT
        with WATCHDOG.op('upload %s' % (tuple(host.shape),), extra=COMPILE_GRACE_S):
            return ttnn.from_torch(host, dtype=dtype, layout=layout, device=self.device, memory_config=config)

    def pool(self, seed, dtype='bf8'):
        key = (seed, dtype)
        if key not in self.pools:
            generator = self.torch.Generator().manual_seed(1000 + seed)
            shape = (self.pool_blocks, self.nkh, BLOCK, HD)
            k = self.torch.randn(shape, generator=generator).to(self.torch.bfloat16)
            v = self.torch.randn(shape, generator=generator).to(self.torch.bfloat16)
            kind = self.ttnn.bfloat8_b if dtype == 'bf8' else self.ttnn.bfloat16
            self.pools[key] = (self.upload(k, kind), self.upload(v, kind))
        return self.pools[key]

    def table(self, seed):
        if seed not in self.tables:
            generator = self.torch.Generator().manual_seed(2000 + seed)
            row = self.torch.randperm(self.pool_blocks, generator=generator)
            self.tables[seed] = self.upload(row.to(self.torch.int32).reshape(1, self.pool_blocks), self.ttnn.int32)
        return self.tables[seed]

    def query(self, rows, seed, variant, qmem):
        key = (rows, seed, variant, qmem)
        if key not in self.queries and qmem == 'l1':
            for old in [other for other in self.queries if other[3] == 'l1']:      # at most one L1 Q alive beside the SDPA CBs
                self.ttnn.deallocate(self.queries.pop(old))
        if key not in self.queries:
            generator = self.torch.Generator().manual_seed(3000 + 17 * seed + rows)
            host = self.torch.randn((1, self.nqh, rows, HD), generator=generator)
            if variant == 'peaky':
                host = host * 8
            self.queries[key] = self.upload(host.to(self.torch.bfloat16), self.ttnn.bfloat8_b, qmem)
        return self.queries[key]

    def start(self, value):
        if value not in self.starts:
            self.starts[value] = self.upload(self.torch.tensor([value], dtype=self.torch.int32), self.ttnn.int32)
        return self.starts[value]

    def config(self, program_word=None, grid=None):
        options = dict(compute_with_storage_grid_size=grid or self.native_grid, exp_approx_mode=False, q_chunk_size=Q_CHUNK,
                       k_chunk_size=Q_CHUNK)
        if program_word is not None:
            options['max_cores_per_head_batch'] = program_word
        return self.ttnn.SDPAProgramConfig(**options)

    def compute(self):
        return self.ttnn.WormholeComputeKernelConfig(math_fidelity=self.ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                                     fp32_dest_acc_en=True, packer_l1_acc=True)

    def call(self, rows, qmem, seed, variant, start, program_word=None, *, grid=None, legacy_start=False, kv_bf16=False,
             record=True, label=None, grace=COMPILE_GRACE_S):
        ttnn = self.ttnn
        k, v = self.pool(seed, 'bf16' if kv_bf16 else 'bf8')
        q = self.query(rows, seed, variant, qmem)
        pages = self.table(seed)
        what = label or ('sdpa %s r%d %s start=%d word=%s' % (self.geometry, rows, qmem, start,
                                                              'stock' if program_word is None else hex(program_word)))
        common = dict(input_tensor_q=q, input_tensor_k=k, input_tensor_v=v, page_table_tensor=pages,
                      compute_kernel_config=self.compute(), program_config=self.config(program_word, grid))
        with WATCHDOG.op(what, extra=grace):
            if legacy_start:
                out = ttnn.transformer.chunked_scaled_dot_product_attention(chunk_start_idx=start, **common)
            else:
                out = ttnn.transformer.chunked_scaled_dot_product_attention(chunk_start_idx_tensor=self.start(start), **common)
        if record and program_word is not None and (program_word & 0xFFFF0000) == PF_TAG:
            self.requested.add((self.geometry, rows, self.pool_blocks, qmem, program_word & 0xFFFF, grid or self.native_grid))
        return out

    def host(self, tensor, label='read back'):
        with WATCHDOG.op(label):
            result = self.ttnn.to_torch(tensor)
        self.ttnn.deallocate(tensor)
        return result

    def sha(self, *args, **kwargs):
        return Q1.digest(self.torch, self.host(self.call(*args, **kwargs)))

    def cache_entries(self):
        counter = getattr(self.device, 'num_program_cache_entries', None)
        return counter() if callable(counter) else None


def arm_list(args, geometry, rows, grid):
    """The arms to run at a shape: oneq arms are skipped (and the refusal expected) when the planner says the factory refuses them."""
    refused = oneq_refusal(geometry, rows, grid)
    return [name for name in args.arms if not (refused and name.startswith('oneq'))]


def sweep(bench, args, report):
    """Every arm equals the stock path, at every case. TP4 first."""
    failures = report['failures']
    for rows in args.rows:
        validate_starts(args.starts, rows)
        arms = arm_list(args, 'tp4', rows, bench.native_grid)
        for qmem in args.q_memory:
            for seed in args.seeds:
                for variant in args.variants:
                    for start in args.starts:
                        shas = {name: bench.sha(rows, qmem, seed, variant, start, word(ARMS[name])) for name in arms}
                        label = case_label(rows, qmem, seed, variant, start)
                        reference = shas.get('stock')
                        differing = sorted(name for name, value in shas.items() if value != reference)
                        report['cases'].append(dict(case=label, rows=rows, context=start + rows, exact=not differing,
                                                    differing=differing, sha256=reference))
                        if differing:
                            failures.append('%s: %s differ from the stock path' % (label, ', '.join(differing)))
            print('rows %d %s: %d cases so far, %d failures' % (rows, qmem, len(report['cases']), len(failures)), flush=True)


def tp2_checks(bench, args, report):
    """TP2's head counts (12 Q heads, 2 KV heads): oneq at 1024 and 512 rows runs and equals stock; 2048 rows (192 chunks) is refused."""
    failures = report['failures']
    for rows in TP2_ROWS:
        validate_starts(TP2_STARTS, rows, bench.pool_blocks)
        arms = arm_list(args, 'tp2', rows, bench.native_grid)
        for start in TP2_STARTS:
            shas = {name: bench.sha(rows, 'dram', 0, 'normal', start, word(ARMS[name])) for name in arms}
            label = case_label(rows, 'dram', 0, 'normal', start, 'tp2')
            reference = shas.get('stock')
            differing = sorted(name for name, value in shas.items() if value != reference)
            report['tp2_cases'].append(dict(case=label, rows=rows, exact=not differing, differing=differing, arms=arms))
            if differing:
                failures.append('%s: %s differ from the stock path' % (label, ', '.join(differing)))
    print('tp2 geometry: %d cases, %d failures so far' % (len(report['tp2_cases']), len(failures)), flush=True)


def grid_checks(bench, args, report):
    """The same bytes on a smaller grid than the device's (an 11 x 10 grid on a 13 x 10 device): oneq / served / stock at a few starts."""
    failures = report['failures']
    for grid in args.grid_list:
        if grid == bench.native_grid:
            continue
        if grid[0] > bench.native_grid[0] or grid[1] > bench.native_grid[1]:
            failures.append('grid %dx%d is larger than the device grid %dx%d' % (grid + bench.native_grid))
            continue
        arms = arm_list(args, 'tp4', 2048, grid)
        for start in GRID_STARTS:
            shas = {name: bench.sha(2048, 'dram', 0, 'normal', start, word(ARMS[name]), grid=grid) for name in arms}
            reference = shas.get('stock')
            differing = sorted(name for name, value in shas.items() if value != reference)
            label = case_label(2048, 'dram', 0, 'normal', start, 'tp4', grid)
            report['grid_cases'].append(dict(case=label, exact=not differing, differing=differing))
            if differing:
                failures.append('%s: %s differ from the stock path' % (label, ', '.join(differing)))
    print('grid passes: %d cases, %d failures so far' % (len(report['grid_cases']), len(failures)), flush=True)


def cache_checks(bench, args, report):
    """Served then oneq at one fresh shape adds exactly one program (the word keys its own program); a chunk_start change adds none."""
    shape = (2048, 'dram', 0, 'normal')
    bench.pool(0)
    bench.table(0)
    bench.query(shape[0], shape[2], shape[3], shape[1])
    starts = [s for s in (0, 2048, 65536, 128) if blocks_needed(s, 2048) <= bench.pool_blocks]
    for start in starts:
        bench.start(start)
    before = bench.cache_entries()
    bench.host(bench.call(*shape, starts[0], word(ARMS['served'])))
    after_served = bench.cache_entries()
    bench.host(bench.call(*shape, starts[0], word(ARMS['oneq'])))
    after_oneq = bench.cache_entries()
    for start in starts[1:]:
        bench.host(bench.call(*shape, start, word(ARMS['oneq'])))
    after_starts = bench.cache_entries()
    result = dict(before=before, after_served=after_served, after_oneq=after_oneq, after_starts=after_starts)
    report['cache'] = result
    print('program cache %r' % (result,), flush=True)
    if None in result.values():
        report['failures'].append('the device has no num_program_cache_entries(): the oneq word keying its own program is unproven')
        return
    if after_oneq != after_served + 1:
        report['failures'].append('cache: the oneq word did not key its own program (cache %d -> %d)' % (after_served, after_oneq))
    if after_starts != after_oneq:
        report['failures'].append('cache: a chunk_start change rebuilt the oneq program (cache %d -> %d)' % (after_oneq, after_starts))


REFUSALS = (
    # (name, geometry, call changes, word flags, rows, needle)
    ('bare 0x8 (no chain bit)', 'tp4', {}, 0x8, 2048, 'unknown or incomplete flags'),
    ('oneq at TP2 2048 rows (192 chunks on the grid)', 'tp2', {}, 0xB, 2048, 'oneq needs one q chunk per core'),
    ('oneq with a legacy int chunk_start', 'tp4', dict(legacy_start=True), 0xB, 2048, 'kv_chain outside its qualified envelope'),
    ('oneq with bf16 K/V', 'tp4', dict(kv_bf16=True), 0xB, 2048, 'kv_chain outside its qualified envelope'),
    ('oneq at an odd chunk count (1920 rows)', 'tp4', {}, 0xB, 1920, 'kv_chain outside its qualified envelope'),
    ('the served chain at an odd chunk count (1920 rows)', 'tp4', {}, 0x3, 1920, 'kv_chain outside its qualified envelope'),
)


def refusals(bench, tp2_bench, args, report):
    """Every refusal is a TT_FATAL raised at program build (a Python exception), never a hang. Run BEFORE the sweep builds these programs:
    the factory checks only on a program-cache miss."""
    results = {}
    for name, geometry, change, flags, rows, needle in REFUSALS:
        target = tp2_bench if geometry == 'tp2' else bench
        change = dict(change)
        try:
            out = target.call(rows, 'dram', 0, 'normal', 2048, PF_TAG | flags, record=False, label='refusal %s' % name, **change)
        except Exception as error:  # noqa: BLE001 - the TT_FATAL surfaces as a RuntimeError
            text = str(error)
            results[name] = dict(refused=True, matched=needle in text, message=text[:400])
        else:
            target.ttnn.deallocate(out)
            results[name] = dict(refused=False, matched=False, message='returned an output')
        print('refusal %-52s %s' % (name, results[name]), flush=True)
        if not (results[name]['refused'] and results[name]['matched']):
            report['failures'].append('refusal: %s was not refused by its [QWEN-SDPA-PF] TT_FATAL (%s): %s'
                                      % (name, needle, results[name]['message']))
    report['refusals'] = results


def stress(bench, args, report):
    """Alternating served / oneq / oneq_noc calls at random starts, each equal to the stock sha of its start (taken first, eagerly)."""
    shape = (2048, 'dram', 0, 'normal')
    expected = {start: bench.sha(*shape, start) for start in args.stress_starts}
    rng = random.Random(7)
    names = ('served', 'oneq', 'oneq', 'oneq_noc')
    drift = []
    for index in range(args.alternations):
        start = rng.choice(args.stress_starts)
        name = names[index % len(names)]
        if bench.sha(*shape, start, word(ARMS[name]), grace=0.0) != expected[start]:
            drift.append((index, start, name))
    report['stress'] = dict(calls=args.alternations, drifted=len(drift), first=drift[:10])
    print('stress %d alternating calls: %d drifted' % (args.alternations, len(drift)), flush=True)
    if drift:
        report['failures'].append('stress: %d of %d alternating calls drifted (first %r)' % (len(drift), args.alternations, drift[0]))


def time_arms(bench, args, report):
    """Interleaved single-call timings, one attention layer per sample: rounds of every (arm, start), the arm order rotated."""
    ttnn, device = bench.ttnn, bench.device
    timing = report['timing']
    for rows in args.time_rows:
        arms = arm_list(args, 'tp4', rows, bench.native_grid)
        arms = [name for name in arms if name in args.time_arms]
        starts = list(args.time_starts)
        validate_starts(starts, rows)
        for qmem in args.time_q_memory:
            samples = {(name, start): [] for name in arms for start in starts}
            for name in arms:
                for start in starts:
                    for _ in range(args.warmup):                         # the first call of a program JIT-compiles it
                        bench.ttnn.deallocate(bench.call(rows, qmem, 0, 'normal', start, word(ARMS[name]), record=name != 'stock'))
                    with WATCHDOG.op('timing warm sync'):
                        ttnn.synchronize_device(device)
            for name, start in schedule(arms, starts, args.rounds):
                with WATCHDOG.op('%s@%d timed' % (name, start)):
                    ttnn.synchronize_device(device)
                    began = time.perf_counter()
                    out = bench.call(rows, qmem, 0, 'normal', start, word(ARMS[name]), record=False, grace=0.0)
                    ttnn.synchronize_device(device)
                    samples[(name, start)].append((time.perf_counter() - began) * 1e3)
                ttnn.deallocate(out)
            block = timing.setdefault(str(rows), {}).setdefault(qmem, {})
            for (name, start), values in samples.items():
                block.setdefault(name, {})[str(start)] = dict(median_ms=statistics.median(values), min_ms=min(values), samples=values)
    verdicts = timing_verdict(timing, args.time_rows[0] if args.time_rows else 2048)
    report['timing_verdict'] = verdicts
    return verdicts


def print_timing(report, rows=2048):
    for qmem, arms in (report.get('timing', {}).get(str(rows)) or {}).items():
        served = arms.get('served') or {}
        for name in ('stock', 'served', 'oneq', 'oneq_noc'):
            for start, entry in sorted((arms.get(name) or {}).items(), key=lambda item: int(item[0])):
                base = (served.get(start) or {}).get('median_ms')
                print('ONEQ TIME rows=%d q=%s arm=%s start=%s context=%d median_ms=%.4f min_ms=%.4f vs_served=%s' % (
                    rows, qmem, name, start, int(start) + rows, entry['median_ms'], entry['min_ms'],
                    '%.3f' % (entry['median_ms'] / base) if base else '-'))
    for qmem, entry in sorted((report.get('timing_verdict') or {}).items()):
        print('ONEQ TIME_VERDICT q=%s %s oneq/served=%s best=%s' % (
            qmem, entry['verdict'], '%.3f' % entry['ratio'] if entry['ratio'] is not None else '-', entry['best']))


def run(args, report):
    import torch
    import ttnn

    with WATCHDOG.op('open device', extra=COMPILE_GRACE_S):
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        if args.maps:
            binary = loaded_binary()
            report['binary'] = binary
            print('binary %s sha256 %s chain_factory=%s oneq_edits=%s' % (binary['path'], binary['sha256'][:16], binary['chain'], binary['oneq']),
                  flush=True)
            if not binary['chain'] or not binary['oneq']:
                report['failures'].append('the loaded _ttnncpp.so lacks the %s: the K64j-OQ graft is not mounted'
                                          % ('[QWEN-SDPA-PF] factory' if not binary['chain'] else 'oneq edits'))
                return
            if args.expect_binary_sha256 and binary['sha256'] != args.expect_binary_sha256:
                report['failures'].append('the loaded _ttnncpp.so is %s, not the expected %s' % (binary['sha256'][:16],
                                                                                              args.expect_binary_sha256[:16]))
                return
        requested = set()
        report['_requested'] = requested
        bench = Bench(ttnn, torch, device, 'tp4', POOL_BLOCKS, requested)
        report['grid'] = list(bench.native_grid)
        report['geometry'] = dict(GEOMETRIES['tp4'], pool_blocks=POOL_BLOCKS)
        print('device grid %dx%d (%d cores)' % (bench.native_grid + (bench.native_grid[0] * bench.native_grid[1],)), flush=True)
        tp2_bench = Bench(ttnn, torch, device, 'tp2', TP2_POOL_BLOCKS, requested) if not args.no_tp2 else None
        if not args.no_controls:
            cache_checks(bench, args, report)
            refusals(bench, tp2_bench or Bench(ttnn, torch, device, 'tp2', TP2_POOL_BLOCKS, requested), args, report)
        sweep(bench, args, report)
        if tp2_bench is not None:
            tp2_checks(tp2_bench, args, report)
        if args.grid_list:
            grid_checks(bench, args, report)
        if args.alternations:
            stress(bench, args, report)
        if not args.no_timing:
            time_arms(bench, args, report)
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_list(text, cast=int):
    return [cast(value, 0) if cast is int else cast(value) for value in text.split(',') if value.strip()]


def parse_grids(text):
    out = []
    for item in parse_list(text, str):
        if item == 'native':
            continue
        match = re.fullmatch(r'([0-9]+)x([0-9]+)', item)
        if not match:
            raise ValueError('grid %r is not NxM or native' % item)
        out.append((int(match.group(1)), int(match.group(2))))
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--expect-binary-sha256', default='', help='the K64j-OQ _ttnncpp.so the run must have mapped (the runner passes it)')
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--rows', default=','.join(map(str, ROWS)))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)))
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--q-memory', default=','.join(Q_MEMORY))
    parser.add_argument('--grids', default='native,11x10', help='extra grids for the grid pass (native = the device grid, always run)')
    parser.add_argument('--alternations', type=int, default=400, help='alternating served/oneq calls against the stock sha (0: skip)')
    parser.add_argument('--stress-starts', default='0,2048,65536,129024')
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--time-rows', default='2048')
    parser.add_argument('--time-arms', default='stock,served,oneq,oneq_noc')
    parser.add_argument('--time-starts', default=','.join(map(str, TIME_STARTS)))
    parser.add_argument('--time-q-memory', default='dram,l1')
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--no-tp2', action='store_true', help='skip the TP2 head-count checks')
    parser.add_argument('--no-controls', action='store_true', help='skip the cache and refusal checks')
    parser.add_argument('--watchdog', type=float, default=120.0, help='seconds per device call; 0 off')
    parser.add_argument('--no-maps', dest='maps', action='store_false', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.arms = parse_list(args.arms, str)
    args.rows = parse_list(args.rows)
    args.starts = parse_list(args.starts)
    args.seeds = parse_list(args.seeds)
    args.variants = parse_list(args.variants, str)
    args.q_memory = parse_list(args.q_memory, str)
    args.stress_starts = parse_list(args.stress_starts)
    args.time_rows = parse_list(args.time_rows)
    args.time_arms = parse_list(args.time_arms, str)
    args.time_starts = parse_list(args.time_starts)
    args.time_q_memory = parse_list(args.time_q_memory, str)
    args.grid_list = parse_grids(args.grids)
    try:
        validate_starts(args.stress_starts, 2048)
        validate_starts(args.time_starts, 2048)
    except ValueError as error:
        parser.error(str(error))
    if 'stock' not in args.arms:
        parser.error('--arms must include stock: it is the reference every other arm is compared with')
    unknown = [name for name in args.arms + args.time_arms if name not in ARMS]
    if unknown:
        parser.error('unknown arm %r (known: %s)' % (unknown, ', '.join(ARMS)))
    for rows in args.rows:
        if rows % Q_CHUNK:
            parser.error('rows %d is not a whole number of 128-row Q chunks' % rows)
        try:
            validate_starts(args.starts, rows)
        except ValueError as error:
            parser.error(str(error))
    for values, known, name in ((args.variants, VARIANTS, 'variant'), (args.q_memory, Q_MEMORY, 'Q memory'),
                                (args.time_q_memory, Q_MEMORY, 'Q memory')):
        if any(value not in known for value in values):
            parser.error('unknown %s in %r' % (name, values))
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(passed=False, arms=args.arms, rows=args.rows, starts=args.starts, seeds=args.seeds, variants=args.variants,
                  q_memory=args.q_memory, failures=[], cases=[], tp2_cases=[], grid_cases=[], timing={},
                  expect_binary_sha256=args.expect_binary_sha256 or None)
    native = Q1.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        payload['requested_programs'] = sorted('%s rows=%d width=%d q=%s flags=%#x grid=%dx%d' % (key[:5] + key[5]) for key in report.get('_requested', ()))
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def on_fire(label):
        try:
            write_report(dict(error='watchdog: %r exceeded %ss' % (label, args.watchdog), passed=False, watchdog=label))
        except Exception:  # noqa: BLE001 - the WATCHDOG line stands
            pass

    WATCHDOG = Q1.Watchdog(args.watchdog, on_fire=on_fire).start()
    try:
        with native:
            run(args, report)
        if report.get('binary') or not args.maps:
            for problem in check_factory_lines(native.text(), report.get('_requested', set())):
                report['failures'].append('factory log: ' + problem)
        report['passed'] = not report['failures'] and bool(report['cases'])
    except Exception as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
        report['traceback'] = traceback.format_exc()[-4000:]
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    if report.get('error'):
        print('ERROR', report['error'])
        print(report.get('traceback', ''))
    exact = sum(1 for case in report['cases'] if case['exact'])
    print_timing(report, args.time_rows[0] if args.time_rows else 2048)
    print('ONEQ_CARD_M verdict=%s cases=%d exact=%d tp2=%d grid=%d failures=%d report=%s native_log=%s' % (
        'PASS' if report['passed'] else 'FAIL', len(report['cases']), exact, len(report['tp2_cases']), len(report['grid_cases']),
        len(report['failures']), args.out, native.path))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
