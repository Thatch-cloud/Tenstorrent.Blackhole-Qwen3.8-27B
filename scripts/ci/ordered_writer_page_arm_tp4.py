"""The page64 arm of ordered_writer_tp4_card_test.py (E1): the card-M proof that the page-parallel K/V writer (kv_page_writer_tp4) leaves the
served writer's bytes - the evidence the lever refuses to engage without. Plan 9 unknown U1: is bfloat8_b read-modify-write idempotent on the
hardware packer?

The served chain applies pack(unpack(.)) once PER ROW of a tile row; the page writer once PER TILE. They agree exactly when pack(unpack(x)) is the
identity on every block the packer writes. This arm settles that on the card three ways, none of which trusts a model of the packer:

  PAGE CASES (kind 'page'). Two identical caches per K/V (S0), the SERVED chained writer on one and the page writer on the other, the same inputs,
      step after step; after EVERY step the COMPLETE caches are read back and compared as bit patterns (served_equal_k / _v). The shapes are the
      stack's: four packed users of 16 consecutive positions on distinct (page, tile row)s, starts at every offset class (inside a tile row, across
      the 32-row tile boundary, across the 64-token page boundary, the last positions of the window), both page-table widths (2,052 and 4,096),
      both sources (the prepared K/V interleaved in DRAM, and in the height-sharded L1 shards the AttnPrep op writes), eager and trace replay (tables
      rewritten in place, and untouched), K and V together. Three payload regimes: 'exact' (bfloat8_b-exact values from zero caches: the one regime a
      host can predict, so a third oracle, host_equal_k / _v), 'random' (non-exact bfloat16 over a random device-packed cache: partial pages) and 'edge'
      (zeros, -0, denormals, an outlier that underflows its block mates, rounding carries, the largest and smallest normals). The page writer is then
      run AGAIN on the same inputs (twice_equal_k / _v: writing the same page twice).
  NOOP CASES (kind 'noop'). The idempotence itself, without the page writer: a device-packed random cache is read back, and each row of it is
      WRITTEN BACK to where it came from (by the served writer, and by the page writer); the complete cache must not change by one bit
      (noop_equal_k / _v). A packer whose pack(unpack(x)) moved any stored block - including the sign of a zero - fails here first.
  NEGATIVE CONTROLS (--page-negative drop|slot). The kernel with a row left out, or the second tile row written as the first: the same cases must FAIL.

Everything is pure integer arithmetic and torch on the host; the device part is run_page_case / run_noop_case (card M, one chip of four).
"""

import hashlib
import json

import ordered_writer_tp4_card_test as card

PAGE_WRITER = 'page64'
KIND_PAGE, KIND_NOOP = 'page', 'noop'
PAGE_KINDS = (KIND_PAGE, KIND_NOOP)
SOURCES = ('dram', 'l1')
REGIMES = ('exact', 'random', 'edge')
NOOP_REGIMES = ('random', 'edge')
GROUPS, GROUP_ROWS, BLOCK_ROWS = 4, 16, 64
SEGMENTS = tuple((group * GROUP_ROWS, (group + 1) * GROUP_ROWS) for group in range(GROUPS))
PAGE_STEPS = 8
SCRATCH_BLOCKS = 64                       # the blocks after the plan's, never assigned to a plan entry: the fill's filler rows write there


def plan_blocks(width):
    """Blocks the plan assigns from: the page-table width (every row of a table carries distinct ids, so a table of `width` entries needs that many)."""
    return width


def total_blocks(width):
    """The cache's blocks at this width (the plan's and the scratch ones): 71 MB at 4,096 entries, 38 MB at 2,052."""
    return plan_blocks(width) + SCRATCH_BLOCKS
PADDED_HEADS = 32
HEAD_DIM = 256
# Start offsets inside a page, cycled: the aligned ones (0, 16, 32, 48), one past them, the ones whose 16 rows cross the 32-row tile boundary (17..31,
# 33.. no: 17-31) or the 64-token page boundary (49..63), and the unaligned middles.
START_OFFSETS = (0, 16, 1, 17, 31, 32, 15, 48, 49, 33, 47, 63, 8, 24, 40, 56, 5, 60)
EDGE_KINDS = ('zeros', 'negzero', 'denormal', 'outlier', 'carry', 'huge', 'tiny')
NOOP_STARTS = (0, 16, 32, 48, 8, 24, 40)
NOOP_SERVED_STEPS = (0, 1, 31, 32, 63)    # the offsets whose readback is compared (every step writes every row; a failure shows at once)
# (regime, source, mode): the page cases of one (width, seed). The matrix covers every regime, both sources and all three modes at both widths.
MATRIX = (('exact', 'dram', 'eager'), ('random', 'l1', 'eager'), ('random', 'l1', 'replay_changed'), ('random', 'dram', 'replay_unchanged'),
          ('edge', 'dram', 'replay_changed'), ('edge', 'l1', 'replay_unchanged'), ('edge', 'l1', 'eager'))
FILL_OFFSETS = 64


def page_case_name(width, source, mode, regime, seed):
    return 'page64-%d-%s-%s-%s-s%d' % (width, source, mode, regime, seed)


def noop_case_name(writer, width, source, regime):
    return 'noop-%s-%d-%s-%s' % (writer, width, source, regime)


def build_page_cases(widths=card.WIDTHS, seeds=card.SEEDS, modes=card.MODES, sources=SOURCES, regimes=REGIMES, noop=True, first_index=0):
    """The page and noop case specs in run order (seed, width, matrix row; then the noop cases)."""
    specs = []
    for seed in seeds:
        for width in widths:
            for regime, source, mode in MATRIX:
                if regime in regimes and source in sources and mode in modes:
                    specs.append(dict(kind=KIND_PAGE, writer=PAGE_WRITER, width=width, source=source, mode=mode, regime=regime, seed=seed,
                                      name=page_case_name(width, source, mode, regime, seed)))
    if noop and 0 in seeds:
        for regime in NOOP_REGIMES:
            if regime in regimes:
                specs.append(dict(kind=KIND_NOOP, writer='chained64', width=card.CONTROL_WIDTH, source='dram', mode='eager', regime=regime,
                                  seed=0, name=noop_case_name('served', card.CONTROL_WIDTH, 'dram', regime)))
                for width in widths:
                    for source in sources:
                        specs.append(dict(kind=KIND_NOOP, writer=PAGE_WRITER, width=width, source=source, mode='eager', regime=regime,
                                          seed=0, name=noop_case_name('page', width, source, regime)))
    for offset, spec in enumerate(specs):
        spec['index'] = first_index + offset
    return specs


def step_count(spec):
    if spec['kind'] == KIND_NOOP:
        return len(NOOP_STARTS) if spec['writer'] == PAGE_WRITER else len(NOOP_SERVED_STEPS)
    return PAGE_STEPS + 1


def required_checks(spec):
    """The (name, step) checks a case must record exact."""
    required = []
    if spec['kind'] == KIND_NOOP:
        for step in range(step_count(spec)):
            required.append(('noop_equal_k', step))
            required.append(('noop_equal_v', step))
        return required
    if spec['regime'] != 'exact':
        required.append(('fill_equal', None))
    for step in range(step_count(spec)):
        for cache in ('k', 'v'):
            required.append(('served_equal_' + cache, step))
            if spec['regime'] == 'exact':
                required.append(('host_equal_' + cache, step))
    for cache in ('k', 'v'):
        required.append(('twice_equal_' + cache, None))
    required.append(('input_unchanged', None))
    required.append(('pages_unchanged', None))
    return required


# ---------------------------------------------------------------------------------------------
# The plan: positions and tables (pure integer arithmetic).
# ---------------------------------------------------------------------------------------------

def group_starts(width, step, seed, steps):
    """The first position of each of the four users at `step` (0 .. steps; the last is the window-end step). Distinct (entry, tile row) per user."""
    entries = card.entries_for(width)
    if step == steps:
        top = (width - 1) * card.BLOCK_SIZE
        # user 0: tile row 0 of the last page; user 1: its last 16 positions (tile row 1, through position width * 64 - 1); user 2: the last page
        # but one; user 3: 16 rows across a page boundary (offsets 49..63 and the next page's first row), high in the table.
        return [top + 16, top + 48, top - card.BLOCK_SIZE + 48, entries[-3] * card.BLOCK_SIZE + 49]
    starts = []
    used = set()
    for group in range(GROUPS):
        for attempt in range(len(entries) * 4):
            entry = entries[(group * 3 + step + seed + attempt) % len(entries)]
            offset = START_OFFSETS[(step * GROUPS + group + seed * 3 + attempt) % len(START_OFFSETS)]
            start = entry * card.BLOCK_SIZE + offset
            if start + GROUP_ROWS > width * card.BLOCK_SIZE:
                continue
            keys = {((start + row) >> 6, ((start + row) >> 5) & 1) for row in range(GROUP_ROWS)}
            if used & keys:
                continue
            used |= keys
            starts.append(start)
            break
        else:
            raise ValueError('no free start for user %d at step %d' % (group, step))
    return starts


def positions_of(starts):
    """The block's 64 positions: user g's rows are starts[g] .. starts[g] + 15 (packed_host_inputs)."""
    return [start + row for start in starts for row in range(GROUP_ROWS)]


def keys_of(starts):
    return [{((start + row) >> 6, ((start + row) >> 5) & 1) for row in range(GROUP_ROWS)} for start in starts]


def conflict_free(starts):
    seen = set()
    for keys in keys_of(starts):
        if seen & keys:
            return False
        seen |= keys
    return True


def hit_entries(all_starts):
    """Every page-table entry any user touches (the entry a position is in, and the next one when the 16 rows cross a page)."""
    hits = set()
    for starts in all_starts:
        for start in starts:
            hits.update(((start + row) >> 6) for row in range(GROUP_ROWS))
    return sorted(hits)


def group_tables(seed, width, hits, table_index=0):
    """(GROUPS, width) tables: each (user, hit entry) a physical block no other has (card.page_table's unique assignment over plan_blocks(width) blocks);
    the rest a distinct filler id. A user's 16 rows all carry its table."""
    return card.page_table(seed * 31 + table_index + 1, GROUPS, width, plan_blocks(width), hits)


def expand_tables(tables):
    """The device page table: (64, width), row r the table of user r // 16."""
    return [list(tables[row // GROUP_ROWS]) for row in range(BLOCK_ROWS)]


def build_page_case(spec):
    """Materialise a page case: tables (two for replay_changed, else one), per-step starts, positions and target blocks, payload seeds."""
    width, seed, mode = spec['width'], spec['seed'], spec['mode']
    steps = []
    all_starts = [group_starts(width, step, seed, PAGE_STEPS) for step in range(PAGE_STEPS + 1)]
    for starts in all_starts:
        if not conflict_free(starts):
            raise ValueError('the plan puts two users on one (page, tile row): %r' % (starts,))
    hits = hit_entries(all_starts)
    tables = [group_tables(seed, width, hits, index) for index in range(2 if mode == 'replay_changed' else 1)]
    for number, starts in enumerate(all_starts):
        table_index = number % len(tables)
        positions = positions_of(starts)
        blocks = [tables[table_index][row // GROUP_ROWS][position >> 6] for row, position in enumerate(positions)]
        steps.append(dict(step=number, table=table_index, starts=starts, positions=positions, blocks=blocks,
                          payload_seeds=[(spec['index'] + 1) * 1000003 + number * 1009 + row for row in range(BLOCK_ROWS)]))
    touched = sorted({block for table in tables for group in range(GROUPS) for entry in hits for block in [table[group][entry]]})
    return dict(spec, tables=tables, steps=steps, hits=hits, touched=touched)


def build_noop_case(spec):
    """A noop case: the blocks it fills and rewrites in place. The page writer's steps put user g on blocks[g] at 16 consecutive offsets from
    NOOP_STARTS[step] of table entry `entry` (high in the table); the served writer's put row r on blocks[r] at offset NOOP_SERVED_STEPS[step]."""
    width = spec['width']
    page = spec['writer'] == PAGE_WRITER
    entry = card.entries_for(width)[-3] if page else 0
    unused, order = card.permutation(spec['seed'] * 977 + width, plan_blocks(width))
    return dict(spec, entry=entry, blocks=order[:GROUPS if page else BLOCK_ROWS])


# ---------------------------------------------------------------------------------------------
# Payloads (host, deterministic).
# ---------------------------------------------------------------------------------------------

def payload_bits(regime, seed, rows=BLOCK_ROWS, kind_shift=0):
    """(rows, 32, 256) int16 bit patterns of bfloat16: head 0 is the K/V row of the regime, heads 1..31 padding (random: a reader that copies a
    padded head into the cache is caught)."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, PADDED_HEADS, HEAD_DIM, generator=generator)
    exponents = torch.randint(-8, 9, (rows, PADDED_HEADS, HEAD_DIM // 16), generator=generator).to(torch.float32)
    scale = torch.pow(torch.full_like(exponents, 2.0), exponents).repeat_interleave(16, dim=2)
    values = (base * scale).to(torch.bfloat16).contiguous().view(torch.int16).clone()
    if regime == 'edge':
        for row in range(rows):
            kind = EDGE_KINDS[(row + seed + kind_shift) % len(EDGE_KINDS)]
            values[row, 0, :] = torch.tensor(edge_row_bits(kind, seed * 131 + row), dtype=torch.int32).to(torch.int16)
    return values


def to_bfloat16(bits):
    import torch

    return bits.contiguous().view(torch.bfloat16)


def _bf16(value):
    import struct

    packed = struct.unpack('>I', struct.pack('>f', value))[0]
    packed += 0x7FFF + ((packed >> 16) & 1)
    return (packed >> 16) & 0xFFFF


def _signed16(value):
    return value - 65536 if value >= 32768 else value


def edge_row_bits(kind, seed):
    """256 signed-int16 bit patterns of one edge regime (the head row)."""
    state = [seed * 2654435761 + 12345]

    def draw():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (state[0] >> 33) & 0x7FFFFFFF

    def gauss():
        total = sum(draw() / 0x7FFFFFFF for unused in range(6)) - 3.0
        return total * 1.4142

    row = []
    for block in range(16):
        for column in range(16):
            if kind == 'zeros':
                value = 0
            elif kind == 'negzero':
                value = 0x8000 if column % 2 else _bf16(gauss())
            elif kind == 'denormal':
                value = (0x0001, 0x8001, 0x007F, 0x807F)[draw() % 4] if column % 3 == 0 else _bf16(gauss() * 1e-30)
            elif kind == 'outlier':
                value = _bf16(2.0 ** 20) if column == block % 16 else _bf16(gauss())
            elif kind == 'carry':
                value = 0x3FFF if column % 2 else 0x3F7F
            elif kind == 'huge':
                value = _bf16((-1 if draw() % 2 else 1) * 1.9921875 * 2.0 ** 120)
            elif kind == 'tiny':
                value = _bf16((-1 if draw() % 2 else 1) * 1.5 * 2.0 ** -120)
            else:
                raise ValueError('unknown edge kind %r' % (kind,))
            row.append(_signed16(value & 0xFFFF))
    return row


def step_payloads(hw, regime, seeds, cache_index):
    """The (64, 32, 256) bfloat16 K (cache_index 0) or V (1) payloads of one step. 'exact' is ordered_cache_hw_plan's bfloat8_b-exact payload (K and V
    on disjoint seeds); the others come from payload_bits."""
    import torch

    if regime == 'exact':
        return hw.step_payloads(dict(payload_seeds=[seed + 500000 * cache_index for seed in seeds]))
    rows = [to_bfloat16(payload_bits(regime, seed + 7919 * cache_index, rows=1, kind_shift=index)) for index, seed in enumerate(seeds)]
    return torch.cat(rows, dim=0)


def fill_payloads(regime, seed, cache_index):
    return to_bfloat16(payload_bits(regime if regime != 'exact' else 'random', seed * 17 + 3 + 100003 * cache_index))


# ---------------------------------------------------------------------------------------------
# Verdict helpers.
# ---------------------------------------------------------------------------------------------

def decide_page(report):
    """PASS / FAIL / NO-DECISION over the page and noop cases of the report ('absent' when it ran none)."""
    plan = [entry for entry in report.get('plan', []) if entry.get('kind') in PAGE_KINDS]
    if not plan:
        return dict(verdict='absent', problems=[])
    if report.get('error'):
        return dict(verdict='NO-DECISION', problems=[str(report['error'])[:200]])
    checks = [check for check in report.get('checks', []) if check.get('kind') in PAGE_KINDS]
    by_case = {}
    for check in checks:
        by_case.setdefault(check['case'], {})[(check['name'], check.get('step'))] = check
    problems = []
    for entry in plan:
        state = (report.get('cases') or {}).get(entry['name']) or {}
        if state.get('error'):
            problems.append('%s raised' % entry['name'])
            continue
        if state.get('skipped'):
            problems.append('%s was cut by the deadline' % entry['name'])
            continue
        recorded = by_case.get(entry['name']) or {}
        missing = [item for item in entry['required'] if tuple(item) not in recorded]
        if missing:
            problems.append('%s lacks %d checks (%s)' % (entry['name'], len(missing), missing[0]))
    if problems:
        return dict(verdict='NO-DECISION', problems=problems[:20])
    inexact = [check for check in checks if not check.get('exact')]
    if inexact:
        first = inexact[0]
        return dict(verdict='FAIL', problems=['%d checks differ (first: %s %s step %s)' % (len(inexact), first['case'], first['name'], first.get('step'))])
    return dict(verdict='PASS', problems=[])


def page_scope(report):
    """'full' when the report ran every regime, source, mode, both widths, seeds 0-2, K and V, and the noop proofs of both writers."""
    requested = report.get('requested') or {}
    page = requested.get('page') or {}
    full = (PAGE_WRITER in (requested.get('writers') or ()) and set(requested.get('widths', ())) >= set(card.WIDTHS)
            and set(requested.get('seeds', ())) >= set(card.SEEDS) and set(requested.get('modes', ())) >= set(card.MODES)
            and set(page.get('sources', ())) >= set(SOURCES) and set(page.get('regimes', ())) >= set(REGIMES) and page.get('noop') is True
            and not page.get('negative'))
    return 'full' if full else 'reduced'


def tally_page(report):
    checks = [check for check in report.get('checks', []) if check.get('kind') in PAGE_KINDS]
    return dict(checks=len(checks), exact=sum(1 for check in checks if check.get('exact')))


def proofs_of(report):
    """{'noop_served' | 'noop_page' | 'twice_page': {checks, exact}} from the report's checks, for the evidence record."""
    proofs = {}
    for label, predicate in (('noop_served', lambda c: c['name'].startswith('noop_equal') and c['writer'] == 'chained64'),
                             ('noop_page', lambda c: c['name'].startswith('noop_equal') and c['writer'] == PAGE_WRITER),
                             ('twice_page', lambda c: c['name'].startswith('twice_equal'))):
        found = [check for check in report.get('checks', []) if check.get('kind') in PAGE_KINDS and predicate(check)]
        proofs[label] = dict(checks=len(found), exact=sum(1 for check in found if check.get('exact')))
    return proofs


# ---------------------------------------------------------------------------------------------
# The device part (card M, one chip of four).
# ---------------------------------------------------------------------------------------------

def shard_config(ttnn):
    """The AttnPrep output's placement: one (32, 256) shard per row on an 8x8 grid, row-major (qwen36_attention_tp._kv_shard_cfg)."""
    return ttnn.create_sharded_memory_config(shape=(ttnn.TILE_SIZE, HEAD_DIM), core_grid=ttnn.CoreGrid(x=8, y=8),
                                             strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                             use_height_and_width_as_shard_shape=True)


def bits_equal(torch, actual, reference):
    return (tuple(actual.shape) == tuple(reference.shape) and actual.dtype == torch.bfloat16 and reference.dtype == torch.bfloat16
            and torch.equal(actual.contiguous().view(torch.int16), reference.contiguous().view(torch.int16)))


class PageRig:
    """What a page or noop case needs beside card.Rig: both writers as host-callable operations over one set of tensors."""

    def __init__(self, rig, hw, kernels, texts, wt, negative=None):
        import kv_page_writer_tp4 as kvpw
        import packed_ordered_cache

        self.rig, self.hw, self.kernels, self.texts, self.wt, self.negative = rig, hw, kernels, texts, wt, negative
        self.kvpw, self.poc = kvpw, packed_ordered_cache
        self.ttnn, self.torch, self.mesh = rig.ttnn, rig.torch, rig.mesh
        self.shard = None

    def served(self, caches, packed, positions, pages):
        for cache, value in zip(caches, packed):
            self.poc.update_chained(self.mesh, cache, value, positions, pages, self.kernels, SEGMENTS, operations=self.ttnn)

    def fill_served(self, caches, packed, positions, pages):
        singles = tuple((row, row + 1) for row in range(BLOCK_ROWS))
        for cache, value in zip(caches, packed):
            self.poc.update_chained(self.mesh, cache, value, positions, pages, self.kernels, singles, operations=self.ttnn)

    def page(self, caches, dram_packed, positions, pages, source):
        ttnn, kvpw = self.ttnn, self.kvpw
        held = []
        try:
            packed, cores = list(dram_packed), None
            if source == 'l1':
                if self.shard is None:
                    self.shard = shard_config(ttnn)
                packed = [ttnn.to_memory_config(value, self.shard) for value in dram_packed]
                held.extend(packed)
                mode, cores = kvpw.source_mode(ttnn, packed[0])
                if mode != 'l1':
                    raise AssertionError('the prepared K/V is not height-sharded in L1')
            kvpw.validate_tensors(ttnn, list(caches), packed, positions, pages, int(pages.shape[1]))
            kvpw.launch(ttnn, self.mesh, dict(caches=list(caches), packed=packed, positions=positions, pages=pages), self.kernels, self.texts,
                        wt=self.wt, source=source, source_cores=cores, ordered=False, negative=self.negative)
        finally:
            for value in held:
                ttnn.deallocate(value)


def read_cache(rig, tensor):
    return rig.host(tensor)


def run_fill(prig, report, name, caches, blocks, regime, seed, width):
    """Write every row of every block in `blocks` through the SERVED writer (device-packed content): 64 offsets, 64 blocks per launch."""
    rig, torch, ttnn = prig.rig, prig.torch, prig.ttnn
    positions = rig.upload(torch.zeros(BLOCK_ROWS, dtype=torch.int32), ttnn.int32)
    pages = rig.upload(torch.zeros(BLOCK_ROWS, width, dtype=torch.int32), ttnn.int32)
    packed = [rig.upload(torch.zeros(1, BLOCK_ROWS, PADDED_HEADS, HEAD_DIM, dtype=torch.bfloat16), ttnn.bfloat16) for unused in caches]
    batches = [blocks[start:start + BLOCK_ROWS] for start in range(0, len(blocks), BLOCK_ROWS)]
    for batch_index, batch in enumerate(batches):
        rows = list(batch) + [plan_blocks(width) + row for row in range(len(batch), BLOCK_ROWS)]
        table = torch.zeros(BLOCK_ROWS, width, dtype=torch.int32)
        table[:, 0] = torch.tensor(rows, dtype=torch.int32)
        rig.replace(table, pages, ttnn.int32)
        for offset in range(FILL_OFFSETS):
            rig.replace(torch.full((BLOCK_ROWS,), offset, dtype=torch.int32), positions, ttnn.int32)
            for index, value in enumerate(packed):
                payload = fill_payloads(regime, seed * 1000 + batch_index * 64 + offset, index)
                rig.replace(payload.unsqueeze(0).contiguous(), value, ttnn.bfloat16)
            prig.fill_served(caches, packed, positions, pages)
    ttnn.synchronize_device(rig.mesh)
    report['cases'][name]['filled_blocks'] = len(blocks)


def run_page_case(prig, hw, case, report):
    """One page case on the open card; appends its checks to report['checks']."""
    ttnn, torch, rig = prig.ttnn, prig.torch, prig.rig
    name, width, regime, source, mode = case['name'], case['width'], case['regime'], case['source'], case['mode']

    def record(check_name, step, exact, **extra):
        report['checks'].append(dict(case=name, kind=KIND_PAGE, writer=PAGE_WRITER, width=width, mode=mode, seed=case['seed'], source=source,
                                     regime=regime, name=check_name, step=step, exact=bool(exact), **extra))

    zero = torch.zeros(total_blocks(width), card.KV_HEADS, card.BLOCK_SIZE, card.HEAD_DIM, dtype=torch.bfloat16)
    cache_s = [rig.upload(zero.clone(), ttnn.bfloat8_b) for unused in range(2)]
    if regime == 'exact':
        cache_p = [rig.upload(zero.clone(), ttnn.bfloat8_b) for unused in range(2)]
    else:
        # S0: both caches device-packed random content over every block the plan touches (the served writer made it), then a copy.
        run_fill(prig, report, name, cache_s, case['touched'], regime, case['seed'], width)
        cache_p = [ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG) for value in cache_s]
        rig.owned.extend(cache_p)
        record('fill_equal', None, all(bits_equal(torch, rig.host(a), rig.host(b)) for a, b in zip(cache_s, cache_p)))
    tables = [torch.tensor(expand_tables(table), dtype=torch.int32) for table in case['tables']]
    pages = rig.upload(tables[0], ttnn.int32)
    first = case['steps'][0]
    positions = rig.upload(torch.tensor(first['positions'], dtype=torch.int32), ttnn.int32)
    packed = [rig.upload(step_payloads(hw, regime, first['payload_seeds'], index).unsqueeze(0).contiguous(), ttnn.bfloat16) for index in range(2)]
    expected = [hw.ExpectedCache(total_blocks(width), heads=card.KV_HEADS) for unused in range(2)] if regime == 'exact' else None
    current, traces = 0, None

    def served_op():
        prig.served(cache_s, packed, positions, pages)

    def page_op():
        prig.page(cache_p, packed, positions, pages, source)

    def execute(operation, key):
        nonlocal traces
        if mode == 'eager':
            operation()
            return
        traces = traces or {}
        if key not in traces:
            operation()
            ttnn.synchronize_device(rig.mesh)
            trace = ttnn.begin_trace_capture(rig.mesh, cq_id=0)
            try:
                operation()
            finally:
                ttnn.end_trace_capture(rig.mesh, trace, cq_id=0)
            traces[key] = trace
        ttnn.execute_trace(rig.mesh, traces[key], cq_id=0, blocking=True)

    try:
        for step in case['steps']:
            payloads = [step_payloads(hw, regime, step['payload_seeds'], index) for index in range(2)]
            if step['table'] != current:
                rig.replace(tables[step['table']], pages, ttnn.int32)
                current = step['table']
            rig.replace(torch.tensor(step['positions'], dtype=torch.int32), positions, ttnn.int32)
            for index in range(2):
                rig.replace(payloads[index].unsqueeze(0).contiguous(), packed[index], ttnn.bfloat16)
            execute(served_op, 'served')
            execute(page_op, 'page')
            ttnn.synchronize_device(rig.mesh)
            for index, label in enumerate(('k', 'v')):
                served_host, page_host = rig.host(cache_s[index]), rig.host(cache_p[index])
                result = bits_equal(torch, page_host, served_host)
                extra = {}
                if not result:
                    differing = (page_host.contiguous().view(torch.int16) != served_host.contiguous().view(torch.int16))
                    blocks = differing.reshape(total_blocks(width), -1).any(dim=1).nonzero().flatten().tolist()
                    extra = dict(differing_blocks=len(blocks), sample_blocks=blocks[:8])
                record('served_equal_' + label, step['step'], result, table=step['table'], **extra)
                if expected is not None:
                    expected[index].apply(expand_tables(case['tables'][step['table']]), step['positions'], payloads[index])
                    record('host_equal_' + label, step['step'], bits_equal(torch, page_host, expected[index].values), table=step['table'])
        before = [rig.host(value) for value in cache_p]
        execute(page_op, 'page')
        ttnn.synchronize_device(rig.mesh)
        for index, label in enumerate(('k', 'v')):
            record('twice_equal_' + label, None, bits_equal(torch, rig.host(cache_p[index]), before[index]))
        unchanged = all(tuple(rig.host(packed[index]).shape) == (1, BLOCK_ROWS, PADDED_HEADS, HEAD_DIM)
                        and torch.equal(rig.host(packed[index]).contiguous().view(torch.int16),
                                        payloads[index].unsqueeze(0).contiguous().view(torch.int16)) for index in range(2))
        record('input_unchanged', None, unchanged)
        actual_pages = rig.host(pages)
        record('pages_unchanged', None, tuple(actual_pages.shape) == tuple(tables[current].shape) and torch.equal(actual_pages, tables[current]))
    finally:
        if traces:
            for trace in traces.values():
                ttnn.release_trace(rig.mesh, trace)
        ttnn.synchronize_device(rig.mesh)
        rig.release()


def run_noop_case(prig, hw, case, report):
    """Write every stored row of a device-packed cache back to where it came from; the complete cache must not change."""
    ttnn, torch, rig = prig.ttnn, prig.torch, prig.rig
    name, width, regime, source = case['name'], case['width'], case['regime'], case['source']
    page_writer = case['writer'] == PAGE_WRITER

    def record(check_name, step, exact, **extra):
        report['checks'].append(dict(case=name, kind=KIND_NOOP, writer=case['writer'], width=width, mode='eager', seed=case['seed'], source=source,
                                     regime=regime, name=check_name, step=step, exact=bool(exact), **extra))

    zero = torch.zeros(total_blocks(width), card.KV_HEADS, card.BLOCK_SIZE, card.HEAD_DIM, dtype=torch.bfloat16)
    caches = [rig.upload(zero.clone(), ttnn.bfloat8_b) for unused in range(2)]
    blocks = case['blocks']
    run_fill(prig, report, name, caches, blocks, regime, case['seed'], width)
    stored = [rig.host(value) for value in caches]
    tables = torch.zeros(BLOCK_ROWS, width, dtype=torch.int32)
    entry = case['entry']
    positions = rig.upload(torch.zeros(BLOCK_ROWS, dtype=torch.int32), ttnn.int32)
    packed = [rig.upload(torch.zeros(1, BLOCK_ROWS, PADDED_HEADS, HEAD_DIM, dtype=torch.bfloat16), ttnn.bfloat16) for unused in range(2)]
    try:
        steps = NOOP_STARTS if page_writer else NOOP_SERVED_STEPS
        for number, value in enumerate(steps):
            if page_writer:
                rows_positions = [entry * card.BLOCK_SIZE + value + row for unused in range(GROUPS) for row in range(GROUP_ROWS)]
                row_blocks = [blocks[row // GROUP_ROWS] for row in range(BLOCK_ROWS)]
                tables.zero_()
                for row in range(BLOCK_ROWS):
                    tables[row, entry] = row_blocks[row]
                offsets = [(entry * card.BLOCK_SIZE + value + row) % card.BLOCK_SIZE for unused in range(GROUPS) for row in range(GROUP_ROWS)]
            else:
                rows_positions = [value] * BLOCK_ROWS
                row_blocks = blocks[:BLOCK_ROWS]
                tables.zero_()
                tables[:, 0] = torch.tensor(row_blocks, dtype=torch.int32)
                offsets = [value] * BLOCK_ROWS
            pages = rig.upload(tables.clone(), ttnn.int32)
            rig.replace(torch.tensor(rows_positions, dtype=torch.int32), positions, ttnn.int32)
            for index in range(2):
                payload = torch.zeros(BLOCK_ROWS, PADDED_HEADS, HEAD_DIM, dtype=torch.bfloat16)
                for row in range(BLOCK_ROWS):
                    payload[row, 0, :] = stored[index][row_blocks[row], 0, offsets[row], :]
                rig.replace(payload.unsqueeze(0).contiguous(), packed[index], ttnn.bfloat16)
            if page_writer:
                prig.page(caches, packed, positions, pages, source)
            else:
                prig.served(caches, packed, positions, pages)
            ttnn.synchronize_device(rig.mesh)
            for index, label in enumerate(('k', 'v')):
                actual = rig.host(caches[index])
                result = bits_equal(torch, actual, stored[index])
                extra = {}
                if not result:
                    differing = (actual.contiguous().view(torch.int16) != stored[index].contiguous().view(torch.int16))
                    changed = differing.reshape(total_blocks(width), -1).any(dim=1).nonzero().flatten().tolist()
                    extra = dict(differing_blocks=len(changed), sample_blocks=changed[:8])
                record('noop_equal_' + label, number, result, **extra)
    finally:
        ttnn.synchronize_device(rig.mesh)
        rig.release()


def design_summary(prig):
    """What the report records about the writer that ran: for the evidence record's binding (kv_page_writer_tp4.design_signature)."""
    kvpw = prig.kvpw
    return dict(design=kvpw.design(prig.wt), design_signature=kvpw.design_signature(prig.wt), wt=prig.wt, units=kvpw.unit_count(prig.wt),
                kernel_sha256=kvpw.source_sha256())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
