#!/usr/bin/env python3
"""One-card sweep of the long-context SDPA decode at the FOUR-CARD per-chip shape: every candidate configuration timed against the
served one at 32k / 64k / 131k / 262k and byte-compared with it, row by row.

WHAT A CHIP DOES AT FOUR CARDS. 6 query heads on 1 KV head of 256, bf8 paged KV (blocks, 1, 64, 256). The packed verify makes
one K64j decode launch per user per attention layer: G8B2, flags 0x23 (tail 0x1 | share 0x2 | extent 0x20), two entries of
8 tokens x 6 heads (48 rows, 2 row tiles), 16 cores per entry, 32 of 110 cores active, 78 idle. One card reproduces that exactly:
SDPA has no collectives, and the shapes do not depend on the mesh. Users run one after another, so a four-user layer is four
launches; the question is how much of the chip's DRAM bandwidth a candidate configuration reaches for the same bytes.

THE ARMS (names are QWEN_FAST_TP4_SDPA's values: sdpa_long_tp.CONFIGS, one table):
  served    G8B2 0x23 per user, the mesh's own worker grid. The reference every other arm is compared with.
  grid8x4, grid8x10, grid11x4, grid4x8
            the same call on another worker grid (placement only): exact by construction, SERVABLE.
  multi     ONE launch for all users, one G16 entry per user (16 tokens x 6 heads = 96 rows, 3 row tiles, flags 0x21, no share):
            16 cores per entry (U <= 6), so every user's partition, chunk ranges and tree are the served ones, and the users run
            concurrently on 16 x U cores. Not servable yet (needs a reader, a mask kernel and a pool-lent table).
  rowsplit  G4B4 per user (4 entries of 4 tokens, 24 rows, 1 row tile, 64 cores, 0x23). Not servable yet.
  ra        the served call with the KV read-ahead flag (0x8: flags 0x2B). Not servable yet (the pinned reader admits 0x23 only).
An arm that raises is DATA (the error is recorded and the sweep goes on); a hang ends the container (the per-call watchdog writes
the partial report and exits 124, and run_card_m.sh prints the reset line).

THE CASES. For each extent E (--extents, a multiple of 256: the user's 256-key family) a one-user case u1@E and a --users case
uN@E (every user at E), then 'mixed' (four users at 262,400 / 131,328 / 65,792 / 33,024) and 'skewed' (262,400 and three at 4,352)
when --users is 4. Every user has its own seeded bf8 K/V in one shared cache with a disjoint shuffled page table, the table
POISONED past E (K = 0, V = +16,384: a read past E moves every row), the narrow tail mask and cur_pos = E - 1.

THE CHECKS, per case and arm:
  exact   every row of every user, compared as int16 against the served arm in the same process: 'differing_rows' 0 is exact. An
          arm with a differing row is NOT-EXACT and is never offered for serving. The served arm's per-user bit hashes are
          recorded, so two containers (two readers, two images) compare without sharing a process.
  live    cur_pos one chunk past E (a read of the poisoned chunk) must change the served output: a call that ignored cur_pos
          would otherwise "match" everything.
  finite  every output is finite.
  factory the native log has the [QWEN-SDPA] line every launched program should have (flags, B, PNHt): a graft that is mounted is
          not necessarily the one executed.
  timing  the arm's launches (every user) as one step, 'calls' steps per trace, replayed 'iterations' times, in --rounds rounds
          with a seeded arm order per round (the rig's load drifts); the paired ratio arm / served is taken per round. Eager
          bursts are the fallback if a trace cannot be captured. GB/s = bytes of K and V actually read (2 x E x 256 x 1.0625 per
          user) over the step time, against about 405 GB/s of DRAM.

The verdict line: 'SDPA_TP4_LONG verdict=PASS|FAIL|NO-DECISION ...'. PASS: the served arm ran, its liveness control moved, every
output was finite, no factory line was missing, and at least one case was timed. The winners are reported (the fastest EXACT arm
per case kind), never applied: a name goes to QWEN_FAST_TP4_SDPA only after the job that proves it in a serving gate.

The helpers above the device part import no ttnn and no torch at module level and are tested on CPU (test_sdpa_tp4_long.py), which
also runs the whole flow on a fake device whose attention honours cur_pos, the mask and the page table per row.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
# In a checkout the neighbours are siblings; in the container everything is mounted flat beside this file (sys.path[0]), and there
# is no third parent to name.
_PARENTS = list(HERE.parents)
for _path in ((HERE.parent / 'k64j', HERE.parent / 'sdpa_decode_qwen')
              + ((_PARENTS[2] / 'scripts' / 'ci',) if len(_PARENTS) > 2 else ())):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import sdpa_long_tp  # noqa: E402

VERDICT = 'SDPA_TP4_LONG'
HEAD_ROWS = 6                      # folded query rows per token at one KV head
HEAD_DIM = 256
PAGE = 64
K_CHUNK = 256
TILE = 32
SCALE = 1.0 / 16
MAGIC = 0x51DEC000
TAIL, SHARE, EXTENT, READAHEAD = 0x1, 0x2, 0x20, 0x8
KV_BYTES_PER_ELEMENT = 1.0625      # bfloat8_b: 1,088 B per 1,024-element tile
DRAM_GBPS = 405.0                  # measured on the TP2 cards; the per-chip op is the same on every card
KV_CORES_PER_ENTRY = 16
POISON_K, POISON_V, POISON_BLOCKS = 0.0, 16384.0, 64
TOKENS = 16                        # tokens of one user in a packed block (T16)
DEFAULT_EXTENTS = (33024, 65792, 131328, 262400)
DEFAULT_STARTS = (240,)
MIXED = (262400, 131328, 65792, 33024)
SKEWED = (262400, 4352, 4352, 4352)
MAX_MULTI_USERS = 6                # B = 7 would drop below 16 cores per entry (the factory gives 110 // B)

# arm -> layout. rows: tokens per entry; entries: entries per user (launch B for per-user arms); one_launch: all users in one
# launch with `entries` entry per user; flags: the sentinel's flag set. Names and grids come from sdpa_long_tp.CONFIGS.
ARMS = {
    'served': dict(rows=8, entries=2, flags=TAIL | SHARE | EXTENT, one_launch=False, risky=False),
    'grid8x4': dict(rows=8, entries=2, flags=TAIL | SHARE | EXTENT, one_launch=False, risky=False),
    'grid8x10': dict(rows=8, entries=2, flags=TAIL | SHARE | EXTENT, one_launch=False, risky=False),
    'grid11x4': dict(rows=8, entries=2, flags=TAIL | SHARE | EXTENT, one_launch=False, risky=False),
    'grid4x8': dict(rows=8, entries=2, flags=TAIL | SHARE | EXTENT, one_launch=False, risky=False),
    'multi': dict(rows=16, entries=1, flags=TAIL | EXTENT, one_launch=True, risky=False),
    'rowsplit': dict(rows=4, entries=4, flags=TAIL | SHARE | EXTENT, one_launch=False, risky=True),
    'ra': dict(rows=8, entries=2, flags=TAIL | SHARE | EXTENT | READAHEAD, one_launch=False, risky=True),
}
REFERENCE = 'served'


# ---------------------------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------------------------

def arm_grid(arm):
    """(x, y) for a grid arm, None for the mesh's own grid."""
    return sdpa_long_tp.CONFIGS[arm]['grid']


def parse_ints(text, name, multiple=None, low=1, high=None):
    values = [int(part) for part in text.split(',') if part.strip()]
    if not values:
        raise ValueError('%s: no values' % name)
    for value in values:
        if value < low or (multiple and value % multiple) or (high is not None and value > high):
            raise ValueError('%s: %d is not an accepted value' % (name, value))
    return values


def parse_arms(text):
    names = [part.strip() for part in text.split(',') if part.strip()]
    if names == ['all']:
        names = list(ARMS)
    unknown = [name for name in names if name not in ARMS]
    if unknown or not names:
        raise ValueError('arms are %s or all, got %r' % (', '.join(ARMS), text))
    if REFERENCE not in names:
        names.insert(0, REFERENCE)           # every arm is compared with, and timed against, the served arm
    return list(dict.fromkeys(names))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extents', default=','.join(map(str, DEFAULT_EXTENTS)), help='the users\' 256-key families')
    parser.add_argument('--users', type=int, default=4, help='users in the multi-user cases (1 skips them)')
    parser.add_argument('--starts', default=','.join(map(str, DEFAULT_STARTS)),
                        help='first token of the 16-token block inside the family (0..240)')
    parser.add_argument('--arms', default='all', help='comma list of %s, or all (served always runs)' % ','.join(ARMS))
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--timing', choices=('trace', 'eager', 'none'), default='trace')
    parser.add_argument('--rounds', type=int, default=5, help='timing rounds, a seeded arm order each')
    parser.add_argument('--iterations', type=int, default=10, help='replays (or bursts) per arm per round')
    parser.add_argument('--calls', type=int, default=8, help='steps per trace (or per eager burst)')
    parser.add_argument('--order-seed', type=int, default=7)
    parser.add_argument('--no-mixed', action='store_true', help='skip the mixed and skewed cases')
    parser.add_argument('--trace-region-bytes', type=int, default=64 << 20)
    parser.add_argument('--device-id', type=int, default=int(os.environ.get('SDPA_DEVICE_ID', '0')))
    parser.add_argument('--deadline-s', type=float, default=0.0, help='stop cleanly between cases after this long (0: none)')
    parser.add_argument('--watchdog', type=float, default=300.0, help='seconds one device step may take before exit 124 (0: off)')
    parser.add_argument('--expect-binary-sha256', default='')
    parser.add_argument('--binary', default='/opt/tt-metal/build_Release/lib/_ttnncpp.so')
    args = parser.parse_args(argv)
    args.extents = parse_ints(args.extents, '--extents', multiple=K_CHUNK, low=K_CHUNK)
    args.starts = parse_ints(args.starts, '--starts', low=0, high=K_CHUNK - TOKENS)
    args.seeds = parse_ints(args.seeds, '--seeds', low=0)
    args.arms = parse_arms(args.arms)
    if not 1 <= args.users <= MAX_MULTI_USERS:
        parser.error('--users is 1..%d (a seventh entry takes the factory below 16 cores per entry)' % MAX_MULTI_USERS)
    for name in ('rounds', 'iterations', 'calls'):
        if getattr(args, name) < 1:
            parser.error('--%s must be at least 1' % name)
    return args


def busiest_chunks(extent, cores=KV_CORES_PER_ENTRY):
    """256-key chunks on the busiest of an entry's `cores` cores: the chunks are split into contiguous ranges."""
    return -(-(extent // K_CHUNK) // cores)


def kv_bytes(extent):
    """K and V bytes one user's launch reads: 2 x E x 256 x 1.0625 (bf8)."""
    return 2 * extent * HEAD_DIM * KV_BYTES_PER_ELEMENT


def gbps(extents, seconds):
    return sum(kv_bytes(extent) for extent in extents) / seconds / 1e9 if seconds > 0 else 0.0


def case_list(extents, users, mixed):
    """[{name, kind, extents}]: u1@E per extent, uN@E per extent, then mixed and skewed. kind groups cases for the winners."""
    cases = [dict(name='u1@%d' % extent, kind='u1', extents=[extent]) for extent in extents]
    if users > 1:
        cases += [dict(name='u%d@%d' % (users, extent), kind='u%d' % users, extents=[extent] * users) for extent in extents]
        if mixed and users == 4:
            cases += [dict(name='mixed', kind='mixed', extents=list(MIXED)), dict(name='skewed', kind='skewed', extents=list(SKEWED))]
    return cases


def extent_positions(extent, start, entries, rows):
    """Entry e's first position: E - 256 + start + e * rows (a block never crosses its family: start <= 240)."""
    return [extent - K_CHUNK + start + slot * rows for slot in range(entries)]


def words_for(arm, positions, extents, users):
    """The cur_pos words an arm sends: E - 1 per entry; every entry of a share bundle carries the bundle's word."""
    spec = ARMS[arm]
    if spec['one_launch']:
        return [extent - 1 for extent in extents]
    return [(positions[0] // K_CHUNK + 1) * K_CHUNK - 1] * spec['entries']


def expected_programs(arm, users):
    """The factory lines this arm's launches must have: {(flags, B, PNHt)}."""
    spec = ARMS[arm]
    entries = users * spec['entries'] if spec['one_launch'] else spec['entries']
    return {(spec['flags'], entries, -(-spec['rows'] * HEAD_ROWS // TILE))}


def launches_per_step(arm, users):
    return 1 if ARMS[arm]['one_launch'] else users


def program_lines(lines):
    return {(int(line['flags'], 16), line['B'], line['PNHt']) for line in lines}


def digest(bits_bytes):
    return hashlib.sha256(bits_bytes).hexdigest()[:16]


def paired_ratios(rounds, arm, reference=REFERENCE):
    """Per round arm / reference step time, for the rounds both ran."""
    return [round_times[arm] / round_times[reference] for round_times in rounds
            if arm in round_times and round_times.get(reference)]


def summarize_timing(rounds, arm, extents):
    """{median_us, best_us, ratio_median, ratio_min, ratio_max, gbps_median} from per-round step seconds."""
    times = [round_times[arm] for round_times in rounds if arm in round_times]
    if not times:
        return None
    ratios = paired_ratios(rounds, arm)
    median = statistics.median(times)
    entry = dict(rounds=len(times), median_us=median * 1e6, best_us=min(times) * 1e6, gbps_median=gbps(extents, median))
    if ratios:
        entry.update(ratio_median=statistics.median(ratios), ratio_min=min(ratios), ratio_max=max(ratios))
    return entry


def fit_line(points):
    """Least squares t = a + b x over [(x, t)]: (a, b), or None with fewer than two distinct x."""
    if len({x for x, _ in points}) < 2:
        return None
    n = len(points)
    mx = sum(x for x, _ in points) / n
    mt = sum(t for _, t in points) / n
    spread = sum((x - mx) ** 2 for x, _ in points)
    slope = sum((x - mx) * (t - mt) for x, t in points) / spread
    return mt - slope * mx, slope


def fits(report):
    """Per arm, over the one-user cases: t = a + b * busiest_chunks (microseconds; b is the per-chunk cost the design models)."""
    result = {}
    for case in report.get('cases', []):
        if case['kind'] != 'u1':
            continue
        for arm, state in case['arms'].items():
            timing = state.get('timing')
            if timing and timing.get('median_us') is not None:
                result.setdefault(arm, []).append((busiest_chunks(case['extents'][0]), timing['median_us']))
    out = {}
    for arm, points in result.items():
        line = fit_line(points)
        if line:
            out[arm] = dict(fixed_us=line[0], per_chunk_us=line[1], points=len(points))
    return out


def winners(report):
    """{case kind: {arm, ratio_median, gbps}}: the fastest arm per case kind that is exact on every case of that kind, by the
    median paired ratio over the cases (served excluded; it is the reference)."""
    by_kind = {}
    for case in report.get('cases', []):
        for arm, state in case['arms'].items():
            if arm == REFERENCE:
                continue
            by_kind.setdefault(case['kind'], {}).setdefault(arm, []).append(state)
    best = {}
    for kind, arms in by_kind.items():
        for arm, states in arms.items():
            if any(state.get('status') != 'ok' or state.get('differing_rows') != 0 for state in states):
                continue
            ratios = [state['timing']['ratio_median'] for state in states if state.get('timing', {}) and
                      state['timing'].get('ratio_median') is not None]
            if len(ratios) != len(states):
                continue
            mean_ratio = statistics.mean(ratios)
            if kind not in best or mean_ratio < best[kind]['ratio_mean']:
                best[kind] = dict(arm=arm, ratio_mean=mean_ratio,
                                  gbps=statistics.mean(state['timing']['gbps_median'] for state in states))
    return best


def decide(report):
    """PASS / FAIL / NO-DECISION. FAIL: a served output is not finite, or the liveness control did not move. NO-DECISION: an
    error outside an arm, no served arm, a missing factory line, a deadline cut with nothing run, or nothing timed (timing on)."""
    problems = list(report.get('failures', []))
    cases = report.get('cases', [])
    verdict_problems, fail = [], []
    served = [case['arms'].get(REFERENCE, {}) for case in cases]
    if not cases or any(state.get('status') != 'ok' for state in served):
        verdict_problems.append('the served arm did not run in every case')
    for case in cases:
        live = case.get('liveness')
        if live is not None and not live.get('moved'):
            fail.append('%s: cur_pos one chunk past E did not change the served output' % case['name'])
        for arm, state in case['arms'].items():
            if state.get('finite') is False:
                fail.append('%s/%s: a non-finite output' % (case['name'], arm))
    if report.get('timing_mode') != 'none' and not any(state.get('timing') for case in cases for state in case['arms'].values()):
        verdict_problems.append('nothing was timed')
    if fail:
        return dict(verdict='FAIL', problems=fail + problems)
    if problems or verdict_problems:
        return dict(verdict='NO-DECISION', problems=problems + verdict_problems)
    return dict(verdict='PASS', problems=[])


def verdict_line(report):
    cases = report.get('cases', [])
    arms = {}
    for case in cases:
        for arm, state in case['arms'].items():
            arms.setdefault(arm, []).append(state)
    exact = sorted(arm for arm, states in arms.items()
                   if all(state.get('status') == 'ok' and state.get('differing_rows') == 0 for state in states))
    ran = sorted(arm for arm, states in arms.items() if any(state.get('status') == 'ok' for state in states))
    best = winners(report)
    parts = ['%s verdict=%s' % (VERDICT, report['decision']['verdict']), 'cases=%d' % len(cases), 'arms_ran=%s' % ','.join(ran),
             'arms_exact=%s' % ','.join(exact)]
    for kind in sorted(best):
        parts.append('winner_%s=%s ratio=%.3f gbps=%.0f' % (kind, best[kind]['arm'], best[kind]['ratio_mean'], best[kind]['gbps']))
    gb = [state['timing']['gbps_median'] for case in cases if case['kind'] == 'u1' and case['extents'][0] == max(
        report.get('extents') or [0]) for arm, state in case['arms'].items() if arm == REFERENCE and state.get('timing')]
    if gb:
        parts.append('served_gbps_longest_u1=%.0f' % gb[0])
    return ' '.join(parts)


# ---------------------------------------------------------------------------------------------
# The device part.
# ---------------------------------------------------------------------------------------------

class Deadline:
    def __init__(self, seconds):
        self.started, self.seconds = time.time(), seconds

    def reached(self):
        return bool(self.seconds) and time.time() - self.started > self.seconds


class Watchdog:
    """If one device step runs longer than `seconds` the partial report is written and the process exits 124 (the harness
    runner's timeout code), so a hang costs the rest of the container's time, not the report."""

    def __init__(self, seconds, on_expiry):
        self.seconds, self.on_expiry = seconds, on_expiry
        self.label, self.since = None, 0.0
        self.lock = threading.Lock()
        if seconds:
            threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while True:
            time.sleep(2.0)
            with self.lock:
                label, since = self.label, self.since
            if label is not None and time.time() - since > self.seconds:
                try:
                    self.on_expiry(label)
                finally:
                    os._exit(124)

    def op(self, label):
        watchdog = self

        class Guard:
            def __enter__(self):
                with watchdog.lock:
                    watchdog.label, watchdog.since = label, time.time()

            def __exit__(self, *exc):
                with watchdog.lock:
                    watchdog.label = None
                return False

        return Guard()


class Cache:
    """One case's paged K/V: every user's clean blocks (a shuffled disjoint slice each) and 64 poison blocks, ONE KV head,
    64-key pages, bf8; and each user's page-table row over capacity / 64 pages, poisoned past E."""

    def __init__(self, rig, extents, seed):
        ttnn, torch = rig.ttnn, rig.torch
        self.rig, self.extents = rig, list(extents)
        self.capacity = max(extents) + K_CHUNK          # one chunk of poisoned pages past the longest family (the liveness read)
        self.width = self.capacity // PAGE
        generator = torch.Generator().manual_seed(4000 + seed)
        shape = (1, PAGE, HEAD_DIM)
        clean = [extent // PAGE for extent in extents]
        keys, values = [], []
        for count in clean:
            keys.append((torch.randn((count,) + shape, generator=generator) * 2).to(torch.bfloat16))
            values.append(torch.randn((count,) + shape, generator=generator).to(torch.bfloat16))
        keys.append(torch.full((POISON_BLOCKS,) + shape, POISON_K).to(torch.bfloat16))
        values.append(torch.full((POISON_BLOCKS,) + shape, POISON_V).to(torch.bfloat16))
        total = sum(clean)
        self.poison = list(range(total, total + POISON_BLOCKS))
        offset, self.rows = 0, []
        for count in clean:
            order = (torch.randperm(count, generator=generator) + offset).to(torch.int32)
            row = torch.empty(self.width, dtype=torch.int32)
            row[:count] = order
            for index in range(count, self.width):
                row[index] = self.poison[(index - count) % POISON_BLOCKS]
            self.rows.append(row)
            offset += count
        self.k = rig.upload(torch.cat(keys), ttnn.bfloat8_b)
        self.v = rig.upload(torch.cat(values), ttnn.bfloat8_b)

    def close(self):
        for tensor in (self.k, self.v):
            self.rig.ttnn.deallocate(tensor)


def query_block(torch, user, seed):
    """(96, 256) bf16: user's 16 tokens x 6 heads, token-major (at one KV head the fold is a reshape)."""
    generator = torch.Generator().manual_seed(1000 + seed + 31 * user)
    return torch.randn(TOKENS * HEAD_ROWS, HEAD_DIM, generator=generator).to(torch.bfloat16)


class Rig:
    def __init__(self, ttnn, torch, device, args, watchdog):
        self.ttnn, self.torch, self.device, self.args, self.watchdog = ttnn, torch, device, args, watchdog

    def upload(self, host, dtype=None):
        ttnn = self.ttnn
        dtype = ttnn.bfloat16 if dtype is None else dtype
        layout = ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT
        return ttnn.from_torch(host, device=self.device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def grid(self, arm):
        fixed = arm_grid(arm)
        if fixed is not None:
            return fixed
        mesh = self.device.compute_with_storage_grid_size()
        return (mesh.x, mesh.y)


class Call:
    """One arm on one case: the uploaded inputs, the launches of one step, and the unfold of their outputs to (users, 96, 256)."""

    def __init__(self, rig, arm, cache, case, seed, start):
        self.rig, self.arm, self.cache, self.start = rig, arm, cache, start
        ttnn, torch = rig.ttnn, rig.torch
        import k64j_nkv1_spike as spike

        spec = ARMS[arm]
        self.spec, self.owned, self.launches = spec, [], []
        extents = case['extents']
        self.users = len(extents)
        sentinel = MAGIC | spec['flags']
        self.config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=rig.grid(arm), exp_approx_mode=False,
                                             q_chunk_size=sentinel, k_chunk_size=K_CHUNK)
        queries = [query_block(torch, user, seed) for user in range(self.users)]

        def put(host, dtype=None):
            tensor = rig.upload(host, dtype)
            self.owned.append(tensor)
            return tensor

        if spec['one_launch']:
            masks = [spike.narrow_mask(torch, extent_positions(extent, start, 1, TOKENS), extent, TOKENS)
                     for extent in extents]
            self.launches.append(dict(
                query=put(torch.stack(queries).reshape(1, self.users, TOKENS * HEAD_ROWS, HEAD_DIM).contiguous()),
                pages=put(torch.stack([cache.rows[user] for user in range(self.users)]).contiguous(), ttnn.int32),
                words=put(torch.tensor(words_for(arm, None, extents, self.users), dtype=torch.int32), ttnn.int32),
                host_words=words_for(arm, None, extents, self.users),
                mask=put(torch.cat(masks).contiguous())))
        else:
            for user, extent in enumerate(extents):
                positions = extent_positions(extent, start, spec['entries'], spec['rows'])
                self.launches.append(dict(
                    query=put(queries[user].reshape(1, spec['entries'], spec['rows'] * HEAD_ROWS, HEAD_DIM).contiguous()),
                    pages=put(torch.stack([cache.rows[user]] * spec['entries']).contiguous(), ttnn.int32),
                    words=put(torch.tensor(words_for(arm, positions, extents, self.users), dtype=torch.int32), ttnn.int32),
                    host_words=words_for(arm, positions, extents, self.users),
                    mask=put(spike.narrow_mask(torch, positions, extent, spec['rows']))))

    def launch(self, entry, words=None):
        ttnn = self.rig.ttnn
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            entry['query'], self.cache.k, self.cache.v, page_table_tensor=entry['pages'],
            cur_pos_tensor=entry['words'] if words is None else words, is_causal=False, attn_mask=entry['mask'], scale=SCALE,
            program_config=self.config, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def step(self):
        return [self.launch(entry) for entry in self.launches]

    def unfold(self, hosts):
        """(users, 96, 256) from the step's host outputs: token-major rows, so every layout is a reshape of the user's 96 rows."""
        torch = self.rig.torch
        spec = self.spec
        if spec['one_launch']:
            block = hosts[0].reshape(1, self.users, -1, HEAD_DIM)[:, :, :TOKENS * HEAD_ROWS]
            return block[0].contiguous()
        rows = spec['rows'] * HEAD_ROWS
        return torch.stack([host.reshape(1, spec['entries'], -1, HEAD_DIM)[:, :, :rows].reshape(TOKENS * HEAD_ROWS, HEAD_DIM)
                            for host in hosts])

    def read(self):
        ttnn = self.rig.ttnn
        outputs = self.step()
        hosts = [ttnn.to_torch(output) for output in outputs]
        for output in outputs:
            ttnn.deallocate(output)
        return self.unfold(hosts)

    def close(self):
        for tensor in self.owned:
            try:
                self.rig.ttnn.deallocate(tensor)
            except BaseException:  # noqa: BLE001 - best effort on the way out
                pass
        self.owned = []


def bits_of(torch, tensor):
    return tensor.to(torch.bfloat16).contiguous().view(torch.int16)


def differing_rows(torch, left, right):
    """How many (user, row) pairs differ as int16 between two (users, 96, 256) outputs."""
    if tuple(left.shape) != tuple(right.shape):
        return int(left.shape[0] * left.shape[1])
    return int((bits_of(torch, left) != bits_of(torch, right)).any(dim=-1).sum())


def liveness(rig, served):
    """cur_pos one chunk past E for the first launch: the poisoned chunk must change the output. -> dict(moved=bool)."""
    ttnn, torch = rig.ttnn, rig.torch
    entry = served.launches[0]
    base = ttnn.to_torch(served.launch(entry))
    extra = rig.upload(torch.tensor([word + K_CHUNK for word in entry['host_words']], dtype=torch.int32), ttnn.int32)
    try:
        moved = ttnn.to_torch(served.launch(entry, words=extra))
    finally:
        ttnn.deallocate(extra)
    return dict(moved=bool((bits_of(torch, base) != bits_of(torch, moved)).any()))


def capture_trace(rig, call, args):
    """`calls` steps in one trace. -> (trace, every captured output list, whether a replay equals the eager read)."""
    ttnn = rig.ttnn
    eager = call.read()
    trace = ttnn.begin_trace_capture(rig.device, cq_id=0)
    captured = []
    try:
        for _ in range(args.calls):
            captured.append(call.step())
    finally:
        ttnn.end_trace_capture(rig.device, trace, cq_id=0)
    try:
        ttnn.execute_trace(rig.device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(rig.device)
        replayed = call.unfold([ttnn.to_torch(output) for output in captured[-1]])
        return trace, captured, differing_rows(rig.torch, eager, replayed) == 0
    except BaseException:
        ttnn.release_trace(rig.device, trace)
        raise


def time_trace(rig, call, trace):
    ttnn = rig.ttnn
    start = time.perf_counter()
    for _ in range(rig.args.iterations):
        ttnn.execute_trace(rig.device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(rig.device)
    return (time.perf_counter() - start) / (rig.args.iterations * rig.args.calls)


def time_eager(rig, call):
    ttnn = rig.ttnn
    start = time.perf_counter()
    for _ in range(rig.args.iterations):
        for _ in range(rig.args.calls):
            for output in call.step():
                ttnn.deallocate(output)
    ttnn.synchronize_device(rig.device)
    return (time.perf_counter() - start) / (rig.args.iterations * rig.args.calls)


def run_case(rig, args, report, case, seed, deadline, write):
    torch, ttnn = rig.torch, rig.ttnn
    state = dict(name=case['name'], kind=case['kind'], extents=case['extents'], seed=seed, arms={})
    report['cases'].append(state)
    cache = Cache(rig, case['extents'], seed)
    calls, traces = {}, {}
    try:
        start = args.starts[0]
        served_rows = None
        for arm in args.arms:
            if deadline.reached():
                report['deadline'] = 'stopped in %s before %s' % (case['name'], arm)
                break
            if arm == 'multi' and len(case['extents']) < 2:
                state['arms'][arm] = dict(status='skipped', error='needs two or more users')
                continue
            arm_state = state['arms'][arm] = dict(status='error')
            try:
                with rig.watchdog.op('%s/%s setup' % (case['name'], arm)):
                    call = Call(rig, arm, cache, case, seed, start)
                calls[arm] = call
                with rig.watchdog.op('%s/%s run' % (case['name'], arm)):
                    rows = call.read()
                arm_state['finite'] = bool(torch.isfinite(rows.float()).all())
                arm_state['programs'] = sorted(list(key) for key in expected_programs(arm, len(case['extents'])))
                report['requested'].update(expected_programs(arm, len(case['extents'])))
                if arm == REFERENCE:
                    served_rows = rows
                    arm_state['user_hashes'] = [digest(bits_of(torch, rows[user]).numpy().tobytes()) for user in range(rows.shape[0])]
                    arm_state['differing_rows'] = 0
                    with rig.watchdog.op('%s liveness' % case['name']):
                        state['liveness'] = liveness(rig, call)
                else:
                    arm_state['differing_rows'] = differing_rows(torch, served_rows, rows)
                    arm_state['rows'] = int(rows.shape[0] * rows.shape[1])
                    if arm_state['differing_rows']:
                        print(json.dumps(dict(case=case['name'], arm=arm, differing_rows=arm_state['differing_rows'])), flush=True)
                arm_state['status'] = 'ok'
            except BaseException as error:  # noqa: BLE001 - a refusal is data; the sweep continues
                arm_state['error'] = '%s: %s' % (type(error).__name__, str(error)[:400])
                arm_state['traceback'] = traceback.format_exc()
                print(arm_state['traceback'], flush=True)
                if arm == REFERENCE:
                    break
            write()
        if args.timing != 'none' and REFERENCE in calls and state['arms'][REFERENCE].get('status') == 'ok':
            timed = [arm for arm in args.arms if arm in calls and state['arms'][arm].get('status') == 'ok']
            modes = {}
            for arm in timed:
                modes[arm] = 'eager'
                if args.timing == 'trace':
                    try:
                        with rig.watchdog.op('%s/%s trace capture' % (case['name'], arm)):
                            trace, captured, match = capture_trace(rig, calls[arm], args)
                        traces[arm] = (trace, captured)
                        modes[arm] = 'trace'
                        state['arms'][arm]['trace_equal'] = bool(match)
                    except BaseException as error:  # noqa: BLE001 - eager bursts are the fallback
                        state['arms'][arm]['trace_error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
            rounds = []
            for index in range(args.rounds):
                if deadline.reached():
                    report['deadline'] = 'stopped in %s timing, round %d' % (case['name'], index)
                    break
                order = list(timed)
                random.Random(args.order_seed + index).shuffle(order)
                round_times = {}
                for arm in order:
                    with rig.watchdog.op('%s/%s timing' % (case['name'], arm)):
                        round_times[arm] = (time_trace(rig, calls[arm], traces[arm][0]) if modes[arm] == 'trace'
                                            else time_eager(rig, calls[arm]))
                rounds.append(round_times)
            state['timing_rounds_us'] = [{arm: value * 1e6 for arm, value in entry.items()} for entry in rounds]
            for arm in timed:
                summary = summarize_timing(rounds, arm, case['extents'])
                if summary:
                    summary['mode'] = modes[arm]
                    summary['busiest_chunks'] = max(busiest_chunks(extent) for extent in case['extents'])
                    state['arms'][arm]['timing'] = summary
        write()
    finally:
        for trace, captured in traces.values():
            try:
                ttnn.release_trace(rig.device, trace)
            except BaseException:  # noqa: BLE001
                pass
            for outputs in captured:
                for output in outputs:
                    try:
                        ttnn.deallocate(output)
                    except BaseException:  # noqa: BLE001
                        pass
        for call in calls.values():
            call.close()
        cache.close()


def write_report(args, report):
    payload = {key: value for key, value in report.items() if not key.startswith('_') and key != 'requested'}
    payload['requested_programs'] = sorted(list(key) for key in report['requested'])
    payload['fits'] = fits(report)
    payload['winners'] = winners(report)
    args.out.write_text(json.dumps(payload, indent=2, default=str))


def sha256_of(path):
    digester = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digester.update(block)
    return digester.hexdigest()


def check_binary(args, report):
    """Record the installed op binary's sha256 and hold it to --expect-binary-sha256: the K64j graft must be the one in the image."""
    try:
        got = sha256_of(args.binary)
    except OSError as error:
        report['binary'] = dict(path=args.binary, error=str(error))
        if args.expect_binary_sha256:
            report['failures'].append('cannot read %s to check it: %s' % (args.binary, error))
        return
    report['binary'] = dict(path=args.binary, sha256=got)
    if args.expect_binary_sha256 and got != args.expect_binary_sha256:
        report['failures'].append('%s is %s, not the expected graft %s' % (args.binary, got, args.expect_binary_sha256))


def check_factory_lines(report, text, card):
    """Every program an arm launched has its [QWEN-SDPA] factory line in the native log (flags, B, PNHt)."""
    lines = card.factory_lines(text)
    seen = program_lines(lines)
    report['factory_lines'] = sorted({(line['flags'], line['B'], line['PNHt'], line['kv_share'], line['scratch_slots'],
                                       line['cb_bytes']) for line in lines})
    for key in sorted(report['requested']):
        if key not in seen:
            report['failures'].append('no [QWEN-SDPA] factory line for flags=0x%x B=%d PNHt=%d' % key)


def run(args, report):
    import torch
    import ttnn

    import test_sdpa_decode_qwen_card_m as card

    if os.environ.get(card.SCRATCH_ENV) != '1':
        report['failures'].append('%s=1 is required: the G8 programs do not fit L1 without the compact scratch' % card.SCRATCH_ENV)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    check_binary(args, report)
    native = card.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    deadline = Deadline(args.deadline_s)

    def expired(label):
        report['failures'].append('watchdog: %s ran longer than %.0f s; the container exits 124' % (label, args.watchdog))
        report['decision'] = dict(verdict='NO-DECISION', problems=report['failures'])
        report['verdict_line'] = '%s verdict=NO-DECISION watchdog=%s' % (VERDICT, label)
        write_report(args, report)

    watchdog = Watchdog(args.watchdog, expired)
    with native:
        options = dict(device_id=args.device_id, l1_small_size=24576)
        if args.timing == 'trace':
            options['trace_region_size'] = args.trace_region_bytes
        device = ttnn.open_device(**options)
        try:
            try:
                device.enable_program_cache()
            except Exception:  # noqa: BLE001 - default-on in newer runtimes
                pass
            grid = device.compute_with_storage_grid_size()
            report['grid'] = [grid.x, grid.y]
            rig = Rig(ttnn, torch, device, args, watchdog)
            for case in case_list(args.extents, args.users, not args.no_mixed):
                for seed in args.seeds:
                    if deadline.reached():
                        report['deadline'] = 'stopped before %s' % case['name']
                        break
                    try:
                        run_case(rig, args, report, case, seed, deadline, lambda: write_report(args, report))
                    except BaseException as error:  # noqa: BLE001 - a case that cannot be built is data; the sweep continues
                        report['failures'].append('%s: %s: %s' % (case['name'], type(error).__name__, str(error)[:300]))
                        print(traceback.format_exc(), flush=True)
                        write_report(args, report)
        finally:
            ttnn.close_device(device)
    check_factory_lines(report, native.text(), card)


def main(argv=None):
    args = parse_args(argv)
    report = dict(plan='SDPA decode at the four-card per-chip shape: every configuration against the served one',
                  argv=list(sys.argv[1:] if argv is None else argv), extents=args.extents, users=args.users,
                  starts=args.starts, seeds=args.seeds, arms=args.arms, timing_mode=args.timing, rounds=args.rounds,
                  iterations=args.iterations, calls=args.calls, cases=[], failures=[], requested=set(),
                  env={name: os.environ.get(name) for name in ('QWEN_SDPA_TREE_SCRATCH_ROUNDS', 'TT_METAL_WATCHER')},
                  dram_gbps_reference=DRAM_GBPS, expect_binary_sha256=args.expect_binary_sha256)
    try:
        run(args, report)
    except BaseException as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
        report['traceback'] = traceback.format_exc()
    report['decision'] = decide(report)
    if report.get('error'):
        report['decision'] = dict(verdict='NO-DECISION', problems=[report['error']] + report['decision']['problems'])
    report['verdict_line'] = verdict_line(report)
    write_report(args, report)
    for problem in report['decision']['problems']:
        print('PROBLEM', problem)
    print(report['verdict_line'], flush=True)
    return 0 if report['decision']['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
