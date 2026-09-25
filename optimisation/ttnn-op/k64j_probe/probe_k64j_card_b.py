"""K64j P0 (c2-serve-for-real-plan.md section 2.3; risks R1 and R2) on the qualification card: the decode SDPA takes
its extent at RUN time and reproduces the compile-time split byte for byte, and its UINT32_MAX "skip this user"
holds - measured on the binary that serves today, with no build.

WHAT IT DECIDES. K64j would let the packed 64-row block serve users at arbitrary positions by giving the served
non-causal replay call (tail mask, fixed 256-key chunks) a runtime cur_pos word, E - 1 with E = (start // 256 + 1)
* 256, instead of the compile-time cur_pos_base = St * 32 - 1. The one kernel path that already takes its position
at run time is the CAUSAL one: reader_decode_all.cpp:113-157 reads a per-entry word from the cur_pos tensor into
c_8 (the writer's copy) and c_15 (compute's), and the reader (:167-178), the writer (writer_decode_all.cpp:117-157)
and the compute kernel (sdpa_flash_decode.cpp:139-177) then split the keys with get_workload_for_core(cur_pos) and
return at once on UINT32_MAX. Every cur_pos
in [E - 256, E - 1] gives exactly the split of the non-causal call at capacity E (split_model.py, all 513 families
x every core count, on CPU). What else differs between the two programs is known and controlled here:
  - the causal writer GENERATES the final chunk's mask (generate_mask, dataflow_common.hpp:215-305: +0.0 up to
    cur_pos, 0xFF80 (-inf) after) where the non-causal call READS the provided one; the probe's provided masks
    hold the same bits at the same columns, so both add the same values on the final chunk only (legacy adds
    +0.0 on the others: x + 0.0 == x);
  - a causal call with <= 16 bf16 heads uses a 16 x 32 HALF-tile Q (factory :449). K64j stays non-causal (full
    tile), so the decisive shapes have 24, 48 or 96 folded rows; the 12-row (native, single token) shape is run and
    RECORDED;
  - the page table is C / 64 wide (C = the served 131,328 keys) instead of E / 64, and every page past E is
    POISONED (K = 0, V = +16384: any read of one moves every row of the output), so an equal output also proves
    the runtime split reads nothing past its extent;
  - the cores per head depend on B (16 for B <= 3, 13 at B = 4), so each reference has the call's own B.

Sections (--sections, default all):
  S  the plan's P0: single-position users at arbitrary, DIFFERENT positions in ONE call. B = --batch entries (3:
     16 cores per head, as every served replay shape), each a user with its own page table (C / 64 wide, poisoned
     past its E) at p = E - 256 + s for E in --extents (K1's six: 2,304 ... 131,328) and s in --starts (+0, +7,
     +127, +240, +255), --rows 2 (two query tokens at p, 24 folded rows: full tile, DECISIVE) and 1 (12 rows:
     half tile, RECORDED), --variants normal and peaky (rows aimed at 8 visible keys each, so the running max
     moves across chunks, cores and tree rounds), --seeds. One causal call per group of B (E, s) pairs, against
     per entry: the non-causal LEGACY call at capacity E (B copies of that entry: its Q, its first E / 64 pages
     and the mask -inf past p in the last 256 columns) - DECISIVE; and the qwen modes on the same inputs (tail
     0x1, tail+share 0x3 where the binary has them) against legacy - RECORDED and warned (12 and 24 rows were
     never qualified for them). Liveness, per group and entry with E < C: the causal call at cur_pos = E (one key into the
     next chunk, a poisoned page) must move the entry against cur_pos = E - 1 - the plan's K3 "cur_pos off by one
     chunk must change the output", and the proof that the poison is live.
  E  the replay shapes at cur_pos = E - 1: G4B3 (4-row groups, B = 3, 48 rows, PNHt 2) and G8B2 (8-row groups,
     B = 2, 96 rows, PNHt 3), every entry the same user, causal on the poisoned C-wide table against the legacy
     non-causal call at capacity E with a zero mask (nothing is masked at E - 1) - DECISIVE - plus the served
     flags on the same inputs (0x1, 0x3; 0x7 on G8 when the binary has the q-slice), and the cur_pos = E
     liveness.
  T  one TRACE of the causal call (rows 2, B entries, clean C-wide tables), replayed --trace-families times with
     the cur_pos tensor rewritten between replays (copy_host_to_device_tensor, as attention_replay.stage does the
     positions word) so the extent moves across that many 256-key families; every replay == the eager causal
     call at its positions (every entry), and == the compile-time legacy call at the entry's E (--trace-references:
     entry 0 by default, all entries, or none) - all DECISIVE. Then UINT32_MAX skip patterns replayed in the same trace: the live entries == eager.
  K  R2, eager: rows 2, B entries at three families, cur_pos = UINT32_MAX (-1 as int32) on the entries of each
     pattern in SKIP_PATTERNS (and a B = 1 call that skips its only user). No hang (the watchdog), the live
     entries == the all-live call - DECISIVE - and each skipped entry's rows: written or left unwritten (a NaN
     tensor of the output's shape is freed just before, so unwritten rows show as NaN: RECORDED, it is what K64j's
     consumers must expect).
  D  the dynamic chunk, RECORDED: causal k_chunk_size 0 (the native decode's config, docker/qwen-c2-graft/graft/
     attention/tp.py:737-742) against k_chunk_size 256 at the same positions, rows 1 and 2. With the default
     compute config max_dynamic_chunk_size is 8 (fp32_dest_acc_en false), so the native chunk is 256 keys too.
  N  the flag bit and the refusal K64j relaxes, RECORDED: 0x20 (proposed for K64j; 0x10 is the card tests'
     unknown-flag control) must be refused as unknown by the loaded binary ('[QWEN-SDPA] unknown flags'), and a
     qwen sentinel on a causal call is refused ('modes are non-causal ... take no cur_pos tensor',
     apply_factory_qwen.py:71-72).
  timing (unless --no-timing; RECORDED, never a failure): eager medians, --rounds interleaved: causal at E - 1 on
     the C-wide table against the legacy call at capacity E and at C (the runtime extent pays for E keys, not C),
     and the skip saving at cur_pos C - 1 (0 / 1 / 2 / B entries skipped).

Verdict: one 'K64J_P0 verdict=...' line.
  GO           every DECISIVE comparison is byte-equal (torch.equal on int16 views), every liveness control
               moved, no failure: the runtime position reproduces the compile-time split on this binary, and the
               skip holds. K64j's gates (K1-K4) still have to prove the new non-causal branch.
  NO-GO        a decisive comparison differs on an otherwise valid run: R1 or R2 is real on this binary (the plan's
               fallback is P1, decision D-b).
  NO-DECISION  a failure (wrong binary or kernels, no compact scratch, a served mode that differs from legacy, a
               requested qwen program without its factory line, an error, the watchdog), a dead liveness control,
               or no decisive comparison ran.

Failures: the loaded _ttnncpp.so is not --expect-binary-sha256; the stock kernels in the mounted op directory are
not the ones this reading is of (49a05926 / 734c90c0 / d24769bd / e4623a22 / 1b52c60d, --kernel-root);
QWEN_SDPA_TREE_SCRATCH_ROUNDS is not 1 (the G8 legacy calls need the compact scratch); a served mode that differs
from legacy; a requested qwen program with no '[QWEN-SDPA] flags=' line; a non-finite reference; the watchdog.

RUN with run_card_b.sh only (QUAL_CARD, default card B; the serving pair is refused without ALLOW_SERVING_CARD=1),
in the C2 image with the served graft (K64i) mounted as the arm mounts it and a fresh kernel cache. The helpers
above the device section import no ttnn and are tested on CPU by test_k64j_probe.py, which also runs the whole
flow on a fake ttnn whose broken variants the controls must catch.
"""

import argparse
import json
import mmap
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / 'sdpa_decode_qwen'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import probe_k1_card_b as k1  # noqa: E402 - the watchdog, the output address, summary, file hashes
import split_model  # noqa: E402 - the split, the cores per head, the stale-writer hazard
import test_sdpa_decode_qwen_card_m as card  # noqa: E402 - fold, query, digest, NativeLog, binary markers

PROBE = 'K64J_P0'
PLAN = 'c2-serve-for-real-plan.md section 2.3 (P0, K1 capacities, K3 skip); risks R1, R2'
CAPACITY = 131328                                         # the served geometry: 2,052 pages of 64 keys
EXTENTS = (2304, 16896, 33024, 65792, 98560, 131328)      # K1's six families (plan 2.3, hardware iterations)
STARTS = (0, 7, 127, 240, 255)                            # p = E - 256 + s
SEEDS = (0, 1, 2)
VARIANTS = ('normal', 'peaky')
ROWS = (2, 1)                                             # query tokens per entry: 2 decisive, 1 recorded
DECISIVE_ROWS = 2
BATCH = 3
E_SHAPES = {'G4B3': (4, 3), 'G8B2': (8, 2)}               # the replay's bundles: rows per fold group, entries
SECTIONS = ('S', 'E', 'T', 'K', 'D', 'N')
TRACE_FAMILIES = 64
SKIP_PATTERNS = ((0,), (1,), (2,), (0, 2), (0, 1, 2))
SKIP_EXTENTS = (2304, 65792, 131328)
POISON_K, POISON_V = 0.0, 16384.0                         # exact in bf8; any read moves every row
POISON_BLOCKS = 64                                        # card.num_blocks's +64 spare blocks
MAGIC, LEGACY = card.MAGIC, card.LEGACY
TAIL, SHARE, SLICE = 0x1, 0x2, 0x4
K64J_FLAG = 0x20                                          # proposed; the harness only checks it is unknown today
UNKNOWN_NEEDLE = '[QWEN-SDPA] unknown flags'
CAUSAL_NEEDLE = 'modes are non-causal'
SLICE_BINARY_MARKER = b'[QWEN-SDPA] q-slice rows_per_kv='
K64I_TTNNCPP_SHA256 = 'cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4'
KERNEL_ROOT = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels'
# The stock decode kernels (tt-metal 9f9cd4fd; build_k64e.sh:48-51 pins the first four): the causal and the
# legacy programs this probe runs are built from exactly these, and the reading in the docstring is of them.
STOCK_KERNELS = {
    'dataflow/reader_decode_all.cpp': '49a05926b437e2ca90d7e01c60e85a6b11f333375a6159fa02c9ff6f78af764e',
    'dataflow/writer_decode_all.cpp': '734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8',
    'compute/sdpa_flash_decode.cpp': 'd24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f',
    'dataflow/dataflow_common.hpp': 'e4623a2254559eaec4450ebfab0f9c5732e02acfe4d8126bd5eeb7efe0fdc608',
    'rt_args_common.hpp': '1b52c60d78ada6f08effd326c2ed2407b3a74cf0db2353fadbe51b088610aec8',
}
RECORDED_KERNELS = ('dataflow/reader_decode_qwen.cpp', 'compute/sdpa_flash_decode_qwen.cpp',
                    'dataflow/reader_decode_qwen_slice.cpp', 'dataflow/writer_decode_qwen_slice.cpp')
OPEN_EXTRA_S = 600.0
ENV_RECORDED = (card.SCRATCH_ENV, 'TT_METAL_WATCHER', 'TT_METAL_CACHE', 'TT_METAL_HOME')
DECISIVE_KINDS = ('split_vs_legacy', 'extent_vs_legacy', 'trace_vs_eager', 'trace_vs_legacy', 'skip_live',
                  'trace_skip_live')

clock = time.perf_counter
WATCHDOG = k1.Watchdog(0)


# ---------------------------------------------------------------------------------------------
# Pure helpers (no ttnn).
# ---------------------------------------------------------------------------------------------

def check_capacity(value, name='capacity'):
    if type(value) is not int or value < split_model.K_CHUNK or value % split_model.K_CHUNK:
        raise ValueError('%s must be a positive multiple of %d, got %r' % (name, split_model.K_CHUNK, value))
    return value


def position_pairs(extents, starts):
    """(E, s, p) for every extent and start: p = E - 256 + s, a position in family E."""
    return [(extent, start, extent - split_model.K_CHUNK + start) for extent in extents for start in starts]


def group_calls(pairs, batch):
    """The pairs in calls of `batch` entries. Each call's entries walk the pair list with a stride of
    len(pairs) // batch, so one call mixes families; the last calls wrap around to stay full."""
    if batch < 1:
        raise ValueError('batch must be >= 1')
    count = -(-len(pairs) // batch)
    stride = max(1, len(pairs) // batch)
    return [[pairs[(index + slot * stride) % len(pairs)] for slot in range(batch)] for index in range(count)]


def trace_positions(capacity, families, batch, seed):
    """`families` replays of `batch` positions whose families spread over every family the capacity serves: replay
    i's entry b sits in family (i * (F - 1) // (families - 1) + b * F // batch) mod F, at a random offset. The
    first replay's entries are all in the first families, the last's in the last ones."""
    table = split_model.families(capacity)
    count = len(table)
    rng = random.Random(9000 + seed)
    plan = []
    for index in range(families):
        base = 0 if families == 1 else index * (count - 1) // (families - 1)
        entry = []
        for slot in range(batch):
            extent = table[(base + slot * count // batch) % count]
            entry.append(extent - split_model.K_CHUNK + rng.randrange(split_model.K_CHUNK))
        plan.append(tuple(entry))
    return plan


def distinct_families(plan):
    return len({split_model.extent(position) for positions in plan for position in positions if position >= 0})


def skip_positions(positions, pattern):
    """The positions with the pattern's entries set to -1 (UINT32_MAX as int32: skip this user)."""
    return tuple(-1 if slot in pattern else position for slot, position in enumerate(positions))


def poisoned_row(torch, table, extent, capacity, poison):
    """One user's C / 64-wide page-table row: the first E / 64 pages its own, the rest cycling through the poison
    blocks. Only pages past the extent are poisoned, so the runtime split must read none of them."""
    pages = capacity // card.PAGE
    keep = extent // card.PAGE
    row = torch.empty(pages, dtype=torch.int32)
    row[:keep] = table[:keep]
    tail = pages - keep
    if tail:
        row[keep:] = torch.tensor([poison[index % len(poison)] for index in range(tail)], dtype=torch.int32)
    return row


def position_mask(torch, positions, extents, rows, width, zero=False):
    """(B, 1, rows * 12, width) bf16: zeros, and in the LAST 256 columns, which are the cache positions
    [E - 256, E) of each entry's family, -inf where the position is past the entry's p (every folded row of an
    entry sits at p: the causal writer's generated mask, dataflow_common.hpp:215-305). width is E for the full
    mask (legacy reads every chunk of it) or 256 for the narrow one; zero=True gives the all-zero mask."""
    batch = len(positions)
    mask = torch.zeros(batch, 1, rows * 12, width, dtype=torch.float32)
    if not zero:
        for slot, (position, extent) in enumerate(zip(positions, extents)):
            cache = torch.arange(extent - split_model.K_CHUNK, extent, dtype=torch.int64)
            mask[slot, 0, :, width - split_model.K_CHUNK:] = torch.where(
                cache > position, torch.tensor(float('-inf')), torch.tensor(0.0))[None, :]
    return mask.to(torch.bfloat16)


def rows_query(torch, batch, rows, seed, variant, keys=None, tables=None, positions=None, salt=0):
    """(1, batch, rows * 12, 256) bf16, each entry `rows` query tokens folded KV-head major
    (attention_head_fold.fold_query, card.fold_tokens). 'peaky': each token head is 6 x the unit vectors of 8
    keys the entry can see (positions <= p, through its page table) plus 0.1 x noise, so those scores dominate
    wherever they fall. 'zeroq': every fourth folded row zero (+-0 scores)."""
    generator = torch.Generator().manual_seed(7000 + 97 * seed + salt)
    tokens = torch.randn(batch, rows, 12, card.HEAD_DIM, generator=generator)
    if variant == 'peaky':
        if keys is None or tables is None or positions is None:
            raise ValueError('Peaky queries need the host keys, the tables and the positions')
        tokens *= 0.1
        for slot in range(batch):
            visible = int(positions[slot]) + 1
            for token in range(rows):
                for head in range(12):
                    kv = head // 6
                    for position in torch.randint(0, visible, (8,), generator=generator).tolist():
                        vector = keys[int(tables[slot][position // card.PAGE]), kv, position % card.PAGE].float()
                        tokens[slot, token, head] += 6 * vector / vector.norm().clamp_min(1e-3)
    elif variant == 'zeroq':
        pass
    elif variant != 'normal':
        raise ValueError('Unknown query variant %r' % (variant,))
    folded = torch.cat([card.fold_tokens(tokens[slot:slot + 1]) for slot in range(batch)], dim=1)
    if variant == 'zeroq':
        folded[0, :, ::4] = 0.0
    return folded.to(torch.bfloat16)


def moved_rows(torch, left, right):
    """Folded rows (per entry) with any differing element: {entry: count}."""
    a, b = card.int16_view(torch, left), card.int16_view(torch, right)
    moved = (a != b).any(dim=-1)[0]                        # (B, rows)
    return {slot: int(moved[slot].sum()) for slot in range(moved.shape[0])}


def comparison(section, kind, label, differing, decisive, **extra):
    entry = dict(section=section, kind=kind, label=label, differing=int(differing), decisive=bool(decisive))
    entry.update(extra)
    return entry


def tally(comparisons):
    """Per kind: runs, equal, decisive."""
    out = {}
    for entry in comparisons:
        row = out.setdefault(entry['kind'], dict(runs=0, equal=0, decisive=entry['decisive']))
        row['runs'] += 1
        row['equal'] += entry['differing'] == 0
    return out


def decide(report):
    """The verdict from the report's comparisons, liveness, failures and error."""
    comparisons = report.get('comparisons', [])
    decisive = [entry for entry in comparisons if entry['decisive']]
    differing = [entry for entry in decisive if entry['differing']]
    reasons = []
    if report.get('failures'):
        reasons.append('measurement invalid: %d failures' % len(report['failures']))
    if report.get('error'):
        reasons.append('error: %s' % report['error'])
    dead = [entry for entry in report.get('liveness', []) if not entry['live']]
    if dead:
        reasons.append('%d liveness controls did not move (%s): the poison or the off-by-one control is dead'
                       % (len(dead), ', '.join(entry['label'] for entry in dead[:4])))
    if not decisive:
        reasons.append('no decisive comparison ran')
    if reasons:
        verdict = 'NO-DECISION'
    elif differing:
        verdict = 'NO-GO'
    else:
        verdict = 'GO'
    return dict(verdict=verdict, reasons=reasons, decisive=len(decisive), decisive_differing=len(differing),
                first_differing=[entry['label'] for entry in differing[:6]])


def verdict_line(report):
    decision = report['decision']
    counts = tally(report.get('comparisons', []))
    words = [PROBE, 'verdict=%s' % decision['verdict']]

    def part(name, kinds):
        runs = sum(counts.get(kind, {}).get('runs', 0) for kind in kinds)
        equal = sum(counts.get(kind, {}).get('equal', 0) for kind in kinds)
        return '%s=%d/%d' % (name, equal, runs) if runs else '%s=none' % name

    words.append(part('split', ('split_vs_legacy',)))
    words.append(part('extent', ('extent_vs_legacy',)))
    words.append(part('trace', ('trace_vs_eager', 'trace_vs_legacy')))
    words.append(part('skip', ('skip_live', 'trace_skip_live')))
    words.append(part('served', ('served_vs_legacy',)))
    words.append(part('served_pnht1', ('served_unqualified_vs_legacy',)))
    liveness = report.get('liveness', [])
    words.append('live=%d/%d' % (sum(1 for entry in liveness if entry['live']), len(liveness)))
    words.append(part('half_tile', ('half_tile_vs_legacy',)))
    words.append(part('dynamic_chunk', ('dynamic_vs_fixed',)))
    skipped = report.get('skip_written')
    if skipped:
        words.append('skipped_rows=%s' % skipped)
    words.append('flag_0x20=%s' % report.get('flag_0x20', 'n/a'))
    words.append('families=%d' % report.get('trace_families_distinct', 0))
    words.append('binary_stage=%s' % report.get('binary', {}).get('stage', '?'))
    if decision['first_differing']:
        words.append('first_differing=%s' % json.dumps(decision['first_differing']))
    if decision['reasons']:
        words.append('reasons=%s' % json.dumps(decision['reasons']))
    return ' '.join(words)


def split_predictions(extents, capacity, batches):
    """What split_model says the hardware should do, recorded beside the measurements."""
    out = []
    for extent in extents:
        for batch in sorted(set(batches)):
            cores = split_model.cores_per_head(batch)
            live = split_model.split(extent - 1, cores)
            out.append(dict(extent=extent, batch=batch, cores_per_head=cores, chunks=live['num_chunks'],
                            busiest_core_chunks=max(end - start for start, end in live['ranges']),
                            same_split_as_capacity_call=split_model.same_split(extent - 1, extent, cores),
                            stale_writer_blocks=len(split_model.stale_writer_hangs(extent - 1, capacity, cores))))
    return out


def slice_marker(path):
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return view.find(SLICE_BINARY_MARKER) >= 0


def binary_stage(markers, sliced):
    """4: K64i (q-slice); 3: K64f/K64g (share); 1: K64e (tail); 0: stock, no [QWEN-SDPA] branch."""
    stage = card.binary_stage(markers)
    if sliced:
        if stage != 3:
            raise RuntimeError('A q-slice literal without the stage-3 markers: %r' % (markers,))
        return 4
    return stage


def q_slice_saves(rows, kv_heads=card.KV_HEADS):
    """pooled_attention_replay.q_slice_saves, the stage-4 factory's rule (F14/F15): each KV head's rows * 6 folded
    rows span row tiles [floor(h G / 32), ceil((h + 1) G / 32)); 0x4 builds only when the widest span is narrower
    than Q's own row tiles (6-8 rows per group; 1-5 rows save nothing and are refused)."""
    folded = rows * 12
    if kv_heads <= 1 or folded % kv_heads:
        return False
    per_kv = folded // kv_heads
    tile = split_model.TILE
    widest = max(-(-((head + 1) * per_kv) // tile) - (head * per_kv) // tile for head in range(kv_heads))
    return widest < -(-folded // tile)


def served_flags(stage, rows, batch):
    """The served modes this binary can run on a shape, beside legacy: tail (stage >= 1), tail+share (stage >= 3,
    B > 1), tail+share+slice (stage 4, where the slice saves a tile: the served 8-row groups)."""
    flags = []
    if stage >= 1:
        flags.append(TAIL)
    if stage >= 3 and batch > 1:
        flags.append(TAIL | SHARE)
    if stage >= 4 and batch > 1 and q_slice_saves(rows):
        flags.append(TAIL | SHARE | SLICE)
    return flags


def program_key(capacity, batch, mask_width, flags):
    return (flags, batch, capacity // split_model.TILE, mask_width // split_model.TILE)


def missing_programs(lines, requested):
    """Requested qwen programs (program_key) without a '[QWEN-SDPA] flags=' factory line: graft mounted is not graft
    executed. PNHt is not compared (under 0x4 the factory prints the slice)."""
    found = {(int(line['flags'], 16), line['B'], line['St'], line['mask_width_t']) for line in lines}
    return sorted(key for key in requested if key not in found)


# ---------------------------------------------------------------------------------------------
# Device harness.
# ---------------------------------------------------------------------------------------------

class Pool:
    """One seed's bf8 paged K/V pool (C / 64 + 64 blocks, 2 KV heads, 64-key pages): the first C / 64 blocks
    random (K ~ 2 N(0, 1), V ~ N(0, 1)), the last 64 the poison (K = 0, V = +16384); and one page table per user
    (a random permutation of the clean blocks)."""

    def __init__(self, ttnn, torch, device, capacity, seed, users, report):
        self.ttnn, self.torch, self.device, self.capacity, self.report = ttnn, torch, device, capacity, report
        clean = capacity // card.PAGE
        blocks = clean + POISON_BLOCKS
        generator = torch.Generator().manual_seed(4000 + seed)
        keys = torch.randn(blocks, card.KV_HEADS, card.PAGE, card.HEAD_DIM, generator=generator) * 2
        values = torch.randn(blocks, card.KV_HEADS, card.PAGE, card.HEAD_DIM, generator=generator)
        keys[clean:] = POISON_K
        values[clean:] = POISON_V
        self.keys = keys.to(torch.bfloat16)
        self.poison = list(range(clean, blocks))
        self.tables = [torch.randperm(clean, generator=generator).to(torch.int32) for _ in range(users)]
        self.k = self.v = None
        self.k = self.upload(self.keys, ttnn.bfloat8_b)
        self.v = self.upload(values.to(torch.bfloat16), ttnn.bfloat8_b)

    def upload(self, host, dtype=None):
        ttnn = self.ttnn
        dtype = ttnn.bfloat16 if dtype is None else dtype
        layout = ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT
        with WATCHDOG.op('upload %s' % (tuple(host.shape),)):
            return ttnn.from_torch(host, device=self.device, dtype=dtype, layout=layout,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def config(self, sentinel=LEGACY, k_chunk=split_model.K_CHUNK):
        grid = self.device.compute_with_storage_grid_size()
        return self.ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False,
                                           q_chunk_size=sentinel, k_chunk_size=k_chunk)

    def launch(self, query, pages, *, causal, cur_pos=None, mask=None, sentinel=LEGACY, k_chunk=split_model.K_CHUNK,
               label='sdpa', expect_program=True):
        """One device call; returns the output tensor (the caller reads and frees it). A qwen sentinel's program is
        recorded as requested (its factory line must appear) unless the call is a refusal control."""
        options = dict(is_causal=causal, scale=card.SCALE, program_config=self.config(sentinel, k_chunk),
                       memory_config=self.ttnn.DRAM_MEMORY_CONFIG)
        if mask is not None:
            options['attn_mask'] = mask
        if cur_pos is not None:
            options['cur_pos_tensor'] = cur_pos
        if sentinel != LEGACY and expect_program:
            self.report['_requested'].add(program_key(int(pages.shape[1]) * card.PAGE, int(query.shape[1]),
                                                      int(mask.shape[3]), sentinel & 0xFF))
        with WATCHDOG.op(label):
            return self.ttnn.transformer.paged_scaled_dot_product_attention_decode(query, self.k, self.v, pages,
                                                                                   **options)

    def host(self, tensor, label='read back'):
        with WATCHDOG.op(label):
            result = self.ttnn.to_torch(tensor)
        self.ttnn.deallocate(tensor)
        return result

    def run(self, *args, **kwargs):
        return self.host(self.launch(*args, **kwargs), 'read back %s' % kwargs.get('label', 'sdpa'))

    def positions(self, values):
        return self.upload(self.torch.tensor(list(values), dtype=self.torch.int32), self.ttnn.int32)

    def pages_causal(self, users, extents, poison=True):
        """(B, C / 64): entry b's user table, poisoned past its extent (or clean everywhere)."""
        torch = self.torch
        rows = []
        for user, extent in zip(users, extents):
            table = self.tables[user]
            rows.append(poisoned_row(torch, table, extent, self.capacity, self.poison) if poison
                        else table[:self.capacity // card.PAGE].clone())
        return self.upload(torch.stack(rows).contiguous(), self.ttnn.int32)

    def pages_reference(self, user, extent, batch):
        """(B, E / 64): the user's first E / 64 pages, repeated (the compile-time call at capacity E)."""
        row = self.tables[user][:extent // card.PAGE]
        return self.upload(row.repeat(batch, 1).contiguous(), self.ttnn.int32)

    def poison_output(self, shape):
        """N-S0: a NaN tensor of the output's shape, uploaded and freed: its address, which the next output of this
        size usually takes, so rows a kernel leaves unwritten show as NaN."""
        tensor = self.upload(self.torch.full(shape, float('nan'), dtype=self.torch.bfloat16))
        try:
            return k1.buffer_address(tensor)
        finally:
            self.ttnn.deallocate(tensor)

    def close(self):
        for tensor in (self.k, self.v):
            if tensor is not None:
                self.ttnn.deallocate(tensor)
        self.k = self.v = None


class Scope:
    """Device tensors to free when a case ends."""

    def __init__(self, ttnn):
        self.ttnn, self.tensors = ttnn, []

    def keep(self, tensor):
        self.tensors.append(tensor)
        return tensor

    def close(self):
        for tensor in self.tensors:
            self.ttnn.deallocate(tensor)
        self.tensors = []


def slot(output, index):
    return output[:, index:index + 1]


def record(report, entry, verbose=True):
    report['comparisons'].append(entry)
    if entry['differing'] and entry['kind'] == 'served_vs_legacy':
        report['failures'].append('%s: served mode differs from legacy in %d elements (the mounted graft is not the '
                                  'qualified one)' % (entry['label'], entry['differing']))
    elif entry['differing'] and entry['kind'] == 'served_unqualified_vs_legacy':
        report['warnings'].append('%s: a qwen mode on a never-qualified row count differs from legacy in %d elements '
                                  '(recorded; not what K64j serves)' % (entry['label'], entry['differing']))
    if verbose:
        print('%-18s %-60s %s' % (entry['kind'], entry['label'],
                                  'equal' if not entry['differing'] else 'DIFFERS(%d)' % entry['differing']),
              flush=True)


def finite_or_fail(torch, report, label, output):
    if not bool(torch.isfinite(output.float()).all()):
        report['failures'].append('%s: the reference output is not finite (the inputs are broken)' % label)


def liveness(torch, report, section, label, before, after, slots):
    """cur_pos = E against E - 1 (one poisoned key more): every listed entry must move."""
    moved = moved_rows(torch, before, after)
    for index in slots:
        rows = before.shape[2]
        entry = dict(section=section, label='%s/entry%d' % (label, index), moved_rows=moved[index], rows=rows,
                     live=moved[index] > 0)
        report['liveness'].append(entry)
        if moved[index] < rows:
            report['warnings'].append('%s: the off-by-one-chunk control moved %d of %d rows' % (entry['label'],
                                                                                              moved[index], rows))


def section_split(ttnn, torch, pool, seed, args, report):
    """S: single-position users at different positions in one call."""
    stage = report['binary']['stage']
    pairs = position_pairs(args.extents, args.starts)
    for rows in args.rows:
        decisive = rows == DECISIVE_ROWS
        kind = 'split_vs_legacy' if decisive else 'half_tile_vs_legacy'
        for index, group in enumerate(group_calls(pairs, args.batch)):
            extents = [extent for extent, _start, _p in group]
            positions = [p for _extent, _start, p in group]
            users = list(range(args.batch))
            base = 'S/rows%d/seed%d/call%d' % (rows, seed, index)
            scope = Scope(ttnn)
            try:
                pages = scope.keep(pool.pages_causal(users, extents))
                cur_pos = scope.keep(pool.positions(positions))
                references = {}
                for slot_index, (extent, start, p) in enumerate(group):
                    references[slot_index] = dict(
                        pages=scope.keep(pool.pages_reference(users[slot_index], extent, args.batch)),
                        wide=scope.keep(pool.upload(position_mask(torch, [p] * args.batch, [extent] * args.batch,
                                                                  rows, extent))))
                for variant in args.variants:
                    host_query = rows_query(torch, args.batch, rows, seed, variant, pool.keys,
                                            [pool.tables[user] for user in users], positions, salt=index)
                    query = scope.keep(pool.upload(host_query))
                    causal = pool.run(query, pages, causal=True, cur_pos=cur_pos,
                                      label='%s/%s causal' % (base, variant))
                    for slot_index, (extent, start, p) in enumerate(group):
                        label = '%s/%s/entry%d/E%d+%d' % (base, variant, slot_index, extent, start)
                        ref = references[slot_index]
                        repeated = scope.keep(pool.upload(host_query[:, slot_index:slot_index + 1]
                                                          .repeat(1, args.batch, 1, 1).contiguous()))
                        legacy = pool.run(repeated, ref['pages'], causal=False, mask=ref['wide'],
                                          label=label + ' legacy')
                        finite_or_fail(torch, report, label, legacy)
                        record(report, comparison('S', kind, label,
                                                  card.differing(torch, slot(causal, slot_index), slot(legacy, 0)),
                                                  decisive, rows=rows, batch=args.batch, extent=extent, start=start,
                                                  position=p, seed=seed, variant=variant))
                        for flags in served_flags(stage, rows, args.batch):
                            # 12 / 24 folded rows (PNHt 1) were never qualified for the qwen modes (the replay
                            # serves 48 and 96): recorded and warned, never a failure (section E's are).
                            served = pool.run(repeated, ref['pages'], causal=False, mask=ref['wide'],
                                              sentinel=MAGIC | flags, label=label + ' 0x%x' % flags)
                            record(report, comparison('S', 'served_unqualified_vs_legacy', '%s/0x%x' % (label, flags),
                                                      card.differing(torch, served, legacy), False, flags=flags))
                # Liveness: cur_pos = E (one poisoned key more) against E - 1, on the first variant's query.
                live_slots = [slot_index for slot_index, extent in enumerate(extents) if extent < pool.capacity]
                if live_slots:
                    host_query = rows_query(torch, args.batch, rows, seed, args.variants[0], pool.keys,
                                            [pool.tables[user] for user in users], positions, salt=index)
                    query = scope.keep(pool.upload(host_query))
                    last = scope.keep(pool.positions([extent - 1 for extent in extents]))
                    beyond = scope.keep(pool.positions([extent if extent < pool.capacity else extent - 1
                                                        for extent in extents]))
                    before = pool.run(query, pages, causal=True, cur_pos=last, label=base + ' E-1')
                    after = pool.run(query, pages, causal=True, cur_pos=beyond, label=base + ' E')
                    liveness(torch, report, 'S', base, before, after, live_slots)
            finally:
                scope.close()


def section_extent(ttnn, torch, pool, seed, args, report):
    """E: the replay shapes, causal at E - 1 on the poisoned C-wide table against the call at capacity E."""
    stage = report['binary']['stage']
    for name in args.shapes:
        rows, batch = E_SHAPES[name]
        host_query = card.build_query(torch, batch, seed, 'normal', rows=rows)
        for extent in args.extents:
            label = 'E/%s/seed%d/E%d' % (name, seed, extent)
            scope = Scope(ttnn)
            try:
                query = scope.keep(pool.upload(host_query))
                pages = scope.keep(pool.pages_causal([0] * batch, [extent] * batch))
                last = scope.keep(pool.positions([extent - 1] * batch))
                causal = pool.run(query, pages, causal=True, cur_pos=last, label=label + ' causal')
                reference_pages = scope.keep(pool.pages_reference(0, extent, batch))
                wide = scope.keep(pool.upload(position_mask(torch, [extent - 1] * batch, [extent] * batch, rows,
                                                            extent, zero=True)))
                legacy = pool.run(query, reference_pages, causal=False, mask=wide, label=label + ' legacy')
                finite_or_fail(torch, report, label, legacy)
                record(report, comparison('E', 'extent_vs_legacy', label, card.differing(torch, causal, legacy), True,
                                          shape=name, rows=rows, batch=batch, extent=extent, seed=seed))
                for flags in served_flags(stage, rows, batch):
                    served = pool.run(query, reference_pages, causal=False, mask=wide, sentinel=MAGIC | flags,
                                      label=label + ' 0x%x' % flags)
                    record(report, comparison('E', 'served_vs_legacy', '%s/0x%x' % (label, flags),
                                              card.differing(torch, served, legacy), False, flags=flags))
                if extent < pool.capacity:
                    beyond = scope.keep(pool.positions([extent] * batch))
                    after = pool.run(query, pages, causal=True, cur_pos=beyond, label=label + ' E')
                    liveness(torch, report, 'E', label, causal, after, range(batch))
            finally:
                scope.close()


def section_trace(ttnn, torch, device, pool, seed, args, report):
    """T: one trace, the extent rewritten between replays across --trace-families families; then skips."""
    rows, batch = DECISIVE_ROWS, args.batch
    plan = trace_positions(pool.capacity, args.trace_families, batch, seed)
    report['trace_families_distinct'] = max(report.get('trace_families_distinct', 0), distinct_families(plan))
    users = list(range(batch))
    scope = Scope(ttnn)
    trace = None
    try:
        host_query = rows_query(torch, batch, rows, seed, 'normal', salt=500)
        query = scope.keep(pool.upload(host_query))
        pages = scope.keep(pool.pages_causal(users, [pool.capacity] * batch, poison=False))
        cur_pos = scope.keep(pool.positions(plan[0]))
        # Compile first: capture needs the program built (and the eager bytes are the first comparison anyway).
        eager = {plan[0]: pool.run(query, pages, causal=True, cur_pos=cur_pos, label='T eager warm')}
        with WATCHDOG.op('T capture'):
            trace, output = card.capture(ttnn, device, lambda: pool.launch(query, pages, causal=True, cur_pos=cur_pos,
                                                                           label='T capture'))
        scope.keep(output)

        def replay(values, label):
            source = ttnn.from_torch(torch.tensor(list(values), dtype=torch.int32), dtype=ttnn.int32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT)
            with WATCHDOG.op('stage ' + label):
                ttnn.copy_host_to_device_tensor(source, cur_pos)
            with WATCHDOG.op('replay ' + label):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                ttnn.synchronize_device(device)
            with WATCHDOG.op('trace read back ' + label):
                return ttnn.to_torch(output)

        def eager_at(values, label):
            if values not in eager:
                positions = pool.positions(values)
                try:
                    eager[values] = pool.run(query, pages, causal=True, cur_pos=positions, label=label + ' eager')
                finally:
                    ttnn.deallocate(positions)
            return eager[values]

        for index, values in enumerate(plan):
            label = 'T/seed%d/replay%d/%s' % (seed, index, ','.join(map(str, values)))
            got = replay(values, label)
            reference = eager_at(values, label)
            record(report, comparison('T', 'trace_vs_eager', label, card.differing(torch, got, reference), True,
                                      positions=list(values), seed=seed))
            wanted = range(batch) if args.trace_references == 'all' else (
                (0,) if args.trace_references == 'slot0' else ())
            for slot_index in wanted:
                extent = split_model.extent(values[slot_index])
                refscope = Scope(ttnn)
                try:
                    repeated = refscope.keep(pool.upload(host_query[:, slot_index:slot_index + 1]
                                                         .repeat(1, batch, 1, 1).contiguous()))
                    ref_pages = refscope.keep(pool.pages_reference(users[slot_index], extent, batch))
                    wide = refscope.keep(pool.upload(position_mask(torch, [values[slot_index]] * batch,
                                                                   [extent] * batch, rows, extent)))
                    legacy = pool.run(repeated, ref_pages, causal=False, mask=wide, label=label + ' legacy')
                finally:
                    refscope.close()
                record(report, comparison('T', 'trace_vs_legacy', '%s/entry%d/E%d' % (label, slot_index, extent),
                                          card.differing(torch, slot(got, slot_index), slot(legacy, 0)), True,
                                          extent=extent, position=values[slot_index], seed=seed), verbose=False)
        # Skips inside the same trace: the live entries equal the eager all-live call.
        last = plan[-1]
        full = eager_at(last, 'T/skip base')
        for pattern in SKIP_PATTERNS:
            if max(pattern) >= batch:
                continue
            values = skip_positions(last, pattern)
            label = 'T/seed%d/skip%s' % (seed, ''.join(map(str, pattern)))
            got = replay(values, label)
            live = [index for index in range(batch) if index not in pattern]
            differing = sum(card.differing(torch, slot(got, index), slot(full, index)) for index in live)
            record(report, comparison('T', 'trace_skip_live', label, differing, True, pattern=list(pattern),
                                      live=live, seed=seed))
    finally:
        if trace is not None:
            with WATCHDOG.op('release trace'):
                ttnn.release_trace(device, trace)
        scope.close()


def section_skip(ttnn, torch, pool, seed, args, report):
    """K: UINT32_MAX skips, eager, with the output poisoned first."""
    rows, batch = DECISIVE_ROWS, args.batch
    extents = [SKIP_EXTENTS[index % len(SKIP_EXTENTS)] for index in range(batch)]
    extents = [min(extent, pool.capacity) for extent in extents]
    positions = tuple(extent - 1 - 17 * index for index, extent in enumerate(extents))
    users = list(range(batch))
    scope = Scope(ttnn)
    written = report.setdefault('skip_rows', [])
    try:
        host_query = rows_query(torch, batch, rows, seed, 'normal', salt=800)
        query = scope.keep(pool.upload(host_query))
        pages = scope.keep(pool.pages_causal(users, extents))
        live_positions = scope.keep(pool.positions(positions))
        full = pool.run(query, pages, causal=True, cur_pos=live_positions, label='K all live')
        shape = (1, batch, rows * 12, card.HEAD_DIM)
        for pattern in SKIP_PATTERNS:
            if max(pattern) >= batch:
                continue
            label = 'K/seed%d/skip%s' % (seed, ''.join(map(str, pattern)))
            values = scope.keep(pool.positions(skip_positions(positions, pattern)))
            address = pool.poison_output(shape)
            out = pool.launch(query, pages, causal=True, cur_pos=values, label=label)
            reused = None if address is None else k1.buffer_address(out) == address
            got = pool.host(out, 'read back ' + label)
            live = [index for index in range(batch) if index not in pattern]
            differing = sum(card.differing(torch, slot(got, index), slot(full, index)) for index in live)
            record(report, comparison('K', 'skip_live', label, differing, True, pattern=list(pattern), live=live,
                                      seed=seed, poison_address_reused=reused))
            for index in pattern:
                rows_nan = int(torch.isnan(slot(got, index).float()).all(dim=-1).sum())
                state = 'unwritten' if rows_nan == rows * 12 else ('written' if rows_nan == 0 else 'partial')
                if reused is not True:
                    state += '-unpoisoned'
                written.append(dict(label='%s/entry%d' % (label, index), nan_rows=rows_nan, rows=rows * 12,
                                    state=state))
        # One call whose only user is skipped: every core returns at once.
        single_pages = scope.keep(pool.pages_causal([0], [extents[0]]))
        single_query = scope.keep(pool.upload(host_query[:, :1].contiguous()))
        idle = scope.keep(pool.positions([-1]))
        out = pool.launch(single_query, single_pages, causal=True, cur_pos=idle, label='K/idle B=1')
        with WATCHDOG.op('K idle synchronize'):
            ttnn.synchronize_device(pool.device)
        ttnn.deallocate(out)
        report['skip_idle_call'] = 'returned'
    finally:
        scope.close()
    states = sorted({entry['state'] for entry in written})
    report['skip_written'] = ','.join(states) if states else None


def section_dynamic(ttnn, torch, pool, seed, args, report):
    """D: the native dynamic chunk (k_chunk_size 0) against the fixed 256, same positions."""
    pairs = position_pairs(args.extents, args.starts[:1])
    group = group_calls(pairs, args.batch)[0]
    extents = [extent for extent, _start, _p in group]
    positions = [p for _extent, _start, p in group]
    for rows in args.rows:
        scope = Scope(ttnn)
        try:
            query = scope.keep(pool.upload(rows_query(torch, args.batch, rows, seed, 'normal', salt=900)))
            pages = scope.keep(pool.pages_causal(list(range(args.batch)), extents))
            cur_pos = scope.keep(pool.positions(positions))
            fixed = pool.run(query, pages, causal=True, cur_pos=cur_pos, label='D fixed rows%d' % rows)
            dynamic = pool.run(query, pages, causal=True, cur_pos=cur_pos, k_chunk=0, label='D dynamic rows%d' % rows)
            label = 'D/rows%d/seed%d/%s' % (rows, seed, ','.join(map(str, positions)))
            record(report, comparison('D', 'dynamic_vs_fixed', label, card.differing(torch, dynamic, fixed), False,
                                      rows=rows, positions=positions))
        finally:
            scope.close()


def section_flags(ttnn, torch, pool, args, report):
    """N: 0x20 is unknown to the loaded binary; a qwen sentinel on a causal call is refused."""
    stage = report['binary']['stage']
    if stage < 1:
        report['flag_0x20'] = 'n/a(stock)'
        report['refusals'] = {}
        return
    rows, batch = E_SHAPES['G4B3']
    extent = min(args.extents)
    scope = Scope(ttnn)
    results = {}
    try:
        query = scope.keep(pool.upload(card.build_query(torch, batch, 0, 'normal', rows=rows)))
        pages = scope.keep(pool.pages_reference(0, extent, batch))
        wide = scope.keep(pool.upload(position_mask(torch, [extent - 1] * batch, [extent] * batch, rows, extent,
                                                    zero=True)))
        causal_pages = scope.keep(pool.pages_causal([0] * batch, [extent] * batch))
        cur_pos = scope.keep(pool.positions([extent - 1] * batch))
        cases = (('flag 0x20 (K64j, proposed)', dict(query=query, pages=pages, causal=False, mask=wide,
                                                      sentinel=MAGIC | TAIL | K64J_FLAG), UNKNOWN_NEEDLE),
                 ('qwen sentinel on a causal call', dict(query=query, pages=causal_pages, causal=True, cur_pos=cur_pos,
                                                         sentinel=MAGIC | TAIL), CAUSAL_NEEDLE))
        for name, call, needle in cases:
            try:
                out = pool.launch(call['query'], call['pages'], causal=call['causal'], cur_pos=call.get('cur_pos'),
                                  mask=call.get('mask'), sentinel=call['sentinel'], label='N ' + name,
                                  expect_program=False)
            except Exception as error:  # noqa: BLE001 - a TT_FATAL surfaces as RuntimeError
                text = ' '.join(str(error).split())
                results[name] = dict(refused=True, matched=needle in text, message=text[:300])
            else:
                ttnn.deallocate(out)
                results[name] = dict(refused=False, matched=False, message='accepted')
            print('refusal %-34s %s' % (name, results[name]), flush=True)
    finally:
        scope.close()
    report['refusals'] = results
    flag = results['flag 0x20 (K64j, proposed)']
    report['flag_0x20'] = 'unknown' if flag['refused'] and flag['matched'] else (
        'accepted' if not flag['refused'] else 'refused-otherwise')
    if report['flag_0x20'] == 'accepted':
        report['warnings'].append('the loaded binary ACCEPTS flag 0x20: K64j needs another bit')
    causal = results['qwen sentinel on a causal call']
    if not (causal['refused'] and causal['matched']):
        report['warnings'].append('a qwen sentinel on a causal call was not refused as non-causal: %s'
                                  % causal['message'])


def eager_median(ttnn, device, once, args, label):
    samples = []
    with WATCHDOG.span(label, args.watchdog * max(1, args.iters)):
        for _ in range(args.warmup):
            ttnn.deallocate(once())
        ttnn.synchronize_device(device)
        for _ in range(args.iters):
            started = clock()
            out = once()
            ttnn.synchronize_device(device)
            samples.append((clock() - started) * 1e6)
            ttnn.deallocate(out)
    return samples


def section_timing(ttnn, torch, device, pool, args, report):
    """Recorded: the runtime extent's cost against the compile-time calls, and the skip saving."""
    rows, batch = DECISIVE_ROWS, args.batch
    scope = Scope(ttnn)
    rows_out = []
    try:
        query = scope.keep(pool.upload(rows_query(torch, batch, rows, 0, 'normal', salt=1200)))
        users = list(range(batch))
        shapes = []
        for extent in args.extents:
            pages = scope.keep(pool.pages_causal(users, [extent] * batch))
            last = scope.keep(pool.positions([extent - 1] * batch))
            ref_pages = scope.keep(pool.pages_reference(0, extent, batch))
            zero = scope.keep(pool.upload(position_mask(torch, [extent - 1] * batch, [extent] * batch, rows, extent,
                                                        zero=True)))
            shapes.append(('runtime E%d' % extent, extent, (lambda p=pages, c=last: pool.launch(
                query, p, causal=True, cur_pos=c, label='timing'))))
            shapes.append(('compile E%d' % extent, extent, (lambda p=ref_pages, m=zero: pool.launch(
                query, p, causal=False, mask=m, label='timing'))))
        full_pages = scope.keep(pool.pages_causal(users, [pool.capacity] * batch, poison=False))
        top = tuple([pool.capacity - 1] * batch)
        for pattern in ((), (0,), (0, 1), tuple(range(batch))):
            values = scope.keep(pool.positions(skip_positions(top, pattern)))
            shapes.append(('skip%d' % len(pattern), pool.capacity, (lambda c=values: pool.launch(
                query, full_pages, causal=True, cur_pos=c, label='timing'))))
        medians = {name: [] for name, _extent, _once in shapes}
        samples = {name: [] for name, _extent, _once in shapes}
        for index in range(args.rounds):
            turn = index % len(shapes)
            for name, _extent, once in shapes[turn:] + shapes[:turn]:
                got = eager_median(ttnn, device, once, args, 'timing %s round %d' % (name, index))
                medians[name].append(statistics.median(got))
                samples[name].extend(got)
        for name, extent, _once in shapes:
            rows_out.append(dict(name=name, extent=extent, eager=k1.summary(samples[name]),
                                 round_medians=medians[name]))
            print('timing %-16s %.1f us' % (name, statistics.median(samples[name])), flush=True)
    finally:
        scope.close()
    by = {row['name']: row['eager']['median_us'] for row in rows_out}
    ratios = {}
    for extent in args.extents:
        runtime, compile_time = by.get('runtime E%d' % extent), by.get('compile E%d' % extent)
        if runtime and compile_time:
            ratios['E%d' % extent] = runtime / compile_time
    report['timing'] = dict(rows=rows_out, runtime_over_compile=ratios,
                            skip_us={name: by[name] for name in by if name.startswith('skip')})


def check_binary(args, report):
    path, markers = card.loaded_binary()
    sliced = slice_marker(path)
    stage = binary_stage(markers, sliced)
    sha = k1.file_sha256(path)
    report['binary'] = dict(path=path, sha256=sha, markers=markers, slice=sliced, stage=stage,
                            expected_sha256=args.expect_binary_sha256 or None)
    print('binary %s sha256 %s stage=%d markers=%s slice=%s' % (path, sha[:16], stage, markers, sliced), flush=True)
    if args.expect_binary_sha256 and sha != args.expect_binary_sha256:
        report['failures'].append('the loaded _ttnncpp.so is %s, not the expected %s (read the launched argv)'
                                  % (sha[:16], args.expect_binary_sha256[:16]))
        return False
    return True


def check_kernels(root, report):
    found = {}
    ok = True
    for name, expected in STOCK_KERNELS.items():
        path = Path(root) / name
        found[name] = k1.file_sha256(path) if path.is_file() else None
        if found[name] != expected:
            ok = False
            report['failures'].append('%s is %s, not %s: the stock decode kernels are not the ones this probe reads'
                                      % (path, (found[name] or 'missing')[:16], expected[:16]))
    recorded = {}
    for name in RECORDED_KERNELS:
        path = Path(root) / name
        recorded[name] = k1.file_sha256(path) if path.is_file() else None
    report['kernels'] = dict(root=str(root), stock=found, qwen=recorded)
    return ok


def run(args, report):
    import torch
    import ttnn

    failures = report['failures']
    options = dict(device_id=args.device_id, l1_small_size=24576)
    if 'T' in args.sections:
        options['trace_region_size'] = args.trace_region_bytes
    with WATCHDOG.op('open device', extra=OPEN_EXTRA_S):
        device = ttnn.open_device(**options)
    try:
        try:
            device.enable_program_cache()
            report['program_cache_enabled_call'] = True
        except Exception as error:  # noqa: BLE001 - default-on in newer runtimes
            report['program_cache_enabled_call'] = repr(error)[:200]
        if not check_binary(args, report):
            return
        if args.kernel_root and not check_kernels(args.kernel_root, report):
            return
        if os.environ.get(card.SCRATCH_ENV) != '1':
            failures.append('%s=1 is required: the arm sets it, and the G8 legacy calls do not fit L1 without it'
                            % card.SCRATCH_ENV)
            return
        users = max(args.batch, max(batch for _rows, batch in E_SHAPES.values()))
        for seed in args.seeds:
            pool = Pool(ttnn, torch, device, args.capacity, seed, users, report)
            try:
                if 'S' in args.sections:
                    section_split(ttnn, torch, pool, seed, args, report)
                if 'E' in args.sections:
                    section_extent(ttnn, torch, pool, seed, args, report)
                if 'T' in args.sections and seed == args.seeds[0]:
                    section_trace(ttnn, torch, device, pool, seed, args, report)
                if 'K' in args.sections:
                    section_skip(ttnn, torch, pool, seed, args, report)
                if 'D' in args.sections and seed == args.seeds[0]:
                    section_dynamic(ttnn, torch, pool, seed, args, report)
                if 'N' in args.sections and seed == args.seeds[0]:
                    section_flags(ttnn, torch, pool, args, report)
                if not args.no_timing and seed == args.seeds[0]:
                    section_timing(ttnn, torch, device, pool, args, report)
            finally:
                pool.close()
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--capacity', type=int, default=CAPACITY, help='the C-wide page table (keys)')
    parser.add_argument('--extents', default=','.join(map(str, EXTENTS)))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)), help='p = E - 256 + s, 0..255')
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--rows', default=','.join(map(str, ROWS)), help='query tokens per entry in S and D (1, 2)')
    parser.add_argument('--batch', type=int, default=BATCH, help='entries per call in S, T and K (1-4)')
    parser.add_argument('--shapes', default=','.join(E_SHAPES), help='section E shapes: %s' % ', '.join(E_SHAPES))
    parser.add_argument('--sections', default=','.join(SECTIONS), help='any of %s' % ', '.join(SECTIONS))
    parser.add_argument('--trace-families', type=int, default=TRACE_FAMILIES)
    parser.add_argument('--trace-references', choices=('all', 'slot0', 'none'), default='slot0',
                        help='compile-time references per replay: entry 0 (default: one new program per family, '
                             '--trace-families of them), every entry (about 3 x as many JIT builds), or none')
    parser.add_argument('--trace-region-bytes', type=int, default=16 << 20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--watchdog', type=float, default=0, help='seconds per device call before os._exit(3); 0 off')
    parser.add_argument('--expect-binary-sha256', default='', help='the mapped _ttnncpp.so must be this (K64i: %s)'
                        % K64I_TTNNCPP_SHA256[:16])
    parser.add_argument('--kernel-root', default=KERNEL_ROOT, help='the mounted sdpa_decode kernels ("" skips)')
    args = parser.parse_args(argv)

    def ints(text):
        return [int(value) for value in text.split(',') if value.strip()]

    try:
        args.extents = ints(args.extents)
        args.starts = ints(args.starts)
        args.seeds = ints(args.seeds)
        args.rows = ints(args.rows)
    except ValueError as error:
        parser.error(str(error))
    args.variants = [value for value in args.variants.split(',') if value]
    args.shapes = [value for value in args.shapes.split(',') if value]
    args.sections = [value for value in args.sections.split(',') if value]
    try:
        check_capacity(args.capacity)
        for extent in args.extents:
            check_capacity(extent, 'extent')
    except ValueError as error:
        parser.error(str(error))
    if not args.extents or max(args.extents) > args.capacity or len(set(args.extents)) != len(args.extents):
        parser.error('--extents must be distinct and at most --capacity')
    if not args.starts or any(not 0 <= start < split_model.K_CHUNK for start in args.starts):
        parser.error('--starts must be 0..255')
    if not args.seeds or not args.variants or any(v not in ('normal', 'peaky', 'zeroq') for v in args.variants):
        parser.error('--seeds and --variants (normal, peaky, zeroq) must be non-empty')
    if not args.rows or any(rows not in (1, 2) for rows in args.rows):
        parser.error('--rows takes 1 and/or 2')
    if not 1 <= args.batch <= 4:
        parser.error('--batch must be 1..4')
    if any(name not in E_SHAPES for name in args.shapes) or any(name not in SECTIONS for name in args.sections):
        parser.error('unknown shape or section')
    if args.trace_families < 1 or min(args.iters, args.rounds) < 1 or args.warmup < 0:
        parser.error('--trace-families, --iters and --rounds must be >= 1')
    if args.expect_binary_sha256 and (len(args.expect_binary_sha256) != 64
                                      or any(c not in '0123456789abcdef' for c in args.expect_binary_sha256)):
        parser.error('--expect-binary-sha256 must be a full lowercase sha256')
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(probe=PROBE, plan=PLAN, passed=False, argv=list(sys.argv[1:] if argv is None else argv),
                  capacity=args.capacity, extents=args.extents, starts=args.starts, seeds=args.seeds,
                  variants=args.variants, rows=args.rows, batch=args.batch, shapes=args.shapes, sections=args.sections,
                  poison=dict(k=POISON_K, v=POISON_V, blocks=POISON_BLOCKS), k64j_flag='0x%x' % K64J_FLAG,
                  predictions=split_predictions(args.extents, args.capacity,
                                                [args.batch] + [batch for _rows, batch in E_SHAPES.values()]),
                  max_dynamic_chunk_size_default=split_model.max_dynamic_chunk_size(False),
                  env={name: os.environ.get(name) for name in ENV_RECORDED}, watchdog=args.watchdog,
                  failures=[], warnings=[], comparisons=[], liveness=[])
    report['_requested'] = set()
    native = card.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        payload['requested_programs'] = sorted(list(key) for key in report.get('_requested', ()))
        payload['tally'] = tally(report.get('comparisons', []))
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def on_fire(label):
        try:
            write_report(dict(error='watchdog: %r exceeded its budget' % (label,), passed=False))
        except Exception:  # noqa: BLE001 - the main thread may be mid-update; the WATCHDOG line stands
            pass

    WATCHDOG = k1.Watchdog(args.watchdog, on_fire=on_fire).start()
    card.WATCHDOG = WATCHDOG
    k1.WATCHDOG = WATCHDOG
    try:
        try:
            with native:
                run(args, report)
            if report.get('binary', {}).get('stage', 0) >= 1:
                report['factory_lines'] = card.factory_lines(native.text())
                for key in missing_programs(report['factory_lines'], report['_requested']):
                    report['failures'].append('factory log: no [QWEN-SDPA] line for flags=0x%x B=%d St=%d '
                                              'mask_width_t=%d (graft mounted, not executed)' % key)
        except Exception as error:  # noqa: BLE001
            report['error'] = '%s: %s' % (type(error).__name__, error)
        report['decision'] = decide(report)
        report['verdict_line'] = verdict_line(report)
        report['passed'] = report['decision']['verdict'] == 'GO'
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    for warning in report['warnings']:
        print('WARN', warning)
    if report.get('error'):
        print('ERROR', report['error'])
    print(report.get('verdict_line', PROBE + ' verdict=NO-DECISION'), flush=True)
    print('SDPA_K64J_P0 passed=%s comparisons=%d liveness=%d failures=%d warnings=%d report=%s native_log=%s' % (
        report['passed'], len(report['comparisons']), len(report['liveness']), len(report['failures']),
        len(report['warnings']), args.out, native.path), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
