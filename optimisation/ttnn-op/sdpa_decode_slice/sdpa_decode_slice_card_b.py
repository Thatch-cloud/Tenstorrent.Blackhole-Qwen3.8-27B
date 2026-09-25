"""K64i card-B qualification: the head-sliced decode SDPA (flag 0x4, K1a) and the share leader's read-ahead
(flag 0x8, K1b) against the SERVED tail+share call (0x3), byte for byte (k1-sdpa-head-slice-design.md section 7).

WHAT IT PROVES. With ~/opgraft-K64i mounted as the arm mounts it, every paged_scaled_dot_product_attention_decode
call whose q_chunk_size sentinel carries 0x4 and/or 0x8 returns the SAME BYTES, on every kept row of both KV
heads, as the same call without them, as the served 0x3 call and as legacy (q_chunk_size 0); the reference role
(K64g, the served graft) records the served outputs' sha256 so the candidate proves its 0x0-0x3 and legacy
outputs are K64g's bytes too. Then it times 0x3 against 0x7 (K1a), 0xB (K1b alone) and 0xF (both).

Inputs are the stage-3 card test's (../sdpa_decode_qwen/test_sdpa_decode_qwen_card_m.py, whose host helpers this
imports): the bf8 paged pool, ONE page table repeated to every bundle row, the refresh-formula mask, Q folded
KV-head-major (row = kv*G + t*6 + j, G = rows*6). Capacities 2,304 / 33,024 / 66,048 / 131,328 keys (1 / 9 /
17 / 33 chunks on each head's busiest core); seeds 0-4; variants normal, peaky, zeroq; starts +0, +7, +240.

Shapes (rows per group x 12 folded rows per entry, B entries; 2 KV heads per chip):
  G8B2   the served call (8-row groups, T16 = one batch-2 bundle): legacy, 0x0-0x7, 0xA, 0xB, 0xF
  G8B3   T32 bundles (B=3, twin bands 9 of 10 rows): the same modes
  G8B1   T8 segments (B=1: 0x2 and 0x8 inert, distinct programs all the same): the same modes
  G7B2   7-row groups (84 rows: head 1 starts at row 42, mid-tile): legacy, 0x3, 0x7, 0xB, 0xF
  G16B1  E-S1b: 16-row groups (192 rows, 6 -> 3 tiles). Its unsliced form needs 1,804,352 B of CBs and does not
         fit L1, so only 0x4 and 0x5 run, compared per token against G8B2 legacy over the same 16 tokens.

Checks (candidate; each output compared with torch.equal on int16 views, per KV head's rows):
  E-S1   every flag set == its non-slice twin (flags & 0x3) == legacy, on both heads' rows, for G8 at B = 2, 3
         and 1 and 7-row groups; 0xB builds with the stock writer (the review's F16/F17 fix) and == 0x3
  E-S1b  G16B1 0x4 / 0x5 unfolded per token vs G8B2 legacy unfolded: expected exact, RECORDED (a difference is a
         warning, not a failure: G16 is not served; the design records it as the stage-3 test recorded E4)
  N-S0   before EVERY qwen call a NaN (0x7FC0) tensor of the output's shape is uploaded and freed; the call's
         output must take that address (re-poisoned up to --poison-retries times) and hold no NaN: an output
         row the sliced writer skipped cannot pass on a stale correct row of an earlier call
  N-S1   a +320 spike planted in the final chunk (its first column, visible to every row) at row 40 (head 0,
         tile 1), 56 (head 1, tile 1) and 90 (head 1, tile 2), one entry at a time (b = 0, then 1): legacy moves
         exactly that row; 0x5, 0x7 and 0xF equal legacy on the planted mask (R8's row offset and its batch
         stride); a spike in chunk 0 (every row) moves legacy, not 0x7 / 0xF (tail reads the final chunk only)
  N-S2   distinct page-table rows: 0x7 / 0xF == legacy with equal rows (the leader's), every twin != legacy on
         its own row (G8B2 and G8B3)
  N-S3   refusals: 0x4 on 4-row groups (no saving), unknown flag 0x10, 0x8 without 0x2 (0x9, 0xD), 0x4 on a
         causal call, a mask whose rows are not Q's (validate's or F15's text)
  N-S4   program cache: --alternations pairs of 0x5 / 0x7 on distinct rows (they differ), 0x3 / 0x7 and 0x7 / 0xF
         on equal rows (identical): each output equals its mode's first
  N-S5   one trace (legacy, 0x3, 0x7, 0xF, 0xB on G8B2; 0x7, 0xF on G8B3; 0x5 on G8B1; distinct rows)
         replayed --trace-soak times, every replay == eager
  log    exactly the requested programs have '[QWEN-SDPA] flags=' lines (PNHt = the slice under 0x4,
         kv_share, scratch_slots=4, the modelled cb_bytes), and every stage-4 program its '[QWEN-SDPA] q-slice'
         line (rows_per_kv, pnht_full, slice_tiles, readahead): graft mounted is not graft executed
  timing eager (median of --iters, --rounds interleaved) and a traced --trace-calls replay, per call, of 0x3 /
         0x7 / 0xB / 0xF on G8B2 and G8B3 and 0x1 / 0x5 on G8B1 at --timing-capacities. Acceptance (section 7):
         the better of 0x7 / 0xF <= 0.78 x 0x3 at 131,328 keys (>= 180 us saved); > 0.90 x 0x3 is the stop
         rule. Recorded beside the S0.K1 probe's prediction; never a failure (the rig's load skews).

--role reference (on K64g) runs legacy and 0x0-0x3 only and records every output's sha256; --role candidate
--reference <that report> asserts its legacy and 0x0-0x3 outputs are those bytes (a report that is not a PASSING
reference-role report is refused at the arguments; without --reference the K1_CARD line ends reference=none, and
exact=PASS then means self-consistency only). Failures: the loaded binary
is not the role's build (stage 4 carries '[QWEN-SDPA] q-slice rows_per_kv='; K64g does not), a mounted kernel
is not the recorded one, no compact scratch (QWEN_SDPA_TREE_SCRATCH_ROUNDS=1, as the arm sets it), any check
above, the watchdog. Prints one 'K1_CARD ...' line and one 'SDPA_K1_CARD passed=...' line.

RUN with run_card_b.sh only (QUAL_CARD, default card B; the serving pair only with ALLOW_SERVING_CARD=1). The
helpers above WATCHDOG import no ttnn and are tested on CPU by test_sdpa_decode_slice_card_b.py.
"""

import argparse
import json
import mmap
import os
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / 'sdpa_decode_qwen'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import probe_k1_card_b as probe  # noqa: E402 - watchdog, poison, eager/traced timing, slopes
import slice_index_model as model  # noqa: E402 - the slice rule and the CB bytes
import test_sdpa_decode_qwen_card_m as card  # noqa: E402 - masks, queries, fold, Case, digest

CARD = 'K64i card-B qualification'
MAGIC, LEGACY = card.MAGIC, card.LEGACY
TAIL, SHARE, SLICE, READAHEAD = 0x1, 0x2, 0x4, 0x8
SERVED = TAIL | SHARE
CAPACITIES = (2304, 33024, 66048, 131328)
DECISION_CAPACITY = 131328
CALLS_PER_REPLAY = 64
ACCEPT, STOP = 0.78, 0.90                       # section 7: candidate acceptance and the stop rule
# name -> (rows per fold group, group offsets (the mask formula's), role)
SHAPES = {
    'G8B2': (8, (0, 8), 'served'),
    'G8B3': (8, (0, 8, 16), 'T32 bundles'),
    'G8B1': (8, (0,), 'T8 segments'),
    'G7B2': (7, (0, 7), '7-row groups'),
    'G16B1': (16, (0,), 'E-S1b, sliced only'),
}
# E-S1 (design section 7): on every G8 shape each of 0x4-0x7 against its twin 0x0-0x3 and legacy, and with K1b
# 0xA, 0xB and 0xF (at B=1, 0x2 and 0x8 are inert, but they are distinct programs and must still be exact).
G8_MODES = (0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0xA, 0xB, 0xF)
MODES = {
    'G8B2': G8_MODES,
    'G8B3': G8_MODES,
    'G8B1': G8_MODES,
    'G7B2': (0x3, 0x7, 0xB, 0xF),
    'G16B1': (0x4, 0x5),
}
NO_LEGACY = ('G16B1',)                          # 1,804,352 B unsliced: never called without 0x4
TIMING = {'G8B2': (0x3, 0x7, 0xB, 0xF), 'G8B3': (0x3, 0x7, 0xB, 0xF), 'G8B1': (0x1, 0x5)}
TIMING_BASE = {'G8B2': 0x3, 'G8B3': 0x3, 'G8B1': 0x1}
PLANT_ROWS = (40, 56, 90)                       # head 0 tile 1; head 1 tile 1; head 1 tile 2 (G8)
# The plant: +320 (+20 after the op's 1/16 scale) at one visible key. A single -inf removes ~1e-6 of a row's
# softmax weight at 131,328 keys, which bf16 does not show, so that control would be dead; a spike makes that
# key dominate the row, so legacy moves exactly the planted row at every capacity.
PLANT_VALUE = 320.0
PLANT_FLAGS = (0x5, 0x7, 0xF)
TWIN_FLAGS = (0x7, 0xF)
TRACE_OPS = (('G8B2', None), ('G8B2', 0x3), ('G8B2', 0x7), ('G8B2', 0xF), ('G8B2', 0xB), ('G8B3', 0x7),
             ('G8B3', 0xF), ('G8B1', 0x5))
ALTERNATIONS = ((0x5, 0x7, True), (0x3, 0x7, False), (0x7, 0xF, False))   # (a, b, on distinct rows: must differ)
# N-S3: (name, shape rows, batches, flags, needles (any), causal, mask kind)
REFUSALS = (
    ('q-slice on 4-row groups (B=3)', 4, 3, 0x5, ('[QWEN-SDPA] q-slice saves no tile',), False, 'wide'),
    ('q-slice on 4-row groups (B=2, share)', 4, 2, 0x7, ('[QWEN-SDPA] q-slice saves no tile',), False, 'wide'),
    ('unknown flag 0x10', 8, 2, 0x11, ('[QWEN-SDPA] unknown flags',), False, 'wide'),
    ('read-ahead without share (0x9)', 8, 2, 0x9, ('[QWEN-SDPA] KV read-ahead needs KV share',), False, 'wide'),
    ('read-ahead without share (0xD)', 8, 2, 0xD, ('[QWEN-SDPA] KV read-ahead needs KV share',), False, 'wide'),
    ('q-slice on a causal call', 8, 2, 0x5, ('modes are non-causal',), True, None),
    ('a mask with fewer rows than Q', 8, 2, 0x5, ("[QWEN-SDPA] q-slice needs a mask with Q's",
                                                  'Expect same number of padded heads in mask as in Q'), False, 'short'),
)
SLICE_BINARY_MARKER = b'[QWEN-SDPA] q-slice rows_per_kv='
KERNELS = {
    'dataflow/reader_decode_qwen.cpp': '280a847fae833891dffff1057d67b999288a183386cd058e3e1614755ce3499b',
    'compute/sdpa_flash_decode_qwen.cpp': '8776fcc7420c6f27a9c7ae06c54c391225a00ce78322c5397970d74a5063ca8a',
}
SLICE_KERNELS = {
    'dataflow/reader_decode_qwen_slice.cpp': '0f5a019ccc06ca603bb4ed44c77cc66f9cb3e5bd35193810eb127f13c9f5631f',
    'dataflow/writer_decode_qwen_slice.cpp': 'ac6cf815c34df85a9d39593d95f28eb2da0cb5b37c232633e65bf3ed116925f4',
}
K64G_TTNNCPP_SHA256 = probe.K64G_TTNNCPP_SHA256
# The S0.K1 probe on card B (2026-09-24, K64g): the G4 B2 tail+share proxy of K1a, the compute reference and
# the served call, eager, at 131,328 keys; saving per call = served - proxy.
PROBE_CARD_B = dict(date='2026-09-24', report='~/kwork64/k1probe/card-b/probe-20260924T120742.json', verdict='GO',
                    scenario='B', enable='0x4,0x8', proxy_us=696.3, compute_us=563.8, served_us=796.5,
                    proxy_over_served=0.874, saving_us=100.2, slopes_us_per_chunk=dict(served=20.27, proxy=18.24,
                                                                                         compute=14.64))
ENV_RECORDED = (card.SCRATCH_ENV, 'TT_METAL_WATCHER', 'TT_METAL_CACHE', 'TT_METAL_HOME')


# ---------------------------------------------------------------------------------------------
# Pure helpers (no ttnn).
# ---------------------------------------------------------------------------------------------

def sentinel(flags):
    return LEGACY if flags is None else MAGIC | flags


def flag_name(flags):
    return 'legacy' if flags is None else '0x%x' % flags


def twin_of(flags):
    """The same call without the stage-4 flags: 0x4 -> 0x0, 0x7 -> 0x3, 0xB -> 0x3, 0xF -> 0x3."""
    return flags & (TAIL | SHARE)


def folded_rows(shape):
    return SHAPES[shape][0] * 12


def batches_of(shape):
    return len(SHAPES[shape][1])


def head_rows(shape, head):
    """The folded rows of KV head `head` (the kept rows of that head's cores): [head*G, (head+1)*G)."""
    rows_per_kv = folded_rows(shape) // card.KV_HEADS
    return head * rows_per_kv, (head + 1) * rows_per_kv


def reference_modes(shape):
    """What the reference (K64g, stage 3) can run: legacy and the flags without 0x4 / 0x8."""
    if shape in NO_LEGACY:
        return ()
    return (None,) + tuple(flags for flags in MODES[shape] if not flags & (SLICE | READAHEAD))


def program_key(capacity, batches, query_rows, mask_width, flags):
    """The factory's F4 line for one call: (capacity, B, PNHt as built, mask_width_t, flags)."""
    return (capacity, batches, model.program_pnht(query_rows, flags), mask_width // card.TILE, flags)


def slice_key(query_rows, batches, flags):
    """The F18 line one call's program prints, or None: (rows_per_kv, pnht_full, slice_tiles, readahead)."""
    readahead = bool(flags & READAHEAD) and bool(flags & SHARE) and batches > 1
    if not (flags & SLICE or readahead):
        return None
    rule = model.slice_rule(query_rows)
    return (rule['rows_per_kv'], rule['pnht_full'], model.program_pnht(query_rows, flags), readahead)


def describe(key):
    return 'cap=%d B=%d PNHt=%d mask_width_t=%d flags=0x%x' % key


def describe_slice(key):
    return 'rows_per_kv=%d pnht_full=%d slice_tiles=%d readahead=%s' % (key[0], key[1], key[2], str(key[3]).lower())


SLICE_LINE = re.compile(r'\[QWEN-SDPA\] q-slice rows_per_kv=([0-9]+) pnht_full=([0-9]+) slice_tiles=([0-9]+) '
                             r'readahead=(true|false)')


def slice_lines(text):
    return [(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) == 'true') for m in SLICE_LINE.finditer(text)]


def check_logs(text, requested, requested_slices):
    """Every requested program has its F4 line with the modelled fields and no unrequested one appears; every
    stage-4 program its F18 line and no other F18 line appears. Returns the problems."""
    problems = []
    lines = card.factory_lines(text)
    found = {}
    for line in lines:
        found.setdefault(card.line_key(line), []).append(line)
    for key in sorted(requested):
        capacity, batches, pnht, _width, flags = key
        if key not in found:
            problems.append('%s: no [QWEN-SDPA] flags= line' % describe(key))
            continue
        expected = dict(kv_share='true' if flags & SHARE and batches > 1 else 'false',
                        scratch_slots=card.SCRATCH_SLOTS, cb_bytes=model.cb_bytes(capacity, pnht))
        for line in found[key]:
            wrong = {name: (line[name], value) for name, value in expected.items() if line[name] != value}
            if wrong:
                problems.append('%s: (found, expected) %r' % (describe(key), wrong))
                break
    stray = sorted(set(found) - set(requested))
    if stray:
        problems.append('%d [QWEN-SDPA] programs nobody requested: %s' % (len(stray), '; '.join(map(describe, stray))))
    seen = set(slice_lines(text))
    for key in sorted(requested_slices):
        if key not in seen:
            problems.append('%s: no [QWEN-SDPA] q-slice line (a stage-4 program was not built)' % describe_slice(key))
    stray_slices = sorted(seen - set(requested_slices))
    if stray_slices:
        problems.append('[QWEN-SDPA] q-slice lines nobody requested: %s' % '; '.join(map(describe_slice, stray_slices)))
    return problems


def acceptance(ratio):
    if ratio is None:
        return 'none'
    if ratio <= ACCEPT:
        return 'ACCEPT'
    if ratio > STOP:
        return 'STOP'
    return 'BETWEEN'


def analyse_timing(rows, capacity=None):
    """Per shape, mode and basis: the per-call medians, each over its shape's base (0x3; 0x1 at B=1), and the
    per-chunk slopes; then section 7's verdict on G8B2 at `capacity` (the better of 0x7 and 0xF against 0x3)."""
    capacity = DECISION_CAPACITY if capacity is None else capacity
    times = {}
    for row in rows:
        for basis in ('eager', 'trace'):
            stats = row.get(basis)
            if stats and stats.get('median_us') is not None:
                times.setdefault(basis, {}).setdefault(row['shape'], {}).setdefault(row['flags'], {})[row['capacity']] = \
                    stats['median_us']
    out = dict(ratios={}, slopes={}, verdict={})
    for basis, shapes in times.items():
        for shape, modes in shapes.items():
            base = modes.get('0x%x' % TIMING_BASE[shape], {})
            for flags, per_capacity in modes.items():
                out['slopes'].setdefault(basis, {}).setdefault(shape, {})[flags] = probe.slopes(per_capacity)
                out['ratios'].setdefault(basis, {}).setdefault(shape, {})[flags] = {
                    cap: us / base[cap] for cap, us in per_capacity.items() if base.get(cap)}
        g8 = shapes.get('G8B2', {})
        served = g8.get('0x3', {}).get(capacity)
        verdict = dict(capacity=capacity, served_us=served)
        for name, flags in (('k1a', '0x7'), ('k1b', '0xb'), ('k1ab', '0xf')):
            us = g8.get(flags, {}).get(capacity)
            verdict[name + '_us'] = us
            verdict[name + '_ratio'] = us / served if us is not None and served else None
        candidates = [(verdict[name + '_us'], name) for name in ('k1a', 'k1ab') if verdict[name + '_us'] is not None]
        if served and candidates:
            best_us, best = min(candidates)
            verdict.update(best=best, best_us=best_us, best_ratio=best_us / served, saving_us=served - best_us,
                           saving_ms_per_replay=(served - best_us) * CALLS_PER_REPLAY / 1000.0,
                           acceptance=acceptance(best_us / served),
                           probe_k1a_ratio=PROBE_CARD_B['proxy_over_served'],
                           served_over_probe=served / PROBE_CARD_B['served_us'])
        else:
            verdict['acceptance'] = 'none'
        out['verdict'][basis] = verdict
    return out


def verdict_line(report, basis='eager'):
    words = ['K1_CARD', 'role=%s' % report.get('role'), 'exact=%s' % ('PASS' if report.get('passed') else 'FAIL'),
             'cases=%d' % len(report.get('cases', [])), 'failures=%d' % len(report.get('failures', []))]
    verdict = report.get('timing_analysis', {}).get('verdict', {}).get(basis)
    if verdict and verdict.get('served_us'):
        words += ['timing=%s' % verdict['acceptance'], 'basis=%s' % basis, 'cap=%d' % verdict['capacity'],
                  'served_0x3=%.1fus' % verdict['served_us']]
        for name, flags in (('k1a', '0x7'), ('k1b', '0xB'), ('k1ab', '0xF')):
            if verdict.get(name + '_us') is not None:
                words.append('%s_%s=%.1fus(r=%.3f)' % (name, flags, verdict[name + '_us'], verdict[name + '_ratio']))
        if verdict.get('best'):
            words += ['saving=%.1fus/call' % verdict['saving_us'], 'replay=%.1fms' % -verdict['saving_ms_per_replay'],
                      'rules=accept<=%.2f,stop>%.2f' % (ACCEPT, STOP)]
        words.append('probe_k1a=%.1fus(r=%.3f)' % (PROBE_CARD_B['proxy_us'], PROBE_CARD_B['proxy_over_served']))
    else:
        words.append('timing=none')
    if report.get('role') == 'candidate':
        # Whether exact=PASS includes "legacy and 0x0-0x3 are K64g's bytes" (design section 8, card B), or only
        # the self-consistency of this binary.
        words.append('reference=%s' % (Path(report['reference']).name if report.get('reference') else 'none'))
    return ' '.join(words)


def file_sha256(path):
    return probe.file_sha256(path)


def binary_has_slice(path):
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return view.find(SLICE_BINARY_MARKER) >= 0


# ---------------------------------------------------------------------------------------------
# Device harness (the qualification card only).
# ---------------------------------------------------------------------------------------------

WATCHDOG = probe.Watchdog(0)


class Runner:
    """One device session: the Case of the current capacity and seed, the requested programs, the poison."""

    def __init__(self, ttnn, torch, device, args, report):
        self.ttnn, self.torch, self.device, self.args, self.report = ttnn, torch, device, args, report
        self.requested, self.requested_slices = set(), set()
        self.case = None

    def record(self, query, mask, flags, batches):
        if flags is None:
            return
        width = self.case.capacity if mask is None else mask.shape[3]
        try:
            self.requested.add(program_key(self.case.capacity, batches, query.shape[2], width, flags))
        except ValueError:
            return                                      # a refusal: the factory builds nothing
        key = slice_key(query.shape[2], batches, flags)
        if key is not None:
            self.requested_slices.add(key)

    def call(self, query, mask, flags, *, pages=None, record=True, **options):
        batches = query.shape[1]
        if record:
            self.record(query, mask, flags, batches)
        return self.case.call(query, mask, sentinel(flags), pages=pages, record=False, **options)

    def poisoned(self, query, mask, flags, *, pages=None):
        """N-S0: poison the output's address, call, check the output took it; up to --poison-retries tries.
        Returns (host output, reused: True/False/None when unknowable, tries)."""
        shape = (1, query.shape[1], query.shape[2], card.HEAD_DIM)
        tries = max(1, self.args.poison_retries)
        for attempt in range(tries):
            address = probe.poison(self.ttnn, self.torch, self.case, shape)
            out = self.call(query, mask, flags, pages=pages)
            reused = None if address is None else probe.buffer_address(out) == address
            if reused is not False or attempt == tries - 1:
                return self.case.host(out), reused, attempt + 1
            self.ttnn.deallocate(out)
        raise AssertionError('unreachable')


def output_check(torch, label, flags, out, reused, tries, report):
    """No NaN (a row nobody wrote shows the poison) and the poisoned address taken (else a stale row could hide)."""
    nan = int(torch.isnan(out.float()).sum())
    entry = dict(nan_elements=nan, poison_address_reused=reused, poison_tries=tries)
    if nan:
        report['failures'].append('%s/%s: %d output elements are NaN - rows the kernels left unwritten (N-S0)'
                                  % (label, flag_name(flags), nan))
    if reused is False:
        text = '%s/%s: the output never took the poisoned address (%d tries): an unwritten row could hide' % (
            label, flag_name(flags), tries)
        (report['failures'] if flags & SLICE else report['warnings']).append(text)
    return entry


def per_head_differing(torch, shape, left, right):
    return [card.differing(torch, left[:, :, lo:hi], right[:, :, lo:hi])
            for lo, hi in (head_rows(shape, head) for head in range(card.KV_HEADS))]


def shape_inputs(torch, case, shape, tokens, seed, variant, begin):
    """(query, mask) hosts for `shape`: G8B2's query is drawn; G8B1 and G16B1 fold its tokens (same tokens);
    G8B3 and G7B2 are their own draws."""
    rows, offsets, _role = SHAPES[shape]
    if shape == 'G8B2':
        query = card.build_query(torch, 2, seed, variant, case.keys, case.table, rows=8)
    elif shape in ('G8B1', 'G16B1'):
        query = card.fold_entries(torch, tokens, (0,), rows)
    else:
        query = card.build_query(torch, len(offsets), seed, variant, case.keys, case.table, rows=rows)
    return query, card.build_mask(torch, case.capacity, begin, offsets, rows=rows)


def record_sha(torch, label, out, args, report):
    sha = card.digest(torch, out)
    report['shas'][label] = sha
    if args.reference_shas is not None:
        expected = args.reference_shas.get(label)
        if expected is None:
            report['failures'].append('%s: not in the reference report (rerun the reference with this harness)' % label)
        elif expected != sha:
            report['failures'].append('%s: differs from the reference (K64g) bytes' % label)
    return sha


def equality_case(runner, capacity, seed, variant, start):
    """E-S1 and E-S1b for one capacity x seed x variant x start; every tensor it uploads is freed."""
    ttnn, torch, case, args, report = runner.ttnn, runner.torch, runner.case, runner.args, runner.report
    base = 'cap%d/seed%d/%s/start+%d' % (capacity, seed, variant, start)
    begin = capacity - 256 + start
    q8 = card.build_query(torch, 2, seed, variant, case.keys, case.table, rows=8)
    tokens = card.unfold_entries(torch, q8, 8)
    entry = dict(label=base, shapes={})
    legacy_g8 = None
    local = []
    try:
        for shape in args.shapes:
            query_host, mask_host = shape_inputs(torch, case, shape, tokens, seed, variant, begin)
            query, mask = case.upload(query_host, keep=False), case.upload(mask_host, keep=False)
            local += [query, mask]
            label = '%s/%s' % (shape, base)
            batches = batches_of(shape)
            outputs = {}
            result = dict(modes={})
            if shape not in NO_LEGACY:
                legacy = case.host(runner.call(query, mask, None))
                outputs[None] = legacy
                result['legacy_sha256'] = record_sha(torch, '%s/legacy' % label, legacy, args, report)
                if not bool(torch.isfinite(legacy.float()).all()):
                    report['failures'].append('%s: the legacy output is not finite (the inputs are broken)' % label)
                if shape == 'G8B2':
                    legacy_g8 = legacy
            modes = reference_modes(shape)[1:] if args.role == 'reference' else MODES[shape]
            for flags in modes:
                out, reused, tries = runner.poisoned(query, mask, flags)
                outputs[flags] = out
                mode = output_check(torch, label, flags, out, reused, tries, report)
                if not flags & (SLICE | READAHEAD):
                    mode['sha256'] = record_sha(torch, '%s/%s' % (label, flag_name(flags)), out, args, report)
                result['modes'][flag_name(flags)] = mode
            for flags in modes:
                mode, out = result['modes'][flag_name(flags)], outputs[flags]
                if None in outputs:
                    mode['legacy_differing_per_head'] = per_head_differing(torch, shape, out, outputs[None])
                    if any(mode['legacy_differing_per_head']):
                        report['failures'].append('%s: %s differs from legacy (per head %r)' % (
                            label, flag_name(flags), mode['legacy_differing_per_head']))
                twin = twin_of(flags)
                if twin != flags and twin in outputs:
                    mode['twin'] = flag_name(twin)
                    mode['twin_differing_per_head'] = per_head_differing(torch, shape, out, outputs[twin])
                    if any(mode['twin_differing_per_head']):
                        report['failures'].append('%s: %s differs from its non-slice twin %s (per head %r)' % (
                            label, flag_name(flags), flag_name(twin), mode['twin_differing_per_head']))
            if shape in NO_LEGACY and legacy_g8 is not None:
                reference_tokens = card.unfold_entries(torch, legacy_g8, 8)
                for flags in modes:
                    moved = card.differing(torch, card.unfold_entries(torch, outputs[flags], SHAPES[shape][0]),
                                           reference_tokens)
                    result['modes'][flag_name(flags)]['g8b2_legacy_per_token_differing'] = moved
                    report['e_s1b'].append(dict(label=label, flags=flag_name(flags), differing=moved))
                    if moved:
                        report['warnings'].append('%s: %s unfolded differs from G8B2 legacy per token in %d elements '
                                                  '(E-S1b, recorded)' % (label, flag_name(flags), moved))
            elif shape in NO_LEGACY and args.role != 'reference':
                report['warnings'].append('%s: E-S1b needs G8B2 in --shapes (its per-token reference)' % label)
            entry['shapes'][shape] = result
    finally:
        for tensor in local:
            ttnn.deallocate(tensor)
    report['cases'].append(entry)
    print('%-40s shapes=%s failures=%d' % (base, ','.join(entry['shapes']), len(report['failures'])), flush=True)


def planted_mask(torch, capacity, begin, offsets, rows, entry=None, row=None, column=None):
    """The refresh-formula mask with PLANT_VALUE at (entry, row, column): column defaults to the final chunk's
    first (visible to every row); entry / row None plant every entry / row."""
    mask = card.build_mask(torch, capacity, begin, offsets, rows=rows).float()
    column = capacity - 256 if column is None else column
    mask[slice(None) if entry is None else entry, 0, slice(None) if row is None else row, column] = PLANT_VALUE
    return mask.to(torch.bfloat16)


def controls(runner, capacity):
    """N-S1, N-S2, N-S3, N-S4 and N-S5 at one capacity (the first seed)."""
    ttnn, torch, case, args, report = runner.ttnn, runner.torch, runner.case, runner.args, runner.report
    failures = report['failures']
    begin = capacity - 256 + args.starts[0]
    seed = args.seeds[0]
    result = dict(capacity=capacity, planted=[], twins={}, refusals={}, alternations=[], trace=None)
    local = []

    def upload(host, dtype=None):
        tensor = case.upload(host, dtype, keep=False)
        local.append(tensor)
        return tensor

    try:
        rows, offsets, _ = SHAPES['G8B2']
        q8h = card.build_query(torch, 2, seed, 'normal', rows=rows)
        q8, m8 = upload(q8h), upload(card.build_mask(torch, capacity, begin, offsets, rows=rows))
        legacy = case.host(runner.call(q8, m8, None))
        # N-S1: one planted row, one entry at a time.
        for entry in range(2):
            for row in PLANT_ROWS:
                planted = upload(planted_mask(torch, capacity, begin, offsets, rows, entry, row))
                legacy_planted = case.host(runner.call(q8, planted, None))
                diff = (card.int16_view(torch, legacy_planted) != card.int16_view(torch, legacy)).any(dim=-1)[0]
                moved = sorted((int(b), int(r)) for b, r in diff.nonzero().tolist())
                item = dict(entry=entry, row=row, legacy_moved_rows=moved, flags={})
                if moved != [(entry, row)]:
                    failures.append('cap%d: the spike planted at entry %d row %d moved legacy rows %r (expected exactly '
                                    'that row: the control is dead or not row-local)' % (capacity, entry, row, moved[:6]))
                for flags in PLANT_FLAGS:
                    out, reused, tries = runner.poisoned(q8, planted, flags)
                    output_check(torch, 'cap%d/planted' % capacity, flags, out, reused, tries, report)
                    per_head = per_head_differing(torch, 'G8B2', out, legacy_planted)
                    item['flags'][flag_name(flags)] = per_head
                    if any(per_head):
                        failures.append('cap%d: %s on the mask planted at entry %d row %d differs from legacy (per head '
                                        '%r): the slice read the wrong mask rows (R8)' % (capacity, flag_name(flags),
                                                                                           entry, row, per_head))
                result['planted'].append(item)
        chunk0 = upload(planted_mask(torch, capacity, begin, offsets, rows, column=0))
        legacy_chunk0 = case.host(runner.call(q8, chunk0, None))
        if not card.differing(torch, legacy_chunk0, legacy):
            failures.append('cap%d: the chunk-0 plant did not move legacy (control dead)' % capacity)
        for flags in TWIN_FLAGS:
            plain = case.host(runner.call(q8, m8, flags))
            moved = card.differing(torch, case.host(runner.call(q8, chunk0, flags)), plain)
            result.setdefault('chunk0', {})[flag_name(flags)] = moved
            if moved:
                failures.append('cap%d: %s read a non-final mask chunk (%d elements moved by a chunk-0 plant)'
                                % (capacity, flag_name(flags), moved))
        # N-S2: twins read no K/V of their own.
        for shape in ('G8B2', 'G8B3'):
            rows_s, offsets_s, _ = SHAPES[shape]
            batches = len(offsets_s)
            query = q8 if shape == 'G8B2' else upload(card.build_query(torch, batches, seed, 'normal', rows=rows_s))
            mask = m8 if shape == 'G8B2' else upload(card.build_mask(torch, capacity, begin, offsets_s, rows=rows_s))
            distinct = case.pages(batches, distinct=True)
            leader_rows = legacy if shape == 'G8B2' else case.host(runner.call(query, mask, None))
            legacy_distinct = case.host(runner.call(query, mask, None, pages=distinct))
            for flags in TWIN_FLAGS:
                out, reused, tries = runner.poisoned(query, mask, flags, pages=distinct)
                output_check(torch, 'cap%d/%s/distinct' % (capacity, shape), flags, out, reused, tries, report)
                same = card.differing(torch, out, leader_rows)
                twins = [card.differing(torch, out[:, b:b + 1], legacy_distinct[:, b:b + 1]) > 0 for b in range(1, batches)]
                result['twins']['%s/%s' % (shape, flag_name(flags))] = dict(differing_from_leader_row=same,
                                                                            twins_differ=twins)
                if same:
                    failures.append('cap%d/%s: %s with distinct page-table rows is not legacy on the leader row (%d)'
                                    % (capacity, shape, flag_name(flags), same))
                if not all(twins):
                    failures.append('cap%d/%s: %s twin output equals legacy on its own row - the twin read its own K/V '
                                    'or the control is dead (%r)' % (capacity, shape, flag_name(flags), twins))
        # N-S3: refusals (the factory's TT_FATALs fire before any program is built).
        for name, rows_r, batches, flags, needles, causal, mask_kind in REFUSALS:
            offsets_r = tuple(rows_r * index for index in range(batches))
            query = upload(card.build_query(torch, batches, 0, 'normal', rows=rows_r))
            if mask_kind == 'wide':
                mask = upload(card.build_mask(torch, capacity, begin, offsets_r, rows=rows_r))
            elif mask_kind == 'short':
                mask = upload(torch.zeros(batches, 1, rows_r * 12 - 32, capacity, dtype=torch.bfloat16))
            else:
                mask = None
            options = dict(pages=case.pages(batches))
            if causal:
                positions = upload(torch.full((batches,), capacity - 1, dtype=torch.int32), ttnn.int32)
                options.update(is_causal=True, cur_pos_tensor=positions)
            try:
                out = runner.call(query, mask, flags, record=False, **options)
            except Exception as error:  # noqa: BLE001 - TT_FATAL surfaces as RuntimeError
                text = ' '.join(str(error).split())
                outcome = dict(refused=True, matched=any(needle in text for needle in needles), message=text[:300])
            else:
                ttnn.deallocate(out)
                outcome = dict(refused=False, matched=False, message='call was accepted')
            result['refusals'][name] = outcome
            print('refusal cap=%d %-40s %s' % (capacity, name, outcome), flush=True)
            if not (outcome['refused'] and outcome['matched']):
                failures.append('cap%d: %s was not refused with %r: %s' % (capacity, name, needles, outcome['message']))
        # N-S4: the program cache.
        distinct2 = case.pages(2, distinct=True)
        for first_flags, second_flags, on_distinct in ALTERNATIONS:
            pages = distinct2 if on_distinct else None
            first, drift = {}, 0
            for index in range(2 * args.alternations):
                flags = (first_flags, second_flags)[index % 2]
                out = case.host(runner.call(q8, m8, flags, pages=pages))
                if flags not in first:
                    first[flags] = out
                elif card.differing(torch, first[flags], out):
                    drift += 1
            if not first:
                continue
            modes_differ = card.differing(torch, first[first_flags], first[second_flags]) > 0
            item = dict(modes='%s/%s' % (flag_name(first_flags), flag_name(second_flags)), distinct_rows=on_distinct,
                        calls=2 * args.alternations, drifted=drift, modes_differ=modes_differ)
            result['alternations'].append(item)
            print('alternation cap=%d %s' % (capacity, item), flush=True)
            if drift:
                failures.append('cap%d: %d alternating %s calls drifted from their mode\'s first output' % (
                    capacity, drift, item['modes']))
            if modes_differ != on_distinct:
                failures.append('cap%d: %s outputs %s on %s rows (expected %s)' % (
                    capacity, item['modes'], 'differ' if modes_differ else 'agree',
                    'distinct' if on_distinct else 'equal', 'differ' if on_distinct else 'agree'))
        # N-S5: one trace of every stage-4 program beside the served ones, replayed.
        if args.trace_soak:
            inputs = {}
            for shape in sorted({shape for shape, _flags in TRACE_OPS}):
                rows_s, offsets_s, _ = SHAPES[shape]
                batches = len(offsets_s)
                query = q8 if shape == 'G8B2' else upload(card.build_query(torch, batches, seed, 'normal', rows=rows_s))
                mask = m8 if shape == 'G8B2' else upload(card.build_mask(torch, capacity, begin, offsets_s, rows=rows_s))
                inputs[shape] = (query, mask, case.pages(batches, distinct=True) if batches > 1 else None)

            def launch():
                return [runner.call(inputs[shape][0], inputs[shape][1], flags, pages=inputs[shape][2])
                        for shape, flags in TRACE_OPS]

            eager = [case.host(tensor) for tensor in launch()]
            distinguishable = card.differing(torch, eager[0], eager[2]) > 0     # legacy vs 0x7 on distinct rows
            with WATCHDOG.op('capture cap=%d' % capacity):
                trace, outputs = card.capture(ttnn, runner.device, launch)
            mismatches = []
            try:
                for replay in range(args.trace_soak):
                    with WATCHDOG.op('trace replay %d cap=%d' % (replay, capacity)):
                        ttnn.execute_trace(runner.device, trace, cq_id=0, blocking=True)
                    for (shape, flags), output, expected in zip(TRACE_OPS, outputs, eager):
                        with WATCHDOG.op('trace read back %s %s' % (shape, flag_name(flags))):
                            got = ttnn.to_torch(output)
                        if card.differing(torch, got, expected):
                            mismatches.append((replay, shape, flag_name(flags)))
            finally:
                with WATCHDOG.op('release trace cap=%d' % capacity):
                    ttnn.release_trace(runner.device, trace)
                for output in outputs:
                    ttnn.deallocate(output)
            result['trace'] = dict(replays=args.trace_soak, ops=['%s/%s' % (s, flag_name(f)) for s, f in TRACE_OPS],
                                   mismatch_count=len(mismatches), mismatches=mismatches[:20],
                                   legacy_vs_0x7_distinguishable=distinguishable)
            print('trace cap=%d replays=%d mismatches=%d distinguishable=%s' % (
                capacity, args.trace_soak, len(mismatches), distinguishable), flush=True)
            if mismatches:
                failures.append('cap%d: %d trace replay outputs differ from eager (first %r)' % (
                    capacity, len(mismatches), mismatches[0]))
            if not distinguishable:
                failures.append('cap%d: legacy and 0x7 agree on distinct page-table rows in the trace (control dead)'
                                % capacity)
    finally:
        for tensor in local:
            ttnn.deallocate(tensor)
    report['controls'].append(result)


def timing(runner, capacity):
    ttnn, torch, case, args, report = runner.ttnn, runner.torch, runner.case, runner.args, runner.report
    begin = capacity - 256 + args.starts[0]
    local, runs = [], []
    try:
        for shape, modes in TIMING.items():
            if shape not in args.shapes:
                continue
            rows, offsets, _ = SHAPES[shape]
            query = case.upload(card.build_query(torch, len(offsets), args.seeds[0], 'normal', rows=rows), keep=False)
            mask = case.upload(card.build_mask(torch, capacity, begin, offsets, rows=rows), keep=False)
            local += [query, mask]
            runs += [(shape, flags, query, mask) for flags in modes]
        samples = {index: [] for index in range(len(runs))}
        rounds = {index: [] for index in range(len(runs))}
        for turn in range(args.rounds):
            order = list(range(len(runs)))
            order = order[turn % len(order):] + order[:turn % len(order)]      # no mode always runs first
            for index in order:
                shape, flags, query, mask = runs[index]
                got = probe.eager_samples(ttnn, runner.device, lambda q=query, m=mask, f=flags: runner.call(q, m, f),
                                          args, 'timing cap=%d %s %s round %d' % (capacity, shape, flag_name(flags), turn))
                samples[index].extend(got)
                rounds[index].append(statistics.median(got))
        for index, (shape, flags, query, mask) in enumerate(runs):
            row = dict(capacity=capacity, shape=shape, flags=flag_name(flags), chunks=probe.busiest_chunks(capacity),
                       eager=probe.summary(samples[index]), eager_round_medians=rounds[index], trace=None)
            if args.replays:
                runner.record(query, mask, flags, query.shape[1])
                row['trace'], _sha = probe.traced(ttnn, torch, runner.device, case, query, mask, sentinel(flags), args,
                                                  'cap=%d %s %s' % (capacity, shape, flag_name(flags)))
            report['timing'].append(row)
            print('timing cap=%d %-5s %-4s eager %.1f us%s' % (
                capacity, shape, flag_name(flags), row['eager']['median_us'],
                '' if row['trace'] is None else ' trace %.1f us/call' % row['trace']['median_us']), flush=True)
    finally:
        for tensor in local:
            ttnn.deallocate(tensor)


def check_binary(args, report):
    path, markers = card.loaded_binary()
    stage = card.binary_stage(markers)
    sha = file_sha256(path)
    has_slice = binary_has_slice(path)
    report['binary'] = dict(path=path, sha256=sha, markers=markers, stage=stage, slice=has_slice,
                            expected_sha256=args.expect_binary_sha256 or None)
    print('binary %s sha256 %s stage=%d slice=%s' % (path, sha[:16], stage, has_slice), flush=True)
    if stage != 3:
        report['failures'].append('the loaded _ttnncpp.so is stage %d, not a KV-share build: the graft is not mounted' % stage)
        return False
    if args.role == 'candidate' and not has_slice:
        report['failures'].append("the loaded _ttnncpp.so lacks %r: it is not K64i (read the launched argv)"
                                  % SLICE_BINARY_MARKER.decode())
        return False
    if args.role == 'reference' and has_slice:
        report['failures'].append('the reference role needs the served K64g, but the loaded binary is a stage-4 build')
        return False
    if args.expect_binary_sha256 and sha != args.expect_binary_sha256:
        report['failures'].append('the loaded _ttnncpp.so is %s, not the expected %s' % (sha[:16], args.expect_binary_sha256[:16]))
        return False
    return True


def check_kernels(root, role, report):
    expected = dict(KERNELS, **(SLICE_KERNELS if role == 'candidate' else {}))
    found = {}
    for name, sha in expected.items():
        path = Path(root) / name
        found[name] = file_sha256(path) if path.is_file() else None
        if found[name] != sha:
            report['failures'].append('%s is %s, not the recorded %s' % (path, (found[name] or 'missing')[:16], sha[:16]))
    report['kernels'] = dict(root=str(root), sha256=found)
    return all(found[name] == sha for name, sha in expected.items())


def run(args, report):
    import torch
    import ttnn

    options = dict(device_id=args.device_id, l1_small_size=24576)
    if args.role == 'candidate' and ((args.trace_soak and args.controls) or (not args.no_timing and args.replays)):
        options['trace_region_size'] = args.trace_region_bytes
    with WATCHDOG.op('open device', extra=probe.OPEN_EXTRA_S):
        device = ttnn.open_device(**options)
    try:
        try:
            device.enable_program_cache()
            report['program_cache_enabled_call'] = True
        except Exception as error:  # noqa: BLE001 - default-on in newer runtimes
            report['program_cache_enabled_call'] = repr(error)[:200]
        if not check_binary(args, report):
            return
        if args.kernel_root and not check_kernels(args.kernel_root, args.role, report):
            return
        if os.environ.get(card.SCRATCH_ENV) != '1':
            report['failures'].append('%s=1 is required: the arm sets it, and the G8 legacy calls do not fit L1 without it'
                                      % card.SCRATCH_ENV)
            return
        runner = Runner(ttnn, torch, device, args, report)
        report['_runner'] = runner
        for capacity in args.capacities:
            for seed in args.seeds:
                runner.case = card.Case(ttnn, torch, device, capacity, seed)
                try:
                    for variant in args.variants:
                        for start in args.starts:
                            equality_case(runner, capacity, seed, variant, start)
                    if args.role == 'candidate' and seed == args.seeds[0]:
                        if args.controls:
                            controls(runner, capacity)
                        if not args.no_timing and capacity in args.timing_capacities:
                            timing(runner, capacity)
                finally:
                    runner.case.close()
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--role', choices=('reference', 'candidate'), default='candidate')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--capacities', default=','.join(map(str, CAPACITIES)))
    parser.add_argument('--seeds', default=','.join(map(str, card.SEEDS)))
    parser.add_argument('--variants', default=','.join(card.VARIANTS))
    parser.add_argument('--starts', default=','.join(map(str, card.STARTS)))
    parser.add_argument('--shapes', default=','.join(SHAPES), help='from %s' % ', '.join(SHAPES))
    parser.add_argument('--no-controls', dest='controls', action='store_false', help='skip N-S1..N-S5')
    parser.add_argument('--alternations', type=int, default=500, help='pairs per N-S4 alternation (1,000 calls)')
    parser.add_argument('--trace-soak', type=int, default=200, help='N-S5 replays per capacity (0: none)')
    parser.add_argument('--poison-retries', type=int, default=3)
    parser.add_argument('--timing-capacities', default=None, help='default: every --capacities entry')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--trace-calls', type=int, default=16)
    parser.add_argument('--trace-warmup', type=int, default=2)
    parser.add_argument('--replays', type=int, default=10, help='timed trace replays per mode (0: eager only)')
    parser.add_argument('--trace-region-bytes', type=int, default=32 << 20)
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--decision-capacity', type=int, default=DECISION_CAPACITY,
                        help='the capacity the section-7 acceptance is read at (131,328 keys)')
    parser.add_argument('--watchdog', type=float, default=0, help='seconds per device call before os._exit(3); 0 off')
    parser.add_argument('--reference', help='a reference-role report whose legacy / 0x0-0x3 shas the candidate reproduces')
    parser.add_argument('--expect-binary-sha256', default='')
    parser.add_argument('--kernel-root', default=probe.KERNEL_ROOT, help='the mounted sdpa_decode kernels ("" skips)')
    args = parser.parse_args(argv)
    args.capacities = [int(value) for value in args.capacities.split(',') if value]
    args.seeds = [int(value) for value in args.seeds.split(',') if value]
    args.variants = [value for value in args.variants.split(',') if value]
    args.starts = [int(value) for value in args.starts.split(',') if value]
    args.shapes = [value for value in args.shapes.split(',') if value]
    args.timing_capacities = (list(args.capacities) if args.timing_capacities is None
                              else [int(value) for value in args.timing_capacities.split(',') if value])
    if any(name not in SHAPES for name in args.shapes) or not args.shapes or len(set(args.shapes)) != len(args.shapes):
        parser.error('unknown or repeated --shapes %r (known: %s)' % (args.shapes, ', '.join(SHAPES)))
    for capacity in args.capacities + args.timing_capacities:
        try:
            card.num_blocks(capacity)
        except ValueError as error:
            parser.error(str(error))
    if any(capacity < 768 for capacity in args.capacities):
        parser.error('capacities below 768 leave no room for the tail and a planted chunk 0')
    if any(variant not in card.VARIANTS for variant in args.variants) or any(not 0 <= s <= 240 for s in args.starts):
        parser.error('unknown variant or start outside the family')
    if not (args.seeds and args.variants and args.starts and args.capacities):
        parser.error('empty sweep')
    if min(args.iters, args.rounds, args.trace_calls) < 1 or min(args.warmup, args.trace_warmup, args.replays,
                                                                 args.alternations, args.trace_soak) < 0:
        parser.error('--iters, --rounds and --trace-calls must be >= 1; the rest >= 0')
    if args.expect_binary_sha256 and (len(args.expect_binary_sha256) != 64
                                      or any(c not in '0123456789abcdef' for c in args.expect_binary_sha256)):
        parser.error('--expect-binary-sha256 must be a full lowercase sha256')
    if args.role == 'reference' and args.reference:
        parser.error('--reference is the candidate\'s input')
    args.reference_shas, args.reference_binary = None, None
    if args.reference:
        # Only a PASSING reference-role report is the served bytes: a failed one (a watchdog, a legacy that is not
        # finite, a 0x0-0x3 that differs from legacy on K64g) or a candidate's report would pass wrong bytes on.
        payload = json.loads(Path(args.reference).read_text())
        shas = payload.get('shas') or {}
        if payload.get('role') != 'reference' or payload.get('passed') is not True or not shas:
            parser.error("--reference %s is not a passing reference-role report (role=%r, passed=%r, %d shas): rerun "
                         "'run_card_b.sh reference'" % (args.reference, payload.get('role'), payload.get('passed'),
                                                        len(shas)))
        args.reference_shas = shas
        args.reference_binary = (payload.get('binary') or {}).get('sha256')
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(card=CARD, design='k1-sdpa-head-slice-design.md section 7', role=args.role, passed=False,
                  argv=list(sys.argv[1:] if argv is None else argv), capacities=args.capacities, seeds=args.seeds,
                  variants=args.variants, starts=args.starts, shapes=args.shapes, reference=args.reference,
                  env={name: os.environ.get(name) for name in ENV_RECORDED}, watchdog=args.watchdog,
                  thresholds=dict(accept=ACCEPT, stop=STOP, calls_per_replay=CALLS_PER_REPLAY),
                  probe_card_b=PROBE_CARD_B, failures=[], warnings=[], cases=[], controls=[], timing=[], shas={},
                  e_s1b=[], reference_binary_sha256=args.reference_binary)
    if args.role == 'candidate' and args.reference is None:
        report['warnings'].append('no --reference: legacy and 0x0-0x3 were NOT compared with the served K64g bytes '
                                  '(the K1_CARD line says reference=none); run run_card_b.sh reference first')
    native = card.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        runner = report.get('_runner')
        if runner is not None:
            payload['requested_programs'] = sorted(describe(key) for key in runner.requested)
            payload['requested_slice_programs'] = sorted(describe_slice(key) for key in runner.requested_slices)
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def on_fire(label):
        try:
            write_report(dict(error='watchdog: %r exceeded its budget' % (label,), passed=False))
        except Exception:  # noqa: BLE001 - the main thread may be mid-update; the WATCHDOG line stands
            pass

    WATCHDOG = probe.Watchdog(args.watchdog, on_fire=on_fire).start()
    card.WATCHDOG = probe.WATCHDOG = WATCHDOG            # Case's uploads, calls, read-backs; the timing helpers
    try:
        try:
            with native:
                run(args, report)
            runner = report.get('_runner')
            if runner is not None:
                text = native.text()
                report['factory_line_count'] = len(card.factory_lines(text))
                report['slice_line_count'] = len(slice_lines(text))
                for problem in check_logs(text, runner.requested, runner.requested_slices):
                    report['failures'].append('factory log: ' + problem)
            if report['timing']:
                report['timing_analysis'] = analyse_timing(report['timing'], args.decision_capacity)
        except Exception as error:  # noqa: BLE001
            report['error'] = '%s: %s' % (type(error).__name__, error)
        report['passed'] = not report['failures'] and not report.get('error') and bool(report['cases'])
        report['verdict_line'] = verdict_line(report)
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    for warning in report['warnings']:
        print('WARN', warning)
    if report.get('error'):
        print('ERROR', report['error'])
    print(report['verdict_line'], flush=True)
    print('SDPA_K1_CARD passed=%s role=%s cases=%d controls=%d timing_rows=%d failures=%d warnings=%d report=%s '
          'native_log=%s' % (report['passed'], args.role, len(report['cases']), len(report['controls']),
                             len(report['timing']), len(report['failures']), len(report['warnings']), args.out,
                             native.path), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
