"""K64j CB2b (s2-design.md W10b, section 6.2): the REAL S2 extent readers (scripts/ci/extent_attention_replay.py) on
one card, over lent storage built as serving_buffer_pool lends it, against the compile-time served call.

WHAT IT DECIDES. W1's ExtentSegmentReader and PackedExtentReplayReader serve a packed user at any position >= 128 from
one captured program: per segment a pool-lent full-width page table and cur_pos (E - 1) per bundle, a reader-owned
positions word (start & 255) and a reader-owned narrow (2, 1, 96, 256) mask that the PINNED mask kernel
(attention_mask_replay.cpp) writes in-trace at capacity 256. CB2a (k64j_card_b.py: K2, X7, Z) proved the kernel calls
on their own; this proves the reader code that issues them, through the classes themselves:

  R1  DECISIVE (design Q1: the pinned mask kernel has never run at capacity 256 on hardware). prepare_narrow and the
      pinned execute, eager and captured in a trace, at G8B2, G4B3 and G4B1 (--r1-geometries), for every word of
      --r1-words (s & 255): the device mask == extent_attention_replay.narrow_mask_host bit for bit, that mirror being
      checked on the host against k64j_card_b.served_mask at capacity 256 (an independent transliteration). The mask
      is poisoned (NaN) before every run, so an equal read proves the kernel rewrote every tile. And after every R2
      replay, each segment's own narrow masks, written in-trace by the reader's refresh, against the mirror at the
      start this harness staged (r1_reader).
  S   DECISIVE (design A6). Right after construction: each segment's word == [start & 255, 0 x 7], each bundle's
      cur_pos == [E - 1, E - 1] and its lent table == the host table repeated per entry, read back from the device and
      compared with values this harness computes itself (never the module's extent_values); the lent storage reads
      zero before construction. The same after every restage (word and cur_pos; the tables when the restage wrote
      them).
  R2  DECISIVE (design A2). One PackedExtentReplayReader of four T16 segments (the M3 block: LAYOUT(16, 8), one G8B2
      bundle each), built at the plan's first assignment and CALLED EAGERLY WITHOUT A RESTAGE (the state model_batch
      keeps when the capture start matches: r2_construction_vs_wide), then captured ONCE per seed (shared_masks(1) and
      the call) and replayed across --r2-families distinct 256-key families (the named ones first - 256, 2,304,
      16,640, 65,792, 131,328 - then evenly spread ones; 56 by default, so more than 50), the segments rotating
      through them at starts E - 256 + r for r in --r2-residues (a family-256 segment starts at 128 + r % 128, the
      live floor; the last family at most C - 16). Each assignment is restaged --r2-restages times: first per segment
      with its table (the lent table POISONED past E - K = 0, V = +16,384 - so a read past E moves every row), then
      word and cur_pos only (the block's PackedExtentReplayReader.stage). Every replay's segment == the 0x7
      compile-time call at capacity E with the table truncated to E / 64 and the WIDE mask carrying the real -inf tail
      (k64j_card_b.wide_mask), folded and unfolded on the host (r2_trace_vs_wide), and == the same reader called
      eagerly on the same staged state (r2_trace_vs_eager).
  R4  DECISIVE. At each variant's first assignment (the small families) and again at the first assignment that holds
      a segment at C (idle segments in one trace with a live segment at C, the served layout near 131k), idle
      segments (--idle-patterns: which segments go idle, the k-th of them at start 32 * (k % 2), page 0 tile row 0 or
      1) on the zero table at E = 256, in the same trace: the live segments == the all-live replay bit for bit
      (r4_live_unchanged), every idle row finite (r4_idle_finite), and each idle segment == the 0x7 call at capacity
      256 on the zero table with the wide mask at its start (r4_idle_vs_wide).
  liveness (per seed and variant, at the first assignment; a dead control is NO-DECISION): a segment below C with its
      lent cur_pos rewritten to [E, E] (one poisoned chunk more) must move its rows (cur_pos_live: the kernel reads
      the lent cur_pos), and its 0x7 reference with a ZERO wide mask must differ from the real one (mask_live: the
      masked keys matter, so an unapplied mask could not pass R2).

ONE CHIP, NOT TWO (a deviation from the design's "on both chips"). The readers and the pinned device_layout_dma and
prepare_narrow hard-require two chips ('Two chip-local metadata buffers required', 'Both chips required') and a p150a
is one chip. TwoChipView presents the one chip as a 1x2 mesh to the reader code only: get_device_tensors returns the
one shard twice, and a MeshProgramDescriptor the code builds for chips 0 and 1 launches chip 0's program (chip 1's,
built from the same shard addresses, is dropped and counted as phantom). It is sys.modules['ttnn'] only around the
calls into the reader classes, so the pinned modules' function-local `import ttnn` resolves to it; everything else
it forwards to ttnn unchanged. Nothing of the reader is replaced or patched but its log function (_pindiag), which is
teed into the report. Chip 1 of the serving pair is left to the model-level gates on M+A (G3, G3b, the extent audit).
WHAT THIS CANNOT SEE: chip 1's programs (prepare_narrow's mask program, device_layout_dma's folds) are built from chip
0's shard addresses and never launched, so a chip-indexing error in either - chip 1 given chip 0's metadata buffer,
word or mask, or chip 1's own ones never written - passes here. Chip 1 is covered only by W3's extent audit (the
two-chip readback of every segment's word and cur_pos and, in rotation, its narrow masks and tables) and by G3 / G3b
on M+A; the audit's chip-1 mask comparison is therefore decisive there (its MISMATCH line fails the arm, design 2.2
step 7), never advisory. The verdict line says chips=1of2.

THE CODE UNDER TEST comes from this checkout's scripts/ci, which run_card_b.sh mounts at /bench/ci (--ci-root): every
module must be loaded from there, the four pinned sources must keep their frozen bytes, and the sha256 of
extent_attention_replay.py (what W7's evidence pins), pooled_attention_replay.py and serving_buffer_pool.py are
recorded; the first is also on the verdict line (extent_sha256=).

FOR W7 (design W7: "the extent readers refuse construction unless admitted() holds"). This harness constructs the
readers with no admission - its PASS is part of the admission's evidence - and that evidence pins
extent_attention_replay.py's sha256 as this run recorded it. So the admitted() refusal belongs in the readers' caller
(model_batch, which builds the PackedExtentReplayReader, or packed_verifier), never in extent_attention_replay.py: a
refusal there changes the pinned bytes, and a rerun of this harness would then be refused by the refusal itself
(otherwise it needs a bypass for this harness, and CB2b rerun on the new bytes). W7's evidence takes the line
'K64J_READER verdict=PASS scope=full ... chips=1of2 ... extent_sha256=<the pinned sha>' - a reduced scope, another
sha or a missing chips=1of2 is not CB2b's evidence.

The log: the reader's 0x27 program (B = 2, C / 32, 8 mask tiles) and every 0x7 reference must log their F4 lines, the
0x27 one followed by exactly one F22 line (entries=2 kv_share=true q_slice=true); each construction must log one
'[PINDIAG] extent replay engaged segments=4 flags=0x27,0x27,0x27,0x27 mask=narrow capacity=C' line and, per segment,
one '[PINDIAG] sdpa qwen-modes ... flags=['0x27'] mask=narrow' line.

Verdict: one 'K64J_READER verdict=...' line.
  PASS         every decisive comparison byte-equal, every liveness control moved, no failure, nothing decisive cut by
               the deadline. scope=full when the run covered the design's set (every section; R1 at the three
               geometries and the residues 0, 7, 127, 240, 255; R2 over more than 50 families including the five
               named, at C = 131,328, for both variants; R4 at both idle starts; the block run of each of seeds 0, 1
               and 2 complete); scope=reduced otherwise (the WATCHER pass, a narrowed run, one seed). Only a full-scope
               PASS is CB2b's evidence.
  FAIL         a decisive comparison differs on an otherwise valid run.
  NO-DECISION  a failure (the wrong binary or kernels, no compact scratch, QWEN_FAST_SDPA_MODES other than
               tail,share,slice, a module not from --ci-root, a pinned source changed, the host mirrors disagreeing, a
               requested program without its factory lines, a missing PINDIAG line, a section that raised, the
               watchdog), a dead liveness control, a decisive section cut by the deadline, SIGTERM, or a requested
               section without a decisive comparison.

RUN with run_card_b.sh and K64J_HARNESS=extent_reader, WATCHER=1 first; on card M through the cardm action (card B is
reserved for another agent). The helpers above the device part import no ttnn and are tested on CPU by
test_extent_reader_card_b.py, which also runs the whole flow - the real reader classes included - on a fake one-chip
ttnn, and the broken variants each section must catch.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / 'k64j_probe', HERE.parent / 'sdpa_decode_qwen'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import k64j_card_b as card_b  # noqa: E402 - the pool, the served and wide calls, the factory log, the binary checks

probe, split_model = card_b.probe, card_b.split_model
card, k1 = probe.card, probe.k1

READER = 'K64J_READER'
PLAN = ('s2-design.md W10b / CB2b (section 6.2): R1, R2, R4 and the construction staging, through the real extent '
        'readers')
CAPACITY = card_b.CAPACITY                     # the served table: 2,052 pages of 64 keys
K_CHUNK = card_b.K_CHUNK
CI_ROOT = '/bench/ci'                          # where run_card_b.sh mounts this checkout's scripts/ci
SECTIONS = ('R1', 'S', 'R2', 'R4')
BLOCK_SECTIONS = ('S', 'R2', 'R4')             # one reader per seed serves these
DECISIVE_RUNS = ('R1', 'block')
R1_GEOMETRIES = {'G8B2': (8, 2, 0), 'G4B3': (4, 3, 0), 'G4B1': (4, 1, 0)}     # rows per group, groups, first offset
R1_WORDS = (0, 7, 32, 127, 128, 240, 255)
DESIGN_RESIDUES = (0, 7, 127, 240, 255)        # s mod 256 (design W10b R1, W10a)
SEGMENTS = ((0, 16), (16, 32), (32, 48), (48, 64))    # the M3 block: four T16 users
USERS, SEGMENT_ROWS, GROUP_ROWS = 4, 16, 8
SERVED_ROWS, SERVED_BATCH, SERVED_OFFSETS = card_b.SERVED_ROWS, card_b.SERVED_BATCH, card_b.SERVED_OFFSETS
SERVED_FLAGS, COMPILE_FLAGS = card_b.SERVED_FLAGS, card_b.COMPILE_FLAGS         # 0x27 (the reader), 0x7 (v235)
MIN_LIVE_START = card_b.MIN_LIVE_START
R2_NAMED = (256, 2304, 16640, 65792, 131328)
R2_FAMILIES = 56
R2_MIN_FAMILIES = 51                           # design W10b R2: "restaged across more than 50 families"
R2_RESTAGES = 2
IDLE_STARTS = card_b.IDLE_STARTS               # (0, 32): page 0, tile row 0 or 1
IDLE_FAMILY = K_CHUNK
IDLE_PATTERNS = '3,2+3,0,0+1'                  # segments that go idle, per replay (at most two: MAX_IDLE_SEGMENTS)
MAX_IDLE = 2
SEEDS = (0, 1, 2)
VARIANTS = ('normal', 'peaky')
SDPA_MODES_ENV = 'QWEN_FAST_SDPA_MODES'
SDPA_MODES = ('share', 'slice', 'tail')        # the image's QWEN_FAST_SDPA_MODES (v235); the reader adds extent
ENGAGED_MARKER = '[PINDIAG] extent replay engaged'
MODES_MARKER = '[PINDIAG] sdpa qwen-modes'
# The pinned sources the reader runs (target_t16_attention_gate.SOURCES, frozen at 8c102b20; test_extent_reader_card_b
# keeps these equal to test_extent_attention_replay.StructureTests.SOURCES).
PINNED = {
    'attention_mask_replay.py': '3e431742e35a2b94b4a02a60fa334a93a44a471eaefcacd25e52fbafdf03361f',
    'attention_mask_replay.cpp': 'e10cae1d6fe97f9b1509ac5ef918f6e7eda8d51bfbd77dcfd9e95662bb838af8',
    'attention_fold_dma.py': '5ce9d7d1590be2a9739a01d7604037f9fe70556594396067025bf1b3188151e5',
    'attention_fold_dma.cpp': '066fa6709127dcddbcdc033de9f0e0ad59a2c6756ceba3a99c5b0fd94cf26ab9',
}
RECORDED_SOURCES = ('extent_attention_replay.py', 'pooled_attention_replay.py', 'serving_buffer_pool.py',
                    'attention_head_fold.py', 'gdn_multitoken_conv.py')
MODULES = (('extent', 'extent_attention_replay'), ('mask', 'attention_mask_replay'), ('fold', 'attention_fold_dma'),
           ('head_fold', 'attention_head_fold'), ('pooled', 'pooled_attention_replay'),
           ('pool', 'serving_buffer_pool'), ('conv', 'gdn_multitoken_conv'))
SECTION_KINDS = {
    'R1': ('r1_eager', 'r1_trace', 'r1_reader'),
    'S': ('staging_word', 'staging_cur_pos', 'staging_table', 'restage_word', 'restage_cur_pos', 'restage_table'),
    'R2': ('r2_construction_vs_wide', 'r2_trace_vs_wide', 'r2_trace_vs_eager'),
    'R4': ('r4_live_unchanged', 'r4_idle_finite', 'r4_idle_vs_wide'),
}
DECISIVE_KINDS = tuple(kind for kinds in SECTION_KINDS.values() for kind in kinds)
DEADLINE_MARGIN_S = probe.DEADLINE_MARGIN_S
OPEN_EXTRA_S = probe.OPEN_EXTRA_S
ENV_RECORDED = probe.ENV_RECORDED + (SDPA_MODES_ENV,)


# ---------------------------------------------------------------------------------------------
# Pure helpers (no ttnn).
# ---------------------------------------------------------------------------------------------

def extent(start):
    """The 256-key family a row at `start` reads, computed here (never the module's extent_values)."""
    return split_model.extent(start)


def expected_word(start):
    """The positions word a segment at `start` must hold: [start & 255, 0 x 7] (the kernel runs at capacity 256)."""
    return [start & (K_CHUNK - 1)] + [0] * 7


def expected_cur_pos(start, entries=SERVED_BATCH):
    """Its bundle's cur_pos: E - 1 in every slot (K64j F20: one word per entry)."""
    return [extent(start) - 1] * entries


def live_start(family, residue, capacity, rows=SEGMENT_ROWS):
    """A live ticket's start in `family` at s mod 256 = residue: E - 256 + r, lifted to 128 + r % 128 below the live
    floor (only family 256) and held to capacity - rows in the last family (the ticket must fit the table)."""
    start = family - K_CHUNK + residue
    if start < MIN_LIVE_START:
        start = MIN_LIVE_START + residue % (K_CHUNK - MIN_LIVE_START)
    start = min(start, capacity - rows)
    if extent(start) != family or start < MIN_LIVE_START:
        raise ValueError('no live %d-row ticket in family %d at residue %d within %d keys' % (rows, family, residue,
                                                                                            capacity))
    return start


def family_plan(capacity, named, count):
    """The distinct families R2 replays, in order: the named ones that fit the table, then evenly spread multiples of
    256 in [512, capacity] up to `count` in all."""
    ordered = []
    for family in named:
        probe.check_capacity(family, 'family')
        if family <= capacity and family not in ordered:
            ordered.append(family)
    if count <= len(ordered):
        return ordered[:max(count, 1)]
    spare = [family for family in range(2 * K_CHUNK, capacity + 1, K_CHUNK) if family not in ordered]
    want = count - len(ordered)
    if want > len(spare):
        raise ValueError('%d families asked, only %d fit a %d-key table' % (count, len(ordered) + len(spare), capacity))
    if want == 1:
        picks = [spare[len(spare) // 2]]
    else:
        picks = [spare[round(index * (len(spare) - 1) / (want - 1))] for index in range(want)]
    if len(set(picks)) != len(picks):
        raise ValueError('the spread families collide')
    return ordered + picks


def replay_plan(families, residues, restages, capacity, users=USERS):
    """R2's replays in order. Assignment a gives segment j the family families[(a * users + j) % n] (every family once,
    the last assignment wrapping), restaged `restages` times: restage k at residue residues[(a + j + k) % len], the
    first with the tables (a new family per segment), the later ones word and cur_pos only."""
    if not families or not residues or restages not in (1, 2):
        raise ValueError('families, residues and one or two restages required')
    plan = []
    for assignment in range(-(-len(families) // users)):
        chosen = [families[(assignment * users + segment) % len(families)] for segment in range(users)]
        for restage in range(restages):
            starts = [live_start(family, residues[(assignment + segment + restage) % len(residues)], capacity)
                      for segment, family in enumerate(chosen)]
            plan.append(dict(assignment=assignment, restage=restage, families=chosen, starts=starts,
                             tables=restage == 0))
    return plan


def idle_entries(plan, capacity):
    """The indices of the plan entries R4 runs at: the first (the first assignment, the small families) and the first
    that holds a segment at C (idle segments in one trace with a live segment at C: the served layout near 131k)."""
    chosen = [0]
    at_capacity = next((index for index, entry in enumerate(plan) if capacity in entry['families']), None)
    if at_capacity is not None and at_capacity not in chosen:
        chosen.append(at_capacity)
    return chosen


def parse_patterns(text, users=USERS):
    """'3,2+3,0' -> [(3,), (2, 3), (0,)]: per replay the segments that go idle (one or two, never all), in segment
    order, the order packed_verifier.idle_inputs numbers them in."""
    patterns = []
    for part in [value for value in text.split(',') if value]:
        segments = tuple(sorted(int(value) for value in part.split('+')))
        if (not 1 <= len(segments) <= MAX_IDLE or len(set(segments)) != len(segments)
                or any(not 0 <= segment < users for segment in segments) or len(segments) >= users):
            raise ValueError('an idle pattern is one or two distinct segments of 0..%d, got %r' % (users - 1, part))
        patterns.append(segments)
    return patterns


def idle_assignment(pattern):
    """{segment: start}: the k-th idle segment in segment order at 32 * (k % 2), page 0 tile row k % 2
    (packed_verifier.idle_inputs, with E = 256's family start 0 in place of its C - 256, design 2.1)."""
    return {segment: IDLE_STARTS[index % 2] for index, segment in enumerate(sorted(pattern))}


def run_tag(seed, name):
    return '%s/seed%d' % (name, seed)


def section_runs(args):
    """R1 on the first seed only (the mask does not depend on the pool); the block (S, R2, R4) on every seed."""
    runs = []
    if 'R1' in args.sections:
        runs.append((args.seeds[0], 'R1'))
    if set(BLOCK_SECTIONS) & set(args.sections):
        runs += [(seed, 'block') for seed in args.seeds]
    return runs


def elements_differing(torch, got, expected):
    """Differing integer elements (int32 words, tables)."""
    got, expected = torch.as_tensor(got).reshape(-1).long(), torch.as_tensor(expected).reshape(-1).long()
    if got.shape != expected.shape:
        return max(got.numel(), expected.numel())
    return int((got != expected).sum())


def tally(comparisons):
    return probe.tally(comparisons)


def scope(report):
    """('full', []) when the run covered the design's set, else ('reduced', what fell short)."""
    short = []
    if report.get('capacity') != CAPACITY:
        short.append('capacity')
    if set(SECTIONS) - set(report.get('sections') or ()):
        short.append('sections')
    ran = report.get('r1_run') or {}
    if set(R1_GEOMETRIES) - set(ran) or any(set(DESIGN_RESIDUES) - set(words) for words in ran.values()):
        short.append('r1')
    families = set(report.get('r2_families_replayed') or ())
    if len(families) < R2_MIN_FAMILIES or set(R2_NAMED) - families:
        short.append('r2_families')
    if set(VARIANTS) - set(report.get('variants_run') or ()):
        short.append('variants')
    if set(IDLE_STARTS) - set(report.get('idle_starts_run') or ()):
        short.append('idle_starts')
    if set(SEEDS) - set(report.get('seeds_run') or ()):
        short.append('seeds')
    return ('full' if not short else 'reduced'), short


def decide(report):
    """PASS / FAIL / NO-DECISION from the comparisons, liveness, failures, error and deadline."""
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
        reasons.append('%d liveness controls did not move (%s)' % (len(dead), ', '.join(entry['label']
                                                                                       for entry in dead[:4])))
    deadline = report.get('deadline') or {}
    cut = [tag for tag in deadline.get('skipped', []) if tag.split('/')[0] in DECISIVE_RUNS]
    if cut:
        reasons.append('the deadline (%ss) cut %d decisive section runs (%s)'
                       % (deadline.get('seconds'), len(cut), ', '.join(cut[:6])))
    kinds = {entry['kind'] for entry in decisive}
    empty = [name for name in (report.get('sections') or ()) if not kinds & set(SECTION_KINDS.get(name, ()))]
    if empty:
        reasons.append('no decisive comparison of %s ran' % ','.join(empty))
    if not decisive:
        reasons.append('no decisive comparison ran')
    if reasons:
        verdict = 'NO-DECISION'
    elif differing:
        verdict = 'FAIL'
    else:
        verdict = 'PASS'
    extent_scope, short = scope(report)
    return dict(verdict=verdict, reasons=reasons, decisive=len(decisive), decisive_differing=len(differing),
                first_differing=[entry['label'] for entry in differing[:6]], scope=extent_scope, scope_short=short)


def verdict_line(report):
    decision = report['decision']
    counts = tally(report.get('comparisons', []))
    words = [READER, 'verdict=%s' % decision['verdict'], 'scope=%s' % decision.get('scope', 'reduced')]

    def part(name, kinds):
        runs = sum(counts.get(kind, {}).get('runs', 0) for kind in kinds)
        equal = sum(counts.get(kind, {}).get('equal', 0) for kind in kinds)
        return '%s=%d/%d' % (name, equal, runs) if runs else '%s=none' % name

    words.append(part('r1', ('r1_eager', 'r1_trace')))
    words.append(part('r1_reader', ('r1_reader',)))
    words.append(part('staging', SECTION_KINDS['S']))
    words.append(part('r2', ('r2_construction_vs_wide', 'r2_trace_vs_wide')))
    words.append(part('r2_trace', ('r2_trace_vs_eager',)))
    words.append(part('r4', SECTION_KINDS['R4']))
    liveness = report.get('liveness', [])
    words.append('live=%d/%d' % (sum(1 for entry in liveness if entry['live']), len(liveness)))
    words.append('families=%d' % len(report.get('r2_families_replayed') or ()))
    view = report.get('two_chip_view') or {}
    if view:
        words.append('chips=1of2 phantom=%d' % view.get('phantom_programs', 0))
    source = ((report.get('modules') or {}).get('sha256') or {}).get('extent_attention_replay.py')
    if source:
        words.append('extent_sha256=%s' % source)
    if decision.get('scope_short'):
        words.append('scope_short=%s' % ','.join(decision['scope_short']))
    if report.get('sections_failed'):
        words.append('sections_failed=%s' % ','.join(report['sections_failed']))
    if (report.get('deadline') or {}).get('skipped'):
        words.append('deadline_skipped=%d' % len(report['deadline']['skipped']))
    if decision['first_differing']:
        words.append('first_differing=%s' % json.dumps(decision['first_differing']))
    if decision['reasons']:
        words.append('reasons=%s' % json.dumps(decision['reasons']))
    return ' '.join(words)


def pindiag_problems(lines, capacity, segments=len(SEGMENTS)):
    """What one construction's PINDIAG lines lack: exactly one engaged line naming the four 0x27 segments, the
    narrow mask and the capacity, and one sdpa qwen-modes line per segment with flags ['0x27'] and mask=narrow."""
    problems = []
    engaged = [line for line in lines if line.startswith(ENGAGED_MARKER)]
    want = '%s segments=%d flags=%s mask=narrow capacity=%d' % (ENGAGED_MARKER, segments,
                                                                ','.join(['0x%x' % SERVED_FLAGS] * segments), capacity)
    if engaged != [want]:
        problems.append('expected one %r line, got %r' % (want, engaged))
    modes = [line for line in lines if line.startswith(MODES_MARKER + ' modes=')]
    good = [line for line in modes if "flags=['0x%x'] mask=narrow" % SERVED_FLAGS in line
            and 'modes=extent,share,slice,tail ' in line and ' capacity=%d ' % capacity in line]
    if len(modes) != segments or len(good) != segments:
        problems.append('expected %d %s lines with flags=[\'0x27\'] and mask=narrow, got %r' % (segments, MODES_MARKER,
                                                                                                modes))
    return problems


def check_log(report, text):
    """The factory's lines for every requested program (F4), and every 0x20 program's single F22 line."""
    lines = card.factory_lines(text)
    report['factory_lines'] = lines
    for key in probe.missing_programs(lines, report['_requested']):
        report['failures'].append('factory log: no [QWEN-SDPA] line for flags=0x%x B=%d St=%d mask_width_t=%d (graft '
                                  'mounted, not executed)' % key)
    events = card_b.extent_lines(text)
    report['extent_lines'] = [fields for kind, fields in events if kind == 'F22']
    for problem in card_b.extent_line_problems(events):
        report['failures'].append('extent log: %s' % problem)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# The one chip, presented to the reader code as two.
# ---------------------------------------------------------------------------------------------

class ViewProgram:
    """A MeshProgramDescriptor as the reader code builds it: one ProgramDescriptor per chip of the view, at mesh
    coordinate (0, chip). realise() makes the real one-chip descriptor from chip 0's once, and reuses it after."""

    def __init__(self, view):
        self.view, self.entries, self.realised = view, {}, None

    def __setitem__(self, key, program):
        if self.realised is not None:
            raise RuntimeError('TwoChipView: a launched program was changed')
        first, last = key
        if first != last or first[0] != 0 or not 0 <= first[1] < self.view.chips:
            raise ValueError('TwoChipView: one chip coordinate (0, 0..%d) per entry, got %r' % (self.view.chips - 1,
                                                                                              key))
        if first[1] in self.entries:
            raise ValueError('TwoChipView: chip %d programmed twice' % first[1])
        self.entries[first[1]] = program

    def realise(self):
        if self.realised is None:
            if sorted(self.entries) != list(range(self.view.chips)):
                raise ValueError('TwoChipView: a program for every chip of the view required, got chips %r'
                                 % sorted(self.entries))
            ttnn = self.view.ttnn
            real = ttnn.MeshProgramDescriptor()
            coordinate = ttnn.MeshCoordinate(0, 0)
            real[ttnn.MeshCoordinateRange(coordinate, coordinate)] = self.entries[0]
            self.realised = real
            self.view.realised += 1
            self.view.phantom += len(self.entries) - 1
        return self.realised


class TwoChipView:
    """ttnn for the reader code, on one chip shown as two. get_device_tensors gives the one shard `chips` times; a
    MeshProgramDescriptor is a ViewProgram whose chip-0 program alone is launched; MeshCoordinate and
    MeshCoordinateRange are plain tuples for it. Everything else is ttnn's. installed() makes it sys.modules['ttnn']
    for the pinned modules' function-local imports, and restores what was there."""

    def __init__(self, ttnn, chips=2):
        self.__dict__.update(ttnn=ttnn, chips=chips, realised=0, phantom=0, launches=0)

    def __getattr__(self, name):
        return getattr(self.__dict__['ttnn'], name)

    def get_device_tensors(self, tensor):
        shards = list(self.ttnn.get_device_tensors(tensor))
        if len(shards) != 1:
            raise RuntimeError('TwoChipView presents ONE chip as %d; this tensor has %d shards' % (self.chips,
                                                                                                 len(shards)))
        return shards * self.chips

    @staticmethod
    def MeshCoordinate(row, column):  # noqa: N802 - mirrors ttnn
        return (row, column)

    @staticmethod
    def MeshCoordinateRange(first, last):  # noqa: N802 - mirrors ttnn
        return (tuple(first), tuple(last))

    def MeshProgramDescriptor(self):  # noqa: N802 - mirrors ttnn
        return ViewProgram(self)

    def generic_op(self, tensors, program):
        if isinstance(program, ViewProgram):
            program = program.realise()
        self.launches += 1
        return self.ttnn.generic_op(tensors, program)

    @contextmanager
    def installed(self):
        saved = sys.modules.get('ttnn')
        sys.modules['ttnn'] = self
        try:
            yield self
        finally:
            if saved is None:
                sys.modules.pop('ttnn', None)
            else:
                sys.modules['ttnn'] = saved


@contextmanager
def tee_pindiag(modules, sink):
    """Each module's _pindiag also appends its line to `sink` (the lines still go where they went)."""
    saved = [(module, module._pindiag) for module in modules]

    def tee(original):
        def log(text):
            sink.append(text)
            original(text)
        return log

    for module, original in saved:
        module._pindiag = tee(original)
    try:
        yield sink
    finally:
        for module, original in saved:
            module._pindiag = original


def load_modules(ci_root, report):
    """The code under test, from `ci_root` (this checkout's scripts/ci as the runner mounts it; '' takes sys.path as
    it is): each module's file, the pinned sources' frozen bytes, the recorded shas. None, with failures, when any
    of it is wrong."""
    root = None
    if ci_root:
        root = Path(ci_root).resolve()
        if str(root) not in sys.path[:1]:
            sys.path.insert(0, str(root))
    modules, files, ok = {}, {}, True
    for key, name in MODULES:
        module = importlib.import_module(name)
        modules[key] = module
        path = Path(module.__file__).resolve()
        files[name] = str(path)
        if root is not None and path.parent != root:
            report['failures'].append('%s was loaded from %s, not from --ci-root %s (the code under test is this '
                                      'checkout\'s)' % (name, path, root))
            ok = False
    directory = Path(modules['extent'].__file__).resolve().parent
    shas = {}
    for name in tuple(PINNED) + RECORDED_SOURCES:
        path = directory / name
        shas[name] = file_sha256(path) if path.is_file() else None
    for name, digest in PINNED.items():
        if shas[name] != digest:
            report['failures'].append('pinned source %s is %s, not its frozen %s'
                                      % (name, (shas[name] or 'missing')[:16], digest[:16]))
            ok = False
    report['modules'] = dict(ci_root=str(root) if root else None, files=files, sha256=shas)
    return type('Modules', (), modules) if ok else None


# ---------------------------------------------------------------------------------------------
# Device harness.
# ---------------------------------------------------------------------------------------------

def stage_host(ttnn, device, destination, value, dtype, layout, label):
    """A host value into a device tensor in place, as the readers stage theirs (replicated host tensor, then
    copy_host_to_device_tensor)."""
    source = ttnn.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(device))
    with probe.WATCHDOG.op('stage ' + label):
        ttnn.copy_host_to_device_tensor(source, destination)


def read(ttnn, tensor, label):
    """The tensor on the host. A replicated tensor this harness or the reader uploaded may want a mesh composer on
    some runtimes; its one device shard never does (verify_t2_card_m reads so), so that is the fallback."""
    with probe.WATCHDOG.op('read back ' + label):
        try:
            return ttnn.to_torch(tensor)
        except Exception:  # noqa: BLE001 - retried on the one shard; a second failure is the section's
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 1:
                raise
            return ttnn.to_torch(shards[0])


def sync(ttnn, device, label):
    with probe.WATCHDOG.op('synchronize ' + label):
        ttnn.synchronize_device(device)


def record(report, entry, verbose=True):
    probe.record(report, entry, verbose)


def lend_storage(operations, mesh, torch, pool_module, page_width, users=USERS, rows=SEGMENT_ROWS,
                 group_rows=GROUP_ROWS):
    """serving_buffer_pool's extent storage for (users, rows), allocated as its constructor allocates it (per user,
    per bundle of the extent layout: a zeroed (entries, page_width) table, then a zeroed (entries,) cur_pos; int32,
    row-major, interleaved DRAM, replicated) and wrapped in its PackedExtentStorage."""
    tensors = []

    def allocate(shape):
        value = operations.from_torch(torch.zeros(shape, dtype=torch.int32), device=mesh, dtype=operations.int32,
                                      layout=operations.ROW_MAJOR_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
                                      mesh_mapper=operations.ReplicateTensorToMesh(mesh))
        tensors.append(value)
        return value

    try:
        tables, cur_pos = [], []
        for user in range(users):
            tables.append([])
            cur_pos.append([])
            for entries in pool_module.extent_bundle_batches(rows, group_rows):
                tables[-1].append(allocate((entries, page_width)))
                cur_pos[-1].append(allocate((entries,)))
        return pool_module.PackedExtentStorage(users, rows, tables, cur_pos)
    except BaseException:
        for value in tensors:
            operations.deallocate(value)
        raise


def mirrors_agree(torch, extent_module, rows, batches, offset, words, report, geometry):
    """The module's narrow_mask_host against k64j_card_b.served_mask at capacity 256, on the host: two
    transliterations of attention_mask_replay.cpp:18-33 that must agree before either is an oracle."""
    for word in words:
        mirror = extent_module.narrow_mask_host(word, rows, batches, offset)
        other = card_b.served_mask(torch, word, K_CHUNK, rows=rows, batches=batches, offset=offset)
        if card.differing(torch, mirror, other):
            report['failures'].append('R1/%s: the host mirrors of the mask kernel disagree at word %d '
                                      '(narrow_mask_host vs k64j_card_b.served_mask): no oracle' % (geometry, word))
            return False
    return True


def section_r1(ttnn, torch, device, mods, view, args, report):
    """R1: the pinned mask kernel at capacity 256 through prepare_narrow, eager and in a trace, against the mirror."""
    x, mask_replay = mods.extent, mods.mask
    ran = report.setdefault('r1_run', {})
    for name in args.r1_geometries:
        rows, batches, offset = R1_GEOMETRIES[name]
        if not mirrors_agree(torch, x, rows, batches, offset, args.r1_words, report, name):
            continue
        shape = (batches, 1, rows * 12, K_CHUNK)
        owned, trace = [], None
        try:
            with view.installed(), probe.WATCHDOG.op('R1 %s prepare' % name):
                positions = x._upload(view, device, torch.zeros(8, dtype=torch.int32), view.int32)
                owned.append(positions)
                mask = x._upload(view, device, torch.zeros(shape, dtype=torch.bfloat16), view.bfloat16)
                owned.append(mask)
                program = x.prepare_narrow(device, positions, mask, rows=rows, batches=batches, offset=offset)

            def launch():
                with view.installed():
                    mask_replay.execute(positions, mask, program)

            def one(word, mode):
                label = 'R1/%s/word%d/%s' % (name, word, mode)
                probe.DEADLINE.check(label)
                stage_host(ttnn, device, mask, torch.full(shape, float('nan'), dtype=torch.bfloat16), ttnn.bfloat16,
                           ttnn.TILE_LAYOUT, label + ' poison')
                stage_host(ttnn, device, positions, torch.tensor(expected_word(word), dtype=torch.int32), ttnn.int32,
                           ttnn.ROW_MAJOR_LAYOUT, label + ' word')
                with probe.WATCHDOG.op(label):
                    if mode == 'eager':
                        launch()
                    else:
                        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                sync(ttnn, device, label)
                got = read(ttnn, mask, label)
                expected = x.narrow_mask_host(word, rows, batches, offset)
                record(report, probe.comparison('R1', 'r1_' + mode, label, card.differing(torch, got, expected), True,
                                                geometry=name, word=word))

            for word in args.r1_words:
                one(word, 'eager')
            with probe.WATCHDOG.op('R1 %s capture' % name):
                trace, _ = card.capture(ttnn, device, launch)
            for word in args.r1_words:
                one(word, 'trace')
            ran[name] = list(args.r1_words)
        finally:
            if trace is not None:
                ttnn.release_trace(device, trace)
            for value in owned:
                ttnn.deallocate(value)


class Block:
    """One seed's packed block: the pool (the seed's bf8 K / V, the poison, a page table per user), the lent storage,
    the PackedExtentReplayReader over it, the (1, 64, 12, 256) query, the trace, and what is staged now."""

    def __init__(self, ttnn, torch, device, mods, view, seed, args, report):
        self.ttnn, self.torch, self.device, self.mods, self.view = ttnn, torch, device, mods, view
        self.seed, self.args, self.report = seed, args, report
        self.capacity = args.capacity
        self.width = args.capacity // card.PAGE
        self.pool = self.storage = self.reader = self.query = self.trace = self.output = None
        self.starts, self.tables, self.tokens = [None] * USERS, [None] * USERS, [None] * USERS
        self.kwargs = None
        self.token_cache = {}

    # host values ----------------------------------------------------------------------------
    def table(self, segment, family):
        """The segment's (1, C / 64) table: its user's pages up to E, the poison blocks past it."""
        pool = self.pool
        return probe.poisoned_row(self.torch, pool.tables[segment], family, self.capacity, pool.poison)[None]

    def idle_table(self):
        return self.torch.zeros(1, self.width, dtype=self.torch.int32)

    def segment_tokens(self, segment, start, variant):
        """(1, 16, 12, 256): k64j_card_b.token_query per row position (peaky aims at the user's own keys)."""
        key = (segment, start, variant)
        if key not in self.token_cache:
            pool, torch = self.pool, self.torch
            peaky = variant == 'peaky'
            rows = [card_b.token_query(torch, self.seed, variant, position, keys=pool.keys if peaky else None,
                                       table=pool.tables[segment] if peaky else None)
                    for position in range(start, start + SEGMENT_ROWS)]
            self.token_cache[key] = torch.stack(rows)[None]
        return self.token_cache[key]

    # device ---------------------------------------------------------------------------------
    def build(self, first):
        ttnn, torch, view, report = self.ttnn, self.torch, self.view, self.report
        x, pool_module = self.mods.extent, self.mods.pool
        self.pool = card_b.ExtentPool(ttnn, torch, self.device, self.capacity, self.seed, USERS, report)
        self.pool.output = self.args.output_memory
        with probe.WATCHDOG.op('lend storage seed%d' % self.seed):
            self.storage = lend_storage(view, self.device, torch, pool_module, self.width).take()
        nonzero = sum(int(read(ttnn, value, 'lent storage').long().abs().sum() != 0) for value in self.storage.tensors)
        if nonzero:
            report['failures'].append('block/seed%d: %d lent storage tensors are not zero before construction'
                                      % (self.seed, nonzero))
        tables = [self.table(segment, family) for segment, family in enumerate(first['families'])]
        before = len(report['pindiag'])
        with view.installed(), probe.WATCHDOG.op('construct the extent reader seed%d' % self.seed):
            self.reader = x.PackedExtentReplayReader(view, self.device, SEGMENTS, self.width, tables,
                                                     storage=self.storage.segment_storage(), max_group_rows=GROUP_ROWS,
                                                     starts=tuple(first['starts']))
        for problem in pindiag_problems(report['pindiag'][before:], self.capacity):
            report['failures'].append('block/seed%d: PINDIAG: %s' % (self.seed, problem))
        report['_requested'].add(probe.program_key(self.capacity, SERVED_BATCH, K_CHUNK, SERVED_FLAGS))
        self.starts, self.tables = list(first['starts']), tables
        report.setdefault('extent_reader', dict(flags=['0x%x' % value for reader in self.reader.readers
                                                for value in reader.sdpa_modes_applied],
                                         segments=len(self.reader.readers), capacity=self.reader.capacity,
                                         borrowed=len(self.reader.borrowed)))

    def check_staging(self, phase, segments=None, tables=True):
        """S: each segment's word, cur_pos and (when `tables`) lent tables, read back, against this harness's own
        values for the starts and tables it staged."""
        ttnn, torch, report = self.ttnn, self.torch, self.report
        for segment in (range(USERS) if segments is None else segments):
            own, start = self.reader.readers[segment], self.starts[segment]
            label = 'S/seed%d/%s/segment%d/start%d' % (self.seed, phase, segment, start)
            kind = 'staging' if phase == 'construction' else 'restage'
            got = read(ttnn, own.positions, label + ' word')
            record(report, probe.comparison('S', kind + '_word', label + '/word',
                                            elements_differing(torch, got, expected_word(start)), True, start=start))
            for index, ((bundle, pages, _mask, _config), positions) in enumerate(zip(own.metadata, own.cur_pos)):
                got = read(ttnn, positions, label + ' cur_pos')
                record(report, probe.comparison('S', kind + '_cur_pos', '%s/bundle%d/cur_pos' % (label, index),
                                                elements_differing(torch, got, expected_cur_pos(start, len(bundle))),
                                                True, start=start))
                if tables:
                    want = self.tables[segment][:, :self.width].repeat(len(bundle), 1)
                    got = read(ttnn, pages, label + ' table')
                    record(report, probe.comparison('S', kind + '_table', '%s/bundle%d/table' % (label, index),
                                                    elements_differing(torch, got, want), True, start=start))

    def write_query(self):
        block = self.torch.cat(self.tokens, dim=1).contiguous()
        stage_host(self.ttnn, self.device, self.query, block, self.ttnn.bfloat16, self.ttnn.TILE_LAYOUT,
                   'query seed%d' % self.seed)

    def restage(self, starts, tables=None, segments=None):
        """Through the reader: per segment with its table (ExtentSegmentReader.stage), or the block's word and
        cur_pos only (PackedExtentReplayReader.stage)."""
        with self.view.installed(), probe.WATCHDOG.op('restage seed%d' % self.seed):
            if tables is None:
                self.reader.stage(tuple(starts))
            else:
                for segment in (range(USERS) if segments is None else segments):
                    self.reader.readers[segment].stage(starts[segment], table=tables[segment])
        for segment in (range(USERS) if segments is None else segments):
            self.starts[segment] = starts[segment]
            if tables is not None:
                self.tables[segment] = tables[segment]

    def forward(self):
        with self.view.installed():
            with self.reader.shared_masks(1):
                return self.reader(self.query, self.pool.k, self.pool.v, **self.kwargs)

    def eager(self, label):
        with self.view.installed(), probe.WATCHDOG.op(label):
            output = self.reader(self.query, self.pool.k, self.pool.v, **self.kwargs)
        try:
            return read(self.ttnn, output, label)
        finally:
            self.ttnn.deallocate(output)

    def capture(self):
        with probe.WATCHDOG.op('capture seed%d' % self.seed):
            self.trace, self.output = card.capture(self.ttnn, self.device, self.forward)
        self.report['captures'] = self.report.get('captures', 0) + 1

    def replay(self, label):
        with probe.WATCHDOG.op(label):
            self.ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=True)
        sync(self.ttnn, self.device, label)
        return read(self.ttnn, self.output, label)

    def reference(self, tokens, start, family, table_row, label, zero=False):
        """The 0x7 compile-time call at capacity `family`: the ticket folded on the host (two eight-row groups), the
        table truncated to E / 64 and repeated per entry, the wide mask with the real -inf tail (or zero), then
        unfolded on the host: (1, 16, 12, 256)."""
        torch, pool = self.torch, self.pool
        folded = card.fold_entries(torch, tokens, SERVED_OFFSETS, SERVED_ROWS)
        pages = table_row.reshape(-1)[:family // card.PAGE].repeat(SERVED_BATCH, 1).contiguous()
        mask = (torch.zeros(SERVED_BATCH, 1, SERVED_ROWS * 12, family, dtype=torch.bfloat16) if zero
                else card_b.wide_mask(torch, start, family))
        tensors = []
        try:
            tensors.append(pool.upload(folded))
            tensors.append(pool.upload(pages, self.ttnn.int32))
            tensors.append(pool.upload(mask))
            output = pool.served_run(tensors[0], tensors[1], tensors[2], COMPILE_FLAGS, label=label)
        finally:
            for value in tensors:
                self.ttnn.deallocate(value)
        return card.unfold_entries(torch, output, SERVED_ROWS)

    def reader_masks(self, label):
        """R1 through the reader: each segment's narrow masks after the replay's in-trace refresh."""
        torch, x = self.torch, self.mods.extent
        for segment, own in enumerate(self.reader.readers):
            start = self.starts[segment]
            for index, (bundle, _pages, mask, _config) in enumerate(own.metadata):
                got = read(self.ttnn, mask, label + ' mask')
                expected = x.narrow_mask_host(start & (K_CHUNK - 1), bundle[0]['rows'], len(bundle),
                                              bundle[0]['offset'])
                record(self.report, probe.comparison(
                    'R1', 'r1_reader', '%s/segment%d/bundle%d/start%d' % (label, segment, index, start),
                    card.differing(torch, got, expected), True, start=start), verbose=False)

    def close(self):
        ttnn = self.ttnn
        try:
            if self.trace is not None:
                ttnn.release_trace(self.device, self.trace)
                self.trace = None
            for value in (self.output, self.query):
                if value is not None:
                    ttnn.deallocate(value)
            self.output = self.query = None
            if self.reader is not None:
                with self.view.installed():
                    self.reader.close()
            if self.storage is not None:
                self.storage.release()
                for value in self.storage.tensors:
                    ttnn.deallocate(value)
                self.storage = None
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool = None


def rows_of(output, segment):
    first, last = SEGMENTS[segment]
    return output[:, first:last]


def section_block(ttnn, torch, device, mods, view, seed, args, report):
    """S, R2 and R4 on one seed's block: construction staging, the construction-state call, one capture, the replays
    across the plan, the idle patterns and the liveness controls."""
    wants = set(args.sections) & set(BLOCK_SECTIONS)
    plan = replay_plan(args.families, args.r2_residues, args.r2_restages, args.capacity)
    idle_at = idle_entries(plan, args.capacity)
    if 'R2' not in wants:
        plan = [plan[index] for index in idle_at]         # what R4 replays at (S uses the first alone)
        idle_at = list(range(len(plan)))
    block = Block(ttnn, torch, device, mods, view, seed, args, report)
    replayed = set(report.get('r2_families_replayed') or ())
    try:
        block.build(plan[0])
        block.check_staging('construction')
        if not {'R2', 'R4'} & wants:
            return
        first = plan[0]
        with view.installed():
            block.query = mods.extent._upload(view, device, torch.zeros(1, USERS * SEGMENT_ROWS, 12, card.HEAD_DIM,
                                                                        dtype=torch.bfloat16), view.bfloat16)
        block.kwargs = dict(page_table_tensor=None, cur_pos_tensor=None, scale=card_b.NATIVE_SCALE,
                            program_config=None, memory_config=block.pool.output_memory())
        variant = args.variants[0]
        block.tokens = [block.segment_tokens(segment, start, variant) for segment, start in enumerate(first['starts'])]
        block.write_query()
        # The construction state, never restaged (model_batch skips the initial stage when the capture start matches).
        label = 'R2/seed%d/%s/construction' % (seed, variant)
        got = block.eager(label)
        if 'R2' in wants:
            for segment in range(USERS):
                reference = block.reference(block.tokens[segment], first['starts'][segment],
                                            first['families'][segment], block.pool.tables[segment],
                                            '%s/segment%d ref' % (label, segment))
                family = first['families'][segment]
                record(report, probe.comparison(
                    'R2', 'r2_construction_vs_wide', '%s/segment%d/E%d' % (label, segment, family),
                    card.differing(torch, rows_of(got, segment), reference), True, seed=seed, variant=variant,
                    extent=family))
        block.capture()
        for variant in args.variants:
            for index, entry in enumerate(plan):
                label = 'R2/seed%d/%s/a%d.%d' % (seed, variant, entry['assignment'], entry['restage'])
                probe.DEADLINE.check(label)
                if entry['tables']:
                    block.restage(entry['starts'], [block.table(segment, family)
                                                    for segment, family in enumerate(entry['families'])])
                else:
                    block.restage(entry['starts'])
                block.tokens = [block.segment_tokens(segment, start, variant)
                                for segment, start in enumerate(entry['starts'])]
                block.write_query()
                block.check_staging('restage', tables=entry['tables'])
                got = block.replay(label)
                if 'R1' in args.sections:
                    block.reader_masks(label)
                if 'R2' in wants:
                    eager = block.eager(label + ' eager')
                    record(report, probe.comparison('R2', 'r2_trace_vs_eager', label, card.differing(torch, got, eager),
                                                    True, seed=seed, variant=variant))
                    for segment in range(USERS):
                        family, start = entry['families'][segment], entry['starts'][segment]
                        reference = block.reference(block.tokens[segment], start, family, block.pool.tables[segment],
                                                    '%s/segment%d ref' % (label, segment))
                        record(report, probe.comparison(
                            'R2', 'r2_trace_vs_wide', '%s/segment%d/E%d/s%d' % (label, segment, family, start),
                            card.differing(torch, rows_of(got, segment), reference), True, seed=seed, variant=variant,
                            extent=family, start=start))
                        replayed.add(family)
                    report['r2_families_replayed'] = sorted(replayed)
                    report['variants_run'] = sorted(set(report.get('variants_run') or ()) | {variant})
                if index == 0:
                    liveness(block, entry, got, variant, label)
                if index in idle_at and 'R4' in wants:
                    section_idle(block, entry, got, variant, label)
        report['seeds_run'] = sorted(set(report.get('seeds_run') or ()) | {seed})
    finally:
        block.close()


def liveness(block, entry, good, variant, label):
    """cur_pos_live and mask_live on the first segment whose family is below C."""
    torch, report = block.torch, block.report
    segment = next((index for index, family in enumerate(entry['families']) if family < block.capacity), None)
    if segment is None:
        report['warnings'].append('%s: no segment below C, no liveness control here' % label)
        return
    family, start = entry['families'][segment], entry['starts'][segment]
    own = block.reader.readers[segment]
    for positions in own.cur_pos:
        stage_host(block.ttnn, block.device, positions, torch.tensor([family] * positions.shape[0], dtype=torch.int32),
                   block.ttnn.int32, block.ttnn.ROW_MAJOR_LAYOUT, label + ' cur_pos E')
    try:
        moved = block.eager(label + ' cur_pos=E')
    finally:
        block.restage(list(block.starts), segments=None)
    differing = card.differing(torch, rows_of(moved, segment), rows_of(good, segment))
    report['liveness'].append(dict(section='R2', label='%s/segment%d/cur_pos_live' % (label, segment),
                                   kind='cur_pos_live', differing=differing, live=differing > 0, extent=family))
    real = block.reference(block.tokens[segment], start, family, block.pool.tables[segment], label + ' mask ref')
    zero = block.reference(block.tokens[segment], start, family, block.pool.tables[segment], label + ' zero-mask ref',
                           zero=True)
    differing = card.differing(torch, real, zero)
    report['liveness'].append(dict(section='R2', label='%s/segment%d/mask_live' % (label, segment), kind='mask_live',
                                   differing=differing, live=differing > 0, extent=family))


def section_idle(block, entry, baseline, variant, label):
    """R4: per pattern, idle segments on the zero table at E = 256 in the same trace."""
    torch, report = block.torch, block.report
    for pattern in block.args.idle_patterns:
        idle = idle_assignment(pattern)
        tag = '%s/idle%s' % (label, '+'.join(str(segment) for segment in pattern))
        probe.DEADLINE.check(tag)
        starts = list(block.starts)
        tables = list(block.tables)
        for segment, start in idle.items():
            starts[segment], tables[segment] = start, block.idle_table()
            block.tokens[segment] = block.segment_tokens(segment, start, variant)
        block.restage(starts, tables, segments=sorted(idle))
        block.write_query()
        block.check_staging('restage', segments=sorted(idle))
        got = block.replay(tag)
        for segment in range(USERS):
            if segment in idle:
                rows = rows_of(got, segment)
                finite = int((~torch.isfinite(rows.float())).sum())
                record(report, probe.comparison('R4', 'r4_idle_finite', '%s/segment%d' % (tag, segment), finite, True,
                                                start=idle[segment]))
                reference = block.reference(block.tokens[segment], idle[segment], IDLE_FAMILY, block.idle_table(),
                                            '%s/segment%d ref' % (tag, segment))
                record(report, probe.comparison('R4', 'r4_idle_vs_wide', '%s/segment%d/start%d' % (
                    tag, segment, idle[segment]), card.differing(torch, rows, reference), True, start=idle[segment]))
                report['idle_starts_run'] = sorted(set(report.get('idle_starts_run') or ()) | {idle[segment]})
            else:
                differing = card.differing(torch, rows_of(got, segment), rows_of(baseline, segment))
                record(report, probe.comparison('R4', 'r4_live_unchanged', '%s/segment%d' % (tag, segment), differing,
                                                True))
        # Back to the assignment's live state.
        restore = sorted(idle)
        for segment in restore:
            block.tokens[segment] = block.segment_tokens(segment, entry['starts'][segment], variant)
        block.restage(list(entry['starts']), [block.table(segment, family)
                                              for segment, family in enumerate(entry['families'])], segments=restore)
        block.write_query()


def run(args, report, checkpoint=None):
    import torch
    import ttnn

    mods = load_modules(args.ci_root, report)
    if mods is None:
        return
    options = dict(device_id=args.device_id, l1_small_size=24576)
    if {'R1', 'R2', 'R4'} & set(args.sections):
        options['trace_region_size'] = args.trace_region_bytes
    with probe.WATCHDOG.op('open device', extra=OPEN_EXTRA_S):
        device = ttnn.open_device(**options)
    try:
        try:
            device.enable_program_cache()
            report['program_cache_enabled_call'] = True
        except Exception as error:  # noqa: BLE001 - default-on in newer runtimes
            report['program_cache_enabled_call'] = repr(error)[:200]
        grid = device.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        if not card_b.check_binary(args, report):
            return
        if args.kernel_root and not card_b.check_kernels(args.kernel_root, report):
            return
        if os.environ.get(card.SCRATCH_ENV) != '1':
            report['failures'].append('%s=1 is required: the extent reader refuses eight-row groups without it (the '
                                      'pinned reader\'s G8 precondition)' % card.SCRATCH_ENV)
            return
        try:
            modes = sorted(mods.pooled.sdpa_modes())
        except ValueError as error:
            modes = repr(error)
        if modes != list(SDPA_MODES):
            report['failures'].append('%s must be %s, the image\'s (the reader adds extent itself); got %r'
                                      % (SDPA_MODES_ENV, ','.join(('tail', 'share', 'slice')), modes))
            return
        view = TwoChipView(ttnn)
        try:
            with tee_pindiag((mods.extent, mods.pooled), report['pindiag']):
                run_sections(ttnn, torch, device, mods, view, args, report, checkpoint)
        finally:
            report['two_chip_view'] = dict(chips_physical=1, chips_presented=view.chips,
                                           programs_realised=view.realised, phantom_programs=view.phantom,
                                           launches=view.launches)
    finally:
        with probe.WATCHDOG.op('close device'):
            ttnn.close_device(device)


def run_sections(ttnn, torch, device, mods, view, args, report, checkpoint=None):
    """Every run, each isolated: one that raises is a failure and the run goes on; the deadline stops it and lists
    the rest; the report is checkpointed after every run."""
    runs = section_runs(args)
    report['plan_runs'] = [run_tag(seed, name) for seed, name in runs]
    done, failed = report.setdefault('sections_done', []), report.setdefault('sections_failed', [])
    requested = report.get('_requested')
    for index, (seed, name) in enumerate(runs):
        tag = run_tag(seed, name)
        if isinstance(requested, card_b.RequestLog):
            requested.section = name
        try:
            probe.DEADLINE.check(tag)
            if name == 'R1':
                section_r1(ttnn, torch, device, mods, view, args, report)
            else:
                section_block(ttnn, torch, device, mods, view, seed, args, report)
        except probe.DeadlineReached as reached:
            skipped = [run_tag(later_seed, later) for later_seed, later in runs[index:]]
            report['deadline'] = dict(seconds=probe.DEADLINE.seconds, reached_at=str(reached), skipped=skipped)
            report['warnings'].append('deadline %ss reached at %s: %d section runs not run (%s)'
                                      % (probe.DEADLINE.seconds, reached, len(skipped), ', '.join(skipped)))
            print('DEADLINE reached at %s: skipped %s' % (reached, ', '.join(skipped)), flush=True)
            break
        except Exception as error:  # noqa: BLE001 - isolate the run; a TT_FATAL surfaces as RuntimeError
            failed.append(tag)
            report['failures'].append('%s: %s' % (tag, probe.one_line(error)))
            print('SECTION FAILED %s: %s' % (tag, probe.one_line(error)), flush=True)
        else:
            done.append(tag)
        finally:
            if isinstance(requested, card_b.RequestLog):
                requested.section = None
            if checkpoint is not None:
                checkpoint(tag)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--capacity', type=int, default=CAPACITY, help='the C-wide lent table (keys)')
    parser.add_argument('--sections', default=','.join(SECTIONS), help='any of %s' % ','.join(SECTIONS))
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--variants', default=','.join(VARIANTS), help='normal and/or peaky')
    parser.add_argument('--r1-geometries', default=','.join(R1_GEOMETRIES), help='any of %s' % ','.join(R1_GEOMETRIES))
    parser.add_argument('--r1-words', default=','.join(map(str, R1_WORDS)), help='mask words 0..255 (s & 255)')
    parser.add_argument('--r2-named', default=','.join(map(str, R2_NAMED)),
                        help='families R2 replays first (those above --capacity are dropped)')
    parser.add_argument('--r2-families', type=int, default=R2_FAMILIES,
                        help='distinct families R2 replays in all (the named ones, then evenly spread ones)')
    parser.add_argument('--r2-residues', default=','.join(map(str, DESIGN_RESIDUES)), help='s mod 256 of the starts')
    parser.add_argument('--r2-restages', type=int, default=R2_RESTAGES,
                        help='1: per segment with tables; 2: then word and cur_pos only')
    parser.add_argument('--idle-patterns', default=IDLE_PATTERNS,
                        help='R4: per replay, the segments that go idle ("3,2+3": segment 3; then 2 and 3)')
    parser.add_argument('--output-memory', choices=card_b.OUTPUT_MEMORY, default=card_b.OUTPUT_MEMORY[0],
                        help='the calls\' output memory (the model passes L1, tp.py:753)')
    parser.add_argument('--trace-region-bytes', type=int, default=32 << 20)
    parser.add_argument('--watchdog', type=float, default=0, help='seconds per device call before os._exit(3); 0 off')
    parser.add_argument('--deadline-s', type=float, default=0,
                        help='stop cleanly between device calls this long after start; 0 off')
    parser.add_argument('--expect-binary-sha256', default='',
                        help='the mapped _ttnncpp.so must be this (build_k64j.sh\'s K64J_TTNNCPP_SHA256; required)')
    parser.add_argument('--kernel-root', default=card_b.KERNEL_ROOT, help='the mounted sdpa_decode kernels ("" skips)')
    parser.add_argument('--ci-root', default=CI_ROOT,
                        help='the scripts/ci the code under test must come from ("" takes sys.path as it is)')
    args = parser.parse_args(argv)

    def ints(text):
        return [int(value) for value in text.split(',') if value.strip()]

    try:
        args.seeds = ints(args.seeds)
        args.r1_words = ints(args.r1_words)
        args.r2_named = ints(args.r2_named)
        args.r2_residues = ints(args.r2_residues)
        args.idle_patterns = parse_patterns(args.idle_patterns)
        probe.check_capacity(args.capacity)
    except ValueError as error:
        parser.error(str(error))
    args.sections = [value for value in args.sections.split(',') if value]
    args.variants = [value for value in args.variants.split(',') if value]
    args.r1_geometries = [value for value in args.r1_geometries.split(',') if value]
    if not args.sections or any(name not in SECTIONS for name in args.sections):
        parser.error('--sections must be a non-empty subset of %s' % ','.join(SECTIONS))
    if not args.seeds or not args.variants or any(value not in VARIANTS for value in args.variants):
        parser.error('--seeds and --variants (normal, peaky) must be non-empty')
    if any(name not in R1_GEOMETRIES for name in args.r1_geometries) or ('R1' in args.sections
                                                                         and not args.r1_geometries):
        parser.error('--r1-geometries must be a subset of %s' % ','.join(R1_GEOMETRIES))
    if ('R1' in args.sections and not args.r1_words) or any(not 0 <= word < K_CHUNK for word in args.r1_words):
        parser.error('--r1-words must be 0..255')
    if not args.r2_residues or any(not 0 <= residue < K_CHUNK for residue in args.r2_residues):
        parser.error('--r2-residues must be non-empty and 0..255')
    if args.r2_restages not in (1, 2) or args.r2_families < 1:
        parser.error('--r2-restages must be 1 or 2 and --r2-families at least 1')
    if args.capacity < 2 * K_CHUNK:
        parser.error('--capacity must hold at least two families')
    try:
        args.families = family_plan(args.capacity, args.r2_named, args.r2_families)
        replay_plan(args.families, args.r2_residues, args.r2_restages, args.capacity)
    except ValueError as error:
        parser.error(str(error))
    if args.deadline_s < 0:
        parser.error('--deadline-s must be >= 0')
    if args.expect_binary_sha256 and (len(args.expect_binary_sha256) != 64
                                      or any(c not in '0123456789abcdef' for c in args.expect_binary_sha256)):
        parser.error('--expect-binary-sha256 must be a full lowercase sha256')
    return args


def term_handler(report, write_report):
    """SIGTERM: record the error, write the partial report (NO-DECISION) at once and unwind."""

    def on_term(signum, _frame):
        if report.get('terminated'):
            return
        report['terminated'] = signum
        report['error'] = ('terminated by signal %d during %r (the container timeout: --deadline-s did not stop the '
                           'run first)' % (signum, probe.WATCHDOG.label))
        try:
            decision = decide(report)
            write_report(dict(decision=decision, verdict_line=verdict_line(dict(report, decision=decision)),
                              in_progress='terminated'))
        except Exception:  # noqa: BLE001 - best effort; main writes again after the unwinding
            pass
        raise probe.Terminated(signum)

    return on_term


def main(argv=None):
    args = parse_args(argv)
    probe.DEADLINE = probe.Deadline(args.deadline_s)
    report = dict(reader=READER, plan=PLAN, passed=False, argv=list(sys.argv[1:] if argv is None else argv),
                  capacity=args.capacity, sections=args.sections, seeds=args.seeds, variants=args.variants,
                  r1_geometries=args.r1_geometries, r1_words=args.r1_words, r2_families=args.families,
                  r2_residues=args.r2_residues, r2_restages=args.r2_restages,
                  idle_patterns=[list(pattern) for pattern in args.idle_patterns], output_memory=args.output_memory,
                  segments=[list(span) for span in SEGMENTS], served=dict(flags='0x%x' % SERVED_FLAGS,
                  reference_flags='0x%x' % COMPILE_FLAGS, rows=SERVED_ROWS, batch=SERVED_BATCH,
                  offsets=list(SERVED_OFFSETS), k_chunk_size=K_CHUNK),
                  poison=dict(k=probe.POISON_K, v=probe.POISON_V, blocks=probe.POISON_BLOCKS),
                  deadline_s=args.deadline_s, env={name: os.environ.get(name) for name in ENV_RECORDED},
                  watchdog=args.watchdog, pindiag=[], failures=[], warnings=[], comparisons=[], liveness=[])
    report['_requested'] = card_b.RequestLog()
    native = card.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        payload['requested_programs'] = sorted(list(key) for key in report.get('_requested', ()))
        payload['tally'] = tally(report.get('comparisons', []))
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def checkpoint(tag):
        try:
            write_report(dict(in_progress='after %s' % tag))
        except Exception as error:  # noqa: BLE001 - the final write still comes
            report['warnings'].append('checkpoint write after %s failed: %s' % (tag, probe.one_line(error)))

    def on_fire(label):
        try:
            write_report(dict(error='watchdog: %r exceeded its budget' % (label,), passed=False))
        except Exception:  # noqa: BLE001 - the WATCHDOG line stands
            pass

    watchdog = k1.Watchdog(args.watchdog, on_fire=on_fire).start()
    probe.WATCHDOG = card.WATCHDOG = k1.WATCHDOG = watchdog
    installed, previous = probe.install_signal(signal.SIGTERM, term_handler(report, write_report))
    try:
        try:
            with native:
                run(args, report, checkpoint)
            if report.get('binary', {}).get('stage', 0) >= 1:
                check_log(report, native.text())
        except probe.Terminated:
            pass                                                # report['error'] was set by the handler
        except Exception as error:  # noqa: BLE001
            report['error'] = '%s: %s' % (type(error).__name__, error)
        report['decision'] = decide(report)
        report['verdict_line'] = verdict_line(report)
        report['passed'] = report['decision']['verdict'] == 'PASS'
    finally:
        if installed:
            probe.install_signal(signal.SIGTERM, previous)
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    for warning in report['warnings']:
        print('WARN', warning)
    if report.get('error'):
        print('ERROR', report['error'])
    print(report.get('verdict_line', READER + ' verdict=NO-DECISION'), flush=True)
    print('EXTENT_READER_CARD passed=%s comparisons=%d liveness=%d failures=%d warnings=%d report=%s native_log=%s' % (
        report['passed'], len(report['comparisons']), len(report['liveness']), len(report['failures']),
        len(report['warnings']), args.out, native.path), flush=True)
    if report.get('terminated'):
        return 128 + int(report['terminated'])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
