"""P1: the pair drafter's row-1 draft SDPA on the qualification card, op level (QWEN_FAST_PAIR_ROW_EXACT).

WHAT IT DECIDES. The packed pair drafter drafts a user in pair ROW 1 (block rows 16-31) differently from the
same user alone or in row 0: deterministically, whoever the partner is, and never better (h1a-draft-race.md
sections 2-3). The code narrows it to the pair's draft SDPA. M1: row 1's first live chunk merges against a
state carried from 65 fully masked chunks. This probe settles that on the served kernel, and it qualifies the
fix (pair_row_exact.py, the GQA fold) bit for bit and in time.

Card B is one p150a, so execute_proposal (TP2) cannot run here. Every case is the draft SDPA, or a row-local
draft op, called exactly as the served code calls it: draft_attention.draft_sdpa, pair_row_exact.fold_attention,
draft_shared_head.local_head_candidates, the branch's program configurations. The operands are seeded bf16
tensors of the served shapes.

Operands, per seed and regime. For users a and b: a 2048-row cache K/V (1, 4, 2048, 128); 16 live K/V rows and
16 pad K/V rows (the single-user trace's live block is [live | pad], the pair's is [live a | live b]); 16 query
rows and 16 pad query rows. The regimes:
  normal       N(0, 1)
  peaked       queries x4
  negative     every visible key of the first live chunk scores below zero (queries |x|, cache keys 0-31 -|x|)
  partner100   user a's keys and values x100; user b's bits unchanged

Cases. Every comparison is bitwise (int16 views), per user block of 16 rows.
  C0  alone: user u's single-user call. K = [cache | live | pad] (2080), the served single-user mask.
  C1  the served pair [a | b]. K/V are assembled ON DEVICE exactly as draft_attention_branch does
      (key_value_plan's slices and concat); the packed mask is (1, 1, 32, 4160). Row 0 vs C0(a), row 1 vs C0(b).
  C2  the pair swapped, [b | a]: which row differs follows the row, not the user.
  C3  face: C0 with the query and mask row halves swapped (the user at rows 16-31, K = 2080). Shows an SDPA face
      asymmetry, if there is one, without masked leading chunks.
  C4  leading masked chunks: C0 with n in {1, 2, 8, 65} fully masked 32-key chunks in front. Their contents are
      zeros, the partner's cache, or the partner's cache x100. M1 predicts a difference that depends on n and not
      on the contents.
  C5  assembly: the device-assembled pair K/V read back against the host concatenation of the same pieces.
  C6  the fixes, each on C1's assembled K/V, each row vs C0:
        r1g          pair_row_exact.fold_attention, the served fix
        r1g-noshift  no row shift, a per-head mask; relies on face symmetry
        b2           batch 2, with copies
        two-calls    two SDPAs on the two segments
  C7  the head: linear on a random (5120, 124160) shard (bf16 and bf8), then local_head_candidates' chunked
      top-16. The same 16 rows are fed three ways: as a 16-row input (the single-user path's slice), at rows 0-15
      of a 32-row input, and at rows 16-31 of it. Logits and top-16 values and indices are compared, and ties at
      the 16th place are counted.
  C8  face sweep: every row-local draft op with its served program configuration, the same 16 rows at rows 0-15
      and at rows 16-31. The ops: rms_norm, the seven projection matmul configurations, typecast,
      rotary_embedding_hf, the head rms_norm, silu, multiply and add.
  P   partner independence: C1's row 1 in the normal and the partner100 regime (same seed, b's bits identical).

Timing uses C1's operands at seed 0. Per variant (served, r1g, r1g-noshift, b2, two-calls) it times one "layer",
which is the K/V assembly plus the attention and any merge, and also the attention alone. Eager: the median of
--iters synchronised calls per round, over --rounds rounds that interleave the variants. Trace: --trace-layers
layers captured in one trace, replayed --replays times, per layer.

Verdicts, on one PAIR_ROW_PROBE line:
  mechanism  M1             C1 row 0 equal, row 1 different; C2 follows the row; C3 equal; C4 different at 65
                            and the same bits whatever the masked contents; C5 equal; P equal
             SDPA-FACE      C3 differs (R1g still fixes the SDPA; C7 and C8 say what else)
             NOT-SDPA       C1 row 1 equals C0: the SDPA is exact, so take P2 (M+A) for the rest
             DATA-MOVEMENT  C5 differs
             UNEXPECTED     anything else (row 0 differs, the partner matters, C4 never differs ...)
  fix        R1G-EXACT      every C6 r1g row equals C0, in every seed and regime; otherwise R1G-DIFFERS
  head       EQUAL or DIFFERS (C7); rowlocal EQUAL or DIFFERS(ops) (C8)
  timing     served and r1g per layer, eager and trace, and the change

Failures (passed=False):
  - the mapped _ttnncpp.so is not --expect-binary-sha256;
  - the SDPA sources under TT_METAL_HOME are not the T16 admission's pinned ones (then the served kernel is not
    the one being run);
  - a non-finite C0;
  - the watchdog fired.

RUN with run_card_b.sh only. The helpers above Watchdog import no ttnn and are tested on CPU by
test_pair_row_probe.py, which also drives the whole device flow against a torch stand-in.
"""

import argparse
from contextlib import contextmanager
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROBE = 'P1'
CONTEXT, BLOCK, SPAN, PAIR_KEYS = 2048, 16, 2080, 4160
HEADS, KV_HEADS, HEAD_DIM = 16, 4, 128
HIDDEN, VOCAB_SHARD, INTERMEDIATE_SHARD = 5120, 124160, 8704
REGIMES = ('normal', 'peaked', 'negative', 'partner100')
LEADING = (1, 2, 8, 65)
CONTENTS = ('zeros', 'partner', 'partner100')
VARIANTS = ('served', 'r1g', 'r1g-noshift', 'b2', 'two-calls')
FIXES = VARIANTS[1:]
HEAD_DTYPES = ('bf16', 'bf8')
# The T16 admission's pins (dflash_t16_native_attention_gate.ORIGINAL); test_pair_row_probe keeps them equal. The
# draft SDPA's kernels are JIT-compiled from these sources, so a run on any other tree does not test the served
# kernel.
SDPA = 'ttnn/cpp/ttnn/operations/transformer/sdpa/'
PINNED_SOURCES = {
    SDPA + 'device/sdpa_program_factory.cpp': 'fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783',
    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h':
        '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181',
    SDPA + 'device/kernels/compute/compute_common.hpp': '3fb5da2440c3bf90ebceb8acd55424c7739339de4c6b02db83836e1e3414fa19',
    SDPA + 'device/kernels/compute/sdpa.cpp': 'a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0',
}
# Recorded, not required: the rest of the T16 admission's NATIVE_SOURCES.
RECORDED_SOURCES = tuple(SDPA + name for name in (
    'device/sdpa_device_operation.cpp', 'device/sdpa_device_operation_types.hpp',
    'device/kernels/compute/compute_streaming.hpp', 'device/kernels/dataflow/reader_interleaved.cpp',
    'device/kernels/dataflow/writer_interleaved.cpp'))
# The served modules the cases call; run_card_b.sh mounts this checkout's copies beside the harness.
MODULES = ('pair_row_exact', 'draft_attention', 'dflash_batched_mask', 'draft_shared_head')
OPEN_EXTRA_S = 600.0         # the first open JITs the firmware into the fresh kernel cache
ENV_RECORDED = ('TT_METAL_HOME', 'TT_METAL_CACHE', 'TT_METAL_WATCHER', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS')

clock = time.perf_counter    # module level so the CPU dry run can drive a model clock


# ---------------------------------------------------------------------------------------------
# Pure helpers (no ttnn).
# ---------------------------------------------------------------------------------------------

def build_fixture(torch, seed, regime):
    """Every host operand of one (seed, regime), bf16. The draws are in a fixed order and a regime only transforms
    them afterwards, so user b's bits are the same in every regime but peaked and negative, and user a's
    only differ in partner100 by the x100."""
    if regime not in REGIMES:
        raise ValueError('unknown regime %r' % (regime,))
    generator = torch.Generator().manual_seed(seed)
    draw = lambda *shape: torch.randn(*shape, generator=generator)
    fixture = dict(seed=seed, regime=regime)
    for user in 'ab':
        fixture[user] = dict(
            cache={name: draw(1, KV_HEADS, CONTEXT, HEAD_DIM) for name in 'kv'},
            live={name: draw(1, KV_HEADS, BLOCK, HEAD_DIM) for name in 'kv'},
            pad={name: draw(1, KV_HEADS, BLOCK, HEAD_DIM) for name in 'kv'},
            query=draw(1, HEADS, BLOCK, HEAD_DIM), query_pad=draw(1, HEADS, BLOCK, HEAD_DIM))
    for user in 'ab':
        entry = fixture[user]
        if regime == 'peaked':
            entry['query'] = entry['query'] * 4
            entry['query_pad'] = entry['query_pad'] * 4
        elif regime == 'negative':
            entry['query'] = entry['query'].abs()
            entry['cache']['k'][:, :, :32] = -entry['cache']['k'][:, :, :32].abs()
        elif regime == 'partner100' and user == 'a':
            for part in ('cache', 'live', 'pad'):
                entry[part] = {name: value * 100 for name, value in entry[part].items()}
    for user in 'ab':
        entry = fixture[user]
        for part in ('cache', 'live', 'pad'):
            entry[part] = {name: value.bfloat16() for name, value in entry[part].items()}
        entry['query'], entry['query_pad'] = entry['query'].bfloat16(), entry['query_pad'].bfloat16()
    return fixture


def single_operands(torch, fixture, user):
    """(query, keys, values) of `user`'s single-user trace: its rows then its pad rows; K = [cache | live | pad]."""
    entry = fixture[user]
    query = torch.cat([entry['query'], entry['query_pad']], dim=2)
    keys = {name: torch.cat([entry['cache'][name], entry['live'][name], entry['pad'][name]], dim=2) for name in 'kv'}
    return query, keys['k'], keys['v']


def pair_operands(torch, fixture, order):
    """The pair's host operands in `order` ((a, b) or (b, a)): the 32-row query, the 32-row live block and the two
    caches - what the device assembly (key_value_plan) starts from - and the host concatenation it must equal."""
    from dflash_batched_mask import key_value_plan

    first, second = (fixture[user] for user in order)
    query = torch.cat([first['query'], second['query']], dim=2)
    block = {name: torch.cat([first['live'][name], second['live'][name]], dim=2) for name in 'kv'}
    caches = [first['cache'], second['cache']]
    plan, spans, key_rows = key_value_plan([CONTEXT, CONTEXT], BLOCK)
    expected = {}
    for name in 'kv':
        pieces = []
        for part in plan:
            if part['kind'] == 'cached':
                pieces.append(caches[part['user']][name])
                continue
            start = part['source'].start if part['kind'] == 'live' else 0
            pieces.append(block[name][:, :, start:start + part['rows']])
        expected[name] = torch.cat(pieces, dim=2)
    if key_rows != PAIR_KEYS:
        raise AssertionError('the pair plan must cover 4160 keys')
    return dict(query=query, block=block, caches=caches, plan=plan, expected=expected)


def single_mask():
    """The served single-user mask, (1, 1, 32, 2080): pair_row_exact.fold_mask's."""
    from dflash_batched_mask import batched_attention_mask

    return batched_attention_mask([CONTEXT], BLOCK)


def packed_mask():
    from dflash_batched_mask import batched_attention_mask

    return batched_attention_mask([CONTEXT, CONTEXT], BLOCK)


def swapped_mask(torch):
    """The single mask with its row halves swapped: the live rule at rows 16-31, the pad rule at rows 0-15."""
    mask = single_mask()
    return torch.cat([mask[:, :, BLOCK:], mask[:, :, :BLOCK]], dim=2).contiguous()


def leading_mask(torch, chunks):
    """`chunks` fully masked 32-key chunks, then the single mask."""
    mask = single_mask()
    front = torch.full((1, 1, 32, 32 * chunks), float('-inf'), dtype=torch.bfloat16)
    return torch.cat([front, mask], dim=3).contiguous()


def leading_keys(torch, fixture, user, chunks, content):
    """C4's K and V: `chunks` x 32 masked keys, then `user`'s single-user K/V."""
    query, keys, values = single_operands(torch, fixture, user)
    partner = fixture['b' if user == 'a' else 'a']['cache']
    fronts = {}
    for name in 'kv':
        if content == 'zeros':
            fronts[name] = torch.zeros((1, KV_HEADS, 32 * chunks, HEAD_DIM), dtype=torch.bfloat16)
        elif content == 'partner':
            fronts[name] = partner[name][:, :, :32 * chunks].clone()
        elif content == 'partner100':
            fronts[name] = (partner[name][:, :, :32 * chunks].float() * 100).bfloat16()
        else:
            raise ValueError('unknown content %r' % (content,))
    # 65 chunks is 2080 keys: more than the partner's 2048-row cache holds, so its tail repeats from the start
    for name in 'kv':
        if fronts[name].shape[2] < 32 * chunks:
            repeat = (32 * chunks + fronts[name].shape[2] - 1) // fronts[name].shape[2]
            fronts[name] = fronts[name].repeat(1, 1, repeat, 1)[:, :, :32 * chunks].contiguous()
    return (query, torch.cat([fronts['k'], keys], dim=2).contiguous(),
            torch.cat([fronts['v'], values], dim=2).contiguous())


def noshift_mask(torch):
    """R1g without the row shift: (1, 32, 32, 2080), heads 8h+j (user a) the single mask, heads 8h+4+j (user b,
    left at rows 16-31) the row-swapped one."""
    single, swapped = single_mask(), swapped_mask(torch)
    heads = []
    for head in range(2 * HEADS):
        heads.append(single if (head % 8) < 4 else swapped)
    return torch.cat(heads, dim=1).contiguous()


def int16(torch, tensor):
    return tensor.contiguous().to(torch.bfloat16).view(torch.int16)


def digest(torch, tensor):
    return hashlib.sha256(int16(torch, tensor).numpy().tobytes()).hexdigest()


def compare(torch, actual, expected):
    """Bitwise, as bf16: equal, differing elements, total, max_abs (float32)."""
    actual, expected = actual.to(torch.bfloat16), expected.to(torch.bfloat16)
    if tuple(actual.shape) != tuple(expected.shape):
        return dict(equal=False, differing=None, total=None, max_abs=None,
                    shapes=[list(actual.shape), list(expected.shape)])
    differing = int((int16(torch, actual) != int16(torch, expected)).sum())
    difference = (actual.float() - expected.float()).abs()
    finite = torch.isfinite(difference)
    return dict(equal=differing == 0, differing=differing, total=int(actual.numel()),
                max_abs=float(difference[finite].max()) if bool(finite.any()) else None)


def rows(tensor, block):
    """User block `block` (0: rows 0-15, 1: rows 16-31) of a (..., 32, D) tensor."""
    return tensor[..., block * BLOCK:(block + 1) * BLOCK, :]


def summary(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return dict(median_us=statistics.median(ordered), min_us=ordered[0], mean_us=statistics.mean(ordered),
                stdev_us=statistics.stdev(ordered) if len(ordered) > 1 else 0.0, n=len(ordered))


def _all(entries, key='equal'):
    values = [entry.get(key) for entry in entries]
    return bool(values) and all(value is True for value in values)


def _count(entries, key='equal'):
    return '%d/%d' % (sum(1 for entry in entries if entry.get(key) is True), len(entries))


def decide(report):
    """The verdicts from report['cases'] (and report['timing']): see the module docstring."""
    cases = [case for case in report.get('cases', []) if not case.get('error')]
    errors = [case for case in report.get('cases', []) if case.get('error')]
    by = lambda name: [case for case in cases if case['case'] == name]
    c1_row0 = [case['rows'][0] for case in by('C1')]
    c1_row1 = [case['rows'][1] for case in by('C1')]
    c2_row0 = [case['rows'][0] for case in by('C2')]
    c2_row1 = [case['rows'][1] for case in by('C2')]
    c3 = [case['rows'][0] for case in by('C3')]
    c5 = [case['assembly'] for case in by('C5')]
    c4 = by('C4')
    leading = sorted(set(report.get('leading') or LEADING))
    c4_at = {n: [case['rows'][0] for case in c4 if case['chunks'] == n and case['content'] == 'zeros']
             for n in leading}
    groups = {}
    for case in c4:
        groups.setdefault((case['seed'], case['regime'], case['chunks']), set()).add(case['digest'])
    c4_content_free = bool(groups) and all(len(digests) == 1 for digests in groups.values())
    fixes = {variant: [row for case in by('C6') if case['variant'] == variant for row in case['rows']]
             for variant in FIXES}
    partner = by('P')
    verdict = dict(
        c1=dict(row0=_count(c1_row0), row1=_count(c1_row1),
                row1_differing=[row.get('differing') for row in c1_row1]),
        c2=dict(row0=_count(c2_row0), row1=_count(c2_row1)), c3=_count(c3), c5=_count(c5),
        c4={str(n): _count(entries) for n, entries in c4_at.items()}, c4_content_free=c4_content_free,
        c6={variant: _count(entries) for variant, entries in fixes.items()},
        partner=_count(partner), errors=len(errors))
    if not c1_row0 or not c5:
        mechanism = 'NO-DECISION'
    elif not _all(c5):
        mechanism = 'DATA-MOVEMENT'
    elif not _all(c1_row0):
        mechanism = 'UNEXPECTED'
    elif _all(c1_row1):
        mechanism = 'NOT-SDPA'
    elif c3 and not _all(c3):
        mechanism = 'SDPA-FACE'
    elif (leading and c4_at[leading[-1]] and not any(entry['equal'] for entry in c4_at[leading[-1]])
          and c4_content_free and (not partner or _all(partner)) and _all(c2_row0) and not _all(c2_row1)):
        mechanism = 'M1'
    else:
        mechanism = 'UNEXPECTED'
    verdict['mechanism'] = mechanism
    verdict['fix'] = 'R1G-EXACT' if _all(fixes['r1g']) else ('R1G-DIFFERS' if fixes['r1g'] else 'NO-DECISION')
    heads = by('C7')
    verdict['head'] = ('EQUAL' if heads and all(case['equal'] for case in heads)
                       else 'DIFFERS(%s)' % ','.join(case['dtype'] for case in heads if not case['equal'])
                       if heads else 'NOT-RUN')
    verdict['head_ties'] = sum(case.get('ties', 0) for case in heads)
    faces = by('C8')
    verdict['rowlocal'] = ('EQUAL' if faces and all(case['equal'] for case in faces)
                           else 'DIFFERS(%s)' % ','.join(case['op'] for case in faces if not case['equal'])
                           if faces else 'NOT-RUN')
    timing = {}
    for row in report.get('timing', []):
        if row.get('median_us') is not None:
            timing.setdefault(row['basis'], {})[(row['variant'], row['scope'])] = row['median_us']
    for basis, values in timing.items():
        served, fixed = values.get(('served', 'layer')), values.get(('r1g', 'layer'))
        if served and fixed:
            verdict['%s_served_us' % basis] = served
            verdict['%s_r1g_us' % basis] = fixed
            verdict['%s_change' % basis] = fixed / served - 1
    return verdict


def verdict_line(verdict):
    words = ['PAIR_ROW_PROBE', 'mechanism=%s' % verdict['mechanism'], 'fix=%s' % verdict['fix'],
             'head=%s' % verdict['head'], 'rowlocal=%s' % verdict['rowlocal'],
             'c1=row0:%s,row1:%s' % (verdict['c1']['row0'], verdict['c1']['row1']),
             'c2=row0:%s,row1:%s' % (verdict['c2']['row0'], verdict['c2']['row1']),
             'c3=%s' % verdict['c3'], 'c4=%s' % ','.join('%s:%s' % item for item in verdict['c4'].items()),
             'c4_content_free=%s' % verdict['c4_content_free'], 'c5=%s' % verdict['c5'],
             'partner=%s' % verdict['partner'],
             'c6=%s' % ','.join('%s:%s' % item for item in verdict['c6'].items())]
    for basis in ('eager', 'trace'):
        if '%s_served_us' % basis in verdict:
            words.append('%s=served:%.1fus,r1g:%.1fus(%+.1f%%)' % (
                basis, verdict['%s_served_us' % basis], verdict['%s_r1g_us' % basis],
                100 * verdict['%s_change' % basis]))
    if verdict.get('errors'):
        words.append('case_errors=%d' % verdict['errors'])
    return ' '.join(words)


def file_sha256(path):
    digest_ = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest_.update(block)
    return digest_.hexdigest()


def check_sources(root, report):
    """The SDPA sources the JIT builds the draft SDPA from: the pinned four must be the T16 admission's."""
    found = {}
    for name in (*PINNED_SOURCES, *RECORDED_SOURCES):
        path = Path(root) / name
        found[name] = file_sha256(path) if path.is_file() else None
    report['sdpa_sources'] = found
    wrong = [name for name, expected in PINNED_SOURCES.items() if found[name] != expected]
    for name in wrong:
        report['failures'].append('%s is %s, not the T16 admission\'s %s (the served draft SDPA is not the one run)'
                                  % (name, (found[name] or 'missing')[:16], PINNED_SOURCES[name][:16]))
    return not wrong


def loaded_binary(maps='/proc/self/maps'):
    """The _ttnncpp.so this process mapped."""
    paths = sorted({line.split()[-1] for line in Path(maps).read_text().splitlines()
                    if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    return paths[0]


def module_files():
    """The served modules this run imported, with their sha256 (the checkout's, mounted beside the harness)."""
    out = {}
    for name in MODULES:
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        out[name] = dict(path=path, sha256=file_sha256(path) if path and Path(path).is_file() else None)
    return out


# ---------------------------------------------------------------------------------------------
# Device harness.
# ---------------------------------------------------------------------------------------------

class Watchdog:
    """A per-device-call deadline: a hung NoC handshake cannot be interrupted from Python, so the poller prints
    WATCHDOG, writes the partial report and os._exit(3)s. A faulthandler backstop (a C thread) dumps every stack
    ('Timeout (') and exits 1 at the budget plus `grace` when a blocking call holds the GIL. A timed loop runs
    inside one span, so arming is not charged to each call."""

    def __init__(self, seconds, on_fire=None, grace=60.0, backstop=True, exit=os._exit):
        self.seconds, self.on_fire, self.grace, self.exit = seconds, on_fire, grace, exit
        self.backstop = bool(backstop and seconds)
        self.label, self.deadline, self.coarse = None, None, False
        self.lock = threading.Lock()

    def start(self):
        if self.seconds:
            threading.Thread(target=self.poll, name='pair-row-watchdog', daemon=True).start()
        return self

    def arm(self, seconds):
        if self.backstop:
            try:
                faulthandler.dump_traceback_later(seconds + self.grace, exit=True, file=sys.stdout)
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    def cancel(self):
        if self.backstop:
            try:
                faulthandler.cancel_dump_traceback_later()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    @contextmanager
    def span(self, label, seconds):
        if not self.seconds or self.coarse:
            yield
            return
        with self.op(label, extra=max(0.0, seconds - self.seconds)):
            self.coarse = True
            try:
                yield
            finally:
                self.coarse = False

    @contextmanager
    def op(self, label, extra=0.0):
        if not self.seconds or self.coarse:
            yield
            return
        budget = self.seconds + extra
        with self.lock:
            outer = (self.label, self.deadline)
            self.label, self.deadline = label, time.monotonic() + budget
        self.arm(budget)
        try:
            yield
        finally:
            with self.lock:
                self.label, self.deadline = outer
                remaining = None if outer[1] is None else max(outer[1] - time.monotonic(), 0.0)
            if remaining is None:
                self.cancel()
            else:
                self.arm(remaining)

    def check(self):
        with self.lock:
            label, deadline = self.label, self.deadline
        if label is None or time.monotonic() < deadline:
            return False
        sys.stdout.write('WATCHDOG: %r did not return within its budget (%ss); exiting 3 (docker rm -f, then reset '
                         'this card only, by the runner\'s printed reset command)\n' % (label, self.seconds))
        sys.stdout.flush()
        try:
            if self.on_fire is not None:
                self.on_fire(label)
        finally:
            self.exit(3)
        return True

    def poll(self):
        while not self.check():
            time.sleep(1.0)


WATCHDOG = Watchdog(0)


class Session:
    """One open device: uploads, read-backs and the served calls, each under the watchdog."""

    def __init__(self, ttnn, torch, device):
        self.ttnn, self.torch, self.device = ttnn, torch, device

    def upload(self, host, dtype=None, row_major=False):
        ttnn = self.ttnn
        with WATCHDOG.op('upload %s' % (tuple(host.shape),)):
            return ttnn.from_torch(host, dtype=dtype or ttnn.bfloat16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT,
                                   device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def host(self, tensor):
        with WATCHDOG.op('read back'):
            return self.ttnn.to_torch(tensor)

    def free(self, tensors):
        seen = set()
        for tensor in tensors:
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            try:
                self.ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001 - a view already freed with its buffer
                pass

    def retainer(self, owned, *inputs):
        """owned.append for a tensor a variant made. A view of an input (ttnn.reshape of the assembled K/V or the query
        shares its buffer) is never owned: free() would deallocate the input under every later variant and round."""
        kept = {tensor.buffer_address() for tensor in inputs}

        def retain(tensor):
            if tensor.buffer_address() not in kept:
                owned.append(tensor)
            return tensor

        return retain

    def sync(self):
        with WATCHDOG.op('synchronize'):
            self.ttnn.synchronize_device(self.device)

    def sdpa(self, query, key, value, mask):
        """The served single-call draft SDPA (draft_attention.draft_sdpa)."""
        from draft_attention import draft_sdpa

        with WATCHDOG.op('draft_sdpa %s' % (tuple(key.shape),)):
            return draft_sdpa(self.ttnn, query, key, value, mask)

    def folded(self, query, key, value, mask):
        """draft_sdpa's configuration on shapes draft_sdpa refuses (32 heads, batch 2): pair_row_exact.folded_sdpa."""
        from pair_row_exact import folded_sdpa

        with WATCHDOG.op('folded_sdpa %s' % (tuple(query.shape),)):
            return folded_sdpa(self.ttnn, query, key, value, mask)

    def stage(self, pair, owned):
        """The pair's device inputs to the K/V assembly: the two caches and the shared 32-row live block."""
        caches = [{name: self.upload(cache[name]) for name in 'kv'} for cache in pair['caches']]
        block = {name: self.upload(pair['block'][name]) for name in 'kv'}
        owned.extend([*caches[0].values(), *caches[1].values(), *block.values()])
        return dict(caches=caches, block=block)

    def assemble(self, pair, staged, owned):
        """The pair's K/V exactly as draft_attention_branch assembles them: each user's cache, its own 16 live rows
        sliced out of the shared block, then pad rows sliced from the block's rows 0-15, concatenated. Device work
        only: `staged` holds the uploaded caches and block (stage())."""
        ttnn = self.ttnn
        caches, block = staged['caches'], staged['block']
        heads = {}
        for name in 'kv':
            pieces = []
            for part in pair['plan']:
                if part['kind'] == 'cached':
                    pieces.append(caches[part['user']][name])
                    continue
                start = part['source'].start if part['kind'] == 'live' else 0
                piece = ttnn.slice(block[name], (0, 0, start, 0), (1, KV_HEADS, start + part['rows'], HEAD_DIM))
                owned.append(piece)
                pieces.append(piece)
            with WATCHDOG.op('assemble %s' % name):
                heads[name] = ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            owned.append(heads[name])
        return heads['k'], heads['v']

    def shifted(self, query, owned):
        """The query with its row halves swapped (user b's rows at 0-15)."""
        ttnn = self.ttnn
        upper = ttnn.slice(query, (0, 0, 0, 0), (1, HEADS, BLOCK, HEAD_DIM))
        lower = ttnn.slice(query, (0, 0, BLOCK, 0), (1, HEADS, 2 * BLOCK, HEAD_DIM))
        out = ttnn.concat([lower, upper], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.extend([upper, lower, out])
        return out

    def merge(self, first, second, owned, *, second_rows=0):
        """Rows 0-15 of `first` then rows second_rows..second_rows+16 of `second`, as (1, 16, 32, 128)."""
        ttnn = self.ttnn
        top = ttnn.slice(first, (0, 0, 0, 0), (1, HEADS, BLOCK, HEAD_DIM))
        bottom = ttnn.slice(second, (0, 0, second_rows, 0), (1, HEADS, second_rows + BLOCK, HEAD_DIM))
        out = ttnn.concat([top, bottom], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.extend([top, bottom, out])
        return out

    def attend(self, variant, query, key, value, masks, owned):
        """One variant's attention over the assembled pair K/V: the packed (1, 16, 32, 128) output."""
        ttnn = self.ttnn
        if variant == 'served':
            out = self.sdpa(query, key, value, masks['packed'])
            owned.append(out)
            return out
        if variant == 'r1g':
            from pair_row_exact import fold_attention

            with WATCHDOG.op('fold_attention'):
                return fold_attention(ttnn, query, key, value, masks['single'], self.retainer(owned, query, key, value),
                                      mask_validated=True)
        from pair_row_exact import fold_keys

        retain = self.retainer(owned, query, key, value)
        if variant == 'r1g-noshift':
            grouped = retain(ttnn.reshape(query, (KV_HEADS, 4, 32, HEAD_DIM)))
            doubled = retain(ttnn.concat([grouped, grouped], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            folded = retain(ttnn.reshape(doubled, (1, 2 * HEADS, 32, HEAD_DIM)))
            out = retain(self.folded(folded, fold_keys(ttnn, key, retain), fold_keys(ttnn, value, retain),
                                     masks['noshift']))
            regrouped = retain(ttnn.reshape(out, (KV_HEADS, 8, 32, HEAD_DIM)))
            top = retain(ttnn.slice(regrouped, (0, 0, 0, 0), (KV_HEADS, 4, BLOCK, HEAD_DIM)))
            bottom = retain(ttnn.slice(regrouped, (0, 4, BLOCK, 0), (KV_HEADS, 8, 2 * BLOCK, HEAD_DIM)))
            halves = [retain(ttnn.reshape(part, (1, HEADS, BLOCK, HEAD_DIM))) for part in (top, bottom)]
            return retain(ttnn.concat(halves, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        segments = {name: [retain(ttnn.slice(tensor, (0, 0, SPAN * user, 0), (1, KV_HEADS, SPAN * (user + 1), HEAD_DIM)))
                           for user in range(2)] for name, tensor in (('k', key), ('v', value))}
        shifted = self.shifted(query, owned)
        if variant == 'b2':
            queries = retain(ttnn.concat([query, shifted], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            keys = retain(ttnn.concat(segments['k'], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            values = retain(ttnn.concat(segments['v'], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            out = retain(self.folded(queries, keys, values, masks['batch2']))
            first = retain(ttnn.slice(out, (0, 0, 0, 0), (1, HEADS, 32, HEAD_DIM)))
            second = retain(ttnn.slice(out, (1, 0, 0, 0), (2, HEADS, 32, HEAD_DIM)))
            return self.merge(first, second, owned)
        if variant == 'two-calls':
            first = retain(self.sdpa(query, segments['k'][0], segments['v'][0], masks['single']))
            second = retain(self.sdpa(shifted, segments['k'][1], segments['v'][1], masks['single']))
            return self.merge(first, second, owned)
        raise ValueError('unknown variant %r' % (variant,))


def upload_masks(session, torch):
    return dict(single=session.upload(single_mask()), packed=session.upload(packed_mask()),
                swapped=session.upload(swapped_mask(torch)), noshift=session.upload(noshift_mask(torch)),
                batch2=session.upload(torch.cat([single_mask()] * 2, dim=0).contiguous()))


def record(report, entry):
    report['cases'].append(entry)
    rows_ = entry.get('rows') or []
    text = ' '.join('row%d=%s' % (index, 'equal' if row.get('equal') else 'DIFFERS(%s)' % row.get('differing'))
                    for index, row in enumerate(rows_))
    extra = ''.join(' %s=%s' % (key, entry[key]) for key in ('variant', 'chunks', 'content', 'op', 'dtype')
                    if key in entry)
    print('case %s seed=%s regime=%s%s %s%s' % (entry['case'], entry.get('seed'), entry.get('regime'), extra, text,
                                                 ' ERROR %s' % entry['error'] if entry.get('error') else ''),
          flush=True)


def guarded(report, entry, body):
    """Run one case; an exception is the case's result, never the run's end (except the watchdog's exit)."""
    try:
        body(entry)
    except Exception as error:  # noqa: BLE001
        entry['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
    record(report, entry)


def sdpa_cases(session, torch, fixture, masks, args, report, alone):
    """C0-C6 and P for one (seed, regime). `alone` collects C0's rows by (seed, regime, user)."""
    seed, regime = fixture['seed'], fixture['regime']
    key = lambda user: (seed, regime, user)

    def c0(user):
        def body(entry):
            owned = []
            try:
                query, keys, values = single_operands(torch, fixture, user)
                operands = [session.upload(value) for value in (query, keys, values)]
                owned.extend(operands)
                out = session.sdpa(*operands, masks['single'])
                owned.append(out)
                result = session.host(out)
                if not bool(torch.isfinite(result.float()).all()):
                    report['failures'].append('C0 seed=%d regime=%s user %s: a non-finite output' % (seed, regime, user))
                alone[key(user)] = rows(result, 0).clone()
                entry['digest'] = digest(torch, rows(result, 0))
            finally:
                session.free(owned)
        guarded(report, dict(case='C0', seed=seed, regime=regime, user=user), body)

    c0('a')
    c0('b')

    def pair_case(name, order):
        def body(entry):
            owned = []
            try:
                pair = pair_operands(torch, fixture, order)
                query = session.upload(pair['query'])
                owned.append(query)
                keys, values = session.assemble(pair, session.stage(pair, owned), owned)
                out = session.attend('served', query, keys, values, masks, owned)
                result = session.host(out)
                entry['rows'] = [compare(torch, rows(result, block), alone[key(user)])
                                 for block, user in enumerate(order)]
                entry['digests'] = [digest(torch, rows(result, block)) for block in range(2)]
                if name == 'C1':
                    assembled = [compare(torch, session.host(tensor), pair['expected'][part])
                                 for tensor, part in ((keys, 'k'), (values, 'v'))]
                    guarded(report, dict(case='C5', seed=seed, regime=regime,
                                         assembly=dict(equal=all(item['equal'] for item in assembled),
                                                       differing=sum(item['differing'] or 0 for item in assembled))),
                            lambda entry_: None)
                    for variant in FIXES:
                        guarded(report, dict(case='C6', seed=seed, regime=regime, variant=variant),
                                lambda entry_, variant=variant: fix_case(entry_, variant, query, keys, values))
            finally:
                session.free(owned)
        guarded(report, dict(case=name, seed=seed, regime=regime, order=''.join(order)), body)

    def fix_case(entry, variant, query, keys, values):
        owned = []
        try:
            out = session.attend(variant, query, keys, values, masks, owned)
            result = session.host(out)
            entry['rows'] = [compare(torch, rows(result, block), alone[key(user)]) for block, user in enumerate('ab')]
        finally:
            session.free(owned)

    pair_case('C1', ('a', 'b'))
    pair_case('C2', ('b', 'a'))

    def c3(user):
        def body(entry):
            owned = []
            try:
                query, keys, values = single_operands(torch, fixture, user)
                swapped = torch.cat([rows(query, 1), rows(query, 0)], dim=2).contiguous()
                operands = [session.upload(value) for value in (swapped, keys, values)]
                owned.extend(operands)
                out = session.sdpa(*operands, masks['swapped'])
                owned.append(out)
                result = session.host(out)
                # the user's rows now sit at rows 16-31
                entry['rows'] = [compare(torch, rows(result, 1), alone[key(user)])]
            finally:
                session.free(owned)
        guarded(report, dict(case='C3', seed=seed, regime=regime, user=user), body)

    c3('a')
    c3('b')

    if args.leading:
        for chunks in args.leading:
            for content in CONTENTS:
                def body(entry, chunks=chunks, content=content):
                    owned = []
                    try:
                        operands = [session.upload(value) for value in leading_keys(torch, fixture, 'b', chunks, content)]
                        mask = session.upload(leading_mask(torch, chunks))
                        owned.extend([*operands, mask])
                        out = session.sdpa(*operands, mask)
                        owned.append(out)
                        result = session.host(out)
                        entry['rows'] = [compare(torch, rows(result, 0), alone[key('b')])]
                        entry['digest'] = digest(torch, rows(result, 0))
                    finally:
                        session.free(owned)
                guarded(report, dict(case='C4', seed=seed, regime=regime, chunks=chunks, content=content), body)


def partner_cases(report):
    """P: C1's row 1 (user b) in the partner100 regime against the normal one, seed by seed."""
    c1 = {(case['seed'], case['regime']): case for case in report['cases']
          if case['case'] == 'C1' and not case.get('error') and case.get('digests')}
    for (seed, regime), case in sorted(c1.items()):
        if regime != 'partner100' or (seed, 'normal') not in c1:
            continue
        equal = case['digests'][1] == c1[(seed, 'normal')]['digests'][1]
        record(report, dict(case='P', seed=seed, regime=regime, equal=equal,
                            rows=[dict(equal=equal, differing=None if equal else 'some')]))


def head_cases(session, torch, args, report):
    """C7: the head's linear and chunked top-16, the same 16 rows alone (16-row input), at rows 0-15 and at rows
    16-31 of a 32-row input."""
    from draft_shared_head import candidate_chunks, local_head_candidates

    ttnn = session.ttnn
    generator = torch.Generator().manual_seed(args.head_seed)
    ours = torch.randn(1, 1, BLOCK, HIDDEN, generator=generator).bfloat16()
    theirs = torch.randn(1, 1, BLOCK, HIDDEN, generator=generator).bfloat16()
    weight = torch.empty(HIDDEN, VOCAB_SHARD, dtype=torch.bfloat16)
    for start in range(0, HIDDEN, 512):
        weight[start:start + 512] = (torch.randn(min(512, HIDDEN - start), VOCAB_SHARD, generator=generator)
                                     * HIDDEN ** -0.5).bfloat16()
    for name in args.head_dtypes:
        def body(entry, name=name):
            owned = []
            try:
                dtype = ttnn.bfloat16 if name == 'bf16' else ttnn.bfloat8_b
                with WATCHDOG.op('upload head weight %s' % name, extra=OPEN_EXTRA_S):
                    shard = ttnn.from_torch(weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=session.device,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                owned.append(shard)
                top = session.upload(torch.cat([ours, theirs], dim=2).contiguous())
                bottom = session.upload(torch.cat([theirs, ours], dim=2).contiguous())
                alone = ttnn.slice(top, (0, 0, 0, 0), (1, 1, BLOCK, HIDDEN))
                owned.extend([top, bottom, alone])
                results = {}
                for layout, tensor, block in (('alone', alone, 0), ('row0', top, 0), ('row1', bottom, 1)):
                    with WATCHDOG.op('head %s %s' % (name, layout)):
                        logits = ttnn.linear(tensor, shard)
                        owned.append(logits)
                        chunks = local_head_candidates(ttnn, logits, owned)
                        host_logits = session.host(logits)
                    results[layout] = dict(
                        logits=rows(host_logits, block),
                        values=[rows(session.host(chunk['values']), block) for chunk in chunks],
                        indices=[rows(session.host(chunk['indices']).to(torch.int32), block) for chunk in chunks])
                reference = results['alone']
                verdicts = {}
                for layout in ('row0', 'row1'):
                    got = results[layout]
                    verdicts[layout] = dict(
                        logits=compare(torch, got['logits'], reference['logits']),
                        values=all(compare(torch, mine, theirs_)['equal']
                                   for mine, theirs_ in zip(got['values'], reference['values'])),
                        indices=all(torch.equal(mine, theirs_) for mine, theirs_ in zip(got['indices'], reference['indices'])))
                entry['layouts'] = {layout: dict(logits_equal=result['logits']['equal'],
                                                 logits_differing=result['logits']['differing'],
                                                 values_equal=result['values'], indices_equal=result['indices'])
                                    for layout, result in verdicts.items()}
                entry['equal'] = all(result['logits']['equal'] and result['values'] and result['indices']
                                     for result in verdicts.values())
                logits = reference['logits'].float().reshape(BLOCK, VOCAB_SHARD)
                ties = 0
                for start, stop in candidate_chunks():
                    ordered = logits[:, start:stop].sort(dim=-1, descending=True).values
                    ties += int((ordered[:, 15] == ordered[:, 16]).sum())
                entry['ties'] = ties
                entry['rows'] = [dict(equal=verdicts['row0']['logits']['equal']),
                                 dict(equal=verdicts['row1']['logits']['equal'])]
            finally:
                session.free(owned)
        guarded(report, dict(case='C7', dtype=name, equal=False), body)


def face_cases(session, torch, args, report):
    """C8: every row-local draft op with its served configuration, the same 16 rows at rows 0-15 and at 16-31."""
    ttnn = session.ttnn
    generator = torch.Generator().manual_seed(args.face_seed)
    draw = lambda *shape: torch.randn(*shape, generator=generator)
    kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                              fp32_dest_acc_en=True, packer_l1_acc=False)
    projection = ttnn.bfloat8_b if args.projection_dtype == 'bf8' else ttnn.bfloat16

    def program(grid, columns):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid, in0_block_w=4,
            out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=columns, fuse_batch=True,
            fused_activation=None, mcast_in0=True)

    def matmul(width, columns, grid, per_core, dtype):
        weight = (draw(width, columns) * width ** -0.5).bfloat16()

        def run(inputs, owned):
            shard = ttnn.from_torch(weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=session.device,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
            owned.append(shard)
            return ttnn.matmul(inputs[0], shard, dtype=ttnn.float32, compute_kernel_config=kernel,
                               program_config=program(grid, per_core), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return [((1, 1, 32, width), 'bf16')], run

    def norm(shape, weight_shape):
        weight = (1 + 0.1 * draw(*weight_shape)).bfloat16()

        def run(inputs, owned):
            device_weight = session.upload(weight, row_major=True)
            owned.append(device_weight)
            return ttnn.rms_norm(inputs[0], epsilon=1e-6, weight=device_weight, compute_kernel_config=kernel,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return [(shape, 'bf16')], run

    ops = {
        'rms_norm-hidden': norm((1, 1, 32, HIDDEN), (1, 1, 160, 32)),
        'matmul-conv': matmul(HIDDEN, 1280, (8, 5), 1, ttnn.bfloat16),
        'matmul-q': matmul(HIDDEN, 2048, (8, 8), 1, projection),
        'matmul-kv': matmul(HIDDEN, 512, (8, 8), 1, projection),
        'matmul-o': matmul(2048, HIDDEN, (8, 10), 2, projection),
        'matmul-gate': matmul(HIDDEN, INTERMEDIATE_SHARD, (8, 10), 4, projection),
        'matmul-down': matmul(INTERMEDIATE_SHARD, HIDDEN, (8, 10), 2, projection),
        'matmul-selector': matmul(HIDDEN, 256, (8, 1), 1, ttnn.bfloat16),
        'typecast': ([((1, 1, 32, HIDDEN), 'fp32')], lambda inputs, owned: ttnn.typecast(inputs[0], ttnn.bfloat16)),
        'rotary': ([((1, HEADS, 32, HEAD_DIM), 'fp32'), ((1, 1, 32, HEAD_DIM), 'fp32'), ((1, 1, 32, HEAD_DIM), 'fp32')],
                   lambda inputs, owned: ttnn.experimental.rotary_embedding_hf(*inputs, is_decode_mode=False,
                       compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG)),
        'rms_norm-head': norm((1, HEADS, 32, HEAD_DIM), (1, 1, 4, 32)),
        'silu': ([((1, 1, 32, INTERMEDIATE_SHARD), 'fp32')],
                 lambda inputs, owned: ttnn.silu(inputs[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)),
        'multiply': ([((1, 1, 32, INTERMEDIATE_SHARD), 'fp32')] * 2,
                     lambda inputs, owned: ttnn.multiply(inputs[0], inputs[1], dtype=ttnn.float32)),
        'add': ([((1, 1, 32, HIDDEN), 'fp32')] * 2,
                lambda inputs, owned: ttnn.add(inputs[0], inputs[1], dtype=ttnn.float32)),
    }
    for name in args.face_ops or list(ops):
        shapes, run = ops[name]

        def body(entry, shapes=shapes, run=run):
            owned = []
            try:
                blocks = []
                for shape, kind in shapes:
                    full = draw(*shape)
                    blocks.append((full, kind))
                outputs = []
                for placement in (0, 1):
                    inputs = []
                    for full, kind in blocks:
                        ours, theirs = rows(full, 0), rows(full, 1)
                        value = torch.cat([ours, theirs] if placement == 0 else [theirs, ours], dim=2).contiguous()
                        tensor = session.upload(value if kind == 'fp32' else value.bfloat16(),
                                                dtype=ttnn.float32 if kind == 'fp32' else None)
                        owned.append(tensor)
                        inputs.append(tensor)
                    with WATCHDOG.op('face %s' % entry['op']):
                        out = run(inputs, owned)
                    owned.append(out)
                    outputs.append(rows(session.host(out), placement))
                result = dict(compare(torch, outputs[1].float(), outputs[0].float()))
                if outputs[0].dtype == torch.float32:
                    left, right = outputs[0].contiguous().view(torch.int32), outputs[1].contiguous().view(torch.int32)
                    result.update(equal=bool(torch.equal(left, right)), differing=int((left != right).sum()))
                entry['rows'] = [result]
                entry['equal'] = result['equal']
            finally:
                session.free(owned)
        guarded(report, dict(case='C8', op=name, equal=False), body)


def timing(session, torch, args, report):
    """Per variant, eager and traced: one layer (assembly + attention + merge) and the attention alone."""
    ttnn = session.ttnn
    fixture = build_fixture(torch, 0, 'normal')
    pair = pair_operands(torch, fixture, ('a', 'b'))
    masks = upload_masks(session, torch)
    query = session.upload(pair['query'])
    base = []
    staged = session.stage(pair, base)
    fixed_keys, fixed_values = session.assemble(pair, staged, base)

    def layer(variant, owned):
        keys, values = session.assemble(pair, staged, owned)
        return session.attend(variant, query, keys, values, masks, owned)

    def attention(variant, owned):
        return session.attend(variant, query, fixed_keys, fixed_values, masks, owned)

    scopes = dict(layer=layer, attention=attention)
    variants = list(args.variants)
    try:
        for scope, run in scopes.items():
            samples = {variant: [] for variant in variants}
            for variant in variants:        # warm: every program compiled before anything is timed or captured
                owned = []
                run(variant, owned)
                session.sync()
                session.free(owned)
            for index in range(args.rounds):
                turn = index % len(variants)
                for variant in variants[turn:] + variants[:turn]:
                    with WATCHDOG.span('eager %s %s round %d' % (scope, variant, index), args.watchdog):
                        for _ in range(args.iters):
                            owned = []
                            started = clock()
                            run(variant, owned)
                            ttnn.synchronize_device(session.device)
                            samples[variant].append((clock() - started) * 1e6)
                            session.free(owned)
            for variant in variants:
                row = dict(basis='eager', scope=scope, variant=variant, **(summary(samples[variant]) or {}))
                report['timing'].append(row)
                print('timing eager %-9s %-12s %.1f us' % (scope, variant, row.get('median_us', float('nan'))), flush=True)
            if not args.replays:
                continue
            for variant in variants:
                owned = []
                with WATCHDOG.op('capture %s %s' % (scope, variant)):
                    trace = ttnn.begin_trace_capture(session.device, cq_id=0)
                    try:
                        for _ in range(args.trace_layers):
                            run(variant, owned)
                    finally:
                        ttnn.end_trace_capture(session.device, trace, cq_id=0)
                replays = []
                try:
                    with WATCHDOG.span('replay %s %s' % (scope, variant), args.watchdog):
                        for index in range(args.trace_warmup + args.replays):
                            started = clock()
                            ttnn.execute_trace(session.device, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(session.device)
                            if index >= args.trace_warmup:
                                replays.append((clock() - started) * 1e6 / args.trace_layers)
                finally:
                    with WATCHDOG.op('release trace'):
                        ttnn.release_trace(session.device, trace)
                    session.free(owned)
                row = dict(basis='trace', scope=scope, variant=variant, **(summary(replays) or {}))
                report['timing'].append(row)
                print('timing trace %-9s %-12s %.1f us/layer' % (scope, variant, row.get('median_us', float('nan'))),
                      flush=True)
    finally:
        session.free([query, *masks.values(), *base])


def check_binary(args, report):
    path = loaded_binary()
    sha = file_sha256(path)
    report['binary'] = dict(path=path, sha256=sha, expected_sha256=args.expect_binary_sha256 or None)
    print('binary %s sha256 %s' % (path, sha[:16]), flush=True)
    if args.expect_binary_sha256 and sha != args.expect_binary_sha256:
        report['failures'].append('the loaded _ttnncpp.so is %s, not the expected %s (read the launched argv)'
                                  % (sha[:16], args.expect_binary_sha256[:16]))
        return False
    return True


def run(args, report, ttnn=None, torch=None):
    if torch is None:
        import torch
    if ttnn is None:
        import ttnn
    options = dict(device_id=args.device_id, l1_small_size=24576)
    if args.replays and not args.no_timing:
        options['trace_region_size'] = args.trace_region_bytes
    with WATCHDOG.op('open device', extra=OPEN_EXTRA_S):
        device = ttnn.open_device(**options)
    try:
        try:
            device.enable_program_cache()
            report['program_cache_enabled_call'] = True
        except Exception as error:  # noqa: BLE001 - default-on in newer runtimes
            report['program_cache_enabled_call'] = repr(error)[:200]
        if not args.skip_binary_check and not check_binary(args, report):
            return
        if args.tt_metal_home and not check_sources(args.tt_metal_home, report):
            return
        import pair_row_exact  # noqa: F401 - the mounted modules, recorded before any case runs
        import draft_attention  # noqa: F401
        import dflash_batched_mask  # noqa: F401
        import draft_shared_head  # noqa: F401
        report['modules'] = module_files()
        session = Session(ttnn, torch, device)
        masks = upload_masks(session, torch)
        alone = {}
        try:
            for seed in args.seeds:
                for regime in args.regimes:
                    fixture = build_fixture(torch, seed, regime)
                    sdpa_cases(session, torch, fixture, masks, args, report, alone)
            partner_cases(report)
        finally:
            session.free(list(masks.values()))
        if args.head_dtypes:
            head_cases(session, torch, args, report)
        if not args.no_faces:
            face_cases(session, torch, args, report)
        if not args.no_timing:
            timing(session, torch, args, report)
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--regimes', default=','.join(REGIMES))
    parser.add_argument('--leading', default=','.join(map(str, LEADING)),
                        help='C4 fully masked leading chunk counts ("" skips C4)')
    parser.add_argument('--head-dtypes', default=','.join(HEAD_DTYPES), help='C7 weight dtypes ("" skips C7)')
    parser.add_argument('--head-seed', type=int, default=0)
    parser.add_argument('--no-faces', action='store_true', help='skip C8')
    parser.add_argument('--face-ops', default='', help='C8 ops to run (default: all)')
    parser.add_argument('--face-seed', type=int, default=0)
    parser.add_argument('--projection-dtype', choices=('bf8', 'bf16'), default='bf8',
                        help='the draft projections\' dtype in C8 (bf8: QWEN_FAST_DRAFT_BF8=1, as every current arm)')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--iters', type=int, default=20, help='synchronised eager calls per variant per round')
    parser.add_argument('--rounds', type=int, default=5, help='eager rounds, interleaving the variants')
    parser.add_argument('--trace-layers', type=int, default=5, help='layers per captured trace (the draft\'s five)')
    parser.add_argument('--trace-warmup', type=int, default=2)
    parser.add_argument('--replays', type=int, default=50, help='timed trace replays per variant (0: no trace)')
    parser.add_argument('--trace-region-bytes', type=int, default=64 << 20)
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--watchdog', type=float, default=0, help='seconds per device call before os._exit(3); 0 off')
    parser.add_argument('--expect-binary-sha256', default='')
    parser.add_argument('--skip-binary-check', action='store_true', help='the CPU dry run only')
    parser.add_argument('--tt-metal-home', default=os.environ.get('TT_METAL_HOME', ''),
                        help='where the SDPA sources are pinned ("" skips the check)')
    args = parser.parse_args(argv)
    split = lambda text: [value for value in text.split(',') if value]
    try:
        args.seeds = [int(value) for value in split(args.seeds)]
        args.leading = [int(value) for value in split(args.leading)]
    except ValueError as error:
        parser.error(str(error))
    args.regimes, args.head_dtypes = split(args.regimes), split(args.head_dtypes)
    args.face_ops, args.variants = split(args.face_ops), split(args.variants)
    if not args.seeds or any(regime not in REGIMES for regime in args.regimes) or not args.regimes:
        parser.error('--regimes from %s, and at least one seed' % ', '.join(REGIMES))
    if any(chunks < 1 or chunks > 65 for chunks in args.leading):
        parser.error('--leading counts are 1..65')
    if any(name not in HEAD_DTYPES for name in args.head_dtypes):
        parser.error('--head-dtypes from %s' % ', '.join(HEAD_DTYPES))
    if any(name not in VARIANTS for name in args.variants) or 'served' not in args.variants or 'r1g' not in args.variants:
        parser.error('--variants from %s, with served and r1g' % ', '.join(VARIANTS))
    if min(args.iters, args.rounds, args.trace_layers) < 1 or min(args.trace_warmup, args.replays) < 0:
        parser.error('--iters, --rounds and --trace-layers must be >= 1; --trace-warmup and --replays >= 0')
    if args.expect_binary_sha256 and (len(args.expect_binary_sha256) != 64
                                      or any(c not in '0123456789abcdef' for c in args.expect_binary_sha256)):
        parser.error('--expect-binary-sha256 must be a full lowercase sha256')
    return args


def main(argv=None, ttnn=None, torch=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(probe=PROBE, design='pair row exact diagnosis section 3 (P1)', passed=False,
                  argv=list(sys.argv[1:] if argv is None else argv), seeds=args.seeds, regimes=args.regimes,
                  leading=args.leading, head_dtypes=args.head_dtypes, variants=args.variants,
                  env={name: os.environ.get(name) for name in ENV_RECORDED}, watchdog=args.watchdog,
                  failures=[], warnings=[], cases=[], timing=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = dict(report)
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def on_fire(label):
        try:
            write_report(dict(error='watchdog: %r exceeded its budget' % (label,), passed=False))
        except Exception:  # noqa: BLE001 - the main thread may be mid-update; the WATCHDOG line stands
            pass

    WATCHDOG = Watchdog(args.watchdog, on_fire=on_fire).start()
    try:
        try:
            run(args, report, ttnn=ttnn, torch=torch)
        except Exception as error:  # noqa: BLE001
            report['error'] = '%s: %s' % (type(error).__name__, error)
        try:
            report['verdict'] = decide(report)
            report['verdict_line'] = verdict_line(report['verdict'])
        except Exception as error:  # noqa: BLE001 - the measurements are still written
            report['error'] = report.get('error') or 'analysis: %s: %s' % (type(error).__name__, error)
        report['passed'] = not report['failures'] and not report.get('error') and bool(report['cases'])
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    for warning in report['warnings']:
        print('WARN', warning)
    if report.get('error'):
        print('ERROR', report['error'])
    print(report.get('verdict_line', 'PAIR_ROW_PROBE mechanism=NO-DECISION'), flush=True)
    print('PAIR_ROW_PROBE_DONE passed=%s cases=%d timing_rows=%d failures=%d warnings=%d report=%s' % (
        report['passed'], len(report['cases']), len(report['timing']), len(report['failures']),
        len(report['warnings']), args.out), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
