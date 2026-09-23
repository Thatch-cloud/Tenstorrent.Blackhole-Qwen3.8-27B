"""M1 of the prefill ranking: is the chunked-SDPA quadratic term bytes-bound or compute-bound?

The served C1 prefill spends ~22.7 s of each 131,072-token prompt in the chunked SDPA quadratic
term (16 full-attention layers, 64 chunks). Lever #1 (K/V prefix sharing in the chunked SDPA
reader) is worth 7-14 s only if that term is limited by the bytes each core reads; if it is
limited by compute, #1 is worth ~0 and the compute knobs (#1b) are the lever instead. This bench
answers that on ONE card with no model: it times the served call alone, then moves one knob at a
time and fits the slope of the time against the prefix length (chunk_start).

THE SERVED CALL (grafted attention/tp.py, forward_prefill_paged, run 35816715775):
    ttnn.transformer.chunked_scaled_dot_product_attention(
        input_tensor_q=q8, input_tensor_k=k_paged, input_tensor_v=v_paged,
        page_table_tensor=sdpa_page_table, chunk_start_idx_tensor=chunk_start_idx_tensor,
        compute_kernel_config=tpc.COMPUTE_HIFI2, program_config=SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),   # 11 x 10
            exp_approx_mode=False, q_chunk_size=128, k_chunk_size=128))
  - Q (1, 12, 2048, 256): 12 local heads at TP2, typecast to bf8 under QWEN_SDPA_BF8=1 (the arm
    hard-sets it), so the served Q and paged K/V are all bf8.
  - K/V pools (blocks, 2, 64, 256): 2 local KV heads, 64-token pages (GQA 6:1).
  - The page table is padded to a multiple of 32 blocks (extra blocks masked by causality).
  - COMPUTE_HIFI2 = HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True.
  - The FLEXIBLE path: chunk_start comes from a device int32 tensor, so one program serves every
    chunk_start (q/k chunk fixed at 128).

ARMS (one knob each against the baseline) and how to read them (ranking, section 3, M1):
  baseline     q128/k128, bf8 Q and KV                        the served call
  bf16_kv      bf16 K/V (Q bf8)                               bytes x 1.88, compute unchanged
  bf16_qkv     bf16 Q and K/V (the non-bf8 serving mode)      fallback if the op wants one dtype
  exp_approx   exp_approx_mode=True                           compute only
  fp32_off     fp32_dest_acc_en=False                         compute only
  q256_2048    q256/k128 at 2048 rows   pre-registered: no gain or a regression (48 busy cores:
                                        causal chunked SDPA hands out Q chunks in PAIRS)
  q256_4096    q256/k128 at 4096 rows   96 busy cores again, and half the K/V bytes per token
  - bytes-bound:   the bf16/bf8 slope ratio follows 1.88 and q256_4096 roughly halves the
                   per-token slope -> build lever #1.
  - compute-bound: exp_approx (or fp32_off) moves the slope and the bytes barely matter ->
                   #1 is worth ~0; take #1b through a token gate.
The slope is ms per 1k keys of prefix, per call; for the 4096-row arm it is also given per 2048
query rows (per token), which is the figure the reading rules compare.

Timing: every (arm, chunk_start) pair is warmed up (program compile), then the pairs are timed
in interleaved rounds (arm-major order rotated per round) so card load drifts spread over every
arm alike; each sample is one call bracketed by ttnn.synchronize_device. The median per pair is
reported. Single device, no CCL.

Every ttnn/torch import is local to the device functions: the helpers import and unit-test under
plain CPython (test_sdpa_prefill_bench.py).

K0 additions (sdpa-prefill-share-spec.md 3.7 and 7.1; all off by default, so a plain run is the
bench as it was):
  --sha           after timing, every (arm, start) is called twice more and the sha256 of the int16
                  view of ttnn.to_torch(out) is recorded (report 'sha256', 'sha_stable') and printed
                  as 'M1 SHA <arm>@<start> <hex> stable=0|1'. K0c must equal stock; K0a/b must not.
  --watchdog-s N  per device call: a call not back within N s prints 'WATCHDOG: ...', writes the
                  partial report and exits 3 (a faulthandler backstop 60 s later covers a call that
                  holds the GIL: 'Timeout (...)!' and exit 1). The runner then resets card M.
                  Budgets on top of N: each arm's FIRST warmup +780 s (it JIT-compiles the
                  program's kernels on a fresh cache, under rig CI load: a false alarm costs a
                  session and a card reset), later warmups, open_device (firmware build on a fresh
                  cache) and build_inputs (host bf8 tilize of two K/V pools) +240 s.
  --coords-out P  worker_core_from_logical_core for every grid core, as JSON (the fixture
                  fixtures/cardm_worker_coords.json, captured in the K0 session).
  --kernel-elf K  after the run, a digest of kernel K's compiled ELFs in $TT_METAL_CACHE
                  (.../kernels/K/<hash>/<risc>/*.elf; sha256 over the PT_LOAD segments, i.e. what
                  is loaded on the core, not the debug info), printed as
                  'M1 KERNEL_ELF K <hex|none> files=N'. K0c's output equals stock by design, so its
                  proof that the mounted reader was compiled is its reader ELF differing from stock's
                  (with stock's reproducible run to run: the session's closing stock2).

Prefill-chain additions (sdpa-prefill-share-spec.md 3.7 / 6.2-6.3; optimisation/ttnn-op/sdpa_prefill_chain;
all off by default, and the chain arms need the K64g graft: run_m1.sh KOPGRAFT_PF=~/opgraft-K64g):
  arms chain / chain_b / chain_o / chain_bo
                  the baseline call with SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0001 /
                  0x5EFA0003 / 0x5EFA0005 / 0x5EFA0007 (the G6 K/V chain; 0x2 injector read cadence,
                  0x4 NoC-cost chain order). Never in the default --arms.
  --program-word HEX[,HEX]
                  one extra arm per word, named word_0x<hex>: the baseline call with that word (test
                  flags 0x100 / 0x200 need QWEN_SDPA_PF_TEST=1 in the container; refusal checks).
  --q-memory {dram,l1}
                  where Q lives (the model's Q is L1-interleaved: the Q2 acceptance placement).
  --page-blocks N the page-table width in blocks (the model's 2080 at 131k), >= what the starts need,
                  a multiple of 32; the K/V pool has the same number of blocks.
  --verify-log    fds 1/2 go to <out>.native.log for the run; the factory's
                  '[QWEN-SDPA-PF] flags=' lines are parsed (report 'pf_log') and printed as
                  'M1 PF_LOG ...': exactly one per chain arm's flags (one program per shape, page
                  width and flags), chains=16 members=96 at 2048 rows, none for a flags word no arm used.
  --k0b32-slope S the K0b32 slope (ms per 1k keys) Q2's rule (a) compares against (default: the K0
                  session's 0.2116). With the baseline and a chain arm timed, 'M1 Q2: ...' reports the
                  best chain arm, per-step times (slope x 64 us) and rules (a) best <= 1.10 x K0b32,
                  (b) best <= 0.85 x baseline, (c) intercept <= baseline + 0.2 ms, (d) every chain arm's
                  output equals the baseline's at every start (needs --sha; without it Q2 FAILs).
"""

import argparse
import contextlib
import faulthandler
import hashlib
import json
import math
import os
import re
import statistics
import struct
import sys
import threading
import time
from pathlib import Path

NH = 12            # local Q heads at TP2
NKV = 2            # local KV heads at TP2
HD = 256
BLOCK = 64         # page size
ROWS = 2048        # prefill chunk
GRID = (11, 10)    # Blackhole p150 compute_with_storage_grid_size
STARTS = (0, 32768, 65536, 129024)
COMPILE_GRACE_S = 240.0         # extra watchdog budget: later warmups, open_device, build_inputs
FIRST_COMPILE_GRACE_S = 780.0   # each arm's first warmup: the JIT compile of its kernels (900 s at --watchdog-s 120)
BF8_BYTES_PER_ELEMENT = 1088 / 1024   # bfloat8_b tile: 1024 mantissa bytes + 64 exponent bytes
BF16_BYTES_PER_ELEMENT = 2.0
EXPECTED_BF16_RATIO = BF16_BYTES_PER_ELEMENT / BF8_BYTES_PER_ELEMENT   # 1.882

# Reading-rule thresholds (the ranking gives the directions; these are the cut points).
BYTES_BF16_RATIO_MIN = 1.5      # bf16/bf8 slope ratio at or above this: bytes move the slope
BYTES_Q256_4096_MAX = 0.65      # per-token slope ratio at or below this: halving bytes/token pays
COMPUTE_BF16_RATIO_MAX = 1.15   # bf16/bf8 ratio at or below this: bytes barely matter
COMPUTE_KNOB_MAX = 0.90         # exp_approx or fp32_off at or below this: compute moves the slope

ARMS = (
    dict(name='baseline', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=False, fp32_dest=True),
    dict(name='bf16_kv', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8', kv_dtype='bf16',
         exp_approx=False, fp32_dest=True),
    dict(name='bf16_qkv', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf16', kv_dtype='bf16',
         exp_approx=False, fp32_dest=True),
    dict(name='exp_approx', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=True, fp32_dest=True),
    dict(name='fp32_off', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=False, fp32_dest=False),
    dict(name='q256_2048', rows=2048, q_chunk=256, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=False, fp32_dest=True),
    dict(name='q256_4096', rows=4096, q_chunk=256, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=False, fp32_dest=True),
)
ARM_NAMES = tuple(arm['name'] for arm in ARMS)

# Prefill lever #1: the baseline call plus the G6 K/V chain word (sdpa_prefill_chain/apply_factory_pf.py).
PF_TAG = 0x5EFA0000
PF_TEST_FLAGS = 0x300
CHAIN_ARMS = (
    dict(ARMS[0], name='chain', program_word=PF_TAG | 0x1),
    dict(ARMS[0], name='chain_b', program_word=PF_TAG | 0x3),
    dict(ARMS[0], name='chain_o', program_word=PF_TAG | 0x5),
    dict(ARMS[0], name='chain_bo', program_word=PF_TAG | 0x7),
)
CHAIN_ARM_NAMES = tuple(arm['name'] for arm in CHAIN_ARMS)
K0B32_SLOPE = 0.2116          # ms per 1k keys: K0b32, card M, image A' 1b9b6445 (Q2 rule (a))
STEP_US_PER_SLOPE = 64.0      # per-step time (us) = slope (ms / 1k keys) x 0.128 k / 2 steps x 1000
Q2_FORWARD_MAX = 1.10
Q2_BASELINE_MAX = 0.85
Q2_INTERCEPT_MAX_MS = 0.2
Q2_PREFER_MARGIN = 0.02       # a flag set beyond 0x1 wins only with a >= 2% lower slope


def word_arm(word):
    """The baseline call with max_cores_per_head_batch = word (--program-word)."""
    if not 0 <= word < (1 << 32):
        raise ValueError('program word %r is not a 32-bit value' % (word,))
    return dict(ARMS[0], name='word_%#x' % word, program_word=word)


def arm_by_name(name):
    for arm in ARMS + CHAIN_ARMS:
        if arm['name'] == name:
            return dict(arm)
    if name.startswith('word_0x'):
        return word_arm(int(name[len('word_'):], 16))
    raise ValueError('unknown arm %r (known: %s)' % (name, ', '.join(ARM_NAMES + CHAIN_ARM_NAMES)))


def validate(arm, starts, block=BLOCK):
    """The factory's preconditions this bench relies on: chunk_start a multiple of q_chunk (the
    flexible path's contract) and of the page size, rows a whole number of q chunks."""
    if arm['rows'] % arm['q_chunk'] or arm['rows'] % 32:
        raise ValueError('%s: rows %d not a multiple of q_chunk %d' % (arm['name'], arm['rows'], arm['q_chunk']))
    for start in starts:
        if type(start) is not int or start < 0 or start % arm['q_chunk'] or start % block:
            raise ValueError('%s: chunk_start %r must be a non-negative multiple of %d and %d'
                             % (arm['name'], start, arm['q_chunk'], block))


def blocks_for(tokens, block=BLOCK):
    """Pages covering `tokens`, padded to a multiple of 32 blocks as forward_prefill_paged pads."""
    needed = -(-tokens // block)
    return -(-needed // 32) * 32


def pool_blocks(arms, starts, block=BLOCK):
    """One K/V pool and one page table serve every arm: sized for the largest start + rows."""
    return blocks_for(max(starts) + max(arm['rows'] for arm in arms), block)


def q_shape(arm):
    return (1, NH, arm['rows'], HD)


def kv_shape(blocks):
    return (blocks, NKV, BLOCK, HD)


def busy_cores(arm, cores=GRID[0] * GRID[1], heads=NH):
    """Busy cores for causal chunked SDPA: Q chunks are handed out in pairs when a head has an
    even number of them (sdpa_program_factory.cpp:391-405 in the TT-Sim tree), so the work units
    are heads x q_chunks / 2. q128@2048 -> 96, q256@2048 -> 48, q256@4096 -> 96."""
    q_chunks = arm['rows'] // arm['q_chunk']
    units = heads * q_chunks // 2 if q_chunks % 2 == 0 else heads * q_chunks
    return min(units, cores)


def kv_bytes(arm, start):
    """K+V bytes the causal chunked SDPA reads for one call if nothing is shared: every Q head's
    every Q chunk reads the keys up to its own last row (prefix + causal part)."""
    per_element = BF16_BYTES_PER_ELEMENT if arm['kv_dtype'] == 'bf16' else BF8_BYTES_PER_ELEMENT
    q_chunks = arm['rows'] // arm['q_chunk']
    keys = sum(start + (index + 1) * arm['q_chunk'] for index in range(q_chunks))
    return NH * keys * HD * 2 * per_element


def fit(points):
    """Least-squares line through (chunk_start, ms): slope in ms per 1k keys, intercept in ms."""
    points = [(x, y) for x, y in points if y is not None and math.isfinite(y)]
    if len(points) < 2 or len({x for x, _ in points}) < 2:
        return None
    xs = [x / 1000.0 for x, _ in points]
    ys = [y for _, y in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    return dict(slope_ms_per_1k_keys=slope, intercept_ms=mean_y - slope * mean_x, points=len(points))


def slopes(results):
    """{arm: fit + per-token slope} from {arm: {start: median_ms or None}}."""
    table = {}
    for name, by_start in results.items():
        line = fit(sorted((int(start), ms) for start, ms in by_start.items()))
        if line is None:
            table[name] = None
            continue
        rows = arm_by_name(name)['rows']
        line['slope_ms_per_1k_keys_per_2048_rows'] = line['slope_ms_per_1k_keys'] * ROWS / rows
        table[name] = line
    return table


def _ratio(table, name, base='baseline', per_token=False):
    key = 'slope_ms_per_1k_keys_per_2048_rows' if per_token else 'slope_ms_per_1k_keys'
    if not table.get(name) or not table.get(base) or table[base][key] <= 0:
        return None
    return table[name][key] / table[base][key]


def verdict(table):
    """Apply the ranking's M1 reading rules to the fitted slopes."""
    bf16 = _ratio(table, 'bf16_kv')
    bf16_source = 'bf16_kv'
    if bf16 is None:
        bf16, bf16_source = _ratio(table, 'bf16_qkv'), 'bf16_qkv'
    ratios = dict(bf16=bf16, bf16_source=bf16_source if bf16 is not None else None,
                  exp_approx=_ratio(table, 'exp_approx'), fp32_off=_ratio(table, 'fp32_off'),
                  q256_2048=_ratio(table, 'q256_2048'),
                  q256_4096_per_token=_ratio(table, 'q256_4096', per_token=True),
                  expected_bf16_if_bytes_bound=EXPECTED_BF16_RATIO)
    reasons = []
    if table.get('baseline') is None or bf16 is None:
        return dict(verdict='incomplete', ratios=ratios,
                    reasons=['the baseline or both bf16 arms have no slope; nothing to compare'])
    knobs = [value for value in (ratios['exp_approx'], ratios['fp32_off']) if value is not None]
    compute_moves = bool(knobs) and min(knobs) <= COMPUTE_KNOB_MAX
    q4096 = ratios['q256_4096_per_token']
    if bf16 >= BYTES_BF16_RATIO_MIN and (q4096 is None or q4096 <= BYTES_Q256_4096_MAX):
        name = 'bytes-bound'
        reasons.append('bf16/bf8 slope ratio %.2f >= %.2f (1.88 if purely bytes)' % (bf16, BYTES_BF16_RATIO_MIN))
        if q4096 is None:
            reasons.append('q256_4096 has no slope; the halving check could not run')
        else:
            reasons.append('q256_4096 per-token slope ratio %.2f <= %.2f' % (q4096, BYTES_Q256_4096_MAX))
        action = 'build lever #1 (chunked SDPA K/V prefix sharing)'
    elif bf16 <= COMPUTE_BF16_RATIO_MAX and compute_moves:
        name = 'compute-bound'
        reasons.append('bf16/bf8 slope ratio %.2f <= %.2f: bytes barely matter' % (bf16, COMPUTE_BF16_RATIO_MAX))
        reasons.append('a compute knob moves the slope to %.2f <= %.2f' % (min(knobs), COMPUTE_KNOB_MAX))
        action = 'lever #1 is worth ~0; take #1b (exp_approx / fp32_dest off) through a token gate'
    else:
        name = 'mixed'
        reasons.append('bf16/bf8 slope ratio %.2f, compute knobs %s, q256_4096 per-token %s: '
                       'neither reading rule holds cleanly' % (
                           bf16, ', '.join('%.2f' % value for value in knobs) or 'n/a',
                           'n/a' if q4096 is None else '%.2f' % q4096))
        action = 'no clean call: size #1 from the bf16 ratio (bytes share ~ (ratio - 1) / 0.88)'
    q2048 = ratios['q256_2048']
    prereg = None if q2048 is None else ('confirmed' if q2048 >= 0.95 else 'refuted')
    if prereg is not None:
        reasons.append('pre-registered q256@2048 no-gain: %s (ratio %.2f)' % (prereg, q2048))
    return dict(verdict=name, action=action, ratios=ratios, reasons=reasons, q256_2048_preregistration=prereg)


def verdict_line(result):
    ratios = result['ratios']
    show = lambda value: 'n/a' if value is None else '%.3f' % value
    return ('M1 VERDICT: %s | bf16/bf8=%s (%s) exp_approx=%s fp32_off=%s q256@2048=%s q256@4096/token=%s | %s'
            % (result['verdict'], show(ratios['bf16']), ratios.get('bf16_source') or '-', show(ratios['exp_approx']),
               show(ratios['fp32_off']), show(ratios['q256_2048']), show(ratios['q256_4096_per_token']),
               result.get('action', '')))


def schedule(arms, starts, rounds):
    """Interleaved timing order: every round visits every (arm, start), the arm order rotated."""
    order = []
    for index in range(rounds):
        shift = index % len(arms)
        rotated = arms[shift:] + arms[:shift]
        for arm in rotated:
            for start in starts:
                order.append((arm['name'], start))
    return order


class Watchdog:
    """Per-call host watchdog (spec 3.7 --watchdog-s). A device call still running `seconds` after
    its guard was entered never returns (a hung kernel): print one WATCHDOG line, run `on_fire`
    (the partial report) and end the process with exit 3, since nothing after a hung call can run.
    The poll thread needs the GIL; if a blocking ttnn call holds it, the faulthandler backstop
    (a C thread) dumps the stacks and exits 1 at seconds + grace. seconds <= 0: no thread, no
    backstop, guard() is a no-op (the bench as it was)."""

    def __init__(self, seconds, on_fire=None, clock=time.monotonic, exit=os._exit, poll=0.5, grace=60.0,
                 start=True, backstop=True, out=None):
        self.seconds = float(seconds or 0)
        self.on_fire, self.clock, self.exit, self.poll, self.grace = on_fire, clock, exit, poll, grace
        self.backstop, self.out = backstop and self.seconds > 0, out
        self.what, self.deadline, self.fired, self.budget = None, None, False, self.seconds
        self.lock = threading.Lock()
        if self.seconds > 0 and start:
            threading.Thread(target=self._loop, name='m1-watchdog', daemon=True).start()

    @contextlib.contextmanager
    def guard(self, what, extra=0.0):
        """Arm for one device call; `extra` seconds on top (warmups: the JIT compile is not a hang)."""
        if self.seconds <= 0:
            yield
            return
        budget = self.seconds + extra
        with self.lock:
            self.what, self.deadline, self.budget = what, self.clock() + budget, budget
        if self.backstop:
            try:
                faulthandler.dump_traceback_later(budget + self.grace, exit=True)
            except (RuntimeError, ValueError, OSError):
                pass
        try:
            yield
        finally:
            with self.lock:
                self.what, self.deadline = None, None
            if self.backstop:
                faulthandler.cancel_dump_traceback_later()

    def check(self):
        with self.lock:
            due = self.deadline is not None and self.clock() >= self.deadline and not self.fired
            what, budget = self.what, self.budget
            if due:
                self.fired = True
        if due:
            print('WATCHDOG: %s did not return within %.0f s; exit 3 (reset card M before the next run)'
                  % (what, budget), file=self.out or sys.stdout, flush=True)
            if self.on_fire is not None:
                try:
                    self.on_fire(what)
                except Exception:  # noqa: BLE001 - the exit must happen whatever the report does
                    pass
            self.exit(3)
        return due

    def _loop(self):
        while True:
            time.sleep(self.poll)
            self.check()


def output_sha(torch, tensor):
    """(sha256 of the int16 view, dtype, shape) of one host output (ttnn.to_torch(out)): a bit-exact
    identity that NaN payloads cannot fool (spec 3.7 --sha)."""
    host = tensor.contiguous()
    digest = hashlib.sha256(host.view(torch.int16).numpy().tobytes()).hexdigest()
    return digest, str(host.dtype), list(host.shape)


def sha_lines(report):
    """'M1 SHA <arm>@<start> <hex> stable=0|1', one per timed pair, for grep-level comparison."""
    stable = report.get('sha_stable') or {}
    return ['M1 SHA %s %s stable=%d' % (key, value, int(bool(stable.get(key))))
            for key, value in sorted((report.get('sha256') or {}).items())]


def worker_coords(ttnn, device, grid):
    """worker_core_from_logical_core for every grid core, in the factory's linear order (core i =
    {i % grid_x, i // grid_x}, the reader RT-arg loop): the coordinate fixture the G6 chain order
    (flag 0x4) is costed on (spec 5.1 fixtures/cardm_worker_coords.json)."""
    cores = []
    for index in range(grid[0] * grid[1]):
        logical = ttnn.CoreCoord(index % grid[0], index // grid[0])
        worker = device.worker_core_from_logical_core(logical)
        cores.append(dict(i=index, logical=[index % grid[0], index // grid[0]], worker=[int(worker.x), int(worker.y)]))
    return dict(grid=list(grid), order='i -> logical (i % grid_x, i // grid_x)',
                source='device.worker_core_from_logical_core', cores=cores)


ELF_MAGIC = bytes((0x7F,)) + b'ELF'
PT_LOAD = 1


def elf_load_digest(data):
    """(sha256, bytes) over an ELF's entry point and PT_LOAD segments (vaddr, filesz, memsz, file
    bytes): what the core runs, without the debug info, symbol tables or LTO section names that
    may differ build to build. None if `data` is not a well-formed ELF32/ELF64."""
    if len(data) < 52 or data[:4] != ELF_MAGIC or data[4] not in (1, 2) or data[5] not in (1, 2):
        return None
    end = '<' if data[5] == 1 else '>'
    try:
        if data[4] == 1:
            entry, phoff = struct.unpack_from(end + 'II', data, 24)
            phentsize, phnum = struct.unpack_from(end + 'HH', data, 42)
            layout, pick = end + 'IIIIIIII', (0, 1, 2, 4, 5)       # type offset vaddr paddr filesz memsz flags align
        else:
            entry, phoff = struct.unpack_from(end + 'QQ', data, 24)
            phentsize, phnum = struct.unpack_from(end + 'HH', data, 54)
            layout, pick = end + 'IIQQQQQQ', (0, 2, 3, 5, 6)       # type flags offset vaddr paddr filesz memsz align
        if phentsize < struct.calcsize(layout):
            return None
        digest, loaded = hashlib.sha256(struct.pack('<Q', entry)), 0
        for index in range(phnum):
            fields = struct.unpack_from(layout, data, phoff + index * phentsize)
            p_type, offset, vaddr, filesz, memsz = (fields[i] for i in pick)
            if p_type != PT_LOAD:
                continue
            segment = data[offset:offset + filesz]
            if len(segment) != filesz:
                return None
            digest.update(struct.pack('<QQQ', vaddr, filesz, memsz))
            digest.update(segment)
            loaded += filesz
    except struct.error:
        return None
    return digest.hexdigest(), loaded


def kernel_elf_report(root, name):
    """--kernel-elf: every compiled ELF of kernel `name` under the JIT cache `root`
    (<root>/.../kernels/<name>/<hash>/<risc>/*.elf), each digested by elf_load_digest, and one
    digest over them keyed by the path from 'kernels/<name>/' on (so the cache's own top-level
    directory name does not enter it). Never raises: a missing cache or ELF gives digest None."""
    report = dict(name=name, root=root, digest=None, files=[], other_files=[])
    try:
        if not root or not os.path.isdir(root):
            report['error'] = 'no kernel cache directory %r' % (root,)
            return report
        marker = 'kernels/%s/' % name
        for directory, _, names in os.walk(root):
            for file_name in names:
                path = os.path.join(directory, file_name)
                slashed = '/' + os.path.relpath(path, root).replace(os.sep, '/')
                at = slashed.rfind('/' + marker)
                if at < 0:
                    continue
                key = slashed[at + 1:]
                if not file_name.endswith('.elf'):
                    report['other_files'].append(key)
                    continue
                data = Path(path).read_bytes()
                loaded = elf_load_digest(data)
                report['files'].append(dict(path=key, file_sha256=hashlib.sha256(data).hexdigest(),
                                            load_sha256=None if loaded is None else loaded[0],
                                            load_bytes=None if loaded is None else loaded[1]))
        report['files'].sort(key=lambda entry: entry['path'])
        report['other_files'] = sorted(report['other_files'])[:40]
        if report['files']:
            combined = hashlib.sha256()
            for entry in report['files']:
                combined.update(('%s %s' % (entry['path'], entry['load_sha256'] or 'raw:' + entry['file_sha256'])
                                 + chr(10)).encode('utf-8'))
            report['digest'] = combined.hexdigest()
    except Exception as error:  # noqa: BLE001 - evidence only, never the run's outcome
        report['error'] = '%s: %s' % (type(error).__name__, error)
    return report


def kernel_elf_line(report):
    return 'M1 KERNEL_ELF %s %s files=%d' % (report['name'], report.get('digest') or 'none', len(report.get('files') or ()))


def q2_exact(chains, sha256, sha_stable=None):
    """Rule (d): every production chain arm's output equals the baseline's, bit for bit, at every
    timed start (--sha digests, report 'sha256' keyed 'arm@start'), and every digest was stable over
    its two calls. None without digests (a run without --sha cannot pass Q2)."""
    if not sha256:
        return None, ['no --sha digests: exactness unchecked']
    problems = []
    starts = sorted({int(key.rsplit('@', 1)[1]) for key in sha256 if key.startswith('baseline@')})
    if not starts:
        return False, ['no baseline digest']
    for name in sorted(chains):
        for start in starts:
            key, base = '%s@%d' % (name, start), 'baseline@%d' % start
            if sha256.get(key) is None:
                problems.append('%s: no digest' % key)
            elif sha256[key] != sha256[base]:
                problems.append('%s differs from the baseline output' % key)
    for key, stable in sorted((sha_stable or {}).items()):
        if not stable and (key.startswith('baseline@') or key.split('@', 1)[0] in chains):
            problems.append('%s: the two calls gave different outputs' % key)
    return not problems, problems


def q2_verdict(table, k0b32_slope=K0B32_SLOPE, sha256=None, sha_stable=None):
    """Spec 6.3's pass rules on one run's fitted slopes (None without the baseline and a chain arm):
    the best production chain arm (no test flags) against K0b32 (a), the baseline (b) and the
    baseline intercept (c); (d) every production chain arm's output equals the baseline's at every
    start (--sha; without digests d is unchecked and Q2 does not pass); per-step times; and which
    chain arms beat 'chain' (0x1) by >= 2%."""
    base = table.get('baseline')
    chains = {}
    for name, line in table.items():
        word = arm_by_name(name).get('program_word')
        if line and word is not None and (word & 0xFFFF0000) == PF_TAG and not word & PF_TEST_FLAGS:
            chains[name] = line
    if not base or not chains:
        return None
    best = min(chains, key=lambda name: chains[name]['slope_ms_per_1k_keys'])
    slope = chains[best]['slope_ms_per_1k_keys']
    exact, exact_problems = q2_exact(chains, sha256, sha_stable)
    rules = dict(a=slope <= Q2_FORWARD_MAX * k0b32_slope, b=slope <= Q2_BASELINE_MAX * base['slope_ms_per_1k_keys'],
                 c=chains[best]['intercept_ms'] <= base['intercept_ms'] + Q2_INTERCEPT_MAX_MS, d=exact)
    step_us = {name: line['slope_ms_per_1k_keys'] * STEP_US_PER_SLOPE for name, line in chains.items()}
    step_us['baseline'] = base['slope_ms_per_1k_keys'] * STEP_US_PER_SLOPE
    reference = chains.get('chain')
    better = sorted(name for name, line in chains.items() if name != 'chain' and reference is not None
                    and line['slope_ms_per_1k_keys'] <= (1 - Q2_PREFER_MARGIN) * reference['slope_ms_per_1k_keys'])
    return dict(best=best, best_slope=slope, baseline_slope=base['slope_ms_per_1k_keys'], k0b32_slope=k0b32_slope,
                best_over_k0b32=slope / k0b32_slope, best_over_baseline=slope / base['slope_ms_per_1k_keys'],
                intercept_delta_ms=chains[best]['intercept_ms'] - base['intercept_ms'], rules=rules,
                passed=all(value is True for value in rules.values()), step_us=step_us, beat_chain_by_2pct=better,
                exact_problems=exact_problems)


def q2_line(q2):
    exact = q2['rules'].get('d')
    return ('M1 Q2: %s | best=%s slope=%.4f (x%.3f K0b32, x%.3f baseline) intercept%+.3f ms step=%.2f us '
            '(baseline %.2f us) a=%d b=%d c=%d d(exact)=%s beat_0x1_by_2pct=%s%s' % (
                'PASS' if q2['passed'] else 'FAIL', q2['best'], q2['best_slope'], q2['best_over_k0b32'],
                q2['best_over_baseline'], q2['intercept_delta_ms'], q2['step_us'][q2['best']], q2['step_us']['baseline'],
                q2['rules']['a'], q2['rules']['b'], q2['rules']['c'], '-' if exact is None else int(exact),
                ','.join(q2['beat_chain_by_2pct']) or '-',
                '' if not q2.get('exact_problems') else ' | ' + '; '.join(q2['exact_problems'][:4])))


PF_LOG = '[QWEN-SDPA-PF] flags='
PF_LOG_LINE = re.compile(r'\[QWEN-SDPA-PF\] flags=(0x[0-9a-f]+) kv_chain=1 chains=([0-9]+) members=([0-9]+) '
                         r'order=([a-z]+)')


def pf_log_lines(text):
    """The factory F4 lines: [dict(flags, chains, members, order)]."""
    return [dict(flags=int(m.group(1), 16), chains=int(m.group(2)), members=int(m.group(3)), order=m.group(4))
            for m in PF_LOG_LINE.finditer(text)]


def check_pf_log(lines, arms, rows=ROWS):
    """[problems]: exactly one line per chain arm's flags (one program each: one shape, one page
    width per run), none for flags no arm used, chains=16 members=96 at 2048 rows, the order the
    0x4 bit asks for."""
    problems = []
    wanted = sorted({arm['program_word'] & 0xFFFF for arm in arms
                     if arm.get('program_word') is not None and (arm['program_word'] & 0xFFFF0000) == PF_TAG})
    counts = {}
    for line in lines:
        counts[line['flags']] = counts.get(line['flags'], 0) + 1
        if rows == 2048 and (line['chains'], line['members']) != (16, 96):
            problems.append('flags %#x: chains=%d members=%d, expected 16 / 96' % (line['flags'], line['chains'],
                                                                                line['members']))
        if line['order'] != ('noc' if line['flags'] & 0x4 else 'raster'):
            problems.append('flags %#x: order=%s' % (line['flags'], line['order']))
    for flags in wanted:
        if counts.get(flags, 0) != 1:
            problems.append('flags %#x: %d factory lines, expected exactly 1' % (flags, counts.get(flags, 0)))
    for flags in sorted(set(counts) - set(wanted)):
        problems.append('flags %#x: %d factory lines nobody asked for' % (flags, counts[flags]))
    return problems


class NativeLog:
    """fds 1 and 2 (tt-logger's sinks) to a file for the run; Python's own prints keep going to the
    original stdout/stderr (as test_sdpa_decode_qwen_card_m.NativeLog)."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.saved = [os.dup(1), os.dup(2)]
        self.handle = open(self.path, 'wb')
        os.dup2(self.handle.fileno(), 1)
        os.dup2(self.handle.fileno(), 2)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = os.fdopen(os.dup(self.saved[0]), 'w', buffering=1)
        sys.stderr = os.fdopen(os.dup(self.saved[1]), 'w', buffering=1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = self.stdout, self.stderr
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        for fd in self.saved:
            os.close(fd)
        self.handle.close()
        return False

    def text(self):
        return self.path.read_bytes().decode('utf-8', 'replace') if self.path.is_file() else ''


def format_table(results, table, starts):
    lines = ['%-11s %6s %5s' % ('arm', 'rows', 'cores') + ''.join(' %10s' % ('@%dk' % (s // 1024)) for s in starts)
             + ' %12s %14s' % ('ms/1k keys', 'per 2048 rows')]
    names = [arm['name'] for arm in ARMS if arm['name'] in results] + [name for name in results if name not in ARM_NAMES]
    for arm in (arm_by_name(name) for name in names):
        cells = ''.join(' %10s' % ('ERR' if results[arm['name']].get(str(s)) is None
                                   else '%.3f' % results[arm['name']][str(s)]) for s in starts)
        line = table.get(arm['name'])
        tail = (' %12s %14s' % ('n/a', 'n/a') if line is None else
                ' %12.4f %14.4f' % (line['slope_ms_per_1k_keys'], line['slope_ms_per_1k_keys_per_2048_rows']))
        lines.append('%-11s %6d %5d' % (arm['name'], arm['rows'], busy_cores(arm)) + cells + tail)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------
# Device side. ttnn/torch are imported inside these functions only.
# ---------------------------------------------------------------------------------------------

def _dtype(ttnn, name):
    return ttnn.bfloat8_b if name == 'bf8' else ttnn.bfloat16


def page_table_blocks(arms, starts, page_blocks=None, block=BLOCK):
    """The pool / page-table width: what the starts need, or --page-blocks (>= that, a multiple of 32)."""
    needed = pool_blocks(arms, starts, block)
    if page_blocks is None:
        return needed
    if page_blocks < needed or page_blocks % 32:
        raise ValueError('--page-blocks %d must be >= %d (the largest start + rows) and a multiple of 32'
                         % (page_blocks, needed))
    return page_blocks


def build_inputs(ttnn, torch, device, arms, starts, seed=0, q_memory='dram', page_blocks=None):
    """One K/V pool per KV dtype, one scattered page table, one Q per (rows, dtype)."""
    blocks = page_table_blocks(arms, starts, page_blocks)
    q_config = ttnn.L1_MEMORY_CONFIG if q_memory == 'l1' else ttnn.DRAM_MEMORY_CONFIG
    generator = torch.Generator().manual_seed(seed)
    inputs = dict(blocks=blocks, pools={}, queries={}, starts={})
    for kv_dtype in sorted({arm['kv_dtype'] for arm in arms}):
        pool = []
        for _ in range(2):
            host = torch.randn(kv_shape(blocks), generator=generator, dtype=torch.float32).to(torch.bfloat16)
            pool.append(ttnn.from_torch(host, dtype=_dtype(ttnn, kv_dtype), layout=ttnn.TILE_LAYOUT, device=device,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG))
        inputs['pools'][kv_dtype] = tuple(pool)
    permutation = torch.randperm(blocks, generator=generator).to(torch.int32).reshape(1, blocks)
    inputs['page_table'] = ttnn.from_torch(permutation, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for arm in arms:
        key = (arm['rows'], arm['q_dtype'])
        if key not in inputs['queries']:
            host = torch.randn(q_shape(arm), generator=generator, dtype=torch.float32).to(torch.bfloat16)
            inputs['queries'][key] = ttnn.from_torch(host, dtype=_dtype(ttnn, arm['q_dtype']), layout=ttnn.TILE_LAYOUT,
                                                     device=device, memory_config=q_config)
    for start in starts:
        inputs['starts'][start] = ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32,
                                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return inputs


def make_call(ttnn, device, inputs, arm, start, grid, start_mode):
    """A zero-argument closure making the served call for one (arm, chunk_start)."""
    k_pool, v_pool = inputs['pools'][arm['kv_dtype']]
    query = inputs['queries'][(arm['rows'], arm['q_dtype'])]
    compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                               fp32_dest_acc_en=arm['fp32_dest'], packer_l1_acc=True)
    config = dict(compute_with_storage_grid_size=grid, exp_approx_mode=arm['exp_approx'],
                  q_chunk_size=arm['q_chunk'], k_chunk_size=arm['k_chunk'])
    if arm.get('program_word') is not None:
        config['max_cores_per_head_batch'] = arm['program_word']   # the G6 chain word (K64g factory F1)
    program = ttnn.SDPAProgramConfig(**config)
    common = dict(input_tensor_q=query, input_tensor_k=k_pool, input_tensor_v=v_pool,
                  page_table_tensor=inputs['page_table'], compute_kernel_config=compute, program_config=program)
    if start_mode == 'tensor':
        return lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
            chunk_start_idx_tensor=inputs['starts'][start], **common)
    return lambda: ttnn.transformer.chunked_scaled_dot_product_attention(chunk_start_idx=start, **common)


def output_digests(ttnn, torch, calls, guard):
    """--sha: two fresh calls per timed (arm, start); the first call's digest, and whether the second
    matched it (a K0c-vs-stock mismatch means nothing unless stock is stable run to run)."""
    digests, stable, meta = {}, {}, {}
    for key in sorted(calls, key=lambda pair: (pair[0], pair[1])):
        seen = []
        for attempt in range(2):
            with guard('%s@%d sha %d' % (key[0], key[1], attempt)):
                out = calls[key]()
                host = ttnn.to_torch(out)
                ttnn.deallocate(out)
            seen.append(output_sha(torch, host))
        label = '%s@%d' % key
        digests[label], stable[label] = seen[0][0], seen[0][0] == seen[1][0]
        meta[label] = dict(dtype=seen[0][1], shape=seen[0][2])
    return dict(sha256=digests, sha_stable=stable, sha_meta=meta)


def run(options, watchdog=None):
    import torch
    import ttnn

    guard = (watchdog or Watchdog(0)).guard
    arms = [arm_by_name(name) for name in options.arms]
    for arm in arms:
        validate(arm, options.starts)
    with guard('open_device', extra=COMPILE_GRACE_S):      # a fresh TT_METAL_CACHE: the firmware builds here
        device = ttnn.open_device(device_id=options.device_id, l1_small_size=24576)
    report = dict(passed=False, arms=arms, starts=list(options.starts), start_mode=options.start_mode,
                  warmup=options.warmup, rounds=options.rounds, q_memory=getattr(options, 'q_memory', 'dram'),
                  page_blocks=getattr(options, 'page_blocks', None))
    try:
        grid_size = device.compute_with_storage_grid_size()
        grid = (grid_size.x, grid_size.y)
        report['grid'] = list(grid)
        if grid != GRID:
            report['grid_note'] = 'grid %r differs from the served 11 x 10; busy-core figures assume 11 x 10' % (grid,)
        coords_out = getattr(options, 'coords_out', None)
        if coords_out:
            try:
                with guard('worker coordinates'):
                    coords = worker_coords(ttnn, device, grid)
                coords_out.parent.mkdir(parents=True, exist_ok=True)
                coords_out.write_text(json.dumps(coords, indent=1) + chr(10), encoding='utf-8', newline=chr(10))
                report['coords_out'] = str(coords_out)
            except Exception as error:  # noqa: BLE001 - the fixture must never cost the timing run
                report['coords_error'] = '%s: %s' % (type(error).__name__, error)
        with guard('build_inputs', extra=COMPILE_GRACE_S):     # host bf8 tilize of two K/V pools, 8 CPUs, CI load
            inputs = build_inputs(ttnn, torch, device, arms, options.starts, seed=options.seed,
                                  q_memory=getattr(options, 'q_memory', 'dram'),
                                  page_blocks=getattr(options, 'page_blocks', None))
        report['pool_blocks'] = inputs['blocks']
        calls, errors, paths = {}, {}, {}
        for arm in arms:
            for index, start in enumerate(options.starts):
                key = (arm['name'], start)
                mode = options.start_mode
                call = make_call(ttnn, device, inputs, arm, start, grid, mode)
                # The arm's first warmup JIT-compiles its program on a fresh cache.
                grace = FIRST_COMPILE_GRACE_S if index == 0 else COMPILE_GRACE_S
                try:
                    with guard('%s@%d warmup' % key, extra=grace):
                        for _ in range(options.warmup):
                            ttnn.deallocate(call())
                        ttnn.synchronize_device(device)
                except Exception as error:  # noqa: BLE001 - one bad arm must not end the sweep
                    if mode == 'tensor' and options.fallback_scalar:
                        mode = 'scalar'
                        call = make_call(ttnn, device, inputs, arm, start, grid, mode)
                        try:
                            with guard('%s@%d warmup (scalar)' % key, extra=FIRST_COMPILE_GRACE_S):
                                for _ in range(options.warmup):
                                    ttnn.deallocate(call())
                                ttnn.synchronize_device(device)
                            errors[key] = 'tensor path failed (%s: %s); timed on the scalar path' % (
                                type(error).__name__, error)
                        except Exception as second:  # noqa: BLE001
                            errors[key] = '%s: %s' % (type(second).__name__, second)
                            continue
                    else:
                        errors[key] = '%s: %s' % (type(error).__name__, error)
                        continue
                calls[key], paths[key] = call, mode
        samples = {key: [] for key in calls}
        for name, start in schedule(arms, list(options.starts), options.rounds):
            key = (name, start)
            if key not in calls:
                continue
            with guard('%s@%d timed' % key):
                ttnn.synchronize_device(device)
                began = time.perf_counter()
                out = calls[key]()
                ttnn.synchronize_device(device)
                samples[key].append((time.perf_counter() - began) * 1e3)
                ttnn.deallocate(out)
        finite = {}
        for arm in arms:
            key = (arm['name'], options.starts[0])
            if key in calls:
                with guard('%s@%d finite' % key):
                    out = calls[key]()
                    finite[arm['name']] = bool(torch.isfinite(ttnn.to_torch(out).float()).all())
                    ttnn.deallocate(out)
        digests = output_digests(ttnn, torch, calls, guard) if getattr(options, 'sha', False) else {}
        results = {arm['name']: {str(start): (statistics.median(samples[(arm['name'], start)])
                                              if samples.get((arm['name'], start)) else None)
                                 for start in options.starts} for arm in arms}
        table = slopes(results)
        outcome = verdict(table)
        q2 = q2_verdict(table, getattr(options, 'k0b32_slope', K0B32_SLOPE), digests.get('sha256'),
                        digests.get('sha_stable'))
        report.update(
            passed=bool(calls) and all(value is not None for value in results.get('baseline', {}).values()),
            median_ms=results, slopes=table, verdict=outcome, verdict_line=verdict_line(outcome),
            samples_ms={'%s@%d' % key: values for key, values in samples.items()},
            paths={'%s@%d' % key: value for key, value in paths.items()},
            errors={'%s@%d' % key: value for key, value in errors.items()}, finite_output=finite,
            busy_cores={arm['name']: busy_cores(arm) for arm in arms},
            kv_gbytes_per_call={arm['name']: {str(s): kv_bytes(arm, s) / 1e9 for s in options.starts} for arm in arms},
            table=format_table(results, table, list(options.starts)), **digests)
        if q2 is not None:
            report.update(q2=q2, q2_line=q2_line(q2))
    finally:
        with guard('close_device'):
            ttnn.close_device(device)
    return report


def parse_list(text, cast=str):
    return [cast(value) for value in text.split(',') if value.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True, help='JSON report path')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--arms', default=','.join(ARM_NAMES))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)))
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=9, help='interleaved timing rounds (samples per pair)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start-mode', choices=('tensor', 'scalar'), default='tensor',
                        help='tensor = the served flexible path (device chunk_start_idx_tensor)')
    parser.add_argument('--no-fallback-scalar', dest='fallback_scalar', action='store_false',
                        help='do not retry a failing tensor-path pair on the scalar path')
    parser.add_argument('--sha', action='store_true',
                        help='sha256 of every output per (arm, start), int16 view of ttnn.to_torch(out)')
    parser.add_argument('--watchdog-s', type=float, default=0.0,
                        help='per device call: WATCHDOG and exit 3 if a call is not back in this many s (0 = off)')
    parser.add_argument('--coords-out', type=Path, default=None,
                        help='also write worker_core_from_logical_core for every grid core (JSON) here')
    parser.add_argument('--kernel-elf', default=None, metavar='KERNEL',
                        help="after the run, digest this kernel's compiled ELFs in $TT_METAL_CACHE (PT_LOAD bytes)")
    parser.add_argument('--program-word', default='',
                        help='extra arms: the baseline call with max_cores_per_head_batch = each HEX word (comma list)')
    parser.add_argument('--q-memory', choices=('dram', 'l1'), default='dram', help='Q placement (the model: l1)')
    parser.add_argument('--page-blocks', type=int, default=None, help='page-table width in blocks (the model: 2080)')
    parser.add_argument('--verify-log', action='store_true',
                        help="capture fds 1/2 to <out>.native.log and check the factory's [QWEN-SDPA-PF] lines")
    parser.add_argument('--k0b32-slope', type=float, default=K0B32_SLOPE, help='Q2 rule (a) reference slope')
    options = parser.parse_args(argv)
    options.arms = parse_list(options.arms)
    for word in parse_list(options.program_word, lambda text: int(text, 0)):
        name = word_arm(word)['name']
        if name not in options.arms:
            options.arms.append(name)
    options.starts = parse_list(options.starts, int)
    for name in options.arms:
        arm_by_name(name)
    if options.rounds < 1 or options.warmup < 1:
        parser.error('--rounds and --warmup must be at least 1')
    if options.watchdog_s < 0:
        parser.error('--watchdog-s must be >= 0')

    def write_report(report):
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8', newline='\n')

    watchdog = Watchdog(options.watchdog_s, on_fire=lambda what: write_report(
        dict(passed=False, error='WATCHDOG: %s did not return within %.0f s' % (what, options.watchdog_s),
             watchdog=what)))
    report = dict(passed=False)
    native = NativeLog(options.out.with_name(options.out.name + '.native.log')) if options.verify_log else None
    try:
        if native is not None:
            options.out.parent.mkdir(parents=True, exist_ok=True)
            with native:
                report = run(options, watchdog)
        else:
            report = run(options, watchdog)
    except Exception as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        if native is not None:
            lines = pf_log_lines(native.text())
            problems = check_pf_log(lines, [arm_by_name(name) for name in options.arms])
            report['pf_log'] = dict(path=str(native.path), lines=lines, problems=problems)
            if problems:
                report['passed'] = False
        if options.kernel_elf:
            report['kernel_elf'] = kernel_elf_report(os.environ.get('TT_METAL_CACHE'), options.kernel_elf)
        write_report(report)
    if report.get('table'):
        print(report['table'])
    for key, value in sorted((report.get('errors') or {}).items()):
        print('ERROR %s: %s' % (key, value))
    for line in sha_lines(report):
        print(line)
    if report.get('kernel_elf'):
        print(kernel_elf_line(report['kernel_elf']))
    if report.get('coords_error'):
        print('COORDS ERROR: %s' % report['coords_error'])
    if report.get('pf_log') is not None:
        for line in report['pf_log']['lines']:
            print('M1 PF_LOG flags=%#x chains=%d members=%d order=%s' % (line['flags'], line['chains'], line['members'],
                                                                        line['order']))
        print('M1 PF_LOG %s' % ('ok' if not report['pf_log']['problems']
                                else 'FAIL: ' + '; '.join(report['pf_log']['problems'])))
    if report.get('q2_line'):
        print(report['q2_line'])
    print(report.get('verdict_line') or 'M1 VERDICT: incomplete | %s' % report.get('error', 'no result'))
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
