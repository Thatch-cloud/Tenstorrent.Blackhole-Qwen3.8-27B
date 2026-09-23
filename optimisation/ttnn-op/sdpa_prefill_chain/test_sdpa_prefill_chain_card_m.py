"""Card-M qualification of the [QWEN-SDPA-PF] G6 K/V chain (sdpa-prefill-share-spec.md 6.2, Q1).

The served causal chunked paged SDPA (the model's forward_prefill_paged call: 12 Q heads, 2 KV
heads, head dim 256, q/k chunk 128, bf8 Q/K/V, HiFi2 with fp32 dest, the flexible path with a
device chunk_start tensor) against the same call with SDPAProgramConfig.max_cores_per_head_batch =
0x5EFA0000 | flags on the K64g graft. Byte-for-byte (sha256 of the int16 view of ttnn.to_torch).

ROLES (run_card_m_pf.sh runs each in its own container on card M only):
  reference   the stock image, no graft: the sha of every served output in the matrix
  candidate   the graft mounted, in this order: Q1.7 program cache (served then chain at one fresh
              shape adds exactly one cache entry, no entry across chunk_start); Q1.6 served ==
              reference (every swept case must be in the reference report); Q1.5 every chain flag
              (0x1, 0x3, 0x5, 0x7: every flag set Q2 may choose) == reference; Q1.8 the six refusals
              (TT_FATAL, no hang), BEFORE anything builds a test-flag program: the factory checks
              only on a program-cache miss, so a refusal whose program is already cached is
              reported as untestable instead of silently passing or failing; Q1.4 mutation M-A
              (0x103: all 192 (head, q) units differ from the served output, not just the 32
              injector units); Q1.9 1,000 alternating served / chain calls and 200 replays of one
              captured chain call while copy_host_to_device_tensor rewrites chunk_start; then the
              factory log (one line per chain program - one per (rows, page width, Q memory, flags):
              DRAM and L1 Q are separate programs - none for a served one, chains=16 members=96 at
              2048 rows). The watcher pass (WATCHER=1) is this role narrowed by the runner.
  hang        Q1.3, LAST in its session: a served pre-check call on the same shape must return
              first (card M healthy, inputs uploaded, compute and writer JIT-warm), then
              QWEN_SDPA_PF_TEST=1, flags 0x203 (the sink withholds its last credit), rows 2048,
              start 2048. It must never return: the per-call watchdog prints WATCHDOG on the planted
              call or its read back and exits 3 (a faulthandler backstop exits 1 with 'Timeout ('
              if a blocking ttnn call holds the GIL); under the watcher the device reports QWDV (the
              sink) and QWDC (its upstream) and the assert. A watchdog on any other operation is not
              a pass. If the call returns, the role FAILS (exit 1).

MATRIX (spec 6.2 item 5): rows {2048, 1024, 512} x starts {0, 128, 1920, 2048, 4224, 63488, 65664,
126976} x seeds {0, 1, 2} x Q variants {normal, peaky (Q x 8), zeroq} x page tables {perm (random
permutation of the pool), identity} x page-table widths {fit (the starts' need, padded to 32 blocks:
2016), model (2080)} x Q memory {dram, l1}. Odd C (1, 33, 513) and C 15/16 cover slot-parity flips
at the unit boundary. One K/V pool of 2080 blocks per seed.

Everything the C++ side prints goes to <out>.native.log (NativeLog); the helpers above run() import
no ttnn and are unit-tested on CPU (test_sdpa_prefill_chain_sources.py, with a fake ttnn).

    python3 -B test_sdpa_prefill_chain_card_m.py --role reference --out /results/ref.json
    python3 -B test_sdpa_prefill_chain_card_m.py --role candidate --reference /results/ref.json --out /results/cand.json
"""

import argparse
from contextlib import contextmanager
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

NH, NKV, HD, BLOCK, Q_CHUNK, TILE = 12, 2, 256, 64, 128, 32
POOL_BLOCKS = 2080                      # the model's page-table width at 131k
ROWS = (2048, 1024, 512)
STARTS = (0, 128, 1920, 2048, 4224, 63488, 65664, 126976)
SEEDS = (0, 1, 2)
VARIANTS = ('normal', 'peaky', 'zeroq')
TABLES = ('perm', 'identity')
WIDTHS = ('fit', 'model')
Q_MEMORY = ('dram', 'l1')
PF_TAG = 0x5EFA0000
PRODUCTION_FLAGS = (0x1, 0x3, 0x5, 0x7)  # every flag set the Q2 bench arms can choose (chain .. chain_bo)
DEFAULT_FLAGS = 0x1                     # K0: the chain alone (0x2 buys nothing at the served read-ahead)
MUTATION_FLAGS = 0x103
HANG_FLAGS = 0x203
HANG_CALL_LABEL = 'planted hang 0x203'
HANG_READBACK_LABEL = 'planted hang read back'
HANG_LABELS = (HANG_CALL_LABEL, HANG_READBACK_LABEL)   # the only watchdog labels that pass Q1.3
HANG_ARMED = 'planted hang: armed'      # printed after the served pre-check returned (run_card_m_pf.sh greps it)
HANG_CALL_GRACE_S = 300.0               # the planted call JIT-compiles only the chain reader (compute/writer warm)
UNKNOWN_WORD = PF_TAG | 0x8
TEST_ENV = 'QWEN_SDPA_PF_TEST'
UNITS_2048 = NH * (2048 // Q_CHUNK)     # 192 (head, q) units
INJECTOR_UNITS = 32                     # 16 injectors x 2 units
TRACE_STARTS = (0, 2048, 126976, 128)
BINARY_MARKER = b'[QWEN-SDPA-PF] flags='
FACTORY_LINE = re.compile(r'\[QWEN-SDPA-PF\] flags=(0x[0-9a-f]+) kv_chain=1 chains=([0-9]+) members=([0-9]+) '
                          r'order=([a-z]+)')
# Q1.8: (name, what the call changes from the served chain call, the TT_FATAL text it must raise).
REFUSALS = (
    ('non-causal call', dict(causal=False), 'kv_chain outside its qualified envelope'),
    ('legacy int chunk_start', dict(legacy_start=True), 'kv_chain outside its qualified envelope'),
    ('fp32 dest off (streaming compute)', dict(fp32_dest=False), 'kv_chain outside its qualified envelope'),
    ('bf16 K/V', dict(kv_bf16=True), 'kv_chain outside its qualified envelope'),
    ('unknown flag 0x5EFA0008', dict(word=UNKNOWN_WORD), 'unknown or incomplete flags'),
    ('test flags without %s' % TEST_ENV, dict(word=PF_TAG | MUTATION_FLAGS, clear_test_env=True), 'test-only flags'),
)


# ---------------------------------------------------------------------------------------------
# Host helpers (no ttnn): the matrix, labels, inputs, digests, log checks.
# ---------------------------------------------------------------------------------------------

def word(flags):
    return PF_TAG | flags


def blocks_needed(start, rows, block=BLOCK):
    return -(-(start + rows) // block)


def page_width(kind, rows, starts):
    """'fit': the largest start's need padded to 32 blocks (forward_prefill_paged's pad); 'model': 2080."""
    if kind == 'model':
        return POOL_BLOCKS
    return -(-blocks_needed(max(starts), rows) // 32) * 32


def validate_starts(starts, rows):
    for start in starts:
        if start < 0 or start % Q_CHUNK or start % BLOCK:
            raise ValueError('chunk_start %d must be a non-negative multiple of %d' % (start, Q_CHUNK))
        if blocks_needed(start, rows) > POOL_BLOCKS:
            raise ValueError('chunk_start %d + rows %d exceeds the %d-block pool' % (start, rows, POOL_BLOCKS))


def group_label(rows, width, table, qmem, seed, variant):
    return 'r%d-%s-%s-%s-s%d-%s' % (rows, width, table, qmem, seed, variant)


def case_label(group, start):
    return '%s@%d' % (group, start)


def groups(args):
    """Every (rows, width, table, qmem, seed, variant) group the run covers, in run order."""
    out = []
    for rows in args.rows:
        for width in args.widths:
            for table in args.tables:
                for qmem in args.q_memory:
                    for seed in args.seeds:
                        for variant in args.variants:
                            out.append((rows, width, table, qmem, seed, variant))
    return out


def int16_view(torch, tensor):
    return tensor.to(torch.bfloat16).contiguous().view(torch.int16)


def digest(torch, tensor):
    return hashlib.sha256(int16_view(torch, tensor).numpy().tobytes()).hexdigest()


def unit_digests(torch, tensor, q_chunk=Q_CHUNK):
    """{(head, q): sha} over the (1, NH, rows, HD) output, one per 128-row Q chunk of a head."""
    view = int16_view(torch, tensor)
    heads, rows = view.shape[1], view.shape[2]
    return {(h, q): hashlib.sha256(view[0, h, q * q_chunk:(q + 1) * q_chunk].contiguous().numpy().tobytes()).hexdigest()
            for h in range(heads) for q in range(rows // q_chunk)}


def mutation_verdict(served_units, mutated_units):
    """(passed, differing units, message). All 192 must differ; only the 32 injector units
    differing means the receivers read DRAM themselves (the chain is not active)."""
    differing = sum(1 for key, value in served_units.items() if mutated_units.get(key) != value)
    total = len(served_units)
    if differing == total:
        return True, differing, 'all %d units differ' % total
    if differing == INJECTOR_UNITS:
        return False, differing, 'only the %d injector units differ: the receivers read DRAM (chain not active)' % differing
    return False, differing, '%d of %d units differ' % (differing, total)


def factory_lines(text):
    return [dict(flags=int(m.group(1), 16), chains=int(m.group(2)), members=int(m.group(3)), order=m.group(4))
            for m in FACTORY_LINE.finditer(text)]


def check_factory_lines(lines, requested):
    """requested: {(rows, width_blocks, q_memory, flags)} chain programs the run built (one line
    each: the program cache is on, so a second call on a program never re-runs the factory; a DRAM
    and an L1 Q are two programs - the Q tensor spec and its accessor CT args differ). A 2048-row
    program has 16 chains of 6; every line's order follows flag 0x4."""
    problems = []
    by_flags = {}
    for line in lines:
        by_flags[line['flags']] = by_flags.get(line['flags'], 0) + 1
        if line['order'] != ('noc' if line['flags'] & 0x4 else 'raster'):
            problems.append('flags %#x: order=%s' % (line['flags'], line['order']))
        if line['members'] != 6 * line['chains']:
            problems.append('flags %#x: %d members in %d chains (groups of 6 expected)' % (line['flags'], line['members'],
                                                                                        line['chains']))
    wanted = {}
    for _rows, _width, _qmem, flags in requested:
        wanted[flags] = wanted.get(flags, 0) + 1
    for flags in sorted(set(wanted) | set(by_flags)):
        if by_flags.get(flags, 0) != wanted.get(flags, 0):
            problems.append('flags %#x: %d factory lines for %d requested programs' % (flags, by_flags.get(flags, 0),
                                                                                    wanted.get(flags, 0)))
    if any(key[0] == 2048 for key in requested) and not any(l['chains'] == 16 and l['members'] == 96 for l in lines):
        problems.append('no chains=16 members=96 line for a 2048-row program')
    return problems


def expected_chains(rows):
    """G6 groups at `rows` (12 heads, 2 KV heads, 110 cores): 16 at 2048, 8 at 1024, 4 at 512."""
    return {2048: (16, 96), 1024: (8, 48), 512: (4, 24)}.get(rows)


def loaded_binary(maps='/proc/self/maps'):
    paths = sorted({line.split()[-1] for line in Path(maps).read_text().splitlines()
                    if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    data = Path(paths[0]).read_bytes()
    return paths[0], hashlib.sha256(data).hexdigest(), BINARY_MARKER in data


class Watchdog:
    """Per-device-call deadline: a call not back within `seconds` prints WATCHDOG, runs on_fire (the
    partial report) and os._exit(3)s - a hung chain cannot be interrupted from Python otherwise.

    The poll thread needs the GIL; a blocking ttnn call (to_torch on a hung program) may hold it, so
    each op also arms a faulthandler backstop (a C thread, as sdpa_prefill_bench.Watchdog does): at
    the op's budget + `grace` it dumps every thread's stack ('Timeout (h:mm:ss)!') and exits 1.
    run_card_m_pf.sh reads exit 3 + WATCHDOG, or exit 1 + 'Timeout (', as "the call never returned".
    Nested ops re-arm the backstop for the outer op's remaining time on exit."""

    def __init__(self, seconds, *, on_fire=None, stream=None, exit=os._exit, clock=time.monotonic, grace=60.0,
                 backstop=True):
        self.seconds, self.on_fire, self.stream, self.exit, self.clock = seconds, on_fire, stream, exit, clock
        self.grace, self.backstop = grace, bool(backstop and seconds)
        self.label, self.deadline, self.fired = None, None, False
        self.lock = threading.Lock()
        self.thread = None
        self.armed = []                         # backstop timeouts armed, in order (tests read it)

    def start(self):
        if self.seconds and self.thread is None:
            self.thread = threading.Thread(target=self.poll, name='pf-watchdog', daemon=True)
            self.thread.start()
        return self

    def arm_backstop(self, seconds):
        if not self.backstop:
            return
        self.armed.append(seconds + self.grace)
        try:
            faulthandler.dump_traceback_later(seconds + self.grace, exit=True, file=self.stream or sys.stdout)
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass

    def cancel_backstop(self):
        if not self.backstop:
            return
        self.armed.append(None)
        try:
            faulthandler.cancel_dump_traceback_later()
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass

    @contextmanager
    def op(self, label, extra=0.0):
        if not self.seconds:
            yield
            return
        budget = self.seconds + extra
        with self.lock:
            outer = (self.label, self.deadline)
            self.label, self.deadline = label, self.clock() + budget
        self.arm_backstop(budget)
        try:
            yield
        finally:
            with self.lock:
                self.label, self.deadline = outer
                remaining = None if outer[1] is None else max(outer[1] - self.clock(), 0.0)
            if remaining is None:
                self.cancel_backstop()
            else:
                self.arm_backstop(remaining)

    def check(self):
        with self.lock:
            label, deadline = self.label, self.deadline
        if label is None or self.clock() < deadline:
            return False
        self.fired = True
        stream = self.stream or sys.stdout
        try:
            stream.write('WATCHDOG: %r did not return within %ss; exiting 3 (docker rm -f, then tt-smi -r card M '
                         'only, then a passing stock smoke call)\n' % (label, self.seconds))
            stream.flush()
            if self.on_fire is not None:
                self.on_fire(label)
        finally:
            self.exit(3)
        return True

    def poll(self):
        while not self.check():
            time.sleep(0.5)


WATCHDOG = Watchdog(0)
COMPILE_GRACE_S = 780.0     # a program's first call JIT-compiles its kernels on a fresh cache


class NativeLog:
    """fds 1 and 2 (tt-logger's sinks) to a file; Python's own prints keep the original stdout."""

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


# ---------------------------------------------------------------------------------------------
# Device part: card M only.
# ---------------------------------------------------------------------------------------------

class Bench:
    """One device: the K/V pools per seed, page tables per (seed, table, width), queries per
    (rows, seed, variant, qmem), chunk_start tensors per start; and the call itself."""

    def __init__(self, ttnn, torch, device, requested=None):
        self.ttnn, self.torch, self.device = ttnn, torch, device
        self.requested = requested if requested is not None else set()
        self.built = set()      # program_key() of every call that returned: those programs are cached
        self.pools, self.tables, self.queries, self.starts = {}, {}, {}, {}
        grid = device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)

    def upload(self, host, dtype, memory='dram', layout=None):
        ttnn = self.ttnn
        config = ttnn.L1_MEMORY_CONFIG if memory == 'l1' else ttnn.DRAM_MEMORY_CONFIG
        layout = layout or (ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT)
        with WATCHDOG.op('upload %s' % (tuple(host.shape),), extra=COMPILE_GRACE_S):
            return ttnn.from_torch(host, dtype=dtype, layout=layout, device=self.device, memory_config=config)

    def pool(self, seed, dtype='bf8'):
        key = (seed, dtype)
        if key not in self.pools:
            generator = self.torch.Generator().manual_seed(1000 + seed)
            shape = (POOL_BLOCKS, NKV, BLOCK, HD)
            k = self.torch.randn(shape, generator=generator).to(self.torch.bfloat16)
            v = self.torch.randn(shape, generator=generator).to(self.torch.bfloat16)
            kind = self.ttnn.bfloat8_b if dtype == 'bf8' else self.ttnn.bfloat16
            self.pools[key] = (self.upload(k, kind), self.upload(v, kind))
        return self.pools[key]

    def dense_kv(self, rows, seed):
        """Non-paged (1, NKV, rows, HD) K/V for the non-causal refusal (a valid plain SDPA call)."""
        key = ('dense', rows, seed)
        if key not in self.pools:
            generator = self.torch.Generator().manual_seed(4000 + seed)
            k = self.torch.randn((1, NKV, rows, HD), generator=generator).to(self.torch.bfloat16)
            v = self.torch.randn((1, NKV, rows, HD), generator=generator).to(self.torch.bfloat16)
            self.pools[key] = (self.upload(k, self.ttnn.bfloat8_b), self.upload(v, self.ttnn.bfloat8_b))
        return self.pools[key]

    def table(self, seed, kind, width):
        key = (seed, kind, width)
        if key not in self.tables:
            if kind == 'perm':
                generator = self.torch.Generator().manual_seed(2000 + seed)
                row = self.torch.randperm(POOL_BLOCKS, generator=generator)[:width]
            else:
                row = self.torch.arange(width)
            self.tables[key] = self.upload(row.to(self.torch.int32).reshape(1, width), self.ttnn.int32)
        return self.tables[key]

    def query(self, rows, seed, variant, qmem):
        key = (rows, seed, variant, qmem)
        if key not in self.queries and qmem == 'l1':
            # At most one L1 Q alive: 27 of them (up to 6.7 MB each) would not fit beside the SDPA CBs.
            for old in [other for other in self.queries if other[3] == 'l1']:
                self.ttnn.deallocate(self.queries.pop(old))
        if key not in self.queries:
            generator = self.torch.Generator().manual_seed(3000 + 17 * seed + rows)
            host = self.torch.randn((1, NH, rows, HD), generator=generator)
            if variant == 'peaky':
                host = host * 8
            elif variant == 'zeroq':
                host = host * 0
            self.queries[key] = self.upload(host.to(self.torch.bfloat16), self.ttnn.bfloat8_b, qmem)
        return self.queries[key]

    def start(self, value):
        if value not in self.starts:
            self.starts[value] = self.upload(self.torch.tensor([value], dtype=self.torch.int32), self.ttnn.int32)
        return self.starts[value]

    def config(self, program_word=None):
        options = dict(compute_with_storage_grid_size=self.grid, exp_approx_mode=False, q_chunk_size=Q_CHUNK,
                       k_chunk_size=Q_CHUNK)
        if program_word is not None:
            options['max_cores_per_head_batch'] = program_word
        return self.ttnn.SDPAProgramConfig(**options)

    def compute(self, fp32_dest=True):
        return self.ttnn.WormholeComputeKernelConfig(math_fidelity=self.ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                                     fp32_dest_acc_en=fp32_dest, packer_l1_acc=True)

    @staticmethod
    def program_key(rows, width, qmem, program_word=None, *, legacy_start=False, causal=True, fp32_dest=True,
                    kv_bf16=False):
        """What selects a device program here: the tensor specs (rows, page width, Q memory, K/V dtype),
        the path (causal, legacy int start), the compute config and the program word. Table, seed, Q
        values and chunk_start are data of one program."""
        return (rows, width, qmem, program_word, causal, legacy_start, fp32_dest, kv_bf16)

    def call(self, rows, width, table, qmem, seed, variant, start, program_word=None, *, legacy_start=False,
             causal=True, fp32_dest=True, kv_bf16=False, record=True, label=None, grace=COMPILE_GRACE_S):
        ttnn = self.ttnn
        k, v = self.pool(seed, 'bf16' if kv_bf16 else 'bf8')
        q = self.query(rows, seed, variant, qmem)
        pages = self.table(seed, table, width)
        if record and program_word is not None and (program_word & 0xFFFF0000) == PF_TAG:
            self.requested.add((rows, width, qmem, program_word & 0xFFFF))
        key = self.program_key(rows, width, qmem, program_word, legacy_start=legacy_start, causal=causal,
                               fp32_dest=fp32_dest, kv_bf16=kv_bf16)
        what = label or ('sdpa r%d w%d %s %s start=%d word=%s' % (
            rows, width, table, qmem, start, 'served' if program_word is None else hex(program_word)))
        with WATCHDOG.op(what, extra=grace):
            if not causal:
                dense_k, dense_v = self.dense_kv(rows, seed)
                out = ttnn.transformer.scaled_dot_product_attention(
                    q, dense_k, dense_v, is_causal=False, program_config=self.config(program_word),
                    compute_kernel_config=self.compute(fp32_dest))
            else:
                common = dict(input_tensor_q=q, input_tensor_k=k, input_tensor_v=v, page_table_tensor=pages,
                              compute_kernel_config=self.compute(fp32_dest), program_config=self.config(program_word))
                if legacy_start:
                    out = ttnn.transformer.chunked_scaled_dot_product_attention(chunk_start_idx=start, **common)
                else:
                    out = ttnn.transformer.chunked_scaled_dot_product_attention(
                        chunk_start_idx_tensor=self.start(start), **common)
        self.built.add(key)
        return out

    def host(self, tensor, label='read back'):
        with WATCHDOG.op(label):
            result = self.ttnn.to_torch(tensor)
        self.ttnn.deallocate(tensor)
        return result

    def sha(self, *args, **kwargs):
        return digest(self.torch, self.host(self.call(*args, **kwargs)))

    def cache_entries(self):
        counter = getattr(self.device, 'num_program_cache_entries', None)
        return counter() if callable(counter) else None


def sweep(bench, args, report):
    """reference: record every served sha. candidate: served == reference, every flag == reference."""
    failures = report['failures']
    reference = report.get('_reference') or {}
    for rows, width_kind, table, qmem, seed, variant in groups(args):
        validate_starts(args.starts, rows)
        width = page_width(width_kind, rows, args.starts)
        group = group_label(rows, width_kind, table, qmem, seed, variant)
        for start in args.starts:
            label = case_label(group, start)
            served = bench.sha(rows, width, table, qmem, seed, variant, start)
            if args.role == 'reference':
                report['reference_sha256'][label] = served
                continue
            row = dict(case=label, served=served, served_matches_reference=None, flags={})
            if label in reference:
                row['served_matches_reference'] = served == reference[label]
                if served != reference[label]:
                    failures.append('%s: the graft served output differs from the stock reference (Q1.6)' % label)
            elif args.reference:
                failures.append('%s: not in the reference report %s (Q1.6 unchecked for this case; take a '
                                'reference that covers the matrix)' % (label, args.reference))
            for flags in args.flags:
                got = bench.sha(rows, width, table, qmem, seed, variant, start, word(flags))
                expected = reference.get(label, served)
                row['flags']['%#x' % flags] = got == expected
                if got != expected:
                    failures.append('%s: chain %#x output differs from the served reference (Q1.5)' % (label, flags))
            report['cases'].append(row)
        print('group %s: %d starts done, %d failures so far' % (group, len(args.starts), len(failures)), flush=True)


def mutation(bench, args, report):
    """Q1.4 M-A: flags 0x103 (the injector reads the other KV head for k_chunk 0) at rows 2048,
    start 2048: every one of the 192 (head, q) units must differ from the served output."""
    if os.environ.get(TEST_ENV) != '1':
        report['failures'].append('mutation M-A needs %s=1 in the container (run_card_m_pf.sh sets it)' % TEST_ENV)
        return
    shape = (2048, page_width('fit', 2048, args.starts), 'perm', 'dram', 0, 'normal', 2048)
    served = unit_digests(bench.torch, bench.host(bench.call(*shape)))
    mutated = unit_digests(bench.torch, bench.host(bench.call(*shape, word(MUTATION_FLAGS))))
    passed, differing, message = mutation_verdict(served, mutated)
    report['mutation'] = dict(flags='%#x' % MUTATION_FLAGS, differing_units=differing, units=len(served), passed=passed,
                              message=message)
    print('mutation M-A: %s' % message, flush=True)
    if not passed:
        report['failures'].append('mutation M-A: %s' % message)


def cache_checks(bench, args, report):
    """Q1.7, run FIRST in the candidate role (before the sweep builds these programs): served then
    chain at one fresh shape adds exactly one program (the default hash covers
    max_cores_per_head_batch); the chain across chunk_start adds none. Inputs are uploaded first so
    only the SDPA programs move the count."""
    shape = (2048, page_width('fit', 2048, args.starts), 'perm', 'dram', 0, 'normal')
    bench.pool(0)
    bench.table(0, 'perm', shape[1])
    bench.query(2048, 0, 'normal', 'dram')
    for start in (0, 2048, 126976, 128):
        if blocks_needed(start, 2048) <= shape[1]:
            bench.start(start)
    before = bench.cache_entries()
    bench.host(bench.call(*shape, 0))
    after_served = bench.cache_entries()
    bench.host(bench.call(*shape, 0, word(DEFAULT_FLAGS)))
    after_chain = bench.cache_entries()
    for start in (2048, 126976, 128):
        if blocks_needed(start, 2048) <= shape[1]:
            bench.host(bench.call(*shape, start, word(DEFAULT_FLAGS)))
    after_starts = bench.cache_entries()
    result = dict(before=before, after_served=after_served, after_chain=after_chain, after_starts=after_starts)
    report['cache'] = result
    print('program cache %r' % (result,), flush=True)
    if None in result.values():
        report['failures'].append('the device has no num_program_cache_entries(): Q1.7 hash coverage unproven')
        return
    if after_chain != after_served + 1:
        report['failures'].append('Q1.7: the chain word did not key its own program (cache %d -> %d)'
                                  % (after_served, after_chain))
    if after_starts != after_chain:
        report['failures'].append('Q1.7: a chunk_start change rebuilt the chain program (cache %d -> %d)'
                                  % (after_chain, after_starts))


def refusals(bench, args, report):
    """Q1.8: every refusal is a TT_FATAL raised at program build (a Python exception), never a hang."""
    results = {}
    shape = (2048, page_width('fit', 2048, args.starts), 'perm', 'dram', 0, 'normal', 2048)
    for name, change, needle in REFUSALS:
        change = dict(change)
        program_word = change.pop('word', word(DEFAULT_FLAGS))
        clear = change.pop('clear_test_env', False)
        key = bench.program_key(shape[0], shape[1], shape[3], program_word, **change)
        if key in bench.built:
            # The factory (F1's environment check, F4's envelope) runs only on a program-cache miss:
            # a cached program would be reused and prove nothing either way.
            results[name] = dict(refused=False, matched=False, message='untestable: program %r is already cached '
                                 'in this process (run the refusals before anything builds it)' % (key,))
            print('refusal %-40s %s' % (name, results[name]), flush=True)
            report['failures'].append('Q1.8: %s: %s' % (name, results[name]['message']))
            continue
        saved = os.environ.pop(TEST_ENV, None) if clear else None
        try:
            out = bench.call(*shape, program_word, record=False, label='refusal %s' % name, **change)
        except Exception as error:  # noqa: BLE001 - the TT_FATAL surfaces as a RuntimeError
            text = str(error)
            results[name] = dict(refused=True, matched=needle in text, message=text[:400])
        else:
            bench.ttnn.deallocate(out)
            results[name] = dict(refused=False, matched=False, message='returned an output')
        finally:
            if saved is not None:
                os.environ[TEST_ENV] = saved
        print('refusal %-40s %s' % (name, results[name]), flush=True)
        if not (results[name]['refused'] and results[name]['matched']):
            report['failures'].append('Q1.8: %s was not refused by its [QWEN-SDPA-PF] TT_FATAL: %s'
                                      % (name, results[name]['message']))
    report['refusals'] = results


def stress(bench, args, report):
    """Q1.9 part 1: alternating served / chain calls at random starts, each equal to the served sha of
    its start (taken first, eagerly)."""
    shape = (2048, page_width('fit', 2048, args.starts), 'perm', 'dram', 0, 'normal')
    expected = {start: bench.sha(*shape, start) for start in args.starts}
    rng = random.Random(7)
    drift = []
    for index in range(args.alternations):
        start = rng.choice(args.starts)
        program_word = None if index % 2 == 0 else word(DEFAULT_FLAGS)
        got = bench.sha(*shape, start, program_word, record=program_word is not None)
        if got != expected[start]:
            drift.append((index, start, 'served' if program_word is None else hex(program_word)))
    report['stress'] = dict(calls=args.alternations, drifted=len(drift), first=drift[:10])
    print('stress %d alternating calls: %d drifted' % (args.alternations, len(drift)), flush=True)
    if drift:
        report['failures'].append('Q1.9: %d of %d alternating calls drifted (first %r)' % (len(drift), args.alternations,
                                                                                          drift[0]))


def trace_replay(bench, args, report):
    """Q1.9 part 2: one captured chain call, replayed while copy_host_to_device_tensor rewrites its
    chunk_start tensor; every replay equals the eager chain output at that start."""
    ttnn, torch = bench.ttnn, bench.torch
    shape = (2048, page_width('fit', 2048, args.starts), 'perm', 'dram', 0, 'normal')
    starts = [s for s in TRACE_STARTS if blocks_needed(s, 2048) <= shape[1]]
    eager = {start: bench.sha(*shape, start, word(DEFAULT_FLAGS)) for start in starts}
    bench.requested.add((2048, shape[1], 'dram', DEFAULT_FLAGS))
    device_start = bench.upload(torch.tensor([starts[0]], dtype=torch.int32), ttnn.int32)
    k, v = bench.pool(0)
    q = bench.query(2048, 0, 'normal', 'dram')
    pages = bench.table(0, 'perm', shape[1])

    def launch():
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=q, input_tensor_k=k, input_tensor_v=v, page_table_tensor=pages,
            chunk_start_idx_tensor=device_start, compute_kernel_config=bench.compute(),
            program_config=bench.config(word(DEFAULT_FLAGS)))

    with WATCHDOG.op('trace warm', extra=COMPILE_GRACE_S):
        ttnn.deallocate(launch())
    trace = ttnn.begin_trace_capture(bench.device, cq_id=0)
    try:
        output = launch()
    finally:
        ttnn.end_trace_capture(bench.device, trace, cq_id=0)
    mismatches = []
    try:
        for replay in range(args.trace_replays):
            start = starts[replay % len(starts)]
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32,
                                                            layout=ttnn.ROW_MAJOR_LAYOUT), device_start)
            with WATCHDOG.op('trace replay %d' % replay):
                ttnn.execute_trace(bench.device, trace, cq_id=0, blocking=True)
            with WATCHDOG.op('trace read back'):
                got = digest(torch, ttnn.to_torch(output))
            if got != eager[start]:
                mismatches.append((replay, start))
    finally:
        ttnn.release_trace(bench.device, trace)
        ttnn.deallocate(output)
    report['trace'] = dict(replays=args.trace_replays, starts=starts, mismatches=len(mismatches), first=mismatches[:10])
    print('trace %d replays over starts %s: %d mismatches' % (args.trace_replays, starts, len(mismatches)), flush=True)
    if mismatches:
        report['failures'].append('Q1.9: %d trace replays differ from eager (first %r)' % (len(mismatches), mismatches[0]))


def planted_hang(bench, args, report):
    """Q1.3: must never return. Returning is the failure; the watchdog's exit 3 is the pass."""
    if os.environ.get(TEST_ENV) != '1':
        report['failures'].append('the planted hang needs %s=1 in the container' % TEST_ENV)
        return
    shape = (2048, page_width('fit', 2048, args.starts), 'perm', 'dram', 0, 'normal', 2048)
    # The served pre-check: card M runs and the inputs are uploaded (a watchdog on 'open device' or an
    # upload is then never mistaken for the planted hang), and compute + writer are JIT-warm (the
    # chain program shares their binaries), so the planted call compiles only the chain reader.
    served = bench.sha(*shape)
    report['hang'] = dict(precheck_served_sha256=served, armed=True, returned=False)
    print('%s: served pre-check returned (%s); flags %#x, rows 2048, start 2048; PASS = the watchdog fires on %r '
          'or %r (%ss)' % (HANG_ARMED, served[:16], HANG_FLAGS, HANG_CALL_LABEL, HANG_READBACK_LABEL, args.watchdog),
          flush=True)
    began = time.monotonic()
    out = bench.call(*shape, word(HANG_FLAGS), label=HANG_CALL_LABEL, grace=HANG_CALL_GRACE_S)
    print('planted hang: enqueued; reading back (must never return)', flush=True)
    bench.host(out, label=HANG_READBACK_LABEL)
    report['hang'].update(returned=True, seconds=time.monotonic() - began)
    report['failures'].append('Q1.3 / K-7: the planted hang RETURNED a result after %.1f s (a bounded wait fell '
                              'through): the design may not go near serving cards' % report['hang']['seconds'])


def run(args, report):
    import torch
    import ttnn

    options = dict(device_id=args.device_id, l1_small_size=24576)
    if args.role == 'candidate' and args.trace_replays:
        options['trace_region_size'] = args.trace_region_bytes
    with WATCHDOG.op('open device', extra=COMPILE_GRACE_S):
        device = ttnn.open_device(**options)
    try:
        path, sha, has_chain = loaded_binary() if args.maps else ('?', '?', None)
        report['binary'] = dict(path=path, sha256=sha, chain_factory=has_chain)
        print('binary %s sha256 %s chain_factory=%s' % (path, sha[:16], has_chain), flush=True)
        if args.role == 'reference' and has_chain:
            report['failures'].append('the reference role must run on the stock binary (it carries the chain factory)')
            return
        if args.role in ('candidate', 'hang') and has_chain is False:
            report['failures'].append('the loaded _ttnncpp.so lacks the [QWEN-SDPA-PF] factory: the graft is not mounted')
            return
        requested = set()
        report['_requested'] = requested
        bench = Bench(ttnn, torch, device, requested)
        if args.role == 'hang':
            planted_hang(bench, args, report)
            return
        if args.role == 'candidate' and not args.no_controls:
            cache_checks(bench, args, report)
        sweep(bench, args, report)
        if args.role == 'candidate':
            if not args.no_controls:
                refusals(bench, args, report)       # before M-A builds the 0x103 program the env refusal uses
            if not args.no_mutation:
                mutation(bench, args, report)
            if args.alternations:
                stress(bench, args, report)
            if args.trace_replays:
                trace_replay(bench, args, report)
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_list(text, cast=int):
    return [cast(value, 0) if cast is int else cast(value) for value in text.split(',') if value.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--role', choices=('reference', 'candidate', 'hang'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', help='a reference-role report whose served shas the candidate must reproduce')
    parser.add_argument('--image', default=None, help='the image this runs in (recorded; a --reference must match it)')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--rows', default=','.join(map(str, ROWS)))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)))
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--tables', default=','.join(TABLES))
    parser.add_argument('--widths', default=','.join(WIDTHS))
    parser.add_argument('--q-memory', default=','.join(Q_MEMORY))
    parser.add_argument('--flags', default=','.join('%#x' % flags for flags in PRODUCTION_FLAGS))
    parser.add_argument('--alternations', type=int, default=1000, help='Q1.9 alternating served/chain calls (0: skip)')
    parser.add_argument('--trace-replays', type=int, default=200, help='Q1.9 trace replays (0: skip)')
    parser.add_argument('--trace-region-bytes', type=int, default=16 << 20)
    parser.add_argument('--no-mutation', action='store_true', help='skip M-A (e.g. without QWEN_SDPA_PF_TEST=1)')
    parser.add_argument('--no-controls', action='store_true', help='skip the cache and refusal checks')
    parser.add_argument('--watchdog', type=float, default=120.0, help='seconds per device call; 0 off')
    parser.add_argument('--no-maps', dest='maps', action='store_false', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.rows = parse_list(args.rows)
    args.starts = parse_list(args.starts)
    args.seeds = parse_list(args.seeds)
    args.flags = parse_list(args.flags)
    args.variants = parse_list(args.variants, str)
    args.tables = parse_list(args.tables, str)
    args.widths = parse_list(args.widths, str)
    args.q_memory = parse_list(args.q_memory, str)
    for rows in args.rows:
        if rows % (2 * Q_CHUNK):
            parser.error('rows %d: the chain needs an even number of 128-row Q chunks' % rows)
        validate_starts(args.starts, rows)
    for values, known, name in ((args.variants, VARIANTS, 'variant'), (args.tables, TABLES, 'table'),
                                (args.widths, WIDTHS, 'width'), (args.q_memory, Q_MEMORY, 'Q memory')):
        if any(value not in known for value in values):
            parser.error('unknown %s in %r' % (name, values))
    if any(flags & ~0x7 or not flags & 0x1 for flags in args.flags):
        parser.error('--flags are production flags 0x1..0x7')
    if args.role == 'candidate' and not args.reference:
        print('no --reference: chain outputs are compared with this run\'s served outputs only (Q1.6 not checked)')
    return args


def narrow(args):
    """True when the matrix is not the full default one (a WATCHER=1 or hand-narrowed run): a narrow
    reference cannot serve a full candidate."""
    return (args.rows != list(ROWS) or args.starts != list(STARTS) or args.seeds != list(SEEDS)
            or args.variants != list(VARIANTS) or args.tables != list(TABLES) or args.widths != list(WIDTHS)
            or args.q_memory != list(Q_MEMORY))


def load_reference(path, image, report):
    """The reference report's served shas; a reference that did not pass, or ran in another image
    than this run's --image, is a failure (its shas would not be the stock outputs of this image)."""
    loaded = json.loads(Path(path).read_text())
    report['reference_image'] = loaded.get('image')
    report['reference_narrow'] = loaded.get('narrow')
    if loaded.get('passed') is not True:
        report['failures'].append('the reference report %s did not pass (passed=%r)' % (path, loaded.get('passed')))
    if image and loaded.get('image') != image:
        report['failures'].append('the reference report %s ran in image %r, this run in %r'
                                  % (path, loaded.get('image'), image))
    return loaded.get('reference_sha256') or {}


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(passed=False, role=args.role, rows=args.rows, starts=args.starts, seeds=args.seeds,
                  variants=args.variants, tables=args.tables, widths=args.widths, q_memory=args.q_memory,
                  flags=['%#x' % flags for flags in args.flags], reference=args.reference, failures=[], cases=[],
                  reference_sha256={}, test_env=os.environ.get(TEST_ENV), image=args.image, narrow=narrow(args))
    if args.reference:
        report['_reference'] = load_reference(args.reference, args.image, report)
    native = NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        payload['requested_programs'] = sorted('rows=%d width=%d q=%s flags=%#x' % key
                                               for key in report.get('_requested', ()))
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def on_fire(label):
        extra = dict(error='watchdog: %r exceeded %ss' % (label, args.watchdog), passed=False, watchdog=label)
        if args.role == 'hang':
            # Only the planted call or its read back is the planted hang; a stall anywhere before it (open
            # device, an upload, the served pre-check) is a sick card, not a qualified bounded wait.
            extra.update(passed=label in HANG_LABELS, hang=dict(report.get('hang') or {}, returned=False, watchdog=label))
        try:
            write_report(extra)
        except Exception:  # noqa: BLE001 - the WATCHDOG line stands
            pass

    if report['failures']:                      # a bad reference: refuse before touching card M
        write_report()
        for failure in report['failures']:
            print('FAIL', failure)
        print('SDPA_PF_CARD_M role=%s passed=False (refused before opening the device) report=%s' % (args.role, args.out))
        return 1
    WATCHDOG = Watchdog(args.watchdog, on_fire=on_fire).start()
    try:
        with native:
            run(args, report)
        if args.role == 'candidate':
            lines = factory_lines(native.text())
            report['factory_lines'] = lines
            for problem in check_factory_lines(lines, report.get('_requested', set())):
                report['failures'].append('factory log: ' + problem)
        report['passed'] = not report['failures'] and (bool(report['cases']) or bool(report['reference_sha256']))
    except Exception as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    if report.get('error'):
        print('ERROR', report['error'])
    print('SDPA_PF_CARD_M role=%s passed=%s cases=%d references=%d failures=%d report=%s native_log=%s' % (
        args.role, report['passed'], len(report['cases']), len(report['reference_sha256']), len(report['failures']),
        args.out, native.path))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
