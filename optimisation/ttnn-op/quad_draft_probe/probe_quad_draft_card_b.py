"""Q0: the card-B probe for Q4, one four-user 64-row draft pass (QWEN_FAST_QUAD_DRAFT; quad-draft-plan.md section 5).

WHAT IT DECIDES. Whether the quad pass can be byte-exact per user (E2) and pays enough to build. Every comparison
is bitwise on int views (int16 for bf16, int32 for fp32, int64 for indices) and carries a sensitivity control; a
case whose control does not fire is INCONCLUSIVE, never a pass.

Card B is one p150a, so no TP2 pass runs here: every case is one draft op, or the draft attention, called as the
served code calls it (the served modules are this checkout's, mounted beside the harness), on seeded operands of
the served per-chip shapes. The candidates under test are in quad_candidates.py (the quad K/V plan and fold, the
dense control mask, the one-chip conv programs) and quad_conv_io.cpp (the 64-row conv I/O kernel, E1/E1b), both
probe-local copies of what the plan would ship.

STEPS (--steps, default Q0a,Q0b,Q0c,Q0d,Q0e; 'cores' is its own pass, see below)
  Q0a  Row-local ops widened to 64 rows. Input X = [pair-A block ; pair-B block]. Each op on X against today's
       32-row call on each half, concatenated: the nine explicit matmul programs at per_core_M=2 (conv-kernel, q,
       k, v, o, gate, up, down, selector: the served grids, per_core_N, bf8/bf16 dtypes and per-chip shard shapes;
       --shards independent weight shards each, the two chips' shards by default), the hidden and head (q, k)
       rms_norms, rotary (q and k heads), create heads (with its k|v concat) and concat heads, typecast (both
       directions), add, silu, multiply and embedding. --seeds x --regimes (normal, peaked, negative, row0x100:
       tile row 0 x100). Per case: widened == halves; row swap: widened(swap X) == swap(widened X); control
       PERTURB: X with one element of tile row 1 negated must change the widened result (the op reads tile row 1,
       the comparison is live). Per matmul program, the plan's K-order control: today's call with in0_block_w=2
       against 4 must differ in at least one case (--matmul-control in0, the plan's rule; 'either' also accepts
       the K-split control: K halves on two matmuls and an fp32 add, run on shard 0, always recorded).
       Core counts come from the 'cores' pass (--cores-report); the plan wants the norms on 2 / 32 / 8 cores.
       Kill: a FAIL in q/k/v/o/gate/up/down stops Q4 (kill=STOP-Q4); any other FAIL keeps that op split per half.
  Q0b  The four-way fold (quad_candidates.quad_fold_attention: pair_row_exact's fold_query/unfold_output/
       folded_sdpa, unchanged, plus a head-axis concat) on the device-assembled 12-piece K/V (the plan's pads
       from rows 32p), against the pair fold (pair_row_exact.fold_attention on each pair's own assembly: the
       probe's C6 r1g) and against single-user (C0, draft_attention.draft_sdpa). 4 users x 16 heads x --seeds x
       --b-regimes (normal, peaked, negative, partner100: users 0 and 2 x100). C5: the assembled K/V against the
       two PAIR assemblies' host bytes concatenated (so it checks R2's pads, not the quad plan against itself).
       P: users 1 and 3 are bit-identical in partner100 and normal. Negative control: the
       dense, unfolded 4-segment call (mask (1, 1, 64, 8320)) must differ from single-user for users 1-3 in at
       least 80% of cases (the pair probe's: served row 1 differed 21/24).
  Q0c  Q0c-lite: the H0 candidate concat. Each chunk's top-16 values and uint16 indices of two 32-row halves
       (draft_shared_head.local_head_candidates, as served) concatenated on dim 2 on the device, against the
       halves read separately; the fallback casts the indices to uint32 first. Control: the halves concatenated
       in the other order must differ.
  Q0d  The conv at 64 rows: E1 (80 workers x 4 pages) and E1b (110 workers, 3/2 pages; the 11x10 grid) against
       two served 32-row calls, the users' seams at rows 0/16/32/48. Negative control: tile row 1's seam word
       without bit 16 (so row 48, user 3's first row, carries user 2's last) must differ, per case. Every output
       is NaN-filled before its call (Session.conv poison): an unwritten page can never read back a freed
       buffer's correct bytes.
  Q0e  Traced timing, arms interleaved: --rounds (>= 5) rounds x --replays (>= 20) replays of each arm, each
       capture holding --trace-reps repetitions. The quad attention layer (assembly, fold, SDPA, unfold) against
       two pair layers; each widened matmul, norm, rotary, head op, SwiGLU and small op against its two 32-row
       calls; E1b and E1 against two served conv calls. The plan's device-saving model (saving_model): per item
       (2 - s) x t32, t32 the V138 per-pass time and s = 2 t64 / t(32x2) from card B; the attention layer from
       card B directly (5 layers x (two pair layers - the quad layer)); CCL at s = 1.8; gaps 1 us per op over
       the plan's 1056 -> ~740 ops; the candidate concat -0.03. An item counts only as far as this run proved it
       (bounded_saving): saving_ms counts the items whose Q0a ops / Q0b / Q0d variant PASSED, saving_upper_ms also
       the unproven ones, and a FAILED item counts nothing in either (it stays split; a conv that is neither E1b
       nor E1 is C0, -0.4). Go at >= 5.0 ms/round proven; 3.5-5.0 proven only if E1b (Q0d) and Q0c-lite pass;
       stop (kill) when even the upper bound is below 3.5; otherwise INCONCLUSIVE.
       Both timing biases are known and left in: the per-replay host dispatch and sync (divided by --trace-reps)
       is added to both arms, which pulls s toward 2 on the few-us arms (conservative); the extra inter-op gaps of
       the 32x2 arms are in s and also in the modelled gaps item (optimistic, at most the 0.32 ms of that item).
  cores  With TT_METAL_DEVICE_PROFILER=1 (run_card_b.sh runs it as its own process first): each Q0a op once at
       64 rows and once at 32, the device profiler read around each call, the distinct cores per program from the
       raw device log. The main pass folds it into Q0a with --cores-report.

OUTPUT. One line per case, then one verdict line per step that ran:
  QUAD_PROBE step=Q0a verdict=PASS|FAIL|INCONCLUSIVE ...
and last, one JSON line (the summary: verdicts, kill, go, saving, report path). --out holds the full report.
A step with too little coverage for the plan (Q0a/Q0b: 3 seeds and all four regimes) is at best INCONCLUSIVE.

FAILURES (passed=False): the mapped _ttnncpp.so is not --expect-binary-sha256 (run_card_b.sh defaults it to
K64i, cf54d716); the SDPA sources under TT_METAL_HOME are not the T16 admission's; --expect-matmul-kernel-sha256
set and not met; the watchdog fired. The image's op-source trees are hashed and recorded (R1: the matmul kernel is
JIT-compiled from TT_METAL_HOME, the factory is in the .so), and a matmul compute kernel that is not the retained
BMM export is a warning.

RUN with run_card_b.sh only. The helpers above Watchdog import no ttnn and are tested on CPU by
test_quad_draft_probe.py, which also drives every step against a torch stand-in and injects the faults the
controls guard against. --decide-only REPORT re-reads a report and prints its verdicts (e.g. under
--matmul-control either).
"""

import argparse
from contextlib import contextmanager
import csv
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
# In the container the served modules sit beside the harness (/bench); in a checkout they are scripts/ci's.
CI = HERE.parents[2] / 'scripts' / 'ci' if len(HERE.parents) > 2 else None
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if CI is not None and CI.is_dir() and str(CI) not in sys.path:
    sys.path.append(str(CI))

import quad_candidates as quad  # noqa: E402

PROBE = 'Q0'
STEPS = ('Q0a', 'Q0b', 'Q0c', 'Q0d', 'Q0e')
REGIMES_A = ('normal', 'peaked', 'negative', 'row0x100')
REGIMES_B = ('normal', 'peaked', 'negative', 'partner100')
COVERAGE_SEEDS = 3                     # the plan's 3 seeds x 4 regimes for Q0a and Q0b
CONTEXT, BLOCK, SPAN = quad.CONTEXT, quad.BLOCK, quad.SPAN
HEADS, KV_HEADS, HEAD_DIM = quad.HEADS, quad.KV_HEADS, quad.HEAD_DIM
# The per-chip served widths (tests shrink them).
HIDDEN, INTERMEDIATE, Q_WIDTH, KV_WIDTH = 5120, 8704, 2048, 512
CONV_COLUMNS, SELECTOR_COLUMNS, EMBED_WIDTH = 1280, 256, 2560
LOGIT_SHARD = 124160
MATMUL_CONTROLS = ('in0', 'either')
KILL_OPS = ('matmul-q', 'matmul-k', 'matmul-v', 'matmul-o', 'matmul-gate', 'matmul-up', 'matmul-down')
NORM_CORES = {'rms_norm-hidden': 2, 'rms_norm-q': 32, 'rms_norm-k': 8}
DENSE_FIRE = 0.8
SAVING_GO, SAVING_STOP = 5.0, 3.5
MIN_ROUNDS, MIN_REPLAYS = 5, 20
# The plan's section 2.2 anchors: t32 per pass (V138, ms) and which timing arms give s. 'attention' is measured
# absolutely; 'ccl', 'gaps' and 'candidate_concat' are modelled as the plan says.
MODEL = {
    'mlp': dict(t32_ms=2.12, arms=('mm-gate', 'mm-up', 'mm-down')),
    'swiglu': dict(t32_ms=0.26, arms=('swiglu',)),
    'qkvo': dict(t32_ms=0.83, arms=('mm-q', 'mm-k', 'mm-v', 'mm-o')),
    'conv_projection': dict(t32_ms=0.46, arms=('mm-conv',)),
    'selector': dict(t32_ms=0.04, arms=('mm-selector',)),
    'hidden_norm': dict(t32_ms=0.89, arms=('norm-hidden',)),
    'head_norm_rope': dict(t32_ms=0.15, arms=('norm-q', 'norm-k', 'rotary-q', 'rotary-k')),
    'heads': dict(t32_ms=0.33, arms=('heads-create', 'heads-concat')),
    'small_ops': dict(t32_ms=0.55, arms=('small',)),
    'conv': dict(t32_ms=3.54, arms=('conv',)),
}
CCL_T32_MS, CCL_S = 0.22, 1.8
GAP_US, PAIR_OPS, QUAD_OPS = 1.0, 1056, 740
CANDIDATE_CONCAT_MS = -0.03
DRAFT_LAYERS = 5
# Which Q0a ops prove each timed item. The plan (section 5, Q0a): "any mismatch keeps that op split per half", so an
# item whose ops are not all PASS cannot be counted as saved; Q0b's fallback (QUAD_SDPA=pairs) saves nothing on the
# attention layer; the conv item is the variant Q0d proved, else C0's cost (R4, two 32-row calls per half).
ITEM_OPS = {
    'mlp': ('matmul-gate', 'matmul-up', 'matmul-down'),
    'swiglu': ('typecast-down', 'typecast-up', 'silu', 'multiply'),
    'qkvo': ('matmul-q', 'matmul-k', 'matmul-v', 'matmul-o'),
    'conv_projection': ('matmul-conv',),
    'selector': ('matmul-selector',),
    'hidden_norm': ('rms_norm-hidden',),
    'head_norm_rope': ('rms_norm-q', 'rms_norm-k', 'rotary-q', 'rotary-k'),
    'heads': ('create-heads', 'concat-heads'),
    'small_ops': ('typecast-down', 'typecast-up', 'add'),
}
C0_CONV_MS = -0.4
# The T16 admission's pins (dflash_t16_native_attention_gate.ORIGINAL): Q0b runs the draft SDPA, JIT-compiled from
# these sources.
SDPA = 'ttnn/cpp/ttnn/operations/transformer/sdpa/'
PINNED_SOURCES = {
    SDPA + 'device/sdpa_program_factory.cpp': 'fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783',
    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h':
        '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181',
    SDPA + 'device/kernels/compute/compute_common.hpp': '3fb5da2440c3bf90ebceb8acd55424c7739339de4c6b02db83836e1e3414fa19',
    SDPA + 'device/kernels/compute/sdpa.cpp': 'a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0',
}
# R1: the matmul kernel is JIT-compiled from the image's TT_METAL_HOME; the plan's argument read a retained export
# (t16-matched runner-evidence 35092212895). A differing image kernel is a warning (risk 12), a pin a failure.
OPS_ROOT = 'ttnn/cpp/ttnn/operations/'
BMM_KERNEL = OPS_ROOT + 'matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp'
BMM_EVIDENCE_SHA256 = 'ef0d62e39d6d72d95eb023dc74000b1b501f46aeb9c6bfeb9d8952333c2bef88'
OP_TREES = ('matmul', 'normalization', 'experimental/transformer', 'embedding', 'eltwise/unary', 'eltwise/binary',
            'copy', 'data_movement/concat', 'data_movement/slice', 'reduction/topk', 'generic')
K64I_SHA256 = 'cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4'
# The served modules the cases call; run_card_b.sh mounts this checkout's copies beside the harness.
MODULES = ('pair_row_exact', 'draft_attention', 'dflash_batched_mask', 'draft_shared_head', 'draft_head_preparation',
           'dflash_t16_native_attention', 'draft_mlp', 'quad_candidates')
KERNEL_FILES = ('quad_conv_io.cpp', 'draft_convolution_fused_io.cpp', 'draft_convolution_fused_compute.cpp')
OPEN_EXTRA_S = 600.0
CORES_GAP_S = 1.0          # the cores pass's pause between measured calls (cores_from_log groups runs by it)
ENV_RECORDED = ('TT_METAL_HOME', 'TT_METAL_CACHE', 'TT_METAL_WATCHER', 'TT_METAL_DEVICE_PROFILER',
                'TT_METAL_PROFILER_DIR', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS')

clock = time.perf_counter    # module level so the CPU dry run can drive a model clock


# ---------------------------------------------------------------------------------------------
# Pure helpers (no ttnn).
# ---------------------------------------------------------------------------------------------

def matmul_table():
    """The nine explicit-config matmul programs: K, N, grid, per_core_N, weight dtype ('proj' follows
    --projection-dtype: QWEN_FAST_DRAFT_BF8=1 in every current arm)."""
    return {
        'matmul-conv': dict(width=HIDDEN, columns=CONV_COLUMNS, grid=(8, 5), per_core_n=1, dtype='bf16'),
        'matmul-q': dict(width=HIDDEN, columns=Q_WIDTH, grid=(8, 8), per_core_n=1, dtype='proj'),
        'matmul-k': dict(width=HIDDEN, columns=KV_WIDTH, grid=(8, 8), per_core_n=1, dtype='proj'),
        'matmul-v': dict(width=HIDDEN, columns=KV_WIDTH, grid=(8, 8), per_core_n=1, dtype='proj'),
        'matmul-o': dict(width=Q_WIDTH, columns=HIDDEN, grid=(8, 10), per_core_n=2, dtype='proj'),
        'matmul-gate': dict(width=HIDDEN, columns=INTERMEDIATE, grid=(8, 10), per_core_n=4, dtype='proj'),
        'matmul-up': dict(width=HIDDEN, columns=INTERMEDIATE, grid=(8, 10), per_core_n=4, dtype='proj'),
        'matmul-down': dict(width=INTERMEDIATE, columns=HIDDEN, grid=(8, 10), per_core_n=2, dtype='proj'),
        'matmul-selector': dict(width=HIDDEN, columns=SELECTOR_COLUMNS, grid=(8, 1), per_core_n=1, dtype='bf16'),
    }


def op_inputs():
    """Q0a: {op: (family, [(64-row shape, kind)])}. kind: bf16, fp32 (full), fp32r (fp32 of bf16 values, as the
    served typecasts feed them), ids (uint32 row-major, rows on the last axis). Rows sit on axis -2 otherwise."""
    table = {name: ('matmul', [((1, 1, 64, spec['width']), 'bf16')]) for name, spec in matmul_table().items()}
    table.update({
        'rms_norm-hidden': ('norm', [((1, 1, 64, HIDDEN), 'bf16')]),
        'rms_norm-q': ('norm', [((1, HEADS, 64, HEAD_DIM), 'bf16')]),
        'rms_norm-k': ('norm', [((1, KV_HEADS, 64, HEAD_DIM), 'bf16')]),
        'rotary-q': ('rotary', [((1, HEADS, 64, HEAD_DIM), 'fp32r'), ((1, 1, 64, HEAD_DIM), 'cos'),
                                ((1, 1, 64, HEAD_DIM), 'sin')]),
        'rotary-k': ('rotary', [((1, KV_HEADS, 64, HEAD_DIM), 'fp32r'), ((1, 1, 64, HEAD_DIM), 'cos'),
                                ((1, 1, 64, HEAD_DIM), 'sin')]),
        'create-heads': ('heads', [((1, 1, 64, HEADS * HEAD_DIM), 'bf16'), ((1, 1, 64, KV_HEADS * HEAD_DIM), 'bf16'),
                                   ((1, 1, 64, KV_HEADS * HEAD_DIM), 'bf16')]),
        'concat-heads': ('heads', [((1, HEADS, 64, HEAD_DIM), 'bf16')]),
        'typecast-down': ('eltwise', [((1, 1, 64, HIDDEN), 'fp32')]),
        'typecast-up': ('eltwise', [((1, 1, 64, INTERMEDIATE), 'bf16')]),
        'add': ('eltwise', [((1, 1, 64, HIDDEN), 'fp32r'), ((1, 1, 64, HIDDEN), 'fp32r')]),
        'silu': ('eltwise', [((1, 1, 64, INTERMEDIATE), 'fp32r')]),
        'multiply': ('eltwise', [((1, 1, 64, INTERMEDIATE), 'fp32r'), ((1, 1, 64, INTERMEDIATE), 'fp32r')]),
        'embedding': ('embedding', [((1, 64), 'ids')]),
    })
    return table


def op_names():
    return list(op_inputs())


def row_axis(kind):
    return -1 if kind == 'ids' else -2


def bf16_round(torch, value):
    return value.bfloat16().float()


def draw_inputs(torch, name, seed, regime, *, vocab=8192):
    """One Q0a case's host inputs at 64 rows. The draws are fixed per (op, seed); a regime only transforms input 0
    afterwards: peaked x4, negative -|x|, row0x100 tile row 0 x100 (ids: repeated, high, and one repeated id in
    tile row 0). RoPE tables are two pairs' real tables (rope_tables at two positions), concatenated."""
    from draft_head_preparation import rope_tables

    family, inputs = op_inputs()[name]
    generator = torch.Generator().manual_seed(1000003 * (seed + 1) + 7919 * op_names().index(name))
    values = []
    for index, (shape, kind) in enumerate(inputs):
        if kind == 'ids':
            values.append(torch.randint(0, vocab, shape, generator=generator, dtype=torch.int64))
        elif kind in ('cos', 'sin'):
            start = 4096 * (seed + 1)
            tables = [rope_tables(start + 64 * half, 32) for half in range(2)]
            part = 0 if kind == 'cos' else 1
            values.append(torch.cat([tables[0][part], tables[1][part]], dim=2).float())
        else:
            value = torch.randn(*shape, generator=generator)
            values.append(value if kind == 'fp32' else bf16_round(torch, value))
    first, kind = values[0], inputs[0][1]
    if kind == 'ids':
        if regime == 'peaked':
            first = first % 8
        elif regime == 'negative':
            first = vocab - 1 - (first % 64)
        elif regime == 'row0x100':
            first = first.clone()
            first[..., :32] = first[..., :1]
    elif regime == 'peaked':
        first = first * 4
    elif regime == 'negative':
        first = -first.abs()
    elif regime == 'row0x100':
        first = first.clone()
        first[..., :32, :] = first[..., :32, :] * 100
    if kind in ('bf16', 'fp32r'):
        first = bf16_round(torch, first)
    values[0] = first
    return values


def draw_weights(torch, name, seed, shard):
    """A matmul program's weight shard, (K, N) scaled by K^-0.5, drawn per (program, seed, shard)."""
    spec = matmul_table()[name]
    generator = torch.Generator().manual_seed(2000003 * (seed + 1) + 104729 * (shard + 1) + op_names().index(name))
    return (torch.randn(spec['width'], spec['columns'], generator=generator) * spec['width'] ** -0.5).bfloat16()


def take_rows(value, kind, half):
    axis = row_axis(kind)
    size = value.shape[axis] // 2
    return value.narrow(axis, half * size, size).contiguous()


def swap_rows(value, kind):
    """The two 32-row halves exchanged (pair B's block first)."""
    axis = row_axis(kind)
    return _swap(value, axis, value.shape[axis] // 2)


def _swap(value, axis, size):
    import torch

    axis = axis % value.dim()
    return torch.cat([value.narrow(axis, size, size), value.narrow(axis, 0, size)], dim=axis).contiguous()


def perturb(torch, value, kind, *, vocab=8192):
    """Input 0 with one element of tile row 1 changed: negated (1.0 if it is zero); an id moved by one."""
    out = value.clone()
    if kind == 'ids':
        out[..., 32 + 5] = (out[..., 32 + 5] + 1) % vocab
        return out
    index = (0,) * (out.dim() - 2) + (32 + 5, 7)
    out[index] = -out[index] if float(out[index]) != 0 else 1.0
    return out


def bits(torch, tensor):
    """The comparison view: int32 for fp32, int16 for bf16/fp16, int64 for integer tensors."""
    tensor = tensor.contiguous()
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.view(torch.int16)
    if tensor.is_floating_point():
        return tensor.to(torch.float32).view(torch.int32)
    return tensor.to(torch.int64)


def compare(torch, actual, expected):
    """Bitwise: equal, differing elements, total, max_abs (float)."""
    if tuple(actual.shape) != tuple(expected.shape):
        return dict(equal=False, differing=None, total=None, max_abs=None,
                    shapes=[list(actual.shape), list(expected.shape)])
    if actual.dtype != expected.dtype and (actual.is_floating_point() or expected.is_floating_point()):
        return dict(equal=False, differing=None, total=int(actual.numel()), max_abs=None,
                    dtypes=[str(actual.dtype), str(expected.dtype)])
    differing = int((bits(torch, actual) != bits(torch, expected)).sum())
    difference = (actual.double() - expected.double()).abs()
    finite = torch.isfinite(difference)
    return dict(equal=differing == 0, differing=differing, total=int(actual.numel()),
                max_abs=float(difference[finite].max()) if bool(finite.any()) else None)


def compare_all(torch, actual, expected):
    """compare() over parallel lists of outputs, folded into one result."""
    results = [compare(torch, left, right) for left, right in zip(actual, expected)]
    if len(actual) != len(expected) or not results:
        return dict(equal=False, differing=None, total=None, max_abs=None)
    return dict(equal=all(result['equal'] for result in results),
                differing=None if any(result['differing'] is None for result in results)
                else sum(result['differing'] for result in results),
                total=None if any(result['total'] is None for result in results)
                else sum(result['total'] for result in results),
                max_abs=max((result['max_abs'] for result in results if result['max_abs'] is not None), default=None))


def digest(torch, tensor):
    return hashlib.sha256(bits(torch, tensor).numpy().tobytes()).hexdigest()


def summary(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return dict(median_us=statistics.median(ordered), min_us=ordered[0], mean_us=statistics.mean(ordered),
                stdev_us=statistics.stdev(ordered) if len(ordered) > 1 else 0.0, n=len(ordered))


def build_quad_fixture(torch, seed, regime):
    """Q0b's operands, bf16, for users 0-3: a 2048-row cache K/V (1, 4, 2048, 128), 16 live and 16 pad K/V rows (a
    single-user trace's live block is [live | pad]), 16 query and 16 pad query rows. The draws are in a fixed order
    and a regime only transforms them afterwards: peaked queries x4; negative queries |x| and cache keys 0-31 -|x|;
    partner100 users 0 and 2 (each pair's row 0) K/V x100, so users 1 and 3 keep their bits."""
    if regime not in REGIMES_B:
        raise ValueError('unknown regime %r' % (regime,))
    generator = torch.Generator().manual_seed(seed)
    draw = lambda *shape: torch.randn(*shape, generator=generator)
    users = []
    for _ in range(quad.USERS):
        users.append(dict(
            cache={name: draw(1, KV_HEADS, CONTEXT, HEAD_DIM) for name in 'kv'},
            live={name: draw(1, KV_HEADS, BLOCK, HEAD_DIM) for name in 'kv'},
            pad={name: draw(1, KV_HEADS, BLOCK, HEAD_DIM) for name in 'kv'},
            query=draw(1, HEADS, BLOCK, HEAD_DIM), query_pad=draw(1, HEADS, BLOCK, HEAD_DIM)))
    for index, entry in enumerate(users):
        if regime == 'peaked':
            entry['query'], entry['query_pad'] = entry['query'] * 4, entry['query_pad'] * 4
        elif regime == 'negative':
            entry['query'] = entry['query'].abs()
            entry['cache']['k'][:, :, :32] = -entry['cache']['k'][:, :, :32].abs()
        elif regime == 'partner100' and index in (0, 2):
            for part in ('cache', 'live', 'pad'):
                entry[part] = {name: value * 100 for name, value in entry[part].items()}
    for entry in users:
        for part in ('cache', 'live', 'pad'):
            entry[part] = {name: value.bfloat16() for name, value in entry[part].items()}
        entry['query'], entry['query_pad'] = entry['query'].bfloat16(), entry['query_pad'].bfloat16()
    return dict(seed=seed, regime=regime, users=users)


def single_operands(torch, fixture, user):
    """User u's single-user trace: its rows then its pad rows; K = [cache | live | pad] (2080)."""
    entry = fixture['users'][user]
    query = torch.cat([entry['query'], entry['query_pad']], dim=2)
    keys = {name: torch.cat([entry['cache'][name], entry['live'][name], entry['pad'][name]], dim=2) for name in 'kv'}
    return query, keys['k'], keys['v']


def pair_operands(torch, fixture, pair):
    """Pair p's served operands (users 2p, 2p + 1): the 32-row query and live block, the two caches, the
    assembly plan (key_value_plan) and its host concatenation."""
    from dflash_batched_mask import key_value_plan

    first, second = (fixture['users'][user] for user in (2 * pair, 2 * pair + 1))
    query = torch.cat([first['query'], second['query']], dim=2)
    block = {name: torch.cat([first['live'][name], second['live'][name]], dim=2) for name in 'kv'}
    caches = [first['cache'], second['cache']]
    plan, _, key_rows = key_value_plan([CONTEXT, CONTEXT], BLOCK)
    if key_rows != 2 * SPAN:
        raise AssertionError('the pair plan must cover 4160 keys')
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
    return dict(query=query, block=block, caches=caches, plan=plan, expected=expected)


def quad_operands(torch, fixture):
    """The quad's operands: the 64-row query and live block and the four caches. `expected` is what C5 holds
    the device assembly to: the two PAIR assemblies' host bytes, concatenated (pair 0's segments, then pair 1's),
    so C5 checks the plan's pads (R2) as well as the data movement - not the quad plan against itself."""
    users = fixture['users']
    query = torch.cat([entry['query'] for entry in users], dim=2)
    block = {name: torch.cat([entry['live'][name] for entry in users], dim=2) for name in 'kv'}
    caches = [entry['cache'] for entry in users]
    pairs = [pair_operands(torch, fixture, pair)['expected'] for pair in range(2)]
    expected = {name: torch.cat([pairs[0][name], pairs[1][name]], dim=2).contiguous() for name in 'kv'}
    return dict(query=query, block=block, caches=caches, expected=expected)


def single_mask():
    from dflash_batched_mask import batched_attention_mask

    return batched_attention_mask([CONTEXT], BLOCK)


def user_rows(tensor, user):
    """User u's 16 rows of a (..., 64, D) quad output (or of a 32-row output, u in 0-1)."""
    return tensor[..., BLOCK * user:BLOCK * (user + 1), :]


def conv_reference(torch, hidden, dynamic, base, seams):
    """The fused conv kernel's arithmetic on the host, per tile row: shifted = the row above within the tile row
    unless the row's seam bit is set; out = four bf16-rounded fp32 terms (base0 * hidden, dynamic0 * hidden,
    base1 * shifted, dynamic1 * shifted) added in that order, each sum rounded to bf16. `seams` is one word per
    tile row. Rows beyond the block's rows are zero."""
    rows = hidden.shape[2]
    parts = []
    for tile_row in range((rows + 31) // 32):
        start, stop = 32 * tile_row, min(rows, 32 * tile_row + 32)
        value = hidden[..., start:stop, :].float()
        shifted = torch.zeros_like(value)
        for row in range(1, stop - start):
            if not (seams[tile_row] >> row) & 1:
                shifted[..., row, :] = value[..., row - 1, :]
        expand = [part[..., start:stop, :].float().repeat_interleave(16, dim=-1) for part in dynamic]
        bases = [part.float().expand_as(value) for part in base]
        out = torch.zeros_like(value)
        for term in (bases[0] * value, expand[0] * value, bases[1] * shifted, expand[1] * shifted):
            out = (out + term.bfloat16().float()).bfloat16().float()
        parts.append(out)
    return torch.cat(parts, dim=2).bfloat16()


# ---- verdicts -------------------------------------------------------------------------------

def _count(flags):
    flags = list(flags)
    return '%d/%d' % (sum(1 for flag in flags if flag is True), len(flags))


def _coverage(report, seeds_key, regimes_key, regimes):
    seeds = report.get(seeds_key) or []
    ran = report.get(regimes_key) or []
    return len(set(seeds)) >= COVERAGE_SEEDS and set(regimes) <= set(ran)


def norm_cores(report):
    """{'hidden': n, 'q': n, 'k': n} from the cores pass (the widened call's largest program), or None."""
    cores = report.get('cores') or {}
    if cores.get('_status') != 'measured':
        return None
    out = {}
    for name, short in (('rms_norm-hidden', 'hidden'), ('rms_norm-q', 'q'), ('rms_norm-k', 'k')):
        counts = (cores.get(name) or {}).get('widened') or []
        out[short] = max(counts) if counts else None
    return out


def decide_q0a(report, policy='in0'):
    entries = [case for case in report.get('cases', []) if case.get('step') == 'Q0a']
    names = [name for name in op_names() if any(case.get('op') == name for case in entries)]
    per_op = {}
    for name in names:
        cases = [case for case in entries if case['op'] == name]
        good = [case for case in cases if not case.get('error')]
        errors = len(cases) - len(good)
        mismatch = [case for case in good if not case.get('equal') or not case.get('swap_equal')]
        perturb_ok = bool(good) and all(case.get('perturb_fires') is True for case in good)
        row = dict(cases=_count(case.get('equal') for case in good), swap=_count(case.get('swap_equal') for case in good),
                   perturb=_count(case.get('perturb_fires') for case in good), errors=errors, reasons=[])
        control_ok = True
        if name.startswith('matmul-'):
            in0 = [case.get('k_in0_fires') for case in good if 'k_in0_fires' in case]
            split = [case.get('k_split_fires') for case in good if 'k_split_fires' in case]
            row.update(k_in0=_count(in0), k_split=_count(split))
            fired = any(flag is True for flag in in0) or (policy == 'either' and any(flag is True for flag in split))
            control_ok = fired
            if not fired:
                row['reasons'].append('k-control')
        if mismatch:
            verdict = 'FAIL'
            row['reasons'].append('mismatch')
        elif errors or not good or not perturb_ok or not control_ok:
            verdict = 'INCONCLUSIVE'
            if errors:
                row['reasons'].append('errors')
            if good and not perturb_ok:
                row['reasons'].append('perturb-control')
        else:
            verdict = 'PASS'
        row['verdict'] = verdict
        per_op[name] = row
    verdicts = [row['verdict'] for row in per_op.values()]
    complete = set(names) == set(op_names())
    covered = complete and _coverage(report, 'seeds', 'regimes', REGIMES_A)
    if 'FAIL' in verdicts:
        verdict = 'FAIL'
    elif verdicts and all(value == 'PASS' for value in verdicts) and covered:
        verdict = 'PASS'
    else:
        verdict = 'INCONCLUSIVE'
    kill = 'STOP-Q4' if any(per_op.get(name, {}).get('verdict') == 'FAIL' for name in KILL_OPS) else 'none'
    cores = norm_cores(report)
    return dict(verdict=verdict, ops=per_op, ran='%d/%d' % (len(names), len(op_names())), covered=covered,
                kill=kill, policy=policy, split=[name for name, row in per_op.items() if row['verdict'] != 'PASS'],
                norm_cores=cores, norm_split=None if cores is None else (cores.get('hidden') or 0) >= 2,
                norm_cores_expected={key.split('-')[-1]: value for key, value in NORM_CORES.items()})


def decide_q0b(report):
    entries = [case for case in report.get('cases', []) if case.get('step') == 'Q0b']
    good = [case for case in entries if not case.get('error')]
    errors = len(entries) - len(good)
    quads = [user for case in good if case['case'] == 'quad' for user in case['users']]
    dense = [user for case in good if case['case'] == 'dense' for user in case['users']]
    pairs = [row for case in good if case['case'] == 'pair' for row in case['rows']]
    c5 = [case.get('equal') for case in good if case['case'] == 'C5']
    partner = [case.get('equal') for case in good if case['case'] == 'P']
    vs_pair = [user['vs_pair']['equal'] for user in quads]
    vs_single = [user['vs_single']['equal'] for user in quads]
    heads = [user.get('heads_equal', 0) for user in quads]
    later = [not user['vs_single']['equal'] for user in dense if user['user'] >= 1]
    fired = sum(1 for flag in later if flag)
    rate = fired / len(later) if later else 0.0
    result = dict(quad_vs_pair=_count(vs_pair), quad_vs_single=_count(vs_single),
                  pair_vs_single=_count(row['equal'] for row in pairs), c5=_count(c5), partner=_count(partner),
                  heads='%d/%d' % (sum(heads), HEADS * len(quads)), dense='%d/%d' % (fired, len(later)),
                  dense_rate=rate, dense_user0=_count(not user['vs_single']['equal'] for user in dense
                                                      if user['user'] == 0), errors=errors, reasons=[])
    covered = _coverage(report, 'seeds', 'b_regimes', REGIMES_B)
    if (not all(vs_pair) or not all(vs_single) or (c5 and not all(c5)) or (partner and not all(partner))):
        verdict = 'FAIL'
        for label, flags in (('quad-vs-pair', vs_pair), ('quad-vs-single', vs_single), ('c5', c5), ('partner', partner)):
            if flags and not all(flags):
                result['reasons'].append(label)
    elif not quads or errors or not c5 or rate < DENSE_FIRE or not covered or not partner:
        verdict = 'INCONCLUSIVE'
        if errors:
            result['reasons'].append('errors')
        if rate < DENSE_FIRE:
            result['reasons'].append('dense-control')
        if not covered:
            result['reasons'].append('coverage')
        if not partner:
            result['reasons'].append('no-partner-case')
    else:
        verdict = 'PASS'
    result['verdict'] = verdict
    result['fallback'] = 'QUAD_SDPA=pairs' if verdict != 'PASS' else None
    return result


def decide_q0c(report):
    entries = [case for case in report.get('cases', []) if case.get('step') == 'Q0c']
    good = [case for case in entries if not case.get('error')]
    errors = len(entries) - len(good)
    values = [case.get('values_equal') for case in good]
    direct = [case.get('indices_equal') for case in good]
    wide = [case.get('indices_u32_equal') for case in good]
    control = [case.get('control_fires') for case in good]
    result = dict(values=_count(values), indices=_count(direct), indices_u32=_count(wide), control=_count(control),
                  errors=errors, fallback=None, reasons=[])
    if not good:
        verdict = 'INCONCLUSIVE'
        result['reasons'].append('no-cases')
    elif not all(value is True for value in values):
        verdict = 'FAIL'
        result['reasons'].append('values')
    elif not all(value is True for value in direct) and not all(value is True for value in wide):
        verdict = 'FAIL'
        result['reasons'].append('indices')
    elif errors or not all(value is True for value in control):
        verdict = 'INCONCLUSIVE'
        result['reasons'].append('errors' if errors else 'order-control')
    else:
        verdict = 'PASS'
        if not all(value is True for value in direct):
            result['fallback'] = 'indices-uint32'
    result['verdict'] = verdict
    return result


def decide_q0d(report):
    entries = [case for case in report.get('cases', []) if case.get('step') == 'Q0d']
    out = dict(reasons=[])
    variants = {}
    for variant in ('E1', 'E1b'):
        cases = [case for case in entries if case.get('variant') == variant]
        good = [case for case in cases if not case.get('error')]
        errors = len(cases) - len(good)
        equal = [case.get('equal') for case in good]
        control = [case.get('control_fires') for case in good]
        if good and not all(value is True for value in equal):
            verdict = 'FAIL'
        elif not good or errors or not all(value is True for value in control):
            verdict = 'INCONCLUSIVE'
        else:
            verdict = 'PASS'
        variants[variant] = dict(verdict=verdict, equal=_count(equal), control=_count(control), errors=errors)
    served = [case.get('served_reference_equal') for case in entries if case.get('variant') == 'served32'
              and not case.get('error')]
    out['served_vs_reference'] = _count(served)
    out.update(variants)
    e1, e1b = variants['E1']['verdict'], variants['E1b']['verdict']
    if e1b == 'PASS':
        verdict, conv = 'PASS', '110'
    elif e1 == 'PASS' and e1b != 'FAIL':
        verdict, conv = 'INCONCLUSIVE', '80'
        out['reasons'].append('E1b-not-proven')
    elif e1 == 'PASS':
        verdict, conv = 'FAIL', '80'
        out['reasons'].append('E1b-differs')
    elif 'FAIL' in (e1, e1b):
        verdict, conv = 'FAIL', 'halves'
    else:
        verdict, conv = 'INCONCLUSIVE', None
    out['verdict'] = verdict
    out['conv'] = conv
    return out


def arm_medians(timing):
    return {row['arm']: row['median_us'] for row in timing if row.get('median_us') is not None}


def saving_model(timing, *, conv_variant='E1b'):
    """The plan's device saving per 4-live round (ms), from Q0e's per-arm medians (us per repetition). Each item's
    s = 2 x t64 / t(32x2) summed over its arms; saving (2 - s) x t32 (V138). Returns items, total and missing."""
    medians = arm_medians(timing)
    items, missing = {}, []
    for item, spec in MODEL.items():
        arms = [('conv-%s' % conv_variant) if arm == 'conv' else arm for arm in spec['arms']]
        wide = [medians.get('%s/64' % arm) for arm in arms]
        halves = [medians.get('%s/32x2' % arm) for arm in arms]
        if any(value is None for value in wide + halves) or not sum(halves):
            missing.append(item)
            continue
        s = 2 * sum(wide) / sum(halves)
        items[item] = dict(s=s, t32_ms=spec['t32_ms'], saving_ms=(2 - s) * spec['t32_ms'], source='card-B s x V138')
    quad_layer, pair_layers = medians.get('attention/64'), medians.get('attention/32x2')
    if quad_layer is None or pair_layers is None:
        missing.append('attention')
    else:
        items['attention'] = dict(quad_us=quad_layer, pairs_us=pair_layers,
                                  saving_ms=DRAFT_LAYERS * (pair_layers - quad_layer) / 1000.0,
                                  source='card-B absolute, 5 layers')
    items['ccl'] = dict(s=CCL_S, t32_ms=CCL_T32_MS, saving_ms=(2 - CCL_S) * CCL_T32_MS, source='model (R3)')
    items['gaps'] = dict(saving_ms=(PAIR_OPS - QUAD_OPS) * GAP_US / 1000.0, source='model, 1 us per op')
    items['candidate_concat'] = dict(saving_ms=CANDIDATE_CONCAT_MS, source='model')
    total = sum(item['saving_ms'] for item in items.values())
    return dict(items=items, total_ms=total, missing=missing, conv_variant=conv_variant)


def bounded_saving(timing, verdicts):
    """The device saving per 4-live round that this run has PROVEN (lower) and the most it could still prove (upper).
    An item counts in `lower` only when its correctness step passed in the same run (ITEM_OPS in Q0a; attention in
    Q0b; the conv as the variant Q0d proved, else C0), and in `upper` unless that step FAILED. Without this a failed
    or unproven op would still be credited with its widened timing (the E1b conv's 1.77 ms after E1b failed Q0d),
    so the go line could pass, and the stop line miss, on savings the shipped pass cannot have."""
    models = {variant: saving_model(timing, conv_variant=variant) for variant in ('E1b', 'E1')}
    table, missing = {}, []

    def entry(status, value):
        return dict(status=status, lower=value if status == 'proven' else 0.0,
                    upper=0.0 if status == 'failed' else value)

    def measured(variant, item):
        found = models[variant]['items'].get(item)
        return None if found is None else found['saving_ms']

    ops = (verdicts.get('Q0a') or {}).get('ops')
    for item, names in ITEM_OPS.items():
        value = measured('E1b', item)
        if value is None:
            missing.append(item)
            continue
        states = [None if ops is None else (ops.get(name) or {}).get('verdict') for name in names]
        status = 'failed' if 'FAIL' in states else ('proven' if all(state == 'PASS' for state in states)
                                                     else 'unproven')
        table[item] = entry(status, value)
    value = measured('E1b', 'attention')
    if value is None:
        missing.append('attention')
    else:
        table['attention'] = entry({'PASS': 'proven', 'FAIL': 'failed'}.get(
            (verdicts.get('Q0b') or {}).get('verdict'), 'unproven'), value)
    q0d = verdicts.get('Q0d') or {}
    states = {variant: (q0d.get(variant) or {}).get('verdict') for variant in ('E1b', 'E1')}
    conv = {variant: measured(variant, 'conv') for variant in ('E1b', 'E1')}
    proven = next((variant for variant in ('E1b', 'E1') if states[variant] == 'PASS'), None)
    possible = next((variant for variant in ('E1b', 'E1') if states[variant] != 'FAIL' and conv[variant] is not None),
                    None)
    if proven is not None and conv[proven] is None:
        missing.append('conv')
    else:
        lower = C0_CONV_MS if proven is None else conv[proven]
        upper = max(lower, C0_CONV_MS if possible is None else conv[possible])
        table['conv'] = dict(status='proven:%s' % proven if proven else ('unproven' if possible else 'C0'),
                             lower=lower, upper=upper)
    for item in ('ccl', 'gaps', 'candidate_concat'):
        value = models['E1b']['items'][item]['saving_ms']
        table[item] = dict(status='model', lower=value, upper=value)
    return dict(items=table, missing=missing, lower_ms=sum(row['lower'] for row in table.values()),
                upper_ms=sum(row['upper'] for row in table.values()))


def decide_q0e(report, verdicts):
    """Go at >= 5.0 ms/round PROVEN in this run; 3.5-5.0 proven only with E1b (Q0d) and Q0c-lite passing; stop (the
    kill) only when even the upper bound is below 3.5; anything the unproven items could still move is INCONCLUSIVE."""
    timing = report.get('timing') or []
    if not timing:
        return dict(verdict='INCONCLUSIVE', reasons=['no-timing'], saving_ms=None, band=None)
    bounds = bounded_saving(timing, verdicts)
    short = (report.get('rounds') or 0) < MIN_ROUNDS or (report.get('replays') or 0) < MIN_REPLAYS
    lower, upper = bounds['lower_ms'], bounds['upper_ms']
    result = dict(saving_ms=lower, saving_upper_ms=upper, missing=bounds['missing'],
                  items={name: dict(status=row['status'], lower=round(row['lower'], 3), upper=round(row['upper'], 3))
                         for name, row in bounds['items'].items()},
                  short=short, reasons=[])
    e1b = ((verdicts.get('Q0d') or {}).get('E1b') or {}).get('verdict')
    lite = (verdicts.get('Q0c') or {}).get('verdict')
    if upper < SAVING_STOP:
        band = 'stop'
    elif lower >= SAVING_GO:
        band = 'go'
    elif lower >= SAVING_STOP:
        band = 'conditional'
    else:
        band = 'unproven'
    result['band'] = band
    unproven = sorted(name for name, row in bounds['items'].items() if row['upper'] > row['lower'] + 1e-12)
    result['unproven'] = unproven
    if bounds['missing'] or short:
        result['verdict'] = 'INCONCLUSIVE'
        result['reasons'].append('missing:%s' % ','.join(bounds['missing']) if bounds['missing'] else 'short')
    elif band == 'stop':
        result['verdict'] = 'FAIL'
        result['reasons'].append('stop')
    elif band == 'go':
        result['verdict'] = 'PASS'
    else:
        if band == 'conditional' and e1b == 'PASS' and lite == 'PASS':
            result['verdict'] = 'PASS'
        elif upper < SAVING_GO and 'FAIL' in (e1b, lite):
            result['verdict'] = 'FAIL'   # the best still provable lands in the conditional band, its condition failed
        else:
            result['verdict'] = 'INCONCLUSIVE'
            if unproven:
                result['reasons'].append('unproven:%s' % ','.join(unproven))
        result['reasons'].append('conditional:E1b=%s,Q0c-lite=%s' % (e1b, lite))
    return result


def decide(report, policy='in0'):
    """Every step's verdict that has cases (or timing), keyed by step."""
    steps = report.get('steps') or list(STEPS)
    out = {}
    if 'Q0a' in steps:
        out['Q0a'] = decide_q0a(report, policy)
    if 'Q0b' in steps:
        out['Q0b'] = decide_q0b(report)
    if 'Q0c' in steps:
        out['Q0c'] = decide_q0c(report)
    if 'Q0d' in steps:
        out['Q0d'] = decide_q0d(report)
    if 'Q0e' in steps:
        out['Q0e'] = decide_q0e(report, out)
    return out


def verdict_lines(verdicts):
    lines = []
    for step in STEPS:
        verdict = verdicts.get(step)
        if verdict is None:
            continue
        name = 'Q0c-lite' if step == 'Q0c' else step
        words = ['QUAD_PROBE', 'step=%s' % name, 'verdict=%s' % verdict['verdict']]
        if step == 'Q0a':
            words.append('ops=%s' % verdict['ran'])
            words.append('pass=%d' % sum(1 for row in verdict['ops'].values() if row['verdict'] == 'PASS'))
            words.append('kill=%s' % verdict['kill'])
            words.append('control=%s' % verdict['policy'])
            words.append('split=%s' % (','.join(verdict['split']) or 'none'))
            cores = verdict['norm_cores']
            words.append('norm_cores=%s' % ('unmeasured' if cores is None else ','.join(
                '%s:%s' % (key, cores.get(key)) for key in ('hidden', 'q', 'k'))))
            if not verdict['covered']:
                words.append('coverage=partial')
            for name_, row in verdict['ops'].items():
                if row['verdict'] != 'PASS':
                    words.append('%s=%s(%s)' % (name_, row['verdict'], '+'.join(row['reasons']) or 'none'))
        elif step == 'Q0b':
            for key in ('quad_vs_pair', 'quad_vs_single', 'pair_vs_single', 'c5', 'partner', 'heads', 'dense'):
                words.append('%s=%s' % (key, verdict[key]))
            words.append('dense_rate=%.2f' % verdict['dense_rate'])
            if verdict.get('fallback'):
                words.append('fallback=%s' % verdict['fallback'])
        elif step == 'Q0c':
            for key in ('values', 'indices', 'indices_u32', 'control'):
                words.append('%s=%s' % (key, verdict[key]))
            if verdict.get('fallback'):
                words.append('fallback=%s' % verdict['fallback'])
        elif step == 'Q0d':
            for variant in ('E1', 'E1b'):
                row = verdict[variant]
                words.append('%s=%s(equal:%s,control:%s,errors:%d)' % (variant, row['verdict'], row['equal'],
                                                                       row['control'], row['errors']))
            words.append('conv=%s' % verdict['conv'])
            words.append('served_vs_reference=%s' % verdict['served_vs_reference'])
        elif step == 'Q0e':
            if verdict.get('saving_ms') is not None:
                words.append('saving_ms=%.2f' % verdict['saving_ms'])
                words.append('saving_upper_ms=%.2f' % verdict['saving_upper_ms'])
                words.append('band=%s' % verdict['band'])
            if verdict.get('short'):
                words.append('short=True')
        if verdict.get('reasons'):
            words.append('reasons=%s' % ','.join(verdict['reasons']))
        errors = verdict.get('errors')
        if errors:
            words.append('case_errors=%d' % errors)
        lines.append(' '.join(words))
    return lines


def summary_json(report, verdicts, out_path):
    """The final line: one JSON object."""
    steps = {step: verdicts[step]['verdict'] for step in STEPS if step in verdicts}
    q0a = verdicts.get('Q0a') or {}
    q0e = verdicts.get('Q0e') or {}
    kill = q0a.get('kill', 'none')
    if q0e.get('band') == 'stop' and q0e.get('verdict') == 'FAIL':
        kill = 'STOP-Q4' if kill == 'none' else kill
    # A run with a recorded failure (a wrong binary, unpinned sources, a non-finite single-user output that the
    # quad could match bit for bit) is never a go, whatever its steps say.
    go = (bool(steps) and all(value == 'PASS' for value in steps.values()) and set(STEPS) <= set(steps)
          and kill == 'none' and report.get('passed') is True)
    return dict(summary='QUAD_PROBE', probe=PROBE, passed=report.get('passed', False), steps=steps, kill=kill, go=go,
                saving_ms=q0e.get('saving_ms'), saving_upper_ms=q0e.get('saving_upper_ms'),
                conv=(verdicts.get('Q0d') or {}).get('conv'),
                sdpa='fold' if steps.get('Q0b') == 'PASS' else 'pairs',
                failures=len(report.get('failures', [])), error=report.get('error'), report=str(out_path))


def file_sha256(path):
    digest_ = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest_.update(block)
    return digest_.hexdigest()


def tree_digest(root):
    """sha256 over (relative path, file sha256) of every file under `root`, sorted; None when absent."""
    root = Path(root)
    if not root.is_dir():
        return None
    digest_ = hashlib.sha256()
    count = 0
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        digest_.update(path.relative_to(root).as_posix().encode() + b'\0' + file_sha256(path).encode() + b'\n')
        count += 1
    return dict(sha256=digest_.hexdigest(), files=count)


def check_sources(root, report, *, expect_matmul_kernel=''):
    """The SDPA sources the JIT builds the draft SDPA from must be the T16 admission's pinned ones; the op trees
    the widened ops JIT from are recorded, and the matmul compute kernel is compared with the retained export."""
    found = {name: (file_sha256(Path(root) / name) if (Path(root) / name).is_file() else None)
             for name in PINNED_SOURCES}
    report['sdpa_sources'] = found
    ok = True
    for name, expected in PINNED_SOURCES.items():
        if found[name] != expected:
            report['failures'].append('%s is %s, not the T16 admission\'s %s (the served draft SDPA is not the one run)'
                                      % (name, (found[name] or 'missing')[:16], expected[:16]))
            ok = False
    bmm = Path(root) / BMM_KERNEL
    kernel = file_sha256(bmm) if bmm.is_file() else None
    report['matmul_kernel'] = dict(path=BMM_KERNEL, sha256=kernel, evidence_sha256=BMM_EVIDENCE_SHA256)
    report['op_trees'] = {name: tree_digest(Path(root) / OPS_ROOT / name) for name in OP_TREES}
    if kernel != BMM_EVIDENCE_SHA256:
        report['warnings'].append('the image\'s matmul compute kernel is %s, not the retained BMM export %s: the R1 '
                                  'argument read another text (risk 12); the byte proof here stands on its own'
                                  % ((kernel or 'missing')[:16], BMM_EVIDENCE_SHA256[:16]))
    if expect_matmul_kernel and kernel != expect_matmul_kernel:
        report['failures'].append('the matmul compute kernel is %s, not the expected %s'
                                  % ((kernel or 'missing')[:16], expect_matmul_kernel[:16]))
        ok = False
    return ok


def loaded_binary(maps='/proc/self/maps'):
    paths = sorted({line.split()[-1] for line in Path(maps).read_text().splitlines()
                    if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    return paths[0]


def module_files():
    out = {}
    for name in MODULES:
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        out[name] = dict(path=path, sha256=file_sha256(path) if path and Path(path).is_file() else None)
    for name in KERNEL_FILES:
        path = kernel_path(name)
        out[name] = dict(path=str(path), sha256=file_sha256(path) if path.is_file() else None)
    return out


def kernel_path(name):
    """A kernel source: beside the harness (the container's /bench), else this checkout's scripts/ci."""
    for directory in (HERE, HERE.parents[2] / 'scripts' / 'ci' if len(HERE.parents) > 2 else HERE):
        if (Path(directory) / name).is_file():
            return Path(directory) / name
    return HERE / name


def read_device_log(path):
    """{run host id: {(core_x, core_y)}} from a raw device-profiler log (profile_log_device.csv)."""
    lines = Path(path).read_text(errors='replace').splitlines()
    start = next((index for index, line in enumerate(lines) if 'core_x' in line), None)
    if start is None:
        return {}
    reader = csv.DictReader(lines[start:], skipinitialspace=True)
    fields = [field.strip() for field in (reader.fieldnames or [])]
    reader.fieldnames = fields
    run = next((field for field in fields if field.lower() == 'run host id'), None) or next(
        (field for field in fields if 'run' in field.lower() and 'id' in field.lower()), None)
    if run is None:
        return {}
    out = {}
    for row in reader:
        try:
            key, x, y = int(row[run]), int(row['core_x']), int(row['core_y'])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(key, set()).add((x, y))
    return out


def read_device_runs(path):
    """(chip MHz, {run host id: dict(cores={(x, y)}, start=first cycle)}) from a raw device-profiler log."""
    lines = Path(path).read_text(errors='replace').splitlines()
    start = next((index for index, line in enumerate(lines) if 'core_x' in line), None)
    mhz = 1350.0
    for line in lines[:start or 0]:
        if 'CHIP_FREQ[MHz]:' in line:
            try:
                mhz = float(line.split('CHIP_FREQ[MHz]:')[1].split(',')[0].strip())
            except ValueError:
                pass
    if start is None:
        return mhz, {}
    reader = csv.DictReader(lines[start:], skipinitialspace=True)
    fields = [field.strip() for field in (reader.fieldnames or [])]
    reader.fieldnames = fields
    run = next((field for field in fields if field.lower() == 'run host id'), None)
    clock_field = next((field for field in fields if field.lower().startswith('time[')), None)
    if run is None or clock_field is None:
        return mhz, {}
    out = {}
    for row in reader:
        try:
            key, x, y, cycle = int(row[run]), int(row['core_x']), int(row['core_y']), int(row[clock_field])
        except (KeyError, TypeError, ValueError):
            continue
        entry = out.setdefault(key, dict(cores=set(), start=cycle))
        entry['cores'].add((x, y))
        entry['start'] = min(entry['start'], cycle)
    return mhz, out


def cores_from_log(cores, paths, gap_s=None):
    """Resolve the cores pass from the device log a runtime writes only when the device closes (P6's does: every
    per-call ReadDeviceProfiler read found no file, and profile_log_device.csv appeared at close_device).

    The pass runs every call once to warm the program cache, pauses 3 x CORES_GAP_S, then runs the measured calls
    in cores['_calls'] order with a CORES_GAP_S pause before each. Runs are grouped where consecutive program
    starts are more than half a gap apart; the measured calls are the last len(calls) groups (cached programs run
    back to back, so no JIT compile can split them). Accepted only when the measured phase's per-program core
    counts repeat the warm-up phase's exactly (the same programs on the same cores); anything else leaves the
    status unavailable, never a guessed count."""
    gap_s = CORES_GAP_S if gap_s is None else gap_s
    calls = [tuple(call) for call in cores.get('_calls') or []]
    if not calls:
        return
    path = next((Path(item) for item in paths if Path(item).is_file()), None)
    if path is None:
        cores['_status'] = 'unavailable: no device log at close in %s' % ', '.join(map(str, paths))
        return
    mhz, runs = read_device_runs(path)
    cores['_log'] = str(path)
    threshold = gap_s * mhz * 1e6 / 2
    groups = []
    for run in sorted(runs, key=lambda key: (runs[key]['start'], key)):
        if groups and runs[run]['start'] - runs[groups[-1][-1]]['start'] <= threshold:
            groups[-1].append(run)
        else:
            groups.append([run])
    if len(groups) < len(calls):
        cores['_status'] = ('unavailable: %d program groups in the device log for %d measured calls'
                            % (len(groups), len(calls)))
        return
    measured = groups[len(groups) - len(calls):]
    warm = [run for group in groups[:len(groups) - len(calls)] for run in group]
    counts = lambda keys: [len(runs[key]['cores']) for key in keys]
    flat = [run for group in measured for run in group]
    if counts(warm) != counts(flat):
        cores['_status'] = ('unavailable: the measured calls\' %d programs do not repeat the warm-up\'s %d on the '
                            'same cores (the grouping is not trusted)' % (len(flat), len(warm)))
        return
    for (name, label), group in zip(calls, measured):
        row = cores.get(name)
        row = dict(row) if isinstance(row, dict) and 'error' not in row else {}
        row[label] = counts(group)
        cores[name] = row
    cores['_status'] = 'measured'
    cores['_source'] = 'device log at close: %d groups, %d warm-up programs, %d measured' % (
        len(groups), len(warm), len(flat))


def device_log_paths(environ=None):
    environ = os.environ if environ is None else environ
    roots = []
    if environ.get('TT_METAL_PROFILER_DIR'):
        roots.append(Path(environ['TT_METAL_PROFILER_DIR']))
    if environ.get('TT_METAL_HOME'):
        roots.append(Path(environ['TT_METAL_HOME']) / 'generated' / 'profiler')
    candidates = []
    for root in roots:
        candidates.extend([root / '.logs' / 'profile_log_device.csv', root / 'profile_log_device.csv'])
    return candidates


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
            threading.Thread(target=self.poll, name='quad-draft-watchdog', daemon=True).start()
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

    def upload_kind(self, host, kind):
        ttnn = self.ttnn
        if kind == 'ids':
            return self.upload(host.to(self.torch.int32), dtype=ttnn.uint32, row_major=True)
        if kind == 'bf16':
            return self.upload(host.bfloat16())
        return self.upload(host.float(), dtype=ttnn.float32)

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
        """owned.append for a tensor a variant made. A view of an input (ttnn.reshape of the assembled K/V or the
        query shares its buffer) is never owned: free() would deallocate the input under every later variant and
        round (commit 04c59e49)."""
        kept = {tensor.buffer_address() for tensor in inputs}

        def retain(tensor):
            if tensor.buffer_address() not in kept:
                owned.append(tensor)
            return tensor

        return retain

    def sync(self):
        with WATCHDOG.op('synchronize'):
            self.ttnn.synchronize_device(self.device)

    def kernel(self):
        ttnn = self.ttnn
        return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                fp32_dest_acc_en=True, packer_l1_acc=False)

    def grid(self):
        try:
            size = self.device.compute_with_storage_grid_size()
            return int(size.x), int(size.y)
        except Exception:  # noqa: BLE001
            return None

    def stage(self, caches, block, owned):
        staged_caches = [{name: self.upload(cache[name]) for name in 'kv'} for cache in caches]
        staged_block = {name: self.upload(block[name]) for name in 'kv'}
        for cache in staged_caches:
            owned.extend(cache.values())
        owned.extend(staged_block.values())
        return staged_caches, staged_block

    def assemble_pair(self, plan, caches, block, owned):
        """The pair's K/V exactly as draft_attention_branch assembles them (its pads from the block's rows 0-15)."""
        ttnn = self.ttnn
        heads = {}
        for name in 'kv':
            pieces = []
            for part in plan:
                if part['kind'] == 'cached':
                    pieces.append(caches[part['user']][name])
                    continue
                start = part['source'].start if part['kind'] == 'live' else 0
                piece = ttnn.slice(block[name], (0, 0, start, 0), (1, KV_HEADS, start + part['rows'], HEAD_DIM))
                owned.append(piece)
                pieces.append(piece)
            with WATCHDOG.op('assemble pair %s' % name):
                heads[name] = ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            owned.append(heads[name])
        return heads['k'], heads['v']

    def assemble_quad(self, caches, block, owned):
        with WATCHDOG.op('assemble quad'):
            return quad.assemble_quad(self.ttnn, caches, block, keeper(owned))

    def conv(self, variant, inputs, seams, owned, *, compute_groups=None, poison=False):
        """One fused-conv call (quad_candidates.conv_program) on [hidden, dynamic0, dynamic1, base0, base1].

        poison=True (every Q0d comparison) writes the output buffer full of NaN before the call. The served
        ttnn.empty leaves DRAM as it was, and Q0d runs E1 then E1b on byte-identical inputs back to back: the
        allocator hands E1b the addresses E1's outputs just freed, so an E1b kernel that wrote no page (or skipped
        some) would read back E1's correct output and E1's carried control, and PASS without having run. Timing
        (inside a trace capture, where a host write is not allowed) keeps ttnn.empty."""
        ttnn = self.ttnn
        shape = tuple(inputs[0].shape)
        if poison:
            output = self.upload(self.torch.full(shape, float('nan'), dtype=self.torch.bfloat16))
        else:
            output = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.append(output)
        tensors = [*inputs, output]
        shards = []
        for value in tensors:
            parts = ttnn.get_device_tensors(value)
            if len(parts) != 1:
                raise RuntimeError('one device shard per tensor required (this harness opens one chip)')
            shards.append(parts[0])
        program = quad.conv_program(ttnn, shards, variant, kernel_dir=kernel_path(quad.QUAD_CONV_KERNEL).parent,
                                    served_dir=kernel_path(quad.SERVED_CONV_IO).parent, seams=seams,
                                    compute_groups=compute_groups)
        with WATCHDOG.op('conv %s' % variant):
            ttnn.generic_op(tensors, program)
        return output


def record(report, entry):
    report['cases'].append(entry)
    keys = ('op', 'seed', 'regime', 'shard', 'case', 'variant', 'user', 'pair')
    head = ' '.join('%s=%s' % (key, entry[key]) for key in keys if key in entry)
    flags = []
    for key in ('equal', 'swap_equal', 'perturb_fires', 'k_in0_fires', 'k_split_fires', 'control_fires',
                'values_equal', 'indices_equal', 'indices_u32_equal', 'served_reference_equal'):
        if key in entry:
            flags.append('%s=%s' % (key, entry[key]))
    if entry.get('users'):
        flags.append('users=%s' % ','.join('%d:%s' % (user['user'], 'eq' if user.get('vs_single', {}).get('equal')
                                                      and user.get('vs_pair', {'equal': True}).get('equal') else 'DIFF')
                                           for user in entry['users']))
    print('case %s %s %s%s' % (entry.get('step'), head, ' '.join(flags),
                               ' ERROR %s' % entry['error'] if entry.get('error') else ''), flush=True)


def keeper(owned):
    """A retain callable for the served functions: owned.append, returning the tensor (append returns None)."""

    def keep(tensor):
        owned.append(tensor)
        return tensor

    return keep


def guarded(report, entry, body):
    """Run one case; an exception is the case's result, never the run's end (except the watchdog's exit)."""
    try:
        body(entry)
    except Exception as error:  # noqa: BLE001
        entry['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
    record(report, entry)


# ---- Q0a --------------------------------------------------------------------------------------

class RowOps:
    """The Q0a ops, called as the served code calls them, at 32 or 64 rows. `state` holds a matmul's uploaded weight
    shard (and its K halves for the K-split control) or an op's small uploaded parameters."""

    def __init__(self, session, torch, args):
        self.session, self.torch, self.args = session, torch, args
        self.ttnn = session.ttnn

    def weight_dtype(self, spec):
        ttnn = self.ttnn
        if spec['dtype'] == 'bf16' or self.args.projection_dtype == 'bf16':
            return ttnn.bfloat16
        return ttnn.bfloat8_b

    def prepare(self, name, seed, shard, *, ksplit=False):
        ttnn, torch = self.ttnn, self.torch
        state = dict(owned=[])
        family = op_inputs()[name][0]
        if family == 'matmul':
            spec = matmul_table()[name]
            weight = draw_weights(torch, name, seed, shard)
            dtype = self.weight_dtype(spec)
            state['weight'] = self.session.upload(weight, dtype=dtype)
            state['owned'].append(state['weight'])
            if ksplit:
                half = spec['width'] // 2
                state['top'] = self.session.upload(weight[:half].contiguous(), dtype=dtype)
                state['bottom'] = self.session.upload(weight[half:].contiguous(), dtype=dtype)
                state['owned'].extend([state['top'], state['bottom']])
        elif family == 'norm':
            generator = torch.Generator().manual_seed(31 * (seed + 1) + op_names().index(name))
            shape = (1, 1, HIDDEN // 32, 32) if name == 'rms_norm-hidden' else (1, 1, HEAD_DIM // 32, 32)
            weight = (1 + 0.1 * torch.randn(*shape, generator=generator)).bfloat16()
            state['weight'] = self.session.upload(weight, row_major=True)
            state['owned'].append(state['weight'])
        elif family == 'embedding':
            generator = torch.Generator().manual_seed(37 * (seed + 1))
            table = torch.randn(self.args.embedding_vocab, EMBED_WIDTH, generator=generator).bfloat16()
            state['table'] = self.session.upload(table, row_major=True)
            state['owned'].append(state['table'])
        return state

    def matmul(self, name, value, weight, rows, owned, *, in0_block_w=4):
        ttnn = self.ttnn
        spec = matmul_table()[name]
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=spec['grid'],
            in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32,
            per_core_N=spec['per_core_n'], fuse_batch=True, fused_activation=None, mcast_in0=True)
        return ttnn.matmul(value, weight, dtype=ttnn.float32, compute_kernel_config=self.session.kernel(),
                           program_config=program, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def run(self, name, tensors, rows, state, owned, config):
        """The op on uploaded inputs; returns a tuple of outputs (owned by the caller)."""
        ttnn = self.ttnn
        kernel = self.session.kernel()
        memory = ttnn.DRAM_MEMORY_CONFIG
        family = op_inputs()[name][0]
        if family == 'matmul':
            if config.get('ksplit'):
                spec = matmul_table()[name]
                half = spec['width'] // 2
                left = ttnn.slice(tensors[0], (0, 0, 0, 0), (1, 1, rows, half))
                right = ttnn.slice(tensors[0], (0, 0, 0, half), (1, 1, rows, spec['width']))
                owned.extend([left, right])
                first = self.matmul(name, left, state['top'], rows, owned)
                second = self.matmul(name, right, state['bottom'], rows, owned)
                owned.extend([first, second])
                return (ttnn.add(first, second, dtype=ttnn.float32),)
            return (self.matmul(name, tensors[0], state['weight'], rows, owned,
                                in0_block_w=config.get('in0_block_w', 4)),)
        if family == 'norm':
            return (ttnn.rms_norm(tensors[0], epsilon=1e-6, weight=state['weight'], compute_kernel_config=kernel,
                                  memory_config=memory),)
        if family == 'rotary':
            return (ttnn.experimental.rotary_embedding_hf(*tensors, is_decode_mode=False, compute_kernel_config=kernel,
                                                          memory_config=memory),)
        if name == 'create-heads':
            combined = ttnn.concat([tensors[1], tensors[2]], dim=3, memory_config=memory)
            owned.append(combined)
            return tuple(ttnn.experimental.nlp_create_qkv_heads(tensors[0], combined, num_heads=HEADS,
                                                                num_kv_heads=KV_HEADS, transpose_k_heads=False,
                                                                memory_config=memory))
        if name == 'concat-heads':
            return (ttnn.experimental.nlp_concat_heads(tensors[0], memory_config=memory),)
        if name == 'typecast-down':
            return (ttnn.typecast(tensors[0], ttnn.bfloat16),)
        if name == 'typecast-up':
            return (ttnn.typecast(tensors[0], ttnn.float32),)
        if name == 'add':
            return (ttnn.add(tensors[0], tensors[1], dtype=ttnn.float32),)
        if name == 'silu':
            return (ttnn.silu(tensors[0], memory_config=memory),)
        if name == 'multiply':
            return (ttnn.multiply(tensors[0], tensors[1], dtype=ttnn.float32),)
        if name == 'embedding':
            local = ttnn.embedding(tensors[0], state['table'], layout=ttnn.TILE_LAYOUT, memory_config=memory)
            owned.append(local)
            return (ttnn.reshape(local, (1, 1, rows, EMBED_WIDTH)),)
        raise ValueError('unknown op %r' % (name,))

    def upload(self, name, values):
        kinds = [kind for _, kind in op_inputs()[name][1]]
        return [self.session.upload_kind(value, 'fp32' if kind in ('cos', 'sin', 'fp32r') else kind)
                for value, kind in zip(values, kinds)]

    def call(self, name, values, rows, state, **config):
        """Upload `values`, run the op at `rows`, read every output back, free everything the call made."""
        session = self.session
        owned = []
        try:
            tensors = self.upload(name, values)
            owned.extend(tensors)
            with WATCHDOG.op('Q0a %s rows=%d %s' % (name, rows, config or '')):
                outputs = self.run(name, tensors, rows, state, owned, config)
            owned.extend(outputs)
            return [session.host(output) for output in outputs]
        finally:
            session.free(owned)


def halves_of(values, name):
    kinds = [kind for _, kind in op_inputs()[name][1]]
    return [[take_rows(value, kind, half) for value, kind in zip(values, kinds)] for half in range(2)]


def joined(torch, first, second):
    return [torch.cat([left, right], dim=-2) for left, right in zip(first, second)]


def swapped_outputs(torch, outputs):
    return [_swap(output, output.dim() - 2, output.shape[-2] // 2) for output in outputs]


def rowlocal_cases(session, torch, args, report):
    ops = RowOps(session, torch, args)
    names = args.ops or op_names()
    for name in names:
        family, inputs = op_inputs()[name]
        kinds = [kind for _, kind in inputs]
        shards = args.shards if family == 'matmul' else 1
        for seed in args.seeds:
            for shard in range(shards):
                ksplit = family == 'matmul' and shard == 0 and not args.no_ksplit
                state = None
                try:
                    with WATCHDOG.op('prepare %s' % name, extra=OPEN_EXTRA_S):
                        state = ops.prepare(name, seed, shard, ksplit=ksplit)
                except Exception as error:  # noqa: BLE001
                    for regime in args.regimes:
                        record(report, dict(step='Q0a', op=name, seed=seed, regime=regime, shard=shard,
                                            error='prepare: %s: %s' % (type(error).__name__, str(error)[:200])))
                    continue
                try:
                    for regime in args.regimes:
                        def body(entry, name=name, seed=seed, regime=regime, state=state, ksplit=ksplit,
                                 kinds=kinds, family=family):
                            values = draw_inputs(torch, name, seed, regime, vocab=args.embedding_vocab)
                            wide = ops.call(name, values, 64, state)
                            parts = [ops.call(name, half, 32, state) for half in halves_of(values, name)]
                            halves = joined(torch, *parts)
                            result = compare_all(torch, wide, halves)
                            entry.update(equal=result['equal'], differing=result['differing'], total=result['total'],
                                         max_abs=result['max_abs'])
                            swapped_in = [swap_rows(value, kind) for value, kind in zip(values, kinds)]
                            swapped = ops.call(name, swapped_in, 64, state)
                            entry['swap_equal'] = compare_all(torch, swapped, swapped_outputs(torch, wide))['equal']
                            moved = list(values)
                            moved[0] = perturb(torch, values[0], kinds[0], vocab=args.embedding_vocab)
                            perturbed = ops.call(name, moved, 64, state)
                            entry['perturb_fires'] = not compare_all(torch, perturbed, halves)['equal']
                            if family == 'matmul':
                                blocked = joined(torch, *[ops.call(name, half, 32, state, in0_block_w=2)
                                                          for half in halves_of(values, name)])
                                entry['k_in0_fires'] = not compare_all(torch, blocked, halves)['equal']
                                if ksplit:
                                    split = joined(torch, *[ops.call(name, half, 32, state, ksplit=True)
                                                            for half in halves_of(values, name)])
                                    entry['k_split_fires'] = not compare_all(torch, split, halves)['equal']
                        guarded(report, dict(step='Q0a', op=name, seed=seed, regime=regime, shard=shard), body)
                finally:
                    session.free(state['owned'])


def cores_pass(session, torch, args, report):
    """'cores': each Q0a op once at 64 rows and once at 32, the device profiler read around each call; the
    distinct cores of every new program in the raw device log."""
    ttnn = session.ttnn
    ops = RowOps(session, torch, args)
    cores = dict(_status='unavailable', _log=None)
    report['cores'] = cores
    paths = device_log_paths()
    seen = set()

    def read():
        session.sync()
        with WATCHDOG.op('ReadDeviceProfiler'):
            ttnn.ReadDeviceProfiler(session.device)
        for path in paths:
            if path.is_file():
                cores['_log'] = str(path)
                return read_device_log(path)
        return {}

    try:
        seen.update(read())
    except Exception as error:  # noqa: BLE001
        cores['_status'] = 'unavailable: %s: %s' % (type(error).__name__, str(error)[:200])
        return
    cores['_calls'] = []
    plans = []
    try:
        for name in args.ops or op_names():
            try:
                state = ops.prepare(name, args.seeds[0], 0)
            except Exception as error:  # noqa: BLE001
                cores[name] = dict(error='prepare: %s: %s' % (type(error).__name__, str(error)[:200]))
                continue
            values = draw_inputs(torch, name, args.seeds[0], 'normal', vocab=args.embedding_vocab)
            plans.append((name, state, (('widened', values, 64), ('half', halves_of(values, name)[0], 32))))
        # Warm-up: every call once, so the measured calls below run cached programs back to back. A runtime that
        # writes the device log only at close (P6) is resolved by cores_from_log from the pauses between the
        # measured calls, which a JIT compile inside a call would otherwise split.
        failed = set()
        for name, state, calls in plans:
            for label, inputs, rows in calls:
                owned = []
                try:
                    tensors = ops.upload(name, inputs)
                    owned.extend(tensors)
                    with WATCHDOG.op('cores warm %s %s' % (name, label)):
                        owned.extend(ops.run(name, tensors, rows, state, owned, {}))
                    session.sync()
                except Exception as error:  # noqa: BLE001
                    cores[name] = dict(error='warm: %s: %s' % (type(error).__name__, str(error)[:200]))
                    failed.add(name)
                finally:
                    session.free(owned)
        seen.update(read())
        time.sleep(3 * CORES_GAP_S)
        for name, state, calls in plans:
            if name in failed:
                continue
            row = {}
            try:
                for label, inputs, rows in calls:
                    owned = []
                    try:
                        # Uploads first and flushed, then the pause, so the new programs after it are the op's own.
                        tensors = ops.upload(name, inputs)
                        owned.extend(tensors)
                        seen.update(read())
                        time.sleep(CORES_GAP_S)
                        with WATCHDOG.op('cores %s %s' % (name, label)):
                            outputs = ops.run(name, tensors, rows, state, owned, {})
                        owned.extend(outputs)
                        cores['_calls'].append([name, label])
                        after = read()
                        new = sorted(key for key in after if key not in seen)
                        seen.update(new)
                        row[label] = [len(after[key]) for key in new]
                    finally:
                        session.free(owned)
                cores[name] = row
                print('cores %s widened=%s half=%s' % (name, row['widened'], row['half']), flush=True)
            except Exception as error:  # noqa: BLE001
                cores[name] = dict(error='%s: %s' % (type(error).__name__, str(error)[:200]))
    finally:
        for _, state, _ in plans:
            session.free(state['owned'])
    measured = [row for key, row in cores.items() if not key.startswith('_') and row.get('widened')]
    cores['_status'] = 'measured' if measured else ('unavailable: no device log at %s'
                                                   % ', '.join(map(str, paths)) if not cores['_log']
                                                   else 'unavailable: no programs in the log')


# ---- Q0b --------------------------------------------------------------------------------------

def attention_cases(session, torch, args, report, digests):
    """Q0b for every seed and regime; `digests` collects quad user digests for P."""
    from draft_attention import draft_sdpa
    from dflash_t16_native_attention import validate_mask
    from pair_row_exact import fold_attention, folded_sdpa

    ttnn = session.ttnn
    mask_host = single_mask()
    validate_mask(mask_host)              # the unchanged single-user rule, before upload (section 3.3)
    masks = dict(single=session.upload(mask_host), dense=session.upload(quad.dense_quad_mask(torch)))
    try:
        for seed in args.seeds:
            for regime in args.b_regimes:
                fixture = build_quad_fixture(torch, seed, regime)
                alone = {}
                for user in range(quad.USERS):
                    def c0(entry, user=user):
                        owned = []
                        try:
                            operands = [session.upload(value) for value in single_operands(torch, fixture, user)]
                            owned.extend(operands)
                            with WATCHDOG.op('draft_sdpa C0 user %d' % user):
                                out = draft_sdpa(ttnn, *operands, masks['single'])
                            owned.append(out)
                            result = session.host(out)
                            if not bool(torch.isfinite(result.float()).all()):
                                # A NaN reference could be matched bit for bit by a NaN quad: never a reference.
                                report['failures'].append('C0 seed=%d regime=%s user %d: a non-finite output'
                                                          % (seed, regime, user))
                                raise RuntimeError('C0: a non-finite single-user output is no reference')
                            alone[user] = result[..., :BLOCK, :].clone()
                            entry['digest'] = digest(torch, alone[user])
                        finally:
                            session.free(owned)
                    guarded(report, dict(step='Q0b', case='C0', seed=seed, regime=regime, user=user), c0)
                folded_pairs = {}
                for pair in range(2):
                    def pair_case(entry, pair=pair):
                        owned = []
                        try:
                            operands = pair_operands(torch, fixture, pair)
                            query = session.upload(operands['query'])
                            owned.append(query)
                            caches, block = session.stage(operands['caches'], operands['block'], owned)
                            keys, values = session.assemble_pair(operands['plan'], caches, block, owned)
                            variant = []
                            try:
                                with WATCHDOG.op('fold_attention pair %d' % pair):
                                    out = fold_attention(ttnn, query, keys, values, masks['single'],
                                                         session.retainer(variant, query, keys, values),
                                                         mask_validated=True)
                                result = session.host(out)
                            finally:
                                session.free(variant)
                            rows_ = []
                            for index in range(2):
                                user = 2 * pair + index
                                folded_pairs[user] = user_rows(result, index).clone()
                                rows_.append(dict(user=user, **compare(torch, folded_pairs[user], alone[user])))
                            entry['rows'] = rows_
                        finally:
                            session.free(owned)
                    guarded(report, dict(step='Q0b', case='pair', seed=seed, regime=regime, pair=pair), pair_case)

                def quad_case(entry):
                    owned = []
                    try:
                        operands = quad_operands(torch, fixture)
                        query = session.upload(operands['query'])
                        owned.append(query)
                        caches, block = session.stage(operands['caches'], operands['block'], owned)
                        keys, values = session.assemble_quad(caches, block, owned)
                        variant = []
                        try:
                            with WATCHDOG.op('quad_fold_attention'):
                                out = quad.quad_fold_attention(ttnn, query, keys, values, masks['single'],
                                                               session.retainer(variant, query, keys, values),
                                                               mask_validated=True)
                            result = session.host(out)
                        finally:
                            session.free(variant)
                        users = []
                        for user in range(quad.USERS):
                            got = user_rows(result, user)
                            single = compare(torch, got, alone[user])
                            heads_equal = sum(1 for head in range(HEADS)
                                              if compare(torch, got[:, head], alone[user][:, head])['equal'])
                            paired = compare(torch, got, folded_pairs[user])
                            digests[(seed, regime, user)] = digest(torch, got)
                            users.append(dict(user=user, vs_single=single, vs_pair=paired, heads_equal=heads_equal))
                        entry['users'] = users

                        def c5(entry_):
                            assembled = [compare(torch, session.host(tensor), operands['expected'][name])
                                         for tensor, name in ((keys, 'k'), (values, 'v'))]
                            entry_.update(equal=all(item['equal'] for item in assembled),
                                          differing=sum(item['differing'] or 0 for item in assembled))
                        guarded(report, dict(step='Q0b', case='C5', seed=seed, regime=regime), c5)

                        def dense(entry_):
                            variant_ = []
                            try:
                                with WATCHDOG.op('dense 4-segment sdpa'):
                                    out_ = folded_sdpa(ttnn, query, keys, values, masks['dense'])
                                variant_.append(out_)
                                result_ = session.host(out_)
                            finally:
                                session.free(variant_)
                            entry_['users'] = [dict(user=user, vs_single=compare(torch, user_rows(result_, user),
                                                                                  alone[user]))
                                               for user in range(quad.USERS)]
                        guarded(report, dict(step='Q0b', case='dense', seed=seed, regime=regime), dense)
                    finally:
                        session.free(owned)
                guarded(report, dict(step='Q0b', case='quad', seed=seed, regime=regime), quad_case)
    finally:
        session.free(list(masks.values()))
    for seed in args.seeds:
        if 'partner100' not in args.b_regimes or 'normal' not in args.b_regimes:
            break
        for user in (1, 3):
            normal, loud = digests.get((seed, 'normal', user)), digests.get((seed, 'partner100', user))
            if normal is None or loud is None:
                continue
            record(report, dict(step='Q0b', case='P', seed=seed, regime='partner100', user=user, equal=normal == loud))


# ---- Q0c-lite ---------------------------------------------------------------------------------

def draw_logits(torch, seed, regime):
    generator = torch.Generator().manual_seed(9001 + seed)
    halves = [torch.randn(1, 1, 32, LOGIT_SHARD, generator=generator) for _ in range(2)]
    if regime == 'peaked':
        halves = [half * 4 for half in halves]
    elif regime == 'negative':
        halves = [-half.abs() for half in halves]
    elif regime == 'row0x100':
        halves[0] = halves[0] * 100
    return [half.bfloat16() for half in halves]


def candidate_cases(session, torch, args, report):
    from draft_shared_head import local_head_candidates

    ttnn = session.ttnn
    for seed in args.seeds:
        for regime in args.regimes:
            def body(entry, seed=seed, regime=regime):
                owned = []
                try:
                    halves = [session.upload(half) for half in draw_logits(torch, seed, regime)]
                    owned.extend(halves)
                    with WATCHDOG.op('local_head_candidates'):
                        chunks = [local_head_candidates(ttnn, half, owned) for half in halves]
                    expected = []
                    for top, bottom in zip(*chunks):
                        values = torch.cat([session.host(top['values']), session.host(bottom['values'])], dim=-2)
                        indices = torch.cat([session.host(top['indices']).to(torch.int64),
                                             session.host(bottom['indices']).to(torch.int64)], dim=-2)
                        expected.append((values, indices))

                    def read(concatenated):
                        return [(session.host(chunk['values']), session.host(chunk['indices']).to(torch.int64))
                                for chunk in concatenated]

                    with WATCHDOG.op('candidate concat'):
                        direct = read(quad.concat_candidates(ttnn, chunks[0], chunks[1], keeper(owned)))
                    if not expected or len(direct) != len(expected):   # all() over an empty zip is True
                        raise RuntimeError('the concat returned %d chunks, the halves carry %d'
                                           % (len(direct), len(expected)))
                    entry['values_equal'] = all(compare(torch, got[0], want[0])['equal']
                                                for got, want in zip(direct, expected))
                    entry['indices_equal'] = all(torch.equal(got[1], want[1]) for got, want in zip(direct, expected))
                    try:
                        with WATCHDOG.op('candidate concat uint32'):
                            wide = read(quad.concat_candidates(ttnn, chunks[0], chunks[1], keeper(owned),
                                                               indices_u32=True))
                        entry['indices_u32_equal'] = all(torch.equal(got[1], want[1])
                                                         for got, want in zip(wide, expected))
                    except Exception as error:  # noqa: BLE001 - the fallback's own failure, not the case's
                        entry['indices_u32_equal'] = None
                        entry['u32_error'] = '%s: %s' % (type(error).__name__, str(error)[:200])
                    with WATCHDOG.op('candidate concat control'):
                        swapped = read(quad.concat_candidates(ttnn, chunks[1], chunks[0], keeper(owned)))
                    entry['control_fires'] = not all(compare(torch, got[0], want[0])['equal'] and torch.equal(got[1], want[1])
                                                     for got, want in zip(swapped, expected))
                finally:
                    session.free(owned)
            guarded(report, dict(step='Q0c', seed=seed, regime=regime), body)


# ---- Q0d --------------------------------------------------------------------------------------

def draw_conv(torch, seed, regime):
    generator = torch.Generator().manual_seed(7001 + seed)
    hidden = torch.randn(1, 1, 64, quad.HIDDEN, generator=generator)
    if regime == 'peaked':
        hidden = hidden * 4
    elif regime == 'negative':
        hidden = -hidden.abs()
    elif regime == 'row0x100':
        hidden[..., :32, :] = hidden[..., :32, :] * 100
    dynamic = [(torch.randn(1, 1, 64, 320, generator=generator) * 0.1).bfloat16() for _ in range(2)]
    base = [(torch.randn(1, 1, 1, quad.HIDDEN, generator=generator) * 0.5).bfloat16() for _ in range(2)]
    return hidden.bfloat16(), dynamic, base


QUAD_SEAMS = ((0, 16), (16, 32), (32, 48), (48, 64))
PAIR_SEAMS = ((0, 16), (16, 32))


def conv_cases(session, torch, args, report):
    grid = session.grid()
    low, high = quad.seam_words(QUAD_SEAMS, 64)
    served_word = quad.seam_word(PAIR_SEAMS, 32)
    control = (low, high & ~(1 << 16))
    for seed in args.seeds:
        for regime in args.regimes:
            hidden, dynamic, base = draw_conv(torch, seed, regime)
            state = dict()

            def served(entry, seed=seed, regime=regime):
                owned = []
                try:
                    halves = []
                    for half in range(2):
                        rows_ = slice(32 * half, 32 * half + 32)
                        inputs = [session.upload(hidden[..., rows_, :].contiguous()),
                                  *[session.upload(part[..., rows_, :].contiguous()) for part in dynamic],
                                  *[session.upload(part) for part in base]]
                        owned.extend(inputs)
                        halves.append(session.host(session.conv('served32', inputs, served_word, owned,
                                                                poison=True)))
                    joined_ = torch.cat(halves, dim=-2)
                    if not bool(torch.isfinite(joined_.float()).all()):
                        # A page the served call left unwritten still holds the NaN poison; a variant that skipped
                        # the same page would match it bit for bit, so this is no reference.
                        raise RuntimeError('the served 32-row calls left non-finite (unwritten) pages')
                    state['halves'] = joined_
                    reference = conv_reference(torch, hidden, dynamic, base, (served_word, served_word))
                    entry['served_reference_equal'] = compare(torch, state['halves'], reference)['equal']
                finally:
                    session.free(owned)
            guarded(report, dict(step='Q0d', variant='served32', seed=seed, regime=regime), served)
            for variant in ('E1', 'E1b'):
                def body(entry, variant=variant):
                    if 'halves' not in state:
                        raise RuntimeError('the served 32-row calls did not run')
                    if grid is not None and not quad.fits_grid(variant, grid):
                        raise RuntimeError('the compute grid %s cannot hold %s (%s)'
                                           % (grid, variant, quad.CONV_VARIANTS[variant]['grid']))
                    owned = []
                    try:
                        inputs = [session.upload(hidden), *[session.upload(part) for part in dynamic],
                                  *[session.upload(part) for part in base]]
                        owned.extend(inputs)
                        wide = session.host(session.conv(variant, inputs, (low, high), owned, poison=True))
                        result = compare(torch, wide, state['halves'])
                        entry.update(equal=result['equal'], differing=result['differing'])
                        carried = session.host(session.conv(variant, inputs, control, owned, poison=True))
                        entry['control_fires'] = not compare(torch, carried, state['halves'])['equal']
                    finally:
                        session.free(owned)
                guarded(report, dict(step='Q0d', variant=variant, seed=seed, regime=regime), body)


# ---- Q0e --------------------------------------------------------------------------------------

def timing_arms(session, torch, args, base):
    """{arm: callable(owned)}: every arm's device work, on inputs uploaded once into `base`. Arms come in pairs:
    '<item>/64' (the widened call) and '<item>/32x2' (today's call on each half)."""
    from draft_mlp import swiglu_device
    from pair_row_exact import fold_attention

    ttnn = session.ttnn
    ops = RowOps(session, torch, args)
    arms = {}

    def upload_all(values, name):
        kinds = [kind for _, kind in op_inputs()[name][1]]
        tensors = [session.upload_kind(value, 'fp32' if kind in ('cos', 'sin', 'fp32r') else kind)
                   for value, kind in zip(values, kinds)]
        base.extend(tensors)
        return tensors

    def pair_arm(item, name, state):
        values = draw_inputs(torch, name, 0, 'normal', vocab=args.embedding_vocab)
        wide = upload_all(values, name)
        halves = [upload_all(half, name) for half in halves_of(values, name)]

        def run(inputs, rows):
            def call(owned):
                outputs = ops.run(name, inputs, rows, state, owned, {})
                owned.extend(outputs)
            return call
        arms['%s/64' % item] = run(wide, 64)
        first, second = run(halves[0], 32), run(halves[1], 32)
        arms['%s/32x2' % item] = lambda owned: (first(owned), second(owned))

    for name, item in (('matmul-conv', 'mm-conv'), ('matmul-q', 'mm-q'), ('matmul-k', 'mm-k'), ('matmul-v', 'mm-v'),
                       ('matmul-o', 'mm-o'), ('matmul-gate', 'mm-gate'), ('matmul-up', 'mm-up'),
                       ('matmul-down', 'mm-down'), ('matmul-selector', 'mm-selector'),
                       ('rms_norm-hidden', 'norm-hidden'), ('rms_norm-q', 'norm-q'), ('rms_norm-k', 'norm-k'),
                       ('rotary-q', 'rotary-q'), ('rotary-k', 'rotary-k'), ('create-heads', 'heads-create'),
                       ('concat-heads', 'heads-concat')):
        state = ops.prepare(name, 0, 0)
        base.extend(state['owned'])
        pair_arm(item, name, state)

    generator = torch.Generator().manual_seed(4242)
    gates = [session.upload(bf16_round(torch, torch.randn(1, 1, rows, INTERMEDIATE, generator=generator)),
                            dtype=ttnn.float32) for rows in (64, 32, 32)]
    ups = [session.upload(bf16_round(torch, torch.randn(1, 1, rows, INTERMEDIATE, generator=generator)),
                          dtype=ttnn.float32) for rows in (64, 32, 32)]
    wides = [session.upload(torch.randn(1, 1, rows, HIDDEN, generator=generator), dtype=ttnn.float32)
             for rows in (64, 32, 32)]
    base.extend([*gates, *ups, *wides])

    def swiglu(index):
        return lambda owned: swiglu_device(ttnn, gates[index], ups[index], keeper(owned))

    def small(index):
        def call(owned):
            rounded = ttnn.typecast(wides[index], ttnn.bfloat16)
            owned.append(rounded)
            owned.append(ttnn.add(wides[index], wides[index], dtype=ttnn.float32))
        return call

    arms['swiglu/64'] = swiglu(0)
    arms['swiglu/32x2'] = lambda owned: (swiglu(1)(owned), swiglu(2)(owned))
    arms['small/64'] = small(0)
    arms['small/32x2'] = lambda owned: (small(1)(owned), small(2)(owned))

    fixture = build_quad_fixture(torch, 0, 'normal')
    mask = session.upload(single_mask())
    base.append(mask)
    operands = quad_operands(torch, fixture)
    quad_query = session.upload(operands['query'])
    base.append(quad_query)
    quad_caches, quad_block = session.stage(operands['caches'], operands['block'], base)
    pairs = []
    for pair in range(2):
        pair_ops = pair_operands(torch, fixture, pair)
        query = session.upload(pair_ops['query'])
        base.append(query)
        caches, block = session.stage(pair_ops['caches'], pair_ops['block'], base)
        pairs.append((pair_ops['plan'], query, caches, block))

    def quad_layer(owned):
        keys, values = quad.assemble_quad(ttnn, quad_caches, quad_block, keeper(owned))
        quad.quad_fold_attention(ttnn, quad_query, keys, values, mask, session.retainer(owned, quad_query, keys, values),
                                 mask_validated=True)

    def pair_layers(owned):
        for plan, query, caches, block in pairs:
            keys, values = session.assemble_pair(plan, caches, block, owned)
            fold_attention(ttnn, query, keys, values, mask, session.retainer(owned, query, keys, values),
                           mask_validated=True)

    arms['attention/64'] = quad_layer
    arms['attention/32x2'] = pair_layers

    hidden, dynamic, bases = draw_conv(torch, 0, 'normal')
    wide_inputs = [session.upload(hidden), *[session.upload(part) for part in dynamic],
                   *[session.upload(part) for part in bases]]
    half_inputs = [[session.upload(hidden[..., 32 * half:32 * half + 32, :].contiguous()),
                    *[session.upload(part[..., 32 * half:32 * half + 32, :].contiguous()) for part in dynamic],
                    *[session.upload(part) for part in bases]] for half in range(2)]
    base.extend(wide_inputs + half_inputs[0] + half_inputs[1])
    words = quad.seam_words(QUAD_SEAMS, 64)
    served_word = quad.seam_word(PAIR_SEAMS, 32)
    grid = session.grid()
    for variant in ('E1', 'E1b'):
        if grid is None or quad.fits_grid(variant, grid):
            arms['conv-%s/64' % variant] = (lambda owned, variant=variant:
                                             session.conv(variant, wide_inputs, words, owned))

    def served_halves(owned):
        for inputs in half_inputs:
            session.conv('served32', inputs, served_word, owned)

    arms['conv-E1/32x2'] = served_halves
    arms['conv-E1b/32x2'] = served_halves
    return arms


def timing(session, torch, args, report):
    """Q0e: every arm warmed eagerly, captured once (--trace-reps repetitions per capture), then --rounds rounds that
    interleave the arms (rotating the order), --replays timed replays per arm per round."""
    ttnn = session.ttnn
    base = []
    traces = {}
    try:
        with WATCHDOG.op('Q0e setup', extra=OPEN_EXTRA_S):
            arms = timing_arms(session, torch, args, base)
        names = [name for name in arms if not args.arms or name.split('/')[0] in args.arms]
        for name in names:
            owned = []
            try:
                with WATCHDOG.op('warm %s' % name):
                    arms[name](owned)
                session.sync()
            except Exception as error:  # noqa: BLE001
                report['timing'].append(dict(arm=name, error='warm: %s: %s' % (type(error).__name__, str(error)[:200])))
            finally:
                session.free(owned)
        names = [name for name in names if not any(row.get('arm') == name for row in report['timing'])]
        for name in names:
            owned = []
            try:
                with WATCHDOG.op('capture %s' % name):
                    trace = ttnn.begin_trace_capture(session.device, cq_id=0)
                    try:
                        for _ in range(args.trace_reps):
                            arms[name](owned)
                    finally:
                        ttnn.end_trace_capture(session.device, trace, cq_id=0)
                traces[name] = (trace, owned)
            except Exception as error:  # noqa: BLE001
                session.free(owned)
                report['timing'].append(dict(arm=name, error='capture: %s: %s' % (type(error).__name__, str(error)[:200])))
        names = [name for name in names if name in traces]
        samples = {name: [] for name in names}
        for name in names:
            with WATCHDOG.op('trace warm-up %s' % name):
                for _ in range(args.trace_warmup):
                    ttnn.execute_trace(session.device, traces[name][0], cq_id=0, blocking=False)
                    ttnn.synchronize_device(session.device)
        for index in range(args.rounds):
            turn = index % max(1, len(names))
            for name in names[turn:] + names[:turn]:
                with WATCHDOG.span('replay %s round %d' % (name, index), args.watchdog):
                    for _ in range(args.replays):
                        started = clock()
                        ttnn.execute_trace(session.device, traces[name][0], cq_id=0, blocking=False)
                        ttnn.synchronize_device(session.device)
                        samples[name].append((clock() - started) * 1e6 / args.trace_reps)
        for name in names:
            row = dict(arm=name, **(summary(samples[name]) or {}))
            report['timing'].append(row)
            print('timing %-22s %.1f us' % (name, row.get('median_us', float('nan'))), flush=True)
    finally:
        for name, (trace, owned) in traces.items():
            try:
                with WATCHDOG.op('release trace %s' % name):
                    ttnn.release_trace(session.device, trace)
            finally:
                session.free(owned)
        session.free(base)


# ---- run --------------------------------------------------------------------------------------

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
    if 'Q0e' in args.steps:
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
        if args.tt_metal_home and not check_sources(args.tt_metal_home, report,
                                                    expect_matmul_kernel=args.expect_matmul_kernel_sha256):
            return
        import pair_row_exact  # noqa: F401 - the mounted modules, recorded before any case runs
        import draft_attention  # noqa: F401
        import dflash_batched_mask  # noqa: F401
        import draft_shared_head  # noqa: F401
        import draft_head_preparation  # noqa: F401
        import dflash_t16_native_attention  # noqa: F401
        import draft_mlp  # noqa: F401
        report['modules'] = module_files()
        session = Session(ttnn, torch, device)
        report['grid'] = session.grid()
        if 'cores' in args.steps:
            cores_pass(session, torch, args, report)
        if 'Q0a' in args.steps:
            rowlocal_cases(session, torch, args, report)
        if 'Q0b' in args.steps:
            attention_cases(session, torch, args, report, {})
        if 'Q0c' in args.steps:
            candidate_cases(session, torch, args, report)
        if 'Q0d' in args.steps:
            conv_cases(session, torch, args, report)
        if 'Q0e' in args.steps:
            timing(session, torch, args, report)
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=False)
    parser.add_argument('--steps', default=','.join(STEPS), help='from %s and cores' % ', '.join(STEPS))
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--regimes', default=','.join(REGIMES_A), help='Q0a, Q0c-lite and Q0d regimes')
    parser.add_argument('--b-regimes', default=','.join(REGIMES_B), help='Q0b regimes')
    parser.add_argument('--ops', default='', help='Q0a ops (default: all %d)' % len(op_names()))
    parser.add_argument('--shards', type=int, default=2, help='independent weight shards per matmul program')
    parser.add_argument('--no-ksplit', action='store_true', help='skip the K-split matmul control')
    parser.add_argument('--matmul-control', choices=MATMUL_CONTROLS, default='in0',
                        help='in0: the plan\'s in0_block_w control decides; either: in0 or the K-split control')
    parser.add_argument('--projection-dtype', choices=('bf8', 'bf16'), default='bf8')
    parser.add_argument('--embedding-vocab', type=int, default=8192,
                        help='rows of the seeded embedding table (a gather: the served 248320/2 is not needed)')
    parser.add_argument('--cores-report', type=Path, default=None, help='the cores pass\'s report, folded into Q0a')
    parser.add_argument('--arms', default='', help='Q0e items to time (default: all)')
    parser.add_argument('--rounds', type=int, default=5, help='Q0e rounds, interleaving the arms (plan: >= 5)')
    parser.add_argument('--replays', type=int, default=20, help='Q0e timed replays per arm per round (plan: >= 20)')
    parser.add_argument('--trace-reps', type=int, default=5, help='repetitions of an arm inside its captured trace')
    parser.add_argument('--trace-warmup', type=int, default=2)
    parser.add_argument('--trace-region-bytes', type=int, default=256 << 20)
    parser.add_argument('--watchdog', type=float, default=0, help='seconds per device call before os._exit(3); 0 off')
    parser.add_argument('--expect-binary-sha256', default='')
    parser.add_argument('--expect-matmul-kernel-sha256', default='')
    parser.add_argument('--skip-binary-check', action='store_true', help='the CPU dry run only')
    parser.add_argument('--tt-metal-home', default=os.environ.get('TT_METAL_HOME', ''),
                        help='where the SDPA sources are pinned and the op trees hashed ("" skips)')
    parser.add_argument('--decide-only', type=Path, default=None, help='re-decide a saved report and print it')
    args = parser.parse_args(argv)
    split = lambda text: [value for value in text.split(',') if value]
    try:
        args.seeds = [int(value) for value in split(args.seeds)]
    except ValueError as error:
        parser.error(str(error))
    args.steps, args.regimes, args.b_regimes = split(args.steps), split(args.regimes), split(args.b_regimes)
    args.ops, args.arms = split(args.ops), split(args.arms)
    if args.decide_only is not None:
        return args
    if args.out is None:
        parser.error('--out is required')
    if not args.steps or any(step not in STEPS + ('cores',) for step in args.steps):
        parser.error('--steps from %s and cores' % ', '.join(STEPS))
    if 'cores' in args.steps and len(args.steps) != 1:
        parser.error('the cores pass runs alone (--steps cores), under TT_METAL_DEVICE_PROFILER=1')
    if not args.seeds or not args.regimes or any(regime not in REGIMES_A for regime in args.regimes):
        parser.error('--regimes from %s, and at least one seed' % ', '.join(REGIMES_A))
    if not args.b_regimes or any(regime not in REGIMES_B for regime in args.b_regimes):
        parser.error('--b-regimes from %s' % ', '.join(REGIMES_B))
    if any(name not in op_names() for name in args.ops):
        parser.error('--ops from %s' % ', '.join(op_names()))
    if args.shards < 1 or args.embedding_vocab < 128:
        parser.error('--shards >= 1 and --embedding-vocab >= 128')
    if min(args.rounds, args.replays, args.trace_reps) < 1 or args.trace_warmup < 0:
        parser.error('--rounds, --replays and --trace-reps must be >= 1; --trace-warmup >= 0')
    for name in ('expect_binary_sha256', 'expect_matmul_kernel_sha256'):
        value = getattr(args, name)
        if value and (len(value) != 64 or any(c not in '0123456789abcdef' for c in value)):
            parser.error('--%s must be a full lowercase sha256' % name.replace('_', '-'))
    return args


def finish(report, args, out_path):
    """Verdicts, lines and the summary for a finished (or re-read) report."""
    verdicts = decide(report, args.matmul_control)
    report['verdicts'] = verdicts
    lines = verdict_lines(verdicts)
    report['verdict_lines'] = lines
    return verdicts, lines, summary_json(report, verdicts, out_path)


def main(argv=None, ttnn=None, torch=None):
    global WATCHDOG
    args = parse_args(argv)
    if args.decide_only is not None:
        report = json.loads(args.decide_only.read_text())
        _, lines, summary_ = finish(report, args, args.decide_only)
        for line in lines:
            print(line)
        print(json.dumps(summary_, sort_keys=True), flush=True)
        return 0
    report = dict(probe=PROBE, design='quad-draft-plan.md section 5 (Q0)', passed=False,
                  argv=list(sys.argv[1:] if argv is None else argv), steps=args.steps, seeds=args.seeds,
                  regimes=args.regimes, b_regimes=args.b_regimes, shards=args.shards, rounds=args.rounds,
                  replays=args.replays, trace_reps=args.trace_reps, matmul_control=args.matmul_control,
                  env={name: os.environ.get(name) for name in ENV_RECORDED}, watchdog=args.watchdog,
                  failures=[], warnings=[], cases=[], timing=[])
    if args.cores_report is not None:
        try:
            loaded = json.loads(Path(args.cores_report).read_text())
            report['cores'] = loaded.get('cores') or dict(_status='unavailable: the cores report has none')
            report['cores_report'] = dict(path=str(args.cores_report), sha256=file_sha256(args.cores_report))
        except Exception as error:  # noqa: BLE001
            report['cores'] = dict(_status='unavailable: %s: %s' % (type(error).__name__, str(error)[:200]))
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
    lines, summary_ = [], None
    try:
        try:
            run(args, report, ttnn=ttnn, torch=torch)
        except Exception as error:  # noqa: BLE001
            report['error'] = '%s: %s' % (type(error).__name__, error)
        cores = report.get('cores')
        if 'cores' in args.steps and isinstance(cores, dict) and cores.get('_status') != 'measured':
            try:
                cores_from_log(cores, device_log_paths())   # the device is closed: a close-only log is written now
            except Exception as error:  # noqa: BLE001
                cores['_status'] = 'unavailable: %s: %s' % (type(error).__name__, str(error)[:200])
            if cores.get('_status') == 'measured':
                for name in args.ops or op_names():
                    row = cores.get(name) or {}
                    if 'widened' in row:
                        print('cores %s widened=%s half=%s (device log at close)'
                              % (name, row.get('widened'), row.get('half')), flush=True)
        report['passed'] =(not report['failures'] and not report.get('error')
                            and bool(report['cases'] or report['timing'] or report.get('cores')))
        try:
            _, lines, summary_ = finish(report, args, args.out)
        except Exception as error:  # noqa: BLE001 - the measurements are still written
            report['error'] = report.get('error') or 'analysis: %s: %s' % (type(error).__name__, error)
            report['passed'] = False
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    for warning in report['warnings']:
        print('WARN', warning)
    if report.get('error'):
        print('ERROR', report['error'])
    for line in lines:
        print(line, flush=True)
    if summary_ is None:
        summary_ = dict(summary='QUAD_PROBE', probe=PROBE, passed=False, steps={}, kill='none', go=False,
                        error=report.get('error'), report=str(args.out))
    print(json.dumps(summary_, sort_keys=True), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
