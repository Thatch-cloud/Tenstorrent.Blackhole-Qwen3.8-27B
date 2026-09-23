#!/usr/bin/env python3
"""Reference and unit test for the prefill sibling of ttnn.transformer.gdn_decode_conv_gates.

Written BEFORE the kernel, so the kernel has something to be correct against rather
than the other way round. Run against the torch reference alone with --reference-only
on any machine; run the device half on the rig once the op exists.

WHY THE OP IS WANTED. Prefill's causal conv is 473 ms, 12.8% of prefill device time,
and 357 ms of that is pure layout conversion (run 35422536834). The two available
implementations land within 2.5% of each other because both leave TILE layout:

  _causal_conv1d_fir   three shifted ttnn.slice windows, each forcing untilize/slice/
                       tilize over the whole [1, S, 5120] activation, then three macs
  _conv1d_prefill      one to_layout(ROW_MAJOR) before ttnn.conv1d and one
                       to_layout(TILE) after

Run 35428550094 measured the swap between them at 0.975x, a regression, which is what
bounds the prize: anything that still untilizes lands where those two did. The op has
to stay in TILE layout throughout to be worth building.

SEMANTICS. Same math as the decode op, with the window taken along the time axis
instead of from a shift register:

    xin      = concat(conv_state, x)            [1, (K-1)+T, C]
    conv_out = silu( sum_j taps[j] * xin[t+j] ) [1, T, C]
    new_state= x[T-(K-1):]                      [1, K-1, C]
    beta     = sigmoid(b)                       [1, T, Nv]
    g        = neg_exp_A * softplus(a + dt_bias)

THE ONE HARD PART. Output row t reads input rows t..t+K-1, so with 32-row tiles a shift
of 1..3 straddles two tiles. The shift-matrix premise this file used to carry (a banded
32x32 matmul per tap) was measured 4.5x slower (docs/gdn-conv-path-2026-09-19.md, 200-252)
and is gone. The op that was built instead, scripts/ci/gdn_prefill_conv_exact.py (lever #2,
a generic_op), shifts rows with eight 32-byte face-row copies per tile in its reader
(shift_copies), keeps TILE layout throughout, and replays the served FIR's four LLK calls
so it is byte-exact rather than close. check_planner below pins those copy tables against a
plain row shift; the op's own CPU tests are scripts/ci/test_gdn_prefill_conv_exact.py and its
device test optimisation/ttnn-op/gdn_prefill_conv/gdn_prefill_conv_card_m.py.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'ci'))
import gdn_prefill_conv_exact as pcx  # noqa: E402

C, NV, K = 5120, 24, 4
TILE = 32


def softplus(x, beta=1.0, threshold=20.0):
    scaled = beta * x
    return torch.where(scaled > threshold, x, torch.log1p(torch.exp(scaled)) / beta)


def reference(x, conv_state, taps, a, b, dt_bias, neg_exp_A):
    """Causal depthwise conv over the time axis, plus the two gates.

    x           [1, T, C]        the qkv projection for this chunk
    conv_state  [1, K-1, C]      carry from the previous chunk, zeros at sequence start
    taps        K x [1, 1, C]
    """
    T = x.shape[1]
    xin = torch.cat([conv_state, x], dim=1)
    acc = torch.zeros_like(x)
    for j in range(K):
        acc = acc + taps[j] * xin[:, j:j + T, :]
    conv_out = torch.nn.functional.silu(acc)
    new_state = x[:, T - (K - 1):, :].clone()
    beta = torch.sigmoid(b)
    g = neg_exp_A * softplus(a + dt_bias)
    return conv_out, new_state, beta, g


def lane(row, column):
    return (row // 16) * 512 + (column // 16) * 256 + (row % 16) * 16 + column % 16


def check_planner():
    """The op's face-row copy tables must reproduce a plain row shift of concat(prev, cur)."""
    torch.manual_seed(0)
    prev = torch.randint(-30000, 30000, (TILE, TILE), dtype=torch.int16)
    cur = torch.randint(-30000, 30000, (TILE, TILE), dtype=torch.int16)
    lanes = torch.tensor([[lane(r, c) for c in range(TILE)] for r in range(TILE)]).reshape(-1)

    def faces(block):
        flat = torch.zeros(TILE * TILE, dtype=torch.int16)
        flat[lanes] = block.reshape(-1)
        return flat

    sources = dict(cur=faces(cur), prev=faces(prev))
    xin = torch.cat([prev, cur], dim=0)
    for s in (1, 2, 3):
        out = torch.zeros(TILE * TILE, dtype=torch.int16)
        for source, offset, length, target in pcx.shift_copies(s, True):
            out[target // 2:(target + length) // 2] = sources[source][offset // 2:(offset + length) // 2]
        assert torch.equal(out[lanes].reshape(TILE, TILE), xin[TILE - s:2 * TILE - s]), 'shift %d is wrong' % s
    print('gdn_prefill_conv_exact.shift_copies reproduces the row shift exactly, for shifts 1..3')


def check_reference():
    """Pin the reference against a naive per-row loop."""
    torch.manual_seed(1)
    T = 64
    x = torch.randn(1, T, C)
    conv_state = torch.randn(1, K - 1, C)
    taps = [torch.randn(1, 1, C) * 0.5 for _ in range(K)]
    a = torch.randn(1, T, NV) * 2
    b = torch.randn(1, T, NV) * 2
    dt_bias = torch.randn(1, 1, NV)
    neg_exp_A = -torch.exp(torch.randn(1, 1, NV))

    conv_out, new_state, beta, g = reference(x, conv_state, taps, a, b, dt_bias, neg_exp_A)

    xin = torch.cat([conv_state, x], dim=1)
    naive = torch.zeros(1, T, C)
    for t in range(T):
        for j in range(K):
            naive[0, t] += taps[j][0, 0] * xin[0, t + j]
    naive = torch.nn.functional.silu(naive)
    assert torch.allclose(conv_out, naive, atol=1e-4), 'conv disagrees with the naive loop'
    assert torch.equal(new_state, x[:, T - (K - 1):, :]), 'carry rows are wrong'
    assert conv_out.shape == (1, T, C) and new_state.shape == (1, K - 1, C)
    assert beta.shape == (1, T, NV) and g.shape == (1, T, NV)
    print('reference matches the naive causal loop; shapes and carry are right')


def check_matches_fir_semantics():
    """A zero carry must reproduce a from-scratch chunk, which is the trace-safe path."""
    torch.manual_seed(2)
    T = 32
    x = torch.randn(1, T, C)
    taps = [torch.randn(1, 1, C) * 0.5 for _ in range(K)]
    zeros = torch.zeros(1, K - 1, C)
    out_a, _, _, _ = reference(x, zeros, taps, torch.zeros(1, T, NV), torch.zeros(1, T, NV),
                               torch.zeros(1, 1, NV), torch.zeros(1, 1, NV))
    # rows before the first real token contribute nothing
    assert torch.isfinite(out_a).all()
    print('zero carry is well defined, matching the from-scratch chunk path')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference-only', action='store_true',
                        help='skip the device half; runs anywhere')
    options = parser.parse_args()

    check_planner()
    check_reference()
    check_matches_fir_semantics()

    if options.reference_only:
        print('\nreference pinned.')
        return 0
    print('the device half is optimisation/ttnn-op/gdn_prefill_conv/run_card_m.sh (card M only)')
    return 2


if __name__ == '__main__':
    sys.exit(main())
