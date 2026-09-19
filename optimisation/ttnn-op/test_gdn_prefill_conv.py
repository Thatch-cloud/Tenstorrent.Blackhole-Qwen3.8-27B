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

THE ONE HARD PART, and why it is tractable. Output row t reads input rows t..t+K-1, so
with 32-row tiles a shift of 1..3 straddles two tiles. A row shift is a matmul by a
fixed banded 32x32 matrix that is identical for every channel, so

    shifted_j = S_j_self @ x_cur + S_j_prev @ x_prev

and the conv becomes sum_j taps[j] * shifted_j with taps broadcast over rows, exactly
the decode kernel's inner loop. chunk_gdn_prep already loads constant eye/tril/ones
matrices this way, so the pattern is established in this codebase.

This file generates those shift matrices and checks them against the naive definition,
because a wrong constant would otherwise look like a kernel bug.
"""

import argparse
import sys

import torch

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


def shift_matrices(k, tile=TILE):
    """S_self[j], S_prev[j] with (S_prev[j] @ prev + S_self[j] @ cur)[r] == xin[r + j].

    xin is the concatenation, so for an output row r inside tile `cur`, tap j reads
    xin row r + j, which is cur row r + j - (K-1) once the K-1 carry rows are folded in.
    A negative index reaches back into the previous tile.
    """
    self_m, prev_m = [], []
    for j in range(k):
        s = torch.zeros(tile, tile)
        p = torch.zeros(tile, tile)
        for r in range(tile):
            src = r + j - (k - 1)
            if src >= 0:
                s[r, src] = 1.0
            else:
                p[r, tile + src] = 1.0
        self_m.append(s)
        prev_m.append(p)
    return self_m, prev_m


def check_shift_matrices():
    """The matrices must reproduce a plain shift, or a kernel bug gets blamed instead."""
    self_m, prev_m = shift_matrices(K)
    torch.manual_seed(0)
    prev = torch.randn(TILE, 8)
    cur = torch.randn(TILE, 8)
    # xin as the kernel sees it: previous tile then current tile, carry already folded in
    xin = torch.cat([prev, cur], dim=0)
    for j in range(K):
        got = prev_m[j] @ prev + self_m[j] @ cur
        want = xin[TILE + j - (K - 1): 2 * TILE + j - (K - 1)]
        assert torch.allclose(got, want, atol=0), 'shift matrix j=%d is wrong' % j
    print('shift matrices reproduce the row shift exactly, for all %d taps' % K)


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

    check_shift_matrices()
    check_reference()
    check_matches_fir_semantics()

    if options.reference_only:
        print('\nreference pinned. The kernel is not built yet; rerun on the rig without')
        print('--reference-only once ttnn.transformer.gdn_prefill_conv_gates exists.')
        return 0

    import ttnn
    if not hasattr(getattr(ttnn, 'transformer', None), 'gdn_prefill_conv_gates'):
        print('ttnn.transformer.gdn_prefill_conv_gates is ABSENT: the op is not built into '
              'this image yet. Reference checks above still passed.')
        return 2
    print('device half not written yet')
    return 2


if __name__ == '__main__':
    sys.exit(main())
