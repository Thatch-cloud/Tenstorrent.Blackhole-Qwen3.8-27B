# SPDX-License-Identifier: Apache-2.0
# Mounted into the tt-serving image over
#   /opt/tt-metal/models/demos/blackhole/qwen36/tests/test_gdn_decode_multi.py
# See docs/speculative-decoding.md K4 / §10j.
# Markers like `§10y` refer to numbered entries in the design log this work was
# recorded in. That log is not public; docs/speculative-decoding.md summarises the
# measurements, dead ends and reasoning the markers point at.
"""K4 — a **narrow** multi-token GDN decode, `T = K+1` tokens from the live decode state.

§10j established that the existing multi-token path (chunk-prefill-with-carry) is only usable at
`T ≥ 64`, where it costs 2.50 decode-steps and speculative decoding is break-even. The win needs
`T = 2…4` at ~1.1–1.3× a single decode. That is what this builds.

**It needs no new kernel.** Reading `forward_decode`, the cost is in three places that are all
batchable over `T`:

* `_project_qkvzab(x, S, …)` — takes the row count `S` as a parameter and, for `S <= TILE_SIZE`,
  uses the tuned 1D decode matmul. Feeding it `T` rows reads the weights **once**.
* `_row_proj` — same, same threshold.
* `tt_all_reduce` — one cross-device reduction per call regardless of `T`.

Only two steps are genuinely serial, and both are state-sized rather than weight-sized: the conv
shift-register (`K-1` copies + `K` MACs on `[1, B, qkv_dim_tp]`) and the recurrence itself
(`[B, Nv, Dk, Dv]`, 1.5 MiB fp32). So `f ≈ 1 + (T-1)·(small/total)` rather than the `f = T` that
naively chaining `forward_decode` would cost — and chaining is *worse* than the 64-wide chunk
(4× vs 2.46× at T=4), which is why this is the piece that matters.

Built as a free function over the live layer rather than a patch to `tt/gdn/tp.py`, matching how
K2's stager was validated before proposing anything upstream.

Correctness is judged on **agreement with `T` sequential `forward_decode` calls**, per token and
on the final recurrent state, with a shifted-token control — without the control a high PCC
proves nothing, since adjacent GDN outputs can be similar (the SCORR-11 failure mode).
"""
import os
import sys

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.tt_transformers.tt.ccl import tt_all_reduce

T_TOK = int(os.environ.get("MD_T", "4"))
PCC_OK = float(os.environ.get("MD_PCC_OK", "0.99"))


def forward_decode_multi(gdn, x, T, on_token=None):
    """`T` tokens of ONE user through GDN, continuing the live decode state.

    x: `[1, 1, T, dim]` or `[1, T, dim]`, replicated. Returns `[1, 1, T, dim]`.

    `on_token(t)` is called after token `t`'s recurrence, with the state exactly as it stands
    having consumed tokens `0…t`. That is the hook speculative decoding needs: verify advances
    the state by all `T` tokens but only `n` are accepted, so the state has to come back to
    `n`. Recomputing the accepted prefix instead would add another `f(n)` of GDN — at `K=2` that
    turns 1.53× into ~1.07× — which is why K2 stages rather than replays.

    The row axis that `forward_decode` uses for *users* is reused here for *tokens*. That is
    exactly what makes the projections batch: they are position-agnostic. The conv window and the
    recurrence are not, so those run per token over the same live buffers `forward_decode` uses,
    which also means the state is left where a subsequent `forward_decode` expects it.
    """
    tw, Nk, Nv, Dk, Dv = gdn.tw, gdn.Nk, gdn.Nv, gdn.Dk, gdn.Dv
    _L1 = ttnn.L1_MEMORY_CONFIG
    from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, _softplus_add

    if gdn.conv_states is None:
        gdn.reset_state()
    if len(x.shape) == 4:
        x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
    assert x.shape[-2] == T, f"expected {T} rows, got {list(x.shape)}"
    assert T <= tpc.TILE_SIZE, (
        f"T={T} exceeds TILE_SIZE={tpc.TILE_SIZE}; above it the projections leave the tuned "
        "decode path and the MLP switches to the AG-matmul arm (§10i), which is the whole cost "
        "problem this is meant to avoid."
    )

    # ---- one weight read for all T tokens ----
    qkv, z, a, b = gdn._project_qkvzab(x, T, out_mc=_L1)

    kd, qd = gdn.key_dim_tp, gdn.qkv_dim_tp
    st = gdn.conv_states
    rf = Nv // Nk
    Bmax = gdn.B
    outs = []
    for t in range(T):
        # ---- serial: conv shift-register, one token wide ----
        #
        # The state buffers are `Bmax` wide because they are sized for the *user* batch, and
        # verify runs one user. So each token is a width-1 bucketed decode inside a Bmax-wide
        # state -- exactly the `B < Bmax` path `forward_decode` already takes, reused rather than
        # reimplemented: pad to full width, run the same full-width shift register, slice row 0.
        qkv_t = ttnn.slice(qkv, (0, t, 0), (1, t + 1, qd), memory_config=_L1)
        if Bmax > 1:
            qkv_p = ttnn.pad(qkv_t, [(0, 0), (0, Bmax - 1), (0, 0)], value=0.0, memory_config=_L1)
            ttnn.deallocate(qkv_t)
            qkv_t = qkv_p
        for j in range(gdn.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv_t, st[gdn.K - 1])
        ttnn.deallocate(qkv_t)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
        for j in range(1, gdn.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)

        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, 1, kd)), (1, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, 1, 2 * kd)), (1, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, 1, qd)), (1, Nv, Dv))
        ttnn.deallocate(conv)
        q = ttnn.reshape(ttnn.repeat_interleave(q, rf, dim=1), (1, 1, Nv, Dk), memory_config=_L1)
        k = ttnn.reshape(ttnn.repeat_interleave(k, rf, dim=1), (1, 1, Nv, Dk), memory_config=_L1)
        v = ttnn.reshape(v, (1, 1, Nv, Dv), memory_config=_L1)

        b_t = ttnn.slice(b, (0, t, 0), (1, t + 1, Nv), memory_config=_L1)
        a_t = ttnn.slice(a, (0, t, 0), (1, t + 1, Nv), memory_config=_L1)
        beta = ttnn.reshape(ttnn.sigmoid(b_t, memory_config=_L1), (1, 1, Nv))
        ttnn.deallocate(b_t)
        g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_t, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a_t)
        g = ttnn.reshape(g, (1, 1, Nv))

        # ---- serial: the recurrence, over the live state ----
        init_state = gdn.rec_state if Bmax == 1 else gdn._slice_along(gdn.rec_state, 0, 0, 1)
        o_t, new_rec = recurrent_gated_delta_rule_decode_ttnn(
            q, k, v, beta, g, scale=gdn.scale, initial_state=init_state, device=gdn.mesh,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        if init_state is not gdn.rec_state:
            ttnn.deallocate(init_state)
        if gdn._stable_state:
            if Bmax == 1:
                ttnn.copy(new_rec, gdn.rec_state)
                ttnn.deallocate(new_rec)
            else:
                gdn._write_recurrent_state_prefix(new_rec, 1)
        else:
            gdn.rec_state = new_rec
        if on_token is not None:
            on_token(t)
        outs.append(ttnn.reshape(o_t, (1, 1, gdn.value_dim_tp)))

    ttnn.deallocate(qkv)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    # ---- batched again from here: norm, gate, out-projection, one all-reduce ----
    out_f = outs[0] if T == 1 else ttnn.concat(outs, dim=1)
    out_r = ttnn.reshape(out_f, (T, Nv, Dv))
    out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
    ttnn.deallocate(out_r)
    out_f = ttnn.reshape(out_n, (1, T, gdn.value_dim_tp))
    ttnn.deallocate(out_n)
    gated = _silu_mul(out_f, z, _L1)
    ttnn.deallocate(out_f)
    ttnn.deallocate(z)

    partial = gdn._row_proj(gated, tw["out"])
    ttnn.deallocate(gated)
    partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
    return tt_all_reduce(
        partial, gdn.mesh, gdn.tt_ccl, cluster_axis=0, dim=3,
        topology=gdn.args.ccl_topology(),
    )


def forward_decode_multi_batched(gdn, x, T, on_token=None):
    """Same contract as `forward_decode_multi`, but only the recurrence stays serial.

    The first version treated the conv1d as sequential because `forward_decode` does — a
    shift-register advanced once per token. **It is not sequential.** Output `t` is
    `Σ_j taps[j] · X[t+j]` over `X = [carry_{K-1} … carry_1, x_0 … x_{T-1}]`, so all `T` outputs
    come from `K` slices and `K` MACs over the whole window at once, rather than `T × 2K` ops.
    The gates (`sigmoid`, `softplus`) are elementwise and batch the same way.

    That matters because §10k measured the serial part at **62% of a whole decode step** — the
    figure that says a fused kernel is needed. Most of that 62% was the conv and the gates, not
    the recurrence, and none of it needed to be serial.

    What genuinely cannot batch: the recurrence, which reads the state it just wrote.
    """
    tw, Nk, Nv, Dk, Dv = gdn.tw, gdn.Nk, gdn.Nv, gdn.Dk, gdn.Dv
    _L1 = ttnn.L1_MEMORY_CONFIG
    from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, _softplus_add

    if gdn.conv_states is None:
        gdn.reset_state()
    if len(x.shape) == 4:
        x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
    assert T <= tpc.TILE_SIZE, f"T={T} exceeds TILE_SIZE={tpc.TILE_SIZE}"

    qd, kd, K, Bmax = gdn.qkv_dim_tp, gdn.key_dim_tp, gdn.K, gdn.B
    qkv, z, a, b = gdn._project_qkvzab(x, T, out_mc=_L1)

    # ---- batched conv: one window, K slices + K MACs, independent of T ----
    st = gdn.conv_states
    carry = [ttnn.slice(st[j], (0, 0, 0), (1, 1, qd), memory_config=_L1) for j in range(1, K)]
    win = ttnn.concat(carry + [qkv], dim=1)                     # [1, T+K-1, qd]
    for c in carry:
        ttnn.deallocate(c)
    conv = ttnn.multiply(ttnn.slice(win, (0, 0, 0), (1, T, qd), memory_config=_L1),
                         tw["conv_taps"][0], memory_config=_L1)
    for j in range(1, K):
        conv = ttnn.mac(ttnn.slice(win, (0, j, 0), (1, j + T, qd), memory_config=_L1),
                        tw["conv_taps"][j], conv)
    conv = ttnn.silu(conv, memory_config=_L1)                   # [1, T, qd]

    # ---- batched gates ----
    beta_all = ttnn.sigmoid(b, memory_config=_L1)               # [1, T, Nv]
    g_all = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    rf = Nv // Nk
    outs = []
    for t in range(T):
        q = ttnn.reshape(ttnn.slice(conv, (0, t, 0), (1, t + 1, kd)), (1, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, t, kd), (1, t + 1, 2 * kd)), (1, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, t, 2 * kd), (1, t + 1, qd)), (1, Nv, Dv))
        q = ttnn.reshape(ttnn.repeat_interleave(q, rf, dim=1), (1, 1, Nv, Dk), memory_config=_L1)
        k = ttnn.reshape(ttnn.repeat_interleave(k, rf, dim=1), (1, 1, Nv, Dk), memory_config=_L1)
        v = ttnn.reshape(v, (1, 1, Nv, Dv), memory_config=_L1)
        beta = ttnn.reshape(ttnn.slice(beta_all, (0, t, 0), (1, t + 1, Nv)), (1, 1, Nv))
        g = ttnn.reshape(ttnn.slice(g_all, (0, t, 0), (1, t + 1, Nv)), (1, 1, Nv))

        init_state = gdn.rec_state if Bmax == 1 else gdn._slice_along(gdn.rec_state, 0, 0, 1)
        o_t, new_rec = recurrent_gated_delta_rule_decode_ttnn(
            q, k, v, beta, g, scale=gdn.scale, initial_state=init_state, device=gdn.mesh,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        if init_state is not gdn.rec_state:
            ttnn.deallocate(init_state)
        if gdn._stable_state:
            if Bmax == 1:
                ttnn.copy(new_rec, gdn.rec_state)
                ttnn.deallocate(new_rec)
            else:
                gdn._write_recurrent_state_prefix(new_rec, 1)
        else:
            gdn.rec_state = new_rec
        if on_token is not None:
            # The stager reads `conv_states`, so they have to be materialised at this token
            # before the slot is taken -- K copies, paid only when staging is on.
            _sync_conv_states(gdn, win, t, T, qd, K, Bmax)
            on_token(t)
        outs.append(ttnn.reshape(o_t, (1, 1, gdn.value_dim_tp)))

    if on_token is None:
        _sync_conv_states(gdn, win, T - 1, T, qd, K, Bmax)
    ttnn.deallocate(win)
    ttnn.deallocate(conv)
    ttnn.deallocate(beta_all)
    ttnn.deallocate(g_all)

    out_f = outs[0] if T == 1 else ttnn.concat(outs, dim=1)
    out_r = ttnn.reshape(out_f, (T, Nv, Dv))
    out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
    ttnn.deallocate(out_r)
    out_f = ttnn.reshape(out_n, (1, T, gdn.value_dim_tp))
    ttnn.deallocate(out_n)
    gated = _silu_mul(out_f, z, _L1)
    ttnn.deallocate(out_f)
    ttnn.deallocate(z)
    partial = gdn._row_proj(gated, tw["out"])
    ttnn.deallocate(gated)
    partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
    return tt_all_reduce(partial, gdn.mesh, gdn.tt_ccl, cluster_axis=0, dim=3,
                         topology=gdn.args.ccl_topology())


_DR = ttnn.DRAM_MEMORY_CONFIG
_M = 32   # the kernel's mini-batch row count: `Mt = k.shape[2] / TILE_HEIGHT`, and Mt=0 is silent

# Deliberately-wrong arms of the fused path, so the test can be shown to discriminate rather
# than merely to agree with itself. `none` is the real one.
#   nobeta32 — drop the /32 that pays for the 32 replicated k rows (construction R)
#   gqa      — map value head n to k-head `n % Nk` instead of `n // rf`
#   noexp    — hand the kernel exp(g) as if it did not apply exp itself
MD_CTRL = os.environ.get("MD_CTRL", "none")


def _rel_l2(got, ref):
    """||got - ref|| / ||ref||.

    PCC is close to useless on the recurrent state: it is scale-invariant, the state is
    dominated by a few large components, and on ~800k elements `compute_pcc`'s float32
    accumulation rounds *above* 1.0 (the reference composition reports 1.000098 against
    itself). Relative L2 is scale-sensitive and is what the state verdict rests on.
    """
    g, r = got.flatten().float(), ref.flatten().float()
    return float((g - r).norm() / (r.norm() + 1e-30))


def _l2norm_scale(x, extra=None):
    """`l2_norm_ttnn(x, dim=-1)`, optionally folded with a scalar, forced to DRAM.

    The epsilon must be `1e-6 / K`, not `1e-6`: `rms_norm` adds eps to the MEAN of squares, so
    pre-dividing is what turns it back into eps on the SUM — a true L2 norm. On random
    activations the difference is invisible; on a near-zero row it is not.

    `l2_norm_ttnn` itself returns L1 for `shape[1] <= 512` and the op's validator TT_FATALs on
    L1 operands, which is the only reason this is a local copy rather than a call.
    """
    Kd = x.shape[-1]
    n = ttnn.rms_norm(x, epsilon=1e-6 / Kd, memory_config=_DR)
    s = Kd**-0.5 if extra is None else (Kd**-0.5) * extra
    out = ttnn.multiply(n, s, memory_config=_DR)
    ttnn.deallocate(n)
    return out


def _state_slice(states, t, Nv, Dk, Dv):
    """`states[:, t]` as `[1, Nv, Dk, Dv]`, plus whether the result owns its buffer.

    `ttnn.slice` RETURNS THE INPUT TENSOR when the slice covers everything, and `ttnn.reshape`
    keeps that alias — measured: identity slice and its reshape both report the input's buffer
    address, and deallocating the view leaves the original `is_allocated() == False`. At `T == 1`
    the final-state slice is exactly the identity, so freeing it silently destroyed the `states`
    tensor the stager had just been handed. It cost nothing at T >= 2 and failed only at T = 1,
    which is the mirror image of the "always test a loop at T >= 2" rule: also test it at 1.
    """
    s = ttnn.slice(states, (0, t, 0, 0), (Nv, t + 1, Dk, Dv), memory_config=_DR)
    owned = s.buffer_address() != states.buffer_address()
    return ttnn.reshape(s, (1, Nv, Dk, Dv)), owned


def _head_major_rows(x, H, T, D):
    """`[1, T, H, D]` (token-major, head inside one tile row) -> `[H, T, 32, D]` (head-major).

    This is the whole integration problem. The kernel wants the head axis in dim 0 — one head per
    core — and `M = 32` real token rows per step; the model hands us 24 heads packed *inside* a
    single 32-row tile. `permute` moves those sub-tile rows out into separate pages, and `repeat`
    fills all 32 M rows with copies of the one live row (construction R).

    Replicating the row rather than zero-padding it is deliberate. `Mt` is `k.shape[2] / 32`, so
    a 1-row k gives `Mt = 0` and every compute loop runs zero times — the op returns with the
    state unchanged and no error. Padding to 32 instead would make correctness depend on rows
    1..31 of a *device op's* output tile being zero, which nothing guarantees. With all 32 rows
    live and `beta/32`, `k^T @ delta` sums 32 copies of 1/32 of the wanted rank-1 term: exact up
    to fp32 rounding, bias-free because 1/32 is a power of two, and every row of `o` is then
    correct so the row-0 slice is safe regardless of tile-pad semantics.

    NOT `ttnn.bcast(ones, row, MUL, H)`, which is what the obvious reading of the design says:
    measured against torch it lands 3.1e-2 off on fp32 inputs — bf16 precision — and would have
    silently thrown away most of the fp32 state accuracy the kernel exists to provide.
    `permute` + `repeat` are bit-exact (max abs err 0.0).
    """
    p = ttnn.permute(x, (2, 1, 0, 3), memory_config=_DR)      # [H, T, 1, D]
    r = ttnn.repeat(p, ttnn.Shape([1, 1, _M, 1]))             # [H, T, 32, D]
    ttnn.deallocate(p)
    assert list(r.shape) == [H, T, _M, D], f"head-major rows {list(r.shape)}"
    return r


def _head_major_scalar(x, H, T):
    """`[1, T, H]` -> `[H, T, 1, 1]`, the kernel's per-head-per-token scalar layout.

    `T` tokens have to move out of one tile row into `T` separate pages; the 4-D reshape is free
    (leading dims only) and the permute does the scatter.
    """
    r4 = ttnn.reshape(x, (1, T, H, 1))
    s = ttnn.permute(r4, (2, 1, 0, 3), memory_config=_DR)     # [H, T, 1, 1]
    assert list(s.shape) == [H, T, 1, 1], f"head-major scalar {list(s.shape)}"
    return s


def forward_decode_multi_fused(gdn, x, T, stage=None):
    """Same contract as the other two, with the recurrence in ONE dispatch of the fused kernel.

    `ttnn._ttnn.operations.transformer.gdn_recurrent_step` computes all `T` steps of

        h1 = state*exp(g);  delta = (v - k@h1)*beta;  state' = h1 + k^T@delta;  o = q@state'

    for every head at once, one head per core, and returns **all `T` intermediate states**. That
    last part is what makes the `on_token` staging hook redundant: the snapshots speculative
    verify needs to roll back to are a by-product of the recurrence rather than `T*(1+K)` device
    copies per layer. `stage` is the out-dict that receives them.

    Three things `recurrent_gated_delta_rule_decode_ttnn` did internally now belong to the
    caller — fp32 typecast, the L2 norm of q and k, and `gdn.scale` on q. All three are per-row,
    so they commute with both the token axis and the GQA fan-out and hoist out of the loop: 7
    device ops for the whole call instead of ~10 per token.

    Use the `_ttnn` path, not `ttnn.transformer.*` — the latter is a curated re-export and does
    not pick up new bindings.
    """
    tw, Nk, Nv, Dk, Dv = gdn.tw, gdn.Nk, gdn.Nv, gdn.Dk, gdn.Dv
    _L1 = ttnn.L1_MEMORY_CONFIG
    from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, _softplus_add

    if gdn.conv_states is None:
        gdn.reset_state()
    if len(x.shape) == 4:
        x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
    assert T <= tpc.TILE_SIZE, f"T={T} exceeds TILE_SIZE={tpc.TILE_SIZE}"
    # The op takes h as [BH,1,K,V] -- one [K,V] state per head, ONE user. `gdn.B` is not the user
    # count though, it is the bucket WIDTH: the traced serving path rounds T up to a power of two,
    # so verify's single user runs inside a Bmax=4 state buffer. Handle that the way both
    # reference impls do -- slice user row 0 on the way in, `_write_recurrent_state_prefix` on the
    # way out -- rather than refusing, which would make the traced measurement impossible.
    assert gdn.rec_state.dtype == ttnn.float32, (
        f"the op validates fp32 operands; rec_state is {gdn.rec_state.dtype} "
        "(QWEN35_GDN_STATE_BF16=1?)"
    )
    assert os.environ.get("QWEN35_GDN_DECODE_BF16") != "1", "the op has no bf16 mode"

    qd, kd, K, Bmax = gdn.qkv_dim_tp, gdn.key_dim_tp, gdn.K, gdn.B
    qkv, z, a, b = gdn._project_qkvzab(x, T, out_mc=_L1)

    # ---- batched conv and gates: unchanged from `forward_decode_multi_batched` ----
    st = gdn.conv_states
    _fast = _fast_carry_enabled() and getattr(gdn, "_spec_carry", None) is not None
    if _fast:
        # The K-1 taps already sit contiguously in the persistent buffer, so the window is one
        # concat: no per-tap slice, no deallocs.
        # Clone the carry into the window rather than concatenating the live buffer. The
        # end-of-call sync writes `_spec_carry` from `win`, and `commit_prefix` later reads
        # `stage["win"]` rows n..n+K-2 -- which for small n ARE the carry rows. If `win` aliases
        # the buffer, that write corrupts the very rows the rollback reads, and the rollback
        # PCC collapses to ~-0.5 while the forward itself still looks perfect.
        win = ttnn.concat([ttnn.clone(gdn._spec_carry), qkv], dim=1)   # [1, T+K-1, qd]
    else:
        carry = [ttnn.slice(st[j], (0, 0, 0), (1, 1, qd), memory_config=_L1) for j in range(1, K)]
        win = ttnn.concat(carry + [qkv], dim=1)                 # [1, T+K-1, qd]
        for c in carry:
            ttnn.deallocate(c)
    conv = ttnn.multiply(ttnn.slice(win, (0, 0, 0), (1, T, qd), memory_config=_L1),
                         tw["conv_taps"][0], memory_config=_L1)
    for j in range(1, K):
        conv = ttnn.mac(ttnn.slice(win, (0, j, 0), (1, j + T, qd), memory_config=_L1),
                        tw["conv_taps"][j], conv)
    conv = ttnn.silu(conv, memory_config=_L1)                   # [1, T, qd]

    # Computed in the model dtype exactly as the reference does, then typecast — the reference
    # also runs sigmoid/softplus in bf16 and only widens inside the recurrence.
    beta_all = ttnn.sigmoid(b, memory_config=_L1)               # [1, T, Nv]
    g_all = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    # ---- head blocks, built once for all T tokens ----
    rf = Nv // Nk
    q_bf = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
    k_bf = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
    v_bf = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, qd)), (1, T, Nv, Dv))
    ttnn.deallocate(conv)
    # GQA fan-out before the permute: head n takes k-head n // rf. `n % Nk` or an off-by-one
    # still correlates strongly on random data (the SCORR-11 failure mode) -- only the head-shift
    # control in the test catches it.
    if MD_CTRL == "gqa":
        # `repeat` tiles [h0..h7, h0..h7, h0..h7] where `repeat_interleave` gives
        # [h0,h0,h0, h1,h1,h1, …] -- i.e. exactly the `n % Nk` mapping.
        q_bf = ttnn.repeat(q_bf, ttnn.Shape([1, 1, rf, 1]))
        k_bf = ttnn.repeat(k_bf, ttnn.Shape([1, 1, rf, 1]))
    else:
        q_bf = ttnn.repeat_interleave(q_bf, rf, dim=2)          # [1, T, Nv, Dk]
        k_bf = ttnn.repeat_interleave(k_bf, rf, dim=2)

    # The L1->DRAM move the op's validator demands is folded into the mandatory fp32 typecast.
    q32 = ttnn.typecast(q_bf, ttnn.float32, memory_config=_DR)
    k32 = ttnn.typecast(k_bf, ttnn.float32, memory_config=_DR)
    v32 = ttnn.typecast(v_bf, ttnn.float32, memory_config=_DR)
    for t_ in (q_bf, k_bf, v_bf):
        ttnn.deallocate(t_)
    k_n = _l2norm_scale(k32)                       # L2 only, NO scale
    q_n = _l2norm_scale(q32, gdn.scale)            # L2 then * scale; folded to 1/Dk exactly
    ttnn.deallocate(q32)
    ttnn.deallocate(k32)

    k_m = _head_major_rows(k_n, Nv, T, Dk)
    q_m = _head_major_rows(q_n, Nv, T, Dk)
    v_m = _head_major_rows(v32, Nv, T, Dv)
    ttnn.deallocate(k_n)
    ttnn.deallocate(q_n)
    ttnn.deallocate(v32)

    beta32 = ttnn.typecast(beta_all, ttnn.float32, memory_config=_DR)
    g32 = ttnn.typecast(g_all, ttnn.float32, memory_config=_DR)
    ttnn.deallocate(beta_all)
    ttnn.deallocate(g_all)
    # beta/32 pays for the 32 replicated k rows (construction R above).
    beta_sc = beta32 if MD_CTRL == "nobeta32" else ttnn.multiply(beta32, 1.0 / _M, memory_config=_DR)
    beta_s = _head_major_scalar(beta_sc, Nv, T)
    g_sc = ttnn.exp(g32, memory_config=_DR) if MD_CTRL == "noexp" else g32
    g_s = _head_major_scalar(g_sc, Nv, T)          # raw log-decay; the kernel applies exp
    for _t in {id(beta_sc): beta_sc, id(beta32): beta32, id(g_sc): g_sc, id(g32): g32}.values():
        ttnn.deallocate(_t)

    # [1,Nv,K,V] -> [Nv,1,K,V] is pure metadata: both index pages as (head*Kt + r)*Vt + c.
    init_state = gdn.rec_state if Bmax == 1 else gdn._slice_along(gdn.rec_state, 0, 0, 1)
    h0 = ttnn.reshape(init_state, (Nv, 1, Dk, Dv))
    o_h, states = ttnn._ttnn.operations.transformer.gdn_recurrent_step(h0, g_s, k_m, v_m, beta_s, q_m)
    for t_ in (g_s, k_m, v_m, beta_s, q_m):
        ttnn.deallocate(t_)
    if init_state is not gdn.rec_state:
        ttnn.deallocate(init_state)

    # ---- back to token-major ----
    o_row = ttnn.slice(o_h, (0, 0, 0, 0), (Nv, T, 1, Dv), memory_config=_DR)
    ttnn.deallocate(o_h)
    o_tn = ttnn.permute(o_row, (2, 1, 0, 3), memory_config=_DR)          # [1, T, Nv, Dv]
    ttnn.deallocate(o_row)

    # ---- commit the final state; `_stable_state` needs rec_state to keep its address ----
    last, owned = _state_slice(states, T - 1, Nv, Dk, Dv)
    if gdn._stable_state:
        if Bmax == 1:
            ttnn.copy(last, gdn.rec_state)
        else:
            gdn._write_recurrent_state_prefix(last, 1)
        if owned:
            ttnn.deallocate(last)
    else:
        # Never hand out an alias of `states`: the stager holds that, and whichever of the two is
        # freed first would take the other with it.
        gdn.rec_state = last if owned else ttnn.clone(last, memory_config=_DR)
    if _fast:
        # Both: the carry for the next fused call (2 ops) AND conv_states, so a width-1
        # `forward_decode` after this call still sees the truth. The saving that matters is in
        # `commit_prefix`, which fires on rejection and is where the 12 ops repeat.
        _sync_spec_carry(gdn, win, T - 1, qd, K)
    _sync_conv_states(gdn, win, T - 1, T, qd, K, Bmax)

    if stage is not None:
        # Owned by the caller until `commit_prefix`; deallocating either here would hand the
        # stager dangling buffers.
        stage["states"] = states
        stage["win"] = win
    else:
        ttnn.deallocate(states)
        ttnn.deallocate(win)

    out_r = ttnn.reshape(o_tn, (T, Nv, Dv))
    out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
    ttnn.deallocate(out_r)
    out_f = ttnn.reshape(out_n, (1, T, gdn.value_dim_tp))
    ttnn.deallocate(out_n)
    gated = _silu_mul(out_f, z, _L1)
    ttnn.deallocate(out_f)
    ttnn.deallocate(z)
    partial = gdn._row_proj(gated, tw["out"])
    ttnn.deallocate(gated)
    partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
    return tt_all_reduce(partial, gdn.mesh, gdn.tt_ccl, cluster_axis=0, dim=3,
                         topology=gdn.args.ccl_topology())


def commit_prefix(gdn, stage, n, T, fused_only=False):
    """Roll the layer back to having consumed exactly `n` of the `T` speculated tokens.

    `states[:, n-1]` is the state after tokens `0…n-1`, which is precisely `on_token(n-1)`'s
    contract — so this replaces the stager at zero device-copy cost.

    `n == 0` is NOT supported: the kernel does not emit the seed state. In speculative verify the
    first token is the target model's own next token and is always accepted, so `n >= 1`; if that
    ever stops holding, snapshot `rec_state` before the call.
    """
    assert 1 <= n <= T, f"n={n} outside 1..{T}"
    part, owned = _state_slice(stage["states"], n - 1, gdn.Nv, gdn.Dk, gdn.Dv)
    if gdn.B == 1:
        ttnn.copy(part, gdn.rec_state)
    else:
        gdn._write_recurrent_state_prefix(part, 1)
    if owned:
        ttnn.deallocate(part)   # at T == n == 1 this IS `stage["states"]` -- see _state_slice
    # `fused_only=True` is the caller stating that nothing will read `conv_states` before the
    # next fused call -- which lets the carry collapse from 12 device ops per layer to 2. It is
    # NOT inferable here: `gdn.forward_decode` reads `conv_states[0..K-1]` and the fused path
    # reads only `_spec_carry`, so getting it wrong is silent staleness, not an error. Default
    # False keeps the always-correct path.
    if fused_only and getattr(gdn, "_spec_carry", None) is not None:
        _sync_spec_carry(gdn, stage["win"], n - 1, gdn.qkv_dim_tp, gdn.K)
    else:
        _sync_conv_states(gdn, stage["win"], n - 1, T, gdn.qkv_dim_tp, gdn.K, gdn.B)


def _fast_carry_enabled():
    """Off by default so §10ab's measurements reproduce byte-for-byte."""
    return os.environ.get("SPEC_FAST_CARRY", "0") == "1"


def alloc_spec_carry(gdns):
    """One persistent [1, K-1, qkv_dim_tp] carry per layer.

    Allocated before any trace capture and never rebound: the captured forward reads this
    address, so a reallocation would silently detach the trace from the buffer it reads --
    exactly the failure `reset_state_inplace` exists to avoid for `rec_state`.
    """
    for g in gdns:
        if g.conv_states is None:
            g.reset_state()
        rows = [ttnn.slice(g.conv_states[j], (0, 0, 0), (1, 1, g.qkv_dim_tp),
                           memory_config=ttnn.L1_MEMORY_CONFIG) for j in range(1, g.K)]
        buf = ttnn.concat(rows, dim=1) if len(rows) > 1 else rows[0]
        g._spec_carry = ttnn.clone(buf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(rows) > 1:
            ttnn.deallocate(buf)
        for r in rows:
            ttnn.deallocate(r)
    return [g._spec_carry for g in gdns]


def _sync_spec_carry(gdn, win, t, qd, K):
    """Carry as of token `t`, in two ops instead of twelve.

    The fused path reads taps 1..K-1, which are `win[t+1 .. t+K-1]` -- contiguous, so one slice
    covers all of them and one copy lands them. Tap 0 is not written at all: the fused carry
    build never reads `conv_states[0]`.
    """
    src = ttnn.slice(win, (0, t + 1, 0), (1, t + K, qd), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.copy(src, gdn._spec_carry)
    ttnn.deallocate(src)


def _sync_conv_states(gdn, win, t, T, qd, K, Bmax):
    """Write the conv window as of token `t` back into `gdn.conv_states`.

    `win` is `[carry_{K-1} … carry_1, x_0 … x_{T-1}]`, so `win[i]` holds token `i-(K-1)` and the
    tap `j` for token `t` — which is token `t-(K-1)+j` — sits at `win[t+j]`. Only needed where something reads `conv_states` -- a stager slot, or the
    end of the call so the next one carries correctly.
    """
    for j in range(K):
        s = ttnn.slice(win, (0, t + j, 0), (1, t + j + 1, qd), memory_config=ttnn.L1_MEMORY_CONFIG)
        if Bmax > 1:
            s_p = ttnn.pad(s, [(0, 0), (0, Bmax - 1), (0, 0)], value=0.0,
                           memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(s)
            s = s_p
        ttnn.copy(s, gdn.conv_states[j])
        ttnn.deallocate(s)


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_decode_multi(mesh_device, reset_seeds, ensure_gc, request):
    os.environ.setdefault("HF_MODEL", model_path())
    B = 1
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    gdn = TPGatedDeltaNet(
        mesh_device, args, load_gdn_weights_tp(mesh_device, load_gdn_layer(args.CKPT_DIR, li), args), tt_ccl
    )
    gdn._stable_state = True
    if _fast_carry_enabled():
        alloc_spec_carry([gdn])
        assert getattr(gdn, '_spec_carry', None) is not None, 'fast carry did not allocate'
        print(f'MD fast_carry shape={list(gdn._spec_carry.shape)}', flush=True)
    comp = tp_composer(mesh_device)
    # 3-way: `plain` and `batched` are the validated references the fused path is judged
    # against and are never touched. MD_BATCHED=1 is still honoured so older runners keep working.
    _IMPLS = {"plain": forward_decode_multi, "batched": forward_decode_multi_batched,
              "fused": forward_decode_multi_fused}
    md_impl = os.environ.get("MD_IMPL", "batched" if os.environ.get("MD_BATCHED") == "1" else "plain")
    assert md_impl in _IMPLS, f"MD_IMPL={md_impl} not in {sorted(_IMPLS)}"
    impl = _IMPLS[md_impl]
    print(f"MD devices={nd} dim={args.dim} T={T_TOK} TILE={tpc.TILE_SIZE} K(conv)={gdn.K} "
          f"Nk={gdn.Nk} Nv={gdn.Nv} Dk={gdn.Dk} Dv={gdn.Dv} impl={impl.__name__}", flush=True)

    toks = [torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16) for _ in range(T_TOK)]
    tt_toks = [replicate_to_device(mesh_device, t) for t in toks]

    # ---- reference: T sequential decodes ----
    #
    # Every INTERMEDIATE state is captured, not just the last one. The old test only checked the
    # final state, which would pass even if every intermediate were garbage -- and with the fused
    # kernel the intermediates are the entire point, since they are what speculative rollback
    # restores from.
    gdn.reset_state()
    ref, ref_states = [], []
    for t in range(T_TOK):
        o = gdn.forward_decode(tt_toks[t])
        ref.append(ttnn.to_torch(o, mesh_composer=comp)[0, 0].float().reshape(-1))
        ttnn.deallocate(o)
        ref_states.append(ttnn.to_torch(gdn.rec_state, mesh_composer=comp).float())
    ref_state = ref_states[-1]

    # ---- multi-token, same tokens, same starting state ----
    gdn.reset_state()
    x = torch.cat([t.reshape(1, 1, 1, args.dim) for t in toks], dim=2)   # [1,1,T,dim]
    stage = {} if impl is forward_decode_multi_fused else None
    out = impl(gdn, replicate_to_device(mesh_device, x), T_TOK, **({"stage": stage} if stage is not None else {}))
    got = ttnn.to_torch(out, mesh_composer=comp).float()
    got = got.reshape(-1, got.shape[-1])
    multi = [got[i][: args.dim] if got.shape[-1] != args.dim else got[i] for i in range(T_TOK)]
    multi_state = ttnn.to_torch(gdn.rec_state, mesh_composer=comp).float()

    pccs = [compute_pcc(ref[t], multi[t]) for t in range(T_TOK)]
    for t, p in enumerate(pccs):
        print(f"MD pos={t} pcc={p:.6f}", flush=True)
    sp = compute_pcc(ref_state, multi_state)
    print(f"MD final_rec_state pcc={sp:.6f}", flush=True)

    # Control: the same outputs compared against the wrong token. Without this a high PCC is not
    # evidence -- adjacent GDN outputs on random inputs can still correlate.
    # At T=1 the "shifted" token is (0+1)%1 = 0, i.e. the output compared against itself, so the
    # control is vacuous and scores 1.0 for every impl including the two references. Say so
    # rather than asserting on it -- a control that cannot fail is not a control.
    ctrl = [compute_pcc(ref[t], multi[(t + 1) % T_TOK]) for t in range(T_TOK)] if T_TOK > 1 else []
    print(f"MD control shifted_max={(max(ctrl) if ctrl else float('nan')):.6f} "
          f"matched_min={min(pccs):.6f}"
          + ("" if ctrl else "  [SKIPPED: degenerate at T=1, the shift maps onto itself]"),
          flush=True)

    # ---- per-token INTERMEDIATE state, with a head-shift control ----
    #
    # The head-shift control is the only thing that catches a GQA mapping error. Head n must take
    # k-head n//rf; `n % Nk` or an off-by-one still produces output that correlates strongly with
    # the reference on random data. Rolling the head axis by one shows what a wrong mapping looks
    # like on this metric, so a high matched PCC is evidence rather than an assumption.
    st_pccs, st_ctrl, st_rel, st_rel_ctrl = [], [], [], []
    if stage is not None:
        for t in range(T_TOK):
            part, owned = _state_slice(stage["states"], t, gdn.Nv, gdn.Dk, gdn.Dv)
            g_st = ttnn.to_torch(part, mesh_composer=comp).float()
            if owned:
                ttnn.deallocate(part)
            st_pccs.append(compute_pcc(ref_states[t], g_st))
            st_ctrl.append(compute_pcc(ref_states[t], g_st.roll(1, dims=1)))
            st_rel.append(_rel_l2(g_st, ref_states[t]))
            st_rel_ctrl.append(_rel_l2(g_st.roll(1, dims=1), ref_states[t]))
            print(f"MD state t={t} pcc={st_pccs[-1]:.6f} rel={st_rel[-1]:.3e} "
                  f"headshift_ctrl_pcc={st_ctrl[-1]:.6f} headshift_ctrl_rel={st_rel_ctrl[-1]:.3e}",
                  flush=True)
        ratio = min(st_rel_ctrl) / max(max(st_rel), 1e-12)
        print(f"MD state_series min_pcc={min(st_pccs):.6f} max_rel={max(st_rel):.3e} "
              f"headshift_max_pcc={max(st_ctrl):.6f} headshift_min_rel={min(st_rel_ctrl):.3e} "
              f"rel_ratio={ratio:.1f}x", flush=True)

    o_rel = [_rel_l2(multi[t], ref[t]) for t in range(T_TOK)]
    o_rel_ctrl = [_rel_l2(multi[(t + 1) % T_TOK], ref[t]) for t in range(T_TOK)] if T_TOK > 1 else []
    print(f"MD out max_rel={max(o_rel):.3e} "
          + (f"shifted_min_rel={min(o_rel_ctrl):.3e} "
             f"rel_ratio={min(o_rel_ctrl) / max(max(o_rel), 1e-12):.1f}x" if o_rel_ctrl
             else "[shifted control skipped: degenerate at T=1]"), flush=True)

    print(f"MD VERDICT impl={impl.__name__} ctrl={MD_CTRL} min_pcc={min(pccs):.6f} "
          f"out_max_rel={max(o_rel):.3e} state_pcc={sp:.6f} "
          f"ctrl_max={(max(ctrl) if ctrl else float('nan')):.6f}"
          + (f" per_tok_state_min={min(st_pccs):.6f} per_tok_state_max_rel={max(st_rel):.3e} "
             f"headshift_max={max(st_ctrl):.6f}" if st_pccs else ""), flush=True)
    if MD_CTRL != "none":
        print(f"MD NOTE ctrl={MD_CTRL} is a DELIBERATELY WRONG arm; a pass here would mean the "
              "test cannot see the thing it claims to test", flush=True)

    assert not ctrl or max(ctrl) < min(pccs), (
        f"control did not separate: shifted {max(ctrl):.4f} >= matched {min(pccs):.4f}"
    )
    assert min(pccs) > PCC_OK, f"multi-token decode != sequential decode ({min(pccs):.6f})"
    assert sp > PCC_OK, f"final recurrent state diverged ({sp:.6f})"
    if st_pccs:
        assert min(st_pccs) > PCC_OK, (
            f"an intermediate state diverged ({min(st_pccs):.6f}) -- speculative rollback "
            "restores from these, so the final-state check alone is not enough"
        )
        assert max(st_ctrl) < min(st_pccs), (
            f"head-shift control did not separate: {max(st_ctrl):.4f} >= {min(st_pccs):.4f}; "
            "the per-token state PCC cannot distinguish a GQA head-mapping error"
        )
        # The metric that actually carries the state verdict. PCC saturates here.
        assert max(st_rel) < 1e-2, f"intermediate state relative L2 {max(st_rel):.3e} too large"
        assert min(st_rel_ctrl) > 10 * max(st_rel), (
            f"head-shift control is only {min(st_rel_ctrl) / max(st_rel):.1f}x further away in "
            "relative L2; the state check cannot discriminate"
        )


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_fused_rollback(mesh_device, reset_seeds, ensure_gc, request):
    """`commit_prefix(n)` must leave the layer exactly as `n` sequential decodes would.

    This is the whole reason the fused kernel is worth wiring in: `states` [Nv,T,K,V] already
    holds every snapshot the speculative verify step needs, so rolling back to `n` accepted
    tokens costs one slice and one copy instead of the stager's `T*(1+K)` device copies per layer.

    Judged the way rollback is actually used — restore, then decode one MORE token and compare
    against the sequential run. That exercises the conv window as well as the recurrent state; a
    test that only reads back `rec_state` would pass with `_sync_conv_states` deleted.
    """
    if os.environ.get("MD_IMPL", "fused") != "fused":
        import pytest

        pytest.skip("rollback is a property of the fused path only")
    os.environ.setdefault("HF_MODEL", model_path())
    B, T = 1, T_TOK
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    gdn = TPGatedDeltaNet(
        mesh_device, args, load_gdn_weights_tp(mesh_device, load_gdn_layer(args.CKPT_DIR, li), args), tt_ccl
    )
    gdn._stable_state = True
    if _fast_carry_enabled():
        alloc_spec_carry([gdn])
        assert getattr(gdn, '_spec_carry', None) is not None, 'fast carry did not allocate'
        print(f'MD fast_carry shape={list(gdn._spec_carry.shape)}', flush=True)
    comp = tp_composer(mesh_device)

    toks = [torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16) for _ in range(T)]
    probe = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    tt_toks = [replicate_to_device(mesh_device, t) for t in toks]
    tt_probe = replicate_to_device(mesh_device, probe)

    # Reference: n sequential decodes, then the probe token.
    ref = {}
    for n in range(1, T + 1):
        gdn.reset_state()
        for t in range(n):
            ttnn.deallocate(gdn.forward_decode(tt_toks[t]))
        o = gdn.forward_decode(tt_probe)
        ref[n] = ttnn.to_torch(o, mesh_composer=comp)[0, 0].float().reshape(-1)
        ttnn.deallocate(o)

    # Fused: one multi-token call, then roll back to each n in turn.
    gdn.reset_state()
    stage = {}
    x = torch.cat([t.reshape(1, 1, 1, args.dim) for t in toks], dim=2)
    ttnn.deallocate(forward_decode_multi_fused(gdn, replicate_to_device(mesh_device, x), T, stage=stage))

    got = {}
    _fo = os.environ.get("MD_FUSED_ONLY") == "1"
    for n in range(1, T + 1):
        commit_prefix(gdn, stage, n, T, fused_only=_fo)
        o = gdn.forward_decode(tt_probe)
        got[n] = ttnn.to_torch(o, mesh_composer=comp)[0, 0].float().reshape(-1)
        ttnn.deallocate(o)

    matched = [compute_pcc(ref[n], got[n]) for n in range(1, T + 1)]
    # Control: rollback to n judged against the WRONG acceptance count. Without it a high PCC
    # proves nothing, since consecutive prefixes give similar continuations.
    mism = [compute_pcc(ref[n], got[m]) for n in range(1, T + 1) for m in range(1, T + 1) if n != m]
    for i, n in enumerate(range(1, T + 1)):
        print(f"MD rollback n={n} pcc={matched[i]:.6f} rel={_rel_l2(got[n], ref[n]):.3e}", flush=True)
    print(f"MD ROLLBACK VERDICT T={T} min_matched={min(matched):.6f} "
          f"max_mismatched={(max(mism) if mism else float('nan')):.6f}", flush=True)

    assert min(matched) > PCC_OK, f"commit_prefix does not reproduce n sequential decodes ({min(matched):.6f})"
    if mism:
        assert max(mism) < min(matched), (
            f"rollback control did not separate: wrong-n {max(mism):.4f} >= right-n {min(matched):.4f}"
        )


@torch.no_grad()
@parametrize_mesh_tp()
def test_l2norm_scale_matches_reference(mesh_device, reset_seeds, ensure_gc, request):
    """`_l2norm_scale` must be `l2_norm_ttnn(x, dim=-1)` — the caller now owns this step.

    Writing `epsilon=1e-6` instead of `epsilon=1e-6/Dk` is invisible on random activations (eps
    is negligible next to the norm) and wrong by a factor of 128 on a near-zero row, so it has to
    be checked against the function it replaced AND against torch.
    """
    Dk, T, H = 128, 4, 24
    x = torch.randn(1, T, H, Dk)
    # Row 0 is driven to ~1e-4, so sum(x^2) ~ 1e-6 is comparable to eps and the two epsilon
    # conventions differ by the full factor of Dk. On plain random activations they do not, which
    # is exactly why this trap survives an ordinary test.
    x[0, 0, 0] *= 1e-4
    cmp_ = tp_composer(mesh_device)
    xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG,
                         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

    mine = ttnn.to_torch(_l2norm_scale(xt), mesh_composer=cmp_)[..., :Dk].float()
    theirs = ttnn.to_torch(l2_norm_ttnn(xt, dim=-1), mesh_composer=cmp_)[..., :Dk].float()
    want = x / (x.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    # The wrong convention: eps on the MEAN of squares instead of on the sum.
    bad = ttnn.to_torch(
        ttnn.multiply(ttnn.rms_norm(xt, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                      Dk**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG), mesh_composer=cmp_
    )[..., :Dk].float()
    r_ref, r_torch = _rel_l2(mine, theirs), _rel_l2(mine, want)
    r_bad = _rel_l2(bad[0, 0, 0], want[0, 0, 0])
    r_mine_row0 = _rel_l2(mine[0, 0, 0], want[0, 0, 0])
    print(f"MD l2norm vs_l2_norm_ttnn rel={r_ref:.3e} pcc={compute_pcc(mine, theirs):.8f} | "
          f"vs_torch rel={r_torch:.3e} pcc={compute_pcc(mine, want):.8f} | "
          f"tiny_row: mine_rel={r_mine_row0:.3e} wrong_eps_rel={r_bad:.3e}", flush=True)
    # Against `l2_norm_ttnn` the bar is BIT-EXACT: the fused path must inherit the reference's
    # numerics here, not merely approximate them. Measured 0.0.
    assert r_ref == 0.0, f"does not reproduce l2_norm_ttnn bit-for-bit ({r_ref:.3e})"
    # Against torch the bar is much looser, and the gap is NOT this function's doing: since it
    # matches `l2_norm_ttnn` exactly, the residual ~4e-3 is `ttnn.rms_norm`'s own accuracy on
    # fp32 input, which both the fused and the composed path pay identically. Worth knowing,
    # because it is probably the dominant term in the fused path's state error -- larger than
    # anything the kernel contributes.
    assert r_torch < 1e-2, f"does not reproduce a true L2 norm ({r_torch:.3e})"
    assert r_bad > 100 * max(r_mine_row0, 1e-12), (
        f"the eps control does not separate ({r_bad:.3e} vs {r_mine_row0:.3e}); this test cannot "
        "see an eps-on-the-mean mistake"
    )
