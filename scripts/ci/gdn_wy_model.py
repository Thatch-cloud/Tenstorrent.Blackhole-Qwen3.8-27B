"""gdn_wy_model: CPU models of the gated delta rule's verify forms, for the NON-EXACT window WY probe (docs/gdn-wy-probe.md).

RESEARCH ONLY. The window form computes a different arithmetic from the served per-token recurrence (gdn_seq_block, K5-A), so its
bytes are not the served bytes: it can never run in a serving or traffic profile under the byte-identity contract. Nothing here
imports ttnn and nothing here is imported by the serving path.

Two forms of one layer's recurrence over a block of rows (per user, per value head; S is [dk, dv]):

  sequential   the served chain: per token, decay, read, delta, rank-one update, read-out; the state rounded to bf16 after every
               token (per-token snapshots, the commit copies snapshot n).
  window WY    one aligned window of W rows (default 16) at a time, the chunked WY / UT block form of flash-linear-attention:
               A = (I + tril_strict(beta K K^T o Gamma))^-1, u = A beta V, w = A (beta gamma K), V~ = u - w S0,
               O = (gamma Q) S0 + tril(Q K^T o Gamma) V~, and the state after n rows S_n = gamma_n S0 + sum_i exp(G_n - G_i) k_i v~_i^T,
               with the state rounded to bf16 once per window boundary (at commit), not per token. A block of 32 rows is TWO chained
               windows: window two starts from window one's committed bf16 state. Decays are always in log space (exp(G_t - G_i)); the
               literal g_t / g_i ratio of the TreeWY paper under-flows to 0/0 under strong decay and is kept only as `ratio=True`.

Rounding models (labels used in the reports): fp64 (ground truth); fp32; bf16 (bf16 inputs, fp32 arithmetic, bf16 state: the served
class); bf16+tf32 (every multiply's operands also rounded to TF32, the Tensix srcA/srcB width, ESTIMATED); bf16mid (WY only: every WY
intermediate stored as bf16, as bf16 circular buffers would hold it). These are CPU models of the device class, not bit-faithful
Tensix emulation, and the TT kernel does not exist yet (docs/gdn-wy-probe.md, "What remains").
"""

import math

import torch

NK, NV, DK, DV, T, USERS, GDN_LAYERS = 4, 12, 128, 128, 16, 8, 48   # the four-card shard: 4 key heads, 12 value heads per card
TAU = 4.485    # committed tokens per user per round, pooled (docs/tp4-lookup.md)
F64, F32 = torch.float64, torch.float32
TILE = 32


# ---------------------------------------------------------------- rounding helpers

def bf16(x):
    return x.to(torch.bfloat16).to(x.dtype)


def tf32(x):
    """Round-to-nearest-even of a float32 tensor to 10 mantissa bits (TF32, the Tensix srcA / srcB width)."""
    if x.dtype != F32:
        return x
    i = x.contiguous().view(torch.int32).to(torch.int64)
    i = (i + 0xFFF + ((i >> 13) & 1)) & ~0x1FFF
    return i.to(torch.int32).view(torch.float32)


class Model:
    def __init__(self, name, dtype, state_bf16=False, tf32_ops=False, mid_bf16=False):
        self.name, self.dtype = name, dtype
        self.state_bf16, self.tf32_ops, self.mid_bf16 = state_bf16, tf32_ops, mid_bf16

    def st(self, x):
        return bf16(x) if self.state_bf16 else x

    def op(self, x):
        return tf32(x) if self.tf32_ops else x

    def mid(self, x):
        return bf16(x) if self.mid_bf16 else x


MODELS = {
    'fp64': Model('fp64', F64),
    'fp32': Model('fp32', F32),
    'bf16': Model('bf16', F32, state_bf16=True),
    'bf16+tf32': Model('bf16+tf32', F32, state_bf16=True, tf32_ops=True),
    'bf16mid': Model('bf16mid', F32, state_bf16=True, mid_bf16=True),
}


# ---------------------------------------------------------------- inputs

def gate_constants(seed, nv=NV):
    g = torch.Generator().manual_seed(seed)
    # per-head decay scale, log-uniform over [1e-3, 4]: alpha from ~0.999 (slow memory) to ~0.02 (fast)
    return torch.exp(torch.empty(nv).uniform_(math.log(1e-3), math.log(4.0), generator=g))


def draw_rows(gen, rows, regime, consts, users=USERS, nk=NK, nv=NV):
    """Raw layer inputs for `rows` tokens, bf16-valued float32: q,k (B,T,nk,dk), v,z (B,T,nv,dv), g,beta (B,T,nv).

    regime 'model': silu-shaped q/k/v, sigmoid beta, g = -consts * softplus(.) / ln 2 (the Qwen3-Next parameterisation
    -exp(A_log) softplus(a + dt_bias), with a per-head scale spread over three decades); 'R1' and 'R2' are the regimes of
    gdn_seq_block_device_test (R1: unit normal, beta U(0,1), g -U(0,1); R2: wide range, g -U(0,20))."""
    def randn(*s):
        return torch.randn(*s, generator=gen)

    def rand(*s):
        return torch.rand(*s, generator=gen)

    if regime == 'model':
        silu = torch.nn.functional.silu
        q, k, v = silu(randn(users, rows, nk, DK)), silu(randn(users, rows, nk, DK)), silu(randn(users, rows, nv, DV))
        beta = torch.sigmoid(randn(users, rows, nv))
        g = -consts * torch.nn.functional.softplus(randn(users, rows, nv)) / math.log(2.0)
    elif regime == 'R1':
        q, k, v = randn(users, rows, nk, DK), randn(users, rows, nk, DK), randn(users, rows, nv, DV)
        beta, g = rand(users, rows, nv), -rand(users, rows, nv)
    elif regime == 'R2':
        q, k, v = 8 * randn(users, rows, nk, DK), 8 * randn(users, rows, nk, DK), 8 * randn(users, rows, nv, DV)
        beta, g = rand(users, rows, nv), -20 * rand(users, rows, nv)
    else:
        raise ValueError(regime)
    z = randn(users, rows, nv, DV)
    raw = dict(q=q, k=k, v=v, z=z, beta=beta, g=g)
    return {key: bf16(val.float()) for key, val in raw.items()}


def cat_rows(parts):
    return {key: torch.cat([p[key] for p in parts], dim=1) for key in parts[0]}


def take_rows(raw, start, stop):
    return {key: val[:, start:stop] for key, val in raw.items()}


def prep(raw, norm_w, dtype):
    """Prologue (row-independent): GQA expand, q/k L2 norms, layout (B, H, T, D)."""
    nk, nv = raw['q'].shape[2], raw['v'].shape[2]
    q = raw['q'].to(dtype).repeat_interleave(nv // nk, dim=2)
    k = raw['k'].to(dtype).repeat_interleave(nv // nk, dim=2)
    qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    tr = lambda x: x.transpose(1, 2).contiguous()
    return dict(qn=tr(qn), kn=tr(kn), v=tr(raw['v'].to(dtype)), z=tr(raw['z'].to(dtype)),
                g=raw['g'].to(dtype).transpose(1, 2).contiguous(), beta=raw['beta'].to(dtype).transpose(1, 2).contiguous(),
                w=norm_w.to(dtype))


def rows_of(p, a, b=None):
    """Rows [a, b) of a prepped block (rows [0, a) when b is None)."""
    if b is None:
        a, b = 0, a
    return {key: (val if key == 'w' else val[:, :, a:b].contiguous()) for key, val in p.items()}


# ---------------------------------------------------------------- the served form: sequential, per-token snapshots

def seq_round(S, p, n, m, pre_out=None):
    """The repo reference's per-token order (decay, read, delta, rank-one update, read-out) over rows 0..n-1.
    Returns outputs (B,H,n,DV), the n per-token state snapshots, and the final state."""
    alpha = torch.exp(p['g'])
    outs, snaps = [], []
    for t in range(n):
        S = m.op(S) * m.op(alpha[:, :, t, None, None])
        kt, qt = p['kn'][:, :, t], p['qn'][:, :, t]
        read = torch.einsum('bhk,bhkv->bhv', m.op(kt), m.op(S))
        delta = m.op(p['v'][:, :, t] - read) * m.op(p['beta'][:, :, t, None])
        S = S + m.op(kt)[..., :, None] * m.op(delta)[..., None, :]
        if pre_out is not None:   # sensitivity: read-out from the unrounded state (before the bf16 pack)
            pre_out.append(torch.einsum('bhk,bhkv->bhv', m.op(qt), m.op(S)))
        S = m.st(S)
        outs.append(torch.einsum('bhk,bhkv->bhv', m.op(qt), m.op(S)))
        snaps.append(S)
    return torch.stack(outs, 2), snaps, S


# ---------------------------------------------------------------- the candidate: one aligned window, WY / UT block form

def wy_window(S0, p, m, ratio=False):
    """One window = every row of `p` (HF torch_chunk_gated_delta_rule op order, chunk size = rows of p).
    ratio=True uses TreeWY eq. (2)'s literal g_t / g_i of cumulative decays instead of exp(G_t - G_i)."""
    Tn = p['g'].shape[-1]
    G = torch.cumsum(p['g'], -1)
    lower = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))
    strict = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool), -1)
    gam = torch.exp(G)
    if ratio:
        Gam = (gam[..., :, None] / gam[..., None, :]).masked_fill(~lower, 0.0)
    else:
        Gam = torch.exp((G[..., :, None] - G[..., None, :]).masked_fill(~lower, float('-inf')))
    kn, qn = p['kn'], p['qn']
    kb, vb = kn * p['beta'][..., None], p['v'] * p['beta'][..., None]
    attn = (-(m.op(kb) @ m.op(kn).transpose(-1, -2)) * Gam).masked_fill(~strict, 0.0)
    for i in range(1, Tn):   # forward substitution: attn <- (I + diag(beta) (Gam o K K^T)_strict)^-1 - I
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :i, None].clone() * attn[..., :i, :i]).sum(-2)
    attn = m.mid(attn + torch.eye(Tn, dtype=attn.dtype))
    u = m.mid(m.op(attn) @ m.op(vb))
    w = m.mid(m.op(attn) @ m.op(kb * gam[..., None]))
    vnew = m.mid(u - m.op(w) @ m.op(S0))                      # pseudo-values V~ (T, dv)
    qk = (m.op(qn) @ m.op(kn).transpose(-1, -2)) * Gam        # lower-inclusive by Gam's zeros
    o = m.op(qn * gam[..., None]) @ m.op(S0) + m.op(qk) @ m.op(vnew)
    return o, dict(G=G, gam=gam, vnew=vnew, kn=kn)


def wy_commit(S0, ctx, n, m):
    """TreeWY eq. (3): S_n = gamma_n S0 + sum_{i<=n} exp(G_n - G_i) k_i v~_i^T (log-space decays), rounded to the state format once."""
    G = ctx['G']
    Gn = G[..., n - 1:n]
    kd = ctx['kn'][..., :n, :] * torch.exp(Gn - G[..., :n])[..., None]
    S = m.op(S0) * m.op(torch.exp(Gn)[..., None]) + m.op(kd).transpose(-1, -2) @ m.op(ctx['vnew'][..., :n, :])
    return m.st(S)


def wy_verify(S0, p, n, m, window=T):
    """Verify rows 0..n-1 of `p` as aligned windows of `window` rows, chained through the committed (bf16) boundary state.
    Every window computes all of its rows (the kernel's fixed shape); only the n committed rows are returned.
    Returns (outputs (B,H,n,DV), state after n rows, windows used)."""
    total = p['g'].shape[-1]
    S, pos, outs, windows = S0, 0, [], 0
    while pos < n:
        stop = min(pos + window, total)
        o, ctx = wy_window(S, rows_of(p, pos, stop), m)
        take = min(stop - pos, n - pos)
        outs.append(o[:, :, :take])
        S = wy_commit(S, ctx, take, m)
        pos += take
        windows += 1
    return torch.cat(outs, 2), S, windows


# ---------------------------------------------------------------- layer output (norm, gate) and compares

def gated(o, p, n, round_bf16=True):
    xn = o * torch.rsqrt((o * o).sum(-1, keepdim=True) + 128e-6) * DV ** 0.5
    y = xn * p['w'] * torch.nn.functional.silu(p['z'][:, :, :n])
    return bf16(y.float()) if round_bf16 else y


def cmp(a, b):
    """Error of `a` against reference `b`, machine readable. rel_to_max: max abs error over the reference's largest magnitude;
    max_elem_rel: worst per-element relative error over elements above 1e-3 of that magnitude."""
    a64, b64 = a.double(), b.double()
    d = (a64 - b64).abs()
    finite = bool(torch.isfinite(a64).all())
    scale = float(b64.abs().max())
    big = b64.abs() > 1e-3 * scale
    return dict(max_abs=float(d.max()), rel_to_max=float(d.max()) / scale if scale else float('nan'),
                rel_l2=float(d.norm() / b64.norm()) if float(b64.norm()) else float('nan'),
                max_elem_rel=float((d[big] / b64.abs()[big]).max()) if bool(big.any()) else 0.0,
                bitwise_equal_frac=float((a64 == b64).double().mean()), elements=a.numel(),
                differing=int((a64 != b64).sum()), finite=finite)


def short(c):
    return {k: (float('%.4g' % v) if isinstance(v, float) else v) for k, v in c.items()}


def json_safe(value):
    """NaN and Infinity are not JSON: every non-finite float becomes null, so a strict consumer (jq, JSON.parse) can read a report."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def state_gap(a, b):
    """(max abs difference over the reference's largest magnitude, finite) of two states: the cheap form of cmp() for every cycle."""
    a64, b64 = a.double(), b.double()
    scale = float(b64.abs().max())
    gap = float((a64 - b64).abs().max())
    return (gap / scale if scale else float('nan')), bool(torch.isfinite(a64).all())


def bits_equal(a, b):
    """Bit-for-bit equality of two tensors of the same dtype (NaN payloads and -0 included)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {torch.float32: torch.int32, torch.float64: torch.int64, torch.bfloat16: torch.int16}[a.dtype]
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


# ---------------------------------------------------------------- acceptance distribution

def geometric_p(tau=TAU, rows=T):
    """The per-proposal acceptance probability p with E[commit] = 1 + sum_{j=1}^{rows-1} p^j = tau (commit is capped at rows)."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 1 + sum(mid ** j for j in range(1, rows)) < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def draw_commits(gen, cycles, p_accept, rows=T, hist=None):
    """Committed tokens per round, `cycles` of them. Default: the seed plus a geometric run of accepted proposals, capped at rows
    (the fit to the pooled tau when p_accept = geometric_p()). With `hist` ({n: count}, e.g. the lab's per-round histogram) the
    lengths are drawn from that histogram instead."""
    if hist:
        sizes = sorted(int(k) for k in hist)
        weights = torch.tensor([float(hist[k] if k in hist else hist[str(k)]) for k in sizes], dtype=torch.float64)
        idx = torch.multinomial(weights / weights.sum(), cycles, replacement=True, generator=gen)
        return [sizes[i] for i in idx.tolist()]
    u = torch.rand(cycles, rows - 1, generator=gen)
    acc = (u < p_accept).int().cumprod(1).sum(1)
    return (1 + acc).tolist()


# ---------------------------------------------------------------- cost accounting

def counts(n_commit, rows=T, dk=DK, dv=DV):
    """FLOPs and bytes per (user, value head) per verify round at `rows` rows (one window), bf16 storage."""
    t, n = rows, n_commit
    seq_flops = t * (dk * dv + 2 * dk * dv + 2 * dv + 2 * dk * dv + 2 * dk * dv)   # decay, read, delta, update, read-out
    tri = sum(2 * i * i for i in range(1, t))                                        # forward substitution on T x T
    wy_verify_f = (t * dk + 2 * t * t * dk          # beta*k, (k beta) K^T
                   + 3 * t * t + t                  # Gamma (diff, exp, mul), cumsum
                   + tri
                   + t * dv + 2 * t * t * dv        # beta*v, u = A^-1 (beta v)
                   + 2 * t * dk + 2 * t * t * dk    # (beta gamma) k, w = A^-1 (beta gamma k)
                   + 2 * t * dk * dv + t * dv       # w S0, v_new = u - w S0
                   + 2 * t * t * dk + t * t         # (q K^T) o Gamma
                   + t * dk + 2 * t * dk * dv       # (gamma q) S0
                   + 2 * t * t * dv + t * dv)       # (qk) V~, add
    wy_commit_f = 2 * n * dk + 2 * n * dk * dv + 2 * dk * dv     # k scaling, K^T V~, gamma_n S0 + add
    st_b = dk * dv * 2
    inputs = 2 * t * dk * 2 + 2 * t * dv * 2 + 2 * t * 2          # q,k (per core copy), v,z, g,beta (bf16)
    out_b = t * dv * 2
    seq_bytes = dict(read_state=st_b, inputs=inputs, snapshots=t * st_b, outputs=out_b, commit_copy=2 * st_b)
    wy_bytes = dict(read_state=st_b, inputs=inputs, outputs=out_b, vtilde_stash_fp32=t * dv * 4,
                    commit_reread_state=st_b, commit_read_k_vtilde=t * dk * 2 + t * dv * 4, commit_write_state=st_b)
    return dict(n_commit=n, rows=t, seq=dict(flops=seq_flops, bytes=seq_bytes, bytes_total=sum(seq_bytes.values())),
                wy=dict(flops_verify=wy_verify_f, flops_commit=wy_commit_f, flops_total=wy_verify_f + wy_commit_f,
                        bytes=wy_bytes, bytes_total=sum(wy_bytes.values()),
                        bytes_total_commit_fused=sum(wy_bytes.values()) - st_b - (t * dk * 2 + t * dv * 4) - t * dv * 4))


def tile_ops_wy(rows=T, dk=DK, dv=DV):
    """ESTIMATED 32x32-tile operation count for one (user, head) WY window (rows pad to one tile row)."""
    mm = lambda m, n, k: math.ceil(m / TILE) * math.ceil(n / TILE) * math.ceil(k / TILE)
    parts = dict(kkT=mm(rows, rows, dk), qkT=mm(rows, rows, dk), inverse_by_doubling=3 * mm(rows, rows, rows) + 4 * mm(rows, rows, rows) + 4,
                 u=mm(rows, dv, rows), w=mm(rows, dk, rows), wS0=mm(rows, dv, dk), qS0=mm(rows, dv, dk), qkV=mm(rows, dv, rows),
                 elementwise=6 + 6 * math.ceil(dv / TILE), commit_KtV=mm(dk, dv, rows),
                 commit_scale_add=2 * (dk // TILE) * (dv // TILE), gate_exp_cumsum=3)
    return dict(parts=parts, total=sum(parts.values()))


# The planned per-core circular-buffer table of ONE (user, head) window on one core, in bytes (a 32x32 tile is 2,048 B in bf16,
# 4,096 B in fp32). The window's T=16 rows occupy the top half of one tile row (rows 16-31 are padding the kernel must keep finite and,
# for the causality gate, zero). This is a PLAN for the probe's SRAM check, not a measurement: the kernel is not built.
def sram_plan(dk=DK, dv=DV, windows=2):
    bf, f32 = 2 * TILE * TILE, 4 * TILE * TILE
    kt, vt = dk // TILE, dv // TILE          # tiles across the key and value width
    cbs = dict(
        state_in=kt * vt * bf,               # S0 (bf16), 16 tiles
        state_out=kt * vt * bf,              # the window-boundary commit (bf16), 16 tiles
        k=kt * bf, q=kt * bf,                # one tile row of k and of q (rows padded to 32)
        v=vt * bf, z=vt * bf,                # one tile row of v and of the gate input z
        beta_g=2 * bf,                       # beta and g columns
        gram_kk=1 * f32, gram_qk=1 * f32,    # the 32x32 (T x T) Gram tiles, fp32 intermediates
        decay_gamma=1 * f32,                 # Gamma = exp(G_t - G_i), lower-inclusive
        attn_inverse=2 * f32,                # A^-1 (the doubling scratch ping-pongs two tiles)
        u=vt * f32, w=kt * f32,              # UT products
        vtilde=vt * f32,                     # V~ (fp32 stash, also read by the commit)
        out=vt * bf,                         # O rows
        scratch=4 * f32)
    total = sum(cbs.values())
    return dict(per_cb_bytes=cbs, per_window_total_bytes=total, windows_chained=windows,
                chained_total_bytes=total + (kt * vt * bf if windows > 1 else 0),   # the second window reads the first's boundary state
                note='plan, not a measurement; K5-A holds 630,784 B of circular buffers a core (docs/tp4-recurrence-split.md); the card harness reports the kernel own figure beside this plan')


def dram_bytes_per_core(windows=2, dk=DK, dv=DV, rows=T):
    """Plan: DRAM bytes one core moves for `windows` chained windows of one (user, head): state in once, state out once (the boundary state
    stays in L1 between windows), inputs and outputs per window, and the fp32 V~ stash per window only if the commit is not fused."""
    st = dk * dv * 2
    per_window_io = 2 * rows * dk * 2 + 2 * rows * dv * 2 + 2 * rows * 2 + rows * dv * 2
    return dict(state_in=st, state_out=st, window_io=per_window_io * windows, total=2 * st + per_window_io * windows,
                sequential_two_windows=2 * (st + per_window_io + rows * st + 2 * st))
