#!/usr/bin/env python3
"""gdn_wy_numerics: E1 of the NON-EXACT window WY probe (docs/gdn-wy-probe.md), on CPU, for a CI job.

RESEARCH ONLY. It measures how far the aligned-window WY / UT block form of the gated delta rule (gdn_wy_model.wy_verify) sits from the
served sequential per-token recurrence and from fp64, at the TP4 per-card shard (4 key heads, 12 value heads, dk = dv = 128, 8 users),
and whether the window form keeps the properties the byte-identity contract needs. It cannot qualify anything for serving: the window
form is a different arithmetic, so its bytes are not the served bytes (this report says so in `contract`, whatever the numbers are).

Run it in CI (the qwen-gdn-wy-numerics workflow, tag experiment/gdn-wy-numerics-v*) and never trust a laptop's numbers: a laptop may
flip bits at rest. The local run is a smoke test (`--smoke`).

  python gdn_wy_numerics.py --out report.json                       # the full E1 for the default case
  python gdn_wy_numerics.py --regimes R1 --rows 32 --commit-dist stress --out report.json
  python gdn_wy_numerics.py --smoke --out smoke.json                # seconds, tiny scale, never a result

Sections (--sections): crosscheck (this module's fp64 sequential form against the repo's reference in gdn_seq_block_device_test, when
that file is importable as text), single (one round, every rounding model: error per element against fp64 and against the served form,
by row), causality (rows 0..t and the committed state bit for bit when the later rows are real, zero or random; and the NaN-poison
leak), packed_solo (a batch of users against each user alone, bit for bit), drift (>= 1,000 accept / reject cycles, committed lengths
from the tau distribution, every form carries its own state), segmentation (one committed stream split into rounds two ways),
counts (FLOPs, bytes, tile operations, SRAM plan).

Inputs. Real layer 0 / 23 / 47 activations are not in the CI image (the weights are private and large): `--fixture DIR` reads
layer<N>.npz files (keys q,k,v,z,beta,g as float32 bf16-valued arrays shaped (users, rows, heads, dim) / (users, rows, heads), at least
--rows * 2 rows long, optional S0 (users, nv, dk, dv) and norm_w (dv)), as a dev card or the card harness's real-text path dumps them;
regime `fixture:layer0` then draws from the file. Without fixtures the regimes are synthetic: `model` (the Qwen3-Next parameterisation
shapes), `R1`, `R2` (gdn_seq_block_device_test's own). Committed lengths come from the geometric fit to the pooled tau 4.485 (the lab's
per-round histogram is private; `--commit-hist FILE` takes it as {"n": count}) or, with --commit-dist stress, from p = 0.97 so that
the second window of a 32-row block is exercised.

The report is one JSON object (kind gdn-wy-numerics); the last stdout line is a one-line summary, then the same object's `acceptance`.
Exit code 0 when the E1 criterion holds, 1 when it does not, 2 on a usage error. The criterion (E1): the window form's error against
fp64 is no worse than the served chain's (ratio <= 1.00, strict: a tie a hair over fails and `e1_worst_ratio` says by how much), in the
bf16 class, for the state and the output, over every drift case, and nothing is non-finite. It is a research criterion: `contract.byte_identical_to_served` is false whenever any byte differs, and that is what
rules the form out of serving.
"""

import argparse
import itertools
import json
import os
import platform
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gdn_wy_model as M  # noqa: E402
from gdn_wy_model import (DK, DV, F32, F64, MODELS, NK, NV, T, TAU, USERS, bf16, bits_equal, cat_rows, cmp, draw_commits,  # noqa: E402
                          draw_rows, gate_constants, gated, geometric_p, prep, rows_of, seq_round, short, take_rows, wy_commit,
                          wy_verify, wy_window)

KIND = 'gdn-wy-numerics'
SCHEMA = 1
SECTIONS = ('crosscheck', 'single', 'causality', 'packed_solo', 'drift', 'segmentation', 'counts')
CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
E1_RATIO_MAX = 1.0       # window error / served-chain error against fp64 must not exceed this
STRESS_P = 0.97
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- input sources

class Source:
    """Rows of one layer's raw inputs: synthetic (a regime) or a fixture file (cursor over its rows, wrapping)."""

    def __init__(self, regime, seed, users, nk, nv, fixture_dir=None):
        self.regime, self.users, self.nk, self.nv = regime, users, nk, nv
        self.consts = gate_constants(1000 + seed, nv)
        self.fixture = None
        if regime.startswith('fixture:'):
            import numpy as np
            path = os.path.join(fixture_dir or '.', regime.split(':', 1)[1] + '.npz')
            data = np.load(path)
            self.fixture = {key: torch.from_numpy(data[key].astype('float32')) for key in ('q', 'k', 'v', 'z', 'beta', 'g')}
            self.s0 = torch.from_numpy(data['S0'].astype('float32')) if 'S0' in data.files else None
            self.norm_w = torch.from_numpy(data['norm_w'].astype('float32')) if 'norm_w' in data.files else None
            self.length = self.fixture['g'].shape[1]
            self.cursor = {}
            if self.fixture['g'].shape[0] != users or self.fixture['v'].shape[2] != nv or self.fixture['q'].shape[2] != nk:
                raise ValueError('fixture %s: users/heads %s do not match the run (%d, %d, %d)' % (
                    path, tuple(self.fixture['v'].shape), users, nk, nv))

    def draw(self, gen, rows, lane=0):
        if self.fixture is None:
            return draw_rows(gen, rows, self.regime, self.consts, self.users, self.nk, self.nv)
        # two lanes: lane 0 the stream, lane 1 the rejected-draft rows (drawn from the second half of the file)
        start = self.cursor.get(lane, (self.length // 2) * lane)
        idx = [(start + i) % self.length for i in range(rows)]
        self.cursor[lane] = (start + rows) % self.length
        return {key: bf16(val[:, idx].float()) for key, val in self.fixture.items()}

    def initial_state(self, seed, users, norm_w_default):
        """(S0 bf16-valued fp32 (users, nv, dk, dv), norm_w): a fixture's own, else warmed up on 256 rows of the source in fp64."""
        if self.fixture is not None and self.s0 is not None:
            return bf16(self.s0.float()), (self.norm_w if self.norm_w is not None else norm_w_default)
        gen = torch.Generator().manual_seed(seed)
        S = torch.zeros(users, self.nv, DK, DV, dtype=F64)
        raw = self.draw(gen, 256)
        _, _, S = seq_round(S, prep(raw, torch.ones(DV), F64), 256, MODELS['fp64'])
        return bf16(S.float()), norm_w_default


def make_source(regime, seed, args):
    return Source(regime, seed, args.users, args.nk, args.nv, args.fixture)


def commit_draw(args, seed, cycles, rows):
    gen = torch.Generator().manual_seed(seed)
    if args.commit_hist:
        with open(args.commit_hist, encoding='utf-8') as handle:
            hist = json.load(handle)
        return draw_commits(gen, cycles, 0.0, rows, hist=hist), dict(source='histogram', file=os.path.basename(args.commit_hist))
    if args.commit_dist == 'stress':
        return draw_commits(gen, cycles, STRESS_P, rows), dict(source='geometric-stress', p_accept=STRESS_P)
    p = geometric_p(TAU, T)
    return draw_commits(gen, cycles, p, rows), dict(source='geometric-fit-to-pooled-tau', tau=TAU, p_accept=p)


# ---------------------------------------------------------------- (a) cross-check against the repo's reference

def crosscheck_repo_reference(seed=17):
    """At the TP2 shape the repo reference hard-codes (8 key heads, 24 value heads), regime R1 inputs as regime_inputs draws them:
    the repo's fp64 function (read from gdn_seq_block_device_test.py as text, so no ttnn import) against this module's seq_round + gated."""
    path = os.path.join(HERE, 'gdn_seq_block_device_test.py')
    with open(path, encoding='utf-8') as handle:
        src = handle.read()
    start = src.index('def reference(torch, qkv, beta, gate, initial, z, norm_w):')
    end = src.index('\ndef reference_error', start)
    ns = {'ROWS': T}
    exec(src[start:end], ns)
    gen = torch.Generator().manual_seed(seed)
    randn = lambda *s: torch.randn(*s, generator=gen)
    rand = lambda *s: torch.rand(*s, generator=gen)
    norm_w = (1 + randn(1, 1, 128) * 0.1).bfloat16()
    initial = randn(1, 24, 128, 128) * 0.05
    qkv, beta, gate, z = randn(1, T, 5120), rand(1, T, 24), -rand(1, T, 24), randn(1, T, 3072)
    qkv, beta, gate, initial, z = (x.bfloat16() for x in (qkv, beta, gate, initial, z))
    out_ref, states_ref, _ = ns['reference'](torch, qkv, beta, gate, initial, z, norm_w)
    raw = dict(q=qkv[:, :, 0:1024].float().reshape(1, T, 8, 128), k=qkv[:, :, 1024:2048].float().reshape(1, T, 8, 128),
               v=qkv[:, :, 2048:5120].float().reshape(1, T, 24, 128), z=z.float().reshape(1, T, 24, 128),
               beta=beta.float(), g=gate.float())
    p = prep(raw, norm_w[0, 0].float(), F64)
    o, snaps, _ = seq_round(initial.double().clone(), p, T, MODELS['fp64'])
    y = gated(o, p, T, round_bf16=False)
    return dict(file='scripts/ci/gdn_seq_block_device_test.py', function='reference',
                output=short(cmp(y.transpose(1, 2).reshape(1, T, 3072), out_ref)),
                states=short(cmp(torch.stack([s[0] for s in snaps]), states_ref)))


# ---------------------------------------------------------------- experiments

def single_round(regime, args, seed=1):
    """One verify round at `rows` rows: every rounding model, the window form (chained windows) against the served form and fp64."""
    rows = args.rows
    src = make_source(regime, 1000 + seed, args)
    gen = torch.Generator().manual_seed(seed)
    norm_w = bf16(1 + 0.1 * torch.randn(DV, generator=gen))
    if regime == 'R2':
        S0 = bf16(4 * torch.randn(args.users, args.nv, DK, DV, generator=gen))
    else:
        S0, norm_w = src.initial_state(seed + 7, args.users, norm_w)
    raw = src.draw(gen, rows)
    res = dict(regime=regime, rows=rows)
    p64 = prep(raw, norm_w, F64)
    o64, snaps64, _ = seq_round(S0.double(), p64, rows, MODELS['fp64'])
    y64 = gated(o64, p64, rows)
    wy_o64, wy_s64, _ = wy_verify(S0.double(), p64, rows, MODELS['fp64'])
    res['fp64_wy_vs_seq'] = dict(output=short(cmp(wy_o64, o64)), state_after_n=short(cmp(wy_s64, snaps64[-1])))
    for name in ('fp32', 'bf16', 'bf16+tf32', 'bf16mid'):
        m = MODELS[name]
        p = prep(raw, norm_w, m.dtype)
        seq_m = MODELS['bf16'] if name == 'bf16mid' else m      # bf16mid is a WY-only variant
        pre = []
        o_s, snaps, _ = seq_round(S0.to(m.dtype), p, rows, seq_m, pre_out=pre)
        o_pre = torch.stack(pre, 2)
        o_w, s_w, windows = wy_verify(S0.to(m.dtype), p, rows, m)
        # the state after each committed n (a = 1..rows), not only after all rows: one commit per a from the first window's context where a <= 16
        states_w = torch.stack([wy_verify(S0.to(m.dtype), p, a, m)[1] for a in range(1, rows + 1)])
        states_s = torch.stack(snaps)
        y_s, y_w = gated(o_s, p, rows), gated(o_w, p, rows)
        per_row = [cmp(o_w[:, :, r], o_s[:, :, r])['rel_to_max'] for r in range(rows)]
        res[name] = dict(
            windows=windows,
            wy_vs_seq=dict(core_output=short(cmp(o_w, o_s)), gated_bf16_output=short(cmp(y_w, y_s)),
                           states_a1_to_n=short(cmp(states_w, states_s)), state_a_last=short(cmp(states_w[-1], states_s[-1])),
                           core_output_rel_by_row=[float('%.3g' % x) for x in per_row],
                           core_output_vs_seq_readout_before_pack=short(cmp(o_w, o_pre)),
                           gated_bf16_vs_seq_readout_before_pack=short(cmp(y_w, gated(o_pre, p, rows)))),
            seq_vs_fp64=dict(core_output=short(cmp(o_s, o64)), gated_bf16_output=short(cmp(y_s, y64)),
                             states=short(cmp(states_s, torch.stack(snaps64)))),
            wy_vs_fp64=dict(core_output=short(cmp(o_w, o64)), gated_bf16_output=short(cmp(y_w, y64)),
                            states=short(cmp(states_w, torch.stack(snaps64)))))
        # T = 1: the non-speculative decode step itself
        p1 = rows_of(p, 1)
        o1s, _, S1s = seq_round(S0.to(m.dtype), p1, 1, seq_m)
        o1w, S1w, _ = wy_verify(S0.to(m.dtype), p1, 1, m, window=1)
        res[name]['t1_wy_vs_seq'] = dict(core_output=short(cmp(o1w, o1s)), state=short(cmp(S1w, S1s)))
        # TreeWY eq. (2) literal ratio form: non-finite values under strong decay
        o_r, ctx_r = wy_window(S0.to(m.dtype), rows_of(p, min(rows, T)), m, ratio=True)
        s_r = wy_commit(S0.to(m.dtype), ctx_r, min(rows, T), m)
        res[name]['ratio_form'] = dict(nonfinite_outputs=int((~torch.isfinite(o_r)).sum()), outputs=o_r.numel(),
                                       nonfinite_state_last=int((~torch.isfinite(s_r)).sum()),
                                       min_cum_log_decay=float(ctx_r['G'].min()))
    return res


def median_merge(runs):
    """Leaf-wise median over repeated runs (numbers), elementwise for lists; other leaves from the first run."""
    first = runs[0]
    if isinstance(first, dict):
        return {k: median_merge([r[k] for r in runs]) for k in first}
    if isinstance(first, list):
        return [median_merge([r[i] for r in runs]) for i in range(len(first))]
    if isinstance(first, bool) or not isinstance(first, (int, float)):
        return first
    return sorted(runs)[len(runs) // 2]


def causality(regime, args, seed=5):
    """Rows 0..t and the state after t + 1 rows, bit for bit, when the later rows of the block are the real draft rows, zeros, other
    random data, or absent (a shorter block); and whether NaN / Inf in the later rows leaks into rows 0..t. A kernel that reads a
    padded or stale tile row must pass the first three; the poison probe says which operations multiply later rows by a masked zero."""
    rows = args.rows
    src = make_source(regime, 2000 + seed, args)
    gen = torch.Generator().manual_seed(seed)
    norm_w = bf16(1 + 0.1 * torch.randn(DV, generator=gen))
    S0, norm_w = src.initial_state(seed + 7, args.users, norm_w)
    raw = src.draw(gen, rows)
    other = src.draw(torch.Generator().manual_seed(seed + 1), rows, lane=1)
    out = dict(regime=regime, rows=rows, forms={})
    for form, mname in (('seq', 'bf16'), ('wy', 'bf16'), ('wy', 'fp32')):
        m = MODELS[mname]
        cases = []
        for t in sorted({0, 3, 7, min(14, rows - 2)}):
            n = t + 1
            variants = {}
            for label in ('real', 'zeros', 'random', 'poison_nan', 'poison_inf'):
                mixed = {key: val.clone() for key, val in raw.items()}
                if label == 'zeros':
                    for key in mixed:
                        mixed[key][:, n:] = 0
                elif label == 'random':
                    for key in mixed:
                        mixed[key][:, n:] = other[key][:, n:]
                elif label.startswith('poison'):
                    for key in ('q', 'k', 'v'):
                        mixed[key][:, n:] = float('nan') if label == 'poison_nan' else float('inf')
                p = prep(mixed, norm_w, m.dtype)
                if form == 'seq':
                    o, snaps, _ = seq_round(S0.to(m.dtype), p, rows, m)
                    variants[label] = (o[:, :, :n], snaps[n - 1])
                else:
                    # the whole window is computed (rows after n are part of the fixed-shape launch), then n rows are committed
                    o_all, ctx = wy_window(S0.to(m.dtype), rows_of(p, min(rows, T)), m)
                    variants[label] = (o_all[:, :, :n], wy_commit(S0.to(m.dtype), ctx, n, m))
            # absent: a shorter block of exactly n rows
            p_short = prep(take_rows(raw, 0, n), norm_w, m.dtype)
            if form == 'seq':
                o, snaps, _ = seq_round(S0.to(m.dtype), p_short, n, m)
                variants['absent'] = (o, snaps[-1])
            else:
                o_all, ctx = wy_window(S0.to(m.dtype), rows_of(p_short, n), m)
                variants['absent'] = (o_all[:, :, :n], wy_commit(S0.to(m.dtype), ctx, n, m))
            ref = variants['real']
            case = dict(t=t, rows_committed=n)
            for label, (o, S) in variants.items():
                if label == 'real':
                    continue
                case[label] = dict(outputs_bitwise_equal=bits_equal(o, ref[0]), state_bitwise_equal=bits_equal(S, ref[1]),
                                   outputs_finite=bool(torch.isfinite(o).all()), state_finite=bool(torch.isfinite(S).all()))
            cases.append(case)
        required = ('zeros', 'random')
        out['forms']['%s_%s' % (form, mname)] = dict(
            cases=cases,
            causal_for_real_zero_random=all(c[v]['outputs_bitwise_equal'] and c[v]['state_bitwise_equal'] for c in cases for v in required),
            absent_bitwise_equal=all(c['absent']['outputs_bitwise_equal'] and c['absent']['state_bitwise_equal'] for c in cases),
            poison_leaks_into_committed_rows=any(not c[v]['outputs_finite'] or not c[v]['state_finite'] for c in cases
                                                 for v in ('poison_nan', 'poison_inf')))
    return out


def packed_solo(regime, args, seed=6):
    """A batch of users against each user alone, bit for bit, with a different committed length per user, and the batch in a second
    order: the property a 4-user launch must have against four 1-user launches. A CPU matmul library may pick different blocking for
    different batch sizes, so a difference here is a CPU-model fact, reported; the card harness checks the kernel's own bytes."""
    rows, users = args.rows, args.users
    src = make_source(regime, 3000 + seed, args)
    gen = torch.Generator().manual_seed(seed)
    norm_w = bf16(1 + 0.1 * torch.randn(DV, generator=gen))
    S0, norm_w = src.initial_state(seed + 7, users, norm_w)
    raw = src.draw(gen, rows)
    commits = draw_commits(torch.Generator().manual_seed(seed + 3), users, geometric_p(TAU, T), min(rows, T))
    out = dict(regime=regime, rows=rows, users=users, commits=commits, forms={})

    def run(kind, m, sel):
        """outputs and committed states for the users in `sel` (a list of user indices), as one batch."""
        sub = {key: val[sel] for key, val in raw.items()}
        p = prep(sub, norm_w, m.dtype)
        S = S0[sel].to(m.dtype)
        results = []
        if kind == 'seq':
            o, snaps, _ = seq_round(S, p, rows, m)
            for i, u in enumerate(sel):
                results.append((o[i:i + 1, :, :commits[u]], snaps[commits[u] - 1][i:i + 1]))
        else:
            o, ctx = wy_window(S, rows_of(p, min(rows, T)), m)
            for i, u in enumerate(sel):
                ci = dict(G=ctx['G'][i:i + 1], kn=ctx['kn'][i:i + 1], vnew=ctx['vnew'][i:i + 1])
                results.append((o[i:i + 1, :, :commits[u]], wy_commit(S[i:i + 1], ci, commits[u], m)))
        return results

    for kind, mname in (('seq', 'bf16'), ('wy', 'bf16'), ('wy', 'bf16+tf32')):
        m = MODELS[mname]
        everyone = list(range(users))
        packed = run(kind, m, everyone)
        reversed_ = run(kind, m, everyone[::-1])[::-1]
        solo = [run(kind, m, [u])[0] for u in everyone]
        same = lambda a, b: all(bits_equal(x[0], y[0]) and bits_equal(x[1], y[1]) for x, y in zip(a, b))
        out['forms']['%s_%s' % (kind, mname)] = dict(packed_equals_solo=same(packed, solo), packed_equals_reversed_batch=same(packed, reversed_),
                                                     users_differing=[u for u in everyone if not (bits_equal(packed[u][0], solo[u][0]) and bits_equal(packed[u][1], solo[u][1]))])
    return out


def draw_stream_and_state(src, args, seed, rows):
    gen = torch.Generator().manual_seed(seed)
    norm_w = bf16(1 + 0.1 * torch.randn(DV, generator=gen))
    S0, norm_w = src.initial_state(seed + 7, args.users, norm_w)
    return S0, norm_w


def drift(regime, args, seed=2):
    """Accept / reject cycles: each method carries its own state; inputs identical; committed n from the tau distribution."""
    rows, cycles = args.rows, args.cycles
    src = make_source(regime, seed, args)
    S0, norm_w = draw_stream_and_state(src, args, seed, rows)
    commits, commit_meta = commit_draw(args, seed + 11, cycles, rows)
    stream_gen = torch.Generator().manual_seed(seed + 13)
    reject_gen = torch.Generator().manual_seed(seed + 17)
    methods = dict(seq64=('seq', 'fp64'), wy64=('wy', 'fp64'), seq32=('seq', 'fp32'), wy32=('wy', 'fp32'),
                   seqbf=('seq', 'bf16'), wybf=('wy', 'bf16'), wybfmid=('wy', 'bf16mid'),
                   seqtf=('seq', 'bf16+tf32'), wytf=('wy', 'bf16+tf32'))
    state = {k: S0.to(MODELS[mm].dtype).clone() for k, (_, mm) in methods.items()}
    pairs = dict(wy32_vs_seq32=('wy32', 'seq32'), wybf_vs_seqbf=('wybf', 'seqbf'), wybfmid_vs_seqbf=('wybfmid', 'seqbf'),
                 wytf_vs_seqtf=('wytf', 'seqtf'), wy64_vs_seq64=('wy64', 'seq64'), wybf_vs_seqbf_o32=('wybf', 'seqbf_o32'),
                 seq32_vs_fp64=('seq32', 'seq64'), wy32_vs_fp64=('wy32', 'seq64'), seqbf_vs_fp64=('seqbf', 'seq64'),
                 wybf_vs_fp64=('wybf', 'seq64'), seqtf_vs_fp64=('seqtf', 'seq64'), wytf_vs_fp64=('wytf', 'seq64'))
    track = {k: dict(state=[], out_rel=[], gated_diff=0, gated_total=0, out_max_rel=0.0, state_max_rel=0.0,
                     nonfinite_cycles=0) for k in pairs}
    stream = []
    windows_used = [0, 0]
    t0 = time.time()
    for c in range(cycles):
        n = commits[c]
        good = src.draw(stream_gen, n)
        if c < args.seg_cycles:
            stream.append(good)
        raw = cat_rows([good, src.draw(reject_gen, rows - n, lane=1)]) if n < rows else good
        preps = {mm: prep(raw, norm_w, MODELS[mm].dtype) for mm in ('fp64', 'fp32', 'bf16', 'bf16+tf32', 'bf16mid')}
        outs, gouts, new = {}, {}, {}
        for k, (kind, mm) in methods.items():
            m, p = MODELS[mm], preps[mm]
            if kind == 'seq':
                pre = [] if k == 'seqbf' else None
                o, _, new[k] = seq_round(state[k], p, n, m, pre_out=pre)   # == snapshot n of the rows-row verify
                if pre:
                    outs['seqbf_o32'] = torch.stack(pre, 2)
                    gouts['seqbf_o32'] = gated(outs['seqbf_o32'], p, n)
            else:
                o, new[k], w = wy_verify(state[k], p, n, m)
                if k == 'wybf':
                    windows_used[w - 1] += 1
            outs[k], gouts[k] = o, gated(o, p, n)
        state = new
        for name, (a, b) in pairs.items():
            tr = track[name]
            so = cmp(outs[a], outs[b])
            sg = cmp(gouts[a], gouts[b])
            tr['gated_diff'] += sg['differing']
            tr['gated_total'] += sg['elements']
            tr['nonfinite_cycles'] += 0 if (so['finite'] and sg['finite']) else 1
            tr['out_max_rel'] = max(tr['out_max_rel'], so['rel_to_max'])
            if (c + 1) in CHECKPOINTS or (c + 1) % 25 == 0:
                ss = cmp(state[a], state[b if b != 'seqbf_o32' else 'seqbf'])
                tr['state_max_rel'] = max(tr['state_max_rel'], ss['rel_to_max'])
                if not ss['finite']:
                    tr['nonfinite_cycles'] += 1
                if (c + 1) in CHECKPOINTS and (c + 1) <= cycles:
                    tr['state'].append(dict(cycle=c + 1, **short(ss)))
                    tr['out_rel'].append(dict(cycle=c + 1, core_rel_to_max=float('%.4g' % so['rel_to_max']),
                                              gated_bf16_differing_frac=float('%.4g' % (sg['differing'] / sg['elements']))))
        if (c + 1) % 100 == 0:
            print('  drift %s cycle %d (%.0fs)' % (regime, c + 1, time.time() - t0), flush=True)
    for tr in track.values():
        tr['gated_differing_frac_all_cycles'] = tr.pop('gated_diff') / tr.pop('gated_total')
    return dict(regime=regime, rows=rows, cycles=cycles, commit_distribution=commit_meta, mean_commit=sum(commits) / len(commits),
                commit_hist={str(i): commits.count(i) for i in range(1, rows + 1)}, tokens=sum(commits),
                rounds_with_window_one_only=windows_used[0], rounds_with_two_windows=windows_used[1],
                seconds=time.time() - t0, pairs=track), dict(stream=stream, S0=S0, norm_w=norm_w, src=src, commits=commits[:args.seg_cycles])


def segmentation(regime, ctx, args, seed=3):
    """The same committed token stream split into rounds two ways (the drift's acceptance draw A, and an independent draw B from the
    same distribution): per method, outputs at every position and states at every common commit point, A against B. The sequential
    form must be bitwise invariant; the window form is measured (a different split changes which rows share a window)."""
    rows = args.rows
    stream = cat_rows(ctx['stream'])
    L = stream['g'].shape[1]
    commitsA = ctx['commits']
    commitsB, meta = [], None
    gB = torch.Generator().manual_seed(seed + 19)
    while sum(commitsB) < L:
        drawn, meta = commit_draw(args, int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gB)), 1, rows)
        commitsB += drawn
    commitsB[-1] -= sum(commitsB) - L
    commitsB = [c for c in commitsB if c > 0]
    norm_w, src = ctx['norm_w'], ctx['src']

    def run(kind, mname, commits, rseed):
        m = MODELS[mname]
        rgen = torch.Generator().manual_seed(rseed)
        S = ctx['S0'].to(m.dtype).clone()
        pos, outs, states = 0, [], {}
        for n in commits:
            good = take_rows(stream, pos, pos + n)
            raw = cat_rows([good, src.draw(rgen, rows - n, lane=1)]) if n < rows else good
            p = prep(raw, norm_w, m.dtype)
            if kind == 'seq':
                o, _, S = seq_round(S, p, n, m)
            else:
                o, S, _ = wy_verify(S, p, n, m)
            outs.append(o)
            pos += n
            states[pos] = S
        return torch.cat(outs, 2), states

    common = sorted(set(itertools.accumulate(commitsA)) & set(itertools.accumulate(commitsB)))
    res = dict(regime=regime, tokens=L, roundsA=len(commitsA), roundsB=len(commitsB), common_commit_points=len(common))
    for kind, mname in (('seq', 'fp32'), ('wy', 'fp32'), ('seq', 'bf16'), ('wy', 'bf16')):
        oA, sA = run(kind, mname, commitsA, 101)
        oB, sB = run(kind, mname, commitsB, 202)
        st = [cmp(sA[pt], sB[pt]) for pt in common]
        res['%s_%s' % (kind, mname)] = dict(
            outputs_A_vs_B=short(cmp(oA, oB)),
            outputs_bitwise_identical=bits_equal(oA, oB),
            states_at_common_points=dict(max_rel_to_max=max((s['rel_to_max'] for s in st), default=0.0),
                                         points_bitwise_identical=sum(1 for s in st if s['differing'] == 0),
                                         points=len(st), last_point=short(st[-1]) if st else None))
    return res


# ---------------------------------------------------------------- the verdict

def acceptance(report):
    """E1: the window form's error against fp64 no worse than the served chain's (bf16 class), state and output, in every drift case,
    nothing non-finite. Also the contract fields: what rules the form out of serving, whatever E1 says."""
    problems, rows = [], []
    for key, drift_case in report.get('drift', {}).items():
        pairs = drift_case['pairs']
        for what in ('state_max_rel', 'out_max_rel'):
            seq_err, wy_err = pairs['seqbf_vs_fp64'][what], pairs['wybf_vs_fp64'][what]
            ratio = wy_err / seq_err if seq_err else float('inf') if wy_err else 1.0
            rows.append(dict(case=key, metric=what, seq_vs_fp64=seq_err, wy_vs_fp64=wy_err, ratio=ratio, ok=ratio <= E1_RATIO_MAX))
            if ratio > E1_RATIO_MAX:
                problems.append('%s %s: window error %.4g vs served %.4g (ratio %.3g > %.2f)' % (key, what, wy_err, seq_err, ratio, E1_RATIO_MAX))
        for name, tr in pairs.items():
            if tr['nonfinite_cycles']:
                problems.append('%s %s: %d cycles with non-finite values' % (key, name, tr['nonfinite_cycles']))
    contract_differing = [(key, case['pairs']['wybf_vs_seqbf']['gated_differing_frac_all_cycles'])
                          for key, case in report.get('drift', {}).items()]
    packed_ok = all(f['packed_equals_solo'] for case in report.get('packed_solo', {}).values() for f in case['forms'].values()) \
        if report.get('packed_solo') else None
    seg = {key: case['wy_bf16']['outputs_bitwise_identical'] for key, case in report.get('segmentation', {}).items()}
    return dict(
        criterion='E1: window error vs fp64 <= %.2f x served-chain error (bf16 class), state and output, every drift case, all finite' % E1_RATIO_MAX,
        e1_pass=not problems and bool(rows), e1_worst_ratio=max((r['ratio'] for r in rows), default=None), e1_rows=rows, problems=problems,
        contract=dict(byte_identical_to_served=bool(contract_differing) and all(f == 0.0 for _, f in contract_differing),
                      gated_bf16_differing_fraction_wybf_vs_seqbf=dict(contract_differing),
                      packed_equals_solo=packed_ok, segmentation_invariant_wy_bf16=seg,
                      serving_eligible=False,
                      reason='the window form is a different arithmetic: its bytes are not the served bytes (docs/gdn-wy-probe.md)'))


# ---------------------------------------------------------------- driver

def counts_section(args):
    return dict(per_user_head_round=dict(tau=M.counts(TAU), worst=M.counts(T), window_rows=T),
                tile_ops_wy_estimate=M.tile_ops_wy(), tile_ops_seq_served='124 per token x 16 = 1984 (gdn_seq_block.py docstring)',
                sram_plan_two_windows=M.sram_plan(windows=2), dram_plan_two_windows=M.dram_bytes_per_core(windows=2))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sections', default=','.join(SECTIONS))
    ap.add_argument('--regimes', default='model,R1', help='comma list of model, R1, R2, fixture:<name>')
    ap.add_argument('--rows', type=int, default=T, help='rows of the verify block: 16 (one window) or 32 (two chained windows)')
    ap.add_argument('--commit-dist', choices=('tau', 'stress'), default='tau')
    ap.add_argument('--commit-hist', default='', help='JSON {"n": count} of committed lengths (the lab histogram) instead of the fit')
    ap.add_argument('--cycles', type=int, default=1000)
    ap.add_argument('--seg-cycles', type=int, default=300)
    ap.add_argument('--users', type=int, default=USERS)
    ap.add_argument('--nk', type=int, default=NK)
    ap.add_argument('--nv', type=int, default=NV)
    ap.add_argument('--fixture', default='')
    ap.add_argument('--repeat', type=int, default=1, help='single-round repeats (leaf-wise median)')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--label', default='', help='free text for the report, e.g. the matrix cell')
    ap.add_argument('--smoke', action='store_true', help='tiny scale for a local smoke run (never a result)')
    a = ap.parse_args(argv)
    a.sections = [s for s in a.sections.split(',') if s]
    a.regimes = [r for r in a.regimes.split(',') if r]
    if any(s not in SECTIONS for s in a.sections):
        ap.error('--sections names only %s' % (SECTIONS,))
    if a.rows not in (16, 32):
        ap.error('--rows is 16 or 32')
    if a.smoke:
        a.users, a.nk, a.nv, a.cycles, a.seg_cycles = 2, 1, 3, 12, 8
    if a.nv % a.nk or a.users < 1 or a.cycles < 1:
        ap.error('--nv must be a multiple of --nk; --users and --cycles positive')
    return a


def main(argv=None):
    args = parse(argv)
    torch.set_num_threads(args.threads or os.cpu_count() or 2)
    t0 = time.time()
    report = dict(kind=KIND, schema=SCHEMA, label=args.label, smoke=args.smoke,
                  scope='NON-EXACT research probe: the aligned-window WY form against the served sequential chain; never serves',
                  geometry=dict(nk_tp=args.nk, nv_tp=args.nv, dk=DK, dv=DV, T=T, rows=args.rows, window_rows=T, users=args.users,
                                gdn_layers=M.GDN_LAYERS, tau=TAU),
                  argv=sys.argv[1:] if argv is None else list(argv), torch=torch.__version__, python=platform.python_version(),
                  source_sha=os.environ.get('GITHUB_SHA', ''), regimes=args.regimes, commit_dist=args.commit_dist,
                  inputs=dict(real_layers=bool(args.fixture), fixture=os.path.basename(args.fixture) if args.fixture else None,
                              note='synthetic regimes unless fixtures are given: the weights are not in CI'))
    todo = args.sections

    def save():
        with open(args.out, 'w', newline='\n') as handle:
            json.dump(report, handle, indent=1)

    if 'crosscheck' in todo:
        report['repo_reference_crosscheck'] = crosscheck_repo_reference()
    if 'single' in todo:
        report['single_round'] = {}
        for regime in args.regimes:
            print('single round', regime, flush=True)
            report['single_round'][regime] = median_merge([single_round(regime, args) for _ in range(args.repeat)])
        report['single_round_repeats'] = args.repeat
        save()
    if 'causality' in todo:
        report['causality'] = {regime: causality(regime, args) for regime in args.regimes}
    if 'packed_solo' in todo:
        report['packed_solo'] = {regime: packed_solo(regime, args) for regime in args.regimes}
        save()
    if 'counts' in todo:
        report['counts'] = counts_section(args)
    report['drift'], report['segmentation'] = {}, {}
    if 'drift' in todo or 'segmentation' in todo:
        for regime in args.regimes:
            print('drift', regime, flush=True)
            report['drift'][regime], ctx = drift(regime, args)
            if 'segmentation' in todo:
                print('segmentation', regime, flush=True)
                report['segmentation'][regime] = segmentation(regime, ctx, args)
            save()
    if 'drift' not in todo:
        report.pop('drift')
    if 'segmentation' not in todo:
        report.pop('segmentation')
    report['acceptance'] = acceptance(report)
    report['seconds'] = time.time() - t0
    save()
    acc = report['acceptance']
    print('GDN_WY_NUMERICS e1=%s rows=%d regimes=%s byte_identical_to_served=%s serving_eligible=false smoke=%s' % (
        'PASS' if acc['e1_pass'] else 'FAIL', args.rows, ','.join(args.regimes), acc['contract']['byte_identical_to_served'], args.smoke))
    for problem in acc['problems']:
        print('  problem: ' + problem)
    print(json.dumps(dict(kind=KIND, e1_pass=acc['e1_pass'], e1_worst_ratio=acc['e1_worst_ratio'], rows=args.rows,
                          out=os.path.basename(args.out)), sort_keys=True))
    return 0 if acc['e1_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
