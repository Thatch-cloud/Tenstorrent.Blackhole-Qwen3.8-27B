"""CPU EMULATION of the arithmetic of the K-split against the sequential (oneq) kernel, in the waiver's statistic (ksplit_numerics.error_stats: error in bf16 ulps of the row's scale).

It is NOT the card: it rounds to bf16 where the pinned kernels store bf16 and keeps fp32 elsewhere, on random K / V, and it exists for one purpose, to put the a-priori reading of the
bound (docs/tp4-prefill3-ksplit.md, the waiver card) on numbers before a card is touched: how large the K-split's difference from the sequential call is when the ONLY differences are
the roundings the doc names. Every figure it prints is an ESTIMATE of the order of magnitude, and card M's measurement replaces it.

THE TWO ARITHMETICS (one q chunk of 128 rows, B blocks of 128 keys, head dim 256, softmax scale 1/16; no causal mask: the diagonal stamp touches one K block of the last slice only):

  sequential (oneq / stock)   one pass over all B blocks. Per block: s = scale x q.k; m = max(m, rowmax s); the running max and the correction factor exp(m_old - m) in the statistics format (bf16
                              by default in the pinned factory), the running sum in fp32 (sum_df), P = exp(s - m) in fp32 (qk_im_df); the accumulator O = O x factor + P.V rounded to bf16 per
                              block (out_im_df is bf16); at the end out = bf16(O / sum).
  ksplit                      E slices, slice s the blocks [s B / E, (s+1) B / E): each is the same pass with FP32 statistics (the lever's PK4), ending in out_s = bf16(O / sum) and
                              lse_s = max + log(sum) (fp32). The slices are folded IN ORDER by the pinned ring merge, every intermediate in bf16:
                              d = bf16(prev - cur); t = bf16(sigmoid(cur_lse - prev_lse) x d); out = bf16(prev - t); lse = prev_lse - logsigmoid(prev_lse - cur_lse) (fp32).

Assumptions that card M can break (the formats are read from the served factory text, sdpa_program_factory.fd8c0676.cpp lines 712-715: im_df bf16, stats_df bf16, qk_im_df and sum_df fp32 with
fp32 dest accumulation; set --fp32-accumulator for the other reading of the accumulator): O is rounded once per block (the pinned kernel packs the rescaled accumulator and the matmul result
through bf16 CBs: up to two roundings), the exp is exact (the pinned kernel's exp is an approximation: a few 1e-3 relative in the approximate mode; the lever REFUSES that mode, the sequential call does not), the dot products are
fp32 (the card's are HiFi2 on bf8 inputs, the SAME for both arithmetics).

    python3 -B ksplit_emulation.py [--blocks 8 64 256] [--slices 2 4] [--seeds 3] [--peaky 1.0 4.0]       needs torch (the scripts/ci venv); prints one markdown table
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / 'scripts' / 'ci'))

import ksplit_numerics as numerics  # noqa: E402
import ksplit_planner as planner  # noqa: E402

ROWS, BLOCK, DIM = 128, 128, 256
SCALE = 1.0 / 16.0                                  # 1 / sqrt(256): exact in bf16 (the lever's guard)


def rb(x, torch):
    """Round to bf16 and back."""
    return x.to(torch.bfloat16).to(torch.float32)


def one_pass(q, k, v, blocks, torch, stats_bf16, fp32_accumulator=False):
    """(normalised output rounded to bf16, lse fp32) of the pass over the key blocks `blocks` (a range). `stats_bf16`: the running max and the correction factor are stored in bf16 (the
    pinned factory's default; the K-split makes them fp32); the running sum is fp32 in both (sum_df)."""
    stat = (lambda x: rb(x, torch)) if stats_bf16 else (lambda x: x)
    acc = (lambda x: x) if fp32_accumulator else (lambda x: rb(x, torch))
    m = torch.full((q.shape[0],), float('-inf'))
    total = torch.zeros(q.shape[0])
    out = torch.zeros(q.shape[0], v.shape[1])
    for j in blocks:
        kb, vb = k[j * BLOCK:(j + 1) * BLOCK], v[j * BLOCK:(j + 1) * BLOCK]
        s = (q @ kb.T) * SCALE
        new_m = stat(torch.maximum(m, s.amax(dim=-1)))
        factor = stat(torch.exp(m - new_m)) if m.isfinite().all() else torch.zeros_like(m)
        p = torch.exp(s - new_m[:, None])
        total = total * factor + p.sum(dim=-1)
        out = acc(out * factor[:, None] + p @ vb)
        m = new_m
    lse = m + torch.log(total)
    return rb(out / total[:, None], torch), lse


def ring_merge(prev_out, prev_lse, cur_out, cur_lse, torch):
    """The pinned ring merge with every intermediate in bf16 (the lse in fp32)."""
    sig = torch.sigmoid(cur_lse - prev_lse)
    d = rb(prev_out - cur_out, torch)
    t = rb(d * sig[:, None], torch)
    out = rb(prev_out - t, torch)
    lse = prev_lse - torch.nn.functional.logsigmoid(prev_lse - cur_lse)
    return out, lse


def sequential(q, k, v, nblocks, torch, fp32_accumulator=False):
    return one_pass(q, k, v, range(nblocks), torch, True, fp32_accumulator)[0]


def ksplit(q, k, v, nblocks, slices, torch, fp32_accumulator=False):
    eff = planner.effective_slices(nblocks, slices)
    out = lse = None
    for s in range(eff):
        begin = planner.slice_begin(nblocks, s, eff)
        end = planner.slice_begin(nblocks, s + 1, eff) if s + 1 < eff else nblocks
        cur_out, cur_lse = one_pass(q, k, v, range(begin, end), torch, False, fp32_accumulator)
        out, lse = (cur_out, cur_lse) if s == 0 else ring_merge(out, lse, cur_out, cur_lse, torch)
    return out


def reference(q, k, v, nblocks, torch):
    s = (q.double() @ k[:nblocks * BLOCK].double().T) * SCALE
    return torch.softmax(s, dim=-1) @ v[:nblocks * BLOCK].double()


def measure(nblocks, slices, seed, peaky, torch, fp32_accumulator=False):
    gen = torch.Generator().manual_seed(seed)
    q = rb(torch.randn(ROWS, DIM, generator=gen) * peaky, torch)
    k = rb(torch.randn(nblocks * BLOCK, DIM, generator=gen), torch)
    v = rb(torch.randn(nblocks * BLOCK, DIM, generator=gen), torch)
    ref = reference(q, k, v, nblocks, torch).float()
    seq = sequential(q, k, v, nblocks, torch, fp32_accumulator)
    split = ksplit(q, k, v, nblocks, slices, torch, fp32_accumulator)
    return {'ksplit_vs_oneq': numerics.error_stats(split, seq, torch), 'ksplit_vs_ref': numerics.error_stats(split, ref, torch), 'oneq_vs_ref': numerics.error_stats(seq, ref, torch),
            'slices': planner.effective_slices(nblocks, slices)}


def table(blocks, slices, seeds, peaks, fp32_accumulator=False):
    import torch
    torch.set_num_threads(2)
    lines = ['| blocks (context) | S (E) | q scale | ksplit-oneq max / rms / >2u | ksplit-ref max / rms | oneq-ref max / rms | rms ratio ksplit/oneq vs ref |', '|---:|---:|---:|---|---|---|---:|']
    for nblocks in blocks:
        for nslices in slices:
            for peaky in peaks:
                rows = [measure(nblocks, nslices, seed, peaky, torch, fp32_accumulator) for seed in range(seeds)]

                def mean(key, stat):
                    return sum(row[key][stat] for row in rows) / len(rows)

                def peak(key, stat):
                    return max(row[key][stat] for row in rows)
                lines.append('| %d (%dk) | %d (%d) | %.1f | %.2f / %.2f / %.3f | %.2f / %.2f | %.2f / %.2f | %.2f |' % (
                    nblocks, nblocks * BLOCK // 1000, nslices, rows[0]['slices'], peaky,
                    peak('ksplit_vs_oneq', 'max_ulps'), mean('ksplit_vs_oneq', 'rms_ulps'), mean('ksplit_vs_oneq', 'frac_gt2'),
                    peak('ksplit_vs_ref', 'max_ulps'), mean('ksplit_vs_ref', 'rms_ulps'), peak('oneq_vs_ref', 'max_ulps'), mean('oneq_vs_ref', 'rms_ulps'),
                    mean('ksplit_vs_ref', 'rms_ulps') / max(mean('oneq_vs_ref', 'rms_ulps'), 1e-9)))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--blocks', type=int, nargs='+', default=[8, 64, 256])
    parser.add_argument('--slices', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--peaky', type=float, nargs='+', default=[1.0, 4.0])
    parser.add_argument('--fp32-accumulator', action='store_true')
    options = parser.parse_args(argv)
    print(table(options.blocks, options.slices, options.seeds, options.peaky, options.fp32_accumulator))
    return 0


if __name__ == '__main__':
    sys.exit(main())
