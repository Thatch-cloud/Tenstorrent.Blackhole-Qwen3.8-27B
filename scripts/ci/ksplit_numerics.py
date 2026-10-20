"""The numbers of the K-split numerics waiver (docs/tp4-prefill3-ksplit.md, section "The numerics waiver"): the error statistic, the PROVISIONAL bound, the noise-floor reference. One source for
the audit at the model (sdpa_pf_ksplit_tp.py, QWEN_FAST_SDPA_PF_KSPLIT_AUDIT) and the card-M harness (optimisation/ttnn-op/sdpa_prefill_ksplit/ksplit_card_m.py), so the audit can
never judge by a rule the qualification did not use. Torch is imported by the functions that need it.

THE STATISTIC. The K-split's attention output differs from the oneq call's in the last bits (a different association of the softmax: partial results merged with the pinned ring merge).
An element's error is measured in units of the bf16 ulp of its ROW'S SCALE: for the row (one head, one query row, 256 columns) with scale R = max |reference|, the unit is
2^(floor(log2 R) - 7), the spacing of bf16 numbers just below 2R. So 1.0 is "one bf16 step of the row's largest element", a rounding of the largest element is at most 0.5 and of any other
element at most that. The statistic is the distribution of |ksplit - reference| / unit over all elements of a call: max, rms, a high quantile, the share above 1 and 2 units, and the worst
row's own max. It is the quantity a downstream bf16 consumer of the output cannot tell apart from rounding when it stays near 1.

THE BOUND IS PROVISIONAL. BOUND below is the a-priori reading of the merge's roundings (derived in the doc: one bf16 round trip of each slice's normalised output, three bf16 roundings in each
merge, the final rounding), NOT a measurement. It must be replaced by the numbers card M measures (ksplit vs oneq per context, and both against the float reference) before the owner approves the
waiver; until then it is only the audit's tripwire at a gate. NOISE says the second half of the rule: against the float reference the K-split must be no further than NOISE times the oneq call's
own distance (the sequential kernel's accumulator is rounded to bf16 at every one of up to ~2,000 blocks; a split has fewer sequential steps per slice).
"""

import math

# a-priori, provisional (see the module docstring and the doc's waiver card)
BOUND = {'max_ulps': 8.0, 'rms_ulps': 1.5, 'frac_gt2': 0.02}
NOISE = {'rms': 1.25, 'max': 1.5}
BF16_MANTISSA_BITS = 7
SCALE_FLOOR = 2.0 ** -120


def row_unit(scale, torch):
    """Tensor of bf16 ulp units for row scales (a tensor of the same shape as `scale`): 2^(floor(log2 scale) - 7); a zero row has the unit of SCALE_FLOOR."""
    return torch.pow(2.0, torch.floor(torch.log2(scale.clamp_min(SCALE_FLOOR))) - BF16_MANTISSA_BITS)


def error_stats(candidate, reference, torch):
    """The statistic of `candidate` against `reference` (same shape [..., rows, cols], any float dtype). -> dict of floats and ints."""
    if tuple(candidate.shape) != tuple(reference.shape):
        raise ValueError('shapes %s and %s differ' % (tuple(candidate.shape), tuple(reference.shape)))
    a, b = candidate.to(torch.float32), reference.to(torch.float32)
    nonfinite = int((~torch.isfinite(a)).sum())
    diff = (a - b).abs()
    unit = row_unit(b.abs().amax(dim=-1, keepdim=True), torch)
    ulps = diff / unit
    flat = ulps.reshape(-1)
    count = flat.numel()
    top = max(1, int(math.ceil(count * 0.001)))
    p999 = float(torch.topk(flat, top).values.min()) if count else 0.0
    rows_max = ulps.amax(dim=-1).reshape(-1)
    norm = float(b.norm())
    return dict(elements=count, nonfinite=nonfinite, max_abs=float(diff.max()) if count else 0.0, rel_l2=float(diff.norm()) / norm if norm > 0 else 0.0,
                max_ulps=float(flat.max()) if count else 0.0, rms_ulps=float(flat.pow(2).mean().sqrt()) if count else 0.0, p999_ulps=p999,
                frac_gt1=float((flat > 1.0).float().mean()) if count else 0.0, frac_gt2=float((flat > 2.0).float().mean()) if count else 0.0,
                row_max_p99_ulps=float(torch.quantile(rows_max, 0.99)) if rows_max.numel() <= 16_000_000 else float(rows_max.max()))


def within_bound(stats, bound=None):
    """[problem] when `stats` (error_stats of ksplit vs oneq) is outside the bound; [] when inside."""
    bound = BOUND if bound is None else bound
    problems = []
    if stats['nonfinite']:
        problems.append('%d non-finite elements' % stats['nonfinite'])
    for key in ('max_ulps', 'rms_ulps', 'frac_gt2'):
        if stats[key] > bound[key]:
            problems.append('%s %.4g > %.4g' % (key, stats[key], bound[key]))
    return problems


def noise_problems(candidate_to_reference, baseline_to_reference, noise=None):
    """[problem] when the candidate (ksplit) is further from the float reference than the baseline (oneq) by more than the NOISE ratios."""
    noise = NOISE if noise is None else noise
    problems = []
    if candidate_to_reference['rms_ulps'] > noise['rms'] * max(baseline_to_reference['rms_ulps'], 1e-9):
        problems.append('rms %.4g vs the baseline %.4g (ratio limit %.2f)' % (candidate_to_reference['rms_ulps'], baseline_to_reference['rms_ulps'], noise['rms']))
    if candidate_to_reference['max_ulps'] > noise['max'] * max(baseline_to_reference['max_ulps'], 1e-9):
        problems.append('max %.4g vs the baseline %.4g (ratio limit %.2f)' % (candidate_to_reference['max_ulps'], baseline_to_reference['max_ulps'], noise['max']))
    return problems


def reference_attention(q, k, v, start, scale, torch, chunk=128):
    """Float32 causal attention for a chunked-prefill call: q [heads, rows, dim] at absolute row `start`, k and v [length >= start + rows, dim] (ONE KV head shared by every q head).
    -> [heads, rows, dim]. Blocked over q chunks (never builds more than [heads, chunk, length] scores)."""
    heads, rows, _dim = q.shape
    out = torch.empty_like(q, dtype=torch.float32)
    q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    for index in range(rows // chunk):
        low = start + index * chunk
        high = low + chunk
        scores = torch.einsum('hqd,kd->hqk', q32[:, index * chunk:(index + 1) * chunk], k32[:high]) * scale
        keys = torch.arange(high)[None, :]
        future = keys > (low + torch.arange(chunk))[:, None]
        scores = scores.masked_fill(future[None], float('-inf'))
        out[:, index * chunk:(index + 1) * chunk] = torch.softmax(scores, dim=-1) @ v32[:high]
    return out


def gather_pages(pool, table, length, head=0):
    """The logical K or V rows [length, dim] of a paged pool [blocks, kv_heads, block, dim] through a page table row (block ids)."""
    block = pool.shape[2]
    needed = -(-length // block)
    pages = pool[table[:needed].long(), head]
    return pages.reshape(needed * block, pool.shape[3])[:length]
