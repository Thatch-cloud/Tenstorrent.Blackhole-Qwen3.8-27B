"""CPU cost model of long-context prefill on the four-card TP4 serving stack (stdlib only): one 2,048-token chunk per op class, the whole step of a
returning agent turn, and what each candidate change is worth (docs/tp4-prefill-2.md).

    python3 -B scripts/ci/prefill_cost_model.py chunk            the per-class table of one chunk at 4k / 32k / 128k / 160k / 254k of context
    python3 -B scripts/ci/prefill_cost_model.py step 120832 3112  one step: start of the step (a multiple of 2,048) and its new tokens
    python3 -B scripts/ci/prefill_cost_model.py levers 140000 3500 the candidate list priced at one step

EVERY NUMBER HERE IS EITHER MEASURED (M) OR AN ESTIMATE (E); the label is in the name of the table it lives in.

M  PF1 (device profile, run 38051000905): four p150a cards, 13x10 workers each, the production profile, one 131,072-token prompt as 64 chunks of 2,048 rows. Per-chunk device
   kernel milliseconds by op class, median of the four chips (chips agree within 0.1 percent), fitted per class as a + b x context(k tokens) over chunks 1-63 (R^2 1.000 for
   the total and for SDPA). Chunk 0 lost its profiler markers, so a fit intercept stands for it. The unprofiled twin of the same prompt measured the wall of a chunk as
   5.5 ms + 1.1286 x kernel ms (63 chunks); the stretch is the cards' 300 W board limit (loaded AICLK ~1,210 MHz against 1,345 in the profile) plus host dispatch gaps.
M  oneq (docs/tp4-ship4.md, optimisation/ttnn-op/sdpa_prefill_oneq): the card-M ratio of the one-q-chunk-per-core attention call to the paired one at TP4, per layer, 2,048 rows
   (W1 run 38066157833, Q1 run 38066212228: 360 of 360 byte-exact). The ratio is what multiplies PF1's SDPA fit on the ship-4 stack.
M  PRODUCTION FIT: 1,608 'lever N route' lines of one 1.85-hour window of the ship-4 stack serving the Swift checkpoint (1,024 finished prompts of which 996 returning turns),
   fitted as milliseconds against the step's start context: a 2,048-row chunk pass, and the end-of-prompt pass per masked tail bucket (the tail's own forward plus logits, the
   host snapshot of the GDN scratch and the decode-slot write). Aggregates only; no request, user or host is named.
M  HANDOFF: the same PF1 profile, chip 0, after the last chunk: the host snapshot of the B=1 GDN scratch (240 blocking reads) opened a 48.6 ms gap, and the 48-layer decode-slot
   write (1,919 device ops: tilize, untilize, slice, concat, copy) ran 115.2 ms for 8.0 ms of device kernels. That run was profiled (every op dispatch carries the profiler's host
   cost), so it is the upper end; production's end-of-prompt pass minus a chunk pass is 25-65 ms at 40-200k of context, which bounds the production hand-off from above.
E  everything else (the levers' savings) is arithmetic on the rows above and says so.
"""

import argparse
import sys

CHUNK = 2048
WALL_FIXED_MS = 5.5
WALL_PER_KERNEL = 1.1286

# M PF1: non-SDPA kernel ms per chunk by group (the class fits' intercepts; their slopes are zero within 0.005 ms per 1k tokens of context)
NON_SDPA_MS = (
    ('MATMUL', 119.29, 'gdn_in 27.44, attn_in 9.51, mlp gate/up 48.98 (fused all-gather matmul, 76 cores), gdn_out 15.03, attn_out 2.59, mlp_down 15.73 (10x10 grids)'),
    ('GDN', 48.71, 'conv 4.80, gates 0.59, prep 16.48, scan 26.84 (the scan keeps 48 of 130 cores busy: 12 v-heads x 4 column tiles)'),
    ('COLLECTIVE', 48.86, 'reduce-scatters after the gdn (fp32, 25.4), mlp (17.4) and attention (2.7) out-projections, the norm statistics gather (3.3)'),
    ('GLUE', 20.39, 'gdn split/layout/state copies 7.7, head/attention splits 3.5, the gate/up repack (GenericOp) 8.3, rope 0.8, kv fill 0.1'),
    ('NORM', 12.18, 'layernorm pre/post all-gather 9.2, gated norm 2.1, qk norm 0.9'),
    ('ELTWISE', 11.72, 'residual adds 5.9, z-gate 5.2, attention gate 0.7'),
)
SDPA_A_MS = 5.05            # PF1 SDPA fit, paired causal pairing as profiled: a + b x context(k)
SDPA_B_MS_PER_K = 3.3841
NON_SDPA_TOTAL_MS = sum(row[1] for row in NON_SDPA_MS)

# M card-M, per attention layer at TP4, 2,048 rows, Q in DRAM: (context in k tokens, served ms, oneq ms)
ONEQ_CARD_M = ((4.0, 0.96, 0.69), (34.0, 8.73, 4.58), (67.0, 17.06, 8.73), (131.0, 33.17, 16.80), (199.0, 50.26, 25.30), (254.0, 64.27, 32.34))
BLOCK_STEP_US = 13.51       # M card-M K0a: one 128x128x256 block on one core, HiFi2, the compute floor
# M what card M's ratio becomes in the served stack. The solo TTFT A/B of the lever (OQ-T1..T4 on the fusion stack: runs 38095202391, 38095525167, 38095893373, 38096201565) measured the 131,072-token
# prompt at 37.0 / 36.9 s without it and 31.2 / 31.1 s with it: 5.9 s of the 7.8 s the profile-anchored estimate gave (an effective SDPA ratio of 0.63, not 0.505: 96 busy cores draw more of a 300 W
# board limit and the rest of the chunk pays for it in clock), and the 32k rung 7.11 s -> 6.8 s, the 4k rung unmoved. The production route lines say the same from the other side: a chunk pass
# costs 2.108 ms more per 1k tokens of context, which is 1.1286 x 3.3841 x 0.552. The model takes the production slope: card M's ratio times 1.09.
ONEQ_IN_SITU = 0.552 / 0.5065
ATTENTION_LAYERS = 16

# M the production fit (ship-4 stack, Swift checkpoint): milliseconds = a + b x (start context in k tokens)
PRODUCTION_CHUNK = (321.1, 2.108)
PRODUCTION_TAIL = {128: (324.0, 1.386), 256: (290.0, 3.305), 512: (329.0, 1.274), 1024: (381.0, 1.251), 2048: (407.0, 1.657)}
MASK_BUCKETS = tuple(sorted(PRODUCTION_TAIL))

# M the hand-off, profiled run (upper end) and the production bound (lower end, see the module docstring)
HANDOFF_PROFILED = {'read_ms': 48.6, 'write_ms': 115.2, 'device_kernel_ms': 8.0, 'ops': 1919, 'tensors': 240, 'layers': 48,
                    'upload_ms_per_layer': 0.39, 'rec_ms_per_layer': 0.15, 'conv_ms_per_layer': 1.67}
HANDOFF_PRODUCTION_MS = (30.0, 60.0, 90.0)       # E low / central / high of the whole read + write on the production stack
CLONE_DISPATCH_US = 50.0                          # E host dispatch of one device-side clone (the profile's median gap between ops is 56 us)


def interpolate(points, x):
    """Piecewise-linear value at x of (x, y) points sorted by x; clamped outside."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def oneq_ratio(context_tokens):
    """M the oneq-over-paired SDPA time ratio at a context (card M, per layer): 0.72 at 4k, 0.52 from 34k, 0.50 beyond 199k. The served stack sees ONEQ_IN_SITU times it."""
    return interpolate([(k, one / served) for k, served, one in ONEQ_CARD_M], context_tokens / 1000.0)


def sdpa_kernel_ms(context_tokens, oneq=True):
    """M/E SDPA kernel ms of one chunk (16 attention layers) at a context; oneq=False is PF1 as profiled (paired), True the ship-4 stack."""
    paired = SDPA_A_MS + SDPA_B_MS_PER_K * context_tokens / 1000.0
    return paired * oneq_ratio(context_tokens) * ONEQ_IN_SITU if oneq else paired


def chunk_kernel_ms(context_tokens, oneq=True):
    return NON_SDPA_TOTAL_MS + sdpa_kernel_ms(context_tokens, oneq)


def chunk_wall_ms(context_tokens, oneq=True):
    """E the wall of one traced-equivalent chunk in the unprofiled run: 5.5 ms + 1.1286 x kernel."""
    return WALL_FIXED_MS + WALL_PER_KERNEL * chunk_kernel_ms(context_tokens, oneq)


def chunk_table(contexts=(4096, 32768, 131072, 163840, 253952), oneq=True):
    """[(context, {group: ms}, kernel ms, wall ms, share of the context-dependent part)] for one 2,048-row chunk starting at each context (E for 254k: the profile stops at 129k)."""
    rows = []
    for context in contexts:
        groups = dict((name, ms) for name, ms, _ in NON_SDPA_MS)
        groups['SDPA'] = sdpa_kernel_ms(context, oneq)
        kernel = sum(groups.values())
        rows.append((context, groups, kernel, WALL_FIXED_MS + WALL_PER_KERNEL * kernel, groups['SDPA'] / kernel))
    return rows


def bucket_for(tail):
    """The masked bucket the model's _mask_bucket_for picks for a tail of `tail` real tokens (0: no tail)."""
    if tail <= 0:
        return 0
    for bucket in MASK_BUCKETS:
        if tail <= bucket:
            return bucket
    raise ValueError('a tail is shorter than a chunk')


def passes(tokens):
    """(full chunks, tail tokens) of a step of `tokens` new tokens, as prefill_traced_chunked splits it."""
    return tokens // CHUNK, tokens % CHUNK


def route_ms(start, tokens, final=True):
    """E/M the 'lever N route' milliseconds of one step from the production fit: its full chunks (each at its own start) plus, for a final step with a tail, the end-of-prompt pass.
    A final step that ends on a chunk boundary has no tail pass: the last chunk then carries the logits and the hand-off, which the fit's chunk pass already averages over."""
    full, tail = passes(tokens)
    total = 0.0
    for index in range(full):
        total += PRODUCTION_CHUNK[0] + PRODUCTION_CHUNK[1] * (start + index * CHUNK) / 1000.0
    if tail and final:
        a, b = PRODUCTION_TAIL[bucket_for(tail)]
        total += a + b * (start + full * CHUNK) / 1000.0
    elif tail:
        raise ValueError('a step that is not final ends on a %d-token boundary' % CHUNK)
    return total


def step_breakdown(start, tokens):
    """E the split of a final step at `start`: the K-stream SDPA (the context-dependent part), the rest of the device work, and host/other, from the PF1 + oneq model; the production
    fit gives the total, the model the shares."""
    full, tail = passes(tokens)
    contexts = [start + i * CHUNK for i in range(full)] + ([start + full * CHUNK] if tail else [])
    sdpa = sum(sdpa_kernel_ms(c) for c in contexts) * WALL_PER_KERNEL
    rest = sum(NON_SDPA_TOTAL_MS for _ in contexts) * WALL_PER_KERNEL
    total = route_ms(start, tokens)
    return dict(total_ms=total, sdpa_ms=sdpa, other_device_ms=rest, host_and_handoff_ms=max(0.0, total - sdpa - rest), passes=len(contexts))


# -- the candidates ----------------------------------------------------------------------------------------------------------------------------------

def handoff_saving_ms(level='central'):
    """E what the device-side clone saves per final step (docs/tp4-prefill-2.md section 5): the 240 blocking reads and the 240 uploads go, 240 device clones come; the slot write's own
    ops (the conv taps' untilize/slice/concat/tilize) stay. Scaled from the profiled split by the production/profile ratio."""
    index = {'low': 0, 'central': 1, 'high': 2}[level]
    total = HANDOFF_PRODUCTION_MS[index]
    profiled = HANDOFF_PROFILED['read_ms'] + HANDOFF_PROFILED['write_ms']
    share_removed = (HANDOFF_PROFILED['read_ms'] + HANDOFF_PROFILED['upload_ms_per_layer'] * HANDOFF_PROFILED['layers']) / profiled
    clone_ms = HANDOFF_PROFILED['tensors'] * CLONE_DISPATCH_US / 1000.0 * (total / profiled)
    return max(0.0, total * share_removed - clone_ms)


def undrop_start(start, tokens):
    """E exact resume P0 (tp4/exact-end): the first step resumes one chunk higher, at the 2,048 boundary the cap already publishes; the step loses its first full chunk."""
    if tokens < 2 * CHUNK and tokens % CHUNK == 0:
        return start, tokens
    if tokens < CHUNK:
        return start, tokens
    return start + CHUNK, tokens - CHUNK


SPLIT_K_CAP = 4              # E the most cores that share one q chunk's K stream before the partial-softmax merge costs what it saves
SPLIT_K_EFFICIENCY = 0.8     # E the share of the ideal split a merge step leaves
HEADS_LOCAL = 6              # TP4: six local q heads on one KV head
WORKER_CORES = 130           # the 13x10 grid
TRACED_TAIL_KEEP = 0.8       # E the share of (eager floor - device time) a traced tail pass would really recover


def passes_of(start, tokens):
    """[(context of the pass, rows)] of a final step: its full chunks, then the masked tail's bucket."""
    full, tail = passes(tokens)
    out = [(start + index * CHUNK, CHUNK) for index in range(full)]
    if tail:
        out.append((start + full * CHUNK, bucket_for(tail)))
    return out


def split_k_slices(rows):
    """E how many cores can share one q chunk's K stream in a pass of `rows` rows: whole slices only (a pass whose q chunks already fill most of the grid cannot be split at all)."""
    busy = HEADS_LOCAL * (rows // 128)
    return max(1, min(SPLIT_K_CAP, WORKER_CORES // busy))


def split_k_saving_ms(start, tokens):
    """E what a K-split (flash-decoding) attention would save on a step, per pass: the K stream of a q chunk is serial on one core (oneq is already at that floor), so a pass whose q chunks leave
    cores idle can share each stream between S cores and merge S partial (output, log-sum-exp) pairs. A 2,048-row pass keeps 96 of 130 cores busy: S = 1, nothing to gain. A 1,024-row one keeps 48
    (S = 2), a 512-row one 24 (S = 4 at the cap). The merge makes it a tolerance-class change."""
    saved = 0.0
    for context, rows in passes_of(start, tokens):
        slices = split_k_slices(rows)
        effective = 1.0 + (slices - 1) * SPLIT_K_EFFICIENCY
        saved += sdpa_kernel_ms(context) * WALL_PER_KERNEL * (1.0 - 1.0 / effective)
    return saved


def traced_tail_saving_ms(start, tokens):
    """E what tracing the masked tail pass would save: the eager pass costs its fixed floor (production fit) while its device work is the non-SDPA time of its rows plus the end-of-prompt
    work (~86 ms: the fit's tail floor minus its chunk floor); only a bucket whose device work is below the floor gains."""
    full, tail = passes(tokens)
    if not tail:
        return 0.0
    bucket = bucket_for(tail)
    device = NON_SDPA_TOTAL_MS * WALL_PER_KERNEL * bucket / CHUNK + (PRODUCTION_TAIL[2048][0] - PRODUCTION_CHUNK[0])
    return max(0.0, TRACED_TAIL_KEEP * (PRODUCTION_TAIL[bucket][0] - device))


def lever_table(start, tokens):
    """[(name, saving ms low, central, high, class, cost, risk, label)] priced at one final step. The classes: E0 = byte-exact by construction, E1 = exact against a split twin, T = a
    tolerance-class numerics change (owner waiver), K = needs a new kernel. Costs are engineering days (E); the label says what the saving rests on."""
    full, tail = passes(tokens)
    base = route_ms(start, tokens)
    rows = []
    new_start, new_tokens = undrop_start(start, tokens)
    p0 = base - route_ms(new_start, new_tokens) if (new_start, new_tokens) != (start, tokens) else 0.0
    rows.append(('P0 undrop + drafter-window carry (tp4/exact-end, built)', p0 * 0.9, p0, p0 * 1.05, 'E0', '0 (built)', 'low: one card window', 'E: production fit minus one chunk pass'))
    rows.append(('PF2-H device-side GDN slot hand-off (built here)', handoff_saving_ms('low'), handoff_saving_ms('central'), handoff_saving_ms('high'),
                 'E0', '3 (built)', 'low', 'E: profile split scaled to production'))
    if tail and bucket_for(tail) == 256:
        a256, b256 = PRODUCTION_TAIL[256]
        a512, b512 = PRODUCTION_TAIL[512]
        c = (start + full * CHUNK) / 1000.0
        gain = (a256 + b256 * c) - (a512 + b512 * c)
        rows.append(('tail bucket 256 -> 512 above 16k of context', gain * 0.7, gain, gain * 1.1, 'E1', '1', 'low', 'M: the 256 bucket pairs its two q chunks per head on 6 cores'))
    else:
        rows.append(('tail bucket 256 -> 512 above 16k of context', 0.0, 0.0, 0.0, 'E1', '1', 'low', 'M: only a 129-256 token tail pays it'))
    rows.append(('QWEN_GDN_PREFILL_MMRS=1 (fused out-projection + reduce-scatter)', 0.0, 12.0 * (full + (1 if tail else 0)), 25.0 * (full + (1 if tail else 0)), 'unproven (fp32 sum order)', '1 (flag exists)',
                 'medium: deadlocked on 1x2, TP4 unproven', 'E: PF1 gdn_out 15.0 + rs_gdn 25.4 ms a chunk, overlap'))
    split = split_k_saving_ms(start, tokens)
    rows.append(('split-K SDPA over the idle cores, tolerance class', 0.0, split, 1.4 * split, 'T + K', '15-25', 'high: new kernel + numerics waiver', 'E: per pass, whole slices of min(%d, 130 // busy cores); none for a 2,048-row pass' % SPLIT_K_CAP))
    traced = traced_tail_saving_ms(start, tokens)
    rows.append(('traced masked tails (valid_len as a device scalar in the GDN ops)', 0.0, traced, 1.25 * traced, 'E0 + K', '10-15', 'high: trace region, GDN op arguments',
                 'E: the eager floor (~320 ms) minus the device work of a small bucket'))
    return rows


def format_chunk_table(oneq=True):
    names = [row[0] for row in NON_SDPA_MS] + ['SDPA']
    lines = ['| context | ' + ' | '.join(names) + ' | kernel ms | wall ms | SDPA share |', '|---|' + '---:|' * (len(names) + 3)]
    for context, groups, kernel, wall, share in chunk_table(oneq=oneq):
        lines.append('| %s | ' % format(context, ',') + ' | '.join('%.1f' % groups[name] for name in names) + ' | %.1f | %.1f | %.0f%% |' % (kernel, wall, 100 * share))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('chunk')
    step = sub.add_parser('step')
    step.add_argument('start', type=int)
    step.add_argument('tokens', type=int)
    levers = sub.add_parser('levers')
    levers.add_argument('start', type=int)
    levers.add_argument('tokens', type=int)
    arguments = parser.parse_args(argv)
    if arguments.command == 'chunk':
        print('ship-4 stack (oneq attention):')
        print(format_chunk_table(True))
        print('\nPF1 as profiled (paired attention):')
        print(format_chunk_table(False))
        return 0
    start = arguments.start - arguments.start % CHUNK
    if arguments.command == 'step':
        info = step_breakdown(start, arguments.tokens)
        print('step at %d, %d tokens: %.0f ms over %d passes; SDPA %.0f ms (%.0f%%), other device %.0f ms, host/hand-off/other %.0f ms'
              % (start, arguments.tokens, info['total_ms'], info['passes'], info['sdpa_ms'], 100 * info['sdpa_ms'] / info['total_ms'], info['other_device_ms'], info['host_and_handoff_ms']))
        return 0
    base = route_ms(start, arguments.tokens)
    print('step at %d, %d tokens: %.0f ms' % (start, arguments.tokens, base))
    for name, low, central, high, klass, cost, risk, label in lever_table(start, arguments.tokens):
        print('%-72s %6.0f %6.0f %6.0f ms (%4.1f%%)  %-12s cost %s days  %s  [%s]' % (name, low, central, high, 100 * central / base, klass, cost, risk, label))
    return 0


if __name__ == '__main__':
    sys.exit(main())
