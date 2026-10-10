"""CPU model of the [QWEN-SDPA-PF] oneq work split: the host planner (factory PS0-PS5), the per-core block counts,
the SDPA time model and the TTFT estimate. No ttnn, no torch.

THE WORK SPLIT (sdpa_program_factory.cpp, pinned tt-metal 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9, lines 389-410 of the
served file; q_chunk_remapping.hpp for the kernels' side)
  The flat space is (batch, head, q chunk): total = B * NQH * q_num_chunks positions. Core i gets the contiguous range
  [start_i, start_i + count_i). Paired (causal, even q_num_chunks): the unit is a PAIR of positions, so a core gets 2
  consecutive positions and the kernels' zigzag remap (pos 2k -> q chunk k, pos 2k+1 -> q chunk N-1-k, per head)
  turns them into one light and one heavy q chunk. Not paired: one position per core while total <= cores.
  The chain (F4) then groups the cores whose WHOLE unit list is the same (batch, kv_head, q chunk) sequence: they read
  byte-identical K/V streams.

  TP4 (6 local Q heads, 1 KV head), 2048 rows, q/k chunk 128 -> 96 q chunks:
    paired  48 busy cores x 2 chunks, 8 chains of 6   (the profile's 'kv_chain=1 chains=8 members=48')
    oneq    96 busy cores x 1 chunk,  16 chains of 6

THE TIME MODEL (prefill profile run 38051000905, analysis in the programme's notes): a core spends T_STEP = 13.51 us per
128x128 block (card-M K0a, the compute floor) plus ~90 us per layer; the critical path is the busiest core, and a q chunk
q at context C chunks (C = chunk_start / 128) has C + q + 1 blocks. The model matches the measured paired SDPA at chunks
1, 32 and 63 within 0.3 percent (750.4 / 14177.3 / 27604.4 us measured, 752.0 / 14153.9 / 27555.8 modelled).
"""

import itertools

TILE = 32
T_STEP_US = 13.51              # per 128x128 block, the compute floor (K0a)
LAYER_OVERHEAD_US = 90.0       # per attention layer and chunk: launch, drain, the diagonal tail
ATTENTION_LAYERS = 16          # Qwen 3.x: 16 full-attention layers of 64
WALL_PER_DEVICE = 1.1286       # unprofiled wall per chunk = 5.5 ms + 1.1286 x kernel ms (profile run 38051000905)
CHUNK_TOKENS = 2048            # the model's prefill chunk (rows of one SDPA call)
T_STEP_PESSIMISTIC_US = 16.5   # the chain step Q2 measured at the 16-chain TP2 geometry (card M, K64g, Q in L1)

PF_TAG = 0x5EFA0000
FLAG_CHAIN, FLAG_INJ_BATCH, FLAG_NOC_ORDER, FLAG_ONEQ = 0x1, 0x2, 0x4, 0x8
FATAL_TEXT = '[QWEN-SDPA-PF] oneq needs one q chunk per core: {} q chunks on {} cores'
NOC_GRID = (17, 12)


def zigzag(pos, n):
    """q_chunk_remapping.hpp linear_to_zigzag, inside one head."""
    return pos // 2 if pos % 2 == 0 else n - 1 - pos // 2


def decompose(index, q_num_chunks, nqh, zigzag_on=True):
    """(nb, nq, q) of a flat position (decompose_global_q_index)."""
    if zigzag_on:
        head, pos = divmod(index, q_num_chunks)
        index = head * q_num_chunks + zigzag(pos, q_num_chunks)
    return index // (nqh * q_num_chunks), (index // q_num_chunks) % nqh, index % q_num_chunks


def one_q_word(word):
    """Whether a program word asks for oneq: the PF tag, the chain bit and the oneq bit (the factory's early decode)."""
    return (word & 0xFFFF0000) == PF_TAG and (word & (FLAG_CHAIN | FLAG_ONEQ)) == (FLAG_CHAIN | FLAG_ONEQ)


def split(num_cores, total, pair):
    """(base, extra_cores, extra) of the factory's work split."""
    if pair:
        pairs = total // 2
        return (pairs // num_cores) * 2, pairs % num_cores, 2
    return total // num_cores, total % num_cores, 1


def ranges(num_cores, total, pair):
    """[(global_q_start, global_q_count)] per linear core, with the reader runtime-arg loop's clamp."""
    base, extra_cores, extra = split(num_cores, total, pair)
    out = []
    for i in range(num_cores):
        start = i * base + min(i, extra_cores) * extra
        count = base + (extra if i < extra_cores else 0)
        if start >= total:
            start, count = total, 0
        elif start + count > total:
            count = total - start
        out.append((start, count))
    return out


def physical(logical, grid):
    """The planner's stand-in for device->worker_core_from_logical_core (the host-exec harness uses the same map)."""
    return (logical % grid[0] + 1, logical // grid[0] + 2)


def order_cost(phys, order, noc_grid=NOC_GRID):
    """F4's cost: direction-agnostic torus Manhattan distance over consecutive members, uint32 arithmetic."""
    total = 0
    for a, b in zip(order, order[1:]):
        (ax, ay), (bx, by) = phys[a], phys[b]
        dx, dy = abs(ax - bx), abs(ay - by)
        total += min(dx, (noc_grid[0] - dx) % (1 << 32)) + min(dy, (noc_grid[1] - dy) % (1 << 32))
    return total


def plan(nqh=6, nkh=1, rows=2048, q_chunk=128, grid=(13, 10), batch=1, oneq=False, noc_order=False, causal=True):
    """The factory's decisions for one call.

    -> dict(num_cores, total, pair, oneq, ranges, units (per core [(nb, nq, q)]), busy, max_per_core, q_buffer_factor,
            chains {key: [cores in chain order]}, chain_count, member_count, refusal (None or the TT_FATAL text))."""
    q_num_chunks = rows // q_chunk
    num_cores = grid[0] * grid[1]
    total = batch * nqh * q_num_chunks
    pair = causal and q_num_chunks % 2 == 0 and not oneq
    spans = ranges(num_cores, total, pair)
    base, extra_cores, extra = split(num_cores, total, pair)
    max_per_core = base + (extra if extra_cores > 0 else 0)
    units = [[decompose(start + u, q_num_chunks, nqh, causal) for u in range(count)] for start, count in spans]
    refusal = None
    if oneq and max_per_core != 1:
        refusal = FATAL_TEXT.format(total, num_cores)
    q_per_kv = nqh // nkh
    groups = {}
    for core, list_ in enumerate(units):
        if not list_:
            continue
        key = tuple(value for nb, nq, q in list_ for value in (nb, nq // q_per_kv, q))
        groups.setdefault(key, []).append(core)
    chains = {}
    for key in sorted(groups):
        members = groups[key]
        if len(members) < 2:
            continue
        order = list(range(len(members)))
        if noc_order:
            phys = [physical(core, grid) for core in members]
            best, best_cost = order, order_cost(phys, order)
            for probe in itertools.permutations(range(len(members))):
                cost = order_cost(phys, probe)
                if cost < best_cost:
                    best, best_cost = list(probe), cost
            order = best
        chains[key] = [members[index] for index in order]
    return dict(num_cores=num_cores, total=total, q_num_chunks=q_num_chunks, pair=pair, oneq=oneq, ranges=spans,
                units=units, busy=sum(1 for _s, count in spans if count), max_per_core=max_per_core,
                q_buffer_factor=2 if max_per_core > 1 else 1, chains=chains, chain_count=len(chains),
                member_count=sum(len(members) for members in chains.values()), refusal=refusal,
                nqh=nqh, nkh=nkh, rows=rows, q_chunk=q_chunk, grid=grid)


def blocks(unit_q, context_tokens, q_chunk=128):
    """128x128 blocks of the q chunk `unit_q` of a call at chunk_start = context_tokens: the k chunks 0..C+q."""
    return context_tokens // q_chunk + unit_q + 1


def core_blocks(the_plan, context_tokens):
    """Blocks per core (sum over its units) at a context."""
    q_chunk = the_plan['q_chunk']
    return [sum(blocks(q, context_tokens, q_chunk) for _nb, _nq, q in units) for units in the_plan['units']]


def critical_blocks(the_plan, context_tokens):
    return max(core_blocks(the_plan, context_tokens) or [0])


def layer_us(the_plan, context_tokens, step_us=T_STEP_US, overhead_us=LAYER_OVERHEAD_US):
    """Modelled device time of one attention layer's SDPA call at a context."""
    return step_us * critical_blocks(the_plan, context_tokens) + overhead_us


def chunk_starts(prompt_tokens, chunk=CHUNK_TOKENS):
    """The model's full-chunk starts of a prompt (a tail chunk of fewer than `chunk` rows is not modelled)."""
    return [c * chunk for c in range(prompt_tokens // chunk)]


def prompt_device_s(the_plan, prompt_tokens, step_us=T_STEP_US, overhead_us=LAYER_OVERHEAD_US, layers=ATTENTION_LAYERS):
    """Modelled SDPA device seconds of one prompt (all chunks, all attention layers)."""
    return sum(layers * layer_us(the_plan, start, step_us, overhead_us) for start in chunk_starts(prompt_tokens)) / 1e6


def ttft_saving_s(prompt_tokens, paired=None, oneq=None, step_oneq_us=T_STEP_US, wall=WALL_PER_DEVICE):
    """Estimated solo-TTFT saving in wall seconds: (paired - oneq) SDPA device seconds x the wall factor."""
    paired = paired or plan()
    oneq = oneq or plan(oneq=True)
    saved = prompt_device_s(paired, prompt_tokens) - prompt_device_s(oneq, prompt_tokens, step_oneq_us)
    return saved * wall


def estimate_table(prompts=(32768, 131072, 253952), **kwargs):
    """[(prompt tokens, chunks, saving at the floor step, saving at the pessimistic step)] for the report."""
    rows = []
    for tokens in prompts:
        rows.append((tokens, tokens // CHUNK_TOKENS, ttft_saving_s(tokens, step_oneq_us=T_STEP_US, **kwargs),
                     ttft_saving_s(tokens, step_oneq_us=T_STEP_PESSIMISTIC_US, **kwargs)))
    return rows
