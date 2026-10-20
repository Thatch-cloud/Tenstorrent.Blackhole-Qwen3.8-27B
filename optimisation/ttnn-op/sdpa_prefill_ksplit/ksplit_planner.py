"""CPU model of the [QWEN-SDPA-PF] K-split (flag 0x10): the slice count, the slice-major core assignment, the K block ranges, the hand-off topology, the time model.
No ttnn, no torch (stdlib only; it reuses oneq_planner for the one-q-chunk-per-core layout it multiplies).

WHAT THE LEVER DOES (docs/tp4-prefill3-ksplit.md, docs/tp4-prefill-2.md section 7). A pass of R rows at TP4 (6 local Q heads, 1 KV head, q/k chunk 128) has
`total = 6 * R / 128` q chunks. The oneq call (flag 0x8) puts each on its own core: `total` busy cores, each streaming its q chunk's K/V blocks one after another at
13.51 us a block. With flag 0x10 the factory gives each q chunk S consecutive SLICES, S = min(4, cores // total), and slice s streams the K blocks
[s * B // E, (s + 1) * B // E) of the chunk's B = C + q + 1 blocks (E = effective_slices(B, S) of them are active). The slices run the pinned compute kernel's RING path:
slice 0 with ring_iter = 0, every later slice with ring_iter = 1 and the previous slice's (normalised output, log-sum-exp) in cb_prev_out / cb_lse_in, so the fold is
slice 0, then 1, ... in order, one pinned ring merge per hop (out = prev - sigmoid(cur_lse - prev_lse) * (prev - cur); lse = prev_lse - logsigmoid(prev_lse - cur_lse)).
The last active slice (the one that carries the diagonal block) writes the output.

THE DEVICE-SIDE ARITHMETIC (the three kernels and this module carry the SAME integer functions; test_ksplit_sources holds the C++ copies byte for byte):
  blocks            B = ceil(min((chunk_start_t + q_chunk * Sq + Sq), Skt) / Sk)   (the K chunks of the q chunk: C + q + 1)
  effective_slices  E = max(1, min(S, B // 4))      a slice keeps at least 4 blocks: below that the hop costs more than it saves
  slice_begin       s * B // E
  active            s < E
This is runtime on purpose: the chunk start is a device tensor (the flexible chunked path), so the host cannot know B.

CORE ASSIGNMENT (slice-major). Cores [s * total, (s + 1) * total) are slice s; inside a slice core u has the flat position u of the oneq split (the kernels' zigzag
decompose gives (head, q chunk)). So a chain group is the oneq group (6 heads of one q chunk) of one slice: byte-identical K/V sub-streams, S times as many chains of the same length.
The hand-off partner of core i is core i + total (the next slice, same head and q chunk).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sdpa_prefill_oneq'))

import oneq_planner as oneq  # noqa: E402

FLAG_KSPLIT = 0x10
FLAG_KS_HANG = 0x400                  # test only (QWEN_SDPA_PF_TEST=1): the receiving slice withholds the hand-off credit
KS_TEST_FLAGS = FLAG_KS_HANG
PF_TAG = oneq.PF_TAG
Q_CHUNK = 128
SLICE_CAP = 4                         # most slices of one q chunk: beyond it the hops cost what the split saves
MIN_BLOCKS_PER_SLICE = 4              # the device keeps at least this many blocks in a slice (effective_slices)
QUALIFIED_ROWS = (512, 1024)          # the rows the host lever engages: 2048 -> one slice (nothing to split); 256 and 128 are not chain-qualified (graft _QWEN_PF_ROWS)
OUT_TILES = 32                        # Sq_chunk_t (4) x vDHt (8) tiles of the normalised output
LSE_TILES = 4                         # Sq_chunk_t tiles, one column each
OUT_TILE_BYTES = 2048                 # a bf16 32x32 tile
LSE_TILE_BYTES = 4096                 # an fp32 32x32 tile (the stats CBs are fp32 under the lever)

FATAL_SLICES = '[QWEN-SDPA-PF] ksplit needs at least 2 slices: {} q chunks on {} cores'
FATAL_SCALE = '[QWEN-SDPA-PF] ksplit needs a scale exact in bf16'
FATAL_ENVELOPE = '[QWEN-SDPA-PF] ksplit outside its qualified envelope'
LOG_TEXT = '[QWEN-SDPA-PF] ksplit=1 slices={} q_chunks={} cores={} busy={} chains={} members={}'
ENVELOPE_TEXT = 'kv_chain outside its qualified envelope'

# The time model (estimates; the oneq constants are card-M K0a). A hop = the 80 KB partial over the NoC plus the pinned ring merge on the receiving slice.
T_STEP_US = oneq.T_STEP_US
LAYER_OVERHEAD_US = oneq.LAYER_OVERHEAD_US
HOP_US = 18.0                         # E: ~5 us for 80 KB at ~16 GB/s of one unicast write + ~12 us for 32 tiles of sub / mul / sub and the fp32 sigmoid and logsigmoid
HOP_US_PESSIMISTIC = 40.0


# ---- the device arithmetic ---------------------------------------------------------------------------------------------------------------------

def blocks_of(chunk_start, q, q_chunk=Q_CHUNK):
    """B: the K blocks of the q chunk `q` (per head) of a call at chunk_start tokens (the unclamped value: Skt never clamps a tail pass)."""
    return chunk_start // q_chunk + q + 1


def effective_slices(blocks, slices):
    """E (qwen_ks::effective_slices): how many slices of S take part in a q chunk of `blocks` blocks."""
    return max(1, min(slices, blocks // MIN_BLOCKS_PER_SLICE))


def slice_begin(blocks, slice_index, eff):
    """qwen_ks::slice_begin: the first block of a slice (uint32 arithmetic on the device; the product stays below 2^32)."""
    return (slice_index * blocks) // eff


def slice_range(blocks, slice_index, slices):
    """(begin, end) of a slice for a q chunk of `blocks` blocks, or None for an idle slice (slice >= E)."""
    eff = effective_slices(blocks, slices)
    if slice_index >= eff:
        return None
    return slice_begin(blocks, slice_index, eff), slice_begin(blocks, slice_index + 1, eff)


def ranges(blocks, slices):
    """[(begin, end) per active slice]; contiguous, disjoint, covering [0, blocks), the last one carries block blocks - 1 (the diagonal)."""
    eff = effective_slices(blocks, slices)
    return [slice_range(blocks, s, slices) for s in range(eff)]


# ---- the host side -----------------------------------------------------------------------------------------------------------------------------

def split_k_slices(rows, nqh=6, cores=130, q_chunk=Q_CHUNK, cap=SLICE_CAP):
    """S, the factory's slice count for a pass of `rows` rows: min(cap, cores // busy) with busy = nqh * rows / q_chunk (one batch). 1 means the lever does not engage."""
    if rows <= 0 or rows % q_chunk:
        return 1
    busy = nqh * (rows // q_chunk)
    return max(1, min(cap, cores // busy))


def engaged_rows(rows, nqh=6, cores=130):
    """Whether the host lever sets flag 0x10 for a call of `rows` rows: the chain-qualified rows that the split can take twice or more."""
    return rows in QUALIFIED_ROWS and split_k_slices(rows, nqh, cores) >= 2


def plan(nqh=6, nkh=1, rows=1024, q_chunk=Q_CHUNK, grid=(13, 10), slices=None, noc_order=False):
    """The factory's decisions for one flag-0x10 call.

    slices: None = the factory's own S; an int = a request (the card harness asks to see the refusals: S above the cap, a pass that needs more cores than the grid has).
    -> dict(num_cores, total, slices, busy, refusal (None or the text), per-core records, chains {key: [cores in chain order]}, edges [(sender core, receiver core)], ...)"""
    chunks = rows // q_chunk
    num_cores = grid[0] * grid[1]
    base = oneq.plan(nqh, nkh, rows, q_chunk, grid, oneq=True, noc_order=noc_order)
    total = base['total']
    chosen = split_k_slices(rows, nqh, num_cores, q_chunk) if slices is None else slices
    refusal = None
    if chunks % 2:
        refusal = ENVELOPE_TEXT
    elif base['refusal']:
        refusal = base['refusal']
    elif slices is not None and slices > SLICE_CAP:
        refusal = FATAL_SLICES.format(total, num_cores) + ' (S=%d is above the cap %d)' % (slices, SLICE_CAP)
    elif chosen < 2 or total * chosen > num_cores:
        refusal = FATAL_SLICES.format(total, num_cores)
    out = dict(num_cores=num_cores, total=total, q_num_chunks=chunks, slices=chosen, busy=total * chosen if not refusal else 0, refusal=refusal,
               nqh=nqh, nkh=nkh, rows=rows, q_chunk=q_chunk, grid=grid, cores=[], chains={}, edges=[])
    if refusal:
        return out
    per_kv = nqh // nkh
    groups = {}
    for core in range(num_cores):
        if core >= total * chosen:
            out['cores'].append(dict(core=core, busy=False, slice=0, unit=None, prev=None, next=None))
            continue
        slice_index, position = divmod(core, total)
        _nb, nq, q = base['units'][position][0]
        record = dict(core=core, busy=True, slice=slice_index, unit=(0, nq, q), prev=core - total if slice_index > 0 else None,
                      next=core + total if slice_index + 1 < chosen else None, range_start=position, range_count=1)
        out['cores'].append(record)
        groups.setdefault((0, nq // per_kv, q, slice_index), []).append(core)
        if record['next'] is not None:
            out['edges'].append((core, record['next']))
    for key in sorted(groups):
        members = groups[key]
        if len(members) >= 2:
            out['chains'][key] = members
    out['chain_count'] = len(out['chains'])
    out['member_count'] = sum(len(members) for members in out['chains'].values())
    return out


def core_blocks(the_plan, chunk_start):
    """Blocks each core streams at a chunk start (0 for an idle core or an idle slice)."""
    result = []
    for record in the_plan['cores']:
        if not record['busy']:
            result.append(0)
            continue
        _nb, _nq, q = record['unit']
        span = slice_range(blocks_of(chunk_start, q, the_plan['q_chunk']), record['slice'], the_plan['slices'])
        result.append(0 if span is None else span[1] - span[0])
    return result


def critical_blocks(the_plan, chunk_start):
    return max(core_blocks(the_plan, chunk_start) or [0])


def hops(the_plan, chunk_start):
    """The longest relay chain (merge hops) of any q chunk at this chunk start: E - 1 for the chunk with the most active slices."""
    best = 0
    for q in range(the_plan['q_num_chunks']):
        best = max(best, effective_slices(blocks_of(chunk_start, q, the_plan['q_chunk']), the_plan['slices']) - 1)
    return best


def layer_us(the_plan, chunk_start, step_us=T_STEP_US, overhead_us=LAYER_OVERHEAD_US, hop_us=HOP_US):
    """Modelled device time of one attention layer's SDPA call: the busiest slice, the layer overhead, and the relay tail of the chunk that has the most hops."""
    return step_us * critical_blocks(the_plan, chunk_start) + overhead_us + hop_us * hops(the_plan, chunk_start)


def oneq_layer_us(rows, chunk_start, nqh=6, grid=(13, 10), step_us=T_STEP_US, overhead_us=LAYER_OVERHEAD_US):
    """The same call at one q chunk per core (the baseline the lever is measured against): the heaviest q chunk's blocks."""
    base = oneq.plan(nqh, 1, rows, Q_CHUNK, grid, oneq=True)
    return oneq.layer_us(base, chunk_start, step_us, overhead_us)


def layer_ratio(rows, chunk_start, nqh=6, grid=(13, 10), hop_us=HOP_US):
    """ksplit over oneq, per layer (< 1 is a win); 1.0 when the lever does not take the call."""
    mine = plan(nqh, 1, rows, Q_CHUNK, grid)
    if mine['refusal']:
        return 1.0
    return layer_us(mine, chunk_start, hop_us=hop_us) / oneq_layer_us(rows, chunk_start, nqh, grid)


def handoff_bytes():
    """Bytes one hop moves: (normalised output, log-sum-exp)."""
    return OUT_TILES * OUT_TILE_BYTES, LSE_TILES * LSE_TILE_BYTES
