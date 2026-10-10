"""QWEN_FAST_OCTO_DRAFT (default off, gate only): ONE eight-seat, eight-row draft pass for the octo rounds, in place of the two four-seat quad passes.

THE COST IT REMOVES. An octo round (QWEN_FAST_OCTO, serving_octo) verifies eight seats x eight rows in one 64-row pass but still drafts with the two
four-seat quad passes of the M3 blocks (QWEN_FAST_QUAD_DRAFT_BLOCKS=2): two 64-row traces, two launches, two readbacks, each drafting 16 rows per seat of
which the host keeps seven (GreedySession.propose(max_rows=8)). Measured on cards (O3, the Lever N octo twin, 8 live): the early draft is 30 ms at the median
(43 ms at the 75th percentile) of a 140 ms round. The pass is op-latency bound, not weight bound, so one 64-row pass over eight seats costs well under two.

WHAT RUNS. The quad pass (quad_draft.py / quad_draft_tp.py) is the template and is NOT edited. This module is its eight-seat, eight-row twin, a new pass object
(`OctoPass`, the `quad=` argument execute_proposal and the two draft branches already take) over the same drafter layers (the T16 ones: the layer parameters, the
cached K/V banks, the fused conv, the shared head are all block independent):
  - eight users x EIGHT rows (seed + seven mask rows) in a 64-row block: user u at rows 8u..8u+7; every row-local op, the nine matmul programs at per_core_M = 2,
    the K/V projection, the rotary, the fused conv (the sha-pinned quad_conv_io.cpp reads a seam word per tile row, a bit per row: seams at 8, 16 and 24 are the same
    kernel), the collectives, the learned norm and selector projection and the 32-row-half LM head are the quad's, call for call;
  - the K/V: a 24-piece plan, each user's 2048 cached rows, its 8 live rows [8u, 8u + 8) of the block and 24 pad rows (any finite rows: the mask hides them) from its
    half, so every user's key segment is 2080 keys with its live keys at 2048..2055, exactly the single-user T8 layout;
  - the draft SDPA: the quad's four-way fold generalised to eight users. Each 32-row half holds four users; the half is rotated by 0, 8, 16 and 24 rows so that
    user k of the half sits at rows 0-7, exactly where the single-user T8 trace has its eight query rows, the four rotations are grouped on the head axis
    (folded head 8*group*h + group*u + j, u = 4p + k), and the SDPA runs at 8 x the query heads over 8 x the KV heads (64 / 16 at four cards: the program shape
    the pair's quad proved on card B), reading the SAME single-user mask (1, 1, 32, 2080) for every head: the T8 one (rows 0-7 see the 2048 cached keys and
    their 8 live keys, every other row exactly one key). Every (query head, KV head) work unit is then user u's single-user T8 unit - the same 8 query rows at the
    same tile rows, the same keys in the same order, the same mask tiles, the same 65 key chunks - so the fold is a re-indexing, never another sum;
  - the readback: the quad's (18 reads at four cards), each 32-row half merged as the quad merges it, the halves stitched [half 0 | dummy row | half 1] and split
    by dflash_packed_proposal.split_selection(users=8, block_rows=8, block_width=64): eight users' (features, candidates, scores), selected seven tokens each.

WHAT IT IS NOT. It is NOT the T16 draft cut to seven rows: a T8 block has eight live keys where the T16 block has sixteen, and the drafter's attention is
bidirectional inside the block (dflash_batched_mask.batched_attention_mask: every live query sees every live key), so rows 1-7 of a T8 block are not rows 1-7 of a
T16 block. The drafter is checkpointed at 16 rows; the target verifies every proposal, so the text is the target's own greedy decode either way and the cost of a
different draft is TAU ONLY. That is the card question this lever asks (job OD1/OD2: committed tokens per round against the T16-cut control), and it is why the
proof of exactness here is "the fold re-indexes the single-user T8 computation bit for bit" (test_octo_draft) and not "equal to the quad's tokens".

WHEN IT RUNS. dflash_packed_proposal_coordinator.PackedProposalCoordinator._prepare_octo, before the quads, only for a round the packed step planned as an octo round (the
hook passes octo_draft=True only then) with all eight slots live and packable (four packable pairs). 6 and 7 live octo rounds keep the quads (cut to seven rows). A refusal
(refusal()), a capture short of DRAM, or a failure falls the round back to the quads (FALLBACK_LINE); GIVE_UP_FAILURES in a row disable it for the process (DISABLED_MARKER, which
the smoke rule fails on). NOT qualified: no card has run this module (serving_octo.UNQUALIFIED_ITEMS names the job).

THE FLAG, read at the attach and each round, never at import: QWEN_FAST_OCTO_DRAFT strict 0 or 1 (unset is 0); anything else raises ValueError naming it. It needs
QWEN_FAST_OCTO=live|alternate (serving_octo.octo_admission refuses it otherwise), the quad flags the octo block already requires, QWEN_FAST_PIPELINED_PROPOSALS=1 and
QWEN_FAST_PACKED_PROPOSAL=1 (the coordinator), and is refused beside the draft book / parked engines and the quad shadow audit (their traces and pairs are the quad's).
QWEN_FAST_OCTO_DRAFT_CONV 110|80|halves (default 110: the quad's QWEN_FAST_QUAD_CONV, same kernel).

Stdlib and torch only at call time (torch inside functions), importable on py 3.7 beside the quad twin.
"""

import os
import time
from types import SimpleNamespace

import quad_draft as _pinned
import quad_draft_tp as _quad
import tp_shapes
from quad_draft import log_line

FLAG = 'QWEN_FAST_OCTO_DRAFT'
CONV_FLAG = 'QWEN_FAST_OCTO_DRAFT_CONV'
# The drafter layers this pass reuses are the T16 ones (their prepared parameters carry block_rows 16): the pass's own block is eight rows.
PREPARED_BLOCK = _pinned.BLOCK
CONTEXT, SPAN, HEAD_DIM, HIDDEN = _pinned.CONTEXT, _pinned.SPAN, _pinned.HEAD_DIM, _pinned.HIDDEN
USERS, BLOCK, ROWS = 8, 8, 64
HALF_USERS, HALVES = 4, 2                   # users per 32-row tile half, halves per block
PAD = SPAN - CONTEXT - BLOCK                # 24: pad keys per user segment (masked)
PROPOSALS = BLOCK - 1                       # seven proposals per seat
SLOTS = tuple(range(USERS))
PAIRS = tuple(zip(SLOTS[::2], SLOTS[1::2]))  # the pool-slot pairs whose live banks the pass binds
GIVE_UP_FAILURES = _pinned.GIVE_UP_FAILURES
# A fresh capture's DRAM per chip, estimated as a quad's (450 MiB, measured 375 MB) plus the larger K/V assembly the 24-piece plan retains (two 8.5 MB tensors a layer
# over five layers); the octo_draft_built ledger point measures the real figure.
OCTO_CAPTURE_BYTES_EST = 520 * 2 ** 20

MARKER = '[PINDIAG] octo draft engaged'
DISABLED_MARKER = '[PINDIAG] octo draft disabled'
ROUND_LINE = '[OCTO-DRAFT] round={round} built={built} ms={ms}'
FALLBACK_LINE = '[OCTO-DRAFT] fallback round={round} reason={reason}'
ADMITTED_LINE = '[OCTO-DRAFT] admitted rows=64 users=8 block=8 proposals=7 (gate only)'
REFUSED_LINE = '[OCTO-DRAFT] refused'
_NOTED = []


def enabled(environ=None):
    """QWEN_FAST_OCTO_DRAFT, strictly: unset or '0' is off, '1' is on, anything else (an empty value included) is a configuration error naming the flag."""
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    return value == '1'


def conv_mode(environ=None):
    """QWEN_FAST_OCTO_DRAFT_CONV: '110' (unset, the default), '80' or 'halves' (quad_draft.CONV_MODES)."""
    value = (os.environ if environ is None else environ).get(CONV_FLAG) or '110'
    if value not in _pinned.CONV_MODES:
        raise ValueError('%s must be one of %s, got %r' % (CONV_FLAG, '|'.join(_pinned.CONV_MODES), value))
    return value


def heads():
    """(query heads, KV heads, GQA group, octo query heads, octo KV heads) per chip at the width this process serves at: (8, 2, 4, 64, 16) at four cards."""
    query, key = tp_shapes.active().draft_heads, tp_shapes.active().draft_kv_heads
    return query, key, query // key, USERS * query, USERS * key


def note(slots, conv, *, log=None):
    """MARKER once per process, at the first octo bucket that captured and replayed. Returns whether it logged."""
    if _NOTED:
        return False
    _NOTED.append(tuple(slots))
    (log or log_line)('%s slots=[%s] heads=%d/%d rows=%d block=%d conv=%s' % (
        MARKER, ','.join(str(slot) for slot in slots), heads()[3], heads()[4], ROWS, BLOCK, conv))
    return True


def capture_bytes():
    """What a fresh octo capture's headroom (and its ledger reading) asks."""
    return OCTO_CAPTURE_BYTES_EST


# ---------------------------------------------------------------------------------------------
# The K/V plan and the single-user T8 mask.
# ---------------------------------------------------------------------------------------------

def require_octo(contexts, block_rows):
    """The prepared layers are the T16 ones (block_rows 16 in their parameters); the pass packs eight users at the full 2048-row history."""
    if type(block_rows) is not int or block_rows != PREPARED_BLOCK or tuple(contexts) != (CONTEXT,) * USERS:
        raise ValueError('%s packs eight %d-row T8 users at the %d-row history over the prepared T%d layers only, not contexts %r at %r rows'
                         % (FLAG, BLOCK, CONTEXT, PREPARED_BLOCK, tuple(contexts), block_rows))


def octo_key_value_plan():
    """The 24 pieces of the key axis, in order: for each user u (half p = u // 4) its cached bank (2048 rows), its live rows [8u, 8u + 8) of the 64-row block, then 24
    pad rows from rows [32p, 32p + 24) of the block. The pad is masked (every visible key of a row is a cached or live one), so any finite rows do; the half's own are
    the ones nearest the user."""
    plan = []
    for user in range(USERS):
        half = user // HALF_USERS
        plan.append(dict(kind='cached', user=user, rows=CONTEXT))
        plan.append(dict(kind='live', user=user, rows=BLOCK, source=slice(BLOCK * user, BLOCK * (user + 1))))
        plan.append(dict(kind='pad', user=user, rows=PAD, source=slice(32 * half, 32 * half + PAD)))
    if sum(piece['rows'] for piece in plan) != USERS * SPAN:
        raise AssertionError('The octo plan must cover 16640 keys')
    return plan


def key_value_plan(contexts, block_rows):
    """draft_attention_branch's (plan, spans, key_rows) for the octo pass: octo_key_value_plan and one 2080-key span per user, its rows the user's eight of the block."""
    require_octo(contexts, block_rows)
    spans = [dict(user=user, context=CONTEXT, offset=SPAN * user, span=SPAN, rows=slice(BLOCK * user, BLOCK * (user + 1)),
                  keys=slice(SPAN * user, SPAN * (user + 1))) for user in range(USERS)]
    return octo_key_value_plan(), spans, USERS * SPAN


def octo_host_mask():
    """The single-user T8 mask (1, 1, 32, 2080): rows 0-7 see the 2048 cached keys and the 8 live ones, every other row exactly one key; validated as the quad's is
    (dflash_t16_native_attention.validate_mask holds for it: rows 8-15 and 16-31 each see one key). The same for every head of the folded SDPA."""
    from dflash_batched_mask import batched_attention_mask
    from dflash_t16_native_attention import validate_mask

    host_mask = batched_attention_mask([CONTEXT], BLOCK)
    validate_mask(host_mask)
    return host_mask


def octo_users(devices, context=CONTEXT):
    return [dict(position=device.position, history_rows=context) for device in devices]


def octo_rope(users):
    """The pass's rope.q and live_k: each 32-row half's packed_rope_tables over its four users at EIGHT rows (query: 4 x 8 rows; key: four 2080-key segments, from which the live
    rows are cut by live_key_rope_from), the halves concatenated on the row axis. Returns (query tables, live key tables), each (cos, sin) of (1, 1, 64, 128)."""
    import torch

    from dflash_batched_mask import live_key_rope_from, packed_rope_tables

    users = list(users)
    if len(users) != USERS:
        raise ValueError('Eight users required')
    query, live = [], []
    for half in range(HALVES):
        members = users[HALF_USERS * half:HALF_USERS * (half + 1)]
        tables = packed_rope_tables(members, BLOCK)
        query.append(tables['q'])
        live.append(live_key_rope_from(tables['k'], members, BLOCK))
    joined = lambda parts: tuple(torch.cat([part[index] for part in parts], dim=2) for index in (0, 1))
    return joined(query), joined(live)


# ---------------------------------------------------------------------------------------------
# The draft SDPA: the quad's fold generalised to eight users (four per 32-row half).
# ---------------------------------------------------------------------------------------------

def validate_octo(operations, query, key, value, mask):
    query_heads, key_heads = heads()[:2]
    if (tuple(query.shape) != (1, query_heads, ROWS, HEAD_DIM) or tuple(key.shape) != (1, key_heads, USERS * SPAN, HEAD_DIM)
            or tuple(value.shape) != tuple(key.shape) or tuple(mask.shape) != (1, 1, 32, SPAN)):
        raise ValueError('The octo fold takes the (1, %d, 64, 128) query, the eight-segment (1, %d, 16640, 128) keys and values and the single-user (1, 1, 32, 2080) mask'
                         % (query_heads, key_heads))
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 octo draft attention operands required')
    if any(tensor.layout != operations.TILE_LAYOUT or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG
           for tensor in (query, key, value, mask)):
        raise ValueError('Interleaved tiled DRAM operands required')


def octo_fold_query(operations, query, retain):
    """(1, q, 64, 128) -> (1, 8q, 32, 128). Each tile-aligned half holds four users' eight rows; it is rotated by 0, 8, 16 and 24 rows (the quarters in a cyclic order) so that
    user k of the half sits at rows 0-7, as in its single-user trace, then read as (kv, group, 32, 128) and the four rotations are concatenated on dim 1, and the two halves
    after them: folded head (8 * group) * h + group * u + j with u = 4p + k. The GQA group then maps it to KV head 8h + u: user u's segment of head h."""
    query_heads, key_heads, group = heads()[:3]
    memory = operations.DRAM_MEMORY_CONFIG
    halves = []
    for half in range(HALVES):
        rows = retain(operations.slice(query, (0, 0, 32 * half, 0), (1, query_heads, 32 * (half + 1), HEAD_DIM)))
        quarters = [retain(operations.slice(rows, (0, 0, BLOCK * k, 0), (1, query_heads, BLOCK * (k + 1), HEAD_DIM))) for k in range(HALF_USERS)]
        rotated = [rows]
        for k in range(1, HALF_USERS):
            rotated.append(retain(operations.concat(quarters[k:] + quarters[:k], dim=2, memory_config=memory)))
        groups = [retain(operations.reshape(value, (key_heads, group, 32, HEAD_DIM))) for value in rotated]
        halves.append(retain(operations.concat(groups, dim=1, memory_config=memory)))
    combined = retain(operations.concat(halves, dim=1, memory_config=memory))
    return retain(operations.reshape(combined, (1, heads()[3], 32, HEAD_DIM)))


def octo_fold_keys(operations, tensor, retain):
    """(1, kv, 16640, 128) -> (1, 8 * kv, 2080, 128), a view: KV head 8h + u is user u's segment of head h."""
    return retain(operations.reshape(tensor, (1, heads()[4], SPAN, HEAD_DIM)))


def octo_unfold_output(operations, output, retain):
    """(1, 8q, 32, 128) -> (1, q, 64, 128): user u's rows 0-7 of its group of folded heads, read back to its eight rows of the block (concatenated per half, then the halves)."""
    query_heads, key_heads, group = heads()[:3]
    memory = operations.DRAM_MEMORY_CONFIG
    grouped = retain(operations.reshape(output, (key_heads, heads()[3] // key_heads, 32, HEAD_DIM)))
    halves = []
    for half in range(HALVES):
        users = []
        for k in range(HALF_USERS):
            user = HALF_USERS * half + k
            part = retain(operations.slice(grouped, (0, group * user, 0, 0), (key_heads, group * (user + 1), BLOCK, HEAD_DIM)))
            users.append(retain(operations.reshape(part, (1, query_heads, BLOCK, HEAD_DIM))))
        halves.append(retain(operations.concat(users, dim=2, memory_config=memory)))
    return retain(operations.concat(halves, dim=2, memory_config=memory))


def fold_attention(operations, query, key, value, mask, retain, *, mask_validated=False):
    """The octo pass's draft attention: every (Q head, KV head) unit is that user's single-user T8 unit (see the module docstring), only the compile-time head counts
    differ (64 / 16 at four cards)."""
    from pair_row_exact import folded_sdpa

    if mask_validated is not True:
        raise ValueError('Validate the single-user host mask before upload and replay')
    validate_octo(operations, query, key, value, mask)
    folded = octo_fold_query(operations, query, retain)
    keys = octo_fold_keys(operations, key, retain)
    values = octo_fold_keys(operations, value, retain)
    attention = retain(folded_sdpa(operations, folded, keys, values, mask))
    return octo_unfold_output(operations, attention, retain)


def octo_head_map():
    """{folded query head: (h, u, j, kv head)}: the fold's head arithmetic at this width."""
    query_heads, key_heads, group, octo_query, _ = heads()
    span = octo_query // key_heads
    return {span * h + group * u + j: (h, u, j, USERS * h + u)
            for h in range(key_heads) for u in range(USERS) for j in range(group)}


class OctoPass(_quad.QuadPass):
    """What execute_proposal and the two branches take from the pass (their `quad` keyword) at eight users x eight rows: the quad's 64 rows, 64-row helpers, conv and head,
    with this module's K/V plan and SDPA, and `users` / `block` for the two places that counted four users of 16 rows (execute_proposal's guard and its MLP seams)."""

    users = USERS
    block = BLOCK
    mask_rows = SPAN
    key_value_plan = staticmethod(key_value_plan)

    def __init__(self, conv='110'):
        super().__init__('fold', conv)

    def attention(self, operations, query, key, value, mask, retain, *, mask_validated=False):
        return fold_attention(operations, query, key, value, mask, retain, mask_validated=mask_validated)


# ---------------------------------------------------------------------------------------------
# The readback.
# ---------------------------------------------------------------------------------------------

def read_octo_outputs(device, outputs, reference=False):
    """read_quad_outputs at eight users x eight rows: the same reads (chunks x chips x values and indices, plus the replicated features), each 32-row half merged by the
    four-card merge_chunk_candidates(block_rows=32), stitched [half 0 | a dummy row | half 1] (the dummy is merged index 31 - block row 32, user 4's anchor - which no
    user slice reads) and split with split_selection(users=8, block_rows=8, block_width=64): eight users' (features, candidates, scores). The tp4/round-host READ lever
    (QWEN_FAST_TP4_ROUND_HOST_READ) applies as it does to the quad; `reference=True` is always today's."""
    import torch

    from dflash_packed_proposal import report_rejected_outputs, split_selection
    from draft_shared_head_tp import merge_chunk_candidates
    import round_host
    import verify_prestage

    fast = not reference and round_host.read_enabled()
    if fast:
        merge_chunk_candidates = round_host.merge_chunk_candidates
    operations = device.operations
    chips = tp_shapes.chip_count()
    stamps = [time.perf_counter()] if verify_prestage.hostgap_log_enabled() else None
    halves = ([], [])
    for chunk in outputs.chunks:
        values = operations.get_device_tensors(chunk['values'])
        indices = operations.get_device_tensors(chunk['indices'])
        if len(values) != chips or len(indices) != chips:
            raise AssertionError('%s learned head shards required' % tp_shapes.all_chips())
        for chip in range(chips):
            host_values = operations.to_torch(values[chip]).float().reshape(ROWS, 16)
            host_indices = operations.to_torch(indices[chip]).long().reshape(ROWS, 16)
            for half in range(HALVES):
                rows = slice(32 * half, 32 * (half + 1))
                halves[half].append(dict(chip=chip, start=chunk['start'], stop=chunk['stop'], values=host_values[rows], indices=host_indices[rows]))

    def merged(host_chunks):
        try:
            return merge_chunk_candidates(host_chunks, block_rows=32)
        except ValueError as failure:
            report_rejected_outputs(device, outputs, host_chunks, failure)
            raise

    if stamps is not None:
        stamps.append(time.perf_counter())
    merged_halves = [merged(list(part)) for part in halves]
    if stamps is not None:
        stamps.append(time.perf_counter())
    candidates = torch.cat([merged_halves[0][0], torch.zeros_like(merged_halves[0][0][:, :1]), merged_halves[1][0]], dim=1)
    unary = torch.cat([merged_halves[0][1], torch.zeros_like(merged_halves[0][1][:, :1]), merged_halves[1][1]], dim=1)
    shards = operations.get_device_tensors(outputs.projected)
    guard = round_host.guard_full() if fast else True
    parts = [operations.to_torch(value) for value in (shards if guard else shards[:1])]
    if stamps is not None:
        stamps.append(time.perf_counter())
    if len(shards) != chips or (guard and any(not torch.equal(parts[0], other) for other in parts[1:])):
        raise AssertionError('Replicated learned selector features differ')
    hidden = parts[0].reshape(1, ROWS, 256)
    selection = split_selection(hidden, candidates, unary, USERS, BLOCK, block_width=ROWS)
    if stamps is not None:
        finished = time.perf_counter()
        verify_prestage.add_collect_split((stamps[1] - stamps[0] + stamps[3] - stamps[2]) * 1000, (stamps[2] - stamps[1] + finished - stamps[3]) * 1000)
    return selection


def select_octo_outputs(device, outputs, seeds, counts):
    """read_octo_outputs, then today's per-user selector (select_packed): the unbatched selection."""
    from dflash_packed_proposal import select_packed

    return select_packed(read_octo_outputs(device, outputs, reference=True), seeds, counts, device.predecessors, device.successors)


# ---------------------------------------------------------------------------------------------
# The trace.
# ---------------------------------------------------------------------------------------------

class PreparedOctoDFlashProposal:
    """One traced eight-user, 64-row, T8 proposal over pool slots 0-7 (the quad's trace at eight users). One bucket, (2048,) * 8, built lazily at the first prepare_device:
    placeholders (ids (1, 64), the single-user T8 mask (1, 1, 32, 2080) - the pool's pre-trace mask for slots 0-7 when it holds one - and rope.q and live_k 2 x (1, 1, 64, 128)),
    cached_history the pool's live banks (fused_commit.live_bank_history per pair) or, without QWEN_FAST_FUSED_COMMIT_LIVE_BANKS, eight users' placeholder banks refreshed
    every round, then an eager warm-up, the capture, one blocking replay, the marker and the octo_draft_built ledger point. devices[0] owns the weights (all eight share
    them). collect(), adopt() and finish() take `which` in 0..7. No shadow audit: nothing but this pass drafts T8 (the pair and quad traces draft T16), so there is no twin
    to compare with; exactness of the fold is the CPU proof, and the lever's own cost is tau, read from the committed tokens per round."""

    def __init__(self, devices, *, conv=None):
        devices = tuple(devices)
        if len(devices) != USERS:
            raise ValueError('Eight devices required')
        first = devices[0]
        if any(device.operations is not first.operations or device.mesh is not first.mesh for device in devices):
            raise ValueError('All eight devices must share one mesh and runtime')
        if any(device.block_rows != PREPARED_BLOCK or not getattr(device, 'native_proposal_attention', False) for device in devices):
            raise ValueError('All eight devices must run the qualified native-proposal T16 layers')
        if any(device.kv_history is None for device in devices):
            raise ValueError('All eight devices require a committed K/V cache')
        self.devices = devices
        self.operations, self.mesh = first.operations, first.mesh
        self.block_rows = BLOCK
        self.quad = OctoPass(conv_mode() if conv is None else conv)
        self.buckets, self.owned = {}, []
        self.closed = False
        self._pending = None
        self._audit = None
        self.last_built = False
        self.round_number = None

    @property
    def device_a(self):
        """The weight and codebook owner (select_round groups by its predecessors)."""
        return self.devices[0]

    def pair_label(self):
        from dflash_proposal_trace import _slot_of

        return [_slot_of(device) for device in self.devices]

    def _upload(self, value, *, identifiers=False):
        operations = self.operations
        tensor = operations.from_torch(value, device=self.mesh, dtype=operations.uint32 if identifiers else operations.bfloat16,
            layout=operations.ROW_MAJOR_LAYOUT if identifiers else operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
        self.owned.append(tensor)
        return tensor

    def _protected(self, bucket=None):
        """What no temporary of the pass or of an update may ever queue for release: every device's feature history pair, the placeholders, the lent live banks and the
        pool's mask when the bucket borrowed it."""
        protected = []
        for device in self.devices:
            protected.extend((device.history, device.spare_history))
        protected.extend(self.owned)
        if bucket is not None:
            protected.extend(value for cache in bucket.cached_history for layer in cache for value in layer.values())
            protected.extend(getattr(bucket, 'lent_mask', ()))
        return protected

    def _live_banks(self):
        from fused_commit import live_bank_history

        banks = []
        for first, second in PAIRS:
            pair = live_bank_history(self.devices[first], self.devices[second])
            if pair is None:
                raise ValueError('The octo draft reads the pool live banks: QWEN_FAST_FUSED_COMMIT_LIVE_BANKS and pooled five-layer caches required')
            banks.extend(pair)
        return banks

    def _placeholder_banks(self):
        """cached_history without live banks: per user (device order) and layer a k / v pair of zeros (1, kv, 2048, 128), refreshed from each user's active bank every round."""
        import torch

        kv = tp_shapes.active().draft_kv_heads
        return [[{name: self._upload(torch.zeros((1, kv, CONTEXT, HEAD_DIM), dtype=torch.bfloat16)) for name in ('k', 'v')}
                 for _ in device.kv_history.active] for device in self.devices]

    def _bucket(self):
        key = (CONTEXT,) * USERS
        bucket = self.buckets.get(key)
        if bucket is not None:
            self.last_built = False
            return bucket
        from dflash_packed_proposal import packed_identifiers
        from dflash_proposal_trace import (borrow_pooled_mask, close_draft_audit, compare_draft_audit, open_draft_audit, pool_outputs,
                                           release_draft_audit, traced_pass)
        from gdn_multitoken_conv import addresses, release_owned

        operations, device = self.operations, self.devices[0]
        placeholder_mark = len(self.owned)
        bind_live = _quad.live_banks_requested()
        try:
            host_mask = octo_host_mask()
            query, live = octo_rope([dict(position=CONTEXT, history_rows=CONTEXT)] * USERS)
            pooled_mask = borrow_pooled_mask(device, self.pair_label(), host_mask, operations, self.mesh, log=log_line)
            bucket = SimpleNamespace(context=key, host_mask=host_mask,
                identifiers=self._upload(packed_identifiers([0] * USERS, BLOCK, block_width=ROWS), identifiers=True),
                mask=pooled_mask if pooled_mask is not None else self._upload(host_mask),
                rope=dict(q=tuple(self._upload(value) for value in query), live_k=tuple(self._upload(value) for value in live)),
                cached_history=self._live_banks() if bind_live else self._placeholder_banks(), trace=None, outputs=None,
                owned=[], tokens=None, consumed=set(), parts=None, live_banks=bind_live, draft_audit=None)
            if pooled_mask is not None:
                bucket.lent_mask = (pooled_mask,)
            bucket.inputs = [bucket.identifiers, bucket.mask, *bucket.rope['q'], *bucket.rope['live_k'],
                *(value for cache in bucket.cached_history for layer in cache for value in layer.values())]
            bucket.addresses = [addresses(operations, value) for value in bucket.inputs]
            device.validated_native_proposal_masks.add(addresses(operations, bucket.mask))
            self.buckets[key] = bucket
            if bind_live:
                from fused_commit import note_live_banks

                note_live_banks(self.pair_label(), bucket.context)
            self._update(bucket, (0,) * USERS)
            transient, retain = device.temporaries(self._protected(bucket))
            try:
                warm = self._execute(bucket, transient, retain)
                pooled_outputs = pool_outputs(device, self.pair_label(), warm, operations, log=log_line)
                operations.synchronize_device(self.mesh)
            finally:
                release_owned(operations, transient)
            bucket.owned, retain = device.temporaries(self._protected(bucket))
            from attention_batch import capture_operation

            audit_scope = bucket.draft_audit = open_draft_audit('quad')
            try:
                bucket.trace, bucket.outputs = capture_operation(operations, self.mesh,
                    lambda: traced_pass(operations, lambda: self._execute(bucket, bucket.owned, retain), pooled_outputs))
            finally:
                close_draft_audit(audit_scope)
            operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=True)
            compare_draft_audit(operations, bucket)
            note(self.pair_label(), self.quad.conv)
            import memory_ledger

            memory_ledger.record('octo_draft_built', point='slots=%s' % ','.join(str(slot) for slot in self.pair_label()),
                                 octo_placeholders=list(self.owned), octo_intermediates=list(bucket.owned))
        except BaseException:
            self.buckets.pop(key, None)
            built = locals().get('bucket')
            if built is not None:
                device.validated_native_proposal_masks.discard(addresses(operations, built.mask))
                if built.trace is not None:
                    operations.release_trace(self.mesh, built.trace)
                    built.trace = None
                release_owned(operations, built.owned)
                release_draft_audit(operations, built.draft_audit)
                built.draft_audit = None
            leaked = self.owned[placeholder_mark:]
            del self.owned[placeholder_mark:]
            release_owned(operations, leaked)
            raise
        self.last_built = True
        return bucket

    def _execute(self, bucket, owned, retain):
        rope = dict(q=bucket.rope['q'], live_k=bucket.rope['live_k'])
        return self.devices[0].execute_proposal(bucket.identifiers, None, bucket.mask, rope, context=None, pack=octo_users(self.devices),
            cached_history=bucket.cached_history, owned=owned, retain=retain, stage=lambda name, **values: None, audit=False, quad=self.quad)

    def _update(self, bucket, seeds, *, defer_finish=False):
        from dflash_packed_proposal import note_round_b1, packed_identifiers, round_b1_audit_enabled
        from dflash_proposal_trace import _audit_live_key_rope, pair_mask_refresh_enabled
        from gdn_multitoken_conv import addresses, release_owned

        devices, operations = self.devices, self.operations
        if any(device.history_rows != CONTEXT for device in devices):
            raise ValueError("Octo proposal replay requires every user at the bucket's committed context")
        if any(device.kv_history.pending is not None or device.position - device.history_rows < 0 for device in devices):
            raise ValueError('Octo proposal replay requires a fully committed matching K/V frontier')
        users = octo_users(devices)
        note_round_b1('octo-update')
        query, live = octo_rope(users)
        if round_b1_audit_enabled():
            for half in range(HALVES):
                rows = tuple(table[:, :, 32 * half:32 * (half + 1)] for table in live)
                _audit_live_key_rope(rows, users[HALF_USERS * half:HALF_USERS * (half + 1)], BLOCK)
        sources = [packed_identifiers(list(seeds), BLOCK, block_width=ROWS), *query, *live]
        destinations = [bucket.identifiers, *bucket.rope['q'], *bucket.rope['live_k']]
        if pair_mask_refresh_enabled():
            sources.append(bucket.host_mask)
            destinations.append(bucket.mask)
        for value, destination in zip(sources, destinations, strict=True):
            payload = operations.from_torch(value, dtype=destination.dtype, layout=destination.layout, mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(payload, destination)
        protected = self._protected()
        for device in devices:
            protected.extend((*device.kv_history.owned, *device.kv_history.borrowed))
        protected.extend(getattr(bucket, 'lent_mask', ()))
        owned, retain = devices[0].temporaries(protected)
        kv = tp_shapes.active().draft_kv_heads
        live_banks = getattr(bucket, 'live_banks', False)

        def copy_cache():
            # Without live banks: every user's active bank (whichever side of the four-card slide's swap it is on) into the pass's own placeholder. With them: only a
            # device whose live bank sits on the pool's spare side is copied into the pool's active bank, which the pass reads (the quad's normalisation).
            normalised = 0
            for device, cache in zip(devices, bucket.cached_history, strict=True):
                for active, destination in zip(device.kv_history.active, cache, strict=True):
                    for name in ('k', 'v'):
                        if live_banks and active[name] is destination[name]:
                            continue
                        value = retain(operations.slice(active[name], (0, 0, 0, 0), (1, kv, CONTEXT, HEAD_DIM)))
                        operations.copy(value, destination[name])
                        normalised += 1
            if live_banks and normalised:
                from fused_commit import LIVE_BANKS_NORMALISED, log_line as fused_log

                fused_log('%s pair=%s normalised=%d' % (LIVE_BANKS_NORMALISED, self.pair_label(), normalised))
        if defer_finish:
            try:
                copy_cache()
            except BaseException:
                release_owned(operations, owned)
                raise
            return owned
        try:
            copy_cache()
            operations.synchronize_device(self.mesh)
            if [addresses(operations, value) for value in bucket.inputs] != bucket.addresses:
                raise AssertionError('Prepared octo proposal input addresses moved')
        finally:
            release_owned(operations, owned)

    def prepare_device(self, seeds):
        """Enqueue the pass's copies and its replay without waiting on the device. False when there is nothing to prewarm: closed, or any device closed, mid-publication or
        under audit."""
        if self.closed:
            return False
        if self._pending is not None:
            self.discard_pending()
        seeds = tuple(seeds)
        if len(seeds) != USERS:
            raise ValueError('One anchor per octo user required')
        if any(device.closed or device.pending is not None or device.progress is not None for device in self.devices):
            return False
        bucket = self._bucket()
        owned = self._update(bucket, seeds, defer_finish=True)
        self.operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=False)
        self._pending = (seeds, bucket, owned)
        return True

    def has_pending(self, which, seed):
        if self._pending is None:
            return False
        seeds, bucket, _ = self._pending
        if which in bucket.consumed:
            return False
        return seed == seeds[which]

    def finish(self, which, count):
        """The quad's finish() for eight users: the first call checks the addresses, releases the transients and selects (unless the round's batched selection was adopted);
        each user reads its own tokens. A pass of seven proposals cannot answer a wider ticket: that is an error, never a short ticket."""
        if self._pending is None:
            raise ValueError('No prepared octo proposal is pending')
        if type(count) is not int or not 1 <= count <= PROPOSALS:
            raise ValueError('The octo draft pass proposes %d tokens a seat, not %r' % (PROPOSALS, count))
        seeds, bucket, owned = self._pending
        if bucket.tokens is None:
            from dflash_proposal_trace import compare_draft_audit
            from gdn_multitoken_conv import addresses, release_owned

            operations = self.operations
            moved = [addresses(operations, value) for value in bucket.inputs] != bucket.addresses
            release_owned(operations, owned)
            if moved:
                raise AssertionError('Prepared octo proposal input addresses moved')
            compare_draft_audit(operations, bucket)
            bucket.tokens = select_octo_outputs(self.devices[0], bucket.outputs, seeds, (PROPOSALS,) * USERS)
        tokens = bucket.tokens[which]
        bucket.consumed.add(which)
        if len(bucket.consumed) == USERS:
            self._pending = None
            bucket.tokens, bucket.consumed, bucket.parts = None, set(), None
        return tokens[:count]

    def collect(self):
        """QWEN_FAST_ROUND_B1 (C1), as the quad's: the address check, the transient release and the readback, with the seeds and counts finish() would select with."""
        if self._pending is None:
            raise ValueError('No prepared octo proposal is pending')
        seeds, bucket, owned = self._pending
        if bucket.tokens is not None or bucket.consumed:
            raise ValueError('This octo proposal was already selected')
        from dflash_proposal_trace import compare_draft_audit
        from gdn_multitoken_conv import addresses, release_owned

        operations = self.operations
        self._pending = (seeds, bucket, [])
        moved = [addresses(operations, value) for value in bucket.inputs] != bucket.addresses
        release_owned(operations, owned)
        if moved:
            raise AssertionError('Prepared octo proposal input addresses moved')
        compare_draft_audit(operations, bucket)
        bucket.parts = read_octo_outputs(self.devices[0], bucket.outputs)
        return dict(parts=bucket.parts, seeds=seeds, counts=(PROPOSALS,) * USERS)

    def audit_selection(self):
        """QWEN_FAST_ROUND_B1_AUDIT (C1): the tokens finish() would have selected itself."""
        if self._pending is None:
            raise ValueError('No prepared octo proposal is pending')
        seeds, bucket, _ = self._pending
        return select_octo_outputs(self.devices[0], bucket.outputs, seeds, (PROPOSALS,) * USERS)

    def adopt(self, tokens):
        if self._pending is None:
            raise ValueError('No prepared octo proposal is pending')
        _, bucket, _ = self._pending
        tokens = tuple(tokens)
        if bucket.tokens is not None or bucket.consumed or len(tokens) != USERS:
            raise ValueError('All eight users of one pending octo proposal must adopt one selection')
        bucket.tokens = tokens

    def run_audit(self, round_number):
        """The coordinator's shadow-audit hook: there is none for this pass (nothing else drafts T8)."""
        return None

    def discard_pending(self):
        if self._pending is None:
            return
        from gdn_multitoken_conv import release_owned

        _, bucket, owned = self._pending
        self._pending = None
        bucket.tokens, bucket.consumed, bucket.parts = None, set(), None
        release_owned(self.operations, owned)

    def close(self):
        """Traces released, then the buckets' buffers, then the drafter audit's held pairs (a trace writes them until it is released, so they are freed last)."""
        if self.closed:
            return
        from dflash_proposal_trace import release_draft_audit
        from gdn_multitoken_conv import addresses, release_owned

        scopes = [getattr(bucket, 'draft_audit', None) for bucket in self.buckets.values()]
        self.operations.synchronize_device(self.mesh)
        self.discard_pending()
        for bucket in self.buckets.values():
            if bucket.trace is not None:
                self.operations.release_trace(self.mesh, bucket.trace)
                bucket.trace = None
            self.devices[0].validated_native_proposal_masks.discard(addresses(self.operations, bucket.mask))
            release_owned(self.operations, bucket.owned)
            bucket.owned.clear()
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.buckets.clear()
        self.closed = True
        for scope in scopes:
            release_draft_audit(self.operations, scope)


# ---------------------------------------------------------------------------------------------
# Admission (the attach) and the coordinator's engage-time checks.
# ---------------------------------------------------------------------------------------------

REQUIRED_FLAGS = ('QWEN_FAST_PACKED_PROPOSAL', 'QWEN_FAST_PIPELINED_PROPOSALS', 'QWEN_FAST_PAIR_ROW_EXACT', 'QWEN_FAST_ROUND_B1')
EXCLUDED_FLAGS = (
    ('QWEN_FAST_PARKED_ENGINES', 'the parked-engine draft book binds the pair and quad traces to the slots; the octo draft trace is not in it'),
    ('QWEN_FAST_PARKED_DRAFTS', 'the parked-engine draft book binds the pair and quad traces to the slots; the octo draft trace is not in it'),
    ('QWEN_FAST_QUAD_DRAFT_AUDIT', 'the quad shadow audit replays the first block\'s pair traces; the octo pass has no twin to compare with'),
    ('QWEN_FAST_DRAFT_SINGLES_AUDIT', 'the singles audit compares a batched draft with the members\' own T16 single-user drafts; the octo pass drafts T8'),
    ('QWEN_FAST_PAIR_MASK_AUDIT', 'the mask audit reads the quad/pair masks; the octo pass has its own T8 mask'),
)


UNQUALIFIED_ITEMS = (
    ("the T8 draft's tau: committed tokens per octo round against the same shape drafted by the two T16 quads cut to seven rows (the target's text is the same either way)", 'OD1 OD2'),
    ("the 24-piece K/V plan and the four-way-per-half fold: SDPA at 64 query / 16 KV heads, eight-row concat and slice in TILE layout, run in a trace", 'OD1'),
    ("one eight-seat capture's DRAM beside the two quads' (octo_draft_built in the memory ledger)", 'OD1'),
)
UNQUALIFIED_LINE = '[OCTO-DRAFT] UNQUALIFIED (gate only) ({index}/{count}): {text} [job {job}]'


def admission_problems(environ=None):
    """Every reason QWEN_FAST_OCTO_DRAFT=1 cannot be admitted under `environ` ([] when it can, or when the flag is off). The value is strict (enabled raises on a bad one)."""
    environ = os.environ if environ is None else environ
    if not enabled(environ):
        return []
    problems = []
    if environ.get('QWEN_FAST_OCTO', 'off') not in ('live', 'alternate'):
        problems.append('%s=1 needs QWEN_FAST_OCTO=live|alternate (the pass drafts the octo rounds)' % FLAG)
    for name in REQUIRED_FLAGS:
        if environ.get(name) != '1':
            problems.append('%s=%s, not 1: the pass is a packed proposal trace the coordinator selects in its batched round' % (name, environ.get(name, '(unset)')))
    for name, why in EXCLUDED_FLAGS:
        if environ.get(name, '0') not in ('0', ''):
            problems.append('%s=%s: %s' % (name, environ.get(name), why))
    if tp_shapes.chip_count(environ) != _quad.TP4:
        problems.append('%s serves QWEN_FAST_TP=4 only' % FLAG)
    needs = _quad.live_banks_missing(environ)
    if needs:
        problems.append('%s=1 needs %s' % (_quad.LIVE_BANKS_FLAG, ','.join('%s=1' % name for name in needs)))
    conv_mode(environ)
    return problems


def admission(environ=None, *, log=None):
    """None while QWEN_FAST_OCTO_DRAFT is off. Else the record of what is admitted, or ValueError naming EVERY reason the attach is refused. serving_octo.octo_admission folds
    admission_problems into its own list (so the octo shape, the gate run and the quad flags are settled in the same refusal); serving_runtime calls this for the flag set with
    QWEN_FAST_OCTO off, which refuses it by name."""
    environ = os.environ if environ is None else environ
    if not enabled(environ):
        return None
    log = log_line if log is None else log
    problems = admission_problems(environ)
    if problems:
        for problem in problems:
            log('%s: %s' % (REFUSED_LINE, problem))
        raise ValueError('%s=1 is refused: %s' % (FLAG, '; '.join(problems)))
    log_admitted(log)
    return dict(rows=ROWS, users=USERS, block=BLOCK, proposals=PROPOSALS)


def log_admitted(log=None):
    """The admission line and one UNQUALIFIED line per card question (its own marker: serving_octo's six are counted by the octo judge and stay six)."""
    log = log_line if log is None else log
    for index, (text, job) in enumerate(UNQUALIFIED_ITEMS, 1):
        log(UNQUALIFIED_LINE.format(index=index, count=len(UNQUALIFIED_ITEMS), text=text, job=job))
    log(ADMITTED_LINE)


def refusal(devices, batched, environ=None):
    """Why the octo draft pass cannot serve these eight packable devices (a permanent configuration reason: the coordinator disables it with the reason), or None."""
    environ = os.environ if environ is None else environ
    if tp_shapes.chip_count(environ) != _quad.TP4:
        return 'the octo draft serves QWEN_FAST_TP=4 only'
    missing = [name for name in REQUIRED_FLAGS if environ.get(name) != '1']
    if missing:
        return 'requires ' + ','.join('%s=1' % name for name in missing)
    needs = _quad.live_banks_missing(environ)
    if needs:
        return '%s=1 needs %s (the live bank must never move)' % (_quad.LIVE_BANKS_FLAG, ','.join('%s=1' % name for name in needs))
    if batched is None:
        return 'requires the batched selection (QWEN_FAST_ROUND_B1=1)'
    if len(devices) != USERS:
        return 'the octo draft needs %d devices, not %d' % (USERS, len(devices))
    first = devices[0]
    if any(device.operations is not first.operations or device.mesh is not first.mesh for device in devices):
        return 'the eight devices do not share one mesh'
    if tp_shapes.mesh_width(first.mesh, environ) is None:
        return 'the eight devices are not on the (1, %d) mesh' % _quad.TP4
    if any(getattr(device, 'block_rows', None) != PREPARED_BLOCK for device in devices):
        return 'the eight devices are not T16 (the prepared layers the pass reuses)'
    if any(not getattr(device, 'fused_convolution', False) for device in devices):
        return 'requires the fused learned convolution'
    if any(device.layers is not first.layers or device.predecessors is not first.predecessors or device.successors is not first.successors for device in devices):
        return 'the eight devices do not share one draft weight set'
    conv = conv_mode(environ)
    if _pinned.grid_fits(first.mesh, conv) is False:
        return 'the compute grid cannot hold %s=%s' % (CONV_FLAG, conv)
    return None


def elapsed_ms(started):
    return '%.1f' % ((time.perf_counter() - started) * 1000)
