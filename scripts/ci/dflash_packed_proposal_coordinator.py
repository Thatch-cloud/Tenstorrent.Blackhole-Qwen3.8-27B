"""QWEN_FAST_PACKED_PROPOSAL round-level wiring: pair eligible bridges by their fixed
pool slot (dflash_packed_proposal.DRAFT_PAIRS), run one traced packed pass per
full pair (dflash_proposal_trace.PreparedPackedDFlashProposal) instead of two separate
single-user passes, and hand each pair's per-user result back through the SAME
has_pending(seed)/finish(count) surface DFlashDevice.propose() already calls on
device.proposal_capture - so neither DFlashDevice nor a bridge's own drafts() needs to
change at all.

Entirely inert with the flag off: PackedProposalCoordinator is only ever constructed
from inside serving_worker_hook.FastWorkerHook._drafts's QWEN_FAST_PACKED_PROPOSAL
branch, so a hook that never takes that branch never creates one, never stores one on
itself, and never touches a device's proposal_capture - the flag-off path is exactly
prepare_pipelined_drafts(), unchanged.

Packing is gated to history_rows == 2048 on BOTH paired devices (steady-state decode,
past the prefill ramp) - see PreparedPackedDFlashProposal's own docstring for why a
narrower, still-growing history is deliberately left unpacked rather than recaptured
every round.
"""

import os
import time


AUDIT_FLAG = 'QWEN_FAST_PACKED_AUDIT'
AUDIT_LINE = '[PACKED-PROPOSE] round={round} pairs={pairs} propose_ms={propose_ms}'
# One line per pair whose first (or any) traced execution raised - run 35581352016
# (image v73): a pair formed the moment packable() first saw both devices at
# history_rows == 2048, and dflash_device.DFlashDevice.execute_proposal refused the
# packed cached_history it was handed ('Every prepared learned layer requires a
# committed K/V cache', dflash_device.py:661-662). See packable()'s own docstring for
# the precondition this line's fallback pair failed anyway, on hardware, despite
# passing every host-side check this module can perform.
PAIR_FALLBACK_LINE = '[PACKED-PROPOSE] pair={pair} fallback={fallback}'
# One line whenever _ensure_single_user rebuilds a device's own single-user
# PreparedDFlashProposal after _release_single_user freed it - a fallback after
# release is exactly the case run 35585107688 needs visible.
RECAPTURE_LINE = '[PACKED-PROPOSE] recapture slot={slot}'
PACKED_CONTEXT = 2048

# S2 W12 (design 4, W12): which users draft on their own single-user capture in a round, and why - under
# QWEN_FAST_EXTENT_REPLAY=1 (the c2-packed profiles) with QWEN_FAST_PACKED_AUDIT=1, one line per round that has
# any. A round's slots and reasons are parallel lists, in the order the singles were prepared:
#   ramp          a pair member below the steady-state 2048-row history (the prefill ramp): packs from 2048 on
#   unpackable    a pair at 2048 that packable() still refuses (no native proposal path, K/V history or capture)
#   absent        the pair's other slot has no user this round
#   split         QWEN_FAST_PAIRS_PACKED_ONLY split the pair for a round the packed step serves sequentially
#   dram_reserve  a fresh pair capture refused for DRAM headroom (PAIR_FALLBACK_LINE says how much)
#   failure       the pair's traced pass raised this round (PAIR_FALLBACK_LINE names it)
#   declined      the pair's prepare_device answered False
#   unslotted     a device with no pool slot, which never pairs
# Rounds the quad serves have no singles. Nothing else changes: the line only reports what prepare() did.
EXTENT_FLAG = 'QWEN_FAST_EXTENT_REPLAY'
SINGLES_LINE = '[PACKED-PROPOSE] singles round={round} slots={slots} reasons={reasons}'
SINGLE_REASONS = ('ramp', 'unpackable', 'absent', 'split', 'dram_reserve', 'failure', 'declined', 'unslotted')
# S2 W12 (Report 5 W1): a pair bucket exists only at the steady-state context. packable() gates both users to
# history_rows == 2048 and the trace keys its buckets by (history_rows_a, history_rows_b), so any other key is a
# regression that would capture a new bucket nearly every round of the ramp (and hold its DRAM).
PAIR_BUCKET_CONTEXT = (PACKED_CONTEXT, PACKED_CONTEXT)


def pooled_draft_mask_shapes(users, block_rows):
    """S2 (serving_buffer_pool's `draft_masks=`, which serving_runtime passes at attach under
    QWEN_FAST_EXTENT_REPLAY=1): {slot group: mask shape} for every packed draft a pool of `users` scheduler slots
    can build - each fixed pair (dflash_packed_proposal.DRAFT_PAIRS) whose slots all exist, at the one
    bucket a pair builds (PAIR_BUCKET_CONTEXT, QWEN_FAST_PAIR_ROW_EXACT deciding the fold as the bucket will),
    and the quad over slots 0-3 while QWEN_FAST_QUAD_DRAFT is on - so each borrows a mask allocated before any
    trace instead of uploading its own after them (M0). {} with QWEN_FAST_PACKED_PROPOSAL off: no pair forms."""
    refuse_quad_blocks(users)
    if os.environ.get('QWEN_FAST_PACKED_PROPOSAL') != '1':
        return {}
    from dflash_packed_proposal import DRAFT_PAIRS
    from dflash_proposal_trace import pair_host_mask

    shapes = {}
    pair_shape = tuple(pair_host_mask(*PAIR_BUCKET_CONTEXT, block_rows)[0].shape)
    for group in DRAFT_PAIRS:
        if max(group) < users:
            shapes[tuple(group)] = pair_shape
    if os.environ.get(QUAD_DRAFT_FLAG, '0') != '0':
        import quad_draft

        if max(quad_draft.SLOTS) < users and block_rows == quad_draft.BLOCK:
            shapes[tuple(quad_draft.SLOTS)] = tuple(quad_draft.quad_host_mask().shape)
        if quad_blocks_requested():
            for slots, _ in _pooled_quad_blocks(quad_draft, users, block_rows):
                shapes.setdefault(slots, tuple(quad_draft.quad_host_mask().shape))
    if octo_draft_requested():
        import octo_draft_tp

        if max(octo_draft_tp.SLOTS) < users and block_rows == octo_draft_tp.PREPARED_BLOCK:
            shapes[tuple(octo_draft_tp.SLOTS)] = tuple(octo_draft_tp.octo_host_mask().shape)
    return shapes


def refuse_quad_blocks(users):
    """The attach's refusal of QWEN_FAST_QUAD_DRAFT_BLOCKS (serving_runtime calls it whenever the flag is requested, with or without
    QWEN_FAST_EXTENT_REPLAY): whenever it is requested, whatever else is set (QWEN_FAST_QUAD_DRAFT off included),
    ValueError by name - not a later round that silently skips the flag. Nothing when it is off."""
    if not quad_blocks_requested():
        return
    import quad_draft

    reason = quad_blocks_refusal(quad_draft, users)
    if reason is not None:
        raise ValueError('%s: %s' % (QUAD_BLOCKS_FLAG, reason))


def _pooled_quad_blocks(quad_draft, users, block_rows):
    """QWEN_FAST_QUAD_DRAFT_BLOCKS: the per-block quads a pool of `users` slots holds buffers for. A value it cannot serve, or a pool that is
    not eight slots, refuses the attach by name (ValueError) rather than leave a quad that can never form."""
    reason = quad_blocks_refusal(quad_draft, users)
    if reason is not None:
        raise ValueError('%s: %s' % (QUAD_BLOCKS_FLAG, reason))
    if block_rows != quad_draft.BLOCK:
        raise ValueError('%s: the quads are %d-row T16 blocks, not %r' % (QUAD_BLOCKS_FLAG, quad_draft.BLOCK, block_rows))
    return [(slots, pairs) for slots, pairs in quad_block_groups(quad_draft) if max(slots) < users]


# What a draft pass's head leaves per candidate chunk (draft_shared_head: top-16 values and indices) and the width
# of its selector projection (dflash_device.execute_proposal): the pooled output shapes' last dimensions.
TOP_CANDIDATES = 16
SELECTOR_WIDTH = 256
# A single-user pass pads its block to the 32-row pass before the selector projection (execute_proposal).
PASS_ROWS = 32


def pooled_draft_output_shapes(users, block_rows):
    """S2 v86 (serving_buffer_pool's `draft_outputs=`, which serving_runtime passes at attach under
    QWEN_FAST_EXTENT_REPLAY=1): {slot group: dict(chunks=, head=, projected=)} for every traced draft a pool of
    `users` scheduler slots can build - each slot's single-user draft (its head at block_rows rows, the projection
    at the 32-row pass), and, with QWEN_FAST_PACKED_PROPOSAL on, each fixed pair whose slots all exist (the 32-row
    block) and the quad over slots 0-3 (64 rows) while QWEN_FAST_QUAD_DRAFT is on - so every trace copies its head
    outputs into buffers allocated before any trace, which no other trace's replay can write (run 36416471352: a
    fresh pair's outputs sat in an older single-user trace's holes, and its replay, later in the same round,
    overwrote them before the read). The chunks are draft_shared_head.candidate_chunks()."""
    from draft_shared_head import candidate_chunks

    chunks = tuple(candidate_chunks())

    def spec(head_rows, projected_rows):
        return dict(chunks=chunks, head=(1, 1, head_rows, TOP_CANDIDATES),
                    projected=(1, 1, projected_rows, SELECTOR_WIDTH))

    refuse_quad_blocks(users)
    shapes = {(slot,): spec(block_rows, max(block_rows, PASS_ROWS)) for slot in range(users)}
    if os.environ.get('QWEN_FAST_PACKED_PROPOSAL') != '1':
        return shapes
    from dflash_packed_proposal import BLOCK_WIDTH, DRAFT_PAIRS

    for group in DRAFT_PAIRS:
        if max(group) < users:
            shapes[tuple(group)] = spec(BLOCK_WIDTH, BLOCK_WIDTH)
    if os.environ.get(QUAD_DRAFT_FLAG, '0') != '0':
        import quad_draft

        if max(quad_draft.SLOTS) < users and block_rows == quad_draft.BLOCK:
            shapes[tuple(quad_draft.SLOTS)] = spec(quad_draft.ROWS, quad_draft.ROWS)
        if quad_blocks_requested():
            for slots, _ in _pooled_quad_blocks(quad_draft, users, block_rows):
                shapes.setdefault(slots, spec(quad_draft.ROWS, quad_draft.ROWS))
    if octo_draft_requested():
        import octo_draft_tp

        if max(octo_draft_tp.SLOTS) < users and block_rows == octo_draft_tp.PREPARED_BLOCK:
            shapes[tuple(octo_draft_tp.SLOTS)] = spec(octo_draft_tp.ROWS, octo_draft_tp.ROWS)
    return shapes


DRAM_RESERVE_FLAG = 'QWEN_FAST_PACKED_PROPOSAL_DRAM_RESERVE_MB'
DRAM_RESERVE_DEFAULT_MB = 256

# Variable-user packed rounds, M0's fallback (QWEN_FAST_PAIRS_PACKED_ONLY=1, default off): in a
# round the packed step's policy will not serve as one pass (serving_worker_hook._drafts'
# packed_rows is None: a user finished, or one is within a block round of its budget), every
# member of a slot pair proposes on its own single-user capture, as an unpaired bridge does -
# the pair trace runs only in packed rounds. For the case M0's mask audit finds the mask intact
# while the pair users' tail acceptance still collapses: single-proposal users hold 2.76-4.0
# tokens per step in those same steps. The hook passes packed_round only while the flag is on;
# PAIRS_PACKED_ONLY_MARKER is logged once per process, the first round it unpairs a pair.
PAIRS_PACKED_ONLY_FLAG = 'QWEN_FAST_PAIRS_PACKED_ONLY'
PAIRS_PACKED_ONLY_MARKER = '[PINDIAG] pairs packed only'
_PAIRS_PACKED_ONLY_NOTED = []


def pairs_packed_only_enabled(environ=None):
    """QWEN_FAST_PAIRS_PACKED_ONLY=1."""
    return (os.environ if environ is None else environ).get(PAIRS_PACKED_ONLY_FLAG) == '1'


def unpair_groups(groups, round_number):
    """QWEN_FAST_PAIRS_PACKED_ONLY: every member of every group on its own, in group order.
    Logs PAIRS_PACKED_ONLY_MARKER once per process, the first time a full pair is split."""
    pairs = [list(group) for group in groups if len(group) == 2]
    if pairs and not _PAIRS_PACKED_ONLY_NOTED:
        _PAIRS_PACKED_ONLY_NOTED.append(round_number)
        message = '%s: round=%s unpaired=%s (the policy serves this round sequentially)' % (
            PAIRS_PACKED_ONLY_MARKER, round_number, pairs)
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    return [(slot,) for group in groups for slot in group]


# Q4 (quad_draft.py; QWEN_FAST_QUAD_DRAFT, default off): with all four users live and packable, one 64-row
# quad pass replaces the two pair passes. Read at each round; '0' or unset and nothing here imports quad_draft,
# every round is today's.
QUAD_DRAFT_FLAG = 'QWEN_FAST_QUAD_DRAFT'
# tp4/next-5 (quad_draft_tp.QUADS, default off): at eight seats one 64-row quad per block of four slots, (0, 1, 2, 3) and (4, 5, 6, 7), each with
# its own trace, state and capture headroom check. '' and '0' are off - the quad is then exactly the one over slots 0-3 above, and nothing
# here reads quad_draft_tp. Any other value reaches quad_draft_tp.blocks_refusal, which names why it is not served ('2' is the only one that is).
QUAD_BLOCKS_FLAG = 'QWEN_FAST_QUAD_DRAFT_BLOCKS'


def quad_blocks_requested(environ=None):
    """QWEN_FAST_QUAD_DRAFT_BLOCKS set to anything but '' or '0'."""
    return (os.environ if environ is None else environ).get(QUAD_BLOCKS_FLAG, '') not in ('', '0')


def quad_block_groups(quad_draft):
    """The per-block quads: ((slots, pairs), ...) from the twin's QUADS - ((0, 1, 2, 3), ((0, 1), (2, 3))), ((4, 5, 6, 7), ((4, 5), (6, 7)))."""
    return tuple((tuple(slots), tuple(zip(slots[::2], slots[1::2]))) for slots in quad_draft.QUADS)


def quad_blocks_refusal(quad_draft, users=None, environ=None):
    """Why the per-block quads cannot serve this process, or None: quad_draft_tp.blocks_refusal where the module has it, and a named reason
    where it does not (the pair's pinned quad_draft has no QUADS)."""
    refuse = getattr(quad_draft, 'blocks_refusal', None)
    if refuse is None:
        return 'the two-quad draft serves QWEN_FAST_TP=4 only (this quad_draft has no QUADS)'
    return refuse(users, environ)

# QWEN_FAST_OCTO_DRAFT (octo_draft_tp.py; gate only, default off): at an octo round (QWEN_FAST_OCTO) with all eight seats live, ONE eight-seat, eight-row, 64-row draft pass in place of the
# two quad passes. Read here without importing the module (a pair process never loads the four-card twins); the attach validates the value strictly (octo_draft_tp.enabled), and the hook
# passes octo_draft=True to prepare() only for a round the packed step planned for the octo block. Off, none of this runs and prepare() is called as it always was.
OCTO_DRAFT_FLAG = 'QWEN_FAST_OCTO_DRAFT'
OCTO_RELEASED_LINE = '[OCTO-DRAFT] released slots={slots}'


def octo_draft_requested(environ=None):
    """QWEN_FAST_OCTO_DRAFT set to anything but '' or '0'."""
    return (os.environ if environ is None else environ).get(OCTO_DRAFT_FLAG, '') not in ('', '0')


# QWEN_FAST_DRAFT_SINGLES_AUDIT (draft_singles_audit.py; default off): every batched group of a round - a packed pair or the
# quad - against its members' own single-user drafts, prepared with the same seeds in the same round and compared bit for bit
# after the fence. Read at each round; unset or empty and nothing here imports it, every round is today's. Under the flag the
# single-user captures are kept (_release_single_user releases nothing): the audit replays them.
SINGLES_AUDIT_FLAG = 'QWEN_FAST_DRAFT_SINGLES_AUDIT'


def _singles_audit_on():
    """Unset, empty and '0' are off (the repo's usual off value); anything else is on and validated by draft_singles_audit."""
    return os.environ.get(SINGLES_AUDIT_FLAG, '') not in ('', '0')


# One packed pair's own placeholder buffers at the only packable geometry (steady
# state, 2048-row context, T16 block_rows=16): identifiers (1,32) uint32 (~128 B,
# negligible), mask (1,1,32,4160) bf16 (~266 KB), rope.q (2x(1,1,32,128) bf16,
# ~16 KB), rope.k (2x(1,1,4160,128) bf16, ~2.03 MB), rope.live_k (2x(1,1,32,128)
# bf16, ~16 KB), cached_history (2 users x 5 layers x 2 k/v x (1,4,2048,128) bf16,
# ~40 MB - the dominant term). Precisely computed from dflash_proposal_trace.
# PreparedPackedDFlashProposal._bucket's own upload calls and dflash_batched_mask.
# segments' span formula (see the report for the full derivation) - a LOWER BOUND:
# it does not include the trace's own internal compute/intermediate buffers
# (embeddings, 5 layers of attention and MLP intermediates) that capture_operation
# retains for replay, which are not sized from source here.
PAIR_CAPTURE_PLACEHOLDER_BYTES = 128 + 266_240 + 16_384 + 2_129_920 + 16_384 + 41_943_040

# The (1,1,2048,5120) bf16 transient every prepare_publication call allocates and
# releases (the 'padded'/'combined' tensor, general or fused_steady_state path
# alike, dflash_device.py) - exactly the shape and size of the allocation that
# killed run 35585107688's engine (20,971,520 B). A fallback's own single-user
# commit needs this same allocation regardless of packing, so the reserve below
# accounts for BOTH paired users needing it in the worst case (the pair capture
# fails and both fall back to their own commit in the same round).
PREPARE_PUBLICATION_TRANSIENT_BYTES = 2048 * 5120 * 2


def estimated_pair_capture_bytes():
    return PAIR_CAPTURE_PLACEHOLDER_BYTES + 2 * PREPARE_PUBLICATION_TRANSIENT_BYTES


# S2 W6d: one single-user PreparedDFlashProposal bucket at the 2048 context, the one _ensure_single_user
# rebuilds (dflash_proposal_trace.PreparedDFlashProposal.__init__'s uploads, T16): identifiers (1,16) uint32
# (64 B), history (1,1,2080,5120) bf16 (21,299,200 B), mask (1,1,32,2080) bf16 (133,120 B), rope.q
# 2x(1,1,32,128) bf16 (16,384 B), rope.k 2x(1,1,2080,128) bf16 (1,064,960 B) and cached_history 5 layers x
# k/v x (1,4,2048,128) bf16 (20,971,520 B). A LOWER BOUND, like the pair's: the trace's retained
# intermediates are not sized here (the after point measures the real figure).
SINGLE_CAPTURE_PLACEHOLDER_BYTES = 64 + 21_299_200 + 133_120 + 16_384 + 1_064_960 + 20_971_520


def estimated_single_capture_bytes():
    return SINGLE_CAPTURE_PLACEHOLDER_BYTES + PREPARE_PUBLICATION_TRANSIENT_BYTES


# S2 W6c: the line PackedProposalCoordinator.release_closed logs at every detach it runs for.
RELEASED_LINE = '[PACKED-PROPOSE] released quad={quad} pairs={pairs}'
# S2 (s2-design.md): W6c and W6d run only under this flag, read at each use. The attach refuses a value other
# than '0' or '1' (serving_request_factory.extent_replay_enabled, W7); here only '1' turns them on, so a round
# never raises over it.
EXTENT_REPLAY_FLAG = 'QWEN_FAST_EXTENT_REPLAY'


def extent_memory_points(environ=None):
    """QWEN_FAST_EXTENT_REPLAY=1."""
    return (os.environ if environ is None else environ).get(EXTENT_REPLAY_FLAG) == '1'


# Engine reuse (QWEN_FAST_PARKED_ENGINES=1, serving_parked_engines; read at each use, and only '1' turns it on - the attach refuses any other
# value). A parked device is not closed when its request ends: its slot's next request rebinds it, and every rebind advances its
# rebind_generation. So under the flag
#   - PARKED_RELEASED_LINE is logged when a departing request's parked device has its drafter traces retired (release_parked, which the worker
#     hook reaches through serving_parked_engines.after_park) - E1, without the draft book;
#   - a pair or quad trace records its members' generations when it is captured, and _cached_trace, _retire_quad and release_closed treat a
#     member whose generation moved on like a closed one: a rebound device never replays a trace captured for its previous request, even when
#     release_parked did not run (the backstop). The quads of the eight-seat path live in quad_blocks, and are keyed here with the pairs;
#   - _ensure_single_user rebuilds under the single 2048 bucket (serving_request_factory.single_proposal_bucket), whatever the device's
#     position: a device rebound below 2048 would otherwise get one narrower bucket, and the first round its history outgrew it would raise
#     'Committed history exceeds prepared request contexts';
#   - the capture headroom and the ledger's readings ask the captures' measured bytes (serving_prefill_admission.measured_*), by chip count.
# With the DRAFT BOOK registered (QWEN_FAST_PARKED_DRAFTS=1, "2c"; serving_parked_engines.DraftTraceBook) the pair and block-quad traces are
# bound to the pool slots for the process: the coordinator's `pairs` and `quad_blocks` ARE the book's dicts, every close of a trace routes
# through the book (retire), no generation is recorded or checked, no single-user capture is ever released, and hook close drops views and
# counters only. Off, none of this runs and every call is today's.
PARKED_ENGINES_FLAG = 'QWEN_FAST_PARKED_ENGINES'
PARKED_RELEASED_LINE = '[PACKED-PROPOSE] released parked slot={slot} quad={quad} pairs={pairs}'
BOOK_QUAD_LINE = '[PINDIAG] parked draft book quad slots={slots} captured_at_round={round}'

_DRAFT_BOOKS = []


def register_draft_book(book):
    """Register the process's draft trace book (2c). Returns the callable that removes it. One at a time."""
    if _DRAFT_BOOKS:
        raise ValueError('A draft trace book is already registered')
    _DRAFT_BOOKS.append(book)

    def unregister():
        if book in _DRAFT_BOOKS:
            _DRAFT_BOOKS.remove(book)

    return unregister


def draft_book():
    """The registered draft trace book, or None (every process without QWEN_FAST_PARKED_DRAFTS=1)."""
    return _DRAFT_BOOKS[0] if _DRAFT_BOOKS else None


def parked_engines_on(environ=None):
    """QWEN_FAST_PARKED_ENGINES=1."""
    return (os.environ if environ is None else environ).get(PARKED_ENGINES_FLAG) == '1'


def generation(device):
    """A device's rebind_generation (serving_parked_engines.rebind_device advances it); 0 for one never rebound."""
    return getattr(device, 'rebind_generation', 0)


def generations(devices):
    return tuple(generation(device) for device in devices)


def single_capture_bytes():
    """What a single-capture rebuild's ledger reading counts on: under QWEN_FAST_PARKED_ENGINES=1 the measured
    serving_prefill_admission.measured_single_capture_bytes (by chip count), else estimated_single_capture_bytes(), as always."""
    if parked_engines_on():
        import serving_prefill_admission

        return serving_prefill_admission.measured_single_capture_bytes()
    return estimated_single_capture_bytes()


def pair_capture_bytes():
    """What a fresh pair capture's headroom (and its ledger reading) asks: under QWEN_FAST_PARKED_ENGINES=1 the measured bytes, else
    estimated_pair_capture_bytes(), as always."""
    if parked_engines_on():
        import serving_prefill_admission

        return serving_prefill_admission.measured_pair_capture_bytes()
    return estimated_pair_capture_bytes()


def quad_capture_bytes():
    """What a fresh quad capture's headroom (and its ledger reading) asks: under QWEN_FAST_PARKED_ENGINES=1 the measured bytes, else
    quad_draft.QUAD_CAPTURE_BYTES_EST, as always."""
    if parked_engines_on():
        import serving_prefill_admission

        return serving_prefill_admission.measured_quad_capture_bytes()
    import quad_draft

    return quad_draft.QUAD_CAPTURE_BYTES_EST


def ledger_before(op, estimate, point):
    """S2 W6d: memory_ledger.before ahead of a capture, only under QWEN_FAST_EXTENT_REPLAY=1 (and a no-op there
    unless QWEN_FAST_MEMORY_LEDGER=1). Unset, nothing is imported or read."""
    if not extent_memory_points():
        return None
    import memory_ledger

    return memory_ledger.before(op, estimate=estimate, point=point)


def ledger_after(token):
    """S2 W6d: memory_ledger.after for a token ledger_before returned; None does nothing."""
    if token is None:
        return None
    import memory_ledger

    return memory_ledger.after(token)


def dram_reserve_bytes(environ=None):
    """QWEN_FAST_PACKED_PROPOSAL_DRAM_RESERVE_MB: an integer number of megabytes,
    default 256. Read at each round rather than at import, so the value a test
    sets is the one the round sees."""
    value = (os.environ if environ is None else environ).get(DRAM_RESERVE_FLAG, str(DRAM_RESERVE_DEFAULT_MB))
    try:
        megabytes = int(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be an integer number of megabytes' % DRAM_RESERVE_FLAG)
    if megabytes < 0:
        raise ValueError('%s must not be negative' % DRAM_RESERVE_FLAG)
    return megabytes * 1024 * 1024


def dram_headroom(device):
    """The smallest largest-contiguous-free-DRAM-block across this device's chips,
    or None if the allocator statistics are unavailable. serving_buffer_pool.
    dram_statistics's own contract is 'a diagnostic, never a gate': unavailable
    statistics here fall through to packing exactly as the reserve check being off
    would, rather than refusing to ever pack in an environment (or a host test)
    that cannot read the allocator at all."""
    from serving_buffer_pool import dram_statistics

    tensor = getattr(device, 'history', None)
    if tensor is None:
        return None
    statistics = dram_statistics(device.operations, tensor)
    if isinstance(statistics, dict):
        return None
    return min(chip['largest_free'] for chip in statistics)


def dram_reading(device):
    """S2: the smallest free and the smallest largest-contiguous-free DRAM block across this device's chips,
    dict(free=, largest_free=), or None when the statistics are unavailable (dram_headroom's contract)."""
    from serving_buffer_pool import dram_statistics

    tensor = getattr(device, 'history', None)
    if tensor is None:
        return None
    statistics = dram_statistics(device.operations, tensor)
    if isinstance(statistics, dict):
        return None
    return dict(free=min(chip['free'] for chip in statistics),
                largest_free=min(chip['largest_free'] for chip in statistics))


def capture_headroom(device, estimate, reserve=True):
    """(short, reading) for a fresh capture of `estimate` bytes, plus the packed reserve unless `reserve` is False
    (the pair release's threshold is a headroom already): short is () when it fits or the statistics cannot be read
    (reading None: a diagnostic, never a gate).

    With QWEN_FAST_EXTENT_REPLAY unset this is the rule every profile before S2 runs, call for call and in the same
    order (the reserve is read only once the headroom is): the largest free block (dram_headroom) against the need,
    short ('largest_free',) below it, reading dict(largest_free=). Under it (the S2 profiles) it is the admission's
    split (serving_prefill_admission.split_short; gate v79, run 36368363993, whose engines landed in the holes
    departed users left): the free less the stranded bytes against the need, and the largest block against the
    reserve plus the largest buffer; reading dict(free=, largest_free=). v79's quad builds read 1.515, 1.272 and
    1.514 GB free beside 1322.8, 1079.7 and 1320.7 MB blocks. The old rule needs a 740.3 MB block for a quad
    (450 MiB and the 256 MiB reserve), so a fourth user admitted beside a 1079.7 MB block could take only 339 MB of
    it before its quad is refused. The split needs 740.3 MB of the free less the stranded 300 MB (about 1.21 GB
    there) and a 396.4 MB block, so the quad keeps forming whether the new engine lands in the holes (five
    replacement engines in v79, v70 and v71 did) or takes up to 683 MB of the block. For a pair (354.7 MB) the
    split is no looser than the old rule: it adds the free term, and its block is 396.4 MB rather than 354.7."""
    if not extent_memory_points():
        headroom = dram_headroom(device)
        if headroom is None:
            return (), None
        need = estimate + (dram_reserve_bytes() if reserve else 0)
        return (() if headroom >= need else ('largest_free',)), dict(largest_free=headroom)
    reading = dram_reading(device)
    if reading is None:
        return (), None
    import serving_prefill_admission

    packed_reserve = dram_reserve_bytes()
    need = estimate + (packed_reserve if reserve else 0)
    return serving_prefill_admission.split_short(reading['free'], reading['largest_free'], need,
                                                 packed_reserve), reading


def audit_enabled(environ=None):
    return (os.environ if environ is None else environ).get(AUDIT_FLAG) == '1'


def singles_enabled(environ=None):
    """SINGLES_LINE: QWEN_FAST_EXTENT_REPLAY=1 (the S2 profiles) and QWEN_FAST_PACKED_AUDIT=1. Off, the log is
    exactly today's."""
    environ = os.environ if environ is None else environ
    return environ.get(EXTENT_FLAG) == '1' and audit_enabled(environ)


def unpacked_reason(device_a, device_b):
    """None when packable(device_a, device_b); else why the pair drafts singly: 'ramp' while either member's
    history is below the steady-state context, 'unpackable' for any other refusal of packable()."""
    if packable(device_a, device_b):
        return None
    if (getattr(device_a, 'history_rows', None) != PACKED_CONTEXT
            or getattr(device_b, 'history_rows', None) != PACKED_CONTEXT):
        return 'ramp'
    return 'unpackable'


def check_pair_buckets(trace, pair):
    """Raise AssertionError when a pair trace holds a bucket at any context but PAIR_BUCKET_CONTEXT (S2 W12)."""
    off = sorted(tuple(key) for key in (getattr(trace, 'buckets', None) or {}) if tuple(key) != PAIR_BUCKET_CONTEXT)
    if off:
        raise AssertionError('[PACKED-PROPOSE] pair %s holds buckets at contexts %s: a pair bucket exists only at %s '
                             '(packable() gates both users to history_rows == %d)'
                             % (list(pair), off, PAIR_BUCKET_CONTEXT, PACKED_CONTEXT))


def audit_log(message, **values):
    """One loguru INFO line, brace-formatted; plain print where loguru is absent
    (host tests) - dflash_device.pindiag's own pattern."""
    try:
        from loguru import logger
    except ImportError:
        print(message.format(**values), flush=True)
        return
    logger.info(message, **values)


def _committed_kv_history(device):
    """The exact precondition dflash_device.DFlashDevice.execute_proposal itself
    checks before accepting ANY packed cached_history for this device
    (dflash_device.py:661-662): `self.kv_history is None or len(cached_history) !=
    len(self.layers)` raises 'Every prepared learned layer requires a committed K/V
    cache'. There is no separate kv_history.committed flag anywhere in draft_kv_history.
    py or dflash_device.py - DraftKVHistory.active is populated once, at construction,
    from the prefill features, not by any later commit, so this is a structural check
    (kv_history exists and holds one active bank per prepared draft layer), not a
    timing one. Checked here so a device this admits is a device execute_proposal's
    OWN guard would structurally accept too - though see PAIR_FALLBACK_LINE and
    PackedProposalCoordinator.prepare(): run 35581352016 hit that same guard anyway,
    on hardware, for a pair both members of which pass every check in this function."""
    kv_history = getattr(device, 'kv_history', None)
    if kv_history is None:
        return False
    active = getattr(kv_history, 'active', None)
    layers = getattr(device, 'layers', None)
    return active is not None and layers is not None and len(active) == len(layers)


def packable(device_a, device_b):
    """Both paired devices at the permanent steady-state context, both running the
    qualified native-proposal path, both carrying a committed K/V history
    (_committed_kv_history, the same precondition execute_proposal itself checks),
    and both carrying a real single-user proposal_capture (QWEN_FAST_EAGER_PROPOSAL
    off) for _PackedCaptureView to fall through to - a device with none
    (proposal_capture is None: DFlashDevice.prepare_device already declines it
    unconditionally) must never be wrapped, or a later round that falls back to its
    single-user path (its partner finished, breaking the pair) would delegate
    has_pending()/finish()/propose() onto None and crash. draft_kv_history.
    DraftKVHistory requires history_rows == min(position, 2048), so history_rows ==
    2048 is monotonic and permanent once reached - this never flips back to False for
    a pair that has started packing.

    Passing every check here is NOT a guarantee execute_proposal will accept the
    pack - PackedProposalCoordinator.prepare() treats a pair's first traced execution
    as fallible regardless, and falls the pair back to single-user for that one round
    on any exception (PAIR_FALLBACK_LINE)."""
    return (getattr(device_a, 'history_rows', None) == PACKED_CONTEXT
            and getattr(device_b, 'history_rows', None) == PACKED_CONTEXT
            and getattr(device_a, 'native_proposal_attention', False)
            and getattr(device_b, 'native_proposal_attention', False)
            and _committed_kv_history(device_a)
            and _committed_kv_history(device_b)
            and getattr(device_a, 'proposal_capture', None) is not None
            and getattr(device_b, 'proposal_capture', None) is not None)


class _PackedCaptureView:
    """Transparent proxy for device.proposal_capture, installed on each half of a
    packed pair for the rounds it packs. Presents has_pending(seed)/finish(count)
    over the shared PreparedPackedDFlashProposal when THIS round's packed
    prepare_device() is still pending, and falls through to the device's own
    original single-user capture for anything else - a seed this round's pack did
    not prepare, or any other proposal_capture method DFlashDevice.propose()/close()
    reaches. has_pending(seed) and the finish(count) that follows it are always
    called back to back with nothing in between (DFlashDevice.propose()), so caching
    which side matched between the two calls is safe."""

    def __init__(self, trace, which, original):
        self._trace, self._which, self._original = trace, which, original
        self._matched = False

    def has_pending(self, seed):
        self._matched = self._trace.has_pending(self._which, seed)
        if self._matched:
            return True
        original_has_pending = getattr(self._original, 'has_pending', None)
        return bool(callable(original_has_pending) and original_has_pending(seed))

    def finish(self, count):
        if self._matched:
            self._matched = False
            return self._trace.finish(self._which, count)
        return self._original.finish(count)

    def propose(self, seed, count):
        return self._original.propose(seed, count)

    def prepare_device(self, seed):
        return self._original.prepare_device(seed)

    def discard_pending(self):
        # The shared trace's own pending is this pair's, not just this side's -
        # PackedProposalCoordinator.prepare() discards it directly (both sides) on a
        # phase-A failure, bypassing this proxy entirely, so this only ever needs to
        # forward to the per-device single-user capture underneath.
        original_discard = getattr(self._original, 'discard_pending', None)
        if callable(original_discard):
            original_discard()

    def close(self):
        # The shared trace outlives any one device - PackedProposalCoordinator owns
        # closing it (at hook teardown, or when a pair is retired) - so a per-device
        # close() must only close this device's OWN single-user capture underneath.
        # _original is None whenever PackedProposalCoordinator._release_single_user
        # has freed it and no fallback has needed it back since (this pair packed
        # every round from then until the request itself ended) - nothing to close.
        if self._original is not None:
            return self._original.close()


def _install(device, trace, which):
    """Wrap device.proposal_capture in a _PackedCaptureView for `which` side of
    `trace`, unwrapping any previous view first so repeated rounds never nest -
    always exactly one proxy layer over the true single-user PreparedDFlashProposal."""
    current = device.proposal_capture
    original = current._original if isinstance(current, _PackedCaptureView) else current
    device.proposal_capture = _PackedCaptureView(trace, which, original)


class PackedProposalCoordinator:
    """Owns one PreparedPackedDFlashProposal per DRAFT_PAIRS fixed slot pair for
    the life of this object (one per FastWorkerHook, created lazily on first use),
    rebuilding a pair's trace only when the devices occupying that pair's slots
    change identity (a finished request releases its slot, a new one acquires it) -
    never once per round. A slot pair with only one member active this round, or not
    yet at the packable steady-state context, runs each member's OWN
    device.prepare_device() unchanged, exactly as prepare_pipelined_drafts always
    has; this coordinator never touches a bridge outside a full, packable pair.

    self.pairs[pair] holds (device_a, device_b, trace, released): `released` is True
    once the pair's first successful capture has freed both devices' own single-user
    PreparedDFlashProposal (_release_single_user) - run 35585107688: a packed pair's
    placeholder set duplicates the two single-user traced-proposal placeholder sets,
    so keeping all three allocated at once is pure DRAM duplication once the pair is
    doing the same job. _ensure_single_user rebuilds a released capture, once, the
    first time ANY later round needs the single-user fallback again (the pair's own
    capture failing, or the pair breaking up when a partner finishes)."""

    def __init__(self):
        # The draft book (2c): when one is registered its dicts ARE this coordinator's, so a trace is listed once and closed once.
        self._book = draft_book()
        self.pairs = {} if self._book is None else self._book.pairs
        self.rounds = 0
        # QWEN_FAST_QUAD_DRAFT (quad_draft.py): (devices, trace, released) of the quad over slots 0-3, its
        # consecutive failures, whether it gave up, and how many rounds it served. Read only with the flag on.
        self.quad = None
        self.quad_failures = 0
        self.quad_disabled = False
        self.quad_rounds = 0
        # QWEN_FAST_QUAD_DRAFT_BLOCKS: the per-block quads' own state, keyed by slot tuple - (devices, trace, released) as self.quad, the
        # consecutive failures, and the blocks given up (slots -> reason). Empty, and nothing reads them, with the flag off.
        self.quad_blocks = {} if self._book is None else self._book.quad_blocks
        self.quad_block_failures = {}
        self.quad_blocked = {}
        # QWEN_FAST_OCTO_DRAFT: (devices, trace, released) of the eight-seat pass over slots 0-7, its consecutive failures, the reason it was blocked (None: not), and how many rounds
        # it served. Read only with the flag on.
        self.octo = None
        self.octo_failures = 0
        self.octo_blocked = None
        self.octo_rounds = 0
        # Engine reuse (QWEN_FAST_PARKED_ENGINES=1, no book; empty otherwise): {pair group, 'quad' or the block's slots: the members' rebind_generation
        # when the trace was captured}.
        self.generations = {}
        # QWEN_FAST_DRAFT_SINGLES_AUDIT: the rounds that ran a batched group (the audit's own count).
        self.singles_audit_candidates = 0

    @property
    def book(self):
        """The draft book while it is still registered, else None: the kill switch unregisters it, and from then on this coordinator is today's (its
        dicts, which were the book's, are empty and its own)."""
        return self._book if self._book is not None and self._book in _DRAFT_BOOKS else None

    def close(self):
        if self.book is not None:
            # 2c: the traces are the book's, bound to the slots for the process. A hook's close drops the views its devices wear and resets the
            # per-hook give-up state (a drain gives a blocked quad another chance); the traces stay.
            self.book.drop_views()
            self.quad_failures = 0
            self.quad_disabled = False
            self.quad_block_failures.clear()
            self.quad_blocked.clear()
            self.generations.clear()
            return
        for _, _, trace, _ in self.pairs.values():
            trace.close()
        self.pairs.clear()
        if self.quad is not None:
            self.quad[1].close()
            self.quad = None
        for state in self.quad_blocks.values():
            state[1].close()
        self.quad_blocks.clear()
        if self.octo is not None:
            self.octo[1].close()
            self.octo = None
        self.generations.clear()

    # -- every close of a pair or block-quad trace goes through these two: with the draft book registered they retire through it (closed once,
    # -- unlisted, the surviving members' views unwrapped), without it they close the trace as the code always did.
    def _close_pair(self, group):
        if self.book is not None:
            self.book.retire('pair', group)
        else:
            # Close first, then unlist: a close that raises leaves the entry listed, as the code always did.
            self.pairs[group][2].close()
            del self.pairs[group]
        self.generations.pop(group, None)

    def _close_quad_block(self, slots):
        if self.book is not None:
            self.book.retire('quad', slots)
        else:
            self.quad_blocks[slots][1].close()
            del self.quad_blocks[slots]
        self.generations.pop(slots, None)

    def _rebound(self, key, devices):
        """Engine reuse, E1: whether a trace recorded under `key` was captured for an earlier request of one of `devices` - its generation has
        moved on since. Always False with QWEN_FAST_PARKED_ENGINES off, and with the draft book (whose traces outlive requests by design)."""
        return self.book is None and parked_engines_on() and self.generations.get(key) != generations(devices)

    def _note_generations(self, key, devices):
        if self.book is None and parked_engines_on():
            self.generations[key] = generations(devices)

    def release_closed(self):
        """S2 W6c: close, now, every proposal trace a closed device was captured for - the quad if any of its
        four devices has closed (_retire_quad's test) and every pair trace bound to one. Called from
        FastWorkerHook.detach (serving_worker_hook.release_dead_proposals, QWEN_FAST_EXTENT_REPLAY=1 only) right
        after the departing request's device closed. Otherwise the quad (0.3-0.45 GB per chip, est.) is retired
        only in the next draft round's _prepare_quad and a stale pair only when its slots re-form, and a
        replacement's prefill and engine build, which run first, see about 0.5 GB less (s2-design.md section 3.1).
        Live traces are kept, and so is each surviving device's released single-user capture: _ensure_single_user
        rebuilds it the first round it is needed. Logs RELEASED_LINE every time it runs. Under QWEN_FAST_PARKED_ENGINES=1 (no book) a member
        rebound since the capture counts as closed (_rebound); with the book, a closed member's traces are retired through it."""
        quad = 0
        if self.quad is not None and (any(getattr(device, 'closed', False) for device in self.quad[0])
                                      or self._rebound('quad', self.quad[0])):
            self.quad[1].close()
            self.quad = None
            self.generations.pop('quad', None)
            quad = 1
        for slots, state in list(self.quad_blocks.items()):
            if any(getattr(device, 'closed', False) for device in state[0]) or self._rebound(slots, state[0]):
                self._close_quad_block(slots)
                quad += 1
        pairs = []
        for group, (device_a, device_b, trace, _) in list(self.pairs.items()):
            if (getattr(device_a, 'closed', False) or getattr(device_b, 'closed', False)
                    or self._rebound(group, (device_a, device_b))):
                self._close_pair(group)
                pairs.append(list(group))
        if self.octo is not None and (any(getattr(device, 'closed', False) for device in self.octo[0]) or self._rebound('octo', self.octo[0])):
            self._close_octo()
        audit_log(RELEASED_LINE, quad=quad, pairs=pairs)
        return dict(quad=quad, pairs=pairs)

    def release_parked(self, device):
        """Engine reuse, E1 (QWEN_FAST_PARKED_ENGINES=1, no book; serving_parked_engines.after_park, from FastWorkerHook.detach): close, now,
        every proposal trace `device` was captured into - the quad when it is one of the four, each block quad it is in, and each pair it is
        half of - as release_closed closes a closed device's. A parked device stays open for its slot's next request, so `closed` never marks
        these dead; left alone they would hold their DRAM until the slots re-form (and a rebound member's generation retires them then). The
        partners keep their released singles, which _ensure_single_user rebuilds the first round each drafts alone. Logs PARKED_RELEASED_LINE.
        With the book nothing is released: the traces are the slots' for the process."""
        if self.book is not None:
            return dict(quad=0, pairs=[])
        quad = 0
        if self.quad is not None and any(member is device for member in self.quad[0]):
            self.quad[1].close()
            self.quad = None
            self.generations.pop('quad', None)
            quad = 1
        for slots, state in list(self.quad_blocks.items()):
            if any(member is device for member in state[0]):
                self._close_quad_block(slots)
                quad += 1
        pairs = []
        for group, (device_a, device_b, trace, _) in list(self.pairs.items()):
            if device_a is device or device_b is device:
                self._close_pair(group)
                pairs.append(list(group))
        if self.octo is not None and any(member is device for member in self.octo[0]):
            self._close_octo()
        audit_log(PARKED_RELEASED_LINE, slot=getattr(getattr(device, 'pool_slot', None), 'index', None), quad=quad, pairs=pairs)
        return dict(quad=quad, pairs=pairs)

    def _cached_trace(self, pair, device_a, device_b):
        """The pair's already-captured trace if the SAME two devices still occupy
        it and neither has closed, else None - a fresh capture (and the DRAM
        reserve check that must gate one) is needed either way."""
        cached = self.pairs.get(pair)
        if cached is None:
            return None
        old_a, old_b, trace, released = cached
        if old_a is device_a and old_b is device_b and not device_a.closed and not device_b.closed:
            if self._rebound(pair, (device_a, device_b)):
                return None
            return trace
        return None

    def _trace_for(self, pair, device_a, device_b):
        trace = self._cached_trace(pair, device_a, device_b)
        if trace is not None:
            return trace
        if pair in self.pairs:
            self._close_pair(pair)
        from dflash_proposal_trace import PreparedPackedDFlashProposal

        trace = PreparedPackedDFlashProposal(device_a, device_b)
        self.pairs[pair] = (device_a, device_b, trace, False)
        self._note_generations(pair, (device_a, device_b))
        return trace

    def _release_single_user(self, device):
        """Free device's own single-user PreparedDFlashProposal (reached either
        directly, if no pair has ever formed for it, or through the _PackedCaptureView
        _install put in its place) once its pair has captured successfully - see the
        class docstring for why keeping both is pure duplication. Marks the device so
        _ensure_single_user knows this - and ONLY this - is why its capture (or its
        view's _original) reads None afterwards; a device whose capture is None for
        any OTHER reason (QWEN_FAST_EAGER_PROPOSAL, or one this coordinator never
        touched at all) must never be treated as needing a rebuild. Idempotent: a
        device whose capture is already released is left alone.

        Under QWEN_FAST_DRAFT_SINGLES_AUDIT nothing is released: the audit replays every member's own single-user capture. With the draft book
        (2c) nothing is released either: the singles are kept for the process, so no rebind and no admission ever rebuilds one."""
        if _singles_audit_on() or self.book is not None:
            return
        capture = device.proposal_capture
        if isinstance(capture, _PackedCaptureView):
            if capture._original is not None:
                capture._original.close()
                capture._original = None
                device._packed_capture_released = True
        elif capture is not None:
            capture.close()
            device.proposal_capture = None
            device._packed_capture_released = True

    def _ensure_single_user(self, device):
        """Rebuild device's own single-user PreparedDFlashProposal if
        _release_single_user freed it (device._packed_capture_released, set ONLY by
        that method - never inferred from proposal_capture being None alone, which
        is also the ordinary QWEN_FAST_EAGER_PROPOSAL state this coordinator must
        leave untouched) - reached either as device.proposal_capture directly (no
        pair ever formed for this device) or as the _original a _PackedCaptureView
        falls through to (a pair released it, then broke up or its capture failed
        this round) - device.prepare_device() delegates to whichever of the two it
        finds, and DFlashDevice.propose()'s own fallback (self.proposal_capture.
        propose(...), reached through the SAME view when has_pending() answers
        False) would otherwise run onto None. max_new_tokens=1 reproduces the SAME
        single 2048-row bucket the original ladder would have narrowed to by now
        regardless of its real value - dflash_proposal_inputs.proposal_contexts(
        position, max_new_tokens) returns exactly (2048,) whenever position >= 2048
        (steady state, the only state packable() ever admits a device in) for ANY
        max_new_tokens satisfying its own bound (1 <= max_new_tokens <= 262144 -
        position - 32) - the real original value is not stored anywhere on
        DFlashDevice and is not needed to reproduce its ladder."""
        if not getattr(device, '_packed_capture_released', False):
            return False
        capture = device.proposal_capture
        view = capture if isinstance(capture, _PackedCaptureView) else None
        from dflash_proposal_trace import PreparedDFlashProposal

        # S2 W6d: this rebuild checks no headroom, so the ledger reads the allocator either side of it.
        ledger_token = ledger_before('single', single_capture_bytes(),
                                     'slot=%s' % getattr(getattr(device, 'pool_slot', None), 'index', None))
        try:
            if parked_engines_on():
                # Engine reuse: a parked device is rebound at any position, so the one 2048 bucket its builds capture, never the narrower one the
                # position alone would give below 2048.
                from serving_request_factory import single_proposal_bucket

                with single_proposal_bucket():
                    rebuilt = PreparedDFlashProposal(device, max_new_tokens=1)
            else:
                rebuilt = PreparedDFlashProposal(device, max_new_tokens=1)
        finally:
            ledger_after(ledger_token)
        if view is not None:
            view._original = rebuilt
        else:
            device.proposal_capture = rebuilt
        device._packed_capture_released = False
        if audit_enabled():
            audit_log(RECAPTURE_LINE, slot=getattr(getattr(device, 'pool_slot', None), 'index', None))
        return True

    def prepare(self, bridges, packed_round=None, while_waiting=None, after_reads=None, octo_draft=False):
        """Phase A, packed variant: pair eligible bridges by fixed pool slot, run one
        traced pass per full packable pair, prepare_device() unchanged for anything
        left over (an unpaired bridge, a lone survivor of a broken pair, a pair not
        yet past the ramp), then one shared fence covering every device this round
        actually touched - prepare_pipelined_drafts' own contract, so phase B's
        per-bridge drafts() loop needs no changes at all to read any of it back.

        A device that raises while being prepared fails the round exactly as its own
        drafts() call would have, but only after every other already-prepared
        device's enqueued work is fenced and its pending discarded - mirroring
        prepare_pipelined_drafts' own exception path.

        `packed_round` (QWEN_FAST_PAIRS_PACKED_ONLY only; the hook passes nothing
        otherwise): False when the packed step's policy serves this round sequentially,
        and then, with the flag on, no pair forms - every member prepares on its own
        (unpair_groups).

        `while_waiting` (round-fence plan H1a, verify_prestage.WhileWaiting; the hook passes
        it only when a flag is on and the coming round is the block's packed round): called
        just before this round's one fence, after every pair and single is enqueued, so its
        host work hides under their device time - and only when there is a fence to follow
        it. An Exception it raises is handed to its `drop` and never fails the round. Right
        after the fence its `fenced` is called (the fence has drained everything enqueued
        before it).

        `after_reads` (round-fence plan H2, early_draft.EarlyDraft.coordinator_options; passed only
        under QWEN_FAST_GDN_AFTER_PAIRS): called once every device this round prepared has been read
        back - inside select_round, after the pairs' collect and before the selection - so the packed
        block's deferred GDN commit traces queue behind the pairs and their readback, not ahead of
        them. Only when the batched pairs cover every prepared device (QWEN_FAST_ROUND_B1, no single
        whose phase-B finish() still reads); otherwise it is not called here and the early draft
        flushes at its end.

        `octo_draft` (QWEN_FAST_OCTO_DRAFT; the hook passes it, True, only for a round the packed step planned for the octo block and only with the flag on): try the
        one eight-seat, eight-row pass (_prepare_octo) before the quads; a round it does not serve (fewer than eight live and packable seats, a refusal, a failure) goes on
        to the quads and pairs below exactly as it would have."""
        from dflash_packed_proposal import pair_slots
        from serving_worker_hook import phase, pipelined_device
        import round_host
        import verify_prestage

        # tp4/round-host (ledger): where the draft's host time goes - the quads' launch, the window, the fence. Flags unset, no-ops.
        ledger = round_host.ledger
        self.rounds += 1
        round_number = self.rounds
        entries = []
        for bridge in bridges:
            device = pipelined_device(bridge)
            if device is None:
                continue
            slot = getattr(getattr(device, 'pool_slot', None), 'index', None)
            entries.append(dict(bridge=bridge, device=device, slot=slot, seed=bridge.request.session.seed))
        by_slot = {entry['slot']: entry for entry in entries if entry['slot'] is not None}
        unpaired = [entry for entry in entries if entry['slot'] is None]
        groups = pair_slots(by_slot) if by_slot else []
        # S2 W12: the slots a QWEN_FAST_PAIRS_PACKED_ONLY split leaves on their own this round.
        split = set()
        if packed_round is False and pairs_packed_only_enabled():
            split = {slot for group in groups if len(group) == 2 for slot in group}
            groups = unpair_groups(groups, round_number)

        prepared, fence = [], None
        # S2 W12: (slot, reason) of every device this round prepares on its own capture (SINGLES_LINE).
        singles = []
        pair_labels, pair_ms = [], []
        # QWEN_FAST_ROUND_B1 (C1): the traces this round prepared, selected together after
        # the fence below. None with the flag off, and nothing then reads it.
        batched = [] if os.environ.get('QWEN_FAST_ROUND_B1') == '1' else None
        # QWEN_FAST_QUAD_DRAFT: this round's quad (_prepare_quad), None in every round it does not serve.
        quad = None
        # QWEN_FAST_QUAD_DRAFT_BLOCKS: this round's per-block quads (_prepare_quad_blocks), empty in every round it does not serve.
        block_quads = []
        # QWEN_FAST_OCTO_DRAFT: this round's eight-seat pass (_prepare_octo), None in every round it does not serve.
        octo = None
        # QWEN_FAST_DRAFT_SINGLES_AUDIT: this round's audit (draft_singles_audit.Round), None in every round it does not run.
        singles_audit = None

        def prepare_single(entry, reason):
            singles.append((entry['slot'], reason))
            device = entry['device']
            # A no-op unless a pair had already released this device's own single-
            # user capture (PackedProposalCoordinator._release_single_user) and this
            # is the first round since that needs it back - see _ensure_single_user's
            # own docstring for why this must run before prepare_device() below.
            self._ensure_single_user(device)
            if device.prepare_device(entry['seed']):
                prepared.append(device)
                return True
            return False

        try:
            ledger.mark('quads0')
            octo = self._prepare_octo(groups, by_slot, round_number, batched) if octo_draft else None
            if octo is None and self.octo is not None:
                # An earlier octo round's proposal that nothing consumed (the round changed shape after its drafts): drop it now, so no device still wearing its view is answered
                # by it - the view matches on the seed alone, and a seven-proposal pass cannot answer a wider ticket. A no-op when nothing is pending.
                self.octo[1].discard_pending()
            if octo is not None:
                # One pass drafted all eight seats at eight rows: no quad and no pair runs this round.
                prepared.extend(octo['devices'])
                fence = octo['fence']
                groups = []
                ledger.mark('quads1')
            elif os.environ.get(QUAD_DRAFT_FLAG, '0') != '0' and quad_blocks_requested():
                # One quad per block of four slots; the pairs of a block that did not form stay with `groups`.
                # tp4/fx-wph (QWEN_FAST_TP4_HOSTGAP_LOG, _PROBE; hostgap_instr, imported only with them): the launches as one span and, every 16th eight-live round, a timed
                # synchronize right after them (2Q measured directly; the round is a probe round). Both unset, this is the call and the mark.
                probe = verify_prestage.hostgap_log_enabled() and os.environ.get('QWEN_FAST_TP4_HOSTGAP_PROBE', '0') != '0'
                if probe:
                    import hostgap_instr

                    launch_started = hostgap_instr.begin_launch()
                with verify_prestage.hostgap_span('launch'):
                    block_quads, groups = self._prepare_quad_blocks(groups, by_slot, round_number, batched, prepared)
                ledger.mark('quads1')
                if block_quads:
                    fence = block_quads[0]['fence']
                    if probe and hostgap_instr.probe_due(len(block_quads)):
                        hostgap_instr.sync_probe(fence, round_number, len(block_quads), launch_started)
            elif os.environ.get(QUAD_DRAFT_FLAG, '0') != '0':
                # Q4: all four live and packable - one 64-row pass, and no pair runs. Anything else (or a quad
                # that fails, refuses or gave up) leaves `groups` to today's pairs below.
                quad = self._prepare_quad(groups, by_slot, round_number, batched)
                if quad is not None:
                    prepared.extend(quad['devices'])
                    fence = quad['fence']
                    groups = []
            for group in groups:
                if len(group) == 2:
                    slot_a, slot_b = group
                    entry_a, entry_b = by_slot[slot_a], by_slot[slot_b]
                    device_a, device_b = entry_a['device'], entry_b['device']
                    single_reason = unpacked_reason(device_a, device_b)
                    if single_reason is None:
                        fresh_build = self._cached_trace(group, device_a, device_b) is None
                        headroom_ok = True
                        if fresh_build:
                            # Only a FRESH capture allocates new placeholder buffers -
                            # replaying an already-built trace does not - so the
                            # reserve check only ever gates the first round a pair
                            # forms, never every round after (run 35585107688). Under
                            # the S2 flag the check is the admission's split
                            # (capture_headroom, gate v79).
                            short, _ = capture_headroom(device_a, pair_capture_bytes())
                            if short:
                                headroom_ok = False
                                single_reason = 'dram_reserve'
                                if audit_enabled():
                                    audit_log(PAIR_FALLBACK_LINE, pair=[slot_a, slot_b], fallback='dram_reserve')
                        if headroom_ok:
                            started = time.perf_counter() if audit_enabled() else None
                            trace = self._trace_for(group, device_a, device_b)
                            from dflash_proposal_trace import pair_mask_audit_enabled

                            if pair_mask_audit_enabled():
                                # QWEN_FAST_PAIR_MASK_AUDIT (M0): this round's number for the
                                # trace's '[PACKED-PROPOSE] mask round=' lines.
                                trace.round_number = round_number
                            ids = '%s,%s' % (entry_a['bridge'].request.session.request_id,
                                             entry_b['bridge'].request.session.request_id)
                            # S2 W6d: a fresh pair capture allocates; the ledger reads either side of it.
                            ledger_token = (ledger_before('pair', pair_capture_bytes(),
                                                          'slots=%s,%s' % (slot_a, slot_b)) if fresh_build else None)
                            try:
                                ready = phase('propose_pair', ids,
                                              lambda: trace.prepare_device(entry_a['seed'], entry_b['seed']))
                            except Exception as failure:
                                # The four-user gate must never lose the engine to a
                                # proposal-path refusal (run 35581352016: packable() passed
                                # every host-side check yet execute_proposal still raised on
                                # the pair's first traced execution) - this pair falls back
                                # to single-user for THIS round only. _bucket()'s own
                                # exception handling already removed any partially-built
                                # bucket (and every placeholder it uploaded, run
                                # 35585107688) before re-raising, so the trace object itself
                                # stays valid; discard_pending() is a safe no-op here
                                # (prepare_device() never reached setting self._pending
                                # before its own _bucket() call raised) but is called
                                # anyway, matching every other failure path in this module.
                                # packable() is re-evaluated fresh next round - a permanent-
                                # shaped failure (see packable()'s own docstring) then just
                                # falls back every round, harmlessly, rather than being
                                # retried once and blacklisted.
                                trace.discard_pending()
                                if audit_enabled():
                                    audit_log(PAIR_FALLBACK_LINE, pair=[slot_a, slot_b],
                                              fallback='%s: %s' % (type(failure).__name__, str(failure)[:160]))
                                ready = False
                                single_reason = 'failure'
                            else:
                                single_reason = 'declined'
                            ledger_after(ledger_token)
                        else:
                            ready = False
                        if ready:
                            _install(device_a, trace, 'a')
                            _install(device_b, trace, 'b')
                            prepared.extend((device_a, device_b))
                            if batched is not None:
                                batched.append(([slot_a, slot_b], trace))
                            if fence is None:
                                fence = (device_a.operations, device_a.mesh)
                            # S2 W12: after the pair is among `prepared` and the fence named, so a failure here
                            # takes the exception path below (fence, the shared trace's pending dropped once).
                            check_pair_buckets(trace, group)
                            if started is not None:
                                pair_labels.append([slot_a, slot_b])
                                pair_ms.append((time.perf_counter() - started) * 1000)
                            cached = self.pairs.get(group)
                            if cached is not None and not cached[3]:
                                self._release_single_user(device_a)
                                self._release_single_user(device_b)
                                self.pairs[group] = (cached[0], cached[1], cached[2], True)
                            continue
                    for entry in (entry_a, entry_b):
                        if prepare_single(entry, single_reason) and fence is None:
                            fence = (entry['device'].operations, entry['device'].mesh)
                else:
                    entry = by_slot[group[0]]
                    if prepare_single(entry, 'split' if group[0] in split else 'absent') and fence is None:
                        fence = (entry['device'].operations, entry['device'].mesh)
            for entry in unpaired:
                if prepare_single(entry, 'unslotted') and fence is None:
                    fence = (entry['device'].operations, entry['device'].mesh)
        except BaseException:
            if fence is not None:
                fence[0].synchronize_device(fence[1])
            # A packed pair's two devices share ONE trace - discard it once, not once
            # per device, or the second discard_pending() call raises into an
            # exception handler that is already unwinding one.
            discarded_traces = set()
            for device in prepared:
                capture = device.proposal_capture
                if isinstance(capture, _PackedCaptureView):
                    if id(capture._trace) not in discarded_traces:
                        discarded_traces.add(id(capture._trace))
                        capture._trace.discard_pending()
                else:
                    discard = getattr(capture, 'discard_pending', None)
                    if callable(discard):
                        discard()
            raise
        singles_module = None
        if batched and _singles_audit_on():
            # QWEN_FAST_DRAFT_SINGLES_AUDIT: every member of every batched group also drafts on its own single-user capture,
            # with the seed the batched pass got, enqueued behind it and before the round's one fence.
            import draft_singles_audit as singles_module

            self.singles_audit_candidates += 1
            if singles_module.selected(self.singles_audit_candidates):
                singles_audit = singles_module.start(batched, by_slot, round_number, self._ensure_single_user)
        if fence is not None and while_waiting is not None:
            ledger.mark('window0')
            with verify_prestage.hostgap_span('window'):
                run_while_waiting(while_waiting)
            ledger.mark('window1')
        if fence is not None:
            ledger.mark('fence0')
            with verify_prestage.hostgap_span('fence'):
                fence[0].synchronize_device(fence[1])
            ledger.mark('fence1')
            if while_waiting is not None:
                note_fenced(while_waiting)
        if singles_audit is not None:
            # After the fence and before anything else is enqueued: the raw reads of the batched traces and the singles.
            singles_module.read(singles_audit)
        try:
            if batched:
                if after_reads is not None and reads_covered(batched, prepared):
                    # Round-fence plan H2: the deferred GDN commits go in after the pairs' readback.
                    select_round(batched, prepared, round_number, after_collect=after_reads)
                else:
                    select_round(batched, prepared, round_number)
        except BaseException:
            if singles_audit is not None:
                singles_audit.close()
            raise
        if quad is not None:
            # QWEN_FAST_QUAD_DRAFT_AUDIT: the pair traces replayed beside the quad, compared after the selection.
            quad['trace'].run_audit(round_number)
        for formed in block_quads:
            formed['trace'].run_audit(round_number)
        if singles_audit is not None:
            singles_module.finish(singles_audit)
        if audit_enabled() and pair_labels:
            audit_log(AUDIT_LINE, round=round_number, pairs=pair_labels,
                      propose_ms=['%.1f' % value for value in pair_ms])
        if singles and singles_enabled():
            audit_log(SINGLES_LINE, round=round_number, slots=[slot for slot, _ in singles],
                      reasons=[reason for _, reason in singles])
        return prepared

    # -- QWEN_FAST_OCTO_DRAFT (octo_draft_tp.py) -----------------------------------------------------------
    def _close_octo(self):
        """Close the eight-seat pass's trace and forget it (its DRAM is the pairs' and the engines' again)."""
        slots = ','.join(str(slot) for slot in range(len(self.octo[0])))
        self.octo[1].close()
        self.octo = None
        self.generations.pop('octo', None)
        audit_log(OCTO_RELEASED_LINE, slots=slots)

    def _retire_octo(self, by_slot):
        """Close the pass's trace once a device it was built for has closed or a slot holds another device: it can never replay again. A device merely absent this round keeps it."""
        state = self.octo
        if state is None:
            return
        import octo_draft_tp

        devices = state[0]
        if any(device.closed for device in devices) or any(
                slot in by_slot and by_slot[slot]['device'] is not device
                for slot, device in zip(octo_draft_tp.SLOTS, devices)) or self._rebound('octo', devices):
            self._close_octo()

    def _block_octo(self, round_number, reason):
        """Give up on the pass for the process: close its trace and log DISABLED_MARKER (the gate fails on it). The quads keep serving."""
        import octo_draft_tp

        if self.octo_blocked is not None:
            return
        self.octo_blocked = reason
        if self.octo is not None:
            self._close_octo()
        audit_log('{marker} round={round} failures={failures} reason={reason}', marker=octo_draft_tp.DISABLED_MARKER, round=round_number,
                  failures=self.octo_failures, reason=str(reason).replace(' ', '_')[:160])

    def _prepare_octo(self, groups, by_slot, round_number, batched):
        """Prepare the one eight-seat, eight-row pass over slots 0-7, or None for the quads. Engages only when the groups are exactly the four pairs (0, 1) .. (6, 7), every pair is
        packable, octo_draft_tp.refusal finds nothing and it has not given up. A fresh build needs octo_draft_tp.capture_bytes() plus the packed reserve of free DRAM
        (capture_headroom). A failure falls the round back to the quads (FALLBACK_LINE); GIVE_UP_FAILURES in a row block it for the process. On success each device wears a
        view of the pass (the first success releases every single-user capture still held, as the quad's does) and the pass joins the round's batched selection."""
        import octo_draft_tp
        from serving_worker_hook import phase

        octo_draft_tp.enabled()
        self._retire_octo(by_slot)
        if self.octo_blocked is not None or [tuple(group) for group in groups] != list(octo_draft_tp.PAIRS):
            return None
        entries = [by_slot[slot] for slot in octo_draft_tp.SLOTS]
        devices = [entry['device'] for entry in entries]
        if not all(packable(devices[first], devices[second]) for first, second in octo_draft_tp.PAIRS):
            return None
        reason = octo_draft_tp.refusal(devices, batched)
        if reason is not None:
            self._block_octo(round_number, reason)
            return None
        state = self.octo
        trace = state[1] if state is not None and all(old is new for old, new in zip(state[0], devices)) else None
        fresh = trace is None or not trace.buckets
        if fresh:
            short, reading = capture_headroom(devices[0], octo_draft_tp.capture_bytes())
            if short:
                reason = 'dram_reserve:headroom=%d' % reading['largest_free']
                if 'free' in reading:
                    reason += ':free=%d:short=%s' % (reading['free'], '+'.join(short))
                audit_log(octo_draft_tp.FALLBACK_LINE, round=round_number, reason=reason)
                return None
        if trace is None:
            if state is not None:
                state[1].close()
            trace = octo_draft_tp.PreparedOctoDFlashProposal(devices)
            self.octo = (tuple(devices), trace, False)
            self._note_generations('octo', devices)
        seeds = [entry['seed'] for entry in entries]
        ids = ','.join(str(entry['bridge'].request.session.request_id) for entry in entries)
        ledger_token = ledger_before('octo', octo_draft_tp.capture_bytes(), 'slots=' + ','.join(str(slot) for slot in octo_draft_tp.SLOTS)) if fresh else None
        started = time.perf_counter()
        try:
            ready = phase('propose_octo', ids, lambda: trace.prepare_device(seeds))
        except Exception as failure:
            ledger_after(ledger_token)
            trace.discard_pending()
            self.octo_failures += 1
            audit_log(octo_draft_tp.FALLBACK_LINE, round=round_number, reason=('%s:%s' % (type(failure).__name__, str(failure)[:120])).replace(' ', '_'))
            if self.octo_failures >= octo_draft_tp.GIVE_UP_FAILURES:
                self._block_octo(round_number, 'consecutive_failures=%d' % self.octo_failures)
            return None
        ledger_after(ledger_token)
        if not ready:
            return None
        built_ms = octo_draft_tp.elapsed_ms(started)
        self.octo_failures = 0
        self.octo_rounds += 1
        try:
            for which, device in enumerate(devices):
                _install(device, trace, which)
            held = self.octo
            if not held[2]:
                for device in devices:
                    self._release_single_user(device)
                self.octo = (held[0], trace, True)
        except BaseException:
            devices[0].operations.synchronize_device(devices[0].mesh)
            trace.discard_pending()
            raise
        batched.append((list(octo_draft_tp.SLOTS), trace))
        audit_log(octo_draft_tp.ROUND_LINE, round=round_number, built=int(trace.last_built), ms=built_ms)
        return dict(devices=devices, fence=(devices[0].operations, devices[0].mesh), trace=trace)

    # -- QWEN_FAST_QUAD_DRAFT (quad_draft.py) --------------------------------------------------------------
    def _retire_quad(self, by_slot, slots=None):
        """Close the quad trace once a device it was built for has closed or a slot holds another device (a
        finished request's slot reused): it can never replay again, and its capture is 0.3-0.45 GB per chip (est.)
        the pairs need back. A device merely absent this round keeps it. `slots` names a per-block quad
        (QWEN_FAST_QUAD_DRAFT_BLOCKS); None is the quad over slots 0-3 in self.quad, as ever."""
        state = self.quad if slots is None else self.quad_blocks.get(slots)
        if state is None:
            return
        import quad_draft

        devices, trace, _ = state
        if any(device.closed for device in devices) or any(
                slot in by_slot and by_slot[slot]['device'] is not device
                for slot, device in zip(quad_draft.SLOTS if slots is None else slots, devices)) or self._rebound(
                    'quad' if slots is None else slots, devices):
            if slots is None:
                trace.close()
                self.quad = None
                self.generations.pop('quad', None)
            else:
                self._close_quad_block(slots)

    def _disable_quad(self, round_number, reason):
        """Give up on the quad for the process: close its trace and log DISABLED_MARKER (the gate fails on it)."""
        import quad_draft

        if self.quad_disabled:
            return
        self.quad_disabled = True
        if self.quad is not None:
            self.quad[1].close()
            self.quad = None
        audit_log('{marker} round={round} failures={failures} reason={reason}', marker=quad_draft.DISABLED_MARKER,
                  round=round_number, failures=self.quad_failures, reason=str(reason).replace(' ', '_')[:160])

    def _quad_audit_pairs(self, devices, seeds, round_number, pairs=None):
        """QWEN_FAST_QUAD_DRAFT_AUDIT: both pair traces (built here if they are not), prepared with the quad's seeds
        before the round's fence, so they read the same live banks, seeds and RoPE. A reason string when they
        cannot be - the audit line then reads equal=0."""
        import quad_draft
        from dflash_proposal_trace import pair_mask_audit_enabled

        prepared = []
        try:
            for position, group in enumerate(quad_draft.PAIRS):
                # `pairs` (a per-block quad's slot pairs) names the pool slots; `group` is the position within the quad's devices.
                slot_group = group if pairs is None else tuple(pairs[position])
                trace = self._trace_for(slot_group, devices[group[0]], devices[group[1]])
                if pair_mask_audit_enabled():
                    trace.round_number = round_number
                if not trace.prepare_device(seeds[group[0]], seeds[group[1]]):
                    raise ValueError('pair %s declined to prepare' % (slot_group,))
                prepared.append((list(slot_group), trace))
        except Exception as failure:
            for _, trace in prepared:
                trace.discard_pending()
            return 'pairs-unavailable:%s' % type(failure).__name__
        return prepared

    def _prepare_quad_blocks(self, groups, by_slot, round_number, batched, prepared):
        """QWEN_FAST_QUAD_DRAFT_BLOCKS: one _prepare_quad per block of four slots (quad_draft_tp.QUADS), each on its own state. Returns
        (the quads formed this round, in block order, and the groups they leave to the pairs). A quad that forms is appended to
        `prepared` here, at once, so a later block's failure leaves the earlier one's devices where the caller's exception path finds
        them. A refusal the process cannot serve (the value, the width) blocks every quad by name, before any capture."""
        import quad_draft

        reason = quad_blocks_refusal(quad_draft)
        if reason is not None:
            blocks = quad_block_groups(quad_draft) if hasattr(quad_draft, 'QUADS') else ((tuple(quad_draft.SLOTS), ()),)
            for slots, _ in blocks:
                self._block_quad(slots, round_number, reason)
            return [], groups
        formed, remaining = [], list(groups)
        try:
            for slots, pairs in quad_block_groups(quad_draft):
                quad = self._prepare_quad(remaining, by_slot, round_number, batched, slots=slots, pairs=pairs)
                if quad is None:
                    continue
                prepared.extend(quad['devices'])
                formed.append(quad)
                remaining = [group for group in remaining if tuple(group) not in pairs]
        except BaseException:
            # A later block raised before the caller holds a fence: the earlier block's replay may still be running, so fence it here
            # before the caller's handler discards its pending work and releases its transients.
            if formed:
                operations, mesh = formed[0]['fence']
                operations.synchronize_device(mesh)
            raise
        return formed, remaining

    def _block_quad(self, slots, round_number, reason):
        """Give up on one per-block quad for the process: close its trace and log DISABLED_MARKER with its slots (the gate fails on it).
        The other block keeps serving."""
        import quad_draft

        if slots in self.quad_blocked:
            return
        self.quad_blocked[slots] = reason
        if slots in self.quad_blocks:
            self._close_quad_block(slots)
        audit_log('{marker} round={round} failures={failures} reason={reason} slots={slots}', marker=quad_draft.DISABLED_MARKER,
                  round=round_number, failures=self.quad_block_failures.get(slots, 0),
                  reason=str(reason).replace(' ', '_')[:160], slots=','.join(str(slot) for slot in slots))

    def _prepare_quad(self, groups, by_slot, round_number, batched, slots=None, pairs=None):
        """Q4: prepare the four users' one 64-row pass, or None for today's pairs. Engages only when the groups are
        exactly [(0, 1), (2, 3)], both pairs are packable, quad_draft.refusal finds nothing and it has not given up.
        A fresh build needs quad_draft.QUAD_CAPTURE_BYTES_EST plus the packed reserve of free DRAM (capture_headroom:
        in the largest block, or under the S2 flag the admission's split). A failure falls the round back to the
        pairs (FALLBACK_LINE); GIVE_UP_FAILURES in a row disable it for the process. On success each device wears a
        view of the quad, the first success releases every single-user capture still held (R6; the pair traces are
        kept, unless the quad leaves under PAIR_RELEASE_BELOW_BYTES free - capture_headroom again - and no audit
        needs them), and the quad joins the round's batched selection.

        `slots` and `pairs` (QWEN_FAST_QUAD_DRAFT_BLOCKS, from _prepare_quad_blocks; both None is the quad over slots 0-3, every line
        below as it always was): a per-block quad. It engages when both its pairs are among `groups` (the other block's need not be),
        keeps its state in self.quad_blocks[slots], and a refusal or two failures in a row block that quad only (_block_quad)."""
        import quad_draft
        from serving_worker_hook import phase

        block = slots is not None
        if not block:
            slots, pairs = tuple(quad_draft.SLOTS), tuple(quad_draft.PAIRS)
        quad_draft.enabled()
        self._retire_quad(by_slot, slots if block else None)
        if block:
            if slots in self.quad_blocked or not all(pair in [tuple(group) for group in groups] for pair in pairs):
                return None
        elif self.quad_disabled or [tuple(group) for group in groups] != list(quad_draft.PAIRS):
            return None
        entries = [by_slot[slot] for slot in slots]
        devices = [entry['device'] for entry in entries]
        if not (packable(devices[0], devices[1]) and packable(devices[2], devices[3])):
            return None
        reason = quad_draft.refusal(devices, batched)
        if reason is not None:
            if block:
                self._block_quad(slots, round_number, reason)
            else:
                self._disable_quad(round_number, reason)
            return None
        state = self.quad_blocks.get(slots) if block else self.quad
        trace = None
        if state is not None and all(old is new for old, new in zip(state[0], devices)):
            trace = state[1]
        fresh = trace is None or not trace.buckets
        if fresh:
            # A fresh capture - a new quad, or the same four devices' quad whose last build failed (its _bucket
            # released what that attempt built) - needs the headroom; replaying a built quad allocates nothing. Under
            # the S2 flag the headroom is the admission's split (capture_headroom, gate v79).
            # Per quad: a second block's check reads the free DRAM after the first block's capture landed, so each needs one
            # quad's bytes (quad_draft_tp.blocks_capture_need adds the two up; the attach itself makes no DRAM check).
            short, reading = capture_headroom(devices[0], quad_capture_bytes())
            if short:
                reason = 'dram_reserve:headroom=%d' % reading['largest_free']
                if 'free' in reading:
                    reason += ':free=%d:short=%s' % (reading['free'], '+'.join(short))
                audit_log(quad_draft.FALLBACK_LINE, round=round_number, reason=reason)
                return None
        if trace is None:
            if state is not None:
                if block:
                    self._close_quad_block(slots)
                else:
                    state[1].close()
            trace = quad_draft.PreparedQuadDFlashProposal(devices)
            if block:
                self.quad_blocks[slots] = (tuple(devices), trace, False)
                self._note_generations(slots, devices)
            else:
                self.quad = (tuple(devices), trace, False)
                self._note_generations('quad', devices)
        from dflash_proposal_trace import pair_mask_audit_enabled

        if pair_mask_audit_enabled():
            trace.round_number = round_number
        seeds = [entry['seed'] for entry in entries]
        ids = ','.join(str(entry['bridge'].request.session.request_id) for entry in entries)
        # S2 W6d: a fresh quad capture allocates; the ledger reads either side of it.
        ledger_token = ledger_before('quad', quad_capture_bytes(),
                                     'slots=' + ','.join(str(slot) for slot in slots)) if fresh else None
        started = time.perf_counter()
        try:
            ready = phase('propose_quad', ids, lambda: trace.prepare_device(seeds))
        except Exception as failure:
            ledger_after(ledger_token)
            # As a pair's failure (above): this round falls back to today's pairs - never the engine. The trace
            # stays valid (its _bucket released what the attempt built); two in a row give up for good.
            trace.discard_pending()
            if block:
                failures = self.quad_block_failures[slots] = self.quad_block_failures.get(slots, 0) + 1
            else:
                self.quad_failures += 1
                failures = self.quad_failures
            audit_log(quad_draft.FALLBACK_LINE, round=round_number,
                      reason=('%s:%s' % (type(failure).__name__, str(failure)[:120])).replace(' ', '_'))
            if failures >= quad_draft.GIVE_UP_FAILURES:
                if block:
                    self._block_quad(slots, round_number, 'consecutive_failures=%d' % failures)
                else:
                    self._disable_quad(round_number, 'consecutive_failures=%d' % failures)
            return None
        ledger_after(ledger_token)
        if not ready:
            return None
        built_ms = quad_draft.elapsed_ms(started)
        if block:
            self.quad_block_failures[slots] = 0
        else:
            self.quad_failures = 0
        self.quad_rounds += 1
        try:
            for which, device in enumerate(devices):
                _install(device, trace, which)
            held = self.quad_blocks[slots] if block else self.quad
            if not held[2]:
                for device in devices:
                    self._release_single_user(device)
                if block:
                    self.quad_blocks[slots] = (held[0], trace, True)
                else:
                    self.quad = (held[0], trace, True)
            audited = quad_draft.audit_selected(self.quad_rounds)
            if trace.last_built and not audited:
                # Under the S2 flag the release reads the admission's split too (capture_headroom, gate v79): the
                # pairs go when the free less the stranded bytes is below PAIR_RELEASE_BELOW_BYTES, or the largest
                # block below the reserve plus the largest buffer.
                short, reading = capture_headroom(devices[0], quad_draft.PAIR_RELEASE_BELOW_BYTES, reserve=False)
                if short:
                    released = [list(group) for group in pairs if group in self.pairs]
                    for group in pairs:
                        if group in self.pairs:
                            self._close_pair(group)
                    if released and 'free' in reading:
                        audit_log(quad_draft.RELEASE_SPLIT_LINE, pairs=released, headroom=reading['largest_free'],
                                  free=reading['free'], short='+'.join(short))
                    elif released:
                        audit_log(quad_draft.RELEASE_LINE, pairs=released, headroom=reading['largest_free'])
            if audited:
                # The quad's outputs as its replay left them, fenced and read before the pair replays
                # (snapshot_audit): a pair replay that writes into them is then named, never compared.
                try:
                    trace.snapshot_audit()
                except Exception as failure:
                    trace.attach_audit('snapshot-unavailable:%s' % type(failure).__name__)
                else:
                    trace.attach_audit(self._quad_audit_pairs(devices, seeds, round_number, pairs if block else None))
        except BaseException:
            # The quad's replay is enqueued and not yet among `prepared`: fence it and drop its pending (and any
            # attached audit pairs) here, as the caller's own exception path does for everything it prepared.
            devices[0].operations.synchronize_device(devices[0].mesh)
            trace.discard_pending()
            raise
        batched.append((list(slots), trace))
        if block and self.book is not None and trace.last_built:
            audit_log(BOOK_QUAD_LINE, slots=','.join(str(slot) for slot in slots), round=round_number)
        if audit_enabled():
            audit_log(quad_draft.ROUND_LINE, round=round_number, built=int(trace.last_built), ms=built_ms)
        return dict(devices=devices, fence=(devices[0].operations, devices[0].mesh), trace=trace)


# QWEN_FAST_ROUND_B1 (C1), under QWEN_FAST_PACKED_AUDIT=1: one line per round, the host
# time of the pairs' readback (collect) and of the batched selection, and how many selector
# calls that took (one unless two pairs lend different codebooks).
SELECT_LINE = '[PACKED-SELECT] round={round} pairs={pairs} users={users} calls={calls} collect_ms={collect_ms} select_ms={select_ms}'


def _discard_round(prepared):
    """The exception path of PackedProposalCoordinator.prepare, for a failure after its
    fence: every prepared device's pending work is dropped, a shared pair trace once.

    Unlike that path, a device wearing a _PackedCaptureView also has the view's own
    single-user capture discarded (the view forwards discard_pending to it): a device whose
    pair did not pack this round is prepared through prepare_single, and its pending then
    lives in that capture, not in the pair trace. A no-op for a device the pair prepared
    (its single-user capture holds nothing pending, or was released)."""
    discarded_traces = set()
    for device in prepared:
        capture = device.proposal_capture
        if isinstance(capture, _PackedCaptureView):
            if id(capture._trace) not in discarded_traces:
                discarded_traces.add(id(capture._trace))
                capture._trace.discard_pending()
            capture.discard_pending()
        else:
            discard = getattr(capture, 'discard_pending', None)
            if callable(discard):
                discard()


def reads_covered(batched, prepared):
    """Round-fence plan H2: whether the batched pairs' collect reads back every device this round
    prepared - no single left whose phase-B finish() would read behind whatever is enqueued next."""
    # A quad trace (QWEN_FAST_QUAD_DRAFT) reads back all four of its devices; a pair trace has no `devices`.
    paired = {id(device) for _, trace in batched
              for device in getattr(trace, 'devices', None) or (trace.device_a, trace.device_b)}
    return all(id(device) in paired for device in prepared)


def select_round(batched, prepared, round_number, after_collect=None):
    """QWEN_FAST_ROUND_B1 (C1): after the round's one fence, read every prepared pair back
    (PreparedPackedDFlashProposal.collect) and select all of their users in ONE FP64
    selector call (dflash_packed_proposal.select_packed_batched), where each pair's first
    finish() in phase B used to select its own two users - and adopt the tokens into each
    trace, so phase B's finish() returns them unchanged. Pairs are batched only when they
    select against the very same codebook objects (the lent draft weights), checked by
    identity; a pair that does not share them gets a call of its own.

    A failure here fails the round as it would have failed in phase B, after dropping
    every prepared device's pending work - the device queue is already fenced.

    Under QWEN_FAST_ROUND_B1_AUDIT each pair is then read back and selected again through
    PreparedPackedDFlashProposal.audit_selection (select_device_outputs, the flag-off path)
    and must give the tokens it adopted (dflash_packed_proposal.ROUND_B1_AUDIT_FLAG).

    `after_collect` (round-fence plan H2, the coordinator's after_reads): called right after every
    pair's readback, before the selection - its time is in neither collect_ms nor select_ms. A failure
    there fails the round like any other here."""
    from dflash_packed_proposal import note_round_b1, round_b1_audit_enabled, select_packed_batched

    audit = round_b1_audit_enabled()
    started = time.perf_counter()
    import round_host
    import verify_prestage

    ledger = round_host.ledger
    ledger.mark('collect0', started)

    hostgap = verify_prestage.hostgap_log_enabled()
    cpu_started = verify_prestage.thread_ms() if hostgap else 0.0
    if hostgap:
        verify_prestage.take_scratch('collect')
    try:
        with verify_prestage.hostgap_span('collect'):
            collected = [(labels, trace, trace.collect()) for labels, trace in batched]
        collected_at = select_started = time.perf_counter()
        ledger.mark('collect1', collected_at)
        cpu_collected = verify_prestage.thread_ms() if hostgap else 0.0
        if after_collect is not None:
            after_collect()
            select_started = time.perf_counter()
        ledger.mark('flush1', select_started)
        groups = []
        for labels, trace, result in collected:
            device = trace.device_a
            for group in groups:
                if group['predecessors'] is device.predecessors and group['successors'] is device.successors:
                    group['members'].append((trace, result))
                    break
            else:
                groups.append(dict(predecessors=device.predecessors, successors=device.successors,
                                   members=[(trace, result)]))
        for group in groups:
            parts = [part for _, result in group['members'] for part in result['parts']]
            seeds = [seed for _, result in group['members'] for seed in result['seeds']]
            counts = [count for _, result in group['members'] for count in result['counts']]
            tokens = select_packed_batched(parts, seeds, counts, group['predecessors'], group['successors'])
            offset = 0
            for trace, result in group['members']:
                users = len(result['parts'])
                trace.adopt(tokens[offset:offset + users])
                if audit:
                    _audit_selection(trace, tokens[offset:offset + users])
                offset += users
    except BaseException:
        _discard_round(prepared)
        raise
    ledger.mark('select1')
    note_round_b1('batched-select')
    if audit:
        from dflash_packed_proposal import note_round_b1_audit

        note_round_b1_audit()
    if hostgap:
        # Stage 0: the readback's reads against its merge (quad blocks only; 0 on the pair path), and the thread CPU time of the
        # collect and of the rest (the H2 flush, then the selection; the collect's blocking reads are mostly waiting, not CPU).
        read_ms, merge_ms = verify_prestage.take_scratch('collect', [0.0, 0.0])
        verify_prestage.log_line('%s round=%d read_ms=%.2f merge_ms=%.2f collect_cpu_ms=%.2f select_cpu_ms=%.2f' % (
            verify_prestage.HOSTGAP_SELECT_MARKER, round_number, read_ms, merge_ms, cpu_collected - cpu_started,
            verify_prestage.thread_ms() - cpu_collected))
    if audit_enabled():
        finished = time.perf_counter()
        audit_log(SELECT_LINE, round=round_number, pairs=[labels for labels, _, _ in collected],
                  users=sum(len(result['parts']) for _, _, result in collected), calls=len(groups),
                  collect_ms='%.2f' % ((collected_at - started) * 1000),
                  select_ms='%.2f' % ((finished - select_started) * 1000))


def run_while_waiting(while_waiting):
    """Round-fence plan H1a: the window's callable, just before the round's fence. An Exception
    goes to its `drop` (the pre-stage's snapshot is dropped, the next verify takes the full
    path) and never fails the round; a failing `drop` is swallowed too."""
    try:
        while_waiting()
    except Exception as failure:
        drop = getattr(while_waiting, 'drop', None)
        if callable(drop):
            try:
                drop(failure)
            except Exception:
                pass


def note_fenced(while_waiting):
    """Round-fence plan H1a: right after the round's fence, the window callable's `fenced` (it
    arms the packed block's next replay under QWEN_FAST_ROUND_FENCES). A failure only leaves
    that replay to pay its own fence, so it never fails the round."""
    fenced = getattr(while_waiting, 'fenced', None)
    if callable(fenced):
        try:
            fenced()
        except Exception as failure:
            try:
                audit_log('[PACKED-PRESTAGE-WINDOW] fenced failed: {failure}', failure=repr(failure)[:160])
            except Exception:
                pass


def _audit_selection(trace, adopted):
    """QWEN_FAST_ROUND_B1_AUDIT (C1): the pair's own flag-off selection against the batched
    tokens it adopted."""
    from dflash_packed_proposal import round_b1_audit_count, round_b1_audit_mismatch

    reference = tuple(tuple(tokens) for tokens in trace.audit_selection())
    adopted = tuple(tuple(tokens) for tokens in adopted)
    if reference != adopted:
        round_b1_audit_mismatch('C1', 'batched=%s per-user=%s' % (adopted, reference))
    round_b1_audit_count('select', len(adopted))
