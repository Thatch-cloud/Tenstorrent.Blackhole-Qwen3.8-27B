"""Packed proposal inputs and per-user selection for the shared 32-row block.

One proposal pass carries every packed user. The anchor/draft identifiers, the
selector features and the vocabulary-head candidates are all laid out in block
order, so each user owns a contiguous run of `block_rows` rows and its results are
a slice rather than a separate pass.

`merge_chunk_candidates` drops the anchor row and returns rows 1..block_rows-1, so
with a 32-row block the returned index of user u's first draft is u * block_rows:
user 0 holds block rows 1..15 at indices 0..14, and user 1 holds block rows 17..31
at indices 16..30. Getting that offset wrong silently attributes one user's
proposals to another, which is why it is pinned by a test rather than a comment.

BLOCK WIDTH. The device's proposal block is 32 rows, and that is intrinsic to it, not
to the layout arithmetic here: `execute_proposal` embeds and pads 32 rows and builds
32-row RoPE tables (dflash_device.py), the packed mask is (1, 1, 32, K) (dflash_batched_
mask.py), the shared head reads (32, 16) candidates back and merges at block_rows=32
(draft_shared_head.py), the pool's query buffer is (1, 1, 32, 2048) (serving_buffer_pool.
py) and the T16 native attention admits at most two users, pinned by its source-hashed
simulator evidence (dflash_t16_native_attention.py, dflash_t16_native_scope.py). So four
T16 users are proposed in TWO 32-row passes, two users each, while their verify shares
one 64-row block (M3). The layout helpers below take `block_width` so the 64-row layout
- four T16 users, anchors at rows 0, 16, 32, 48 - is defined and tested for the day
the device grows a 64-row proposal block; `propose_packed` and `select_device_outputs`
stay at the device's 32.
"""

import os

DRAFT_FILLER = 248070
BLOCK_WIDTH = 32
BLOCK_WIDTHS = (32, 64)

# QWEN_FAST_PACKED_PROPOSAL: opt-in wiring of propose_packed() into serving as two
# per-round traced two-user passes (dflash_packed_proposal_trace.PreparedPackedDFlashProposal,
# serving_worker_hook.prepare_pipelined_drafts), instead of four serial single-user
# passes. Default OFF; anything but '0'/'1' is a configuration error rather than a
# silent fallback - the gdn_user_batch.enabled() pattern (scripts/ci/gdn_user_batch.py).
PACKED_PROPOSAL_FLAG = 'QWEN_FAST_PACKED_PROPOSAL'

# The same fixed slot pairing the packed VERIFY side already uses under
# QWEN_FAST_FOUR_AS_TWO (serving_runtime.py): 'block A over pool slots (0, 1), block B
# over (2, 3)'. Reusing it here means a request's pairing never depends on which of the
# two packed passes - propose or verify - happens to run first.
FOUR_AS_TWO_PAIRS = ((0, 1), (2, 3))

# QWEN_FAST_ROUND_B1: build 1 of the round host-phase cuts (the 4 x 131k packed round's
# ~82 ms outside the verify trace) - only the cuts that are token-exact by construction
# (E1) and change no lifetime across steps:
#   C1  one batched FP64 selection for every packed pair after the proposal fence
#       (select_packed_batched below; dflash_packed_proposal_coordinator);
#   C2  O(1) retain() bookkeeping (dflash_device.indexed_temporaries,
#       draft_kv_history.indexed_temporaries);
#   C7  the fused steady-state feature-history write, which nothing on the served
#       path reads, is skipped and the history poisoned (dflash_device);
#   C8  the pair update stops uploading the two key RoPE tables its trace never reads
#       and builds the pair's RoPE once (dflash_proposal_trace, dflash_batched_mask);
#   M0a host timing splits inside publication, logged under QWEN_FAST_PACKED_AUDIT.
# Off by default and read at each use, never at import. With it unset every path is
# the one that ran before it existed.
ROUND_B1_FLAG = 'QWEN_FAST_ROUND_B1'
# Logged once per process, at the first B1 path that runs; the m3native gate requires it
# whenever the arm passes the flag (lever_n_m3native_gate.required_flag_markers).
ROUND_B1_MARKER = '[PINDIAG] round b1 engaged'
_ROUND_B1_NOTED = []


def round_b1_enabled(environ=None):
    """QWEN_FAST_ROUND_B1=1."""
    return (os.environ if environ is None else environ).get(ROUND_B1_FLAG) == '1'


def note_round_b1(site):
    """ROUND_B1_MARKER, once per process, naming the B1 path that ran first."""
    if _ROUND_B1_NOTED:
        return False
    _ROUND_B1_NOTED.append(site)
    message = '%s site=%s cuts=C1,C2,C7,C8,M0a' % (ROUND_B1_MARKER, site)
    try:
        from loguru import logger
    except ImportError:
        print(message, flush=True)
        return True
    logger.info('{}', message)
    return True


# QWEN_FAST_ROUND_B1_AUDIT=1, read only on B1 paths (so only with QWEN_FAST_ROUND_B1=1): a
# shadow check of the B1 cuts the gate's final text cannot see - under exact greedy
# verification a wrong draft token only lowers acceptance, and the per-round [PACKED] lines
# differ run to run. Host work only, beside the served result, never instead of it:
#   C1  each pair is read back and selected again through select_device_outputs - the
#       flag-off path - and its tokens compared with the batched ones it adopted;
#   C2  every retain() decision is made again by the list scan it replaced
#       (scanned_decision), every DraftKVHistory scope's release list again by the list
#       filter, and the cached borrowed addresses are re-read and compared on every reuse;
#   C8  live_key_rope is rebuilt and compared bit for bit with the rows sliced from the
#       pair's one table build.
# (C7's stale history needs none: every reader of it already raises.) The first mismatch logs
# ROUND_B1_AUDIT_MISMATCH and raises - a raise the pair-proposal fallback may swallow, which
# is why the gate fails on the line itself. After each audited batched selection it logs
# ROUND_B1_AUDIT_MARKER with the running counts, which the gate requires. It spends back
# what B1 saves, so it belongs on a correctness arm, never a timed one.
ROUND_B1_AUDIT_FLAG = 'QWEN_FAST_ROUND_B1_AUDIT'
ROUND_B1_AUDIT_MARKER = '[PINDIAG] round b1 audit'
ROUND_B1_AUDIT_MISMATCH = '[PINDIAG] round b1 audit mismatch'
ROUND_B1_AUDIT_COUNTS = ('select', 'rope', 'retain', 'borrowed', 'release')
_ROUND_B1_AUDIT = dict(rounds=0, **dict.fromkeys(ROUND_B1_AUDIT_COUNTS, 0))


def round_b1_audit_enabled(environ=None):
    """QWEN_FAST_ROUND_B1_AUDIT=1 beside QWEN_FAST_ROUND_B1=1."""
    environ = os.environ if environ is None else environ
    return environ.get(ROUND_B1_FLAG) == '1' and environ.get(ROUND_B1_AUDIT_FLAG) == '1'


def _log_line(message):
    try:
        from loguru import logger
    except ImportError:
        print(message, flush=True)
        return
    logger.info('{}', message)


def round_b1_audit_count(name, count=1):
    _ROUND_B1_AUDIT[name] += count


def round_b1_audit_mismatch(cut, detail):
    """Log ROUND_B1_AUDIT_MISMATCH for `cut` and raise."""
    message = '%s cut=%s %s' % (ROUND_B1_AUDIT_MISMATCH, cut, detail)
    _log_line(message[:280])
    raise AssertionError(message)


def note_round_b1_audit():
    """ROUND_B1_AUDIT_MARKER once per audited batched selection: '<n> exact=True' and the
    comparisons made so far, all of which matched (a mismatch raised before this line)."""
    _ROUND_B1_AUDIT['rounds'] += 1
    _log_line('%s %d exact=True %s' % (ROUND_B1_AUDIT_MARKER, _ROUND_B1_AUDIT['rounds'], ' '.join(
        '%s=%d' % (name, _ROUND_B1_AUDIT[name]) for name in ROUND_B1_AUDIT_COUNTS)))


def scanned_decision(protected_ids, identity):
    """The decision the flag-off list scan in DFlashDevice.temporaries and
    DraftKVHistory.temporaries makes for `identity`, text for text: 'kept' when it is a
    protected identity, 'refused' when any one chip's address matches the same chip of a
    protected identity, else 'queued'. The scan zips strictly, so identities of unequal
    length raise there; that is 'unequal' here, which no set lookup can answer."""
    if identity not in protected_ids:
        try:
            if any(any(left == right for left, right in zip(identity, other, strict=True)) for other in protected_ids):
                return 'refused'
        except ValueError:
            return 'unequal'
        return 'queued'
    return 'kept'


def audit_borrowed(fresh, cached):
    """QWEN_FAST_ROUND_B1_AUDIT (C2): the borrowed addresses read now against the cached ones
    a scope is about to reuse."""
    if list(fresh) != list(cached):
        round_b1_audit_mismatch('C2', 'cached borrowed addresses are stale (%d tensors)' % len(fresh))
    round_b1_audit_count('borrowed')


def audit_retain(protected_ids, identity, decided):
    """QWEN_FAST_ROUND_B1_AUDIT (C2): the set lookups' decision against the list scan's."""
    expected = scanned_decision(protected_ids, identity)
    if decided != expected:
        round_b1_audit_mismatch('C2', 'retain %s where the scan decides %s' % (decided, expected))
    round_b1_audit_count('retain')


def audited_retain(retain, owned, identify, protected_ids):
    """`retain`, with each decision it makes (kept, queued, refused - read off what it did)
    checked against scanned_decision over the same protected identities; `identify` is the
    caller's own addresses(operations, value)."""
    def audited(value):
        identity = identify(value)
        before = len(owned)
        try:
            retain(value)
        except ValueError:
            audit_retain(protected_ids, identity, 'refused')
            raise
        audit_retain(protected_ids, identity, 'queued' if len(owned) > before else 'kept')
        return value
    return audited


def audit_release(expected, released):
    """QWEN_FAST_ROUND_B1_AUDIT (C2): a DraftKVHistory scope's release list against the list
    filter's, value for value and in order."""
    if len(expected) != len(released) or any(mine is not theirs for mine, theirs in zip(released, expected)):
        round_b1_audit_mismatch('C2', 'scope releases %d values where the scan releases %d'
                                % (len(released), len(expected)))
    round_b1_audit_count('release')


def same_bits(left, right):
    """Equal dtype, shape and bits (a float compared as its integer image)."""
    import torch

    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    image = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32,
             torch.float64: torch.int64}.get(left.dtype)
    if image is None:
        return torch.equal(left, right)
    return torch.equal(left.contiguous().view(image), right.contiguous().view(image))


def packed_proposal_enabled(environ=None):
    """QWEN_FAST_PACKED_PROPOSAL=1. Read at each round rather than at import, so the
    switch a test flips is the one the round sees."""
    value = (os.environ if environ is None else environ).get(PACKED_PROPOSAL_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % PACKED_PROPOSAL_FLAG)
    return value == '1'


def pair_slots(active):
    """Group active pool-slot indices into FOUR_AS_TWO_PAIRS packed rounds.

    Each fixed pair with BOTH members in `active` packs into one two-element group; a
    fixed pair with only ONE active member degrades to a one-element group for that
    member alone. A slot is never paired across the other fixed pair - so one finished
    neighbour (slot 1 gone, say) never silently recombines slot 0 with slot 2 or 3 just
    because both happen to be active. Order follows FOUR_AS_TWO_PAIRS, not `active`'s
    own order; an empty `active` returns no groups at all.
    """
    valid = {slot for pair in FOUR_AS_TWO_PAIRS for slot in pair}
    active = set(active)
    if any(type(slot) is not int or slot not in valid for slot in active):
        raise ValueError('Every active slot must be one of the fixed FOUR_AS_TWO_PAIRS slots %r' % (valid,))
    groups = []
    for pair in FOUR_AS_TWO_PAIRS:
        present = tuple(slot for slot in pair if slot in active)
        if present:
            groups.append(present)
    return groups


def validate_block_width(block_width):
    if type(block_width) is not int or block_width not in BLOCK_WIDTHS:
        raise ValueError('A %s-row proposal block width required' % ' or '.join(map(str, BLOCK_WIDTHS)))
    return block_width


def packed_identifiers(seeds, block_rows=16, *, filler=DRAFT_FILLER, block_width=BLOCK_WIDTH):
    """The (1, block_width) anchor/draft token ids the embedding consumes, in block order."""
    import torch

    validate_block_width(block_width)
    seeds = tuple(seeds)
    if (type(block_rows) is not int or block_rows not in (8, 16, 32)
            or not seeds or len(seeds) * block_rows > block_width
            or any(type(seed) is not int or not 0 <= seed < 248320 for seed in seeds)):
        raise ValueError('One committed anchor per packed user and a fitting block required')
    rows = []
    for seed in seeds:
        rows.extend([seed, *([filler] * (block_rows - 1))])
    rows.extend([filler] * (block_width - len(rows)))
    return torch.tensor([rows], dtype=torch.int64)


def user_slices(users, block_rows=16, *, block_width=BLOCK_WIDTH):
    """Where each user's drafts sit in the merged candidate and selector tensors."""
    validate_block_width(block_width)
    if (type(block_rows) is not int or type(users) is not int
            or not (1 <= users and users * block_rows <= block_width)):
        raise ValueError('A fitting number of packed users required')
    return [dict(user=index, block=slice(index * block_rows, (index + 1) * block_rows),
                 drafts=slice(index * block_rows, index * block_rows + block_rows - 1),
                 selector=slice(index * block_rows + 1, (index + 1) * block_rows))
            for index in range(users)]


def split_selection(projected_hidden, candidates, unary, users, block_rows=16, *, block_width=BLOCK_WIDTH):
    """Per-user (selector features, candidate ids, candidate scores).

    `projected_hidden` is the replicated selector projection over the whole block,
    shaped (1, block_width, 256). `candidates` and `unary` come straight from
    merge_chunk_candidates(block_rows=block_width), shaped (1, block_width - 1, 16).
    """
    validate_block_width(block_width)
    if (projected_hidden.ndim != 3 or projected_hidden.shape[:2] != (1, block_width)
            or candidates.ndim != 3 or candidates.shape[:2] != (1, block_width - 1)
            or unary.shape != candidates.shape):
        raise ValueError('Whole-block selector features and merged %d-row candidates required' % block_width)
    parts = []
    for part in user_slices(users, block_rows, block_width=block_width):
        parts.append(dict(user=part['user'],
                          hidden=projected_hidden[:, part['selector'], :],
                          candidates=candidates[:, part['drafts']],
                          unary=unary[:, part['drafts']]))
        if parts[-1]['hidden'].shape[1] != block_rows - 1:
            raise AssertionError('Each user contributes block_rows - 1 drafts')
        if parts[-1]['candidates'].shape[1] != block_rows - 1:
            raise AssertionError('Each user owns block_rows - 1 merged candidate rows')
    return parts


def select_packed(parts, seeds, counts, predecessors, successors):
    """Run the shared selector once per user over its own slice of the block."""
    import torch

    from draft_selector import select_active_candidates

    seeds, counts = tuple(seeds), tuple(counts)
    if len(parts) != len(seeds) or len(parts) != len(counts):
        raise ValueError('One anchor and one proposal count per packed user required')
    chosen = []
    for part, seed, count in zip(parts, seeds, counts):
        if type(count) is not int or not 1 <= count <= part['candidates'].shape[1]:
            raise ValueError('Bounded proposal count within this user block required')
        tokens, _ = select_active_candidates(part['hidden'], part['candidates'], part['unary'],
            predecessors, successors, torch.tensor([seed], dtype=torch.int64))
        chosen.append(tuple(int(token) for token in tokens[0, :count]))
    return tuple(chosen)


def select_packed_batched(parts, seeds, counts, predecessors, successors):
    """QWEN_FAST_ROUND_B1 (C1): select_packed's result from ONE selector call over every
    user in `parts`, instead of one call per user.

    Bit-identical by construction: every step of draft_selector's greedy FP64 search
    works row by row - the codebook gathers, the elementwise products, the sum over the
    contiguous rank dimension of one (user, candidate) row, the first-max argmax and the
    gather - so no user's arithmetic ever sees another user's rows. torch.unique over
    the union of every user's ids only renumbers them: the rows it gathers out of the
    codebooks hold the same values. The per-user count check is select_packed's own,
    run for every user before the call. The price is shared failure: an operand one
    user's call would have refused refuses them all, where today it fails that round
    one user later. test_dflash_round_b1.py pins the equality, ties included."""
    import torch

    from draft_selector import select_active_candidates

    parts, seeds, counts = tuple(parts), tuple(seeds), tuple(counts)
    if not parts or len(parts) != len(seeds) or len(parts) != len(counts):
        raise ValueError('One anchor and one proposal count per packed user required')
    for part, count in zip(parts, counts):
        if type(count) is not int or not 1 <= count <= part['candidates'].shape[1]:
            raise ValueError('Bounded proposal count within this user block required')
    hidden = torch.cat([part['hidden'] for part in parts], dim=0)
    candidates = torch.cat([part['candidates'] for part in parts], dim=0)
    unary = torch.cat([part['unary'] for part in parts], dim=0)
    tokens, _ = select_active_candidates(hidden, candidates, unary, predecessors, successors,
        torch.tensor(seeds, dtype=torch.int64))
    return tuple(tuple(int(token) for token in tokens[index, :count]) for index, count in enumerate(counts))


def propose_packed(device, slots, seeds, counts, *, stage=None):
    """One proposal pass over every packed user, returning each user's tokens.

    `slots` carry the per-user state DFlashDevice holds singly today: the absolute
    frontier, the committed history length, and the per-layer cached K/V. The shared
    weights, mesh and prepared layers come from `device`, so packing costs one pass
    rather than one pass per user.

    The cached path never reads a history tensor, so none is built: at two 2048-row
    users that would be 42 MB of DRAM nobody touches.

    The pass is the device's 32-row block (module docstring, BLOCK WIDTH): at most two
    T16 users per call, so M3's four T16 users take two calls.
    """
    import torch

    from dflash_batched_mask import batched_attention_mask, live_key_rope, packed_rope_tables
    from dflash_t16_native_attention import validate_mask
    from gdn_multitoken_conv import addresses, release_owned

    operations = device.operations
    slots = tuple(slots)
    seeds, counts = tuple(seeds), tuple(counts)
    if (not slots or 32 % len(slots) or device.closed or device.pending is not None
            or len(seeds) != len(slots) or len(counts) != len(slots)):
        raise ValueError('An idle device, one anchor and count per slot, and a block-dividing slot count required')
    block_rows = 32 // len(slots)
    if block_rows != device.block_rows:
        raise ValueError('Packed slots must divide the block into this device rows')
    users = [dict(position=slot['position'], history_rows=slot['history_rows']) for slot in slots]
    contexts = [user['history_rows'] for user in users]
    caches = [slot['kv_history'] for slot in slots]
    if any(cache is None or len(cache) != len(device.layers) for cache in caches):
        raise ValueError('Every packed slot needs a committed K/V cache for each prepared layer')

    stage = stage or (lambda name, **values: None)
    owned, retain = device.temporaries([device.history, device.spare_history, *device.owned])

    def upload(value, dtype=None, row_major=False):
        return retain(operations.from_torch(value, device=device.mesh, dtype=dtype or operations.bfloat16,
            layout=operations.ROW_MAJOR_LAYOUT if row_major else operations.TILE_LAYOUT,
            memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ReplicateTensorToMesh(device.mesh)))

    try:
        stage('upload-packed-anchor-and-mask')
        identifiers = upload(packed_identifiers(seeds, block_rows),
            dtype=operations.uint32, row_major=True)
        host_mask = batched_attention_mask(contexts, block_rows)
        validate_mask(host_mask, contexts=contexts)
        mask = upload(host_mask)
        # execute_proposal refuses a native proposal mask it has not seen validated,
        # and the address is the identity it checks.
        device.validated_native_proposal_masks.add(addresses(operations, mask))
        stage('prepare-packed-rope')
        tables = packed_rope_tables(users, block_rows)
        rope = dict(q=tuple(upload(value) for value in tables['q']),
                    k=tuple(upload(value) for value in tables['k']),
                    live_k=tuple(upload(value) for value in live_key_rope(users, block_rows)))
        outputs = device.execute_proposal(identifiers, None, mask, rope, context=None, pack=users,
            cached_history=caches, owned=owned, retain=retain, stage=stage)
        stage('synchronize-packed-head')
        operations.synchronize_device(device.mesh)
        stage('read-packed-candidates')
        return select_device_outputs(device, outputs, seeds, counts, len(slots), block_rows)
    finally:
        stage('release-packed-temporaries')
        operations.synchronize_device(device.mesh)
        release_owned(operations, owned)


def select_device_outputs(device, outputs, seeds, counts, users, block_rows):
    """Read the shared head back once and split it by user.

    The readback is the device's 32-row block: (32, 16) candidates per chip and chunk,
    merged at block_rows=32, a (1, 32, 256) selector projection (BLOCK WIDTH above)."""
    import torch

    from draft_shared_head import merge_chunk_candidates

    operations = device.operations
    host_chunks = []
    for chunk in outputs.chunks:
        values = operations.get_device_tensors(chunk['values'])
        indices = operations.get_device_tensors(chunk['indices'])
        if len(values) != 2 or len(indices) != 2:
            raise AssertionError('Both learned head shards required')
        for chip in range(2):
            host_chunks.append(dict(chip=chip, start=chunk['start'], stop=chunk['stop'],
                values=operations.to_torch(values[chip]).float().reshape(32, 16),
                indices=operations.to_torch(indices[chip]).long().reshape(32, 16)))
    candidates, unary = merge_chunk_candidates(host_chunks, block_rows=32)
    parts = [operations.to_torch(value) for value in operations.get_device_tensors(outputs.projected)]
    if len(parts) != 2 or not torch.equal(*parts):
        raise AssertionError('Replicated learned selector features differ')
    hidden = parts[0].reshape(1, 32, 256)
    return select_packed(split_selection(hidden, candidates, unary, users, block_rows),
        seeds, counts, device.predecessors, device.successors)


def read_device_outputs(device, outputs, users, block_rows):
    """QWEN_FAST_ROUND_B1 (C1): select_device_outputs' readback, merge, replicated-selector
    check and per-user split, without the selection - so every pair of a round can be read
    and then all of their users selected in one call. A copy of select_device_outputs'
    body up to its return, so the flag-off function stays the text that ran before;
    test_dflash_round_b1.ReadDeviceOutputsTests pins the two bodies statement for statement,
    and the B1 shadow audit (QWEN_FAST_ROUND_B1_AUDIT) re-reads each pair through
    select_device_outputs itself."""
    import torch

    from draft_shared_head import merge_chunk_candidates

    operations = device.operations
    host_chunks = []
    for chunk in outputs.chunks:
        values = operations.get_device_tensors(chunk['values'])
        indices = operations.get_device_tensors(chunk['indices'])
        if len(values) != 2 or len(indices) != 2:
            raise AssertionError('Both learned head shards required')
        for chip in range(2):
            host_chunks.append(dict(chip=chip, start=chunk['start'], stop=chunk['stop'],
                values=operations.to_torch(values[chip]).float().reshape(32, 16),
                indices=operations.to_torch(indices[chip]).long().reshape(32, 16)))
    candidates, unary = merge_chunk_candidates(host_chunks, block_rows=32)
    parts = [operations.to_torch(value) for value in operations.get_device_tensors(outputs.projected)]
    if len(parts) != 2 or not torch.equal(*parts):
        raise AssertionError('Replicated learned selector features differ')
    hidden = parts[0].reshape(1, 32, 256)
    return split_selection(hidden, candidates, unary, users, block_rows)
