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

DRAFT_FILLER = 248070
BLOCK_WIDTH = 32
BLOCK_WIDTHS = (32, 64)


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
