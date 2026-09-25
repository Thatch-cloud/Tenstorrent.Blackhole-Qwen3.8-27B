"""Assemble a packed verify block from several single-user verifier participants.

Each serving request already owns everything a packed user needs. `VerifierEngine`
holds that request's `pages` (its own page table), `position` (its own frontier)
and, per bucket, `checkpoints` - one `helper.allocate()` per GDN layer. The only
thing a packed block adds is a per-user CARRIED recurrent state, because the row
axis is time for the 48 GDN layers and each user's recurrence must resume where
that user left off rather than where the user packed above it ended.

So a participant is a request's engine plus one carried state per GDN layer, and
packing is assembling those into the contract `ModelBatch` validates.

The weights are read once for the whole block either way: each GDN layer runs one
input projection across all rows and one output projection over the whole block,
and the 16 attention layers are per row. That is what packing buys, and it is the
reason a packed block is worth the bookkeeping.
"""

GDN_LAYERS = 48


def participant(engine, rows, prefix, checkpoints, slots):
    """One user's contribution to a packed block."""
    if (type(rows) is not int or rows < 1 or type(prefix) is not int or not 0 <= prefix <= rows
            or len(checkpoints or ()) != GDN_LAYERS or len(slots or ()) != GDN_LAYERS):
        raise ValueError('Each packed participant needs its rows, an accepted prefix within them, '
                         'and one GDN checkpoint and carried state per linear layer')
    pages = engine.pages
    if getattr(pages, 'ndim', None) != 2 or pages.shape[0] != 1:
        raise ValueError('Each packed participant contributes its own single page table')
    return dict(start=engine.position, rows=rows, pages=pages, prefix=prefix,
                checkpoints=tuple(checkpoints), slots=tuple(slots))


def allocate_slots(helpers):
    """One carried GDN state per linear layer, seeded from the live layer state.

    `helper.allocate()` is the same compact five-tensor snapshot the engine already
    uses for its checkpoints, and `helper.save` seeds it from whatever that layer
    currently holds - which, at the moment a request is admitted to a pack, is that
    request's own state.
    """
    if len(helpers) != GDN_LAYERS:
        raise ValueError('One carried state per linear layer required')
    slots = []
    for helper in helpers:
        snapshot = helper.allocate()
        helper.save(snapshot)
        slots.append(snapshot)
    return tuple(slots)


def build_pack(participants, *, block_rows=32):
    """Validate a set of participants into the pack `ModelBatch` takes."""
    users = tuple(participants)
    if not users:
        raise ValueError('A packed block needs at least one participant')
    total = sum(user['rows'] for user in users)
    # 64: the M3 block, four T16 participants (docs/packed-device-step-plan-2026-09-20.md section 5).
    if total != block_rows or block_rows not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError('Packed participants must fill exactly one legal block width')
    if len({id(user['pages']) for user in users}) != len(users):
        raise ValueError('Each packed user needs its own page table; a shared one means a shared cache')
    if len({id(slot) for user in users for slot in user['slots']}) != len(users) * GDN_LAYERS:
        raise ValueError('Carried GDN states must not be shared between packed users')
    return [dict(user) for user in users]


def release_slots(operations, slots):
    """Free carried states allocated by allocate_slots."""
    from gdn_multitoken_conv import release_owned

    release_owned(operations, [tensor for snapshot in slots for tensor in snapshot])
