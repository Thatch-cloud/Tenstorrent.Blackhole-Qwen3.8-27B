"""Persistent per-request draft history storage, allocated once before any request trace.

WHY. A request's verify trace (verifier_engine.py, capture_operation) bakes the device
address of every intermediate its captured forward allocates - including the ones the
forward frees again before capture ends (model_batch.py release_owned, finish_output).
Once capture finishes those addresses are free, so the NEXT request's persistent draft
buffers - DFlashDevice.history and spare_history, kept for the request's whole life -
are allocated into exactly those holes, and every later replay of the first request's
trace writes its intermediates over them. Each chip's allocator reuses its freed memory
independently, so the two replicas of a replicated buffer end up holding different,
finite data. That is the shape of the failure runs 35477522469 and 35479238722
measured: with two users stepped in turn, the second-admitted user's second proposal
found its replicated selector projection differing between the chips on 8184 of 8192
elements, every value finite. Excluding the per-request proposal trace did not change
it, because the verify trace still replays. TT-Metal's allocator warns about precisely
this: buffers allocated while a trace exists may be corrupted by its replay. The
publication prefix pool met the same failure and the same cure before
(docs/experiment-execution.md, "first native traced correction overwrites the second
prefix at layer19/chip0"; feature_prefix.py): allocate before any trace, then lend.

WHAT. One history pair per scheduler slot, allocated at serving attach - before any
request exists, hence before any request trace - and lent to one DFlashDevice for
that device's lifetime. A slot is handed over zeroed and returned on device close;
acquisition fails loudly when every slot is lent, and pooled addresses are checked
unchanged at both ends of the loan.

THE WEIGHTS COME FIRST. The draft weights a proposal reads - five layers of attention
and MLP parameters, the projection, the norms and the selector projection - are the
first thing a request allocates, so they take the lowest hole an earlier request's
trace left, before the history does; and on the cached serving path the proposal
reads them and the K/V cache rather than `history` at all. They are hoisted rather
than pooled: one PreparedDraftWeights (dflash_device.py) is uploaded at attach and
lent to every device. This pool covers the history pair, which commit_publication
writes and which the eager proposal path reads.

NOT POOLED HERE. The DraftKVHistory active/spare K/V (draft_kv_history.py, allocated
inline in its constructor with no injection point) stays per request in this pass;
pooling it needs DraftKVHistory to accept pre-allocated active/spare tensors in place
of the pad/zeros_like pair it allocates.
"""

from dflash_device import pindiag
from gdn_multitoken_conv import addresses, release_owned
from serving_fast_policy import NATIVE_GDN_SLOTS


HISTORY_SHAPE = (1, 1, 2048, 5120)


def overlaps(left, right):
    return any(first == second for first, second in zip(left, right, strict=True))


class HistorySlot:
    """One request's history pair; the addresses are recorded at allocation and never move."""

    def __init__(self, pool, index, history, spare_history):
        self.pool, self.index = pool, index
        self.history, self.spare_history = history, spare_history
        self.tensors = (history, spare_history)
        self.addresses = tuple(addresses(pool.operations, value) for value in self.tensors)
        self.lent = False
        self.owner = None

    def verify(self):
        current = tuple(addresses(self.pool.operations, value) for value in self.tensors)
        if current != self.addresses:
            raise AssertionError('Pooled draft history slot %d moved: %r became %r'
                                 % (self.index, self.addresses, current))

    def release(self):
        self.pool.release(self)


class ServingBufferPool:
    def __init__(self, operations, mesh, *, users):
        import torch

        if type(users) is not int or not 1 <= users <= NATIVE_GDN_SLOTS:
            raise ValueError('Explicit scheduler request count within the %d native GDN slots required'
                             % NATIVE_GDN_SLOTS)
        self.operations, self.mesh, self.users = operations, mesh, users
        self.owned, self.slots = [], []
        self.closed = False
        try:
            protected = []
            for index in range(users):
                pair = []
                for name in ('history', 'spare_history'):
                    value = operations.from_torch(torch.zeros(HISTORY_SHAPE, dtype=torch.bfloat16),
                        device=mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                        memory_config=operations.DRAM_MEMORY_CONFIG,
                        mesh_mapper=operations.ReplicateTensorToMesh(mesh))
                    self.owned.append(value)
                    current = addresses(operations, value)
                    if any(overlaps(current, other) for other in protected):
                        raise ValueError('Pooled draft buffers must own independent chip storage')
                    protected.append(current)
                    pair.append(value)
                self.slots.append(HistorySlot(self, index, *pair))
            operations.synchronize_device(mesh)
        except BaseException:
            self.close()
            raise

    def acquire(self, *, owner='unnamed'):
        if self.closed:
            raise ValueError('Closed serving buffer pool cannot lend a slot')
        slot = next((candidate for candidate in self.slots if not candidate.lent), None)
        if slot is None:
            raise ValueError('All %d pooled draft history slots are already lent; a %dth request '
                             'cannot be admitted' % (self.users, self.users + 1))
        slot.verify()
        # Zeroed on every loan, so a returned slot cannot carry one request's history
        # into the next, and the spare reads exactly as the zeros_like it replaces.
        for value in slot.tensors:
            self.operations.full_like(value, 0.0, optional_tensor=value)
        slot.lent, slot.owner = True, owner
        # So the confirming run shows the slot in use, not a fresh allocation.
        pindiag('[PINDIAG] pool slot {} acquired for {} at {}', slot.index, owner, slot.addresses)
        return slot

    def release(self, slot):
        if self.closed:
            raise ValueError('Closed serving buffer pool cannot take a slot back')
        if slot.pool is not self or not any(slot is candidate for candidate in self.slots):
            raise ValueError('Only a slot lent by this pool may be returned to it')
        if not slot.lent:
            raise ValueError('Pooled draft history slot %d is not lent' % slot.index)
        slot.verify()
        pindiag('[PINDIAG] pool slot {} released by {}', slot.index, slot.owner)
        slot.lent, slot.owner = False, None

    def describe(self):
        return dict(users=self.users, shape=list(HISTORY_SHAPE),
            bytes_per_slot=2 * 2 * HISTORY_SHAPE[2] * HISTORY_SHAPE[3],
            slots=[dict(index=slot.index, lent=slot.lent, owner=slot.owner,
                        addresses=[list(value) for value in slot.addresses]) for slot in self.slots])

    def close(self):
        if self.closed:
            return
        self.closed = True
        lent = [(slot.index, slot.owner) for slot in self.slots if slot.lent]
        self.operations.synchronize_device(self.mesh)
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.slots.clear()
        if lent:
            raise ValueError('Serving buffer pool closed with slots %r still lent' % lent)
