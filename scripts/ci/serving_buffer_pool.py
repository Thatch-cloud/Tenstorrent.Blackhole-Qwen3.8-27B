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

AND THE K/V. Run 35481466425, on the history pool alone: after one request's step the
shard check found the OTHER request's kv_history[0].k differing between the chips
while its checked weights and its pooled history were still bit-identical. The pooled
buffers survived the replay; the one persistent buffer still allocated per request did
not. So each slot also carries the five draft layers' K/V banks - active and spare, k
and v, (1, 4, 2048, 128) each - and DraftKVHistory adopts them through its `storage=`
instead of padding and zeros_like-ing its own.

AND THE VERIFIER. The rule that closes the class: nothing a request keeps across steps
may be allocated after any trace that will replay exists. After the K/V, the audit of
2026-09-20 found these still allocated per request, after an earlier request's traces:
DraftKVHistory's zero query input; the verifier's initial GDN snapshots, its carried
slot-zero state (allocated before its OWN capture, which protects it from nothing an
earlier request captured), one GDN checkpoint set per captured width, the feature taps
and MTP hidden the target copies into per width, and the fixed-shape inputs its
ModelBatch fixture uploads - tokens, positions, the page table and its singleton, the
per-row positions and the rotary tables. Given `helpers` (the 48 ActiveSnapshot
helpers) the pool allocates all of it per slot as well, as a VerifierSlot: two GDN
snapshot sets (initial, carry) plus one BucketSlot per capture the engine can ask for
(verifier_engine.capture_bucket_rows bounds the widths and their multiplicity over
every position), each with its checkpoint set, its feature taps and its fixture
inputs at that width and the pool's page-table width. VerifierEngine and ModelBatch
take them through `storage=`, borrow, and free none of them.

AND THE REPLAY PAGE TABLES. With everything above pooled, runs 35492676194 and
35493208438 still had the second-admitted user right for its first verify and wrong
from its second - after the first user's first replay. The GDN audit found the one
per-request buffer left that is static across steps: the replay attention reader's
per-bundle page tables (attention_replay.py), uploaded at capture and rewritten only
when the scheduler's blocks change (serving_page_binding.py). A replay of the earlier
request's trace writes over them and every attention layer of the later request reads
the wrong KV pages from its next verify on. They are not the request's page table:
each is (batches, capacity // 64) for the capture's 256-position native chunk family,
which the request's position fixes at admission and the pool cannot know at attach.
So every replay-width bucket carries one page-table set per family the reader can be
captured in (attention_replay.family_capacities, cut to the pool's page-table width),
`batch.replay_pages[capacity]`, and the fixture picks its family's. The row tables need
nothing: unpacked, every row's table is the pooled singleton page table.
"""

from types import SimpleNamespace

from attention_replay import bundle_batches, family_capacities
from dflash_device import pindiag
from gdn_multitoken_conv import addresses, release_owned
from serving_fast_policy import NATIVE_GDN_SLOTS


HISTORY_SHAPE = (1, 1, 2048, 5120)
KV_SHAPE = (1, 4, 2048, 128)
QUERY_SHAPE = (1, 1, 32, 2048)
FEATURE_WIDTH = 5120
DRAFT_LAYERS = 5
GDN_LAYERS = 48
SIDES, HEADS = ('active', 'spare'), ('k', 'v')
BATCH_INPUTS = ('tokens', 'positions', 'pages', 'singleton_pages', 'cos', 'sin')


def overlaps(left, right):
    return any(first == second for first, second in zip(left, right, strict=True))


def tensor_bytes(shape, itemsize=2):
    count = 1
    for size in shape:
        count *= size
    return itemsize * count


def bank_tensors(kv):
    """Every K/V tensor of a slot in one fixed order: by layer, active then spare, k then v."""
    return [bank[side][head] for bank in kv for side in SIDES for head in HEADS]


def snapshot_tensors(snapshots):
    """Every tensor of one GDN snapshot set - 48 layers, each the helper's live-state list - in order."""
    return [value for snapshot in snapshots for value in snapshot]


class BucketSlot:
    """One captured verify width of one request: its GDN checkpoints, the feature taps and
    MTP hidden the target copies into, and the fixture inputs whose shapes the trace bakes -
    including, at replay widths, the reader's per-bundle page tables for every family."""

    def __init__(self, rows, checkpoints, target_features, mtp_hidden, batch):
        self.rows = rows
        self.checkpoints = [list(snapshot) for snapshot in checkpoints]
        self.target_features = tuple(target_features)
        self.mtp_hidden = mtp_hidden
        self.batch = batch
        self.taken = False
        self.tensors = (*snapshot_tensors(self.checkpoints), *self.target_features,
                        *(() if mtp_hidden is None else (mtp_hidden,)), *self.batch_tensors())
        # Fully rewritten before any read: the fixture stages every input at construction
        # and before every verify (the replay reader its page tables at construction);
        # only the tiled BF16 buffers are zeroed on loan.
        self.zeroed = (*snapshot_tensors(self.checkpoints), *self.target_features,
                       *(() if mtp_hidden is None else (mtp_hidden,)), batch.cos, batch.sin)

    def replay_tables(self):
        """Every pooled replay page table of the bucket, by family then bundle."""
        tables = getattr(self.batch, 'replay_pages', None) or {}
        return tuple(table for capacity in sorted(tables) for table in tables[capacity])

    def batch_tensors(self):
        return (*(getattr(self.batch, name) for name in BATCH_INPUTS), *self.batch.singleton_positions,
                *self.replay_tables())


class VerifierSlot:
    """A request's verifier storage: initial and carried GDN state, and one BucketSlot per capture."""

    def __init__(self, initial, carry, buckets):
        self.initial = [list(snapshot) for snapshot in initial]
        self.carry = [list(snapshot) for snapshot in carry]
        self.buckets = tuple(buckets)
        self.tensors = (*snapshot_tensors(self.initial), *snapshot_tensors(self.carry),
                        *(value for bucket in self.buckets for value in bucket.tensors))
        self.zeroed = (*snapshot_tensors(self.initial), *snapshot_tensors(self.carry),
                       *(value for bucket in self.buckets for value in bucket.zeroed))

    def take(self, rows):
        """The first free bucket of exactly this width; a request wanting more than the pool
        holds is refused here, at admission, rather than allocating after a trace."""
        for bucket in self.buckets:
            if bucket.rows == rows and not bucket.taken:
                bucket.taken = True
                return bucket
        raise ValueError('Pooled verifier storage has no free %d-row bucket: the slot holds widths %r'
                         % (rows, [bucket.rows for bucket in self.buckets]))

    def reset(self):
        for bucket in self.buckets:
            bucket.taken = False


class HistorySlot:
    """One request's history pair, K/V banks, query and verifier storage; the addresses are
    recorded at allocation and never move."""

    def __init__(self, pool, index, history, spare_history, kv, query=None, verifier=None, bytes_allocated=0):
        self.pool, self.index = pool, index
        self.history, self.spare_history, self.kv = history, spare_history, tuple(kv)
        self.query, self.verifier = query, verifier
        self.bytes = bytes_allocated
        self.tensors = (history, spare_history, *bank_tensors(self.kv),
                        *(() if query is None else (query,)),
                        *(() if verifier is None else verifier.tensors))
        self.zeroed = (history, spare_history, *bank_tensors(self.kv),
                       *(() if query is None else (query,)),
                       *(() if verifier is None else verifier.zeroed))
        self.addresses = tuple(addresses(pool.operations, value) for value in self.tensors)
        self.lent = False
        self.owner = None

    def verify(self):
        current = tuple(addresses(self.pool.operations, value) for value in self.tensors)
        if current != self.addresses:
            moved = [(index, before, after) for index, (before, after) in enumerate(zip(self.addresses, current, strict=True))
                     if before != after][:4]
            raise AssertionError('Pooled draft history slot %d moved: %r became %r (first moved: %r)'
                                 % (self.index, self.addresses[:2], current[:2], moved))

    def release(self):
        self.pool.release(self)

    def describe(self):
        operations = self.pool.operations
        banks = self.addresses[2:2 + 4 * len(self.kv)]
        report = dict(index=self.index, lent=self.lent, owner=self.owner,
            addresses=[list(value) for value in self.addresses[:2]],
            kv=[{side: {head: list(banks[layer * 4 + offset * 2 + position]) for position, head in enumerate(HEADS)}
                 for offset, side in enumerate(SIDES)} for layer in range(len(self.kv))])
        if self.query is not None:
            report['query'] = list(addresses(operations, self.query))
        if self.verifier is not None:
            def first(snapshots):
                return list(addresses(operations, snapshots[0][0]))

            report['verifier'] = dict(
                initial=first(self.verifier.initial), carry=first(self.verifier.carry),
                buckets=[dict(rows=bucket.rows, taken=bucket.taken, checkpoints=first(bucket.checkpoints),
                              target_features=[list(addresses(operations, value)) for value in bucket.target_features],
                              mtp_hidden=None if bucket.mtp_hidden is None else list(addresses(operations, bucket.mtp_hidden)),
                              batch={name: list(addresses(operations, getattr(bucket.batch, name))) for name in BATCH_INPUTS},
                              singleton_positions=[list(addresses(operations, value)) for value in bucket.batch.singleton_positions],
                              replay_pages={capacity: [list(addresses(operations, table)) for table in tables]
                                            for capacity, tables in sorted((getattr(bucket.batch, 'replay_pages', None) or {}).items())})
                         for bucket in self.verifier.buckets])
        return report


class ServingBufferPool:
    def __init__(self, operations, mesh, *, users, helpers=None, page_width=None, bucket_rows=(),
                 feature_taps=0, rope=None, mtp_hidden=False, replay_group_rows=4, replay_capacities=None):
        import torch

        if type(users) is not int or not 1 <= users <= NATIVE_GDN_SLOTS:
            raise ValueError('Explicit scheduler request count within the %d native GDN slots required'
                             % NATIVE_GDN_SLOTS)
        bucket_rows = tuple(bucket_rows)
        if helpers is not None:
            helpers = tuple(helpers)
            if (len(helpers) != GDN_LAYERS or any(not callable(getattr(helper, 'allocate', None)) for helper in helpers)
                    or type(page_width) is not int or page_width < 1
                    or not bucket_rows or any(type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32) for rows in bucket_rows)
                    or type(feature_taps) is not int or feature_taps < 0 or not callable(rope)
                    or type(mtp_hidden) is not bool):
                raise ValueError('Verifier storage needs all %d GDN helpers, a page-table width, the capture widths, '
                                 'a feature tap count and the rotary table builder' % GDN_LAYERS)
            # The replay reader's page tables: one set per native chunk family the request
            # can be captured in - by default every family the regime admits that the
            # page table can hold - each bundle's sized by the reader's own grouping.
            if type(replay_group_rows) is not int or replay_group_rows not in (4, 8):
                raise ValueError('Replay group width must be integer four or eight')
            admitted = family_capacities(page_width=page_width)
            if replay_capacities is None:
                replay_capacities = admitted
            replay_capacities = tuple(replay_capacities)
            if (len(set(replay_capacities)) != len(replay_capacities)
                    or any(type(capacity) is not int or capacity not in admitted for capacity in replay_capacities)):
                raise ValueError('Replay families must be distinct native chunk capacities the page table holds: %r'
                                 % (admitted,))
        elif (page_width is not None or bucket_rows or feature_taps or rope is not None or mtp_hidden
                or replay_group_rows != 4 or replay_capacities is not None):
            raise ValueError('Verifier storage geometry without the GDN helpers that shape it')
        self.operations, self.mesh, self.users = operations, mesh, users
        self.helpers, self.page_width, self.bucket_rows = helpers, page_width, bucket_rows
        self.feature_taps, self.mtp_hidden = feature_taps, mtp_hidden
        self.replay_group_rows, self.replay_capacities = replay_group_rows, replay_capacities
        self.owned, self.slots = [], []
        self.closed = False
        try:
            protected = []
            counted = [0]

            def adopt(value, count):
                self.owned.append(value)
                current = addresses(operations, value)
                if any(overlaps(current, other) for other in protected):
                    raise ValueError('Pooled draft buffers must own independent chip storage')
                protected.append(current)
                counted[0] += count
                return value

            def allocate(shape, dtype=None, layout=None, mapper=None, itemsize=2):
                """Replicated tiled BF16 zeros by default; the fixture inputs are row-major
                integers and the feature taps are sharded on the feature axis, like the
                buffers they replace (model_batch.py, verifier_engine.py)."""
                host = torch.zeros(shape, dtype=torch.bfloat16 if dtype is None else torch.int32)
                value = operations.from_torch(host, device=mesh,
                    dtype=operations.bfloat16 if dtype is None else dtype,
                    layout=operations.TILE_LAYOUT if layout is None else layout,
                    memory_config=operations.DRAM_MEMORY_CONFIG,
                    mesh_mapper=operations.ReplicateTensorToMesh(mesh) if mapper is None else mapper)
                return adopt(value, tensor_bytes(shape, itemsize))

            def snapshot_set():
                """One slot-zero snapshot per GDN layer, exactly as the engine allocated its own."""
                snapshots = []
                for helper in helpers:
                    snapshot = [adopt(value, tensor_bytes(tuple(value.shape))) for value in helper.allocate()]
                    snapshots.append(snapshot)
                return snapshots

            def bucket_slot(rows):
                checkpoints = snapshot_set()
                features = [allocate((1, 1, rows, FEATURE_WIDTH), mapper=operations.ShardTensorToMesh(mesh, dim=3))
                            for tap in range(feature_taps)]
                hidden = allocate((1, 1, rows, FEATURE_WIDTH)) if mtp_hidden else None
                integers = dict(dtype=operations.int32, layout=operations.ROW_MAJOR_LAYOUT, itemsize=4)
                batch = SimpleNamespace(
                    tokens=allocate((rows, 1), **dict(integers, dtype=operations.uint32)),
                    positions=allocate((rows,), **integers),
                    pages=allocate((rows, page_width), **integers),
                    singleton_pages=allocate((1, page_width), **integers),
                    singleton_positions=[allocate((1,), **integers) for row in range(rows)],
                    # Replay widths only (ModelBatch replays from eight rows): per family,
                    # one (batches, capacity // 64) table per reader bundle, in bundle order.
                    replay_pages={capacity: [allocate((batches, capacity // 64), **integers)
                                             for batches in bundle_batches(rows, capacity, max_group_rows=replay_group_rows)]
                                  for capacity in (replay_capacities if rows >= 8 else ())})
                cos, sin = rope(torch.arange(rows, dtype=torch.int32))
                batch.cos, batch.sin = (adopt(value, tensor_bytes(tuple(value.shape))) for value in (cos, sin))
                return BucketSlot(rows, checkpoints, features, hidden, batch)

            for index in range(users):
                counted[0] = 0
                pair = [allocate(HISTORY_SHAPE) for name in ('history', 'spare_history')]
                # The K/V banks DraftKVHistory keeps for the request's whole life - the
                # buffer run 35481466425 found overwritten while the pooled pair survived.
                kv = [{side: {head: allocate(KV_SHAPE) for head in HEADS} for side in SIDES}
                      for layer in range(DRAFT_LAYERS)]
                query = verifier = None
                if helpers is not None:
                    query = allocate(QUERY_SHAPE)
                    verifier = VerifierSlot(snapshot_set(), snapshot_set(), [bucket_slot(rows) for rows in bucket_rows])
                self.slots.append(HistorySlot(self, index, *pair, kv, query, verifier, counted[0]))
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
        for value in slot.zeroed:
            self.operations.full_like(value, 0.0, optional_tensor=value)
        if slot.verifier is not None:
            slot.verifier.reset()
        slot.lent, slot.owner = True, owner
        # So the confirming run shows the slot in use, not a fresh allocation; every
        # bank's address is in the stage line printed at attach.
        pindiag('[PINDIAG] pool slot {} acquired for {}: history at {}, {} K/V banks from {}',
                slot.index, owner, slot.addresses[:2], 4 * len(slot.kv), slot.addresses[2])
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
        if slot.verifier is not None:
            slot.verifier.reset()

    def describe(self):
        draft_bytes = 2 * tensor_bytes(HISTORY_SHAPE) + 4 * DRAFT_LAYERS * tensor_bytes(KV_SHAPE)
        report = dict(users=self.users, shape=list(HISTORY_SHAPE), kv_shape=list(KV_SHAPE), layers=DRAFT_LAYERS,
            bytes_per_slot=draft_bytes, slots=[slot.describe() for slot in self.slots])
        if self.helpers is not None:
            slot_bytes = self.slots[0].bytes if self.slots else draft_bytes
            replay_bytes = sum(tensor_bytes(tuple(table.shape), 4) for bucket in self.slots[0].verifier.buckets
                               for table in bucket.replay_tables()) if self.slots else 0
            report.update(bytes_per_slot=slot_bytes, draft_bytes_per_slot=draft_bytes,
                verifier_bytes_per_slot=slot_bytes - draft_bytes - tensor_bytes(QUERY_SHAPE),
                query_bytes_per_slot=tensor_bytes(QUERY_SHAPE), query_shape=list(QUERY_SHAPE),
                page_width=self.page_width, bucket_rows=list(self.bucket_rows),
                gdn_snapshot_sets=2 + len(self.bucket_rows), feature_taps=self.feature_taps,
                mtp_hidden=self.mtp_hidden, replay_group_rows=self.replay_group_rows,
                replay_capacities=list(self.replay_capacities), replay_page_bytes_per_slot=replay_bytes)
        return report

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
