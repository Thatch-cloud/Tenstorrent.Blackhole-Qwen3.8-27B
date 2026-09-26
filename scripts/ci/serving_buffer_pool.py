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
captured in (pooled_attention_replay.family_capacities, cut to the pool's page-table
width), `batch.replay_pages[capacity]`, and the fixture picks its family's and builds
the pooled reader over them (attention_replay.py itself is frozen-recipe evidence and
stays byte-exact). The row tables need nothing: unpacked, every row's table is the
pooled singleton page table.

AND THE PACKED BLOCK'S. The packed verify block (packed_verifier.py) is built at attach,
after this pool and before any request, so what it allocates itself is pre-trace by
construction; its attention, though, is one replay reader PER USER (M1b,
pooled_attention_replay.PackedReplayAttentionReader), and those readers' per-bundle page
tables are the same kind of static-across-steps buffer the request buckets pool. They
come from here too: per packed shape the serving block can take - M1 two T16 users in
the 32-row block, M3 four T16 users in the 64-row block, keyed (users, rows_per_user) -
one table set per user per family, each shaped as a rows_per_user-row reader bundles it
(`packed_replay(users, rows)`), lent once to the block for the pool's life. The block
restages every user's table into them before each verify, so they are not zeroed. The
serving runtime names the one shape its block takes (`packed_shapes=`); by default the
pool holds every default shape its slot count can fill.

AND TWO BLOCKS OF THE SAME SHAPE. QWEN_FAST_FOUR_AS_TWO builds two 32-row M1 blocks over
one four-slot pool instead of the single 64-row M3 block - block A over slots (0, 1),
block B over (2, 3) - and each needs its OWN lent (2, 16) table set: one set is sized for
one block's two users, not both blocks' four. `packed_shapes=` still names the shape once
(a shape repeated there is refused, same as ever); the multiplicity is a separate
`packed_replicas={(2, 16): 2}`, which builds that many independent sets under the one key.
`packed_replay(2, 16)` then lends the first untaken one each time it is asked, so the two
blocks' own `PackedVerifierEngine.__init__` calls - made in turn, each immediately
`take()`-ing what it got back - end up with two different sets without either naming which.

AND THE EXTENT READERS' (S2, `extent_replay=True`, under QWEN_FAST_EXTENT_REPLAY=1). The S2 block
serves each user at its own 256-key family through one captured program (K64j flag 0x20,
extent_attention_replay.py): no family is fixed at capture, so no per-family table set is needed.
Instead each packed user, per bundle of the extent readers' layout, borrows one FULL-WIDTH table
(batches, page_width) - the same shape as today's table at the widest family C = page_width * 64 -
and one (batches,) int32 cur_pos word per entry, which K64j reads in-trace (F20: row-major int32,
interleaved, padded_shape[-1] == batches). Both are state kept across steps, the class this pool
exists for, so both are allocated here, zeroed, before any trace, and lent once as a
PackedExtentStorage (`packed_extent(users, rows)`). The per-family tables are not built at all:
the family set collapses to C, and the per-request bucket slots must hold no replay width (they
capture at 1, 2 and 4 rows beside the block), since their pinned readers would need the families.
"""

from types import SimpleNamespace

from attention_head_fold import parallel_groups
from dflash_device import pindiag
from pooled_attention_replay import bundle_batches, family_capacities
from gdn_multitoken_conv import addresses, release_owned
from packed_shapes import BLOCK_ROWS as PACKED_BLOCK_WIDTHS
from serving_fast_policy import NATIVE_GDN_SLOTS


HISTORY_SHAPE = (1, 1, 2048, 5120)
KV_SHAPE = (1, 4, 2048, 128)
# The draft cache's zero query input (draft_kv_history.QUERY_SHAPE): the draft proposes
# in 32-row passes whatever the verify block's width, so this is not a verify-block pin.
QUERY_SHAPE = (1, 1, 32, 2048)
FEATURE_WIDTH = 5120
DRAFT_LAYERS = 5
GDN_LAYERS = 48
SIDES, HEADS = ('active', 'spare'), ('k', 'v')
BATCH_INPUTS = ('tokens', 'positions', 'pages', 'singleton_pages', 'cos', 'sin')
# The packed shapes the serving block takes by default, (users, rows_per_user): M1 two
# T16 users in the 32-row block, M3 four T16 users in the 64-row block
# (packed_shapes.serving_shape). M2's (4, 8) is still accepted when given explicitly.
PACKED_REPLAY_SHAPES = ((2, 16), (4, 16))
PACKED_BLOCK_ROWS = max(PACKED_BLOCK_WIDTHS)
# The extent readers bundle every segment as the native chunk grouping at position 256
# (extent_attention_replay.LAYOUT); test_extent_attention_replay pins this copy equal to it.
EXTENT_LAYOUT_START = 256


def extent_bundle_batches(rows, group_rows):
    """How many groups each bundle of a `rows`-row extent segment packs, in bundle order: the first
    dimension of its lent table and the length of its cur_pos."""
    return tuple(len(bundle) for bundle in parallel_groups(EXTENT_LAYOUT_START, rows, max_group_rows=group_rows))


def default_packed_shapes(users):
    """The packed shapes a pool for `users` scheduler slots can be asked to serve."""
    return tuple(shape for shape in PACKED_REPLAY_SHAPES if shape[0] <= users)


def validate_packed_shapes(shapes):
    try:
        shapes = tuple(tuple(shape) for shape in shapes)
    except TypeError:
        raise ValueError('Packed replay shapes must be (users, rows_per_user) pairs') from None
    if (len(set(shapes)) != len(shapes)
            or any(len(shape) != 2 or any(type(value) is not int for value in shape) or shape[0] < 1
                   or shape[1] not in (8, 16, 32) or shape[0] * shape[1] not in PACKED_BLOCK_WIDTHS
                   for shape in shapes)):
        raise ValueError('Packed replay shapes must be distinct (users, rows_per_user) pairs of T8/T16/T32 users '
                         'filling one legal block width up to %d rows' % PACKED_BLOCK_ROWS)
    return shapes


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


class PackedReplayTables:
    """The packed block's per-user replay page tables for one shape: per family, one table
    list per user, shaped as that user's rows_per_user-row reader bundles them (M1b,
    pooled_attention_replay.PackedReplayAttentionReader). Lent once, to the block."""

    def __init__(self, users, rows, replay_pages):
        self.users, self.rows = users, rows
        self.replay_pages = {capacity: [list(tables) for tables in per_user] for capacity, per_user in replay_pages.items()}
        self.taken = False
        self.tensors = tuple(self.tables())

    def tables(self):
        """Every table, by family then user then bundle."""
        return tuple(table for capacity in sorted(self.replay_pages)
                     for tables in self.replay_pages[capacity] for table in tables)

    def take(self):
        if self.taken:
            raise ValueError('The pooled replay page tables for %d x T%d packed users are already lent'
                             % (self.users, self.rows))
        self.taken = True
        return self

    def release(self):
        self.taken = False

    def describe(self, operations):
        return dict(users=self.users, rows=self.rows, taken=self.taken,
            bytes=sum(tensor_bytes(tuple(table.shape), 4) for table in self.tensors),
            replay_pages={capacity: [[list(addresses(operations, table)) for table in tables] for tables in per_user]
                          for capacity, per_user in sorted(self.replay_pages.items())})


class PackedExtentStorage:
    """The S2 block's per-user extent storage for one shape: per user, per bundle of the extent
    layout, a full-width (batches, page_width) page table and a (batches,) int32 cur_pos
    (extent_attention_replay.ExtentSegmentReader's `storage`). Lent once, to the block; restaged by
    the block before every verify and at reader construction, so never zeroed on loan."""

    def __init__(self, users, rows, tables, cur_pos):
        tables = tuple(tuple(per_user) for per_user in tables)
        cur_pos = tuple(tuple(per_user) for per_user in cur_pos)
        if (len(tables) != users or len(cur_pos) != users
                or any(len(user_tables) != len(user_positions) for user_tables, user_positions in zip(tables, cur_pos))):
            raise ValueError('Extent storage needs one table and one cur_pos per bundle for each of %d users' % users)
        self.users, self.rows = users, rows
        self.tables, self.cur_pos = tables, cur_pos
        self.taken = False
        self.tensors = tuple(tensor for user in range(users) for pair in zip(tables[user], cur_pos[user])
                             for tensor in pair)

    def segment_storage(self):
        """Per user, its (table, cur_pos) per bundle: one PackedExtentReplayReader segment's `storage`."""
        return tuple(tuple(zip(self.tables[user], self.cur_pos[user])) for user in range(self.users))

    def take(self):
        if self.taken:
            raise ValueError('The pooled extent storage for %d x T%d packed users is already lent' % (self.users, self.rows))
        self.taken = True
        return self

    def release(self):
        self.taken = False

    def describe(self, operations):
        return dict(users=self.users, rows=self.rows, taken=self.taken,
            bytes=sum(tensor_bytes(tuple(tensor.shape), 4) for tensor in self.tensors),
            tables=[[list(addresses(operations, table)) for table in per_user] for per_user in self.tables],
            cur_pos=[[list(addresses(operations, positions)) for positions in per_user] for per_user in self.cur_pos])


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


def dram_statistics(operations, tensor):
    """Each chip's DRAM allocator figures, in bytes over all banks, read from the chips a
    device tensor spans: allocated, free, the largest free block (the largest buffer that
    could still be allocated) and the total. A diagnostic, never a gate: a ttnn without
    the memory view, or one that refuses it, reports the reason instead of raising."""
    try:
        report = []
        for chip, shard in enumerate(operations.get_device_tensors(tensor)):
            view = operations.get_memory_view(shard.device(), operations.BufferType.DRAM)
            banks = int(view.num_banks)
            report.append(dict(chip=chip, banks=banks,
                allocated=int(view.total_bytes_allocated_per_bank) * banks,
                free=int(view.total_bytes_free_per_bank) * banks,
                largest_free=int(view.largest_contiguous_bytes_free_per_bank) * banks,
                total=int(view.total_bytes_per_bank) * banks))
        return report
    except BaseException as failure:
        return dict(unavailable='%s: %s' % (type(failure).__name__, str(failure)[:120]))


def format_dram(statistics):
    """One line for the log: per chip, allocated / free / largest free block of the total."""
    if isinstance(statistics, dict):
        return 'unavailable (%s)' % statistics.get('unavailable', 'no statistics')
    gigabyte, megabyte = 1e9, 1e6
    return '; '.join('chip%d allocated=%.2fGB free=%.2fGB largest_free=%.1fMB of %.2fGB'
                     % (chip['chip'], chip['allocated'] / gigabyte, chip['free'] / gigabyte,
                        chip['largest_free'] / megabyte, chip['total'] / gigabyte) for chip in statistics)


def dram_line(pool):
    """The pool's DRAM statistics formatted for a [PINDIAG] line; never raises."""
    try:
        statistics = getattr(pool, 'dram_statistics', None)
        if not callable(statistics):
            return 'unavailable (pool without device statistics)'
        return format_dram(statistics())
    except BaseException as failure:
        return 'unavailable (%s: %s)' % (type(failure).__name__, str(failure)[:120])


class ServingBufferPool:
    def dram_statistics(self):
        """The DRAM allocator's figures per chip, read through the pool's first buffer."""
        if self.closed or not self.owned:
            return dict(unavailable='no pooled buffer to read the chips through')
        return dram_statistics(self.operations, self.owned[0])

    def __init__(self, operations, mesh, *, users, helpers=None, page_width=None, bucket_rows=(),
                 feature_taps=0, rope=None, mtp_hidden=False, replay_group_rows=4, replay_capacities=None,
                 packed_shapes=None, packed_replicas=None, packed_replay_group_rows=None, extent_replay=False):
        import torch

        if type(extent_replay) is not bool:
            raise ValueError('Extent replay storage must be selected by an explicit bool')
        if type(users) is not int or not 1 <= users <= NATIVE_GDN_SLOTS:
            raise ValueError('Explicit scheduler request count within the %d native GDN slots required'
                             % NATIVE_GDN_SLOTS)
        bucket_rows = tuple(bucket_rows)
        if helpers is not None:
            # The packed block's per-user replay tables: by default every packed shape this
            # many scheduler slots can fill, or an explicit tuple of (users, rows_per_user)
            # (serving_runtime names the one shape of the block it builds over this pool).
            packed_shapes = validate_packed_shapes(default_packed_shapes(users) if packed_shapes is None else packed_shapes)
            # How many INDEPENDENT table sets to lend for each shape, keyed the same way -
            # 1 unless named. Two blocks of the same shape (QWEN_FAST_FOUR_AS_TWO's pair of
            # 32-row M1 blocks) each need their own lent set, but `packed_shapes` itself
            # still names each distinct shape once (validate_packed_shapes keeps refusing a
            # shape repeated there): the multiplicity is this separate mapping instead.
            if packed_replicas is None:
                packed_replicas = {}
            else:
                packed_replicas = dict(packed_replicas)
                if (any(shape not in packed_shapes for shape in packed_replicas)
                        or any(type(count) is not int or count < 1 for count in packed_replicas.values())):
                    raise ValueError('Packed replay replica counts must be positive integers naming shapes the pool holds')
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
            # The packed block's own grouping, independent of the per-request bucket
            # slots' replay_group_rows above: None (the default) keeps today's behaviour
            # of bundling the packed tables at the SAME width as the bucket slots; named
            # explicitly, only PackedReplayTables (below) bundles at it - the per-request
            # engines (serving_request_factory.py) always capture at replay_group_rows=4
            # and must never see this value.
            if packed_replay_group_rows is None:
                packed_replay_group_rows = replay_group_rows
            elif type(packed_replay_group_rows) is not int or packed_replay_group_rows not in (4, 8):
                raise ValueError('Packed replay group width must be integer four or eight')
            if extent_replay:
                # S2: one full-width table set at C = page_width * 64 and nothing per family. The
                # extent readers never ask validate_ticket, so C need not be an admitted family; the
                # bucket slots must hold no replay width, whose pinned readers would need the families.
                extent_capacity = page_width * 64
                if page_width % 4:
                    raise ValueError('Extent replay needs a page table of whole 256-key families, got width %d'
                                     % page_width)
                if replay_capacities is not None and tuple(replay_capacities) != (extent_capacity,):
                    raise ValueError('Extent replay pools its tables at C = %d only, not %r'
                                     % (extent_capacity, tuple(replay_capacities)))
                if any(rows >= 8 for rows in bucket_rows):
                    raise ValueError('Extent replay pools no per-family replay tables: capture widths %r include a '
                                     'replay width (8 or more rows)' % (bucket_rows,))
                replay_capacities = (extent_capacity,)
            else:
                admitted = family_capacities(page_width=page_width)
                if replay_capacities is None:
                    replay_capacities = admitted
                replay_capacities = tuple(replay_capacities)
                if (len(set(replay_capacities)) != len(replay_capacities)
                        or any(type(capacity) is not int or capacity not in admitted for capacity in replay_capacities)):
                    raise ValueError('Replay families must be distinct native chunk capacities the page table holds: %r'
                                     % (admitted,))
        elif (page_width is not None or bucket_rows or feature_taps or rope is not None or mtp_hidden
                or replay_group_rows != 4 or replay_capacities is not None or packed_shapes or packed_replicas
                or packed_replay_group_rows is not None or extent_replay):
            raise ValueError('Verifier storage geometry without the GDN helpers that shape it')
        else:
            packed_shapes, packed_replicas = (), {}
        self.operations, self.mesh, self.users = operations, mesh, users
        self.helpers, self.page_width, self.bucket_rows = helpers, page_width, bucket_rows
        self.feature_taps, self.mtp_hidden = feature_taps, mtp_hidden
        self.replay_group_rows, self.replay_capacities = replay_group_rows, replay_capacities
        self.packed_replay_group_rows = packed_replay_group_rows
        self.packed_shapes, self.packed_replicas = packed_shapes, packed_replicas
        self.extent_replay = extent_replay
        self.owned, self.slots = [], []
        self.packed, self.packed_bytes = {}, 0
        self.extent = {}
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
            # The packed block's per-user replay page tables (PackedReplayTables): per shape,
            # per family, one (batches, capacity // 64) table per bundle of a rows_per_user-row
            # reader, for every user of the shape. Not per slot: one block serves them all -
            # or, with `packed_replicas` naming more than one, one independent set per block
            # of that shape (QWEN_FAST_FOUR_AS_TWO's two 32-row blocks each lend their own).
            for count, rows in packed_shapes:
                if extent_replay:
                    # Per user, per bundle of the extent layout: the full-width table, then its
                    # cur_pos, zeroed, independent on both chips (adopt), before any trace.
                    self.extent[(count, rows)] = []
                    batches = extent_bundle_batches(rows, packed_replay_group_rows)
                    for replica in range(packed_replicas.get((count, rows), 1)):
                        counted[0] = 0
                        integers = dict(dtype=operations.int32, layout=operations.ROW_MAJOR_LAYOUT, itemsize=4)
                        tables, cur_pos = [], []
                        for user in range(count):
                            tables.append([])
                            cur_pos.append([])
                            for entries in batches:
                                tables[-1].append(allocate((entries, page_width), **integers))
                                cur_pos[-1].append(allocate((entries,), **integers))
                        self.extent[(count, rows)].append(PackedExtentStorage(count, rows, tables, cur_pos))
                        self.packed_bytes += counted[0]
                    continue
                self.packed[(count, rows)] = []
                for replica in range(packed_replicas.get((count, rows), 1)):
                    counted[0] = 0
                    integers = dict(dtype=operations.int32, layout=operations.ROW_MAJOR_LAYOUT, itemsize=4)
                    tables = {capacity: [[allocate((batches, capacity // 64), **integers)
                                          for batches in bundle_batches(rows, capacity, max_group_rows=packed_replay_group_rows)]
                                         for user in range(count)]
                              for capacity in replay_capacities}
                    self.packed[(count, rows)].append(PackedReplayTables(count, rows, tables))
                    self.packed_bytes += counted[0]
            operations.synchronize_device(mesh)
        except BaseException:
            self.close()
            raise

    def packed_replay(self, users, rows):
        """The packed block's replay page tables for (users, rows_per_user), to be taken once.

        With one set for the shape (every shape but a named replica count) this is exactly
        as before, taken or not. With several - QWEN_FAST_FOUR_AS_TWO's pair of 32-row
        blocks - the first untaken set is handed out, so each block calling this in turn
        gets its OWN set; once every set is taken, the last one is returned again and its
        own `take()` raises 'already lent', exactly as the single-set case always did."""
        if self.closed:
            raise ValueError('Closed serving buffer pool cannot lend packed replay page tables')
        if self.extent_replay:
            raise ValueError('The pool holds S2 extent storage (extent_replay) and no per-family packed replay '
                             'tables: ask packed_extent')
        candidates = self.packed.get((users, rows))
        if not candidates:
            raise ValueError('The pool holds packed replay page tables for shapes %r; %r was asked for'
                             % (sorted(self.packed), (users, rows)))
        return next((tables for tables in candidates if not tables.taken), candidates[-1])

    def packed_extent(self, users, rows):
        """The S2 block's extent storage for (users, rows_per_user), to be taken once: the first
        untaken set, as packed_replay lends its tables."""
        if self.closed:
            raise ValueError('Closed serving buffer pool cannot lend packed extent storage')
        if not self.extent_replay:
            raise ValueError('The pool was built without extent_replay: it holds per-family packed replay tables, '
                             'no extent storage')
        candidates = self.extent.get((users, rows))
        if not candidates:
            raise ValueError('The pool holds packed extent storage for shapes %r; %r was asked for'
                             % (sorted(self.extent), (users, rows)))
        return next((storage for storage in candidates if not storage.taken), candidates[-1])

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
                replay_capacities=list(self.replay_capacities), replay_page_bytes_per_slot=replay_bytes,
                packed_replay_group_rows=self.packed_replay_group_rows,
                packed_shapes=[list(shape) for shape in self.packed_shapes], packed_replay_bytes=self.packed_bytes,
                packed_replay=[tables.describe(self.operations) for group in self.packed.values() for tables in group],
                # Only under extent_replay: flag off, the attach line reads exactly as before.
                **({} if not self.extent_replay else dict(extent_replay=True, packed_extent=[
                    storage.describe(self.operations) for group in self.extent.values() for storage in group])))
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
        self.packed.clear()
        self.extent.clear()
        if lent:
            raise ValueError('Serving buffer pool closed with slots %r still lent' % lent)
