"""The replay attention reader over pooled page tables, beside the pinned reader.

WHY A SEPARATE MODULE. attention_replay.py is frozen-recipe evidence: the T16 attention
gates (target_t16_attention_gate.SOURCES) and the combined runtime's admission check
hash every source their reports name, and a changed attention_replay.py refused run
35495227738 (image v42) at its first admission - 'Combined runtime component source
differs: attention_replay.py'. That evidence is not regenerated cheaply, so the pinned
reader keeps its bytes at 8c102b20 and the pooled behaviour lives here, in a module the
image overrides the way it overrides model_batch.py.

WHAT THE POOL CLOSES. The per-bundle page tables are the one thing the replay reader
keeps across steps: (batches, capacity // 64) each, uploaded once at capture, read by all
sixteen attention layers inside the trace, rewritten only when the scheduler's blocks
change (serving_page_binding.VerifierPageBinding.refresh). The mask is recomputed inside
every replay from the positions word and that word is restaged before every verify
(verifier_inputs.stage_inputs), so those two are safe; a page table that an EARLIER
request's verify replay writes over - allocated after that trace was captured, into a
hole the trace baked for an intermediate it freed (serving_buffer_pool.py) - stays wrong
from the next verify on, sending every attention layer to another user's pages. Runs
35492676194 and 35493208438: the second-admitted user's text was right for its first
verify and wrong from its second, after the first user's first replay.

HOW. PooledReplayAttentionReader is the pinned reader with its page-table upload
intercepted. The base asks its `upload` callable for, in order, the 1-D int32 positions
word, then per bundle the 2-D int32 page table and the BF16 mask; the interception hands
back the lent table for every 2-D int32 request and forwards the rest, so metadata,
positions, masks, programs, refresh, shared_masks and __call__ are the base's, and
serving_page_binding.py and verifier_engine.py see the same reader. The tables come
through `storage=`, allocated at attach before any trace; the request's page table is
staged into them at construction the way refresh rewrites them, they are kept in
`borrowed`, and close() frees only what the base uploaded.
"""

from attention_head_fold import parallel_groups
from attention_mask_replay import validate_ticket
from attention_replay import ReplayAttentionReader
from gdn_multitoken_conv import addresses


# The largest native chunk family any validate_ticket regime admits (the simulator's
# context ladder tops out at 262144 + 256); family_capacities probes up to it.
MAX_FAMILY_CAPACITY = 262400


def family_start(capacity, *, short_context=False):
    """The first position of a native chunk family, exactly as the reader computes it."""
    return max(128, capacity - 256) if short_context else capacity - 256


def family_capacities(*, short_context=False, page_width=None):
    """Every native chunk family a replay reader can be captured in, in order: the
    capacities validate_ticket admits in the current regime, cut to the ones a
    `page_width`-column page table can hold (the reader slices its `capacity // 64`
    columns from the request's table). The serving pool allocates one page-table set
    per family because the family is fixed by the request's position at capture, which
    the pool cannot know at attach."""
    if page_width is not None and (type(page_width) is not int or page_width < 1):
        raise ValueError('Positive integer page-table width required')
    capacities = []
    for capacity in range(256, MAX_FAMILY_CAPACITY + 1, 256):
        if page_width is not None and capacity // 64 > page_width:
            break
        # Probed with the narrowest replay width every regime admits (short context is T8 only).
        try:
            validate_ticket(family_start(capacity, short_context=short_context), 8, capacity,
                            short_context=short_context)
        except ValueError:
            continue
        capacities.append(capacity)
    return tuple(capacities)


def bundle_batches(rows, capacity, *, max_group_rows=4, short_context=False):
    """How many groups each bundle of a `rows`-row reader in this family packs - the
    first dimension of its page table - in bundle order. Computed from the family's
    first position exactly as the reader does, so the pool's tables fit it."""
    first = family_start(capacity, short_context=short_context)
    return tuple(len(bundle) for bundle in parallel_groups(first, rows, max_group_rows=max_group_rows))


def validate_storage(operations, storage, bundles, capacity):
    """Pooled page tables, one per bundle in order, of exactly the geometry the trace
    will bake: (batches, capacity // 64), row-major int32. Or None to upload."""
    if storage is None:
        return None
    storage = list(storage)
    if len(storage) != len(bundles):
        raise ValueError('Pooled replay page tables must number the reader bundles: %d for %d'
                         % (len(storage), len(bundles)))
    for table, bundle in zip(storage, bundles, strict=True):
        shape = (len(bundle), capacity // 64)
        if (tuple(table.shape) != shape or table.dtype != operations.int32
                or table.layout != operations.ROW_MAJOR_LAYOUT):
            raise ValueError('Pooled replay page table must be a row-major int32 of shape %r' % (shape,))
    return storage


class PooledReplayAttentionReader(ReplayAttentionReader):
    def __init__(self, operations, mesh, rows, capacity, pages_host, upload, *, storage, max_group_rows=4,
                 short_context=False):
        # The base's geometry checks, first, so the lent tables can be checked against
        # the bundles they must fit before anything is uploaded.
        if (type(rows) is not int or rows not in (8, 16, 32) or type(capacity) is not int
                or type(max_group_rows) is not int or max_group_rows not in (4, 8) or type(short_context) is not bool):
            raise ValueError('Pooled replay reader requires an explicit T8/T16/T32 bucket, an integer capacity '
                             'and four- or eight-row groups')
        first = family_start(capacity, short_context=short_context)
        validate_ticket(first, rows, capacity, short_context=short_context)
        bundles = parallel_groups(first, rows, max_group_rows=max_group_rows)
        tables = validate_storage(operations, storage, bundles, capacity)
        if tables is None:
            raise ValueError('Pooled replay reader needs the lent page tables; ReplayAttentionReader uploads its own')
        self.borrowed = []

        def intercept(value, dtype=None):
            # The base uploads, in order: the 1-D positions word (int32), then per bundle
            # the 2-D page table (int32) and the BF16 mask. Only the tables are lent.
            if dtype == operations.int32 and value.ndim == 2:
                if len(self.borrowed) >= len(tables):
                    raise AssertionError('The pinned reader asked for more page tables than it has bundles')
                table = tables[len(self.borrowed)]
                if tuple(table.shape) != tuple(value.shape):
                    raise AssertionError('The pinned reader asked for a page table of shape %r; the pool lent %r'
                                         % (tuple(value.shape), tuple(table.shape)))
                self.borrowed.append(table)
                return table
            return upload(value) if dtype is None else upload(value, dtype)

        super().__init__(operations, mesh, rows, capacity, pages_host, intercept, max_group_rows=max_group_rows,
                         short_context=short_context)
        self._disown_lent()
        try:
            if len(self.borrowed) != len(tables):
                raise AssertionError('The pinned reader took %d of the %d lent page tables' % (len(self.borrowed), len(tables)))
            self.stage_pages(pages_host)
        except BaseException:
            self.close()
            raise

    def _disown_lent(self):
        """The base appends every table its upload hands back to `owned`; they are the pool's."""
        lent = {id(table) for table in self.borrowed}
        self.owned[:] = [value for value in self.owned if id(value) not in lent]

    def stage_pages(self, pages_host):
        """The request's page table into the lent tables, the way VerifierPageBinding.refresh
        rewrites them on a block change: in place, fenced, addresses checked unchanged."""
        operations = self.operations
        before = [addresses(operations, entry[1]) for entry in self.metadata]
        try:
            for bundle, pages, mask, config in self.metadata:
                host = pages_host[:, :self.capacity // 64].repeat(len(bundle), 1).contiguous()
                source = operations.from_torch(host, dtype=operations.int32, layout=operations.ROW_MAJOR_LAYOUT,
                    mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
                operations.copy_host_to_device_tensor(source, pages)
            operations.synchronize_device(self.mesh)
            if [addresses(operations, entry[1]) for entry in self.metadata] != before:
                raise AssertionError('Staging replaced a pooled replay page table')
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if getattr(self, 'closed', True):
            return
        # Also on the base constructor's own failure path, which calls close() with the
        # tables it was handed still in `owned`.
        self._disown_lent()
        super().close()
