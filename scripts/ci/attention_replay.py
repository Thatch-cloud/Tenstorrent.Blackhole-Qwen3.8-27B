"""Experimental fixed-family attention reader; not wired into request serving.

THE PAGE TABLES ARE THE ONE THING HERE A REQUEST KEEPS. Per bundle the reader holds
a (batches, capacity // 64) page table, uploaded once and rewritten only when the
scheduler's block allocation changes (serving_page_binding.VerifierPageBinding.refresh);
the mask is recomputed inside every replay from the positions word, and that word is
restaged before every verify (verifier_inputs.stage_inputs). So a page table that an
EARLIER request's verify replay writes over - allocated after that trace was captured,
into a hole the trace baked for an intermediate it freed (serving_buffer_pool.py) -
stays wrong from the next verify on, sending every attention layer to another user's
pages. Runs 35492676194 and 35493208438: the second-admitted user's text was right for
its first verify and wrong from its second, after the first user's first replay. Pooled
serving lends the tables through `storage=`, allocated at attach before any trace; the
reader stages the host page table into them and never frees them.
"""

from contextlib import contextmanager
import os

from attention_head_fold import parallel_groups
from attention_mask_replay import execute as refresh_mask, prepare, validate_ticket
from attention_parallel import execute
from gdn_multitoken_conv import addresses, release_owned


# The largest native chunk family any validate_ticket regime admits (the simulator's
# context ladder tops out at 262144 + 256); family_capacities probes up to it.
MAX_FAMILY_CAPACITY = 262400


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
            validate_ticket(max(128, capacity - 256) if short_context else capacity - 256, 8, capacity,
                            short_context=short_context)
        except ValueError:
            continue
        capacities.append(capacity)
    return tuple(capacities)


def bundle_batches(rows, capacity, *, max_group_rows=4, short_context=False):
    """How many groups each bundle of a `rows`-row reader in this family packs - the
    first dimension of its page table - in bundle order. Computed from the family's
    first position exactly as the reader does, so the pool's tables fit it."""
    first = max(128, capacity - 256) if short_context else capacity - 256
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


class ReplayAttentionReader:
    def __init__(self, operations, mesh, rows, capacity, pages_host, upload, *, max_group_rows=4,
                 short_context=False, storage=None):
        import torch

        if type(rows) is not int or rows not in (8, 16, 32):
            raise ValueError('Replay reader requires an explicit T8/T16/T32 bucket')
        if type(max_group_rows) is not int or max_group_rows not in (4, 8):
            raise ValueError('Replay group width must be explicitly four or eight')
        if short_context and max_group_rows != 4:
            raise ValueError('Short-context qualification requires four-row groups')
        if max_group_rows == 8 and os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1':
            raise ValueError('Eight-row replay requires process-fixed compact native scratch')
        if type(capacity) is not int:
            raise ValueError('Integer capacity required')
        first = max(128, capacity - 256) if short_context else capacity - 256
        validate_ticket(first, rows, capacity, short_context=short_context)
        self.short_context = short_context
        if pages_host.ndim != 2 or pages_host.shape[0] != 1 or pages_host.shape[1] < capacity // 64:
            raise ValueError('One complete native cache page table required')
        bundles = parallel_groups(first, rows, max_group_rows=max_group_rows)
        # Pooled page tables (serving_buffer_pool.BucketSlot.batch.replay_pages): staged
        # into and read here, never freed here. Checked before anything is uploaded.
        storage = validate_storage(operations, storage, bundles, capacity)
        self.operations, self.mesh = operations, mesh
        self.rows, self.capacity, self.max_group_rows = rows, capacity, max_group_rows
        self.owned, self.borrowed, self.metadata, self.programs = [], [], [], []
        self.closed = False
        self.failed = False
        self.calls, self.refresh_calls = 0, 0
        self.mask_scope = None
        self.audit = None
        self.start = first
        grid = mesh.compute_with_storage_grid_size()
        try:
            words = torch.zeros(8, dtype=torch.int32)
            words[0] = self.start
            self.positions = upload(words, operations.int32)
            self.owned.append(self.positions)
            for index, bundle in enumerate(bundles):
                count = bundle[0]['rows']
                host = pages_host[:, :capacity // 64].repeat(len(bundle), 1).contiguous()
                if storage is None:
                    pages = upload(host, operations.int32)
                    self.owned.append(pages)
                else:
                    pages = storage[index]
                    self.borrowed.append(pages)
                mask = upload(torch.zeros(len(bundle), 1, count * 12, capacity, dtype=torch.bfloat16))
                self.owned.append(mask)
                config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)
                self.metadata.append((bundle, pages, mask, config))
                self.programs.append(prepare(mesh, self.positions, mask, rows=count, batches=len(bundle),
                    offset=bundle[0]['offset'], capacity=capacity, short_context=short_context))
            if storage is not None:
                self.stage_pages(pages_host)
        except BaseException:
            self.close()
            raise

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

    def validate(self, start):
        if self.closed:
            raise RuntimeError('Replay reader is closed')
        if self.failed:
            raise RuntimeError('Replay reader is poisoned after a failed operation')
        validate_ticket(start, self.rows, self.capacity, short_context=self.short_context)

    def stage(self, start):
        import torch

        self.validate(start)
        if self.mask_scope is not None:
            raise RuntimeError('Cannot stage positions during a shared-mask forward')
        operations = self.operations
        words = torch.zeros(8, dtype=torch.int32)
        words[0] = start
        before = addresses(operations, self.positions)
        source = operations.from_torch(words, dtype=operations.int32, layout=operations.ROW_MAJOR_LAYOUT,
            mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
        try:
            operations.copy_host_to_device_tensor(source, self.positions)
            operations.synchronize_device(self.mesh)
            if addresses(operations, self.positions) != before:
                raise AssertionError('Staging replaced captured position addresses')
        except BaseException:
            self.failed = True
            raise
        self.start = start

    def refresh(self):
        self.validate(self.start)
        for entry, program in zip(self.metadata, self.programs, strict=True):
            if self.short_context:
                self.operations.full_like(entry[2], 0.0, optional_tensor=entry[2])
            refresh_mask(self.positions, entry[2], program)
            self.refresh_calls += 1

    @contextmanager
    def shared_masks(self, expected_calls):
        self.validate(self.start)
        if type(expected_calls) is not int or not 1 <= expected_calls <= 16:
            raise ValueError('Explicit one-to-sixteen attention call budget required')
        if self.mask_scope is not None:
            raise RuntimeError('Shared-mask forwards cannot nest')
        self.mask_scope = self.calls + expected_calls
        try:
            self.refresh()
            yield
            if self.calls != self.mask_scope:
                raise AssertionError('Shared-mask forward did not consume its exact attention call budget')
        except BaseException:
            self.failed = True
            raise
        finally:
            self.mask_scope = None

    def __call__(self, query, keys, values, *, page_table_tensor=None, cur_pos_tensor=None, **kwargs):
        self.validate(self.start)
        if tuple(query.shape) != (1, self.rows, 12, 256):
            raise ValueError('Replay query geometry changed')
        owned = []
        protected = {addresses(self.operations, value) for value in (query, keys, values)}
        try:
            if self.mask_scope is None:
                self.refresh()
            elif self.calls >= self.mask_scope:
                raise AssertionError('Shared-mask forward exceeded its attention call budget')
            result = execute(self.mesh, self.operations, query, keys, values, self.metadata, owned,
                scale=kwargs['scale'], memory_config=kwargs['memory_config'])
            protected.add(addresses(self.operations, result))
            if self.audit is not None:
                self.audit.capture(query, keys, values, result,
                    page_table_tensor=page_table_tensor, cur_pos_tensor=cur_pos_tensor, **kwargs)
            self.calls += 1
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            release_owned(self.operations, [value for value in owned if addresses(self.operations, value) not in protected])

    def close(self):
        if self.closed:
            return
        if self.mask_scope is not None:
            raise RuntimeError('Cannot close a reader during a shared-mask forward')
        if self.audit is not None:
            self.audit.close()
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.closed = True
