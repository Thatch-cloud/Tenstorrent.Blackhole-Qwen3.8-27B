"""Experimental fixed-family attention reader; not wired into request serving."""

from contextlib import contextmanager
import os

from attention_head_fold import parallel_groups
from attention_mask_replay import execute as refresh_mask, prepare, validate_ticket
from attention_parallel import execute
from gdn_multitoken_conv import addresses, release_owned


class ReplayAttentionReader:
    def __init__(self, operations, mesh, rows, capacity, pages_host, upload, *, max_group_rows=4,
                 short_context=False):
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
        self.operations, self.mesh = operations, mesh
        self.rows, self.capacity, self.max_group_rows = rows, capacity, max_group_rows
        self.owned, self.metadata, self.programs = [], [], []
        self.closed = False
        self.failed = False
        self.calls, self.refresh_calls = 0, 0
        self.mask_scope = None
        self.start = first
        grid = mesh.compute_with_storage_grid_size()
        try:
            words = torch.zeros(8, dtype=torch.int32)
            words[0] = self.start
            self.positions = upload(words, operations.int32)
            self.owned.append(self.positions)
            for bundle in parallel_groups(self.start, rows, max_group_rows=max_group_rows):
                count = bundle[0]['rows']
                pages = upload(pages_host[:, :capacity // 64].repeat(len(bundle), 1).contiguous(), operations.int32)
                self.owned.append(pages)
                mask = upload(torch.zeros(len(bundle), 1, count * 12, capacity, dtype=torch.bfloat16))
                self.owned.append(mask)
                config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)
                self.metadata.append((bundle, pages, mask, config))
                self.programs.append(prepare(mesh, self.positions, mask, rows=count, batches=len(bundle),
                    offset=bundle[0]['offset'], capacity=capacity, short_context=short_context))
        except BaseException:
            self.close()
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
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.closed = True
