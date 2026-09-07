"""Experimental fixed-family attention reader; not wired into request serving."""

from attention_head_fold import parallel_groups
from attention_mask_replay import execute as refresh_mask, prepare, validate_ticket
from attention_parallel import execute
from gdn_multitoken_conv import addresses, release_owned


class ReplayAttentionReader:
    def __init__(self, operations, mesh, rows, capacity, pages_host, upload):
        import torch

        if type(rows) is not int or rows not in (8, 16, 32):
            raise ValueError('Replay reader requires an explicit T8/T16/T32 bucket')
        if type(capacity) is not int:
            raise ValueError('Integer capacity required')
        validate_ticket(capacity - 256, rows, capacity)
        if pages_host.ndim != 2 or pages_host.shape[0] != 1 or pages_host.shape[1] < capacity // 64:
            raise ValueError('One complete native cache page table required')
        self.operations, self.mesh = operations, mesh
        self.rows, self.capacity = rows, capacity
        self.owned, self.metadata, self.programs = [], [], []
        self.closed = False
        self.failed = False
        self.calls, self.refresh_calls = 0, 0
        self.start = capacity - 256
        grid = mesh.compute_with_storage_grid_size()
        try:
            words = torch.zeros(8, dtype=torch.int32)
            words[0] = self.start
            self.positions = upload(words, operations.int32)
            self.owned.append(self.positions)
            for bundle in parallel_groups(self.start, rows):
                count = bundle[0]['rows']
                pages = upload(pages_host[:, :capacity // 64].repeat(len(bundle), 1).contiguous(), operations.int32)
                self.owned.append(pages)
                mask = upload(torch.zeros(len(bundle), 1, count * 12, capacity, dtype=torch.bfloat16))
                self.owned.append(mask)
                config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)
                self.metadata.append((bundle, pages, mask, config))
                self.programs.append(prepare(mesh, self.positions, mask, rows=count, batches=len(bundle),
                    offset=bundle[0]['offset'], capacity=capacity))
        except BaseException:
            self.close()
            raise

    def validate(self, start):
        if self.closed:
            raise RuntimeError('Replay reader is closed')
        if self.failed:
            raise RuntimeError('Replay reader is poisoned after a failed operation')
        validate_ticket(start, self.rows, self.capacity)

    def stage(self, start):
        import torch

        self.validate(start)
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

    def __call__(self, query, keys, values, *, page_table_tensor=None, cur_pos_tensor=None, **kwargs):
        self.validate(self.start)
        if tuple(query.shape) != (1, self.rows, 12, 256):
            raise ValueError('Replay query geometry changed')
        owned = []
        protected = {addresses(self.operations, value) for value in (query, keys, values)}
        try:
            for entry, program in zip(self.metadata, self.programs, strict=True):
                refresh_mask(self.positions, entry[2], program)
                self.refresh_calls += 1
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
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.closed = True
