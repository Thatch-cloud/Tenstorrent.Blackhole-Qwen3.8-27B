"""Prepared native slice traces into one fixed MTP hidden-row buffer."""

from attention_batch import capture_operation
from gdn_multitoken_conv import addresses


class MTPHiddenRows:
    def __init__(self, operations, mesh, sources):
        import torch

        self.operations, self.mesh = operations, mesh
        self.sources, self.traces = {}, {}
        self.ready = self.closed = False
        for source in sources:
            shape = tuple(source.shape)
            if (shape[:2] != (1, 1) or len(shape) != 4 or shape[2] not in (1, 2, 4, 8, 16, 32)
                    or shape[3] != 5120 or source.dtype != operations.bfloat16 or source.layout != operations.TILE_LAYOUT
                    or source.memory_config() != operations.DRAM_MEMORY_CONFIG):
                raise ValueError('Pinned replicated verifier hidden rows required')
            identity = tuple(addresses(operations, source))
            if len(identity) != 2 or any(any(left == right for left, right in zip(identity, existing))
                                         for existing in self.sources):
                raise ValueError('Unique two-chip source allocations required')
            self.sources[identity] = source
        if not self.sources:
            raise ValueError('At least one verifier bucket required')
        self.destination = operations.from_torch(torch.zeros((1, 1, 1, 5120), dtype=torch.bfloat16),
            device=mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
            memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(mesh))
        self.destination_ids = tuple(addresses(operations, self.destination))
        if len(self.destination_ids) != 2 or any(any(left == right for left, right in zip(identity, self.destination_ids))
                                               for identity in self.sources):
            operations.deallocate(self.destination)
            raise ValueError('Row destination must not alias source on either chip')

    def execute(self, source, row):
        if source.shape[2] == 1:
            self.operations.copy(source, self.destination)
            return self.destination
        result = self.operations.slice(source, (0, 0, row, 0), (1, 1, row + 1, 5120),
            memory_config=self.operations.DRAM_MEMORY_CONFIG)
        source_ids, result_ids = addresses(self.operations, source), addresses(self.operations, result)
        if source_ids != result_ids and any(left == right for left, right in zip(source_ids, result_ids, strict=True)):
            raise ValueError('Slice must not partially alias its source across chips')
        try:
            self.operations.copy(result, self.destination)
        finally:
            if source_ids != result_ids:
                self.operations.deallocate(result)
        return self.destination

    def prepare(self):
        if self.ready or self.closed:
            raise ValueError('Row reader must be prepared exactly once')
        try:
            for source in self.sources.values():
                for row in range(source.shape[2]):
                    self.execute(source, row)
            self.operations.synchronize_device(self.mesh)
            for identity, source in self.sources.items():
                for row in range(source.shape[2]):
                    self.traces[identity, row], unused = capture_operation(self.operations, self.mesh,
                        lambda source=source, row=row: self.execute(source, row))
            self.ready = True
        except BaseException:
            self.close()
            raise

    def __call__(self, source, row):
        identity = tuple(addresses(self.operations, source))
        if (not self.ready or self.closed or type(row) is not int or (identity, row) not in self.traces
                or self.sources[identity] is not source
                or tuple(addresses(self.operations, self.destination)) != self.destination_ids):
            raise ValueError('Live prepared source, row and fixed destination required')
        self.operations.execute_trace(self.mesh, self.traces[identity, row], cq_id=0, blocking=True)
        return self.destination

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        for trace in self.traces.values():
            self.operations.release_trace(self.mesh, trace)
        self.operations.deallocate(self.destination)
        self.closed = True
