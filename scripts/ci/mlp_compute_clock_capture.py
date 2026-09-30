"""Persistent single-core L1 sample pages; allocate before any trace capture."""

from mlp_compute_clock import decode
from mlp_compute_clock_projection import sample_memory, validate_buffer


class ComputeClockCapture:
    def __init__(self, operations, mesh, owned):
        import torch

        self.operations, self.mesh = operations, mesh
        self.pending, self.records = False, []
        poison = torch.full((3, 64), 0xffffffff, dtype=torch.int64)
        options = dict(dtype=operations.uint32, layout=operations.ROW_MAJOR_LAYOUT,
            mesh_mapper=operations.ReplicateTensorToMesh(mesh))
        self.payload = operations.from_torch(poison, **options)
        self.buffer = operations.from_torch(poison, device=mesh,
            memory_config=sample_memory(operations), **options)
        owned.append(self.buffer)
        validate_buffer(operations, mesh, self.buffer)
        self.identity = self.addresses()

    def addresses(self):
        return tuple(shard.buffer_address() for shard in self.operations.get_device_tensors(self.buffer))

    def prepare(self):
        if self.pending:
            raise ValueError('Previous compute sample must be collected')
        validate_buffer(self.operations, self.mesh, self.buffer)
        if self.addresses() != self.identity:
            raise ValueError('Persistent compute sample bindings changed')
        self.operations.synchronize_device(self.mesh)
        self.operations.copy_host_to_device_tensor(self.payload, self.buffer)
        self.operations.synchronize_device(self.mesh)
        self.pending = True

    def collect(self, label):
        if not self.pending:
            raise ValueError('Poison-before-execution required')
        try:
            self.operations.synchronize_device(self.mesh)
            if self.addresses() != self.identity:
                raise ValueError('Persistent compute sample bindings changed')
            samples = []
            for chip, shard in enumerate(self.operations.get_device_tensors(self.buffer)):
                pages = self.operations.to_torch(shard).tolist()
                if len(pages) != 3:
                    raise ValueError('Three separate compute processor pages required')
                for processor, words in enumerate(pages):
                    samples.extend(dict(record, chip=chip, worker=0) for record in decode(words, processor))
            if len(samples) != 36:
                raise ValueError('Six intervals on three processors on both chips required')
            result = dict(label=label, poisoned_before_execution=True, samples=samples)
            self.records.append(result)
            return result
        finally:
            self.pending = False

    def reject_missing_execution(self):
        self.prepare()
        try:
            self.collect('missing-execution')
        except ValueError as error:
            if 'Missing, malformed' not in str(error):
                raise
            return True
        raise AssertionError('Poisoned compute pages passed without execution')
