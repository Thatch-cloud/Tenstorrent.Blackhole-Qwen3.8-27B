"""Caller-owned sample pages with poison-before-replay freshness checks."""

from mlp_clock_projection import validate_buffers
from mlp_clock_samples import decode


class ClockCapture:
    def __init__(self, operations, mesh, owned):
        import torch

        self.operations, self.mesh = operations, mesh
        self.buffers, self.payloads = [], []
        self.pending = False
        self.records = []
        for rows in (2, 1):
            poison = torch.full((rows, 32), 0xffffffff, dtype=torch.int64)
            options = dict(dtype=operations.uint32, layout=operations.ROW_MAJOR_LAYOUT,
                mesh_mapper=operations.ReplicateTensorToMesh(mesh))
            self.payloads.append(operations.from_torch(poison, **options))
            tensor = operations.from_torch(poison, device=mesh,
                memory_config=operations.DRAM_MEMORY_CONFIG, **options)
            owned.append(tensor)
            self.buffers.append(tensor)
        self.buffers = tuple(self.buffers)
        validate_buffers(mesh, self.buffers)
        self.identity = self.addresses()

    def addresses(self):
        return tuple(tuple((shard.device().id(), shard.buffer_address())
            for shard in self.operations.get_device_tensors(tensor)) for tensor in self.buffers)

    def prepare(self):
        if self.pending:
            raise ValueError('Previous sample execution has not been collected')
        validate_buffers(self.mesh, self.buffers)
        if self.addresses() != self.identity:
            raise ValueError('Persistent sample bindings changed')
        self.operations.synchronize_device(self.mesh)
        for payload, tensor in zip(self.payloads, self.buffers, strict=True):
            self.operations.copy_host_to_device_tensor(payload, tensor)
        self.operations.synchronize_device(self.mesh)
        self.pending = True

    def collect(self, label):
        if not self.pending:
            raise ValueError('Poison-before-execution required')
        try:
            self.operations.synchronize_device(self.mesh)
            if self.addresses() != self.identity:
                raise ValueError('Persistent sample bindings changed')
            samples = []
            for tensor, role, workers in zip(self.buffers, ('input', 'weights'), ((0, 1), (0,)), strict=True):
                for chip, shard in enumerate(self.operations.get_device_tensors(tensor)):
                    pages = self.operations.to_torch(shard).tolist()
                    for worker in workers:
                        samples.extend(dict(record, chip=chip, role=role)
                            for record in decode(pages[worker], role, worker))
            if len(samples) != 20:
                raise ValueError('All ten samples on both chips required')
            record = dict(label=label, samples=samples, poisoned_before_execution=True)
            self.records.append(record)
            return record
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
        raise AssertionError('Poisoned samples passed without executing a kernel')
