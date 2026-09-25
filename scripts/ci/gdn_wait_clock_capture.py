"""Caller-owned, poisoned recurrence clock pages retained through trace release."""

from gdn_wait_clock import decode
from mlp_compute_clock_projection import sample_memory, validate_buffer


class WaitClockCapture:
    def __init__(self, operations, mesh, owned, *, token=8):
        import torch

        if type(token) is not int or not 0 <= token < 16:
            raise ValueError('Explicit T16 sample token required')
        self.operations, self.mesh, self.token = operations, mesh, token
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
            raise ValueError('Previous recurrence sample must be collected')
        validate_buffer(self.operations, self.mesh, self.buffer)
        if self.addresses() != self.identity:
            raise ValueError('Persistent recurrence sample bindings changed')
        self.operations.synchronize_device(self.mesh)
        self.operations.copy_host_to_device_tensor(self.payload, self.buffer)
        self.operations.synchronize_device(self.mesh)
        self.pending = True

    def collect(self, label):
        if not self.pending:
            raise ValueError('Poison-before-execution required')
        try:
            self.operations.synchronize_device(self.mesh)
            validate_buffer(self.operations, self.mesh, self.buffer)
            if self.addresses() != self.identity:
                raise ValueError('Persistent recurrence sample bindings changed')
            samples = []
            for chip, shard in enumerate(self.operations.get_device_tensors(self.buffer)):
                pages = self.operations.to_torch(shard).tolist()
                if len(pages) != 3:
                    raise ValueError('Three separate recurrence processor pages required')
                for processor, words in enumerate(pages):
                    samples.extend(dict(record, chip=chip, worker=0)
                        for record in decode(words, processor, self.token))
            if len(samples) != 42:
                raise ValueError('Seven intervals on three processors on both chips required')
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
            if 'Missing, out-of-order' not in str(error):
                raise
            return True
        raise AssertionError('Poisoned recurrence pages passed without execution')
