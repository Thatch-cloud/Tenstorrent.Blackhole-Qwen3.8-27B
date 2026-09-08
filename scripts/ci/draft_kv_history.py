"""Double-buffered projected draft history with explicit accepted-prefix publication."""

from contextlib import contextmanager
from types import SimpleNamespace

from draft_head_preparation import rope_tables
from draft_kv_projection import project_key_value
from gdn_multitoken_conv import addresses, release_owned


class DraftKVHistory:
    def __init__(self, operations, mesh, parameters, features, *, position, history_rows):
        import torch

        parameters = tuple(parameters)
        if (type(position) is not int or not 1 <= position <= 262112
                or type(history_rows) is not int or history_rows != min(position, 2048)
                or not 1 <= len(parameters) <= 5):
            raise ValueError('Bounded absolute draft frontier and explicit learned layers required')
        self.operations, self.mesh, self.parameters = operations, mesh, parameters
        self.position, self.history_rows = position, history_rows
        self.owned, self.active, self.spare = [], [], []
        self.checks = []
        self.pending, self.closed = None, False
        try:
            self.query = self.upload(torch.zeros((1, 1, 32, 2048), dtype=torch.bfloat16))
            self.owned.append(self.query)
            with self.temporaries([features]) as retain:
                inputs, tables = self.project_inputs(features, history_rows, position - history_rows, retain)
                for parameter in parameters:
                    result = project_key_value(operations, inputs, self.query, tables, retain, parameters=parameter)
                    active, spare = {}, {}
                    for name in ('k', 'v'):
                        valid = retain(operations.slice(result[name], (0, 0, 0, 0), (1, 4, history_rows, 128)))
                        active[name] = retain(operations.pad(valid, [(0, 0), (0, 0), (0, 2048 - history_rows), (0, 0)], 0.0))
                        self.owned.append(active[name])
                        spare[name] = operations.zeros_like(active[name])
                        self.owned.append(spare[name])
                    self.active.append(active)
                    self.spare.append(spare)
            operations.synchronize_device(mesh)
        except BaseException:
            self.close()
            raise

    def upload(self, value):
        operations = self.operations
        return operations.from_torch(value, device=self.mesh, dtype=operations.bfloat16,
            layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))

    @contextmanager
    def temporaries(self, protected):
        owned = []
        identities = [addresses(self.operations, value) for value in [*self.owned, *protected]]
        def retain(value):
            identity = addresses(self.operations, value)
            if identity not in identities:
                if any(any(left == right for left, right in zip(identity, other, strict=True)) for other in identities):
                    raise ValueError('Draft cache temporary partially aliases borrowed storage')
                owned.append(value)
            return value
        try:
            yield retain
        finally:
            persistent = [*identities, *(addresses(self.operations, value) for value in self.owned)]
            release_owned(self.operations, [value for value in owned if addresses(self.operations, value) not in persistent])

    def project_inputs(self, features, count, start, retain):
        operations = self.operations
        if (type(count) is not int or not 1 <= count <= 2048 or len(features.shape) != 4
                or tuple(features.shape)[:2] != (1, 1) or features.shape[2] < count
                or features.shape[3] != 5120 or features.dtype != operations.bfloat16):
            raise ValueError('Complete replicated projected BF16 feature rows required')
        padded_rows = ((count + 31) // 32) * 32
        valid = retain(operations.slice(features, (0, 0, 0, 0), (1, 1, count, 5120)))
        inputs = retain(operations.pad(valid, [(0, 0), (0, 0), (0, padded_rows - count), (0, 0)], 0.0))
        host = rope_tables(start, padded_rows)
        for table in host:
            table[..., count:, :] = 0
        tables = tuple(retain(self.upload(table)) for table in host)
        return inputs, tables

    def prepare(self, features, prefix, *, position):
        if (self.closed or self.pending is not None or type(position) is not int or position != self.position
                or type(prefix) is not int or not 1 <= prefix <= 32 or position + prefix > 262144):
            raise ValueError('One accepted-prefix cache update at the current committed frontier required')
        operations = self.operations
        rows = min(2048, self.history_rows + prefix)
        with self.temporaries([features]) as retain:
            inputs, tables = self.project_inputs(features, prefix, position, retain)
            for parameter, active, spare in zip(self.parameters, self.active, self.spare, strict=True):
                result = project_key_value(operations, inputs, self.query, tables, retain, parameters=parameter)
                for name in ('k', 'v'):
                    historical = retain(operations.slice(active[name], (0, 0, 0, 0), (1, 4, self.history_rows, 128)))
                    accepted = retain(operations.slice(result[name], (0, 0, 0, 0), (1, 4, prefix, 128)))
                    combined = retain(operations.concat([historical, accepted], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
                    tail = retain(operations.slice(combined, (0, 0, self.history_rows + prefix - rows, 0),
                        (1, 4, self.history_rows + prefix, 128)))
                    padded = retain(operations.pad(tail, [(0, 0), (0, 0), (0, 2048 - rows), (0, 0)], 0.0))
                    operations.copy(padded, spare[name])
            operations.synchronize_device(self.mesh)
        self.pending = SimpleNamespace(position=position, prefix=prefix, rows=rows, status='prepared')
        return self.pending

    def commit(self, publication):
        if (self.closed or publication is not self.pending or publication.status != 'prepared'
                or publication.position != self.position):
            raise ValueError('Only the current prepared draft cache may commit')
        self.active, self.spare = self.spare, self.active
        self.position += publication.prefix
        self.history_rows = publication.rows
        publication.status = 'committed'
        self.pending = None

    def discard(self, publication):
        if publication.status == 'committed':
            return
        if self.closed or publication is not self.pending or publication.status != 'prepared':
            raise ValueError('Only the current prepared draft cache may be discarded')
        publication.status = 'discarded'
        self.pending = None

    def audit(self, features):
        import torch

        if self.closed or self.pending is not None:
            raise ValueError('Only a committed open draft cache may be audited')
        operations = self.operations
        with self.temporaries([features]) as retain:
            inputs, tables = self.project_inputs(features, self.history_rows, self.position - self.history_rows, retain)
            for layer, (parameter, active) in enumerate(zip(self.parameters, self.active, strict=True)):
                expected = project_key_value(operations, inputs, self.query, tables, retain, parameters=parameter)
                for name in ('k', 'v'):
                    actual_shards = operations.get_device_tensors(active[name])
                    expected_shards = operations.get_device_tensors(expected[name])
                    if len(actual_shards) != 2 or len(expected_shards) != 2:
                        raise AssertionError('Both committed draft-cache shards required')
                    for chip, (actual, reference) in enumerate(zip(actual_shards, expected_shards, strict=True)):
                        left = operations.to_torch(actual)[..., :self.history_rows, :].contiguous()
                        right = operations.to_torch(reference)[..., :self.history_rows, :].contiguous()
                        if not torch.equal(left.view(torch.int16), right.view(torch.int16)):
                            raise AssertionError(f'Committed historical K/V differs: layer={layer}, head={name}, chip={chip}')
                        self.checks.append(dict(position=self.position, rows=self.history_rows, layer=layer, head=name, chip=chip, exact=True))

    def close(self):
        if self.closed:
            return
        if self.pending is not None:
            self.discard(self.pending)
        self.operations.synchronize_device(self.mesh)
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.active.clear()
        self.spare.clear()
        self.closed = True
