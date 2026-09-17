"""Captured learned K/V projection; outputs are borrowed until the next replay."""

from attention_batch import capture_operation
from draft_kv_projection import project_key_value
from gdn_multitoken_conv import addresses, release_owned


class PreparedDraftKVProjection:
    def __init__(self, operations, mesh, parameters, query):
        import torch

        self.operations, self.mesh, self.parameters, self.query = operations, mesh, tuple(parameters), query
        if not 1 <= len(self.parameters) <= 5:
            raise ValueError('Explicit learned draft layers required')
        self.owned, self.inputs, self.outputs = [], [], []
        self.trace, self.closed, self.calls = None, False, 0
        try:
            for width in (5120, 128, 128):
                value = operations.from_torch(torch.zeros((1, 1, 32, width), dtype=torch.bfloat16),
                    device=mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                    memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(mesh))
                self.owned.append(value)
                self.inputs.append(value)
            transient_start = len(self.owned)
            try:
                self.execute()
                operations.synchronize_device(mesh)
            finally:
                release_owned(operations, self.owned[transient_start:])
                del self.owned[transient_start:]
            self.trace, self.outputs = capture_operation(operations, mesh, self.execute)
            self.binding_tensors = [query, *self.inputs, *(layer[name] for layer in self.outputs for name in ('k', 'v'))]
            self.bindings = [addresses(operations, value) for value in self.binding_tensors]
            operations.execute_trace(mesh, self.trace, cq_id=0, blocking=True)
        except BaseException:
            self.close()
            raise

    def retain(self, value):
        identity = addresses(self.operations, value)
        known = [addresses(self.operations, tensor) for tensor in [self.query, *self.owned]]
        if identity not in known:
            if any(any(left == right for left, right in zip(identity, other, strict=True)) for other in known):
                raise ValueError('Captured K/V projection partially aliases borrowed storage')
            self.owned.append(value)
        return value

    def execute(self):
        return [project_key_value(self.operations, self.inputs[0], self.query, self.inputs[1:],
            self.retain, parameters=parameter) for parameter in self.parameters]

    def project(self, features, tables):
        operations = self.operations
        if self.closed or len(tables) != 2:
            raise ValueError('Open captured K/V projection and two absolute rotary tables required')
        sources = [features, *tables]
        if any(tuple(value.shape) != (1, 1, 32, width) or value.dtype != operations.bfloat16
                or value.layout != operations.TILE_LAYOUT or value.memory_config() != operations.DRAM_MEMORY_CONFIG
                for value, width in zip(sources, (5120, 128, 128), strict=True)):
            raise ValueError('Fixed32 BF16 tiled projection inputs required')
        if [addresses(operations, value) for value in self.binding_tensors] != self.bindings:
            raise AssertionError('Captured K/V projection bindings moved')
        for source, destination in zip(sources, self.inputs, strict=True):
            operations.copy(source, destination)
        operations.execute_trace(self.mesh, self.trace, cq_id=0, blocking=True)
        self.calls += 1
        return self.outputs

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        if self.trace is not None:
            self.operations.release_trace(self.mesh, self.trace)
            self.trace = None
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.inputs.clear()
        self.outputs.clear()
        self.closed = True
