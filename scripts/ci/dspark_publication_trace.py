"""Experimental fixed32 learned history projection; borrowed outputs expire on replay."""

from attention_batch import capture_operation
from dspark_history import TensorScope, leaves, project_block
from dspark_projection import require_tensor
from gdn_multitoken_conv import addresses


class PreparedHistoryProjection:
    def __init__(self, operations, mesh, collectives, parameters, layer_weights, features, tables, *, audit=False):
        if type(audit) is not bool:
            raise ValueError('Explicit projection audit policy required')
        self.audit, self.checks = audit, []
        self.operations, self.mesh, self.collectives = operations, mesh, collectives
        self.parameters, self.layer_weights = parameters, tuple(layer_weights)
        self.trace, self.closed, self.outputs = None, False, ()
        self.input_scope = self.output_scope = None
        self.inputs = ()
        self.validate(features, tables)
        if len(self.layer_weights) != 5 or list(mesh.shape) != [1, 2]:
            raise ValueError('Five learned layers on the two-card mesh required')
        self.borrowed = [*parameters.values(), *(value for layer in self.layer_weights for value in layer.values())]
        try:
            self.input_scope = TensorScope(operations, [*self.borrowed, *features, *tables])
            self.inputs = tuple(self.input_scope.retain(operations.clone(value,
                memory_config=operations.DRAM_MEMORY_CONFIG)) for value in (*features, *tables))
            warm = self.owner()
            try:
                warm_outputs = self.execute(warm.retain)
                operations.synchronize_device(mesh)
                expected = self.snapshot(warm_outputs) if audit else None
            finally:
                warm.release()
            self.output_scope = self.owner()
            self.trace, self.outputs = capture_operation(operations, mesh,
                lambda: self.execute(self.output_scope.retain))
            self.binding_tensors = [*self.borrowed, *self.inputs, *leaves(self.outputs)]
            self.bindings = [addresses(operations, value) for value in self.binding_tensors]
            operations.execute_trace(mesh, self.trace, cq_id=0, blocking=True)
            if expected is not None:
                self.compare(expected)
        except BaseException:
            self.close()
            raise

    def validate(self, features, tables):
        if len(features) != 5 or len(tables) != 2:
            raise ValueError('Five fixed32 feature taps and two absolute rotary tables required')
        for value, width in zip((*features, *tables), (2560,) * 5 + (128, 128), strict=True):
            require_tensor(self.operations, value, (1, 1, 32, width), self.operations.bfloat16)

    def owner(self):
        return TensorScope(self.operations, [*self.borrowed, *self.inputs])

    def execute(self, retain):
        return project_block(self.operations, self.mesh, self.collectives, self.inputs[:5],
            self.parameters, self.layer_weights, self.inputs[5:], retain)

    def project(self, features, tables):
        if self.closed or self.trace is None:
            raise ValueError('Live captured history projection required')
        self.validate(features, tables)
        if [addresses(self.operations, value) for value in self.binding_tensors] != self.bindings:
            raise AssertionError('Captured history projection bindings moved')
        for source, destination in zip((*features, *tables), self.inputs, strict=True):
            self.operations.copy(source, destination)
        expected = None
        if self.audit:
            scope = self.owner()
            try:
                expected = self.snapshot(self.execute(scope.retain))
            finally:
                scope.release()
        self.operations.execute_trace(self.mesh, self.trace, cq_id=0, blocking=True)
        if expected is not None:
            self.compare(expected)
        return self.outputs

    def snapshot(self, outputs):
        import torch

        if len(outputs) != 5 or any(len(pair) != 2 for pair in outputs):
            raise AssertionError('All five complete projected K/V pairs required')
        self.operations.synchronize_device(self.mesh)
        result = []
        for tensor in leaves(outputs):
            shards = self.operations.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError('Both projection output shards required')
            for shard in shards:
                value = self.operations.to_torch(shard).clone()
                if tuple(value.shape) != (1, 4, 32, 128) or not torch.isfinite(value).all():
                    raise AssertionError('Complete finite fixed32 projection output required')
                result.append(value)
        return tuple(result)

    def compare(self, expected):
        import torch

        actual = self.snapshot(self.outputs)
        if any(not torch.equal(value, reference) for value, reference in zip(actual, expected, strict=True)):
            raise AssertionError('Captured history projection differs from eager execution')
        self.checks.append(dict(tensors=len(actual), exact=True))

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        if self.trace is not None:
            self.operations.release_trace(self.mesh, self.trace)
            self.trace = None
        if self.output_scope is not None:
            self.output_scope.release()
        if self.input_scope is not None:
            self.input_scope.release()
        self.outputs, self.inputs, self.closed = (), (), True
