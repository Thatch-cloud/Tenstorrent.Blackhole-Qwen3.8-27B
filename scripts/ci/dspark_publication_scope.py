"""Opt-in captured projection adapter; construct before other request traces."""

from contextlib import contextmanager

from dspark_captured_publication import prepare
from dspark_history import TensorScope, leaves
from dspark_projection import require_tensor
from dspark_publication_trace import PreparedHistoryProjection


class CapturedPublicationArm:
    def __init__(self, history, *, audit=False):
        import torch

        if history.closed or history.pending is not None or 'prepare_publication' in vars(history):
            raise ValueError('Idle unmodified fixed history required before publication capture')
        self.history, self.projection = history, None
        operations = history.operations
        scope = TensorScope(operations, leaves(history.layers) + leaves(history.spare_layers))
        try:
            features = tuple(scope.retain(self.upload(torch.zeros(1, 1, 32, 2560, dtype=torch.bfloat16)))
                for tap in range(5))
            tables = tuple(scope.retain(self.upload(value)) for value in history.rotary.tables(history.position, 32))
            self.projection = PreparedHistoryProjection(operations, history.mesh, history.collectives,
                history.parameters, history.layer_weights, features, tables, audit=audit)
        finally:
            scope.release()

    def upload(self, value):
        operations = self.history.operations
        return operations.from_torch(value, device=self.history.mesh, dtype=operations.bfloat16,
            layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ReplicateTensorToMesh(self.history.mesh))

    def publication(self, features, prefix, *, position):
        import torch

        history, operations = self.history, self.history.operations
        history.check_prefix(prefix, position)
        features = tuple(features)
        if len(features) != 5:
            raise ValueError('Five complete verified feature taps required')
        for value in features:
            shape = tuple(value.shape)
            if len(shape) != 4 or shape[:2] != (1, 1) or shape[-1] != 2560 or not prefix <= shape[2] <= 32:
                raise ValueError('Verified feature prefix within fixed32 required')
            require_tensor(operations, value, shape, operations.bfloat16)
        scope = TensorScope(operations, features + leaves(history.layers) + leaves(history.spare_layers)
            + leaves(self.projection.outputs))
        try:
            padded = tuple(scope.retain(operations.pad(value,
                [(0, 0), (0, 0), (0, 32 - value.shape[2]), (0, 0)], 0.0))
                if value.shape[2] != 32 else value for value in features)
            host_tables = (torch.ones(1, 1, 32, 128, dtype=torch.bfloat16),
                torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16))
            for destination, actual in zip(host_tables, history.rotary.tables(position, prefix), strict=True):
                destination[:, :, :prefix] = actual
            tables = tuple(scope.retain(self.upload(value)) for value in host_tables)
            return prepare(history, self.projection, padded, tables, prefix, position=position)
        finally:
            scope.release()

    @contextmanager
    def install(self):
        if 'prepare_publication' in vars(self.history) or self.projection.closed:
            raise ValueError('One live publication adapter required')
        callback = self.publication
        self.history.prepare_publication = callback
        try:
            yield self
        finally:
            unchanged = self.history.prepare_publication is callback
            del self.history.prepare_publication
            self.projection.close()
            if not unchanged:
                raise RuntimeError('Captured publication hook changed during request')
