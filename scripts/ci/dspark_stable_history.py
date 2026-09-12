"""Double-buffered learned history allocated before target trace capture."""

from types import SimpleNamespace

from dspark_history import FullHistoryKV, TensorScope, leaves, project_chunks
from dspark_intake import TAPS
from dspark_prefill import FeatureChunk
from dspark_projection import require_tensor


class StableHistoryKV(FullHistoryKV):
    def __init__(self, operations, mesh, collectives, parameters, layer_weights, chunks, rotary, *, position, capacity):
        if type(capacity) is not int or not position <= capacity <= 8192 or capacity % 32:
            raise ValueError('Tile-aligned fixed capacity must contain the complete request history')
        super().__init__(operations, mesh, collectives, parameters, layer_weights, chunks, rotary, position=position)
        self.capacity, self.spare_layers = capacity, ()
        original = self.layers
        scope = TensorScope(operations, leaves(original))
        try:
            active = tuple(tuple(scope.retain(operations.pad(value,
                [(0, 0), (0, 0), (0, capacity - position), (0, 0)], 0.0)) if capacity != position else
                scope.retain(operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG))
                for value in pair) for pair in original)
            spare = tuple(tuple(scope.retain(operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG))
                for value in pair) for pair in active)
            operations.synchronize_device(mesh)
            self.layers, self.spare_layers = active, spare
        except BaseException:
            scope.release()
            super().close()
            raise
        scope.release(keep=leaves(active) + leaves(spare))
        self.release_layers(original)

    def logical_layers(self, retain):
        if self.closed:
            raise ValueError('Live fixed history required')
        return tuple(tuple(retain(self.operations.slice(value, (0, 0, 0, 0), (1, 4, self.position, 128)))
            if self.position < self.capacity else value for value in pair) for pair in self.layers)

    def check_prefix(self, prefix, position):
        if (self.closed or self.pending is not None or type(position) is not int or position != self.position
                or type(prefix) is not int or not 1 <= prefix <= 32 or position + prefix > self.capacity):
            raise ValueError('One verified prefix within the preallocated full-history capacity required')

    def prepare_projected(self, added, prefix, *, position):
        self.check_prefix(prefix, position)
        if len(added) != 5 or any(len(pair) != 2 for pair in added):
            raise ValueError('All five projected K/V pairs required')
        operations = self.operations
        for value in leaves(added):
            require_tensor(operations, value, (1, 4, prefix, 128), operations.bfloat16)
        scope = TensorScope(operations, leaves(self.layers) + leaves(self.spare_layers) + leaves(added))
        try:
            previous = self.logical_layers(scope.retain)
            for history, delta, spare in zip(previous, added, self.spare_layers, strict=True):
                for old, new, destination in zip(history, delta, spare, strict=True):
                    combined = scope.retain(operations.concat([old, new], dim=2,
                        memory_config=operations.DRAM_MEMORY_CONFIG))
                    padded = scope.retain(operations.pad(combined,
                        [(0, 0), (0, 0), (0, self.capacity - position - prefix), (0, 0)], 0.0))
                    operations.copy(padded, destination)
            operations.synchronize_device(self.mesh)
        finally:
            scope.release()
        self.pending = SimpleNamespace(owner=self, position=position, prefix=prefix,
            layers=self.spare_layers, status='prepared')
        return self.pending

    def prepare_publication(self, features, prefix, *, position):
        self.check_prefix(prefix, position)
        features = tuple(features)
        if len(features) != len(TAPS):
            raise ValueError('All five complete verified feature taps required')
        operations = self.operations
        scope = TensorScope(operations, features + leaves(self.layers) + leaves(self.spare_layers))
        try:
            prepared = []
            for value in features:
                shape = tuple(value.shape)
                if len(shape) != 4 or shape[:2] != (1, 1) or shape[-1] != 2560 or not prefix <= shape[2] <= 32:
                    raise ValueError('Complete verified prefix within one padded feature block required')
                require_tensor(operations, value, shape, operations.bfloat16)
                prepared.append(scope.retain(operations.pad(value, [(0, 0), (0, 0), (0, 32 - shape[2]), (0, 0)], 0.0))
                    if shape[2] != 32 else value)
            added = project_chunks(operations, self.mesh, self.collectives, self.parameters, self.layer_weights,
                (FeatureChunk(position, prefix, tuple(prepared)),), self.rotary, start=position, rows=prefix)
            for value in leaves(added):
                scope.retain(value)
            return self.prepare_projected(added, prefix, position=position)
        finally:
            scope.release()

    def commit_publication(self, publication):
        self.validate_publication(publication)
        self.layers, self.spare_layers = self.spare_layers, self.layers
        self.position += publication.prefix
        publication.status, self.pending = 'committed', None

    def discard_publication(self, publication):
        if getattr(publication, 'owner', None) is self and publication.status == 'committed':
            return
        self.validate_publication(publication)
        publication.status, self.pending = 'discarded', None

    def close(self):
        if self.closed:
            return
        spare, self.spare_layers = self.spare_layers, ()
        try:
            super().close()
        finally:
            self.release_layers(spare)
