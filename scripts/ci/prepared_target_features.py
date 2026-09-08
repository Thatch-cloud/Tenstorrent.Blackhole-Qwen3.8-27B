"""Reusable post-layer copies into preallocated verifier-owned feature buffers."""

from contextlib import contextmanager

from model_batch import instance_overrides


class PreparedTargetFeatures:
    def __init__(self, model, tap_ids, destinations, *, copy, storage_ids):
        self.tap_ids, self.destinations = tuple(tap_ids), tuple(destinations)
        if (not self.tap_ids or len(set(self.tap_ids)) != len(self.tap_ids)
                or len(self.destinations) != len(self.tap_ids)
                or any(type(index) is not int or not 0 <= index < len(model.layers) for index in self.tap_ids)
                or not callable(copy) or not callable(storage_ids)):
            raise ValueError('Complete ordered target taps, destinations and device callbacks required')
        self.model, self.copy, self.storage_ids = model, copy, storage_ids
        self.identities = tuple(tuple(storage_ids(value)) for value in self.destinations)
        if any(not identity or any(set(identity) & set(other) for other in self.identities[:index])
                for index, identity in enumerate(self.identities)):
            raise ValueError('Independent persistent storage required for every feature tap')
        shape = tuple(self.destinations[0].shape)
        if any(tuple(value.shape) != shape for value in self.destinations):
            raise ValueError('Matching destination feature shapes required')
        self.active = self.complete = self.closed = False

    @contextmanager
    def capture(self):
        if self.closed or self.active or hasattr(self.model, '_qwen_target_feature_capture'):
            raise RuntimeError('Only one target feature capture per forward')
        self.active, self.complete = True, False
        seen = set()

        def wrap(index, original, destination, identity):
            def forward(*args, **kwargs):
                if index in seen:
                    raise RuntimeError('Each target feature tap must execute exactly once')
                seen.add(index)
                output = original(*args, **kwargs)
                hidden = output[0] if isinstance(output, tuple) else output
                source_ids = tuple(self.storage_ids(hidden))
                if (tuple(hidden.shape) != tuple(destination.shape) or hidden.dtype != destination.dtype
                        or hidden.layout != destination.layout or tuple(self.storage_ids(destination)) != identity
                        or len(source_ids) != len(identity)
                        or any(set(source_ids) & set(other) for other in self.identities)):
                    raise ValueError('Feature source must match stable independently owned destinations')
                self.copy(hidden, destination)
                return output
            return forward

        bindings = [(self.model, '_qwen_target_feature_capture', self)]
        bindings.extend((self.model.layers[index], 'forward', wrap(index, self.model.layers[index].forward, destination, identity))
            for index, destination, identity in zip(self.tap_ids, self.destinations, self.identities, strict=True))
        try:
            with instance_overrides(bindings):
                yield self
            if seen != set(self.tap_ids):
                raise RuntimeError('Target forward omitted a requested feature tap')
            self.complete = True
        finally:
            self.active = False

    def outputs(self):
        if self.closed or self.active or not self.complete:
            raise RuntimeError('Complete verifier feature capture required')
        if tuple(tuple(self.storage_ids(value)) for value in self.destinations) != self.identities:
            raise ValueError('Verifier feature destinations moved')
        return self.destinations

    def close(self):
        if self.active:
            raise RuntimeError('Cannot close features during their target forward')
        self.closed = True
