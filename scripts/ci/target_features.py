"""Owned post-layer outputs for an explicit target forward; no drafter or feature-cache publication."""

from contextlib import contextmanager

from model_batch import instance_overrides


class LayerOutputCapture:
    def __init__(self, model, layer_ids, *, snapshot, release, storage_ids):
        self.layer_ids = tuple(layer_ids)
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError('A nonempty, unique ordered tap list is required')
        if any(type(index) is not int or not 0 <= index < len(model.layers) for index in self.layer_ids):
            raise ValueError('Tap IDs must index decoder layers directly')
        if not all(callable(callback) for callback in (snapshot, release, storage_ids)):
            raise ValueError('Explicit snapshot, release and storage identity callbacks required')
        self.model, self.snapshot, self.release, self.storage_ids = model, snapshot, release, storage_ids
        self.records = {}
        self.active = self.complete = self.closed = self.started = False

    def wrap(self, index, forward):
        def captured(*args, **kwargs):
            if not self.active or index in self.records:
                raise RuntimeError('Each selected layer must execute once inside its capture scope')
            output = forward(*args, **kwargs)
            hidden = output[0] if isinstance(output, tuple) else output
            owned = self.snapshot(hidden)
            owned_ids = set(self.storage_ids(owned)) if owned is not None else set()
            borrowed_ids = set(self.storage_ids(hidden))
            if not owned_ids or not borrowed_ids or owned_ids & borrowed_ids or any(
                    owned_ids & set(self.storage_ids(value)) for value in self.records.values()):
                raise ValueError('Feature snapshots must have independent ownership')
            self.records[index] = owned
            return output

        return captured

    @contextmanager
    def capture(self):
        if self.started or self.closed:
            raise RuntimeError('Feature capture owns exactly one forward')
        if hasattr(self.model, '_qwen_target_feature_capture'):
            raise RuntimeError('Target already has an active feature capture')
        self.started = self.active = True
        bindings = [(self.model, '_qwen_target_feature_capture', self)]
        try:
            bindings.extend((self.model.layers[index], 'forward', self.wrap(index, self.model.layers[index].forward))
                            for index in self.layer_ids)
            with instance_overrides(bindings):
                yield self
            if set(self.records) != set(self.layer_ids):
                raise AssertionError('Target forward did not produce every selected layer output')
            self.complete = True
        except BaseException:
            self.active = False
            self.close()
            raise
        finally:
            self.active = False

    def outputs(self):
        if not self.complete or self.closed or self.active:
            raise RuntimeError('Only a complete, open feature capture may be consumed')
        return tuple(self.records[index] for index in self.layer_ids)

    def close(self):
        if self.active:
            raise RuntimeError('Cannot release features during their target forward')
        if self.closed:
            return
        self.closed = True
        records, self.records = self.records, {}
        first_error = None
        for value in records.values():
            try:
                self.release(value)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
