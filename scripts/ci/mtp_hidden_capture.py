"""Opt-in fixed-buffer post-final-norm capture, including sharded target heads."""

from contextlib import contextmanager

from model_batch import instance_overrides


class MTPHiddenCapture:
    def __init__(self, model, destination, *, copy, storage_ids):
        if (not callable(copy) or not callable(storage_ids)
                or not callable(getattr(model, '_final_norm_decode', None))):
            raise ValueError('Explicit device copy and storage identity functions required')
        self.model, self.destination = model, destination
        self.copy, self.storage_ids = copy, storage_ids
        self.identities = tuple(storage_ids(destination))
        if not self.identities:
            raise ValueError('Persistent destination storage required')
        self.active = False
        self.complete = False

    @contextmanager
    def capture(self):
        if self.active or hasattr(self.model, '_qwen_mtp_hidden_capture'):
            raise RuntimeError('Only one MTP hidden capture per target forward')
        self.active, self.complete = True, False
        calls = 0
        forward = self.model._final_norm_decode

        def normalized(*args, **kwargs):
            nonlocal calls
            if calls:
                raise RuntimeError('Expected exactly one final normalization')
            calls += 1
            output = forward(*args, **kwargs)
            source_ids = tuple(self.storage_ids(output))
            if (tuple(output.shape) != tuple(self.destination.shape)
                    or output.dtype != self.destination.dtype or output.layout != self.destination.layout
                    or tuple(self.storage_ids(self.destination)) != self.identities
                    or len(source_ids) != len(self.identities)
                    or any(source == target for source, target in zip(source_ids, self.identities))):
                raise ValueError('Stable independent destination matching normalized hidden required')
            self.copy(output, self.destination)
            return output

        try:
            with instance_overrides([(self.model, '_qwen_mtp_hidden_capture', self),
                                     (self.model, '_final_norm_decode', normalized)]):
                yield self
            if calls != 1:
                raise RuntimeError('Target did not execute final normalization')
            self.complete = True
        finally:
            self.active = False

    def output(self):
        if self.active or not self.complete:
            raise RuntimeError('Complete target capture required')
        return self.destination
