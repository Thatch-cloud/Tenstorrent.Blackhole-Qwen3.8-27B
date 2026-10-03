"""Isolated B1 working state; native B8 buffers remain stable and owned by the layer."""

from types import FunctionType, MethodType

from gdn_prefix import decode_projected, gated_decode


class WorkingState:
    def __init__(self, active, operations, compact_dma=False, skip_row_clones=False, hoist_row_layout=False):
        if not active.direct or active.gdn.B != 8 or not active.gdn._stable_state:
            raise ValueError("Expected audited direct snapshots and stable B8 state")
        self.active = active
        self.gdn = active.gdn
        self.operations = operations
        self.compact_dma = compact_dma
        self.skip_row_clones = skip_row_clones
        self.hoist_row_layout = hoist_row_layout
        self.skipped_clones = 0
        self.checkpoint_calls = 0
        self.state = active.allocate()
        self.calls = 0
        self.addresses = self.pointers(self.state)
        native = gated_decode(self.gdn).__func__
        namespace = dict(native.__globals__)
        recurrence = namespace["recurrent_gated_delta_rule_decode_packed_ttnn"]

        def checked(*args, **kwargs):
            if kwargs.get("inplace_state") is not True or kwargs.get("initial_state") is not self.state[0]:
                raise AssertionError("Native recurrence did not request the compact in-place state")
            result = recurrence(*args, **kwargs)
            if self.pointers([result[1]]) != self.pointers([self.state[0]]):
                raise AssertionError("Native recurrence did not return the in-place buffer on both chips")
            self.calls += 1
            return result

        namespace["recurrent_gated_delta_rule_decode_packed_ttnn"] = checked
        self.forward = MethodType(FunctionType(native.__code__, namespace, native.__name__, native.__defaults__,
                                               native.__closure__), self.gdn)

    def pointers(self, tensors):
        result = []
        for tensor in tensors:
            shards = self.operations.get_device_tensors(tensor)
            if len(shards) != 2:
                raise ValueError("Both chips required")
            result.append(tuple(shard.buffer_address() for shard in shards))
        return result

    def save(self, destination):
        if len(destination) != len(self.state):
            raise ValueError("Complete compact checkpoint required")
        if self.compact_dma:
            from gdn_state_copy import copy_compact
            copy_compact(self.state, destination)
            self.checkpoint_calls += 1
            return
        for source, target in zip(self.state, destination, strict=True):
            self.operations.copy(source, target)

    def decode(self, packed, tokens, checkpoint):
        layer = self.gdn
        original = layer.B, layer.rec_state, layer.conv_states
        if original[0] != 8 or original[1] is not self.active.live[0]:
            raise ValueError("Native state binding changed")
        self.active.save(self.state)
        before = self.calls
        try:
            layer.B, layer.rec_state, layer.conv_states = 1, self.state[0], self.state[1:]
            checkpoint(0)
            def skipped():
                self.skipped_clones += 1
            outputs = decode_projected(layer, packed, tokens, checkpoint, self.operations, forward=self.forward,
                                       **({"clone_skipped": skipped} if self.skip_row_clones else {}),
                                       **({"hoist_row_layout": True} if self.hoist_row_layout else {}))
            if self.calls - before != len(tokens) or self.pointers(self.state) != self.addresses:
                raise AssertionError("Every row must update the same compact state in place")
        finally:
            layer.B, layer.rec_state, layer.conv_states = original
        self.active.restore(self.state)
        return outputs

    def close(self):
        for tensor in self.state:
            self.operations.deallocate(tensor)
        self.state.clear()
