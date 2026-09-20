"""Slot-zero GDN snapshots; persistent native state buffers remain owned by the layer."""


class ActiveSnapshot:
    def __init__(self, gdn, operations, direct=False):
        self.gdn = gdn
        self.operations = operations
        self.direct = direct
        if gdn.B != 8 or not gdn._stable_state:
            raise ValueError("Expected stable eight-slot native GDN state")
        self.live = [gdn.rec_state, *gdn.conv_states]
        self.dimensions = [0] + [1] * len(gdn.conv_states)

    def allocate(self):
        operations = self.operations
        result = []
        for tensor, dimension in zip(self.live, self.dimensions, strict=True):
            sliced = self.gdn._slice_along(tensor, dimension, 0, 1)
            result.append(operations.clone(sliced, memory_config=operations.DRAM_MEMORY_CONFIG))
            operations.deallocate(sliced)
        return result

    def save(self, destination):
        if len(destination) != len(self.live):
            raise ValueError("Incomplete active snapshot")
        if self.direct:
            from gdn_state_copy import copy_active
            copy_active(self.live, destination)
            return
        for tensor, dimension, target in zip(self.live, self.dimensions, destination, strict=True):
            sliced = self.gdn._slice_along(tensor, dimension, 0, 1)
            self.operations.copy(sliced, target)
            self.operations.deallocate(sliced)

    def restore(self, source):
        if len(source) != len(self.live):
            raise ValueError("Incomplete active snapshot")
        if self.direct:
            from gdn_state_copy import copy_active
            copy_active(source, self.live)
            return
        operations = self.operations
        self.gdn._write_recurrent_state_prefix(operations.clone(source[0]), 1)
        for target, saved in zip(self.gdn.conv_states, source[1:], strict=True):
            self.gdn._write_index(target, operations.clone(saved), 0, 1)

    def adopt_slot(self, index, *, layer=None):
        """Copy native row `index` of every live tensor into row 0.

        The plugin's batched prefill writes the admitted user's recurrent and conv
        state into its decode slot and leaves the other rows alone, while allocate,
        save and restore here read row 0 unconditionally. Called once at admission,
        before the engine's initial save, so that save reads the user's own state
        (runs 35492676194, 35493208438: the second user decoded from the first's).
        Always the ttnn slice path, whatever `direct` is: the DMA kernel copies
        slot 0 only. Row 0 has nothing to adopt.

        Every slice is verified against a host readback of its live tensor before
        anything is written. The conv states keep all eight slots inside one 32-row
        tile, so a row-k slice with k != 0 is a tile-internal unaligned slice that no
        other path takes (every existing slice starts at row 0); rec_state's dim-0
        slice is page-aligned and checked the same way as insurance. Unconditional:
        once per admission, and it is the proof the token-exact gate run needs.
        Returns the number of chips every tensor was verified on.
        """
        if type(index) is not int or not 0 <= index < self.gdn.B:
            raise ValueError("Native GDN slot index within the eight-slot batch required")
        if index == 0:
            return 0
        operations = self.operations
        names = ["rec_state"] + ["conv_states[%d]" % position for position in range(len(self.gdn.conv_states))]
        sources, chips = [], None
        try:
            for tensor, dimension in zip(self.live, self.dimensions, strict=True):
                sources.append(self.gdn._slice_along(tensor, dimension, index, 1))
            for name, tensor, dimension, source in zip(names, self.live, self.dimensions, sources, strict=True):
                verified = self._verify_row(name, tensor, dimension, source, index, layer)
                if chips is not None and verified != chips:
                    raise ValueError("Live GDN tensors of layer %s span %d and %d chips" % (layer, chips, verified))
                chips = verified
            self.gdn._write_recurrent_state_prefix(operations.clone(sources[0]), 1)
            for target, source in zip(self.gdn.conv_states, sources[1:], strict=True):
                self.gdn._write_index(target, operations.clone(source), 0, 1)
        finally:
            for source in sources:
                operations.deallocate(source)
        return chips

    def _verify_row(self, name, tensor, dimension, source, index, layer):
        """Per chip, the device slice must equal row `index` of the live tensor read
        back whole (the check_shards readback in serving_sequential_step)."""
        import torch

        operations = self.operations
        kind = "Recurrent-state" if dimension == 0 else "Unaligned conv-state"
        where = "layer %s %s" % ("?" if layer is None else layer, name)
        fulls = list(operations.get_device_tensors(tensor))
        parts = list(operations.get_device_tensors(source))
        if not fulls or len(fulls) != len(parts):
            raise ValueError("%s slice at row %d has %d shards against %d live shards: %s"
                             % (kind, index, len(parts), len(fulls), where))
        for chip, (full, part) in enumerate(zip(fulls, parts, strict=True)):
            expected = operations.to_torch(full).narrow(dimension, index, 1).contiguous()
            actual = operations.to_torch(part).contiguous()
            if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
                raise ValueError("%s slice at row %d differs from the row: %s chip %d shape %r %s, row %r %s"
                                 % (kind, index, where, chip, tuple(actual.shape), actual.dtype,
                                    tuple(expected.shape), expected.dtype))
            if not _same_bits(torch, actual, expected):
                difference = (actual.float() - expected.float()).abs()
                raise ValueError("%s slice at row %d differs from the row: %s chip %d differing=%d of %d max_abs=%g"
                                 % (kind, index, where, chip, int((difference > 0).sum()), difference.numel(),
                                    float(difference.max())))
        return len(fulls)


def _same_bits(torch, left, right):
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(left.element_size())
    if width is None:
        return torch.equal(left, right)
    return torch.equal(left.view(width), right.view(width))
