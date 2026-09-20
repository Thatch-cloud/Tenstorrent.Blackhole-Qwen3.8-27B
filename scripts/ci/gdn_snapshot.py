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

    def adopt_slot(self, index):
        """Copy native row `index` of every live tensor into row 0.

        The plugin's batched prefill writes the admitted user's recurrent and conv
        state into its decode slot and leaves the other rows alone, while allocate,
        save and restore here read row 0 unconditionally. Called once at admission,
        before the engine's initial save, so that save reads the user's own state
        (runs 35492676194, 35493208438: the second user decoded from the first's).
        Always the ttnn slice path, whatever `direct` is: the DMA kernel copies
        slot 0 only. Row 0 has nothing to adopt.
        """
        if type(index) is not int or not 0 <= index < self.gdn.B:
            raise ValueError("Native GDN slot index within the eight-slot batch required")
        if index == 0:
            return
        operations = self.operations
        sources = [self.gdn._slice_along(tensor, dimension, index, 1)
                   for tensor, dimension in zip(self.live, self.dimensions, strict=True)]
        try:
            self.gdn._write_recurrent_state_prefix(operations.clone(sources[0]), 1)
            for target, source in zip(self.gdn.conv_states, sources[1:], strict=True):
                self.gdn._write_index(target, operations.clone(source), 0, 1)
        finally:
            for source in sources:
                operations.deallocate(source)
