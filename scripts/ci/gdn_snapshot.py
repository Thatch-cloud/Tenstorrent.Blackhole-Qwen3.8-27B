"""Slot-zero GDN snapshots; persistent native state buffers remain owned by the layer."""


class ActiveSnapshot:
    def __init__(self, gdn, operations):
        self.gdn = gdn
        self.operations = operations
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
        for tensor, dimension, target in zip(self.live, self.dimensions, destination, strict=True):
            sliced = self.gdn._slice_along(tensor, dimension, 0, 1)
            self.operations.copy(sliced, target)
            self.operations.deallocate(sliced)

    def restore(self, source):
        if len(source) != len(self.live):
            raise ValueError("Incomplete active snapshot")
        operations = self.operations
        self.gdn._write_recurrent_state_prefix(operations.clone(source[0]), 1)
        for target, saved in zip(self.gdn.conv_states, source[1:], strict=True):
            self.gdn._write_index(target, operations.clone(saved), 0, 1)
