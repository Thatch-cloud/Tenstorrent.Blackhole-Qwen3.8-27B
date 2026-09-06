"""Retain one verified block's GDN histories until its single commit decision."""

from gdn_multitoken_conv import addresses, release_owned, restore_prefix


class RetainedGDNBlock:
    def __init__(self, rows, operations):
        if type(rows) is not int or rows not in (2, 4, 8, 16):
            raise ValueError('Multirow packed-history block required')
        self.rows, self.operations = rows, operations
        self.records = []
        self.selected_prefix = None
        self.closed = False

    def append(self, state, result, checkpoint):
        if self.closed or self.selected_prefix is not None or len(self.records) >= 48:
            raise ValueError('Cannot append to a closed, committed or complete block')
        if not result.get('packed_checkpoints') or result['states'].shape[0] != self.rows:
            raise ValueError('Every layer must retain all packed prefixes')
        if any(previous.gdn is state.gdn for previous, unused, saved in self.records):
            raise ValueError('Duplicate GDN layer record')
        self.records.append((state, result, checkpoint))

    def commit(self, prefix, *, dma=False, publication=None):
        if self.closed or self.selected_prefix is not None or len(self.records) != 48:
            raise ValueError('Exactly one decision on a complete live block required')
        if type(prefix) is not int or not 0 <= prefix <= self.rows:
            raise ValueError('Commit prefix outside verified rows')
        if publication is not None and (not dma or not callable(publication)):
            raise ValueError('A bound publication callback requires the DMA path')
        for state, result, checkpoint in self.records:
            native = [state.gdn.rec_state, *state.gdn.conv_states]
            if state.gdn.B != 8 or not state.gdn._stable_state or [addresses(self.operations, value) for value in native] != state.native_addresses:
                raise ValueError('Native layer binding changed before commit')
        self.selected_prefix = prefix
        if publication is not None:
            publication(prefix)
            return
        if dma:
            from gdn_commit_dma import publish
            mesh = self.records[0][0].gdn.mesh
            if any(state.gdn.mesh is not mesh for state, result, checkpoint in self.records):
                raise ValueError('All retained layers must belong to the same mesh')
            layers = [[*state.entry, result['states'], *result['packed_conv_states'],
                       state.gdn.rec_state, *state.gdn.conv_states, *checkpoint]
                      for state, result, checkpoint in self.records]
            publish(mesh, layers, prefix)
            return
        for state, result, checkpoint in self.records:
            restore_prefix(self.operations, result, state.entry, checkpoint, prefix)
            state.active.restore(checkpoint)

    def close(self):
        if not self.closed:
            release_owned(self.operations, [value for state, result, checkpoint in self.records for value in result['owned']])
            self.records.clear()
            self.closed = True
