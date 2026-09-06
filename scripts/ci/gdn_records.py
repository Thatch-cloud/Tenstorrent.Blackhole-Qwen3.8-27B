"""Retain GDN histories for one decision per explicitly synchronized trace epoch."""

from gdn_multitoken_conv import addresses, release_owned, restore_prefix


class RetainedGDNBlock:
    def __init__(self, rows, operations):
        if type(rows) is not int or rows not in (2, 4, 8, 16, 32):
            raise ValueError('Multirow packed-history block required')
        self.rows, self.operations = rows, operations
        self.records = []
        self.selected_prefix = None
        self.replay_ready = False
        self.replay_epoch = 0
        self.closed = False

    def append(self, state, result, checkpoint):
        if self.closed or self.selected_prefix is not None or len(self.records) >= 48:
            raise ValueError('Cannot append to a closed, committed or complete block')
        if not result.get('packed_checkpoints') or result['states'].shape[0] != self.rows:
            raise ValueError('Every layer must retain all packed prefixes')
        if any(previous.gdn is state.gdn for previous, unused, saved in self.records):
            raise ValueError('Duplicate GDN layer record')
        self.records.append((state, result, checkpoint))

    def validate_bindings(self):
        for state, result, checkpoint in self.records:
            native = [state.gdn.rec_state, *state.gdn.conv_states]
            if state.gdn.B != 8 or not state.gdn._stable_state or [addresses(self.operations, value) for value in native] != state.native_addresses:
                raise ValueError('Native layer binding changed')

    def bound_mesh(self):
        mesh = self.records[0][0].gdn.mesh
        if any(state.gdn.mesh is not mesh for state, result, checkpoint in self.records):
            raise ValueError('All retained layers must belong to the same mesh')
        return mesh

    def commit(self, prefix, *, dma=False, publication=None, synchronize=False):
        if self.closed or self.selected_prefix is not None or len(self.records) != 48:
            raise ValueError('Exactly one decision on a complete live block required')
        if type(prefix) is not int or not 0 <= prefix <= self.rows:
            raise ValueError('Commit prefix outside verified rows')
        if publication is not None and (not dma or not callable(publication)):
            raise ValueError('A bound publication callback requires the DMA path')
        self.validate_bindings()
        mesh = self.bound_mesh() if dma or synchronize else None
        self.selected_prefix = prefix
        self.replay_ready = False
        if publication is not None:
            publication(prefix)
        elif dma:
            from gdn_commit_dma import publish
            layers = [[*state.entry, result['states'], *result['packed_conv_states'],
                       state.gdn.rec_state, *state.gdn.conv_states, *checkpoint]
                      for state, result, checkpoint in self.records]
            publish(mesh, layers, prefix)
        else:
            for state, result, checkpoint in self.records:
                restore_prefix(self.operations, result, state.entry, checkpoint, prefix)
                state.active.restore(checkpoint)
        if synchronize:
            self.operations.synchronize_device(mesh)
            self.replay_ready = True

    def replay(self, operation):
        if self.closed or not self.replay_ready or self.selected_prefix is None or len(self.records) != 48:
            raise ValueError('A successfully synchronized commit is required before replay')
        if not callable(operation):
            raise ValueError('Bound trace replay operation required')
        self.replay_ready = False
        self.validate_bindings()
        mesh = self.bound_mesh()
        if operation() is not None:
            raise RuntimeError('Replay operation must return None after enqueueing the bound trace')
        self.operations.synchronize_device(mesh)
        self.validate_bindings()
        self.selected_prefix = None
        self.replay_epoch += 1

    def close(self):
        if not self.closed:
            release_owned(self.operations, [value for state, result, checkpoint in self.records for value in result['owned']])
            self.records.clear()
            self.replay_ready = False
            self.closed = True
