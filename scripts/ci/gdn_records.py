"""Retain GDN histories for one decision per explicitly synchronized trace epoch."""

import os

from gdn_multitoken_conv import addresses, release_owned, restore_prefix


def block_histories(result):
    """The histories a decision needs: five per sequence, or five per packed user.

    A packed result is the first segment's result plus `segment_results`; only the
    per-segment ones carry every user's own prefixes, so those are what is retained.
    """
    pieces = result.get('segment_results') or (result,)
    histories = [value for piece in pieces for value in [piece['states'], *piece.get('packed_conv_states', [])]]
    if any(not piece.get('packed_checkpoints') for piece in pieces) or len(histories) != 5 * len(pieces):
        raise ValueError('Packed recurrent and four convolution histories required')
    return histories


def retain_checkpoint_histories(operations, result, output):
    histories = block_histories(result)
    bindings = [addresses(operations, value) for value in [*histories, output]]
    if any(len({binding[chip] for binding in bindings}) != len(bindings) for chip in range(2)):
        raise ValueError('Histories and caller-owned output must be independent on both chips')
    protected = set(bindings)
    owned = {addresses(operations, value): value for value in result['owned']}
    history_addresses = {addresses(operations, value) for value in histories}
    if len(history_addresses) != len(histories) or not history_addresses.issubset(owned):
        raise ValueError('Every independent history must have retained ownership')
    for binding in owned:
        if binding not in protected and any(any(current == saved for current, saved in zip(binding, keep, strict=True))
                                            for keep in protected):
            raise ValueError('Scratch partially aliases a protected history or output')
    release_owned(operations, [value for binding, value in owned.items() if binding not in protected])
    result['owned'] = [value for binding, value in owned.items() if binding in history_addresses]


def validate_packed_record(state, result, carries, rows):
    """A packed layer record: one history set and one block-start entry per user, and
    each user's carry, which is where that user's accepted prefix is committed."""
    segments = tuple(tuple(span) for span in result['segments'])
    pieces = result.get('segment_results') or ()
    entries = getattr(state, 'segment_entries', None) or ()
    if (not segments or segments[0][0] != 0 or segments[-1][1] != rows
            or len(pieces) != len(segments) or len(entries) != len(segments) or len(carries or ()) != len(segments)
            or any(not piece.get('packed_checkpoints') or piece['states'].shape[0] != stop - start
                   for piece, (start, stop) in zip(pieces, segments))
            or any(len(entry) != 5 or len(carry) != 5 for entry, carry in zip(entries, carries))):
        raise ValueError("Every layer must retain each packed user's own prefixes, block-start entry and carried state")
    return segments


# 64 is the M3 packed block: four 16-row segments, decided one user at a time through
# commit_user, whose layers are that user's 16-row histories. An UNPACKED 64-row block
# can be retained but not committed: gdn_commit_dma.validate_shapes and
# gdn_multitoken_conv.restore_prefix both stop at 32 rows, so commit() fails closed.
BLOCK_ROWS = (2, 4, 8, 16, 32, 64)


class RetainedGDNBlock:
    def __init__(self, rows, operations):
        if type(rows) is not int or rows not in BLOCK_ROWS:
            raise ValueError('Multirow packed-history block required')
        self.rows, self.operations = rows, operations
        self.records = []
        # Packed: the users' spans, and this epoch's per-user decisions as they arrive.
        self.segments = None
        self.decisions = {}
        # QWEN_FAST_FAST_COMMIT=1: read once at construction, like every other packed
        # engine flag (packed_verifier.PackedVerifierEngine.pipelined_commits). See
        # commit_user for what it skips.
        self.fast_commit = os.environ.get('QWEN_FAST_FAST_COMMIT') == '1'
        self.poisoned = False
        self.selected_prefix = None
        self.replay_ready = False
        self.replay_epoch = 0
        self.closed = False

    def append(self, state, result, checkpoint):
        """Record one layer. Unpacked, `checkpoint` is where the block's one decision is
        committed; packed, it is one carry per user, where that user's decision is."""
        if self.closed or self.selected_prefix is not None or len(self.records) >= 48:
            raise ValueError('Cannot append to a closed, committed or complete block')
        segments = None
        if result.get('segments') is not None:
            segments = validate_packed_record(state, result, checkpoint, self.rows)
        elif not result.get('packed_checkpoints') or result['states'].shape[0] != self.rows:
            raise ValueError('Every layer must retain all packed prefixes')
        if self.records and segments != self.segments:
            raise ValueError('Every layer must pack the same users')
        if any(previous.gdn is state.gdn for previous, unused, saved in self.records):
            raise ValueError('Duplicate GDN layer record')
        self.segments = segments
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
        if self.segments is not None:
            raise ValueError('A packed block is committed one user at a time')
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

    def validate_segment(self, segment):
        if self.segments is None:
            raise ValueError('Per-user commit applies to a packed block')
        if type(segment) is not int or not 0 <= segment < len(self.segments):
            raise ValueError('Packed user index outside the block')
        return self.segments[segment]

    def segment_layers(self, segment):
        """One packed user's commit layers in record order, the twenty tensors per layer
        that `gdn_commit_dma.prepare` takes: `[*entry, states, *conv histories, rec_state,
        *conv_states, *carry]`. The destination is that user's CARRY, so the accepted
        prefix lands where the next block's decode restores this user from; native slot
        zero is written as a side effect and means nothing after a packed step."""
        self.validate_segment(segment)
        return [[*state.segment_entries[segment], piece['states'], *piece['packed_conv_states'],
                 state.gdn.rec_state, *state.gdn.conv_states, *carries[segment]]
                for state, result, carries in self.records for piece in (result['segment_results'][segment],)]

    def commit_user(self, segment, prefix, *, dma=False, publication=None, synchronize=False):
        """Commit ONE packed user's accepted prefix into that user's own carry.

        Every user decides once per block, in any order. The block is ready to replay
        once every user has decided and a `synchronize=True` decision has fenced the
        outstanding publications, so synchronize on the last user's commit.

        Prefix zero is a no-op on the carry: the decode restored this user from it and
        never advanced it, and the entry the DMA would copy back is that same state.
        """
        if self.closed or self.poisoned or len(self.records) != 48:
            raise ValueError('Exactly one decision per user on a complete live block required')
        start, stop = self.validate_segment(segment)
        if type(prefix) is not int or not 0 <= prefix <= stop - start:
            raise ValueError("Commit prefix outside the user's verified rows")
        if segment in self.decisions:
            raise ValueError('Each packed user decides once per block')
        if publication is not None and (not dma or not callable(publication)):
            raise ValueError('A bound publication callback requires the DMA path')
        # validate_bindings checks the SAME 48 x 5 native buffer addresses (state.gdn.
        # rec_state and conv_states) that packed_verifier.PackedVerifierEngine.
        # validate_bindings already checked once, this round, at the top of verify() -
        # nothing between verify() and any of this round's commit_user calls (each one
        # DMAs into a distinct user's CARRY, never a native buffer's identity) can move
        # them. Called on every one of a round's commit_user calls (once per packed
        # user) it is a genuine but redundant re-check: measured at ~8 ms per user (four
        # users x 48 layers x 5 get_device_tensors host round trips each), it dominates
        # the packed_commit phase's host wall time. Under QWEN_FAST_FAST_COMMIT=1, keep
        # the check on this round's FIRST commit_user call only (`self.decisions` is
        # still empty there) - still catches a binding actually broken before any commit
        # of the round - and skip the three repeats that only re-confirm what verify()
        # already established. Default (fast_commit False) behaviour is unchanged: every
        # commit_user call validates, exactly as before.
        if not self.fast_commit or not self.decisions:
            self.validate_bindings()
        mesh = self.bound_mesh() if dma or synchronize else None
        self.decisions[segment] = prefix
        if len(self.decisions) == len(self.segments):
            self.selected_prefix = tuple(self.decisions[index] for index in range(len(self.segments)))
        self.replay_ready = False
        try:
            if prefix == 0:
                pass
            elif publication is not None:
                publication(prefix)
            elif dma:
                from gdn_commit_dma import publish
                publish(mesh, self.segment_layers(segment), prefix)
            else:
                for state, result, carries in self.records:
                    restore_prefix(self.operations, result['segment_results'][segment],
                                   state.segment_entries[segment], carries[segment], prefix)
        except BaseException:
            # A half-written carry cannot be retried as a fresh decision and must not
            # be replayed over by a later user's fence.
            self.poisoned = True
            raise
        if synchronize:
            self.operations.synchronize_device(mesh)
            self.replay_ready = self.selected_prefix is not None

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
        self.decisions = {}
        self.replay_epoch += 1

    def close(self):
        if not self.closed:
            release_owned(self.operations, [value for state, result, checkpoint in self.records for value in result['owned']])
            self.records.clear()
            self.decisions = {}
            self.replay_ready = False
            self.closed = True
