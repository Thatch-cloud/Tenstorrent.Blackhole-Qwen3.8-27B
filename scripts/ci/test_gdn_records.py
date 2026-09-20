from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_records import RetainedGDNBlock


def operations():
    return SimpleNamespace(get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2,
                           synchronize_device=Mock())


def unpacked_record(mesh, rows):
    live = [object() for slot in range(5)]
    state = SimpleNamespace(gdn=SimpleNamespace(B=8, _stable_state=True, rec_state=live[0], conv_states=live[1:], mesh=mesh),
        native_addresses=[(id(value), id(value)) for value in live], entry=[object()] * 5,
        active=SimpleNamespace(restore=Mock()))
    result = dict(packed_checkpoints=True, states=SimpleNamespace(shape=(rows, 24, 128, 128)), owned=[object()])
    return state, result, [object() for slot in range(5)]


def unpacked_block(count=48, rows=4):
    block = RetainedGDNBlock(rows, operations())
    mesh = object()
    for index in range(count):
        block.append(*unpacked_record(mesh, rows))
    return block


def packed_record(mesh, segments=((0, 16), (16, 32))):
    """One layer of a deferred packed decode as ModelBatch records it: per user, a
    history set, a block-start entry and the carry that user's decision is committed to."""
    live = [object() for slot in range(5)]
    pieces = tuple(dict(packed_checkpoints=True, states=SimpleNamespace(shape=(stop - start, 24, 128, 128)),
                        packed_conv_states=[object() for slot in range(4)], owned=[]) for start, stop in segments)
    state = SimpleNamespace(gdn=SimpleNamespace(B=8, _stable_state=True, rec_state=live[0], conv_states=live[1:], mesh=mesh),
        native_addresses=[(id(value), id(value)) for value in live], entry=[],
        segment_entries=tuple([object() for slot in range(5)] for span in segments), segment_results=pieces,
        active=SimpleNamespace(restore=Mock()))
    result = dict(pieces[0], segments=segments, segment_results=pieces, owned=[object()])
    carries = tuple([object() for slot in range(5)] for span in segments)
    return state, result, carries


def packed_block(count=48, segments=((0, 16), (16, 32))):
    block = RetainedGDNBlock(32, operations())
    mesh = object()
    for index in range(count):
        block.append(*packed_record(mesh, segments))
    return block


class RecordTests(unittest.TestCase):
    def fixture(self, count=48, rows=4):
        return unpacked_block(count, rows)

    def test_t32_retains_every_prefix_and_replays_after_commit(self):
        block = self.fixture(rows=32)
        with patch('gdn_records.restore_prefix'):
            for prefix in range(33):
                block.commit(prefix, synchronize=True)
                block.replay(lambda: None)
        self.assertEqual(block.replay_epoch, 33)

    def test_replay_rearms_only_after_synchronized_commit_and_execution(self):
        block = self.fixture()
        with patch('gdn_records.restore_prefix'):
            for epoch in range(3):
                block.commit(epoch, synchronize=True)
                self.assertTrue(block.replay_ready)
                operation = Mock(return_value=None)
                block.replay(operation)
                operation.assert_called_once_with()
                self.assertIsNone(block.selected_prefix)
                self.assertEqual(block.replay_epoch, epoch + 1)
                with self.assertRaises(ValueError):
                    block.replay(operation)
        self.assertEqual(block.operations.synchronize_device.call_count, 6)

    def test_unsynchronized_or_failed_commit_cannot_replay(self):
        for failure in ('unsynchronized', 'publication', 'synchronization'):
            block = self.fixture()
            if failure == 'synchronization':
                block.operations.synchronize_device.side_effect = RuntimeError('device')
            with patch('gdn_records.restore_prefix', side_effect=RuntimeError('device') if failure == 'publication' else None):
                if failure == 'unsynchronized':
                    block.commit(2)
                else:
                    with self.assertRaises(RuntimeError):
                        block.commit(2, synchronize=True)
            operation = Mock()
            with self.assertRaises(ValueError):
                block.replay(operation)
            operation.assert_not_called()

    def test_failed_or_rebound_replay_is_poisoned(self):
        for failure in ('binding', 'callback', 'synchronization', 'return-value', 'reentrancy'):
            block = self.fixture()
            with patch('gdn_records.restore_prefix'):
                block.commit(2, synchronize=True)
            operation = Mock(return_value=None)
            if failure == 'binding':
                block.records[-1][0].gdn.B = 1
            elif failure == 'callback':
                operation.side_effect = RuntimeError('device')
            elif failure == 'synchronization':
                block.operations.synchronize_device.side_effect = RuntimeError('device')
            elif failure == 'return-value':
                operation.return_value = object()
            else:
                operation.side_effect = lambda: block.replay(lambda: None)
            with self.assertRaises((RuntimeError, ValueError)):
                block.replay(operation)
            with self.assertRaises(ValueError):
                block.replay(lambda: None)
            with self.assertRaises(ValueError):
                block.commit(2)
            self.assertEqual(block.replay_epoch, 0)

    def test_all_layers_commit_once_and_keep_records_until_trace_release(self):
        for prefix in range(5):
            block = self.fixture()
            with patch('gdn_records.restore_prefix') as restore, patch('gdn_records.release_owned') as release:
                block.commit(prefix)
                self.assertEqual(restore.call_count, 48)
                self.assertTrue(all(call.args[-1] == prefix for call in restore.call_args_list))
                for state, result, checkpoint in block.records:
                    state.active.restore.assert_called_once_with(checkpoint)
                release.assert_not_called()
                with self.assertRaises(ValueError):
                    block.commit(prefix)
                block.close()
                block.close()
                release.assert_called_once()

    def test_incomplete_invalid_or_rebound_blocks_do_not_write(self):
        for invalid in ('incomplete', 'prefix', 'binding', 'unstable'):
            block = self.fixture(47 if invalid == 'incomplete' else 48)
            if invalid == 'binding':
                block.records[-1][0].gdn.B = 1
            if invalid == 'unstable':
                block.records[-1][0].gdn._stable_state = False
            with patch('gdn_records.restore_prefix') as restore:
                with self.assertRaises(ValueError):
                    block.commit(True if invalid == 'prefix' else 2)
            restore.assert_not_called()
            self.assertIsNone(block.selected_prefix)

    def test_partial_device_failure_cannot_be_retried_as_a_fresh_commit(self):
        block = self.fixture()
        with patch('gdn_records.restore_prefix', side_effect=RuntimeError('device')):
            with self.assertRaises(RuntimeError):
                block.commit(2)
        with self.assertRaises(ValueError):
            block.commit(2)

    def test_dma_publishes_all_records_in_one_launch(self):
        block = self.fixture()
        mesh = object()
        for state, result, checkpoint in block.records:
            state.gdn.mesh = mesh
            result['packed_conv_states'] = [object() for slot in range(4)]
        with patch('gdn_commit_dma.publish') as publish, patch('gdn_records.restore_prefix') as restore:
            block.commit(2, dma=True)
            publish.assert_called_once()
            self.assertIs(publish.call_args.args[0], mesh)
            self.assertEqual(len(publish.call_args.args[1]), 48)
            self.assertTrue(all(len(layer) == 20 for layer in publish.call_args.args[1]))
            self.assertEqual(publish.call_args.args[2], 2)
            restore.assert_not_called()
            for state, result, checkpoint in block.records:
                state.active.restore.assert_not_called()
        with self.assertRaises(ValueError):
            block.commit(2, dma=True)

    def test_duplicate_or_unpacked_record_rejected(self):
        block = self.fixture(1)
        state, result, checkpoint = block.records[0]
        with self.assertRaises(ValueError):
            block.append(state, result, checkpoint)
        result['packed_checkpoints'] = False
        with self.assertRaises(ValueError):
            block.append(state, result, checkpoint)

    def test_bound_publication_still_requires_native_binding_and_single_decision(self):
        block = self.fixture()
        publication = Mock()
        block.commit(2, dma=True, publication=publication)
        publication.assert_called_once_with(2)
        with self.assertRaises(ValueError):
            block.commit(2, dma=True, publication=publication)
        for invalid in ('binding', 'mode'):
            block = self.fixture()
            if invalid == 'binding':
                block.records[-1][0].gdn.B = 1
            publication = Mock()
            with self.assertRaises(ValueError):
                block.commit(2, dma=invalid != 'mode', publication=publication)
            publication.assert_not_called()


class PackedRecordTests(unittest.TestCase):
    """A packed block is decided one user at a time, each into its own carry.

    The decode restored every user from its carry and never advanced it, so the
    commit writes the accepted prefix there from that user's entry and histories,
    and a prefix of zero has nothing to write. The block replays once every user
    has decided and the last decision was fenced.
    """

    def test_commit_user_publishes_that_users_layers_into_its_own_carry(self):
        block = packed_block()
        with patch('gdn_commit_dma.publish') as publish, patch('gdn_records.restore_prefix') as restore:
            block.commit_user(1, 9, dma=True)
        publish.assert_called_once()
        mesh, layers, prefix = publish.call_args.args
        self.assertIs(mesh, block.records[0][0].gdn.mesh)
        self.assertEqual((len(layers), prefix), (48, 9))
        for (state, result, carries), layer in zip(block.records, layers, strict=True):
            piece = result['segment_results'][1]
            self.assertEqual(layer, [*state.segment_entries[1], piece['states'], *piece['packed_conv_states'],
                                     state.gdn.rec_state, *state.gdn.conv_states, *carries[1]])
            state.active.restore.assert_not_called()
        restore.assert_not_called()
        self.assertEqual(block.decisions, {1: 9})
        self.assertIsNone(block.selected_prefix, 'the block has not decided until every user has')

    def test_the_eager_fallback_restores_that_users_prefix_into_its_carry(self):
        block = packed_block()
        with patch('gdn_records.restore_prefix') as restore:
            block.commit_user(0, 5)
        self.assertEqual(restore.call_count, 48)
        for (state, result, carries), call in zip(block.records, restore.call_args_list, strict=True):
            self.assertEqual(call.args, (block.operations, result['segment_results'][0],
                                         state.segment_entries[0], carries[0], 5))

    def test_prefix_zero_leaves_the_carry_alone_but_counts_as_the_decision(self):
        block = packed_block()
        publication = Mock()
        with patch('gdn_commit_dma.publish') as publish, patch('gdn_records.restore_prefix') as restore:
            block.commit_user(0, 0, dma=True, publication=publication)
            block.commit_user(1, 0)
        publish.assert_not_called()
        publication.assert_not_called()
        restore.assert_not_called()
        self.assertEqual(block.selected_prefix, (0, 0))
        for segment in (0, 1):
            with self.assertRaises(ValueError):
                block.commit_user(segment, 0)

    def test_replay_needs_every_user_decided_and_the_last_decision_fenced(self):
        block = packed_block()
        first, second = Mock(), Mock()
        with self.assertRaises(ValueError):
            block.replay(lambda: None)
        # fencing an early decision does not arm the replay: the other user has not decided
        block.commit_user(1, 9, dma=True, publication=second, synchronize=True)
        second.assert_called_once_with(9)
        self.assertFalse(block.replay_ready)
        with self.assertRaises(ValueError):
            block.replay(lambda: None)
        # and a complete decision that was not fenced is not ready either
        block.commit_user(0, 12, dma=True, publication=first)
        self.assertEqual(block.selected_prefix, (12, 9))
        self.assertFalse(block.replay_ready)
        with self.assertRaises(ValueError):
            block.replay(lambda: None)
        block = packed_block()
        block.commit_user(0, 12, dma=True, publication=first)
        block.commit_user(1, 0, synchronize=True)
        self.assertTrue(block.replay_ready)
        operation = Mock(return_value=None)
        block.replay(operation)
        operation.assert_called_once_with()
        self.assertEqual((block.replay_epoch, block.selected_prefix, block.decisions), (1, None, {}))
        with self.assertRaises(ValueError):
            block.replay(operation)
        # the next epoch decides afresh, in any order
        block.commit_user(1, 3, dma=True, publication=second)
        block.commit_user(0, 0, synchronize=True)
        self.assertEqual((block.selected_prefix, block.replay_ready), ((0, 3), True))
        self.assertEqual(block.operations.synchronize_device.call_count, 3)

    def test_a_failed_publication_poisons_the_block(self):
        block = packed_block()
        with patch('gdn_commit_dma.publish', side_effect=RuntimeError('device')):
            with self.assertRaises(RuntimeError):
                block.commit_user(0, 5, dma=True)
        for segment, prefix in ((0, 5), (1, 0)):
            with self.assertRaises(ValueError):
                block.commit_user(segment, prefix, synchronize=True)
        self.assertFalse(block.replay_ready)
        with self.assertRaises(ValueError):
            block.replay(lambda: None)
        self.assertEqual(block.operations.synchronize_device.call_count, 0)

    def test_decisions_are_bounded_to_the_users_own_rows_and_users(self):
        block = packed_block()
        for segment, prefix in ((2, 0), (-1, 0), (True, 0), (0, 17), (0, -1), (1, True)):
            with self.assertRaises(ValueError):
                block.commit_user(segment, prefix)
        with self.assertRaises(ValueError):
            block.commit_user(0, 5, publication=Mock())
        self.assertEqual(block.decisions, {})

    def test_single_and_per_user_decisions_do_not_cross(self):
        with self.assertRaises(ValueError):
            packed_block().commit(2)
        single = unpacked_block(rows=32)
        with self.assertRaises(ValueError):
            single.commit_user(0, 2)
        with self.assertRaises(ValueError):
            single.segment_layers(0)

    def test_segment_layers_are_per_user_in_record_order(self):
        block = packed_block()
        for segment in (0, 1):
            self.assertEqual([len(layer) for layer in block.segment_layers(segment)], [20] * 48)
        own = {id(value) for segment in (0, 1) for layer in block.segment_layers(segment)
               for value in layer[:10] + layer[15:]}
        self.assertEqual(len(own), 2 * 48 * 15, 'entry, histories and carry are each user\'s own')
        for segment in (2, -1, None):
            with self.assertRaises(ValueError):
                block.segment_layers(segment)

    def test_packed_records_carry_each_users_prefixes_entry_and_carry(self):
        mesh = object()
        block = packed_block(count=0)
        state, result, carries = packed_record(mesh)
        pieces = result['segment_results']
        broken = [
            (state, dict(result, segment_results=pieces[:1]), carries),
            (SimpleNamespace(**dict(vars(state), segment_entries=state.segment_entries[:1])), result, carries),
            (SimpleNamespace(**dict(vars(state), segment_entries=None)), result, carries),
            (state, result, carries[:1]),
            (state, dict(result, segments=((0, 16), (16, 24))), carries),
            (state, dict(result, segment_results=(dict(pieces[0], packed_checkpoints=False), pieces[1])), carries),
            (state, dict(result, segment_results=(dict(pieces[0], states=SimpleNamespace(shape=(32, 24, 128, 128))),
                                                  pieces[1])), carries),
        ]
        for record in broken:
            with self.assertRaises(ValueError):
                block.append(*record)
        self.assertEqual(block.records, [])
        block.append(state, result, carries)
        self.assertEqual(block.segments, ((0, 16), (16, 32)))
        # every layer packs the same users, and packed and single-sequence layers never mix
        with self.assertRaises(ValueError):
            block.append(*packed_record(mesh, segments=((0, 8), (8, 32))))
        with self.assertRaises(ValueError):
            block.append(*unpacked_record(mesh, 32))
        single = unpacked_block(count=1, rows=32)
        with self.assertRaises(ValueError):
            single.append(*packed_record(mesh))
        self.assertEqual((len(block.records), len(single.records)), (1, 1))
