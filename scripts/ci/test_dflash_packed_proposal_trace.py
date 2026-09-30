import unittest
import unittest.mock
from types import SimpleNamespace


class FakeMesh:
    pass


def fake_operations():
    operations = SimpleNamespace(bfloat16='bf16', uint32='u32', TILE_LAYOUT='tile',
        ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram')
    operations.from_torch = unittest.mock.Mock(
        side_effect=lambda *a, dtype=None, layout=None, **k: SimpleNamespace(dtype=dtype, layout=layout))
    operations.ReplicateTensorToMesh = unittest.mock.Mock(return_value='map')
    operations.synchronize_device = unittest.mock.Mock()
    operations.begin_trace_capture = unittest.mock.Mock(side_effect=lambda mesh, cq_id=0: object())
    operations.end_trace_capture = unittest.mock.Mock()
    operations.execute_trace = unittest.mock.Mock()
    operations.release_trace = unittest.mock.Mock()
    operations.copy_host_to_device_tensor = unittest.mock.Mock()
    operations.copy = unittest.mock.Mock()
    operations.slice = unittest.mock.Mock(side_effect=lambda *a, **k: object())
    return operations


def fake_device(operations, mesh, *, position, history_rows, name):
    kv_history = SimpleNamespace(active=[{'k': object(), 'v': object()} for _ in range(5)],
        pending=None, owned=[], borrowed=[])
    return SimpleNamespace(operations=operations, mesh=mesh, block_rows=16,
        native_proposal_attention=True, kv_history=kv_history, position=position,
        history_rows=history_rows, history=None, spare_history=None,
        closed=False, pending=None, progress=None,
        validated_native_proposal_masks=set(),
        temporaries=lambda protected: ([], lambda value: value),
        execute_proposal=unittest.mock.Mock(return_value=SimpleNamespace(projected='projected', chunks=())),
        name=name)


class PreparedPackedDFlashProposalTests(unittest.TestCase):
    def paired(self, *, context_a=300, context_b=300, position_a=4096, position_b=1200):
        mesh = FakeMesh()
        operations = fake_operations()
        device_a = fake_device(operations, mesh, position=position_a, history_rows=context_a, name='a')
        device_b = fake_device(operations, mesh, position=position_b, history_rows=context_b, name='b')
        return operations, device_a, device_b

    def build(self, device_a, device_b):
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned'):
            from dflash_proposal_trace import PreparedPackedDFlashProposal

            return PreparedPackedDFlashProposal(device_a, device_b)

    def test_construction_requires_a_shared_mesh_and_runtime(self):
        operations, device_a, device_b = self.paired()
        device_b.mesh = FakeMesh()
        with self.assertRaises(ValueError):
            self.build(device_a, device_b)

    def test_construction_requires_matching_block_rows(self):
        operations, device_a, device_b = self.paired()
        device_b.block_rows = 32
        with self.assertRaises(ValueError):
            self.build(device_a, device_b)

    def test_construction_requires_native_proposal_attention_on_both(self):
        operations, device_a, device_b = self.paired()
        device_b.native_proposal_attention = False
        with self.assertRaises(ValueError):
            self.build(device_a, device_b)

    def test_construction_requires_a_committed_kv_cache_on_both(self):
        operations, device_a, device_b = self.paired()
        device_b.kv_history = None
        with self.assertRaises(ValueError):
            self.build(device_a, device_b)

    def test_prepare_device_issues_one_packed_call_with_no_history_tensor(self):
        operations, device_a, device_b = self.paired(context_a=300, context_b=300)
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned'):
            trace = self.build(device_a, device_b)
            ready = trace.prepare_device(11, 22)
        self.assertTrue(ready)
        # First use of this geometry: one untraced warm-up execute_proposal() call,
        # then one more inside capture_operation() to record the trace -
        # PreparedDFlashProposal.__init__'s own two-phase build, mirrored for the
        # pair. Both calls carry identical packed arguments; check either.
        self.assertEqual(device_a.execute_proposal.call_count, 2)
        call = device_a.execute_proposal.call_args
        self.assertIsNone(call.args[1], 'the cached packed path builds no history tensor')
        self.assertIsNone(call.kwargs['context'], 'a single context has no meaning packed')
        self.assertEqual([user['history_rows'] for user in call.kwargs['pack']], [300, 300])
        self.assertEqual(len(call.kwargs['cached_history']), 2, 'one cache list per paired user')
        # First bucket build validates its own capture with one blocking replay, then
        # prepare_device()'s own replay is the non-blocking one actually being tested.
        self.assertEqual(operations.execute_trace.call_count, 2)
        operations.execute_trace.assert_called_with(device_a.mesh, unittest.mock.ANY, cq_id=0, blocking=False)

    def test_has_pending_and_finish_split_by_which_side(self):
        operations, device_a, device_b = self.paired(context_a=300, context_b=300)
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned'), \
                unittest.mock.patch('dflash_packed_proposal.select_device_outputs',
                                    return_value=((1, 2, 3), (4, 5))) as select:
            trace = self.build(device_a, device_b)
            trace.prepare_device(11, 22)
            self.assertTrue(trace.has_pending('a', 11))
            self.assertFalse(trace.has_pending('a', 999), 'the wrong seed is never pending')
            self.assertTrue(trace.has_pending('b', 22))
            tokens_a = trace.finish('a', 2)
            self.assertFalse(trace.has_pending('a', 11), "a's own half is consumed")
            self.assertTrue(trace.has_pending('b', 22), "b's half is still outstanding")
            tokens_b = trace.finish('b', 1)
        self.assertEqual(tokens_a, (1, 2))
        self.assertEqual(tokens_b, (4,))
        select.assert_called_once()
        self.assertEqual(select.call_args.args[2], (11, 22))
        self.assertFalse(trace.has_pending('a', 11), 'both halves consumed clears pending entirely')

    def test_finish_without_a_pending_prepare_is_refused(self):
        operations, device_a, device_b = self.paired()
        trace = self.build(device_a, device_b)
        with self.assertRaises(ValueError):
            trace.finish('a', 1)

    def test_the_same_geometry_reuses_one_captured_bucket(self):
        operations, device_a, device_b = self.paired(context_a=300, context_b=300)
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned'), \
                unittest.mock.patch('dflash_packed_proposal.select_device_outputs', return_value=((1,), (2,))):
            trace = self.build(device_a, device_b)
            trace.prepare_device(11, 22)
            trace.finish('a', 1)
            trace.finish('b', 1)
            captures_after_first_round = operations.begin_trace_capture.call_count
            trace.prepare_device(33, 44)
            trace.finish('a', 1)
            trace.finish('b', 1)
        self.assertEqual(captures_after_first_round, 1)
        self.assertEqual(operations.begin_trace_capture.call_count, 1, 'a second round at the same geometry recaptures nothing')
        self.assertEqual(operations.execute_trace.call_count, 3, 'one capture replay plus two prepare_device() replays')

    def test_a_different_geometry_builds_a_second_bucket(self):
        operations, device_a, device_b = self.paired(context_a=300, context_b=300)
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned'), \
                unittest.mock.patch('dflash_packed_proposal.select_device_outputs', return_value=((1,), (2,))):
            trace = self.build(device_a, device_b)
            trace.prepare_device(11, 22)
            trace.finish('a', 1)
            trace.finish('b', 1)
            device_a.history_rows = device_b.history_rows = 500
            trace.prepare_device(33, 44)
            trace.finish('a', 1)
            trace.finish('b', 1)
        self.assertEqual(operations.begin_trace_capture.call_count, 2)
        self.assertEqual(len(trace.buckets), 2)

    def test_discard_pending_releases_without_finishing(self):
        operations, device_a, device_b = self.paired(context_a=300, context_b=300)
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned') as release:
            trace = self.build(device_a, device_b)
            trace.prepare_device(11, 22)
            trace.discard_pending()
            self.assertFalse(trace.has_pending('a', 11))
            with self.assertRaises(ValueError):
                trace.finish('a', 1)
        release.assert_called()

    def test_prepare_device_declines_when_a_paired_device_is_mid_publication(self):
        operations, device_a, device_b = self.paired()
        device_b.pending = object()
        trace = self.build(device_a, device_b)
        self.assertFalse(trace.prepare_device(1, 2))
        device_a.execute_proposal.assert_not_called()

    def test_a_failed_build_releases_every_placeholder_it_uploaded_not_just_the_capture_scope(self):
        """Run 35585107688: a failed bucket build used to leave its identifiers/mask/
        rope/cached_history placeholders allocated forever (only the capture-scoped
        transients were released) - self.owned must be back to its pre-attempt length
        after a failure, and everything uploaded during the attempt must have gone
        through release_owned."""
        operations, device_a, device_b = self.paired(context_a=300, context_b=300)
        device_a.execute_proposal = unittest.mock.Mock(side_effect=RuntimeError('warm-up OOM'))
        released = []
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned',
                                    side_effect=lambda operations, values: released.extend(values)):
            trace = self.build(device_a, device_b)
            before = len(trace.owned)
            with self.assertRaisesRegex(RuntimeError, 'warm-up OOM'):
                trace.prepare_device(11, 22)
        self.assertEqual(len(trace.owned), before, 'no placeholder from the failed attempt stays tracked')
        # identifiers + mask + 2 rope.q + 2 rope.k + 2 rope.live_k + (2 users x 5
        # layers x 2 k/v) cached_history = 1+1+2+2+2+20 = 28 placeholders uploaded
        # for this one failed attempt, every one of them released.
        self.assertEqual(len(released), 28)
        self.assertEqual(trace.buckets, {}, 'the failed bucket never stays cached')

    def test_a_retry_after_a_failed_build_does_not_accumulate_leaked_placeholders(self):
        """Three consecutive failed attempts (run 35585107688's own shape) must leave
        the SAME owned length as one failed attempt - never growing - and, once the
        underlying condition clears, a successful build still only holds ONE
        attempt's worth of placeholders."""
        operations, device_a, device_b = self.paired(context_a=300, context_b=300)
        real_execute = device_a.execute_proposal
        device_a.execute_proposal = unittest.mock.Mock(side_effect=RuntimeError('warm-up OOM'))
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned'):
            trace = self.build(device_a, device_b)
            before = len(trace.owned)
            for _ in range(3):
                with self.assertRaises(RuntimeError):
                    trace.prepare_device(11, 22)
                self.assertEqual(len(trace.owned), before, 'never grows across repeated failed attempts')
            device_a.execute_proposal = real_execute
            with unittest.mock.patch('dflash_packed_proposal.select_device_outputs', return_value=((1,), (2,))):
                self.assertTrue(trace.prepare_device(33, 44))
                trace.finish('a', 1)
                trace.finish('b', 1)
        self.assertGreater(len(trace.owned), before, 'the successful attempt uploaded its own placeholders')
        self.assertEqual(len(trace.buckets), 1)

    def test_close_releases_the_trace_and_validated_masks(self):
        operations, device_a, device_b = self.paired(context_a=300, context_b=300)
        with unittest.mock.patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_proposal_trace.release_owned'):
            trace = self.build(device_a, device_b)
            trace.prepare_device(11, 22)
            self.assertEqual(len(device_a.validated_native_proposal_masks), 1)
            trace.close()
        operations.release_trace.assert_called_once()
        self.assertEqual(len(device_a.validated_native_proposal_masks), 0)
        self.assertTrue(trace.closed)
        trace.close()  # idempotent


if __name__ == '__main__':
    unittest.main()
