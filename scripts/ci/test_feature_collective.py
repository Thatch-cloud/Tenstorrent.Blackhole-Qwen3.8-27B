import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from feature_collective import reduce_projection, gather_add_projection


class FeatureCollectiveTests(unittest.TestCase):
    def test_four_link_policy_reaches_both_collective_paths(self):
        operations, mesh, collectives, value, reduced, gathered = self.fixture()
        with patch('feature_collective.projection_links', return_value=4):
            reduce_projection(operations, mesh, collectives, value)
            self.assertEqual(operations.experimental.reduce_scatter_minimal_async.call_args.kwargs['num_links'], 4)
            self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['num_links'], 4)
            operations.slice = Mock(side_effect=[object(), object()])
            operations.add = Mock(return_value=object())
            gather_add_projection(operations, mesh, collectives, value)
            self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['num_links'], 4)

    def test_trace_ownership_defers_sync_and_temporary_release(self):
        operations, mesh, collectives, value, reduced, gathered = self.fixture()
        left, right, output = object(), object(), object()
        operations.slice = Mock(side_effect=[left, right])
        operations.add = Mock(return_value=output)
        owned = []
        self.assertIs(gather_add_projection(operations, mesh, collectives, value,
            retain_temporaries=owned.append), output)
        self.assertEqual(owned, [gathered, left, right])
        operations.synchronize_device.assert_not_called()
        operations.deallocate.assert_not_called()

    def fixture(self):
        reduced, output = object(), object()
        operations = SimpleNamespace(float32='fp32', DRAM_MEMORY_CONFIG='dram', Topology=SimpleNamespace(Linear='linear'),
            experimental=SimpleNamespace(reduce_scatter_minimal_async=Mock(return_value=reduced),
                all_gather_async=Mock(return_value=output)), synchronize_device=Mock(), deallocate=Mock())
        return operations, SimpleNamespace(shape=[1, 2]), Mock(), SimpleNamespace(shape=(1, 1, 1, 5120), dtype='fp32'), reduced, output

    def test_explicit_collective_chain_owns_only_temporary(self):
        operations, mesh, collectives, value, reduced, output = self.fixture()
        self.assertIs(reduce_projection(operations, mesh, collectives, value), output)
        self.assertIs(operations.experimental.all_gather_async.call_args.args[0], reduced)
        self.assertEqual(operations.experimental.reduce_scatter_minimal_async.call_args.kwargs['dim'], 3)
        self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['num_links'], 1)
        operations.synchronize_device.assert_called_once_with(mesh)
        operations.deallocate.assert_called_once_with(reduced)

    def test_gather_add_uses_raw_transfers_and_float32_add(self):
        operations, mesh, collectives, value, reduced, gathered = self.fixture()
        left, right, output = object(), object(), object()
        operations.slice = Mock(side_effect=[left, right])
        operations.add = Mock(return_value=output)
        self.assertIs(gather_add_projection(operations, mesh, collectives, value), output)
        operations.experimental.reduce_scatter_minimal_async.assert_not_called()
        self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['dim'], 0)
        operations.add.assert_called_once_with(left, right, dtype='fp32', memory_config='dram')
        self.assertEqual([call.args[0] for call in operations.deallocate.call_args_list], [right, left, gathered])

    def test_gather_failure_frees_temporary_and_invalid_shape_never_dispatches(self):
        operations, mesh, collectives, value, reduced, output = self.fixture()
        operations.experimental.all_gather_async.side_effect = RuntimeError('gather failed')
        with self.assertRaises(RuntimeError):
            reduce_projection(operations, mesh, collectives, value)
        operations.deallocate.assert_called_once_with(reduced)
        operations.experimental.reduce_scatter_minimal_async.reset_mock()
        value.dtype = 'bf16'
        with self.assertRaises(ValueError):
            reduce_projection(operations, mesh, collectives, value)
        operations.experimental.reduce_scatter_minimal_async.assert_not_called()

    def test_batched_gather_add_keeps_every_row(self):
        for rows in (8, 32):
            with self.subTest(rows=rows):
                operations, mesh, collectives, value, reduced, gathered = self.fixture()
                value.shape = (1, 1, rows, 5120)
                operations.slice = Mock(side_effect=[object(), object()])
                operations.add = Mock(return_value=object())
                gather_add_projection(operations, mesh, collectives, value)
                self.assertEqual([call.args[1:] for call in operations.slice.call_args_list],
                    [((0, 0, 0, 0), (1, 1, rows, 5120)), ((1, 0, 0, 0), (2, 1, rows, 5120))])

    def test_gather_add_rejects_unsupported_geometry_before_dispatch(self):
        for shape in ((1, 1, 2, 5120), (1, 1, 8, 2560), (1, 8, 5120), (2, 1, 8, 5120)):
            operations, mesh, collectives, value, reduced, gathered = self.fixture()
            value.shape = shape
            with self.assertRaises(ValueError):
                gather_add_projection(operations, mesh, collectives, value)
            operations.experimental.all_gather_async.assert_not_called()


class FourCardTests(unittest.TestCase):
    """QWEN_FAST_TP=4: a (1, 4) mesh, four per-chip partials summed in fixed chip order, and the same call shape."""

    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def fixture(self, rows=8):
        gathered = object()
        operations = SimpleNamespace(float32='fp32', DRAM_MEMORY_CONFIG='dram',
            Topology=SimpleNamespace(Linear='linear', Ring='ring'),
            experimental=SimpleNamespace(reduce_scatter_minimal_async=Mock(return_value='reduced'),
                all_gather_async=Mock(return_value=gathered)), synchronize_device=Mock(), deallocate=Mock())
        value = SimpleNamespace(shape=(1, 1, rows, 5120), dtype='fp32')
        return operations, SimpleNamespace(shape=[1, 4]), Mock(), value, gathered

    def test_the_partials_are_added_in_chip_order(self):
        operations, mesh, collectives, value, gathered = self.fixture()
        slices = [object() for _ in range(4)]
        sums = [object(), object(), object()]
        operations.slice = Mock(side_effect=slices)
        operations.add = Mock(side_effect=sums)
        with patch('feature_collective.projection_links', return_value=2):
            output = gather_add_projection(operations, mesh, collectives, value)
        self.assertIs(output, sums[2])
        self.assertEqual([call.args for call in operations.add.call_args_list],
                         [(slices[0], slices[1]), (sums[0], slices[2]), (sums[1], slices[3])])
        self.assertTrue(all(call.kwargs == dict(dtype='fp32', memory_config='dram')
                            for call in operations.add.call_args_list))
        self.assertEqual([call.args[1] for call in operations.slice.call_args_list],
                         [(chip, 0, 0, 0) for chip in range(4)])
        self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['num_links'], 2)
        self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['topology'], 'linear')
        # the gather, every slice and the two intermediate sums are released; the result is the caller's
        released = [call.args[0] for call in operations.deallocate.call_args_list]
        self.assertEqual(sorted(map(id, released)), sorted(map(id, [gathered, *slices, sums[0], sums[1]])))

    def test_trace_ownership_takes_the_intermediate_sums_too(self):
        operations, mesh, collectives, value, gathered = self.fixture()
        slices = [object() for _ in range(4)]
        sums = [object(), object(), object()]
        operations.slice = Mock(side_effect=slices)
        operations.add = Mock(side_effect=sums)
        owned = []
        with patch('feature_collective.projection_links', return_value=2):
            gather_add_projection(operations, mesh, collectives, value, retain_temporaries=owned.append)
        self.assertEqual(owned, [gathered, *slices, sums[0], sums[1]])
        operations.deallocate.assert_not_called()
        operations.synchronize_device.assert_not_called()

    def test_a_failed_add_frees_the_last_sum_and_every_temporary(self):
        operations, mesh, collectives, value, gathered = self.fixture()
        slices = [object() for _ in range(4)]
        first = object()
        operations.slice = Mock(side_effect=slices)
        operations.add = Mock(side_effect=[first, RuntimeError('add failed')])
        with patch('feature_collective.projection_links', return_value=2):
            with self.assertRaises(RuntimeError):
                gather_add_projection(operations, mesh, collectives, value)
        released = [call.args[0] for call in operations.deallocate.call_args_list]
        self.assertEqual(sorted(map(id, released)), sorted(map(id, [gathered, *slices, first])))

    def test_the_ring_topology_is_only_what_the_switch_asks_for(self):
        operations, mesh, collectives, value, gathered = self.fixture()
        operations.slice = Mock(side_effect=[object() for _ in range(4)])
        operations.add = Mock(return_value=object())
        with patch('feature_collective.projection_links', return_value=2),                 patch.dict(os.environ, {'QWEN_FAST_CCL_TOPOLOGY': 'ring'}):
            gather_add_projection(operations, mesh, collectives, value)
        self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['topology'], 'ring')

    def test_the_reduce_scatter_path_takes_a_four_card_mesh(self):
        operations, mesh, collectives, value, gathered = self.fixture()
        value.shape = (1, 1, 1, 5120)
        with patch('feature_collective.projection_links', return_value=2):
            self.assertIs(reduce_projection(operations, mesh, collectives, value), gathered)
        self.assertEqual(operations.experimental.reduce_scatter_minimal_async.call_args.kwargs['topology'], 'linear')

    def test_the_wrong_mesh_for_the_width_is_refused_before_dispatch(self):
        operations, mesh, collectives, value, gathered = self.fixture()
        for shape in ([1, 2], [2, 2], [4, 1]):
            mesh.shape = shape
            with patch('feature_collective.projection_links', return_value=2):
                with self.assertRaises(ValueError) as failure:
                    gather_add_projection(operations, mesh, collectives, value)
            self.assertIn('TP4', str(failure.exception))
        operations.experimental.all_gather_async.assert_not_called()
        with patch.dict(os.environ, {'QWEN_FAST_TP': '2'}), patch('feature_collective.projection_links', return_value=1):
            mesh.shape = [1, 4]
            with self.assertRaises(ValueError) as failure:
                gather_add_projection(operations, mesh, collectives, value)
        self.assertIn('TP2', str(failure.exception))

    def test_quad_draft_takes_the_same_four_card_sum_at_sixty_four_rows(self):
        import quad_draft

        operations, mesh, collectives, value, gathered = self.fixture()
        value.shape = (1, 1, 64, 5120)
        slices = [object() for _ in range(4)]
        sums = [object(), object(), object()]
        operations.slice = Mock(side_effect=slices)
        operations.add = Mock(side_effect=sums)
        owned = []
        with patch('mesh_link_policy.projection_links', return_value=2):
            output = quad_draft.gather_add_projection(operations, mesh, collectives, value,
                                                      retain_temporaries=owned.append)
        self.assertIs(output, sums[2])
        self.assertEqual([call.args for call in operations.add.call_args_list],
                         [(slices[0], slices[1]), (sums[0], slices[2]), (sums[1], slices[3])])
        self.assertEqual(owned, [gathered, *slices, sums[0], sums[1]])
        self.assertEqual(operations.experimental.all_gather_async.call_args.kwargs['topology'], 'linear')


if __name__ == '__main__':
    unittest.main()
