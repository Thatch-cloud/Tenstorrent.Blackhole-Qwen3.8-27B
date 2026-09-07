from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from feature_collective import reduce_projection, gather_add_projection


class FeatureCollectiveTests(unittest.TestCase):
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
