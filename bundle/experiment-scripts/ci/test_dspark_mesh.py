from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dspark_mesh import gather_partials
from test_dspark_projection import operations, tensor


class DSparkMeshTests(unittest.TestCase):
    def fixture(self):
        runtime = operations()
        gathered = tensor((2,1,32,5120),'fp32')
        parts = [tensor((1,1,32,5120),'fp32') for chip in range(2)]
        runtime.experimental = SimpleNamespace(all_gather_async=MagicMock(return_value=gathered))
        runtime.Topology = SimpleNamespace(Linear='linear')
        runtime.slice = MagicMock(side_effect=parts)
        runtime.to_torch = MagicMock(side_effect=AssertionError('Host staging is forbidden'))
        runtime.from_torch = MagicMock(side_effect=AssertionError('Host staging is forbidden'))
        return runtime,gathered,parts

    def test_peer_order_precision_and_lifetime_are_explicit(self):
        runtime,gathered,parts = self.fixture()
        collectives = MagicMock()
        partial = tensor((1,1,32,5120),'fp32')
        owned = []
        with patch('dspark_mesh.projection_links',return_value=4):
            result = gather_partials(runtime,SimpleNamespace(shape=(1,2)),collectives,partial,
                lambda value:owned.append(value) or value)
        self.assertEqual(result,tuple(parts))
        self.assertEqual(owned,[gathered,*parts])
        self.assertEqual(runtime.experimental.all_gather_async.call_args.args,(partial,))
        self.assertEqual(runtime.experimental.all_gather_async.call_args.kwargs['num_links'],4)
        self.assertEqual(runtime.experimental.all_gather_async.call_args.kwargs['dim'],0)
        self.assertEqual([entry.args[1:] for entry in runtime.slice.call_args_list],
            [((0,0,0,0),(1,1,32,5120)),((1,0,0,0),(2,1,32,5120))])
        runtime.to_torch.assert_not_called()
        runtime.from_torch.assert_not_called()

    def test_invalid_partial_rejects_before_collective_dispatch(self):
        for partial in (tensor((1,1,8,5120),'fp32'),tensor((1,1,32,5120)),tensor((1,1,32,2560),'fp32')):
            runtime,gathered,parts = self.fixture()
            with self.assertRaises(ValueError):
                gather_partials(runtime,SimpleNamespace(shape=(1,2)),MagicMock(),partial,lambda value:value)
            runtime.experimental.all_gather_async.assert_not_called()

    def test_wrong_gather_shape_remains_owned_for_cleanup(self):
        runtime,gathered,parts = self.fixture()
        wrong = tensor((1,1,32,10240),'fp32')
        runtime.experimental.all_gather_async.return_value = wrong
        owned = []
        with patch('dspark_mesh.projection_links',return_value=1):
            with self.assertRaises(ValueError):
                gather_partials(runtime,SimpleNamespace(shape=(1,2)),MagicMock(),tensor((1,1,32,5120),'fp32'),
                    lambda value:owned.append(value) or value)
        self.assertEqual(owned,[wrong])
        runtime.slice.assert_not_called()


if __name__ == '__main__':
    unittest.main()
