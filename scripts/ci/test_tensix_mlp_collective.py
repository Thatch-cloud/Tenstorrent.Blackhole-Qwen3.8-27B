from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tensix_mlp_collective import reduce_partial


class TensixMlpCollectiveTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', L1_MEMORY_CONFIG='l1',
            DRAM_MEMORY_CONFIG='dram', Topology=SimpleNamespace(Linear='linear'),
            experimental=SimpleNamespace(reduce_scatter_minimal_async=Mock()))
        mesh = SimpleNamespace(shape=[1, 2])
        collectives = SimpleNamespace(get_num_links=Mock(return_value=4),
            get_and_cycle_rs_semaphore_handles=Mock(return_value='rs'),
            get_and_cycle_barrier_semaphore_handle=Mock(return_value='barrier'))
        partial = SimpleNamespace(shape=(1, 1, 8, 5120), dtype='bf16', layout='tile',
            memory_config=lambda: 'l1', is_sharded=lambda: False, deallocate=Mock())
        return operations, mesh, collectives, partial

    def test_native_parameters_match_without_deallocating_borrowed_output(self):
        operations, mesh, collectives, partial = self.fixture()
        result = reduce_partial(operations, mesh, collectives, partial, 'linear')
        collectives.get_num_links.assert_called_once_with(0)
        collectives.get_and_cycle_rs_semaphore_handles.assert_called_once_with()
        collectives.get_and_cycle_barrier_semaphore_handle.assert_called_once_with()
        operations.experimental.reduce_scatter_minimal_async.assert_called_once_with(
            partial, persistent_output_buffers=None, dim=3, multi_device_global_semaphore='rs',
            barrier_semaphore='barrier', num_links=4, memory_config='dram', intermediate_memory_config='dram',
            topology='linear', chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2, subdevice_id=None)
        self.assertIs(result, operations.experimental.reduce_scatter_minimal_async.return_value)
        partial.deallocate.assert_not_called()

    def test_drift_fails_before_collective_or_semaphore_use(self):
        for mutation in ('mesh', 'rows', 'dtype', 'layout', 'memory', 'sharded', 'links', 'topology'):
            operations, mesh, collectives, partial = self.fixture()
            topology = 'linear'
            if mutation == 'mesh':
                mesh.shape = [2, 1]
            elif mutation == 'rows':
                partial.shape = (1, 1, 1, 5120)
            elif mutation in ('dtype', 'layout'):
                setattr(partial, mutation, 'invalid')
            elif mutation == 'memory':
                partial.memory_config = lambda: 'dram'
            elif mutation == 'sharded':
                partial.is_sharded = lambda: True
            elif mutation == 'links':
                collectives.get_num_links.return_value = 1
            else:
                topology = 'ring'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reduce_partial(operations, mesh, collectives, partial, topology)
            operations.experimental.reduce_scatter_minimal_async.assert_not_called()
            collectives.get_and_cycle_rs_semaphore_handles.assert_not_called()
            collectives.get_and_cycle_barrier_semaphore_handle.assert_not_called()
            partial.deallocate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
