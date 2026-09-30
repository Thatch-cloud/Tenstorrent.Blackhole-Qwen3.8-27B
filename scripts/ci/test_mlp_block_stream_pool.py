from types import SimpleNamespace
import unittest

from mlp_block_stream_pool import admit_memory, memory_requirement, owned_streams


class StreamPoolTests(unittest.TestCase):
    def fixture(self):
        weights = list(range(64))
        released = []
        view = dict(num_banks=8, total_bytes_free_per_bank=2 ** 30,
            largest_contiguous_bytes_free_per_bank=2 ** 30)
        operations = SimpleNamespace(BufferType=SimpleNamespace(DRAM='dram'),
            synchronize_device=lambda mesh: None, deallocate=released.append,
            get_device_tensors=lambda weight: [SimpleNamespace(buffer_address=lambda: weight,
                device=lambda chip=chip: chip) for chip in range(2)],
            get_memory_view=lambda device, kind: SimpleNamespace(**view))
        return weights, released, view, operations

    def test_bank_padding_and_reserve_are_charged(self):
        requirement = memory_requirement(8)
        self.assertEqual(requirement['logical_stream_bytes_per_chip'], 3220439040)
        self.assertGreater(requirement['allocated_stream_bytes_per_chip'], requirement['logical_stream_bytes_per_chip'])
        self.assertEqual(requirement['required_free_per_bank'],
            requirement['allocated_stream_bytes_per_chip'] // 8 + 2 ** 27)
        for banks in (0, True, -1):
            with self.assertRaises(ValueError):
                memory_requirement(banks)

    def test_admission_rejects_pressure_and_fragmentation(self):
        weights, released, view, operations = self.fixture()
        admit_memory([view, view])
        for field in ('total_bytes_free_per_bank', 'largest_contiguous_bytes_free_per_bank'):
            invalid = dict(view, **{field: 0})
            with self.assertRaises(ValueError):
                admit_memory([view, invalid])

    def test_pool_releases_only_owned_streams_on_request_failure(self):
        weights, released, view, operations = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            with owned_streams(operations, None, weights, packer=lambda mesh, weight: weight + 1000) as (streams, audit):
                self.assertEqual(len(streams), 64)
                self.assertEqual(audit['allocated_layers'], 64)
                raise RuntimeError('request failed')
        self.assertEqual(released, list(reversed(range(1000, 1064))))
        self.assertTrue(audit['released'])
        self.assertTrue(audit['native_bindings_unchanged'])

    def test_partial_allocation_failure_releases_previous_streams(self):
        weights, released, view, operations = self.fixture()

        def packer(mesh, weight):
            if weight == 3:
                raise RuntimeError('allocation failed')
            return weight + 1000

        with self.assertRaisesRegex(RuntimeError, 'allocation failed'):
            with owned_streams(operations, None, weights, packer=packer):
                self.fail('Partial pool must not be exposed')
        self.assertEqual(released, [1002, 1001, 1000])

    def test_pressure_rejects_before_any_allocation(self):
        weights, released, view, operations = self.fixture()
        view['total_bytes_free_per_bank'] = 0
        with self.assertRaises(ValueError):
            with owned_streams(operations, None, weights,
                    packer=lambda *args: self.fail('Must not allocate under pressure')):
                self.fail('Must not admit pressure')
        self.assertEqual(released, [])


if __name__ == '__main__':
    unittest.main()
