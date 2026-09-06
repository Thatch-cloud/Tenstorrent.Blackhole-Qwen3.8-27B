from types import SimpleNamespace
import unittest

from gdn_multitoken_conv import addresses, finish_output, release_owned, restore_prefix, validate_projected


class ConvIntegrationTests(unittest.TestCase):
    def test_output_projection_preserves_rows_and_uses_native_reduce_contract(self):
        calls = []
        partial = SimpleNamespace(shape=(1, 4, 5120))
        output = object()
        gdn = SimpleNamespace(_row_proj=lambda value, weight: partial, tw={'out': object()},
                              mesh='mesh', tt_ccl='ccl', args=SimpleNamespace(ccl_topology=lambda: 'fabric'))
        operations = SimpleNamespace(reshape=lambda value, shape: calls.append(shape) or value,
                                     DRAM_MEMORY_CONFIG='dram')
        def reduce(value, mesh, ccl, **kwargs):
            calls.append((value, mesh, ccl, kwargs))
            return output
        result = dict(output=SimpleNamespace(shape=(1, 4, 3072)), owned=[])
        self.assertIs(finish_output(gdn, result, operations, reduce), result)
        self.assertEqual(calls[0], (1, 1, 4, 5120))
        self.assertEqual(calls[1][1:], ('mesh', 'ccl', dict(cluster_axis=0, dim=3, topology='fabric', memory_config='dram')))
        self.assertIs(result['layer_output'], output)
        self.assertEqual(result['owned'], [output])

    def test_compact_single_sequence_geometry(self):
        states = [SimpleNamespace(shape=(1, 1, 5120)) for index in range(4)]
        for rows in (1, 2, 4, 8, 16):
            for width in (8240, 8256):
                self.assertEqual(validate_projected((1, rows, width), states), rows)
        for shape in ((1, 32, 8256), (2, 2, 8256), (1, 2, 8192)):
            with self.assertRaises(ValueError):
                validate_projected(shape, states)
        with self.assertRaises(ValueError):
            validate_projected((1, 2, 8256), [SimpleNamespace(shape=(1, 8, 5120))] * 4)

    def test_release_deduplicates_shared_wrappers_before_free(self):
        freed = []
        operations = SimpleNamespace(get_device_tensors=lambda tensor: [SimpleNamespace(buffer_address=lambda: tensor)] * 2,
                                     deallocate=freed.append)
        release_owned(operations, [10, 20, 10, 20])
        self.assertEqual(freed, [10, 20])

    def test_two_chips_required(self):
        with self.assertRaises(ValueError):
            addresses(SimpleNamespace(get_device_tensors=lambda tensor: []), 10)

    def restore_fixture(self, rows=2):
        def tensor(shape):
            return SimpleNamespace(shape=shape)
        shapes = [(1, 24, 128, 128), *[(1, 1, 5120)] * 4]
        entry = [tensor(shape) for shape in shapes]
        destinations = [tensor(shape) for shape in shapes]
        result = dict(states=tensor((rows, 24, 128, 128)),
                      conv_prefixes=[[tensor(shape) for shape in shapes[1:]] for token in range(rows)])
        copies, slices, freed = [], [], []
        def sliced(source, start, end, **kwargs):
            value = tensor(shapes[0])
            slices.append((source, start, end, value))
            return value
        operations = SimpleNamespace(
            get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2,
            copy=lambda source, destination: copies.append((source, destination)),
            slice=sliced, deallocate=freed.append, DRAM_MEMORY_CONFIG='dram')
        return operations, result, entry, destinations, copies, slices, freed

    def test_restore_zero_uses_entry_without_slice(self):
        operations, result, entry, destinations, copies, slices, freed = self.restore_fixture()
        restore_prefix(operations, result, entry, destinations, 0)
        self.assertEqual([(id(source), id(destination)) for source, destination in copies],
                         [(id(source), id(destination)) for source, destination in zip(entry, destinations)])
        self.assertEqual(slices, [])
        self.assertEqual(freed, [])

    def test_restore_every_nonzero_prefix_uses_matching_snapshots(self):
        for accepted in (1, 2):
            operations, result, entry, destinations, copies, slices, freed = self.restore_fixture()
            restore_prefix(operations, result, entry, destinations, accepted)
            self.assertEqual(slices[0][1:3], ((accepted - 1, 0, 0, 0), (accepted, 24, 128, 128)))
            self.assertIs(copies[0][0], slices[0][3])
            self.assertEqual([id(source) for source, destination in copies[1:]],
                             [id(value) for value in result['conv_prefixes'][accepted - 1]])
            self.assertIs(freed[0], slices[0][3])

    def test_restore_rejects_invalid_prefix_without_writes(self):
        for accepted in (-1, 3, True, 0.5):
            operations, result, entry, destinations, copies, slices, freed = self.restore_fixture()
            with self.assertRaises(ValueError):
                restore_prefix(operations, result, entry, destinations, accepted)
            self.assertEqual(copies, [])

    def test_restore_refuses_snapshot_alias_and_duplicate_destinations(self):
        for alias in ('entry', 'prefix', 'duplicate'):
            operations, result, entry, destinations, copies, slices, freed = self.restore_fixture()
            destinations[1] = entry[1] if alias == 'entry' else result['conv_prefixes'][0][0] if alias == 'prefix' else destinations[2]
            with self.assertRaises(ValueError):
                restore_prefix(operations, result, entry, destinations, 1)
            self.assertEqual(copies, [])

    def test_restore_all_supported_widths(self):
        for rows in (1, 2, 4, 8, 16):
            for accepted in range(rows + 1):
                operations, result, entry, destinations, copies, slices, freed = self.restore_fixture(rows)
                restore_prefix(operations, result, entry, destinations, accepted)
                self.assertEqual(len(copies), 5)

    def test_restore_keeps_aliased_t1_snapshot_alive(self):
        operations, result, entry, destinations, copies, slices, freed = self.restore_fixture(1)
        operations.slice = lambda *args, **kwargs: result['states']
        restore_prefix(operations, result, entry, destinations, 1)
        self.assertIs(copies[0][0], result['states'])
        self.assertEqual(freed, [])

    def test_restore_rejects_incomplete_convolution_prefix_before_writes(self):
        operations, result, entry, destinations, copies, slices, freed = self.restore_fixture()
        result['conv_prefixes'][0].pop()
        with self.assertRaises(ValueError):
            restore_prefix(operations, result, entry, destinations, 1)
        self.assertEqual(copies, [])

    def test_restore_rejects_alias_on_only_one_chip(self):
        operations, result, entry, destinations, copies, slices, freed = self.restore_fixture()
        operations.get_device_tensors = lambda value: [
            SimpleNamespace(buffer_address=lambda: id(entry[1]) if value is destinations[1] else id(value)),
            SimpleNamespace(buffer_address=lambda: id(value))]
        with self.assertRaises(ValueError):
            restore_prefix(operations, result, entry, destinations, 1)
        self.assertEqual(copies, [])
