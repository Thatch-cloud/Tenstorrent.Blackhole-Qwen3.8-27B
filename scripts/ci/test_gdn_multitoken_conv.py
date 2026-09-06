from types import SimpleNamespace
import unittest

from gdn_multitoken_conv import addresses, release_owned, validate_projected


class ConvIntegrationTests(unittest.TestCase):
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
