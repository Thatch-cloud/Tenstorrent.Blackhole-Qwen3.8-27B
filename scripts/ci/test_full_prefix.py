import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location("full_prefix", Path(__file__).with_name("full-prefix.py"))
full_prefix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_prefix)


class FullPrefixTests(unittest.TestCase):
    def test_native_shape_requires_integer_indexing(self):
        class NativeShape:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                if type(index) is not int:
                    raise TypeError("Native Shape does not accept slices")
                return (8200, 2, 64, 256)[index]

        self.assertEqual(full_prefix.cache_geometry(NativeShape()), (2, 64, 256))

    def test_page_boundary_and_head_order(self):
        values = torch.arange(2 * 3 * 64 * 2).reshape(2, 3, 64, 2)
        for count in (63, 64, 65, 128):
            actual = full_prefix.logical_kv_prefix(values, count)
            for head in range(3):
                expected = torch.cat([values[0, head], values[1, head]])[:count]
                self.assertTrue(torch.equal(actual[head], expected))

    def test_future_page_data_is_excluded(self):
        values = torch.zeros(2, 1, 64, 2)
        before = full_prefix.logical_kv_prefix(values, 65).clone()
        values[1, :, 1:] = 42
        self.assertTrue(torch.equal(before, full_prefix.logical_kv_prefix(values, 65)))
        values[1, :, 0] = 42
        self.assertFalse(torch.equal(before, full_prefix.logical_kv_prefix(values, 65)))

    def test_rejects_invalid_layout_or_length(self):
        for count in (0, 129, True):
            with self.assertRaises(ValueError):
                full_prefix.logical_kv_prefix(torch.zeros(2, 1, 64, 2), count)
        with self.assertRaises(ValueError):
            full_prefix.logical_kv_prefix(torch.zeros(1, 1, 128, 2), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
