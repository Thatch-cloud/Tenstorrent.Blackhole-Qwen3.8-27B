import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from attention_batch import Overlay, SerialAttentionReader, SerialCacheWriter, serial_tail


class AttentionBatchTests(unittest.TestCase):
    def test_sdpa_uses_b1_queries_and_matching_positions(self):
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG="DRAM", slice=Mock(), concat=Mock(), deallocate=Mock(),
            transformer=SimpleNamespace(paged_scaled_dot_product_attention_decode=Mock()))
        reader = SerialAttentionReader(operations, [63, 64, 65, 66], ["pages"] * 4)
        reader(SimpleNamespace(shape=(1, 4, 12, 256)), "keys", "values",
               page_table_tensor="packed", cur_pos_tensor="packed", memory_config="L1", scale=0.0625)
        calls = operations.transformer.paged_scaled_dot_product_attention_decode.call_args_list
        self.assertEqual([call.kwargs["cur_pos_tensor"] for call in calls], [63, 64, 65, 66])
        self.assertTrue(all(call.kwargs["page_table_tensor"] == "pages" for call in calls))
        self.assertTrue(all(call.kwargs["scale"] == 0.0625 for call in calls))
        self.assertEqual([call.args[2] for call in operations.slice.call_args_list],
                         [(1, index + 1, 12, 256) for index in range(4)])
        operations.concat.assert_called_once()
        self.assertEqual(reader.calls, 1)

    def test_sdpa_rejects_wrong_batch_geometry(self):
        reader = SerialAttentionReader(None, [63], ["pages"])
        with self.assertRaises(ValueError):
            reader(SimpleNamespace(shape=(1, 2, 12, 256)), None, None,
                   page_table_tensor=None, cur_pos_tensor=None)

    def fixture(self, rows=4):
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG="DRAM", to_memory_config=Mock(), slice=Mock(),
                                     deallocate=Mock(), experimental=SimpleNamespace(paged_update_cache=Mock()))
        writer = SerialCacheWriter(operations, list(range(rows)), [f"page-{index}" for index in range(rows)], "B1")
        return operations, writer

    def test_writes_ordered_singleton_positions_and_pages(self):
        operations, writer = self.fixture()
        writer("cache", SimpleNamespace(shape=(1, 4, 32, 256)), update_idxs_tensor="packed", page_table="packed")
        calls = operations.experimental.paged_update_cache.call_args_list
        self.assertEqual([call.kwargs["update_idxs_tensor"] for call in calls], [0, 1, 2, 3])
        self.assertEqual([call.kwargs["page_table"] for call in calls], [f"page-{index}" for index in range(4)])
        self.assertEqual([call.args[1][1] for call in operations.slice.call_args_list], [0, 1, 2, 3])
        self.assertEqual(writer.calls, 1)

    def test_rejects_wrong_geometry_before_dispatch(self):
        operations, writer = self.fixture()
        for shape in ((1, 4, 2, 256), (1, 2, 32, 256), (1, 4, 32, 128)):
            with self.assertRaises(ValueError):
                writer(None, SimpleNamespace(shape=shape), update_idxs_tensor=None, page_table=None)
        operations.to_memory_config.assert_not_called()

    def test_rejects_unpaired_or_unsupported_rows(self):
        for positions, pages in (([], []), ([1, 2], [1]), ([1, 2, 3], [1, 2, 3])):
            with self.assertRaises(ValueError):
                SerialCacheWriter(None, positions, pages, None)

    def test_overlay_leaves_native_operations_unchanged(self):
        original = SimpleNamespace(first=1, second=2)
        overlay = Overlay(original, first=3)
        self.assertEqual((overlay.first, overlay.second, original.first), (3, 2, 1))

    def test_local_tail_does_not_patch_class_or_globals(self):
        namespace = {"ttnn": SimpleNamespace(experimental=SimpleNamespace(paged_update_cache=lambda: "native"))}
        exec("def tail(self):\n    return ttnn.experimental.paged_update_cache()", namespace)
        layer_type = type("Layer", (), {"_decode_from_prep": namespace["tail"]})
        layer = layer_type()
        candidate = serial_tail(layer, lambda: "serial", namespace["ttnn"])
        self.assertEqual(candidate(), "serial")
        self.assertEqual(layer._decode_from_prep(), "native")
        self.assertEqual(namespace["ttnn"].experimental.paged_update_cache(), "native")


if __name__ == "__main__":
    unittest.main()
