import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from attention_batch import OrderedCacheWriter, Overlay, SerialAttentionReader, SerialCacheWriter, capture_operation, serial_tail


class AttentionBatchTests(unittest.TestCase):
    def test_capture_closes_and_releases_on_operation_failure(self):
        events = []
        operations = SimpleNamespace(begin_trace_capture=Mock(return_value=17),
            end_trace_capture=Mock(side_effect=lambda *args, **kwargs: events.append('end')),
            release_trace=Mock(side_effect=lambda *args, **kwargs: events.append('release')))
        with self.assertRaisesRegex(RuntimeError, 'cold binary'):
            capture_operation(operations, 'mesh', Mock(side_effect=RuntimeError('cold binary')))
        self.assertEqual(events, ['end', 'release'])

    def test_capture_keeps_successful_trace_owned_by_caller(self):
        operations = SimpleNamespace(begin_trace_capture=Mock(return_value=17), end_trace_capture=Mock(), release_trace=Mock())
        self.assertEqual(capture_operation(operations, 'mesh', lambda: 'output'), (17, 'output'))
        operations.end_trace_capture.assert_called_once_with('mesh', 17, cq_id=0)
        operations.release_trace.assert_not_called()

    def test_ordered_writer_owns_only_its_conversion(self):
        for memory in ('DRAM', 'sharded'):
            operations = SimpleNamespace(DRAM_MEMORY_CONFIG='DRAM', to_memory_config=Mock(), deallocate=Mock())
            writer = OrderedCacheWriter('mesh', operations, 'kernels')
            packed = SimpleNamespace(shape=(1, 2, 32, 256), memory_config=lambda: memory)
            cache, positions, pages = (SimpleNamespace(shape=shape) for shape in ((8, 2, 64, 256), (2,), (2, 4)))
            with patch('ordered_cache.update') as update:
                writer(cache, packed, update_idxs_tensor=positions, page_table=pages)
                self.assertEqual(writer.calls, 1)
                self.assertIs(update.call_args.args[2], operations.to_memory_config.return_value if memory != 'DRAM' else packed)
            if memory == 'DRAM':
                operations.deallocate.assert_not_called()
            else:
                operations.deallocate.assert_called_once_with(operations.to_memory_config.return_value)

    def test_ordered_writer_cleans_conversion_on_failure(self):
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='DRAM', to_memory_config=Mock(), deallocate=Mock())
        writer = OrderedCacheWriter('mesh', operations, 'kernels')
        packed = SimpleNamespace(shape=(1, 2, 32, 256), memory_config=lambda: 'sharded')
        with patch('ordered_cache.update', side_effect=RuntimeError('failed')), self.assertRaises(RuntimeError):
            writer(SimpleNamespace(shape=(8, 2, 64, 256)), packed,
                update_idxs_tensor=SimpleNamespace(shape=(2,)), page_table=SimpleNamespace(shape=(2, 4)))
        operations.deallocate.assert_called_once_with(operations.to_memory_config.return_value)
        self.assertEqual(writer.calls, 0)

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
