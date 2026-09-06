import unittest
from types import SimpleNamespace

from model_batch import compact_gdn_enabled, instance_overrides, validate_checkpoint


class ModelBatchTests(unittest.TestCase):
    def test_compact_path_never_changes_t1_or_default(self):
        self.assertFalse(compact_gdn_enabled(1, True, True, None))
        for rows in (1, 2, 4, 8, 16):
            self.assertFalse(compact_gdn_enabled(rows, False, False, None))
        for rows in (2, 4, 8, 16):
            self.assertTrue(compact_gdn_enabled(rows, True, True, None))

    def test_compact_requires_exact_attention_policy_without_profiling(self):
        for serial_sdpa, profiler in ((False, None), (True, object())):
            with self.assertRaises(ValueError):
                compact_gdn_enabled(16, True, serial_sdpa, profiler)

    def test_prefix_bounds(self):
        for rows in (1, 2, 4, 8, 16):
            for prefix in range(rows + 1):
                validate_checkpoint(rows, prefix)
        for rows, prefix in ((3, 0), (16, -1), (16, 17), (1, True), (True, 0)):
            with self.assertRaises(ValueError):
                validate_checkpoint(rows, prefix)

    def test_existing_instance_attribute_restored_on_failure(self):
        instance = SimpleNamespace(forward="native")
        with self.assertRaisesRegex(RuntimeError, "device"):
            with instance_overrides([(instance, "forward", "candidate")]):
                self.assertEqual(instance.forward, "candidate")
                raise RuntimeError("device")
        self.assertEqual(instance.forward, "native")

    def test_class_attribute_is_never_modified(self):
        layer_type = type("Layer", (), {"forward": "native"})
        first, second = layer_type(), layer_type()
        with instance_overrides([(first, "forward", "candidate")]):
            self.assertEqual(first.forward, "candidate")
            self.assertEqual(second.forward, "native")
        self.assertNotIn("forward", first.__dict__)
        self.assertEqual(first.forward, "native")


if __name__ == "__main__":
    unittest.main()
