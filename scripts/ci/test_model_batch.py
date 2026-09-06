import unittest
from types import SimpleNamespace

from model_batch import compact_gdn_enabled, device_loop_enabled, instance_overrides, validate_checkpoint


class ModelBatchTests(unittest.TestCase):
    def test_t32_static_fixture_has_all_prefixes_and_device_loop(self):
        for prefix in range(33):
            validate_checkpoint(32, prefix)
        self.assertTrue(device_loop_enabled(32, True, True, True, True, True))
        self.assertTrue(compact_gdn_enabled(32, True, True, None))

    def test_packed_checkpoint_experiment_probes_every_multirow_width(self):
        for rows in (1, 2, 4, 8, 16):
            self.assertEqual(device_loop_enabled(rows, True, True, True, True, True), rows > 1)
            self.assertFalse(device_loop_enabled(rows, False, True, True, True, True))

    def test_device_loop_retains_native_t1_and_default(self):
        self.assertFalse(device_loop_enabled(1, True, True, True))
        for rows in (1, 2, 4, 8, 16):
            self.assertFalse(device_loop_enabled(rows, False, False, False))
        for rows in (2, 4, 8, 16):
            self.assertTrue(device_loop_enabled(rows, True, True, True))

    def test_device_loop_requires_exact_previous_control(self):
        for compact, layout in ((False, False), (True, False), (False, True)):
            with self.assertRaises(ValueError):
                device_loop_enabled(4, True, compact, layout)

    def test_compact_prologue_uses_previous_control_for_short_blocks(self):
        for rows in (1, 2, 4):
            self.assertFalse(device_loop_enabled(rows, True, True, True, compact_prologue=True))
        for rows in (8, 16):
            self.assertTrue(device_loop_enabled(rows, True, True, True, compact_prologue=True))

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
