import ast
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from model_batch import ModelBatch, compact_gdn_enabled, device_loop_enabled, instance_overrides, validate_checkpoint


class ModelBatchTests(unittest.TestCase):
    def test_raw_vocabulary_shards_require_explicit_forward_opt_in(self):
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = None
        fixture.gdn_calls = 0
        fixture.norm_batch_calls = 0
        fixture.norm_batch = False
        fixture.working_states, fixture.writers, fixture.readers, fixture.bindings = [], [], [], []
        fixture.compact_gdn = False
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        self.assertEqual(fixture.run(), 'logits')
        self.assertEqual(fixture.model._forward_decode.call_args.kwargs, {})
        self.assertEqual(fixture.run(sharded_logits=True), 'logits')
        self.assertEqual(fixture.model._forward_decode.call_args.kwargs, {'sharded_lm_head': True})
        fixture.norm_batch = True
        with self.assertRaisesRegex(AssertionError, 'all48 GDN'):
            fixture.run()

    def test_full_prefix_propagates_norm_option_to_every_model_fixture(self):
        tree = ast.parse(Path(__file__).with_name('full-prefix.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'ModelBatch']
        self.assertEqual(len(calls), 2)
        for call in calls:
            options = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
            self.assertEqual(options.get('norm_batch'), 'options.norm_batch')
        timing = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == 'measure'
                  and any(keyword.arg == 'packed_checkpoints' for keyword in node.keywords)]
        self.assertEqual(len(timing), 1)
        self.assertIn('norm_batch', [keyword.arg for keyword in timing[0].keywords])

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
