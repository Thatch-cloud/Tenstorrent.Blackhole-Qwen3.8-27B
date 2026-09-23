import ast
from contextlib import nullcontext
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import sys
from unittest.mock import patch

import torch

from model_batch import (ModelBatch, compact_gdn_enabled, device_loop_enabled, instance_overrides, prepare_inputs,
                         validate_checkpoint, validate_pack)


class ModelBatchTests(unittest.TestCase):
    def test_shared_replay_reader_must_engage_for_all_sixteen_layers(self):
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = None
        fixture.gdn_calls = fixture.norm_batch_calls = fixture.user_batched_calls = 0
        fixture.norm_batch = fixture.compact_gdn = False
        fixture.attention_replay = True
        fixture.attention_mask_once = False
        fixture.working_states, fixture.writers, fixture.bindings = [], [], []
        reader = SimpleNamespace(calls=0)
        fixture.readers = [reader] * 16
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            reader.calls += 16
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        self.assertEqual(fixture.run(), 'logits')
        fixture.attention_mask_once = True
        fixture.replay_reader = SimpleNamespace(refresh_calls=0, metadata=['mask'])

        def shared_masks(expected):
            fixture.replay_reader.refresh_calls += 1
            return nullcontext()

        fixture.replay_reader.shared_masks = Mock(side_effect=shared_masks)
        self.assertEqual(fixture.run(), 'logits')
        fixture.replay_reader.shared_masks.assert_called_once_with(16)
        fixture.model._forward_decode = Mock(side_effect=lambda *args, **kwargs: setattr(fixture, 'gdn_calls', fixture.gdn_calls + 48))
        with self.assertRaisesRegex(AssertionError, 'Every selected'):
            fixture.run()
        fixture.model._forward_decode = Mock(side_effect=forward)
        fixture.replay_reader.shared_masks = Mock(return_value=nullcontext())
        with self.assertRaisesRegex(AssertionError, 'exactly once per model forward'):
            fixture.run()

    def test_raw_vocabulary_shards_require_explicit_forward_opt_in(self):
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = None
        fixture.gdn_calls = 0
        fixture.norm_batch_calls = fixture.user_batched_calls = 0
        fixture.norm_batch = False
        fixture.attention_mask_once = False
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
        # 64 is the M3 packed block, four T16 users; 48 and 128 are no block.
        for rows in (1, 2, 4, 8, 16, 32, 64):
            for prefix in range(rows + 1):
                validate_checkpoint(rows, prefix)
        for rows, prefix in ((3, 0), (16, -1), (16, 17), (1, True), (True, 0), (48, 0), (128, 0), (64, 65)):
            with self.assertRaises(ValueError):
                validate_checkpoint(rows, prefix)
        self.assertTrue(compact_gdn_enabled(64, True, True, None))
        self.assertTrue(device_loop_enabled(64, True, True, True, compact_prologue=True, packed_checkpoints=True))

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

    def native_m3_fixture(self, retired_expected=0, active_expected=1):
        """A ModelBatch with two fake two_tile binders - one active, one retired the way
        Lever N M3native retires MLP/GDN-output - and nothing else two_tile touches."""
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = None
        fixture.gdn_calls = fixture.norm_batch_calls = fixture.user_batched_calls = 0
        fixture.norm_batch = fixture.compact_gdn = fixture.attention_replay = fixture.attention_mask_once = False
        fixture.working_states, fixture.writers, fixture.readers, fixture.bindings = [], [], [], []
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)
        fixture.native_m3 = True

        class FakeBinder:
            def __init__(self, label, expected_calls):
                self.label, self.expected_calls, self.calls = label, expected_calls, 0

        active = FakeBinder('full-attention forward', active_expected)
        retired = FakeBinder('MLP forward', retired_expected)
        fixture.two_tile = [active, retired]
        return fixture, active, retired

    def test_native_m3_binder_calls_are_reported_every_round(self):
        fixture, active, retired = self.native_m3_fixture()

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        with patch('dflash_device.pindiag') as marker:
            self.assertEqual(fixture.run(), 'logits')
        marker.assert_called_once()
        template, payload = marker.call_args.args
        self.assertIn('native_m3 binder calls this round', template)
        self.assertEqual(payload, {'full-attention forward': 1, 'MLP forward': 0})

    def test_a_call_leaking_through_a_retired_binder_fails_loudly(self):
        """The exact silent-fallback-to-two-call failure mode native_m3's overlay switch
        exists to catch: something re-binds the retired MLP wrapper and it engages."""
        fixture, active, retired = self.native_m3_fixture()

        def leaking_forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            retired.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=leaking_forward))
        with self.assertRaisesRegex(AssertionError,
                                    'native_m3 retired the MLP forward two-tile wrapper.*1 unexpected call'):
            fixture.run()

    def test_without_native_m3_the_original_two_tile_message_is_unchanged(self):
        fixture, active, retired = self.native_m3_fixture(active_expected=1)
        fixture.native_m3 = False
        retired.expected_calls = 1  # a genuine (non-retired) binder that must engage

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        with self.assertRaisesRegex(AssertionError,
                                    'Every MLP forward of the wide block must take its two-tile form: 0 engaged, 1 expected'):
            fixture.run()

    def test_without_native_m3_no_diagnostic_is_printed(self):
        fixture, active, retired = self.native_m3_fixture(retired_expected=0)
        fixture.native_m3 = False

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        with patch('dflash_device.pindiag') as marker:
            self.assertEqual(fixture.run(), 'logits')
        marker.assert_not_called()

    def native_attn_fixture(self, prep_expected=0, concat_expected=0, active_expected=1):
        """A ModelBatch with three fake two_tile binders - the full-attention forward
        (always required) and QWEN_FAST_NATIVE_ATTN's two retired-wrapper guards
        (TwoTileNativeAttnGuard: sliced attn_decode_prep, two-tile head concat) - and
        nothing else two_tile touches. Mirrors native_m3_fixture; native_attn is its own
        independent switch (a RUNTIME/ENV fact, never a model attribute)."""
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = None
        fixture.gdn_calls = fixture.norm_batch_calls = fixture.user_batched_calls = 0
        fixture.norm_batch = fixture.compact_gdn = fixture.attention_replay = fixture.attention_mask_once = False
        fixture.working_states, fixture.writers, fixture.readers, fixture.bindings = [], [], [], []
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)
        fixture.native_m3 = False
        fixture.native_attn = True

        class FakeBinder:
            def __init__(self, label, expected_calls):
                self.label, self.expected_calls, self.calls = label, expected_calls, 0

        active = FakeBinder('full-attention forward', active_expected)
        prep_guard = FakeBinder('sliced attn_decode_prep', prep_expected)
        concat_guard = FakeBinder('two-tile head concat', concat_expected)
        fixture.two_tile = [active, prep_guard, concat_guard]
        return fixture, active, prep_guard, concat_guard

    def test_native_attn_binder_calls_are_reported_every_round(self):
        fixture, active, prep_guard, concat_guard = self.native_attn_fixture()

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        with patch('dflash_device.pindiag') as marker:
            self.assertEqual(fixture.run(), 'logits')
        marker.assert_called_once()
        template, payload = marker.call_args.args
        self.assertIn('native_m3 binder calls this round', template)
        self.assertEqual(payload, {'full-attention forward': 1, 'sliced attn_decode_prep': 0,
                                   'two-tile head concat': 0})

    def test_a_call_leaking_through_the_prep_guard_fails_loudly(self):
        """The exact silent-fallback-to-the-sliced-prep failure mode QWEN_FAST_NATIVE_ATTN
        exists to catch: something still runs the legacy per-tile attn_decode_prep path."""
        fixture, active, prep_guard, concat_guard = self.native_attn_fixture()

        def leaking_forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            prep_guard.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=leaking_forward))
        with self.assertRaisesRegex(AssertionError,
                                    'retired the sliced attn_decode_prep two-tile wrapper.*1 unexpected call'):
            fixture.run()

    def test_a_call_leaking_through_the_concat_guard_fails_loudly(self):
        """Same failure mode, for the retired two-tile head concat wrapper."""
        fixture, active, prep_guard, concat_guard = self.native_attn_fixture()

        def leaking_forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            concat_guard.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=leaking_forward))
        with self.assertRaisesRegex(AssertionError,
                                    'retired the two-tile head concat two-tile wrapper.*1 unexpected call'):
            fixture.run()

    def test_native_attn_alone_without_native_m3_also_prints_the_diagnostic(self):
        """native_attn is independent of native_m3: either one engaged is enough for
        run()'s per-round [PINDIAG] binder-calls line to fire."""
        fixture, active, prep_guard, concat_guard = self.native_attn_fixture()
        self.assertFalse(fixture.native_m3)
        self.assertTrue(fixture.native_attn)

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        with patch('dflash_device.pindiag') as marker:
            self.assertEqual(fixture.run(), 'logits')
        marker.assert_called_once()

    def test_without_native_attn_a_genuine_binder_message_is_unchanged(self):
        """With native_attn off, a guard configured with a nonzero expectation (the
        shape a genuine, non-retired two-tile binder takes) reports the ordinary
        engaged/expected message, not the retired-wrapper leak message."""
        fixture, active, prep_guard, concat_guard = self.native_attn_fixture(active_expected=1)
        fixture.native_attn = False
        prep_guard.expected_calls = 1  # a genuine (non-retired) binder that must engage

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            active.calls += 1
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        with self.assertRaisesRegex(
                AssertionError,
                'Every sliced attn_decode_prep of the wide block must take its two-tile form: 0 engaged, 1 expected'):
            fixture.run()


if __name__ == "__main__":
    unittest.main()


class FakeShard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class FakeTTNN:
    """Device tensors carry their values, so staging into pooled buffers can be checked exactly."""

    uint32, int32, bfloat16 = 'uint32', 'int32', 'bf16'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'row_major', 'tile', 'dram'

    def __init__(self):
        self.next = [0x1000, 0x900000]
        self.live, self.hosts, self.copies, self.host_copies, self.deallocated = [], [], [], [], []
        self.synchronized = 0

    def allocate(self, shape, dtype, layout, value=None):
        shards = []
        for chip in range(2):
            shards.append(FakeShard(self.next[chip]))
            self.next[chip] += 0x100
        tensor = SimpleNamespace(shape=tuple(shape), dtype=dtype, layout=layout, value=value, shards=shards)
        self.live.append(tensor)
        return tensor

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        if device is None:
            host = SimpleNamespace(shape=tuple(value.shape), dtype=dtype, layout=layout, value=value.clone())
            self.hosts.append(host)
            return host
        return self.allocate(value.shape, dtype, layout, value.clone())

    def copy_host_to_device_tensor(self, host, destination):
        destination.value = host.value.clone()
        self.host_copies.append((host, destination))

    def copy(self, source, destination):
        destination.value = source.value.clone()
        self.copies.append((source, destination))

    def get_device_tensors(self, tensor):
        return tensor.shards

    def deallocate(self, tensor):
        if any(tensor is value for value in self.deallocated):
            raise AssertionError('Double free')
        self.deallocated.append(tensor)

    def synchronize_device(self, mesh):
        self.synchronized += 1

    def ReplicateTensorToMesh(self, mesh):
        return ('replicate', mesh)


def fake_rope(ttnn):
    def rot_mats_decode(mesh, dim, max_seq_len, theta, positions):
        rows = len(positions)
        table = positions.to(torch.float32).reshape(1, rows, 1, 1).repeat(1, 1, 1, dim)
        return (ttnn.allocate((1, rows, 1, dim), 'bf16', 'tile', table.cos().bfloat16()),
                ttnn.allocate((1, rows, 1, dim), 'bf16', 'tile', table.sin().bfloat16()))
    return rot_mats_decode


def pooled_batch(ttnn, rows, page_width=68):
    """A pool bucket's fixture inputs (serving_buffer_pool.BucketSlot.batch), zeroed."""
    def integers(shape, dtype='int32'):
        return ttnn.allocate(shape, dtype, 'row_major', torch.zeros(shape, dtype=torch.int32))

    return SimpleNamespace(tokens=integers((rows, 1), 'uint32'), positions=integers((rows,)),
        pages=integers((rows, page_width)), singleton_pages=integers((1, page_width)),
        singleton_positions=[integers((1,)) for row in range(rows)],
        cos=ttnn.allocate((1, rows, 1, 64), 'bf16', 'tile', torch.zeros(1, rows, 1, 64)),
        sin=ttnn.allocate((1, rows, 1, 64), 'bf16', 'tile', torch.zeros(1, rows, 1, 64)))


class FixtureInputTests(unittest.TestCase):
    """The fixture's inputs, uploaded and owned (as always) or staged into pooled buffers
    allocated before any request's trace, and then borrowed: same values, pooled addresses."""

    def setUp(self):
        self.ttnn = FakeTTNN()
        self.model = SimpleNamespace(mesh_device='mesh', args=SimpleNamespace(rope_head_dim=64, max_seq_len=65536, rope_theta=1e6))
        self.rope = patch.dict(sys.modules, {'models.demos.blackhole.qwen36.tt.attention.rope_tp':
                                             SimpleNamespace(rot_mats_decode=fake_rope(self.ttnn))})
        self.rope.start()
        self.addCleanup(self.rope.stop)

    def inputs(self, rows, start=4096, storage=None, packed=False):
        tokens = [3 + index for index in range(rows)]
        positions = torch.arange(start, start + rows, dtype=torch.int32)
        pages = torch.arange(68, dtype=torch.int32).reshape(1, 68)
        result = prepare_inputs(self.ttnn, self.model, rows, tokens, positions, pages.repeat(rows, 1), pages,
                                storage=storage, packed=packed)
        return result, tokens, positions, pages

    def test_unpooled_inputs_are_uploaded_in_the_fixture_order_and_owned(self):
        result, tokens, positions, pages = self.inputs(4)
        self.assertEqual(result.owned, self.ttnn.live)
        self.assertEqual(result.owned, [result.tokens, result.positions, result.pages, result.singleton_pages,
                                        *result.singleton_positions, result.cos, result.sin])
        self.assertEqual(result.borrowed, [])
        self.assertEqual(result.row_tables, [result.singleton_pages] * 4)
        self.assertEqual((result.tokens.shape, result.tokens.dtype, result.tokens.layout), ((4, 1), 'uint32', 'row_major'))
        self.assertEqual((result.positions.shape, result.positions.dtype), ((4,), 'int32'))
        self.assertEqual((result.pages.shape, result.singleton_pages.shape), ((4, 68), (1, 68)))
        self.assertTrue(torch.equal(result.tokens.value, torch.tensor(tokens, dtype=torch.int32).reshape(4, 1)))
        self.assertTrue(torch.equal(result.positions.value, positions))
        self.assertTrue(torch.equal(result.pages.value, pages.repeat(4, 1)))
        self.assertEqual([value.value.item() for value in result.singleton_positions], [4096, 4097, 4098, 4099])
        self.assertEqual((result.cos.shape, result.cos.dtype, result.cos.layout), ((1, 4, 1, 64), 'bf16', 'tile'))
        # Nothing staged, copied, freed or fenced: the upload path is untouched.
        self.assertEqual((self.ttnn.hosts, self.ttnn.copies, self.ttnn.host_copies, self.ttnn.deallocated, self.ttnn.synchronized),
                         ([], [], [], [], 0))

    def test_pooled_inputs_are_staged_into_the_lent_buffers_with_the_same_values_and_nothing_is_kept(self):
        storage = pooled_batch(self.ttnn, 4)
        pooled = list(self.ttnn.live)
        result, tokens, positions, pages = self.inputs(4, storage=storage)
        self.assertEqual(result.owned, [])
        self.assertEqual(result.borrowed, [storage.tokens, storage.positions, storage.pages, storage.singleton_pages,
                                           *storage.singleton_positions, storage.cos, storage.sin])
        for name in ('tokens', 'positions', 'pages', 'singleton_pages', 'cos', 'sin'):
            self.assertIs(getattr(result, name), getattr(storage, name))
        self.assertEqual(result.singleton_positions, storage.singleton_positions)
        self.assertEqual(result.row_tables, [storage.singleton_pages] * 4)
        # The values the upload path would have put on the device, now in the pooled buffers.
        self.assertTrue(torch.equal(storage.tokens.value, torch.tensor(tokens, dtype=torch.int32).reshape(4, 1)))
        self.assertTrue(torch.equal(storage.positions.value, positions))
        self.assertTrue(torch.equal(storage.pages.value, pages.repeat(4, 1)))
        self.assertTrue(torch.equal(storage.singleton_pages.value, pages))
        self.assertEqual([value.value.item() for value in storage.singleton_positions], [4096, 4097, 4098, 4099])
        native_cos, native_sin = self.ttnn.live[-2:]
        self.assertTrue(torch.equal(storage.cos.value, native_cos.value))
        self.assertTrue(torch.equal(storage.sin.value, native_sin.value))
        self.assertEqual(self.ttnn.copies, [(native_cos, storage.cos), (native_sin, storage.sin)])
        # Host staging, as before every verify; the native tables freed; one fence; no new device buffer survives.
        self.assertEqual([destination for host, destination in self.ttnn.host_copies], result.borrowed[:-2])
        self.assertEqual(self.ttnn.deallocated, [native_cos, native_sin])
        self.assertEqual(self.ttnn.synchronized, 1)
        self.assertEqual([value for value in self.ttnn.live if not any(value is kept for kept in pooled)],
                         [native_cos, native_sin])
        self.assertEqual(result.staged, [])

    def test_pooled_geometry_is_checked_before_anything_is_staged(self):
        cases = {}
        cases['rows'] = pooled_batch(self.ttnn, 2)
        cases['narrow pages'] = pooled_batch(self.ttnn, 4, page_width=67)
        cases['wide pages'] = pooled_batch(self.ttnn, 4, page_width=69)
        wrong = pooled_batch(self.ttnn, 4)
        wrong.tokens = self.ttnn.allocate((4, 1), 'int32', 'row_major')
        cases['token dtype'] = wrong
        wrong = pooled_batch(self.ttnn, 4)
        wrong.positions = self.ttnn.allocate((4,), 'int32', 'tile')
        cases['position layout'] = wrong
        wrong = pooled_batch(self.ttnn, 4)
        del wrong.singleton_positions[-1]
        cases['singletons'] = wrong
        wrong = pooled_batch(self.ttnn, 4)
        wrong.cos = self.ttnn.allocate((1, 2, 1, 64), 'bf16', 'tile')
        cases['rotary rows'] = wrong
        wrong = pooled_batch(self.ttnn, 4)
        del wrong.sin
        cases['missing sin'] = wrong
        for name, storage in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.inputs(4, storage=storage)
        with self.assertRaisesRegex(ValueError, 'packed block'):
            self.inputs(4, storage=pooled_batch(self.ttnn, 4), packed=True)
        self.assertEqual((self.ttnn.hosts, self.ttnn.host_copies, self.ttnn.copies), ([], [], []))

    def test_a_rotary_table_the_pool_cannot_take_is_refused_and_the_native_tables_are_freed(self):
        storage = pooled_batch(self.ttnn, 4)
        storage.cos = self.ttnn.allocate((1, 4, 1, 32), 'bf16', 'tile')
        with self.assertRaisesRegex(ValueError, 'rotary table'):
            self.inputs(4, storage=storage)
        self.assertEqual(len(self.ttnn.deallocated), 2)
        self.assertEqual([value.shape for value in self.ttnn.deallocated], [(1, 4, 1, 64)] * 2)

    def test_close_frees_the_owned_inputs_and_not_the_borrowed_ones(self):
        for borrowed in ([], ['lent']):
            fixture = ModelBatch.__new__(ModelBatch)
            fixture.retained = None
            fixture.working_states, fixture.grouped_readers = [], []
            fixture.operations = SimpleNamespace(deallocate=Mock())
            fixture.buffers, fixture.borrowed = ['owned'], list(borrowed)
            fixture.close()
            fixture.operations.deallocate.assert_called_once_with('owned')
            self.assertEqual((fixture.buffers, fixture.borrowed), ([], []))


class ReplayStorageTests(unittest.TestCase):
    """A pooled fixture takes its capture family's replay page tables from the slot and
    refuses a family the pool does not hold, before any upload; unpooled it uploads."""

    def test_the_fixture_takes_its_familys_tables_and_refuses_a_family_the_pool_lacks(self):
        from model_batch import replay_storage

        self.assertIsNone(replay_storage(None, 4352))
        tables = {4096: ['a'], 4352: ['b', 'c']}
        storage = SimpleNamespace(replay_pages=tables)
        self.assertEqual(replay_storage(storage, 4352), ['b', 'c'])
        self.assertIsNot(replay_storage(storage, 4352), tables[4352])
        with self.assertRaisesRegex(ValueError, r'families \[4096, 4352\]; this capture is in family 4608'):
            replay_storage(storage, 4608)
        # Pooled fixture inputs without the tables: the hole would stay open, so refused.
        with self.assertRaisesRegex(ValueError, 'families None'):
            replay_storage(SimpleNamespace(tokens=None), 4352)
        with self.assertRaisesRegex(ValueError, 'families None'):
            replay_storage(SimpleNamespace(replay_pages=['b']), 4352)

    def test_the_unpooled_call_is_the_pinned_reader_and_the_pooled_one_the_new_module(self):
        """attention_replay.py is frozen-recipe evidence (its bytes at 8c102b20), so the
        pooled reader lives in pooled_attention_replay.py and the unpooled construction
        stays the pinned call exactly as it was."""
        import inspect

        import model_batch

        source = inspect.getsource(model_batch.ModelBatch.__init__)
        self.assertIn('self.replay_reader = ReplayAttentionReader(ttnn, model.mesh_device, self.rows, self.replay_capacity, pages,\n'
                      '                    upload_replay, max_group_rows=self.replay_group_rows, short_context=self.short_context)',
                      source)
        self.assertIn('from pooled_attention_replay import PooledReplayAttentionReader', source)
        self.assertIn('storage=tables', source)
        # The pinned call, the pooled call and, packed, the per-user reader (M1b) over the
        # block's lent table sets - the only three constructions.
        self.assertEqual(source.count('ReplayAttentionReader('), 3)
        self.assertEqual(source.count('PackedReplayAttentionReader('), 1)
        self.assertIn('storage=packed_replay_pages', source)
        self.assertNotIn('storage', source[source.index('= ReplayAttentionReader('):source.index('else:', source.index('= ReplayAttentionReader('))],
                         'the pinned reader takes no storage keyword')
        packed = source[source.index('if self.pack is not None:', source.index('def upload_replay')):source.index('elif tables is None:')]
        # v116: the pinned (unpooled) reader - the warm-up fixture's - takes QWEN_FAST_SDPA_MODES
        # before any forward, as the pooled and packed readers do at construction.
        pinned = source[source.index('elif tables is None:'):source.index('else:', source.index('elif tables is None:'))]
        self.assertIn('apply_sdpa_modes(self.replay_reader, sdpa_modes())', pinned)
        self.assertLess(pinned.index('= ReplayAttentionReader('), pinned.index('apply_sdpa_modes('))
        self.assertIn("PackedReplayAttentionReader(ttnn, model.mesh_device, self.pack['segments']", packed)
        self.assertIn("self.pack['tables']", packed)
        self.assertIn('self.borrowed.extend(self.replay_reader.borrowed)', packed)

    def test_a_pack_carries_each_users_own_table_and_its_replay_family_validates_every_segment(self):
        """Packed, the block's family is the capture position's, as for one request; what
        must lie inside it is each user's segment at that user's own start."""
        from model_batch import packed_replay_family

        users = [dict(start=4100, rows=16, pages=torch.full((1, 68), 7, dtype=torch.int32), prefix=0,
                      checkpoints=['c'] * 48, slots=[['s'] * 5] * 48),
                 dict(start=4200, rows=16, pages=torch.full((1, 68), 11, dtype=torch.int32), prefix=0,
                      checkpoints=['c'] * 48, slots=[['s'] * 5] * 48)]
        pack = validate_pack(users)
        self.assertEqual(len(pack['tables']), 2)
        self.assertIs(pack['tables'][0], users[0]['pages'])
        self.assertIs(pack['tables'][1], users[1]['pages'])
        self.assertEqual(pack['segments'], ((0, 16), (16, 32)))
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(packed_replay_family(4096, pack), 4352)
            self.assertEqual(packed_replay_family(4300, pack), 4352)
            # A user whose segment leaves the family is refused, whichever segment it is.
            for start in (4090, 4340):
                users[1]['start'] = start
                with self.subTest(start=start), self.assertRaises(ValueError):
                    packed_replay_family(4096, validate_pack(users))
            users[1]['start'] = 4200
            with self.assertRaisesRegex(ValueError, 'long-context only'):
                packed_replay_family(4096, pack, short_context=True)

    def test_packed_replay_tables_need_a_packed_replay_fixture(self):
        """The block's lent table sets describe a packed replay fixture: without a pack, or
        without replay attention, they are refused at construction, before any upload."""
        options = dict(serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                       hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True, batch_conv=True,
                       packed_checkpoints=True, retain_records=True, ordered_cache=True, norm_batch=True,
                       commit_only_gdn=True)
        pack = [dict(start=4096 + 16 * index, rows=16, pages=torch.full((1, 68), index + 1, dtype=torch.int32), prefix=0,
                     checkpoints=['c'] * 48, slots=[['s%d' % index] * 5] * 48) for index in range(2)]
        with patch.dict(sys.modules, {'ttnn': SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError, 'packed fixture with replay attention'):
                ModelBatch(SimpleNamespace(), [1] * 32, 4096, torch.zeros(1, 68, dtype=torch.int32), [None] * 48, [None] * 48, 32,
                           packed_replay_pages=[['t'], ['t']], **options)
            with self.assertRaisesRegex(ValueError, 'packed fixture with replay attention'):
                ModelBatch(SimpleNamespace(), [1] * 32, 4096, torch.zeros(1, 68, dtype=torch.int32), [None] * 48, [None] * 48, 32,
                           pack=pack, packed_replay_pages=[['t'], ['t']], attention_replay=False, **options)

    def test_the_unpacked_row_tables_are_the_pooled_singleton(self):
        """Every row's table is the singleton page table, so pooling it pools them."""
        import ast
        import inspect

        import model_batch

        source = inspect.getsource(model_batch.prepare_inputs)
        tree = ast.parse('if True:\n' + source if source.startswith(' ') else source)
        assigned = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Attribute) and target.attr == 'row_tables' for target in node.targets)]
        self.assertEqual(len(assigned), 1)
        self.assertIsInstance(assigned[0].value, ast.IfExp)
        self.assertEqual(ast.unparse(assigned[0].value.orelse), '[result.singleton_pages] * rows')


class RowPagesAssignmentTests(unittest.TestCase):
    """The unpacked path must not read self.row_pages before assigning it.

    A blanket text replacement of `[singleton_pages] * self.rows` rewrote the
    right-hand side of the assignment that defines self.row_pages, so pack=None
    raised AttributeError at construction. The module is not overridden into the
    image, which is the only reason hardware never saw it.
    """

    def test_row_pages_is_defined_before_it_is_read(self):
        import ast
        import inspect

        import model_batch

        source = inspect.getsource(model_batch.ModelBatch.__init__)
        tree = ast.parse('if True:\n' + source if source.startswith(' ') else source)
        assigned = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Attribute) and target.attr == 'row_pages'
                            and isinstance(target.value, ast.Name) and target.value.id == 'self'):
                        assigned = node
                        break
        self.assertIsNotNone(assigned, 'self.row_pages must be assigned in __init__')
        reads = [node for node in ast.walk(assigned.value)
                 if isinstance(node, ast.Attribute) and node.attr == 'row_pages']
        self.assertEqual(reads, [], 'the assignment must not read self.row_pages')


class PackedFixtureTests(unittest.TestCase):
    """A packed retained fixture is decided per user after the verify readback.

    So ModelBatch builds commit-only device-loop states with one entry per user, runs
    every layer's decode deferred, and records each user's carry as that layer's
    commit destination. A packed fixture that retains records without commit-only GDN
    would decide eagerly with placeholder prefixes, so it is refused at construction.
    """

    def pack(self, users=2):
        return [dict(start=100 * (index + 1), rows=16, pages=torch.full((1, 4), index + 1, dtype=torch.int32), prefix=0,
                     checkpoints=['ck%d.%d' % (index, layer) for layer in range(48)],
                     slots=[['slot%d.%d.%d' % (index, layer, part) for part in range(5)] for layer in range(48)])
                for index in range(users)]

    def test_a_packed_retained_fixture_requires_commit_only_gdn(self):
        with patch.dict(sys.modules, {'ttnn': SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError, 'commit-only'):
                ModelBatch(SimpleNamespace(), [1] * 32, 0, torch.zeros(1, 4, dtype=torch.int32), [None] * 48, [None] * 48, 32,
                           serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                           hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True, batch_conv=True,
                           packed_checkpoints=True, retain_records=True, pack=self.pack())

    def fixture(self):
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.rows = 32
        fixture.pack = validate_pack(self.pack())
        fixture.device_loop_gdn = fixture.compact_prologue = fixture.batch_conv = fixture.packed_checkpoints = True
        fixture.commit_only_gdn = True
        fixture.norm_batch = fixture.prefix_zero_reuse = fixture.defer_conv_publication = False
        fixture.operations = SimpleNamespace(reshape=lambda value, shape: SimpleNamespace(shape=shape),
            get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2)
        fixture.working_states, fixture.gdn_calls, fixture.norm_batch_calls, fixture.user_batched_calls = [], 0, 0, 0
        fixture.retained = SimpleNamespace(append=Mock())
        return fixture

    def test_every_layer_decodes_deferred_with_one_entry_per_user_and_records_each_users_carry(self):
        fixture = self.fixture()
        layer = SimpleNamespace(B=8, _stable_state=True, rec_state='rec', conv_states=['c0', 'c1', 'c2', 'c3'])
        helper = SimpleNamespace(direct=True, gdn=layer, live=['rec', 'c0', 'c1', 'c2', 'c3'],
                                 allocate=Mock(side_effect=lambda: [object() for part in range(5)]))
        decoded = dict(commit_only_gdn=True, owned=[], layer_output='reduced')
        with patch.dict(sys.modules, {'models.tt_transformers.tt.ccl': SimpleNamespace(tt_all_reduce='reduce')}), \
                patch('gdn_multitoken.load_kernels', return_value='kernels'), \
                patch('gdn_device_loop_state.DeviceLoopState.decode', return_value=decoded) as decode, \
                patch('gdn_multitoken_conv.finish_output') as finish, \
                patch('gdn_records.retain_checkpoint_histories') as retain:
            forward = fixture.gdn_forward(layer, helper, 'ck3', 3)
            self.assertEqual(forward(SimpleNamespace(shape=(1, 1, 32, 5120))), 'reduced')
        (state,) = fixture.working_states
        self.assertTrue(state.commit_only)
        self.assertEqual((len(state.segment_entries), state.entry, state.state), (2, [], []))
        self.assertEqual(helper.allocate.call_count, 2, 'one block-start entry per user, before any trace')
        decode.assert_called_once()
        self.assertEqual(decode.call_args.args[0].shape, (1, 32, 5120))
        self.assertEqual(decode.call_args.args[1:], (['ck0.3', 'ck1.3'], [0, 0]))
        carries = (fixture.pack['slots'][0][3], fixture.pack['slots'][1][3])
        self.assertEqual(decode.call_args.kwargs, dict(segments=((0, 16), (16, 32)), slots=list(carries), deferred=True))
        finish.assert_called_once_with(layer, decoded, fixture.operations, 'reduce')
        retain.assert_called_once_with(fixture.operations, decoded, 'reduced')
        fixture.retained.append.assert_called_once_with(state, decoded, carries)
        self.assertEqual(fixture.gdn_calls, 1)

    def test_a_packed_block_checkpoints_once_per_user(self):
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = None
        fixture.rows = 32
        fixture.gdn_calls = fixture.norm_batch_calls = fixture.user_batched_calls = 0
        fixture.norm_batch = fixture.attention_mask_once = fixture.skip_row_clones = False
        fixture.compact_gdn = fixture.device_loop_gdn = True
        fixture.writers, fixture.readers, fixture.bindings = [], [], []
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)
        fixture.working_states = [SimpleNamespace(calls=0, checkpoint_calls=0, skipped_clones=0) for layer in range(48)]
        decisions = [1]

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            for state in fixture.working_states:
                state.calls += 1
                state.checkpoint_calls += decisions[0]
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        packed = dict(segments=((0, 16), (16, 32)))
        for fixture.pack, decisions[0] in ((None, 1), (packed, 2)):
            self.assertEqual(fixture.run(), 'logits')
        for fixture.pack, decisions[0] in ((None, 2), (packed, 1)):
            with self.assertRaisesRegex(AssertionError, 'once per user'):
                fixture.run()

    def test_the_user_batched_launch_counts_as_the_selected_row_parallel_norm(self):
        """QWEN_FAST_GDN_USER_BATCH=1 fuses every packed user's norm/gate into one launch per
        layer, reported as user_batched=True with norm_batch=False; with the row-parallel norm
        selected that satisfies the all-48-layers check exactly as the norm-batch adapter does,
        and a layer that reports neither still fails it."""
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = fixture.pack = None
        fixture.rows = 32
        fixture.gdn_calls = fixture.norm_batch_calls = fixture.user_batched_calls = 0
        fixture.norm_batch = True
        fixture.attention_mask_once = fixture.skip_row_clones = False
        fixture.compact_gdn = fixture.device_loop_gdn = True
        fixture.writers, fixture.readers, fixture.bindings = [], [], []
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)
        fixture.working_states = [SimpleNamespace(calls=0, checkpoint_calls=0, skipped_clones=0) for layer in range(48)]
        engaged = ['user_batched']

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            for state in fixture.working_states:
                state.calls += 1
                state.checkpoint_calls += 1
            if engaged[0] == 'user_batched':
                fixture.user_batched_calls += 48
            elif engaged[0] == 'norm_batch':
                fixture.norm_batch_calls += 48
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        for engaged[0] in ('user_batched', 'norm_batch'):
            self.assertEqual(fixture.run(), 'logits')
        engaged[0] = 'neither'
        with self.assertRaisesRegex(AssertionError, 'row-parallel norm must engage in all48'):
            fixture.run()


class UserBatchedMarkerTests(unittest.TestCase):
    """QWEN_FAST_GDN_USER_BATCH: one line per captured forward (run() executes its Python
    only when a forward is captured or run eagerly, never on a trace replay) saying how many
    of the 48 GDN layers took the user-batched launch."""

    TEMPLATE = '[PINDIAG] gdn user_batched calls this captured forward: {} of {} GDN layers'

    def run_forward(self, environ, batched):
        import os

        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = fixture.pack = None
        fixture.rows = 32
        fixture.gdn_calls = fixture.norm_batch_calls = fixture.user_batched_calls = 0
        fixture.norm_batch = True
        fixture.attention_mask_once = fixture.skip_row_clones = False
        fixture.compact_gdn = fixture.device_loop_gdn = True
        fixture.writers, fixture.readers, fixture.bindings = [], [], []
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)
        fixture.working_states = [SimpleNamespace(calls=0, checkpoint_calls=0, skipped_clones=0) for layer in range(48)]

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            for state in fixture.working_states:
                state.calls += 1
                state.checkpoint_calls += 1
            fixture.user_batched_calls += batched
            fixture.norm_batch_calls += 48 - batched
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        clean = {name: value for name, value in os.environ.items() if name != 'QWEN_FAST_GDN_USER_BATCH'}
        with patch.dict(os.environ, dict(clean, **environ), clear=True), patch('dflash_device.pindiag') as marker:
            self.assertEqual(fixture.run(), 'logits')
        return [call.args for call in marker.call_args_list if call.args[0] == self.TEMPLATE]

    def test_with_the_flag_every_captured_forward_reports_its_batched_layers(self):
        self.assertEqual(self.run_forward({'QWEN_FAST_GDN_USER_BATCH': '1'}, 48), [(self.TEMPLATE, 48, 48)])

    def test_with_the_flag_a_forward_that_batched_nothing_says_zero(self):
        self.assertEqual(self.run_forward({'QWEN_FAST_GDN_USER_BATCH': '1'}, 0), [(self.TEMPLATE, 0, 48)])

    def test_unset_and_unbatched_nothing_is_logged(self):
        self.assertEqual(self.run_forward({}, 0), [])
        self.assertEqual(self.run_forward({'QWEN_FAST_GDN_USER_BATCH': '0'}, 0), [])

    def test_a_batched_call_is_reported_even_without_the_flag(self):
        self.assertEqual(self.run_forward({}, 48), [(self.TEMPLATE, 48, 48)])

    def test_the_marker_formats_to_the_gate_string(self):
        self.assertEqual(self.TEMPLATE.format(48, 48),
                         '[PINDIAG] gdn user_batched calls this captured forward: 48 of 48 GDN layers')


class WideBlockTests(unittest.TestCase):
    """The 64-row M3 block (four T16 users) in the fixture: beyond one 32-row tile the
    pinned per-row serial adapters (attention_batch.py) stop, so the K/V write goes tile by
    tile through the ordered writer over the fixture's own cache tiles, the attention through
    the per-user replay readers, and the norm-batch decision is made at the segment's rows -
    or the block is refused before any upload."""

    def pack(self, users=4, rows=16):
        return [dict(start=4096 + 16 * index, rows=rows, pages=torch.full((1, 68), index + 1, dtype=torch.int32), prefix=0,
                     checkpoints=['c'] * 48, slots=[['s%d' % index] * 5] * 48) for index in range(users)]

    def test_a_block_beyond_one_tile_uploads_its_own_cache_tiles(self):
        ttnn = FakeTTNN()
        model = SimpleNamespace(mesh_device='mesh', args=SimpleNamespace(rope_head_dim=64, max_seq_len=65536, rope_theta=1e6))
        positions = torch.cat([torch.arange(4096 + 100 * user, 4096 + 100 * user + 16, dtype=torch.int32) for user in range(4)])
        pages = torch.arange(68, dtype=torch.int32).reshape(1, 68)
        page_rows = torch.cat([torch.full((16, 68), user + 1, dtype=torch.int32) for user in range(4)])
        with patch.dict(sys.modules, {'models.demos.blackhole.qwen36.tt.attention.rope_tp':
                                      SimpleNamespace(rot_mats_decode=fake_rope(ttnn))}):
            result = prepare_inputs(ttnn, model, 64, list(range(64)), positions, page_rows, pages, packed=True)
            narrow = prepare_inputs(ttnn, model, 32, list(range(32)), positions[:32], page_rows[:32], pages, packed=True)
        self.assertEqual([tile.rows for tile in result.cache_tiles], [(0, 32), (32, 64)])
        for tile, (first, last) in zip(result.cache_tiles, ((0, 32), (32, 64)), strict=True):
            self.assertEqual((tile.positions.shape, tile.positions.dtype, tile.positions.layout), ((32,), 'int32', 'row_major'))
            self.assertEqual((tile.pages.shape, tile.pages.dtype, tile.pages.layout), ((32, 68), 'int32', 'row_major'))
            self.assertTrue(torch.equal(tile.positions.value, positions[first:last]))
            self.assertTrue(torch.equal(tile.pages.value, page_rows[first:last]))
            # owned by the fixture, freed with it, never the pool's
            self.assertTrue(any(tile.positions is value for value in result.owned))
            self.assertTrue(any(tile.pages is value for value in result.owned))
        self.assertEqual(result.borrowed, [])
        self.assertEqual(len(result.row_tables), 64)
        self.assertEqual(narrow.cache_tiles, [], 'one tile needs no per-tile metadata')

    def test_the_writer_and_reader_follow_the_block_width(self):
        from model_batch import attention_reader, cache_writer
        from packed_cache_writer import SegmentedOrderedCacheWriter, tile

        ttnn = FakeTTNN()
        tiles = [tile((first, last), ttnn.allocate((32,), 'int32', 'row_major'), ttnn.allocate((32, 68), 'int32', 'row_major'))
                 for first, last in ((0, 32), (32, 64))]
        serial = Mock(return_value='serial-writer')
        with patch('model_batch.OrderedCacheWriter', return_value='ordered-writer') as ordered:
            self.assertEqual(cache_writer(ttnn, 'mesh', 'kernels', ordered_cache=True, cache_tiles=[], serial=serial), 'ordered-writer')
            ordered.assert_called_once_with('mesh', ttnn, 'kernels')
            segmented = cache_writer(ttnn, 'mesh', 'kernels', ordered_cache=True, cache_tiles=tiles, serial=serial)
            self.assertIsInstance(segmented, SegmentedOrderedCacheWriter)
            self.assertEqual((segmented.rows, segmented.page_width, [t.rows for t in segmented.tiles]), (64, 68, [(0, 32), (32, 64)]))
            serial.assert_not_called()
            self.assertEqual(cache_writer(ttnn, 'mesh', 'kernels', ordered_cache=False, cache_tiles=[], serial=serial), 'serial-writer')
            serial.assert_called_once_with()
        serial, grouped = Mock(return_value='serial-reader'), Mock(return_value='grouped-reader')
        self.assertEqual(attention_reader(replay_reader='replay', serial_sdpa=True, grouped_attention=False,
                                          serial=serial, grouped=grouped), 'replay')
        self.assertEqual(attention_reader(replay_reader=None, serial_sdpa=True, grouped_attention=True,
                                          serial=serial, grouped=grouped), 'grouped-reader')
        serial.assert_not_called()
        self.assertEqual(attention_reader(replay_reader=None, serial_sdpa=True, grouped_attention=False,
                                          serial=serial, grouped=grouped), 'serial-reader')
        self.assertIsNone(attention_reader(replay_reader=None, serial_sdpa=False, grouped_attention=False,
                                           serial=serial, grouped=grouped))
        self.assertEqual((serial.call_count, grouped.call_count), (1, 1))

    def test_a_wide_block_without_the_ordered_writer_and_replay_readers_is_refused_before_any_upload(self):
        options = dict(serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                       hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True, batch_conv=True,
                       packed_checkpoints=True, retain_records=True, norm_batch=True, commit_only_gdn=True)
        with patch.dict(sys.modules, {'ttnn': SimpleNamespace()}):
            for name, overrides in (('no ordered cache', dict(ordered_cache=False, attention_replay=False)),
                                    ('no replay attention', dict(ordered_cache=True, attention_replay=False))):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'wider than 32 rows'):
                    ModelBatch(SimpleNamespace(), [1] * 64, 4096, torch.zeros(1, 68, dtype=torch.int32), [None] * 48, [None] * 48, 64,
                               pack=self.pack(), **dict(options, **overrides))
            # the 32-row block takes the serial reader as before, refused only later, by its own checks
            with self.assertRaisesRegex(ValueError, 'commit-only'):
                ModelBatch(SimpleNamespace(), [1] * 32, 4096, torch.zeros(1, 68, dtype=torch.int32), [None] * 48, [None] * 48, 32,
                           pack=self.pack(users=2), **dict(options, ordered_cache=False, attention_replay=False, commit_only_gdn=False))

    def test_the_norm_batch_decision_is_made_at_the_segments_rows(self):
        from model_batch import recurrence_rows

        self.assertEqual(recurrence_rows(32, None, True), 32)
        self.assertEqual(recurrence_rows(64, validate_pack(self.pack()), True), 16)
        self.assertEqual(recurrence_rows(32, validate_pack(self.pack(users=2)), True), 16)
        self.assertEqual(recurrence_rows(32, validate_pack(self.pack(users=4, rows=8)), False), 8)
        # segments whose widths decide differently (4 rows: no norm batch; 16: yes) are refused
        mixed = validate_pack([*self.pack(users=1, rows=4), *self.pack(users=1, rows=4)[:0],
                               dict(start=5000, rows=4, pages=torch.full((1, 68), 9, dtype=torch.int32), prefix=0,
                                    checkpoints=['c'] * 48, slots=[['t'] * 5] * 48),
                               dict(start=6000, rows=8, pages=torch.full((1, 68), 8, dtype=torch.int32), prefix=0,
                                    checkpoints=['c'] * 48, slots=[['u'] * 5] * 48)])
        with self.assertRaisesRegex(ValueError, 'same norm-batch decision'):
            recurrence_rows(16, mixed, True)
        # unpacked, the block IS the recurrence, and 64 unpacked rows fail closed at the decision itself
        from gdn_batched_conv import norm_batch_enabled

        self.assertEqual(recurrence_rows(64, None, True), 64)
        with self.assertRaises(ValueError):
            norm_batch_enabled(recurrence_rows(64, None, True), True)
