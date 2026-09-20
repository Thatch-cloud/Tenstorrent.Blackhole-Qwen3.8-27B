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
                         validate_checkpoint)


class ModelBatchTests(unittest.TestCase):
    def test_shared_replay_reader_must_engage_for_all_sixteen_layers(self):
        fixture = ModelBatch.__new__(ModelBatch)
        fixture.retained = None
        fixture.gdn_calls = fixture.norm_batch_calls = 0
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
        fixture.norm_batch_calls = 0
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
