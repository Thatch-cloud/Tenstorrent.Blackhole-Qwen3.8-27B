"""Host-only prepared lifecycle tests; no TTNN import, device or simulator use."""

from collections import defaultdict
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_vsplit as split
from gdn_vsplit_prepared import PreparedCleanupError, PreparedVSplit


ROOT = Path(os.environ.get('GDN_VSPLIT_SOURCE_ROOT', str(
    Path(__file__).resolve().parents[2] / 'hardware-evidence.local/34009341359'
    / 'qwen-hardware-inventory-34009341359/gdn-source')))


class FakeTensor:
    def __init__(self, shape, address, dtype='bf16', layout='tile', memory='dram'):
        self.shape = shape
        padded = list(shape)
        if layout == 'tile':
            padded[-2:] = [((dimension + 31) // 32) * 32 for dimension in shape[-2:]]
        self.padded_shape = tuple(padded)
        self.address = address
        self.dtype = dtype
        self.layout = layout
        self.memory = memory
        self.allocated = True
        self.contents = 0

    def buffer_address(self):
        if not self.allocated:
            raise RuntimeError('Tensor was externally freed')
        return self.address

    def memory_config(self):
        return self.memory


def tensor(shape, address, dtype='bf16', layout='tile', memory='dram'):
    result = FakeTensor(shape, address, dtype, layout, memory)
    result.shards = [FakeTensor(shape, address + chip * 100000, dtype, layout, memory)
                     for chip in range(2)]
    return result


class FakeBackend:
    bfloat16 = 'bf16'
    float32 = 'fp32'
    TILE_LAYOUT = 'tile'
    ROW_MAJOR_LAYOUT = 'row_major'
    DRAM_MEMORY_CONFIG = 'dram'
    L1_MEMORY_CONFIG = 'l1'
    DataMovementProcessor = SimpleNamespace(RISCV_0=0, RISCV_1=1)
    NOC = SimpleNamespace(RISCV_0_default=0, RISCV_1_default=1)
    MathFidelity = SimpleNamespace(HiFi4='HiFi4')

    class KernelDescriptor(SimpleNamespace):
        SourceType = SimpleNamespace(SOURCE_CODE='source')

    CBDescriptor = CBFormatDescriptor = ProgramDescriptor = SimpleNamespace
    ComputeConfigDescriptor = DataMovementConfigDescriptor = SimpleNamespace
    MeshProgramDescriptor = dict

    @staticmethod
    def RuntimeArgs():
        return defaultdict(dict)

    @staticmethod
    def CoreCoord(*point):
        return point

    CoreRange = MeshCoordinate = MeshCoordinateRange = CoreCoord
    CoreRangeSet = TileDescriptor = Tile = staticmethod(lambda value: value)

    @staticmethod
    def TensorAccessorArgs(value):
        return SimpleNamespace(get_compile_time_args=lambda: [value.address, value.memory])

    def __init__(self):
        self.events = []
        self.allocations = []
        self.empty_calls = 0
        self.enqueue_calls = 0
        self.fail_allocation = None
        self.fail_enqueue = None
        self.fail_fence = False
        self.fail_release = set()
        self.on_enqueue = None
        self.on_allocate = None
        self.on_fence = None
        self.released = []

    @staticmethod
    def get_device_tensors(value):
        return value.shards

    def empty(self, shape, *, device, dtype, layout, memory_config):
        self.empty_calls += 1
        if self.empty_calls == self.fail_allocation:
            raise RuntimeError(f'Allocation {self.empty_calls} failed')
        result = tensor(shape, 1000 + self.empty_calls * 100, dtype, layout, memory_config)
        self.allocations.append(result)
        self.events.append(('allocate', result))
        if self.on_allocate:
            self.on_allocate(result)
        return result

    def generic_op(self, tensors, program):
        self.enqueue_calls += 1
        self.events.append(('enqueue', program, tensors, 0))
        if self.on_enqueue:
            self.on_enqueue(self.enqueue_calls)
        if self.enqueue_calls == self.fail_enqueue:
            raise RuntimeError(f'Enqueue {self.enqueue_calls} failed')

    def synchronize_device(self, mesh):
        self.events.append(('fence', mesh))
        if self.on_fence:
            self.on_fence()
        if self.fail_fence:
            raise RuntimeError('Fence failed')

    def deallocate(self, value):
        self.events.append(('release', value))
        if id(value) in self.fail_release:
            raise RuntimeError(f'Release {value.address} failed')
        if not value.allocated:
            raise AssertionError('Duplicate deallocation')
        value.allocated = False
        for shard in value.shards:
            shard.allocated = False
        self.released.append(value)


class PreparedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = split.load_kernels(ROOT)

    def setUp(self):
        self.backend = FakeBackend()
        self.mesh = SimpleNamespace(shape=(1, 2),
            compute_with_storage_grid_size=lambda: SimpleNamespace(x=12, y=10))
        self.inputs = [tensor(shape, 100 + index) for index, shape in enumerate(
            ((1, 32, 5120), (1, 32, 24), (1, 32, 24), (1, 24, 128, 128),
             (1, 32, 3072), (1, 1, 128)))]
        runtime_patch = patch.object(split, 'validate_runtime')
        self.runtime = runtime_patch.start()
        self.addCleanup(runtime_patch.stop)

    def prepare(self, **kwargs):
        return PreparedVSplit(self.mesh, *self.inputs[:4], z=self.inputs[4], norm_w=self.inputs[5],
                              root=ROOT, operations=self.backend, **dict(experimental=True, **kwargs))

    def assert_inputs_live(self):
        self.assertTrue(all(value.allocated and all(shard.allocated for shard in value.shards)
                            for value in self.inputs))
        self.assertFalse(any(value is original for value in self.backend.released for original in self.inputs))

    def test_prepares_once_then_two_ordered_default_cq_ops_per_run(self):
        with patch.object(split, 'load_kernels', wraps=split.load_kernels) as load, \
                patch.object(split, 'build_program', wraps=split.build_program) as build, \
                patch.object(split, 'stage_spec', wraps=split.stage_spec) as spec:
            prepared = self.prepare()
            self.assertEqual(load.call_count, 1)
            self.assertEqual(build.call_count, 2)
            self.assertEqual([call[0][4] for call in build.call_args_list], ['recurrence', 'norm_gate'])
            spec_count = spec.call_count
            self.assertEqual(self.backend.empty_calls, 3)
            self.assertEqual([event[0] for event in self.backend.events], ['allocate'] * 3)
            outputs = prepared.run()
            for repeat in range(4):
                self.assertIs(prepared.run(), outputs)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(build.call_count, 2)
            self.assertEqual(spec.call_count, spec_count)
            self.assertEqual(self.backend.empty_calls, 3)
            self.assertEqual(self.runtime.call_count, 1)
            enqueues = self.backend.events[3:]
            self.assertEqual([event[0] for event in enqueues], ['enqueue'] * 10)
            for index, event in enumerate(enqueues):
                self.assertIs(event[1], enqueues[index % 2][1])
                self.assertIs(event[2], enqueues[0][2])
                self.assertEqual(event[3], 0)
            self.assertEqual(prepared.rows, 32)
            self.assertIs(outputs[0], prepared.output)
            self.assertIs(outputs[1], prepared.states)
            self.assertIs(outputs[2], prepared.pre_norm)
            self.assertIs(prepared.pre_norm, prepared.bridge)
            self.assert_inputs_live()
            prepared.close()

    def test_kernel_strings_and_two_chip_descriptors_are_unchanged(self):
        prepared = self.prepare()
        prepared.run()
        for event, stage in zip(self.backend.events[3:], ('recurrence', 'norm_gate')):
            self.assertEqual(len(event[1]), 2)
            for descriptor in event[1].values():
                for kernel, role in zip(descriptor.kernels, ('reader', 'writer', 'compute')):
                    self.assertEqual(kernel.kernel_source, self.kernels[stage][role])
                    self.assertEqual(len(kernel.core_ranges), split.stage_spec(stage, 32)['workers'])
        prepared.close()

    def test_all_supported_rows_and_default_shapes(self):
        for rows in (1, 2, 4, 8, 16, 32):
            with self.subTest(rows=rows):
                self.inputs = [tensor(shape, 100 + index) for index, shape in enumerate(
                    ((1, rows, 5120), (1, rows, 24), (1, rows, 24), (1, 24, 128, 128),
                     (1, rows, 3072), (1, 1, 128)))]
                prepared = self.prepare()
                output, states, bridge = prepared.run()
                self.assertEqual(prepared.rows, rows)
                self.assertEqual((output.shape, output.dtype, output.layout, output.memory),
                                 ((1, rows, 3072), 'bf16', 'tile', 'dram'))
                self.assertEqual((states.shape, states.dtype, states.layout, states.memory),
                                 ((rows, 24, 128, 128), 'bf16', 'tile', 'dram'))
                self.assertEqual((bridge.shape, bridge.dtype, bridge.layout, bridge.memory),
                                 ((rows, 1, 96, 32), 'fp32', 'row_major', 'dram'))
                prepared.close()

    def test_l1_changes_only_final_output_and_its_accessor(self):
        prepared = self.prepare(output_memory='l1')
        prepared.run()
        self.assertEqual([value.memory for value in self.backend.allocations], ['dram', 'dram', 'l1'])
        self.assertEqual(prepared.output.memory, 'l1')
        norm_program = self.backend.events[-1][1]
        for chip, descriptor in enumerate(norm_program.values()):
            writer = descriptor.kernels[1]
            address = prepared.output.shards[chip].address
            self.assertEqual(writer.compile_time_args[-4:], [address, 'l1', address, 'l1'])
            for role, kernel in zip(('reader', 'writer', 'compute'), descriptor.kernels):
                self.assertEqual(kernel.kernel_source, self.kernels['norm_gate'][role])
        prepared.close()

    def test_explicit_true_required_before_hashes_or_allocation(self):
        for experimental in (False, None, 1, 'yes'):
            with self.assertRaisesRegex(ValueError, 'explicit experimental=True'):
                PreparedVSplit(self.mesh, *self.inputs[:4], z=self.inputs[4], norm_w=self.inputs[5],
                               operations=self.backend, experimental=experimental)
        self.runtime.assert_not_called()
        self.assertEqual(self.backend.events, [])

    def test_runtime_root_is_independent_of_source_root(self):
        with patch.dict(os.environ, {'TT_METAL_HOME': '/active/runtime'}):
            prepared = self.prepare()
        self.runtime.assert_called_once_with(Path('/active/runtime'))
        prepared.close()

    def test_hash_errors_stop_before_allocation(self):
        original = ValueError('Runtime hash changed')
        self.runtime.side_effect = original
        with self.assertRaises(ValueError) as caught:
            self.prepare()
        self.assertIs(caught.exception, original)
        self.runtime.side_effect = None
        with patch.object(split, 'load_kernels', side_effect=ValueError('Native kernel hash changed')):
            with self.assertRaisesRegex(ValueError, 'Native kernel hash changed'):
                self.prepare()
        self.assertEqual(self.backend.events, [])

    def test_input_geometry_dtype_layout_memory_and_output_policy(self):
        for index, attribute, value in (
            (0, 'shape', (1, 33, 5120)), (1, 'shape', (1, 32, 96)),
            (2, 'dtype', 'fp32'), (3, 'layout', 'row_major'), (3, 'memory', 'l1'),
            (4, 'shape', (1, 32, 96)), (5, 'shape', (1, 1, 32)),
        ):
            with self.subTest(index=index, attribute=attribute):
                original = getattr(self.inputs[index], attribute)
                setattr(self.inputs[index], attribute, value)
                with self.assertRaises(ValueError):
                    self.prepare()
                setattr(self.inputs[index], attribute, original)
        with self.assertRaisesRegex(ValueError, 'interleaved DRAM or L1'):
            self.prepare(output_memory='sharded')
        self.assertEqual(self.backend.events, [])

    def test_mesh_and_both_chip_input_contract_before_allocation(self):
        self.mesh.shape = (2, 1)
        with self.assertRaisesRegex(ValueError, '1x2 mesh'):
            self.prepare()
        self.mesh.shape = (1, 2)
        original_grid = self.mesh.compute_with_storage_grid_size
        self.mesh.compute_with_storage_grid_size = lambda: SimpleNamespace(x=8, y=8)
        with self.assertRaisesRegex(ValueError, '96 worker cores'):
            self.prepare()
        self.mesh.compute_with_storage_grid_size = original_grid
        original_shards = self.inputs[0].shards
        self.inputs[0].shards = original_shards[:1]
        with self.assertRaisesRegex(ValueError, 'Both chips'):
            self.prepare()
        self.inputs[0].shards = original_shards
        self.inputs[1].shards[1].address = self.inputs[0].shards[1].address
        with self.assertRaisesRegex(ValueError, 'aliased fixed address on chip 1'):
            self.prepare()
        self.assertEqual(self.backend.events, [])

    def test_addresses_on_both_chips_are_fixed_and_failure_is_terminal(self):
        for chip in (0, 1):
            with self.subTest(chip=chip):
                prepared = self.prepare()
                address = self.inputs[0].shards[chip].address
                self.inputs[0].shards[chip].address += 10000
                count = self.backend.enqueue_calls
                with self.assertRaisesRegex(ValueError, 'address/signature changed'):
                    prepared.run()
                self.assertTrue(prepared.poisoned)
                self.assertEqual(self.backend.enqueue_calls, count)
                self.inputs[0].shards[chip].address = address
                with self.assertRaisesRegex(RuntimeError, 'poisoned'):
                    prepared.run()
                prepared.close()

    def test_each_output_address_is_checked_before_enqueue(self):
        for output_index in range(3):
            prepared = self.prepare()
            selected = prepared._outputs()[output_index]
            original = selected.shards[1].address
            selected.shards[1].address += 9000
            count = self.backend.enqueue_calls
            with self.assertRaisesRegex(ValueError, 'address/signature changed'):
                prepared.run()
            self.assertEqual(self.backend.enqueue_calls, count)
            selected.shards[1].address = original
            prepared.close()

    def test_shape_padding_dtype_layout_memory_and_shard_count_drift(self):
        for attribute, value in (('shape', (1, 16, 5120)), ('padded_shape', (1, 64, 5120)),
                                 ('dtype', 'fp32'), ('layout', 'row_major'), ('memory', 'l1')):
            with self.subTest(attribute=attribute):
                prepared = self.prepare()
                original = getattr(self.inputs[0], attribute)
                setattr(self.inputs[0], attribute, value)
                count = self.backend.enqueue_calls
                with self.assertRaises(ValueError):
                    prepared.run()
                self.assertEqual(self.backend.enqueue_calls, count)
                setattr(self.inputs[0], attribute, original)
                prepared.close()
        prepared = self.prepare()
        shard = self.inputs[0].shards.pop()
        with self.assertRaisesRegex(ValueError, 'Both chips'):
            prepared.run()
        self.inputs[0].shards.append(shard)
        prepared.close()

    def test_local_shard_metadata_and_mesh_drift(self):
        prepared = self.prepare()
        self.inputs[2].shards[1].dtype = 'fp32'
        with self.assertRaisesRegex(ValueError, 'signature differs on chip 1'):
            prepared.run()
        self.inputs[2].shards[1].dtype = 'bf16'
        prepared.close()
        prepared = self.prepare()
        self.mesh.shape = (2, 1)
        with self.assertRaisesRegex(ValueError, 'mesh signature changed'):
            prepared.run()
        self.mesh.shape = (1, 2)
        prepared.close()

    def test_input_contents_can_change_without_repreparation(self):
        prepared = self.prepare()
        first = prepared.run()
        self.inputs[0].contents = 123
        self.inputs[3].contents = 456
        self.assertIs(prepared.run(), first)
        self.assertFalse(prepared.poisoned)
        prepared.close()

    def test_external_free_is_rejected_before_enqueue(self):
        prepared = self.prepare()
        self.inputs[0].shards[1].allocated = False
        with self.assertRaisesRegex(RuntimeError, 'externally freed'):
            prepared.run()
        self.assertEqual(self.backend.enqueue_calls, 0)
        self.inputs[0].shards[1].allocated = True
        prepared.close()

    def test_validation_repeats_before_second_enqueue(self):
        prepared = self.prepare()
        address = prepared.bridge.shards[1].address
        self.backend.on_enqueue = lambda count: setattr(prepared.bridge.shards[1], 'address', address + 999)
        with self.assertRaisesRegex(ValueError, 'address/signature changed'):
            prepared.run()
        self.assertEqual(self.backend.enqueue_calls, 1)
        self.assertTrue(prepared.poisoned)
        self.assertEqual([event[0] for event in self.backend.events], ['allocate'] * 3 + ['enqueue'])
        prepared.bridge.shards[1].address = address
        prepared.close()

    def test_first_or_second_enqueue_failure_poison_without_fence_or_free(self):
        for failure in (1, 2):
            self.backend = FakeBackend()
            prepared = self.prepare()
            self.backend.fail_enqueue = failure
            with self.assertRaisesRegex(RuntimeError, f'Enqueue {failure} failed'):
                prepared.run()
            self.assertTrue(prepared.poisoned)
            self.assertEqual(self.backend.enqueue_calls, failure)
            self.assertEqual([event[0] for event in self.backend.events], ['allocate'] * 3 + ['enqueue'] * failure)
            with self.assertRaisesRegex(RuntimeError, 'poisoned'):
                prepared.run()
            prepared.close()
            self.assert_inputs_live()

    def test_reentry_poison_even_if_callback_swallows_error(self):
        for action in ('run', 'close'):
            for swallowed in (False, True):
                with self.subTest(action=action, swallowed=swallowed):
                    self.backend = FakeBackend()
                    prepared = self.prepare()

                    def reenter(count):
                        if swallowed:
                            with self.assertRaisesRegex(RuntimeError, 're-entry'):
                                getattr(prepared, action)()
                        else:
                            getattr(prepared, action)()

                    self.backend.on_enqueue = reenter
                    with self.assertRaisesRegex(RuntimeError, 'poisoned'):
                        prepared.run()
                    self.assertEqual(self.backend.enqueue_calls, 1)
                    self.assertTrue(prepared.poisoned)
                    self.assertFalse(any(event[0] in ('fence', 'release') for event in self.backend.events))
                    prepared.close()

    def test_partial_allocation_failures_release_every_successful_allocation(self):
        for failure in (1, 2, 3):
            self.backend = FakeBackend()
            self.backend.fail_allocation = failure
            with self.assertRaisesRegex(RuntimeError, f'Allocation {failure} failed') as caught:
                self.prepare()
            self.assertEqual(caught.exception.cleanup_errors, ())
            self.assertEqual(self.backend.released, list(reversed(self.backend.allocations)))
            self.assertEqual(len(self.backend.released), failure - 1)
            self.assertFalse(any(event[0] in ('fence', 'enqueue') for event in self.backend.events))
            self.assert_inputs_live()

    def test_descriptor_failure_releases_outputs_and_preserves_original_error(self):
        original = RuntimeError('Second program build failed')
        build = split.build_program

        def fail_second(*args):
            if args[4] == 'norm_gate':
                raise original
            return build(*args)

        with patch.object(split, 'build_program', side_effect=fail_second):
            with self.assertRaises(RuntimeError) as caught:
                self.prepare()
        self.assertIs(caught.exception, original)
        self.assertEqual(self.backend.released, list(reversed(self.backend.allocations)))
        self.assertEqual(len(self.backend.released), 3)
        self.assert_inputs_live()

    def test_constructor_cleanup_failures_are_reported_without_masking_and_retryable(self):
        self.backend.fail_allocation = 3
        self.backend.on_allocate = lambda value: self.backend.fail_release.add(id(value))
        with self.assertRaisesRegex(RuntimeError, 'Allocation 3 failed') as caught:
            self.prepare()
        error = caught.exception
        self.assertEqual(len(error.cleanup_errors), 2)
        if hasattr(error, 'add_note'):
            self.assertEqual(len(error.__notes__), 2)
        else:
            self.assertIsInstance(error.__cause__, PreparedCleanupError)
            self.assertEqual(error.__cause__.errors, error.cleanup_errors)
        prepared = error.prepared_operation
        self.assertTrue(prepared.poisoned)
        self.assertFalse(prepared.closed)
        self.assertEqual(len(self.backend.released), 0)
        self.backend.fail_release.clear()
        prepared.close()
        self.assertTrue(prepared.closed)
        self.assertEqual(len(self.backend.released), 2)
        self.assert_inputs_live()

    def test_bad_allocated_placement_is_cleaned_up(self):
        self.backend.on_allocate = lambda value: setattr(value, 'memory', 'l1')
        with self.assertRaisesRegex(ValueError, 'Allocated output signature'):
            self.prepare()
        self.assertEqual(self.backend.released, self.backend.allocations)
        self.assertEqual(len(self.backend.released), 1)
        self.assert_inputs_live()

    def test_allocator_returning_input_is_never_deallocated(self):
        with patch.object(self.backend, 'empty', return_value=self.inputs[0]):
            with self.assertRaisesRegex(ValueError, 'caller-owned input'):
                self.prepare()
        self.assertEqual(self.backend.released, [])
        self.assert_inputs_live()

    def test_close_fences_then_releases_only_owned_outputs_and_is_idempotent(self):
        prepared = self.prepare()
        prepared.run()
        owned = tuple(self.backend.allocations)
        count = len(self.backend.events)
        prepared.close()
        self.assertEqual([event[0] for event in self.backend.events[count:]], ['fence'] + ['release'] * 3)
        self.assertEqual(self.backend.released, list(reversed(owned)))
        self.assertTrue(prepared.closed)
        count = len(self.backend.events)
        prepared.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            prepared.run()
        self.assertEqual(len(self.backend.events), count)
        self.assert_inputs_live()

    def test_close_fence_failure_retains_all_outputs_for_retry(self):
        prepared = self.prepare()
        prepared.run()
        self.backend.fail_fence = True
        with self.assertRaisesRegex(RuntimeError, 'Fence failed'):
            prepared.close()
        self.assertFalse(prepared.closed)
        self.assertTrue(prepared.poisoned)
        self.assertEqual(self.backend.released, [])
        self.assertTrue(all(value.allocated for value in self.backend.allocations))
        with self.assertRaisesRegex(RuntimeError, 'poisoned'):
            prepared.run()
        self.backend.fail_fence = False
        prepared.close()
        self.assertEqual(len(self.backend.released), 3)

    def test_close_attempts_all_releases_and_retries_only_failed_handles(self):
        prepared = self.prepare()
        failed = prepared.states
        self.backend.fail_release.add(id(failed))
        with self.assertRaises(PreparedCleanupError) as caught:
            prepared.close()
        self.assertEqual(len(caught.exception.errors), 1)
        self.assertIs(caught.exception.__cause__, caught.exception.errors[0])
        self.assertEqual(len(self.backend.released), 2)
        self.assertFalse(prepared.closed)
        self.assertTrue(prepared.poisoned)
        self.backend.fail_release.clear()
        prepared.close()
        self.assertEqual(len(self.backend.released), 3)
        self.assertIs(self.backend.released[-1], failed)
        self.assert_inputs_live()


if __name__ == '__main__':
    unittest.main()
