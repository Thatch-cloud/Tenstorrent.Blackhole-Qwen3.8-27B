"""The width-generic GDN state builders and the twins of the pinned helpers.

At the pair every builder is what it was (its own kernel file, no defines, two chip programs, the literal page counts)
and each twin agrees with the pinned function it stands in for. At QWEN_FAST_TP=4 the same builders read the four-card
widths from tp_shapes: 192 recurrent-state pages and 80 conv pages, a 2560-channel conv row, four chip programs, and the
sibling kernels whose page counts arrive as defines. The sibling kernels themselves are held to the pinned ones by text:
each is the pinned source with the pair's literals replaced by the defines, nothing else.
"""

from collections import defaultdict, namedtuple
import itertools
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_commit_dma
import gdn_commit_dma_tp
import gdn_conv_prefix_copy
import gdn_conv_windows
import gdn_multitoken_conv as pinned_conv
import gdn_multitoken_conv_tp
import gdn_records
import gdn_records_tp
import gdn_state_copy
import tp_addresses
import tp_kernels
import tp_shapes

HERE = Path(__file__).parent
Core = namedtuple('Core', 'x y')
PAIR_ENV = dict(os.environ)
PAIR_ENV.pop('QWEN_FAST_TP', None)


def four():
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def pair():
    return patch.dict(os.environ, {}, clear=True)


class FakeTensor:
    def __init__(self, shape, address, chips, accessor=('interleaved',)):
        self.shape, self.address, self.chips, self.accessor = tuple(shape), address, chips, accessor
        self.dtype, self.layout = 'bf16', 'tile'

    def memory_config(self):
        return 'dram'

    def shard(self, chip):
        return SimpleNamespace(buffer_address=lambda: self.address + chip, accessor=self.accessor)


def fake_ttnn(record, chips):
    def generic_op(tensors, program):
        record.append((list(tensors), program))

    addresses = itertools.count(70000, 16)
    return SimpleNamespace(
        deallocate=lambda value: None,
        bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1',
        get_device_tensors=lambda tensor: [tensor.shard(chip) for chip in range(chips)],
        empty=lambda shape, **keywords: FakeTensor(shape, next(addresses), chips),
        CoreRangeSet=list, CoreRange=lambda *values: values, CoreCoord=Core,
        CBDescriptor=SimpleNamespace, CBFormatDescriptor=SimpleNamespace,
        TileDescriptor=lambda value: value, Tile=list,
        MeshProgramDescriptor=dict, ProgramDescriptor=SimpleNamespace,
        MeshCoordinate=lambda *values: values, MeshCoordinateRange=lambda *values: values,
        KernelDescriptor=SimpleNamespace, DataMovementConfigDescriptor=SimpleNamespace,
        DataMovementProcessor=SimpleNamespace(RISCV_0='riscv0'), NOC=SimpleNamespace(RISCV_0_default=0),
        TensorAccessorArgs=lambda shard: SimpleNamespace(get_compile_time_args=lambda: list(shard.accessor)),
        RuntimeArgs=lambda: defaultdict(dict), generic_op=generic_op)


def run(chips, call, *arguments, **keywords):
    record = []
    with patch.dict(sys.modules, {'ttnn': fake_ttnn(record, chips)}):
        call(*arguments, **keywords)
    return record


def kernels_of(program):
    return {key: value.kernels[0] for key, value in program.items()}


def geometry_shapes():
    found = tp_shapes.active()
    return SimpleNamespace(
        found=found, compact=[(1, found.gdn_nv, 128, 128)] + [(1, 1, found.gdn_qkv)] * 4,
        full=[(8, found.gdn_nv, 128, 128)] + [(1, 8, found.gdn_qkv)] * 4)


def state(base, shapes, chips):
    return [FakeTensor(shape, base + 100 * index, chips) for index, shape in enumerate(shapes)]


class KernelSourceTests(unittest.TestCase):
    def test_the_pair_gets_its_own_file_and_no_defines(self):
        with pair():
            self.assertEqual(tp_kernels.source(HERE / 'gdn_commit_dma.cpp'), str(HERE / 'gdn_commit_dma.cpp'))
            self.assertEqual(tp_kernels.defines(), [])

    def test_four_cards_get_the_sibling_and_the_page_counts(self):
        with four():
            self.assertEqual(tp_kernels.source(HERE / 'gdn_commit_dma.cpp'), str(HERE / 'gdn_commit_dma_tp.cpp'))
            self.assertEqual(dict(tp_kernels.defines()),
                             {'QWEN_STATE_PAGES': '192', 'QWEN_CONV_PAGES': '80', 'QWEN_CONV_TASKS': '320'})

    def test_each_sibling_is_the_pinned_kernel_with_the_literals_replaced(self):
        cases = {
            'gdn_commit_dma': [('QWEN_STATE_PAGES', '384'), ('QWEN_CONV_TASKS', '640'), ('QWEN_CONV_PAGES', '160')],
            'gdn_conv_prefix_copy': [('QWEN_CONV_TASKS', '640'), ('QWEN_CONV_PAGES', '160')],
            'gdn_conv_windows': [('QWEN_CONV_PAGES', '160')],
        }
        for stem, macros in cases.items():
            with self.subTest(kernel=stem):
                original = (HERE / (stem + '.cpp')).read_text()
                sibling = (HERE / (stem + '_tp.cpp')).read_text()
                lines = sibling.split('\n')
                self.assertTrue(lines[0].startswith('// Four-card sibling of ' + stem))
                body = '\n'.join(lines[1:])
                for name, literal in macros:
                    self.assertIn('#ifndef %s\n#error' % name, body)
                    body = body.replace('#ifndef %s\n#error "%s is defined by the launch builder (tp_kernels.defines)"\n#endif\n'
                                        % (name, name), '')
                    body = body.replace(name, literal)
                self.assertEqual(body, original)
                for literal in ('384', '640', '160'):
                    stripped = re.sub(r'#error[^\n]*', '', sibling)
                    self.assertNotRegex(stripped, r'\b%s\b' % literal)

    def test_the_index_model_covers_every_page_once_at_both_widths(self):
        # gdn_commit_dma.cpp: worker w of 2 copies state pages w, w + 2, ... and conv tasks the same way
        for pages, conv_pages in ((384, 160), (192, 80)):
            state_seen, task_seen = [], []
            for worker in range(2):
                state_seen.extend(range(worker, pages, 2))
                task_seen.extend(range(worker, 4 * conv_pages, 2))
            self.assertEqual(sorted(state_seen), list(range(pages)))
            self.assertEqual(sorted(task_seen), list(range(4 * conv_pages)))
            slots = {task // conv_pages for task in task_seen}
            self.assertEqual(slots, {0, 1, 2, 3})
            self.assertEqual({task % conv_pages for task in task_seen}, set(range(conv_pages)))


class StateCopyTests(unittest.TestCase):
    def test_the_page_counts_follow_the_width(self):
        with pair():
            self.assertEqual(gdn_state_copy.page_counts([(1, 24, 128, 128)] + [(1, 1, 5120)] * 4),
                             [384, 160, 160, 160, 160])
            with self.assertRaises(ValueError):
                gdn_state_copy.page_counts([(1, 12, 128, 128)] + [(1, 1, 2560)] * 4)
        with four():
            self.assertEqual(gdn_state_copy.page_counts([(1, 12, 128, 128)] + [(1, 1, 2560)] * 4),
                             [192, 80, 80, 80, 80])
            with self.assertRaises(ValueError):
                gdn_state_copy.page_counts([(1, 24, 128, 128)] + [(1, 1, 5120)] * 4)

    def test_a_four_card_copy_launches_one_program_per_chip(self):
        with four():
            shapes = geometry_shapes()
            record = run(4, gdn_state_copy.copy_compact, state(1000, shapes.compact, 4), state(9000, shapes.compact, 4))
        tensors, program = record[0]
        self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)])
        for chip in range(4):
            values = kernels_of(program)[((0, chip), (0, chip))].runtime_args[0][0]
            self.assertEqual(values[:2], [1, 0])
            self.assertEqual([values[4 + 3 * slot] for slot in range(5)], [192, 80, 80, 80, 80])
            self.assertEqual(values[2], 1000 + chip)

    def test_two_chips_are_refused_at_four_cards_and_four_at_the_pair(self):
        with four():
            shapes = geometry_shapes()
            with self.assertRaisesRegex(ValueError, 'All 4 chips required'):
                run(2, gdn_state_copy.copy_compact, state(1000, shapes.compact, 2), state(9000, shapes.compact, 2))
        with pair():
            shapes = geometry_shapes()
            with self.assertRaisesRegex(ValueError, 'Both chips required'):
                run(4, gdn_state_copy.copy_compact, state(1000, shapes.compact, 4), state(9000, shapes.compact, 4))


class ConvLaunchTests(unittest.TestCase):
    def prefix_launch(self, chips):
        shapes = geometry_shapes()
        rows = 4
        windows = [FakeTensor((1, rows, shapes.found.gdn_qkv), 100 + 10 * slot, chips) for slot in range(4)]
        destinations = [FakeTensor((1, 1, shapes.found.gdn_qkv), 500 + 10 * slot, chips) for slot in range(4)]
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=6))
        return run(chips, gdn_conv_prefix_copy.copy_prefix, mesh, windows, destinations, 3)

    def test_the_pair_prefix_copy_is_the_original_kernel_with_no_defines(self):
        with pair():
            record = self.prefix_launch(2)
        program = record[0][1]
        self.assertEqual(len(program), 2)
        for kernel in kernels_of(program).values():
            self.assertEqual(kernel.kernel_source, str(HERE / 'gdn_conv_prefix_copy.cpp'))
            self.assertEqual(kernel.defines, [])

    def test_the_four_card_prefix_copy_uses_the_sibling_and_four_chips(self):
        with four():
            record = self.prefix_launch(4)
        program = record[0][1]
        self.assertEqual(len(program), 4)
        for kernel in kernels_of(program).values():
            self.assertEqual(kernel.kernel_source, str(HERE / 'gdn_conv_prefix_copy_tp.cpp'))
            self.assertEqual(dict(kernel.defines)['QWEN_CONV_PAGES'], '80')
            self.assertEqual(dict(kernel.defines)['QWEN_CONV_TASKS'], '320')

    def test_the_zero_tile_define_rides_beside_the_page_counts(self):
        with four():
            shapes = geometry_shapes()
            windows = [FakeTensor((1, 4, shapes.found.gdn_qkv), 100 + 10 * slot, 4) for slot in range(4)]
            destinations = [FakeTensor((1, 1, shapes.found.gdn_qkv), 500 + 10 * slot, 4) for slot in range(4)]
            mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=6))
            record = run(4, gdn_conv_prefix_copy.copy_prefix, mesh, windows, destinations, 3, reuse_zero_tile=True)
        names = [name for name, value in list(kernels_of(record[0][1]).values())[0].defines]
        self.assertEqual(names[0], 'QWEN_PREFIX_REUSE_ZERO_TILE')
        self.assertIn('QWEN_CONV_PAGES', names)

    def test_the_prefix_shapes_follow_the_width(self):
        with pair():
            self.assertEqual(gdn_conv_prefix_copy.validate_prefix([(1, 4, 5120)] * 4, [(1, 1, 5120)] * 4, 2), 4)
            with self.assertRaises(ValueError):
                gdn_conv_prefix_copy.validate_prefix([(1, 4, 2560)] * 4, [(1, 1, 2560)] * 4, 2)
        with four():
            self.assertEqual(gdn_conv_prefix_copy.validate_prefix([(1, 4, 2560)] * 4, [(1, 1, 2560)] * 4, 2), 4)
            with self.assertRaises(ValueError):
                gdn_conv_prefix_copy.validate_prefix([(1, 4, 5120)] * 4, [(1, 1, 5120)] * 4, 2)

    def window_launch(self, chips, columns):
        shapes = geometry_shapes()
        projected = FakeTensor((1, 4, columns), 10, chips)
        history = [FakeTensor((1, 1, shapes.found.gdn_qkv), 100 + 10 * slot, chips) for slot in range(4)]
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=6))
        return run(chips, gdn_conv_windows.build_windows, mesh, projected, history)

    def test_the_window_kernel_and_output_width_follow_the_width(self):
        with pair():
            record = self.window_launch(2, 8256)
        for kernel in kernels_of(record[0][1]).values():
            self.assertEqual(kernel.kernel_source, str(HERE / 'gdn_conv_windows.cpp'))
            self.assertEqual(kernel.defines, [])
        with four():
            # the builder reads validate_projected from the pinned module at call time: the startup seam rebinds it
            tp_addresses.install()
            self.addCleanup(tp_addresses.uninstall)
            record = self.window_launch(4, 4128)
        program = record[0][1]
        self.assertEqual(len(program), 4)
        outputs = record[0][0][5:]
        self.assertEqual({tuple(value.shape) for value in outputs}, {(1, 4, 2560)})
        for kernel in kernels_of(program).values():
            self.assertEqual(kernel.kernel_source, str(HERE / 'gdn_conv_windows_tp.cpp'))
            self.assertEqual(dict(kernel.defines), {'QWEN_STATE_PAGES': '192', 'QWEN_CONV_PAGES': '80',
                                                    'QWEN_CONV_TASKS': '320'})


def commit_layers(count, rows, chips):
    """`count` complete layer records (20 tensors: entry, history, native, checkpoint) at the served width."""
    shapes = geometry_shapes()
    found = shapes.found
    layout = (shapes.compact + [(rows, found.gdn_nv, 128, 128)] + [(1, rows, found.gdn_qkv)] * 4
              + shapes.full + shapes.compact)
    return [[FakeTensor(shape, 1000 * (layer + 1) + 10 * index, chips) for index, shape in enumerate(layout)]
            for layer in range(count)]


class CommitDmaTests(unittest.TestCase):
    def test_the_twin_validates_exactly_what_the_pinned_module_does_at_the_pair(self):
        with pair():
            for rows in (2, 4, 8, 16, 32):
                layers = [[tuple(tensor.shape) for tensor in layer] for layer in commit_layers(3, rows, 2)]
                for prefix in range(rows + 1):
                    self.assertEqual(gdn_commit_dma_tp.validate_shapes(layers, prefix),
                                     gdn_commit_dma.validate_shapes(layers, prefix))
            bad = [[tuple(tensor.shape) for tensor in commit_layers(1, 8, 2)[0]]]
            for layers, prefix in (([], 0), (bad * 49, 0), (bad, -1), (bad, 9), (bad, True)):
                with self.assertRaises(ValueError) as theirs:
                    gdn_commit_dma.validate_shapes(layers, prefix)
                with self.assertRaises(ValueError) as ours:
                    gdn_commit_dma_tp.validate_shapes(layers, prefix)
                self.assertEqual(str(ours.exception), str(theirs.exception))

    def test_four_cards_accept_their_records_and_refuse_the_pairs(self):
        with four():
            layers = [[tuple(tensor.shape) for tensor in layer] for layer in commit_layers(2, 16, 4)]
            self.assertEqual(gdn_commit_dma_tp.validate_shapes(layers, 5), 16)
        with pair():
            pair_layers = [[tuple(tensor.shape) for tensor in layer] for layer in commit_layers(2, 16, 2)]
        with four():
            with self.assertRaises(ValueError):
                gdn_commit_dma_tp.validate_shapes(pair_layers, 5)

    def launch(self, module, chips, layers=3, rows=8, prefix=5):
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        record = run(chips, module.prepare, mesh, commit_layers(layers, rows, chips), prefix)
        return record

    def test_at_the_pair_the_twin_builds_the_pinned_program(self):
        # prepare() returns the closure that launches; drive it to record the program
        def recorded(module):
            captured = []
            ttnn = fake_ttnn(captured, 2)
            mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
            with patch.dict(sys.modules, {'ttnn': ttnn}):
                module.prepare(mesh, commit_layers(3, 8, 2), 5)()
            return captured[0][1]

        with pair():
            theirs, ours = recorded(gdn_commit_dma), recorded(gdn_commit_dma_tp)
        self.assertEqual(sorted(theirs), sorted(ours))
        for key in theirs:
            left, right = theirs[key].kernels[0], ours[key].kernels[0]
            self.assertEqual(left.kernel_source, right.kernel_source)
            self.assertEqual(left.compile_time_args, right.compile_time_args)
            self.assertEqual(dict(left.runtime_args), dict(right.runtime_args))
            self.assertEqual(right.defines, [])
            self.assertEqual(theirs[key].cbs, ours[key].cbs)

    def test_four_cards_launch_the_sibling_on_four_chips_with_the_page_defines(self):
        captured = []
        ttnn = fake_ttnn(captured, 4)
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        with four(), patch.dict(sys.modules, {'ttnn': ttnn}):
            gdn_commit_dma_tp.prepare(mesh, commit_layers(3, 8, 4), 5)()
        program = captured[0][1]
        self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)])
        for chip in range(4):
            kernel = kernels_of(program)[((0, chip), (0, chip))]
            self.assertEqual(kernel.kernel_source, str(HERE / 'gdn_commit_dma_tp.cpp'))
            self.assertEqual(dict(kernel.defines)['QWEN_STATE_PAGES'], '192')
            first_core = kernel.runtime_args[0][0]
            self.assertEqual(first_core[-2:], [5, 0])
            self.assertEqual(first_core[0], 1000 + chip)

    def test_a_pair_sized_layer_set_is_refused_before_launch_at_four_cards(self):
        captured = []
        with four(), patch.dict(sys.modules, {'ttnn': fake_ttnn(captured, 2)}):
            with self.assertRaises(ValueError):
                gdn_commit_dma_tp.prepare(SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10)),
                                          commit_layers(3, 8, 2), 5)
        self.assertEqual(captured, [])


class ProjectedTests(unittest.TestCase):
    def states(self, channels, count=4):
        return [SimpleNamespace(shape=(1, 1, channels)) for _ in range(count)]

    def test_the_twin_agrees_with_the_pinned_validation_at_the_pair(self):
        cases = [((1, 16, 8256), 5120, 4), ((1, 16, 8240), 5120, 4), ((1, 1, 8256), 5120, 4),
                 ((1, 64, 8256), 5120, 4), ((1, 16, 4128), 5120, 4), ((1, 16, 8256), 2560, 4),
                 ((1, 16, 8256), 5120, 3), ((2, 16, 8256), 5120, 4), ((16, 8256), 5120, 4)]
        with pair():
            for shape, channels, count in cases:
                states = self.states(channels, count)
                try:
                    expected = ('value', pinned_conv.validate_projected(shape, states))
                except ValueError as error:
                    expected = ('error', str(error))
                try:
                    actual = ('value', gdn_multitoken_conv_tp.validate_projected(shape, states))
                except ValueError as error:
                    actual = ('error', str(error))
                self.assertEqual(actual, expected, (shape, channels, count))

    def test_four_cards_take_their_own_row_and_conv_width(self):
        with four():
            for columns in (4120, 4128):
                self.assertEqual(gdn_multitoken_conv_tp.validate_projected((1, 16, columns), self.states(2560)), 16)
            for shape, channels in (((1, 16, 8256), 2560), ((1, 16, 4128), 5120), ((1, 16, 8240), 5120)):
                with self.assertRaises(ValueError):
                    gdn_multitoken_conv_tp.validate_projected(shape, self.states(channels))


class RecordingOperations:
    def __init__(self):
        self.copies, self.freed, self.sliced = [], [], []
        self.DRAM_MEMORY_CONFIG = 'dram'

    def get_device_tensors(self, value):
        return value.shards

    def copy(self, source, destination):
        self.copies.append((source.name, destination.name))

    def deallocate(self, value):
        self.freed.append(value.name)

    def slice(self, value, start, end, memory_config=None):
        made = Tensor('slice%d' % len(self.sliced), (end[0] - start[0],) + tuple(end[1:]), value.chips, 9000 + 100 * len(self.sliced))
        self.sliced.append(made)
        return made


class Tensor:
    def __init__(self, name, shape, chips, base):
        self.name, self.shape, self.chips = name, tuple(shape), chips
        self.shards = [SimpleNamespace(buffer_address=lambda address=base + chip: address) for chip in range(chips)]


class RestoreAndRetainTests(unittest.TestCase):
    def restore_case(self, chips, heads, channels, rows=4):
        entry = [Tensor('entry%d' % index, shape, chips, 100 + 10 * index)
                 for index, shape in enumerate([(1, heads, 128, 128)] + [(1, 1, channels)] * 4)]
        destinations = [Tensor('dest%d' % index, shape, chips, 300 + 10 * index)
                        for index, shape in enumerate([(1, heads, 128, 128)] + [(1, 1, channels)] * 4)]
        prefixes = [[Tensor('p%d_%d' % (token, slot), (1, 1, channels), chips, 500 + 40 * token + 10 * slot)
                     for slot in range(4)] for token in range(rows)]
        result = dict(states=Tensor('states', (rows, heads, 128, 128), chips, 900), conv_prefixes=prefixes)
        return result, entry, destinations

    def test_the_restore_twin_does_what_the_pinned_one_does_at_the_pair(self):
        with pair():
            outcomes = []
            for restore in (pinned_conv.restore_prefix, gdn_multitoken_conv_tp.restore_prefix):
                ops = RecordingOperations()
                result, entry, destinations = self.restore_case(2, 24, 5120)
                restore(ops, result, entry, destinations, 3)
                outcomes.append((ops.copies, ops.freed, [tuple(made.shape) for made in ops.sliced]))
            self.assertEqual(outcomes[0], outcomes[1])
            for restore in (pinned_conv.restore_prefix, gdn_multitoken_conv_tp.restore_prefix):
                ops = RecordingOperations()
                result, entry, destinations = self.restore_case(2, 24, 5120)
                restore(ops, result, entry, destinations, 0)
                self.assertEqual(ops.copies, [('entry%d' % index, 'dest%d' % index) for index in range(5)])
                with self.assertRaises(ValueError):
                    restore(ops, result, entry, destinations, 9)
                with self.assertRaises(ValueError):
                    restore(ops, *self.restore_case(2, 12, 2560)[:1], entry, destinations, 1)

    def test_four_cards_restore_their_own_state_geometry(self):
        with four():
            ops = RecordingOperations()
            result, entry, destinations = self.restore_case(4, 12, 2560)
            gdn_multitoken_conv_tp.restore_prefix(ops, result, entry, destinations, 2)
            self.assertEqual([tuple(made.shape) for made in ops.sliced], [(1, 12, 128, 128)])
            self.assertEqual(ops.copies[0], ('slice0', 'dest0'))
            self.assertEqual(len(ops.copies), 5)
            ops = RecordingOperations()
            pair_result, pair_entry, pair_destinations = self.restore_case(4, 24, 5120)
            with self.assertRaises(ValueError):
                gdn_multitoken_conv_tp.restore_prefix(ops, pair_result, pair_entry, pair_destinations, 2)

    def record_result(self, chips, owned_extra=0):
        states = Tensor('states', (1, 1), chips, 100)
        windows = [Tensor('window%d' % index, (1, 1), chips, 200 + 10 * index) for index in range(4)]
        result = dict(states=states, packed_conv_states=windows, packed_checkpoints=True,
                      owned=[states, *windows, *[Tensor('scratch%d' % index, (1, 1), chips, 600 + 10 * index)
                                                 for index in range(owned_extra)]])
        return result, Tensor('output', (1, 1), chips, 400)

    def test_the_retain_twin_agrees_with_the_pinned_one_and_counts_four_chips(self):
        with pair():
            outcomes = []
            for retain in (gdn_records.retain_checkpoint_histories, gdn_records_tp.retain_checkpoint_histories):
                ops = RecordingOperations()
                result, output = self.record_result(2, owned_extra=2)
                retain(ops, result, output)
                outcomes.append((ops.freed, [value.name for value in result['owned']]))
            self.assertEqual(outcomes[0], outcomes[1])
            self.assertEqual(outcomes[0][0], ['scratch0', 'scratch1'])
        with four():
            ops = RecordingOperations()
            result, output = self.record_result(4, owned_extra=1)
            gdn_records_tp.retain_checkpoint_histories(ops, result, output)
            self.assertEqual(ops.freed, ['scratch0'])
            aliased = self.record_result(4)
            aliased[1].shards[3] = aliased[0]['states'].shards[3]
            with self.assertRaisesRegex(ValueError, 'independent on every chip'):
                gdn_records_tp.retain_checkpoint_histories(RecordingOperations(), *aliased)


class PackedBlockAtFourCardsTests(unittest.TestCase):
    """gdn_user_batch_conv.run_user_batched_projected end to end on a four-chip fake: the conv+gates call carries the
    four-card column offsets, z is the 1536 columns after the 2560 conv channels, and the one recurrence launch goes to
    the four-card sibling, not the pinned pair launch."""

    def operations(self, calls):
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1')
        counter = itertools.count(7016, 16)

        def make(name, shape):
            return Tensor(name, shape, 4, next(counter))

        operations.make = make
        operations.get_device_tensors = lambda value: value.shards
        operations.deallocate = lambda value: calls.append(('free', value.name))
        operations.to_memory_config = lambda value, memory: value

        def record_slice(value, start, stop, memory_config=None):
            calls.append(('slice', start[2], stop[2]))
            return make('z:' + value.name, (1, stop[1] - start[1], stop[2] - start[2]))

        operations.slice = record_slice

        def conv_gates(projected, windows, taps, a, b, dt_bias, neg_exp_A, batch=None, channels=None, a_col=None,
                       b_col=None, memory_config=None):
            calls.append(('conv_gates', batch, channels, a_col, b_col))
            return [make('conv', (1, batch, channels)), make('beta', (1, batch, 12)), make('gate', (1, batch, 12))]

        operations.transformer = SimpleNamespace(gdn_decode_conv_gates=conv_gates)
        return operations

    def test_the_four_card_columns_and_the_sibling_launch(self):
        import gdn_user_batch_conv
        import gdn_user_batch_tp
        calls = []
        operations = self.operations(calls)
        make = operations.make
        groups = [(make('projected%d' % index, (1, 16, 4128)), make('initial%d' % index, (1, 12, 128, 128)),
                   [make('conv%d.%d' % (index, tap), (1, 1, 2560)) for tap in range(4)]) for index in range(4)]
        taps = [make('tap%d' % tap, (1, 1, 2560)) for tap in range(4)]

        def windows(mesh, projected, history):
            return [make('window:%s.%d' % (projected.name, slot), (1, 16, 2560)) for slot in range(4)]

        def launch(mesh, inputs, kernels, ops, output_memory=None):
            calls.append(('launch', len(inputs)))
            return [(make('out%d' % index, (1, 16, 1536)), make('states%d' % index, (16, 12, 128, 128)))
                    for index in range(len(inputs))]

        with four():
            tp_addresses.install()
            self.addCleanup(tp_addresses.uninstall)
            with patch('gdn_conv_windows.build_windows', side_effect=windows),                     patch.object(gdn_user_batch_tp, 'execute', side_effect=launch) as sibling,                     patch('gdn_user_batch.execute') as pinned:
                results = gdn_user_batch_conv.run_user_batched_projected(
                    'mesh', groups, taps, make('dt', (1, 1, 12)), make('nega', (1, 1, 12)), make('norm_w', (1, 1, 128)),
                    'kernels', operations)
        self.assertEqual(sibling.call_count, 1)
        pinned.assert_not_called()
        self.assertEqual([entry for entry in calls if entry[0] == 'conv_gates'],
                         [('conv_gates', 16, 2560, 4096, 4108)] * 4)
        self.assertEqual({entry[1:] for entry in calls if entry[0] == 'slice'}, {(2560, 4096)})
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertEqual(result['packed_users'], 4)
            self.assertEqual(tuple(result['output'].shape), (1, 16, 1536))


if __name__ == '__main__':
    unittest.main()
