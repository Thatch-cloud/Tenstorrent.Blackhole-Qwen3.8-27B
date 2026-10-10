"""QWEN_FAST_OCTO_GLUE8, site 'windows': the T2 packed conv windows at the octo block's EIGHT-row users (gdn_conv_windows_packed8) and its wiring into
gdn_user_batch_conv.run_user_batched_projected.

What the CPU can hold the op to, as gdn_conv_windows_packed's tests hold the sixteen-row one: the trimmed copy map IS the served kernel's token loop at rows == 8 on every byte of
every page, padding included; zeroing each scratch page WHOLE once and rewriting rows 0-7 of both left faces for every window reproduces the served page across a sequence of tasks (the
sixteen-row kernel zeroes faces 2-3 only, which would leave rows 8-15 of faces 0-1 undefined at eight rows - the negative control); the kernel is the pinned one but for the intended
lines; the driver makes one generic_op per call with the served output signature and refuses what it does not serve; and the wiring: flag off the octo block runs exactly the calls it
ran before (the served windows under tp4_vglue.EIGHT_ROW_SERVED, and neither new module imported), flag on one packed8 launch precedes the per-user conv gates, a decline takes the
served path and says so under the glue8 marker. Whether the kernel moves the bytes that way on silicon is a card question (the in-trace T2 audit, G1, at the octo block).

Run: `python -m unittest test_octo_glue8_windows` from scripts/ci.
"""

import difflib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import gdn_conv_windows_packed as pinned
import gdn_seq_block
import gdn_conv_windows_packed8 as windows8
import octo_glue8
import tp4_vglue
import verify_trace_t2 as t2
from test_gdn_conv_windows_packed import DescriptorTTNN, mesh, random_pages, users as make_users

HERE = Path(__file__).resolve().parent
ON = {'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_OCTO_GLUE8': '1', 'QWEN_FAST_TP': '4'}


def served_literal(slot, rows=8):
    """gdn_conv_windows.cpp:12-22 read aloud: per token, the history index, the tile it copies from (0 the piece, h + 1 history h), the source row and the destination row."""
    out = []
    for token in range(rows):
        history = token + slot
        source_row = 0 if history < 4 else history - 4
        out.append((token, history, history + 1 if history < 4 else 0, source_row, token))
    return out


class CopyMapTests(unittest.TestCase):
    def test_the_row_map_is_the_served_loop_at_eight_rows(self):
        for slot in range(4):
            with self.subTest(slot=slot):
                self.assertEqual(pinned.served_row_map(slot, rows=8), served_literal(slot))
                expanded = []
                for kind, source, row, length in windows8.window_copies(slot):
                    for step in range(length // 32):
                        if kind == 'hist':
                            expanded.append((row + step, source, source + 1, 0, row + step))
                        else:
                            expanded.append((row + step, source + step + 4, 0, source + step, row + step))
                self.assertEqual(expanded, served_literal(slot))

    def test_every_offset_and_length_is_a_32_byte_multiple_inside_rows_0_to_7_of_the_left_faces(self):
        for slot in range(4):
            for kind, source, row, length in windows8.window_copies(slot):
                self.assertEqual(length % 32, 0)
                self.assertLessEqual(row * 32 + length, 8 * 32, 'nothing past row 7')
            self.assertEqual(sum(length for kind, source, row, length in windows8.window_copies(slot)), 8 * 32)
            piece_rows = [length // 32 for kind, source, row, length in windows8.window_copies(slot) if kind == 'piece']
            self.assertEqual(piece_rows, [4 + slot], 'the piece block is (4 + s) rows: rows 0 .. 3 + s, all inside the 8 the piece has')
        with self.assertRaises(ValueError):
            windows8.window_copies(4)

    def test_the_trimmed_map_equals_the_served_loop_on_every_byte_padding_included(self):
        piece = random_pages(1)
        histories = [random_pages(10 + h) for h in range(4)]
        served = windows8.reference_windows(piece, histories)
        trimmed = windows8.apply_copies(piece, histories)
        for slot in range(4):
            with self.subTest(slot=slot):
                self.assertTrue(torch.equal(served[slot], trimmed[slot]))
                self.assertTrue(bool((served[slot][:, 512:] == 0).all()), 'faces 2-3 are zero')
                for face in range(2):
                    self.assertTrue(bool((served[slot][:, face * 256 + 8 * 16:(face + 1) * 256] == 0).all()), 'rows 8-15 of the left faces are zero')

    def test_window_row_r_of_slot_s_is_timeline_row_r_plus_s(self):
        piece = torch.full((2, 1024), 777, dtype=torch.int16)
        for row in range(16):
            for face in range(2):
                piece[:, face * 256 + row * 16:face * 256 + row * 16 + 16] = row + 1
        histories = []
        for h in range(4):
            page = torch.full((2, 1024), 777, dtype=torch.int16)
            page[:, 0:16] = 100 + h
            page[:, 256:272] = 100 + h
            histories.append(page)
        timeline = [100, 101, 102, 103] + [row + 1 for row in range(16)]
        for slot, window in enumerate(windows8.reference_windows(piece, histories)):
            for row in range(8):
                for face in range(2):
                    values = window[0, face * 256 + row * 16:face * 256 + row * 16 + 16]
                    self.assertTrue(bool((values == timeline[row + slot]).all()), (slot, row, face))
            self.assertFalse(bool((window == 777).any()), 'nothing past row 7 of the piece or row 0 of a history is read')

    def test_minus_zero_denormals_and_nan_payloads_travel_as_bits(self):
        piece = random_pages(3, pages=4)
        piece[:, :8] = torch.tensor([-32768, 1, -32767, 32705, 127, -32641, 0, 5], dtype=torch.int16)
        histories = [random_pages(4 + h, pages=4) for h in range(4)]
        histories[0][:, 0:4] = torch.tensor([-32768, 1, 32705, -32641], dtype=torch.int16)
        served = windows8.reference_windows(piece, histories)
        trimmed = windows8.apply_copies(piece, histories)
        for slot in range(4):
            self.assertTrue(torch.equal(served[slot], trimmed[slot]))
        self.assertEqual(served[0][0, 0:4].tolist(), [-32768, 1, 32705, -32641])

    def test_a_scratch_zeroed_whole_once_stays_exact_across_a_sequence_of_tasks(self):
        """The kernel's scratch pages are zeroed once per core and every task rewrites rows 0-7 of both left faces of each window: the page after task n is the served page of
        task n whatever task n - 1 left. The negative control: a scratch that starts with junk in rows 8-15 (the sixteen-row kernel zeroes faces 2-3 only and never writes those
        rows, so it could not serve eight rows) fails."""
        pages = 8
        pages_per_task = [(random_pages(100 + task, pages), [random_pages(200 + 4 * task + h, pages) for h in range(4)]) for task in range(6)]

        def run(initial):
            scratch = [initial.clone() for slot in range(4)]
            outputs = []
            for piece, histories in pages_per_task:
                for slot in range(4):
                    for face in range(2):
                        base = face * 256
                        for kind, source, row, length in windows8.window_copies(slot):
                            words = length // 2
                            if kind == 'hist':
                                scratch[slot][:, base + row * 16:base + row * 16 + words] = histories[source][:, base:base + words]
                            else:
                                start = base + source * 16
                                scratch[slot][:, base + row * 16:base + row * 16 + words] = piece[:, start:start + words]
                outputs.append([page.clone() for page in scratch])
            return outputs

        served = [windows8.reference_windows(piece, histories) for piece, histories in pages_per_task]
        zeroed = run(torch.zeros(pages, 1024, dtype=torch.int16))
        for task in range(len(pages_per_task)):
            for slot in range(4):
                self.assertTrue(torch.equal(zeroed[task][slot], served[task][slot]), (task, slot))
        junk = run(random_pages(999, pages))
        self.assertFalse(all(torch.equal(junk[0][slot], served[0][slot]) for slot in range(4)), 'an unzeroed scratch is not the served page')
        faces23_only = torch.zeros(pages, 1024, dtype=torch.int16)
        faces23_only[:, 8 * 16:256] = 5                      # rows 8-15 of face 0 left undefined by a pinned-style zeroing
        self.assertFalse(torch.equal(run(faces23_only)[0][0], served[0][0]))


class KernelTextTests(unittest.TestCase):
    PINNED = (HERE / 'gdn_conv_windows_packed.cpp').read_text(encoding='utf-8')
    TEXT = (HERE / 'gdn_conv_windows_packed8.cpp').read_text(encoding='utf-8')

    def test_the_pinned_kernel_is_untouched_and_the_twin_differs_in_exactly_the_intended_lines(self):
        changed = []
        for line in difflib.ndiff(self.PINNED.splitlines(), self.TEXT.splitlines()):
            if line[:2] in ('- ', '+ ') and not line[2:].strip().startswith('//'):
                changed.append((line[0], line[2:].strip()))
        self.assertEqual(changed, [
            ('+', 'constexpr uint32_t window_rows = 8;'),
            ('-', 'copy_bytes(staged + f * face, destination + f * face + (history - slot) * row, (16 - history + slot) * row);'),
            ('+', 'copy_bytes(staged + f * face, destination + f * face + (history - slot) * row, (window_rows - history + slot) * row);'),
            ('-', 'static_assert(ROWS == 16, "the trimmed windows kernel zeroes faces 2-3 once: rows must be 16");'),
            ('+', 'static_assert(ROWS == 8, "the eight-row windows kernel zeroes the scratch pages once: rows must be 8");'),
            ('-', 'fill_words(scratch_base + index * page + 2 * face, 2 * face, pad_word);'),
            ('+', 'fill_words(scratch_base + index * page, page, pad_word);'),
            ('-', '#error "gdn_conv_windows_packed.cpp: define VTW_ROLE_READER, VTW_ROLE_WRITER or VTW_PORT"'),
            ('+', '#error "gdn_conv_windows_packed8.cpp: define VTW_ROLE_READER, VTW_ROLE_WRITER or VTW_PORT"'),
        ])

    def test_the_whole_scratch_page_is_zeroed_before_the_task_loop_under_rows_8(self):
        writer = self.TEXT[self.TEXT.index('#elif defined(VTW_ROLE_WRITER)'):self.TEXT.index('#elif defined(VTW_PORT)')]
        self.assertIn('static_assert(ROWS == 8', writer)
        self.assertLess(writer.index('static_assert(ROWS == 8'), writer.index('fill_words(scratch_base'))
        self.assertLess(writer.index('fill_words(scratch_base'), writer.index('for (uint32_t t = task_start'))
        self.assertIn('fill_words(scratch_base + index * page, page, pad_word);', writer)

    def test_the_three_roles_and_every_control_define_exist(self):
        for role in pinned.ROLES:
            self.assertIn('defined(%s)' % role, self.TEXT)
        for define in list(pinned.NEGATIVE_CONTROLS.values()) + ['VTW_HIST_ROW', 'VTW_COPY_NOC']:
            self.assertIn('#ifdef %s' % define, self.TEXT)

    def test_the_sibling_source_travels_with_the_module(self):
        self.assertEqual(windows8.source_path().name, 'gdn_conv_windows_packed8.cpp')
        self.assertEqual(windows8.source_path().parent, HERE)
        self.assertNotEqual(windows8.source_sha(), pinned.source_sha(), 'a different kernel keys a different JIT binary')
        self.assertEqual(windows8.RUNTIME_FILES, ('gdn_conv_windows_packed8.py', 'gdn_conv_windows_packed8.cpp'))


class DriverTests(unittest.TestCase):
    def setUp(self):
        windows8.clear_cache()
        pinned.clear_cache()
        self.addCleanup(windows8.clear_cache)
        self.ttnn = DescriptorTTNN()
        self.mesh = mesh()
        patcher = patch.object(gdn_seq_block.batch, 'MAX_USERS', 8)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, group, **options):
        return windows8.build_windows_packed8(self.mesh, group, operations=self.ttnn, **options)

    def test_one_launch_thirty_two_outputs_of_the_served_signature(self):
        group = make_users(self.ttnn, 8, rows=8)
        outputs = self.build(group)
        self.assertEqual(len(self.ttnn.programs()), 1)
        self.assertEqual([len(own) for own in outputs], [4] * 8)
        for own in outputs:
            for window in own:
                self.assertEqual((window.shape, window.dtype, window.layout, window.memory_config()), ((1, 8, 5120), 'bf16', 'tile', 'dram'))
        name, tensors, program = self.ttnn.programs()[0]
        self.assertEqual(len(tensors), 8 + 32 + 32)
        chip0 = program[((0, 0), (0, 0))]
        self.assertEqual([kernel['config'].processor for kernel in chip0['kernels']], ['riscv1', 'riscv0'])
        self.assertTrue(all(kernel['source'].endswith('gdn_conv_windows_packed8.cpp') for kernel in chip0['kernels']))
        self.assertEqual(chip0['kernels'][0]['compile'][:5], [8, 160, 8, pinned.DEFAULTS['nbuf'], 1280])
        runtime = chip0['kernels'][1]['runtime']
        self.assertEqual(len(runtime), 110)
        self.assertEqual(sum(args[1] for x, y, args in runtime), 1280)
        self.assertEqual(sorted(args[1] for x, y, args in runtime)[::-1][:1], [12])
        sha = windows8.source_sha()
        self.assertTrue(all(('VTW_SRC_SHA', '0x' + sha) in kernel['defines'] for kernel in chip0['kernels']))

    def test_each_chip_gets_its_own_addresses_in_the_kernel_layout(self):
        group = make_users(self.ttnn, 8, rows=8)
        outputs = self.build(group)
        program = self.ttnn.programs()[0][2]
        for chip in range(2):
            common = program[((0, chip), (0, chip))]['kernels'][0]['common']
            self.assertEqual(len(common), 8 + 32 + 32)
            for user, (piece, history) in enumerate(group):
                self.assertEqual(common[pinned.common_index(8, 'piece', user)], piece.shards[chip].address)
                for index in range(4):
                    self.assertEqual(common[pinned.common_index(8, 'history', user, index)], history[index].shards[chip].address)
                    self.assertEqual(common[pinned.common_index(8, 'output', user, index)], outputs[user][index].shards[chip].address)

    def test_the_second_call_hits_the_descriptor_cache_and_it_is_not_the_pinned_ones(self):
        first = self.build(make_users(self.ttnn, 8, rows=8))
        second = self.build(make_users(self.ttnn, 8, rows=8))
        self.assertEqual(windows8.cache_size(), 1)
        self.assertEqual(pinned.cache_size(), 0)
        self.assertIsNot(first[0][0], second[0][0])

    def test_four_cards_serve_80_pages_and_2560_channels(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            self.ttnn = DescriptorTTNN(chips=4)
            self.mesh = mesh(4)
            outputs = self.build(make_users(self.ttnn, 8, rows=8))
        chip0 = self.ttnn.programs()[0][2][((0, 0), (0, 0))]
        self.assertEqual(chip0['kernels'][0]['compile'][:5], [8, 80, 8, pinned.DEFAULTS['nbuf'], 640])
        self.assertEqual(outputs[0][0].shape, (1, 8, 2560))

    def test_a_failed_launch_frees_every_output(self):
        self.ttnn.fail = RuntimeError('device')
        with self.assertRaisesRegex(RuntimeError, 'device'):
            self.build(make_users(self.ttnn, 8, rows=8))
        self.assertEqual(len(self.ttnn.deallocated), 32)
        self.assertEqual({id(value) for value in self.ttnn.deallocated}, {id(value) for value in self.ttnn.made})

    def test_aliases_are_refused_with_the_served_text(self):
        group = make_users(self.ttnn, 2, rows=8)
        piece, history = group[0]
        group[0] = (piece, [history[0], history[0], history[2], history[3]])
        with self.assertRaisesRegex(ValueError, 'Immutable input and mutable windows must not alias'):
            self.build(group)
        self.assertEqual(self.ttnn.programs(), [])

    def test_everything_it_does_not_serve_is_refused_before_any_device_work(self):
        cases = {
            'nine users': make_users(self.ttnn, 9, rows=8),
            'sixteen rows': make_users(self.ttnn, 4, rows=16),
            'four rows': make_users(self.ttnn, 4, rows=4),
            'thirty-two rows': make_users(self.ttnn, 4, rows=32),
            'a bad width': make_users(self.ttnn, 8, rows=8, width=8000),
            'three histories': [(p, h[:3]) for p, h in make_users(self.ttnn, 8, rows=8)],
            'a row-major piece': [(self.ttnn.tensor('p', (1, 8, 8240), 'l1', layout='row_major'), h) for p, h in make_users(self.ttnn, 1, rows=8)],
            'mixed history placement': [(p, [self.ttnn.tensor('h', (1, 1, 5120), 'l1')] + h[1:]) for p, h in make_users(self.ttnn, 1, rows=8)],
        }
        for name, group in cases.items():
            with self.subTest(case=name), self.assertRaises(windows8.Unsupported):
                self.build(group)
        self.assertEqual((self.ttnn.made, self.ttnn.programs(), windows8.cache_size()), ([], [], 0))
        self.assertIs(windows8.Unsupported, pinned.Unsupported, 'one exception type: the wiring catches the pinned module\'s own')

    def test_the_user_cap_is_the_width_aware_batch_modules_not_the_pinned_names(self):
        """Card log of the octo block: the pinned op refused it with '8 users outside 1..4' because it reads the pair's gdn_user_batch. Here the cap is gdn_seq_block.batch's."""
        import gdn_user_batch

        self.assertEqual(gdn_user_batch.MAX_USERS, 4)
        with patch.object(gdn_seq_block.batch, 'MAX_USERS', 4):
            with self.assertRaisesRegex(windows8.Unsupported, '8 users outside 1..4'):
                self.build(make_users(self.ttnn, 8, rows=8))
        self.build(make_users(self.ttnn, 8, rows=8))

    def test_the_pinned_op_still_refuses_eight_rows_and_serves_sixteen(self):
        with self.assertRaises(pinned.Unsupported):
            pinned.build_windows_packed(self.mesh, make_users(self.ttnn, 4, rows=8), operations=self.ttnn)
        pinned.build_windows_packed(self.mesh, make_users(self.ttnn, 4, rows=16), operations=self.ttnn)
        self.assertEqual(windows8.cache_size(), 0)

    def test_the_negative_controls_are_defines(self):
        for negative, define in pinned.NEGATIVE_CONTROLS.items():
            windows8.clear_cache()
            self.build(make_users(self.ttnn, 8, rows=8), negative=negative)
            kernels = self.ttnn.programs()[-1][2][((0, 0), (0, 0))]['kernels']
            self.assertTrue(all((define, '1') in kernel['defines'] for kernel in kernels), negative)


class WiringTests(unittest.TestCase):
    """gdn_user_batch_conv.run_user_batched_projected over eight users of eight rows."""

    def setUp(self):
        from test_gdn_user_batch_conv import fake_operations

        self.calls = []
        self.operations = fake_operations(self.calls)
        self.groups = [self.user_group(index, 8) for index in range(8)]
        self.taps = [self.operations.make('tap%d' % tap, (1, 1, 2560)) for tap in range(4)]
        self.extras = (self.operations.make('dt', (1, 1, 24)), self.operations.make('nega', (1, 1, 24)), self.operations.make('norm_w', (1, 1, 128)))
        t2.take()
        t2._LOGGED.clear()
        tp4_vglue.take()
        self.addCleanup(t2.take)
        self.addCleanup(tp4_vglue.take)
        for patcher in (patch.object(tp4_vglue, '_LOGGED_ONCE', set()), patch.object(octo_glue8, '_LOGGED', set()),
                        patch.object(gdn_seq_block.batch, 'MAX_USERS', 8)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def user_group(self, index, rows):
        """One packed user at four cards: the (1, rows, 4120) projection, the recurrent state, four (1, 1, 2560) conv history rows."""
        return (self.operations.make('projected%d' % index, (1, rows, 4120)), self.operations.make('initial%d' % index, (1, 12, 128, 128)),
                [self.operations.make('conv%d.%d' % (index, tap), (1, 1, 2560)) for tap in range(4)])

    def served(self, mesh_, projected, history):
        self.calls.append(('windows', projected.name))
        return [self.operations.make('window:%s.%d' % (projected.name, slot), (1, projected.shape[1], 2560)) for slot in range(4)]

    def packed(self, name):
        def build(mesh_, group, operations=None):
            self.calls.append((name, tuple(piece.name for piece, history in group)))
            return [[self.operations.make('%s:%s.%d' % (name, piece.name, slot), (1, piece.shape[1], 2560)) for slot in range(4)] for piece, history in group]
        return build

    def refuse(self, text):
        import importlib

        def build(mesh_, group, operations=None):
            # the module object in sys.modules NOW: another test module pops and re-imports gdn_conv_windows_packed, and the wiring catches the current class
            raise importlib.import_module('gdn_conv_windows_packed').Unsupported(text)
        return build

    def run_block(self, environ, pinned_builder=None, eight_builder=None, groups=None, patch_eight=True):
        import contextlib

        import gdn_user_batch_conv

        def batched(mesh_, inputs, kernels, ops, output_memory=None):
            self.calls.append(('batched', len(inputs)))
            return [(self.operations.make('out%d' % index, (1, 8, 3072)), self.operations.make('prefix%d' % index, (8, 24, 128, 128)))
                    for index in range(len(inputs))]

        lines, notices = [], []
        with patch.dict('os.environ', environ, clear=True), contextlib.ExitStack() as stack:
            stack.enter_context(patch('gdn_conv_windows.build_windows', side_effect=self.served))
            stack.enter_context(patch('gdn_conv_windows_packed.build_windows_packed', side_effect=pinned_builder or self.packed('packed')))
            if patch_eight:
                stack.enter_context(patch('gdn_conv_windows_packed8.build_windows_packed8', side_effect=eight_builder or self.packed('packed8')))
                stack.enter_context(patch.object(octo_glue8, 'log_line', side_effect=notices.append))
            stack.enter_context(patch.object(gdn_seq_block.batch, 'execute', side_effect=batched))
            stack.enter_context(patch('gdn_user_batch_conv.validate_projected', side_effect=lambda shape, states: shape[1]))
            stack.enter_context(patch.object(t2, 'log_line', side_effect=lines.append))
            stack.enter_context(patch.object(tp4_vglue, 'log_line', side_effect=notices.append))
            results = gdn_user_batch_conv.run_user_batched_projected('mesh', groups or self.groups, self.taps, *self.extras, 'kernels', self.operations)
        return results, lines, notices

    def order(self):
        return [entry[0] for entry in self.calls if entry[0] in ('windows', 'packed', 'packed8', 'conv_gates', 'slice', 'batched')]

    def test_flag_off_the_octo_block_is_todays_and_neither_new_module_is_imported(self):
        for name in ('gdn_conv_windows_packed8', 'octo_glue8'):
            self.addCleanup(sys.modules.__setitem__, name, sys.modules[name])         # the other tests patch these very module objects
            sys.modules.pop(name, None)
        results, lines, notices = self.run_block({'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_TP': '4'}, pinned_builder=self.refuse('8 users outside 1..4'), patch_eight=False)
        self.assertNotIn('gdn_conv_windows_packed8', sys.modules)
        self.assertNotIn('octo_glue8', sys.modules)
        self.assertEqual(self.order(), ['windows', 'conv_gates', 'slice'] * 8 + ['batched'])
        self.assertEqual(lines, [], 'no verify t2 fell back line')
        self.assertEqual(notices, ['%s site=windows reason=8 users outside 1..4' % tp4_vglue.EIGHT_ROW_SERVED])
        self.assertEqual(t2.take(), {'windows_fallback': 1})
        self.assertEqual(tp4_vglue.take(), {})

    def test_flag_zero_is_off_too(self):
        results, lines, notices = self.run_block({'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_OCTO_GLUE8': '0', 'QWEN_FAST_TP': '4'}, pinned_builder=self.refuse('x'))
        self.assertEqual(self.order(), ['windows', 'conv_gates', 'slice'] * 8 + ['batched'])
        self.assertEqual(tp4_vglue.take(), {})

    def test_flag_on_one_packed8_launch_then_the_gates_and_z_per_user(self):
        results, lines, notices = self.run_block(ON)
        self.assertEqual(self.order(), ['packed8'] + ['conv_gates', 'slice'] * 8 + ['batched'])
        self.assertEqual([entry for entry in self.calls if entry[0] == 'packed8'], [('packed8', tuple('projected%d' % user for user in range(8)))])
        self.assertEqual([entry for entry in self.calls if entry[0] == 'packed'], [], 'the sixteen-row op is not called')
        gates = [entry for entry in self.calls if entry[0] == 'conv_gates']
        for index, entry in enumerate(gates):
            self.assertEqual(entry[3], tuple('packed8:projected%d.%d' % (index, slot) for slot in range(4)))
        for index, result in enumerate(results):
            self.assertEqual([w.name for w in result['packed_conv_states']], ['packed8:projected%d.%d' % (index, slot) for slot in range(4)])
        self.assertEqual(t2.take(), {'windows': 1})
        self.assertEqual(tp4_vglue.take(), {'glue8_windows': 1})
        self.assertEqual(notices, [octo_glue8.engaged_line('windows')], 'once per process')
        self.assertEqual(lines, [])

    def test_flag_on_a_second_layer_counts_again_and_logs_once(self):
        self.run_block(ON)
        self.run_block(ON)
        self.assertEqual(tp4_vglue.take(), {'glue8_windows': 2})
        self.assertEqual(t2.take(), {'windows': 2})

    def test_a_declined_native_launch_takes_the_served_path_under_the_glue8_fallback_marker(self):
        results, lines, notices = self.run_block(ON, eight_builder=self.refuse('mixed placement'))
        self.assertEqual(self.order(), ['windows', 'conv_gates', 'slice'] * 8 + ['batched'])
        self.assertEqual(lines, [], 'not the verify t2 fallback line')
        self.assertEqual(notices, [octo_glue8.fallback_line('windows', 'mixed placement')])
        self.assertFalse(any(tp4_vglue.EIGHT_ROW_SERVED in notice for notice in notices))
        self.assertEqual(t2.take(), {'windows_fallback': 1})
        self.assertEqual(tp4_vglue.take(), {'glue8_windows_fallback': 1})

    def test_an_m3_block_with_the_flag_on_still_runs_the_pinned_op(self):
        groups = [self.user_group(index, 16) for index in range(4)]
        results, lines, notices = self.run_block(ON, groups=groups)
        self.assertEqual([entry[0] for entry in self.calls if entry[0] in ('packed', 'packed8')], ['packed'])
        self.assertEqual(tp4_vglue.take(), {})
        self.assertEqual(notices, [])

    def test_without_the_t2_windows_cut_the_flag_changes_nothing(self):
        for environ in ({'QWEN_FAST_OCTO_GLUE8': '1', 'QWEN_FAST_TP': '4'},
                        dict(ON, QWEN_FAST_VERIFY_T2_SKIP='windows')):
            self.calls.clear()
            results, lines, notices = self.run_block(environ)
            self.assertEqual(self.order(), ['windows', 'conv_gates', 'slice'] * 8 + ['batched'])
            self.assertEqual(notices, [])
        self.assertEqual(tp4_vglue.take(), {})

    def test_a_bad_flag_value_raises_at_the_octo_block_and_the_pair_refuses_it(self):
        with self.assertRaises(ValueError):
            self.run_block(dict(ON, QWEN_FAST_OCTO_GLUE8='2'))
        with self.assertRaises(ValueError):
            self.run_block({'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_OCTO_GLUE8': '1'})            # QWEN_FAST_TP unset: the pair

    def test_the_audit_builds_the_served_windows_beside_the_packed8_ones(self):
        results, lines, notices = self.run_block(dict(ON, QWEN_FAST_VERIFY_T2_AUDIT='1'))
        self.assertEqual(sum(1 for entry in self.calls if entry[0] == 'windows'), 8)
        for index, result in enumerate(results):
            self.assertEqual([w.name for w in result['audit_windows']], ['window:projected%d.%d' % (index, slot) for slot in range(4)])
            self.assertEqual([w.name for w in result['packed_conv_states']], ['packed8:projected%d.%d' % (index, slot) for slot in range(4)])

    def test_eight_row_native_is_exactly_the_octo_block_with_the_flag(self):
        import gdn_user_batch_conv as conv

        def groups(users, rows):
            return [(SimpleNamespace(shape=(1, rows, 5120)), None, None) for _ in range(users)]

        with patch.dict('os.environ', {'QWEN_FAST_OCTO_GLUE8': '1', 'QWEN_FAST_TP': '4'}, clear=True):
            self.assertTrue(conv.eight_row_native(groups(8, 8)))
            for users, rows in ((4, 16), (4, 8), (8, 16), (7, 8)):
                self.assertFalse(conv.eight_row_native(groups(users, rows)), (users, rows))
        with patch.dict('os.environ', {'QWEN_FAST_TP': '4'}, clear=True):
            self.assertFalse(conv.eight_row_native(groups(8, 8)))


if __name__ == '__main__':
    unittest.main()
