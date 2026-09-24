"""CPU tests for the verify-trace T2 card-M harness: its matrices, data kinds, raw-page layout,
geometries and verdict helpers; the served-driver re-drives pinned against the served drivers
themselves (gdn_conv_windows.build_windows, ordered_cache.update) on a recording fake; the section
drivers against a torch-only fake Bench; and run_card_m.sh (the qualification card, the A5 image, the one
table mounted file by file, the reference shas, never a reset). The device half runs on the rig only."""

import ast
import contextlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
CI = ROOT / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))

import torch  # noqa: E402

import gdn_conv_windows_packed as vtw  # noqa: E402
import packed_ordered_cache as poc  # noqa: E402
import verify_t2_card_m as card  # noqa: E402
import verify_trace_t2 as t2  # noqa: E402
from test_gdn_conv_windows_packed import DescriptorTTNN, mesh  # noqa: E402

SCRIPT = HERE / 'run_card_m.sh'


def script_text():
    return SCRIPT.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))


class MatrixTests(unittest.TestCase):
    def test_the_windows_matrix_covers_every_axis(self):
        cases = card.windows_matrix()
        self.assertEqual({case['users'] for case in cases}, set(card.USERS))
        self.assertEqual({case['width'] for case in cases}, set(card.WIDTHS))
        self.assertEqual({case['source'] for case in cases}, {'served', 'direct'})
        self.assertEqual({case['placement'] for case in cases}, {'dram', 'l1'})
        self.assertEqual({case['piece'] for case in cases}, set(card.DATA))
        self.assertEqual({case['history'] for case in cases}, set(card.DATA))
        self.assertEqual({case['seed'] for case in cases}, set(card.SEEDS))
        names = [card.windows_case_name(case) for case in cases]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(cases[0], dict(users=4, width=8240, source='served', placement='dram', piece='randn',
                                        history='randn', seed=0), 'the served configuration first')

    def test_the_quick_matrix_is_a_subset_that_keeps_the_signed_zero_and_specials(self):
        quick, full = card.windows_matrix(quick=True), card.windows_matrix()
        self.assertLess(len(quick), len(full))
        self.assertTrue(all(case in full for case in quick))
        self.assertTrue(any(case['piece'] == 'zeros' for case in quick))
        self.assertTrue(any(case['history'] == 'specials' for case in quick))

    def test_the_kv_matrix(self):
        cases = card.kv_matrix()
        self.assertEqual({case['width'] for case in cases}, set(card.KV_WIDTHS))
        self.assertEqual({case['geometry'] for case in cases if case['width'] == 2052}, set(card.KV_GEOMETRIES))
        self.assertNotIn('g2', {case['geometry'] for case in cases if case['width'] != 2052})
        self.assertEqual({case['payload'] for case in cases}, set(card.KV_PAYLOADS))
        self.assertEqual({case['seed'] for case in cases if case['payload'] == 'bf8'}, set(card.SEEDS))
        quick = card.kv_matrix(quick=True)
        self.assertTrue(all(case in cases for case in quick))
        self.assertEqual({case['payload'] for case in quick}, {'bf8', 'zeros'})
        self.assertEqual(card.kv_variants(), [(64, 256), (32, 256), (64, 16), (64, 512)])
        self.assertEqual(card.kv_variants(True), [(64, 256), (32, 256)])

    def test_refusals(self):
        self.assertEqual([case['name'] for case in card.refusal_cases()], ['rows8', 'rows32', 'mixed_history'])


class DataTests(unittest.TestCase):
    def bits(self, kind, rows=16, cols=96):
        return card.make_rows(torch, kind, rows, cols, 0).view(torch.int16).int() & 0xFFFF

    def test_every_kind_is_bf16_and_deterministic(self):
        for kind in card.DATA:
            value = card.make_rows(torch, kind, 16, 96, 3)
            self.assertEqual((tuple(value.shape), value.dtype), ((16, 96), torch.bfloat16), kind)
            self.assertTrue(torch.equal(value.view(torch.int16), card.make_rows(torch, kind, 16, 96, 3).view(torch.int16)))

    def test_the_edge_values_are_really_there(self):
        zeros = self.bits('zeros')
        self.assertTrue(bool((zeros == 0x8000).any()) and bool((zeros == 0).any()))
        denormal = self.bits('denormal')
        self.assertGreater(int((((denormal & 0x7F80) == 0) & ((denormal & 0x7F) != 0)).sum()), 50)
        specials = set((self.bits('specials', 32, 256)).flatten().tolist())
        self.assertLessEqual({0x7FC1, 0x7F81, 0xFFC3, 0x7F80, 0xFF80, 0x8000}, specials)

    def test_tile_pages_is_the_face_order(self):
        matrix = torch.arange(32 * 64, dtype=torch.int16).reshape(32, 64).view(torch.bfloat16)
        pages = card.tile_pages(torch, matrix)
        self.assertEqual(tuple(pages.shape), (2, 1024))
        self.assertEqual(pages[0, 0:16].tolist(), list(range(16)))                 # face 0, row 0
        self.assertEqual(pages[0, 256:272].tolist(), list(range(16, 32)))          # face 1, row 0
        self.assertEqual(pages[0, 512:528].tolist(), list(range(16 * 64, 16 * 64 + 16)))  # face 2, row 16
        self.assertEqual(pages[1, 0:16].tolist(), list(range(32, 48)))             # tile 1, face 0
        self.assertEqual(pages[0, 16:32].tolist(), list(range(64, 80)))            # face 0, row 1

    def test_padding_is_poisoned_and_untile_inverts(self):
        matrix = card.make_rows(torch, 'specials', 16, 8240, 1)
        pages = card.tile_pages(torch, matrix, card.POISON)
        self.assertEqual(tuple(pages.shape), (258, 1024))
        self.assertTrue(torch.equal(card.untile_pages(torch, pages, 16, 8240), matrix.view(torch.int16)))
        padded = card.untile_pages(torch, pages, 32, 8256)
        self.assertTrue(bool((padded[16:] == card.POISON).all()))
        self.assertTrue(bool((padded[:16, 8240:] == card.POISON).all()))

    def test_counts_and_compare(self):
        values = torch.tensor([0, -32768, 1, -32767, 128, 5], dtype=torch.int16)
        self.assertEqual(card.special_counts(torch, values), dict(minus_zero=1, denormal=3))
        a = torch.zeros(3, 1024, dtype=torch.int16)
        b = a.clone()
        b[2, 7] = -32768
        self.assertEqual(card.compare_pages(torch, a, a.clone()), dict(exact=True, differing=0))
        self.assertEqual(card.compare_pages(torch, a, b), dict(exact=False, differing=1, pages=[2], differing_pages=1))
        self.assertFalse(card.compare_pages(torch, a, b[:2])['exact'])

    def test_kv_payload_poisons_the_padding_heads(self):
        for kind in card.KV_PAYLOADS:
            values = card.kv_payload(torch, kind, 64, 1)
            self.assertEqual(tuple(values.shape), (64, 32, 256), kind)
            self.assertTrue(bool((values[:, 2:].view(torch.int16) == card.POISON).all()), kind)
        from ordered_cache_hw_plan import bf8_exact
        self.assertTrue(bf8_exact(card.kv_payload(torch, 'bf8', 8, 0)[:, :2]))

    def test_the_initial_cache_is_bf8_exact_and_nonzero(self):
        from ordered_cache_hw_plan import bf8_exact, payload

        values = card.kv_initial_values(torch, 516, 0, payload)
        self.assertEqual(tuple(values.shape), (528, 2, 64, 256))
        self.assertTrue(bf8_exact(values[:3]))
        self.assertFalse(bool((values == 0).any()))


class GeometryTests(unittest.TestCase):
    def users(self, geometry):
        return [(range(start, start + card.KV_ROWS_PER_USER), table) for start, table in geometry]

    def test_every_geometry_is_disjoint_inside_its_page_table(self):
        for width in card.KV_WIDTHS:
            for name in card.KV_GEOMETRIES:
                for seed in card.SEEDS:
                    geometry = card.kv_geometry(name, width, seed)
                    if geometry is None:
                        self.assertEqual((name, width != 2052), ('g2', True))
                        continue
                    with self.subTest(width=width, geometry=name, seed=seed):
                        self.assertEqual(len(geometry), 4)
                        if name == 'g4':
                            # the served placeholders: one tile row for all four, written as one chain
                            self.assertIsNotNone(t2.kv_conflict(self.users(geometry)))
                            self.assertEqual(card.kv_case_spans(dict(geometry=name)), ((0, 64),))
                        else:
                            self.assertIsNone(t2.kv_conflict(self.users(geometry)))
                            self.assertEqual(card.kv_case_spans(dict(geometry=name)), card.kv_spans())
                        for start, table in geometry:
                            self.assertEqual(len(table), width)
                            self.assertLess((start + 15) // 64, width)
                            self.assertTrue(all(0 <= table[block] < card.kv_pages_total(width)
                                                for block in card.touched_blocks(start)))

    def test_g1_starts_straddle_tile_rows_and_blocks(self):
        rows = set()
        for seed in card.SEEDS:
            for start, table in card.kv_geometry('g1', 2052, seed):
                rows.add(len({(start + index) // 32 for index in range(16)}))
                self.assertIn(start % 64, card.KV_OFFSETS)
        self.assertEqual(rows, {1, 2})

    def test_g2_reads_the_last_table_entries(self):
        blocks = {block for start, table in card.kv_geometry('g2', 2052, 0) for block in card.touched_blocks(start)}
        self.assertLessEqual({2048, 2049, 2050, 2051}, blocks)

    def test_g3_shares_a_page_at_different_tile_rows(self):
        geometry = card.kv_geometry('g3', 1024, 0)
        (start0, table0), (start1, table1) = geometry[:2]
        self.assertEqual(table0[start0 // 64], table1[start1 // 64])
        self.assertNotEqual((start0 % 64) // 32, (start1 % 64) // 32)

    def test_g4_is_the_served_placeholders_the_warm_forward_writes(self):
        geometry = card.kv_geometry('g4', 516, 0)
        self.assertEqual([start for start, table in geometry], [32768] * 4)
        self.assertEqual([set(table) for start, table in geometry], [{0}] * 4)

    def test_the_conflict_geometry_conflicts(self):
        for width in card.KV_WIDTHS:
            for seed in range(5):
                conflict = t2.kv_conflict(self.users(card.kv_conflict_geometry(width, seed)))
                self.assertIsNotNone(conflict, (width, seed))
                self.assertEqual(conflict['users'], (0, 2))

    def test_deconflict_remaps_a_chance_collision(self):
        table = list(range(100))
        geometry = card.deconflict([(0, table), (4, list(table)), (64 * 5, list(table)), (64 * 9, list(table))], 120)
        self.assertIsNone(t2.kv_conflict(self.users(geometry)))
        self.assertEqual(geometry[0][1][0], 0)
        self.assertNotEqual(geometry[1][1][0], 0)

    def test_block_host_and_the_host_prediction(self):
        geometry = [(5, [3, 4]), (70, [6, 7]), (40, [8, 9]), (100, [10, 11])]
        positions, pages = card.kv_block_host(geometry)
        self.assertEqual((len(positions), len(pages)), (64, 64))
        self.assertEqual(positions[:2] + positions[16:18], [5, 6, 70, 71])
        self.assertEqual(pages[16], [6, 7])
        initial = torch.zeros(12, 2, 64, 256, dtype=torch.bfloat16)
        values = torch.arange(64, dtype=torch.float32).reshape(64, 1, 1).expand(64, 32, 256).to(torch.bfloat16) + 1
        predicted = card.predict_cache(torch, initial, geometry, values)
        self.assertEqual(float(predicted[3, 0, 5, 0]), 1.0)
        self.assertEqual(float(predicted[7, 1, 70 % 64, 3]), 17.0)
        self.assertEqual(int((predicted != 0).reshape(12, -1).any(dim=1).sum()), 4)


class VerdictTests(unittest.TestCase):
    def test_the_settings_verdict(self):
        exact = dict(port=True, a=True, b=True)
        self.assertEqual(card.settings_verdict(dict(exact, b=False), dict(a=10.0), 'a')['verdict'], 'inexact')
        self.assertEqual(card.settings_verdict(exact, {}, 'a')['verdict'], 'untimed')
        self.assertEqual(card.settings_verdict(exact, dict(port=100.0, a=40.0, b=39.0), 'a')['verdict'], 'keep')
        change = card.settings_verdict(exact, dict(port=100.0, a=45.0, b=39.0), 'a')
        self.assertEqual((change['verdict'], change['fastest']), ('change', 'b'))

    def test_provenance_needs_every_reference_and_names_a_difference(self):
        shas = {name: '%064x' % index for index, name in enumerate(card.REFERENCE_FILES)}
        self.assertEqual(card.provenance_failures(shas, dict(shas)), [])
        self.assertEqual(len(card.provenance_failures(shas, {})), len(card.REFERENCE_FILES))
        moved = dict(shas, **{'ordered_cache.py': 'f' * 64})
        self.assertEqual(len(card.provenance_failures(moved, shas)), 1)
        self.assertIn('the image has no gdn_conv_windows.cpp',
                      card.provenance_failures(dict(shas, **{'gdn_conv_windows.cpp': None}), shas))

    def test_verdict_needs_a_completed_section_and_no_failure(self):
        self.assertFalse(card.verdict(dict(failures=[], sections_completed=[])))
        self.assertFalse(card.verdict(dict(failures=['x'], sections_completed=['kv'])))
        self.assertTrue(card.verdict(dict(failures=[], sections_completed=['kv'])))

    def test_args(self):
        args = card.parse_args(['--out', 'x.json', '--sections', 'selftest,kv', '--kv-widths', '2052',
                                '--expect', 'ordered_cache.py=' + 'a' * 64])
        self.assertEqual((args.sections, args.kv_widths, args.expect), (['selftest', 'kv'], [2052],
                                                                         {'ordered_cache.py': 'a' * 64}))
        self.assertEqual(args.runtime_files, list(t2.RUNTIME_FILES))
        for bad in (['--sections', 'bogus'], ['--kv-widths', '2048'], ['--expect', 'tp_common.py=' + 'a' * 64],
                    ['--expect', 'ordered_cache.py=abc']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                card.parse_args(['--out', 'x.json'] + bad)

    def test_no_device_import_at_module_level(self):
        source = (HERE / 'verify_t2_card_m.py').read_text(encoding='utf-8')
        head = source[:source.index('# Device part: the qualification card only.')]
        self.assertIsNone(re.search(r'^(import|from) (ttnn|torch)', head, re.M))

    def test_every_device_call_in_the_device_part_runs_under_the_watchdog(self):
        """A hang that first surfaces in an upload, a readback or a restage must still fire the
        watchdog (partial report, exit 3), not wait for the container timeout."""
        source = (HERE / 'verify_t2_card_m.py').read_text(encoding='utf-8')
        start = source.index('# Device part: the qualification card only.')
        tree = ast.parse(source)
        device_calls = {'from_torch', 'to_torch', 'copy_host_to_device_tensor', 'generic_op', 'execute_trace',
                        'synchronize_device', 'begin_trace_capture', 'end_trace_capture', 'slice', 'ReadDeviceProfiler',
                        'close_device'}
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        first_line = source[:start].count(chr(10)) + 1
        bare = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in device_calls and node.lineno > first_line):
                continue
            guarded, parent = False, parents.get(node)
            while parent is not None and not guarded:
                if isinstance(parent, ast.With):
                    guarded = any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
                                  and item.context_expr.func.attr in ('op', 'io') for item in parent.items)
                parent = parents.get(parent)
            if not guarded:
                bare.append('%d %s' % (node.lineno, node.func.attr))
        self.assertEqual(bare, [])


# ---------------------------------------------------------------------------------------------
# The re-drives are the served drivers, descriptor for descriptor.
# ---------------------------------------------------------------------------------------------

class RedriveTests(unittest.TestCase):
    def programs_of(self, ttnn, runs):
        out = []
        for run in runs:
            ttnn.addresses = iter(range(0x900000, 0xA00000, 0x1000))
            ttnn.made.clear()
            run()
            out.append(ttnn.programs()[-1][2])
        return out

    def test_the_windows_redrive_is_build_windows(self):
        from gdn_conv_windows import build_windows

        ttnn = DescriptorTTNN(chips=2)
        piece = ttnn.tensor('piece', (1, 16, 8240), 'l1')
        history = [ttnn.tensor('h%d' % index, (1, 1, 5120)) for index in range(4)]
        kernel = CI / 'gdn_conv_windows.cpp'
        with mock.patch.dict(sys.modules, {'ttnn': ttnn}):
            served, redriven = self.programs_of(ttnn, (
                lambda: build_windows(mesh(), piece, history),
                lambda: card.served_windows_redrive(ttnn, mesh(), piece, history, 2, kernel)))
        self.assertEqual(served, redriven)
        one = DescriptorTTNN(chips=1)
        piece = one.tensor('piece', (1, 16, 8240), 'l1')
        history = [one.tensor('h%d' % index, (1, 1, 5120)) for index in range(4)]
        card.served_windows_redrive(one, mesh(1), piece, history, 1, kernel)
        self.assertEqual(sorted(one.programs()[0][2]), [((0, 0), (0, 0))])
        self.assertEqual(one.programs()[0][2][((0, 0), (0, 0))]['kernels'][0]['source'], str(kernel))

    def test_the_kv_redrive_is_ordered_cache_update(self):
        import ordered_cache

        ttnn = DescriptorTTNN(chips=2)
        cache = ttnn.tensor('cache', (2064, 2, 64, 256), 'dram', 'bf8', 'tile')
        packed = ttnn.tensor('kv', (1, 32, 32, 256), 'dram', 'bf16', 'tile')
        positions = ttnn.tensor('positions', (32,), 'dram', 'int32', 'row_major')
        pages = ttnn.tensor('pages', (32, 2052), 'dram', 'int32', 'row_major')
        kernels = dict(reader='R', writer='W', compute='C')
        with mock.patch.dict(sys.modules, {'ttnn': ttnn}):
            served, redriven = self.programs_of(ttnn, (
                lambda: ordered_cache.update(mesh(), cache, packed, positions, pages, kernels),
                lambda: card.served_kv_redrive(ttnn, mesh(), cache, packed, positions, pages, kernels, 2)))
        self.assertEqual(served, redriven)

    def test_the_copy_program_covers_every_page_once(self):
        ttnn = DescriptorTTNN(chips=1)
        source, destination = ttnn.tensor('s', (1, 16, 8240), 'l1'), ttnn.tensor('d', (258, 512), 'dram', 'int32', 'row_major')
        program = card.copy_pages_program(ttnn, mesh(1), source, destination, 2048, 258, Path('raw_pages.cpp'), 1)
        (kernel,) = program[((0, 0), (0, 0))].kernels
        ranges = sorted(tuple(args) for column in kernel.runtime_args.values() for args in column.values())
        self.assertEqual(sum(count for start, count in ranges), 258)
        covered = sorted(page for start, count in ranges for page in range(start, start + count))
        self.assertEqual(covered, list(range(258)))
        self.assertEqual(kernel.compile_time_args, [2048, 9, 1, 7, 1])
        self.assertEqual(kernel.common_runtime_args, [source.shards[0].address, destination.shards[0].address])
        with self.assertRaises(ValueError):
            card.copy_pages_program(ttnn, mesh(1), source, destination, 1000, 10, Path('raw_pages.cpp'), 1)
        self.assertEqual(card.page_count((1, 32, 8256), True), 258)
        self.assertEqual(card.page_count((2064, 2, 64, 256), True), 2064 * 32)
        self.assertEqual(card.page_count((66048, 272), False), 66048)

    def test_the_copy_kernel_text(self):
        text = (HERE / 'raw_pages.cpp').read_text(encoding='utf-8')
        self.assertIn('get_common_arg_val<uint32_t>(0)', text)
        self.assertIn('TensorAccessorArgs<1>()', text)
        self.assertIn('noc_async_write_barrier();', text)


# ---------------------------------------------------------------------------------------------
# The section drivers against a torch-only Bench.
# ---------------------------------------------------------------------------------------------

class Tensor(SimpleNamespace):
    pass


class FakeBench:
    """Tensors are their raw int16 pages. The served windows are reference_windows, the packed ones
    the trimmed copy map (or a fault); the served K/V write applies rows in row order, the chained
    one per user (and, for the controls, a fault or a race)."""

    def __init__(self, fault=None, race=True, canonicalise=False, hash_runtime_args=False, reuse_addresses=False):
        self.torch, self.vtw, self.poc, self.ttnn = torch, vtw, poc, None
        self.fault, self.race, self.canonicalise = fault, race, canonicalise
        # hash_runtime_args: a program cache keyed on the chain structure too (the capture after
        # the warm forward would compile); reuse_addresses: an allocator handing every call the
        # same addresses (a cache hit that never has to follow new ones)
        self.hash_runtime_args, self.reuse_addresses = hash_runtime_args, reuse_addresses
        self.entries, self.keys, self.calls = 0, set(), 0
        self.empties = 0
        self.replays = 0

    def io(self, label):
        return contextlib.nullcontext()

    def address(self, tensor):
        if self.reuse_addresses:
            return getattr(tensor, 'slot', 0)
        return id(tensor)

    def enter(self, key):
        if key not in self.keys:
            self.keys.add(key)
            self.entries += 1

    def synchronize(self):
        pass

    # traces: capture runs the call once (the fake has no deferred execution) and keeps it;
    # a replay re-runs it over the SAME tensors and copies the new results into the captured ones
    def capture(self, call):
        result = call()
        return dict(call=call, result=result), result

    def replay(self, trace):
        self.replays += 1
        copy_into(trace['result'], trace['call']())

    def release_trace(self, trace):
        pass

    def restage(self, tensor, matrix, poison=None):
        tensor.pages = card.tile_pages(torch, matrix, poison)

    def restage_kv(self, block, tiles, packed, positions, pages, values):
        block[0].values, block[1].values = torch.tensor(positions), torch.tensor(pages)
        for (first, last), (tile_positions, tile_pages_) in zip(((0, 32), (32, 64)), tiles):
            tile_positions.values, tile_pages_.values = block[0].values[first:last], block[1].values[first:last]
        packed.values, packed.pages = values, values.view(torch.int16).reshape(-1, 1024)

    # windows
    def tile_tensor(self, matrix, memory, poison=None):
        return Tensor(pages=card.tile_pages(torch, matrix, poison), rows=matrix.shape[0], memory=memory)

    def windows_inputs(self, case):
        width = case['width']
        hosts = dict(pieces=[], block=None)
        users = []
        block = torch.cat([card.make_rows(torch, case['piece'], 16, width, case['seed'] * 10 + user) for user in range(4)])
        hosts['block'] = block
        for user in range(case['users']):
            matrix = block[16 * user:16 * user + 16].clone()
            if case['source'] == 'served' and self.canonicalise and user % 2:
                bits = matrix.view(torch.int16)
                bits[bits == -32768] = 0
            piece = self.tile_tensor(matrix, 'l1', None if case['source'] == 'served' else card.POISON)
            history = [self.tile_tensor(card.make_rows(torch, case['history'], 1, 5120, case['seed'] * 100 + user * 4 + h),
                                        case['placement'], card.POISON) for h in range(4)]
            users.append((piece, history))
        return users, [], hosts

    def served_windows(self, piece, history):
        return [Tensor(pages=window) for window in vtw.reference_windows(piece.pages[:160], [h.pages[:160] for h in history])]

    def packed_windows(self, users, settings=None, negative=None):
        if any(piece.rows != 16 for piece, history in users) or any(
                len({h.memory for h in history}) != 1 for piece, history in users):
            raise vtw.Unsupported('refused')
        self.enter((vtw.settings_name(settings), negative))
        out = []
        for index, (piece, history) in enumerate(users):
            source = users[(index + 1) % len(users)][0] if negative == 'user' else piece
            windows = vtw.apply_copies(source.pages[:160], [h.pages[:160] for h in history])
            if negative == 'slot':
                windows = windows[1:] + windows[:1]
            if negative == 'hist':
                windows = [window.roll(1, 0) for window in windows]
            if negative == 'pad':
                for window in windows:
                    window[:, 512:] = -1
            if self.fault is not None and vtw.settings_name(settings) == self.fault:
                windows[0] = windows[0].clone()
                windows[0][3, 7] ^= 1
            out.append([Tensor(pages=window, slot=100 + 4 * index + slot) for slot, window in enumerate(windows)])
        return out

    def window_pages(self, tensor):
        return tensor.pages[:160]

    def input_pages(self, tensor):
        return tensor.pages

    def program_entries(self):
        return self.entries

    def release(self, *tensors):
        pass

    # kv
    def kv_initial(self, width, seed):
        from ordered_cache_hw_plan import payload
        return card.kv_initial_values(torch, width, seed, payload)

    def kv_cache(self, values):
        return Tensor(values=values.clone())

    def kv_empty(self, shape):
        """ttnn.empty: whatever the allocator's pages held - different for every buffer."""
        self.empties += 1
        generator = torch.Generator().manual_seed(7000 + self.empties)
        bits = torch.randint(-32768, 32768, tuple(shape), generator=generator, dtype=torch.int32).to(torch.int16)
        return Tensor(values=bits.view(torch.bfloat16))

    def reset_cache(self, pristine, cache):
        cache.values = pristine.values.clone()

    def kv_metadata(self, positions, pages):
        block = (Tensor(values=torch.tensor(positions)), Tensor(values=torch.tensor(pages)))
        tiles = [(Tensor(values=block[0].values[first:first + 32]), Tensor(values=block[1].values[first:first + 32]))
                 for first in (0, 32)]
        return block, tiles

    def kv_input(self, values):
        return Tensor(values=values, pages=values.view(torch.int16).reshape(-1, 1024))

    def int_values(self, tensor):
        return tensor.values

    def raw(self, tensor, page_bytes):
        return tensor.pages

    def write_rows(self, cache, positions, pages, values, order):
        for row in order:
            position, table = int(positions[row]), pages[row]
            cache.values[int(table[position // 64]), :, position % 64, :] = values[row, :2]

    def served_kv(self, cache, packed, tiles):
        self.enter(('served',))
        positions = torch.cat([tile[0].values for tile in tiles])
        pages = torch.cat([tile[1].values for tile in tiles])
        self.write_rows(cache, positions, pages, packed.values, range(64))

    def chained_kv(self, cache, packed, block, tiles, spans, launch_rows=64, cb16_pages=256, negative=None):
        self.calls += 1
        self.enter(('chained', launch_rows, cb16_pages) + ((tuple(spans),) if self.hash_runtime_args else ()))
        positions, pages = block[0].values, block[1].values
        order = [row for first, last in reversed(spans) for row in range(first, last)] if self.race else list(range(64))
        values = packed.values.clone()
        if negative == 'index':
            values[[0, 1]] = values[[1, 0]]
        if negative == 'nochain' and self.race:
            # unchained rows of one tile row race: two read-modify-writes of the same stale tile,
            # the last one to land wins and the other row is lost
            order = [row for row in range(64) if row % 2 == 0]
        self.write_rows(cache, positions, pages, values, order)

    def cache_pages(self, cache):
        return cache.values.view(torch.int16).reshape(-1, 1024).clone()

    def cache_values(self, cache):
        return cache.values


def copy_into(target, value):
    """A replay's results into the captured outputs, tensor by tensor (what the trace's baked
    addresses mean)."""
    if isinstance(target, (list, tuple)):
        for mine, theirs in zip(target, value, strict=True):
            copy_into(mine, theirs)
    elif target is not None:
        for name in ('pages', 'values'):
            if hasattr(value, name):
                setattr(target, name, getattr(value, name))


def run_section(name, bench, **overrides):
    report = dict(failures=[], sections_completed=[], windows_cases=[], windows_refusals=[], windows_negative={},
                  kv_cases=[], kv_negative={}, timing={})
    args = SimpleNamespace(seeds=[0], quick=True, kv_widths=[516], soak=1, iters=2, host_calls=3)
    for key, value in overrides.items():
        setattr(args, key, value)
    with contextlib.redirect_stdout(io.StringIO()):
        card.SECTION_DRIVERS[name](bench, args, report)
    report['sections_completed'].append(name)
    return card.verdict(report), report


class SectionTests(unittest.TestCase):
    def test_an_exact_candidate_passes_the_windows_matrix_and_the_refusals(self):
        passed, report = run_section('windows', FakeBench())
        self.assertTrue(passed, report['failures'])
        self.assertEqual(report['cases_run'], len(card.windows_matrix([0], quick=True)))
        self.assertTrue(all(entry['r1_r2'] and entry['inputs_unchanged'] for entry in report['windows_cases']))
        self.assertEqual([entry['case'] for entry in report['windows_refusals']], ['rows8', 'rows32', 'mixed_history'])
        self.assertTrue(all(entry['refused'] for entry in report['windows_refusals']))
        self.assertEqual(report['windows_exact_by_setting'], {'port': True, 'hist1_noc1_nbuf1': True})

    def test_a_setting_that_flips_one_bit_fails_by_name(self):
        passed, report = run_section('windows', FakeBench(fault='hist1_noc1_nbuf1'))
        self.assertFalse(passed)
        self.assertEqual(report['windows_exact_by_setting'], {'port': True, 'hist1_noc1_nbuf1': False})
        self.assertTrue(all('hist1_noc1_nbuf1' in failure for failure in report['failures']))

    def test_the_slice_canonicalisation_is_recorded_not_judged(self):
        passed, report = run_section('windows', FakeBench(canonicalise=True))
        self.assertTrue(passed, report['failures'])
        users = {(entry['case'], entry['user']) for entry in report['slice_canonicalises']}
        self.assertTrue(users)
        self.assertTrue(all(user in (1, 3) for case, user in users))
        zeros = [entry for entry in report['slice_canonicalises'] if 'pzeros' in entry['case']]
        self.assertTrue(zeros and not all(entry['identical'] for entry in zeros))

    def test_every_negative_control_must_differ(self):
        passed, report = run_section('windows_negative', FakeBench())
        self.assertTrue(passed, report['failures'])
        self.assertEqual(sorted(report['windows_negative']), sorted(card.NEGATIVES))

        class Blind(FakeBench):
            def packed_windows(self, users, settings=None, negative=None):
                return FakeBench.packed_windows(self, users, settings)

        passed, report = run_section('windows_negative', Blind())
        self.assertFalse(passed)
        self.assertEqual(len(report['failures']), len(card.NEGATIVES))

    def test_the_kv_matrix_passes_with_disjoint_chains_and_predicts_the_bf8_cases(self):
        passed, report = run_section('kv', FakeBench())
        self.assertTrue(passed, report['failures'])
        self.assertEqual(report['cases_run'], len([case for case in card.kv_matrix([516], [0], quick=True)]))
        self.assertTrue(all(entry.get('r2', True) for entry in report['kv_cases']))
        self.assertTrue(any('r2' in entry for entry in report['kv_cases']))

    def test_the_kv_controls(self):
        passed, report = run_section('kv_negative', FakeBench())
        self.assertTrue(passed, report['failures'])
        self.assertEqual(sorted(report['kv_negative']), sorted(card.KV_NEGATIVES))
        self.assertEqual(len(report['kv_negative']['nochain']['repeats']), 5)
        self.assertEqual(len(report['kv_negative']['index']['repeats']), 1)
        passed, report = run_section('kv_negative', FakeBench(race=False))
        self.assertFalse(passed)
        self.assertEqual(sorted(failure.split()[3] for failure in report['failures']), ['conflict', 'nochain'])

    def test_the_g4_cases_run_the_warm_forwards_one_chain(self):
        passed, report = run_section('kv', FakeBench(), kv_widths=[516, 2052])
        self.assertTrue(passed, report['failures'])
        chains = {(entry['geometry'], entry['chains']) for entry in report['kv_cases']}
        self.assertEqual(chains, {('g1', 4), ('g2', 4), ('g3', 4), ('g4', 1)})
        self.assertTrue(any(entry['geometry'] == 'g4' and entry.get('r2') for entry in report['kv_cases']))

    def test_kv_state_starts_both_caches_from_the_pristine_pages(self):
        bench = FakeBench()
        initial, pristine, old, new = card.kv_state(bench, 516, 0)
        self.assertTrue(torch.equal(bench.cache_pages(old), bench.cache_pages(pristine)))
        self.assertTrue(torch.equal(bench.cache_pages(new), bench.cache_pages(pristine)))

    def test_the_kv_trace_and_cache_sections_pass_on_uninitialised_buffers(self):
        """ttnn.empty's pages differ per buffer: a section comparing whole caches must start both
        from the pristine copy, or an exact kernel reports a mismatch."""
        for name in ('kv_trace', 'kv_cache'):
            with self.subTest(section=name):
                passed, report = run_section(name, FakeBench())
                self.assertTrue(passed, report['failures'])
        passed, report = run_section('kv_trace', FakeBench())
        self.assertEqual([entry['exact'] for entry in report['kv_trace']], [True] * 3)
        self.assertEqual([entry['table'] for entry in report['kv_trace']], [0, 1, 0])

    def test_the_kv_trace_sees_a_wrong_chain(self):
        class Wrong(FakeBench):
            def chained_kv(self, cache, packed, block, tiles, spans, launch_rows=64, cb16_pages=256, negative=None):
                FakeBench.chained_kv(self, cache, packed, block, tiles, spans, launch_rows, cb16_pages, negative='index')

        passed, report = run_section('kv_trace', Wrong())
        self.assertFalse(passed)
        self.assertTrue(all(failure.startswith('kv trace replay') for failure in report['failures']))

    def test_the_warm_chain_then_the_capture_chains_must_be_one_program(self):
        passed, report = run_section('kv_cache', FakeBench())
        self.assertTrue(passed, report['failures'])
        self.assertEqual(report['kv_cache']['warm_then_capture'], [1, 0])
        passed, report = run_section('kv_cache', FakeBench(hash_runtime_args=True))
        self.assertFalse(passed)
        self.assertEqual(report['kv_cache']['warm_then_capture'], [1, 1])
        self.assertTrue(any('would compile inside its trace' in failure for failure in report['failures']))

    def test_the_windows_trace_checks_two_launches_in_one_trace(self):
        bench = FakeBench()
        passed, report = run_section('windows_trace', bench)
        self.assertTrue(passed, report['failures'])
        self.assertEqual([entry['launches'] for entry in report['windows_trace']], [[True, True]] * 4)
        self.assertEqual(bench.replays, 4)

        class Stale(FakeBench):
            def replay(self, trace):
                self.replays += 1                   # the captured outputs never change

        passed, report = run_section('windows_trace', Stale())
        self.assertFalse(passed)
        self.assertTrue(all('differs from the served launch' in failure for failure in report['failures']))

    def test_the_windows_cache_holds_every_call_so_each_hit_takes_new_addresses(self):
        passed, report = run_section('windows_cache', FakeBench())
        self.assertTrue(passed, report['failures'])
        self.assertEqual(report['windows_cache']['new_entries_per_call'], [1, 0, 0])
        self.assertEqual(report['windows_cache']['fresh_addresses'], [True, True, True])
        passed, report = run_section('windows_cache', FakeBench(reuse_addresses=True))
        self.assertFalse(passed)
        self.assertTrue(any('reused an earlier call' in failure for failure in report['failures']))

    def test_the_timing_section_wiring(self):
        def timed(bench, call, launches, iterations):
            call()
            return dict(median=10.0, min=10.0, samples=iterations)

        with mock.patch.object(card, 'trace_timed', timed), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('TT_METAL_DEVICE_PROFILER', None)
            passed, report = run_section('timing', FakeBench())
        self.assertTrue(passed, report['failures'])
        timing = report['timing']
        self.assertEqual(timing['settings_verdict']['verdict'], 'keep')
        self.assertTrue(timing['port_gate_met'] and timing['defaults_gate_met'] and timing['chained_gate_met'])
        self.assertEqual(timing['host_us_per_call']['calls'], 3)

    def test_trace_timed_captures_the_launches_and_replays_warm_then_timed(self):
        bench = FakeBench()
        calls = []
        result = card.trace_timed(bench, lambda: calls.append(1), 4, 3)
        self.assertEqual(result['samples'], 3)
        self.assertEqual(len(calls), 4 + 4 + 4 * (1 + 3), 'warm, capture, one warm replay and three timed')


class RunScriptTests(unittest.TestCase):
    def test_the_qualification_card_the_a5_image_and_the_refusal(self):
        text = script_text()
        self.assertNotIn('CARD_M=', text)
        self.assertIn('QUAL_CARD_B=blackhole-F36F768B9A5CAFA0', text)          # the embedded qual_card.sh block
        self.assertIn('R=${RESULTS:-$HOME/kwork64/vt2/$QUAL_TAG}', text)
        self.assertIn('IMAGE=${IMAGE:-sha256:126b30dfa72b0e008884f3a8a1cfcb5b7f79eadcdeaeda1cd0f91350dde6ee73}', text)
        self.assertIn('--device "$node"', text)
        self.assertEqual(text.count('--device '), 1)
        self.assertIn('\nqual_card_resolve\nnode=$QUAL_NODE\nqual_refuse_holders\n', text)
        self.assertLess(text.index('\nqual_refuse_holders\n'), text.index('docker run --rm'))
        self.assertIn('qual_reset_hint >&2', text[text.index('docker run --rm'):])
        self.assertNotIn('tt-smi -r', [line.strip() for line in text.splitlines()
                                       if not line.lstrip().startswith(('#', 'echo', '"'))])
        self.assertIn('--network none', text)
        self.assertIn('-e TT_METAL_CACHE=/kcache', text)

    def test_the_table_is_mounted_file_by_file_and_the_references_are_checked(self):
        text = script_text()
        self.assertIn('t.RUNTIME_FILES', text)
        self.assertIn('OM+=(--mount "type=bind,src=$REPO/scripts/ci/$file,dst=/bench/vt2/$file,readonly")', text)
        self.assertNotRegex(text, r'dst=/bench/vt2[,"]')
        self.assertNotRegex(text, r'dst=/experiment-scripts')
        loop = text[text.index('for reference in '):]
        loop = loop[len('for reference in '):loop.index('; do')].replace(chr(92) + chr(10), ' ')
        self.assertEqual(loop.split(), list(card.REFERENCE_FILES), 'the script checks exactly the harness table')
        # the image modules the code under test imports, beside the references themselves
        self.assertLessEqual({'gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'ordered_cache.py', 'packed_cache_writer.py',
                              'gdn_multitoken_conv.py', 'gdn_user_batch.py', 'gdn_device_loop_state.py',
                              'verify_trace_t1.py'}, set(card.REFERENCE_FILES))
        for name in card.REFERENCE_FILES:
            self.assertTrue((CI / name).is_file(), name)
        for helper in ('verify_t2_card_m.py', 'raw_pages.cpp', 'ordered_cache_hw_plan.py'):
            self.assertIn(helper, text)

    def test_the_watcher_pass(self):
        text = script_text()
        block = text[text.index('if [ "${WATCHER:-}" = "1" ]; then'):]
        block = block[:block.index(chr(10) + 'else' + chr(10))]
        self.assertIn('timeout_s=900', block)
        self.assertIn('--quick --no-timing --seeds 0', block)
        self.assertIn('-e TT_METAL_WATCHER=5', block)

    def test_bash_parses_it_and_the_table_snippet_yields_the_four_files(self):
        bash = shutil.which('bash')
        if bash is None or shutil.which('python3') is None:
            self.skipTest('no bash / python3')
        result = subprocess.run([bash, '-n', str(SCRIPT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = [l for l in script_text().splitlines() if l.startswith('mapfile -t op_files')][0]
        with tempfile.TemporaryDirectory() as directory:
            ci = Path(directory, 'scripts', 'ci')
            ci.mkdir(parents=True)
            shutil.copy(CI / 'verify_trace_t2.py', ci / 'verify_trace_t2.py')
            script = ('set -euo pipefail' + chr(10) + 'REPO=' + Path(directory).as_posix() + chr(10) + line + chr(10)
                      + 'printf "RESULT|%s" "${op_files[*]}"' + chr(10))
            result = subprocess.run([bash, '-c', script], capture_output=True, text=True, timeout=120, cwd=directory,
                                    env=dict(os.environ))
        if result.returncode != 0 and ('No module named' in result.stderr or 'No such file' in result.stderr
                                       or 'not found' in result.stderr):
            self.skipTest('python3 here cannot see the temp tree: %s' % result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split('RESULT|')[-1].split(), list(t2.RUNTIME_FILES))


if __name__ == '__main__':
    unittest.main()
