"""CPU tests for draft_slide_inplace_card_b.py (M-F0, round-fence plan S0.1) and run_card_b.sh.

    py -3.11 -B -m unittest discover -s optimisation/ttnn-op/draft_slide_inplace -p 'test_*.py'

Three layers:
  mirrors   the S0.4 argument. Python mirrors of both slide kernels' tile loops are pinned to the kernel
            sources by hash. Out of place and IN PLACE they give the oracle for history 2048 x prefixes
            1..16 and 32, the rows == 2048 entry edges and partial histories. A page-level check finds no
            bank page read after its worker wrote it, for any worker, history and prefix 1..32. The
            negative control (the tiles walked backwards) is caught by both checks and is harmless out
            of place.
  programs  the harness's program is draft_kv_slide.prepare's per-chip program field for field; the
            in-place and multi-bank programs differ only where they must; the refusals hold.
  flow      the whole harness on a fake device whose generic_op runs the mirrors from the runtime args.
            It passes; it moves to the pair form when duplicate io tensors are refused; and it fails on an
            in-place-unsafe kernel, on a program cache that keeps stale runtime args, and on an unknown
            served kernel.
"""

import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CI = ROOT / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))

import torch  # noqa: E402

import draft_slide_inplace_card_b as card  # noqa: E402
import draft_kv_slide  # noqa: E402
import draft_kv_slide_direct  # noqa: E402

SCRIPT = HERE / 'run_card_b.sh'
CARD_M = 'blackhole-CEF5729692C19E6D'
HISTORIES = (1, 31, 32, 33, 1000, 2016, 2017, 2032, 2033, 2040, 2047, 2048)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slide(kind, active, delta, history, prefix, in_place, order='ascending'):
    """Mirror one bank; returns the output bits (the bank itself when in place)."""
    active = active.clone()
    out = active if in_place else torch.full(card.KV_SHAPE, 0x1234, dtype=torch.int16)
    card.mirror_slide(torch, kind, active, delta.clone(), out, history_rows=history, prefix=prefix, order=order)
    return out


# ---------------------------------------------------------------------------------------------
# Sources: the mirrors are only as good as their pins.
# ---------------------------------------------------------------------------------------------

class SourcePinTests(unittest.TestCase):
    def test_the_mirrors_are_pinned_to_both_qualified_kernels(self):
        # A kernel change fails here: re-review the mirror and the in-place argument, then re-pin.
        self.assertEqual(sha(CI / 'draft_kv_slide.cpp'), card.SCALAR_SHA256)
        self.assertEqual(sha(CI / 'draft_kv_slide_direct.cpp'), card.DIRECT_SHA256)
        import draft_kv_slide_gate
        self.assertEqual(set(draft_kv_slide_gate.QUALIFIED_KERNELS.values()), set(card.KERNEL_KINDS))

    def test_the_served_driver_is_the_bundles_and_the_served_kernel_is_the_direct_one(self):
        self.assertEqual(sha(CI / 'draft_kv_slide.py'), card.BUNDLE_DRIVER_SHA256)
        self.assertEqual(card.BUNDLE_KERNEL_SHA256, card.DIRECT_SHA256)
        self.assertEqual(card.KERNEL_KINDS[card.BUNDLE_KERNEL_SHA256], 'direct')

    def test_no_image_copy_list_overrides_the_bundle_slide(self):
        # Why every image carries the bundle's draft_kv_slide.{py,cpp}: nothing copies the checkout's.
        for path in (ROOT / 'docker' / 'qwen-fast-serving.Dockerfile',
                     ROOT / '.github' / 'workflows' / 'qwen-fast-serving-image.yml'):
            with self.subTest(path=path.name):
                self.assertNotIn('draft_kv_slide', path.read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------------------------

class HelperTests(unittest.TestCase):
    def test_geometry_is_the_drivers(self):
        for history in HISTORIES:
            for prefix in range(1, 33):
                self.assertEqual(card.geometry(history, prefix), draft_kv_slide.geometry(history, prefix))
        with self.assertRaises(ValueError):
            card.geometry(2048, 33)

    def test_the_oracle_is_the_drivers_row_map(self):
        active = card.random_bits(torch, card.KV_SHAPE, 3)
        delta = card.random_bits(torch, card.DELTA_SHAPE, 4)
        for history, prefix in ((2048, 1), (2048, 16), (2047, 2), (31, 2), (2016, 32)):
            expected = card.oracle(torch, active, delta, history, prefix)
            for row in (0, 1, 30, 31, 32, 1000, 2015, 2016, 2031, 2032, 2046, 2047):
                source, index = draft_kv_slide.row_source(history, prefix, row)
                want = {'zero': torch.zeros_like(expected[:, :, 0]), 'active': active[:, :, index] if source == 'active'
                        else None, 'delta': delta[:, :, index] if source == 'delta' else None}[source]
                self.assertTrue(torch.equal(expected[:, :, row], want), (history, prefix, row, source))

    def test_random_bits_are_finite_normal_bf16_and_seeded(self):
        bits = card.random_bits(torch, (4, 1024), 7)
        exponent = (bits.to(torch.int32) >> 7) & 0xFF
        self.assertTrue(bool(((exponent >= 1) & (exponent <= 254)).all()))
        self.assertTrue(bool(torch.isfinite(bits.view(torch.bfloat16)).all()))
        self.assertTrue(bool((bits < 0).any()) and bool((bits > 0).any()))
        self.assertTrue(torch.equal(bits, card.random_bits(torch, (4, 1024), 7)))
        self.assertFalse(torch.equal(bits, card.random_bits(torch, (4, 1024), 8)))
        self.assertFalse(bool((bits == card.STALE).any()))

    def test_direct_segments_are_the_direct_oracle(self):
        for history in HISTORIES:
            for prefix in (1, 2, 7, 15, 16, 17, 31, 32):
                shape = card.geometry(history, prefix)
                for tile in range(card.TILES):
                    self.assertEqual(card.direct_segments(history, prefix, shape['drop'], shape['rows'], tile),
                                     draft_kv_slide_direct.segments(history, prefix, tile), (history, prefix, tile))

    def test_io_forms_and_layout_names(self):
        a, d, b, e = 'a', 'd', 'b', 'e'
        self.assertEqual(card.io_list('aliased', [(a, d), (b, e)]), [a, d, a, b, e, b])
        self.assertEqual(card.io_list('pair', [(a, d), (b, e)]), [a, d, b, e])
        self.assertEqual(card.io_list('pair_out_last', [(a, d)]), [d, a])
        self.assertEqual(card.served_io([(a, d, b)]), [a, d, b])
        with self.assertRaises(ValueError):
            card.io_list('both', [(a, d)])
        self.assertEqual([card.layout_name(n) for n in card.LAYOUTS], ['10x16', '5x32', '2x80', '1x160'])
        self.assertEqual(card.groups(list(range(10)), 5), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    def test_compare_locates_the_tile(self):
        base = card.random_bits(torch, card.KV_SHAPE, 5)
        other = base.clone()
        other[0, 2, 40 * 32 + 3, 100] ^= 1
        result = card.compare(torch, other, base)
        self.assertEqual((result['exact'], result['mismatches'], result['first'], result['tiles']),
                         (False, 1, [0, 2, 40 * 32 + 3, 100], [[2, 40]]))
        self.assertTrue(card.compare(torch, base, base.clone())['exact'])

    def test_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory, 'r.json'))
            args = card.parse_args(['--out', out])
            self.assertEqual(args.prefixes, list(range(1, 17)))
            self.assertEqual(args.chips, [0, 1])
            self.assertEqual(args.kernels, ['served', 'scalar', 'direct'])
            self.assertEqual(len(card.case_matrix(args)), (16 + len(card.EDGES)) * 2)   # the plan's 32, plus edges
            self.assertEqual(args.trace_layouts, list(card.LAYOUTS))   # every timed layout can be verified
            quick = card.parse_args(['--out', out, '--quick'])
            self.assertEqual((quick.prefixes, quick.chips, quick.layouts), ([1, 2, 15, 16], [0], [1, 5]))
            self.assertEqual((quick.edges, quick.iters, quick.warmup, quick.trace_replays),
                             ([(2047, 2), (2047, 1)], 5, 1, 1))
            # WATCHER=1 passes --quick, then CARD_B_ARGS: an explicit argument must beat the quick default.
            widened = card.parse_args(['--out', out, '--quick', '--prefixes', '3,4', '--chips', '0,1', '--iters', '9'])
            self.assertEqual((widened.prefixes, widened.chips, widened.iters, widened.layouts),
                             ([3, 4], [0, 1], 9, [1, 5]))
            for history, prefix in card.EDGES:
                self.assertEqual(card.geometry(history, prefix)['rows'], 2048, (history, prefix))
            self.assertIn(0, [card.geometry(h, p)['drop'] for h, p in card.EDGES])   # the window's first fill
            for bad in (['--prefixes', '17'], ['--edges', '2000:16'], ['--layouts', '3'], ['--kernels', 'scalar,served'],
                        ['--sections', 'cases,bogus']):
                with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    card.parse_args(['--out', out] + bad)

    @staticmethod
    def full_report():
        cases = [dict(kernel='served', history=2048, prefix=prefix, chip=chip, inplace_vs_oop={}, all_exact=True)
                 for prefix in range(1, 17) for chip in (0, 1)]
        layouts = ('10x16', '5x32', '2x80')
        return dict(failures=[], cases=cases,
                    multibank=[dict(kernel='served', layout=layout, prefix=16, all_exact=True) for layout in layouts],
                    trace=[dict(kernel='served', layout=layout, all_exact=True, replays=2) for layout in layouts],
                    cache=dict(ok=True), form='aliased',
                    forms={'aliased': dict(accepted=True, exact=True)}, served=dict(kind='direct'),
                    timing=dict(variants={
                        'inplace_2x80': dict(mode='inplace', layout='2x80', traced=dict(pipelined_ms=0.31)),
                        'inplace_10x16': dict(mode='inplace', layout='10x16', traced=dict(pipelined_ms=1.1)),
                        'oop_10x16': dict(mode='oop', layout='10x16', traced=dict(pipelined_ms=0.2))}))

    def test_the_decision(self):
        report = self.full_report()
        decision = card.decide(report)
        self.assertTrue(decision['go'], decision['why_not'])
        self.assertEqual(decision['why_not'], [])
        self.assertTrue(decision['full_matrix'])
        self.assertEqual(decision['best_inplace'], dict(variant='inplace_2x80', ms_per_user=0.31))
        self.assertEqual((decision['cases'], decision['cases_identical']), (32, 32))
        slow = json.loads(json.dumps(report))
        slow['timing']['variants']['inplace_2x80']['traced']['pipelined_ms'] = 0.6
        self.assertFalse(card.decide(slow)['go'])
        broken = json.loads(json.dumps(report))
        broken['cases'][3]['all_exact'] = False
        self.assertFalse(card.decide(broken)['go'])
        self.assertIn('fallback', card.decide(broken)['h1b'])
        uncached = json.loads(json.dumps(report))
        uncached['cache']['ok'] = False
        self.assertFalse(card.decide(uncached)['go'])

    def test_no_go_from_a_narrowed_run_a_failure_or_an_unverified_layout(self):
        narrowed = self.full_report()                  # a --quick or --prefixes run: exact, fast, but thin
        narrowed['cases'] = [case for case in narrowed['cases'] if case['prefix'] in (1, 16) and case['chip'] == 0]
        decision = card.decide(narrowed)
        self.assertFalse(decision['go'])
        self.assertFalse(decision['full_matrix'])
        self.assertIn([2, 0], decision['missing'])
        self.assertTrue(any('matrix incomplete' in reason for reason in decision['why_not']))
        scalar_only = self.full_report()               # the served kernel never ran (missing from the image)
        for case in scalar_only['cases']:
            case['kernel'] = 'scalar'
        self.assertFalse(card.decide(scalar_only)['go'])
        failed = self.full_report()                    # e.g. an unknown served kernel, or a wrong io form
        failed['failures'] = ['served kernel ... is neither qualified slide kernel']
        self.assertFalse(card.decide(failed)['go'])
        untraced = self.full_report()                  # 2x80 fastest but never trace-verified: use 10x16's time
        untraced['trace'] = [entry for entry in untraced['trace'] if entry['layout'] != '2x80']
        decision = card.decide(untraced)
        self.assertEqual(decision['best_inplace'], dict(variant='inplace_10x16', ms_per_user=1.1))
        self.assertEqual(decision['fastest_inplace'], dict(variant='inplace_2x80', ms_per_user=0.31, verified=False))
        self.assertFalse(decision['go'])
        wrong_trace = self.full_report()
        wrong_trace['trace'][2]['all_exact'] = False
        self.assertFalse(card.decide(wrong_trace)['go'])


class WatchdogTests(unittest.TestCase):
    def test_each_op_arms_the_faulthandler_backstop_and_nesting_rearms_the_outer(self):
        # The poll thread needs the GIL; a blocking ttnn call may hold it, so a C-thread backstop must be
        # armed for every op (budget + grace), re-armed for the outer op's remainder, cancelled at the end.
        with mock.patch.object(card.faulthandler, 'dump_traceback_later') as arm, \
                mock.patch.object(card.faulthandler, 'cancel_dump_traceback_later') as cancel:
            watchdog = card.Watchdog(100, grace=60)
            with watchdog.op('open device', extra=300):
                self.assertEqual(arm.call_args_list[-1], mock.call(460, exit=True, file=sys.stdout))
                with watchdog.op('synchronize'):
                    self.assertEqual(arm.call_args_list[-1][0][0], 160)
                self.assertEqual(arm.call_count, 3)
                self.assertTrue(400 < arm.call_args_list[-1][0][0] <= 460)   # the outer op's remainder + grace
                self.assertEqual(watchdog.label, 'open device')
            cancel.assert_called_once_with()
            self.assertIsNone(watchdog.label)
            arm.reset_mock()
            cancel.reset_mock()
            with card.Watchdog(0).op('off'):
                pass
            arm.assert_not_called()
            cancel.assert_not_called()

    def test_a_timed_loop_is_one_span_with_no_per_launch_arming(self):
        # Arming restarts faulthandler's thread (~75 us per arm and cancel): 20 per eager iteration of
        # 10 launches would add ~0.7 ms per user to a 0.5 ms threshold.
        with mock.patch.object(card.faulthandler, 'dump_traceback_later') as arm, \
                mock.patch.object(card.faulthandler, 'cancel_dump_traceback_later') as cancel:
            watchdog = card.Watchdog(100, grace=60)
            with watchdog.span('timing inplace_2x80', card.TIMING_SPAN_S):
                self.assertEqual(arm.call_args_list[-1][0][0], card.TIMING_SPAN_S + 60)
                for _ in range(20):
                    with watchdog.op('timing inplace_2x80'):
                        pass
                self.assertEqual(arm.call_count, 1)
                self.assertEqual(watchdog.label, 'timing inplace_2x80')
            cancel.assert_called_once_with()
            self.assertFalse(watchdog.coarse)
            self.assertIsNone(watchdog.label)
        source = inspect.getsource(card.section_timing)
        self.assertIn('with WATCHDOG.span(', source)
        begin = source.index('with WATCHDOG.span(')
        span = source[begin:source.index('variants[name] = entry', begin)]
        for measured in ('enqueue()', 'bench.sync()', 'bench.capture(enqueue)', 'bench.replay(trace, blocking=False)'):
            self.assertIn(measured, span)


# ---------------------------------------------------------------------------------------------
# Mirrors: the in-place argument (S0.4).
# ---------------------------------------------------------------------------------------------

class MirrorTests(unittest.TestCase):
    CASES = [(2048, prefix) for prefix in list(range(1, 17)) + [32]] + list(card.EDGES) + [
        (31, 2), (1000, 5), (2016, 32)]

    def test_both_kernels_give_the_oracle_out_of_place_and_in_place(self):
        active = card.random_bits(torch, card.KV_SHAPE, 11)
        delta = card.random_bits(torch, card.DELTA_SHAPE, 12)
        for kind in ('scalar', 'direct'):
            for history, prefix in self.CASES:
                with self.subTest(kind=kind, history=history, prefix=prefix):
                    expected = card.oracle(torch, active, delta, history, prefix)
                    self.assertTrue(torch.equal(slide(kind, active, delta, history, prefix, False), expected))
                    self.assertTrue(torch.equal(slide(kind, active, delta, history, prefix, True), expected))

    def test_the_backwards_walk_is_caught_in_place_and_harmless_out_of_place(self):
        active = card.random_bits(torch, card.KV_SHAPE, 13)
        delta = card.random_bits(torch, card.DELTA_SHAPE, 14)
        for kind in ('scalar', 'direct'):
            for prefix in (1, 16):
                with self.subTest(kind=kind, prefix=prefix):
                    expected = card.oracle(torch, active, delta, 2048, prefix)
                    self.assertTrue(torch.equal(slide(kind, active, delta, 2048, prefix, False, 'descending'), expected))
                    wrong = slide(kind, active, delta, 2048, prefix, True, 'descending')
                    result = card.compare(torch, wrong, expected)
                    self.assertFalse(result['exact'])
                    self.assertGreater(result['mismatches'], 0)

    def test_no_bank_page_is_read_after_its_worker_wrote_it(self):
        for kind in ('scalar', 'direct'):
            for history in HISTORIES:
                for prefix in range(1, 33):
                    for worker in (0, 5, 10, 15):
                        self.assertEqual(card.in_place_hazards(kind, history, prefix, worker), [],
                                         (kind, history, prefix, worker))

    def test_the_page_check_sees_the_backwards_walk(self):
        for kind in ('scalar', 'direct'):
            for prefix in (1, 16, 32):
                hazards = card.in_place_hazards(kind, 2048, prefix, 7, order='descending')
                self.assertTrue(hazards, (kind, prefix))

    def test_workers_touch_only_their_own_head_and_column(self):
        for kind in ('scalar', 'direct'):
            for history, prefix in ((2048, 16), (2047, 2), (31, 2)):
                seen = {}
                for worker in range(card.WORKERS):
                    head, column = divmod(worker, card.COLUMNS)
                    pages = set()
                    for tile, reads, write in card.worker_accesses(kind, history, prefix, worker):
                        for source, page in reads:
                            if source == 'bank':
                                pages.add(page)
                            else:
                                self.assertEqual(page, head * card.COLUMNS + column)
                        if write is not None:
                            pages.add(write)
                            self.assertEqual(write, (head * card.TILES + tile) * card.COLUMNS + column)
                    for page in pages:
                        self.assertEqual((page % card.COLUMNS, page // card.COLUMNS // card.TILES), (column, head))
                        self.assertNotIn(page, seen, (kind, worker, seen.get(page)))
                        seen[page] = worker
                self.assertEqual(len([p for p, w in seen.items()]), card.WORKERS * card.TILES)

    def test_the_kernels_agree_with_each_other(self):
        active = card.random_bits(torch, card.KV_SHAPE, 21)
        delta = card.random_bits(torch, card.DELTA_SHAPE, 22)
        for history, prefix in ((2048, 9), (2033, 16)):
            self.assertTrue(torch.equal(slide('scalar', active, delta, history, prefix, True),
                                        slide('direct', active, delta, history, prefix, True)))


# ---------------------------------------------------------------------------------------------
# A fake ttnn: descriptors as data, generic_op running the mirrors, a program cache, traces.
# ---------------------------------------------------------------------------------------------

class Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Coord:
    def __init__(self, x, y):
        self.x, self.y = x, y


class Runtime(dict):
    def __getitem__(self, x):
        return self.setdefault(x, {})


class MeshProgram(dict):
    pass


class FakeTensor:
    def __init__(self, fake, value, on_device, accessor=(1, 0)):
        self.fake, self.value, self.accessor = fake, value, list(accessor)
        self.dtype, self.layout = fake.bfloat16, fake.TILE_LAYOUT
        self.tile = Record(tile_shape=(32, 32), transpose_of_faces=False, transpose_within_face=False)
        self.address = fake.allocate(self) if on_device else None

    @property
    def shape(self):
        return tuple(self.value.shape)

    def memory_config(self):
        return self.fake.DRAM_MEMORY_CONFIG

    def buffer_address(self):
        return self.address


class TwoChip:
    """A replicated two-chip tensor for draft_kv_slide.prepare."""

    def __init__(self, shards):
        self.shards = shards
        self.shape = shards[0].shape


class FakeDevice:
    def __init__(self, fake):
        self.fake = fake

    def compute_with_storage_grid_size(self):
        return Record(x=self.fake.grid[0], y=self.fake.grid[1])

    def num_program_cache_entries(self):
        return len(self.fake.cache)

    def enable_program_cache(self):
        pass


class FakeTtnn:
    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'
    DataMovementProcessor = Record(RISCV_0='riscv0', RISCV_1='riscv1')
    NOC = Record(RISCV_0_default='noc0', RISCV_1_default='noc1')

    def __init__(self, grid=(11, 10), refuse_duplicates=False, stale_cache=False, order='ascending', kinds=None):
        self.grid, self.refuse_duplicates, self.stale_cache, self.order = grid, refuse_duplicates, stale_cache, order
        self.kinds = dict(card.KERNEL_KINDS, **(kinds or {}))
        self.tensors, self.next, self.cache = {}, 0x100000, {}
        self.capturing, self.traces, self.trace_ids = None, {}, 0
        self.executed = 0

    def allocate(self, tensor):
        self.next += 0x400000
        self.tensors[self.next] = tensor
        return self.next

    # device
    def open_device(self, device_id, trace_region_size=0):
        return FakeDevice(self)

    def close_device(self, device):
        pass

    def synchronize_device(self, device):
        pass

    # tensors
    def from_torch(self, value, dtype, layout, device=None, memory_config=None):
        return FakeTensor(self, value.clone(), device is not None)

    def to_torch(self, tensor):
        return tensor.value.clone()

    def copy_host_to_device_tensor(self, host, device_tensor):
        device_tensor.value.copy_(host.value)

    def get_device_tensors(self, tensor):
        return list(tensor.shards) if isinstance(tensor, TwoChip) else [tensor]

    def deallocate(self, tensor):
        pass            # freed DRAM stays addressable (a stale program still writes it); addresses never recur

    def TensorAccessorArgs(self, tensor):
        return Record(get_compile_time_args=lambda: list(tensor.accessor))

    # descriptors
    def CoreCoord(self, x, y):
        return Coord(x, y)

    def CoreRange(self, start, end):
        return ((start.x, start.y), (end.x, end.y))

    def CoreRangeSet(self, ranges):
        return tuple(ranges)

    def Tile(self, dims):
        return tuple(dims)

    def TileDescriptor(self, tile):
        return ('tile', tile)

    def CBFormatDescriptor(self, **kwargs):
        return Record(**kwargs)

    def CBDescriptor(self, **kwargs):
        return Record(**kwargs)

    def RuntimeArgs(self):
        return Runtime()

    def KernelDescriptor(self, **kwargs):
        return Record(runtime_args=None, **kwargs)

    def DataMovementConfigDescriptor(self, **kwargs):
        return Record(**kwargs)

    def ProgramDescriptor(self, kernels, cbs):
        return Record(kernels=kernels, cbs=cbs)

    def MeshProgramDescriptor(self):
        return MeshProgram()

    def MeshCoordinate(self, row, col):
        return (row, col)

    def MeshCoordinateRange(self, start, end):
        return (start, end)

    # the op: the mirrors, per core, from the runtime args
    def generic_op(self, tensors, program):
        if self.refuse_duplicates and len({id(value) for value in tensors}) != len(tensors):
            raise RuntimeError('generic_op: an io tensor is listed twice')
        calls = []
        for coordinate_range, descriptor in program.items():
            (kernel,) = descriptor.kernels
            runtime = {(x, y): list(args) for x, column in kernel.runtime_args.items() for y, args in column.items()}
            key = (kernel.kernel_source, tuple(kernel.compile_time_args), tuple(kernel.core_ranges),
                   tuple(sorted((core, len(args)) for core, args in runtime.items())),
                   tuple((value.shape, value.dtype) for value in tensors), coordinate_range)
            if key in self.cache:
                if self.stale_cache:
                    runtime = self.cache[key]         # a broken override: the first call's args forever
            else:
                self.cache[key] = runtime
            calls.append((kernel.kernel_source, runtime))
        if self.capturing is not None:
            self.capturing.extend(calls)
        else:
            for source, runtime in calls:
                self.execute(source, runtime)
        return tensors[-1]

    def execute(self, source, runtime):
        kind = self.kinds.get(sha(source))
        if kind is None:
            raise RuntimeError('fake device: no mirror for %s' % source)
        for core, args in sorted(runtime.items()):
            active, delta, out, history, prefix, drop, rows, worker = args
            shape = card.geometry(history, prefix)
            if (drop, rows) != (shape['drop'], shape['rows']):
                raise RuntimeError('runtime args disagree with the geometry: %s' % args)
            card.mirror_slide(torch, kind, self.tensors[active].value.view(torch.int16),
                              self.tensors[delta].value.view(torch.int16), self.tensors[out].value.view(torch.int16),
                              history_rows=history, prefix=prefix, workers=[worker], order=self.order)
            self.executed += 1

    # traces
    def begin_trace_capture(self, device, cq_id=0):
        self.trace_ids += 1
        self.capturing = []
        return self.trace_ids

    def end_trace_capture(self, device, trace, cq_id=0):
        self.traces[trace], self.capturing = self.capturing, None

    def execute_trace(self, device, trace, cq_id=0, blocking=True):
        for source, runtime in self.traces[trace]:
            self.execute(source, runtime)

    def release_trace(self, device, trace):
        del self.traces[trace]


def describe(program):
    """A mesh program as plain data."""
    out = {}
    for coordinate_range, descriptor in program.items():
        (kernel,) = descriptor.kernels
        (buffer,) = descriptor.cbs
        out[coordinate_range] = dict(
            source=kernel.kernel_source, cores=list(kernel.core_ranges), compile=list(kernel.compile_time_args),
            runtime={(x, y): list(args) for x, column in kernel.runtime_args.items() for y, args in column.items()},
            processor=kernel.config.processor, noc=kernel.config.noc,
            cb=dict(total=buffer.total_size, cores=list(buffer.core_ranges),
                    formats=[dict(vars(fmt)) for fmt in buffer.format_descriptors]))
    return out


def device_tensor(fake, shape, seed, accessor=(1, 0)):
    return FakeTensor(fake, card.random_bits(torch, shape, seed).view(torch.bfloat16), True, accessor)


class ProgramTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeTtnn()
        self.kernel = str(CI / 'draft_kv_slide.cpp')

    def build(self, banks, in_place, history=2048, prefix=9, grid=(11, 10)):
        return card.build_program(self.fake, grid, self.kernel, banks, history_rows=history, prefix=prefix,
                                  geometry_of=draft_kv_slide.geometry, in_place=in_place)

    def test_one_bank_out_of_place_is_the_served_drivers_program_for_each_chip(self):
        fake = self.fake
        shards = [[device_tensor(fake, shape, 100 + 10 * chip + index) for chip in range(2)]
                  for index, shape in enumerate((card.KV_SHAPE, card.DELTA_SHAPE, card.KV_SHAPE))]
        active, delta, spare = (TwoChip(parts) for parts in shards)
        captured = []
        fake.generic_op = lambda tensors, program: captured.append((tensors, program))
        mesh = FakeDevice(fake)
        with mock.patch.dict(sys.modules, {'ttnn': fake}):
            for history, prefix in ((2048, 16), (2047, 2), (31, 2)):
                captured.clear()
                draft_kv_slide.prepare(mesh, active, delta, spare, history_rows=history, prefix=prefix)()
                (tensors, served), = captured
                self.assertEqual(tensors, [active, delta, spare])
                served = describe(served)
                for chip in range(2):
                    mine = describe(self.build([tuple(parts[chip] for parts in shards)], False, history, prefix))
                    self.assertEqual(list(mine.values()), [served[((0, chip), (0, chip))]], (history, prefix, chip))

    def test_in_place_changes_only_the_output_address_and_accessor(self):
        fake = self.fake
        bank, delta = device_tensor(fake, card.KV_SHAPE, 1), device_tensor(fake, card.DELTA_SHAPE, 2)
        spare = device_tensor(fake, card.KV_SHAPE, 3)
        (oop,) = describe(self.build([(bank, delta, spare)], False)).values()
        (inplace,) = describe(self.build([(bank, delta, bank)], True)).values()
        for core, args in oop['runtime'].items():
            self.assertEqual(inplace['runtime'][core], args[:2] + [bank.address] + args[3:])
        self.assertEqual({k: v for k, v in inplace.items() if k != 'runtime'}, {k: v for k, v in oop.items() if k != 'runtime'})

    def test_a_multi_bank_program_is_disjoint_workers_per_bank(self):
        fake = self.fake
        banks = [(device_tensor(fake, card.KV_SHAPE, 10 + i), device_tensor(fake, card.DELTA_SHAPE, 20 + i)) for i in range(5)]
        (program,) = describe(self.build([(b, d, b) for b, d in banks], True)).values()
        self.assertEqual(program['cores'], [((i % 11, i // 11), (i % 11, i // 11)) for i in range(80)])
        self.assertEqual(program['cb']['cores'], program['cores'])
        for index in range(80):
            bank, delta = banks[index // 16]
            self.assertEqual(program['runtime'][(index % 11, index // 11)],
                             [bank.address, delta.address, bank.address, 2048, 9, 9, 2048, index % 16])
        self.assertEqual(program['cb']['total'], 8192)
        self.assertEqual(program['cb']['formats'][0]['page_size'], 2048)

    def test_the_refusals(self):
        fake = self.fake
        bank, delta = device_tensor(fake, card.KV_SHAPE, 1), device_tensor(fake, card.DELTA_SHAPE, 2)
        spare = device_tensor(fake, card.KV_SHAPE, 3)
        many = [(device_tensor(fake, card.KV_SHAPE, 30 + i), device_tensor(fake, card.DELTA_SHAPE, 40 + i)) for i in range(10)]
        odd = device_tensor(fake, card.KV_SHAPE, 4, accessor=(2, 0))
        wrong = FakeTensor(fake, torch.zeros((1, 4, 1024, 128), dtype=torch.bfloat16), True)
        for label, call in (
                ('10 banks on 110 cores', lambda: self.build([(b, d, b) for b, d in many], True)),
                ('aliased out of place', lambda: self.build([(bank, delta, bank)], False)),
                ('a separate output in place', lambda: self.build([(bank, delta, spare)], True)),
                ('the delta as the bank', lambda: self.build([(delta, delta, delta)], True)),
                ('different accessors', lambda: self.build([(bank, delta, bank), (odd, delta, odd)], True)),
                ('a short bank', lambda: self.build([(wrong, delta, wrong)], True))):
            with self.subTest(label=label), self.assertRaises(card.Unsupported):
                call()
        self.assertEqual(len(describe(self.build([(b, d, b) for b, d in many], True, grid=(13, 13)))), 1)


# ---------------------------------------------------------------------------------------------
# The whole harness on the fake device.
# ---------------------------------------------------------------------------------------------

THIN = ['--prefixes', '1,16', '--chips', '0,1', '--edges', '2047:2', '--layouts', '1,5', '--multibank-prefixes', '16',
        '--trace-layouts', '5', '--trace-replays', '1', '--timing-layouts', '1,5', '--iters', '1', '--warmup', '1']


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.image = root / 'image'                  # what /experiment-scripts/ci holds: the direct kernel
        self.checkout = root / 'bench'
        for path in (self.image, self.checkout / 'scalar', self.checkout / 'direct'):
            path.mkdir(parents=True)
        shutil.copy(CI / 'draft_kv_slide_direct.cpp', self.image / 'draft_kv_slide.cpp')
        shutil.copy(CI / 'draft_kv_slide.py', self.image / 'draft_kv_slide.py')
        shutil.copy(CI / 'draft_kv_slide.cpp', self.checkout / 'scalar' / 'draft_kv_slide.cpp')
        shutil.copy(CI / 'draft_kv_slide_direct.cpp', self.checkout / 'direct' / 'draft_kv_slide.cpp')
        shutil.copy(CI / 'draft_kv_slide.py', self.checkout / 'draft_kv_slide.py')
        self.out = root / 'report.json'

    def tearDown(self):
        self.directory.cleanup()

    def run_harness(self, fake, extra):
        argv = ['--out', str(self.out), '--served-dir', str(self.image), '--checkout-dir', str(self.checkout)] + extra
        with mock.patch.dict(sys.modules, {'ttnn': fake}), contextlib.redirect_stdout(io.StringIO()) as log:
            status = card.main(argv)
        return status, json.loads(self.out.read_text()), log.getvalue()

    def test_every_section_runs_and_passes(self):
        status, report, log = self.run_harness(FakeTtnn(), THIN)
        self.assertEqual(status, 0, report['failures'])
        self.assertTrue(report['passed'])
        self.assertEqual(report['served']['kind'], 'direct')
        self.assertTrue(report['served']['kernel_matches_bundle'] and report['served']['driver_matches_bundle'])
        self.assertEqual([(k['label'], k.get('kind'), k.get('skipped')) for k in report['kernels']],
                         [('served', 'direct', None), ('scalar', 'scalar', None), ('direct', 'direct', 'same bytes as served')])
        self.assertEqual(report['form'], 'aliased')
        self.assertTrue(all(entry['accepted'] and entry['exact'] for entry in report['forms'].values()))
        self.assertEqual(len(report['cases']), 2 * 3 * 2)           # kernels x (2 prefixes + 1 edge) x chips
        self.assertTrue(all(case['all_exact'] for case in report['cases']))
        self.assertEqual(sum('kernels_agree' in case for case in report['cases']), 6)
        self.assertEqual([(m['kernel'], m['layout']) for m in report['multibank']],
                         [('served', '10x16'), ('served', '2x80'), ('scalar', '10x16'), ('scalar', '2x80')])
        cache = report['cache']
        self.assertTrue(cache['ok'] and cache['fresh_addresses_honoured'] and cache['prefix_honoured'])
        self.assertTrue(cache['repeat_adds_no_entry'])
        self.assertEqual([t['all_exact'] for t in report['trace']], [True])
        variants = report['timing']['variants']
        self.assertEqual(sorted(variants), ['inplace_10x16', 'inplace_2x80', 'oop_10x16', 'oop_2x80',
                                            'served_rebuild_10x16'])
        self.assertNotIn('traced', variants['served_rebuild_10x16'])
        self.assertIsNotNone(variants['inplace_2x80']['traced']['pipelined_ms'])
        decision = report['decision']
        self.assertTrue(decision['bytes_identical'])
        self.assertEqual((decision['cases'], decision['cases_identical']), (12, 12))
        self.assertEqual(decision['verified_layouts'], ['2x80'])       # multibank 10x16, 2x80; trace 2x80
        self.assertEqual(decision['best_inplace']['variant'], 'inplace_2x80')
        self.assertFalse(decision['go'])                               # THIN is not the plan's matrix
        self.assertFalse(decision['full_matrix'])
        self.assertTrue(any('matrix incomplete' in reason for reason in decision['why_not']))
        self.assertIn('DECISION go=False', log)
        self.assertIn('not go: served matrix incomplete', log)

    def test_the_watchdog_on_every_section_with_the_backstop(self):
        with mock.patch.object(card.faulthandler, 'dump_traceback_later') as arm, \
                mock.patch.object(card.faulthandler, 'cancel_dump_traceback_later') as cancel:
            status, report, log = self.run_harness(FakeTtnn(), THIN + ['--watchdog', '100'])
        self.assertEqual(status, 0, report['failures'])
        self.assertNotIn('WATCHDOG', log)
        budgets = [call[0][0] for call in arm.call_args_list]
        self.assertEqual(budgets[0], 100 + card.OPEN_EXTRA_S + 60)              # open_device first
        self.assertEqual(budgets.count(card.TIMING_SPAN_S + 60), len(report['timing']['variants']))
        self.assertGreater(cancel.call_count, 0)
        self.assertIsNone(card.WATCHDOG.label)

    def test_a_harness_build_refusal_is_not_recorded_as_a_generic_op_refusal(self):
        # 15 cores cannot hold one bank's 16 workers: the harness refuses to build, so the run errors out
        # instead of reporting "generic_op took no in-place io form".
        with self.assertRaises(card.Unsupported):
            self.run_harness(FakeTtnn(grid=(3, 5)), ['--sections', 'forms', '--kernels', 'served'])
        report = json.loads(self.out.read_text())
        self.assertIn('transport workers', report['error'])
        self.assertEqual(report['forms'], {})
        self.assertFalse(report['passed'])

    def test_an_unknown_cache_entry_count_is_not_a_verified_cache(self):
        fake = FakeTtnn()
        device = FakeDevice(fake)
        device.num_program_cache_entries = mock.Mock(side_effect=AttributeError('no such method'))
        fake.open_device = lambda device_id, trace_region_size=0: device
        status, report, _ = self.run_harness(fake, ['--sections', 'forms,cache', '--kernels', 'served'])
        self.assertEqual(status, 1)
        self.assertFalse(report['cache']['entries_known'])
        self.assertFalse(report['cache']['ok'])
        self.assertTrue(any('num_program_cache_entries' in failure for failure in report['failures']))

    def test_a_runtime_that_refuses_a_listed_twice_tensor_moves_to_the_pair_form(self):
        status, report, _ = self.run_harness(FakeTtnn(refuse_duplicates=True),
                                             ['--sections', 'forms,cases', '--kernels', 'served', '--prefixes', '16',
                                              '--chips', '0', '--edges', ''])
        self.assertEqual(status, 0, report['failures'])
        self.assertFalse(report['forms']['aliased']['accepted'])
        self.assertIn('listed twice', report['forms']['aliased']['error'])
        self.assertEqual(report['form'], 'pair')
        self.assertTrue(all(case['all_exact'] for case in report['cases']))

    def test_an_in_place_unsafe_kernel_fails_the_forms_and_the_cases(self):
        fake = FakeTtnn(order='descending')
        status, report, _ = self.run_harness(fake, ['--sections', 'forms', '--kernels', 'served'])
        self.assertEqual(status, 1)
        self.assertIsNone(report['form'])
        self.assertTrue(any('accepted but the bank is wrong' in failure for failure in report['failures']))
        status, report, _ = self.run_harness(FakeTtnn(order='descending'),
                                             ['--sections', 'cases', '--form', 'aliased', '--kernels', 'served',
                                              '--prefixes', '1,16', '--chips', '0', '--edges', ''])
        self.assertEqual(status, 1)
        self.assertEqual([(c['oop_vs_oracle']['exact'], c['inplace_vs_oop']['exact']) for c in report['cases']],
                         [(True, False), (True, False)])
        self.assertFalse(report['decision']['bytes_identical'])

    def test_a_program_cache_that_keeps_stale_runtime_args_fails(self):
        status, report, _ = self.run_harness(FakeTtnn(stale_cache=True),
                                             ['--sections', 'forms,cache', '--kernels', 'served'])
        self.assertEqual(status, 1)
        self.assertFalse(report['cache']['ok'])
        self.assertFalse(report['cache']['fresh_addresses_honoured'])
        self.assertTrue(any(failure.startswith('program cache') for failure in report['failures']))

    def test_an_unknown_served_kernel_fails_the_run(self):
        kernel = self.image / 'draft_kv_slide.cpp'
        kernel.write_bytes(kernel.read_bytes() + b'// changed\n')
        fake = FakeTtnn(kinds={sha(kernel): 'direct'})
        status, report, log = self.run_harness(fake, ['--sections', 'forms', '--kernels', 'served'])
        self.assertEqual(status, 1)
        self.assertEqual(report['served']['kind'], 'unknown')
        self.assertFalse(report['served']['kernel_matches_bundle'])
        self.assertTrue(any('neither qualified slide kernel' in failure for failure in report['failures']))
        self.assertIn('WARNING', log)


# ---------------------------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------------------------

def find_bash():
    candidates = []
    if os.name == 'nt':
        for root in (os.environ.get('ProgramW6432'), os.environ.get('ProgramFiles'), 'C:/Program Files'):
            if root:
                candidates.append(Path(root) / 'Git' / 'bin' / 'bash.exe')
    found = shutil.which('bash')
    if found and not (os.name == 'nt' and ('system32' in found.lower() or 'windowsapps' in found.lower())):
        candidates.append(Path(found))
    return next((str(path) for path in candidates if path.is_file()), None)


class RunScriptTests(unittest.TestCase):
    def text(self):
        return SCRIPT.read_text(encoding='utf-8')

    def test_the_default_image_is_the_image_of_v192(self):
        gate = (ROOT / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml').read_text(encoding='utf-8')
        line = next(line for line in gate.splitlines() if re.search(r'\*-v192[|)]', line))
        image = re.search(r'image=(sha256:[0-9a-f]{64})', line).group(1)
        self.assertIn('IMAGE=${IMAGE:-%s}' % image, self.text())

    def test_the_launch_the_mounts_and_the_card(self):
        text = self.text()
        self.assertNotIn(chr(13), text)
        self.assertEqual(text.count('--device '), 1)
        self.assertIn('--device "$node"', text)
        self.assertIn('\nqual_card_resolve\nnode=$QUAL_NODE\nqual_refuse_holders\n', text)
        self.assertLess(text.index('\nqual_card_recheck   #'), text.index('\ntimeout -k 30 "$timeout_s" docker run'))
        self.assertIn('qual_reset_hint >&2', text[text.index('docker run --rm'):])
        self.assertIn('R=${RESULTS:-$HOME/kwork64/slide/$QUAL_TAG}', text)
        for source, destination in (('draft_kv_slide.cpp', 'scalar/draft_kv_slide.cpp'),
                                    ('draft_kv_slide_direct.cpp', 'direct/draft_kv_slide.cpp'),
                                    ('draft_kv_slide.py', 'draft_kv_slide.py')):
            self.assertIn('%s:%s' % (source, destination), text)
            self.assertTrue((CI / source).is_file())
        self.assertIn('SM+=(--mount "type=bind,src=$src,dst=/bench/slide/${pair#*:},readonly")', text)
        self.assertIn('--checkout-dir /bench/slide', text)
        self.assertEqual(Path('/bench/slide'), card.CHECKOUT_DIR)
        self.assertNotRegex(text, r'dst=/experiment-scripts')          # the served kernel is the image's own
        self.assertIn('--mount "type=bind,src=$test_file,dst=/bench/draft_slide_inplace_card_b.py,readonly"', text)
        self.assertIn('-e TT_METAL_CACHE=/kcache', text)

    def test_the_watcher_pass(self):
        text = self.text()
        block = text[text.index('if [ "${WATCHER:-}" = "1" ]; then'):]
        block = block[:block.index(chr(10) + 'fi' + chr(10))]
        self.assertIn('-e TT_METAL_WATCHER=5', block)
        self.assertIn('timeout_s=900', block)
        self.assertIn('--quick --no-timing --watchdog', block)

    def test_every_hang_exit_prints_the_reset_hint(self):
        # exit 3 (the watchdog), 124 / 137 (the container cap), and exit 1 with the faulthandler backstop's
        # 'Timeout (' in this run's log (a blocking call held the GIL, so the poller never ran).
        tail = self.text()[self.text().index('status=${PIPESTATUS[0]}'):]
        self.assertIn('3|124|137) hung=1 ;;', tail)
        self.assertIn('''1) grep -qF 'Timeout (' "$R/slide-$stamp.log" && hung=1 ;;''', tail)
        self.assertIn('tee "$R/slide-$stamp.log"', self.text())
        self.assertLess(tail.index('hung=1 ;;'), tail.index('qual_reset_hint >&2'))

    def test_the_qual_card_suite_covers_this_runner(self):
        suite = (CI / 'test_qual_card.py').read_text(encoding='utf-8')
        entry = "OPS / 'draft_slide_inplace' / 'run_card_b.sh'"
        self.assertEqual(suite.count(entry), 3)     # EMBEDDING (block drift), LAUNCH (recheck order), RESOLVE_FIRST

    def test_bash_parses_it_and_it_refuses_card_m(self):
        bash = find_bash()
        if bash is None:
            self.skipTest('bash not found')
        result = subprocess.run([bash, '-n', SCRIPT.as_posix()], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            env = {k: v for k, v in os.environ.items() if k not in ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'RESULTS')}
            env.update(QUAL_CARD=CARD_M, HOME=Path(directory).as_posix(), RESULTS=Path(directory, 'r').as_posix())
            result = subprocess.run([bash, SCRIPT.as_posix()], capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('refusing: QUAL_CARD=%s is card M' % CARD_M, result.stderr)
        self.assertNotIn('docker', result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
