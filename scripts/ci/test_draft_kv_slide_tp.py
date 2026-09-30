"""The four-card K/V slide (draft_kv_slide_tp, QWEN_FAST_TP_KV_SLIDE) against the eager chain it replaces.

Held here, on the CPU:
  - the twin's geometry is the pair's for every (history_rows, prefix), and the kernel it launches is a qualified
    draft_kv_slide.cpp, unedited (the image carries the bundle's direct-DMA kernel; both digests are draft_kv_slide_gate's);
  - the transport's launch: at the pair its programs are exactly draft_kv_slide.prepare's (two chips, sixteen workers, the same
    runtime arguments), at four cards they are four chips x eight workers over (1, 2, ...) banks, and it refuses the pair's
    shapes, aliasing storage and a wrong chip count;
  - EXACTNESS: DraftKVHistory_tp.prepare with the flag on publishes the same spare bank, bit for bit, as the flag-off eager
    chain, for every history_rows in the ramp and at the boundaries x every prefix 1..32, and across the transition to a full
    2048-row window over consecutive commits. The device kernel is stood in for by a host model written from the kernel's own
    per-row rule (draft_kv_slide.cpp: destination row < rows reads logical row + drop from the active bank while it is below
    history_rows, else the delta's row, else zero) - not from the eager chain it is compared with;
  - the flag: unset, '0' and anything but '1' keep the eager chain and call no transport; '1' calls one per (layer, k / v).
The kernel itself runs only on a card: the one-card harness of the S0-S3 window is the hardware half of this test."""

import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from collections import defaultdict, namedtuple

import torch

import draft_kv_history_tp
import draft_kv_slide
import draft_kv_slide_tp
import tp_addresses
import tp_shapes
from tp_test_support import four_cards, pair
from test_draft_tp4 import HistoryOps
from test_pair_row_exact import Device

HERE = Path(__file__).resolve().parent
QUALIFIED_SCALAR_KERNEL = 'bc45d47257c844aff4bf17f478b536a48763578e544597884b7d1620083b6ba1'
QUALIFIED_DIRECT_KERNEL = '1679bbd779add56b4bd445a6b4c51bd3e49c39a8a520dfaddfc3bd9f36667d47'
RAMP = (1, 2, 15, 16, 31, 32, 33, 47, 452, 533, 1000, 2000, 2016, 2040, 2047, 2048)


class GeometryAndKernelTests(unittest.TestCase):
    def test_the_geometry_is_the_pairs_for_every_history_and_prefix(self):
        for history in RAMP:
            for prefix in range(1, 33):
                self.assertEqual(draft_kv_slide_tp.geometry(history, prefix), draft_kv_slide.geometry(history, prefix))

    def test_invalid_geometry_is_refused_as_at_the_pair(self):
        for history, prefix in ((0, 1), (2049, 1), (2048, 0), (2048, 33), (True, 1), (1, 1.0)):
            with self.assertRaises(ValueError):
                draft_kv_slide_tp.geometry(history, prefix)

    def test_the_kernel_is_a_qualified_one_unedited(self):
        """The transport launches draft_kv_slide.cpp beside it. The image carries the bundle's (the direct-DMA kernel: nothing
        copies the checkout's, test_draft_slide_inplace_card_b), the checkout holds the scalar one: each is a qualified digest."""
        self.assertEqual(draft_kv_slide_tp.KERNEL, HERE / 'draft_kv_slide.cpp')
        self.assertIn(hashlib.sha256(draft_kv_slide_tp.KERNEL.read_bytes()).hexdigest(), (QUALIFIED_SCALAR_KERNEL, QUALIFIED_DIRECT_KERNEL))
        self.assertEqual(hashlib.sha256((HERE / 'draft_kv_slide_direct.cpp').read_bytes()).hexdigest(), QUALIFIED_DIRECT_KERNEL)

    def test_both_qualified_kernels_are_generic_in_the_head_count(self):
        """Each kernel derives head and column from the worker index and addresses pages of a 64 x 4 tile grid per head, and
        names no head count: which is what lets the four-card transport launch 4 workers per KV head over 2 heads."""
        for name, delta in (('draft_kv_slide.cpp', 'noc_async_read_tile(head * 4 + column, delta, added);'),
                            ('draft_kv_slide_direct.cpp', ': head * 4 + column;')):
            text = (HERE / name).read_text()
            for line in ('const uint32_t head = worker / 4;', 'const uint32_t column = worker % 4;',
                         'noc_async_write_tile((head * 64 + tile) * 4 + column, spare, output);', delta):
                self.assertIn(line, text, (name, line))
            self.assertNotRegex(text, r'head\s*<\s*\d|heads?\s*=\s*\d|worker\s*[<>]=?\s*\d')

    def test_widths_and_workers_follow_tp_shapes(self):
        with pair():
            self.assertEqual((draft_kv_slide_tp.bank_shape(), draft_kv_slide_tp.delta_shape(), draft_kv_slide_tp.worker_count()),
                             ((1, 4, 2048, 128), (1, 4, 32, 128), 16))
        with four_cards():
            self.assertEqual((draft_kv_slide_tp.bank_shape(), draft_kv_slide_tp.delta_shape(), draft_kv_slide_tp.worker_count()),
                             ((1, 2, 2048, 128), (1, 2, 32, 128), 8))

    def test_the_flag_is_on_only_for_exactly_one(self):
        for value, expected in (('1', True), ('0', False), ('', False), ('true', False), ('2', False)):
            self.assertIs(draft_kv_slide_tp.enabled({draft_kv_slide_tp.FLAG: value}), expected, value)
        self.assertIs(draft_kv_slide_tp.enabled({}), False)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(draft_kv_slide_tp.enabled(), False)
        with patch.dict(os.environ, {draft_kv_slide_tp.FLAG: '1'}):
            self.assertIs(draft_kv_slide_tp.enabled(), True)


# ---- the transport's launch, on a recording stand-in for the ttnn surface it uses ---------------------------------------

class FakeShard:
    tile = SimpleNamespace(tile_shape=(32, 32), transpose_of_faces=False, transpose_within_face=False)
    dtype, layout = 'bf16', 'tile'

    def __init__(self, shape, address, memory='dram'):
        self.shape, self._address, self._memory = tuple(shape), address, memory

    def buffer_address(self):
        return self._address

    def memory_config(self):
        return self._memory


class FakeMeshTensor:
    def __init__(self, shape, chips, base, alias=None):
        self.shards = [FakeShard(shape, (alias if alias is not None else base) + chip * 4096) for chip in range(chips)]


Coord = namedtuple('Coord', 'x y')


class Launch:
    def __init__(self):
        self.programs = []


def fake_ttnn(launch):
    class Program(dict):
        pass

    class Kernel:
        def __init__(self, **options):
            self.__dict__.update(options)

    def generic_op(tensors, program):
        launch.programs.append((tensors, program))

    return SimpleNamespace(
        bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
        get_device_tensors=lambda tensor: tensor.shards,
        CoreCoord=Coord, CoreRange=lambda a, b: (a, b), CoreRangeSet=lambda ranges: tuple(ranges),
        CBDescriptor=lambda **options: dict(options), CBFormatDescriptor=lambda **options: dict(options),
        TileDescriptor=lambda tile: ('tile', tile), Tile=lambda shape: tuple(shape),
        MeshProgramDescriptor=Program, KernelDescriptor=Kernel,
        TensorAccessorArgs=lambda value: SimpleNamespace(get_compile_time_args=lambda: [value.shape[1]]),
        DataMovementConfigDescriptor=lambda **options: dict(options),
        DataMovementProcessor=SimpleNamespace(RISCV_0='r0'), NOC=SimpleNamespace(RISCV_0_default='noc0'),
        RuntimeArgs=lambda: defaultdict(dict), MeshCoordinate=lambda a, b: (a, b),
        MeshCoordinateRange=lambda a, b: (a, b), ProgramDescriptor=lambda **options: dict(options),
        generic_op=generic_op)


def mesh(cores=130):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=13, y=cores // 13))


def summary(launch):
    """Everything the launch fixes, in a comparable form: per chip the kernel source, compile args and every core's runtime args."""
    (tensors, program), = launch.programs
    result = {}
    for key, value in program.items():
        kernel, = value['kernels']
        result[key] = (Path(kernel.kernel_source).name, tuple(kernel.compile_time_args), tuple(kernel.core_ranges),
                       {(x, y): tuple(args) for x, column in kernel.runtime_args.items() for y, args in column.items()},
                       value['cbs'][0]['total_size'])
    return result


class TransportTests(unittest.TestCase):
    def launch(self, module, chips, heads, history, prefix, **override):
        launch = Launch()
        tensors = [FakeMeshTensor((1, heads, 2048, 128), chips, 1000), FakeMeshTensor((1, heads, 32, 128), chips, 500000),
                   FakeMeshTensor((1, heads, 2048, 128), chips, 900000)]
        with patch.dict(sys.modules, {'ttnn': fake_ttnn(launch)}):
            operation = module.prepare(mesh(), *tensors, history_rows=history, prefix=prefix)
            self.assertEqual(launch.programs, [])   # nothing runs until the caller runs it
            operation()
        return launch

    def test_at_the_pair_the_twin_launches_exactly_what_the_pairs_transport_does(self):
        with pair():
            for history, prefix in ((31, 2), (452, 5), (2047, 2), (2048, 1), (2048, 16), (2048, 32)):
                twin = summary(self.launch(draft_kv_slide_tp, 2, 4, history, prefix))
                native = summary(self.launch(draft_kv_slide, 2, 4, history, prefix))
                self.assertEqual(twin, native, (history, prefix))
                self.assertEqual(len(twin), 2)
                self.assertTrue(all(len(value[3]) == 16 for value in twin.values()))

    def test_at_four_cards_it_launches_four_chips_of_eight_workers_over_two_heads(self):
        with four_cards():
            found = summary(self.launch(draft_kv_slide_tp, 4, 2, 452, 5))
        self.assertEqual(sorted(found), [((0, chip), (0, chip)) for chip in range(4)])
        for chip, (source, compile_args, cores, runtime, cb) in enumerate(found.values()):
            self.assertEqual((source, cb, len(cores), len(runtime)), ('draft_kv_slide.cpp', 8192, 8, 8))
            for worker, arguments in enumerate(runtime[(worker % 13, worker // 13)] for worker in range(8)):
                # active, delta, spare addresses of this chip; then history_rows, prefix, drop, rows, worker
                self.assertEqual(arguments, (1000 + chip * 4096, 500000 + chip * 4096, 900000 + chip * 4096, 452, 5, 0, 457, worker))
        with four_cards():
            full = summary(self.launch(draft_kv_slide_tp, 4, 2, 2048, 16))
        self.assertEqual({value[3][(0, 0)][3:7] for value in full.values()}, {(2048, 16, 16, 2048)})

    def test_the_pairs_shapes_and_a_wrong_chip_count_are_refused_at_four_cards(self):
        with four_cards():
            for chips, heads in ((2, 2), (4, 4)):
                with self.assertRaises(ValueError, msg=(chips, heads)):
                    self.launch(draft_kv_slide_tp, chips, heads, 452, 5)

    def test_aliasing_storage_and_too_few_cores_are_refused(self):
        launch = Launch()
        with four_cards(), patch.dict(sys.modules, {'ttnn': fake_ttnn(launch)}):
            same = FakeMeshTensor((1, 2, 2048, 128), 4, 1000)
            with self.assertRaisesRegex(ValueError, 'must not alias'):
                draft_kv_slide_tp.prepare(mesh(), same, FakeMeshTensor((1, 2, 32, 128), 4, 500000), same,
                                          history_rows=452, prefix=5)
            with self.assertRaisesRegex(ValueError, '8 head/column transport workers'):
                draft_kv_slide_tp.prepare(mesh(cores=7 * 1), FakeMeshTensor((1, 2, 2048, 128), 4, 1000),
                                          FakeMeshTensor((1, 2, 32, 128), 4, 500000),
                                          FakeMeshTensor((1, 2, 2048, 128), 4, 900000), history_rows=452, prefix=5)


# ---- exactness: the slide publishes the eager chain's spare bank, bit for bit ---------------------------------------------

def kernel_model(history_rows, prefix, active, delta):
    """The kernel's per-row rule, from draft_kv_slide.cpp: destination row d < rows takes logical row s = d + drop, from the
    active bank while s < history_rows, else from the delta while s - history_rows < prefix; everything else is zero."""
    geometry = draft_kv_slide_tp.geometry(history_rows, prefix)
    source_bank = torch.cat([active, delta, torch.zeros_like(active[:, :, :1])], dim=2)   # rows: active, delta, one zero row
    zero_row = active.shape[2] + delta.shape[2]
    destination = torch.arange(2048)
    logical = destination + geometry['drop']
    live = destination < geometry['rows']
    from_active = live & (logical < history_rows)
    from_delta = live & ~from_active & (logical - history_rows < prefix)
    pick = torch.full((2048,), zero_row)
    pick[from_active] = logical[from_active]
    pick[from_delta] = active.shape[2] + (logical - history_rows)[from_delta]
    return source_bank.index_select(2, pick)


class Recorder:
    """Stands in for draft_kv_slide_tp.prepare on the CPU: records the call, then runs the kernel model when called."""

    def __init__(self):
        self.calls = []

    def __call__(self, mesh, active, delta, spare, *, history_rows, prefix):
        self.calls.append((history_rows, prefix))

        def run():
            spare.value = kernel_model(history_rows, prefix, active.value, delta.value)
        return run


class ExactnessTests(unittest.TestCase):
    KV, LAYERS = 2, 2

    BANKS = {}

    def banks(self, seed):
        """Random bf16 banks for (seed): generated once, cloned per use (torch.randn on 2,048-row banks dominates the run)."""
        if seed not in self.BANKS:
            generator = torch.Generator().manual_seed(seed)
            self.BANKS[seed] = [torch.randn(1, self.KV, 2048, 128, generator=generator).bfloat16() for _ in range(4 * self.LAYERS)]
        return [value.clone() for value in self.BANKS[seed]]

    def cache(self, operations, history_rows, seed):
        raw = self.banks(seed)
        for value in raw[:2 * self.LAYERS]:
            value[:, :, history_rows:, :] = 0   # a valid active window: rows past history_rows are zero, as every published bank holds them
        pieces = iter(Device(value) for value in raw)
        cache = object.__new__(draft_kv_history_tp.DraftKVHistory)
        query = Device(torch.zeros(1, 1, 32, 1024).bfloat16())
        cache.__dict__.update(operations=operations, mesh='mesh', parameters=tuple(object() for _ in range(self.LAYERS)),
                              position=history_rows, history_rows=history_rows, owned=[], checks=[], projection=None,
                              pending=None, closed=False, query=query,
                              active=[dict(k=next(pieces), v=next(pieces)) for _ in range(self.LAYERS)],
                              spare=[dict(k=next(pieces), v=next(pieces)) for _ in range(self.LAYERS)])
        cache.borrowed = [query, *[t for layer in cache.active + cache.spare for t in layer.values()]]
        return cache

    def projector(self, seed):
        generator = torch.Generator().manual_seed(seed)

        def project(operations, inputs, query, tables, retain, parameters):
            make = lambda: Device(torch.randn(1, self.KV, 32, 128, generator=generator).bfloat16())
            return dict(q=Device(torch.zeros(1, 1, 32, 1024)), k=make(), v=make())
        return project

    def run_rounds(self, history_rows, prefixes, slide, seed=7):
        """The banks after each commit of `prefixes` from a window of `history_rows`, with the flag on or off."""
        operations = HistoryOps()
        features = Device(torch.randn(1, 1, 32, 5120, generator=torch.Generator().manual_seed(3)).bfloat16())
        recorder = Recorder()
        seen = []
        environment = {draft_kv_slide_tp.FLAG: '1'} if slide else {}
        with four_cards(), patch.dict(os.environ, environment), \
                patch.object(draft_kv_history_tp, 'project_key_value', self.projector(seed + 1)), \
                patch.object(draft_kv_history_tp.draft_kv_slide_tp, 'prepare', recorder):
            cache = self.cache(operations, history_rows, seed)
            for prefix in prefixes:
                publication = cache.prepare(features, prefix, position=cache.position)
                # a publication writes only the spare: the active bank must be as it was
                cache.commit(publication)
                seen.append((cache.history_rows, cache.position,
                             [(layer['k'].value.clone(), layer['v'].value.clone()) for layer in cache.active]))
        return seen, recorder, operations

    def assert_same(self, eager, slid, label):
        self.assertEqual(len(eager), len(slid), label)
        for (rows_a, position_a, banks_a), (rows_b, position_b, banks_b) in zip(eager, slid, strict=True):
            self.assertEqual((rows_a, position_a), (rows_b, position_b), label)
            for (ka, va), (kb, vb) in zip(banks_a, banks_b, strict=True):
                self.assertTrue(torch.equal(ka.view(torch.int16), kb.view(torch.int16)), label)
                self.assertTrue(torch.equal(va.view(torch.int16), vb.view(torch.int16)), label)

    def test_every_ramp_history_and_prefix_publishes_the_eager_banks_bit_for_bit(self):
        for history in RAMP:
            # every prefix at the tile / capacity boundaries, a spread of them elsewhere
            prefixes = range(1, 33) if history in (1, 31, 32, 33, 2016, 2040, 2047, 2048) else (1, 2, 5, 15, 16, 17, 31, 32)
            for prefix in prefixes:
                label = 'history_rows=%d prefix=%d' % (history, prefix)
                eager, _, _ = self.run_rounds(history, [prefix], slide=False)
                slid, recorder, _ = self.run_rounds(history, [prefix], slide=True)
                self.assert_same(eager, slid, label)
                self.assertEqual(recorder.calls, [(history, prefix)] * (2 * self.LAYERS), label)

    def test_consecutive_commits_across_the_transition_to_a_full_window(self):
        for history, prefixes in ((2000, [16] * 6), (1, [32, 1, 7, 16, 16]), (2040, [3, 5, 8, 1, 16]),
                                  (2047, [1, 1, 32, 32]), (452, [4, 3, 16, 9, 1, 12])):
            eager, _, _ = self.run_rounds(history, prefixes, slide=False, seed=history)
            slid, recorder, _ = self.run_rounds(history, prefixes, slide=True, seed=history)
            self.assert_same(eager, slid, 'from %d: %r' % (history, prefixes))
            if history + sum(prefixes) > 2048:
                self.assertEqual(slid[-1][0], 2048)   # the window did fill and slide
            self.assertEqual(len(recorder.calls), 2 * self.LAYERS * len(prefixes))

    def test_the_flag_off_calls_no_transport_and_runs_the_eager_ops(self):
        _, recorder, operations = self.run_rounds(452, [5, 5], slide=False)
        self.assertEqual(recorder.calls, [])
        self.assertTrue(any(call[0] == 'concat' for call in operations.calls))
        _, recorder, operations = self.run_rounds(452, [5, 5], slide=True)
        self.assertEqual(len(recorder.calls), 8)
        self.assertFalse(any(call[0] == 'concat' for call in operations.calls), 'the slide replaces the six-op chain')
        # ... and it does not slice the active bank at a (history_rows-dependent) shape: no op sees a ramp-shaped bank
        self.assertFalse(any(call[0] == 'slice' and len(call[3]) == 4 and call[3][2] == 452 for call in operations.calls))

    def test_the_slide_flag_alone_never_changes_the_pairs_class(self):
        text = (HERE / 'draft_kv_history.py').read_text()
        self.assertNotIn('draft_kv_slide_tp', text)
        self.assertNotIn('QWEN_FAST_TP_KV_SLIDE', text)


if __name__ == '__main__':
    unittest.main()
