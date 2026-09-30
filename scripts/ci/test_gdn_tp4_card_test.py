"""gdn_tp4_card_test on the CPU: its references held to tile-level simulations of the kernel sources, its widening to the
pair's 24 heads, its verdict logic, and the whole DMA flow through ChipView on a fake device.

What this cannot show is what the card is for: that the kernels move the bytes (the fake's generic_op moves none, so the
flow must end FAIL - a dead device cannot pass)."""

from collections import defaultdict, namedtuple
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import chip_view
import gdn_tp4_card_test as card
import tp_shapes

HERE = Path(__file__).resolve().parent
Core = namedtuple('Core', 'x y')


def four():
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


# ---- tile <-> logical (a bf16 32x32 tile is four 16x16 faces, face-ordered: rows 0-15 | cols 0-15, 16-31, then rows 16-31)

def tile_index(row, column):
    return (row // 16) * 512 + (column // 16) * 256 + (row % 16) * 16 + column % 16


def to_tile(logical):
    """(rows <= 32, 32) -> 1024 elements in tile order (rows past the logical ones zero)."""
    out = torch.zeros(1024, dtype=logical.dtype)
    for row in range(logical.shape[0]):
        for column in range(32):
            out[tile_index(row, column)] = logical[row, column]
    return out


def from_tile(words, rows):
    return torch.stack([torch.stack([words[tile_index(row, column)] for column in range(32)]) for row in range(rows)])


def rand16(*shape):
    return torch.randn(*shape).bfloat16()


class ReferenceSimulationTests(unittest.TestCase):
    """Each logical reference against a direct transcription of the kernel's tile-level loop (the pinned .cpp)."""

    def test_windows_reference_is_the_window_kernel(self):
        """gdn_conv_windows.cpp window<>(): zero scratch, then per token copy the source row's two 16-element face rows."""
        rows, pages = 16, 3
        piece = rand16(1, rows, 32 * pages)
        history = [rand16(1, 1, 32 * pages) for _ in range(4)]
        reference = card.windows_reference(torch, piece, history, rows, 32 * pages)
        for page in range(pages):
            tiles = [to_tile(piece[0, :, 32 * page:32 * page + 32])] + [to_tile(h[0, :, 32 * page:32 * page + 32])
                                                                        for h in history]
            for slot in range(4):
                scratch = torch.zeros(1024, dtype=torch.bfloat16)
                for token in range(rows):
                    hist = token + slot
                    source_row = 0 if hist < 4 else hist - 4
                    source_offset = (source_row // 16) * 512 + (source_row % 16) * 16
                    destination_row = (token // 16) * 512 + (token % 16) * 16
                    tile = hist + 1 if hist < 4 else 0
                    for face in range(2):
                        target = slice(destination_row + face * 256, destination_row + face * 256 + 16)
                        scratch[target] = tiles[tile][source_offset + face * 256:source_offset + face * 256 + 16]
                got = from_tile(scratch, rows)
                self.assertTrue(torch.equal(got, reference[slot][0, :, 32 * page:32 * page + 32]), (page, slot))

    def test_prefix_reference_is_the_prefix_copy_kernel(self):
        """gdn_conv_prefix_copy.cpp: read the window tile, zero the output, copy face rows of row `token` to row 0."""
        rows, pages = 16, 2
        windows = [rand16(1, rows, 32 * pages) for _ in range(4)]
        for prefix in range(1, rows + 1):
            reference = card.prefix_reference(torch, windows, prefix, 32 * pages)
            token = prefix - 1
            offset = (token // 16) * 512 + (token % 16) * 16
            for slot in range(4):
                for page in range(pages):
                    staged = to_tile(windows[slot][0, :, 32 * page:32 * page + 32])
                    output = torch.zeros(1024, dtype=torch.bfloat16)
                    for face in range(2):
                        output[face * 256:face * 256 + 16] = staged[offset + face * 256:offset + face * 256 + 16]
                    got = from_tile(output, 1)
                    self.assertTrue(torch.equal(got, reference[slot][0, :, 32 * page:32 * page + 32]), (prefix, slot))

    def test_commit_reference_is_the_commit_kernel(self):
        """gdn_commit_dma.cpp: state pages move whole (history record prefix - 1, or the entry); each conv page's
        row 0 comes from the history row `token` (or the entry's row 0)."""
        heads, rows, channels = 2, 16, 64
        entry = [rand16(1, heads, 128, 128)] + [rand16(1, 1, channels) for _ in range(4)]
        history = [rand16(rows, heads, 128, 128)] + [rand16(1, rows, channels) for _ in range(4)]
        for prefix in (0, 1, 7, 16):
            published = card.commit_reference(torch, entry, history, prefix, rows)
            token = 0 if prefix == 0 else prefix - 1
            offset = (token // 16) * 512 + (token % 16) * 16
            for slot in range(1, 5):
                source = entry[slot][0] if prefix == 0 else history[slot][0]
                for page in range(channels // 32):
                    staged = to_tile(source[:, 32 * page:32 * page + 32])
                    output = torch.zeros(1024, dtype=torch.bfloat16)
                    for face in range(2):
                        output[face * 256:face * 256 + 16] = staged[offset + face * 256:offset + face * 256 + 16]
                    self.assertTrue(torch.equal(from_tile(output, 1),
                                                published[slot][0, :, 32 * page:32 * page + 32]), (prefix, slot))
            record = entry[0] if prefix == 0 else history[0][prefix - 1:prefix]
            self.assertTrue(torch.equal(published[0], record), prefix)


class WideningTests(unittest.TestCase):
    def test_the_first_twelve_heads_of_the_pair_inputs_are_the_four_card_inputs(self):
        found4, found2 = tp_shapes.geometry(4), tp_shapes.geometry(2)
        generator = torch.Generator().manual_seed(3)
        four_inputs = card.user_inputs(torch, found4, generator)
        spare = card.user_inputs(torch, found4, generator)
        wide = card.widen_to_pair(torch, four_inputs, spare, found4, found2)
        qkv, beta, gate, initial, z = wide
        self.assertEqual([tuple(value.shape) for value in wide],
                         [(1, 16, 5120), (1, 16, 24), (1, 16, 24), (1, 24, 128, 128), (1, 16, 3072)])
        # q heads 0-3 and k heads 0-3 are the four-card ones; v heads 0-11 too; the rest are the spare's
        q4, k4, v12 = four_inputs[0][..., :512], four_inputs[0][..., 512:1024], four_inputs[0][..., 1024:]
        self.assertTrue(torch.equal(qkv[..., :512], q4))
        self.assertTrue(torch.equal(qkv[..., 1024:1536], k4))
        self.assertTrue(torch.equal(qkv[..., 2048:2048 + 1536], v12))
        self.assertTrue(torch.equal(qkv[..., 512:1024], spare[0][..., :512]))
        self.assertTrue(torch.equal(beta[..., :12], four_inputs[1]) and torch.equal(beta[..., 12:], spare[1]))
        self.assertTrue(torch.equal(initial[:, :12], four_inputs[3]) and torch.equal(initial[:, 12:], spare[3]))
        self.assertTrue(torch.equal(z[..., :1536], four_inputs[4]))
        self.assertEqual(card.head_slice(torch, torch.zeros(1, 16, 3072), torch.zeros(16, 24, 128, 128), found4)[1].shape,
                         (16, 12, 128, 128))

    def test_the_value_head_to_key_head_map_is_three_to_one_at_both_widths(self):
        for tp in (2, 4):
            found = tp_shapes.geometry(tp)
            self.assertEqual(found.gdn_nv // found.gdn_nk, 3)


class VerdictTests(unittest.TestCase):
    def report(self, requested, comparisons, sections=None, **extra):
        data = dict(requested=list(requested), comparisons=comparisons, sections=sections or {}, tp=4, **extra)
        data['decision'] = card.decide(data)
        data['tally'] = card.tally(comparisons)
        return data

    def equal(self, section):
        return card.comparison(section, 'k', 'label', True)

    def test_pass_needs_every_requested_section_decisive_and_equal(self):
        found = self.report(card.FOUR_SECTIONS, [self.equal(name) for name in card.FOUR_SECTIONS])
        self.assertEqual(found['decision']['verdict'], 'PASS')
        self.assertEqual(card.scope_of(found), 'full')
        self.assertIn('verdict=PASS scope=full tp=4 chips=1of4', card.verdict_line(found))

    def test_a_differing_comparison_fails(self):
        comparisons = [self.equal(name) for name in card.FOUR_SECTIONS] + [card.comparison('UB', 'k', 'x', False)]
        self.assertEqual(self.report(card.FOUR_SECTIONS, comparisons)['decision']['verdict'], 'FAIL')

    def test_a_raised_or_silent_section_is_no_decision_never_pass(self):
        raised = self.report(['UB', 'K5'], [self.equal('UB')], sections={'K5': {'error': 'boom'}})
        self.assertEqual(raised['decision']['verdict'], 'NO-DECISION')
        silent = self.report(['UB', 'K5'], [self.equal('UB')])
        self.assertEqual(silent['decision']['verdict'], 'NO-DECISION')
        self.assertIn('K5 made no comparison', silent['decision']['problems'])

    def test_a_narrowed_run_is_reduced_and_parity_is_its_own_scope(self):
        narrowed = self.report(['UB'], [self.equal('UB')])
        self.assertEqual(card.scope_of(narrowed), 'reduced')
        parity = self.report(['PARITY'], [self.equal('PARITY')])
        self.assertEqual(card.scope_of(parity), 'parity')
        seeds = self.report(card.FOUR_SECTIONS, [self.equal(name) for name in card.FOUR_SECTIONS], reduced=True)
        self.assertEqual(card.scope_of(seeds), 'reduced')

    def test_arguments(self):
        arguments = card.parse(['--out', 'x.json'])
        self.assertEqual((arguments.sections, arguments.seeds, arguments.prefixes),
                         (list(card.FOUR_SECTIONS), [17, 18, 19], [0, 1, 7, 16]))
        with self.assertRaises(SystemExit):
            card.parse(['--out', 'x.json', '--sections', 'UB,PARITY'])
        with self.assertRaises(SystemExit):
            card.parse(['--out', 'x.json', '--sections', 'NOPE'])
        with self.assertRaises(SystemExit):
            card.parse(['--out', 'x.json', '--prefixes', '17'])
        self.assertEqual(card.parse(['--out', 'x.json', '--sections', 'PARITY']).sections, ['PARITY'])


# ---- a fake device: host-backed tensors, no kernel moves a byte ------------------------------------------------------

class FakeTensor:
    def __init__(self, data, address):
        self.data, self.address = data, address
        self.shape, self.dtype, self.layout = tuple(data.shape), 'bf16', 'tile'

    def memory_config(self):
        return 'dram'

    def buffer_address(self):
        return self.address


class FakeDevice:
    bfloat16, float32, TILE_LAYOUT, ROW_MAJOR_LAYOUT = 'bf16', 'fp32', 'tile', 'row'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'dram', 'l1'
    NOC = SimpleNamespace(RISCV_0_default=0, RISCV_1_default=1)
    DataMovementProcessor = SimpleNamespace(RISCV_0='riscv0', RISCV_1='riscv1')

    def __init__(self):
        self.next = itertools.count(0x10000, 0x1000)
        self.launches = []
        self.freed = []

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        return FakeTensor(value.clone(), next(self.next))

    def to_torch(self, tensor):
        return tensor.data

    def get_device_tensors(self, tensor):
        return [tensor]

    def deallocate(self, tensor):
        self.freed.append(tensor)

    def synchronize_device(self, mesh):
        pass

    def empty(self, shape, **keywords):
        return FakeTensor(torch.zeros(*shape).bfloat16(), next(self.next))

    @staticmethod
    def ReplicateTensorToMesh(mesh):
        return ('replicate', mesh)

    CoreRangeSet = staticmethod(list)
    CoreRange = staticmethod(lambda *values: values)
    CoreCoord = staticmethod(Core)
    CBDescriptor = SimpleNamespace
    CBFormatDescriptor = SimpleNamespace
    TileDescriptor = staticmethod(lambda value: value)
    Tile = staticmethod(list)
    MeshProgramDescriptor = staticmethod(dict)
    ProgramDescriptor = SimpleNamespace
    MeshCoordinate = staticmethod(lambda *values: values)
    MeshCoordinateRange = staticmethod(lambda *values: values)
    KernelDescriptor = SimpleNamespace
    DataMovementConfigDescriptor = SimpleNamespace
    RuntimeArgs = staticmethod(lambda: defaultdict(dict))

    @staticmethod
    def TensorAccessorArgs(shard):
        return SimpleNamespace(get_compile_time_args=lambda: [7])

    def generic_op(self, tensors, program):
        self.launches.append(program)


class FlowTests(unittest.TestCase):
    def rig(self, device):
        report = dict(comparisons=[], sections={})
        return card.Rig(device, torch, SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10)),
                        report)

    def arguments(self, prefixes=(0, 1, 16)):
        return SimpleNamespace(prefixes=list(prefixes), seeds=[17], iterations=0, root=Path('/unused'))

    def test_the_dma_flow_builds_every_launch_and_a_dead_device_cannot_pass(self):
        device = FakeDevice()
        rig = self.rig(device)
        found4 = tp_shapes.geometry(4)
        with four():
            state = card.run_dma(rig, self.arguments(), found4)
        self.assertEqual(state['pages'], dict(state=192, conv=80))
        kinds = {entry['kind'] for entry in rig.comparisons}
        self.assertEqual(kinds, {'state_copy', 'conv_windows', 'conv_prefix_copy', 'commit_native_record',
                                 'commit_native_untouched_slots', 'commit_checkpoint_record', 'commit_native_conv_row0',
                                 'commit_native_conv_other_rows', 'commit_checkpoint_conv_row0'})
        # one state copy, one windows launch, two prefix copies (prefixes 1 and 16), three commits: six launches
        self.assertEqual(len(device.launches), 1 + 1 + 2 + 3)
        for program in device.launches:
            chips = [key for key in program]
            self.assertEqual(chips, [((0, 0), (0, 0))], 'ChipView launches the chip-0 program alone')
        # the untouched-slot and other-row checks pass on a dead device (nothing moved) and everything that moves differs
        # (the prefix copy reads the windows the dead device left zero, so its reference is zero too: not a control)
        moved = [entry for entry in rig.comparisons if entry['kind'] in ('state_copy', 'conv_windows',
                                                                        'commit_native_record',
                                                                        'commit_checkpoint_record')]
        self.assertTrue(moved and all(not entry['equal'] for entry in moved))
        report = dict(requested=['DMA'], comparisons=rig.comparisons, sections={})
        self.assertEqual(card.decide(report)['verdict'], 'FAIL')

    def test_the_dma_flow_seam_is_put_back_after_the_run(self):
        import gdn_multitoken_conv as pinned
        before = pinned.validate_projected
        device = FakeDevice()
        with four():
            card.run_dma(self.rig(device), self.arguments((1,)), tp_shapes.geometry(4))
        self.assertIs(pinned.validate_projected, before)

    def test_the_parity_flow_runs_the_pinned_arm_then_the_siblings_and_compares_every_output(self):
        device = FakeDevice()
        rig = self.rig(device)
        with patch.dict(os.environ, {}, clear=True):
            state = card.run_parity(rig, self.arguments((0, 1)), tp_shapes.geometry(2))
        self.assertEqual(state['pages'], dict(state=384, conv=160))
        # both arms launch the same number of programs; with nothing moved every comparison is equal
        half = len(device.launches) // 2
        self.assertEqual(len(device.launches), 2 * half)
        self.assertTrue(rig.comparisons and all(entry['section'] == 'PARITY' for entry in rig.comparisons))
        sources = [list(program.values())[0].kernels[0].kernel_source for program in device.launches]
        self.assertTrue(any(source.endswith('gdn_commit_dma.cpp') for source in sources[:half]))
        self.assertTrue(any(source.endswith('gdn_commit_dma_tp.cpp') for source in sources[half:]))
        defines = [dict(getattr(list(program.values())[0].kernels[0], 'defines', [])) for program in device.launches[half:]]
        self.assertTrue(all(value.get('QWEN_CONV_PAGES') in (None, '160') for value in defines))
        self.assertTrue(any(value.get('QWEN_STATE_PAGES') == '384' for value in defines))
        import tp_kernels
        self.assertEqual(tp_kernels.source(HERE / 'gdn_commit_dma.cpp'), str(HERE / 'gdn_commit_dma.cpp'),
                         'the source / defines patch is undone')


class ChipViewTests(unittest.TestCase):
    def test_one_shard_is_shown_as_chips_and_only_chip_zero_launches(self):
        device = FakeDevice()
        view = chip_view.ChipView(device, chips=4)
        tensor = device.from_torch(torch.zeros(2, 2))
        self.assertEqual(view.get_device_tensors(tensor), [tensor] * 4)
        program = view.MeshProgramDescriptor()
        for chip in range(4):
            coordinate = view.MeshCoordinate(0, chip)
            program[view.MeshCoordinateRange(coordinate, coordinate)] = SimpleNamespace(chip=chip)
        view.generic_op([tensor], program)
        (launched,) = device.launches
        self.assertEqual({key: value.chip for key, value in launched.items()}, {((0, 0), (0, 0)): 0})
        self.assertEqual((view.launches, view.realised, view.phantom), (1, 1, 3))

    def test_a_missing_or_repeated_chip_and_a_multi_shard_tensor_are_refused(self):
        device = FakeDevice()
        view = chip_view.ChipView(device, chips=4)
        program = view.MeshProgramDescriptor()
        coordinate = view.MeshCoordinate(0, 0)
        program[view.MeshCoordinateRange(coordinate, coordinate)] = SimpleNamespace()
        with self.assertRaises(ValueError):
            program[view.MeshCoordinateRange(coordinate, coordinate)] = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, 'every chip'):
            program.realise()
        with self.assertRaises(ValueError):
            other = view.MeshCoordinate(0, 4)
            view.MeshProgramDescriptor()[view.MeshCoordinateRange(other, other)] = SimpleNamespace()
        two = SimpleNamespace(get_device_tensors=lambda tensor: [tensor, tensor])
        with self.assertRaisesRegex(RuntimeError, 'ONE chip'):
            chip_view.ChipView(two, chips=4).get_device_tensors(object())

    def test_installed_swaps_and_restores_sys_modules_ttnn(self):
        device = FakeDevice()
        saved = sys.modules.get('ttnn')
        view = chip_view.ChipView(device, chips=4)
        with view.installed():
            self.assertIs(sys.modules['ttnn'], view)
        self.assertIs(sys.modules.get('ttnn'), saved)


class MainTests(unittest.TestCase):
    def test_the_report_and_verdict_line_are_written_even_when_the_width_is_wrong(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'r.json'
            with patch.dict(os.environ, {}, clear=True):
                # no torch / ttnn device here: the run stops at the width check (or the import) and says so
                with patch.dict(sys.modules, {'ttnn': SimpleNamespace()}):
                    status = card.main(['--out', str(out)])
            report = json.loads(out.read_text())
            self.assertEqual(status, 1)
            self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
            self.assertTrue(report['verdict_line'].startswith('GDN_TP4 verdict=NO-DECISION'))


if __name__ == '__main__':
    unittest.main()
