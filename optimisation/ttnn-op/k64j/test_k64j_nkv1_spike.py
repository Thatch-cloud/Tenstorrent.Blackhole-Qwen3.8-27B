"""k64j_nkv1_spike on the CPU: its host helpers held to brute-force oracles, its verdict logic, and the whole flow on a fake
device whose attention honours cur_pos and the mask - so a call that ignored either is caught as FAIL, and a faithful one
passes. What only the card can show (that the K64j program builds and is exact at one KV head) is what the spike is for."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parents[2] / 'scripts' / 'ci'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import k64j_nkv1_spike as spike  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_extent_and_positions(self):
        self.assertEqual(spike.extent_of(0), 256)
        self.assertEqual(spike.extent_of(255), 256)
        self.assertEqual(spike.extent_of(256), 512)
        self.assertEqual(spike.bundle_positions(2304, 128, 2, 8), [2048 + 128, 2048 + 136])
        self.assertEqual(spike.bundle_positions(2304, 255, 2, 8), [2048 + 255, 2048 + 255], 'a bundle never crosses')
        self.assertEqual(spike.words_for([2176, 2184], share=True), [2303, 2303])
        self.assertEqual(spike.words_for([2176, 2200 + 256], share=False), [2303, 2559])

    def test_the_narrow_mask_is_the_causal_mask_of_each_folded_row_token_major(self):
        rows, extent = 8, 512
        positions = spike.bundle_positions(extent, 200, 2, rows)
        narrow = spike.narrow_mask(torch, positions, extent, rows)
        self.assertEqual(tuple(narrow.shape), (2, 1, 48, 256))
        causal = spike.causal_mask(torch, positions, extent, rows)
        self.assertEqual(tuple(causal.shape), (2, 1, 48, 512))
        self.assertTrue(torch.equal(causal[..., 256:], narrow), 'the last 256 columns of the causal mask are the tail mask')
        self.assertFalse(bool(causal[..., :256].float().abs().any()), 'every earlier key is visible to every row')
        for entry in range(2):
            for head in range(48):
                row_position = positions[entry] + head // 6
                for column in range(256):
                    masked = bool(torch.isinf(narrow[entry, 0, head, column]))
                    self.assertEqual(masked, 256 + column > row_position, (entry, head, column))

    def test_the_wide_mask_is_zero_before_the_tail_and_the_narrow_mask_after(self):
        positions = spike.bundle_positions(768, 130, 1, 16)
        wide = spike.wide_mask(torch, positions, 768, 16)
        self.assertEqual(tuple(wide.shape), (1, 1, 96, 768))
        self.assertFalse(bool(wide[..., :512].float().abs().any()))
        self.assertTrue(torch.equal(wide[..., 512:], spike.narrow_mask(torch, positions, 768, 16)))

    def test_the_poisoned_row_keeps_the_first_e_pages_and_poisons_the_rest(self):
        table = torch.arange(16, dtype=torch.int32)
        row = spike.poisoned_row(torch, table, 512, 1024, [100, 101, 102])
        self.assertEqual(row[:8].tolist(), list(range(8)))
        self.assertEqual(row[8:].tolist(), [100, 101, 102, 100, 101, 102, 100, 101])
        self.assertEqual(table.tolist(), list(range(16)), 'the source table is not changed')

    def test_program_keys_and_flag_sets(self):
        self.assertEqual(spike.program_key(131328, 2, 8, 256, 0x23), (131328, 2, 2, 8, 0x23))
        self.assertEqual(spike.program_key(2304, 1, 16, 2304, 0x1), (2304, 1, 3, 72, 0x1))
        self.assertEqual(spike.GEOMETRIES['G8B2'][2], 0x23)
        self.assertEqual(spike.GEOMETRIES['G16B1'][2], 0x21)
        for name, (rows, batch, flags) in spike.GEOMETRIES.items():
            self.assertTrue(flags & spike.TAIL and flags & spike.EXTENT and not flags & 0x4, name)
            self.assertEqual(bool(flags & spike.SHARE), batch > 1, name)
        import tp_shapes
        self.assertEqual(spike.HEAD_ROWS, tp_shapes.geometry(4).attn_fold_rows)

    def test_differing_counts_bits(self):
        left = torch.zeros(4).bfloat16()
        right = left.clone()
        right[2] = 1.0
        self.assertEqual(spike.differing(torch, left, left.clone()), 0)
        self.assertEqual(spike.differing(torch, left, right), 1)
        self.assertGreater(spike.differing(torch, left, torch.zeros(5).bfloat16()), 0)


class VerdictTests(unittest.TestCase):
    def report(self, comparisons, **extra):
        report = dict(comparisons=comparisons, geometries={'G8B2': dict(ran=True), 'G16B1': dict(ran=True)},
                      failures=[], extents=list(spike.DEFAULT_EXTENTS), seeds=[0], **extra)
        report['decision'] = spike.decide(report)
        return report

    def entry(self, count=0):
        return spike.comparison('X', 'k', 'l', count, True)

    def test_pass_fail_and_no_decision(self):
        self.assertEqual(self.report([self.entry()])['decision']['verdict'], 'PASS')
        self.assertEqual(self.report([self.entry(), self.entry(3)])['decision']['verdict'], 'FAIL')
        self.assertEqual(self.report([])['decision']['verdict'], 'NO-DECISION')
        refused = self.report([self.entry()])
        refused['geometries']['G16B1'] = dict(ran=False, error='TT_FATAL: not supported')
        self.assertEqual(spike.decide(refused)['verdict'], 'NO-DECISION')
        failing = dict(refused, failures=['no factory line'], geometries={})
        self.assertEqual(spike.decide(failing)['verdict'], 'NO-DECISION')

    def test_scope_and_the_verdict_line(self):
        full = self.report([self.entry()])
        self.assertEqual(spike.scope_of(full), 'full')
        self.assertTrue(spike.verdict_line(full).startswith('K64J_NKV1 verdict=PASS scope=full flags=G16B1:0x21,G8B2:0x23'))
        narrowed = self.report([self.entry()])
        narrowed['extents'] = [2304]
        self.assertEqual(spike.scope_of(narrowed), 'reduced')
        one = self.report([self.entry()])
        one['geometries']['G16B1']['ran'] = False
        self.assertEqual(spike.scope_of(one), 'reduced')

    def test_arguments(self):
        args = spike.parse_args(['--out', 'x.json'])
        self.assertEqual((args.geometries, args.extents, args.seeds), (['G16B1', 'G8B2'], list(spike.DEFAULT_EXTENTS), [0, 1]))
        for bad in (['--geometries', 'G4B3'], ['--extents', '300'], ['--extents', '262144'], ['--starts', '256'],
                    ['--capacity', '100']):
            with self.assertRaises((SystemExit, ValueError)):
                spike.parse_args(['--out', 'x.json'] + bad)

    def test_a_missing_factory_line_is_a_failure(self):
        report = dict(requested={(1024, 2, 2, 8, 0x23), (512, 2, 2, 16, 0x3)}, failures=[])
        line = '[QWEN-SDPA] flags=0x23 B=2 PNHt=2 St=32 mask_width_t=8 kv_share=true scratch_slots=0 cb_bytes=1\n'
        import test_sdpa_decode_qwen_card_m as card
        spike.check_factory_lines(report, line, card)
        self.assertEqual(len(report['failures']), 1)
        self.assertIn('capacity=512', report['failures'][0])
        self.assertEqual(report['factory_lines'], 1)


# ---- a fake device whose attention honours cur_pos and the mask -----------------------------------------------------

class FakeTensor:
    def __init__(self, data, dtype):
        self.data, self.dtype = data, dtype
        self.shape = tuple(data.shape)


class FakeDevice:
    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=11, y=10)

    def enable_program_cache(self):
        pass


class FakeTTNN:
    bfloat8_b, bfloat16, int32 = 'bf8', 'bf16', 'int32'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'row', 'tile', 'dram'

    def __init__(self, ignore_cur_pos=False, ignore_mask=False):
        self.ignore_cur_pos, self.ignore_mask = ignore_cur_pos, ignore_mask
        self.calls = []
        self.transformer = SimpleNamespace(paged_scaled_dot_product_attention_decode=self.attention)

    def open_device(self, **options):
        return FakeDevice()

    def close_device(self, device):
        pass

    def from_torch(self, host, device=None, dtype=None, layout=None, memory_config=None):
        return FakeTensor(host.clone(), dtype)

    def to_torch(self, tensor):
        return tensor.data

    def deallocate(self, tensor):
        pass

    def synchronize_device(self, device):
        pass

    @staticmethod
    def SDPAProgramConfig(**options):
        return SimpleNamespace(**options)

    def attention(self, query, keys, values, *, page_table_tensor, is_causal, attn_mask, scale, program_config,
                  memory_config, cur_pos_tensor=None):
        self.calls.append(dict(sentinel=program_config.q_chunk_size, cur_pos=cur_pos_tensor is not None,
                               mask=tuple(attn_mask.shape)))
        pages, mask = page_table_tensor.data, attn_mask.data
        batch, heads = query.shape[1], query.shape[2]
        output = torch.zeros(1, batch, heads, 256)
        for entry in range(batch):
            width = pages.shape[1] * 64
            if cur_pos_tensor is not None and not self.ignore_cur_pos:
                width = int(cur_pos_tensor.data[0 if batch > 1 and False else entry]) + 1
            key = keys.data[pages[entry, :width // 64].long(), 0].reshape(width, 256).float()
            value = values.data[pages[entry, :width // 64].long(), 0].reshape(width, 256).float()
            scores = query.data[0, entry].float() @ key.T * scale
            columns = mask.shape[-1]
            # ignore_mask drops the NARROW tail mask of the extent call only (a kernel that forgot it), so the extent call
            # disagrees with the compile-time twin that applies its wide one
            if not (self.ignore_mask and columns == 256):
                scores[:, width - columns:width] += mask[entry, 0].float()
            output[0, entry] = torch.softmax(scores, dim=-1) @ value
        return FakeTensor(output.to(torch.bfloat16), 'bf16')


def run_flow(device, extra=()):
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / 'spike.json'
        argv = ['--out', str(out), '--capacity', '1024', '--extents', '512,1024', '--starts', '128,255', '--seeds', '0',
                '--iterations', '2'] + list(extra)
        with patch.dict(sys.modules, {'ttnn': device}), patch.dict('os.environ', {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}):
            import test_sdpa_decode_qwen_card_m as card

            class Quiet:
                def __init__(self, path):
                    self.path = Path(path)

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def text(self):
                    return ''

            with patch.object(card, 'NativeLog', Quiet), patch.object(spike, 'check_factory_lines'):
                status = spike.main(argv)
        return status, json.loads(out.read_text())


class FlowTests(unittest.TestCase):
    def test_a_faithful_device_passes_and_the_report_records_what_ran(self):
        status, report = run_flow(FakeTTNN())
        self.assertEqual(report['decision']['verdict'], 'PASS', report['decision'])
        self.assertEqual(status, 0)
        self.assertEqual({name: state['ran'] for name, state in report['geometries'].items()},
                         {'G8B2': True, 'G16B1': True})
        # G8B2: 2 entries, G16B1: 1; families 512 and 1024 x starts 128 and 255 x one seed
        self.assertEqual(report['tally']['decisive'], (2 + 1) * 2 * 2)
        self.assertEqual(report['tally']['differing'], 0)
        self.assertEqual(len(report['numerics']), 2 * 2 * 2)
        self.assertTrue(all(entry['max_abs'] >= 0 for entry in report['numerics']))
        self.assertEqual(sorted(report['timings']), ['G16B1/E1024', 'G16B1/E512', 'G8B2/E1024', 'G8B2/E512'])
        self.assertIn('K64J_NKV1 verdict=PASS scope=reduced', report['verdict_line'])
        # the calls: extent (flag 0x23 / 0x21 with cur_pos), compile-time (0x3 / 0x1, none), legacy (sentinel 0)
        device = FakeTTNN()
        run_flow(device, ['--iterations', '0'])
        sentinels = {(call['sentinel'], call['cur_pos']) for call in device.calls}
        self.assertEqual(sentinels, {(spike.MAGIC | 0x23, True), (spike.MAGIC | 0x3, False), (0, False),
                                     (spike.MAGIC | 0x21, True), (spike.MAGIC | 0x1, False)})

    def test_a_device_that_ignores_cur_pos_reads_the_poison_and_fails(self):
        status, report = run_flow(FakeTTNN(ignore_cur_pos=True), ['--no-legacy', '--iterations', '0'])
        self.assertEqual(report['decision']['verdict'], 'FAIL')
        self.assertEqual(status, 1)
        self.assertGreater(report['tally']['differing'], 0)

    def test_a_device_that_ignores_the_tail_mask_fails_at_the_boundary_rows(self):
        status, report = run_flow(FakeTTNN(ignore_mask=True), ['--no-legacy', '--iterations', '0'])
        self.assertEqual(report['decision']['verdict'], 'FAIL')
        self.assertGreater(report['tally']['differing'], 0)

    def test_the_legacy_numerics_arm_sees_a_wrong_mask(self):
        _, report = run_flow(FakeTTNN(ignore_mask=True), ['--iterations', '0'])
        self.assertTrue(any(entry['max_abs'] > 0 for entry in report['numerics']))
        _, honest = run_flow(FakeTTNN(), ['--iterations', '0'])
        self.assertTrue(all(entry['max_abs'] < 1e-2 for entry in honest['numerics']), honest['numerics'])

    def test_a_refusal_is_recorded_and_the_run_is_no_decision(self):
        class Refusing(FakeTTNN):
            def attention(self, *arguments, **keywords):
                if keywords['program_config'].q_chunk_size == spike.MAGIC | 0x21:
                    raise RuntimeError('TT_FATAL: extent needs the tail flag')
                return FakeTTNN.attention(self, *arguments, **keywords)

        status, report = run_flow(Refusing(), ['--no-legacy', '--iterations', '0'])
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertIn('TT_FATAL', report['geometries']['G16B1']['error'])
        self.assertTrue(report['geometries']['G8B2']['ran'], 'the other geometry still ran')
        self.assertEqual(status, 1)

    def test_no_compact_scratch_is_a_failure_before_the_device_opens(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'spike.json'
            with patch.dict(sys.modules, {'ttnn': FakeTTNN()}), patch.dict('os.environ', {}, clear=True):
                status = spike.main(['--out', str(out), '--capacity', '1024', '--extents', '512'])
            report = json.loads(out.read_text())
        self.assertEqual(status, 1)
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertTrue(any('QWEN_SDPA_TREE_SCRATCH_ROUNDS' in failure for failure in report['failures']))


if __name__ == '__main__':
    unittest.main()
