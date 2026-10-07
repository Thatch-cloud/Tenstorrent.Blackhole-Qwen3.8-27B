"""CPU tests of the U2 canonical-rule probe: its patterns, its task plan, its comparisons and verdict, and its whole run on a fake ttnn whose served path
applies the rule the kernel claims (and, in the negative arms, a different one).

Run: python -B -m unittest discover -s optimisation/ttnn-op/canon_probe -p 'test_*.py'
"""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
CI = HERE.parent.parent.parent / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))
import canon_probe_card_m as probe  # noqa: E402
import gdn_rows_dma_tp as real_rows_dma  # noqa: E402


def exponent_zero_to_positive_zero(bits):
    unsigned = bits.to(torch.int32) & 0xFFFF
    return torch.where((unsigned & 0x7F80) == 0, torch.zeros_like(bits), bits)


def negative_zero_only(bits):
    return torch.where(bits == -32768, torch.zeros_like(bits), bits)


class Tensor(object):
    def __init__(self, bits, layout='tile'):
        self.bits, self.layout = bits, layout

    @property
    def shape(self):
        return tuple(self.bits.shape)


class FakeTTNN(object):
    bfloat16, TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'row', 'dram'

    def __init__(self, served=exponent_zero_to_positive_zero, upload=None):
        self.served, self.upload, self.deallocated, self.closed = served, upload, 0, False

    def MeshShape(self, *shape):
        return shape

    def open_mesh_device(self, shape):
        return 'mesh'

    def close_mesh_device(self, mesh):
        self.closed = True

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def from_torch(self, tensor, **keywords):
        bits = tensor.contiguous().view(torch.int16).clone()
        return Tensor(self.upload(bits) if self.upload else bits)

    def get_device_tensors(self, tensor):
        return [tensor]

    def to_torch(self, tensor):
        return tensor.bits.contiguous().view(torch.bfloat16)

    def synchronize_device(self, mesh):
        pass

    def deallocate(self, tensor):
        self.deallocated += 1

    def to_layout(self, tensor, layout):
        bits = self.served(tensor.bits) if (tensor.layout, layout) == ('tile', 'row') else tensor.bits.clone()
        return Tensor(bits, layout)

    def slice(self, tensor, start, end, **keywords):
        bits = tensor.bits[start[0]:end[0], start[1]:end[1], start[2]:end[2]].clone()
        if tensor.layout == 'tile' and start[1] % 32 != 0:
            bits = self.served(bits)
        return Tensor(bits, tensor.layout)


def fake_rows_dma(rule=exponent_zero_to_positive_zero):
    """The mover's contract on the fake: a task composes one destination tile from two half-tile sources; the canonical modes apply `rule`."""
    def launch(mesh, sources, destinations, task_list, canon_denorm=True):
        source, destination = sources[0], destinations[0]
        for _destination, page, first, second in task_list:
            tile = slice(page * 32, page * 32 + 32)
            for rows, (index, source_page, mode) in zip((slice(0, 16), slice(16, 32)), (first, second)):
                if mode == 0:
                    destination.bits[:, rows, tile] = 0
                    continue
                half = slice(((mode - 1) & 1) * 16, ((mode - 1) & 1) * 16 + 16)
                piece = source.bits[:, half, source_page * 32:source_page * 32 + 32].clone()
                if mode >= 3:
                    piece = rule(piece) if canon_denorm else negative_zero_only(piece)
                destination.bits[:, rows, tile] = piece

    return SimpleNamespace(mode=real_rows_dma.mode, launch=launch, tp_shapes=SimpleNamespace(chip_count=lambda environ=None: 4))


class PatternTests(unittest.TestCase):
    def test_every_bf16_pattern_exactly_once_in_64_tiles(self):
        signed, as_bf16 = probe.host_patterns(torch)
        self.assertEqual(tuple(signed.shape), (1, 32, 2048))
        self.assertEqual(sorted((signed.reshape(-1).to(torch.int32) & 0xFFFF).tolist()), list(range(65536)))
        self.assertTrue(torch.equal(as_bf16.contiguous().view(torch.int16), signed))
        self.assertEqual(2048 // 32, 64)

    def test_the_model_flushes_exactly_the_exponent_zero_patterns_to_positive_zero(self):
        signed, _ = probe.host_patterns(torch)
        model = probe.canonical_model(torch, signed)
        unsigned = signed.to(torch.int32) & 0xFFFF
        flushed = (unsigned & 0x7F80) == 0
        self.assertEqual(int(flushed.sum()), 256)                      # 128 exponent-zero patterns a sign: +-0 and 127 denormals
        self.assertTrue(bool((model[flushed] == 0).all()))
        self.assertTrue(torch.equal(model[~flushed], signed[~flushed]))
        only = probe.canonical_model(torch, signed, denorm=False)
        self.assertEqual(int((only != signed).sum()), 1)               # -0 alone

    def test_the_task_plan_takes_both_halves_of_every_tile_in_the_asked_modes(self):
        plan = probe.tasks(real_rows_dma, True)
        self.assertEqual(len(plan), 64)
        self.assertEqual(plan[5], (0, 5, (0, 5, 3), (0, 5, 4)))
        self.assertEqual(probe.tasks(real_rows_dma, False)[7], (0, 7, (0, 7, 1), (0, 7, 2)))


class RunTests(unittest.TestCase):
    def run_main(self, ttnn=None, rows_dma=None):
        ttnn = ttnn or FakeTTNN()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'u2.json'
            lines = []
            with patch('builtins.print', side_effect=lambda *a, **k: lines.append(' '.join(map(str, a)))):
                status = probe.main(['--out', str(out)], torch=torch, ttnn=ttnn, rows_dma=rows_dma or fake_rows_dma())
            report = json.loads(out.read_text())
        return status, lines, report, ttnn

    def test_a_mover_with_the_served_rule_passes_on_all_65536_patterns(self):
        status, lines, report, ttnn = self.run_main()
        self.assertEqual((status, report['verdict']), (0, 'PASS'))
        self.assertEqual(report['exercised'], 65536)
        self.assertTrue(all(report['compare'][name]['differing'] == 0 for name in report['compare']), report['compare'])
        self.assertEqual(report['rule_vs_served']['differing'], 0)
        changed = report['served_changes']
        self.assertEqual(changed['patterns'], 255)                # every exponent-zero pattern but +0
        self.assertTrue(changed['all_exponent_zero'] and changed['all_to_positive_zero'])
        self.assertTrue(any(line.startswith('U2_CANON verdict=PASS exercised=65536 raw_differing=0') for line in lines), lines)
        self.assertEqual(json.loads(lines[-1])['kind'], probe.KIND)
        self.assertTrue(ttnn.closed)
        self.assertGreater(ttnn.deallocated, 5)

    def test_a_mover_that_flushes_only_negative_zero_fails_and_names_the_denormals(self):
        status, lines, report, ttnn = self.run_main(rows_dma=fake_rows_dma(rule=negative_zero_only))
        self.assertEqual((status, report['verdict']), (1, 'FAIL'))
        half = report['compare']['canon_half1_vs_trip1']
        self.assertGreater(half['differing'], 0)
        self.assertEqual(half['patterns_total'], 127)                     # rows 16-31 hold the patterns from 0x8000: the 127 negative denormals (-0 agrees)
        self.assertIn('0x8001', half['patterns'])

    def test_a_served_path_that_does_not_flush_denormals_fails_the_model(self):
        status, lines, report, ttnn = self.run_main(ttnn=FakeTTNN(served=negative_zero_only))
        self.assertEqual((status, report['verdict']), (1, 'FAIL'))
        self.assertGreater(report['rule_vs_served']['differing'], 0)

    def test_an_upload_that_alters_patterns_is_a_partial_pass_that_says_how_many_were_not_exercised(self):
        def flush_on_upload(bits):
            return exponent_zero_to_positive_zero(bits)

        status, lines, report, ttnn = self.run_main(ttnn=FakeTTNN(upload=flush_on_upload))
        self.assertEqual((status, report['verdict']), (0, 'PASS-PARTIAL'))
        self.assertEqual(report['exercised'], 65536 - 255)                # -0 and the denormals changed on upload; +0 is unchanged
        self.assertEqual(report['input_roundtrip']['differing'], 255)

    def test_a_section_that_raises_is_not_run(self):
        ttnn = FakeTTNN()

        def broken(tensor, layout):
            raise RuntimeError('no untilize')

        ttnn.to_layout = broken
        status, lines, report, _ = self.run_main(ttnn=ttnn)
        self.assertEqual((status, report['verdict']), (4, 'NOT-RUN'))
        self.assertIn('no untilize', report['error'])

    def test_the_verdict_rule(self):
        good = dict(compare={name: dict(differing=0) for name in ('raw', 'canon_half0_vs_trip0', 'canon_half1_vs_trip1', 'canon_half1_vs_slice1')},
                    rule_vs_served=dict(differing=0), exercised=65536)
        self.assertEqual(probe.verdict(good), ('PASS', 0))
        self.assertEqual(probe.verdict(dict(good, exercised=65000)), ('PASS-PARTIAL', 0))
        self.assertEqual(probe.verdict(dict(good, rule_vs_served=dict(differing=2))), ('FAIL', 1))
        self.assertEqual(probe.verdict({})[0], 'NOT-RUN')


class HarnessTests(unittest.TestCase):
    def test_the_harness_mounts_the_mover_runs_one_card_with_the_serving_hook_off_and_unsets_the_mesh_descriptor(self):
        text = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')
        for fragment in ('gdn_rows_dma_tp.py gdn_rows_dma_tp.cpp tp_shapes.py verify_trace_t1.py',
                         '-e QWEN_C2_SERVING=0 --entrypoint env "$IMAGE" -u TT_MESH_GRAPH_DESC_PATH python3 -B /bench/canon_probe_card_m.py',
                         '--network none', '--cap-drop ALL', 'qual_card_select', 'qual_refuse_holders', 'qual_card_recheck'):
            self.assertIn(fragment, text)
        self.assertNotIn('\r', text)

    def test_every_file_the_harness_mounts_exists(self):
        for name in ('gdn_rows_dma_tp.py', 'gdn_rows_dma_tp.cpp', 'tp_shapes.py', 'verify_trace_t1.py'):
            self.assertTrue((CI / name).is_file(), name)

    def test_it_names_no_host_registry_or_card_outside_the_canonical_block(self):
        text = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')
        body = text[:text.index('# >>> qual_card.sh')] + text[text.index('# <<< qual_card.sh'):]
        for fragment in ('blackhole-', 'zot', '192.168'):
            self.assertNotIn(fragment, body)


if __name__ == '__main__':
    unittest.main()
