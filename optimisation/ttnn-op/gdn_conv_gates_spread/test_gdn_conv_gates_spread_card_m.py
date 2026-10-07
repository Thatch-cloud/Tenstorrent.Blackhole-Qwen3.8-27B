"""CPU tests of the F1 card-M probe: its host data, its compare, its verdict rule, its timing arithmetic and its whole run on a fake ttnn.

Run: python -B -m unittest discover -s optimisation/ttnn-op/gdn_conv_gates_spread -p 'test_*.py'
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
import gdn_conv_gates_spread as real_spread  # noqa: E402
import gdn_conv_gates_spread_card_m as probe  # noqa: E402

QKV, HEADS, A_COL, B_COL, WIDTH = 2560, 12, 4096, 4108, 4120
FOUND = SimpleNamespace(gdn_qkv=QKV, gdn_nv=HEADS, gdn_a_col=A_COL, gdn_b_col=B_COL, gdn_qkvzab=WIDTH)


class Tensor(object):
    def __init__(self, data):
        self.data = data


def serve(x, windows, rows):
    """A stand-in for the op: conv from the windows and x, the gates from the a and b columns, the windows advanced in place."""
    conv = Tensor((windows[1].data + windows[2].data + windows[3].data + x.data[..., :QKV]).to(torch.bfloat16))
    beta = Tensor(x.data[..., A_COL:A_COL + HEADS].clone())
    gate = Tensor(x.data[..., B_COL:B_COL + HEADS].clone())
    old = [window.data.clone() for window in windows]
    windows[0].data, windows[1].data, windows[2].data = old[1], old[2], old[3]
    windows[3].data = x.data[..., :QKV].clone()
    return conv, beta, gate


class FakeTTNN(object):
    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'

    def __init__(self):
        self.synchronized, self.closed = 0, False
        self.transformer = SimpleNamespace(gdn_decode_conv_gates=self.op)

    def MeshShape(self, *shape):
        return shape

    def open_mesh_device(self, shape, **keywords):
        self.open_keywords = keywords
        return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))

    def close_mesh_device(self, mesh):
        self.closed = True

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def from_torch(self, tensor, **keywords):
        return Tensor(tensor.clone())

    def get_device_tensors(self, tensor):
        return [tensor]

    def to_torch(self, tensor):
        return tensor.data

    def synchronize_device(self, mesh):
        self.synchronized += 1

    def deallocate(self, tensor):
        self.deallocated = getattr(self, 'deallocated', 0) + 1

    # a trace is a recorded list of costs (the test's wrappers add one per captured launch); a replay adds the sum to the test clock
    capturing = None
    costs = None
    clock = None

    def begin_trace_capture(self, mesh, cq_id=0):
        self.capturing = []
        return len(getattr(self, 'traces', {}))

    def end_trace_capture(self, mesh, handle, cq_id=0):
        self.traces = getattr(self, 'traces', {})
        self.traces[handle] = self.capturing
        self.capturing = None

    def execute_trace(self, mesh, handle, cq_id=0, blocking=True):
        self.replays = getattr(self, 'replays', []) + [handle]
        if self.clock is not None:
            self.clock['time'] += sum(self.traces[handle])

    def release_trace(self, mesh, handle):
        self.released = getattr(self, 'released', []) + [handle]

    def op(self, x, windows, taps, a, b, dt, neg, batch, memory_config, channels, a_col, b_col):
        return serve(x, windows, batch)


def fake_spread(mode='same'):
    def launch(ttnn, mesh, x, windows, taps, dt, neg, rows, channels, a_col, b_col, chips=None):
        assert chips == 1
        if mode == 'fall_back':
            return None
        conv, beta, gate = serve(x, windows, rows)
        if mode == 'flip_window':
            windows[2].data = windows[2].data.clone()
            windows[2].data.view(torch.int16)[0, 0, 0] ^= 1
        if mode == 'flip_gate':
            gate.data = gate.data.clone()
            gate.data.view(torch.int16)[0, 5, 3] ^= 1
        return conv, beta, gate

    return SimpleNamespace(launch=launch, plan=real_spread.plan, source_sha256=real_spread.source_sha256)


class HostDataTests(unittest.TestCase):
    def test_shapes_and_dtype(self):
        host = probe.host_data(torch, 64, 'random', 17, QKV, WIDTH, HEADS)
        self.assertEqual(tuple(host['x'].shape), (1, 64, WIDTH))
        self.assertEqual([tuple(value.shape) for value in host['windows']], [(1, 64, QKV)] * 4)
        self.assertEqual([tuple(value.shape) for value in host['taps']], [(1, 1, QKV)] * 4)
        self.assertEqual((tuple(host['dt_bias'].shape), tuple(host['neg_exp_A'].shape)), ((1, 1, HEADS), (1, 1, HEADS)))
        self.assertEqual(host['x'].dtype, torch.bfloat16)

    def test_no_infinity_or_nan_in_either_regime(self):
        for regime in probe.REGIMES:
            host = probe.host_data(torch, 128, regime, 3, QKV, WIDTH, HEADS)
            for tensor in (host['x'], *host['windows'], *host['taps']):
                bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
                self.assertFalse(bool(((bits & 0x7F80) == 0x7F80).any()), regime)

    def test_the_edge_regime_carries_the_edge_patterns_and_random_does_not_favour_them(self):
        edge = probe.host_data(torch, 64, 'edge', 5, QKV, WIDTH, HEADS)['x'].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        for pattern in (0x0000, 0x8000, 0x0001, 0x7F7F):
            self.assertTrue(bool((edge == pattern).any()), hex(pattern))
        plain = probe.host_data(torch, 64, 'random', 5, QKV, WIDTH, HEADS)['x'].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        self.assertLess(float((plain == 0).float().mean()), 0.001)
        self.assertGreater(float((edge == 0).float().mean()), 0.02)

    def test_seeded_and_distinct(self):
        first = probe.host_data(torch, 16, 'random', 17, QKV, WIDTH, HEADS)
        again = probe.host_data(torch, 16, 'random', 17, QKV, WIDTH, HEADS)
        other = probe.host_data(torch, 16, 'random', 18, QKV, WIDTH, HEADS)
        self.assertTrue(torch.equal(first['x'], again['x']))
        self.assertFalse(torch.equal(first['x'], other['x']))


class CompareTests(unittest.TestCase):
    def section(self, mode):
        ttnn = FakeTTNN()
        rig = probe.Rig(ttnn, 'mesh', torch, FOUND, fake_spread(mode))
        return probe.compare_case(rig, probe.host_data(torch, 64, 'edge', 9, QKV, WIDTH, HEADS), 64)

    def test_identical_arms_have_no_differing_element_in_any_of_the_seven_tensors(self):
        found = self.section('same')
        self.assertEqual((found['differing'], found['fell_back']), (0, False))
        self.assertEqual(sorted(found['by_tensor']), ['beta', 'conv', 'g', 'window0', 'window1', 'window2', 'window3'])

    def test_a_flipped_bit_in_a_window_or_a_gate_is_counted(self):
        self.assertEqual(self.section('flip_window')['by_tensor']['window2'], 1)
        self.assertEqual(self.section('flip_gate')['by_tensor']['g'], 1)

    def test_a_fall_back_is_a_section_that_cannot_pass(self):
        found = self.section('fall_back')
        self.assertTrue(found['fell_back'])
        self.assertEqual(probe.verdict([found])[0], 'FAIL')

    def test_a_shape_difference_counts_every_element_of_the_larger(self):
        self.assertEqual(probe.differing(torch, torch.zeros(2, 3, dtype=torch.int16), torch.zeros(3, 3, dtype=torch.int16)), 9)

    def test_negative_zero_against_positive_zero_differs_as_int16_bits(self):
        left = torch.tensor([0.0], dtype=torch.bfloat16).view(torch.int16)
        right = torch.tensor([-0.0], dtype=torch.bfloat16).view(torch.int16)
        self.assertEqual(probe.differing(torch, left, right), 1)


class VerdictTests(unittest.TestCase):
    def test_pass_fail_and_not_run(self):
        good = dict(rows=16, differing=0, fell_back=False)
        self.assertEqual(probe.verdict([good, good]), ('PASS', 0))
        self.assertEqual(probe.verdict([good, dict(rows=64, differing=3, fell_back=False)]), ('FAIL', 1))
        self.assertEqual(probe.verdict([good, dict(rows=64, error='RuntimeError: x')]), ('NOT-RUN', 4))
        self.assertEqual(probe.verdict([]), ('NOT-RUN', 4))
        self.assertEqual(probe.verdict([dict(rows=16, differing=0, fell_back=True)])[0], 'FAIL')

    def test_a_section_with_no_differing_key_cannot_pass(self):
        self.assertEqual(probe.verdict([dict(rows=16)])[0], 'FAIL')


class TimingTests(unittest.TestCase):
    def test_per_launch_microseconds_by_arm_from_captured_traces_in_serpentine_order(self):
        ttnn = FakeTTNN()
        spread = fake_spread('same')
        rig = probe.Rig(ttnn, 'mesh', torch, FOUND, spread)
        ticks = {'time': 0.0}
        ttnn.clock = ticks
        calls = []
        real_served, real_spread_launch = rig.served, rig.spreaded

        def served(tensors, rows):
            calls.append('served')
            if ttnn.capturing is not None:
                ttnn.capturing.append(0.00002)
            return real_served(tensors, rows)

        def spreaded(tensors, rows):
            calls.append('spread')
            if ttnn.capturing is not None:
                ttnn.capturing.append(0.00001)
            return real_spread_launch(tensors, rows)

        rig.served, rig.spreaded = served, spreaded
        result = probe.time_case(rig, probe.host_data(torch, 16, 'random', 1, QKV, WIDTH, HEADS), 16, launches=4, rounds=6, clock=lambda: ticks['time'])
        self.assertAlmostEqual(result['served']['median_us'], 20.0, places=3)
        self.assertAlmostEqual(result['spread']['median_us'], 10.0, places=3)
        self.assertAlmostEqual(result['spread_minus_served_us'], -10.0, places=3)
        self.assertEqual((result['served']['n'], result['mode']), (6, 'trace'))
        # a warm call then a four-launch capture per arm, then NO eager launch while timing: the replays carry the timing
        self.assertEqual(calls, ['served'] * 5 + ['spread'] * 5)
        self.assertEqual(ttnn.replays[:2], [0, 1])                      # the untimed replay of each arm
        timed = ttnn.replays[2:]
        self.assertEqual(timed, [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])   # serpentine
        self.assertEqual(sorted(ttnn.released), [0, 1])

    def test_every_device_tensor_is_released_after_a_case(self):
        ttnn = FakeTTNN()
        rig = probe.Rig(ttnn, 'mesh', torch, FOUND, fake_spread('same'))
        probe.compare_case(rig, probe.host_data(torch, 16, 'random', 1, QKV, WIDTH, HEADS), 16)
        self.assertGreater(ttnn.deallocated, 10)

    def test_the_mesh_is_opened_with_a_trace_region(self):
        ttnn = FakeTTNN()
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}), patch('builtins.print'), tempfile.TemporaryDirectory() as directory:
            probe.main(['--out', str(Path(directory) / 'x.json'), '--cases', '16', '--regimes', 'random', '--seeds', '1', '--timing', 'off'],
                       torch=torch, ttnn=ttnn, spread=fake_spread('same'), tp_shapes=SimpleNamespace(active=lambda: FOUND))
        self.assertGreater(ttnn.open_keywords['trace_region_size'], 0)

    def test_summary(self):
        found = probe.summarize([4.0, 1.0, 3.0, 2.0])
        self.assertEqual((found['n'], found['min_us']), (4, 1.0))
        self.assertLessEqual(found['q1_us'], found['median_us'])


class RunTests(unittest.TestCase):
    def run_main(self, mode, extra=(), tp='4'):
        ttnn = FakeTTNN()
        tp_shapes = SimpleNamespace(active=lambda: FOUND)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'f1.json'
            lines = []
            with patch.dict(os.environ, {'QWEN_FAST_TP': tp}), patch('builtins.print', side_effect=lambda *a, **k: lines.append(' '.join(map(str, a)))):
                status = probe.main(['--out', str(out), '--cases', '16,64', '--regimes', 'random,edge', '--seeds', '1', *extra],
                                    torch=torch, ttnn=ttnn, spread=fake_spread(mode), tp_shapes=tp_shapes)
            report = json.loads(out.read_text())
        return status, lines, report, ttnn

    def test_an_exact_run_passes_prints_the_verdict_then_the_json_and_closes_the_mesh(self):
        status, lines, report, ttnn = self.run_main('same', ['--timing', 'off'])
        self.assertEqual(status, 0)
        self.assertTrue(any(line.startswith('GDN_CG_SPREAD verdict=PASS sections=4 differing=0') for line in lines), lines)
        self.assertEqual(json.loads(lines[-1])['kind'], probe.KIND)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(len(report['compare']), 4)
        self.assertTrue(ttnn.closed)
        self.assertEqual(report['grid'], [11, 10])
        self.assertEqual(report['plans']['64']['conv_cores'], 80)
        self.assertEqual(report['plans']['64']['n_gate'], 2)
        self.assertEqual(set(report['new_kernels_sha256']), {'reader', 'writer'})

    def test_the_timing_runs_only_after_a_pass(self):
        with patch.object(probe, 'ROUNDS', 4), patch.object(probe, 'LAUNCHES', 2):
            status, lines, report, ttnn = self.run_main('same')
        self.assertEqual(status, 0)
        self.assertEqual([entry['rows'] for entry in report['timing']], [16, 64])
        status, lines, report, ttnn = self.run_main('flip_window', ['--timing', 'on'])
        self.assertEqual(status, 1)
        self.assertNotIn('timing', report)
        self.assertEqual(report['verdict'], 'FAIL')

    def test_a_fall_back_fails_the_run(self):
        status, lines, report, ttnn = self.run_main('fall_back', ['--timing', 'off'])
        self.assertEqual(status, 1)
        self.assertTrue(all(section['fell_back'] for section in report['compare']))

    def test_the_wrong_width_is_not_run(self):
        status, lines, report, ttnn = self.run_main('same', ['--timing', 'off'], tp='2')
        self.assertEqual(status, 4)
        self.assertEqual(report['verdict'], 'NOT-RUN')
        self.assertIn('QWEN_FAST_TP=4', report['error'])

    def test_bad_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), patch('sys.stderr'):
            for bad in (['--cases', '7'], ['--regimes', 'nan']):
                self.assertEqual(probe.main(['--out', str(Path(directory) / 'x.json'), *bad], torch=torch, ttnn=FakeTTNN(), spread=fake_spread(),
                                            tp_shapes=SimpleNamespace(active=lambda: FOUND)), 2)


class HarnessTests(unittest.TestCase):
    def test_the_harness_mounts_the_module_and_kernels_runs_one_card_with_the_serving_hook_off(self):
        text = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')
        for fragment in ('gdn_conv_gates_spread.py gdn_conv_gates_spread_reader.cpp gdn_conv_gates_spread_writer.cpp tp_shapes.py tp4_vglue.py verify_trace_t1.py',
                         '-e QWEN_FAST_TP=4', '-e QWEN_C2_SERVING=0 --entrypoint env "$IMAGE" -u TT_MESH_GRAPH_DESC_PATH python3 -B /bench/gdn_conv_gates_spread_card_m.py',
                         '--network none', '--cap-drop ALL', 'qual_card_select', 'qual_refuse_holders', 'qual_card_recheck'):
            self.assertIn(fragment, text)
        self.assertNotIn('\r', text)

    def test_every_file_the_harness_mounts_exists(self):
        for name in ('gdn_conv_gates_spread.py', 'gdn_conv_gates_spread_reader.cpp', 'gdn_conv_gates_spread_writer.cpp', 'tp_shapes.py', 'tp4_vglue.py',
                     'verify_trace_t1.py'):
            self.assertTrue((CI / name).is_file(), name)

    def test_it_names_no_host_registry_or_card_outside_the_canonical_block(self):
        text = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')
        body = text[:text.index('# >>> qual_card.sh')] + text[text.index('# <<< qual_card.sh'):]
        for fragment in ('blackhole-', 'zot', '192.168'):
            self.assertNotIn(fragment, body)


if __name__ == '__main__':
    unittest.main()
