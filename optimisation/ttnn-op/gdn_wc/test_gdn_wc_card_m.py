"""CPU tests of the WC card-M probe: its host data, its compare, its sections, its verdict rule, its timing arithmetic, its run script and its whole run on a fake ttnn.

Run: python -B -m unittest discover -s optimisation/ttnn-op/gdn_wc -p 'test_*.py'
"""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CI = ROOT / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))
import gdn_wc as real_wc  # noqa: E402
import gdn_wc_card_m as probe  # noqa: E402

QKV, HEADS, A_COL, B_COL, WIDTH = 2560, 12, 4096, 4108, 4120
FOUND = SimpleNamespace(gdn_qkv=QKV, gdn_nv=HEADS, gdn_a_col=A_COL, gdn_b_col=B_COL, gdn_qkvzab=WIDTH)


class Tensor(object):
    def __init__(self, data):
        self.data = data


def outputs(tensors):
    """A deterministic function of the CURRENT input data standing in for either arm: what a launch would write for these inputs."""
    users = tensors['users']
    x = tensors['x'].data
    made = dict(conv=[], beta=[], g=[], z=[], window=[])
    for user in range(users):
        rows = x[:, 16 * user:16 * user + 16, :]
        history = [value.data for value in tensors['histories'][user]]
        made['conv'].append(Tensor((rows[..., :QKV].float() + history[1].float()).to(torch.bfloat16)))
        made['beta'].append(Tensor(rows[..., A_COL:A_COL + HEADS].clone()))
        made['g'].append(Tensor(rows[..., B_COL:B_COL + HEADS].clone()))
        made['z'].append(Tensor(rows[..., QKV:A_COL].clone()))
        made['window'].append([Tensor(history[slot % 4].clone() if slot < 3 else rows[..., :QKV].clone()) for slot in range(4)])
    return made


class FakeTTNN(object):
    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'

    def __init__(self):
        self.closed, self.deallocated, self.traces, self.recording = False, 0, {}, None
        self.replays = []

    def MeshShape(self, *shape):
        return shape

    def open_mesh_device(self, shape, **keywords):
        self.open_keywords = keywords
        return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10), num_program_cache_entries=lambda: 7)

    def close_mesh_device(self, mesh):
        self.closed = True

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def from_torch(self, tensor, **keywords):
        return Tensor(tensor.clone())

    def copy_host_to_device_tensor(self, staged, device):
        device.data = staged.data.clone()

    def get_device_tensors(self, tensor):
        return [tensor]

    def to_torch(self, tensor):
        return tensor.data

    def synchronize_device(self, mesh):
        pass

    def deallocate(self, tensor):
        self.deallocated += 1

    def begin_trace_capture(self, mesh, cq_id=0):
        self.recording = []
        return len(self.traces)

    def end_trace_capture(self, mesh, handle, cq_id=0):
        self.traces[handle] = self.recording
        self.recording = None

    def execute_trace(self, mesh, handle, cq_id=0, blocking=True):
        self.replays.append(handle)
        for refill in self.traces[handle]:
            refill()

    def release_trace(self, mesh, handle):
        pass


def fake_chain():
    return SimpleNamespace(served_path=lambda ttnn, mesh, found, tensors: outputs(tensors))


def fake_wc(mode='same', ttnn_holder=None):
    """mode: same | fall_back | flip (one element of one window)."""
    def launch(ttnn, mesh, x, histories, taps, dt_bias, neg_exp_A, chips=None):
        assert chips == 1
        if mode == 'fall_back':
            return None
        tensors = dict(users=len(histories), x=x, histories=histories)
        made = outputs(tensors)
        if mode == 'flip':
            made['window'][1][2].data = made['window'][1][2].data.clone()
            made['window'][1][2].data.view(torch.int16)[0, 0, 0] ^= 1
        if ttnn.recording is not None:
            def refill(made=made, tensors=tensors):
                fresh = outputs(tensors)
                for name in ('conv', 'beta', 'g', 'z'):
                    for target, source in zip(made[name], fresh[name]):
                        target.data = source.data
                for target_user, source_user in zip(made['window'], fresh['window']):
                    for target, source in zip(target_user, source_user):
                        target.data = source.data
            ttnn.recording.append(refill)
        return made

    return SimpleNamespace(launch=launch, plan=real_wc.plan, source_sha256=real_wc.source_sha256)


class Rig(probe.Rig):
    pass


def rig(mode='same'):
    ttnn = FakeTTNN()
    mesh = ttnn.open_mesh_device(None)
    return Rig(ttnn, mesh, torch, FOUND, fake_wc(mode), fake_chain()), ttnn


class HostDataTests(unittest.TestCase):
    def test_shapes_and_dtype(self):
        host = probe.host_data(torch, 4, 'random', 17, FOUND)
        self.assertEqual(tuple(host['x'].shape), (1, 64, WIDTH))
        self.assertEqual([[tuple(value.shape) for value in user] for user in host['histories']], [[(1, 1, QKV)] * 4] * 4)
        self.assertEqual([tuple(value.shape) for value in host['taps']], [(1, 1, QKV)] * 4)
        self.assertEqual((tuple(host['dt_bias'].shape), tuple(host['neg_exp_A'].shape)), ((1, 1, HEADS), (1, 1, HEADS)))
        self.assertEqual(host['x'].dtype, torch.bfloat16)
        self.assertEqual(tuple(probe.host_data(torch, 2, 'random', 17, FOUND)['x'].shape), (1, 32, WIDTH))

    def test_no_infinity_or_nan_in_any_regime(self):
        for regime in probe.REGIMES:
            host = probe.host_data(torch, 4, regime, 3, FOUND)
            for tensor in (host['x'], *host['taps'], *[value for user in host['histories'] for value in user]):
                bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
                self.assertFalse(bool(((bits & 0x7F80) == 0x7F80).any()), regime)

    def test_the_edge_regime_carries_the_edge_patterns(self):
        bits = probe.host_data(torch, 4, 'edge', 5, FOUND)['x'].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        for pattern in (0x0000, 0x8000, 0x0001, 0x7F7F):
            self.assertTrue(bool((bits == pattern).any()), hex(pattern))

    def test_seeded_and_distinct(self):
        left, again, other = (probe.host_data(torch, 4, 'random', seed, FOUND) for seed in (1, 1, 2))
        self.assertTrue(torch.equal(left['x'].contiguous().view(torch.int16), again['x'].contiguous().view(torch.int16)))
        self.assertFalse(torch.equal(left['x'].contiguous().view(torch.int16), other['x'].contiguous().view(torch.int16)))


class CompareTests(unittest.TestCase):
    def test_differing_counts_elements_and_shape_mismatches(self):
        left = torch.zeros(2, 3, dtype=torch.int16)
        right = left.clone()
        right[1, 2] = 5
        self.assertEqual(probe.differing(torch, left, left), 0)
        self.assertEqual(probe.differing(torch, left, right), 1)
        self.assertEqual(probe.differing(torch, left, torch.zeros(3, 3, dtype=torch.int16)), 9)

    def test_an_identical_case_is_exact_and_counts_the_tensors(self):
        r, ttnn = rig()
        section = probe.compare_case(r, probe.host_data(torch, 4, 'random', 17, FOUND))
        self.assertEqual((section['differing'], section['fell_back'], section['compared'], section['users']), (0, False, 32, 4))
        self.assertGreater(ttnn.deallocated, 0, 'the case frees what it made')

    def test_two_users_compare_sixteen_tensors(self):
        r, ttnn = rig()
        self.assertEqual(probe.compare_case(r, probe.host_data(torch, 2, 'edge', 17, FOUND))['compared'], 16)

    def test_one_flipped_element_is_found_and_named(self):
        r, ttnn = rig('flip')
        section = probe.compare_case(r, probe.host_data(torch, 4, 'random', 17, FOUND))
        self.assertEqual(section['differing'], 1)
        self.assertEqual(list(section['by_tensor']), ['window2 u1'])

    def test_a_fall_back_is_a_failed_case(self):
        r, ttnn = rig('fall_back')
        section = probe.compare_case(r, probe.host_data(torch, 4, 'random', 17, FOUND))
        self.assertTrue(section['fell_back'])
        self.assertEqual(section['differing'], -1)


class SectionTests(unittest.TestCase):
    def test_the_replay_reads_the_launch_after_the_buffers_were_restaged_with_the_other_dataset(self):
        r, ttnn = rig()
        first, second = probe.host_data(torch, 4, 'random', 17, FOUND), probe.host_data(torch, 4, 'random', 99, FOUND)
        section = probe.replay_section(r, first, second)
        self.assertEqual((section['differing'], section['fell_back']), (0, False))
        self.assertEqual(len(ttnn.replays), 1)

    def test_a_trace_that_baked_the_data_instead_of_the_addresses_is_caught(self):
        r, ttnn = rig()
        ttnn.execute_trace = lambda mesh, handle, cq_id=0, blocking=True: ttnn.replays.append(handle)     # replays nothing: the launch keeps the first data
        first, second = probe.host_data(torch, 4, 'random', 17, FOUND), probe.host_data(torch, 4, 'random', 99, FOUND)
        self.assertGreater(probe.replay_section(r, first, second)['differing'], 0)

    def test_the_cache_must_not_grow_on_repeated_launches(self):
        r, ttnn = rig()
        section = probe.cache_section(r, probe.host_data(torch, 4, 'random', 17, FOUND))
        self.assertEqual((section['before'], section['after_first'], section['after_repeat'], section['stable']), (7, 7, 7, True))
        entries = iter(range(100))
        r.mesh.num_program_cache_entries = lambda: next(entries)
        self.assertFalse(probe.cache_section(r, probe.host_data(torch, 4, 'random', 17, FOUND))['stable'])

    def test_a_mesh_that_does_not_report_the_cache_is_not_a_failure(self):
        r, ttnn = rig()
        del r.mesh.num_program_cache_entries
        self.assertTrue(probe.cache_section(r, probe.host_data(torch, 4, 'random', 17, FOUND))['stable'])

    def test_the_control_differs_for_swapped_histories(self):
        r, ttnn = rig()
        self.assertGreater(probe.control_section(r, probe.host_data(torch, 4, 'random', 17, FOUND))['differing'], 0)

    def test_a_control_that_does_not_differ_fails_the_verdict(self):
        # data in which swapping two users' histories changes nothing leaves the compare blind to a wrong history: the verdict refuses it
        good = dict(differing=0, fell_back=False)
        self.assertEqual(probe.verdict([good], good, dict(stable=True), dict(differing=0, fell_back=False)), ('FAIL', 1))
        r, ttnn = rig()
        host = probe.host_data(torch, 4, 'random', 17, FOUND)
        host['histories'][1] = [value.clone() for value in host['histories'][0]]
        self.assertEqual(probe.control_section(r, host)['differing'], 0)


class VerdictTests(unittest.TestCase):
    good = dict(differing=0, fell_back=False)

    def test_pass(self):
        self.assertEqual(probe.verdict([self.good], self.good, dict(stable=True), dict(differing=3)), ('PASS', 0))

    def test_any_difference_or_fall_back_fails(self):
        for bad in (dict(differing=1, fell_back=False), dict(differing=-1, fell_back=True), dict(differing=0, fell_back=True)):
            self.assertEqual(probe.verdict([self.good, bad], self.good, dict(stable=True), dict(differing=3)), ('FAIL', 1))
        self.assertEqual(probe.verdict([self.good], dict(differing=2, fell_back=False), dict(stable=True), dict(differing=3)), ('FAIL', 1))
        self.assertEqual(probe.verdict([self.good], self.good, dict(stable=False), dict(differing=3)), ('FAIL', 1))

    def test_an_error_or_nothing_run_is_not_run(self):
        self.assertEqual(probe.verdict([], self.good, dict(stable=True), dict(differing=3)), ('NOT-RUN', 4))
        self.assertEqual(probe.verdict([dict(error='x')], self.good, dict(stable=True), dict(differing=3)), ('NOT-RUN', 4))
        self.assertEqual(probe.verdict([self.good], dict(error='x'), dict(stable=True), dict(differing=3)), ('NOT-RUN', 4))


class TimingTests(unittest.TestCase):
    def test_summarize(self):
        found = probe.summarize([4.0, 1.0, 3.0, 2.0])
        self.assertEqual((found['n'], found['min_us']), (4, 1.0))
        self.assertEqual(found['median_us'], 2.5)

    def test_the_timing_arithmetic_is_per_launch_microseconds(self):
        r, ttnn = rig()
        state = {'time': 0.0}
        costs = {0: 240e-6, 1: 24e-6}
        original = ttnn.execute_trace

        def execute(mesh, handle, cq_id=0, blocking=True):
            original(mesh, handle, cq_id, blocking)
            state['time'] += costs[handle]

        ttnn.execute_trace = execute
        result = probe.time_case(r, probe.host_data(torch, 4, 'random', 17, FOUND), launches=24, rounds=5, clock=lambda: state['time'])
        self.assertEqual(result['mode'], 'trace')
        self.assertAlmostEqual(result['served']['median_us'], 10.0, places=1)
        self.assertAlmostEqual(result['wc']['median_us'], 1.0, places=1)
        self.assertAlmostEqual(result['wc_minus_served_us'], -9.0, places=1)


class RunTests(unittest.TestCase):
    def run_main(self, mode='same', environment=None, extra=()):
        ttnn = FakeTTNN()
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, 'report.json')
            environ = dict(os.environ)
            environ.update(environment if environment is not None else {'QWEN_FAST_TP': '4'})
            old = dict(os.environ)
            os.environ.clear()
            os.environ.update(environ)
            try:
                status = probe.main(['--out', out, '--users', '4', '--regimes', 'random', '--seeds', '17', '--timing', 'off', *extra],
                                    torch=torch, ttnn=ttnn, wc=fake_wc(mode), chain=fake_chain(),
                                    tp_shapes=SimpleNamespace(active=lambda: FOUND))
            finally:
                os.environ.clear()
                os.environ.update(old)
            report = None
            if os.path.exists(out):
                with open(out) as handle:
                    report = json.load(handle)
        return status, report, ttnn

    def test_a_clean_run_passes_and_closes_the_mesh(self):
        status, report, ttnn = self.run_main()
        self.assertEqual((status, report['verdict'], report['kind']), (0, 'PASS', 'wc-card-m'))
        self.assertTrue(ttnn.closed)
        self.assertEqual(report['geometry']['channels'], QKV)
        for key in ('compare', 'replay', 'cache', 'control', 'plans', 'new_kernels_sha256'):
            self.assertIn(key, report)
        self.assertEqual(set(report['plans']['4']), {'conv_cores', 'gate_cores', 'z_cores', 'per_core', 'served_per_core', 'cores', 'n_conv'})
        self.assertEqual((report['plans']['4']['conv_cores'], report['plans']['4']['gate_cores'], report['plans']['4']['z_cores']), (80, 2, 28))

    def test_a_difference_fails(self):
        status, report, ttnn = self.run_main('flip')
        self.assertEqual((status, report['verdict']), (1, 'FAIL'))

    def test_a_fall_back_fails(self):
        status, report, ttnn = self.run_main('fall_back')
        self.assertEqual((status, report['verdict']), (1, 'FAIL'))

    def test_without_the_four_card_geometry_nothing_runs(self):
        status, report, ttnn = self.run_main(environment={'QWEN_FAST_TP': '2'})
        self.assertEqual((status, report['verdict']), (4, 'NOT-RUN'))
        self.assertIn('QWEN_FAST_TP=4', report['error'])

    def test_a_bad_argument_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            status = probe.main(['--out', os.path.join(directory, 'r.json'), '--users', '3'], torch=torch, ttnn=FakeTTNN(), wc=fake_wc(),
                                chain=fake_chain(), tp_shapes=SimpleNamespace(active=lambda: FOUND))
        self.assertEqual(status, 2)


class ScriptTests(unittest.TestCase):
    script = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')

    def test_it_mounts_exactly_files_that_exist_and_the_probe_and_the_chain(self):
        names = re.search(r'for file in ([^;]+); do', self.script).group(1).replace('\\\n', ' ').split()
        self.assertTrue(names)
        for name in names:
            self.assertTrue((CI / name).is_file(), name)
        for name in ('gdn_wc.py', 'gdn_wc_reader.cpp', 'gdn_wc_writer.cpp', 'gdn_block_conv_tp.py', 'gdn_conv_windows_packed.py', 'gdn_conv_windows_packed.cpp',
                     'gdn_rows_dma_tp.py', 'gdn_rows_dma_tp.cpp'):
            self.assertIn(name, names)
        self.assertIn('gdn_wc_card_m.py gdn_wc_card_m_chain.py', self.script)
        self.assertTrue((HERE / 'gdn_wc_card_m.py').is_file() and (HERE / 'gdn_wc_card_m_chain.py').is_file())

    def test_it_runs_the_probe_in_a_container_without_network_and_names_no_host(self):
        self.assertIn('--network none', self.script)
        self.assertIn('python3 -B /bench/gdn_wc_card_m.py', self.script)
        self.assertNotRegex(self.script, r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        self.assertNotIn('/home/', self.script)
        self.assertNotIn('f1probe', self.script)

    def test_it_embeds_the_qual_card_block_and_refuses_card_b(self):
        self.assertIn('>>> qual_card.sh', self.script)
        self.assertIn('qual_card_select', self.script)
        self.assertIn('QUAL_RESERVED_CARD', self.script)

    def test_the_chain_helper_is_the_pinned_resident_piece_the_t2_windows_and_the_served_op(self):
        chain = (HERE / 'gdn_wc_card_m_chain.py').read_text(encoding='utf-8')
        for token in ('pinned.resident_piece', 'build_windows_packed', 'gdn_user_batch_conv.conv_gates', 'ttnn.slice(pieces[user]'):
            self.assertIn(token, chain)

    def test_files_are_lf(self):
        for name in ('run_card_m.sh', 'gdn_wc_card_m.py', 'gdn_wc_card_m_chain.py', 'test_gdn_wc_card_m.py'):
            self.assertNotIn(b'\r', (HERE / name).read_bytes(), name)


if __name__ == '__main__':
    unittest.main()
