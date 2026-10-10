"""CPU tests of the S1 card-M probe: its host data, its compare, its verdict rules, its timing arithmetic and its whole run on a fake ttnn.

Run: python -B -m unittest discover -s optimisation/ttnn-op/shard_argmax -p 'test_*.py'
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
import shard_argmax_card_m as probe  # noqa: E402
import tp4_shard_argmax as real_sarg  # noqa: E402

SHARD = probe.SHARD


class Tensor(object):
    def __init__(self, data):
        self.data = data


class FakeTTNN(object):
    bfloat16, uint32, TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'u32', 'tile', 'row_major', 'dram'

    def __init__(self, grid=(11, 10)):
        self.grid = grid
        self.closed = False
        self.deallocated = 0

    def MeshShape(self, *shape):
        return shape

    def open_mesh_device(self, shape, **keywords):
        self.open_keywords = keywords
        grid = self.grid
        return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]))

    def close_mesh_device(self, mesh):
        self.closed = True

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def empty(self, shape, dtype=None, layout=None, device=None, memory_config=None):
        torch_dtype = {'bf16': torch.bfloat16, 'u32': torch.int32}[dtype]
        return Tensor(torch.zeros(tuple(shape), dtype=torch_dtype))

    def from_torch(self, tensor, **keywords):
        return Tensor(tensor.clone())

    def get_device_tensors(self, tensor):
        return [tensor]

    def to_torch(self, tensor):
        return tensor.data

    def synchronize_device(self, mesh):
        pass

    def deallocate(self, tensor):
        self.deallocated += 1

    # a trace is a recorded list of costs (the test's wrappers add one per captured launch); a replay adds the sum to the test clock
    capturing = None
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


def answer(logits, rows):
    """torch.argmax and the element's bits for the (1, 1, rows, SHARD) logits."""
    matrix = logits.data.reshape(rows, SHARD)
    ids = torch.argmax(matrix, dim=1)
    values = matrix[torch.arange(rows), ids]
    return ids, values


def fake_sarg(mode='same'):
    """tp4_shard_argmax with a launch that computes the answer with torch; `mode` breaks one thing in one fold."""
    def launch(ttnn, logits, rows, partials, groups, ids, values, words, grid, fold2, scan_only=False):
        if ttnn.capturing is not None:
            ttnn.capturing.append(ttnn.costs[('scan' if scan_only else ('fold2' if fold2 else 'fold1'))])
        if scan_only:
            return 1
        found_ids, found_values = answer(logits, rows)
        ids.data = torch.zeros(1, 1, 1, 64, dtype=torch.int32)
        values.data = torch.zeros(1, 1, 1, 64, dtype=torch.bfloat16)
        words.data = torch.zeros(1, 1, 1, 64, dtype=torch.int32)
        ids.data[0, 0, 0, :rows] = found_ids.to(torch.int32)
        values.data[0, 0, 0, :rows] = found_values
        packed = real_sarg.pack_words(found_ids, found_values)
        words.data[0, 0, 0, :rows] = packed.where(packed < (1 << 31), packed - (1 << 32)).to(torch.int32)
        if mode == 'flip_id' and fold2:
            ids.data[0, 0, 0, 5] ^= 1
        if mode == 'flip_value_bit' and not fold2:
            values.data.view(torch.int16)[0, 0, 0, 2] ^= 1
        if mode == 'flip_word':
            words.data[0, 0, 0, 3] ^= 1
        if mode == 'dirty_tail' and not fold2 and rows < 64:
            ids.data[0, 0, 0, rows] = 7
        return 3 if fold2 else 2

    return SimpleNamespace(launch=launch, pack_words=real_sarg.pack_words, PAGE_WORDS=32, FOLD2_GROUPS=8, GROUP_PAGE_WORDS=64,
                           column_runs=real_sarg.column_runs, group_plan=real_sarg.group_plan, __file__=real_sarg.__file__)


def fake_t1(mode='same', ttnn=None):
    def served_shards(operations, logits, rows):
        found_ids, found_values = answer(logits, rows)
        values = found_values.clone()
        if mode == 'served_differs':
            values.view(torch.int16)[1] ^= 2
        if mode == 'served_signed_zero':
            pass
        if operations.capturing is not None:
            operations.capturing.append(operations.costs['served'])
        return Tensor(found_ids.to(torch.int32).reshape(1, 1, rows, 1)), Tensor(values.reshape(1, 1, rows, 1))
    return SimpleNamespace(served_shards=served_shards)


class HostDataTests(unittest.TestCase):
    PAIRS = probe.tie_pairs(real_sarg, 110)

    def test_tie_pairs_are_adjacent_columns_across_every_boundary_the_kernels_cut_at(self):
        self.assertTrue(all(b == a + 1 for a, b in self.PAIRS))
        self.assertTrue(all(0 <= a and b < SHARD for a, b in self.PAIRS))
        for needle in ((15, 16), (31, 32), (SHARD - 2, SHARD - 1)):
            self.assertIn(needle, self.PAIRS)
        runs = real_sarg.column_runs(SHARD // 32, 110)
        for first, _last in runs[1:]:
            self.assertIn((32 * first - 1, 32 * first), self.PAIRS)
        self.assertGreater(len(self.PAIRS), 200)
        wide = probe.tie_pairs(real_sarg, 130)
        self.assertNotEqual(wide, self.PAIRS)
        self.assertTrue(all(b == a + 1 and b < SHARD for a, b in wide))

    def test_shapes_and_dtype_for_every_regime(self):
        for regime in probe.REGIMES:
            host = probe.host_logits(torch, 16, regime, 17, self.PAIRS)
            self.assertEqual(tuple(host.shape), (1, 1, 16, SHARD), regime)
            self.assertEqual(host.dtype, torch.bfloat16)

    def test_seeded_and_distinct(self):
        first = probe.host_logits(torch, 16, 'random', 17, self.PAIRS)
        again = probe.host_logits(torch, 16, 'random', 17, self.PAIRS)
        other = probe.host_logits(torch, 16, 'random', 18, self.PAIRS)
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))

    def test_the_ties_regime_plants_two_equal_maxima_in_every_row(self):
        host = probe.host_logits(torch, 64, 'ties', 17, self.PAIRS).reshape(64, SHARD)
        top = host.max(dim=1).values
        self.assertTrue(bool((top == 25.0).all()))
        self.assertTrue(bool(((host == 25.0).sum(dim=1) >= 2).all()))

    def test_the_negatives_regime_is_all_negative_with_a_tied_maximum(self):
        host = probe.host_logits(torch, 32, 'negatives', 3, self.PAIRS).reshape(32, SHARD)
        self.assertTrue(bool((host < 0).all()))
        self.assertTrue(bool(((host == host.max(dim=1, keepdim=True).values).sum(dim=1) >= 2).all()))

    def test_the_zeros_regime_has_both_zeros_at_the_top_in_both_orders(self):
        host = probe.host_logits(torch, 16, 'zeros', 5, self.PAIRS).reshape(16, SHARD)
        bits = probe.bits16(torch, host)
        self.assertTrue(bool(((bits == 0x8000).sum(dim=1) == 1).all()))
        self.assertTrue(bool(((bits == 0x0000).sum(dim=1) == 1).all()))
        order = []
        for row in range(16):
            order.append(int(torch.nonzero(bits[row] == 0x8000)[0, 0]) < int(torch.nonzero(bits[row] == 0x0000)[0, 0]))
        self.assertEqual(set(order), {True, False})

    def test_the_special_regime_carries_infinity_nan_and_minus_infinity_rows(self):
        host = probe.host_logits(torch, 16, 'special', 7, self.PAIRS).reshape(16, SHARD)
        self.assertTrue(bool(torch.isinf(host).any()))
        self.assertTrue(bool(torch.isnan(host).any()))
        self.assertTrue(bool((host[3] == float('-inf')).all()))

    def test_an_unknown_regime_raises(self):
        with self.assertRaises(ValueError):
            probe.host_logits(torch, 4, 'nan', 1, self.PAIRS)


class ReferenceTests(unittest.TestCase):
    def test_reference_is_torch_argmax_and_the_winning_bits(self):
        matrix = torch.zeros(2, SHARD, dtype=torch.bfloat16)
        matrix[0, 10] = 3.0
        matrix[0, 20] = 3.0
        matrix[1, 5] = -1.0
        matrix[1] -= 1.0
        matrix[1, 5] = 1.0
        ids, bits = probe.reference_of(torch, matrix)
        self.assertEqual(ids.tolist(), [10, 5])
        self.assertEqual(bits.tolist(), [int(probe.bits16(torch, torch.tensor([3.0], dtype=torch.bfloat16))[0]),
                                         int(probe.bits16(torch, torch.tensor([1.0], dtype=torch.bfloat16))[0])])

    def test_numbers_equal_treats_signed_zeros_as_equal_and_nan_as_nan(self):
        left = torch.tensor([0x8000, 0x0000, 0x7FC0, 0x3F80, 0x7F80], dtype=torch.int64)
        right = torch.tensor([0x0000, 0x8000, 0x7FC1, 0x3F81, 0x7F80], dtype=torch.int64)
        self.assertEqual(probe.numbers_equal(torch, left, right).tolist(), [True, True, True, False, True])


class CompareTests(unittest.TestCase):
    def sections(self, mode, rows=64, regime='ties', folds=probe.FOLDS, t1_mode='same'):
        ttnn = FakeTTNN()
        rig = probe.Rig(ttnn, 'mesh', torch, fake_sarg(mode), fake_t1(t1_mode), (11, 10))
        rig.reserve()
        pairs = probe.tie_pairs(real_sarg, 110)
        return probe.compare_case(rig, probe.host_logits(torch, rows, regime, 9, pairs), rows, regime, folds), ttnn

    def test_identical_arms_have_no_differing_row_word_or_tail(self):
        found, _ = self.sections('same')
        self.assertEqual([(section['fold'], section['differing'], section['words_differing'], section['served_differing']) for section in found],
                         [('single', 0, 0, 0), ('tree', 0, 0, 0)])
        self.assertEqual(probe.verdict(found), ('PASS', 0))

    def test_a_wrong_id_in_one_fold_names_the_fold_and_the_row_and_fails(self):
        found, _ = self.sections('flip_id')
        by_fold = {section['fold']: section for section in found}
        self.assertEqual((by_fold['single']['differing'], by_fold['tree']['differing']), (0, 1))
        self.assertEqual(by_fold['tree']['differing_rows'], [5])
        self.assertEqual(probe.verdict(found), ('FAIL', 1))

    def test_a_flipped_value_bit_is_a_difference_even_when_it_is_the_same_number(self):
        found, _ = self.sections('flip_value_bit')
        self.assertEqual({section['fold']: section['differing'] for section in found}, {'single': 1, 'tree': 0})

    def test_a_wrong_word_fails_on_its_own(self):
        found, _ = self.sections('flip_word')
        self.assertEqual([section['words_differing'] for section in found], [1, 1])
        self.assertEqual(probe.verdict(found)[0], 'FAIL')

    def test_a_dirty_tail_past_the_live_rows_is_a_word_difference(self):
        found, _ = self.sections('dirty_tail', rows=16)
        self.assertEqual({section['fold']: section['words_differing'] for section in found}, {'single': 1, 'tree': 0})

    def test_a_served_difference_fails_except_in_the_informational_regime(self):
        found, _ = self.sections('same', regime='random', t1_mode='served_differs')
        self.assertTrue(all(section['served_differing'] > 0 for section in found))
        self.assertEqual(probe.verdict(found)[0], 'FAIL')
        found, _ = self.sections('same', regime='special', t1_mode='served_differs')
        self.assertTrue(all(section['informational'] for section in found))
        self.assertEqual(probe.verdict(found)[0], 'PASS')

    def test_an_uncertain_section_does_not_count_either_way(self):
        found, _ = self.sections('flip_id')
        for section in found:
            section['uncertain'] = True
        self.assertEqual(probe.verdict(found)[0], 'PASS')

    def test_every_device_tensor_is_released_after_a_case(self):
        _, ttnn = self.sections('same', rows=16)
        self.assertGreater(ttnn.deallocated, 8)


class VerdictTests(unittest.TestCase):
    def test_pass_fail_and_not_run(self):
        good = dict(rows=64, differing=0, words_differing=0, served_differing=0, fell_back=False)
        self.assertEqual(probe.verdict([good, good]), ('PASS', 0))
        self.assertEqual(probe.verdict([good, dict(good, differing=2)]), ('FAIL', 1))
        self.assertEqual(probe.verdict([good, dict(rows=64, error='RuntimeError: x')]), ('NOT-RUN', 4))
        self.assertEqual(probe.verdict([]), ('NOT-RUN', 4))
        self.assertEqual(probe.verdict([dict(good, fell_back=True)])[0], 'FAIL')

    def test_a_section_with_no_differing_key_cannot_pass(self):
        self.assertEqual(probe.verdict([dict(rows=64)])[0], 'FAIL')

    def test_timing_verdicts(self):
        def entry(served, scan, one, two):
            return dict(rows=64, served=dict(median_us=served), scan=dict(median_us=scan), fold1=dict(median_us=one), fold2=dict(median_us=two))
        self.assertEqual(probe.timing_verdict([entry(1030, 90, 300, 170)]), ('S1-WIN', 'FOLD2-WIN'))
        self.assertEqual(probe.timing_verdict([entry(1030, 90, 300, 290)]), ('S1-WIN', 'FOLD2-NEUTRAL'))
        self.assertEqual(probe.timing_verdict([entry(1030, 90, 300, 400)]), ('S1-WIN', 'FOLD2-LOSS'))
        self.assertEqual(probe.timing_verdict([entry(700, 90, 300, 400)]), ('S1-NO-WIN', 'FOLD2-LOSS'))
        self.assertEqual(probe.timing_verdict([dict(entry(1, 1, 1, 1), rows=32)]), ('NOT-TIMED', 'NOT-TIMED'))


class TimingTests(unittest.TestCase):
    def test_per_launch_microseconds_by_arm_from_captured_traces_in_serpentine_order(self):
        ttnn = FakeTTNN()
        ticks = {'time': 0.0}
        ttnn.clock = ticks
        ttnn.costs = dict(served=0.001, scan=0.00009, fold1=0.0003, fold2=0.00017)
        rig = probe.Rig(ttnn, 'mesh', torch, fake_sarg(), fake_t1(), (11, 10))
        rig.reserve()
        pairs = probe.tie_pairs(real_sarg, 110)
        calls = []
        real_launch, real_served = rig.sarg.launch, rig.t1.served_shards

        def launch(*args, **keywords):
            calls.append(('scan' if keywords.get('scan_only') or (len(args) > 10 and args[10]) else ('tree' if args[9] else 'single'), args[0].capturing is not None))
            return real_launch(*args, **keywords)

        def served(operations, logits, rows):
            calls.append(('served', operations.capturing is not None))
            return real_served(operations, logits, rows)

        rig.sarg.launch, rig.t1.served_shards = launch, served
        result = probe.time_case(rig, probe.host_logits(torch, 16, 'random', 1, pairs), 16, launches=4, rounds=6, clock=lambda: ticks['time'])
        # every arm is called once eagerly (compiled outside the capture), then `launches` times inside its capture, and never while timing
        for arm in ('served', 'scan', 'single', 'tree'):
            self.assertEqual([captured for name, captured in calls if name == arm], [False] + [True] * 4, arm)
        self.assertAlmostEqual(result['served']['median_us'], 1000.0, places=3)
        self.assertAlmostEqual(result['scan']['median_us'], 90.0, places=3)
        self.assertAlmostEqual(result['fold1']['median_us'], 300.0, places=3)
        self.assertAlmostEqual(result['fold2']['median_us'], 170.0, places=3)
        self.assertAlmostEqual(result['fold1_cost_us'], 210.0, places=3)
        self.assertAlmostEqual(result['fold2_cost_us'], 80.0, places=3)
        self.assertAlmostEqual(result['s1_saves_us'], 830.0, places=3)
        self.assertAlmostEqual(result['fold2_gain_us'], 130.0, places=3)
        self.assertEqual((result['served']['n'], result['mode']), (6, 'trace'))
        self.assertEqual(ttnn.replays[:4], [0, 1, 2, 3])                    # the untimed replay of each arm
        timed = ttnn.replays[4:]
        self.assertEqual(timed[:4], [0, 1, 2, 3])                          # serpentine: forward, then reversed
        self.assertEqual(timed[4:8], [3, 2, 1, 0])
        self.assertEqual(sorted(ttnn.released), [0, 1, 2, 3])

    def test_summary(self):
        found = probe.summarize([4.0, 1.0, 3.0, 2.0])
        self.assertEqual((found['n'], found['min_us']), (4, 1.0))
        self.assertLessEqual(found['q1_us'], found['median_us'])

    def test_the_mesh_is_opened_with_a_trace_region(self):
        ttnn = FakeTTNN()
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}), patch('builtins.print'), tempfile.TemporaryDirectory() as directory:
            probe.main(['--out', str(Path(directory) / 'x.json'), '--rows', '16', '--regimes', 'random', '--seeds', '1', '--timing', 'off'],
                       torch=torch, ttnn=ttnn, sarg=fake_sarg(), t1=fake_t1(), tp_shapes=SimpleNamespace(vocab_shard=lambda: SHARD))
        self.assertGreater(ttnn.open_keywords['trace_region_size'], 0)


class RunTests(unittest.TestCase):
    def run_main(self, mode, extra=(), tp='4', grid=(11, 10), t1_mode='same', clock=None):
        ttnn = FakeTTNN(grid)
        ttnn.clock = clock
        ttnn.costs = dict(served=0.001, scan=0.00009, fold1=0.0003, fold2=0.00017)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 's1.json'
            lines = []
            with patch.dict(os.environ, {'QWEN_FAST_TP': tp}), patch('builtins.print', side_effect=lambda *a, **k: lines.append(' '.join(map(str, a)))):
                status = probe.main(['--out', str(out), '--rows', '16,64', '--regimes', 'random,ties', '--seeds', '1', *extra],
                                    torch=torch, ttnn=ttnn, sarg=fake_sarg(mode), t1=fake_t1(t1_mode), tp_shapes=SimpleNamespace(vocab_shard=lambda: SHARD))
            report = json.loads(out.read_text())
        return status, lines, report, ttnn

    def test_an_exact_run_passes_prints_the_verdict_then_the_json_and_closes_the_mesh(self):
        status, lines, report, ttnn = self.run_main('same', ['--timing', 'off'])
        self.assertEqual(status, 0)
        self.assertTrue(any(line.startswith('SHARD_ARGMAX verdict=PASS sections=8 differing=0') for line in lines), lines)
        self.assertEqual(json.loads(lines[-1])['kind'], probe.KIND)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(len(report['compare']), 8)
        self.assertTrue(ttnn.closed)
        self.assertEqual((report['grid'], report['workers']), ([11, 10], 110))
        self.assertEqual(set(report['kernels_sha256']), set(probe.KERNELS))
        self.assertNotIn('timing', report)

    def test_the_grid_comes_from_the_device(self):
        status, lines, report, ttnn = self.run_main('same', ['--timing', 'off'], grid=(13, 10))
        self.assertEqual((report['grid'], report['workers']), ([13, 10], 130))
        self.assertEqual(status, 0)

    def test_the_timing_runs_only_after_a_pass_unless_asked_for_always(self):
        ticks = {'time': 0.0}
        with patch.object(probe, 'ROUNDS', 4), patch.object(probe, 'LAUNCHES', 2), patch.object(probe.time, 'perf_counter', lambda: ticks['time']):
            status, lines, report, ttnn = self.run_main('same', clock=ticks)
            self.assertEqual(status, 0)
            self.assertEqual([entry['rows'] for entry in report['timing']], [16, 64])
            self.assertEqual(report['timing_verdict'], {'s1': 'S1-WIN', 'fold2': 'FOLD2-WIN'})
            self.assertTrue(any(line.startswith('SHARD_ARGMAX timing_verdict s1=S1-WIN fold2=FOLD2-WIN') for line in lines))
            status, lines, report, ttnn = self.run_main('flip_id', ['--timing', 'on'])
            self.assertEqual(status, 1)
            self.assertNotIn('timing', report)
            status, lines, report, ttnn = self.run_main('flip_id', ['--timing', 'always'])
            self.assertEqual(status, 1)
            self.assertIn('timing', report)

    def test_a_section_that_raises_is_not_run(self):
        sarg = fake_sarg()
        sarg.launch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('compile error text'))
        ttnn = FakeTTNN()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 's1.json'
            with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}), patch('builtins.print'):
                status = probe.main(['--out', str(out), '--rows', '16', '--regimes', 'random', '--seeds', '1', '--timing', 'off'],
                                    torch=torch, ttnn=ttnn, sarg=sarg, t1=fake_t1(), tp_shapes=SimpleNamespace(vocab_shard=lambda: SHARD))
            report = json.loads(out.read_text())
        self.assertEqual(status, 4)
        self.assertEqual(report['verdict'], 'NOT-RUN')
        self.assertIn('compile error text', json.dumps(report['compare']))

    def test_the_wrong_width_is_not_run(self):
        status, lines, report, ttnn = self.run_main('same', ['--timing', 'off'], tp='2')
        self.assertEqual(status, 4)
        self.assertEqual(report['verdict'], 'NOT-RUN')
        self.assertIn('QWEN_FAST_TP=4', report['error'])

    def test_bad_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), patch('sys.stderr'):
            for bad in (['--rows', '7'], ['--regimes', 'nan'], ['--folds', 'quad']):
                self.assertEqual(probe.main(['--out', str(Path(directory) / 'x.json'), *bad], torch=torch, ttnn=FakeTTNN(), sarg=fake_sarg(),
                                            t1=fake_t1(), tp_shapes=SimpleNamespace(vocab_shard=lambda: SHARD)), 2)


class HarnessTests(unittest.TestCase):
    TEXT = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')

    def test_the_harness_mounts_the_module_its_three_kernels_and_what_they_import_and_runs_one_card_with_the_serving_hook_off(self):
        for fragment in ('tp4_shard_argmax.py tp4_shard_argmax_scan.cpp tp4_shard_argmax_fold.cpp tp4_shard_argmax_fold2.cpp tp4_sampdraft.py tp_shapes.py '
                         'tp4_vglue.py verify_trace_t1.py',
                         '-e QWEN_FAST_TP=4', '-e QWEN_FAST_TP4_SHARD_VALUES=1',
                         '-e QWEN_C2_SERVING=0 --entrypoint env "$IMAGE" -u TT_MESH_GRAPH_DESC_PATH python3 -B /bench/shard_argmax_card_m.py',
                         '--network none', '--cap-drop ALL', 'qual_card_select', 'qual_refuse_holders', 'qual_card_recheck'):
            self.assertIn(fragment, self.TEXT)
        self.assertNotIn('\r', self.TEXT)

    def test_every_file_the_harness_mounts_exists(self):
        for name in ('tp4_shard_argmax.py', 'tp4_shard_argmax_scan.cpp', 'tp4_shard_argmax_fold.cpp', 'tp4_shard_argmax_fold2.cpp', 'tp4_sampdraft.py',
                     'tp_shapes.py', 'tp4_vglue.py', 'verify_trace_t1.py', 'qual_card.sh'):
            self.assertTrue((CI / name).is_file(), name)

    def test_it_embeds_the_canonical_card_selection_once_and_names_no_host_registry_or_board_outside_it(self):
        import c2_serving_job
        library = (CI / 'qual_card.sh').read_text(encoding='utf-8')
        self.assertTrue(c2_serving_job.embeds_qual_card(self.TEXT, library))      # what the cardm job validation demands of a harness
        body = self.TEXT[:self.TEXT.index('# >>> qual_card.sh')] + self.TEXT[self.TEXT.index('# <<< qual_card.sh'):]
        for fragment in ('blackhole-', 'zot', '192.168', 'thatch@'):
            self.assertNotIn(fragment, body)

    def test_the_container_has_no_network_a_timeout_and_removes_itself(self):
        self.assertIn('timeout -k 30 "$timeout_s" docker run', self.TEXT)
        self.assertIn('docker rm -f', self.TEXT)

    def test_the_kernels_the_probe_hashes_are_the_ones_the_module_launches(self):
        self.assertEqual(set(probe.KERNELS), {real_sarg.SCAN_KERNEL, real_sarg.FOLD_KERNEL, real_sarg.FOLD2_KERNEL})


if __name__ == '__main__':
    unittest.main()
