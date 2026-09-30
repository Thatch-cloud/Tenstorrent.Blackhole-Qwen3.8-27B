import copy
import importlib.util
import io
import json
import os
import re
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

import ordered_cache_hw_plan as hw
from ordered_cache import HASHES, validate_shapes

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / '.github' / 'workflows' / 'qwen-ordered-cache-hw-probe.yml'
CPU_WORKFLOW = REPO / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
PROBE = Path(__file__).resolve().with_name('ordered-cache-hw-probe.py')

PLAN = hw.build_plan()


def case(name):
    return next(entry for entry in PLAN['cases'] if entry['name'] == name)


def passing_report(plan=PLAN):
    checks = []
    for entry in plan['cases']:
        for chip in hw.CHIPS:
            for name in ('pages_uploaded', 'input_unchanged', 'pages_unchanged'):
                checks.append(dict(case=entry['name'], step=None, chip=chip, name=name, exact=True))
                checks.append(dict(case=entry['name'], step=None, chip=chip, name='zero_baseline', exact=True,
                                   **cache_evidence(plan, 0)))
            counts = hw.predicted_counts(entry)
            for step in entry['steps']:
                checks.append(dict(case=entry['name'], step=step['step'], chip=chip, name='complete_cache',
                                   exact=True, **cache_evidence(plan, counts[step['step']])))
    generated = {role: hashlib_hex(role) for role in hw.KERNEL_ROLES}
    return dict(probe='ordered-cache-hw-probe', passed=True, closed_cleanly=True, backend='hardware',
        weight_free=True, prefill=False, plan=copy.deepcopy(plan), plan_sha256=hw.plan_digest(plan),
        native_hashes=dict(HASHES), generated_hashes=generated, ordered_cache_sha256='a' * 64,
        ordered_cache_expected_sha256='a' * 64, wide_page_widths=[2052],
        tt_metal=dict(revision='9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'), checks=checks)


def cache_evidence(plan, predicted):
    return dict(shape=list(plan['cache_shape']), predicted_blocks=predicted, mismatched_blocks=0,
                unpredicted_nonzero_blocks=0, predicted_mismatch_blocks=0)


def hashlib_hex(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


class PlanGeometryTests(unittest.TestCase):
    def test_required_geometry(self):
        self.assertGreaterEqual(PLAN['blocks'], 2060)
        self.assertEqual(PLAN['cache_shape'], [PLAN['blocks'], 2, 64, 256])
        self.assertEqual(PLAN['rows'], 16)
        self.assertEqual({(entry['width'], entry['mode']) for entry in PLAN['cases']},
                         {(1024, 'eager'), (2052, 'eager'), (2052, 'trace')})

    def test_every_case_is_a_shape_the_writer_admits(self):
        """The probe must dispatch through ordered_cache.validate_shapes, not around it."""
        for entry in PLAN['cases']:
            with self.subTest(case=entry['name']):
                rows = PLAN['rows']
                self.assertEqual(validate_shapes(tuple(PLAN['cache_shape']), (1, rows, 32, 256), (rows,),
                                                 (rows, entry['width'])), rows)

    def test_wide_row_stride_is_not_64_byte_aligned(self):
        """8,208-byte page-table rows (8,256 aligned) are the stride the review flagged."""
        self.assertEqual(hw.WIDE_WIDTH * 4, 8208)
        self.assertNotEqual((hw.WIDE_WIDTH * 4) % 64, 0)

    def test_page_rows_are_distinct_permutations_inside_the_cache(self):
        for entry in PLAN['cases']:
            for table in entry['tables']:
                self.assertEqual(len({tuple(row) for row in table}), PLAN['rows'])
                for row in table:
                    self.assertEqual(len(row), entry['width'])
                    self.assertEqual(len(set(row)), entry['width'])
                    self.assertTrue(0 <= min(row) and max(row) < PLAN['blocks'])

    def test_control_table_uses_block_ids_beyond_its_width(self):
        """A 1,024-wide table drawn from 2,064 blocks cannot pass by identity mapping."""
        table = case('control-1024-eager')['tables'][0]
        self.assertTrue(any(max(row) >= 1024 for row in table))

    def test_trace_case_rewrites_the_page_table_between_replays(self):
        entry = case('wide-2052-trace')
        self.assertEqual(len(entry['tables']), 2)
        self.assertNotEqual(entry['tables'][0], entry['tables'][1])
        self.assertEqual([step['table'] for step in entry['steps']][:4], [0, 1, 0, 1])

    def test_tail_entries_hit_by_every_row_in_eager_and_trace(self):
        for name in ('wide-2052-eager', 'wide-2052-trace'):
            entry = case(name)
            for required in (0, 1023, 1024, 2048, 2049, 2050, 2051):
                for row in range(PLAN['rows']):
                    with self.subTest(case=name, entry=required, row=row):
                        self.assertTrue(any(step['entries'][row] == required for step in entry['steps']))
            positions = {position for step in entry['steps'] for position in step['positions']}
            self.assertTrue({131072, 131136, 131200, 131327} <= positions)
            self.assertLess(max(positions), 2052 * 64)

    def test_control_hits_its_first_and_last_entry(self):
        entry = case('control-1024-eager')
        positions = {position for step in entry['steps'] for position in step['positions']}
        self.assertTrue({0, 65535} <= positions)
        self.assertLess(max(positions), 1024 * 64)

    def test_no_two_rows_share_a_target_in_one_step(self):
        for entry in PLAN['cases']:
            for step in entry['steps']:
                self.assertEqual(len(set(zip(step['blocks'], step['offsets']))), PLAN['rows'])

    def test_plan_is_deterministic_and_json_stable(self):
        again = hw.build_plan()
        self.assertEqual(again, PLAN)
        self.assertEqual(json.loads(json.dumps(PLAN)), PLAN)
        self.assertEqual(hw.plan_digest(again), hw.plan_digest(PLAN))

    def test_page_table_prng_is_pinned(self):
        """Pure-integer shuffle: the same table on every Python, so --check can rebuild it."""
        self.assertEqual(hw.page_table(7, 2, 5, 8), hw.page_table(7, 2, 5, 8))
        self.assertNotEqual(hw.page_table(7, 1, 8, 8), hw.page_table(8, 1, 8, 8))
        self.assertEqual(sorted(hw.page_table(3, 1, 8, 8)[0]), list(range(8)))
        with self.assertRaises(ValueError):
            hw.page_table(1, 1, 9, 8)


class PlanValidationTests(unittest.TestCase):
    def mutate(self, change):
        plan = copy.deepcopy(PLAN)
        change(plan)
        return plan

    def assertRefused(self, change, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            hw.validate_plan(self.mutate(change))

    def test_small_cache_refused(self):
        self.assertRefused(lambda plan: plan.update(blocks=2059), '2060')

    def test_too_few_rows_refused(self):
        self.assertRefused(lambda plan: plan.update(rows=4), 'rows')

    def test_duplicate_page_row_refused(self):
        def change(plan):
            entry = plan['cases'][1]
            entry['tables'][0][1] = list(entry['tables'][0][0])
            entry['table_sha256'][0] = hw.table_digest(entry['tables'][0])
        self.assertRefused(change, 'own page row|disagrees')

    def test_block_outside_cache_refused(self):
        def change(plan):
            entry = plan['cases'][1]
            entry['tables'][0][0][5] = plan['blocks']
            entry['table_sha256'][0] = hw.table_digest(entry['tables'][0])
        self.assertRefused(change, 'distinct block ids within the cache')

    def test_tampered_table_refused(self):
        def change(plan):
            row = plan['cases'][1]['tables'][0][0]
            row[0], row[1] = row[1], row[0]
        self.assertRefused(change, 'digest|disagrees')

    def test_position_outside_window_refused(self):
        def change(plan):
            step = plan['cases'][1]['steps'][0]
            step['positions'][0] = 2052 * 64
        self.assertRefused(change, 'inside the page-table window')

    def test_shared_target_refused(self):
        def change(plan):
            step = plan['cases'][1]['steps'][0]
            table = plan['cases'][1]['tables'][step['table']]
            entry = step['entries'][0]
            other = table[1].index(table[0][entry])
            step['positions'][1] = other * 64 + step['offsets'][0]
            step['entries'][1] = other
            step['offsets'][1] = step['offsets'][0]
            step['blocks'][1] = step['blocks'][0]
        self.assertRefused(change, 'one \\(block, offset\\)')

    def test_row_that_never_reads_a_tail_entry_refused(self):
        def change(plan):
            entry = plan['cases'][1]
            for step in entry['steps']:
                if step['entries'][5] == 2051:
                    step['positions'][5] = 2047 * 64 + step['offsets'][5]
                    step['entries'][5] = 2047
                    step['blocks'][5] = entry['tables'][step['table']][5][2047]
        self.assertRefused(change, 'Every row must hit page-table entry 2051')

    def test_missing_trace_case_refused(self):
        self.assertRefused(lambda plan: plan['cases'].pop(2), 'trace')

    def test_missing_control_refused(self):
        self.assertRefused(lambda plan: plan['cases'].pop(0), 'control')


class PayloadTests(unittest.TestCase):
    def test_payload_is_nonzero_bf8_exact_and_seeded(self):
        first = hw.payload(11)
        self.assertEqual(tuple(first.shape), (32, 256))
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertFalse(bool((first == 0).any()))
        self.assertTrue(hw.bf8_exact(first))
        self.assertTrue(torch.equal(first, hw.payload(11)))
        self.assertFalse(torch.equal(first, hw.payload(12)))
        self.assertFalse(torch.equal(first[0], first[1]))

    def test_every_planned_payload_is_bf8_exact(self):
        seeds = [seed for entry in PLAN['cases'] for step in entry['steps'] for seed in step['payload_seeds']]
        values = torch.stack([hw.payload(seed) for seed in seeds])
        self.assertTrue(hw.bf8_exact(values))
        heads = values[:, :2].reshape(len(seeds), -1).float()
        self.assertEqual(len({tuple(row.tolist()) for row in heads}), len(seeds))

    def test_bf8_exact_rejects_mixed_exponent_groups(self):
        group = torch.full((16,), 1.0)
        self.assertTrue(hw.bf8_exact(group))
        group[3] = 2.0 ** -8
        self.assertFalse(hw.bf8_exact(group))
        self.assertTrue(hw.bf8_exact(torch.zeros(32)))
        self.assertFalse(hw.bf8_exact(torch.full((16,), float('nan'))))


class PredictionTests(unittest.TestCase):
    """Small geometry: 6 blocks x 2 heads x 64 rows x 32 columns."""

    def setUp(self):
        self.table = [[4, 1], [2, 5]]
        self.payloads = torch.stack([hw.payload(1, 32, 32), hw.payload(2, 32, 32)])

    def expected(self, positions=(65, 3)):
        cache = hw.ExpectedCache(6, head_dim=32)
        cache.apply(self.table, list(positions), self.payloads)
        return cache

    def native(self, table, positions):
        """A model of the kernel writing through whatever page table it read."""
        cache = torch.zeros(6, 2, 64, 32, dtype=torch.bfloat16)
        for row, position in enumerate(positions):
            cache[table[row][position // 64], :, position % 64] = self.payloads[row, :2]
        return cache

    def test_prediction_writes_heads_zero_and_one_at_pos_mod_64(self):
        cache = self.expected()
        self.assertEqual(cache.predicted, {1, 2})
        self.assertTrue(torch.equal(cache.values[1, :, 1], self.payloads[0, :2]))
        self.assertTrue(torch.equal(cache.values[2, :, 3], self.payloads[1, :2]))
        cache.values[1, :, 1] = 0
        cache.values[2, :, 3] = 0
        self.assertEqual(int(torch.count_nonzero(cache.values)), 0)

    def test_matching_cache_is_exact_in_any_dtype(self):
        cache = self.expected()
        result = hw.compare_cache(self.native(self.table, (65, 3)).to(torch.float32), cache)
        self.assertTrue(result['exact'])
        self.assertEqual(result['predicted_blocks'], 2)

    def test_shared_page_table_misread_is_caught(self):
        """The blind spot: a native writer and a native oracle that both read entry 0 where
        entry 1 was meant agree with each other. The host prediction does not."""
        misread = [[row[0], row[0]] for row in self.table]
        writer, oracle = self.native(misread, (65, 3)), self.native(misread, (65, 3))
        self.assertTrue(torch.equal(writer, oracle))
        result = hw.compare_cache(writer, self.expected())
        self.assertFalse(result['exact'])
        self.assertEqual(result['unpredicted_nonzero_blocks'], 1)
        self.assertEqual(result['predicted_mismatch_blocks'], 1)
        self.assertEqual({sample['block'] for sample in result['samples']}, {1, 4})

    def test_wrong_offset_and_stray_write_and_nan_are_caught(self):
        cache = self.expected()
        wrong = self.native(self.table, (66, 3))
        self.assertEqual(hw.compare_cache(wrong, cache)['predicted_mismatch_blocks'], 1)
        stray = self.native(self.table, (65, 3))
        stray[0, 1, 63, 31] = 1
        result = hw.compare_cache(stray, cache)
        self.assertEqual((result['unpredicted_nonzero_blocks'], result['predicted_mismatch_blocks']), (1, 0))
        self.assertFalse(result['exact'])
        broken = self.native(self.table, (65, 3))
        broken[1, 0, 1, 0] = float('nan')
        self.assertFalse(hw.compare_cache(broken, cache)['exact'])

    def test_padding_head_leak_is_caught(self):
        leaked = self.native(self.table, (65, 3))
        leaked[1, 1, 1] = self.payloads[0, 2]
        self.assertFalse(hw.compare_cache(leaked, self.expected())['exact'])

    def test_shape_mismatch_is_not_exact(self):
        result = hw.compare_cache(torch.zeros(5, 2, 64, 32), self.expected())
        self.assertFalse(result['exact'])
        self.assertEqual(result['samples'][0]['reason'], 'shape')

    def test_chunking_covers_every_block(self):
        cache = self.expected()
        actual = self.native(self.table, (65, 3))
        actual[5, 0, 0, 0] = 2
        for chunk in (1, 4):
            result = hw.compare_cache(actual, cache, chunk=chunk)
            self.assertEqual(result['unpredicted_nonzero_blocks'], 1)
            self.assertFalse(result['exact'])

    def test_extra_write_to_an_unpredicted_block_alone_fails(self):
        """The correct writes plus one duplicate write elsewhere (a stale trace table, or a
        second path through a misread entry): every predicted block is exact, yet the
        verdict must be a failure because an unpredicted block is no longer zero."""
        cache = self.expected()
        actual = self.native(self.table, (65, 3))
        actual[0, :, 1] = self.payloads[0, :2]
        result = hw.compare_cache(actual, cache)
        self.assertEqual(result['predicted_mismatch_blocks'], 0)
        self.assertEqual(result['unpredicted_nonzero_blocks'], 1)
        self.assertFalse(result['exact'])


class ReportTests(unittest.TestCase):
    def test_complete_report_passes(self):
        self.assertEqual(hw.check_report(passing_report(), PLAN), [])
        self.assertTrue(all(entry['passed'] for entry in hw.summarise_cases(passing_report(), PLAN)))

    def test_missing_or_inexact_check_fails_its_case(self):
        report = passing_report()
        report['checks'] = [check for check in report['checks']
                            if not (check['case'] == 'wide-2052-trace' and check['step'] == 10 and check['chip'] == 1)]
        failures = hw.check_report(report, PLAN)
        self.assertEqual(len(failures), 1)
        self.assertIn('wide-2052-trace', failures[0])
        report = passing_report()
        next(check for check in report['checks'] if check['name'] == 'zero_baseline')['exact'] = False
        self.assertTrue(hw.check_report(report, PLAN))
        summary = {entry['name']: entry['passed'] for entry in hw.summarise_cases(report, PLAN)}
        self.assertEqual(summary, {'control-1024-eager': False, 'wide-2052-eager': True, 'wide-2052-trace': True})

    def test_self_reported_exact_without_comparison_evidence_fails(self):
        """check_report must not trust `exact` alone: a cache check needs the plan's cache
        shape, the predicted-block count the plan implies at that step, and zero counters."""
        def complete(report, case_name='wide-2052-eager', step=10, chip=1):
            return next(check for check in report['checks'] if check['name'] == 'complete_cache'
                        and check['case'] == case_name and check['step'] == step and check['chip'] == chip)

        mutations = [
            lambda report: complete(report).update(predicted_blocks=complete(report)['predicted_blocks'] - 1),
            lambda report: complete(report).pop('predicted_blocks'),
            lambda report: complete(report).update(shape=[16, 2, 64, 256]),
            lambda report: complete(report).pop('shape'),
            lambda report: complete(report).update(unpredicted_nonzero_blocks=1),
            lambda report: complete(report).update(predicted_mismatch_blocks=1),
            lambda report: complete(report).update(mismatched_blocks=1),
            lambda report: complete(report, 'wide-2052-trace', 0, 0).pop('mismatched_blocks'),
            lambda report: next(check for check in report['checks'] if check['name'] == 'zero_baseline')
                .update(predicted_blocks=3),
            lambda report: report['checks'].append(dict(complete(report), exact=False)),
        ]
        for index, mutate in enumerate(mutations):
            report = passing_report()
            mutate(report)
            with self.subTest(mutation=index):
                failures = hw.check_report(report, PLAN)
                self.assertTrue(failures)

    def test_predicted_counts_grow_with_the_steps(self):
        counts = hw.predicted_counts(case('wide-2052-eager'))
        self.assertEqual(counts[0], PLAN['rows'])
        self.assertEqual(sorted(counts), list(range(len(case('wide-2052-eager')['steps']))))
        self.assertTrue(all(counts[step] <= counts[step + 1] for step in range(len(counts) - 1)))

    def test_provenance_and_scope_are_required(self):
        mutations = [
            lambda report: report.update(error='RuntimeError: boom'),
            lambda report: report.update(closed_cleanly=False),
            lambda report: report.update(backend='simulator'),
            lambda report: report.update(prefill=True),
            lambda report: report['native_hashes'].pop('reader'),
            lambda report: report.update(generated_hashes={}),
            lambda report: report.update(ordered_cache_expected_sha256='b' * 64),
            lambda report: report.update(wide_page_widths=[]),
            lambda report: report.pop('tt_metal'),
            lambda report: report['plan']['cases'][1]['steps'][0]['positions'].reverse(),
            lambda report: report.update(plan_sha256='0' * 64),
        ]
        for index, mutate in enumerate(mutations):
            report = passing_report()
            mutate(report)
            with self.subTest(mutation=index):
                self.assertTrue(hw.check_report(report, PLAN))

    def test_cli_rechecks_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'report.json')
            # (report, exit status, independent re-check verdict): a probe that reported
            # failure never exits 0 even when the re-check finds nothing missing.
            for report, status, verdict in ((passing_report(), 0, True),
                                            (dict(passing_report(), passed=False), 1, True),
                                            (dict(passing_report(), checks=[]), 1, False)):
                with open(path, 'w', encoding='utf-8', newline='\n') as handle:
                    json.dump(report, handle)
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(hw.main(['--check', path]), status)
                self.assertEqual(json.loads(output.getvalue())['passed'], verdict)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(hw.main(['--check', os.path.join(directory, 'absent.json')]), 1)


def load_probe():
    spec = importlib.util.spec_from_file_location('ordered_cache_hw_probe', PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHost:
    def __init__(self, value):
        self.value = value


class FakeDevice:
    """A replicated tensor: one independent torch tensor per chip."""

    def __init__(self, value, chips):
        self.parts = [value.clone() for unused in range(chips)]
        self.deallocated = False


class FakeMesh:
    def __init__(self):
        self.capturing = None
        self.traces = {}
        self.closed = False

    def enable_program_cache(self):
        pass


class FakeTtnn:
    """Just enough of ttnn for run_probe. BF8 tensors are held as BF16 (the payloads are
    BF8-exact). Ops issued during trace capture are recorded, not run, and execute_trace
    replays them against whatever the device buffers hold at replay time."""

    int32, bfloat16, bfloat8_b = 'int32', 'bfloat16', 'bfloat8_b'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'row_major', 'tile', 'dram'
    FabricConfig = types.SimpleNamespace(FABRIC_1D='fabric_1d')
    __file__ = 'fake-ttnn'

    def __init__(self):
        self.mesh = None
        self.next_trace = 0

    @staticmethod
    def _cast(value, dtype):
        return value.to(torch.int32 if dtype == 'int32' else torch.bfloat16)

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def MeshShape(self, rows, columns):
        return (rows, columns)

    def set_fabric_config(self, config):
        pass

    def open_mesh_device(self, shape, **unused):
        self.mesh = FakeMesh()
        return self.mesh

    def close_mesh_device(self, mesh):
        mesh.closed = True

    def synchronize_device(self, mesh):
        pass

    def from_torch(self, value, dtype, layout, device=None, memory_config=None, mesh_mapper=None):
        value = self._cast(value, dtype)
        return FakeHost(value) if device is None else FakeDevice(value, len(hw.CHIPS))

    def copy_host_to_device_tensor(self, host, destination):
        for part in destination.parts:
            if tuple(part.shape) != tuple(host.value.shape) or part.dtype != host.value.dtype:
                raise RuntimeError('copy_host_to_device_tensor shape/dtype mismatch')
            part.copy_(host.value)

    def get_device_tensors(self, value):
        return list(value.parts)

    def to_torch(self, part):
        return part.clone()

    def deallocate(self, value):
        value.deallocated = True

    def begin_trace_capture(self, mesh, cq_id):
        mesh.capturing = []
        self.next_trace += 1
        return self.next_trace

    def end_trace_capture(self, mesh, trace, cq_id):
        mesh.traces[trace], mesh.capturing = mesh.capturing, None

    def execute_trace(self, mesh, trace, cq_id, blocking):
        for operation in mesh.traces[trace]:
            operation()

    def release_trace(self, mesh, trace):
        del mesh.traces[trace]


class FakeWriter:
    """Stands in for the baked ordered_cache: writes each row through the page table the
    DEVICE holds (per chip), with injectable faults."""

    HASHES = dict(HASHES)
    WIDE_PAGE_WIDTHS = frozenset({hw.WIDE_WIDTH})

    def __init__(self, fault=None, avoid=()):
        self.fault = fault
        self.stray = min(set(range(hw.BLOCKS)) - set(avoid))
        self.calls = 0

    def page(self, chip, row_table, entry):
        if self.fault == 'tail-misread' and entry >= 2048:
            return int(row_table[entry - 2048])
        return int(row_table[entry])

    def write(self, cache, packed, positions, pages_by_chip):
        self.calls += 1
        for chip, part in enumerate(cache.parts):
            table, where, values = pages_by_chip[chip], positions.parts[chip], packed.parts[chip]
            for row in range(table.shape[0]):
                position = int(where[row])
                block = self.page(chip, table[row], position // hw.BLOCK_SIZE)
                part[block, :, position % hw.BLOCK_SIZE, :] = values[0, row, :hw.HEADS, :]
                if self.fault == 'chip1-stray' and chip == 1 and row == 0:
                    part[self.stray, :, position % hw.BLOCK_SIZE, :] = values[0, row, :hw.HEADS, :]

    def update(self, mesh, cache, packed, positions, pages, kernels):
        if mesh.capturing is None:
            self.write(cache, packed, positions, pages.parts)
        elif self.fault == 'stale-trace-table':
            frozen = [part.clone() for part in pages.parts]
            mesh.capturing.append(lambda: self.write(cache, packed, positions, frozen))
        else:
            mesh.capturing.append(lambda: self.write(cache, packed, positions, pages.parts))


class ProbeDriverTests(unittest.TestCase):
    """Runs ordered-cache-hw-probe.run_probe end to end on a fake ttnn at full geometry.
    A driver that read back the wrong tensor, read one chip twice, skipped the op, or
    applied the prediction from the wrong table would fail one of these."""

    @classmethod
    def setUpClass(cls):
        cls.probe = load_probe()
        cls.predicted = {entry['name']: {block for step in entry['steps'] for block in step['blocks']}
                         for entry in PLAN['cases']}
        cls.runs = {}
        for fault in (None, 'tail-misread', 'chip1-stray', 'stale-trace-table'):
            writer = FakeWriter(fault, avoid=set().union(*cls.predicted.values()))
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                output = Path(directory) / 'report.json'
                report, error = cls.probe.run_probe(
                    FakeTtnn(), torch, writer, hw, {role: role for role in hw.KERNEL_ROLES}, output,
                    dict(ordered_cache_sha256='a' * 64, ordered_cache_expected_sha256='a' * 64,
                         tt_metal=dict(revision='fake')))
                written = json.loads(output.read_text(encoding='utf-8'))
            cls.runs[fault] = (report, error, written, writer)

    def failing(self, report):
        return {(check['case'], check['chip']) for check in report['checks']
                if check['name'] == 'complete_cache' and not check['exact']}

    def verdicts(self, report):
        return {entry['name']: entry['passed'] for entry in report['cases']}

    def test_correct_writer_passes_every_case_on_both_chips(self):
        report, error, written, writer = self.runs[None]
        self.assertIsNone(error)
        self.assertEqual(report['failures'], [])
        self.assertTrue(report['passed'])
        self.assertEqual(written['passed'], True)
        self.assertEqual(hw.check_report(written, PLAN), [])
        self.assertEqual(len(report['checks']), len(hw.required_checks(PLAN)))
        eager_steps = sum(len(entry['steps']) for entry in PLAN['cases'] if entry['mode'] == 'eager')
        # Writes executed: one per eager step, the trace warm-up, and one per replay (the
        # capture call is recorded, not run). A skipped op or a skipped replay changes this.
        self.assertEqual(writer.calls, eager_steps + 1 + len(case('wide-2052-trace')['steps']))
        for check in report['checks']:
            if check['name'] == 'complete_cache':
                self.assertEqual(check['shape'], PLAN['cache_shape'])
                self.assertGreater(check['predicted_blocks'], 0)

    def test_tail_misread_fails_both_wide_cases_on_both_chips(self):
        report, error, unused, unused_writer = self.runs['tail-misread']
        self.assertIsNone(error)
        self.assertFalse(report['passed'])
        self.assertEqual(self.verdicts(report),
                         {'control-1024-eager': True, 'wide-2052-eager': False, 'wide-2052-trace': False})
        self.assertEqual(self.failing(report), {(name, chip) for name in ('wide-2052-eager', 'wide-2052-trace')
                                                for chip in hw.CHIPS})

    def test_fault_on_one_chip_fails_only_that_chip(self):
        report, error, unused, unused_writer = self.runs['chip1-stray']
        self.assertIsNone(error)
        self.assertFalse(report['passed'])
        self.assertEqual(self.failing(report), {(entry['name'], 1) for entry in PLAN['cases']})
        stray = [check for check in report['checks'] if check['name'] == 'complete_cache' and not check['exact']]
        self.assertTrue(all(check['predicted_mismatch_blocks'] == 0 and check['unpredicted_nonzero_blocks'] == 1
                            for check in stray))

    def test_replay_of_a_stale_page_table_fails_only_the_trace_case(self):
        report, error, unused, unused_writer = self.runs['stale-trace-table']
        self.assertIsNone(error)
        self.assertEqual(self.verdicts(report),
                         {'control-1024-eager': True, 'wide-2052-eager': True, 'wide-2052-trace': False})
        failed_steps = {check['step'] for check in report['checks']
                        if check['case'] == 'wide-2052-trace' and check['name'] == 'complete_cache'
                        and not check['exact']}
        # Step 0 replays the table it was captured with; step 1 is the first replay after the
        # in-place rewrite, and its wrong writes persist into every later full-cache check.
        self.assertEqual(failed_steps, set(range(1, len(case('wide-2052-trace')['steps']))))


class WorkflowTests(unittest.TestCase):
    """The probe runs inside the serving image: its env must cross with docker -e, and the
    baked /experiment-scripts/ci tree must never be hidden by a mount."""

    def setUp(self):
        self.workflow = WORKFLOW.read_text(encoding='utf-8')

    def test_required_env_is_passed_into_the_container(self):
        for name in hw.PROBE_ENV['required']:
            self.assertRegex(self.workflow, r'-e %s=' % name)
        for name in hw.PROBE_ENV['forbidden']:
            self.assertNotIn('-e %s' % name, self.workflow)

    def test_probe_reads_no_env_beyond_the_declared_set(self):
        source = PROBE.read_text(encoding='utf-8')
        reads = set(re.findall(r"os\.environ(?:\.get\()?\[?\(?'([A-Z0-9_]+)'", source))
        self.assertTrue(reads <= set(hw.PROBE_ENV['required']) | set(hw.PROBE_ENV['forbidden']), reads)

    def test_only_single_new_files_are_mounted_at_bench(self):
        self.assertNotRegex(self.workflow, r'dst=/experiment-scripts')
        mounts = re.findall(r'dst=(/bench/[^,"\s]+)', self.workflow)
        self.assertEqual(sorted(set(mounts)), ['/bench/ordered-cache-hw-probe.py', '/bench/ordered_cache_hw_plan.py'])
        self.assertNotIn('/bench/ordered_cache.py', self.workflow)

    def test_hardware_job_conventions(self):
        self.assertIn('thatch-qwen-p150a-pair', self.workflow)
        self.assertIn('group: qwen-two-p150a-exclusive', self.workflow)
        self.assertIn("tags: ['experiment/ordered-cache-hw-probe-v*']", self.workflow)
        self.assertEqual(self.workflow.count('ordered-cache-hw-probe.py --output'), 1)
        self.assertIn('tt-smi', self.workflow)
        for line in self.workflow.splitlines():
            if line.rstrip().endswith('\\'):
                self.assertNotIn('#', line, line)

    def test_cpu_suite_runs_this_test(self):
        self.assertIn('python -B -m unittest test_ordered_cache_hw_plan', CPU_WORKFLOW.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
