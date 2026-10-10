"""CPU tests of the CCL sweep probe: its grids, its paired statistics and promotion rule, its exactness rule against a fake four-chip runtime whose
collectives sum in a fixed order (and one option that corrupts a bit), its failure containment, its watchdog and its wrapper.

Run: python -B -m unittest discover -s optimisation/ttnn-op/ccl_sweep -p 'test_*.py'     (from the repository root; scripts/ci is put on the path)
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent / 'scripts' / 'ci'))

import ccl_options_tp  # noqa: E402
import ccl_sweep as sweep  # noqa: E402
import tp4_ccl_sweep_probe as wrapper  # noqa: E402

CHIPS = 4


class FakeClock(object):
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class Memory(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class FakeTensor(object):
    def __init__(self, chips, memory):
        self.chips, self.memory = chips, memory
        self.freed = False


def reduce_scatter_sum(chips):
    total = sum(chip.to(torch.float32) for chip in chips).to(torch.bfloat16)
    width = total.shape[-1] // CHIPS
    return [total[..., k * width:(k + 1) * width].contiguous() for k in range(CHIPS)]


class FakeMesh(object):
    shape = (1, CHIPS)

    def get_num_devices(self):
        return CHIPS

    def get_device_ids(self):
        return [3, 1, 0, 2]


class FakeTTNN(object):
    """A four-chip runtime: collectives are exact sums / copies, an op's cost is a function of its keywords (RS_COST / AG_COST), a trace replay advances
    the clock by the sum of its ops' costs plus a fixed launch cost. `corrupt` is a predicate on the keywords: it flips one bit of one chip."""
    bfloat16, TILE_LAYOUT = 'bf16', 'tile'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = Memory('DRAM'), Memory('L1')
    LAUNCH_NS = 30000

    class Topology(object):
        Ring, Linear = 'Ring', 'Linear'

    class FabricConfig(object):
        FABRIC_1D, FABRIC_1D_RING = 'FABRIC_1D', 'FABRIC_1D_RING'

    class ShardStrategy(object):
        WIDTH = 'width'

    class ShardOrientation(object):
        ROW_MAJOR = 'row'

    def __init__(self, clock, rs_cost, ag_cost, corrupt=None, refuse=None, payload_report=None):
        self.clock, self.rs_cost, self.ag_cost = clock, rs_cost, ag_cost
        self.corrupt, self.refuse = corrupt, refuse
        self.payload_report = payload_report
        self.traces, self.capture, self.live = {}, None, 0
        self.memo = {}
        self.fabric = None
        self.events = []
        self.experimental = types.SimpleNamespace(reduce_scatter_minimal_async=self.reduce_scatter,
                                                  reduce_scatter_minimal_async_create_intermediate_buffer=self.interm,
                                                  all_gather_async=self.all_gather)
        self.cluster = types.SimpleNamespace(get_cluster_type=lambda: 'P150_X4')

    # configuration
    def CoreGrid(self, y, x):
        return (y, x)

    def create_sharded_memory_config(self, **kwargs):
        return Memory('WS%s' % (kwargs['shape'],))

    def ShardTensorToMesh(self, mesh, dim):
        return dim

    def FabricRouterConfig(self):
        return types.SimpleNamespace(max_packet_payload_size_bytes=None)

    def set_fabric_config(self, config, router_config=None):
        self.fabric = (config, router_config.max_packet_payload_size_bytes if router_config else None)

    def get_tt_fabric_max_payload_size_bytes(self):
        if self.payload_report is not None:
            return self.payload_report
        return (self.fabric[1] if self.fabric and self.fabric[1] else 4352)

    def open_mesh_device(self, shape, **kwargs):
        self.events.append(('open', tuple(shape.dims), kwargs))
        return FakeMesh()

    def MeshShape(self, *dims):
        return types.SimpleNamespace(dims=dims)

    def close_mesh_device(self, mesh):
        self.events.append(('close',))

    # tensors
    def from_torch(self, host, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        assert mesh_mapper == 0 and host.shape[0] == CHIPS
        return FakeTensor([host[k:k + 1].clone() for k in range(CHIPS)], memory_config)

    def get_device_tensors(self, tensor):
        return list(tensor.chips)

    def to_torch(self, part):
        return part

    def reshape(self, tensor, shape):
        return FakeTensor([chip.reshape(*shape) for chip in tensor.chips], tensor.memory)

    def deallocate(self, tensor):
        tensor.freed = True

    # ops
    def cached(self, kind, tensor, compute):
        """The result of an op is the same for every call on one input: compute it once. The input chip is kept in the entry so its id is not reused."""
        key = (kind, id(tensor.chips[0]))
        if key not in self.memo:
            self.memo[key] = (tensor.chips[0], compute())
        return self.memo[key][1]

    def _cost(self, function, kwargs):
        if self.refuse is not None and self.refuse(kwargs):
            raise RuntimeError('TT_FATAL: this configuration is not supported')
        cost = function(kwargs)
        if self.capture is not None:
            self.capture.append(cost)
        return cost

    def reduce_scatter(self, tensor, **kwargs):
        self._cost(self.rs_cost, kwargs)
        out = list(self.cached('rs', tensor, lambda: reduce_scatter_sum(tensor.chips)))
        if self.corrupt is not None and self.corrupt(kwargs):
            out[1] = out[1].clone()
            out[1].view(-1)[5] = out[1].view(-1)[5] + 1
        return FakeTensor(out, kwargs['memory_config'])

    def interm(self, tensor, dim, topology):
        return FakeTensor([torch.zeros(1)] * CHIPS, 'interm'), FakeTensor([torch.zeros(1)] * CHIPS, 'penult')

    def all_gather(self, tensor, **kwargs):
        self._cost(self.ag_cost, kwargs)
        out = [self.cached('ag', tensor, lambda: torch.cat(tensor.chips, dim=3))] * CHIPS
        if self.corrupt is not None and self.corrupt(kwargs):
            out = [chip.clone() for chip in out]
            out[2].view(-1)[7] = out[2].view(-1)[7] + 1
        return FakeTensor(out, kwargs['memory_config'])

    # traces
    def begin_trace_capture(self, mesh, cq_id=0):
        self.capture = []
        return len(self.traces) + 1

    def end_trace_capture(self, mesh, trace, cq_id=0):
        self.traces[trace] = self.capture
        self.capture = None

    def execute_trace(self, mesh, trace, cq_id=0, blocking=False):
        self.clock.now += int(sum(self.traces[trace]) * 1000) + self.LAUNCH_NS

    def synchronize_device(self, mesh):
        pass

    def release_trace(self, mesh, trace):
        self.traces.pop(trace)


def rs_cost(kwargs):
    cost = 18.0
    cost -= 2.0 if kwargs['num_workers_per_link'] == 1 else 0.0
    cost -= {1: 1.5, 2: 1.2, 5: 1.0}.get(kwargs['chunks_per_sync'], 0.0)
    cost += 4.0 if kwargs['num_links'] == 1 else 0.0
    cost -= 3.0 if kwargs['barrier_semaphore'] is None else 0.0
    cost -= 0.1 if kwargs['persistent_output_buffers'] is not None else 0.0
    cost -= 0.5 if kwargs['memory_config'] is not FakeTTNN.DRAM_MEMORY_CONFIG else 0.0
    return cost


def ag_cost(kwargs):
    cost = 18.4
    cost += 1.0 if kwargs['num_workers_per_link'] == 1 else 0.0
    cost -= 2.5 if kwargs.get('use_broadcast') else 0.0
    cost -= 3.0 if kwargs['barrier_semaphore'] is None else 0.0
    return cost


class Collective(object):
    def __init__(self):
        self.count = 0

    def get_and_cycle_rs_semaphore_handles(self):
        self.count += 1
        return ('rs', self.count % 2)

    def get_and_cycle_ag_semaphore_handles(self):
        self.count += 1
        return ('ag', self.count % 2)

    def get_and_cycle_barrier_semaphore_handle(self):
        self.count += 1
        return ('barrier', self.count % 2)


def model_all_reduce(ttnn):
    def call(tensor, mesh, collective, cluster_axis=0, dim=3, topology='Ring', memory_config=None):
        return FakeTensor(reduce_scatter_sum(tensor.chips), memory_config)
    return call


class Fixture(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.lines = []
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, 'report.json')

    def runtime(self, **kwargs):
        kwargs.setdefault('rs_cost', rs_cost)
        kwargs.setdefault('ag_cost', ag_cost)
        return FakeTTNN(self.clock, **kwargs)

    def options(self, **overrides):
        values = dict(seeds=2, rounds=5, replays=3, quick=True, only='all', skip_probe_only=False)
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def sweep(self, ttnn, **overrides):
        report = dict(scenarios={}, opened=True)
        saved = []

        def save(current):
            saved.append(json.loads(json.dumps(current, default=str)))
        mesh = FakeMesh()
        sweep.run(self.options(**overrides), ttnn, torch, mesh, Collective(), model_all_reduce(ttnn), report, save, log=self.lines.append, clock=self.clock)
        return report, saved

    def rows(self, report, scenario):
        return dict((row['name'], row) for row in report['scenarios'][scenario]['rows'])


class TheGrids(unittest.TestCase):
    def test_every_grid_config_is_a_named_set_of_one_op_the_stack_accepts(self):
        for op in ('rs', 'ag'):
            for quick in (False, True):
                grid = sweep.sweep_grid(op, quick)
                self.assertEqual(grid[0].name, 'served')
                self.assertEqual(grid[0].overrides, {})
                for config in grid[1:]:
                    selection = ccl_options_tp.parse_set(config.name)
                    self.assertEqual(selection.name, config.name)
                    self.assertEqual(dict(selection.rs if op == 'rs' else selection.ag), config.overrides)
                    self.assertTrue(config.overrides)
                    self.assertEqual(len(config.overrides), 1)

    def test_the_full_grid_covers_each_axis_and_the_gather_routes(self):
        names = [config.name for config in sweep.sweep_grid('rs')]
        self.assertEqual(names, ['served', 'rs-l1', 'rs-w1', 'rs-w3', 'rs-w4', 'rs-c1', 'rs-c2', 'rs-c5', 'rs-c20', 'rs-c50', 'rs-b1', 'rs-b3', 'rs-b4'])
        gather = [config.name for config in sweep.sweep_grid('ag')]
        self.assertEqual(gather[:-2], names[:1] + ['ag' + name[2:] for name in names[1:]])
        self.assertEqual(gather[-2:], ['ag-linear', 'ag-bcast'])

    def test_the_grid_never_offers_what_changes_bytes(self):
        for op in ('rs', 'ag'):
            for config in sweep.sweep_grid(op):
                for name in config.overrides:
                    self.assertIn(ccl_options_tp.option_effect(op, name), (ccl_options_tp.EXACT, ccl_options_tp.COPY), (op, name))

    def test_the_probe_only_arms_are_outside_the_grammar(self):
        for op in ('rs', 'ag'):
            arms = sweep.probe_only_grid(op)
            self.assertEqual([arm.probe_only for arm in arms], ['nobar', 'pbuf'])
            for arm in arms:
                with self.assertRaises(ValueError):
                    ccl_options_tp.parse_set(arm.name)
                self.assertEqual(arm.name, '%s-%s' % (op, arm.probe_only if arm.probe_only == 'pbuf' else 'nobar'))

    def test_a_set_of_the_other_op_is_refused(self):
        with self.assertRaises(ValueError):
            sweep.named('rs', 'ag-w1')

    def test_combinations_take_the_best_value_of_each_axis_and_need_two_axes(self):
        def row(name, axis, gain):
            return dict(name=name, axis=axis, promote=True, median_gain_us=gain)
        rows = [row('rs-w1', 'num_workers_per_link', 2.0), row('rs-c1', 'chunks_per_sync', 1.5), row('rs-c5', 'chunks_per_sync', 1.0),
                row('rs-b1', 'num_buffers_per_channel', 0.5), dict(name='rs-l1', axis='num_links', promote=False, median_gain_us=9.0)]
        combos = sweep.combinations('rs', rows)
        self.assertEqual([config.name for config in combos], ['rs-c1+rs-w1', 'rs-b1+rs-c1+rs-w1'])
        self.assertEqual(combos[0].overrides, {'num_workers_per_link': 1, 'chunks_per_sync': 1})
        self.assertEqual(sweep.combinations('rs', rows[:1]), [])
        self.assertEqual(sweep.combinations('rs', []), [])


class TheStatistics(unittest.TestCase):
    def test_the_slope_removes_the_fixed_cost(self):
        high = [sweep.N_HIGH * 18000 + 30000] * 3
        low = [sweep.N_LOW * 18000 + 30000] * 3
        self.assertAlmostEqual(sweep.per_call_us(high, low), 18.0)

    def test_the_paired_read(self):
        verdict = sweep.paired_verdict([18, 18.1, 17.9, 18, 18], [16, 16.2, 16, 16, 16.1], noise_us=0.1)
        self.assertTrue(verdict['faster'])
        self.assertEqual(verdict['wins'], 5)
        self.assertAlmostEqual(verdict['median_gain_us'], 1.9, places=2)

    def test_a_gain_inside_the_noise_or_below_the_floor_does_not_promote(self):
        self.assertFalse(sweep.paired_verdict([18] * 5, [17.8] * 5, 0.05)['faster'])             # 0.2 us < the floor
        self.assertFalse(sweep.paired_verdict([18] * 5, [17.0] * 5, 0.6)['faster'])              # 1.0 us < twice the noise
        self.assertFalse(sweep.paired_verdict([18, 18, 18, 18, 18], [16, 19, 19, 16, 19], 0.1)['faster'])    # wins 2 of 5
        self.assertTrue(sweep.paired_verdict([18] * 5, [16, 16, 16, 16, 19], 0.1)['faster'])      # 4 of 5

    def test_bit_comparison_is_by_pattern_so_minus_zero_differs(self):
        left = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
        right = torch.tensor([-0.0, 1.0], dtype=torch.bfloat16)
        self.assertEqual(sweep.bits_equal(torch, left, right), (2, 1))
        self.assertEqual(sweep.bits_equal(torch, left, left.clone()), (2, 0))
        with self.assertRaises(ValueError):
            sweep.bits_equal(torch, left, torch.zeros(3, dtype=torch.bfloat16))

    def test_the_special_patterns_are_in_the_inputs(self):
        data = sweep.ag_inputs(torch, 0)
        bits = data.view(torch.int16)[:, 0, 0, :len(sweep.SPECIAL_BITS)]
        self.assertEqual(sorted(int(v) & 0xFFFF for v in bits[0]), sorted(sweep.SPECIAL_BITS))
        self.assertEqual(tuple(data.shape), (CHIPS, 1, sweep.ROWS, sweep.SHARD))
        self.assertEqual(tuple(sweep.rs_partials(torch, 0).shape), (CHIPS, 1, sweep.ROWS, sweep.WIDTH))

    def test_the_fingerprints_of_two_reports(self):
        self.assertEqual(sweep.compare_reports({'fingerprints': {'a': '1', 'b': '2'}}, {'fingerprints': {'a': '1', 'b': '3', 'c': '4'}}),
                         {'a': True, 'b': False, 'c': None})


class TheSweep(Fixture):
    def test_the_sweep_finds_the_faster_exact_configs_and_combines_them(self):
        ttnn = self.runtime()
        report, saved = self.sweep(ttnn, only='rs', skip_probe_only=True)
        rows = self.rows(report, 'rs/dram')
        self.assertEqual(rows['served']['status'], 'EXACT')
        self.assertAlmostEqual(rows['served']['base_us'], 18.0, places=2)
        self.assertEqual(rows['served']['noise_us'], 0.0)
        self.assertTrue(rows['rs-w1']['promote'])
        self.assertAlmostEqual(rows['rs-w1']['median_gain_us'], 2.0, places=2)
        self.assertTrue(rows['rs-c1']['promote'])
        self.assertTrue(rows['rs-c5']['promote'])
        self.assertFalse(rows['rs-l1']['promote'])                          # four microseconds slower
        self.assertLess(rows['rs-l1']['median_gain_us'], 0)
        self.assertFalse(rows['rs-b1']['promote'])                          # no change: inside the floor
        combined = [name for name in rows if '+' in name]
        self.assertEqual(combined, ['rs-c1+rs-w1'])
        self.assertAlmostEqual(rows['rs-c1+rs-w1']['median_gain_us'], 3.5, places=2)
        self.assertTrue(rows['rs-c1+rs-w1']['promote'])
        self.assertEqual(len(saved), len(report['scenarios']['rs/dram']['rows']) + 2 * 0 + len(saved) - len(report['scenarios']['rs/dram']['rows']))

    def test_every_scenario_is_swept_and_reported(self):
        report, _ = self.sweep(self.runtime(), skip_probe_only=True)
        self.assertEqual(sorted(report['scenarios']), ['ag/dram-ws', 'ag/l1-dram', 'ag/l1-ws', 'rs/dram', 'rs/l1'])
        for name, scenario in report['scenarios'].items():
            self.assertTrue(all(row['status'] == 'EXACT' for row in scenario['rows']), name)
        self.assertEqual(sorted(report['fingerprints']), sorted(report['scenarios']))
        self.assertTrue(all(len(value) == 64 for value in report['fingerprints'].values()))
        self.assertTrue(self.rows(report, 'ag/l1-ws')['ag-bcast']['promote'])
        self.assertFalse(self.rows(report, 'ag/l1-ws')['ag-w1']['promote'])

    def test_an_option_that_changes_a_bit_is_never_timed_or_promoted(self):
        ttnn = self.runtime(corrupt=lambda kwargs: kwargs.get('chunks_per_sync') == 5)
        report, _ = self.sweep(ttnn, only='rs', skip_probe_only=True)
        rows = self.rows(report, 'rs/dram')
        self.assertEqual(rows['rs-c5']['status'], 'FAIL-BYTES')
        self.assertFalse(rows['rs-c5']['exact'])
        self.assertGreater(rows['rs-c5']['differing'], 0)
        self.assertNotIn('median_gain_us', rows['rs-c5'])
        self.assertFalse(rows['rs-c5']['promote'])
        self.assertNotIn('rs-c5', ' '.join(name for name in rows if '+' in name))
        self.assertTrue(rows['rs-c1']['promote'])

    def test_a_config_the_runtime_refuses_is_an_error_row_and_the_sweep_goes_on(self):
        ttnn = self.runtime(refuse=lambda kwargs: kwargs.get('num_workers_per_link') == 1)
        report, _ = self.sweep(ttnn, only='rs', skip_probe_only=True)
        rows = self.rows(report, 'rs/dram')
        self.assertEqual(rows['rs-w1']['status'], 'ERROR')
        self.assertIn('TT_FATAL', rows['rs-w1']['error'])
        self.assertTrue(rows['rs-c1']['promote'])

    def test_a_served_config_that_is_not_the_sequential_engines_stops_that_scenario(self):
        ttnn = self.runtime(corrupt=lambda kwargs: kwargs.get('num_workers_per_link') == 2 and kwargs.get('chunks_per_sync') == 10)
        report, _ = self.sweep(ttnn, only='rs', skip_probe_only=True)
        self.assertEqual(sorted(report['scenarios']), ['rs/dram', 'rs/l1'])
        for scenario in report['scenarios'].values():
            self.assertEqual([row['name'] for row in scenario['rows']], ['served'])
            self.assertEqual(scenario['rows'][0]['status'], 'FAIL-BYTES')
        self.assertEqual(sweep.verdict(dict(report, opened=True)), ('BASE-INEXACT', 1))
        self.assertEqual(report['unfinished'], ['rs/dram', 'rs/l1'])

    def test_the_probe_only_arms_run_last_are_exact_and_never_promoted(self):
        report, _ = self.sweep(self.runtime(), only='rs')
        rows = report['scenarios']['rs/dram']['rows']
        self.assertEqual([row['name'] for row in rows[-2:]], ['rs-nobar', 'rs-pbuf'])
        by_name = self.rows(report, 'rs/dram')
        self.assertTrue(by_name['rs-nobar']['faster'])
        self.assertFalse(by_name['rs-nobar']['promote'])
        self.assertFalse(by_name['rs-pbuf']['promote'])
        self.assertEqual(by_name['rs-nobar']['probe_only'], 'nobar')
        self.assertAlmostEqual(by_name['rs-nobar']['median_gain_us'], 3.0, places=2)

    def test_skip_probe_only_leaves_them_out(self):
        report, _ = self.sweep(self.runtime(), only='ag', skip_probe_only=True)
        for scenario in report['scenarios'].values():
            self.assertFalse([row for row in scenario['rows'] if row.get('probe_only')])

    def test_the_report_is_saved_after_every_config(self):
        report, saved = self.sweep(self.runtime(), only='ag', skip_probe_only=True)
        counts = [sum(len(scenario['rows']) for scenario in snapshot['scenarios'].values()) for snapshot in saved]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], sum(len(scenario['rows']) for scenario in report['scenarios'].values()))
        self.assertGreater(len(set(counts)), 10)

    def test_every_trace_and_buffer_is_released(self):
        ttnn = self.runtime()
        self.sweep(ttnn)
        self.assertEqual(ttnn.traces, {})

    def test_the_rs_keywords_are_the_models_and_the_gather_keywords_are_distributed_norms(self):
        harness = sweep.Harness(self.runtime(), torch, FakeMesh(), Collective(), None)
        keywords = harness.rs_keywords(sweep.base_config('rs'), FakeTTNN.DRAM_MEMORY_CONFIG)
        self.assertEqual(sorted(keywords), sorted(['persistent_output_buffers', 'dim', 'multi_device_global_semaphore', 'barrier_semaphore', 'num_links',
                                                   'memory_config', 'intermediate_memory_config', 'topology', 'chunks_per_sync', 'num_workers_per_link',
                                                   'num_buffers_per_channel']))
        self.assertEqual((keywords['num_links'], keywords['chunks_per_sync'], keywords['num_workers_per_link'], keywords['num_buffers_per_channel'],
                          keywords['topology'], keywords['dim']), (2, 10, 2, 2, 'Ring', 3))
        import distributed_norm_gather_tp
        gather = harness.ag_keywords(sweep.base_config('ag'), FakeTTNN.DRAM_MEMORY_CONFIG)
        self.assertEqual(set(gather), set(distributed_norm_gather_tp.CENSUS_KEYWORDS))
        self.assertEqual(sweep.Harness(self.runtime(), torch, FakeMesh(), Collective(), None).ag_keywords(
            sweep.named('ag', 'ag-linear+ag-bcast'), FakeTTNN.DRAM_MEMORY_CONFIG)['topology'], 'Linear')

    def test_the_summary_and_the_verdict_line(self):
        report, _ = self.sweep(self.runtime(), only='rs')
        report.update(fabric='FABRIC_1D', payload_actual=4352)
        report['summary'] = sweep.summarise(report)
        text, status = sweep.verdict(report)
        self.assertEqual((text, status), ('DONE', 0))
        line = sweep.verdict_line(report, text)
        self.assertTrue(line.startswith('CCL_SWEEP verdict=DONE fabric=FABRIC_1D payload=4352 scenarios=2 '))
        self.assertIn('rs-c1+rs-w1', line)
        self.assertNotIn('nobar', line)
        self.assertEqual(report['summary']['rs/dram']['best'], 'rs-c1+rs-w1')
        self.assertEqual(sweep.verdict({'opened': False}), ('NOT-MEASURED', 2))
        self.assertEqual(sweep.verdict({'opened': True, 'scenarios': {}}), ('INCOMPLETE', 0))


class TheDeadline(unittest.TestCase):
    def test_a_hung_config_writes_the_partial_report_and_exits_3(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, 'partial.json')
        exits, lines = [], []
        deadline = sweep.Deadline({'scenarios': {'rs/dram': {'rows': []}}}, path, seconds=0.05, exit_function=exits.append, log=lines.append)
        deadline.arm('rs/dram rs-w1')
        import time
        for _ in range(100):
            if exits:
                break
            time.sleep(0.02)
        self.assertEqual(exits, [3])
        with open(path) as handle:
            self.assertEqual(json.load(handle)['watchdog'], 'rs/dram rs-w1')
        self.assertIn('rs/dram rs-w1', lines[0])

    def test_a_disarmed_deadline_never_fires_and_an_unlimited_one_does_nothing(self):
        exits = []
        deadline = sweep.Deadline({}, '/nonexistent/x', seconds=0.05, exit_function=exits.append, log=lambda text: None)
        deadline.arm('x')
        deadline.disarm()
        import time
        time.sleep(0.15)
        self.assertEqual(exits, [])
        sweep.Deadline({}, None).arm('y')


class TheWrapper(Fixture):
    def load(self):
        with open(self.path) as handle:
            return json.load(handle)

    def argv(self, *extra):
        return ['--fabric', 'FABRIC_1D', '--output', self.path, '--quick', '--rounds', '3', '--replays', '3'] + list(extra)

    def modules(self):
        return dict(TT_CCL=lambda mesh: Collective(), tt_all_reduce=model_all_reduce(None), get_num_links=lambda mesh: 2)

    def run_main(self, ttnn, *extra):
        status = wrapper.main(self.argv(*extra), ttnn=ttnn, torch=torch, environ={}, log=self.lines.append, modules=self.modules(), deadlines=False)
        return status, self.load()

    def test_a_run_opens_the_mesh_once_and_ends_with_the_verdict_line_and_json(self):
        ttnn = self.runtime()
        ttnn.clock = self.clock
        status, report = self.run_main(ttnn, '--skip-probe-only', '--only', 'rs')
        self.assertEqual(status, 0)
        self.assertEqual([event[0] for event in ttnn.events], ['open', 'close'])
        self.assertEqual(ttnn.events[0][1], (1, 4))
        self.assertEqual(ttnn.events[0][2]['trace_region_size'], 256 * 1024 * 1024)
        self.assertEqual(ttnn.fabric, ('FABRIC_1D', None))
        self.assertEqual(report['kind'], 'ccl-sweep-quad')
        self.assertEqual(report['verdict'], 'DONE')
        self.assertTrue(report['closed'])
        self.assertEqual(report['payload_actual'], 4352)
        self.assertEqual(report['order'], [3, 1, 0, 2])
        self.assertIn('CCL_SWEEP verdict=DONE', [line for line in self.lines if line.startswith('CCL_SWEEP verdict')][0])
        self.assertEqual(json.loads(self.lines[-1])['verdict'], 'DONE')

    def test_the_ring_fabric_and_a_payload_are_set_before_the_mesh_opens(self):
        ttnn = self.runtime()
        status, report = self.run_main(ttnn, '--skip-probe-only', '--only', 'ag')
        status, report = None, None
        ttnn = self.runtime()
        wrapper_argv = ['--fabric', 'FABRIC_1D_RING', '--output', self.path, '--quick', '--rounds', '3', '--replays', '3', '--payload', '8192',
                        '--skip-probe-only', '--only', 'ag']
        status = wrapper.main(wrapper_argv, ttnn=ttnn, torch=torch, environ={}, log=self.lines.append, modules=self.modules(), deadlines=False)
        report = self.load()
        self.assertEqual(status, 0)
        self.assertEqual(ttnn.fabric, ('FABRIC_1D_RING', 8192))
        self.assertEqual((report['payload_requested'], report['payload_actual'], report['lever_moved']), (8192, 8192, True))

    def test_a_payload_the_runtime_does_not_adopt_measures_nothing(self):
        ttnn = self.runtime(payload_report=4352)
        status, report = self.run_main(ttnn, '--payload', '8192')
        self.assertEqual(status, 2)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertFalse(report['lever_moved'])
        self.assertEqual(report['scenarios'], {})
        self.assertIn('did not move', report['error'])
        self.assertEqual(ttnn.events[-1], ('close',))

    def test_a_payload_below_two_pages_is_refused_before_anything_opens(self):
        ttnn = self.runtime()
        for payload in ('2048', '4095', '15233'):
            status, report = self.run_main(ttnn, '--payload', payload)
            self.assertEqual(status, 2)
            self.assertIn('refused', report['error'])
        self.assertEqual(ttnn.events, [])

    def test_too_few_rounds_are_refused(self):
        ttnn = self.runtime()
        status = wrapper.main(['--output', self.path, '--rounds', '2'], ttnn=ttnn, torch=torch, environ={}, log=self.lines.append, modules=self.modules(),
                              deadlines=False)
        self.assertEqual(status, 2)
        self.assertEqual(ttnn.events, [])

    def test_a_mesh_that_is_not_four_chips_at_two_links_is_not_measured(self):
        ttnn = self.runtime()
        modules = dict(self.modules(), get_num_links=lambda mesh: 4)
        status = wrapper.main(self.argv('--only', 'rs'), ttnn=ttnn, torch=torch, environ={}, log=self.lines.append, modules=modules, deadlines=False)
        report = self.load()
        self.assertEqual(status, 0 if report['scenarios'] else 0)
        self.assertIn('census is four chips at two links', report['error'])
        self.assertEqual(report['scenarios'], {})

    def test_a_served_config_that_differs_is_exit_1(self):
        ttnn = self.runtime(corrupt=lambda kwargs: kwargs.get('num_workers_per_link') == 2)
        status, report = self.run_main(ttnn, '--only', 'rs', '--skip-probe-only')
        self.assertEqual(status, 1)
        self.assertEqual(report['verdict'], 'BASE-INEXACT')

    def test_a_wedged_runtime_stops_the_sweep_after_three_errors_in_a_row_and_the_mesh_is_closed(self):
        ttnn = self.runtime()

        def boom(*args, **kwargs):
            raise RuntimeError('Timed out while waiting for active ethernet core')
        ttnn.begin_trace_capture = boom
        status, report = self.run_main(ttnn)
        self.assertTrue(report['closed'])
        self.assertIn('ethernet-core wedge', report.get('known_failure') or '')
        self.assertIn('3 configs in a row', report['error'])
        self.assertEqual(report['verdict'], 'INCOMPLETE')

    def test_the_descriptor_is_checked_when_the_runtime_is_the_images(self):
        status = wrapper.main(['--output', self.path], environ={'TT_MESH_GRAPH_DESC_PATH': '/nonexistent/other.textproto'}, log=self.lines.append,
                              deadlines=False)
        report = self.load()
        self.assertEqual(status, 2)
        self.assertIn('refused to open', report['error'])


if __name__ == '__main__':
    unittest.main()
