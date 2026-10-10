"""draft_reduce_tp (QWEN_FAST_DRAFT_REDUCE, F-F1): the drafter's gather-add chain with the slices and adds as one launch.

Held here, on the CPU (nothing runs on a card; the audited card jobs are scripts/ci/references/fusion-jobs/WP6):

  - the plan: every tile of the (rows, 5120) block is in exactly one run, runs are contiguous and balanced, the workers never exceed the grid
    the device reports (11 x 10, 13 x 10, or anything else) or the cap, and the cores named are inside the grid;
  - the launch the builder describes, EXECUTED by the transliteration of draft_reduce_tp_io.cpp / _compute.cpp / draft_fuse_out.cpp that reads the
    launch's own runtime arguments, circular buffers and compile-time arguments: its output equals the served slices-and-adds bit for bit on
    every chip, for 32 and 64 rows (and the 1 and 8 rows of the feature projection), on data with zeros of both signs, denormals, infinities,
    cancellation and a hundredfold dynamic range - and a different add order, or a reversed chip order, does NOT (the test can fail);
  - the gather is the served call (same arguments, semaphores cycled once each), the tensors retained and freed are the served ones' minus the
    slices and partial sums, the returned output is the served add's;
  - the .cpp sources read exactly the runtime arguments the builder writes (an argument-index cross-check of the source text);
  - the flag: strict 0 or 1, refused at the pair, an audit without its lever raises; with it off the dispatcher is the served function call for
    call and the new module is not even imported;
  - fall-backs (a shape, dtype, layout, placement or grid it cannot take; a launch that raises) run the served ops and say why once;
  - the audit: exact on the launch, and a flipped bit in the launch's output is a logged mismatch and an AssertionError.

    py -3.11 -B -m unittest test_draft_reduce_tp      (from scripts/ci)
"""

import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import draft_fusion_tp as fusion  # noqa: E402
import draft_reduce_tp as reduce  # noqa: E402
import feature_collective_tp as collective  # noqa: E402
import wp6_fake_device as fake  # noqa: E402

FOUR = {'QWEN_FAST_TP': '4'}
REDUCE_ON = {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_REDUCE': '1'}
AUDIT_ON = {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_REDUCE': '1', 'QWEN_FAST_DRAFT_REDUCE_AUDIT': '1'}


def mesh_for(x=11, y=10, chips=4):
    return SimpleNamespace(shape=[1, chips], compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


COLLECTIVES = SimpleNamespace(get_and_cycle_ag_semaphore_handles=lambda: 'ag', get_and_cycle_barrier_semaphore_handle=lambda: 'barrier')


def bits(tensor):
    return [shard.data.contiguous().view(torch.int32) for shard in tensor.shards]


def same(left, right):
    return all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(bits(left), bits(right), strict=True))


def hard_values(generator, shape, chip=0):
    """fp32 data with the cases an add chain can get wrong: wide dynamic range, zeros of both signs, denormals, near-overflow, infinities of
    opposite signs on different chips (their sum is NaN), and an exact cancellation between chips."""
    value = torch.randn(*shape, generator=generator) * torch.exp(torch.randn(*shape, generator=generator) * 3)
    flat = value.reshape(-1)
    flat[::97] = 0.0
    flat[1::97] = -0.0                                    # the same cells on every chip: a sum of negative zeros is negative zero
    flat[2::101] = 1e-40 if chip % 2 == 0 else -1e-40     # denormals
    flat[4::211] = 3.0e38                                 # near overflow on every chip: the running sum overflows to infinity
    flat[5::307] = float('inf') if chip % 2 == 0 else float('-inf')
    flat[8::113] = 1.5 if chip == 0 else (-1.5 if chip == 1 else 0.0)       # p0 + p1 cancels exactly, then + 0 + 0
    return value


def partials(operations, rows, seed=0, hard=False):
    """A (1, 1, rows, 5120) fp32 projection per chip, the chips' partials."""
    generator = torch.Generator().manual_seed(seed)
    draw = (lambda chip: hard_values(generator, (1, 1, rows, 5120), chip)) if hard else (
        lambda chip: torch.randn(1, 1, rows, 5120, generator=generator))
    return operations.from_chips([draw(chip) for chip in range(operations.chips)], 'fp32')


def served_chain(operations, mesh, collectives, value, **options):
    """The served function itself: feature_collective_tp.served_gather_add_projection at 1, 8 and 32 rows (the pinned body, chip count from
    tp_shapes) and quad_draft.gather_add_projection at the quad's 64."""
    if tuple(value.shape)[2] == 64:
        import quad_draft

        return quad_draft.gather_add_projection(operations, mesh, collectives, value, **options)
    return collective.served_gather_add_projection(operations, mesh, collectives, value, **options)


def fresh(chips=4, grid=(11, 10)):
    operations = fake.install_emulators(fake.FakeOperations(chips), grid)
    fusion.reset()
    return operations


class EnvTestCase(unittest.TestCase):
    environment = FOUR

    def setUp(self):
        patcher = patch.dict(os.environ, self.environment, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        for target in ('mesh_link_policy.projection_links', 'feature_collective_tp.projection_links'):
            links = patch(target, return_value=1)
            links.start()
            self.addCleanup(links.stop)
        self.lines = []
        quiet = patch.object(fusion, 'log_line', side_effect=self.lines.append)
        quiet.start()
        self.addCleanup(quiet.stop)
        fusion.reset()


class PlanTests(unittest.TestCase):
    def test_runs_cover_every_tile_once_contiguously_and_balanced(self):
        for tiles in (1, 5, 109, 110, 111, 160, 272, 320):
            for workers in sorted({1, 2, 7, 64, 110, 130, tiles}):
                if workers > tiles:
                    continue
                runs = fusion.plan_runs(tiles, workers)
                self.assertEqual(len(runs), workers)
                covered = [tile for first, count in runs for tile in range(first, first + count)]
                self.assertEqual(covered, list(range(tiles)))
                counts = [count for _, count in runs]
                self.assertLessEqual(max(counts) - min(counts), 1)
                self.assertGreaterEqual(min(counts), 1)

    def test_the_workers_follow_the_grid_the_device_reports_and_never_exceed_the_cap(self):
        for grid in ((11, 10), (13, 10), (8, 10), (8, 8), (1, 1), (13, 12)):
            for rows, tiles in ((32, 160), (64, 320), (8, 160), (1, 160)):
                found = reduce.plan(rows, grid)
                self.assertEqual(found['tiles'], tiles)
                self.assertEqual(found['stride'], tiles)
                self.assertEqual(found['workers'], min(tiles, grid[0] * grid[1], fusion.WORKER_CAP))
                points = fusion.coordinates(grid, found['workers'])
                self.assertEqual(len(set(points)), found['workers'])
                self.assertTrue(all(0 <= x < grid[0] and 0 <= y < grid[1] for x, y in points))

    def test_the_core_ranges_are_the_same_cores_as_the_coordinates(self):
        operations = fake.FakeOperations()
        for grid, workers in (((11, 10), 110), ((13, 10), 110), ((11, 10), 64), ((13, 10), 5), ((8, 10), 80)):
            ranges = fusion.core_ranges(operations, grid, workers)
            cells = set()
            for (x0, y0), (x1, y1) in ranges:
                cells |= {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
            self.assertEqual(cells, set(fusion.coordinates(grid, workers)))

    def test_what_it_refuses(self):
        for rows in (0, 65, 32.0, None):
            with self.assertRaises(ValueError):
                reduce.plan(rows, (11, 10))
        with self.assertRaises(ValueError):
            reduce.plan(32, (11, 10), chips=5)
        with self.assertRaises(ValueError):
            fusion.plan_runs(3, 4)
        with self.assertRaises(ValueError):
            fusion.coordinates((2, 2), 5)


class ArithmeticTests(EnvTestCase):
    environment = REDUCE_ON

    def run_launch(self, rows, hard, seed=0, grid=(11, 10)):
        operations = fresh(grid=grid)
        mesh = mesh_for(*grid)
        value = partials(operations, rows, seed, hard)
        gathered = operations.all_gather_async(value, dim=0)
        output = reduce.reduce_launch(operations, mesh, gathered, rows, grid, 4)
        reference = reduce.served_sum(operations, gathered, rows, 4, lambda tensor: tensor)
        return operations, output, reference, gathered

    def test_the_launch_is_the_served_slices_and_adds_bit_for_bit_on_every_chip(self):
        for rows in (1, 8, 32, 64):
            for hard in (False, True):
                for grid in ((11, 10), (13, 10), (8, 10)):
                    with self.subTest(rows=rows, hard=hard, grid=grid):
                        operations, output, reference, _ = self.run_launch(rows, hard, seed=rows, grid=grid)
                        self.assertEqual(output.shape, (1, 1, rows, 5120))
                        self.assertEqual(output.dtype, 'fp32')
                        self.assertTrue(same(output, reference))
                        tiles = reduce.tile_rows(rows) * 160
                        self.assertEqual(fake.emulate_reduce.touched, 4 * tiles)         # every tile of every chip written exactly once

    def test_the_hard_data_really_exercises_zeros_nans_and_cancellation(self):
        operations, output, reference, _ = self.run_launch(64, True, seed=3)
        data = output.shards[0].data
        self.assertTrue(torch.isnan(data).any(), 'inf + -inf must occur')
        self.assertTrue(torch.isinf(data).any())
        self.assertTrue((data == 0).any())
        self.assertTrue(((data.view(torch.int32) == torch.tensor(-2 ** 31, dtype=torch.int32))).any(), 'a negative zero result must occur')

    def test_a_different_add_order_does_not_pass_this_test(self):
        operations, output, reference, gathered = self.run_launch(64, False, seed=11)
        pieces = [operations.slice(gathered, (chip, 0, 0, 0), (chip + 1, 1, 64, 5120)) for chip in range(4)]
        left = operations.add(pieces[0], pieces[1], dtype='fp32')
        right = operations.add(pieces[2], pieces[3], dtype='fp32')
        paired = operations.add(left, right, dtype='fp32')                    # (p0 + p1) + (p2 + p3)
        self.assertFalse(same(output, paired), 'the pairwise order must differ from ((p0 + p1) + p2) + p3 on random data')
        backwards = operations.add(operations.add(operations.add(pieces[3], pieces[2], dtype='fp32'), pieces[1], dtype='fp32'),
                                   pieces[0], dtype='fp32')
        self.assertFalse(same(output, backwards), 'a reversed chip order must differ')

    def test_the_output_lives_in_its_own_buffer_and_the_gathered_tensor_is_untouched(self):
        operations, output, reference, gathered = self.run_launch(32, False)
        before = [shard.data.clone() for shard in gathered.shards]
        self.assertTrue(all(torch.equal(shard.data, kept) for shard, kept in zip(gathered.shards, before, strict=True)))
        self.assertNotEqual({shard.address for shard in output.shards} & {shard.address for shard in gathered.shards}, True)


class LaunchTests(EnvTestCase):
    environment = REDUCE_ON

    def run_gather_add(self, rows, *, retain=True, quad=False, hard=False):
        operations = fresh()
        mesh = mesh_for()
        value = partials(operations, rows, rows, hard)
        kept, served_kept = [], []
        served_ops = fresh()
        served_value = served_ops.from_chips([shard.data for shard in value.shards], 'fp32')
        served = served_chain(served_ops, mesh, COLLECTIVES, served_value,
                              **(dict(retain_temporaries=lambda tensor: served_kept.append(tensor) or tensor) if retain else {}))
        result = reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='mlp' if quad else 'feature',
                                   quad=quad, **(dict(retain_temporaries=lambda tensor: kept.append(tensor) or tensor) if retain else {}))
        return operations, served_ops, result, served, kept

    def test_the_result_is_the_served_chains_on_every_chip_at_every_row_count(self):
        for rows, quad in ((1, False), (8, False), (32, False), (64, True)):
            with self.subTest(rows=rows):
                operations, served_ops, result, served, kept = self.run_gather_add(rows, quad=quad, hard=True)
                self.assertTrue(same(result, served))
                self.assertEqual(result.shape, (1, 1, rows, 5120))

    def test_one_gather_one_launch_and_no_slice_or_add(self):
        operations, served_ops, result, served, kept = self.run_gather_add(64, quad=True)
        self.assertEqual(operations.names('all_gather', 'generic_op', 'slice', 'add'), ['all_gather', 'generic_op'])
        self.assertEqual(served_ops.names('all_gather', 'generic_op', 'slice', 'add'),
                         ['all_gather'] + ['slice'] * 4 + ['add'] * 3, 'the served chain: one gather, four slices, three adds')
        self.assertEqual(len(operations.launches), 1)

    def test_the_gather_is_the_served_call(self):
        captured = []

        class Spy(fake.FakeOperations):
            def all_gather_async(self, tensor, **options):
                captured.append(options)
                return super().all_gather_async(tensor, **options)

        for use_launch in (False, True):
            operations = fake.install_emulators(Spy())
            value = partials(operations, 64)
            mesh = mesh_for()
            if use_launch:
                reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='mlp', quad=True,
                                  retain_temporaries=lambda tensor: tensor)
            else:
                served_chain(operations, mesh, COLLECTIVES, value, retain_temporaries=lambda tensor: tensor)
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0], captured[1], 'dim, semaphores, links, topology, workers, buffers: the served call\'s')
        self.assertEqual(captured[0]['dim'], 0)
        self.assertEqual(captured[0]['topology'], 'linear')

    def test_trace_owned_the_gathered_tensor_is_retained_and_nothing_is_freed(self):
        operations, served_ops, result, served, kept = self.run_gather_add(64, quad=True, retain=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].name, 'gathered')
        self.assertEqual([entry for entry in operations.log if entry[0] == 'deallocate'], [])
        self.assertFalse(result.freed)

    def test_eager_the_gathered_tensor_is_freed_and_the_output_is_not(self):
        operations, served_ops, result, served, kept = self.run_gather_add(32, retain=False)
        freed = [entry for entry in operations.log if entry[0] == 'deallocate']
        self.assertEqual(freed, [('deallocate', 'gathered')])
        self.assertFalse(result.freed)
        self.assertGreaterEqual(operations.sync_count, 1)

    def test_a_failing_launch_frees_its_output_and_runs_the_served_adds_on_the_gathered_tensor(self):
        operations = fresh()
        mesh = mesh_for()
        value = partials(operations, 64, 5)
        operations.emulators.insert(0, lambda ops, tensors, program: (_ for _ in ()).throw(RuntimeError('compile failed')))
        kept = []
        result = reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='mlp', quad=True,
                                   retain_temporaries=lambda tensor: kept.append(tensor) or tensor)
        served_ops = fresh()
        reference = served_chain(served_ops, mesh, COLLECTIVES, served_ops.from_chips([s.data for s in value.shards], 'fp32'),
                                 retain_temporaries=lambda tensor: tensor)
        self.assertTrue(same(result, reference))
        self.assertTrue(any(line.startswith(fusion.REDUCE_FALLBACK) and 'the launch failed' in line for line in self.lines))
        empties = [entry for entry in operations.log if entry[0] == 'empty']
        freed = [entry for entry in operations.log if entry[0] == 'deallocate']
        self.assertEqual(len(empties), 1)
        self.assertIn(('deallocate', 'empty'), freed)

    def test_the_chains_counter_reaches_its_milestones_and_each_engaged_line_prints_once(self):
        operations = fresh()
        mesh = mesh_for()
        for _ in range(11):
            reduce.gather_add(operations, mesh, COLLECTIVES, partials(operations, 64), served=served_chain, site='mlp', quad=True,
                              retain_temporaries=lambda tensor: tensor)
        engaged = [line for line in self.lines if line.startswith(fusion.REDUCE_ENGAGED)]
        self.assertEqual([int(re.search(r'chains=(\d+)', line).group(1)) for line in engaged], [1, 5, 10])
        self.assertIn('rows=64', engaged[0])
        self.assertIn('workers=110', engaged[0])


class FallbackTests(EnvTestCase):
    environment = REDUCE_ON

    def attempt(self, **changes):
        operations = fresh()
        mesh = changes.pop('mesh', mesh_for())
        rows = changes.pop('rows', 64)
        value = partials(operations, rows)
        for name, new in changes.items():
            setattr(value, name, new)
        calls = []

        def served(*args, **options):
            calls.append(args[3])
            return args[3]
        result = reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served, site='mlp', quad=changes.pop('quad', True),
                                   retain_temporaries=lambda tensor: tensor)
        return operations, value, result, calls

    def test_each_call_it_cannot_take_goes_to_the_served_function_untouched_and_says_why_once(self):
        cases = (dict(dtype='bf16'), dict(layout='row_major'), dict(shape=(1, 1, 64, 4096)), dict(shape=(1, 2, 64, 5120)),
                 dict(rows=48, shape=(1, 1, 48, 5120)), dict(mesh=mesh_for(chips=2)))
        for case in cases:
            with self.subTest(case=sorted(case)):
                self.lines[:] = []
                fusion.reset()
                operations, value, result, calls = self.attempt(**case)
                self.assertEqual(calls, [value], 'the served function got the very tensor')
                self.assertIs(result, value)
                self.assertEqual(operations.launches, [])
                self.assertEqual(operations.names('all_gather'), [], 'no gather of its own: the served function does it')
                self.assertEqual(len([line for line in self.lines if line.startswith(fusion.REDUCE_FALLBACK)]), 1)

    def test_a_grid_that_cannot_be_read_and_a_quad_without_an_owner_fall_back(self):
        def broken():
            raise RuntimeError('no grid')
        mesh = SimpleNamespace(shape=[1, 4], compute_with_storage_grid_size=broken)
        operations, value, result, calls = self.attempt(mesh=mesh)
        self.assertEqual(calls, [value])
        operations = fresh()
        served_calls = []
        value = partials(operations, 64)
        reduce.gather_add(operations, mesh_for(), COLLECTIVES, value, served=lambda *a, **k: served_calls.append(1), site='mlp', quad=True)
        self.assertEqual(served_calls, [1], 'a quad chain without retain_temporaries is the served function\'s refusal, not ours')

    def test_sixty_four_rows_outside_a_quad_pass_are_not_taken(self):
        operations, value, result, calls = self.attempt(quad=False)
        self.assertEqual(calls, [value])


class FlagTests(unittest.TestCase):
    def test_strict_zero_or_one(self):
        for value in ('', '2', 'true', 'on', '01'):
            with patch.dict(os.environ, {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_REDUCE': value}, clear=True):
                with self.assertRaises(ValueError):
                    fusion.enabled(fusion.REDUCE)
                with self.assertRaises(ValueError):
                    collective._reduce_requested()
        for value, expected in (('0', False), ('1', True)):
            with patch.dict(os.environ, {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_REDUCE': value}, clear=True):
                self.assertIs(fusion.enabled(fusion.REDUCE), expected)
                self.assertIs(collective._reduce_requested(), expected)
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}, clear=True):
            self.assertFalse(fusion.enabled(fusion.REDUCE))

    def test_refused_at_the_pair_for_every_wp6_flag(self):
        for name in fusion.ALL_FLAGS:
            environment = {name: '1'}
            if name in fusion.AUDITS:
                environment[fusion.AUDITS[name]] = '1'
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(ValueError, 'TP4 levers'):
                    fusion.validate()
        with patch.dict(os.environ, {}, clear=True):
            fusion.validate()

    def test_an_audit_without_its_lever_raises(self):
        for audit, lever in fusion.AUDITS.items():
            with patch.dict(os.environ, {'QWEN_FAST_TP': '4', audit: '1'}, clear=True):
                with self.assertRaisesRegex(ValueError, 'compare nothing'):
                    fusion.audit_enabled(audit)
                with self.assertRaises(ValueError):
                    fusion.validate()
            with patch.dict(os.environ, {'QWEN_FAST_TP': '4', audit: '1', lever: '1'}, clear=True):
                self.assertTrue(fusion.audit_enabled(audit))
                fusion.validate()

    def test_off_the_dispatcher_is_the_served_function_call_for_call_and_the_module_is_not_imported(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}, clear=True), patch('mesh_link_policy.projection_links', return_value=1), \
                patch('feature_collective_tp.projection_links', return_value=1):
            saved = sys.modules.pop('draft_reduce_tp', None)
            try:
                logs = []
                for function in (collective.gather_add_projection, collective.served_gather_add_projection):
                    operations = fake.FakeOperations()
                    value = partials(operations, 32)
                    result = function(operations, mesh_for(), COLLECTIVES, value)
                    logs.append((operations.names(), result.shape))
                self.assertEqual(logs[0], logs[1])
                self.assertEqual(logs[0][0], ['all_gather'] + ['slice'] * 4 + ['add'] * 3 + ['deallocate'] * 7,
                                 'the gathered tensor, four slices and two partial sums are freed; the final add is returned')
                self.assertNotIn('draft_reduce_tp', sys.modules)
            finally:
                if saved is not None:
                    sys.modules['draft_reduce_tp'] = saved

    def test_on_the_feature_dispatcher_takes_the_launch(self):
        with patch.dict(os.environ, REDUCE_ON, clear=True), patch('mesh_link_policy.projection_links', return_value=1), \
                patch('feature_collective_tp.projection_links', return_value=1), patch.object(fusion, 'log_line'):
            operations = fresh()
            value = partials(operations, 32, 2, True)
            result = collective.gather_add_projection(operations, mesh_for(), COLLECTIVES, value)
            served_ops = fresh()
            served = collective.served_gather_add_projection(served_ops, mesh_for(), COLLECTIVES, served_ops.from_chips(
                [shard.data for shard in value.shards], 'fp32'))
            self.assertTrue(same(result, served))
            self.assertEqual(len(operations.launches), 1)

    def test_choose_returns_the_served_function_itself_when_off_or_without_a_quad(self):
        def served(*args, **options):
            return None
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}, clear=True):
            self.assertIs(reduce.choose(served, object(), 'mlp'), served)
        with patch.dict(os.environ, REDUCE_ON, clear=True):
            self.assertIs(reduce.choose(served, None, 'mlp'), served, 'the feature dispatcher reads the flag itself')
            self.assertIsNot(reduce.choose(served, object(), 'mlp'), served)


class AuditTests(EnvTestCase):
    environment = AUDIT_ON

    def test_an_exact_launch_logs_the_audit_line_and_frees_what_it_made(self):
        operations = fresh()
        mesh = mesh_for()
        before = len(operations.tensors)
        reduce.gather_add(operations, mesh, COLLECTIVES, partials(operations, 64, 1, True), served=served_chain, site='mlp', quad=True,
                          retain_temporaries=lambda tensor: tensor)
        audit = [line for line in self.lines if line.startswith(fusion.REDUCE_AUDIT_LINE)]
        self.assertEqual(len(audit), 1)
        self.assertRegex(audit[0], r'audit 1 exact=True site=mlp rows=64')
        made = operations.tensors[before:]
        references = [tensor for tensor in made if tensor.name in ('slice', 'add')]
        self.assertEqual(len(references), 7)
        self.assertTrue(all(tensor.freed for tensor in references), 'the audit frees the reference chain')
        self.assertFalse([line for line in self.lines if fusion.REDUCE_MISMATCH in line])

    def test_a_flipped_bit_is_a_logged_mismatch_and_an_assertion(self):
        operations = fresh()
        mesh = mesh_for()
        genuine = operations.emulators[0]

        def corrupt(ops, tensors, program):
            taken = genuine(ops, tensors, program)
            output = tensors[1]
            word = output.shards[2].data.view(torch.int32)
            word.reshape(-1)[12345] ^= 1
            return taken
        operations.emulators[0] = corrupt
        with self.assertRaises(AssertionError):
            reduce.gather_add(operations, mesh, COLLECTIVES, partials(operations, 64), served=served_chain, site='mlp', quad=True,
                              retain_temporaries=lambda tensor: tensor)
        mismatch = [line for line in self.lines if line.startswith(fusion.REDUCE_MISMATCH)]
        self.assertEqual(len(mismatch), 1)
        self.assertIn('chips=[2]', mismatch[0])

    def test_only_the_first_calls_of_each_site_and_row_count_are_audited(self):
        operations = fresh()
        mesh = mesh_for()
        for _ in range(fusion.AUDIT_CALLS + 5):
            reduce.gather_add(operations, mesh, COLLECTIVES, partials(operations, 64), served=served_chain, site='mlp', quad=True,
                              retain_temporaries=lambda tensor: tensor)
        reduce.gather_add(operations, mesh, COLLECTIVES, partials(operations, 64), served=served_chain, site='attention', quad=True,
                          retain_temporaries=lambda tensor: tensor)
        audits = [line for line in self.lines if line.startswith(fusion.REDUCE_AUDIT_LINE)]
        self.assertEqual(len(audits), fusion.AUDIT_CALLS + 1)

    def test_no_audit_inside_a_capture_scope(self):
        operations = fresh()
        mesh = mesh_for()
        with patch.object(fusion, 'capturing', return_value=True):
            reduce.gather_add(operations, mesh, COLLECTIVES, partials(operations, 64), served=served_chain, site='mlp', quad=True,
                              retain_temporaries=lambda tensor: tensor)
        self.assertEqual([line for line in self.lines if line.startswith(fusion.REDUCE_AUDIT_LINE)], [])


class SourceTests(unittest.TestCase):
    """The .cpp sources and the builder agree on the arguments. The transliteration in wp6_fake_device reads the launch the way these files do;
    this holds the files to the indices the builder fills."""

    @staticmethod
    def indices(name):
        text = (HERE / name).read_text()
        return sorted({int(index) for index in re.findall(r'get_arg_val<uint32_t>\((\d+)\)', text)})

    def test_each_kernel_reads_exactly_the_runtime_arguments_the_builder_writes(self):
        operations = fresh()
        mesh = mesh_for()
        value = partials(operations, 64)
        gathered = operations.all_gather_async(value, dim=0)
        with patch.dict(os.environ, REDUCE_ON, clear=True):
            reduce.reduce_launch(operations, mesh, gathered, 64, (11, 10), 4)
        _, program = operations.launches[0]
        chip_program = next(iter(program.values()))
        kernels = fake.kernels_by_name(chip_program)
        for name, kernel in kernels.items():
            words = kernel.runtime_args[0][0]
            self.assertEqual(self.indices(name), list(range(len(words))), name)
        self.assertEqual(kernels['draft_reduce_tp_io.cpp'].compile_time_args, [1, 0, 4])
        self.assertEqual(kernels['draft_fuse_out.cpp'].compile_time_args, [1, 0, 4096])
        self.assertEqual(kernels['draft_reduce_tp_compute.cpp'].compile_time_args, [4])
        self.assertIn('get_compile_time_arg_val(0)', (HERE / 'draft_reduce_tp_compute.cpp').read_text())
        self.assertIn('input_args.next_compile_time_args_offset()', (HERE / 'draft_reduce_tp_io.cpp').read_text())
        self.assertIn('output_args.next_compile_time_args_offset()', (HERE / 'draft_fuse_out.cpp').read_text())

    def test_the_compute_config_unpacks_the_gathered_pages_to_the_destination_as_fp32(self):
        operations = fresh()
        mesh = mesh_for()
        gathered = operations.all_gather_async(partials(operations, 32), dim=0)
        reduce.reduce_launch(operations, mesh, gathered, 32, (11, 10), 4)
        compute = fake.kernels_by_name(next(iter(operations.launches[0][1].values())))['draft_reduce_tp_compute.cpp']
        modes = compute.config.unpack_to_dest_mode
        self.assertEqual(len(modes), 64)
        self.assertEqual([index for index, mode in enumerate(modes) if mode == 'fp32'], [0])
        self.assertTrue(compute.config.fp32_dest_acc_en)
        self.assertFalse(compute.config.math_approx_mode)

    def test_the_kernels_use_the_primitives_of_the_served_ops(self):
        compute = (HERE / 'draft_reduce_tp_compute.cpp').read_text()
        for text in ('add_binary_tile_init();', 'add_binary_tile(0, chip, 0);', 'copy_tile(0, chip, chip);', 'pack_tile(0, 16);',
                     'cb_wait_front(0, chips);', 'cb_pop_front(0, chips);'):
            self.assertIn(text, compute)
        self.assertNotIn('typecast', compute)
        self.assertIn('eltwise_binary_sfpu.h', compute)

    def test_every_runtime_file_exists_and_is_listed(self):
        for name in fusion.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file(), name)
        self.assertEqual(set(reduce.RUNTIME_FILES) - {'draft_fusion_tp.py'}, {'draft_reduce_tp.py', reduce.IO_KERNEL, reduce.COMPUTE_KERNEL, reduce.OUT_KERNEL})


if __name__ == '__main__':
    unittest.main()
