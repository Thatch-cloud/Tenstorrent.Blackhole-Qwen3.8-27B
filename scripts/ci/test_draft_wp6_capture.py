"""The WP6 audits never synchronize, read back or issue an extra op while a trace is being captured.

The first card run of the fusion image (profile ...-fx-all-audit) died at engine start with `TT_FATAL !trace_id_.has_value(): Event Synchronization is not supported
during trace capture`: fused_commit_tp.capture warms its projection eagerly and then CAPTURES it, and the capture reaches feature_collective_tp.gather_add_projection ->
draft_reduce_tp.gather_add -> the audit's synchronize. The audits assumed the first calls of each site were the eager warm pass's; a segment's capture comes before the next
segment's warm, so they were not. The fake device here refuses synchronize_device, to_torch and from_torch exactly while a capture is open (begin_trace_capture ... end_trace_capture),
and every capture goes through attention_batch.capture_operation as the served ones do. Held:

  - draft_fusion_tp.track wraps begin_trace_capture / end_trace_capture once (also through an Overlay's `original`), the depth follows nesting, an exception in the captured
    function and an exception in end_trace_capture, and in_capture() says so;
  - every audit of every lever (reduce, SwiGLU, fused SwiGLU, residual, gate|up, matmul grid) called INSIDE a capture runs its lever's launch alone: no exception, nothing refused by the
    device (capture_attempts is empty), no extra tensor made, one 'skipped its audit inside a trace capture' line per (lever, site), the answer equal to the served one, and the audit
    budget untouched - the eager call after the capture is audited;
  - the whole MLP branch (single-user, quad) captured with all four levers and all four audits on, with and without an eager warm pass first: the same bits as flags off, nothing refused,
    audit lines exactly for the warm pass and skip lines for the capture;
  - a capture that began before track() ever ran (the first WP6 call is inside it) is caught by the audit's barrier: the device's refusal is a skip, any other failure of the
    synchronize is raised;
  - a sequence like the fused commit's (segment 0 warm, segment 0 capture, segment 1 warm, segment 1 capture, with the budget at its edge) audits only the warm calls.

    py -3.11 -B -m unittest test_draft_wp6_capture      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attention_batch  # noqa: E402
import draft_fusion_tp as fusion  # noqa: E402
import draft_gateup_tp as gateup  # noqa: E402
import draft_mlp  # noqa: E402
import draft_mmgrid_tp as mmgrid  # noqa: E402
import draft_reduce_tp as reduce  # noqa: E402
import draft_tail_tp as tail  # noqa: E402
import wp6_fake_device as fake  # noqa: E402
from test_draft_reduce_tp import COLLECTIVES, EnvTestCase, fresh, mesh_for, partials, same, served_chain  # noqa: E402
from test_draft_tail_tp import blocks, projections, same16  # noqa: E402
from test_draft_wp6_branch import FLAGS, Run, output_bits  # noqa: E402
from test_draft_wp6_branch import same as same_bits  # noqa: E402

ALL_AUDITED = {'QWEN_FAST_TP': '4', **{name: '1' for name in fusion.LEVERS}, **{audit: '1' for audit in fusion.AUDITS}}
SKIP = '[PINDIAG] tp4 draft %s skipped its audit inside a trace capture'


def keep(tensor):
    return tensor


def tracked(operations):
    """The realistic order: the levers' first call is eager and tracks the capture calls before any trace is captured (draft_mlp_branch does it at load
    when an audit flag is on)."""
    fusion.track(operations)
    return operations


def captured(operations, mesh, function):
    """The served way to capture: attention_batch.capture_operation (begin_trace_capture, the function, end_trace_capture)."""
    return attention_batch.capture_operation(operations, mesh, function)[1]


class TrackingTests(unittest.TestCase):
    def setUp(self):
        fusion.reset()
        self.addCleanup(fusion.reset)

    def test_track_wraps_once_and_follows_the_capture_depth(self):
        operations = fake.FakeOperations()
        self.assertFalse(fusion.in_capture(operations))
        wrapper = operations.begin_trace_capture
        fusion.track(operations)
        self.assertIs(operations.begin_trace_capture, wrapper, 'a second track() does not wrap again')
        self.assertTrue(getattr(operations.begin_trace_capture, 'tracked_by_wp6', False))
        seen = []
        captured(operations, None, lambda: seen.append(fusion.in_capture()))
        self.assertEqual(seen, [True])
        self.assertFalse(fusion.in_capture())
        self.assertTrue(operations.capture_open is False)

    def test_an_exception_in_the_captured_function_or_in_the_end_still_closes_the_depth(self):
        operations = fake.FakeOperations()
        fusion.track(operations)
        with self.assertRaises(KeyError):
            captured(operations, None, lambda: (_ for _ in ()).throw(KeyError('inside')))
        self.assertFalse(fusion.in_capture())
        genuine = operations.end_trace_capture

        def failing_end(*args, **options):
            genuine(*args, **options)
            raise RuntimeError('end failed')
        failing_end.tracked_by_wp6 = True
        operations.end_trace_capture = failing_end
        with self.assertRaises(RuntimeError):
            attention_batch.capture_operation(operations, None, lambda: None)
        self.assertFalse(fusion.in_capture())

    def test_a_failed_begin_does_not_open_a_depth(self):
        operations = fake.FakeOperations()
        fusion.track(operations)
        operations.capture_open = True
        with self.assertRaises(RuntimeError):
            operations.begin_trace_capture(None, cq_id=0)
        self.assertFalse(fusion.in_capture())

    def test_an_overlay_is_tracked_through_its_original(self):
        operations = fake.FakeOperations()
        overlay = attention_batch.Overlay(operations, marker=1)
        fusion.track(overlay)
        self.assertTrue(getattr(operations.begin_trace_capture, 'tracked_by_wp6', False))
        seen = []
        captured(operations, None, lambda: seen.append(fusion.in_capture(overlay)))
        self.assertEqual(seen, [True])

    def test_an_object_that_never_captures_is_left_alone(self):
        class Bare:
            pass
        bare = Bare()
        fusion.track(bare)
        self.assertFalse(hasattr(bare, 'begin_trace_capture'))
        self.assertFalse(fusion.in_capture(bare))

    def test_the_depth_nests(self):
        operations = fake.FakeOperations()
        fusion.track(operations)
        wrapped_begin, wrapped_end = operations.begin_trace_capture, operations.end_trace_capture
        wrapped_begin(None)
        operations.capture_open = False                    # (the fake allows one at a time; the tracker alone nests)
        wrapped_begin(None)
        self.assertTrue(fusion.in_capture())
        wrapped_end(None, 7)
        self.assertTrue(fusion.in_capture())
        wrapped_end(None, 7)
        self.assertFalse(fusion.in_capture())
        wrapped_end(None, 7)                               # an unmatched end never goes below zero
        self.assertFalse(fusion.in_capture())


class CaptureCase(EnvTestCase):
    environment = ALL_AUDITED

    def skip_lines(self, word):
        return [line for line in self.lines if line.startswith(SKIP % word)]

    def audit_lines(self, marker):
        return [line for line in self.lines if line.startswith(marker)]

    def untouched(self, operations):
        self.assertEqual(operations.capture_attempts, [], 'the device refused nothing: no synchronize, readback or upload inside the capture')


class ReduceInCaptureTests(CaptureCase):
    def test_the_audited_chain_inside_a_capture_runs_the_launch_alone_and_is_not_counted(self):
        operations = tracked(fresh())
        mesh = mesh_for()
        value = partials(operations, 32, 5, True)
        before = len(operations.tensors)
        result = captured(operations, mesh, lambda: reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='feature',
                                                                      quad=False, retain_temporaries=keep))
        self.untouched(operations)
        self.assertEqual(operations.names('slice', 'add'), [], 'no reference chain was issued into the capture')
        self.assertEqual(len([tensor for tensor in operations.tensors[before:] if tensor.name in ('slice', 'add')]), 0)
        self.assertEqual(len(self.skip_lines('reduce')), 1)
        self.assertEqual(self.audit_lines(fusion.REDUCE_AUDIT_LINE), [])
        self.assertEqual(fusion.audited_total(fusion.REDUCE), 0, 'a skipped audit spends none of the budget')
        served_ops = fresh()
        reference = served_chain(served_ops, mesh, COLLECTIVES, served_ops.from_chips([s.data for s in value.shards], 'fp32'),
                                 retain_temporaries=keep)
        self.assertTrue(same(result, reference))
        # the same call outside the capture is audited
        reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='feature', quad=False, retain_temporaries=keep)
        self.assertEqual(len(self.audit_lines(fusion.REDUCE_AUDIT_LINE)), 1)

    def test_the_fused_commit_order_audits_only_the_warm_calls_even_at_the_budget_edge(self):
        """segment 0 warm, segment 0 capture, segment 1 warm, segment 1 capture, ...: the capture of a segment comes before the next warm."""
        operations = fresh()
        mesh = mesh_for()
        value = partials(operations, 32, 6)
        for segment in range(fusion.AUDIT_CALLS + 3):
            reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='feature', quad=False, retain_temporaries=keep)
            captured(operations, mesh, lambda: reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='feature',
                                                                  quad=False, retain_temporaries=keep))
        self.untouched(operations)
        self.assertEqual(len(self.audit_lines(fusion.REDUCE_AUDIT_LINE)), fusion.AUDIT_CALLS, 'the budget went to the warm calls only')
        self.assertEqual(len(self.skip_lines('reduce')), 1, 'one skip line per site')

    def test_a_quad_chain_inside_a_capture(self):
        operations = tracked(fresh())
        mesh = mesh_for()
        value = partials(operations, 64, 7)
        captured(operations, mesh, lambda: reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='mlp', quad=True,
                                                             retain_temporaries=keep))
        self.untouched(operations)
        self.assertEqual(len(self.skip_lines('reduce')), 1)


class TailInCaptureTests(CaptureCase):
    def test_every_tail_launch_inside_a_capture_skips_its_audit(self):
        operations = tracked(fresh())
        mesh = mesh_for()
        gate, up = projections(operations, 64, 64)
        fused = operations.from_chips([torch.cat([g.data, u.data], dim=3) for g, u in zip(gate.shards, up.shards)], 'fp32')
        finished, hidden = blocks(operations, 64)
        outputs = captured(operations, mesh, lambda: (
            tail.swiglu(operations, mesh, gate, up, keep, served=draft_mlp.swiglu_device),
            tail.swiglu_fused(operations, mesh, fused, keep, served=draft_mlp.swiglu_device),
            tail.residual(operations, mesh, finished, hidden, keep)))
        self.untouched(operations)
        self.assertEqual(operations.names('silu', 'multiply', 'typecast', 'add', 'slice'), [], 'no served reference was issued into the capture')
        self.assertEqual(len(self.skip_lines('tail')), 2, 'swiglu and residual (the fused call shares the swiglu site)')
        self.assertEqual(self.audit_lines(fusion.TAIL_AUDIT_LINE), [])
        reference = draft_mlp.swiglu_device(operations, gate, up, keep)
        self.assertTrue(same16(outputs[0], reference))
        self.assertTrue(same16(outputs[1], reference))
        self.assertTrue(same16(outputs[2], tail.served_residual(operations, finished, hidden, keep)))
        tail.residual(operations, mesh, finished, hidden, keep)
        self.assertEqual(len(self.audit_lines(fusion.TAIL_AUDIT_LINE)), 1, 'the eager call after the capture is audited')


class GateUpAndGridInCaptureTests(CaptureCase):
    def test_the_fused_gate_up_audit_and_the_grid_audit_skip_inside_a_capture(self):
        from test_draft_gateup_tp import AuditTests as GateUpScene

        scene = GateUpScene('test_project_fused_runs_one_matmul_at_twice_the_columns_and_audits_it')
        operations, mesh, parameters, prepared, project = scene.scene()
        tracked(operations)
        uploads = len([tensor for tensor in operations.tensors if tensor.name == 'from_torch'])
        captured(operations, mesh, lambda: gateup.project_fused(operations, mesh, parameters, prepared, project, 64))
        self.untouched(operations)
        self.assertEqual(len([tensor for tensor in operations.tensors if tensor.name == 'from_torch']), uploads, 'no separate weights uploaded into a capture')
        self.assertEqual(len([entry for entry in operations.log if entry[0] == 'matmul']), 1, 'the fused matmul only: no served matmuls')
        self.assertEqual(len(self.skip_lines('gateup1')), 1)
        gateup.project_fused(operations, mesh, parameters, prepared, project, 64)
        self.assertEqual(len(self.audit_lines(fusion.GATEUP1_AUDIT_LINE)), 1)

    def test_the_matmul_grid_audit_inside_a_capture_issues_no_second_matmul(self):
        operations = tracked(fresh())
        mesh = mesh_for()
        generator = torch.Generator().manual_seed(1)
        value = operations.from_chips([torch.randn(1, 1, 64, 5120, generator=generator).to(torch.bfloat16) for _ in range(4)], 'bf16')
        weight = operations.from_chips([(torch.randn(5120, 256, generator=generator) * 0.05).to(torch.bfloat16) for _ in range(4)], 'bf16')
        grid = captured(operations, mesh, lambda: mmgrid.grid_for(operations, mesh, value, weight, (8, 10), 2, 64, None, site='test'))
        self.untouched(operations)
        self.assertEqual(operations.programs, [], 'neither the served-grid nor the wide matmul was issued by the audit')
        self.assertEqual(grid, mmgrid.wide_grid(mesh, mmgrid.cores_for((5120, 256), 2)))
        self.assertEqual(len(self.skip_lines('mmgrid')), 1)
        mmgrid.grid_for(operations, mesh, value, weight, (8, 10), 2, 64, None, site='test')
        self.assertEqual(len(self.audit_lines(fusion.MM_GRID_AUDIT_LINE)), 1)


class BranchInCaptureTests(unittest.TestCase):
    def run_branch(self, **options):
        with Run(FLAGS, FLAGS, **options) as run:
            return run.execute(), run

    def control_bits(self, rows, quad):
        with Run(rows=rows, quad=quad) as run:
            return output_bits(run.execute())

    def test_the_whole_branch_captured_with_every_lever_and_every_audit_on_never_touches_the_device(self):
        for rows, quad in ((32, False), (64, True)):
            with self.subTest(rows=rows):
                executed, run = self.run_branch(rows=rows, quad=quad, capture=True)
                self.assertEqual(executed.operations.capture_attempts, [])
                self.assertTrue(same_bits(output_bits(executed), self.control_bits(rows, quad)))
                self.assertEqual([line for line in run.lines if 'audit' in line and 'skipped' not in line], [])
                for word in ('reduce', 'tail', 'gateup1', 'mmgrid'):
                    self.assertTrue([line for line in run.lines if line.startswith(SKIP % word)], word)
                self.assertEqual([line for line in run.lines if 'mismatch' in line or 'fell back' in line], [])

    def test_a_warm_pass_then_the_capture_audits_the_warm_pass_and_skips_the_capture(self):
        executed, run = self.run_branch(rows=64, quad=True, capture=True, warm=True)
        self.assertEqual(executed.operations.capture_attempts, [])
        self.assertTrue(same_bits(output_bits(executed), self.control_bits(64, True)))
        warm_audits = [line for line in run.warm_lines if 'exact=True' in line]
        self.assertTrue(warm_audits)
        all_audits = [line for line in run.lines if 'exact=True' in line]
        self.assertEqual(len(all_audits), len(warm_audits), 'the capture adds no audit line')
        for marker in (fusion.REDUCE_AUDIT_LINE, fusion.TAIL_AUDIT_LINE, fusion.GATEUP1_AUDIT_LINE, fusion.MM_GRID_AUDIT_LINE):
            self.assertTrue([line for line in warm_audits if line.startswith(marker)], marker)
        self.assertEqual([line for line in run.warm_lines if 'skipped' in line], [], 'the warm pass is eager: nothing to skip')
        import draft_wp6_smoke

        env = {name: '1' for name in FLAGS}
        env.update({name + '_AUDIT': '1' for name in FLAGS})
        self.assertEqual(draft_wp6_smoke.problems(env, '\n'.join(run.lines)), [], 'the smoke judge reads the skip lines as neither a pass nor a failure')


class PrepareTracksTests(unittest.TestCase):
    def test_loading_the_mlp_parameters_with_an_audit_flag_on_wraps_the_capture_calls_before_any_trace(self):
        import draft_mlp_branch
        from test_draft_wp6_branch import model, mesh_for as branch_mesh
        from tp_test_support import four_cards

        for audits, expected in (({'QWEN_FAST_DRAFT_TAIL': '1', 'QWEN_FAST_DRAFT_TAIL_AUDIT': '1'}, True), ({'QWEN_FAST_DRAFT_TAIL': '1'}, False), ({}, False)):
            with patch.dict(os.environ, {'QWEN_FAST_TP': '4', **audits}, clear=True), four_cards():
                fusion.reset()
                operations = fake.FakeOperations(4)
                weights, convolution = model()
                draft_mlp_branch.prepare_mlp_branch(operations, branch_mesh(), weights, convolution, lambda value: value)
                self.assertEqual(getattr(operations.begin_trace_capture, 'tracked_by_wp6', False), expected, audits)
        fusion.reset()


class BarrierTests(CaptureCase):
    def test_a_capture_that_began_before_tracking_is_a_skip_through_the_barrier(self):
        operations = fresh()
        mesh = mesh_for()
        value = partials(operations, 32, 3)
        operations.capture_open = True                      # opened without our wrapper seeing it (the first WP6 call is inside the capture): depth 0
        self.assertFalse(fusion.in_capture())
        result = reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site='feature', quad=False, retain_temporaries=keep)
        self.assertEqual(operations.capture_attempts, ['synchronize_device'], 'exactly the one refused synchronize, then nothing')
        self.assertEqual(operations.names('slice', 'add'), [])
        self.assertEqual(len(self.skip_lines('reduce')), 1)
        self.assertEqual(result.shape, (1, 1, 32, 5120))
        self.assertEqual(fusion.audited_total(fusion.REDUCE), 0)

    def test_any_other_failure_of_the_synchronize_is_raised(self):
        operations = fresh()
        mesh = mesh_for()

        def broken(_mesh):
            raise RuntimeError('device lost')
        operations.synchronize_device = broken
        with self.assertRaisesRegex(RuntimeError, 'device lost'):
            fusion.audit_begin(operations, mesh, fusion.REDUCE, 'feature', 32)
        self.assertEqual(fusion.audited_total(fusion.REDUCE), 0)

    def test_a_clean_barrier_counts_the_call_and_stops_at_the_budget(self):
        operations = fresh()
        mesh = mesh_for()
        answers = [fusion.audit_begin(operations, mesh, fusion.TAIL, 'mlp.swiglu', 64) for _ in range(fusion.AUDIT_CALLS + 2)]
        self.assertEqual(answers, [True] * fusion.AUDIT_CALLS + [False] * 2)
        self.assertEqual(operations.sync_count, fusion.AUDIT_CALLS, 'the barrier ran for the audited calls only')

    def test_the_audit_flag_off_never_wraps_or_synchronizes(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_REDUCE': '1'}, clear=True):
            operations = fresh()
            reduce.gather_add(operations, mesh_for(), COLLECTIVES, partials(operations, 32), served=served_chain, site='feature', quad=False,
                              retain_temporaries=keep)
            self.assertFalse(getattr(operations.begin_trace_capture, 'tracked_by_wp6', False))
            self.assertEqual(operations.sync_count, 0)


class SourceTests(unittest.TestCase):
    def test_no_wp6_module_synchronizes_or_reads_back_outside_the_audit_and_the_served_eager_paths(self):
        """The only synchronize_device / to_torch in the lever modules sit in the audit helpers (after audit_begin) or in the served-eager branch
        (retain_temporaries is None), which the served function has too."""
        for name in ('draft_reduce_tp.py', 'draft_tail_tp.py', 'draft_gateup_tp.py', 'draft_mmgrid_tp.py'):
            text = (HERE / name).read_text()
            for marker in ('synchronize_device', 'to_torch', 'chip_bits', 'differing_chips'):
                for number, line in enumerate(text.splitlines(), 1):
                    if marker in line and not line.lstrip().startswith(('#', '"""', "'")):
                        context = text.splitlines()[max(0, number - 25):number]
                        self.assertTrue(any(('def _audit' in item or 'def audit(' in item or 'def _audited' in item
                                             or 'retain_temporaries is None' in item or 'import torch' in item) for item in context),
                                        '%s:%d %s' % (name, number, line.strip()))


if __name__ == '__main__':
    unittest.main()
