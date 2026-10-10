"""The trace-capture gate of the op-fusion programme (capture_rules.py): the rules themselves, and every lever's audit entry point driven through them.

Two audits broke a capture rule of the device on a card (one synchronized inside the fused-commit capture; one launched an audit program that had never run eagerly,
'Cannot load new binaries during trace capture'). The device's rules are: nothing that reads or writes the host while a capture is open, and nothing in a capture that
did not run eagerly before it. This module holds (1) the model of those rules with negative controls, and (2) per lever the lifecycle the stack gives it - warm forward
(eager), capture, replays, the audit after a replay (eager), release - run behind the rules with the lever's audit on. A lever whose audit fails the gate is named by
its test; a known failure is marked expectedFailure with the finding, so the suite goes red the day it is fixed and the marker has to go."""

import contextlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_rules  # noqa: E402
from capture_rules import CaptureViolation, NEW_BINARIES, SYNCHRONIZATION  # noqa: E402


class Tensor:
    def __init__(self, shape, dtype='bf16', layout='tile'):
        self.shape, self.dtype, self.layout = tuple(shape), dtype, layout
        self.shards = [object()]

    def memory_config(self):
        return 'dram'


class TinyOps:
    """The smallest ttnn the rules need: ops, a generic_op, the capture calls, the readbacks, constants and descriptor classes."""

    bfloat16 = 'bf16'

    def __init__(self):
        self.experimental = SimpleNamespace(all_gather_async=self.all_gather_async)
        self.traces = 0
        self.executed = []

    class KernelDescriptor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def matmul(self, a, b, program_config=None, **options):
        return Tensor((a.shape[0], b.shape[1]))

    def add(self, a, b, **options):
        return Tensor(a.shape)

    def all_gather_async(self, tensor, dim=0, **options):
        return Tensor(tensor.shape)

    def generic_op(self, tensors, program):
        return None

    def to_torch(self, tensor):
        return [0]

    def from_torch(self, value, device=None, **options):
        return Tensor((1,))

    def synchronize_device(self, mesh):
        return None

    def deallocate(self, tensor):
        return None

    def get_device_tensors(self, tensor):
        return tensor.shards

    def begin_trace_capture(self, mesh, cq_id=0):
        self.traces += 1
        return self.traces

    def end_trace_capture(self, mesh, trace, cq_id=0):
        return None

    def execute_trace(self, mesh, trace, **options):
        self.executed.append(trace)


def program(source='void kernel_main() {}', args=(1, 2), defines=None, runtime=((1, 1),), buffers=2):
    kernel = TinyOps.KernelDescriptor(kernel_source=source, compile_time_args=list(args), defines=defines or {}, runtime_args=list(runtime))
    return {('range', 0): SimpleNamespace(kernels=[kernel], cbs=[SimpleNamespace(total_size=4096 * buffers)], semaphores=[])}


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.ops = capture_rules.guard(TinyOps())
        self.rules = self.ops.rules
        self.a, self.b = Tensor((32, 64)), Tensor((64, 32))

    def capture(self, body):
        trace = self.ops.begin_trace_capture('mesh')
        try:
            return body()
        finally:
            self.ops.end_trace_capture('mesh', trace)

    def test_an_op_that_ran_eagerly_may_be_captured_and_replayed(self):
        self.ops.matmul(self.a, self.b, program_config='cfg')
        self.capture(lambda: self.ops.matmul(self.a, self.b, program_config='cfg'))
        self.ops.execute_trace('mesh', 1)
        self.rules.assert_clean()
        self.assertEqual((self.rules.eager_launches, self.rules.captured_launches, self.rules.captures), (1, 1, 1))

    def test_an_op_never_run_eagerly_cannot_load_its_binary_in_a_capture(self):
        with self.assertRaises(CaptureViolation) as caught:
            self.capture(lambda: self.ops.matmul(self.a, self.b))
        self.assertIn(NEW_BINARIES, str(caught.exception))
        self.assertEqual(len(self.rules.violations), 1)
        with self.assertRaises(AssertionError):
            self.rules.assert_clean()

    def test_the_key_is_the_ops_shapes_and_configs_not_its_values(self):
        self.ops.matmul(self.a, self.b, program_config='cfg')
        for body in (lambda: self.ops.matmul(Tensor((64, 64)), self.b, program_config='cfg'),         # another shape
                     lambda: self.ops.matmul(Tensor((32, 64), dtype='bf8'), self.b, program_config='cfg'),   # another dtype
                     lambda: self.ops.matmul(self.a, self.b, program_config='other'),                  # another config
                     lambda: self.ops.add(self.a, self.a)):                                           # another op
            with self.assertRaises(CaptureViolation):
                self.capture(body)
        self.capture(lambda: self.ops.matmul(Tensor((32, 64)), Tensor((64, 32)), program_config='cfg'))          # other tensors, same specs: fine
        self.assertEqual(len(self.rules.violations), 4)

    def test_a_generic_op_is_keyed_by_its_kernels_and_ignores_runtime_arguments(self):
        io = [self.a]
        self.ops.generic_op(io, program())
        self.capture(lambda: self.ops.generic_op(io, program(runtime=((9, 9), (8, 8)))))                    # runtime arguments (addresses) are overridden in a cached program
        self.rules.assert_clean()
        for changed in (program(source='void kernel_main() { /* audit */ }'), program(args=(1, 3)), program(defines={'NEGATIVE': '1'}), program(buffers=3)):
            with self.assertRaises(CaptureViolation):
                self.capture(lambda: self.ops.generic_op(io, changed))
        self.assertEqual(len(self.rules.violations), 4)

    def test_a_host_round_trip_inside_a_capture_is_refused_and_outside_it_is_not(self):
        self.ops.synchronize_device('mesh')
        self.ops.to_torch(self.a)
        self.ops.from_torch([0], device='mesh')
        for name, body in (('synchronize_device', lambda: self.ops.synchronize_device('mesh')), ('to_torch', lambda: self.ops.to_torch(self.a)),
                           ('from_torch', lambda: self.ops.from_torch([0], device='mesh'))):
            with self.assertRaises(CaptureViolation) as caught:
                self.capture(body)
            self.assertIn(SYNCHRONIZATION, str(caught.exception), name)
        self.assertEqual(self.rules.attempts, ['synchronize_device', 'to_torch', 'from_torch'])
        self.ops.synchronize_device('mesh')                                          # the capture ended even though its body raised
        self.assertFalse(self.rules.open)

    def test_a_violation_that_the_lever_catches_is_still_recorded(self):
        def swallowed():
            try:
                self.ops.synchronize_device('mesh')
            except RuntimeError:
                return 'skipped'
        self.assertEqual(self.capture(swallowed), 'skipped')
        self.assertEqual(self.rules.violations, ['synchronize_device inside a trace capture'])

    def test_constants_descriptor_classes_namespaces_and_frees_pass_through(self):
        self.assertEqual(self.ops.bfloat16, 'bf16')
        descriptor = self.ops.KernelDescriptor(kernel_source='x')
        self.assertEqual(descriptor.kernel_source, 'x')
        self.ops.experimental.all_gather_async(self.a, dim=3)
        self.capture(lambda: (self.ops.experimental.all_gather_async(self.a, dim=3), self.ops.deallocate(self.a), self.ops.get_device_tensors(self.a)))
        self.rules.assert_clean()
        with self.assertRaises(CaptureViolation):
            self.capture(lambda: self.ops.experimental.all_gather_async(self.a, dim=2))

    def test_the_rules_are_shared_between_wrapped_objects(self):
        other = capture_rules.guard(TinyOps(), self.rules)
        self.ops.matmul(self.a, self.b)
        trace = self.ops.begin_trace_capture('mesh')
        try:
            other.matmul(self.a, self.b)                                              # the warm set is the process's, not one object's
            with self.assertRaises(CaptureViolation):
                other.synchronize_device('mesh')                                      # and so is the open capture
        finally:
            self.ops.end_trace_capture('mesh', trace)
        self.assertEqual(len(self.rules.violations), 1)

    def test_a_second_open_capture_is_refused(self):
        trace = self.ops.begin_trace_capture('mesh')
        try:
            with self.assertRaises(CaptureViolation):
                self.ops.begin_trace_capture('mesh')
        finally:
            self.ops.end_trace_capture('mesh', trace)


class Lifecycle:
    """What a scenario returns: the rules (violations, warm and captured keys) and the lines the lever logged."""

    def __init__(self, rules, lines=(), **facts):
        self.rules, self.lines = rules, list(lines)
        self.__dict__.update(facts)

    def clean(self):
        return self.rules.problems()


# ---------------------------------------------------------------------------------------------------------------------------------
# F-B1, the K/V page writer (WP2): ChainedOrderedCacheWriter builds a page writer per forward. The WARM forward's block has one span (placeholders on one tile
# row): the ordered mode, which carries no audit. The CAPTURED forward's block has the four users' spans: the audited writer, whose prep, served-write-on-the-
# shadow and check programs run in the capture and are replayed with it. audit_round reads the counters after a replay.
# ---------------------------------------------------------------------------------------------------------------------------------

def kvpage_lifecycle(audit, layers=2, source='dram', replays=3):
    import kv_page_writer_tp4 as kvpw
    import packed_ordered_cache as poc
    from test_kv_page_writer_tp4 import FakeTTNN, KERNELS, ON, SEGMENTS, WIDTH, block, mesh, tiles

    class Capturing(FakeTTNN):
        def __init__(self, chips=1):
            FakeTTNN.__init__(self, chips)
            self.next_trace, self.replayed = 0, 0

        def begin_trace_capture(self, device, cq_id=0):
            self.next_trace += 1
            return self.next_trace

        def end_trace_capture(self, device, trace, cq_id=0):
            return None

        def execute_trace(self, device, trace, cq_id=0, blocking=False):
            self.replayed += 1

        def synchronize_device(self, device):
            return None

    environ = dict(ON, **{'QWEN_FAST_KV_PAGE_WRITER_AUDIT': '1'} if audit else {})
    lines = []
    with contextlib.ExitStack() as stack:
        for patch in (mock.patch.dict(os.environ, environ), mock.patch.object(kvpw, 'log_line', lines.append),
                      mock.patch.object(kvpw, 'evidence_problems', return_value=[]), mock.patch.object(kvpw, '_NOTED', set()),
                      mock.patch.object(kvpw, '_REGISTRY', [])):
            stack.enter_context(patch)
        raw = Capturing(chips=1)
        operations = capture_rules.guard(raw)
        device = mesh(chips=1, grid=(13, 10))
        tensors = block(raw, source=source)
        arguments = dict(positions=tensors['positions'], pages=tensors['pages'], tiles=tiles(raw, WIDTH), launch_rows=64)
        call = dict(update_idxs_tensor=SimpleNamespace(shape=(64,)), page_table=SimpleNamespace(shape=(64, WIDTH)))
        # both forwards' writers exist before the capture (the blocks are built at attach)
        warm = poc.ChainedOrderedCacheWriter(device, operations, KERNELS, spans=((0, 64),), **arguments)
        captured = poc.ChainedOrderedCacheWriter(device, operations, KERNELS, spans=SEGMENTS, **arguments)

        def forward(writer):
            for unused in range(layers):
                for cache, packed in zip(tensors['caches'], tensors['packed']):
                    writer(cache, packed, **call)

        forward(warm)
        escaped = None
        trace = operations.begin_trace_capture(device)
        try:
            forward(captured)
        except CaptureViolation as error:
            escaped = error                         # the device would have stopped the engine here; the rules recorded it
        finally:
            operations.end_trace_capture(device, trace)
        units = kvpw.unit_count(captured.page.wt)
        counters = SimpleNamespace(tolist=lambda: [[0, 1 if unit % 2 == 0 else 0, 7, 0] + [0] * 12 for unit in range(units)])
        if escaped is None:
            with mock.patch.object(raw, 'to_torch', lambda shard: counters):
                for number in range(replays):
                    operations.execute_trace(device, trace)
                    if audit:
                        kvpw.audit_round(operations, number)
    return Lifecycle(operations.rules, lines, page=captured.page, replayed=raw.replayed, escaped=escaped)


class KVPageTests(unittest.TestCase):
    def test_the_timed_arm_launches_one_program_in_the_capture_that_the_warm_forward_ran(self):
        found = kvpage_lifecycle(audit=False)
        self.assertEqual(found.clean(), [])
        self.assertEqual(found.replayed, 3)
        self.assertTrue(any(' ordered=0 ' in line for line in found.lines))

    def test_the_timed_arm_with_the_prepared_k_v_in_l1_is_the_same(self):
        self.assertEqual(kvpage_lifecycle(audit=False, source='l1').clean(), [])

    def test_the_audited_arm_launches_nothing_in_the_capture_that_did_not_run_eagerly(self):
        """The first card run of the audit twin (run 38055792581, tp4-fusion-2) died at engine start in audit_before: the audit's prep, its served write on the shadow cache and its
        check ran only in the captured (non-ordered) writer, so none had run eagerly and the capture could not load their binaries ('Cannot load new binaries during trace capture').
        This gate reproduced that failure on the head of tp4/fx-wp2 before ea047a63, whose warm (ordered) writer now runs the whole audit sequence eagerly (Audit.warm)."""
        for source in ('dram', 'l1'):
            with self.subTest(source=source):
                found = kvpage_lifecycle(audit=True, source=source)
                self.assertEqual(found.clean(), [])
                self.assertIsNone(found.escaped)
                self.assertEqual(found.replayed, 3)
                self.assertTrue(found.rules.captured_launches)

    def test_the_audit_state_of_the_warm_writer_is_shared_and_never_read_back(self):
        import kv_page_writer_tp4 as kvpw

        found = kvpage_lifecycle(audit=True)
        self.assertEqual(len(found.rules.captured), 1)
        self.assertEqual(found.page.audit.rounds, 2, 'the captured writer counted its launches (two layers)')
        self.assertTrue(hasattr(kvpw.Audit, 'warm'))


# ---------------------------------------------------------------------------------------------------------------------------------
# WP6 (QWEN_FAST_DRAFT_REDUCE, _TAIL, _GATEUP1, _MM_GRID with their audits): the drafter's MLP branch, single-user (32 rows) and quad (64 rows). The drafter runs inside
# captures of every kind (a bucket's proposal, the fused commit's projection); the fused commit warms a segment eagerly and then captures it, segment by segment.
# ---------------------------------------------------------------------------------------------------------------------------------

def wp6_branch_lifecycle(rows, quad, segments=2, audited=True):
    import attention_batch
    import draft_mlp_branch
    import quad_draft_tp
    import torch
    import wp6_fake_device as fake
    from test_draft_wp6_branch import COLLECTIVES, FLAGS, Run, convolve, mesh_for, model
    from tp_test_support import four_cards

    run = Run(FLAGS, audits=FLAGS if audited else (), rows=rows, quad=quad)
    with run, four_cards():
        raw = fake.install_emulators(fake.FakeOperations(4), (13, 10))
        operations = capture_rules.guard(raw)
        mesh = mesh_for(13, 10)
        weights, convolution = model(0)
        kept = []
        retain = lambda value: kept.append(value) or value          # noqa: E731
        parameters = draft_mlp_branch.prepare_mlp_branch(operations, mesh, weights, convolution, lambda value: value)
        generator = torch.Generator().manual_seed(1)
        hidden = raw.from_chips([torch.randn(1, 1, rows, 5120, generator=generator).to(torch.bfloat16)] * 4, 'bf16')
        extra = {}
        if quad:
            extra = dict(quad=quad_draft_tp.QuadPass(), boundaries=tuple((start, start + 16) for start in range(0, rows, 16)))

        def call():
            return draft_mlp_branch.execute_mlp_branch(operations, mesh, COLLECTIVES, hidden, weights, convolution, retain, parameters=parameters, trace_safe=True,
                                                       convolution_operation=convolve, **extra)

        escaped = None
        for unused in range(segments):
            call()                                                  # the segment's eager warm pass
            try:
                attention_batch.capture_operation(operations, mesh, call)
            except CaptureViolation as error:
                escaped = error
                break
    return Lifecycle(operations.rules, run.lines, escaped=escaped, audit_lines=[line for line in run.lines if ' audit ' in line and 'exact=True' in line],
                     skipped=[line for line in run.lines if 'skipped its audit' in line])


class WP6BranchTests(unittest.TestCase):
    def test_the_single_user_branch_with_all_four_levers_and_audits_passes_the_capture_rules(self):
        found = wp6_branch_lifecycle(32, quad=False)
        self.assertEqual(found.clean(), [])
        self.assertIsNone(found.escaped)
        self.assertTrue(found.audit_lines, 'the eager warm pass audited the levers')
        self.assertTrue(found.rules.captured_launches, 'something was captured')

    def test_the_quad_branch_with_all_four_levers_and_audits_passes_the_capture_rules(self):
        found = wp6_branch_lifecycle(64, quad=True)
        self.assertEqual(found.clean(), [])
        self.assertIsNone(found.escaped)
        self.assertTrue(found.audit_lines)

    def test_the_audits_off_arms_pass_too(self):
        for rows, quad in ((32, False), (64, True)):
            found = wp6_branch_lifecycle(rows, quad, audited=False)
            self.assertEqual(found.clean(), [], (rows, quad))
            self.assertEqual(found.audit_lines, [])


def add_capture_calls(namespace):
    """The capture calls a fake ttnn namespace may lack, so that it can be put behind the rules."""
    state = SimpleNamespace(traces=0, replayed=0)

    def begin(device, cq_id=0):
        state.traces += 1
        return state.traces

    def execute(device, trace, cq_id=0, blocking=False):
        state.replayed += 1

    for name, function in (('begin_trace_capture', begin), ('end_trace_capture', lambda device, trace, cq_id=0: None), ('execute_trace', execute),
                           ('release_trace', lambda device, trace: None), ('synchronize_device', lambda device: None)):
        if not hasattr(namespace, name):
            setattr(namespace, name, function)
    return state


# ---------------------------------------------------------------------------------------------------------------------------------
# WP4 (QWEN_FAST_MLP_CFG with QWEN_FAST_MLP_CFG_AUDIT): the real grafted MLP over the lever's binder, three layers. packed_verifier's order: the warm forward (eager), its
# claim / compare / release, the capture, the capture's claim, then per replay audit_replayed and audit_round, and the release when the fixture closes.
# ---------------------------------------------------------------------------------------------------------------------------------

def mlp_lifecycle(name, audited=True, layers=3, replays=3):
    import tp4_mlp_gateup as lever
    from model_batch import instance_overrides
    from test_tp4_mlp_gateup import Fake, activation, env, model_of

    fake = Fake()
    state = add_capture_calls(fake.ttnn)
    operations = capture_rules.guard(fake.ttnn)
    model = model_of(fake, layers=layers)
    values = {'QWEN_FAST_MLP_CFG': name}
    if audited:
        values['QWEN_FAST_MLP_CFG_AUDIT'] = '1'
    lines = []
    lever.forget_logged()
    escaped = None
    with env(**values), fake.modules(), mock.patch.object(lever, 'log_line', lines.append):
        binder = lever.bindings(model, 64, operations)[0]

        def forward():
            for layer in model.layers:
                layer.feed_forward.forward(activation())

        with instance_overrides(binder.bindings):
            warm = object()
            try:
                forward()
                lever.audit_claim(warm, 'warm')
                lever.audit_round(operations, warm, 0)
            finally:
                lever.audit_claim(warm, 'warm')
                lever.audit_release(operations, warm)
            fixture = object()
            trace = operations.begin_trace_capture('mesh')
            try:
                forward()
            except CaptureViolation as error:
                escaped = error
            finally:
                operations.end_trace_capture('mesh', trace)
                lever.audit_claim(fixture, 'capture')
            if escaped is None:
                for number in range(1, replays + 1):
                    operations.execute_trace('mesh', trace)
                    lever.audit_replayed(fixture)
                    lever.audit_round(operations, fixture, number)
            lever.audit_release(operations, fixture)
    return Lifecycle(operations.rules, lines, escaped=escaped, replayed=state.replayed, held=len(lever._HELD),
                     audit_lines=[line for line in lines if 'audit' in line and 'exact=True' in line])


class MLPTests(unittest.TestCase):
    def test_the_audited_cfg_arm_runs_its_clones_and_comparisons_outside_the_capture(self):
        for name in ('l1', 'g3u4d3', 'g3u3d4'):
            with self.subTest(config=name):
                found = mlp_lifecycle(name)
                self.assertEqual(found.clean(), [])
                self.assertIsNone(found.escaped)
                self.assertEqual(found.replayed, 3)
                self.assertTrue(found.rules.captured_launches, 'the capture launched the lever')
                self.assertEqual(found.held, 0, 'everything the audit held was released')
                self.assertTrue(found.audit_lines, 'an exact=True audit line was logged')

    def test_the_timed_cfg_arm_is_the_same_programs_without_the_audit(self):
        found = mlp_lifecycle('g3u4d3', audited=False)
        self.assertEqual(found.clean(), [])
        self.assertEqual(found.audit_lines, [])


# ---------------------------------------------------------------------------------------------------------------------------------
# WP1 (QWEN_FAST_TP4_SHARD_ARGMAX, _FOLD2, _AUDIT): the sampler's scan and fold launches run INSIDE the verify capture; the scratch is reserved before any capture (at the
# block's warm); the audit holds today's (ids, values) beside the kernel's, produced in the same capture by `served()`, and compares them AFTER a replay.
# ---------------------------------------------------------------------------------------------------------------------------------

def s1_lifecycle(audited=True, fold2=True, rows=64, replays=3):
    import torch
    import tp4_sampdraft
    import tp4_shard_argmax as sarg
    from test_tp4_shard_argmax import FakeOperations, logits_for, mesh_with_grid

    raw = FakeOperations()
    state = add_capture_calls(raw)
    raw.to_torch = lambda part: part.data
    raw.reference_gather = lambda tensor, **options: raw.tensor((1, 1, 1, 64), 'u32', 'row_major')       # today's sampler path, as an op the rules key
    operations = capture_rules.guard(raw)
    mesh = mesh_with_grid(13, 10)
    environ = {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'}
    if audited:
        environ[tp4_sampdraft.SHARD_ARGMAX_AUDIT] = '1'
    if fold2:
        environ[sarg.FOLD2_FLAG] = '1'
    lines, escaped, produced = [], None, {}
    for table in (sarg.REFERENCES, sarg.PRODUCED, sarg.WORDS, sarg._RESERVED):
        table.clear()
    sarg._STATE['rounds'] = 0
    tp4_sampdraft._LOGGED.clear()
    with mock.patch.dict(os.environ, environ), mock.patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
        sarg.reserve(operations, mesh)                                        # the block's warm, before any trace
        logits = logits_for(raw, rows)

        def served():
            ids = operations.reference_gather(logits)
            values = operations.reference_gather(logits)
            return ids, values

        def sample():
            return sarg.sample(operations, logits, rows, served=served if audited else None)

        try:
            warm_ids, warm_values = sample()                                  # the eager warm forward
            sarg.release_audit(operations, warm_values)
            trace = operations.begin_trace_capture(mesh)
            try:
                ids, values = sample()                                        # the captured forward
            except CaptureViolation as error:
                escaped = error
            finally:
                operations.end_trace_capture(mesh, trace)
            if escaped is None:
                reference = sarg.REFERENCES.get(id(values))
                for number in range(replays):
                    operations.execute_trace(mesh, trace)
                    if audited:
                        # the readback of the replayed outputs (eager) and the comparison with today's path
                        wanted_ids = torch.arange(rows, dtype=torch.int32)
                        wanted_values = torch.ones(rows, dtype=torch.float32)
                        for part in operations.get_device_tensors(reference[0]):
                            part.data = wanted_ids
                        for part in operations.get_device_tensors(reference[1]):
                            part.data = wanted_values
                        packed = sarg.pack_words(wanted_ids, wanted_values)
                        wire = packed.where(packed < (1 << 31), packed - (1 << 32)).to(torch.int32)
                        for part in operations.get_device_tensors(sarg.WORDS[id(values)]):
                            part.data = wire
                        sarg.audit_round(operations, values, rows, [wanted_ids] * 4, [wanted_values] * 4)
                sarg.release_audit(operations, values)
        finally:
            sarg.release_reserved(operations, mesh)
    return Lifecycle(operations.rules, lines, escaped=escaped, replayed=state.replayed, audit_lines=[line for line in lines if 'exact=True' in line])


class ShardArgmaxTests(unittest.TestCase):
    def test_the_audited_sampler_with_the_tree_fold_launches_only_what_the_warm_forward_ran(self):
        for fold2 in (False, True):
            with self.subTest(fold2=fold2):
                found = s1_lifecycle(audited=True, fold2=fold2)
                self.assertEqual(found.clean(), [])
                self.assertIsNone(found.escaped)
                self.assertEqual(found.replayed, 3)
                self.assertTrue(found.audit_lines, 'an exact=True line was logged')
                self.assertTrue(found.rules.captured_launches)

    def test_the_timed_sampler_is_the_same_launches(self):
        found = s1_lifecycle(audited=False)
        self.assertEqual(found.clean(), [])
        self.assertEqual(found.audit_lines, [])


# ---------------------------------------------------------------------------------------------------------------------------------
# WP5 (QWEN_FAST_CCL_OPTIONS and _AUDIT): the unit-major reduce-scatter's per-call options and the norms' gather. The audit runs the first calls of each op twice (the
# model's options and the set's) inside every block scope, holds both results and compares them after a replay.
# ---------------------------------------------------------------------------------------------------------------------------------

def ccl_lifecycle(text, audit=2, calls=4, replays=3, gather=False):
    import ccl_options_tp as options
    import tile_collective_tp as collective
    from test_ccl_options_tp import TheGatherShim, partials

    fixture = TheGatherShim('test_outside_a_scope_the_wrapper_is_the_original')
    fixture.setUp()
    try:
        raw = fixture.operations
        state = add_capture_calls(raw)
        operations = capture_rules.guard(raw)
        fixture.wrapper = collective.TileSplitAllReduce(fixture.model, operations)
        for name in ('experimental', 'to_memory_config', 'clone', 'deallocate', 'get_device_tensors', 'to_torch'):
            setattr(fixture.ttnn, name, getattr(operations, name))

        def forward():
            with fixture.scope(text, audit=audit):
                for index in range(calls):
                    fixture.call(partials(64, seed=index))
                    if gather:
                        fixture.forward()

        warm, owner, escaped = object(), object(), None
        try:
            forward()
            collective.audit_claim(warm, 'warm')
            collective.audit_round(operations, warm, 0, log=fixture.lines.append)
        finally:
            collective.audit_claim(warm, 'warm')
            collective.audit_release(operations, warm)
        trace = operations.begin_trace_capture('mesh')
        try:
            forward()
        except CaptureViolation as error:
            escaped = error
        finally:
            operations.end_trace_capture('mesh', trace)
            collective.audit_claim(owner, 'capture')
        if escaped is None:
            for number in range(1, replays + 1):
                operations.execute_trace('mesh', trace)
                collective.audit_replayed(owner)
                collective.audit_round(operations, owner, number, log=fixture.lines.append)
        collective.audit_release(operations, owner)
        lines = list(fixture.lines)
        found = Lifecycle(operations.rules, lines, escaped=escaped, replayed=state.replayed, held=len(collective._HELD),
                          audit_lines=[line for line in lines if line.startswith(options.AUDIT_MARKER) and line.endswith('exact=True')])
    finally:
        fixture.doCleanups()
    return found


class CCLOptionsTests(unittest.TestCase):
    def test_the_audited_reduce_scatter_sets_run_their_twin_calls_in_the_capture_as_in_the_warm_forward(self):
        for text in ('rs-c1', 'served', 'rs-c1+rs-w1'):
            with self.subTest(set=text):
                found = ccl_lifecycle(text)
                self.assertEqual(found.clean(), [])
                self.assertIsNone(found.escaped)
                self.assertEqual(found.replayed, 3)
                self.assertEqual(found.held, 0)
                self.assertTrue(found.audit_lines, found.lines[-3:])

    def test_the_gather_set_with_the_norms_shim_is_the_same(self):
        found = ccl_lifecycle('ag-c1+ag-w1+rs-c1+rs-w1', gather=True)
        self.assertEqual(found.clean(), [])
        self.assertIsNone(found.escaped)
        self.assertTrue(found.audit_lines)

    def test_the_timed_arm_without_the_audit_is_the_same_programs(self):
        found = ccl_lifecycle('rs-c1', audit=0)
        self.assertEqual(found.clean(), [])
        self.assertEqual(found.audit_lines, [])


# ---------------------------------------------------------------------------------------------------------------------------------
# WP7 (QWEN_FAST_DRAFT_PERMUTE, _QKV1, _HEAD64 with their audits). The protocol is quad_draft_tp's _execute: draft_permute_tp.set_pass(True) for the bucket's eager warm pass (the
# served composition runs beside each launch and every pair is byte-compared: a readback), set_pass(False) for the capture (the launches only; "a capture cannot read a tensor back"),
# unmarked everywhere else (no audit). The capture's launches must be the warm pass's.
# ---------------------------------------------------------------------------------------------------------------------------------

def permute_lifecycle(kind, case, replays=3):
    import torch
    import draft_permute_tp as perm
    import tp4_sampdraft
    from test_draft_permute_tp import GEOMETRY, HEADS, DIM, ExecutingOperations, bits, fold_served, kv_operands, served_kv, unfold_served
    from test_pair_row_exact import keep
    from tp_test_support import four_cards

    raw = ExecutingOperations()
    state = add_capture_calls(raw)
    operations = capture_rules.guard(raw)
    lines, escaped = [], None
    environ = {'QWEN_FAST_TP': '4', perm.FLAG: '1', perm.AUDIT_FLAG: '1'}
    with mock.patch.dict(os.environ, environ), four_cards(), mock.patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
        perm._LOGGED.clear()
        perm._CACHE.clear()
        perm._SERVED['depth'] = 0
        owned = []
        if kind == 'kv':
            plan, caches, live = kv_operands(case, seed=3)
            device_caches = [{name: raw.from_logical(value) for name, value in cache.items()} for cache in caches]
            device_live = {name: raw.from_logical(value) for name, value in live.items()}
            reference = served_kv(plan, caches, live)

            def run(served):
                return perm.assemble_kv(operations, plan, device_caches, device_live, keep(owned), served=served, site=case)

            warm_served = lambda name: raw.from_logical(reference[name])            # noqa: E731
        else:
            generator = torch.Generator().manual_seed(2)
            geometry = GEOMETRY[case]
            if kind == 'fold':
                value = bits(generator, (1, HEADS, 32 * geometry['halves'], DIM))
                expected, function = fold_served(case, value), perm.fold_query
            else:
                folded_heads = 2 * geometry['halves'] * geometry['users'] * (HEADS // 2)         # kv heads a chip x halves x users x group
                value = bits(generator, (1, folded_heads, 32, DIM))
                expected, function = unfold_served(case, value), perm.unfold_output
            tensor = raw.from_logical(value)

            def run(served):
                return function(operations, tensor, keep(owned), served=served, site=case, **GEOMETRY[case])

            warm_served = lambda: raw.from_logical(expected)                       # noqa: E731
        previous = perm.set_pass(True)
        try:
            run(warm_served)                                                      # the bucket's eager warm pass
        finally:
            perm.set_pass(previous)
        trace = operations.begin_trace_capture(raw.mesh)
        previous = perm.set_pass(False)
        try:
            run(mock.Mock(side_effect=AssertionError('a capture records the launches only')))
        except CaptureViolation as error:
            escaped = error
        finally:
            perm.set_pass(previous)
            operations.end_trace_capture(raw.mesh, trace)
        if escaped is None:
            for unused in range(replays):
                operations.execute_trace(raw.mesh, trace)
        perm._PASS['eager'] = None
    return Lifecycle(operations.rules, lines, escaped=escaped, replayed=state.replayed,
                     audit_lines=[line for line in lines if line.startswith(perm.AUDIT) and 'exact=True' in line])


class PermuteTests(unittest.TestCase):
    def test_every_kv_assembly_the_drafters_make_launches_in_the_capture_what_the_warm_pass_launched(self):
        for case in ('quad', 'octo', 'pair', 'pair-short'):
            with self.subTest(case=case):
                found = permute_lifecycle('kv', case)
                self.assertEqual(found.clean(), [])
                self.assertIsNone(found.escaped)
                self.assertEqual(found.replayed, 3)
                self.assertEqual(len(found.audit_lines), 1, 'the warm pass audited, the capture did not')
                self.assertEqual(found.rules.captured_launches, 1)

    def test_the_query_fold_and_the_output_unfold_of_every_shape_are_the_same(self):
        for kind in ('fold', 'unfold'):
            for case in ('pair', 'quad', 'octo'):
                with self.subTest(kind=kind, case=case):
                    found = permute_lifecycle(kind, case)
                    self.assertEqual(found.clean(), [])
                    self.assertIsNone(found.escaped)
                    self.assertEqual(len(found.audit_lines), 1)
                    self.assertEqual(found.rules.captured_launches, 1)


def qkv1_lifecycle(replays=3):
    import torch
    import draft_permute_tp as perm
    import draft_qkv_tp as qkv
    import tp4_sampdraft
    from test_draft_qkv_tp import DIM, HEADS, KV, fused_ops
    from test_draft_permute_tp import bits
    from test_pair_row_exact import keep
    from tp_test_support import four_cards

    raw = fused_ops()
    state = add_capture_calls(raw)
    projection = bits(torch.Generator().manual_seed(5), (1, 1, 64, 1536))
    raw.linear = lambda value, weight, **options: raw.from_logical(projection)               # the branch's matmul closure runs this op
    operations = capture_rules.guard(raw)
    prepared = raw.from_logical(bits(torch.Generator().manual_seed(6), (1, 1, 64, 5120)))
    flat = projection[0, 0]
    heads = {name: raw.from_logical(flat[:, low:high].reshape(64, count, DIM).permute(1, 0, 2).reshape(1, count, 64, DIM).contiguous())
             for name, low, high, count in (('q', 0, 1024, HEADS), ('k', 1024, 1280, KV), ('v', 1280, 1536, KV))}
    lines, escaped, owned = [], None, []

    def call(served):
        return qkv.project(operations, prepared, keep(owned), parameters=dict(projections=dict(qkv='fused-weight')),
                           project=lambda value, weight, grid, rows, columns: operations.linear(value, weight, program_config=(grid, rows, columns)),
                           rows=64, served=served, site='quad')

    with mock.patch.dict(os.environ, {'QWEN_FAST_TP': '4', qkv.FLAG: '1', qkv.AUDIT_FLAG: '1'}), four_cards(), \
            mock.patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
        perm._LOGGED.clear()
        perm._CACHE.clear()
        perm._SERVED['depth'] = 0
        previous = perm.set_pass(True)
        try:
            call(lambda: heads)
        finally:
            perm.set_pass(previous)
        trace = operations.begin_trace_capture(raw.mesh)
        previous = perm.set_pass(False)
        try:
            call(mock.Mock(side_effect=AssertionError('a capture records the launches only')))
        except CaptureViolation as error:
            escaped = error
        finally:
            perm.set_pass(previous)
            operations.end_trace_capture(raw.mesh, trace)
        if escaped is None:
            for unused in range(replays):
                operations.execute_trace(raw.mesh, trace)
        perm._PASS['eager'] = None
    return Lifecycle(operations.rules, lines, escaped=escaped, replayed=state.replayed,
                     audit_lines=[line for line in lines if line.startswith(qkv.AUDIT) and 'exact=True' in line])


def head64_lifecycle(replays=3):
    import torch
    import draft_head64_tp as head64
    import draft_permute_tp as perm
    import draft_shared_head_tp
    import tp4_sampdraft
    from test_draft_head64_tp import VOCAB, Tensor, TorchOps
    from tp_test_support import four_cards

    raw = TorchOps()
    state = add_capture_calls(raw)
    columns = torch.arange(VOCAB, dtype=torch.float32)
    raw.linear = lambda left, right: Tensor((left.value[..., :1].double() * columns.double()).float())      # row independent
    operations = capture_rules.guard(raw)
    generator = torch.Generator().manual_seed(4)
    normalized = Tensor(torch.randint(-3, 4, (1, 1, 64, 5120), generator=generator).float())
    weight = None                                                                   # (the stand-in matmul ignores it: 5120 x 62,080 would be 1.2 GB on the host)
    model = SimpleNamespace(num_devices=4, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight=weight)
    lines, escaped = [], None
    with mock.patch.dict(os.environ, {'QWEN_FAST_TP': '4', head64.FLAG: '1', head64.AUDIT_FLAG: '1'}), four_cards(), \
            mock.patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
        perm._LOGGED.clear()
        perm._SERVED['depth'] = 0
        reference = []
        for half in range(2):
            logits = raw.linear(Tensor(normalized.value[:, :, 32 * half:32 * half + 32]), weight)
            reference.append(draft_shared_head_tp.local_head_candidates(raw, logits, []))
        previous = perm.set_pass(True)
        try:
            head64.candidates(operations, model, normalized, [], served=lambda: reference, site='quad')
        finally:
            perm.set_pass(previous)
        trace = operations.begin_trace_capture('mesh')
        previous = perm.set_pass(False)
        try:
            head64.candidates(operations, model, normalized, [], served=mock.Mock(side_effect=AssertionError('a capture records the launches only')), site='quad')
        except CaptureViolation as error:
            escaped = error
        finally:
            perm.set_pass(previous)
            operations.end_trace_capture('mesh', trace)
        if escaped is None:
            for unused in range(replays):
                operations.execute_trace('mesh', trace)
        perm._PASS['eager'] = None
    return Lifecycle(operations.rules, lines, escaped=escaped, replayed=state.replayed,
                     audit_lines=[line for line in lines if line.startswith(head64.AUDIT) and 'exact=True' in line])


class FusedProjectionAndHeadTests(unittest.TestCase):
    def test_the_fused_q_k_v_projection_audits_in_the_warm_pass_and_launches_the_same_in_the_capture(self):
        found = qkv1_lifecycle()
        self.assertEqual(found.clean(), [])
        self.assertIsNone(found.escaped)
        self.assertEqual(len(found.audit_lines), 1)
        self.assertEqual(found.replayed, 3)
        self.assertTrue(found.rules.captured_launches)

    def test_the_64_row_head_audits_in_the_warm_pass_and_runs_the_same_ops_in_the_capture(self):
        found = head64_lifecycle()
        self.assertEqual(found.clean(), [])
        self.assertIsNone(found.escaped)
        self.assertEqual(len(found.audit_lines), 1)
        self.assertTrue(found.rules.captured_launches)


def wp6_fused_commit_lifecycle(segments=6, rows=32, site='feature', quad=False):
    """The fused commit's order (fused_commit_tp.capture): per segment the projection runs eagerly and is then CAPTURED, and the next segment's eager warm follows the capture. The
    reduce is the feature projection's gather (feature_collective_tp.gather_add_projection -> draft_reduce_tp.gather_add): the call that reached the first card failure."""
    import attention_batch
    import draft_fusion_tp as fusion
    import draft_reduce_tp as reduce
    import wp6_fake_device as fake
    from test_draft_reduce_tp import COLLECTIVES, mesh_for, partials, served_chain
    from test_draft_wp6_capture import ALL_AUDITED
    from tp_test_support import four_cards

    escaped = None
    lines = []
    with mock.patch.dict(os.environ, ALL_AUDITED, clear=True), four_cards(), mock.patch.object(fusion, 'log_line', side_effect=lines.append), \
            mock.patch('mesh_link_policy.projection_links', return_value=1), mock.patch('feature_collective_tp.projection_links', return_value=1):
        fusion.reset()
        raw = fake.install_emulators(fake.FakeOperations(4), (11, 10))
        operations = capture_rules.guard(raw)
        mesh = mesh_for()
        value = partials(raw, rows, 6)
        keep = lambda tensor: tensor                                                                  # noqa: E731

        def project():
            return reduce.gather_add(operations, mesh, COLLECTIVES, value, served=served_chain, site=site, quad=quad, retain_temporaries=keep)

        for unused in range(segments):
            project()                                                                                # the segment's eager warm
            try:
                attention_batch.capture_operation(operations, mesh, project)
            except CaptureViolation as error:
                escaped = error
                break
    return Lifecycle(operations.rules, lines, escaped=escaped, audit_lines=[line for line in lines if 'audit' in line and 'exact=True' in line],
                     skipped=[line for line in lines if 'skipped its audit' in line])


class FusedCommitSequenceTests(unittest.TestCase):
    def test_the_feature_projection_gather_audits_the_warm_calls_and_the_captures_launch_what_the_warm_call_ran(self):
        found = wp6_fused_commit_lifecycle()
        self.assertEqual(found.clean(), [])
        self.assertIsNone(found.escaped)
        self.assertTrue(found.audit_lines)
        self.assertTrue(found.skipped, 'the captures skipped their audits')
        self.assertEqual(found.rules.captures, 6)

    def test_the_quad_chain_at_the_mlp_site_is_the_same(self):
        found = wp6_fused_commit_lifecycle(rows=64, site='mlp', quad=True)
        self.assertEqual(found.clean(), [])
        self.assertIsNone(found.escaped)


# ---------------------------------------------------------------------------------------------------------------------------------
# The census of host round trips: every place a lever module of the programme synchronizes, reads a tensor back or writes one from the host, with the reason it cannot be reached
# inside a capture. The scenarios above prove the ones the fakes can drive; this census is what keeps the rest honest - a new call site fails until someone classifies it.
# ---------------------------------------------------------------------------------------------------------------------------------

HOST_CALLS = frozenset(('to_torch', 'synchronize_device', 'from_torch', 'host_buffer', 'copy_host_to_device_tensor', 'synchronize', 'event_synchronize', 'to_device', 'from_device'))
LEVER_MODULES = ('kv_page_writer_tp4', 'tp4_mlp_gateup', 'tp4_mlp_fused', 'tp4_shard_argmax', 'ccl_options_tp', 'tile_collective_tp', 'distributed_norm_gather_tp',
                 'draft_permute_tp', 'draft_qkv_tp', 'draft_head64_tp', 'draft_fusion_tp', 'draft_reduce_tp', 'draft_tail_tp', 'draft_gateup_tp', 'draft_mmgrid_tp', 'round_host',
                 'qwen_device_zeros', 'qwen_lazy_shard')

# (module, function, call) -> why it cannot run in a capture
CLASSIFIED = {
    ('draft_fusion_tp', 'audit_begin', 'synchronize_device'): 'WP6: after in_capture() has said no (a capture skips its audit; the tracked capture calls follow nesting)',
    ('draft_fusion_tp', 'chip_bits', 'to_torch'): 'WP6: the readback of an audit that audit_begin let run (eager)',
    ('draft_gateup_tp', 'audit', 'from_torch'): 'WP6: the audit uploads the separate weights it compares with, after audit_begin',
    ('draft_gateup_tp', 'audit', 'synchronize_device'): 'WP6: inside the audit, after audit_begin',
    ('draft_mmgrid_tp', '_audit', 'synchronize_device'): 'WP6: inside the audit, after audit_begin',
    ('draft_reduce_tp', '_audit', 'synchronize_device'): 'WP6: inside the audit, after audit_begin',
    ('draft_reduce_tp', 'gather_add', 'synchronize_device'): 'WP6: the served-eager branch (retain_temporaries is None), which the served function has too',
    ('draft_tail_tp', '_audited', 'synchronize_device'): 'WP6: inside the audit, after audit_begin',
    ('draft_permute_tp', 'compare', 'to_torch'): 'WP7: only under auditing(): the bucket marked its pass eager (quad_draft_tp._execute); a capture is marked False',
    ('kv_page_writer_tp4', 'Audit.__init__.upload', 'from_torch'): 'WP2: the shadow cache and counters are uploaded when the writer is built, at the block attach, before any capture',
    ('kv_page_writer_tp4', 'audit_round', 'to_torch'): 'WP2: the counters read after a replay (packed_verifier, outside the capture)',
    ('tile_collective_tp', 'audit_round', 'to_torch'): 'WP5: after a replay (packed_verifier), outside the capture',
    ('tp4_mlp_gateup', 'audit_round', 'to_torch'): 'WP4: after a replay (packed_verifier), outside the capture',
    ('tp4_shard_argmax', 'audit_round', 'to_torch'): 'WP1: after a replay (the sampler audit), outside the capture',
    ('qwen_device_zeros', 'DeviceZeros._audit', 'from_device'): 'upload-p0: engine start (the KV pool and buffer pool are built before the first capture)',
    ('qwen_device_zeros', 'DeviceZeros._audit', 'from_torch'): 'upload-p0: engine start',
    ('qwen_device_zeros', 'DeviceZeros.report', 'synchronize'): 'upload-p0: engine start',
    ('qwen_device_zeros', 'kv_twin._allocate_kv_caches_tp', 'synchronize_device'): 'upload-p0: engine start (the KV allocation)',
    ('qwen_device_zeros', 'shard_bytes', 'host_buffer'): 'upload-p0: the audit sampler, engine start',
    ('qwen_lazy_shard', 'audit_load', 'from_device'): 'upload-p0: weight load, engine start',
    ('qwen_lazy_shard', 'audit_load', 'from_torch'): 'upload-p0: weight load, engine start',
}


def host_call_sites():
    import ast

    found = set()

    def walk(node, scope, module):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, scope + [child.name], module)
                continue
            if isinstance(child, ast.Call):
                function = child.func
                name = function.attr if isinstance(function, ast.Attribute) else (function.id if isinstance(function, ast.Name) else None)
                if name in HOST_CALLS:
                    found.add((module, '.'.join(scope) or '<module>', name))
            walk(child, scope, module)

    for module in LEVER_MODULES:
        walk(ast.parse((HERE / (module + '.py')).read_text(encoding='utf-8')), [], module)
    return found


class ReadbackCensusTests(unittest.TestCase):
    def test_every_host_round_trip_of_a_lever_module_is_classified(self):
        found = host_call_sites()
        self.assertEqual(sorted(found - set(CLASSIFIED)), [], 'a new synchronize / readback / host write in a lever module: prove it cannot run inside a capture, then classify it above')
        self.assertEqual(sorted(set(CLASSIFIED) - found), [], 'a classified call site is gone: drop it from the census')

    def test_round_host_touches_no_device(self):
        text = (HERE / 'round_host.py').read_text(encoding='utf-8')
        for word in ('ttnn', 'operations', 'generic_op', 'to_torch', 'synchronize', 'begin_trace_capture'):
            self.assertNotIn(word, text.replace('no kernel, trace, allocation or device command', ''), word)

    def test_the_wrapper_sees_every_call_name_of_the_census(self):
        self.assertTrue(HOST_CALLS <= capture_rules.HOST_ROUND_TRIPS | {'host_buffer', 'from_device'})
        self.assertTrue({'to_torch', 'synchronize_device', 'from_torch', 'copy_host_to_device_tensor'} <= capture_rules.HOST_ROUND_TRIPS)


# Which scenario of this module gates which audit flag of the audited combined twin. A flag that joins that twin without a scenario fails the test below.
GATED = {
    'QWEN_FAST_DRAFT_PERMUTE_AUDIT': 'PermuteTests',
    'QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT': 'ShardArgmaxTests',
    'QWEN_FAST_DRAFT_REDUCE_AUDIT': 'WP6BranchTests',
    'QWEN_FAST_DRAFT_TAIL_AUDIT': 'WP6BranchTests',
    'QWEN_FAST_DRAFT_GATEUP1_AUDIT': 'WP6BranchTests',
    'QWEN_FAST_DRAFT_MM_GRID_AUDIT': 'WP6BranchTests',
    'QWEN_FAST_DRAFT_QKV1_AUDIT': 'FusedProjectionAndHeadTests',
    'QWEN_FAST_DRAFT_HEAD64_AUDIT': 'FusedProjectionAndHeadTests',
    'QWEN_FAST_CCL_OPTIONS_AUDIT': 'CCLOptionsTests',
    'QWEN_FAST_MLP_CFG_AUDIT': 'MLPTests',
    'QWEN_FAST_KV_PAGE_WRITER_AUDIT': 'KVPageTests',
    'QWEN_FAST_DEVICE_ZEROS_AUDIT': 'ReadbackCensusTests',          # engine start: the census holds that no readback of it is reachable in a capture
    'QWEN_FAST_LAZY_SHARD_W_AUDIT': 'ReadbackCensusTests',
    'QWEN_FAST_TP4_ROUND_HOST_AUDIT': 'ReadbackCensusTests',        # host only: round_host touches no device
}
SEQUENCES = ('FusedCommitSequenceTests',)       # the fused commit's warm / capture order, for the WP6 audits at the feature projection


class CoverageTests(unittest.TestCase):
    def test_every_audit_flag_of_the_audited_combined_twin_has_a_scenario_in_this_module(self):
        import json

        import make_fusion_profiles as fusion

        profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
        env = profiles[fusion.NAMESPACE + 'all-audit']['env']
        audits = sorted(name for name, value in env.items() if name.endswith('_AUDIT') and value == '1')
        self.assertEqual([name for name in audits if name not in GATED], [], 'an audit flag joined the audited twin with no capture scenario')
        self.assertEqual([name for name in GATED if name not in audits], [], 'a gated flag is no longer in the audited twin')
        for name in sorted(set(GATED.values()) | set(SEQUENCES)):
            self.assertTrue(isinstance(globals().get(name), type) and issubclass(globals()[name], unittest.TestCase), name)

    def test_the_timed_twin_carries_no_audit_flag_so_none_of_this_runs_in_its_captures(self):
        import json

        import make_fusion_profiles as fusion

        profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
        self.assertEqual([name for name in GATED if name in profiles[fusion.NAMESPACE + 'all']['env']], [])


if __name__ == '__main__':
    unittest.main()
