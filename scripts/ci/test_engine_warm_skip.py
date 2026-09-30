"""Phase-1 quick wins (each default off, each usable without Stage E): the ledger switch and the engine warm skip.

1. QWEN_FAST_MEMORY_LEDGER_OFF=1 (memory_ledger.OFF_FLAG). The image ENV sets QWEN_FAST_MEMORY_LEDGER=1 for the gates,
   so a production profile needs a switch that wins over it. Tests: the enabled() matrix; begin() builds no ledger
   under it and every hook is then a no-op; the flag's strict parser and the policy's check; unset, nothing changed.

2. QWEN_FAST_ENGINE_WARM_SKIP=1 (serving_fast_policy.ENGINE_WARM_SKIP_FLAG, verifier_engine skip_compiled_warm). The
   engine build's warm-up eager forwards (one per captured bucket) compile the bucket's programs; with the flag a
   build skips the warm of a bucket whose key (VerifierEngine.warm_key) an earlier build in the process warmed to
   completion. What the skip may change is only the content of tensors the warm wrote, so the census world (the
   real VerifierEngine, GreedySession, FastRequest and pool over a fake ttnn that logs every access) proves, for the
   warm's every write to a tensor that outlives it:
   - its next effective access is a write (or a free): a replay counts as the trace's first access to it, a capture
     executes nothing, the publish prewarm's reads feed spare banks that are rewritten in full before any read
     (publish_prewarm's docstring) and are set aside by name;
   - the K/V cache is the one tensor whose access the fake cannot split by row, so its argument is the frontier
     induction below, simulated with a negative control;
   - the device events from the first verify capture on (the build's remainder and six rounds) are the same with and
     without the warm, tensor for tensor.
   Off, the parity tests of test_parked_census (the churn trail equals the base commit's) already hold the engine
   byte for byte; here the flag-off engine is also held to run every warm and record nothing.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import memory_ledger  # noqa: E402
import serving_fast_policy as policy  # noqa: E402
import serving_parked_engines as parked  # noqa: E402
import verifier_engine  # noqa: E402
from test_parked_census import World  # noqa: E402
from test_parked_engine_set import make_set  # noqa: E402

SKIP = {policy.ENGINE_WARM_SKIP_FLAG: '1'}


class LedgerSwitchTests(unittest.TestCase):
    def test_off_wins_over_the_image_flag_and_only_off_one_does(self):
        on = {memory_ledger.FLAG: '1'}
        self.assertTrue(memory_ledger.enabled(on))
        self.assertFalse(memory_ledger.enabled({}))
        self.assertFalse(memory_ledger.enabled({memory_ledger.OFF_FLAG: '1'}))
        self.assertFalse(memory_ledger.enabled({**on, memory_ledger.OFF_FLAG: '1'}))
        for value in ('0', ''):
            self.assertTrue(memory_ledger.enabled({**on, memory_ledger.OFF_FLAG: value}), value)

    def test_begin_builds_no_ledger_and_every_hook_is_a_no_op_under_off(self):
        built = []

        class Ledger:
            def __init__(self, *arguments, **options):
                built.append(arguments)
                self.report = 'nowhere'

            def log(self, message):
                pass

            def __getattr__(self, name):
                raise AssertionError('the ledger was used: %s' % name)

        environment = {memory_ledger.FLAG: '1', memory_ledger.OFF_FLAG: '1'}
        with patch.dict('os.environ', environment), patch.object(memory_ledger, 'MemoryLedger', Ledger), \
                patch.object(memory_ledger, '_active', None):
            self.assertIsNone(memory_ledger.begin(object(), object()))
            self.assertIsNone(memory_ledger.active())
            memory_ledger.record('P0', model=object())
            memory_ledger.engine_admitted('request', engine_request=object())
            memory_ledger.first_packed_round(packed_block=object())
            self.assertIsNone(memory_ledger.before('engine', estimate=1))
            self.assertIsNone(memory_ledger.after(None))
            self.assertEqual(built, [])
        with patch.dict('os.environ', {memory_ledger.FLAG: '1'}), patch.object(memory_ledger, 'MemoryLedger', Ledger), \
                patch.object(memory_ledger, '_active', None):
            self.assertIsNotNone(memory_ledger.begin(object(), object()), 'the image flag alone still builds it')
            self.assertEqual(len(built), 1)

    def test_the_flags_parse_strictly_and_the_policy_names_a_bad_value(self):
        self.assertFalse(policy.engine_warm_skip_enabled({}))
        self.assertFalse(policy.engine_warm_skip_enabled({policy.ENGINE_WARM_SKIP_FLAG: '0'}))
        self.assertTrue(policy.engine_warm_skip_enabled(SKIP))
        with self.assertRaises(ValueError):
            policy.engine_warm_skip_enabled({policy.ENGINE_WARM_SKIP_FLAG: 'yes'})
        with self.assertRaises(ValueError):
            policy.quick_win_enabled('QWEN_FAST_SOMETHING_ELSE', {})
        self.assertEqual(policy.quick_win_problems({}), [])
        self.assertEqual(policy.quick_win_problems({'QWEN_FAST_MEMORY_LEDGER_OFF': '1', 'QWEN_FAST_ENGINE_WARM_SKIP': '0'}), [])
        problems = policy.quick_win_problems({'QWEN_FAST_MEMORY_LEDGER_OFF': 'on', 'QWEN_FAST_ENGINE_WARM_SKIP': '2'})
        self.assertEqual(len(problems), 2)
        self.assertIn('QWEN_FAST_MEMORY_LEDGER_OFF', problems[0])


class GateReadsTheSwitchTests(unittest.TestCase):
    def test_the_harness_promises_no_ledger_marker_when_the_ledger_is_off(self):
        import lever_n_m3native_gate as harness
        import real_text_compare

        on = {'QWEN_FAST_MEMORY_LEDGER': '1'}
        self.assertIn('QWEN_FAST_MEMORY_LEDGER', harness.required_flag_markers(on, 4))
        self.assertNotIn('QWEN_FAST_MEMORY_LEDGER', harness.required_flag_markers(
            dict(on, QWEN_FAST_MEMORY_LEDGER_OFF='1'), 4))
        self.assertIn('QWEN_FAST_MEMORY_LEDGER', harness.required_flag_markers(dict(on, QWEN_FAST_MEMORY_LEDGER_OFF='0'), 4))
        # the ledger switch only logs; the warm skip is an exactness claim, so a divergence stays a FAIL
        self.assertIn('QWEN_FAST_MEMORY_LEDGER_OFF', real_text_compare.ARITHMETIC_NEUTRAL)
        self.assertIn('QWEN_FAST_ENGINE_WARM_SKIP', real_text_compare.S2_EXACT_CLAIMS)


class WarmKeyTests(unittest.TestCase):
    model, mesh, sampler = object(), object(), object()

    def engine(self, **changes):
        fields = dict(model=self.model, mesh=self.mesh, commit_only_gdn=True, pages=SimpleNamespace(shape=(1, 68)),
                      norm_batch=True, attention_replay=False, attention_mask_once=False, replay_group_rows=4,
                      short_context=False, native_sampling_rows=True, sampler=self.sampler, target_attention_t16=False,
                      retain_feature_taps=(1, 2), retain_mtp_hidden=False, replay_plan=None)
        fields.update(changes)
        engine = verifier_engine.VerifierEngine.__new__(verifier_engine.VerifierEngine)
        vars(engine).update(fields)
        return engine

    def test_the_key_moves_with_everything_a_warm_forward_compiles_for(self):
        bucket = dict(rows=4, capture_position=1000)
        base = self.engine().warm_key(bucket)
        self.assertEqual(base, self.engine().warm_key(dict(rows=4, capture_position=999999)),
                         'an ordinary bucket\'s position is a device input, not a program parameter')
        moves = [self.engine(pages=SimpleNamespace(shape=(1, 132))), self.engine(norm_batch=False),
                 self.engine(retain_feature_taps=(1,)), self.engine(commit_only_gdn=False), self.engine(model=object()),
                 self.engine(mesh=object()), self.engine(native_sampling_rows=False), self.engine(short_context=True),
                 self.engine(retain_mtp_hidden=True), self.engine(replay_group_rows=8)]
        for changed in moves:
            self.assertNotEqual(changed.warm_key(bucket), base)
        self.assertNotEqual(self.engine().warm_key(dict(rows=2, capture_position=1000)), base)
        # a rows == 1 bucket retains nothing, so commit_only_gdn does not change its key
        one = dict(rows=1, capture_position=5)
        self.assertEqual(self.engine().warm_key(one), self.engine(commit_only_gdn=False).warm_key(one))
        # under a replay plan the buckets are keyed by position, so their key is too
        plan = self.engine(replay_plan=object())
        self.assertNotEqual(plan.warm_key(bucket), plan.warm_key(dict(rows=4, capture_position=1001)))


class Recorder:
    """The census world with the engine's eager forwards and the publish prewarm marked on its trail."""

    def __init__(self, world):
        self.world, self.warm_windows, self.prewarm_windows, self.stack = world, [], [], None

    def __enter__(self):
        import publish_prewarm

        ops = self.world.ops
        original, warm_original = verifier_engine.VerifierEngine.operation, publish_prewarm.warm
        recorder = self

        def operation(engine, *arguments, **options):
            eager = ops.capturing is None
            began = len(ops.trail)
            try:
                return original(engine, *arguments, **options)
            finally:
                if eager:
                    recorder.warm_windows.append((began, len(ops.trail)))

        def warm(*arguments, **options):
            began = len(ops.trail)
            try:
                return warm_original(*arguments, **options)
            finally:
                recorder.prewarm_windows.append((began, len(ops.trail)))

        self.patches = [patch.object(verifier_engine.VerifierEngine, 'operation', operation),
                        patch.object(publish_prewarm, 'warm', warm)]
        for entry in self.patches:
            entry.start()
        return self

    def __exit__(self, *failure):
        for entry in reversed(self.patches):
            entry.stop()
        return False


def effective_accesses(ops, start, tracked, ignored):
    """{serial: 'read' | 'written' | 'freed'} for the tracked tensors: what each one's next effective access is,
    from trail index `start`. Events inside a capture are records (a capture executes nothing on the device); a
    replay is its trace's first access to a tensor (a read when the trace reads it before writing it); reads inside
    the `ignored` windows are set aside."""
    status, capturing = {}, None
    for index, event in enumerate(ops.trail[start:], start):
        kind = event[0]
        if kind == 'begin':
            capturing = event[1]
        elif kind == 'end':
            capturing = None
        elif capturing is not None:
            continue
        elif kind in ('read', 'write', 'free', 'upload'):
            serial = event[1]
            if serial in tracked and serial not in status:
                if kind == 'read' and any(begin <= index < end for begin, end in ignored):
                    continue
                status[serial] = {'read': 'read', 'write': 'written', 'upload': 'written', 'free': 'freed'}[kind]
        elif kind == 'replay':
            trace = next(item for item in ops.traces if item.label == event[1])
            for serial, (how, tensor) in trace.first_access.items():
                if serial in tracked and serial not in status:
                    status[serial] = 'read' if how == 'read' else 'written'
            for serial in trace.writes:
                if serial in tracked and serial not in status:
                    status[serial] = 'written'
    return status


class WarmWritesTests(unittest.TestCase):
    def scenario(self, world):
        first = world.admit('request-a', 4096, 48)
        second = world.admit('request-b', 300, 40)
        for _ in range(6):
            world.replay_block()
            for request in (first, second):
                world.step(request)
        return first, second

    def test_nothing_a_warm_forward_wrote_is_read_before_it_is_rewritten(self):
        with patch.object(verifier_engine, '_warmed', set()), World(tracking=True) as world, Recorder(world) as recorded:
            requests = self.scenario(world)
            ops = world.ops
            self.assertEqual(len(recorded.warm_windows), 6, 'two engines, three warm forwards each')
            labels = {event[1]: event[2] for event in ops.trail if event[0] == 'alloc'}
            examined, verdicts = set(), []
            for begin, end in recorded.warm_windows:
                born = {event[1] for event in ops.trail[begin:end] if event[0] == 'alloc'}
                written = {event[1] for event in ops.trail[begin:end] if event[0] in ('write', 'upload')} - born
                status = effective_accesses(ops, end, written, recorded.prewarm_windows)
                for serial in written:
                    label = labels.get(serial, 'vllm kv cache')
                    examined.add(label)
                    if status.get(serial) == 'read' and label != 'vllm kv cache':
                        verdicts.append((label, serial))
            self.assertEqual(verdicts, [], 'a tensor the warm wrote is read before anything rewrote it')
            # what the warm does reach: the native GDN state, the pooled checkpoints, the feature taps, the fixtures'
            # entry states and the KV cache
            self.assertTrue({'gdn snapshot', 'from_torch', 'vllm kv cache'} <= examined)
            self.assertTrue(any(label.startswith('native gdn') for label in examined))
            for request in requests:
                request.close(request.session.request_id)

    def test_the_only_reads_of_a_warm_write_before_a_rewrite_are_the_prewarms(self):
        """The census has to see them: the feature taps are read by the publish prewarm, whose products are discarded."""
        with patch.object(verifier_engine, '_warmed', set()), World(tracking=True) as world, Recorder(world) as recorded:
            request = world.admit('request-a', 4096, 16)
            ops = world.ops
            labels = {event[1]: event[2] for event in ops.trail if event[0] == 'alloc'}
            begin, end = recorded.warm_windows[0]
            written = {event[1] for event in ops.trail[begin:end] if event[0] == 'write'}
            strict = effective_accesses(ops, end, written, [])
            lenient = effective_accesses(ops, end, written, recorded.prewarm_windows)
            read_strict = {labels.get(serial) for serial, how in strict.items() if how == 'read'}
            read_lenient = {labels.get(serial) for serial, how in lenient.items() if how == 'read'}
            self.assertIn('from_torch', read_strict, 'the taps are read by the prewarm')
            self.assertNotIn('from_torch', read_lenient)
            request.close('request-a')

    def test_every_pooled_checkpoint_and_tap_is_written_by_its_bucket_s_verify_trace_before_anything_reads_it(self):
        """A read of a checkpoint or tap needs a verified ticket (publish refuses otherwise), and the verify replays
        the trace: it writes them first."""
        with patch.object(verifier_engine, '_warmed', set()), World() as world:
            request = world.admit('request-a', 4096, 16)
            engine = request.engine
            for bucket in engine.buckets.values():
                trace = next(item for item in world.ops.traces if item.outputs and bucket['output'] in item.outputs
                             or item.outputs and bucket['output'][0] in item.outputs)
                pooled = [value for snapshot in bucket['checkpoints'] for value in snapshot] + list(bucket['target_features'])
                for value in pooled:
                    self.assertEqual(trace.first_access[value.serial][0], 'write',
                                     '%s of the %d-row bucket is read by its trace before it is written' % (
                                         value.label, bucket['rows']))
            request.close('request-a')


class Frontier:
    """The K/V cache of one request, by position, in the engine's own protocol. The warm forward writes W over
    [P, P + r); a verify of width w at frontier F writes V over [F, F + w) and reads [0, F + w) (a row attends to
    every position up to and including its own); the frontier then advances by the accepted tokens a, 1 <= a <= w."""

    def __init__(self, prompt, warms=(1, 2, 4), verify_writes=None):
        self.cache = {position: 'P' for position in range(prompt)}
        for rows in warms:
            for position in range(prompt, prompt + rows):
                self.cache[position] = 'W'
        self.frontier, self.verify_writes = prompt, verify_writes

    def round(self, width, accepted):
        written = width if self.verify_writes is None else self.verify_writes
        for position in range(self.frontier, self.frontier + written):
            self.cache[position] = 'V'
        seen = [self.cache.get(position) for position in range(self.frontier + width)]
        self.frontier += accepted
        return seen


class KeyValueRangeTests(unittest.TestCase):
    def test_a_verify_never_reads_a_row_the_warm_wrote(self):
        import random

        generator = random.Random(7)
        for prompt in (1, 63, 64, 2048, 4095, 123136):
            cache = Frontier(prompt)
            for _ in range(400):
                width = generator.choice((1, 2, 4))
                seen = cache.round(width, generator.randint(1, width))
                self.assertNotIn('W', seen)
                self.assertNotIn(None, seen)

    def test_the_simulation_sees_a_verify_that_writes_less_than_it_reads(self):
        cache = Frontier(100, verify_writes=1)
        self.assertIn('W', cache.round(4, 1), 'the negative control: a verify that wrote fewer rows than it reads')


class WarmSkipBuildTests(unittest.TestCase):
    def counts(self, request):
        return request.engine.warms_run, request.engine.warms_skipped

    def test_flag_off_warms_every_bucket_and_records_nothing(self):
        with patch.object(verifier_engine, '_warmed', set()) as warmed, World() as world:
            first, second = world.admit('a', 4096, 48), world.admit('b', 300, 40)
            self.assertEqual(self.counts(first), (3, 0))
            self.assertEqual(self.counts(second), (3, 0))
            self.assertEqual(warmed, set())
            self.assertFalse([line for line in world.lines if line.startswith(policy.ENGINE_WARM_MARKER)])
            first.close('a')
            second.close('b')

    def test_flag_on_warms_the_first_build_of_a_shape_and_skips_the_rest(self):
        with patch.object(verifier_engine, '_warmed', set()) as warmed, World(environment=SKIP) as world:
            first, second = world.admit('a', 4096, 48), world.admit('b', 300, 40)
            self.assertEqual(self.counts(first), (3, 0))
            self.assertEqual(self.counts(second), (0, 3))
            self.assertEqual(len(warmed), 3)
            lines = [line for line in world.lines if line.startswith(policy.ENGINE_WARM_MARKER)]
            self.assertEqual([line.split(' request=')[0] for line in lines],
                             [policy.ENGINE_WARM_MARKER + 'run=3 skipped=0', policy.ENGINE_WARM_MARKER + 'run=0 skipped=3'])
            self.assertEqual(world.ops.violations, [])
            first.close('a')
            second.close('b')

    def test_a_bucket_no_earlier_build_warmed_still_warms(self):
        with patch.object(verifier_engine, '_warmed', set()), World(environment=SKIP) as world:
            small = world.admit('a', 4096, 2)
            self.assertEqual(sorted(small.engine.buckets), [1])
            self.assertEqual(self.counts(small), (1, 0))
            small.close('a')
            wider = world.admit('b', 4096, 48)
            self.assertEqual(sorted(wider.engine.buckets), [1, 2, 4])
            self.assertEqual(self.counts(wider), (2, 1), 'the 2- and 4-row buckets were never warmed in this process')
            wider.close('b')

    def test_a_warm_that_fails_is_not_recorded_and_the_next_build_warms_again(self):
        with patch.object(verifier_engine, '_warmed', set()) as warmed, World(environment=SKIP) as world:
            original = verifier_engine.VerifierEngine.operation

            def refuse(engine, *arguments, **options):
                if world.ops.capturing is None:
                    raise RuntimeError('the warm forward died')
                return original(engine, *arguments, **options)

            with patch.object(verifier_engine.VerifierEngine, 'operation', refuse):
                with self.assertRaises(RuntimeError):
                    world.admit('a', 4096, 48)
            self.assertEqual(warmed, set())
            retry = world.admit('b', 4096, 48)
            self.assertEqual(self.counts(retry), (3, 0))
            retry.close('b')

    def test_the_engine_refuses_a_non_boolean_switch(self):
        with patch.object(verifier_engine, '_warmed', set()), World() as world:
            with self.assertRaises(ValueError):
                verifier_engine.VerifierEngine(world.model, None, None, world.helpers, skip_compiled_warm='yes')

    def test_the_skipped_build_runs_no_eager_forward(self):
        with patch.object(verifier_engine, '_warmed', set()), World(environment=SKIP, tracking=True) as world, \
                Recorder(world) as recorded:
            first = world.admit('a', 4096, 48)
            self.assertEqual(len(recorded.warm_windows), 3)
            second = world.admit('b', 300, 40)
            self.assertEqual(len(recorded.warm_windows), 3, 'the second build ran no eager forward')
            first.close('a')
            second.close('b')


def canonical(ops, start):
    """The trail from index `start`, tensors renamed: one born before it by its label and its ordinal among that
    label's allocations, one born after it by first appearance. Trace labels and everything else stay."""
    ordinals, seen, names = {}, {}, {}
    for event in ops.trail:
        if event[0] == 'alloc':
            ordinals[event[1]] = event[2], sum(1 for label, _ in ordinals.values() if label == event[2])
    born_after = {event[1] for event in ops.trail[start:] if event[0] == 'alloc'}
    out = []
    for event in ops.trail[start:]:
        if event[0] in ('alloc', 'read', 'write', 'free', 'upload'):
            serial = event[1]
            if serial in born_after:
                names.setdefault(serial, ('new', len(names)))
                name = names[serial]
            else:
                name = ('old',) + ordinals.get(serial, ('?', serial))
            event = (event[0], name) + tuple(event[2:])
        out.append(event)
    return out


class SkipEquivalenceTests(unittest.TestCase):
    def run_scenario(self, environment):
        with patch.object(verifier_engine, '_warmed', set()), World(environment=environment, tracking=True) as world:
            first = world.admit('a', 4096, 48)
            before = len(world.ops.trail)
            second = world.admit('b', 300, 40)
            begins = [index for index, event in enumerate(world.ops.trail[before:], before) if event[0] == 'begin']
            # the first begin is the drafter's proposal capture, ahead of the warm loop; the second is the engine's
            # first verify capture, after it
            start = begins[1]
            for _ in range(6):
                world.replay_block()
                for request in (first, second):
                    world.step(request)
            trail = canonical(world.ops, start)
            for request in (first, second):
                request.close(request.session.request_id)
            return trail, len(world.ops.trail) - before

    def test_from_the_first_verify_capture_on_the_device_events_are_the_same_with_and_without_the_warm(self):
        skipped, skipped_events = self.run_scenario(SKIP)
        warmed, warmed_events = self.run_scenario({})
        self.assertGreater(len(skipped), 3000)
        self.assertLess(skipped_events, warmed_events, 'the skip removed the warm\'s events and nothing else is asked')
        self.assertEqual(len(skipped), len(warmed))
        self.assertEqual(skipped, warmed)


class ParkedAttachTests(unittest.TestCase):
    def build(self, environ):
        with patch.object(verifier_engine, '_warmed', set()), World() as world:
            engines = make_set(world, environ=environ)
            self.assertEqual(engines.build(), 4)
            lines = [line for line in world.lines if line.startswith(policy.ENGINE_WARM_MARKER)]
            counts = [(entry.engine.warms_run, entry.engine.warms_skipped) for entry in engines.slots]
            self.assertEqual(world.ops.violations, [])
            return lines, counts

    def test_the_attach_warms_the_first_engine_and_skips_the_other_three(self):
        lines, counts = self.build(dict(SKIP, QWEN_FAST_PUBLISH_PREWARM='1'))
        self.assertEqual(counts, [(3, 0), (0, 3), (0, 3), (0, 3)])
        self.assertEqual([line.rsplit(' slot=', 1)[1] for line in lines], ['0', '1', '2', '3'])

    def test_off_the_attach_warms_all_four_and_logs_no_warm_line(self):
        lines, counts = self.build({'QWEN_FAST_PUBLISH_PREWARM': '1'})
        self.assertEqual(counts, [(3, 0)] * 4)
        self.assertEqual(lines, [])


if __name__ == '__main__':
    unittest.main()
