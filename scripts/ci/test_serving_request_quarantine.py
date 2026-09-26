"""D2 request quarantine and D4's first-token terminal, under QWEN_FAST_ANY_REQUEST (plan S1
item 4): a host-side refusal of ONE request's own terms ends it as FINISHED_ABORTED and the
engine lives on (every other refusal stays fatal); a
first token that exhausts max_tokens builds no bridge. Every behaviour is also driven with the
flag OFF, where it must be exactly today's: the refusal fails the engine.

The engine step is simulated in vLLM's order - execute_model, sample_tokens, then the
scheduler's update_from_output - with a fake scheduler whose update_from_output returns
EngineCoreOutputs per client, and fake vLLM output types. The real-vLLM check at the bottom
runs only where vLLM is installed (qwen-fast-vllm-cpu.yml)."""

from contextlib import nullcontext
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import serving_request_quarantine as quarantine
from serving_lifecycle import FastServingLifecycle, prefill_gate
from serving_request_factory import RequestRefused
import test_serving_fast_policy
import test_serving_request_factory
import test_serving_worker_hook

ON = {'QWEN_FAST_ANY_REQUEST': '1'}
OFF = {'QWEN_FAST_ANY_REQUEST': '0'}


class Output(object):
    def __init__(self, request_id, new_token_ids, finish_reason=None):
        self.request_id, self.new_token_ids, self.finish_reason = request_id, new_token_ids, finish_reason


class Outputs(object):
    def __init__(self, outputs=None):
        self.outputs = [] if outputs is None else outputs


KINDS = (Output, Outputs, SimpleNamespace(ABORT='abort'), SimpleNamespace(FINISHED_ABORTED='FINISHED_ABORTED'))


def fake_scheduler_class():
    """A scheduler whose update_from_output emits each scheduled request's sampled tokens the
    way vLLM's does (EngineCoreOutputs per client index), and whose finish_requests records."""

    class FakeScheduler(object):
        def __init__(self):
            self.requests, self.finished = {}, []

        def update_from_output(self, scheduler_output, model_runner_output):
            outputs = {}
            for request_id, tokens in zip(model_runner_output.req_ids, model_runner_output.sampled_token_ids):
                if request_id in self.requests:
                    outputs.setdefault(0, Outputs()).outputs.append(Output(request_id, list(tokens)))
            return outputs

        def finish_requests(self, request_ids, status):
            for request_id in [request_ids] if isinstance(request_ids, str) else request_ids:
                self.requests.pop(request_id, None)
                self.finished.append((request_id, status))

    return FakeScheduler


class QuarantineCase(unittest.TestCase):
    def setUp(self):
        sys.modules.pop(quarantine.HOLDER_KEY, None)
        self.addCleanup(sys.modules.pop, quarantine.HOLDER_KEY, None)
        self.addCleanup(setattr, prefill_gate(), 'held', None)
        self.kinds = patch.object(quarantine, 'vllm_types', return_value=KINDS)
        self.kinds.start()
        self.addCleanup(self.kinds.stop)

    def fixture(self, environ, scheduler_cls=None, max_model_len=4352):
        """The lifecycle fixture of test_serving_lifecycle, with the scheduler class and the
        flag chosen here."""
        worker, bridge, _, decode = test_serving_worker_hook.WorkerHookTests().fixture()
        worker.model_runner.execute_model.return_value = None
        worker.model_runner.sample_tokens.return_value = SimpleNamespace(req_ids=['request'], sampled_token_ids=[[10]])
        _, _, _, arguments = test_serving_request_factory.RequestFactoryTests().fixture()
        sampling = arguments['state'].sampling_params
        bridge.state.sampling_params = sampling
        new = SimpleNamespace(req_id='request', prompt_token_ids=[1] * 4096, num_computed_tokens=0, mm_features=[],
                              prompt_embeds=None, lora_request=None, sampling_params=sampling)
        scheduled = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[new],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={'request': 4096}, total_num_scheduled_tokens=4096)
        capture = SimpleNamespace(capture=Mock(side_effect=lambda: nullcontext()), close=Mock())

        def factory(state, features):
            features.close()
            return bridge

        build = Mock(side_effect=factory)
        config = test_serving_fast_policy.FastPolicyTests().fixture()
        config.model_config.max_model_len = max_model_len
        if scheduler_cls is not None:
            config.scheduler_config.scheduler_cls = scheduler_cls
        with patch.dict('os.environ', environ):
            lifecycle = FastServingLifecycle(worker, config=config, capture_factory=Mock(return_value=capture),
                bridge_factory=build, eos_ids=(99,), cancelled=lambda: False)
        return SimpleNamespace(lifecycle=lifecycle, worker=worker, bridge=bridge, capture=capture, build=build,
                               scheduled=scheduled, sampling=sampling)

    def step(self, case, scheduler):
        """One prefill step in vLLM's order: execute, sample, then the scheduler's update."""
        scheduler.requests['request'] = SimpleNamespace(client_index=0)
        self.assertIsNone(case.worker.execute_model(case.scheduled))
        result = case.worker.sample_tokens(None)
        return result, scheduler.update_from_output(case.scheduled, result)

    def next_request(self, case, finished=('request',)):
        second = SimpleNamespace(**dict(vars(case.scheduled.scheduled_new_reqs[0]), req_id='second'))
        return SimpleNamespace(finished_req_ids=set(finished), scheduled_new_reqs=[second],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={'second': 4096}, total_num_scheduled_tokens=4096)


class BridgeRefusalTests(QuarantineCase):
    def test_off_a_refused_bridge_still_fails_the_engine(self):
        for environ in ({}, OFF):
            with self.subTest(environ=environ):
                case = self.fixture(environ, fake_scheduler_class())
                self.assertIsNone(case.lifecycle.quarantine)
                case.build.side_effect = RequestRefused('Scheduler must own capture warmup pages')
                case.worker.execute_model(case.scheduled)
                with self.assertRaisesRegex(ValueError, 'warmup pages'):
                    case.worker.sample_tokens(None)
                self.assertTrue(case.lifecycle.failed)
                self.assertIsNone(quarantine.holder().installed, 'off, nothing is installed')

    def test_on_a_refused_bridge_ends_that_request_as_aborted_and_the_engine_lives(self):
        scheduler_class = fake_scheduler_class()
        case = self.fixture(ON, scheduler_class)
        self.assertIs(case.lifecycle.quarantine, quarantine)
        case.build.side_effect = RequestRefused('Scheduler must own capture warmup pages')
        scheduler = scheduler_class()
        result, outputs = self.step(case, scheduler)
        # the seed is still this step's output, and the lifecycle is healthy and free
        self.assertEqual(result.sampled_token_ids, [[10]])
        self.assertFalse(case.lifecycle.failed)
        self.assertIsNone(case.lifecycle.request_id)
        self.assertIsNone(prefill_gate().held, 'the prefill gate is free for the next arrival')
        self.assertIsNone(case.lifecycle.hook)
        case.capture.close.assert_called_once()
        # the scheduler aborted it in the same step: its seed goes out with finish_reason abort
        self.assertEqual(scheduler.finished, [('request', 'FINISHED_ABORTED')])
        [output] = outputs[0].outputs
        self.assertEqual((output.request_id, output.new_token_ids, output.finish_reason), ('request', [10], 'abort'))
        self.assertEqual(quarantine.holder().pending, {})
        # the next step names it finished and admits the next request normally
        case.build.side_effect = None
        case.build.return_value = case.bridge
        self.assertIsNone(case.worker.execute_model(self.next_request(case)))
        self.assertEqual(case.lifecycle.request_id, 'second')

    def test_on_a_device_side_bridge_failure_still_fails_the_engine(self):
        for failure in (RuntimeError('trace capture failed'), ValueError('Native GDN buffers changed'),
                        # engine and runner invariants from the factory: plain ValueErrors
                        ValueError('Valid target-selected prefill seed required'),
                        ValueError('Terminal prefill must finish without allocating a verifier')):
            with self.subTest(failure=failure):
                case = self.fixture(ON, fake_scheduler_class())
                case.build.side_effect = failure
                case.worker.execute_model(case.scheduled)
                with self.assertRaises(type(failure)):
                    case.worker.sample_tokens(None)
                self.assertTrue(case.lifecycle.failed)
                self.assertEqual(quarantine.holder().pending, {})

    def test_on_a_refusal_while_others_decode_leaves_their_hook_alone(self):
        scheduler_class = fake_scheduler_class()
        case = self.fixture(ON, scheduler_class)
        case.lifecycle.packed_step = lambda entries, *, cancelled: []
        sampler = case.worker.model_runner.sample_tokens   # the hook replaces it once it is live
        case.worker.execute_model(case.scheduled)
        case.worker.sample_tokens(None)
        hook = case.lifecycle.hook
        self.assertEqual(case.lifecycle.decoding_ids, ['request'])
        second = self.next_request(case, finished=())
        # A refusal of the request's own terms (serving_request_factory.RequestRefused covers
        # only its sampling contract, budget and page table; a bad seed is engine-fatal).
        case.build.side_effect = RequestRefused('Scheduler must own capture warmup pages before verifier allocation')
        sampler.return_value = SimpleNamespace(req_ids=['second'], sampled_token_ids=[[11]])
        case.worker.model_runner.requests['second'] = SimpleNamespace(output_token_ids=[11],
                                                                      prompt_token_ids=[1] * 4096)
        scheduler = scheduler_class()
        scheduler.requests['second'] = SimpleNamespace(client_index=0)
        case.worker.execute_model(second)
        result = case.worker.sample_tokens(None)
        scheduler.update_from_output(second, result)
        self.assertEqual(scheduler.finished, [('second', 'FINISHED_ABORTED')])
        self.assertIs(case.lifecycle.hook, hook)
        self.assertEqual(sorted(hook.bridges), ['request'])
        self.assertEqual(case.lifecycle.decoding_ids, ['request'])
        self.assertFalse(case.lifecycle.failed)


class AdmissionRefusalTests(QuarantineCase):
    def test_off_a_sampling_refusal_at_admission_fails_the_engine_before_the_prefill(self):
        case = self.fixture(OFF, fake_scheduler_class())
        case.sampling.temperature = 0.7
        with self.assertRaisesRegex(ValueError, 'greedy single-sequence'):
            case.worker.execute_model(case.scheduled)
        self.assertTrue(case.lifecycle.failed)
        case.capture.capture.assert_not_called()

    def test_on_it_prefills_then_ends_at_its_first_token_as_aborted_without_a_bridge(self):
        scheduler_class = fake_scheduler_class()
        case = self.fixture(ON, scheduler_class)
        case.sampling.temperature = 0.7
        scheduler = scheduler_class()
        result, outputs = self.step(case, scheduler)
        case.capture.capture.assert_called_once()   # the runner accounted the prefill it scheduled
        case.build.assert_not_called()
        self.assertEqual(result.sampled_token_ids, [[10]])
        self.assertEqual(scheduler.finished, [('request', 'FINISHED_ABORTED')])
        self.assertEqual(outputs[0].outputs[0].finish_reason, 'abort')
        self.assertFalse(case.lifecycle.failed)
        self.assertIsNone(case.lifecycle.refused)

    def test_on_a_refused_request_that_stops_at_its_first_token_anyway_is_not_aborted(self):
        for tokens, max_tokens in (([99], 256), ([10], 1)):
            with self.subTest(tokens=tokens, max_tokens=max_tokens):
                case = self.fixture(ON, fake_scheduler_class())
                case.sampling.temperature = 0.7
                case.sampling.max_tokens = max_tokens
                case.bridge.state.output_token_ids[:] = tokens
                case.worker.model_runner.sample_tokens.return_value.sampled_token_ids = [tokens]
                case.worker.execute_model(case.scheduled)
                case.worker.sample_tokens(None)
                self.assertEqual(quarantine.holder().pending, {})
                self.assertIsNone(case.lifecycle.request_id)
                case.build.assert_not_called()

    def test_on_a_step_shape_refusal_is_still_fatal(self):
        case = self.fixture(ON, fake_scheduler_class())
        case.scheduled.num_scheduled_tokens = {'request': 4096, 'other': 1}
        with self.assertRaisesRegex(ValueError, 'whole or first chunk'):
            case.worker.execute_model(case.scheduled)
        self.assertTrue(case.lifecycle.failed)


class FirstTokenTerminalTests(QuarantineCase):
    def test_on_max_tokens_one_builds_no_bridge_and_frees_the_prefill(self):
        """The platform's warmup (SKILL: max_tokens 1). vLLM stops it at this token."""
        case = self.fixture(ON, fake_scheduler_class())
        case.sampling.max_tokens = 1
        case.worker.execute_model(case.scheduled)
        self.assertEqual(case.worker.sample_tokens(None).sampled_token_ids, [[10]])
        case.build.assert_not_called()
        case.capture.close.assert_called_once()
        self.assertIsNone(case.lifecycle.request_id)
        self.assertIsNone(case.lifecycle.hook)
        self.assertFalse(case.lifecycle.failed)
        self.assertEqual(quarantine.holder().pending, {}, 'finished, not aborted')
        self.assertIsNone(case.worker.execute_model(self.next_request(case)))

    def test_on_a_prompt_whose_first_token_fills_the_context_builds_no_bridge(self):
        case = self.fixture(ON, fake_scheduler_class(), max_model_len=4352)
        case.bridge.state.prompt_token_ids = [1] * 4351
        case.worker.execute_model(case.scheduled)
        case.worker.sample_tokens(None)
        case.build.assert_not_called()

    def test_on_room_left_the_bridge_is_built(self):
        case = self.fixture(ON, fake_scheduler_class())
        case.sampling.max_tokens = 2
        case.worker.execute_model(case.scheduled)
        case.worker.sample_tokens(None)
        case.build.assert_called_once()

    def test_off_max_tokens_one_still_builds_the_bridge_as_today(self):
        case = self.fixture(OFF, fake_scheduler_class())
        case.sampling.max_tokens = 1
        case.worker.execute_model(case.scheduled)
        case.worker.sample_tokens(None)
        case.build.assert_called_once()


class ConsumerTests(QuarantineCase):
    def test_a_consumer_that_never_ran_fails_the_next_step_loudly(self):
        """Registered, but the engine's scheduler is not the class that was wrapped: the next
        step still schedules the request, which must not decode unbridged."""
        case = self.fixture(ON, fake_scheduler_class())
        case.build.side_effect = RequestRefused('refused')
        case.worker.execute_model(case.scheduled)
        case.worker.sample_tokens(None)
        self.assertIn('request', quarantine.holder().pending)
        decode = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=['request']), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={'request': 1}, total_num_scheduled_tokens=1)
        with self.assertRaisesRegex(ValueError, 'never aborted'):
            case.worker.execute_model(decode)
        self.assertTrue(case.lifecycle.failed)

    def test_an_unresolvable_scheduler_keeps_refusals_fatal(self):
        case = self.fixture(ON, 'no_such_module_for_quarantine.Scheduler')
        self.assertIsNone(case.lifecycle.quarantine)
        case.build.side_effect = RequestRefused('refused')
        case.worker.execute_model(case.scheduled)
        with self.assertRaises(RequestRefused):
            case.worker.sample_tokens(None)
        self.assertTrue(case.lifecycle.failed)

    def test_install_wraps_the_named_class_once_and_marks_itself_live_on_first_use(self):
        scheduler_class, log = fake_scheduler_class(), Mock()
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=scheduler_class))
        original = scheduler_class.update_from_output
        name = quarantine.install(config, log=log)
        self.assertTrue(name.endswith('FakeScheduler'))
        self.assertEqual(quarantine.install(config, log=log), name)
        self.assertIs(scheduler_class.update_from_output.__wrapped__, original, 'wrapped exactly once')
        self.assertFalse(quarantine.holder().live)
        scheduler = scheduler_class()
        empty = SimpleNamespace(req_ids=[], sampled_token_ids=[])
        scheduler.update_from_output(None, empty)
        scheduler.update_from_output(None, empty)
        self.assertTrue(quarantine.holder().live)
        lines = [call.args[0] for call in log.call_args_list]
        self.assertEqual(sum('consumer live' in line for line in lines), 1, lines)

    def test_the_scheduler_class_is_the_one_the_config_names(self):
        class Named(object):
            pass

        module = SimpleNamespace(TTScheduler=Named)
        importer = Mock(return_value=module)
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls='vllm_tt_plugin.scheduler.TTScheduler'))
        self.assertIs(quarantine.scheduler_class(config, importer), Named)
        importer.assert_called_once_with('vllm_tt_plugin.scheduler')
        self.assertIs(quarantine.scheduler_class(SimpleNamespace(scheduler_config=SimpleNamespace(
            scheduler_cls=Named))), Named)
        getter = SimpleNamespace(scheduler_cls=None, get_scheduler_cls=lambda: Named)
        self.assertIs(quarantine.scheduler_class(SimpleNamespace(scheduler_config=getter)), Named)
        default = Mock(return_value=SimpleNamespace(Scheduler=Named))
        self.assertIs(quarantine.scheduler_class(SimpleNamespace(scheduler_config=SimpleNamespace()), default), Named)
        default.assert_called_once_with('vllm.v1.core.sched.scheduler')
        with self.assertRaises(ValueError):
            quarantine.scheduler_class(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls='Bare')))

    def test_abort_marks_or_adds_the_finishing_output_and_skips_requests_already_finished(self):
        quarantine.holder().installed = 'x'
        scheduler = fake_scheduler_class()()
        scheduler.requests = {'seeded': SimpleNamespace(client_index=0), 'silent': SimpleNamespace(client_index=2)}
        outputs = {0: Outputs([Output('other', [5]), Output('seeded', [10])])}
        for request_id in ('seeded', 'silent', 'gone'):
            quarantine.register(request_id, 'why')
        log = Mock()
        result = quarantine.abort_quarantined(scheduler, outputs, kinds=KINDS, log=log)
        self.assertIs(result, outputs)
        self.assertEqual([(output.request_id, output.finish_reason) for output in outputs[0].outputs],
                         [('other', None), ('seeded', 'abort')])
        self.assertEqual([(output.request_id, output.new_token_ids, output.finish_reason)
                          for output in outputs[2].outputs], [('silent', [], 'abort')])
        self.assertEqual(scheduler.finished, [('seeded', 'FINISHED_ABORTED'), ('silent', 'FINISHED_ABORTED')])
        self.assertEqual(quarantine.holder().pending, {})
        self.assertIn('already finished', log.call_args_list[-1].args[0])

    def test_nothing_registered_is_a_no_op(self):
        scheduler, outputs = fake_scheduler_class()(), {}
        self.assertIs(quarantine.abort_quarantined(scheduler, outputs, kinds=KINDS), outputs)
        self.assertEqual((outputs, scheduler.finished), ({}, []))

    def test_register_refuses_without_a_consumer_and_unconsumed_forgets_finished_requests(self):
        with self.assertRaisesRegex(ValueError, 'No scheduler consumes'):
            quarantine.register('request', 'why')
        quarantine.holder().installed = 'x'
        quarantine.register('a', 'why')
        quarantine.register('b', 'why')
        scheduled = SimpleNamespace(finished_req_ids={'a'}, scheduled_cached_reqs=SimpleNamespace(req_ids=['b']),
                                    num_scheduled_tokens={'b': 1})
        self.assertEqual(quarantine.unconsumed(scheduled), ['b'])
        self.assertEqual(list(quarantine.holder().pending), ['b'])


PLUGIN_SCHEDULER_FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'plugin_scheduler.py'


def plugin_scheduler_class():
    """TTScheduler as the engine runs it: the TT platform names vllm_tt_plugin.scheduler.TTScheduler,
    a TTScheduler(AsyncScheduler) with placeholder bookkeeping, not vLLM's stock Scheduler. Built
    from fixtures/plugin_scheduler.py - the pinned plugin's class body as probe 35665853903 dumped
    it, the source the lever-N scheduler tests load - executed over the INSTALLED vLLM's own
    AsyncScheduler, request queues and output types. TTSchedulingMode is the one name the reduced
    dump uses without defining; the plugin's scheduler.py defines exactly these three members.
    A fresh class per call, so wrapping it never leaks into another test."""
    import enum
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
    from vllm.v1.request import Request

    class TTSchedulingMode(enum.Enum):
        DEFAULT = 'default'
        PREFILL_ONLY = 'prefill_only'
        DECODE_ONLY = 'decode_only'

    module = types.ModuleType('qwen_quarantine_plugin_scheduler')
    module.__dict__.update(AsyncScheduler=AsyncScheduler, SchedulerOutput=SchedulerOutput, Request=Request,
                           RequestQueue=RequestQueue, create_request_queue=create_request_queue,
                           TTSchedulingMode=TTSchedulingMode)
    exec(compile(PLUGIN_SCHEDULER_FIXTURE.read_text(encoding='utf-8'), str(PLUGIN_SCHEDULER_FIXTURE), 'exec'),
         module.__dict__)
    return module.TTScheduler


class InstalledVllmQuarantineTests(unittest.TestCase):
    """The consumer against the pinned vLLM (0.25.1 in qwen-fast-vllm-cpu.yml) and the scheduler class
    the engine actually builds, TTScheduler(AsyncScheduler): EngineCoreOutput's mutability, the
    per-client output dict and finish_requests on a request carrying async placeholders are all
    vLLM's own here. Skipped wherever vLLM is not installed - which is the PR's CPU suite: this runs
    only on an experiment/fast-vllm-cpu-v* tag, so run one before a c2 hardware run."""

    def setUp(self):
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest('vLLM is not installed')
        sys.modules.pop(quarantine.HOLDER_KEY, None)
        self.addCleanup(sys.modules.pop, quarantine.HOLDER_KEY, None)

    def schedulers(self):
        """The fixture's TTScheduler, and the installed plugin's own when this environment has it
        (qwen-fast-vllm-cpu.yml installs upstream-plugin with pip -e)."""
        classes = [('fixture', plugin_scheduler_class())]
        try:
            from vllm_tt_plugin.scheduler import TTScheduler
        except ImportError:
            pass
        else:
            classes.append(('installed plugin', type('TTScheduler', (TTScheduler,), {})))
        return classes

    def test_the_tt_scheduler_aborts_a_quarantined_request_after_its_first_token(self):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.engine import FinishReason
        from vllm.v1.request import Request, RequestStatus
        import test_serving_scheduler

        for source, scheduler_type in self.schedulers():
            with self.subTest(source=source):
                sys.modules.pop(quarantine.HOLDER_KEY, None)
                self.assertTrue(issubclass(scheduler_type, AsyncScheduler), 'the class the engine builds')
                self.assertEqual(scheduler_type.__name__, 'TTScheduler')
                case = test_serving_scheduler.RealSchedulerTests()
                case.scheduler_type = scheduler_type
                scheduler = case.scheduler()
                log = Mock()
                quarantine.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=scheduler_type)),
                                   log=log)
                request = Request('request', [42] * 4096, SamplingParams(temperature=0, max_tokens=256), None)
                scheduler.add_request(request)
                scheduled = scheduler.schedule()
                self.assertEqual(scheduled.num_scheduled_tokens, {'request': 4096})
                self.assertGreaterEqual(request.num_output_placeholders, 1, "AsyncScheduler's placeholder for the seed")
                quarantine.register('request', 'test')
                outputs = scheduler.update_from_output(scheduled, case.output('request', [100]))
                self.assertIsInstance(outputs, dict, 'EngineCoreOutputs per client index')
                finishing = [output for batch in outputs.values() for output in batch.outputs
                             if output.request_id == 'request']
                self.assertEqual(len(finishing), 1)
                self.assertEqual(finishing[0].new_token_ids, [100])
                self.assertEqual(finishing[0].finish_reason, FinishReason.ABORT, 'EngineCoreOutput took the mutation')
                self.assertEqual(request.status, RequestStatus.FINISHED_ABORTED)
                self.assertNotIn('request', scheduler.requests)
                self.assertEqual(quarantine.holder().pending, {})
                # the positive control the c2 gate greps for: from inside the wrapper, naming the class
                self.assertIn(call('[PINDIAG] request quarantine consumer live in {}', 'TTScheduler'),
                              log.call_args_list)
                following = scheduler.schedule()
                self.assertEqual(following.finished_req_ids, {'request'})
                self.assertEqual(following.total_num_scheduled_tokens, 0)
                # and the seat is free: the next request prefills normally
                replacement = Request('replacement', [43] * 4096, SamplingParams(temperature=0, max_tokens=256), None)
                scheduler.add_request(replacement)
                self.assertEqual(scheduler.schedule().num_scheduled_tokens, {'replacement': 4096})


if __name__ == '__main__':
    unittest.main()
