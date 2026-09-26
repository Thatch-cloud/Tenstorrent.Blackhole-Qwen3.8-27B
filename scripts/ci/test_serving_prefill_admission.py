"""One fresh prompt per prefill step under QWEN_FAST_ANY_REQUEST (serving_prefill_admission).

Run 36211578069 (G4 ladder, profile c2, nine users submitted at once) died in serving_lifecycle._execute
on `Fast serving requires one complete fresh prefill: prefill_slot=None new=[three ids] cached=[]`. The
stock TTScheduler, which the TT platform installs over the fast path's own OneInFlightScheduler, had
batched three fresh prompts into one prefill step.

TTScheduler here is fixtures/plugin_scheduler.py: the pinned plugin's class body (bf77cd63
scheduler.py, the source the lever-N graft tests load). It is executed over a reduced model of vLLM
0.25.1's Scheduler.schedule (FakeVllmScheduler), so the plugin's own prefill/decode split,
_schedule_prefill_only and decode fallback run unmodified. The tests check four things:
- the runtime patch applies lever_n_model_patch.patch_scheduler's rule, call for call, against the
  graft on the same source (GraftParityTests);
- three simultaneous fresh arrivals are admitted one per step, and running decodes are neither dropped
  nor starved (ArrivalTests);
- the lifecycle installs it only under the flag, and the step it produces is one the lifecycle accepts,
  where the unpatched step reproduces the run's refusal (LifecycleTests);
- the same on the INSTALLED vLLM and plugin (InstalledVllmAdmissionTests). That class is skipped where
  vLLM is not installed: it runs in qwen-fast-vllm-cpu.yml.
"""

import enum
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import lever_n_model_patch
import serving_prefill_admission as admission
import serving_request_quarantine as quarantine
import serving_lifecycle
import test_serving_request_quarantine

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'plugin_scheduler.py'
ON = {'QWEN_FAST_ANY_REQUEST': '1'}
OFF = {'QWEN_FAST_ANY_REQUEST': '0'}
C2_BATCHED_TOKENS = 131328   # qwen_c2_profiles.json c2: max-num-batched-tokens


class Queue(list):
    """vLLM's FCFS RequestQueue, as far as the scheduler code under test uses it."""

    def prepend_requests(self, requests):
        self[:0] = list(requests)


def create_request_queue(policy):
    return Queue()


class TTSchedulingMode(enum.Enum):
    """The one name the reduced fixture uses without defining; the plugin defines these three members."""
    DEFAULT = 'default'
    DECODE_ONLY = 'decode_only'
    PREFILL_ONLY = 'prefill_only'


def plugin_class(base, source=None, name='qwen_admission_plugin_scheduler'):
    """TTScheduler from the fixture's text (or `source`, e.g. the grafted text) over `base`. A fresh
    class per call, so patching one never leaks into another test."""
    module = types.ModuleType(name)
    module.__dict__.update(AsyncScheduler=base, SchedulerOutput=object, Request=object, RequestQueue=Queue,
                           create_request_queue=create_request_queue, TTSchedulingMode=TTSchedulingMode,
                           logger=SimpleNamespace(info=lambda *args, **kwargs: None))
    text = FIXTURE.read_text(encoding='utf-8') if source is None else source
    exec(compile(text, str(FIXTURE), 'exec'), module.__dict__)
    return module.TTScheduler


def configured(cls):
    return SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=cls))


class gate(object):
    """The lifecycle's prefill gate holding `held` for the block (None: absent, as before the gate
    existed); whatever was there before is put back afterwards."""

    def __init__(self, held):
        self.held = held

    def __enter__(self):
        self.saved = sys.modules.pop(admission.GATE_KEY, None)
        if self.held is not None:
            holder = types.ModuleType(admission.GATE_KEY)
            holder.held = self.held
            sys.modules[admission.GATE_KEY] = holder
        return self

    def __exit__(self, *failure):
        sys.modules.pop(admission.GATE_KEY, None)
        if self.saved is not None:
            sys.modules[admission.GATE_KEY] = self.saved
        return False


class GateFreeCase(unittest.TestCase):
    """Every test starts with no prefill held, whatever an earlier test module left in the gate."""

    def setUp(self):
        clear = gate(None)
        clear.__enter__()
        self.addCleanup(clear.__exit__, None, None, None)


class FakeRequest(object):
    def __init__(self, request_id, prompt_tokens=4096, sampling_params=None):
        self.request_id, self.prompt_tokens, self.sampling_params = request_id, prompt_tokens, sampling_params
        self.num_computed_tokens = 0
        # No chunked prefill in any c2 profile: a request is never mid-prompt across steps.
        self.is_prefill_chunk = False


class FakeVllmScheduler(object):
    """vLLM 0.25.1's Scheduler.schedule reduced to what a prefill cap touches.

    The running loop schedules every request in `running`; with no chunked prefill each is a decode
    of one token. The waiting loop admits whole prompts in FCFS order while len(running) <
    max_num_running_reqs (scheduler.py:640-645) and the prompt fits the token budget. At the end it
    asserts len(running) <= max_num_running_reqs (scheduler.py:1027)."""

    def __init__(self, max_num_seqs=4, max_num_batched_tokens=C2_BATCHED_TOKENS):
        self.running, self.waiting, self.skipped_waiting = [], Queue(), Queue()
        self.policy = 'fcfs'
        self.max_num_running_reqs = max_num_seqs
        self.max_num_scheduled_tokens = max_num_batched_tokens
        self.requests = {}

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def finish(self, request_id):
        request = self.requests.pop(request_id)
        self.running.remove(request)

    def schedule(self):
        budget, counts, new, cached = self.max_num_scheduled_tokens, {}, [], []
        for request in self.running:
            if budget <= 0:
                break
            counts[request.request_id] = 1
            budget -= 1
            cached.append(request.request_id)
        while (self.waiting or self.skipped_waiting) and budget > 0:
            if len(self.running) >= self.max_num_running_reqs:
                break
            queue = self.skipped_waiting or self.waiting
            request = queue[0]
            if request.prompt_tokens > budget:
                break
            queue.pop(0)
            self.running.append(request)
            counts[request.request_id] = request.prompt_tokens
            budget -= request.prompt_tokens
            new.append(request)
        assert len(self.running) <= self.max_num_running_reqs
        for request in new:
            request.num_computed_tokens = request.prompt_tokens
        return SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id, prompt_token_ids=[1] * request.prompt_tokens,
                                                num_computed_tokens=0, mm_features=[], prompt_embeds=None,
                                                lora_request=None, sampling_params=request.sampling_params)
                                for request in new],
            scheduled_cached_reqs=SimpleNamespace(req_ids=cached), num_scheduled_tokens=counts,
            total_num_scheduled_tokens=sum(counts.values()), finished_req_ids=set(), scheduled_spec_decode_tokens={})

    def update_from_output(self, scheduler_output, model_runner_output):
        return {}


class RecordingBase(object):
    """A base scheduler that records what it is handed mid-call, and can re-queue a request during
    the call as a preemption would."""

    def schedule(self):
        seen = self.qwen_seen
        seen['capacity'] = self.max_num_running_reqs
        seen['waiting'] = list(self.waiting)
        seen['skipped'] = list(self.skipped_waiting)
        seen['running'] = [request.name for request in self.running]
        if self.qwen_requeue:
            self.waiting.append('requeued')
        return SimpleNamespace(total_num_scheduled_tokens=1)


def new_ids(output):
    return [request.req_id for request in output.scheduled_new_reqs]


def cached_ids(output):
    """The decodes a step carries, sorted: the plugin re-appends the hidden decodes after the admitted
    prompt, so `running` order is not arrival order."""
    return sorted(output.scheduled_cached_reqs.req_ids)


def names(requests):
    return sorted(request.request_id for request in requests)


def patched_plugin(log=None):
    cls = plugin_class(FakeVllmScheduler)
    admission.install(configured(cls), log=Mock() if log is None else log)
    return cls


class RuleTests(GateFreeCase):
    def test_the_rule_is_the_grafts(self):
        self.assertEqual(admission.admission(0, None), (1, False), 'nothing in flight: one fresh prompt, queues visible')
        self.assertEqual(admission.admission(0, 'cmpl-a'), (0, True), 'the lifecycle holds a prefill: nobody')
        for held in (None, 'cmpl-a'):
            self.assertEqual(admission.admission(2, held), (2, True), 'partials continue alone, whatever the gate')
        for bad in (-1, 1.0, None, True):
            with self.subTest(partials=bad), self.assertRaises(ValueError):
                admission.admission(bad, None)

    def test_the_written_cap_nets_out_to_the_grafts_capacity(self):
        """The plugin computes max(0, written - decodes); writing min(saved, allowed + decodes) must
        land on the graft's max(0, min(saved - decodes, allowed)) everywhere."""
        for saved in range(0, 9):
            for decodes in range(0, saved + 1):
                for allowed in range(0, 4):
                    written = min(saved, allowed + decodes)
                    self.assertEqual(max(0, written - decodes), admission.waiting_capacity(saved, decodes, allowed),
                                     (saved, decodes, allowed))

    def test_the_gate_is_the_lifecycles(self):
        self.assertEqual(admission.GATE_KEY, serving_lifecycle.PREFILL_GATE_KEY)
        self.assertIsNone(admission.gate_held())
        serving_lifecycle.prefill_gate().held = 'cmpl-a'
        self.assertEqual(admission.gate_held(), 'cmpl-a')

    def test_an_unreadable_gate_reads_as_not_held(self):
        class Broken(object):
            @property
            def held(self):
                raise RuntimeError('gate unreadable')

        self.assertIsNone(admission.gate_held({admission.GATE_KEY: Broken()}))


class GraftParityTests(GateFreeCase):
    """The runtime patch against lever_n_model_patch.patch_scheduler, on the same plugin source and
    the same scheduler states. The check covers what the base scheduler sees mid-call (the capacity,
    each queue, which requests are in `running`) and what is left afterwards (the queues, including a
    request the base scheduler re-queues during the call, `running`, and the capacity)."""

    def observe(self, cls, partials, decodes, held, saved_max, requeue):
        scheduler = cls.__new__(cls)
        scheduler.qwen_seen, scheduler.qwen_requeue = {}, requeue
        scheduler.running = ([SimpleNamespace(name='partial-%d' % i, is_prefill_chunk=True) for i in range(partials)]
                             + [SimpleNamespace(name='decode-%d' % i, is_prefill_chunk=False) for i in range(decodes)])
        scheduler.waiting, scheduler.skipped_waiting = Queue(['w1', 'w2']), Queue(['s1'])
        scheduler.policy, scheduler.max_num_running_reqs = 'fcfs', saved_max
        with gate(held):
            scheduler._schedule_prefill_only()
        return dict(scheduler.qwen_seen, after=dict(
            waiting=list(scheduler.waiting), skipped=list(scheduler.skipped_waiting),
            running=[request.name for request in scheduler.running], capacity=scheduler.max_num_running_reqs))

    def test_the_patch_matches_the_graft_in_every_state(self):
        text = FIXTURE.read_text(encoding='utf-8')
        grafted_text = lever_n_model_patch.patch_scheduler(text)
        compared = 0
        for saved_max in (1, 2, 4, 8):
            for partials in range(0, 3):
                for decodes in range(0, 5):
                    if partials + decodes > saved_max:
                        continue
                    for held in (None, 'cmpl-held'):
                        for requeue in (False, True):
                            state = dict(saved_max=saved_max, partials=partials, decodes=decodes, held=held,
                                         requeue=requeue)
                            with self.subTest(**state):
                                grafted = plugin_class(RecordingBase, grafted_text)
                                runtime = plugin_class(RecordingBase)
                                admission.install(configured(runtime), log=Mock())
                                self.assertEqual(self.observe(runtime, partials, decodes, held, saved_max, requeue),
                                                 self.observe(grafted, partials, decodes, held, saved_max, requeue))
                                compared += 1
        self.assertGreater(compared, 100)

    def test_the_parity_check_can_fail(self):
        """Positive control: the UNPATCHED plugin differs from the graft with nothing in flight and two
        decodes (its waiting loop could admit two fresh prompts; the graft's admits one)."""
        grafted = plugin_class(RecordingBase, lever_n_model_patch.patch_scheduler(FIXTURE.read_text(encoding='utf-8')))
        stock = plugin_class(RecordingBase)
        self.assertEqual(self.observe(grafted, 0, 2, None, 4, False)['capacity'], 1)
        self.assertEqual(self.observe(stock, 0, 2, None, 4, False)['capacity'], 2)


class ArrivalTests(GateFreeCase):
    def arrive(self, scheduler, *request_ids):
        for request_id in request_ids:
            scheduler.add_request(FakeRequest(request_id))

    def test_unpatched_three_simultaneous_arrivals_share_one_prefill_step(self):
        """The negative control: run 36211578069's step shape, from the plugin's own class."""
        scheduler = plugin_class(FakeVllmScheduler)()
        self.arrive(scheduler, 'A', 'B', 'C')
        self.assertEqual(new_ids(scheduler.schedule()), ['A', 'B', 'C'])

    def test_three_simultaneous_arrivals_are_admitted_one_per_step(self):
        scheduler = patched_plugin()()
        self.arrive(scheduler, 'A', 'B', 'C')
        steps = [scheduler.schedule() for _ in range(4)]
        self.assertEqual([(new_ids(step), cached_ids(step)) for step in steps],
                         [(['A'], []), (['B'], []), (['C'], []), ([], ['A', 'B', 'C'])])
        for step in steps[:3]:
            # the lifecycle's step shape for an admission: one new request, nothing cached, its tokens only
            self.assertEqual(set(step.num_scheduled_tokens), set(new_ids(step)))
        self.assertEqual(list(scheduler.waiting), [])
        self.assertEqual(scheduler.max_num_running_reqs, 4, 'the configured capacity is restored after every step')

    def test_running_decodes_are_never_dropped_or_starved(self):
        """A decodes. B, C and D arrive together, then E. On TT a step is all prefill or all decode, so
        each one-prompt prefill step pauses A, as the stock scheduler's single prefill step did. After
        that, every step decodes every running request. With all four seats taken, E's prefill attempt
        admits nothing and the plugin's fallback runs a decode step instead: E waits, and the decoders
        are not held. When a seat frees, E alone is admitted."""
        scheduler = patched_plugin()()
        self.arrive(scheduler, 'A')
        self.assertEqual(new_ids(scheduler.schedule()), ['A'])
        self.assertEqual(cached_ids(scheduler.schedule()), ['A'])
        self.arrive(scheduler, 'B', 'C', 'D')
        steps = [scheduler.schedule() for _ in range(3)]
        self.assertEqual([(new_ids(step), cached_ids(step)) for step in steps], [(['B'], []), (['C'], []), (['D'], [])])
        self.assertEqual(names(scheduler.running), ['A', 'B', 'C', 'D'], 'every admitted request stays running')
        self.arrive(scheduler, 'E')
        for _ in range(3):
            step = scheduler.schedule()
            self.assertEqual((new_ids(step), cached_ids(step)), ([], ['A', 'B', 'C', 'D']))
        self.assertEqual(names(scheduler.waiting), ['E'])
        scheduler.finish('B')
        admitted = scheduler.schedule()
        self.assertEqual((new_ids(admitted), cached_ids(admitted)), (['E'], []))
        self.assertEqual(cached_ids(scheduler.schedule()), ['A', 'C', 'D', 'E'])

    def test_a_held_gate_admits_nobody_and_the_decoders_still_run(self):
        scheduler = patched_plugin()()
        self.arrive(scheduler, 'A')
        scheduler.schedule()
        self.arrive(scheduler, 'B')
        with gate('cmpl-held'):
            step = scheduler.schedule()
        self.assertEqual((new_ids(step), cached_ids(step)), ([], ['A']))
        self.assertEqual(names(scheduler.waiting), ['B'], 'the hidden queue came back')
        self.assertEqual(new_ids(scheduler.schedule()), ['B'])

    def test_the_markers_name_the_class_at_install_and_fire_live_once(self):
        log = Mock()
        cls = plugin_class(FakeVllmScheduler)
        name = admission.install(configured(cls), log=log)
        self.assertEqual(name, 'qwen_admission_plugin_scheduler.TTScheduler')
        scheduler = cls()
        self.arrive(scheduler, 'A', 'B')
        for _ in range(3):
            scheduler.schedule()
        lines = [(entry.args[0], entry.args[1:]) for entry in log.call_args_list]
        self.assertEqual([values for line, values in lines if line == admission.INSTALLED + '{}'], [(name,)])
        self.assertEqual([values for line, values in lines if line == admission.LIVE + '{}'], [('TTScheduler',)])


class InstallTests(GateFreeCase):
    def test_install_wraps_the_named_class_once(self):
        cls, log = plugin_class(FakeVllmScheduler), Mock()
        original = cls._schedule_prefill_only
        name = admission.install(configured(cls), log=log)
        self.assertEqual(admission.install(configured(cls), log=log), name)
        self.assertIs(cls._schedule_prefill_only.__wrapped__, original, 'wrapped exactly once')
        self.assertEqual(sum(entry.args[0] == admission.INSTALLED + '{}' for entry in log.call_args_list), 1)
        subclass = type('Sub', (cls,), {})
        admission.install(configured(subclass), log=log)
        self.assertNotIn(admission.METHOD, vars(subclass), 'a subclass inherits the wrapper, it is not wrapped again')

    def test_the_class_is_the_one_the_config_names(self):
        cls = plugin_class(FakeVllmScheduler)
        importer = Mock(return_value=SimpleNamespace(TTScheduler=cls))
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls='vllm_tt_plugin.scheduler.TTScheduler'))
        admission.install(config, importer=importer, log=Mock())
        importer.assert_called_once_with('vllm_tt_plugin.scheduler')
        self.assertTrue(getattr(cls._schedule_prefill_only, admission.WRAPPED))

    def test_a_scheduler_without_a_prefill_step_is_refused(self):
        """vLLM's stock Scheduler (what an unset scheduler_cls resolves to) has no _schedule_prefill_only."""
        with self.assertRaisesRegex(ValueError, 'has no _schedule_prefill_only'):
            admission.install(configured(test_serving_request_quarantine.fake_scheduler_class()), log=Mock())

    def test_the_queues_and_capacity_are_restored_when_scheduling_raises(self):
        cls = plugin_class(FakeVllmScheduler)
        admission.install(configured(cls), log=Mock())
        scheduler = cls()
        scheduler.add_request(FakeRequest('A'))
        waiting = scheduler.waiting
        with patch.object(FakeVllmScheduler, 'schedule', side_effect=RuntimeError('boom')), gate('cmpl-held'), \
                self.assertRaises(RuntimeError):
            scheduler._schedule_prefill_only()
        self.assertIs(scheduler.waiting, waiting)
        self.assertEqual(names(waiting), ['A'])
        self.assertEqual(scheduler.max_num_running_reqs, 4)


class LifecycleTests(GateFreeCase):
    """The install point: FastServingLifecycle, beside the D2 quarantine consumer, only under the flag."""

    def setUp(self):
        super().setUp()
        sys.modules.pop(quarantine.HOLDER_KEY, None)
        self.addCleanup(sys.modules.pop, quarantine.HOLDER_KEY, None)

    def lifecycle(self, environ, scheduler_cls):
        return test_serving_request_quarantine.QuarantineCase.fixture(self, environ, scheduler_cls)

    def arrivals(self, case, scheduler, *request_ids):
        for request_id in request_ids:
            scheduler.add_request(FakeRequest(request_id, 4096, case.sampling))

    def test_off_the_scheduler_is_untouched_and_simultaneous_arrivals_still_die_as_in_the_run(self):
        """exact (and every profile without the flag) schedules exactly as before, so its engine still
        refuses the batched step. This reproduces run 36211578069's refusal."""
        for environ in ({}, OFF):
            with self.subTest(environ=environ):
                cls = plugin_class(FakeVllmScheduler)
                original = cls._schedule_prefill_only
                case = self.lifecycle(environ, cls)
                self.assertIsNone(case.lifecycle.admission)
                self.assertIs(cls._schedule_prefill_only, original)
                scheduler = cls()
                self.arrivals(case, scheduler, 'A', 'B', 'C')
                with self.assertRaisesRegex(ValueError, r"one complete fresh prefill: prefill_slot=None "
                                                        r"new=\['A', 'B', 'C'\] cached=\[\]"):
                    case.worker.execute_model(scheduler.schedule())
                self.assertTrue(case.lifecycle.failed)

    def test_on_the_lifecycle_installs_it_and_the_admitted_step_is_one_it_serves(self):
        cls = plugin_class(FakeVllmScheduler)
        case = self.lifecycle(ON, cls)
        self.assertEqual(case.lifecycle.admission, 'qwen_admission_plugin_scheduler.TTScheduler')
        self.assertTrue(getattr(cls._schedule_prefill_only, admission.WRAPPED))
        self.assertIs(case.lifecycle.quarantine, quarantine, 'the quarantine consumer is installed beside it')
        scheduler = cls()
        self.arrivals(case, scheduler, 'A', 'B', 'C')
        step = scheduler.schedule()
        self.assertEqual((new_ids(step), cached_ids(step)), (['A'], []))
        self.assertIsNone(case.worker.execute_model(step))
        self.assertFalse(case.lifecycle.failed)
        self.assertEqual(case.lifecycle.request_id, 'A')
        self.assertEqual(names(scheduler.waiting), ['B', 'C'])

    def test_on_a_class_it_cannot_patch_leaves_the_lifecycle_serving_as_before(self):
        for scheduler_cls in (test_serving_request_quarantine.fake_scheduler_class(), 'no_such_module_for_admission.X'):
            with self.subTest(scheduler_cls=scheduler_cls):
                case = self.lifecycle(ON, scheduler_cls)
                self.assertIsNone(case.lifecycle.admission)
                self.assertFalse(case.lifecycle.failed)


class InstalledVllmAdmissionTests(GateFreeCase):
    """Against the pinned vLLM (0.25.1 in qwen-fast-vllm-cpu.yml) and the class the engine builds,
    TTScheduler(AsyncScheduler): vLLM's own waiting loop, request queues, KV admission and
    placeholders. Skipped where vLLM is not installed.

    Built with test_serving_scheduler.RealSchedulerTests (max_num_seqs 1, 4352 batched tokens), then
    given four seats by setting max_num_running_reqs, the only field the scheduling loop reads for
    it (scheduler.py:108, :644, :1027). Three 1024-token prompts fit one step's budget."""

    def setUp(self):
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest('vLLM is not installed')
        super().setUp()

    def schedulers(self):
        """The fixture's TTScheduler over the installed AsyncScheduler, and the installed plugin's own
        when this environment has it (a fresh subclass, so the wrap does not leak)."""
        classes = [('fixture', test_serving_request_quarantine.plugin_scheduler_class())]
        try:
            from vllm_tt_plugin.scheduler import TTScheduler
        except ImportError:
            pass
        else:
            classes.append(('installed plugin', type('TTScheduler', (TTScheduler,), {})))
        return classes

    def build(self, scheduler_type):
        import test_serving_scheduler

        case = test_serving_scheduler.RealSchedulerTests()
        case.scheduler_type = scheduler_type
        scheduler = case.scheduler()
        scheduler.max_num_running_reqs = 4
        return case, scheduler

    def arrive(self, scheduler, *request_ids):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        for request_id in request_ids:
            scheduler.add_request(Request(request_id, [42] * 1024, SamplingParams(temperature=0, max_tokens=256), None))

    def test_unpatched_the_installed_scheduler_batches_simultaneous_arrivals(self):
        for source, scheduler_type in self.schedulers():
            with self.subTest(source=source):
                _, scheduler = self.build(scheduler_type)
                self.arrive(scheduler, 'A', 'B', 'C')
                self.assertEqual(set(scheduler.schedule().num_scheduled_tokens), {'A', 'B', 'C'})

    def test_patched_three_simultaneous_arrivals_are_admitted_one_per_step(self):
        for source, scheduler_type in self.schedulers():
            with self.subTest(source=source):
                log = Mock()
                admission.install(configured(scheduler_type), log=log)
                case, scheduler = self.build(scheduler_type)
                self.arrive(scheduler, 'A', 'B', 'C')
                for request_id in ('A', 'B', 'C'):
                    step = scheduler.schedule()
                    self.assertEqual(step.num_scheduled_tokens, {request_id: 1024})
                    self.assertEqual([request.req_id for request in step.scheduled_new_reqs], [request_id])
                    self.assertEqual(list(step.scheduled_cached_reqs.req_ids), [])
                    scheduler.update_from_output(step, case.output(request_id, [100]))
                decode = scheduler.schedule()
                self.assertEqual(set(decode.num_scheduled_tokens), {'A', 'B', 'C'})
                self.assertEqual(decode.scheduled_new_reqs, [])
                self.assertEqual(scheduler.max_num_running_reqs, 4)
                self.assertIn(call(admission.LIVE + '{}', 'TTScheduler'), log.call_args_list)


if __name__ == '__main__':
    unittest.main()
