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
S2 W6b adds the DRAM admission hold (DramNeedTests, DramHoldTests, and one installed-vLLM test): a prompt the
registered predicate refuses waits while the decodes run and is admitted once it fits; with no decode left it is
admitted anyway; the need is read on the smallest largest free block; and with no predicate, a passing one, an
unavailable reading or a raising predicate, every step is the one the rule above gives. DramTimingTests pin the
W6 review's fixes: nothing is asked with every seat decoding; a reading that still counts a finished request's
engine defers one step without a hold line, and the finished id reaches the decode step the plugin falls back to
(FinishingVllmScheduler models vLLM's hand-off of finished_req_ids); a blocked head is judged with the request
behind it; a released request refused again is held and logged again; and the hold's state stays bounded.
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


class DramRequest(FakeRequest):
    """A request that reports its prompt length as vLLM's Request does (num_prompt_tokens)."""

    @property
    def num_prompt_tokens(self):
        return self.prompt_tokens


class dram(object):
    """The worker's DRAM admission predicate registered for the block (None: absent, as before S2 W6b);
    whatever was there before is put back afterwards."""

    def __init__(self, admits):
        self.admits = admits

    def __enter__(self):
        self.saved = sys.modules.pop(admission.DRAM_KEY, None)
        if self.admits is not None:
            admission.register_dram_predicate(self.admits)
        return self

    def __exit__(self, *failure):
        sys.modules.pop(admission.DRAM_KEY, None)
        if self.saved is not None:
            sys.modules[admission.DRAM_KEY] = self.saved
        return False


class DramFreeCase(GateFreeCase):
    """No prefill held and no DRAM predicate registered, whatever an earlier test left."""

    def setUp(self):
        super().setUp()
        clear = dram(None)
        clear.__enter__()
        self.addCleanup(clear.__exit__, None, None, None)


class Pool(object):
    """A pool whose allocator statistics read (free, largest_free) bytes per chip."""

    def __init__(self, *chips):
        self.chips = chips

    def dram_statistics(self):
        return [dict(chip=index, free=free, largest_free=largest) for index, (free, largest) in enumerate(self.chips)]


MB = 10 ** 6
RESERVE = 256 * 2 ** 20


class DramNeedTests(DramFreeCase):
    def test_the_need_is_the_build_the_long_prompts_prefill_and_the_reserve(self):
        build = 800 * MB + 200 * MB
        self.assertEqual(admission.engine_build_peak(), build)
        self.assertEqual(admission.dram_need(1, RESERVE), build + RESERVE)
        self.assertEqual(admission.dram_need(2047, RESERVE), build + RESERVE)
        for prompt in (2048, 123136, None, '4096'):
            with self.subTest(prompt=prompt):
                self.assertEqual(admission.dram_need(prompt, RESERVE), build + 300 * MB + RESERVE,
                                 'a long or unreadable prompt carries the prefill transient')
        self.assertEqual(admission.backstop_need(RESERVE), build + RESERVE, 'after the prefill, no transient')
        for reserve in (-1, 1.5, None, True):
            with self.subTest(reserve=reserve), self.assertRaises(ValueError):
                admission.dram_need(60, reserve)
            with self.subTest(reserve=reserve), self.assertRaises(ValueError):
                admission.backstop_need(reserve)

    def test_the_reading_is_the_smallest_chips_largest_block_never_the_total_free(self):
        self.assertEqual(admission.largest_free(Pool((5_000 * MB, 700 * MB), (6_000 * MB, 900 * MB))), (700 * MB, None))
        self.assertEqual(admission.largest_free(object()), (None, 'pool without device statistics'))
        unavailable = SimpleNamespace(dram_statistics=lambda: dict(unavailable='no memory view'))
        self.assertEqual(admission.largest_free(unavailable), (None, 'no memory view'))
        broken = SimpleNamespace(dram_statistics=Mock(side_effect=RuntimeError('allocator gone')))
        self.assertEqual(admission.largest_free(broken), (None, 'RuntimeError: allocator gone'))
        largest, reason = admission.largest_free(SimpleNamespace(dram_statistics=lambda: []))
        self.assertIsNone(largest)
        self.assertTrue(reason.startswith('ValueError'), reason)

    def test_the_predicate_refuses_only_a_reading_below_the_need(self):
        short, long = admission.dram_need(60, 0), admission.dram_need(4096, 0)
        admits = admission.dram_predicate(Pool((9_000 * MB, short), (9_000 * MB, short + MB)), 0)
        self.assertEqual(admits(60), (True, dict(largest_free=short, need=short)))
        self.assertEqual(admits(4096), (False, dict(largest_free=short, need=long)))
        # Plenty free in total, but no block large enough: held. The need is never read on `free`.
        fragmented = admission.dram_predicate(Pool((20_000 * MB, 100 * MB), (20_000 * MB, 20_000 * MB)), 0)
        self.assertFalse(fragmented(60)[0])
        self.assertEqual(admission.dram_predicate(object(), 0)(4096),
                         (True, dict(largest_free=None, need=long, unavailable='pool without device statistics')))
        with self.assertRaises(ValueError):
            admission.dram_predicate(Pool(), -1)

    def test_registration_parks_the_predicate_and_its_removal_takes_only_its_own(self):
        modules = {}
        first, second = Mock(), Mock()
        remove_first = admission.register_dram_predicate(first, modules)
        self.assertIs(modules[admission.DRAM_KEY].admits, first)
        self.assertIs(admission.dram_admits(modules), first)
        remove_second = admission.register_dram_predicate(second, modules)
        remove_first()
        self.assertIs(admission.dram_admits(modules), second, 'a later registration is not removed by an earlier one')
        remove_second()
        self.assertNotIn(admission.DRAM_KEY, modules)
        self.assertIsNone(admission.dram_admits(modules))
        with self.assertRaises(ValueError):
            admission.register_dram_predicate('not callable', modules)

        class Broken(object):
            @property
            def admits(self):
                raise RuntimeError('holder unreadable')

        self.assertIsNone(admission.dram_admits({admission.DRAM_KEY: Broken()}))

    def test_the_candidates_are_the_requests_the_waiting_loop_may_admit_this_step(self):
        """skipped_waiting's, then waiting's, in queue order (vLLM's peek_request is the head of that order), up to
        and including the first request that is not blocked: a blocked head is skipped unless promoted in the
        pass, so the loop may admit it or any request up to that one (review W6 defect 4)."""
        short, long = DramRequest('S', 60), DramRequest('W', 4096)
        candidates = admission.admission_candidates
        self.assertEqual(candidates(SimpleNamespace(skipped_waiting=Queue([short]), waiting=Queue([long]))), [short])
        self.assertEqual(candidates(SimpleNamespace(skipped_waiting=Queue(), waiting=Queue([long, short]))), [long])
        self.assertEqual(candidates(SimpleNamespace(waiting=Queue([long]))), [long])
        self.assertEqual(candidates(SimpleNamespace(skipped_waiting=Queue(), waiting=Queue())), [])
        grammar, remote = blocked(DramRequest('G', 60)), blocked(DramRequest('R', 90))
        self.assertEqual(candidates(Blocking(skipped_waiting=Queue([grammar, remote]), waiting=Queue([short, long]))),
                         [grammar, remote, short])
        self.assertEqual(candidates(Blocking(skipped_waiting=Queue([grammar]), waiting=Queue())), [grammar])
        self.assertEqual(candidates(SimpleNamespace(skipped_waiting=Queue([grammar]), waiting=Queue([long]))), [grammar],
                         'a scheduler without _is_blocked_waiting_status blocks nothing')
        self.assertIs(admission.binding_request([grammar, remote, short]), remote)
        self.assertIs(admission.binding_request([short, long, DramRequest('W2', 4096)]), long, 'the first of equals')
        unreadable = SimpleNamespace(request_id='U')
        self.assertIs(admission.binding_request([long, unreadable]), unreadable, 'an unreadable length is the longest')
        self.assertIsNone(admission.binding_request([]))
        self.assertEqual(admission.prompt_tokens(short), 60)
        self.assertEqual(admission.prompt_tokens(SimpleNamespace(prompt_token_ids=[1] * 7)), 7)
        self.assertIsNone(admission.prompt_tokens(SimpleNamespace()))


class DramHoldTests(DramFreeCase):
    """The hold on the plugin's own class (fixtures/plugin_scheduler.py) over the reduced vLLM scheduler."""

    def started(self, log):
        """A scheduler with A decoding."""
        scheduler = patched_plugin(log)()
        scheduler.add_request(DramRequest('A'))
        self.assertEqual(new_ids(scheduler.schedule()), ['A'])
        return scheduler

    @staticmethod
    def lines(log, template):
        return [entry.args[1:] for entry in log.call_args_list if entry.args[0] == template]

    def test_a_prompt_that_does_not_fit_waits_while_the_decodes_run_and_is_admitted_once_it_fits(self):
        log, asked, fits = Mock(), [], [False]

        def admits(prompt):
            asked.append(prompt)
            return fits[0], dict(largest_free=900 * MB, need=1_300 * MB)

        scheduler = self.started(log)
        with dram(admits):
            scheduler.add_request(DramRequest('B', 4096))
            for _ in range(3):
                step = scheduler.schedule()
                self.assertEqual((new_ids(step), cached_ids(step)), ([], ['A']), 'A decodes, B waits')
                self.assertEqual(names(scheduler.waiting), ['B'], 'the hidden queue came back')
                self.assertEqual(scheduler.max_num_running_reqs, 4)
            fits[0] = True
            step = scheduler.schedule()
            self.assertEqual((new_ids(step), cached_ids(step)), (['B'], []))
            self.assertEqual(cached_ids(scheduler.schedule()), ['A', 'B'])
        self.assertEqual(asked, [4096] * 4, 'asked about B at every prefill attempt, about nothing else')
        self.assertEqual(self.lines(log, admission.DRAM_HOLD_LINE), [(4096, '900.0MB', '1300.0MB', 'B', 1)],
                         'one hold line for the held state')
        self.assertEqual(self.lines(log, admission.DRAM_RELEASED_LINE), [(4096, '900.0MB', '1300.0MB', 'B')])
        self.assertEqual(self.lines(log, admission.DRAM_LIFTED_LINE), [])
        self.assertIn(call('[PINDIAG] one fresh prefill per step: partials={} decodes={} gate_held={} allowed={} '
                           'hidden={}', 0, 1, False, 0, True), log.call_args_list, 'the held step logs its decision')

    def test_with_no_decode_left_the_prompt_is_admitted_and_the_backstop_decides(self):
        """Nothing running can finish and free DRAM, so holding would stall the engine for good."""
        log = Mock()
        refuse = lambda prompt: (False, dict(largest_free=900 * MB, need=1_300 * MB))
        scheduler = self.started(log)
        with dram(refuse):
            scheduler.add_request(DramRequest('B', 4096))
            self.assertEqual(cached_ids(scheduler.schedule()), ['A'], 'held while A decodes')
            scheduler.finish('A')
            self.assertEqual(new_ids(scheduler.schedule()), ['B'], 'A finished: B is admitted, still short')
            fresh = patched_plugin(log)()
            fresh.add_request(DramRequest('C', 60))
            self.assertEqual(new_ids(fresh.schedule()), ['C'], 'an idle engine admits at once')
        self.assertEqual(self.lines(log, admission.DRAM_HOLD_LINE), [(4096, '900.0MB', '1300.0MB', 'B', 1)])
        self.assertEqual(self.lines(log, admission.DRAM_LIFTED_LINE),
                         [(4096, '900.0MB', '1300.0MB', 'B'), (60, '900.0MB', '1300.0MB', 'C')])
        self.assertEqual(self.lines(log, admission.DRAM_RELEASED_LINE), [])

    def test_a_held_gate_and_running_partials_are_decided_before_the_predicate_is_asked(self):
        admits = Mock(return_value=(False, dict(largest_free=0, need=1)))
        scheduler = self.started(Mock())
        with dram(admits):
            scheduler.add_request(DramRequest('B'))
            with gate('cmpl-held'):
                self.assertEqual(cached_ids(scheduler.schedule()), ['A'])
            admits.assert_not_called()
            # A running partial prefill continues alone; the predicate is not asked either.
            recording = plugin_class(RecordingBase)
            admission.install(configured(recording), log=Mock())
            partial = recording.__new__(recording)
            partial.qwen_seen, partial.qwen_requeue = {}, False
            partial.running = [SimpleNamespace(name='partial', is_prefill_chunk=True),
                               SimpleNamespace(name='decode', is_prefill_chunk=False)]
            partial.waiting, partial.skipped_waiting = Queue([DramRequest('C')]), Queue()
            partial.policy, partial.max_num_running_reqs = 'fcfs', 4
            partial._schedule_prefill_only()
            self.assertEqual(partial.qwen_seen['running'], ['partial'])
            admits.assert_not_called()

    def test_the_hold_reads_the_pool_through_the_registered_predicate(self):
        """End to end with the worker's own predicate: 1.0 GB in the largest block, no reserve - a short
        prompt fits (need 1.0 GB), a long one does not (need 1.3 GB)."""
        log = Mock()
        pool = Pool((20_000 * MB, 1_000 * MB), (20_000 * MB, 5_000 * MB))
        scheduler = self.started(log)
        with dram(admission.dram_predicate(pool, 0)):
            scheduler.add_request(DramRequest('B', 60))
            self.assertEqual(new_ids(scheduler.schedule()), ['B'])
            scheduler.add_request(DramRequest('C', 4096))
            step = scheduler.schedule()
            self.assertEqual((new_ids(step), cached_ids(step)), ([], ['A', 'B']))
        self.assertEqual(self.lines(log, admission.DRAM_HOLD_LINE), [(4096, '1000.0MB', '1300.0MB', 'C', 2)])

    def scenario(self, admits, log):
        """test_running_decodes_are_never_dropped_or_starved's arrivals, step by step."""
        with dram(admits):
            scheduler = patched_plugin(log)()
            steps = []
            scheduler.add_request(DramRequest('A'))
            steps.extend(scheduler.schedule() for _ in range(2))
            for request_id in 'BCD':
                scheduler.add_request(DramRequest(request_id, 100))
            steps.extend(scheduler.schedule() for _ in range(3))
            scheduler.add_request(DramRequest('E'))
            steps.extend(scheduler.schedule() for _ in range(2))
            scheduler.finish('B')
            steps.extend(scheduler.schedule() for _ in range(2))
            return [(new_ids(step), cached_ids(step)) for step in steps]

    def test_no_predicate_a_passing_one_an_unavailable_reading_or_a_raising_one_schedules_as_before(self):
        reference = self.scenario(None, Mock())
        self.assertEqual(reference[:2], [(['A'], []), ([], ['A'])])
        for name, admits in (('passing', lambda prompt: (True, dict(largest_free=9_000 * MB, need=1_000 * MB))),
                             ('unavailable', admission.dram_predicate(object(), 0)),
                             ('raising', Mock(side_effect=RuntimeError('allocator gone')))):
            with self.subTest(predicate=name):
                log = Mock()
                self.assertEqual(self.scenario(admits, log), reference)
                self.assertEqual(self.lines(log, admission.DRAM_HOLD_LINE), [])
                unavailable = self.lines(log, admission.DRAM_UNAVAILABLE_LINE)
                if name == 'passing':
                    self.assertEqual(unavailable, [])
                else:
                    self.assertEqual([request_id for request_id, _ in unavailable], ['A', 'B', 'C', 'D', 'E'],
                                     'one line per request, never a hold')


BLOCKED = 'WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR'


class Blocking(SimpleNamespace):
    """A scheduler with vLLM's _is_blocked_waiting_status (scheduler.py:1853-1858), over the fixture's statuses."""

    @staticmethod
    def _is_blocked_waiting_status(status):
        return status == BLOCKED


def blocked(request):
    """`request` waiting on its structured-output grammar, as vLLM marks it."""
    request.status = BLOCKED
    return request


def logged(log, template):
    return [entry.args[1:] for entry in log.call_args_list if entry.args and entry.args[0] == template]


class FinishingVllmScheduler(FakeVllmScheduler):
    """FakeVllmScheduler with vLLM 0.25.1's finished_req_ids: finish() records the id (scheduler.py:2108), and
    every schedule() - a pass the plugin then discards as much as one it returns - hands the set to its output and
    starts a new one (:1105, :1210)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finished_req_ids = set()

    def finish(self, request_id):
        super().finish(request_id)
        self.finished_req_ids.add(request_id)

    def schedule(self):
        output = super().schedule()
        output.finished_req_ids, self.finished_req_ids = self.finished_req_ids, set()
        return output


class Worker(object):
    """The DRAM the hold reads, as the worker holds it: one engine per admitted request, freed only when a step's
    output names that request finished, at the start of executing the step (serving_lifecycle._execute detaches
    it and the hook closes its engine). By default the design's figures: 4.37 GB free per chip at attach beside
    the block, 0.80 GB per engine (s2-design.md section 3.1)."""

    def __init__(self, attach=4_370 * MB, engine=800 * MB):
        self.attach, self.engine, self.engines = attach, engine, set()

    def dram_statistics(self):
        largest = self.attach - self.engine * len(self.engines)
        return [dict(chip=0, free=largest, largest_free=largest)]

    def execute(self, step):
        self.engines -= set(step.finished_req_ids)
        self.engines |= set(new_ids(step))
        return step


class DramTimingTests(DramFreeCase):
    """The W6 review's defects in the hold: when it is asked, what a stale reading does, which request it judges,
    what it logs, and what it keeps."""

    @staticmethod
    def finishing(log):
        """The plugin's own class over FinishingVllmScheduler, with the wrapper installed."""
        cls = plugin_class(FinishingVllmScheduler)
        admission.install(configured(cls), log=log)
        return cls()

    def test_with_every_seat_decoding_nothing_is_asked_and_a_hold_after_a_seat_frees_is_logged(self):
        """Defect 1(a): with four users decoding (max-num-seqs 4) a queued fifth is held by the seat limit, so the
        predicate is not asked and no hold line is spent. When a seat frees, a real hold logs at its decode count."""
        log, asked = Mock(), []

        def refuse(prompt):
            asked.append(prompt)
            return False, dict(largest_free=1_170 * MB, need=1_568 * MB)

        scheduler = patched_plugin(log)()
        for request_id in 'ABCD':
            scheduler.add_request(DramRequest(request_id, 100))
            scheduler.schedule()
        self.assertEqual(names(scheduler.running), ['A', 'B', 'C', 'D'])
        with dram(refuse):
            scheduler.add_request(DramRequest('E', 4096))
            for _ in range(3):
                step = scheduler.schedule()
                self.assertEqual((new_ids(step), cached_ids(step)), ([], ['A', 'B', 'C', 'D']))
            self.assertEqual(asked, [], 'every seat decoding: the waiting loop admits nobody, so nothing is asked')
            self.assertEqual(logged(log, admission.DRAM_HOLD_LINE), [])
            scheduler.finish('A')
            step = scheduler.schedule()
            self.assertEqual((new_ids(step), cached_ids(step)), ([], ['B', 'C', 'D']))
        self.assertEqual(asked, [4096])
        self.assertEqual(logged(log, admission.DRAM_HOLD_LINE), [(4096, '1170.0MB', '1568.0MB', 'E', 3)])

    def test_a_replacement_behind_a_finish_is_deferred_one_step_and_the_decode_step_names_the_finish(self):
        """Defect 1(b), on the design's figures: four 120k users hold four engines (1.17 GB left), A finishes and E
        waits. That step's reading still counts A's engine, which the worker frees only when the step's output
        names A. So E is deferred - no hold line - and the decode-only step the plugin falls back to must name A:
        vLLM's first, discarded pass took it (FinishingVllmScheduler), and without it A's engine is never freed and
        E waits for good. E is admitted on the fresh reading of the next step."""
        log, worker = Mock(), Worker()
        scheduler = self.finishing(log)
        with dram(admission.dram_predicate(worker, RESERVE)):
            for request_id in 'ABCD':
                scheduler.add_request(DramRequest(request_id, 120_000))
                self.assertEqual(new_ids(worker.execute(scheduler.schedule())), [request_id])
            self.assertEqual(worker.dram_statistics()[0]['largest_free'], 1_170 * MB)
            scheduler.finish('A')
            scheduler.add_request(DramRequest('E', 110_000))
            deferred = worker.execute(scheduler.schedule())
            self.assertEqual((new_ids(deferred), cached_ids(deferred)), ([], ['B', 'C', 'D']))
            self.assertEqual(deferred.finished_req_ids, {'A'}, 'the decode step names A, so the worker frees it')
            self.assertEqual(worker.dram_statistics()[0]['largest_free'], 1_970 * MB)
            admitted = worker.execute(scheduler.schedule())
            self.assertEqual((new_ids(admitted), admitted.finished_req_ids), (['E'], set()))
        need = '%.1fMB' % (admission.dram_need(110_000, RESERVE) / MB)
        self.assertEqual(logged(log, admission.DRAM_HOLD_LINE), [], 'a stale reading is not a hold')
        self.assertEqual(logged(log, admission.DRAM_DEFERRED_LINE), [(110_000, '1170.0MB', need, 'E', ['A'])])
        self.assertEqual(logged(log, admission.DRAM_CARRIED_LINE), [(['A'],)])
        self.assertEqual(logged(log, admission.DRAM_RELEASED_LINE), [])

    def test_when_the_last_decode_finishes_the_deferred_pass_itself_names_the_finish_once(self):
        """With no decode left the plugin returns the held pass itself, which already names the finish: nothing is
        carried, so the next step does not name it twice."""
        log, worker = Mock(), Worker(attach=1_900 * MB)
        scheduler = self.finishing(log)
        with dram(admission.dram_predicate(worker, RESERVE)):
            scheduler.add_request(DramRequest('A', 120_000))
            self.assertEqual(new_ids(worker.execute(scheduler.schedule())), ['A'])
            scheduler.finish('A')
            scheduler.add_request(DramRequest('E', 110_000))
            deferred = worker.execute(scheduler.schedule())
            self.assertEqual((new_ids(deferred), cached_ids(deferred), deferred.finished_req_ids), ([], [], {'A'}))
            admitted = worker.execute(scheduler.schedule())
            self.assertEqual((new_ids(admitted), admitted.finished_req_ids), (['E'], set()))
        self.assertEqual(len(logged(log, admission.DRAM_DEFERRED_LINE)), 1)
        for template in (admission.DRAM_CARRIED_LINE, admission.DRAM_HOLD_LINE, admission.DRAM_LIFTED_LINE):
            self.assertEqual(logged(log, template), [], template)

    def test_a_forced_prefill_only_step_returns_the_held_pass_as_it_is_so_nothing_is_carried(self):
        """Lane coordination's forced PREFILL_ONLY mode returns the empty prefill pass itself (plugin
        scheduler.py:116-121), which already names the finish; carrying it too would name it twice."""
        log, worker = Mock(), Worker()
        scheduler = self.finishing(log)
        with dram(admission.dram_predicate(worker, RESERVE)):
            for request_id in 'ABCD':
                scheduler.add_request(DramRequest(request_id, 120_000))
                worker.execute(scheduler.schedule())
            scheduler.finish('A')
            scheduler.add_request(DramRequest('E', 110_000))
            scheduler.set_forced_mode(TTSchedulingMode.PREFILL_ONLY)
            forced = worker.execute(scheduler.schedule())
            self.assertEqual((new_ids(forced), cached_ids(forced), forced.finished_req_ids), ([], [], {'A'}))
            scheduler.set_forced_mode(TTSchedulingMode.DEFAULT)
            admitted = worker.execute(scheduler.schedule())
            self.assertEqual((new_ids(admitted), admitted.finished_req_ids), (['E'], set()))
        self.assertEqual(logged(log, admission.DRAM_CARRIED_LINE), [])

    def test_a_blocked_head_is_judged_with_the_request_the_loop_may_admit_behind_it(self):
        """Defect 4: a head vLLM skips for a blocked status (its grammar still compiling) hands the step to the
        request behind it, whose need may carry the prefill transient the head's does not."""
        log, asked = Mock(), []

        def admits(prompt):
            asked.append(prompt)
            need = admission.dram_need(prompt, 0)
            return 1_100 * MB >= need, dict(largest_free=1_100 * MB, need=need)

        with dram(admits):
            waiting = Blocking(skipped_waiting=Queue([blocked(DramRequest('G', 60))]),
                               waiting=Queue([DramRequest('L', 4096)]))
            self.assertTrue(admission.dram_hold(waiting, 2, {}, log))
            self.assertEqual(asked, [4096])
            unblocked = Blocking(skipped_waiting=Queue(), waiting=Queue([DramRequest('S', 60), DramRequest('L', 4096)]))
            self.assertFalse(admission.dram_hold(unblocked, 2, {}, log))
            self.assertEqual(asked, [4096, 60], 'an unblocked head is the only candidate')
        self.assertEqual(logged(log, admission.DRAM_HOLD_LINE), [(4096, '1100.0MB', '1300.0MB', 'L', 2)])

    def test_a_released_request_that_is_refused_again_is_held_and_logged_again(self):
        """Defect 6, and the (request, decodes) key of defect 1: released means the reading fitted, not that the
        request was admitted (a token budget can still keep it waiting), so a new refusal is a new hold; a new
        decode count is a new state."""
        log, fits = Mock(), [False]
        scheduler, state = SimpleNamespace(waiting=Queue([DramRequest('E', 4096)])), {}
        with dram(lambda prompt: (fits[0], dict(largest_free=900 * MB, need=1_300 * MB))):
            self.assertTrue(admission.dram_hold(scheduler, 1, state, log))
            self.assertTrue(admission.dram_hold(scheduler, 1, state, log))
            fits[0] = True
            self.assertFalse(admission.dram_hold(scheduler, 1, state, log))
            fits[0] = False
            self.assertTrue(admission.dram_hold(scheduler, 1, state, log))
            self.assertTrue(admission.dram_hold(scheduler, 2, state, log))
            self.assertTrue(admission.dram_hold(scheduler, 2, state, log))
        self.assertEqual(logged(log, admission.DRAM_HOLD_LINE),
                         [(4096, '900.0MB', '1300.0MB', 'E', 1)] * 2 + [(4096, '900.0MB', '1300.0MB', 'E', 2)])
        self.assertEqual(logged(log, admission.DRAM_RELEASED_LINE), [(4096, '900.0MB', '1300.0MB', 'E')])

    def test_the_state_stays_bounded_however_many_requests_pass(self):
        """Defect 5: one line per distinct state, from state that does not grow with the traffic."""
        log, state = Mock(), {}
        with dram(admission.dram_predicate(object(), 0)):
            for index in range(200):
                for _ in range(2):
                    admission.dram_hold(SimpleNamespace(waiting=Queue([DramRequest('U%d' % index, 60)])), 1, state, log)
        with dram(lambda prompt: (False, dict(largest_free=0, need=1))):
            for index in range(200):
                for _ in range(2):
                    admission.dram_hold(SimpleNamespace(waiting=Queue([DramRequest('L%d' % index, 60)])), 0, state, log)
        self.assertEqual(len(logged(log, admission.DRAM_UNAVAILABLE_LINE)), 200)
        self.assertEqual(len(logged(log, admission.DRAM_LIFTED_LINE)), 200)
        self.assertEqual(sorted(state), ['dram_held', 'dram_noted'])
        self.assertEqual((state['dram_held'], state['dram_noted']), (None, ('lifted', 'L199')))


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

    def test_a_dram_hold_keeps_the_prompt_waiting_while_the_decode_runs(self):
        """S2 W6b on vLLM's own queues and Request (num_prompt_tokens, peek_request): B waits while the
        predicate refuses it, A decodes meanwhile, and B is admitted alone once it fits."""
        for source, scheduler_type in self.schedulers():
            with self.subTest(source=source):
                log, asked, fits = Mock(), [], [False]

                def admits(prompt):
                    asked.append(prompt)
                    return fits[0], dict(largest_free=900 * MB, need=1_300 * MB)

                admission.install(configured(scheduler_type), log=log)
                case, scheduler = self.build(scheduler_type)
                self.arrive(scheduler, 'A')
                step = scheduler.schedule()
                self.assertEqual(step.num_scheduled_tokens, {'A': 1024})
                scheduler.update_from_output(step, case.output('A', [100]))
                with dram(admits):
                    self.arrive(scheduler, 'B')
                    held = scheduler.schedule()
                    self.assertEqual(set(held.num_scheduled_tokens), {'A'})
                    self.assertEqual(held.scheduled_new_reqs, [])
                    scheduler.update_from_output(held, case.output('A', [101]))
                    fits[0] = True
                    admitted = scheduler.schedule()
                    self.assertEqual(admitted.num_scheduled_tokens, {'B': 1024})
                self.assertEqual(asked, [1024, 1024])
                self.assertEqual(scheduler.max_num_running_reqs, 4)
                self.assertIn(call(admission.DRAM_HOLD_LINE, 1024, '900.0MB', '1300.0MB', 'B', 1), log.call_args_list)

    def test_a_deferred_admission_behind_a_finish_leaves_the_finish_in_the_decode_step(self):
        """S2 W6b on vLLM's own finished_req_ids: A and B decode, A finishes, and C waits on a reading that still
        counts A's engine. The prefill pass is held (deferred, not a hold), the plugin falls back to a decode-only
        pass, and that pass names A finished: without the carry vLLM's first, discarded pass took it and the worker
        would never detach A. C is admitted on the next step, which names nothing twice."""
        from vllm.v1.request import RequestStatus

        for source, scheduler_type in self.schedulers():
            with self.subTest(source=source):
                log, fits = Mock(), [False]
                admission.install(configured(scheduler_type), log=log)
                case, scheduler = self.build(scheduler_type)
                for request_id, token in (('A', 100), ('B', 200)):
                    self.arrive(scheduler, request_id)
                    step = scheduler.schedule()
                    self.assertEqual(step.num_scheduled_tokens, {request_id: 1024})
                    scheduler.update_from_output(step, case.output(request_id, [token]))
                scheduler.finish_requests('A', RequestStatus.FINISHED_ABORTED)
                with dram(lambda prompt: (fits[0], dict(largest_free=900 * MB, need=1_300 * MB))):
                    self.arrive(scheduler, 'C')
                    deferred = scheduler.schedule()
                    self.assertEqual(set(deferred.num_scheduled_tokens), {'B'})
                    self.assertEqual(deferred.scheduled_new_reqs, [])
                    self.assertEqual(set(deferred.finished_req_ids), {'A'})
                    scheduler.update_from_output(deferred, case.output('B', [201]))
                    fits[0] = True
                    admitted = scheduler.schedule()
                    self.assertEqual(admitted.num_scheduled_tokens, {'C': 1024})
                    self.assertEqual(set(admitted.finished_req_ids), set())
                self.assertIn(call(admission.DRAM_CARRIED_LINE, ['A']), log.call_args_list)
                self.assertEqual(logged(log, admission.DRAM_HOLD_LINE), [])


if __name__ == '__main__':
    unittest.main()
