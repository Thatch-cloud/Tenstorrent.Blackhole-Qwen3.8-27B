"""Lever N with prefix reuse: what the G1 scheduler graft and its registry do for the merged route (qwen_prefix_scheduler_patch, qwen_prefix_registry),
on the fakes of test_qwen_prefix_scheduler_patch (docs/lever-n-prefix-merged-route.md sections 4 (R2), 5 and 6).

What is held:
  R2         chunked prefill is accepted ONLY beside the installed Lever N cap under sticky sessions (the flag, the schedule wrapper on the class, the
             one-fresh-prefill cap, an integer token budget), every missing piece is named, today's refusal text stays the prefix of the message, the
             threshold and whole-prompt-budget refusals stay absolute, and the install line says chunked=levern;
  peek       the trim's answer for a waiting request, asked before the pass: memoized per schedule() call, no trace in vLLM's prefix-cache statistics, no
             token check or counter counted twice, the same Q the in-pass trim lands on, the in-pass trim capped at it, an unsalted or killed request 0;
  in flight  an admission scheduled for fewer tokens than its prompt leaves an in-flight capture plan that outlives the step, later steps of the request
             are noted from the cached half of the output, the plan ends with the request's final step, and it ends on free, on the kill switch and on a
             rebind; a request that fits one step leaves none; the registry's own counters move;
  names      the marker names the graft reads equal the wrappers' own."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import test_qwen_prefix_scheduler_patch as base
from qwen_prefix_registry import BLOCK, CHUNK, PrefixRegistry
import qwen_prefix_scheduler_patch as graft_module

Quiet, Request, FakeScheduler, tokens = base.Quiet, base.Request, base.FakeScheduler, base.tokens

STICKY = {'QWEN_FAST_STICKY_SESSIONS': '1', 'QWEN_FAST_LEVER_N': '1'}


class ChunkedFake(FakeScheduler):
    """A scheduler whose class carries the markers Lever N's wrappers leave (levern_scheduler.WRAPPED on schedule, serving_prefill_admission.WRAPPED on
    _schedule_prefill_only) and an integer token budget: what the merged route's install check reads."""

    max_num_scheduled_tokens = 262144

    def _schedule_prefill_only(self):
        return self.schedule()

    _schedule_prefill_only._qwen_one_fresh_prefill = True


def with_marker(cls, schedule_marked=True):
    class Marked(cls):
        pass

    function = lambda self: cls.schedule(self)         # noqa: E731
    if schedule_marked:
        function._qwen_levern_schedule = True
    Marked.schedule = function
    return Marked


def chunked_scheduler(sticky_ready=True, **flags):
    scheduler = with_marker(ChunkedFake)()
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.max_num_scheduled_tokens = 262144
    for name, value in flags.items():
        setattr(scheduler, name, value)
    return scheduler


class InstallRuleTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, base.fake_vllm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})
        environment.start()
        self.addCleanup(environment.stop)

    def problems(self, scheduler, environ=STICKY):
        return [problem for problem in graft_module.install_problems(scheduler, environ) if 'chunked' in problem]

    def sticky_scheduler(self):
        scheduler = chunked_scheduler()
        scheduler.num_lookahead_tokens = graft_module.STICKY_LOOKAHEAD
        scheduler.vllm_config.speculative_config = SimpleNamespace(method='dflash', num_speculative_tokens=15)
        return scheduler

    def test_chunked_prefill_is_accepted_beside_the_installed_cap_under_sticky_sessions(self):
        scheduler = self.sticky_scheduler()
        self.assertEqual(self.problems(scheduler), [])
        self.assertEqual(graft_module.levern_chunking_problems(scheduler, STICKY), [])
        self.assertTrue(graft_module.levern_chunked(scheduler, STICKY))
        self.assertEqual(graft_module.install_problems(scheduler, STICKY), [])

    def test_every_missing_piece_is_named_and_todays_text_is_the_prefix_of_the_message(self):
        cases = (
            ('QWEN_FAST_STICKY_SESSIONS=1', lambda s: None, {'QWEN_FAST_LEVER_N': '1'}),
            ('QWEN_FAST_LEVER_N=1', lambda s: None, {'QWEN_FAST_STICKY_SESSIONS': '1'}),
            ("Lever N's schedule wrapper", lambda s: setattr(type(s), 'schedule', lambda self: None), STICKY),
            ('the one-fresh-prefill cap', lambda s: setattr(type(s), '_schedule_prefill_only', lambda self: None), STICKY),
            ('an integer max_num_scheduled_tokens', lambda s: setattr(s, 'max_num_scheduled_tokens', None), STICKY),
            ('an integer max_num_scheduled_tokens', lambda s: setattr(s, 'max_num_scheduled_tokens', 'many'), STICKY),
        )
        for word, mutate, environ in cases:
            with self.subTest(word=word):
                scheduler = self.sticky_scheduler()
                scheduler.__class__ = type('Fresh', (type(scheduler),), {})
                mutate(scheduler)
                found = self.problems(scheduler, environ)
                self.assertEqual(len(found), 1, found)
                self.assertTrue(found[0].startswith('chunked prefill is on: start_pos > 0 would also mean "continue my own suspended scratch" (Lever N)'))
                self.assertIn(word, found[0])

    def test_chunking_without_the_merged_route_is_refused_with_todays_text(self):
        scheduler = FakeScheduler()
        scheduler.scheduler_config.enable_chunked_prefill = True
        found = [problem for problem in graft_module.install_problems(scheduler, {}) if 'chunked' in problem]
        self.assertEqual(len(found), 1)
        self.assertIn('chunked prefill is on', found[0])
        self.assertIn('QWEN_FAST_STICKY_SESSIONS=1', found[0])

    def test_the_threshold_and_the_whole_prompt_budget_refusals_stay_absolute(self):
        scheduler = self.sticky_scheduler()
        scheduler.scheduler_config.long_prefill_token_threshold = 2048
        found = graft_module.install_problems(scheduler, STICKY)
        self.assertTrue(any('long_prefill_token_threshold is 2048' in problem for problem in found), found)
        self.assertEqual(self.problems(scheduler), [])
        scheduler = self.sticky_scheduler()
        scheduler.scheduler_config.max_num_batched_tokens = 4096
        self.assertTrue(any('max_num_batched_tokens' in problem for problem in graft_module.install_problems(scheduler, STICKY)))

    def test_install_succeeds_and_says_chunked_levern_and_without_the_pieces_the_engine_would_not_start(self):
        logs = []
        scheduler = self.sticky_scheduler()
        graft = graft_module.install(scheduler, registry=PrefixRegistry(budget_bytes=1 << 30), kill_switch_path=None,
                                     logger=lambda message, *values: logs.append(message % values if values else message),
                                     stats=graft_module.StatsExport(path='', logger=lambda *a: None), environ=STICKY)
        self.assertTrue(graft.sticky)
        self.assertIn('install sticky=1 lookahead=16 drop_last=False ceiling=floor2048(P-2048)', logs)
        self.assertTrue(any(line.startswith('install chunked=levern') for line in logs), logs)
        with self.assertRaisesRegex(graft_module.PrefixInstallError, 'chunked prefill is on'):
            graft_module.install(chunked_scheduler(), registry=PrefixRegistry(budget_bytes=1 << 30), kill_switch_path=None,
                                 logger=lambda *a: None, stats=graft_module.StatsExport(path='', logger=lambda *a: None), environ={})

    def test_the_marker_names_are_the_wrappers_own(self):
        import levern_policy
        import levern_scheduler
        import serving_prefill_admission as admission

        self.assertEqual(graft_module.LEVERN_SCHEDULE_WRAPPED, levern_scheduler.WRAPPED)
        self.assertEqual(graft_module.ADMISSION_WRAPPED, admission.WRAPPED)
        self.assertEqual(graft_module.LEVERN_ENV, levern_policy.FLAG)

    def test_the_hook_the_stage_writes_is_unchanged(self):
        """The graft is image-baked: its INIT_HOOK bytes and the scheduler pin are what the provenance hashes; this change moves neither."""
        self.assertEqual(graft_module.SCHEDULER_SHA256, 'a1bd6257d3a14c904b41b4795b8e8b4b1b132c70fc3a4db9d4340a128a100de4')
        import hashlib

        # the hook's bytes at the merge base (0cab164f): the stage writes exactly these into the plugin's TTScheduler.__init__
        self.assertEqual(hashlib.sha256(graft_module.INIT_HOOK.encode()).hexdigest(), '09cbba44aa5a55ab9e44fe116deb5121bd8c639ce6db964ae103439fcd22627b')


class PeekTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, base.fake_vllm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})
        environment.start()
        self.addCleanup(environment.stop)
        self.logs = []

    def make(self, kill_switch_path=None):
        scheduler = FakeScheduler()
        manager = scheduler.kv_cache_manager
        manager.prefix_cache_stats = Stats()
        inner = manager.get_computed_blocks

        def recording(request):
            # vLLM's get_computed_blocks records the attempt in prefix_cache_stats (kv_cache_manager.py)
            if manager.prefix_cache_stats is not None:
                manager.prefix_cache_stats.record(request)
            return inner(request)

        manager.get_computed_blocks = recording
        registry = PrefixRegistry(budget_bytes=1 << 40)
        with Quiet():
            registry.enable_mid_loop_capture()
        graft = graft_module.install(scheduler, registry=registry, kill_switch_path=kill_switch_path, logger=lambda *a: self.logs.append(a),
                                     stats=graft_module.StatsExport(path='', logger=lambda *a: None))
        return scheduler, graft

    served = 0

    def conversation(self, scheduler, text):
        """One served turn, as GraftTests.serve: schedule, run the model's side (FakeGdnModel), finish."""
        model = base.FakeGdnModel(scheduler._qwen_prefix.registry)
        PeekTests.served += 1
        name = 'turn-%d' % PeekTests.served
        scheduler.add(Request(name, text))
        out = scheduler.schedule()
        with Quiet():
            for data in out.scheduled_new_reqs:
                model.prefill(data.req_id, scheduler.requests[data.req_id].all_token_ids, data.num_computed_tokens)
        scheduler.finish(scheduler.requests[name])

    def test_the_peek_is_the_q_the_trim_lands_on_and_leaves_no_trace_in_the_statistics(self):
        scheduler, graft = self.make()
        text = tokens(5000, 11)
        self.conversation(scheduler, text)
        graft.registry.begin_step()          # the next schedule() call: the peek is asked inside it
        request = Request('second', text + tokens(3000, 12))
        scheduler.requests['second'] = request
        statistics = scheduler.kv_cache_manager.prefix_cache_stats
        attempts = graft.registry.stats['attempts']
        recorded = statistics.recorded
        q = graft.peek(request)
        self.assertEqual(q, 4096)
        self.assertEqual(statistics.recorded, recorded, 'the peek recorded nothing in vLLM\'s prefix-cache statistics')
        self.assertIs(scheduler.kv_cache_manager.prefix_cache_stats, statistics, 'and put the object back')
        self.assertEqual(graft.registry.stats['attempts'], attempts, 'the registry counts no attempt for a peek')
        self.assertEqual(graft.registry.staged, {}, 'a peek stages no grant')
        blocks, start = scheduler.kv_cache_manager.get_computed_blocks(request)
        self.assertEqual(start, q, 'the in-pass trim lands where the peek said')
        self.assertEqual(statistics.recorded, recorded + 1, 'the in-pass call is the one that records the attempt')

    def test_the_peek_is_memoized_per_call_and_counts_a_token_check_once(self):
        scheduler, graft = self.make()
        text = tokens(5000, 13)
        self.conversation(scheduler, text)
        graft.registry.begin_step()
        request = Request('second', text + tokens(500, 14))
        checks = graft.registry.stats['token_checks']
        first = graft.peek(request)
        self.assertEqual(graft.peek(request), first)
        self.assertEqual(graft.registry.stats['token_checks'], checks + 1)
        scheduler.kv_cache_manager.get_computed_blocks(request)
        self.assertEqual(graft.registry.stats['token_checks'], checks + 1, 'the in-pass trim reuses the remembered verdict')
        scheduler.schedule()
        self.assertEqual(graft.peeked, {}, 'begin_step clears the memo')

    def test_the_in_pass_trim_never_lands_above_the_peek(self):
        scheduler, graft = self.make()
        text = tokens(9000, 15)
        self.conversation(scheduler, text[:2100])
        graft.registry.begin_step()
        self.conversation(scheduler, text[:5000])
        graft.registry.begin_step()
        request = Request('later', text[:5000] + tokens(4000, 16))
        scheduler.requests['later'] = request
        graft.peeked['later'] = 2048          # the cap was computed from a Q of 2048 ...
        blocks, start = scheduler.kv_cache_manager.get_computed_blocks(request)
        self.assertEqual(start, 2048, '... so the trim cannot land on the 4096 vLLM would otherwise grant')
        grant = graft.registry.staged['later']
        self.assertEqual(grant.q, 2048)

    def test_an_unsalted_or_killed_request_peeks_zero(self):
        scheduler, graft = self.make()
        text = tokens(5000, 17)
        self.conversation(scheduler, text)
        self.assertEqual(graft.peek(Request('unsalted', text + tokens(100, 18), salt=None)), 0)
        graft.registry.disable('test')
        self.assertEqual(graft.peek(Request('after-kill', text + tokens(100, 19))), 0)

    def test_the_prefix_kill_switch_engaging_before_the_peek_makes_the_peek_and_the_trim_agree(self):
        # B1: peek read the latched flag but never polled the file; the in-pass trim polled it. A switch that appeared between the two left the cap
        # computed from the hit while the trim landed on 0.
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'prefix.off')
            scheduler, graft = self.make(kill_switch_path=path)
            text = tokens(5000, 21)
            self.conversation(scheduler, text)
            graft.registry.begin_step()
            request = Request('after-the-file', text + tokens(500, 22))
            scheduler.requests['after-the-file'] = request
            with open(path, 'w') as handle:
                handle.write('1')
            graft.kill_switch._last_poll = None          # the once-a-second poll is due
            self.assertEqual(graft.peek(request), 0, 'the peek polls the switch itself')
            blocks, start = scheduler.kv_cache_manager.get_computed_blocks(request)
            self.assertEqual(start, 0, 'and the trim lands where it said')

    def test_a_request_with_no_hit_peeks_zero(self):
        scheduler, graft = self.make()
        self.assertEqual(graft.peek(Request('cold', tokens(5000, 20))), 0)


class Stats(object):
    """vLLM's PrefixCacheStats, as far as get_computed_blocks touches it: record(...)."""

    def __init__(self):
        self.recorded = 0

    def record(self, *args, **kwargs):
        self.recorded += 1


class InflightTests(unittest.TestCase):
    """The registry's in-flight plan, driven the way SchedulerGraft.commit drives it."""

    def setUp(self):
        patcher = mock.patch.dict(sys.modules, base.fake_vllm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})
        environment.start()
        self.addCleanup(environment.stop)

    def make(self):
        scheduler = FakeScheduler()
        registry = PrefixRegistry(budget_bytes=1 << 40)
        with Quiet():
            registry.enable_mid_loop_capture()
        graft = graft_module.install(scheduler, registry=registry, kill_switch_path=None, logger=lambda *a: None,
                                     stats=graft_module.StatsExport(path='', logger=lambda *a: None))
        return scheduler, graft, registry

    def stage(self, registry, request, q, plan):
        registry.begin_step()
        registry.stage(graft_module.Grant(request.request_id, q, q, None, None,
                                          [(pos, request.block_hashes[pos // BLOCK - 1]) for pos in plan], request, None, ()))

    def output(self, admitted=(), cached=(), counts=None):
        return SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id=rid, num_computed_tokens=start) for rid, start in admitted],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[rid for rid, _ in cached], resumed_req_ids=set(),
                                                  num_computed_tokens=[start for _, start in cached]),
            num_scheduled_tokens=dict(counts or {}))

    def test_a_split_admission_leaves_a_plan_that_outlives_its_step_and_ends_with_the_final_step(self):
        scheduler, graft, registry = self.make()
        request = Request('r', tokens(9000, 21))
        self.stage(registry, request, 0, [6144])
        graft.commit(self.output(admitted=[('r', 0)], counts={'r': 2048}))
        self.assertIn('r', registry.inflight)
        self.assertEqual(registry.planned('r'), [6144])
        self.assertEqual(registry.stats['inflight_started'], 1)
        for start in (2048, 4096):
            registry.begin_step()
            self.assertIn('r', registry.inflight, 'the grant is gone, the plan is not')
            self.assertIsNone(registry.grant_for('r'))
            graft.commit(self.output(cached=[('r', start)], counts={'r': 2048}))
            self.assertFalse(registry.inflight['r'].final)
        registry.begin_step()
        graft.commit(self.output(cached=[('r', 6144)], counts={'r': 2856}))
        self.assertTrue(registry.inflight['r'].final)
        self.assertIn('r', registry.inflight, 'the last step still runs its captures after this commit')
        registry.begin_step()
        self.assertNotIn('r', registry.inflight)
        self.assertEqual(registry.stats['inflight_dropped'], 1)

    def test_a_request_that_fits_one_step_or_has_no_plan_leaves_none(self):
        scheduler, graft, registry = self.make()
        request = Request('whole', tokens(5000, 22))
        self.stage(registry, request, 0, [4096])
        graft.commit(self.output(admitted=[('whole', 0)], counts={'whole': 5000}))
        self.assertEqual(registry.inflight, {})
        other = Request('no-plan', tokens(9000, 23))
        self.stage(registry, other, 0, [])
        graft.commit(self.output(admitted=[('no-plan', 0)], counts={'no-plan': 2048}))
        self.assertEqual(registry.inflight, {})

    def test_a_commit_without_the_token_counts_keeps_no_plan(self):
        scheduler, graft, registry = self.make()
        request = Request('r', tokens(9000, 24))
        self.stage(registry, request, 0, [6144])
        registry.commit({'r': 0})
        self.assertEqual(registry.inflight, {})

    def test_a_capture_from_the_plan_is_stored_and_counted_a_free_drops_the_plan(self):
        scheduler, graft, registry = self.make()
        request = Request('r', tokens(9000, 25))
        self.stage(registry, request, 0, [6144])
        graft.commit(self.output(admitted=[('r', 0)], counts={'r': 2048}))
        registry.begin_step()
        stored = registry.capture('r', 6144, rec=['r'], carry=['c'], nbytes=8, loop_pos=6144)
        self.assertIsNotNone(stored)
        self.assertEqual(registry.stats['inflight_captures'], 1)
        self.assertEqual(stored.pos, 6144)
        self.assertEqual(stored.token_ids.tolist() if hasattr(stored.token_ids, 'tolist') else list(stored.token_ids), request.all_token_ids[:6144])
        scheduler._free_request(request)
        self.assertNotIn('r', registry.inflight)
        self.assertIsNone(registry.capture('r', 6144, rec=['r'], carry=['c'], nbytes=8, loop_pos=6144))

    def test_the_kill_switch_and_a_rebind_drop_every_plan(self):
        scheduler, graft, registry = self.make()
        request = Request('r', tokens(9000, 26))
        self.stage(registry, request, 0, [6144])
        graft.commit(self.output(admitted=[('r', 0)], counts={'r': 2048}))
        registry.disable('kill switch')
        self.assertEqual(registry.inflight, {})
        registry2 = PrefixRegistry(budget_bytes=1 << 40)
        scheduler2 = FakeScheduler()
        registry2.bind(scheduler2)
        registry2.inflight['x'] = object()
        del scheduler2
        import gc

        gc.collect()
        registry2.bind(FakeScheduler())
        self.assertNotIn('x', registry2.inflight, 'a dead owner is replaced and the registry starts clean')

    def test_the_snapshot_and_the_stat_names_carry_the_counters(self):
        registry = PrefixRegistry(budget_bytes=1 << 30)
        snapshot = registry.snapshot()
        for name in ('inflight_started', 'inflight_captures', 'inflight_dropped', 'inflight_now'):
            self.assertIn(name, snapshot)

    def test_a_cached_request_that_is_not_in_flight_changes_nothing(self):
        scheduler, graft, registry = self.make()
        graft.commit(self.output(cached=[('decoder', 4096)], counts={'decoder': 1}))
        self.assertEqual(registry.inflight, {})


if __name__ == '__main__':
    unittest.main()
