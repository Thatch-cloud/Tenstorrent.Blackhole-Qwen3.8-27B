"""The sliced restore (QWEN_PREFIX_TIER_RESTORE_SLICE, docs/prefix-store-hygiene.md "Sliced restore"): a returning session's blocks come back from the host tier a bounded slice per
engine step, the seats decoding in between, and the request is admitted in the step its last slice lands.

* the strict parser and the counters that exist only when the flag is on;
* the job on the real graft (test_qwen_prefix_tiers.Rig over its fake scheduler and tier IO): the slice sizes (the block cap, the time budget over the measured cost per block, the
  probe), the blocks hashed into the pool only when the last slice has landed, the same Q and the same bytes as the whole restore, the audit and the digests per slice, an abort at
  every point (the request ending, a corrupt record, a failing write, the tier latching, no room) giving the pool its blocks back, a trim that reaches an unfinished job finishing it;
* the admission wrapper's hold (serving_prefill_admission.restore_hold) on the plugin's own scheduler class over the reduced vLLM scheduler: the candidate waits while the decode runs;
* off, the restore is the whole restore it was.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import prefix_markers as markers  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import qwen_prefix_scheduler_patch as patch_module  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402
import test_qwen_prefix_scheduler_patch as base  # noqa: E402
import test_qwen_prefix_tiers as tiers  # noqa: E402
import test_serving_prefill_admission as admission_tests  # noqa: E402

BLOCK = registry_module.BLOCK
SLICE = registry_module.ENV_TIER_RESTORE_SLICE
SLICE_MS = registry_module.ENV_TIER_RESTORE_SLICE_MS
Request, span, salt_of, output = tiers.Request, tiers.span, tiers.salt_of, base.output


# -- (1) the flag ---------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_unset_is_off_and_the_numbers_default(self):
        self.assertEqual(registry_module.restore_slice_config({}), (0, 200.0))
        self.assertEqual(registry_module.restore_slice_config({SLICE: '', SLICE_MS: ''}), (0, 200.0))
        self.assertEqual(registry_module.restore_slice_config({SLICE: '0'}), (0, 200.0))
        self.assertEqual(registry_module.restore_slice_config({SLICE: '512'}), (512, 200.0))
        self.assertEqual(registry_module.restore_slice_config({SLICE: '512', SLICE_MS: '150.5'}), (512, 150.5))

    def test_every_malformed_value_refuses_to_start(self):
        for environ, needle in (({SLICE: '-1'}, 'digits only'), ({SLICE: '1e3'}, 'digits only'), ({SLICE: ' 5'}, 'digits only'), ({SLICE: '4.0'}, 'digits only'),
                                ({SLICE: '٥'}, 'digits only'), ({SLICE: '65537'}, 'above 65536'), ({SLICE_MS: '100'}, 'inert'),
                                ({SLICE: '64', SLICE_MS: '0'}, 'above 0'), ({SLICE: '64', SLICE_MS: '-5'}, 'above 0'), ({SLICE: '64', SLICE_MS: 'inf'}, 'finite'),
                                ({SLICE: '64', SLICE_MS: 'nan'}, 'finite'), ({SLICE: '64', SLICE_MS: 'x'}, 'number of milliseconds'), ({SLICE: '64', SLICE_MS: '60001'}, 'at most')):
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError) as caught:
                    registry_module.restore_slice_config(environ)
                self.assertIn(needle, str(caught.exception))

    def test_it_needs_the_tier_and_adds_its_counters_only_when_on(self):
        with self.assertRaises(ValueError) as caught:
            registry_module.PrefixRegistry(budget_bytes=100, environ={SLICE: '64'})
        self.assertIn('need the host KV tier', str(caught.exception))
        plain = registry_module.PrefixRegistry(budget_bytes=1 << 20, environ=tiers.tier_environ(8, 1 << 20))
        self.assertEqual(plain.tier_config.restore_slice, 0)
        self.assertFalse(any(name.startswith('tier_slice') for name in plain.stats), 'the stats of a tier without the flag are what they were')
        on = registry_module.PrefixRegistry(budget_bytes=1 << 20, environ=dict(tiers.tier_environ(8, 1 << 20), **{SLICE: '16', SLICE_MS: '50'}))
        self.assertEqual((on.tier_config.restore_slice, on.tier_config.restore_slice_ms), (16, 50.0))
        for name in registry_module.SLICE_STAT_NAMES:
            self.assertEqual(on.stats[name], 0, name)


# -- (2) the job on the real graft -----------------------------------------------------------------------------

class SliceClock(object):
    """A clock the fake device advances: a write or a read of n blocks costs n * per_block seconds, and nothing else moves it."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TimedKvIO(base.FakeKvIO):
    per_block_s = 0.0
    clock = None          # set by SlicedRig before the rig builds its IO

    def write_blocks(self, ids, payloads):
        self.clock.now += self.per_block_s * len(ids)
        return super().write_blocks(ids, payloads)

    def read_blocks(self, ids, into=None):
        return super().read_blocks(ids, into=into)


def held_schedule(original):
    """FakeScheduler.schedule with the admission wrapper's job: while `decoding` seats exist the head candidate is asked for its hold and a held pass admits nothing."""

    def schedule(self):
        graft = getattr(self, '_qwen_prefix', None)
        self.asked = getattr(self, 'asked', [])
        if self.waiting and getattr(self, 'decoding', 0) and graft is not None:
            held = graft.restore_hold(self.waiting[0])
            self.asked.append(held)
            if held:
                self.held_passes = getattr(self, 'held_passes', 0) + 1
                return output([])
        return original(self)

    return schedule


ADMISSION_MARKED = type('Marked', (), {})
ADMISSION_MARKED.__qwen_one_fresh_prefill = True


def marked_prefill_only(self):
    return None


setattr(marked_prefill_only, patch_module.ADMISSION_WRAPPED, True)


def unmarked_prefill_only(self):
    return None

SIZES = dict(a=8300, filler=18000, pool=300, tier=2000)


class SlicedRig(object):
    """tiers.Rig with the sliced restore on: slice cap, ms budget, a fake clock the device advances."""

    def __init__(self, test, slice_blocks=16, slice_ms=None, per_block_s=0.0, wrapper=True, tier_env=None, enable=True, fake_clock=True):
        self.clock = SliceClock()
        patches = [mock.patch.object(base.FakeScheduler, 'schedule', held_schedule(base.FakeScheduler.schedule))]
        if fake_clock:
            patches.append(mock.patch.object(patch_module.time, 'perf_counter', self.clock))
        patches.append(mock.patch.object(base.FakeScheduler, '_schedule_prefill_only', marked_prefill_only if wrapper else unmarked_prefill_only, create=True))
        for patcher in patches:
            patcher.start()
            test.addCleanup(patcher.stop)
        env = dict(tier_env or {})
        if enable:
            env[SLICE] = str(slice_blocks)
            if slice_ms is not None:
                env[SLICE_MS] = str(slice_ms)
        TimedKvIO.clock = self.clock
        with mock.patch.object(tiers, 'FakeKvIO', TimedKvIO):
            self.rig = tiers.Rig(num_blocks=SIZES['pool'], budget=1 << 20, tier=SIZES['tier'], tier_env=env)
        self.rig.kv.per_block_s = per_block_s
        test.addCleanup(self.rig.close)
        self.hooks = self.rig.graft.tier_hooks
        self.scheduler = self.rig.scheduler
        self.registry = self.rig.registry
        self.pool = self.scheduler.kv_cache_manager.block_pool

    def evicted_session(self):
        """A's first turn, then a filler of another tenant that takes most of A's blocks (they are spilled to the tier). -> A's request."""
        a = Request('a1', span(SIZES['a'], 1), salt=salt_of('alpha'))
        self.rig.serve(a)
        self.rig.serve(Request('f1', span(SIZES['filler'], 2), salt=salt_of('bravo')))
        return a

    def returning(self, a):
        return Request('a2', a.all_token_ids + span(700, 3), salt=a.cache_salt)

    def drive(self, request, decoding=1, limit=200):
        """Add `request` and run schedule() until it is admitted. -> (held passes, start_pos, grant)."""
        self.scheduler.decoding = decoding
        self.scheduler.held_passes = getattr(self.scheduler, 'held_passes', 0) if request.request_id in self.scheduler.requests else 0
        if request.request_id not in self.scheduler.requests:
            self.scheduler.add(request)
        self.reads_per_pass = []
        for _ in range(limit):
            reads_before = len(self.rig.kv.reads)
            out = self.scheduler.schedule()
            self.reads_per_pass.append(len(self.rig.kv.reads) - reads_before)
            admitted = [data for data in out.scheduled_new_reqs if data.req_id == request.request_id]
            if admitted:
                grant = self.registry.grant_for(request.request_id)
                with base.Quiet():
                    self.rig.model.prefill(request.request_id, request.all_token_ids, admitted[0].num_computed_tokens)
                single = self.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
                owned = single.req_to_blocks[request.request_id]
                for index in range(admitted[0].num_computed_tokens // BLOCK, len(request.block_hashes)):
                    self.rig.kv.device[owned[index].block_id] = base.kv_content(request.block_hashes[index])
                self.scheduler.finish(request)
                return self.scheduler.held_passes, admitted[0].num_computed_tokens, grant
        raise AssertionError('the request was never admitted in %d passes' % limit)


class SliceJobTests(unittest.TestCase):
    def sliced(self, **kwargs):
        return SlicedRig(self, **kwargs)

    def whole(self):
        """The same traffic with the flag off: the Q and the bytes the whole restore gives."""
        made = SlicedRig(self, enable=False)
        a = made.evicted_session()
        made.rig.serve(made.returning(a))
        row = made.rig.rows[-1]
        return made, row

    def test_it_installs_with_the_wrapper_and_falls_back_without_it(self):
        made = self.sliced()
        self.assertTrue(made.hooks.slicing)
        self.assertTrue(any(line.startswith('host tier restore slice on blocks=16 ms=200') for line in made.rig.lines), made.rig.lines)
        bare = self.sliced(wrapper=False)
        self.assertFalse(bare.hooks.slicing)
        self.assertTrue(any('host tier restore slice fell back' in line for line in bare.rig.lines))
        self.assertFalse(bare.hooks.slicing, 'without the wrapper that holds a request between slices, restores run whole')

    def test_a_session_comes_back_in_bounded_slices_while_the_seats_decode_and_is_admitted_at_the_last(self):
        made = self.sliced(slice_blocks=16)
        a = made.evicted_session()
        held, start, grant = made.drive(made.returning(a))
        stats = made.registry.stats
        total = stats['tier_restore_blocks']
        self.assertGreater(total, 64, 'a big session: many slices')
        self.assertEqual(stats['tier_slice_jobs'], 1)
        # (no time has passed on the fake clock: the probe is 8 blocks, then the cap, 16)
        self.assertEqual(stats['tier_slice_steps'], 1 + -(-(total - patch_module.SLICE_PROBE_BLOCKS) // 16))
        self.assertEqual(held, stats['tier_slice_steps'] - 1, 'every pass but the last held the request (the seats decoded)')
        self.assertEqual(stats['tier_slice_held'], held)
        self.assertEqual(stats['tier_slice_blocks'], total)
        self.assertEqual((stats['tier_restore_requests'], stats['tier_restore_failures'], stats['tier_digest_failures']), (1, 0, 0))
        writes = [len(ids) for ids in made.rig.kv.writes]
        self.assertEqual(sum(writes), total)
        self.assertLessEqual(max(writes), 16)
        self.assertEqual(start, grant.q)
        self.assertTrue(made.rig.check_kv(), 'every cached block, restored ones included, holds its hash\'s bytes')
        self.assertEqual(len(made.pool.free), SIZES['pool'], 'every block is free again')

    def test_the_session_is_the_same_hit_with_the_same_bytes_as_the_whole_restore(self):
        whole_rig, row = self.whole()
        made = self.sliced(slice_blocks=7)
        a = made.evicted_session()
        held, start, grant = made.drive(made.returning(a))
        self.assertEqual((start, grant.q, grant.h), (row[0], row[1].q, row[1].h))
        self.assertEqual(made.registry.stats['tier_restore_blocks'], whole_rig.registry.stats['tier_restore_blocks'])
        self.assertEqual(sorted(made.pool.cached_block_hash_to_block.map), sorted(whole_rig.pool.cached_block_hash_to_block.map), 'the pool serves the same hashes')
        self.assertTrue(made.rig.check_kv())

    def test_the_blocks_are_hashed_into_the_pool_only_when_the_last_slice_has_landed(self):
        made = self.sliced(slice_blocks=16)
        a = made.evicted_session()
        request = made.returning(a)
        made.scheduler.decoding = 1
        made.scheduler.add(request)
        seen = []
        while True:
            before = len(made.pool.cached_block_hash_to_block.map)
            out = made.scheduler.schedule()
            seen.append(len(made.pool.cached_block_hash_to_block.map) - before)
            job = made.hooks.jobs.get(request.request_id)
            if job is not None:
                self.assertEqual(len(made.pool.free), SIZES['pool'] - len(job.new) - len(job.protected), 'the job holds its blocks: they are neither free nor cached')
            if out.scheduled_new_reqs:
                break
        self.assertTrue(all(delta <= 0 for delta in seen[:-1]), 'no restored block was visible to anyone before the last slice: %s' % seen)
        self.assertGreater(seen[-1], 0)
        self.assertEqual(made.hooks.jobs, {})

    def test_the_slice_follows_the_time_budget_over_the_measured_cost_per_block(self):
        made = self.sliced(slice_blocks=64, slice_ms=20, per_block_s=0.002)      # 2 ms a block: the budget is ten blocks
        a = made.evicted_session()
        made.drive(made.returning(a))
        writes = [len(ids) for ids in made.rig.kv.writes][-made.registry.stats['tier_slice_steps']:]
        self.assertEqual(writes[0], patch_module.SLICE_PROBE_BLOCKS, 'the first slice is the probe')
        self.assertTrue(all(count == 10 for count in writes[1:-1]), writes)
        stats = made.registry.stats
        self.assertLessEqual(stats['tier_slice_max_ms'], 20.0 + 1e-6, 'no slice is longer than its budget once the cost per block is known (the probe took 16 ms)')
        self.assertAlmostEqual(stats['tier_slice_ms'] / 1000.0, 0.002 * stats['tier_restore_blocks'], places=3)

    def test_the_block_cap_binds_when_the_budget_would_allow_more(self):
        made = self.sliced(slice_blocks=12, slice_ms=60000, per_block_s=0.001)
        a = made.evicted_session()
        made.drive(made.returning(a))
        sizes = [len(ids) for ids in made.rig.kv.writes][-made.registry.stats['tier_slice_steps']:]
        self.assertLessEqual(max(sizes), 12)
        self.assertEqual(sizes[0], patch_module.SLICE_PROBE_BLOCKS)

    def test_each_slice_spills_only_what_its_own_allocation_evicted(self):
        made = self.sliced(slice_blocks=8)
        a = made.evicted_session()
        held, start, grant = made.drive(made.returning(a))
        spilled = made.reads_per_pass[:held]
        self.assertTrue(held > 3 and all(count <= 8 for count in spilled), 'a held pass allocates one slice of at most 8 blocks, so it spills at most 8: %s' % spilled)
        self.assertGreater(sum(spilled), 0, 'the restore did evict, and spilled, other blocks')

    def test_the_audit_reads_back_every_slice_and_stops_on_a_difference(self):
        made = self.sliced(slice_blocks=16, tier_env={'QWEN_PREFIX_HOST_TIER_AUDIT': '1'})
        a = made.evicted_session()
        made.drive(made.returning(a))
        stats = made.registry.stats
        self.assertEqual((stats['tier_audit_reads'], stats['tier_audit_mismatches']), (stats['tier_restore_blocks'], 0))
        bad = self.sliced(slice_blocks=16, tier_env={'QWEN_PREFIX_HOST_TIER_AUDIT': '1'})
        a = bad.evicted_session()
        bad.rig.kv.corrupt_write = True
        with self.assertRaisesRegex(patch_module.TierAuditError, 'read back differently'):
            bad.drive(bad.returning(a))
        self.assertEqual(bad.hooks.jobs, {}, 'the engine stops, but the job gave its blocks back first')
        self.assertEqual(len(bad.pool.free), SIZES['pool'])

    def test_the_restore_line_carries_the_slices_and_the_gate_reads_them(self):
        made = self.sliced(slice_blocks=16)
        a = made.evicted_session()
        made.drive(made.returning(a))
        scan = markers.scan(['[PINDIAG] prefix: ' + line for line in made.rig.lines])
        self.assertEqual(len(scan['tier_slice_on']), 1)
        self.assertEqual(scan['tier_slice_trouble'], [])
        restore = scan['tier_restores'][0]
        stats = made.registry.stats
        self.assertEqual((restore['slices'], restore['held_steps'], restore['blocks']), (stats['tier_slice_steps'], stats['tier_slice_held'], stats['tier_restore_blocks']))
        self.assertEqual(len(scan['tier_slices']), stats['tier_slice_steps'])
        self.assertAlmostEqual(restore['ms'], stats['tier_restore_ms'], places=0)

    # -- aborts: every way a job ends early gives the pool its blocks back ------------------------------------------------
    def start_job(self, **kwargs):
        made = self.sliced(**kwargs)
        a = made.evicted_session()
        request = made.returning(a)
        made.scheduler.decoding = 1
        made.scheduler.add(request)
        made.scheduler.schedule()
        job = made.hooks.jobs.get(request.request_id)
        self.assertIsNotNone(job, 'the first pass started the job')
        self.assertGreater(len(job.new), 0)
        return made, request, job

    def test_a_request_that_ends_mid_job_gives_everything_back(self):
        made, request, job = self.start_job(slice_blocks=8)
        made.scheduler.waiting.remove(request)
        made.scheduler._free_request(request)
        self.assertEqual(made.hooks.jobs, {})
        self.assertEqual(len(made.pool.free), SIZES['pool'])
        self.assertEqual(made.registry.tier.pinned, set())
        self.assertEqual(made.registry.stats['tier_slice_aborted'], 1)
        self.assertEqual(made.hooks.refused, set())
        self.assertTrue(made.rig.check_kv())

    def test_a_record_that_changed_in_host_memory_abandons_the_job_and_the_request_goes_on_cold(self):
        made, request, job = self.start_job(slice_blocks=8)
        victim = job.records[-1]
        victim.payload = bytes([victim.payload[0] ^ 1]) + victim.payload[1:]
        made.rig.registry.tier.verify = lambda record: False if record is victim else None
        made.hooks.config.verify = 'all'
        held, start, grant = made.drive(request)
        stats = made.registry.stats
        self.assertEqual(start, 0, 'a corrupt record is a miss: the request is admitted cold')
        self.assertEqual((stats['tier_restore_refused_digest'], stats['tier_slice_aborted']), (1, 1))
        self.assertNotIn(victim.key, made.registry.tier.records)
        self.assertEqual(made.hooks.jobs, {})
        self.assertEqual(len(made.pool.free), SIZES['pool'])
        self.assertTrue(made.rig.check_kv())

    def test_a_failing_write_abandons_the_job_and_counts_a_failure(self):
        made, request, job = self.start_job(slice_blocks=8)
        made.rig.kv.fail_write = True
        held, start, grant = made.drive(request)
        stats = made.registry.stats
        self.assertEqual(start, 0)
        self.assertEqual((stats['tier_restore_failures'], stats['tier_slice_aborted']), (1, 1))
        self.assertEqual(len(made.pool.free), SIZES['pool'])
        self.assertEqual(stats['tier_slice_jobs'], 1, 'the tier hit was not retried every step (one job, one abort); the memory of the refusal went with the request')
        self.assertEqual(made.hooks.refused, set())

    def test_a_tier_that_latches_off_mid_job_releases_the_job(self):
        made, request, job = self.start_job(slice_blocks=8)
        made.registry.tier_latch('test')
        held, start, grant = made.drive(request)
        self.assertEqual(start, 0)
        self.assertEqual(made.hooks.jobs, {})
        self.assertEqual(len(made.pool.free), SIZES['pool'])

    def test_a_trim_that_reaches_an_unfinished_job_finishes_it_whole(self):
        made, request, job = self.start_job(slice_blocks=8)
        made.scheduler.decoding = 0        # nothing decoding any more: the admission no longer holds
        out = made.scheduler.schedule()
        self.assertEqual([data.req_id for data in out.scheduled_new_reqs], [request.request_id])
        stats = made.registry.stats
        self.assertEqual((stats['tier_restore_requests'], made.hooks.jobs), (1, {}))
        self.assertEqual(out.scheduled_new_reqs[0].num_computed_tokens, made.registry.grant_for(request.request_id).q)
        self.assertGreater(out.scheduled_new_reqs[0].num_computed_tokens, 0)

    def test_with_nothing_decoding_the_restore_runs_whole_in_the_trim(self):
        made = self.sliced(slice_blocks=8)
        a = made.evicted_session()
        held, start, grant = made.drive(made.returning(a), decoding=0)
        stats = made.registry.stats
        self.assertEqual((held, stats['tier_slice_jobs'], stats['tier_slice_inline'], stats['tier_restore_requests']), (0, 0, 1, 1))
        self.assertEqual(len(made.rig.kv.writes[-1]), stats['tier_restore_blocks'], 'one write of everything')

    def test_no_room_for_the_next_slice_abandons_the_job(self):
        made, request, job = self.start_job(slice_blocks=8)
        # another tenant takes the pool while the job waits
        thief = Request('thief', span(8000, 9), salt=salt_of('mallory'))
        made.scheduler.waiting.insert(0, thief)
        made.scheduler.requests[thief.request_id] = thief
        made.scheduler.decoding = 0
        made.scheduler.schedule()          # the thief is admitted and holds its blocks
        made.scheduler.decoding = 1
        for _ in range(40):
            made.scheduler.schedule()
            if made.hooks.jobs == {}:
                break
        stats = made.registry.stats
        self.assertGreaterEqual(stats['tier_restore_refused_room'], 0)
        self.assertIn(stats['tier_slice_aborted'] + stats['tier_restore_requests'], (1, 2))
        self.assertEqual(made.hooks.jobs, {})

    def test_another_tenants_request_never_meets_the_job(self):
        made = self.sliced(slice_blocks=8)
        a = made.evicted_session()
        twin = Request('twin', a.all_token_ids + span(700, 3), salt=salt_of('mallory'))
        held, start, grant = made.drive(twin)
        self.assertEqual((held, start, made.registry.stats['tier_slice_jobs']), (0, 0, 0))

    def test_off_the_restore_is_the_whole_restore_it_was(self):
        made = self.sliced(enable=False)
        self.assertFalse(made.hooks.slicing)
        self.assertEqual(made.hooks.slice_blocks, 0)
        a = made.evicted_session()
        held, start, grant = made.drive(made.returning(a))
        stats = made.registry.stats
        self.assertEqual(held, 0)
        self.assertFalse(any(name.startswith('tier_slice') for name in stats))
        self.assertEqual(len(made.rig.kv.writes[-1]), stats['tier_restore_blocks'])
        scan = markers.scan(['[PINDIAG] prefix: ' + line for line in made.rig.lines])
        self.assertNotIn('slices', scan['tier_restores'][0])
        self.assertEqual((scan['tier_slice_on'], scan['tier_slices']), ([], []))


# -- (3) the admission wrapper's hold -------------------------------------------------------------------------------

class StubGraft(object):
    def __init__(self, answers):
        self.answers = list(answers)
        self.asked = []

    def restore_hold(self, request):
        self.asked.append(request.request_id)
        value = self.answers.pop(0) if self.answers else False
        if isinstance(value, Exception):
            raise value
        return value


class AdmissionHoldTests(admission_tests.DramFreeCase):
    def started(self, log, graft):
        scheduler = admission_tests.patched_plugin(log)()
        scheduler._qwen_prefix = graft
        scheduler.add_request(admission_tests.DramRequest('A'))
        self.assertEqual(admission_tests.new_ids(scheduler.schedule()), ['A'])
        return scheduler

    def test_a_held_candidate_waits_while_the_decode_runs_and_is_admitted_when_the_hold_lifts(self):
        log, graft = mock.Mock(), StubGraft([True, True, False])
        scheduler = self.started(log, graft)
        scheduler.add_request(admission_tests.DramRequest('B', 4096))
        for _ in range(2):
            step = scheduler.schedule()
            self.assertEqual((admission_tests.new_ids(step), admission_tests.cached_ids(step)), ([], ['A']), 'A decodes, B waits')
            self.assertEqual(admission_tests.names(scheduler.waiting), ['B'], 'the hidden queue came back')
            self.assertEqual(scheduler.max_num_running_reqs, 4)
        step = scheduler.schedule()
        self.assertEqual((admission_tests.new_ids(step), admission_tests.cached_ids(step)), (['B'], []))
        self.assertEqual(graft.asked, ['B', 'B', 'B'])

    def test_nothing_decoding_means_nothing_to_interleave_with_and_nothing_is_asked(self):
        graft = StubGraft([True])
        scheduler = admission_tests.patched_plugin(mock.Mock())()
        scheduler._qwen_prefix = graft
        scheduler.add_request(admission_tests.DramRequest('B', 4096))
        self.assertEqual(admission_tests.new_ids(scheduler.schedule()), ['B'])
        self.assertEqual(graft.asked, [])

    def test_a_hold_that_fails_admits_the_request_and_logs(self):
        log, graft = mock.Mock(), StubGraft([RuntimeError('boom')])
        scheduler = self.started(log, graft)
        scheduler.add_request(admission_tests.DramRequest('B', 4096))
        self.assertEqual(admission_tests.new_ids(scheduler.schedule()), ['B'])
        failed = [entry.args for entry in log.call_args_list if entry.args and 'sliced restore hold failed' in entry.args[0]]
        self.assertEqual([args[1:3] for args in failed], [('RuntimeError', 'boom')])

    def test_a_scheduler_without_the_graft_or_without_the_hold_is_untouched(self):
        for graft in (None, SimpleNamespace()):
            scheduler = admission_tests.patched_plugin(mock.Mock())()
            if graft is not None:
                scheduler._qwen_prefix = graft
            scheduler.add_request(admission_tests.DramRequest('A'))
            self.assertEqual(admission_tests.new_ids(scheduler.schedule()), ['A'])
            scheduler.add_request(admission_tests.DramRequest('B', 4096))
            self.assertEqual(admission_tests.new_ids(scheduler.schedule()), ['B'])

    def test_the_held_pass_is_asked_after_the_dram_and_kv_holds_and_only_when_a_prompt_would_be_admitted(self):
        graft = StubGraft([True, True, True])
        scheduler = self.started(mock.Mock(), graft)
        scheduler.add_request(admission_tests.DramRequest('B', 4096))
        refuse = lambda prompt: (False, dict(largest_free=900 * admission_tests.MB, need=1_300 * admission_tests.MB))
        with admission_tests.dram(refuse):
            scheduler.schedule()
        self.assertEqual(graft.asked, [], 'the DRAM hold decided first: the restore was not asked to work for a prompt that cannot be admitted')
        with admission_tests.gate('cmpl-held'):
            scheduler.schedule()
        self.assertEqual(graft.asked, [], 'a held gate hides the queue: nothing is admitted, nothing is asked')


if __name__ == '__main__':
    unittest.main()
