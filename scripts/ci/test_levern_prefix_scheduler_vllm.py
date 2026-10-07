"""Lever N with prefix reuse: the merged scheduler side PROVED on real vLLM 0.25.1 objects (qwen-fast-vllm-cpu.yml; skipped where vLLM is not installed).

test_levern_prefix_scheduler holds the runtime on a reduced model of vLLM's Scheduler.schedule and test_qwen_prefix_scheduler_vllm holds the G1 graft on real vLLM
with chunking off. This module runs BOTH together, as the merged profile does: the plugin's own TTScheduler as the image's stage patches it (the pinned
scheduler.py with the graft hook, staged by qwen_prefix_scheduler_patch.stage), the class wrapped by serving_prefill_admission.install under
QWEN_FAST_LEVER_N=1 beside QWEN_PREFIX_REUSE=1 (the one-fresh-prefill cap, the merged runtime, the KV reservation, the quarantine consumer), then the graft
installed on the instance with QWEN_FAST_STICKY_SESSIONS=1 and chunked prefill ON beside the DFlash lookahead of 16 (R2: the install accepts chunking only
there). vLLM's real Scheduler, KVCacheManager, BlockPool and Request run underneath. A fake TT model stands in for the device: MergedModel carries a GDN state
that is a hash chain over the 2,048-token chunks it ran (test_qwen_prefix_scheduler_patch's chain), resolves every step's source as levern_route does
(COLD, CHECKPOINT from the committed grant, SCRATCH from its owner, PARKED from its park), asserts that the state each step starts from equals the COLD chain
of the request's own tokens at that position, takes the planned captures with loop_pos == pos (through the registry's in-flight plan), and records what it wrote.

What is proved:
  V1  a conversation of five turns, each extending the last by 64 answer tokens and 2,600 new ones, with a decoder running:
        - every turn's hit Q, vLLM's raw hit h and the planned captures are what prefix_judge.Oracle(sticky=True) predicts (the same as with Lever N off);
        - every step of every turn is levern_policy.plan_from(P, Q): a hit arrives split (CHECKPOINT, then SCRATCH steps), every non-final end on a 2,048
          boundary at or below S_last, the final step holding the drafter window;
        - the state the final step ends in equals the cold chain over the whole prompt, so a hit equals its cold twin;
        - after every turn the registry holds exactly the keys, positions and states a run with Lever N off holds (chunking off, same prompts);
        - the peek leaves vLLM's prefix-cache statistics alone (the in-pass call records each attempt once), no request is preempted, none holds more
          blocks than it reserved, every block is free again at the end;
  D1  the negative control: a hit at Q = S_last with the cap computed from (P, 0) is refused by verify() as a step past S_last, and with the peek it is one
      final step;
  V2  a hit turn arriving during a cold long prefill is admitted at the next prefill pass (the cold one waits no more than that step), parked on the
      host by the model, resumes from PARKED at exactly the position it stopped at, and both end in the cold chain; at most two partials; no preemption;
  Q   a partial prefill whose suspended state is gone is quarantined BEFORE the pass (never scheduled, no block allocated for it), finished as
      FINISHED_ABORTED by the D2 consumer on this very scheduler, the decoder keeps decoding, every block is free again, and no cached block was published
      before its tokens were written.
Run with VLLM_USE_V2_MODEL_RUNNER=0 (as the other installed-vLLM suites)."""

import collections
import os
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import levern_policy  # noqa: E402
import levern_scheduler  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402
import serving_request_quarantine as quarantine  # noqa: E402
import test_qwen_prefix_scheduler_vllm as proof  # noqa: E402
from test_qwen_prefix_scheduler_patch import INITIAL_STATE, ModelAssertion, chunk_state, cold_state  # noqa: E402
from test_qwen_prefix_scheduler_vllm import (BLOCK, CHUNK, DEFAULT_BLOCKS, MAX_MODEL_LEN, SALT, VLLM_ERROR, StickyEnv, attach_ledger,  # noqa: E402
                                            tokens)
from test_qwen_prefix_scheduler_vllm import setUpModule, tearDownModule  # noqa: E402,F401  (the module-level fixtures stage the graft package)

if VLLM_ERROR is None:
    from vllm.config import SchedulerConfig, VllmConfig
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
    from vllm.v1.request import RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

MERGED_ENV = {'QWEN_FAST_LEVER_N': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1', 'QWEN_FAST_ANY_REQUEST': '1',
              'QWEN_FAST_LEVERN_ROUNDS': '1', 'QWEN_FAST_LEVERN_TTFT_TARGET_S': '0', 'QWEN_FAST_LEVERN_PARK': 'host',
              'QWEN_FAST_LEVERN_SHORT_TOKENS': '16384', 'QWEN_FAST_KV_RESERVATION': '1'}
GRAFT_ENV = {'QWEN_FAST_STICKY_SESSIONS': '1', 'QWEN_FAST_LEVER_N': '1'}
FLAGS = levern_policy.ALL_FLAGS + (admission.STEPS_FLAG, 'QWEN_FAST_KV_RESERVATION', 'QWEN_PREFIX_REUSE', 'QWEN_FAST_STICKY_SESSIONS')
DECODER_ANSWER = 20000


class MergedModel(object):
    """The merged route's side of every step, on the hash-chain GDN (the module docstring)."""

    def __init__(self, registry):
        self.registry = registry
        self.owner = None
        self.state = None
        self.parks = {}
        self.written = {}
        self.final = {}
        self.sources = collections.defaultdict(list)
        self.steps = collections.defaultdict(list)
        self.max_parks = 0

    def publish(self):
        holder = levern_policy.state_holder()
        holder.owner = self.owner
        holder.parks = {request_id: position for request_id, (position, _) in self.parks.items()}
        holder.route = True

    def step(self, request_id, prompt, start, end, total):
        registry = self.registry
        grant = registry.grant_for(request_id)
        if start == 0:
            source, state = 'COLD', INITIAL_STATE
            if grant is not None and grant.q != 0:
                raise ModelAssertion('a cold start beside a grant at Q=%d' % grant.q)
        elif grant is not None and grant.q == start:
            source = 'CHECKPOINT'
            if not grant.checkpoint.matches(prompt[0:start]):
                raise ModelAssertion('row %s: the checkpoint tokens differ from the prompt below %d' % (request_id, start))
            state = grant.checkpoint.rec
            registry.note_restore(1.0)
        elif self.owner == (request_id, start):
            source, state = 'SCRATCH', self.state
        elif request_id in self.parks and self.parks[request_id][0] == start:
            source, state = 'PARKED', self.parks.pop(request_id)[1]
        else:
            raise ModelAssertion('row %s [%d, %d): no source holds the state after %d tokens (grant=%r owner=%r parks=%r)' % (
                request_id, start, end, start, None if grant is None else grant.q, self.owner, sorted(self.parks)))
        if state != cold_state(prompt, start):
            raise ModelAssertion('row %s: the state it %s at %d is not the cold chain\'s' % (request_id, source, start))
        if self.owner is not None and self.owner[0] != request_id:
            # the route parks the prefill suspended on the scratch for the one that takes it (admission v2)
            self.parks[self.owner[0]] = (self.owner[1], self.state)
            self.max_parks = max(self.max_parks, len(self.parks))
        last = end // CHUNK * CHUNK
        plan = [position for position in registry.planned(request_id) if start < position <= last]
        for index in range(start // CHUNK, end // CHUNK):
            state = chunk_state(state, prompt[index * CHUNK:(index + 1) * CHUNK])
            done = (index + 1) * CHUNK
            if done in plan:
                registry.capture(request_id, done, rec=state, carry=b'carry', nbytes=1, ms=0.5, loop_pos=done)
        self.written[request_id] = end
        if end >= total:
            self.owner = self.state = None
            self.final[request_id] = state
        else:
            self.owner, self.state = (request_id, end), state
        self.sources[request_id].append(source)
        self.steps[request_id].append((start, end))
        self.publish()

    def forget(self, request_id):
        if self.owner is not None and self.owner[0] == request_id:
            self.owner = self.state = None
        self.parks.pop(request_id, None)
        self.publish()


class MergedDrive(object):
    """schedule -> the merged route's fake model -> update_from_output, as EngineCore.step does it (sync), recording what the proof reads."""

    def __init__(self, env, scheduler, state, name, ledger=None):
        self.env, self.scheduler, self.state = env, scheduler, state
        self.registry = state.registry
        self.model = MergedModel(self.registry)
        self.ledger = {} if ledger is None else ledger
        self.records = []
        self.grants = collections.defaultdict(list)
        self.preempted = set()
        self.over = 0
        self.rng = random.Random(name)
        self.unwritten = []
        self.model.publish()

    def add(self, request_id, prompt, max_tokens, salt=SALT):
        request = self.env.request(request_id, prompt, max_tokens, salt)
        self.scheduler.add_request(request)
        return request

    def prefills_of(self, output):
        found = []
        for data in output.scheduled_new_reqs:
            start = data.num_computed_tokens
            found.append((data.req_id, start, start + output.num_scheduled_tokens[data.req_id]))
        cached = output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            request = self.scheduler.requests[request_id]
            start = cached.num_computed_tokens[index]
            if start < request.num_prompt_tokens:
                found.append((request_id, start, start + output.num_scheduled_tokens[request_id]))
        return found

    def step(self):
        import serving_kv_reservation as kv

        scheduler = self.scheduler
        output = scheduler.schedule()
        self.preempted |= set(getattr(output, 'preempted_req_ids', None) or ())
        for request in scheduler.requests.values():
            if getattr(request, 'num_preemptions', 0) or request.status == RequestStatus.PREEMPTED:
                self.preempted.add(request.request_id)
        for request in scheduler.running:
            held = len(scheduler.kv_cache_manager.get_block_ids(request.request_id)[0])
            self.over = max(self.over, held - kv.request_reservation(request, scheduler.max_model_len))
        prefills = self.prefills_of(output)
        if len(prefills) > 1:
            raise AssertionError('two prefill rows in one step: %r' % (prefills,))
        sampled = {}
        for request_id, start, end in prefills:
            request = scheduler.requests[request_id]
            grant = self.registry.grant_for(request_id)
            if grant is not None:
                self.grants[request_id].append(grant)
            total = request.num_prompt_tokens
            self.model.step(request_id, list(request.prompt_token_ids), start, end, total)
            sampled[request_id] = [] if end < total else [self.rng.randrange(1000, 200000)]
        self.records.append(('prefill', prefills[0]) if prefills else ('decode', len(output.num_scheduled_tokens)))
        if not output.num_scheduled_tokens:
            runner_output = EMPTY_MODEL_RUNNER_OUTPUT
        else:
            ids = list(output.num_scheduled_tokens)
            runner_output = ModelRunnerOutput(req_ids=ids, req_id_to_index=dict((request_id, index) for index, request_id in enumerate(ids)),
                                              sampled_token_ids=[sampled.get(request_id, [self.rng.randrange(1000, 200000)]) for request_id in ids])
        scheduler.update_from_output(output, runner_output)
        for request_id in [value for value in list(self.model.written) if value not in scheduler.requests]:
            self.model.forget(request_id)
        self.check_published()
        return output

    def check_published(self):
        """No cached block was published before its tokens were written."""
        pool = self.scheduler.kv_cache_manager.block_pool
        for block_id, (request_id, index, _) in self.ledger.items():
            block = pool.blocks[block_id]
            if block.block_hash is not None and (index + 1) * BLOCK > self.model.written.get(request_id, 0):
                self.unwritten.append((request_id, index, self.model.written.get(request_id, 0)))

    def run_until(self, done, limit=4000):
        for _ in range(limit):
            if done():
                return
            self.step()
        raise AssertionError('the scheduler did not reach the condition in %d steps' % limit)

    def finished(self, request_id):
        return lambda: request_id not in self.scheduler.requests

    def prefill_steps(self, request_id):
        return [(start, end) for kind, value in self.records if kind == 'prefill' for rid, start, end in [value] if rid == request_id]


class MergedEnv(StickyEnv):
    """StickyEnv (DFlash with 15 proposals, 64-token blocks, prefix caching, the 65,536-token window) with chunked prefill ON and the whole-window token
    budget: the merged profile's engine argv (max-num-batched-tokens equal to max-model-len, enable-chunked-prefill)."""

    def __init__(self):
        super(MergedEnv, self).__init__()
        base = self.vllm_config
        scheduler_config = SchedulerConfig(max_num_seqs=6, max_num_batched_tokens=MAX_MODEL_LEN, max_model_len=MAX_MODEL_LEN, is_encoder_decoder=False,
                                           enable_chunked_prefill=True, async_scheduling=False)
        self.vllm_config = VllmConfig(model_config=base.model_config, device_config=base.device_config, scheduler_config=scheduler_config,
                                      cache_config=base.cache_config, parallel_config=base.parallel_config, speculative_config=base.speculative_config)
        self.vllm_config.cache_config.num_gpu_blocks = DEFAULT_BLOCKS
        register_all_kvcache_specs(self.vllm_config)
        self.structured = StructuredOutputManager(self.vllm_config)

    def merged(self, merged=True, park=True, reservation=True, blocks=None):
        """(scheduler, graft, runtime log, ledger): the class wrapped first (the cap, the runtime, the reservation, the consumer), the graft then installed
        on the instance, in the order the engine runs them."""
        scheduler_type = type('TTSchedulerMerged', (self.scheduler_cls,), {})
        lines = []
        environ = dict(MERGED_ENV)
        if not park:
            environ['QWEN_FAST_LEVERN_PARK'] = '0'
        if not reservation:
            environ['QWEN_FAST_KV_RESERVATION'] = '0'
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=scheduler_type))
        log = lambda message, *values: lines.append((message.format(*values) if '{}' in message else message % values) if values else message)  # noqa: E731
        with mock.patch.dict(os.environ, environ):
            for name in FLAGS:
                if name not in environ:
                    os.environ.pop(name, None)
            admission.install(config, log=log)
            quarantine.install(config, log=log)
        levern_policy.reset_state()
        scheduler, state = self.make_sticky(environ=dict(GRAFT_ENV), scheduler_cls=scheduler_type, num_blocks=blocks)
        ledger = {}
        attach_ledger(scheduler, ledger)
        return scheduler, state, lines, ledger


def reference_registry(env, prompts):
    """The registry a run with Lever N OFF leaves behind: the same prompts, chunking off, the graft's sticky plan, one request at a time."""
    scheduler, state = StickyEnv().make_sticky()
    drive = proof.Drive(env, scheduler, state, 'reference')
    for index, prompt in enumerate(prompts):
        drive.add('ref%d' % index, prompt, 1)
        drive.run()
    return state.registry


def registry_keys(registry):
    """What a registry holds, independent of the block-hash bytes (vLLM seeds the first block's parent hash per initialisation, so two environments
    key the same prefix differently): every checkpoint's position, the token ids it was captured from and its state, sorted."""
    return sorted((checkpoint.pos, bytes(checkpoint.token_ids), checkpoint.rec) for checkpoint in registry.entries.values())


@unittest.skipIf(VLLM_ERROR is not None, 'vLLM is not importable here (%s)' % VLLM_ERROR)
class MergedOnRealVllmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MergedEnv()

    def setUp(self):
        self.env.logs[:] = []
        self.addCleanup(levern_policy.reset_state)
        shared = quarantine.holder()
        saved = shared.installed, dict(shared.pending), shared.live
        self.addCleanup(lambda: (setattr(shared, 'installed', saved[0]), setattr(shared, 'pending', saved[1]), setattr(shared, 'live', saved[2])))

    def build(self, **kwargs):
        scheduler, state, lines, ledger = self.env.merged(**kwargs)
        drive = MergedDrive(self.env, scheduler, state, 'merged', ledger)
        return scheduler, state, lines, drive

    def decoder(self, drive, name='d0'):
        return drive.add(name, tokens(100, name), DECODER_ANSWER, salt=None)

    # -- the install ------------------------------------------------------------------------------------------------------------------------
    def test_the_graft_installs_chunked_beside_the_cap_and_the_runtime_is_the_merged_one(self):
        scheduler, state, lines, drive = self.build()
        self.assertIn('install sticky=1 lookahead=16 drop_last=True ceiling=floor2048(P-2048)', self.env.logs)
        self.assertTrue(any(line.startswith('install chunked=levern') for line in self.env.logs), self.env.logs)
        self.assertIsNotNone(scheduler.vllm_config.scheduler_config.enable_chunked_prefill)
        self.assertTrue(scheduler.vllm_config.scheduler_config.enable_chunked_prefill)
        self.assertEqual(scheduler.max_num_scheduled_tokens, MAX_MODEL_LEN)
        self.assertTrue(getattr(type(scheduler).schedule, levern_scheduler.WRAPPED, False))
        self.assertTrue(getattr(type(scheduler)._schedule_prefill_only, admission.WRAPPED, False))
        self.assertIs(scheduler.__dict__['_qwen_prefix'], state)

    def test_without_the_cap_beside_it_the_graft_refuses_chunked_prefill_and_the_engine_does_not_start(self):
        plain = type('TTSchedulerPlain', (self.env.scheduler_cls,), {})
        with self.assertRaisesRegex(self.env.graft.PrefixInstallError, 'chunked prefill is on'):
            self.env.make_sticky(environ=dict(GRAFT_ENV), scheduler_cls=plain)
        with self.assertRaisesRegex(self.env.graft.PrefixInstallError, 'the merged route accepts it only beside'):
            self.env.make_sticky(environ={'QWEN_FAST_STICKY_SESSIONS': '1'}, scheduler_cls=plain)

    # -- V1 ----------------------------------------------------------------------------------------------------------------------------------
    def test_a_conversation_splits_every_turn_exactly_and_matches_the_sticky_oracle_and_the_unchunked_run(self):
        import prefix_judge

        from vllm.v1.metrics.stats import PrefixCacheStats

        # vLLM replaces the manager's PrefixCacheStats object every time it collects the step's stats, so the count is taken on the class
        recorded, original_record = [], PrefixCacheStats.record

        def counting(this, *args, **kwargs):
            recorded.append(1)
            return original_record(this, *args, **kwargs)

        patcher = mock.patch.object(PrefixCacheStats, 'record', counting)
        patcher.start()
        self.addCleanup(patcher.stop)
        scheduler, state, lines, drive = self.build()
        oracle = prefix_judge.Oracle(sticky=True)
        self.decoder(drive)
        drive.run_until(lambda: 'd0' in drive.model.final)
        for _ in range(3):
            drive.step()
        prompt, prompts, hits = tokens(9000, 'chain-0'), [], []
        for turn in range(5):
            rid = 'turn%d' % turn
            request = drive.add(rid, prompt, 64)
            prompts.append(list(prompt))
            drive.run_until(drive.finished(rid))
            expected = oracle.admit(SALT, list(prompt))
            steps = drive.prefill_steps(rid)
            total = len(prompt)
            self.assertEqual(steps[0][0], expected['q'], 'turn %d resumes where the oracle says' % turn)
            self.assertEqual(steps, levern_policy.plan_from(total, expected['q'], decoding=True), 'turn %d steps' % turn)
            self.assertEqual(drive.model.sources[rid][0], 'CHECKPOINT' if expected['q'] else 'COLD')
            self.assertTrue(all(source == 'SCRATCH' for source in drive.model.sources[rid][1:]))
            grant = drive.grants[rid][0]
            self.assertEqual((grant.h, grant.q, grant.capture_positions()), (expected['h'], expected['q'], expected['plan']), 'turn %d' % turn)
            self.assertEqual(drive.model.final[rid], cold_state(list(prompt), total // CHUNK * CHUNK), 'turn %d ends in the cold chain' % turn)
            for position in expected['plan']:
                self.assertIsNotNone(state.registry.get(request.block_hashes[position // BLOCK - 1]), 'turn %d took its checkpoint at %d' % (turn, position))
            if len(steps) > 1:
                self.assertEqual(steps[-1][0], levern_policy.final_start(total))
            hits.append(expected['q'])
            prompt = list(prompt) + [100000 + turn] * 64 + tokens(2600, 'chain-%d' % (turn + 1))
        self.assertTrue(all(hit > 0 for hit in hits[1:]), hits)
        self.assertGreater(sum(1 for turn in range(1, 5) if len(drive.prefill_steps('turn%d' % turn)) > 1), 0, 'some hit arrived split')
        # the registry equals a run with Lever N off
        reference = reference_registry(self.env, prompts)
        self.assertEqual(registry_keys(state.registry), registry_keys(reference))
        # the peek recorded nothing: vLLM's statistics hold one attempt per in-pass trim, which the registry counts too
        self.assertEqual(len(recorded), state.registry.stats['attempts'])
        self.assertGreater(len(recorded), 5)
        self.assertEqual(drive.preempted, set())
        self.assertLessEqual(drive.over, 0)
        self.assertEqual(drive.unwritten, [])
        self.assertEqual(state.registry.stats['token_mismatches'], 0)
        self.assertEqual(state.registry.stats['commit_mismatch'], 0)
        self.assertGreater(state.registry.stats['inflight_captures'], 0, 'a checkpoint was taken from an in-flight plan')
        import serving_kv_reservation as kv

        scheduler.finish_requests('d0', RequestStatus.FINISHED_ABORTED)
        drive.step()
        self.assertEqual(scheduler.get_num_unfinished_requests(), 0)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), kv.pool_blocks(scheduler), 'every block is free again')

    # -- D1 ----------------------------------------------------------------------------------------------------------------------------------
    def conversation_to_s_last(self, drive):
        first = tokens(9000, 's-last')
        drive.add('first', first, 1)
        drive.run_until(drive.finished('first'))
        return list(first) + tokens(500, 's-last-more')

    def test_a_hit_at_s_last_is_one_final_step_with_the_peek(self):
        scheduler, state, lines, drive = self.build()
        self.decoder(drive)
        drive.run_until(lambda: 'd0' in drive.model.final)
        second = self.conversation_to_s_last(drive)
        self.assertEqual(levern_policy.final_start(len(second)), 6144)
        drive.add('second', second, 1)
        drive.run_until(drive.finished('second'))
        self.assertEqual(drive.prefill_steps('second'), [(6144, 9500)])
        self.assertEqual(drive.model.sources['second'], ['CHECKPOINT'])
        self.assertEqual(drive.model.final['second'], cold_state(second, 9500 // CHUNK * CHUNK))

    def test_the_cap_computed_from_zero_is_refused_as_a_step_past_s_last_the_negative_control(self):
        scheduler, state, lines, drive = self.build()
        self.decoder(drive)
        drive.run_until(lambda: 'd0' in drive.model.final)
        second = self.conversation_to_s_last(drive)
        state.peek = lambda request: 0          # the old cap: the budget of (P, 0); `state` is the graft bound to this scheduler
        drive.add('second', second, 1)
        with self.assertRaisesRegex(ValueError, 'past S_last=6144'):
            drive.run_until(drive.finished('second'))
        self.assertTrue(any('REFUSED' in line for line in lines), lines)

    # -- V2 ----------------------------------------------------------------------------------------------------------------------------------
    def test_a_hit_turn_arriving_during_a_cold_long_prefill_is_admitted_at_the_next_boundary_and_the_long_resumes(self):
        scheduler, state, lines, drive = self.build()
        self.decoder(drive)
        base = tokens(9000, 'agent')
        drive.add('agent-1', base, 1)
        drive.run_until(drive.finished('agent-1'))
        cold = tokens(40000, 'cold-long')
        drive.add('cold', cold, 1, salt='tenant-b')
        drive.run_until(lambda: len(drive.prefill_steps('cold')) >= 4)
        before = len(drive.prefill_steps('cold'))
        turn = list(base) + [100000] * 64 + tokens(3000, 'agent-2')
        drive.add('agent-2', turn, 1)
        drive.run_until(lambda: bool(drive.prefill_steps('agent-2')))
        self.assertEqual(len(drive.prefill_steps('cold')), before, 'the hit turn waited no cold step: it was admitted at the next prefill pass')
        self.assertEqual(levern_policy.state_holder().owner[0], 'agent-2')
        self.assertEqual(levern_policy.state_holder().parks, {'cold': drive.prefill_steps('cold')[-1][1]})
        drive.run_until(drive.finished('agent-2'))
        expected_hit = levern_policy.plan_from(len(turn), 6144, decoding=True)
        self.assertEqual(drive.prefill_steps('agent-2'), expected_hit)
        self.assertEqual(drive.model.sources['agent-2'], ['CHECKPOINT'] + ['SCRATCH'] * (len(expected_hit) - 1))
        drive.run_until(drive.finished('cold'))
        steps = drive.prefill_steps('cold')
        self.assertEqual(steps, levern_policy.plan(len(cold), decoding=True), 'the long prefill resumed exactly where it stopped: no step lost or replayed')
        self.assertEqual(drive.model.sources['cold'][before], 'PARKED')
        self.assertEqual(drive.model.sources['cold'].count('PARKED'), 1)
        self.assertEqual(drive.model.final['cold'], cold_state(cold, len(cold) // CHUNK * CHUNK))
        self.assertEqual(drive.model.final['agent-2'], cold_state(turn, len(turn) // CHUNK * CHUNK))
        self.assertEqual(drive.model.max_parks, 1)
        self.assertEqual(levern_policy.state_holder().parks, {})
        self.assertEqual(drive.preempted, set())
        self.assertLessEqual(drive.over, 0)
        self.assertEqual(drive.unwritten, [])
        self.assertEqual(scheduler.__dict__['_qwen_prefix'].registry.stats['commit_mismatch'], 0)

    def test_without_the_short_lane_the_same_hit_turn_waits_for_the_whole_cold_prefill(self):
        scheduler, state, lines, drive = self.build(park=False)
        self.decoder(drive)
        base = tokens(9000, 'agent')
        drive.add('agent-1', base, 1)
        drive.run_until(drive.finished('agent-1'))
        cold = tokens(40000, 'cold-long')
        drive.add('cold', cold, 1, salt='tenant-b')
        drive.run_until(lambda: len(drive.prefill_steps('cold')) >= 4)
        turn = list(base) + [100000] * 64 + tokens(3000, 'agent-2')
        drive.add('agent-2', turn, 1)
        drive.run_until(lambda: bool(drive.prefill_steps('agent-2')))
        self.assertEqual(drive.prefill_steps('cold')[-1][1], len(cold), 'v1: the hit turn started only after the cold prefill finished')

    # -- Q -----------------------------------------------------------------------------------------------------------------------------------
    def test_a_partial_whose_state_is_gone_is_quarantined_before_the_pass_and_the_engine_lives_on(self):
        scheduler, state, lines, drive = self.build()
        self.decoder(drive)
        drive.run_until(lambda: 'd0' in drive.model.final)
        cold = tokens(30000, 'lost')
        drive.add('lost', cold, 1, salt='tenant-b')
        drive.run_until(lambda: len(drive.prefill_steps('lost')) >= 3)
        scheduled_before = len(drive.prefill_steps('lost'))
        blocks_held = len(scheduler.kv_cache_manager.get_block_ids('lost')[0])
        drive.model.owner = drive.model.state = None            # the state the continuation needs is gone
        drive.model.publish()
        drive.run_until(drive.finished('lost'), limit=50)
        self.assertEqual(len(drive.prefill_steps('lost')), scheduled_before, 'no step was scheduled for it after its state was lost')
        self.assertEqual(scheduler.requests.get('lost'), None)
        self.assertTrue(any('quarantine' in line and 'lost' in line for line in lines), lines)
        self.assertGreater(blocks_held, 0)
        self.assertIn('d0', scheduler.requests, 'the decoder was not touched')
        for _ in range(3):
            drive.step()
        self.assertEqual(drive.records[-1][0], 'decode')
        self.assertEqual(drive.unwritten, [])
        self.assertEqual(drive.preempted, set())
        scheduler.finish_requests('d0', RequestStatus.FINISHED_ABORTED)
        drive.step()
        self.assertEqual(scheduler.get_num_unfinished_requests(), 0)
        pool = scheduler.kv_cache_manager.block_pool
        import serving_kv_reservation as kv

        self.assertEqual(pool.get_num_free_blocks(), kv.pool_blocks(scheduler), 'every block is free again')


if __name__ == '__main__':
    unittest.main()
