"""The KV reservation rule PROVED against the installed vLLM scheduler (0.25.1 in qwen-fast-vllm-cpu.yml; skipped where vLLM is not installed).

serving_kv_reservation says a request that will hold at most r = ceil((prompt + max_tokens + 32) / 64) + 1 blocks may be admitted
only when the running requests' reservations plus r fit the pool, and that under that rule vLLM's own block manager never needs to
preempt (the fast path cannot survive one: the preempted request's engine, snapshot and traces are live on the cards). This module is
that proof, on vLLM's real Scheduler, real KVCacheManager and real BlockPool, with the speculative lookahead the TT plugin runs under
(15 draft tokens: num_lookahead_tokens 16):

  * RandomTrafficTests: seeded random arrivals, prompt lengths, output budgets and acceptances of 1 to 16 tokens per round, on a pool
    smaller than the traffic's demand. Under the wrapper (QWEN_FAST_KV_RESERVATION=1) every step's SchedulerOutput.preempted_req_ids is
    empty, no request is ever in the PREEMPTED status or counts a preemption, no request ever holds more blocks than it reserved,
    every request finishes, and the hold is exercised (the traffic does outrun the pool).
  * NegativeControlTests: the same checks FAIL where they should - the rule off, and the rule with its margin removed (lookahead 0,
    spare 0, a pool packed to the margin-free sums) preempt on a deterministic three-request scenario. A proof whose negative control
    does not preempt proves nothing.
  * NullBlockTests: whether vLLM spends one block of the override on its null block - the premise of pool_blocks (the count less 1)
    and of the 131k profiles' arithmetic (8 x 2,052 = 16,416 is one block short of every seat at full length if it does).

Run with VLLM_USE_V2_MODEL_RUNNER=0 (as the other installed-vLLM suites); built like test_serving_scheduler.RealSchedulerTests."""

import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import torch
    import vllm  # noqa: F401
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False

import serving_kv_reservation as kv  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402

HOOK_SWITCHES = ('QWEN_PREFIX_REUSE', 'QWEN_FAST_STICKY_SESSIONS')
WINDOW = 6144
MAX_STEPS = 6000
CONTROL_PROMPT, CONTROL_ANSWER = 1024, 508        # 1,532 tokens: 23 blocks and 60 tokens, so a 16-token lookahead crosses a boundary


def configured(cls):
    from types import SimpleNamespace

    return SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=cls))


@skipUnless(HAVE_VLLM, 'vLLM is not installed')
class VllmCase(TestCase):
    """A real vLLM Scheduler with `blocks` KV blocks of 64 tokens and `seats` running slots."""

    def scheduler_classes(self):
        import test_serving_request_quarantine

        classes = [('fixture', test_serving_request_quarantine.plugin_scheduler_class())]
        try:
            from vllm_tt_plugin.scheduler import TTScheduler
        except ImportError:
            pass
        else:
            classes.append(('installed plugin', type('TTScheduler', (TTScheduler,), {})))
        return classes

    def build(self, scheduler_type, blocks, seats=8, window=WINDOW):
        from transformers import GPT2Config
        from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig,
                                 VllmConfig)
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import init_none_hash
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa: F401
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
        from vllm.v1.structured_output import StructuredOutputManager

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        GPT2Config(n_positions=8192, n_embd=256, n_layer=1, n_head=4).save_pretrained(temporary.name)
        model = ModelConfig(model=temporary.name, dtype='float32', max_model_len=window, skip_tokenizer_init=True, seed=0)
        speculative = SpeculativeConfig(model='ngram', num_speculative_tokens=15)
        speculative.method = 'dflash'
        config = VllmConfig(
            model_config=model, device_config=DeviceConfig(device='cpu'),
            scheduler_config=SchedulerConfig(max_num_seqs=seats, max_num_batched_tokens=window, max_model_len=window,
                                             is_encoder_decoder=False, enable_chunked_prefill=False, async_scheduling=False,
                                             watermark=0.0),
            cache_config=CacheConfig(block_size=64, enable_prefix_caching=False), parallel_config=ParallelConfig(),
            speculative_config=speculative)
        config.cache_config.num_gpu_blocks = blocks
        cache = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(['layer'], FullAttentionSpec(block_size=64, num_kv_heads=2, head_size=256, dtype=torch.bfloat16))])
        register_all_kvcache_specs(config)
        init_none_hash(get_hash_fn_by_name(config.cache_config.prefix_caching_hash_algo))
        saved = {name: os.environ.pop(name, None) for name in HOOK_SWITCHES}
        try:
            scheduler = scheduler_type(config, cache, StructuredOutputManager(config), block_size=64)
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value
        scheduler.use_v2_model_runner = False
        self.assertEqual(scheduler.num_lookahead_tokens, 16)
        scheduler.max_num_running_reqs = seats
        return scheduler

    def install(self, scheduler_type, on=True):
        log = Mock()
        with patch.dict(os.environ, {kv.FLAG: '1' if on else '0'}):
            admission.install(configured(scheduler_type), log=log)
        return log

    @staticmethod
    def output(ids, tokens):
        from vllm.v1.outputs import ModelRunnerOutput

        return ModelRunnerOutput(req_ids=list(ids), req_id_to_index={name: index for index, name in enumerate(ids)},
                                 sampled_token_ids=[tokens[name] for name in ids], logprobs=None, prompt_logprobs_dict={},
                                 pooler_output=[])

    @staticmethod
    def preemptions(scheduler, scheduled):
        from vllm.v1.request import RequestStatus

        found = set(getattr(scheduled, 'preempted_req_ids', None) or ())
        for request in scheduler.requests.values():
            if getattr(request, 'num_preemptions', 0) or request.status == RequestStatus.PREEMPTED:
                found.add(request.request_id)
        return found

    def drive(self, scheduler, arrivals, rng, check_reservation=True, stop_on_preemption=False):
        """Run `arrivals` ([(step, id, prompt, max_tokens)]) to completion. Returns (finished ids, preempted ids seen, the most
        blocks any request held beyond its reservation, holds seen by the wrapper is the caller's to read)."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import SchedulerOutput  # noqa: F401
        from vllm.v1.outputs import DraftTokenIds
        from vllm.v1.request import Request

        pending = sorted(arrivals)
        finished, preempted, over = set(), set(), 0
        for step in range(MAX_STEPS):
            while pending and pending[0][0] <= step:
                _, name, prompt, max_tokens = pending.pop(0)
                scheduler.add_request(Request(name, [42] * prompt, SamplingParams(temperature=0, max_tokens=max_tokens), None))
            ids, proposals = [], []
            for request in scheduler.running:
                remaining = request.max_tokens - len(request.output_token_ids)
                if request.num_output_tokens >= 1 and remaining > 1:
                    count = rng.randint(1, min(15, remaining - 1))
                    ids.append(request.request_id)
                    proposals.append(list(range(101, 101 + count)))
            if ids:
                scheduler.update_draft_token_ids(DraftTokenIds(ids, proposals))
            scheduled = scheduler.schedule()
            seen = self.preemptions(scheduler, scheduled)
            if seen:
                preempted |= seen
                if stop_on_preemption:
                    return finished, preempted, over
            if check_reservation:
                for request in scheduler.running:
                    held = len(scheduler.kv_cache_manager.get_block_ids(request.request_id)[0])
                    over = max(over, held - kv.request_reservation(request, scheduler.max_model_len))
            if not scheduled.num_scheduled_tokens:
                if not pending and not scheduler.running and not scheduler.waiting and not getattr(
                        scheduler, 'skipped_waiting', None):
                    break
                continue
            new = {entry.req_id for entry in scheduled.scheduled_new_reqs}
            tokens = {}
            for name, scheduled_tokens in scheduled.num_scheduled_tokens.items():
                request = scheduler.requests[name]
                spec = (scheduled.scheduled_spec_decode_tokens or {}).get(name) or []
                if name in new or request.num_computed_tokens - scheduled_tokens < request.num_prompt_tokens:
                    tokens[name] = [100]
                else:
                    committed = rng.randint(1, len(spec) + 1)
                    committed = max(1, min(committed, request.max_tokens - len(request.output_token_ids)))
                    tokens[name] = list(range(200, 200 + committed))
            order = list(scheduled.num_scheduled_tokens)
            scheduler.update_from_output(scheduled, self.output(order, tokens))
            finished |= {name for name in order if name not in scheduler.requests}
        return finished, preempted, over


def arrivals_for(seed, count, pool_blocks):
    """Seeded traffic whose total demand outruns the pool: bursts of arrivals, prompts from tiny to long, budgets from 2 to 1,200."""
    rng = random.Random(seed)
    out, step = [], 0
    longest = (pool_blocks - 2) * 64 - 64 - kv.LOOKAHEAD_TOKENS - 2
    for index in range(count):
        step += rng.choice((0, 0, 0, 1, 2, 5))
        prompt = rng.choice((rng.randint(1, 200), rng.randint(200, 1500), rng.randint(1500, 3500)))
        prompt = max(1, min(prompt, longest - 2))
        answer = max(2, min(rng.choice((rng.randint(2, 40), rng.randint(40, 400), rng.randint(400, 1200))), longest - prompt))
        out.append((step, 'r%d-%d' % (seed, index), prompt, answer))
    return out


class RandomTrafficTests(VllmCase):
    SEEDS = tuple(range(24))
    POOL = 260                      # blocks: 8 seats of mid-size requests cannot all fit

    def test_under_the_rule_vllm_never_preempts_and_no_request_outgrows_its_reservation(self):
        for source, scheduler_type in self.scheduler_classes():
            log = self.install(scheduler_type)
            for seed in self.SEEDS:
                with self.subTest(source=source, seed=seed):
                    scheduler = self.build(scheduler_type, self.POOL)
                    arrivals = arrivals_for(seed, 14, scheduler.kv_cache_manager.block_pool.num_gpu_blocks)
                    for _, _, prompt, answer in arrivals:
                        self.assertLessEqual(kv.request_blocks(prompt, answer), kv.pool_blocks(scheduler))
                    finished, preempted, over = self.drive(scheduler, arrivals, random.Random(1000 + seed))
                    self.assertEqual(preempted, set(), 'vLLM preempted: the reservation rule is not enough')
                    self.assertLessEqual(over, 0, 'a request held more blocks than it reserved')
                    self.assertEqual(finished, {entry[1] for entry in arrivals}, 'every request finishes (progress)')
                    self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), kv.pool_blocks(scheduler),
                                     'every block is back when the traffic has drained')
            holds = len([entry for entry in log.call_args_list if str(entry.args[0]).startswith(kv.HOLD_PREFIX)])
            with self.subTest(source=source):
                self.assertGreater(holds, 0, 'the traffic never outran the pool: the proof exercised no hold')

    def test_the_accepted_token_counts_cover_one_to_sixteen_and_the_finals_hit_every_rounding(self):
        # prompt + max_tokens swept over a whole block's residues, so the last round's lookahead lands on every offset
        for source, scheduler_type in self.scheduler_classes():
            self.install(scheduler_type)
            for residue in range(0, 64, 3):
                with self.subTest(source=source, residue=residue):
                    scheduler = self.build(scheduler_type, 90, seats=3)
                    base = 640 + residue
                    arrivals = [(0, 'a%d-%d' % (residue, index), base + index, 400) for index in range(5)]
                    finished, preempted, over = self.drive(scheduler, arrivals, random.Random(residue))
                    self.assertEqual((preempted, finished), (set(), {entry[1] for entry in arrivals}))
                    self.assertLessEqual(over, 0)


class NegativeControlTests(VllmCase):
    """Three requests whose prompt + answer is 1,532 tokens (23 blocks and 60 tokens): each reserves 26 blocks under the rule and
    24 without its margin, and each reaches 25 blocks at its last round (the lookahead crosses the block boundary)."""

    PROMPT, ANSWER = CONTROL_PROMPT, CONTROL_ANSWER
    ARRIVALS = [(0, 'n%d' % index, CONTROL_PROMPT, CONTROL_ANSWER) for index in range(3)]

    def test_the_real_rule_admits_two_holds_the_third_and_never_preempts(self):
        self.assertEqual(kv.request_blocks(self.PROMPT, self.ANSWER), 26)
        for source, scheduler_type in self.scheduler_classes():
            with self.subTest(source=source):
                log = self.install(scheduler_type)
                scheduler = self.build(scheduler_type, 73, seats=8)       # 72 usable: 2 x 26 fits, 3 x 26 = 78 does not
                finished, preempted, over = self.drive(scheduler, self.ARRIVALS, random.Random(5))
                self.assertEqual((preempted, finished), (set(), {'n0', 'n1', 'n2'}))
                self.assertLessEqual(over, 0)
                self.assertTrue([entry for entry in log.call_args_list if str(entry.args[0]).startswith(kv.HOLD_PREFIX)])

    def test_with_the_rule_off_the_same_traffic_is_preempted(self):
        for source, scheduler_type in self.scheduler_classes():
            with self.subTest(source=source):
                self.install(scheduler_type, on=False)
                scheduler = self.build(scheduler_type, 73, seats=8)
                finished, preempted, over = self.drive(scheduler, self.ARRIVALS, random.Random(5), stop_on_preemption=True)
                self.assertTrue(preempted, 'the negative control did not preempt: the proof proves nothing')

    def test_with_the_margin_removed_the_same_traffic_is_preempted(self):
        for source, scheduler_type in self.scheduler_classes():
            with self.subTest(source=source), patch.object(kv, 'LOOKAHEAD_TOKENS', 0), patch.object(kv, 'SPARE_BLOCKS', 0):
                self.assertEqual(kv.request_blocks(self.PROMPT, self.ANSWER), 24)
                self.install(scheduler_type)
                scheduler = self.build(scheduler_type, 73, seats=8)       # 72 usable: 3 x 24 fits exactly with no margin
                finished, preempted, over = self.drive(scheduler, self.ARRIVALS, random.Random(5), stop_on_preemption=True)
                self.assertTrue(preempted, 'a rule without its margin did not preempt: the margin is untested')


class NullBlockTests(VllmCase):
    def test_vllm_spends_one_block_of_the_override_on_its_null_block(self):
        for source, scheduler_type in self.scheduler_classes():
            with self.subTest(source=source):
                scheduler = self.build(scheduler_type, 128, seats=1)
                pool = scheduler.kv_cache_manager.block_pool
                self.assertEqual(pool.num_gpu_blocks, 128)
                self.assertEqual(pool.get_num_free_blocks(), 128 - kv.NULL_BLOCKS)
                self.assertEqual(kv.pool_blocks(scheduler), 127)

    def test_the_install_check_passes_on_the_installed_vllm(self):
        self.assertEqual(kv.install_check(), [])


if __name__ == '__main__':
    main()
