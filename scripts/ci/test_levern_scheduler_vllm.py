"""Lever N at TP4: the scheduler cap PROVED against the installed vLLM scheduler (0.25.1 in qwen-fast-vllm-cpu.yml; skipped where vLLM is not installed).

test_levern_scheduler holds the cap and the alternation on a reduced model of vLLM's Scheduler.schedule. This module runs the same wrappers over vLLM's real Scheduler, real
KVCacheManager and real BlockPool, with the speculative lookahead the TT plugin runs under (15 draft tokens: num_lookahead_tokens 16), chunked prefill ON and
max-num-batched-tokens equal to the window - the profile's engine argv - and proves what the reduced model can only assume:

  * the attribute the cap writes, Scheduler.max_num_scheduled_tokens, exists on 0.25.1 and is the budget schedule() starts from;
  * a cold prompt of 9,000 tokens is split by that cap alone into the steps levern_policy.plan names - [0, 2048), [2048, 4096), [4096, 6144) and the final step
    [6144, 9000) - each non-final end on a 2,048 boundary, the partial staying in `running` with is_prefill_chunk between its steps, one decode step serving every decoder
    between two prefill steps (QWEN_FAST_LEVERN_ROUNDS=1), the budget put back after every call, no preemption, every request finishing;
  * the prompt's seed is sampled only at the last step (the intermediate rows carry no token);
  * with the flag off the same prompt is one step, as it is today.
Run with VLLM_USE_V2_MODEL_RUNNER=0 (as the other installed-vLLM suites)."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import main, skipUnless
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

import levern_policy  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402
import test_serving_kv_reservation_vllm as reservation  # noqa: E402

WINDOW = 32768
BLOCKS = 800
LEVERN = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1', 'QWEN_FAST_KV_RESERVATION': '1'}
OFF = {'QWEN_FAST_LEVER_N': '0', 'QWEN_FAST_KV_RESERVATION': '1'}
FLAGS = levern_policy.ALL_FLAGS + (admission.STEPS_FLAG, 'QWEN_FAST_KV_RESERVATION')


@skipUnless(HAVE_VLLM, 'vLLM is not installed')
class LevernVllmCase(reservation.VllmCase):
    def build_chunked(self, scheduler_type, chunked=True, blocks=BLOCKS, seats=8, window=WINDOW):
        from transformers import GPT2Config
        from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig,
                                 VllmConfig)
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import init_none_hash
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
        from vllm.v1.structured_output import StructuredOutputManager

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        GPT2Config(n_positions=max(8192, window), n_embd=256, n_layer=1, n_head=4).save_pretrained(temporary.name)
        model = ModelConfig(model=temporary.name, dtype='float32', max_model_len=window, skip_tokenizer_init=True, seed=0)
        speculative = SpeculativeConfig(model='ngram', num_speculative_tokens=15)
        speculative.method = 'dflash'
        config = VllmConfig(
            model_config=model, device_config=DeviceConfig(device='cpu'),
            scheduler_config=SchedulerConfig(max_num_seqs=seats, max_num_batched_tokens=window, max_model_len=window,
                                             is_encoder_decoder=False, enable_chunked_prefill=chunked, async_scheduling=False,
                                             watermark=0.0),
            cache_config=CacheConfig(block_size=64, enable_prefix_caching=False), parallel_config=ParallelConfig(),
            speculative_config=speculative)
        config.cache_config.num_gpu_blocks = blocks
        cache = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(['layer'], FullAttentionSpec(block_size=64, num_kv_heads=2, head_size=256, dtype=torch.bfloat16))])
        register_all_kvcache_specs(config)
        init_none_hash(get_hash_fn_by_name(config.cache_config.prefix_caching_hash_algo))
        saved = {name: os.environ.pop(name, None) for name in reservation.HOOK_SWITCHES}
        try:
            scheduler = scheduler_type(config, cache, StructuredOutputManager(config), block_size=64)
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value
        scheduler.use_v2_model_runner = False
        scheduler.max_num_running_reqs = seats
        return scheduler

    def installed(self, scheduler_type, environ):
        log = Mock()
        with patch.dict(os.environ, environ):
            for name in FLAGS:
                if name not in environ:
                    os.environ.pop(name, None)
            admission.install(reservation.configured(scheduler_type), log=log)
        return log

    def step_records(self, scheduler, cold, decoders, limit=400):
        """Run the engine loop until `cold` finishes its prompt and every decoder has a token; one record per step: (kind, request, tokens, start, budget seen)."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        records, seeded = [], set()
        for name in decoders:
            scheduler.add_request(Request(name, [42] * 100, SamplingParams(temperature=0, max_tokens=4000), None))
        for _ in range(limit):
            if all(name in seeded for name in decoders):
                break
            self.advance(scheduler, seeded, records)
        records.clear()
        scheduler.add_request(Request(cold[0], [7] * cold[1], SamplingParams(temperature=0, max_tokens=20), None))
        for _ in range(limit):
            self.advance(scheduler, seeded, records)
            if cold[0] in seeded:
                break
        return records

    def advance(self, scheduler, seeded, records):
        before = {name: scheduler.requests[name].num_computed_tokens for name in scheduler.requests}
        scheduled = scheduler.schedule()
        budget_after = getattr(scheduler, 'max_num_scheduled_tokens', None)
        if not scheduled.num_scheduled_tokens:
            return
        tokens, sampled = {}, {}
        for name, count in scheduled.num_scheduled_tokens.items():
            request = scheduler.requests[name]
            tokens[name] = count
            prompt_done = before.get(name, 0) + count >= request.num_prompt_tokens
            if not prompt_done:
                sampled[name] = []                       # an intermediate prefill row carries no token
            elif name not in seeded:
                sampled[name] = [100]
                seeded.add(name)
            else:
                sampled[name] = [200]
        order = list(scheduled.num_scheduled_tokens)
        prefill = [name for name in order if name in {entry.req_id for entry in scheduled.scheduled_new_reqs}
                   or (name in before and scheduler.requests[name].num_prompt_tokens > before[name] and tokens[name] > 1)]
        records.append(dict(kind='prefill' if prefill else 'decode', request=prefill[0] if prefill else None, tokens=sum(tokens.values()),
                            start=before.get(prefill[0], 0) if prefill else None, seats=len(order), budget_after=budget_after,
                            preempted=self.preemptions(scheduler, scheduled)))
        scheduler.update_from_output(scheduled, self.output(order, sampled))


@skipUnless(HAVE_VLLM, 'vLLM is not installed')
class CapOnTheRealSchedulerTests(LevernVllmCase):
    def test_the_budget_attribute_exists_and_is_the_window(self):
        for label, scheduler_type in self.scheduler_classes():
            with self.subTest(label):
                scheduler = self.build_chunked(scheduler_type)
                self.assertIsInstance(getattr(scheduler, 'max_num_scheduled_tokens', None), int)
                self.assertEqual(scheduler.max_num_scheduled_tokens, WINDOW)

    def test_a_cold_prompt_is_split_at_chunk_boundaries_between_decode_rounds(self):
        plan = levern_policy.plan(9000)
        self.assertEqual(plan, [(0, 2048), (2048, 4096), (4096, 6144), (6144, 9000)])
        for label, scheduler_type in self.scheduler_classes():
            with self.subTest(label):
                self.installed(scheduler_type, LEVERN)
                scheduler = self.build_chunked(scheduler_type)
                records = self.step_records(scheduler, ('cold', 9000), ['d0', 'd1', 'd2'])
                prefill = [(row['start'], row['start'] + row['tokens']) for row in records if row['kind'] == 'prefill']
                self.assertEqual(prefill, plan)
                self.assertEqual([row['kind'] for row in records], ['prefill', 'decode', 'prefill', 'decode', 'prefill', 'decode', 'prefill'])
                for row in records:
                    if row['kind'] == 'decode':
                        self.assertEqual(row['seats'], 3, 'a decode step serves every decoder and not the partial prefill')
                    self.assertEqual(row['budget_after'], WINDOW, 'the token budget is put back after every call')
                    self.assertEqual(row['preempted'], set())
                self.assertTrue(all(end % 2048 == 0 for _, end in prefill[:-1]))

    def test_the_prefill_alone_runs_in_solo_steps(self):
        for label, scheduler_type in self.scheduler_classes():
            with self.subTest(label):
                self.installed(scheduler_type, dict(LEVERN, QWEN_FAST_LEVERN_ROUNDS='2'))
                scheduler = self.build_chunked(scheduler_type)
                records = self.step_records(scheduler, ('cold', 40000), [])
                prefill = [(row['start'], row['start'] + row['tokens']) for row in records if row['kind'] == 'prefill']
                self.assertEqual(prefill, levern_policy.plan(40000, decoding=False))
                self.assertEqual({row['kind'] for row in records}, {'prefill'})

    def test_with_the_flag_off_the_same_prompt_is_one_step(self):
        for label, scheduler_type in self.scheduler_classes():
            with self.subTest(label):
                self.installed(scheduler_type, OFF)
                scheduler = self.build_chunked(scheduler_type, chunked=False)
                records = self.step_records(scheduler, ('cold', 9000), ['d0', 'd1', 'd2'])
                prefill = [row for row in records if row['kind'] == 'prefill']
                self.assertEqual([(row['start'], row['tokens']) for row in prefill], [(0, 9000)])

    def test_nothing_is_preempted_and_every_request_finishes_under_the_reservation(self):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        for label, scheduler_type in self.scheduler_classes():
            with self.subTest(label):
                self.installed(scheduler_type, LEVERN)
                scheduler = self.build_chunked(scheduler_type, blocks=420)
                arrivals = [(0, 'd%d' % index, 300, 60) for index in range(4)] + [(3, 'cold-a', 9000, 8), (4, 'cold-b', 6500, 8), (9, 'late', 700, 12)]
                pending, seeded, records, over = sorted(arrivals), set(), [], 0
                for step in range(4000):
                    while pending and pending[0][0] <= step:
                        _, name, prompt, answer = pending.pop(0)
                        scheduler.add_request(Request(name, [42] * prompt, SamplingParams(temperature=0, max_tokens=answer), None))
                    self.advance(scheduler, seeded, records)
                    for request in scheduler.running:
                        held = len(scheduler.kv_cache_manager.get_block_ids(request.request_id)[0])
                        over = max(over, held - reservation.kv.request_reservation(request, scheduler.max_model_len))
                    if not pending and not scheduler.requests:
                        break
                self.assertEqual(scheduler.requests, {}, 'every request finished')
                self.assertEqual(set().union(*[row['preempted'] for row in records]), set())
                self.assertLessEqual(over, 0, 'no request held more blocks than it reserved')
                self.assertTrue(any(row['kind'] == 'decode' for row in records) and any(row['kind'] == 'prefill' for row in records))


if __name__ == '__main__':
    main()
