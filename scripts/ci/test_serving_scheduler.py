from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from transformers import GPT2Config
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from serving_page_binding import validate_initial_capture_pages
from serving_vllm_contract import admit_scheduler_output


class RealSchedulerTests(unittest.TestCase):
    scheduler_type = Scheduler

    def scheduler(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        GPT2Config(n_positions=8192, n_embd=256, n_layer=1, n_head=4).save_pretrained(directory)
        model = ModelConfig(model=directory, dtype='float32', max_model_len=4352,
            skip_tokenizer_init=True, seed=0)
        speculative = SpeculativeConfig(model='ngram', num_speculative_tokens=15)
        speculative.method = 'dflash'
        config = VllmConfig(model_config=model, device_config=DeviceConfig(device='cpu'),
            scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4352,
                max_model_len=4352, is_encoder_decoder=False, enable_chunked_prefill=False,
                async_scheduling=False, watermark=0.0),
            cache_config=CacheConfig(block_size=64, enable_prefix_caching=False),
            parallel_config=ParallelConfig(), speculative_config=speculative)
        config.cache_config.num_gpu_blocks = 128
        cache = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(['layer'], FullAttentionSpec(block_size=64,
                num_kv_heads=2, head_size=256, dtype=torch.bfloat16))])
        register_all_kvcache_specs(config)
        scheduler = self.scheduler_type(config, cache, StructuredOutputManager(config), block_size=64)
        scheduler.use_v2_model_runner = False
        self.assertEqual(scheduler.num_lookahead_tokens, 16)
        return scheduler

    def output(self, request_id, tokens):
        return ModelRunnerOutput(req_ids=[request_id], req_id_to_index={request_id: 0},
            sampled_token_ids=[tokens], logprobs=None, prompt_logprobs_dict={}, pooler_output=[])

    def test_actual_prefill_pages_and_decode_rejection_accounting(self):
        scheduler = self.scheduler()
        parameters = SamplingParams(temperature=0, max_tokens=256)
        request = Request('request', [42] * 4096, parameters, None)
        scheduler.add_request(request)
        scheduled = scheduler.schedule()
        self.assertEqual(scheduled.num_scheduled_tokens, {'request': 4096})
        blocks = tuple(scheduled.scheduled_new_reqs[0].block_ids[0])
        self.assertGreaterEqual(len(blocks), 65)
        pages = torch.full((1, 68), blocks[0], dtype=torch.int32)
        pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        validate_initial_capture_pages(pages, blocks, position=4096, output_budget=256)
        scheduler.update_from_output(scheduled, self.output('request', [100]))
        frontier, emitted = 4096, 1
        acceptance = iter((1, 16, 9, 1, 16, 16, 4))
        while emitted < 256:
            rows = max(width for width in (1, 2, 4, 8, 16) if width <= 256 - emitted)
            committed = min(next(acceptance, rows), rows)
            proposals = list(range(101, 100 + rows))
            scheduler.update_draft_token_ids(DraftTokenIds(['request'], [proposals]))
            scheduled = scheduler.schedule()
            ticket = SimpleNamespace(request_id='request', position=frontier, tokens=(100, *proposals))
            prepared = SimpleNamespace(closed=False, cancelled=False, busy=False,
                engine=SimpleNamespace(phase='idle'), session=SimpleNamespace(
                    phase='pending', pending=ticket, request_id='request', position=frontier))
            self.assertIs(admit_scheduler_output(prepared, scheduled), ticket)
            current_blocks = tuple(scheduler.kv_cache_manager.get_block_ids('request')[0])
            self.assertEqual(current_blocks[:len(blocks)], blocks)
            self.assertGreaterEqual(len(current_blocks) * 64, frontier + rows)
            self.assertLessEqual(len(current_blocks), 68)
            blocks = current_blocks
            scheduler.update_from_output(scheduled, self.output('request', list(range(200, 200 + committed))))
            frontier += committed
            emitted += committed
            self.assertEqual(request.num_computed_tokens, frontier)
            self.assertEqual(len(request.output_token_ids), emitted)
        self.assertEqual(request.status, RequestStatus.FINISHED_LENGTH_CAPPED)
        self.assertNotIn('request', scheduler.requests)
        finished = scheduler.schedule()
        self.assertEqual(finished.finished_req_ids, {'request'})
        self.assertEqual(finished.total_num_scheduled_tokens, 0)
        parameters = SamplingParams(temperature=0, max_tokens=256)
        replacement = Request('replacement', [43] * 4096, parameters, None)
        scheduler.add_request(replacement)
        scheduled = scheduler.schedule()
        self.assertEqual(scheduled.num_scheduled_tokens, {'replacement': 4096})
        self.assertGreaterEqual(len(scheduled.scheduled_new_reqs[0].block_ids[0]), 65)
        scheduler.update_from_output(scheduled, self.output('replacement', [100]))
        scheduler.finish_requests('replacement', RequestStatus.FINISHED_ABORTED)
        self.assertNotIn('replacement', scheduler.requests)
        self.assertEqual(scheduler.schedule().finished_req_ids, {'replacement'})


class TTPluginSchedulerTests(RealSchedulerTests):
    def scheduler(self):
        from vllm_tt_plugin.scheduler import TTScheduler

        self.scheduler_type = TTScheduler
        return super().scheduler()
