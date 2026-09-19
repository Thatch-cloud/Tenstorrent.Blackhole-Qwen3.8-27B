"""T5b: can two prefills be delivered to `_execute` one at a time?

T5a established that two concurrent prefill captures are structurally impossible:
`PrefillWindowCapture.capture()` installs itself as an attribute on the MODEL and
keeps a single `cursor` asserting absolute chunk order for one sequence. So
holding N captures is out.

But the capture is only live *during* prefill - `_sample` drops it once the bridge
is built - so if the scheduler hands `_execute` one new request per step, the
single-capture constraint is never violated and batching becomes a scheduling
question rather than capture surgery.

This drives the REAL vLLM scheduler, the same way test_serving_scheduler does, and
asks three things:

  1. with `max_num_seqs=2` and two queued requests, does one `schedule()` return
     both in `scheduled_new_reqs`, or one?
  2. if both, does anything bound the count - is there a knob that makes prefills
     arrive singly?
  3. once request A is decoding, does request B arrive as a fresh prefill in a step
     whose `scheduled_cached_reqs` is non-empty? `_execute` refuses that today, and
     it is the shape serialised prefills would actually take.

Answering on CPU in seconds, because the fast local suite makes that possible.
"""

import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import GPT2Config
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


def build(directory, max_num_seqs):
    GPT2Config(n_positions=8192, n_embd=256, n_layer=1, n_head=4).save_pretrained(directory)
    model = ModelConfig(model=directory, dtype='float32', max_model_len=4352,
                        skip_tokenizer_init=True, seed=0)
    speculative = SpeculativeConfig(model='ngram', num_speculative_tokens=15)
    speculative.method = 'dflash'
    config = VllmConfig(
        model_config=model, device_config=DeviceConfig(device='cpu'),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=8704,
                                         max_model_len=4352, is_encoder_decoder=False,
                                         enable_chunked_prefill=False, async_scheduling=False,
                                         watermark=0.0),
        cache_config=CacheConfig(block_size=64, enable_prefix_caching=False),
        parallel_config=ParallelConfig(), speculative_config=speculative)
    config.cache_config.num_gpu_blocks = 512
    cache = KVCacheConfig(num_blocks=512, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(['layer'], FullAttentionSpec(block_size=64, num_kv_heads=2,
                                                      head_size=256, dtype=torch.bfloat16))])
    register_all_kvcache_specs(config)
    scheduler = Scheduler(config, cache, StructuredOutputManager(config), block_size=64)
    scheduler.use_v2_model_runner = False
    return scheduler


def output(req_ids, token):
    return ModelRunnerOutput(req_ids=list(req_ids),
                             req_id_to_index={r: i for i, r in enumerate(req_ids)},
                             sampled_token_ids=[[token] for _ in req_ids],
                             logprobs=None, prompt_logprobs_dict={}, pooler_output=[])


def describe(scheduled):
    return dict(new=[r.req_id for r in scheduled.scheduled_new_reqs],
                cached=list(getattr(scheduled.scheduled_cached_reqs, 'req_ids', []) or []),
                total_tokens=scheduled.total_num_scheduled_tokens)


def main():
    parameters = SamplingParams(temperature=0, max_tokens=256)
    findings = {}

    # 1 and 2: both queued at once, at max_num_seqs 2 and 1
    for seqs in (2, 1):
        with TemporaryDirectory() as directory:
            scheduler = build(directory, seqs)
            for name in ('A', 'B'):
                scheduler.add_request(Request(name, [42] * 3000, parameters, None))
            first = describe(scheduler.schedule())
            findings['both_queued_max_num_seqs_%d' % seqs] = first
            print('max_num_seqs=%d, two queued -> new=%s cached=%s tokens=%d'
                  % (seqs, first['new'], first['cached'], first['total_tokens']))

    # 3: A already decoding, B arrives afterwards
    with TemporaryDirectory() as directory:
        scheduler = build(directory, 2)
        scheduler.add_request(Request('A', [42] * 3000, parameters, None))
        first = scheduler.schedule()
        step1 = describe(first)
        scheduler.update_from_output(first, output(['A'], 100))
        scheduler.add_request(Request('B', [42] * 3000, parameters, None))
        second = scheduler.schedule()
        step2 = describe(second)
        findings['a_decoding_then_b'] = dict(step1=step1, step2=step2)
        print('A prefill  -> new=%s cached=%s' % (step1['new'], step1['cached']))
        print('B arrives  -> new=%s cached=%s' % (step2['new'], step2['cached']))

    print()
    both = findings['both_queued_max_num_seqs_2']['new']
    print('VERDICT')
    if len(both) > 1:
        print('  one schedule() returns BOTH prefills, so _execute sees two new reqs')
        print('  and serialising needs the scheduler constrained, not just the lifecycle')
    else:
        print('  prefills already arrive one per step: serialisation is free')
    mixed = findings['a_decoding_then_b']['step2']
    if mixed['new'] and mixed['cached']:
        print('  a later prefill arrives ALONGSIDE a decoding request: _execute refuses')
        print('  that shape today (scheduled_cached_reqs non-empty), so it must change')
    return 0


if __name__ == '__main__':
    sys.exit(main())
