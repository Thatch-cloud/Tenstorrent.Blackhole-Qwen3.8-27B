"""T5b-ii: what must admit_scheduler_output accept for a prefill during decode?

The decode-side contract the hook runs under refuses any new request:

    if (scheduled.scheduled_new_reqs or scheduled.finished_req_ids
            or list(cached.req_ids) != [request_id] ...):
        raise ValueError('Single resident decode request at the exact scheduled frontier required')

Relaxing it is only sensible if the step it would then admit is one the device can
actually execute. The earlier probe saw `new=['B'] cached=['A']` with A merely
resident. The real case is A in SPECULATIVE decode, holding a verifier block of
proposals, when B arrives with a full prompt.

So this reports the exact SchedulerOutput for that case:

  - does one step carry BOTH A's verifier block and B's whole prefill?
  - what does num_scheduled_tokens hold, and does the total match their sum?
  - are A's proposals still in scheduled_spec_decode_tokens alongside B's prefill?
  - is B's prefill delivered whole, or chunked?

If one step carries both, the contract cannot simply be widened: `execute_scheduled`
runs `request.step()` for ONE ticket, and a prefill and a verify are different
shapes on the device. The step would have to be split or B deferred. If instead the
scheduler holds B back while A has proposals in flight, relaxing the contract is
enough and the change is small.

CPU only: no device, no weights.
"""

import sys
from tempfile import TemporaryDirectory

import torch
from transformers import GPT2Config
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
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


def show(label, scheduled):
    cached = scheduled.scheduled_cached_reqs
    new = [r.req_id for r in scheduled.scheduled_new_reqs]
    print('%-26s new=%s cached=%s counts=%s total=%s spec=%s'
          % (label, new, list(getattr(cached, 'req_ids', []) or []),
             dict(scheduled.num_scheduled_tokens), scheduled.total_num_scheduled_tokens,
             dict(scheduled.scheduled_spec_decode_tokens)))
    return new, list(getattr(cached, 'req_ids', []) or [])


def main():
    parameters = SamplingParams(temperature=0, max_tokens=256)
    with TemporaryDirectory() as directory:
        scheduler = build(directory, 2)
        scheduler.add_request(Request('A', [42] * 3000, parameters, None))
        first = scheduler.schedule()
        show('A prefill', first)
        scheduler.update_from_output(first, output(['A'], 100))

        # A now holds proposals, which is what the verifier block is made of
        proposals = list(range(101, 116))
        scheduler.update_draft_token_ids(DraftTokenIds(['A'], [proposals]))
        decode = scheduler.schedule()
        show('A speculative decode', decode)
        scheduler.update_from_output(decode, output(['A'], 200))

        # B arrives while A is mid-speculation
        scheduler.update_draft_token_ids(DraftTokenIds(['A'], [proposals]))
        scheduler.add_request(Request('B', [42] * 3000, parameters, None))
        mixed = scheduler.schedule()
        new, cached = show('B arrives mid-spec', mixed)

        print()
        print('VERDICT')
        if new and cached:
            counts = dict(mixed.num_scheduled_tokens)
            print('  ONE step carries B\\'s prefill AND A\\'s decode: counts=%s' % counts)
            print('  execute_scheduled runs request.step() for ONE ticket, and a prefill')
            print('  and a verify are different device shapes, so widening the contract is')
            print('  NOT sufficient - the step must be split or B deferred.')
        elif new and not cached:
            print('  the scheduler holds A back and runs B\\'s prefill alone, so relaxing')
            print('  the contract to allow a prefill step between decodes is enough.')
        else:
            print('  B was not scheduled at all while A holds proposals: the scheduler')
            print('  already serialises, and only the lifecycle contract needs to change.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
