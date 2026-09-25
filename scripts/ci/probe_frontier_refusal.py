"""Which clause refused the first solo decode step in run 35484349353?

v38 was the first run to get past admission under the watcher. The first request
prefilled, its verifier built, its first eager proposal completed on the device,
and then the very next step - `[PHASE] execute total=16 new=0 cached=1 spec=1`,
one resident request, sixteen rows - was refused by

    serving_vllm_packed.ordered_tickets  ->  line 31
    'Resident decode requests at the exact scheduled frontier required'

That raise guards five things at once (new requests, finished ids, preempted ids,
structured output, encoder inputs) and the message names none of them. The
single-request contract in serving_vllm_contract carries the same five, and the
single-user run 35475120459 passed them ten times, so something about THIS step
differed. The engine's stats line just before it said Running 1, Waiting 0: the
second prompt had not reached the scheduler yet, so the step was built with one
running request and no waiting one - the opposite order from v35, where B was
prefilled before A's first decode.

This replays both orders against the real scheduler classes on CPU, at the fp2u
geometry (33024 context, 32768-token prompts, 1032 blocks of 64, 15 draft tokens,
two sequences), and runs the repo's contract on the outputs it produces:

  - stock vllm Scheduler, as the control
  - the plugin's TTScheduler
  - serving_one_in_flight.OneInFlightScheduler, the class serving actually installs

For every step it prints the fields the clause reads, then calls
admit_packed_scheduler_output with a ticket at the scheduler's own frontier, so
the refusal (if any) is reproduced with its real inputs.

CPU only: no device, no weights.
"""

import inspect
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
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

CONTEXT = 33024
PROMPT = 32768
BLOCKS = 1032
BLOCK = 64
DRAFTS = 15
PROPOSALS = list(range(101, 101 + DRAFTS))


def build(directory, scheduler_type=None):
    GPT2Config(n_positions=65536, n_embd=256, n_layer=1, n_head=4).save_pretrained(directory)
    model = ModelConfig(model=directory, dtype='float32', max_model_len=CONTEXT,
                        skip_tokenizer_init=True, seed=0)
    speculative = SpeculativeConfig(model='ngram', num_speculative_tokens=DRAFTS)
    speculative.method = 'dflash'
    config = VllmConfig(
        model_config=model, device_config=DeviceConfig(device='cpu'),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=CONTEXT,
                                         max_model_len=CONTEXT, is_encoder_decoder=False,
                                         enable_chunked_prefill=False, async_scheduling=False,
                                         watermark=0.0),
        cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=False),
        parallel_config=ParallelConfig(), speculative_config=speculative)
    config.cache_config.num_gpu_blocks = BLOCKS
    cache = KVCacheConfig(num_blocks=BLOCKS, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(['layer'], FullAttentionSpec(block_size=BLOCK, num_kv_heads=2,
                                                      head_size=256, dtype=torch.bfloat16))])
    register_all_kvcache_specs(config)
    kind = scheduler_type or Scheduler
    scheduler = kind(config, cache, StructuredOutputManager(config), block_size=BLOCK)
    scheduler.use_v2_model_runner = False
    return scheduler


def output(req_ids, token):
    return ModelRunnerOutput(req_ids=list(req_ids),
                             req_id_to_index={r: i for i, r in enumerate(req_ids)},
                             sampled_token_ids=[[token] for _ in req_ids],
                             logprobs=None, prompt_logprobs_dict={}, pooler_output=[])


def fields(scheduled):
    """Every value the refusing clause reads, plus the frontier."""
    cached = scheduled.scheduled_cached_reqs
    return dict(
        new=[r.req_id for r in scheduled.scheduled_new_reqs],
        cached=list(getattr(cached, 'req_ids', []) or []),
        frontier=list(getattr(cached, 'num_computed_tokens', []) or []),
        resumed=sorted(getattr(cached, 'resumed_req_ids', ()) or ()),
        resumed_flags=list(getattr(cached, 'resumed_from_preemption', []) or []),
        counts=dict(scheduled.num_scheduled_tokens),
        total=scheduled.total_num_scheduled_tokens,
        spec={k: len(v) for k, v in dict(scheduled.scheduled_spec_decode_tokens).items()},
        finished=sorted(scheduled.finished_req_ids or ()),
        preempted=getattr(scheduled, 'preempted_req_ids', 'ABSENT'),
        structured=getattr(scheduled, 'has_structured_output_requests', 'ABSENT'),
        encoder=getattr(scheduled, 'scheduled_encoder_inputs', 'ABSENT'))


def show(label, scheduled):
    values = fields(scheduled)
    print('%-34s new=%s cached=%s frontier=%s counts=%s total=%s spec=%s'
          % (label, values['new'], values['cached'], values['frontier'], values['counts'],
             values['total'], values['spec']))
    print('%-34s finished=%s preempted=%r structured=%r encoder=%r resumed=%s flags=%s'
          % ('', values['finished'], values['preempted'], values['structured'],
             values['encoder'], values['resumed'], values['resumed_flags']))
    return values


def fake_request(request_id, position, tokens):
    """The least a request needs to satisfy serving_vllm_contract.prepared_ticket."""
    ticket = SimpleNamespace(request_id=request_id, position=position, tokens=list(tokens))
    session = SimpleNamespace(pending=ticket, phase='pending', request_id=request_id, position=position)
    return SimpleNamespace(closed=False, cancelled=False, busy=False, session=session,
                           engine=SimpleNamespace(phase='idle'))


def contract(label, scheduled):
    """Run the repo's packed contract on this exact output, at its own frontier."""
    from serving_vllm_packed import admit_packed_scheduler_output

    values = fields(scheduled)
    requests = []
    for request_id, frontier, in zip(values['cached'], values['frontier']):
        rows = values['counts'].get(request_id, 0)
        requests.append(fake_request(request_id, frontier, [100] + PROPOSALS[:max(0, rows - 1)]))
    try:
        entries = admit_packed_scheduler_output(requests, scheduled)
        result = 'ADMITTED %s' % [entry['request_id'] for entry in entries]
    except BaseException as error:
        result = 'REFUSED: %s: %s' % (type(error).__name__, error)
    clause = dict(new=bool(scheduled.scheduled_new_reqs), finished=bool(scheduled.finished_req_ids),
                  preempted=bool(getattr(scheduled, 'preempted_req_ids', None)),
                  structured=bool(getattr(scheduled, 'has_structured_output_requests', False)),
                  encoder=bool(getattr(scheduled, 'scheduled_encoder_inputs', {})),
                  empty=not values['cached'])
    print('%-34s contract: %s' % (label, result))
    print('%-34s clause truth: %s' % ('', {k: v for k, v in clause.items() if v} or 'all false'))
    return result, clause


def scenario(label, scheduler_type, late):
    """A prefills and proposes; B arrives after (late) or before A's first decode."""
    parameters = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    with TemporaryDirectory() as directory:
        scheduler = build(directory, scheduler_type)
        scheduler.add_request(Request('A', [1000 + i % 64 for i in range(PROMPT)], parameters, None))
        first = scheduler.schedule()
        show('%s A prefill' % label, first)
        scheduler.update_from_output(first, output(['A'], 100))
        scheduler.update_draft_token_ids(DraftTokenIds(['A'], [PROPOSALS]))
        if not late:
            scheduler.add_request(Request('B', [1000 + i % 64 for i in range(PROMPT)], parameters, None))
        second = scheduler.schedule()
        values = show('%s step 2 (%s B)' % (label, 'no' if late else 'with'), second)
        results = []
        if values['cached'] and not values['new']:
            results.append(contract('%s step 2' % label, second))
        ids = values['new'] + values['cached']
        if ids:
            scheduler.update_from_output(second, output(ids, 200))
            for name in ids:
                scheduler.update_draft_token_ids(DraftTokenIds([name], [PROPOSALS]))
        if late:
            scheduler.add_request(Request('B', [1000 + i % 64 for i in range(PROMPT)], parameters, None))
        for index in range(3, 7):
            step = scheduler.schedule()
            values = show('%s step %d' % (label, index), step)
            if values['cached'] and not values['new']:
                results.append(contract('%s step %d' % (label, index), step))
            ids = values['new'] + values['cached']
            if not ids:
                print('%-34s empty step, stopping' % '')
                break
            scheduler.update_from_output(step, output(ids, 200 + index))
            for name in ids:
                scheduler.update_draft_token_ids(DraftTokenIds([name], [PROPOSALS]))
        return results


def plugin_sources():
    """The plugin's mode negotiation, since it decides prefill-or-decode and what
    the fallback carries across a discarded pass."""
    try:
        import vllm_tt_plugin.scheduler as module
    except BaseException as error:
        print('plugin scheduler module unavailable: %s' % error)
        return
    for name, value in sorted(vars(module).items()):
        if not inspect.isclass(value):
            continue
        for method in ('_local_prefill_intent', '_negotiate_forced_mode', 'schedule', '_has_capacity'):
            function = vars(value).get(method)
            if function is None:
                continue
            try:
                source = inspect.getsource(function).splitlines()
            except BaseException:
                continue
            print('----- %s.%s (%d lines) -----' % (name, method, len(source)))
            for line in source[:70]:
                print(line)
            if len(source) > 70:
                print('... truncated')


def main():
    plugin_sources()
    print()
    kinds = [('stock', None)]
    try:
        from vllm_tt_plugin.scheduler import TTScheduler
        kinds.append(('TTSched', TTScheduler))
        from serving_one_in_flight import one_in_flight_scheduler
        kinds.append(('OneInFlight', one_in_flight_scheduler(TTScheduler)))
    except BaseException as error:
        print('plugin scheduler unavailable: %s' % error)
    summary = {}
    for label, kind in kinds:
        for late in (True, False):
            name = '%s/%s' % (label, 'late-B' if late else 'early-B')
            print('=== %s ===' % name)
            try:
                summary[name] = scenario(label, kind, late)
            except BaseException as error:
                print('%s failed: %s: %s' % (name, type(error).__name__, error))
                summary[name] = None
            print()

    print('VERDICT')
    for name, results in summary.items():
        if results is None:
            print('  %-22s scenario itself failed' % name)
            continue
        refused = [(clause) for result, clause in results if result.startswith('REFUSED')]
        if not results:
            print('  %-22s no solo/packed decode step was produced' % name)
        elif refused:
            print('  %-22s REFUSED at %d of %d decode steps; truthy clause fields: %s'
                  % (name, len(refused), len(results),
                     sorted({k for clause in refused for k, v in clause.items() if v})))
        else:
            print('  %-22s every decode step admitted (%d)' % (name, len(results)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
