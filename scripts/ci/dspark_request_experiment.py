"""Allocated-hardware full-request DSpark screen, reusing the loaded target and verified learned parameters."""

import json
import math
import os
from pathlib import Path
import time
from contextlib import nullcontext

from dspark_hardware_gate import digest
from dspark_projection import tensor_digest
from gdn_multitoken_conv import addresses


PREREQUISITES = {
    'dspark-full-attention-simulator.json': '7fa3290673df7b77aaed954ab55d683caedbcc51e32da4eb10df8eb292829499',
    'dspark-wide-layout-simulator.json': '3746601ab4c6b45b5287b1e41e2bc99d6a74cc25cd9764c92c24d710510dc6ef',
    'dspark-history-bank-simulator.json': '72870b5018ab69fdbc5f86f4ceed6fa334a98bd76931d41536a0379b757523e0',
}
METADATA_ONLY = {'dspark-history-bank-simulator.json': frozenset(('full_dspark_request.py',))}


def request_preflight(directory, *, prepared_proposals=False):
    if type(prepared_proposals) is not bool:
        raise ValueError('Explicit prepared-proposal qualification policy required')
    directory = Path(directory)
    specifications = {
        'dspark-full-attention-simulator.json': dict(eager_checks=4, replay_checks=6, input_checks=40,
            fixture_controls=6, stale_controls=2),
        'dspark-wide-layout-simulator.json': dict(eager_checks=28, replay_checks=28, input_checks=56, stale_controls=14),
        'dspark-history-bank-simulator.json': dict(bank_checks=260, view_checks=80, trace_checks=8,
            binding_checks=4, stale_controls=20),
    }
    prerequisites, metadata_only = {}, {}
    for name, counts in specifications.items():
        path = directory / name
        if digest(path) != PREREQUISITES[name]:
            raise ValueError('Exact independently audited full-history/wider-layout report required')
        report = json.loads(path.read_text())
        if (path.with_suffix('.exit-status').read_text().strip() != '0'
                or report.get('passed') is not True or report.get('closed_cleanly') is not True
                or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
                or report.get('sources') != report.get('sources_after')
                or report.get('native_sources') != report.get('native_sources_after')
                or {key: len(report.get(key, [])) for key in counts} != counts):
            raise ValueError('Complete clean unchanged full-history/wider-layout simulator evidence required')
        for field in counts:
            flag = 'exact' if field == 'input_checks' else 'detected' if field in ('fixture_controls', 'stale_controls') else 'passed'
            if any(record.get(flag) is not True for record in report[field]):
                raise ValueError('Every full-history/wider-layout comparison must pass')
        for source, checksum in report['sources'].items():
            if source in METADATA_ONLY.get(name, ()):
                metadata_only[source] = dict(recorded_sha256=checksum, current_sha256=digest(directory / source),
                    scope='Recorded by bank probe but not imported or executed; integration requires hardware request audits')
            elif source != '../../optimisation/sim/run-dispatch-probe.sh' and digest(directory / source) != checksum:
                raise ValueError('Qualified full-history/wider-layout source changed: ' + source)
        prerequisites[name] = digest(path)
    if prepared_proposals:
        from dspark_proposal_gate import qualify
        from dspark_commit_gate import qualify as qualify_commit
        prerequisites.update(qualify(directory))
        prerequisites.update(qualify_commit(directory))
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix in ('.py', '.cpp', '.hpp', '.h')]
    harness = directory.parents[1] / 'speculative-decoding/harness'
    if not (harness / 'greedy_session.py').is_file():
        raise ValueError('Complete request harness must be mounted before hardware preflight')
    sources = {path.name: digest(path) for path in sorted(files)}
    sources.update({'../../speculative-decoding/harness/' + path.name: digest(path) for path in sorted(harness.glob('*.py'))})
    return dict(request_prerequisites=prerequisites, sources=sources, simulator_metadata_only_sources=metadata_only)


def summarize(requests):
    if (len(requests) != 3 or [value.get('instrumented_timing') for value in requests] != [True, False, False]
            or any(value.get('exact') is not True or value.get('state_exact') is not True
                or value.get('inactive_exact') is not True for value in requests)):
        raise ValueError('One feature-audited and two exact timed full requests required')
    first = requests[0]
    policies = [(value.get('dspark', {}).get('proposal_trace', False), value.get('commit_only_gdn', False))
        for value in requests]
    if any(any(type(flag) is not bool for flag in policy) or policy != policies[0] for policy in policies):
        raise ValueError('One unchanged proposal and target-state policy per measured arm required')
    if any(value['prompt_tokens'] != first['prompt_tokens'] or value['emitted'] != first['emitted']
            or type(value['length']) is not int or value['length'] != len(value['prompt_tokens'])
            or value['committed_decode_tokens'] != len(value['emitted']) - 1 for value in requests):
        raise ValueError('Every complete request must reproduce identical target tokens at the same context')
    timed = requests[1:]
    for value in timed:
        if any(type(value[field]) not in (int, float) or not math.isfinite(value[field]) or value[field] < 0
                for field in ('decode_ms', 'prefill_ms', 'prefill_setup_decode_ms', 'feature_setup_ms', 'engine_setup_ms')):
            raise ValueError('Finite nonnegative complete request timings required')
    tokens = sum(value['committed_decode_tokens'] for value in timed)
    milliseconds = sum(value['decode_ms'] for value in timed)
    prefill_ms = sum(value['prefill_ms'] for value in timed)
    if tokens < 1 or milliseconds <= 0 or prefill_ms <= 0:
        raise ValueError('Positive committed-token, complete decode-cycle and prefill measurements required')
    proposed = sum(value['proposed'] for value in timed)
    accepted = sum(value['accepted'] for value in timed)
    return dict(pp=1000 * sum(value['length'] for value in timed) / prefill_ms, ctx=first['length'],
        committed_tg=1000 * tokens / milliseconds, streams=1, verifier_rows=16, draft_queries=15,
        committed_tokens=tokens, proposed=proposed, accepted=accepted,
        acceptance=accepted / proposed if proposed else None,
        measured_requests=2, audit_requests=1, mean_setup_inclusive_ms=sum(value['prefill_setup_decode_ms'] for value in timed) / 2,
        mean_feature_setup_ms=sum(value['feature_setup_ms'] for value in timed) / 2,
        mean_verifier_setup_ms=sum(value['engine_setup_ms'] for value in timed) / 2,
        proposal_trace=policies[0][0], commit_only_gdn=policies[0][1],
        held_out_coding_quality=False, serving_qualified=False)


def warm_native_control(generator, kv_cache, report, progress):
    progress('warm_native_control_before_fresh_request_prefill')
    started = time.perf_counter()
    generator.warmup_model_decode(kv_cache=kv_cache, enable_trace=True, max_batch_size=1,
        num_blocks=1024, can_sample_on_device=False)
    traces = generator.trace_ids_decode[False]
    if not traces or any(value is None for value in traces.values()):
        raise AssertionError('Native control must not first capture a state-mutating trace inside gold decode')
    report['native_control_warmup'] = dict(milliseconds=(time.perf_counter() - started) * 1000,
        trace_count=len(traces), before_fresh_prefill=True, charged_to_candidate_decode=False)


def cache_formats(operations, caches, recurrent):
    def describe(values):
        records = []
        for index, value in enumerate(values):
            shards = operations.get_device_tensors(value)
            if len(shards) != 2:
                raise ValueError('Cache format inventory requires both physical chips')
            records.append(dict(index=index, shards=[dict(dtype=str(shard.dtype),
                shape=list(shard.shape)) for shard in shards]))
        return records

    return dict(sdpa_bf8_environment=os.environ.get('QWEN_SDPA_BF8', '0'),
        attention_kv=describe(caches), gdn_state=describe(recurrent))


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, *, prompt, context, variants=False,
        native_attention_variants=False, profile_verifier=False, norm_scatter_variants=False,
        target_attention_variants=False, combined_variants=False, mlp_down=False, mlp_equal_footprint=False):
    import torch
    from full_dspark_request import measure_dspark_request
    from full_request import terminal_ids
    from gdn_snapshot import ActiveSnapshot
    from models.common.sampling.generator import SamplingGenerator
    from sampling_link_policy import sampler_links

    eos = terminal_ids(Path(os.environ['MODEL_WEIGHTS_DIR']), model.args.vocab_size)
    sampler = SamplingGenerator(args=model.args, mesh_device=model.mesh_device, tt_ccl=collectives)
    sampler.set_trace_bucket(1)
    layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    helpers = [ActiveSnapshot(layer, operations, direct=True) for layer in layers]
    recurrent = [value for layer in layers for value in (layer.rec_state, *layer.conv_states)]
    caches = [value for pair in model._paged_kv_caches for value in pair]
    if len(helpers) != 48 or len(recurrent) != 240 or len(caches) != 32:
        raise ValueError('Complete native hybrid target state required')
    report['target_cache_formats'] = cache_formats(operations, caches, recurrent)
    bindings = [addresses(operations, value) for value in (*recurrent, *caches)]

    def host(value):
        shards = operations.get_device_tensors(value)
        if len(shards) != 2:
            raise AssertionError('Both actual target-state shards required')
        return [operations.to_torch(shard).clone() for shard in shards]

    def live_digest():
        if [addresses(operations, value) for value in (*recurrent, *caches)] != bindings:
            raise AssertionError('Native target-state bindings changed')
        return [tensor_digest(shard) for value in recurrent for shard in host(value)]

    def kv_digest(valid):
        if type(valid) is not int or not 1 <= valid <= 65536:
            raise ValueError('Explicit valid target KV prefix required')
        result = []
        for value in caches:
            for start in range(0, math.ceil(valid / 64), 64):
                end = min(start + 64, math.ceil(valid / 64))
                sliced = operations.slice(value, (start, 0, 0, 0), (end, value.shape[1], 64, value.shape[3]))
                try:
                    for shard in host(sliced):
                        logical = shard.permute(1, 0, 2, 3).reshape(shard.shape[1], -1, shard.shape[3])
                        result.append(tensor_digest(logical[:, :min((end - start) * 64, valid - start * 64)]))
                finally:
                    if addresses(operations, sliced) != addresses(operations, value):
                        operations.deallocate(sliced)
        return result

    def inactive_digest():
        return [tensor_digest(shard[1:] if index % 5 == 0 else shard[:, 1:])
            for index, value in enumerate(recurrent) for shard in host(value)]

    def prefill(tokens):
        generator.prev_page_table = None
        logits, unused = generator.prefill_forward(torch.tensor([tokens], dtype=torch.int32), pages, kv_cache,
            [len(tokens)], empty_slots=[0], enable_trace=False)
        return int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax())

    def decode(token, position, traced):
        if traced and not generator.trace_ids_decode[False]:
            raise AssertionError('Cold native trace capture is forbidden inside the measured control')
        output = generator.decode_forward(tokens=torch.tensor([[token]], dtype=torch.int32),
            start_pos=torch.tensor([position], dtype=torch.int32), page_table=pages, kv_cache=kv_cache,
            enable_trace=traced, read_from_device=True)
        return (output[0] if isinstance(output, tuple) else output).clone()

    from dspark_request_variants import SCHEDULE, POLICIES, summarize_variants
    if type(mlp_equal_footprint) is not bool or (mlp_equal_footprint and not mlp_down):
        raise ValueError('Equal-footprint diagnostic requires the down-only MLP experiment')
    if type(mlp_down) is not bool or (mlp_down and not target_attention_variants):
        raise ValueError('Down-only MLP requires the matched folded-attention experiment')
    if (type(native_attention_variants) is not bool or type(profile_verifier) is not bool
            or type(norm_scatter_variants) is not bool
            or type(target_attention_variants) is not bool
            or type(combined_variants) is not bool
            or sum((native_attention_variants, variants, profile_verifier, norm_scatter_variants, target_attention_variants, combined_variants)) > 1):
        raise ValueError('Choose one explicit matched experiment')
    if native_attention_variants or profile_verifier or norm_scatter_variants or target_attention_variants or combined_variants:
        from dspark_native_request_variants import SCHEDULE, POLICIES, summarize_variants
        from dspark_native_fixed_gate import qualify
        qualify(Path(__file__).parent)
    if norm_scatter_variants or combined_variants:
        from dspark_norm_request_variants import SCHEDULE, POLICIES, summarize_variants
        from gdn_norm_scatter_scope import scoped_reader
    if target_attention_variants or combined_variants:
        from dspark_target_attention_variants import SCHEDULE, POLICIES, summarize_variants
        from target_t16_attention_gate import qualify as qualify_target
        qualify_target(Path(__file__).parent)
    if combined_variants:
        from dspark_combined_variants import SCHEDULE, POLICIES, summarize_variants
    if mlp_down:
        from dspark_mlp_down_variants import SCHEDULE, POLICIES, summarize_variants
        from dram_mlp_down_scope import scoped_down
        from models.tt_transformers.tt.ccl import tt_all_reduce

    if type(variants) is not bool:
        raise ValueError('Explicit matched proposal experiment selection required')
    schedule = SCHEDULE if variants or native_attention_variants or norm_scatter_variants or target_attention_variants or combined_variants else tuple(('eager', audit) for audit in (True, False, False))
    if profile_verifier:
        schedule = (('native', True),)
    report['coding_context'], report['request_checks'] = context, []
    report['sampler_links'] = 4
    control_warmed = False
    with sampler_links(sampler.tt_sampling, 4):
        for ordinal, (arm, audit) in enumerate(schedule):
            if not audit and not control_warmed:
                warm_native_control(generator, kv_cache, report, progress)
                control_warmed = True
            progress(f'full_request_{ordinal}_{arm}_' + ('feature_audit' if audit else 'timed'))
            from native_draft_sdpa import precise_draft_kernel
            native = POLICIES[arm].get('native_attention', False)
            with (precise_draft_kernel(os.environ['TT_METAL_HOME']) if native else nullcontext()) as kernel_audit, \
                    (scoped_reader() if (norm_scatter_variants or combined_variants) and arm == 'scatter' else nullcontext()) as norm_audit, \
                    (scoped_down(operations, model, tt_all_reduce, enabled=arm == 'down')
                     if mlp_down and (arm == 'down' or mlp_equal_footprint) else nullcontext()) as down_audit:
                result = measure_dspark_request(operations, model, sampler, prompt, pages, helpers, collectives=collectives,
                    parameters=parameters, layer_weights=layer_weights, predecessor=predecessor, successor=successor, rotary=rotary,
                    prefill=prefill, decode=decode, live_digest=live_digest, kv_digest=kv_digest, inactive_digest=inactive_digest,
                    eos_ids=eos, audit_features=audit, max_new_tokens=256 if target_attention_variants or combined_variants else 257, **POLICIES[arm],
                    **(dict(profile_verifier=True) if profile_verifier else {}))
                if native:
                    result['native_attention_kernel'] = kernel_audit
            if mlp_down:
                result['down_mlp'] = down_audit
                result['down_mlp_equal_footprint'] = mlp_equal_footprint
                if down_audit is not None:
                    result['prefill_setup_decode_ms'] += down_audit['setup_ms']
            if norm_scatter_variants or combined_variants:
                result['norm_scatter_kernel'] = norm_audit
            if target_attention_variants or combined_variants:
                from dspark_target_attention_variants import validate_route
                validate_route(result, 'parallel' if combined_variants or mlp_down else arm)
            result['arm'] = arm
            report['request_checks'].append(result)
            progress(f'full_request_{ordinal}_complete')
    if profile_verifier:
        report.update(instrumented_timing=True, correctness_only=True, profile_family='dspark',
            ctx_tokens=len(prompt), drafter_history_rows=len(prompt), proposal_rows=15, pp=None, committed_tg=None)
        return
    if variants or native_attention_variants or norm_scatter_variants or target_attention_variants or combined_variants:
        report['request_comparison'] = summarize_variants(report['request_checks'])
        report['request_summary'] = report['request_comparison']['arms'][
            'down' if mlp_down else 'scatter' if combined_variants else 'parallel' if target_attention_variants else 'scatter' if norm_scatter_variants else 'native' if native_attention_variants else 'trace_commit']
    else:
        report['request_summary'] = summarize(report['request_checks'])
    report.update(ctx_tokens=len(prompt), drafter_history_rows=len(prompt), proposal_rows=15,
        pp=report['request_summary']['pp'], committed_tg=report['request_summary']['committed_tg'])
