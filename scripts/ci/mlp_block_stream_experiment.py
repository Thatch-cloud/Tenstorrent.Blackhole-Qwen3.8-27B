"""Complete DFlash native-draft requests with one explicit transport comparison."""

import hashlib
import os
from pathlib import Path

from dflash_native_comparison_report import SCHEDULE, summarize
from mlp_block_stream_pool import owned_streams
from mlp_block_stream_runtime import require_hardware


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    from full_dflash_request import load_dflash_fixtures, summarize_dflash_requests
    from dflash_combined_request import measure_combined_dflash
    from dflash_t16_native_scope import admit
    from drafter_request_environment import prepare
    from drafter_request_metadata import record_request
    from dspark_request_experiment import warm_native_control, cache_formats
    from sampling_link_policy import sampler_links, audit as audit_sampling
    from cumulative_register_runtime import runtime_scope
    from cumulative_norm_runtime import require_active
    from dspark_request_limit import request_limit

    require_hardware(os.environ)
    pipeline_flag = os.environ.get('QWEN_BULK_PIPELINE_EXPERIMENT', '0')
    if pipeline_flag not in ('0', '1'):
        raise ValueError('Explicit zero/one bulk-pipeline experiment required')
    bulk_pipeline = pipeline_flag == '1'
    publication_flag = os.environ.get('QWEN_DRAFT_KV_SLIDE_EXPERIMENT', '0')
    if publication_flag not in ('0', '1') or (publication_flag == '1' and bulk_pipeline):
        raise ValueError('One explicit publication or weight transport experiment required')
    kv_publication = publication_flag == '1'
    progressive_flag = os.environ.get('QWEN_PROGRESSIVE_INPUT_EXPERIMENT', '0')
    if progressive_flag not in ('0', '1') or (progressive_flag == '1' and (bulk_pipeline or not kv_publication)):
        raise ValueError('Progressive comparison requires serial weights and fixed KV publication')
    progressive_input = progressive_flag == '1'
    prompt = options.get('prompt', ())
    if (any(os.environ.get(name) != '1' for name in
            ('QWEN_CUMULATIVE_NORM', 'QWEN_CUMULATIVE_REGISTER', 'QWEN_CUMULATIVE_MLP_DOWN'))
            or len(prompt) != 4096 or request_limit(options.get('max_new_tokens'), short_default=True) != 256
            or report.get('streams') != 1):
        raise ValueError('Complete matched 4K single-stream promoted target required')
    require_active()
    directory = Path(__file__).parent
    runtime_root = os.environ['TT_METAL_HOME']
    native_evidence = directory / 'dflash-t16-native-evidence'
    admission = admit(native_evidence, directory, runtime_root)
    if kv_publication:
        from draft_kv_slide_gate import qualify as qualify_publication

        publication_evidence = directory / 'draft-kv-slide-evidence'
        publication_admission = qualify_publication(directory, publication_evidence)
    if bulk_pipeline:
        from mlp_block_stream_pipeline_preload import preload

        pipeline_admission = preload(directory, runtime_root)
    if progressive_input:
        from mlp_progressive_input_preload import preload as preload_progressive
        from draft_kv_slide_gate import DIRECT_REPORT_SHA256

        if publication_admission.get('report_sha256') != DIRECT_REPORT_SHA256:
            raise ValueError('Progressive comparison requires direct DMA in both arms')
        progressive_admission = preload_progressive(directory, runtime_root)
    fabric = audit_sampling(runtime_root, {**os.environ, 'QWEN_FABRIC_LINK_PROBE': '1'})
    if report.get('sampling_link_sources') != fabric:
        raise ValueError('Loaded sampling runtime differs from four-link admission')
    fixtures = load_dflash_fixtures('/experiment-dflash-fixture')

    def fingerprints():
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob('*')) if path.suffix in ('.py', '.cpp', '.hpp', '.h')}

    sources = fingerprints()
    sampler, helpers, callbacks = prepare(operations, generator, model, collectives, pages, kv_cache)
    recurrent = [value for layer in model.layers if not layer.is_full_attention
                 for value in (layer.attention.rec_state, *layer.attention.conv_states)]
    report['target_cache_formats'] = cache_formats(operations,
        [value for pair in model._paged_kv_caches for value in pair], recurrent)
    report.update(coding_context=options['context'], request_checks=[], request_output_limit=256, sampler_links=4)
    if bulk_pipeline:
        report['weight_comparison_policy'] = 'serial-vs-bulk-pipeline'
    if kv_publication:
        report['weight_comparison_policy'] = 'serial-weights-kv-publication'
    if progressive_input:
        report['weight_comparison_policy'] = 'progressive-input-fixed-publication'
    warm_native_control(generator, kv_cache, report, progress)
    weights = [layer.feed_forward.weights.w_gate_up for layer in model.layers]
    mesh = model.layers[0].feed_forward.device
    try:
        progress('block_stream_pool_prepare')
        with owned_streams(operations, mesh, weights) as (streams, pool):
            report['block_stream_pool'] = pool
            progress('block_stream_pool_ready')
            with sampler_links(sampler.tt_sampling, 4), runtime_scope(directory, runtime_root=runtime_root):
                for ordinal, (enabled, audited) in enumerate(SCHEDULE):
                    progress(f'block_stream_{ordinal}_' + ('candidate' if enabled else 'control'))
                    stream_options = (dict(evidence=directory / 'block-stream-evidence', streams=streams)
                        if enabled or bulk_pipeline or kv_publication else None)
                    if enabled and bulk_pipeline:
                        stream_options['pipeline_evidence'] = directory / 'bulk-pipeline-evidence'
                    if enabled and progressive_input:
                        stream_options['progressive_evidence'] = directory / 'progressive-input-evidence'
                    result = measure_combined_dflash(operations, model, sampler, prompt, pages, helpers,
                        directory=directory, runtime_root=runtime_root, fixtures=fixtures,
                        native_attention_evidence=native_evidence,
                        block_stream=stream_options,
                        audit_features=audited, max_new_tokens=256, **callbacks,
                        **(dict(kv_publication_evidence=publication_evidence) if kv_publication and (enabled or progressive_input) else {}))
                    if kv_publication and not enabled and not progressive_input:
                        result['draft_kv_slide'] = dict(enabled=False, restored=True, serving_defaults_changed=False)
                    record_request(report, result, 'dflash2', sampler.tt_sampling, fabric)
                    if (result.get('native_proposal_attention') is not True
                            or ('block_stream' in result) is not (enabled or bulk_pipeline or kv_publication)
                            or result.get('block_stream', {}).get('bulk_pipeline', False) is not (enabled and bulk_pipeline)
                            or result.get('block_stream', {}).get('progressive_input', False) is not (enabled and progressive_input)):
                        raise ValueError('Executed request differs from scheduled transport policy')
                    if audited:
                        summarize_dflash_requests([result], audit_only=True)
                    progress(f'block_stream_{ordinal}_complete')
        report.update(block_stream_comparison=summarize(report['request_checks'],
            weight_transport=not (bulk_pipeline or kv_publication), bulk_pipeline=bulk_pipeline,
            kv_publication=kv_publication and not progressive_input, progressive_input=progressive_input),
            pp=None, committed_tg=None, ctx_tokens=4096, scope=__doc__, performance_promoted=False)
    finally:
        report['drafter_comparison_sources'] = sources
        report['drafter_comparison_sources_after'] = fingerprints()
        if report['drafter_comparison_sources_after'] != sources:
            raise ValueError('Comparison sources changed during complete requests')
        if admit(native_evidence, directory, runtime_root) != admission:
            raise ValueError('Native proposal admission changed during comparison')
        if bulk_pipeline and preload(directory, runtime_root) != pipeline_admission:
            raise ValueError('Bulk pipeline admission changed during comparison')
        if progressive_input and preload_progressive(directory, runtime_root) != progressive_admission:
            raise ValueError('Progressive input admission changed during comparison')
        if kv_publication and qualify_publication(directory, publication_evidence) != publication_admission:
            raise ValueError('K/V publication admission changed during comparison')
        if audit_sampling(runtime_root, {**os.environ, 'QWEN_FABRIC_LINK_PROBE': '1'}) != fabric:
            raise ValueError('Sampling provenance changed during comparison')
