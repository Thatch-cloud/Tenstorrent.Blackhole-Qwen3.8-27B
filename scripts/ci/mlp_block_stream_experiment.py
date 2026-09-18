"""Complete DFlash native-draft requests, varying only target weight transport."""

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
                    result = measure_combined_dflash(operations, model, sampler, prompt, pages, helpers,
                        directory=directory, runtime_root=runtime_root, fixtures=fixtures,
                        native_attention_evidence=native_evidence,
                        block_stream=(dict(evidence=directory / 'block-stream-evidence', streams=streams) if enabled else None),
                        audit_features=audited, max_new_tokens=256, **callbacks)
                    record_request(report, result, 'dflash2', sampler.tt_sampling, fabric)
                    if result.get('native_proposal_attention') is not True or ('block_stream' in result) is not enabled:
                        raise ValueError('Executed request differs from scheduled transport policy')
                    if audited:
                        summarize_dflash_requests([result], audit_only=True)
                    progress(f'block_stream_{ordinal}_complete')
        report.update(block_stream_comparison=summarize(report['request_checks'], weight_transport=True),
            pp=None, committed_tg=None, ctx_tokens=4096, scope=__doc__, performance_promoted=False)
    finally:
        report['drafter_comparison_sources'] = sources
        report['drafter_comparison_sources_after'] = fingerprints()
        if report['drafter_comparison_sources_after'] != sources:
            raise ValueError('Comparison sources changed during complete requests')
        if admit(native_evidence, directory, runtime_root) != admission:
            raise ValueError('Native proposal admission changed during comparison')
        if audit_sampling(runtime_root, {**os.environ, 'QWEN_FABRIC_LINK_PROBE': '1'}) != fabric:
            raise ValueError('Sampling provenance changed during comparison')
