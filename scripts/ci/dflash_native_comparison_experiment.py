"""Combined DFlash composed/native ABBA; hardware execution requires explicit allocation."""

import hashlib
import os
from pathlib import Path

from dflash_native_comparison_report import SCHEDULE, summarize


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

    prompt = options.get('prompt', ())
    required = ('QWEN_DFLASH_NATIVE_COMPARISON', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS',
                'QWEN_CUMULATIVE_NORM', 'QWEN_CUMULATIVE_REGISTER', 'QWEN_CUMULATIVE_MLP_DOWN')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(prompt) != 4096 or request_limit(options.get('max_new_tokens'), short_default=True) != 256
            or report.get('streams') != 1):
        raise ValueError('Explicitly allocated unprofiled combined T16 DFlash comparison required')
    require_active()
    directory = Path(__file__).parent
    runtime_root = os.environ['TT_METAL_HOME']
    evidence = directory / 'dflash-t16-native-evidence'
    admission = admit(evidence, directory, runtime_root)
    fabric_sources = audit_sampling(runtime_root, {**os.environ, 'QWEN_FABRIC_LINK_PROBE': '1'})
    if report.get('sampling_link_sources') != fabric_sources:
        raise ValueError('Loaded runtime sampling provenance differs from the comparison')
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
    try:
        with sampler_links(sampler.tt_sampling, 4), runtime_scope(directory, runtime_root=runtime_root):
            for ordinal, (native, audited) in enumerate(SCHEDULE):
                progress(f'dflash_attention_{ordinal}_' + ('native' if native else 'composed'))
                result = measure_combined_dflash(operations, model, sampler, prompt, pages, helpers,
                    directory=directory, runtime_root=runtime_root, fixtures=fixtures,
                    audit_features=audited, max_new_tokens=256, **callbacks,
                    **(dict(native_attention_evidence=evidence) if native else {}))
                record_request(report, result, 'dflash2', sampler.tt_sampling, fabric_sources)
                if result.get('native_proposal_attention') is not native:
                    raise ValueError('Measured request did not execute the scheduled attention policy')
                if audited:
                    summarize_dflash_requests([result], audit_only=True)
                progress(f'dflash_attention_{ordinal}_complete')
        comparison = summarize(report['request_checks'])
        report.update(dflash_native_comparison=comparison, pp=None, committed_tg=None,
            ctx_tokens=4096, scope=__doc__, performance_promoted=False)
    finally:
        report['drafter_comparison_sources'] = sources
        report['drafter_comparison_sources_after'] = fingerprints()
        if report['drafter_comparison_sources_after'] != sources:
            raise ValueError('Comparison sources changed during complete requests')
        if admit(evidence, directory, runtime_root) != admission:
            raise ValueError('Native proposal admission changed during comparison')
        if audit_sampling(runtime_root, {**os.environ, 'QWEN_FABRIC_LINK_PROBE': '1'}) != fabric_sources:
            raise ValueError('Sampling provenance changed during comparison')
