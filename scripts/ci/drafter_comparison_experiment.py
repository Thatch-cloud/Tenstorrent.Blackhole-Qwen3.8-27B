"""Two fresh audits and ABBA complete requests on one loaded promoted target."""

import hashlib
import os
from pathlib import Path

from drafter_comparison_report import SCHEDULE, summarize


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import full_dspark_request
    from full_dflash_request import load_dflash_fixtures
    from dflash_combined_request import measure_combined_dflash
    from drafter_request_environment import prepare
    from dspark_request_experiment import warm_native_control, cache_formats
    from sampling_link_policy import sampler_links, audit as audit_sampling
    from drafter_request_metadata import record_request
    from native_draft_sdpa import precise_draft_kernel
    from cumulative_t16_scope import scoped_cumulative_t16, validate_request
    from cumulative_register_runtime import runtime_scope
    from cumulative_norm_runtime import require_active, select_norm
    from cumulative_norm_validation import validate_norm_history
    from frozen_ladder_requests import validate_audit
    from gdn_shared_qk_variants import POLICIES
    import gdn_shared_qk_variants
    from compact_score_gate import qualify as qualify_compact, REPORT_SHA256 as COMPACT_SHA256
    from gdn_direct_window_gate import qualify as qualify_windows, REPORT_SHA256 as WINDOW_SHA256
    from mlp_down_grid_gate import qualify as qualify_down, REPORT_SHA256 as DOWN_SHA256
    from dspark_score_layout_hardware_gate import qualify as qualify_scores
    from dspark_score_layout_hardware_audit import audit as audit_scores
    from dspark_request_limit import request_limit

    prompt = options.get('prompt', ())
    required = ('QWEN_DRAFTER_COMPARISON', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS',
                'QWEN_CUMULATIVE_NORM', 'QWEN_CUMULATIVE_REGISTER', 'QWEN_CUMULATIVE_MLP_DOWN')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(prompt) != 4096 or request_limit(options.get('max_new_tokens'), short_default=True) != 256
            or report.get('streams') != 1):
        raise ValueError('Allocated unprofiled 4K single-stream combined drafter comparison required')
    require_active()
    directory = Path(__file__).parent
    runtime_root = os.environ['TT_METAL_HOME']
    fabric_sources = audit_sampling(runtime_root, {**os.environ, 'QWEN_FABRIC_LINK_PROBE': '1'})
    if report.get('sampling_link_sources') != fabric_sources:
        raise ValueError('Loaded runtime sampling provenance differs from the matched request')
    fixtures = load_dflash_fixtures('/experiment-dflash-fixture')
    windows = qualify_windows(directory, directory / 'gdn-direct-window-evidence')
    compact = qualify_compact(directory, directory / 'compact-score-evidence')
    down = qualify_down(directory / 'mlp-down-grid-evidence', directory, runtime_root)
    qualify_scores(directory)
    report['score_layout_hardware_audit'] = audit_scores(operations, model.mesh_device, predecessor, successor)

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
        with sampler_links(sampler.tt_sampling, 4), runtime_scope(directory, runtime_root=runtime_root) as register:
            for ordinal, (drafter, audited) in enumerate(SCHEDULE):
                progress(f'combined_drafter_{ordinal}_{drafter}_' + ('audit' if audited else 'timed'))
                common = dict(callbacks, audit_features=audited, max_new_tokens=256)
                if drafter == 'dspark':
                    with select_norm('scatter'), scoped_cumulative_t16(windows, compact, directory,
                            down_admission=down) as audit, register(), precise_draft_kernel(runtime_root) as kernel_audit:
                        result = full_dspark_request.measure_dspark_request(operations, model, sampler, prompt,
                            pages, helpers, collectives=collectives, parameters=parameters,
                            layer_weights=layer_weights, predecessor=predecessor, successor=successor,
                            rotary=rotary, score_layout_evidence=report['score_layout_hardware_audit'],
                            **common, **POLICIES['publication'])
                        result['native_attention_kernel'] = kernel_audit
                    record_request(report, result, drafter, sampler.tt_sampling, fabric_sources)
                    validate_request(result, audit)
                    validate_norm_history(result, 'scatter')
                    result['gdn_direct_window'] = dict(direct=True, hits=audit['direct']['hits'],
                        report_sha256=WINDOW_SHA256, restored=True)
                    result['mlp_down_grid'] = dict(wider_down=True, hits=audit['down']['hits'],
                        report_sha256=DOWN_SHA256, restored=True)
                    result['compact_score'] = dict(compact=True, hits=audit['compact']['calls'],
                        report_sha256=COMPACT_SHA256, restored=True)
                    result['arm'] = 'publication'
                    if audited:
                        validate_audit(result)
                    else:
                        gdn_shared_qk_variants.validate_route(result, 'publication')
                else:
                    result = measure_combined_dflash(operations, model, sampler, prompt, pages, helpers,
                        directory=directory, runtime_root=runtime_root, fixtures=fixtures, **common)
                    record_request(report, result, drafter, sampler.tt_sampling, fabric_sources)
                    if audited:
                        from full_dflash_request import summarize_dflash_requests
                        summarize_dflash_requests([result], audit_only=True)
                progress(f'combined_drafter_{ordinal}_complete')
        comparison = summarize(report['request_checks'])
        report.update(drafter_comparison=comparison, pp=None, committed_tg=None,
            ctx_tokens=4096, scope=__doc__, performance_promoted=False)
    finally:
        report['drafter_comparison_sources'] = sources
        report['drafter_comparison_sources_after'] = fingerprints()
        if report['drafter_comparison_sources_after'] != sources:
            raise ValueError('Drafter comparison sources changed during complete requests')
        if audit_sampling(runtime_root, {**os.environ, 'QWEN_FABRIC_LINK_PROBE': '1'}) != fabric_sources:
            raise ValueError('Sampling provenance changed during complete requests')
