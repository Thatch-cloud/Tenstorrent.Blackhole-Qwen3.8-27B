"""One loaded winning T16 runtime, two audited arms and complete ABBA timing."""

from contextlib import nullcontext
import hashlib
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

from mlp_k_block_gate import qualify, REPORT_SHA256
from mlp_k_block_comparison import SCHEDULE, summarize


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import dspark_request_experiment
    import frozen_ladder_requests
    import full_dspark_request
    import fused_t16_scope
    import dspark_fusion_variants
    import gdn_shared_qk_variants

    if (os.environ.get('QWEN_MLP_K_BLOCK') != '1'
            or os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(options.get('prompt', ())) != 4096 or report.get('streams') != 1
            or options.get('captured_publication') is not True):
        raise ValueError('Explicit allocated unprofiled winning 4K comparison required')
    directory = Path(__file__).parent
    evidence = qualify(directory, directory / 'k-block-evidence')
    spec = importlib.util.spec_from_file_location('qualified_k_block_candidate',
        directory / 'mlp-k-block-candidate/fused_1d.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    original_measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route
    files = ('mlp_k_block_gate.py', 'mlp_k_block_comparison.py', 'mlp_k_block_experiment.py',
        'mlp_k_block.py', 'mlp-k-block-candidate/fused_1d.py',
        'mlp-k-block-candidate/fused_1d_input.cpp', 'mlp-k-block-candidate/fused_1d_weights.cpp')
    def fingerprints():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}
    sources = fingerprints()
    calls, finished = [], []

    def measure(*args, **kwargs):
        index = len(calls)
        if index >= len(SCHEDULE):
            raise ValueError('Unexpected additional request')
        enabled, audit = SCHEDULE[index]
        if kwargs.get('audit_features') is not audit or any(kwargs.get(name) is not True for name in (
                'gdn_shared_qk', 'fused_t16_mlp', 'captured_publication', 'target_attention_t16',
                'commit_only_gdn', 'proposal_trace', 'native_attention', 'score_layout')):
            raise ValueError('Both arms must retain the complete winning runtime')
        calls.append((enabled, audit))
        with (patch.object(fused_t16_scope, 'FusedProjection', candidate.FusedProjection) if enabled else nullcontext()), \
                (patch.object(fused_t16_scope, 'qualify_simulator', lambda: evidence) if enabled else nullcontext()):
            result = original_measure(*args, **kwargs)
        result['mlp_k_block'] = dict(k32=enabled,
            simulator_report_sha256=REPORT_SHA256 if enabled else None)
        return result

    def route(value, arm):
        identity = value.get('mlp_k_block', {})
        enabled = identity.get('k32')
        if (type(enabled) is not bool or arm != 'publication'
                or identity.get('simulator_report_sha256') != (REPORT_SHA256 if enabled else None)):
            raise ValueError('Explicit source-bound weight-reader identity required')
        with (patch.object(dspark_fusion_variants, 'REPORT_SHA256', REPORT_SHA256) if enabled else nullcontext()):
            original_route(value, arm)

    def finish(result, summarize_requests):
        comparison = summarize(result['request_checks'], summarize_requests,
            frozen_ladder_requests.validate_audit, route)
        result.update(k_block_comparison=comparison, pp=None, committed_tg=None,
            ctx_tokens=4096, fresh_context_audit=True,
            scope='Same winning T16 recipe with unchanged versus k32 MLP reduction block width')
        finished.append(True)

    try:
        with patch.object(full_dspark_request, 'measure_dspark_request', measure), \
                patch.object(gdn_shared_qk_variants, 'validate_route', route), \
                patch.object(frozen_ladder_requests, 'finish', finish):
            dspark_request_experiment.run_loaded_requests(operations, generator, model, collectives, tokenizer,
                pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary, report, progress, **options)
        if calls != list(SCHEDULE) or finished != [True]:
            raise ValueError('Full two-audit and ABBA comparison required')
    finally:
        report['k_block_sources'] = sources
        report['k_block_sources_after'] = fingerprints()
        if report['k_block_sources_after'] != sources or qualify(directory, directory / 'k-block-evidence') != evidence:
            raise ValueError('K-block qualification or sources changed during execution')
