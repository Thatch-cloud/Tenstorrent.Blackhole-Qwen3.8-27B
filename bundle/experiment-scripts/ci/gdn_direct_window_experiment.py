"""Matched combined T16 requests; retain HiFi4 drafting and norm prefetch in both arms."""

from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

from gdn_direct_window_comparison import SCHEDULE, summarize
from gdn_direct_window_gate import qualify, REPORT_SHA256
from gdn_direct_window_scope import scoped_direct_windows


FILES = ('gdn_direct_window_experiment.py', 'gdn_direct_window_comparison.py', 'gdn_direct_window_gate.py',
    'gdn_direct_window_scope.py', 'gdn_direct_window.py', 'gdn-direct-window-probe.py',
    'gdn_direct_window_device.py', 'gdn_direct_window_report.py', 'gdn_direct_window_hardware_sources.py',
    'gdn_direct_window_hardware_device.py', 'gdn_direct_window_hardware_batch.py')


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import dspark_request_experiment
    import frozen_ladder_requests
    import full_dspark_request
    from gdn_shared_qk_variants import validate_route

    if (any(os.environ.get(name) != '1' for name in ('QWEN_GDN_DIRECT_WINDOW',
            'QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS'))
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(options.get('prompt', ())) != 4096 or report.get('streams') != 1
            or options.get('captured_publication') is not True):
        raise ValueError('Explicit allocated unprofiled winning 4K input comparison required')
    directory = Path(__file__).parent
    evidence = qualify(directory, directory / 'gdn-direct-window-evidence')
    original_measure = full_dspark_request.measure_dspark_request

    def fingerprints():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES}

    sources = fingerprints()
    calls, finished = [], []

    def measure(*args, **kwargs):
        if len(calls) >= len(SCHEDULE):
            raise ValueError('Unexpected additional gdn-direct-window request')
        enabled, audited = SCHEDULE[len(calls)]
        if (kwargs.get('audit_features') is not audited
                or any(kwargs.get(name) is not True for name in ('gdn_shared_qk', 'fused_t16_mlp',
                    'captured_publication', 'target_attention_t16', 'commit_only_gdn',
                    'proposal_trace', 'native_attention', 'score_layout'))
                or any(kwargs.get(name) for name in ('t32', 'banked_proposal', 'profile_verifier', 'profile_drafter'))):
            raise ValueError('Both direct-window arms must retain the complete winning T16 runtime')
        calls.append((enabled, audited))
        audit = None
        try:
            with (scoped_direct_windows(evidence, directory) if enabled else nullcontext()) as audit:
                result = original_measure(*args, **kwargs)
        finally:
            report.setdefault('window_route_diagnostics', []).append(dict(
                request_index=len(calls) - 1, candidate=enabled, scope=dict(audit) if audit is not None else None))
        hits = audit['hits'] if audit is not None else 0
        loads = len(result.get('gdn_shared_qk', {}).get('loads', []))
        report['window_route_diagnostics'][-1]['shared_qk_builds'] = loads
        if enabled and (not audit['restored'] or type(hits) is not int or hits < 48 or hits % 48
                or hits != loads):
            raise ValueError(f'T16 window route mismatch: hits={hits}, shared_qk_builds={loads}, scope={audit}')
        result['gdn_direct_window'] = dict(direct=enabled, hits=hits,
            report_sha256=REPORT_SHA256 if enabled else None, restored=True)
        return result

    def finish(result, summarize_requests):
        comparison = summarize(result['request_checks'], summarize_requests,
            frozen_ladder_requests.validate_audit, validate_route)
        result.update(gdn_direct_window_comparison=comparison, pp=None, committed_tg=None,
            ctx_tokens=4096, fresh_context_audit=True,
            scope='Same winning T16/HiFi4 runtime; direct convolution-window construction only')
        finished.append(True)

    try:
        with patch.object(full_dspark_request, 'measure_dspark_request', measure), \
                patch.object(frozen_ladder_requests, 'finish', finish):
            dspark_request_experiment.run_loaded_requests(operations, generator, model, collectives, tokenizer,
                pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary, report, progress, **options)
        if calls != list(SCHEDULE) or finished != [True]:
            raise ValueError('Two fresh audits and complete ABBA timing required')
    finally:
        report['gdn_direct_window_sources'] = sources
        report['gdn_direct_window_sources_after'] = fingerprints()
        if (report['gdn_direct_window_sources_after'] != sources
                or qualify(directory, directory / 'gdn-direct-window-evidence') != evidence):
            raise ValueError('Copy sources or runtime admission changed during execution')
