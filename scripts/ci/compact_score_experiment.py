"""Matched combined T16 requests; retain HiFi4 drafting and norm prefetch in both arms."""

from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

from compact_score_comparison import SCHEDULE, summarize
from compact_score_gate import qualify, REPORT_SHA256
from compact_score_scope import scoped_compact_scores


FILES = ('compact_score_experiment.py', 'compact_score_comparison.py', 'compact_score_gate.py',
    'compact_score_scope.py', 'compact_score_hardware_sources.py', 'compact_score_report.py',
    'compact_score_device.py', 'compact_markov.py', 'compact-score-probe.py',
    'compact_score_hardware_device.py', 'compact_score_hardware_markov.py',
    'compact_score_io.cpp', 'compact_score_compute.cpp', 'compact_score_reduce.cpp')


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import dspark_request_experiment
    import frozen_ladder_requests
    import full_dspark_request
    from gdn_shared_qk_variants import validate_route

    if (any(os.environ.get(name) != '1' for name in ('QWEN_COMPACT_SCORE_HARDWARE',
            'QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS'))
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(options.get('prompt', ())) != 4096 or report.get('streams') != 1
            or options.get('captured_publication') is not True):
        raise ValueError('Explicit allocated unprofiled winning 4K input comparison required')
    directory = Path(__file__).parent
    evidence = qualify(directory, directory / 'compact-score-evidence')
    original_measure = full_dspark_request.measure_dspark_request

    def fingerprints():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES}

    sources = fingerprints()
    calls, finished = [], []

    def measure(*args, **kwargs):
        if len(calls) >= len(SCHEDULE):
            raise ValueError('Unexpected additional compact-score request')
        enabled, audited = SCHEDULE[len(calls)]
        if (kwargs.get('audit_features') is not audited
                or any(kwargs.get(name) is not True for name in ('gdn_shared_qk', 'fused_t16_mlp',
                    'captured_publication', 'target_attention_t16', 'commit_only_gdn',
                    'proposal_trace', 'native_attention', 'score_layout'))
                or any(kwargs.get(name) for name in ('t32', 'banked_proposal', 'profile_verifier', 'profile_drafter'))):
            raise ValueError('Both compact-score arms must retain the complete winning T16 runtime')
        calls.append((enabled, audited))
        audit = None
        try:
            with (scoped_compact_scores(evidence, directory) if enabled else nullcontext()) as audit:
                result = original_measure(*args, **kwargs)
        finally:
            report.setdefault('compact_route_diagnostics', []).append(dict(
                request_index=len(calls) - 1, candidate=enabled, scope=dict(audit) if audit is not None else None))
        hits = audit['calls'] if audit is not None else 0
        loads = result.get('score_layout', {}).get('calls', 0)
        report['compact_route_diagnostics'][-1]['score_layout_calls'] = loads
        if enabled and (not audit['restored'] or type(hits) is not int or hits < 1
                or hits != loads):
            raise ValueError(f'Compact score route mismatch: hits={hits}, score_layout_calls={loads}, scope={audit}')
        result['compact_score'] = dict(compact=enabled, hits=hits,
            report_sha256=REPORT_SHA256 if enabled else None, restored=True)
        return result

    def finish(result, summarize_requests):
        comparison = summarize(result['request_checks'], summarize_requests,
            frozen_ladder_requests.validate_audit, validate_route)
        result.update(compact_score_comparison=comparison, pp=None, committed_tg=None,
            ctx_tokens=4096, fresh_context_audit=True,
            scope='Same winning T16/HiFi4 runtime; draft score selection only')
        finished.append(True)

    try:
        with patch.object(full_dspark_request, 'measure_dspark_request', measure), \
                patch.object(frozen_ladder_requests, 'finish', finish):
            dspark_request_experiment.run_loaded_requests(operations, generator, model, collectives, tokenizer,
                pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary, report, progress, **options)
        if calls != list(SCHEDULE) or finished != [True]:
            raise ValueError('Two fresh audits and complete ABBA timing required')
    finally:
        report['compact_score_sources'] = sources
        report['compact_score_sources_after'] = fingerprints()
        if (report['compact_score_sources_after'] != sources
                or qualify(directory, directory / 'compact-score-evidence') != evidence):
            raise ValueError('Copy sources or runtime admission changed during execution')
