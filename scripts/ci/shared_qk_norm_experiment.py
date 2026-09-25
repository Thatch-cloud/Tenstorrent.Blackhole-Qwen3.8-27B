"""Complete winning-runtime requests comparing only the GDN norm reader."""

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

from shared_qk_norm_comparison import summarize


SOURCES = ('shared_qk_norm_experiment.py', 'shared_qk_norm_comparison.py',
    'shared_qk_norm_comparison_scope.py', 'shared_qk_norm_scatter.py',
    'shared_qk_norm_scatter_gate.py', 'gdn_norm_scatter.py')


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import dspark_request_experiment
    import frozen_ladder_requests
    import gdn_shared_qk_variants

    if (os.environ.get('QWEN_SHARED_QK_NORM_COMPARISON') != '1'
            or os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(options.get('prompt', ())) != 4096 or report.get('streams') != 1
            or options.get('captured_publication') is not True):
        raise ValueError('Explicit allocated unprofiled winning 4K comparison required')
    directory = Path(__file__).parent

    def fingerprints():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCES}

    before = fingerprints()
    finished = []

    def finish(result, summarize_requests):
        comparison = summarize(result['request_checks'], summarize_requests,
            frozen_ladder_requests.validate_audit, gdn_shared_qk_variants.validate_route)
        result.update(norm_reader_comparison=comparison, pp=None, committed_tg=None,
            ctx_tokens=4096, fresh_context_audit=True,
            comparison_axis='Winning prefetched versus direct-scatter norm reader; all other runtime options fixed',
            scope=__doc__)
        finished.append(True)

    try:
        with patch.object(frozen_ladder_requests, 'finish', finish):
            dspark_request_experiment.run_loaded_requests(operations, generator, model, collectives, tokenizer,
                pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary, report, progress, **options)
        if finished != [True]:
            raise ValueError('Complete norm reader comparison required')
    finally:
        report['norm_reader_sources'] = before
        report['norm_reader_sources_after'] = fingerprints()
        if report['norm_reader_sources_after'] != before:
            raise ValueError('Norm comparison sources changed during execution')
