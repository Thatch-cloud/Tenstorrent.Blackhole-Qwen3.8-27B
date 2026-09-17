"""Explicit offline combined T16 cache experiment; no serving defaults."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from prefill_prefix_lookup import PrefixIdentity
from prefill_prefix_session import offline_prefix_session


SOURCES = (
    'prefill_prefix_experiment.py', 'prefill_prefix_session.py', 'prefill_prefix_controller.py',
    'prefill_prefix_residency.py', 'prefill_prefix_lookup.py', 'prefill_prefix_boundary.py',
    'prefill_prefix_features.py', 'prefill_prefix_resume.py', 'prefill_gdn_checkpoint.py',
    'prefill_checkpoint_allocation.py', 'full_dspark_request.py', 'dspark_request_experiment.py',
)


def fingerprints():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCES}


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import dspark_request_experiment
    import frozen_ladder_requests
    import full_dspark_request

    prompt = options.get('prompt', ())
    if (os.environ.get('QWEN_PREFIX_CACHE_EXPERIMENT') != '1'
            or os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('TT_METAL_SIMULATOR')
            or len(prompt) != 4096 or options.get('captured_publication') is not True
            or options.get('profile_verifier') or options.get('profile_drafter')
            or report.get('streams') != 1):
        raise ValueError('Explicit allocated offline 4K combined cache qualification required')
    target = {name: report[name] for name in ('target_index_sha256', 'target_config_sha256', 'target_sources')}
    if not report.get('parameter_sha256') or not report.get('sources') or not report.get('native_sources'):
        raise ValueError('Loaded target, learned drafter and native recipe evidence required')
    sources = fingerprints()
    identity = PrefixIdentity(checksum(target), checksum(report['parameter_sha256']),
        checksum(dict(scripts=report['sources'], native=report['native_sources'], cache=sources)), uuid4().hex, 0)
    capacity = report['target_capacity']
    if (pages.shape != (1, capacity['page_count']) or capacity['block_size'] != 64
            or pages[0].tolist() != list(range(capacity['page_count']))
            or capacity['cache_blocks'] < capacity['page_count']):
        raise ValueError('Exclusive contiguous offline target page allocation required')
    original_measure = full_dspark_request.measure_dspark_request
    original_finish = frozen_ladder_requests.finish
    measure_calls, finished = [], []

    def finish(result, summarize):
        original_finish(result, lambda requests: summarize(requests, prefix_cached=True))
        summary = result['request_summary']
        result.update(prefix_cache_experiment=True,
            effective_cached_pp=summary['effective_cached_pp'], suffix_request_pp=summary['suffix_request_pp'],
            comparison_axis='Same combined T16 recipe with explicit 2048-token prefix reuse',
            scope='Combined cached-prefill qualification; cold-control output/state audits and complete-cycle TG')
        finished.append(True)

    progress('allocate_prefix_checkpoint_before_native_warmup')
    try:
        with offline_prefix_session(operations, generator, model, identity, pages,
                prefix_position=2048, reserved_pages=list(range(capacity['page_count'])),
                inactive_pages=list(range(capacity['page_count'], capacity['cache_blocks']))) as factory:
            def measure(*args, **kwargs):
                if kwargs.get('cached_prefill_factory') is not None or kwargs.get('t32'):
                    raise ValueError('One T16 prefix factory must own the request')
                measure_calls.append(kwargs.get('audit_features'))
                return original_measure(*args, **kwargs, cached_prefill_factory=factory)

            with patch.object(full_dspark_request, 'measure_dspark_request', measure), \
                    patch.object(frozen_ladder_requests, 'finish', finish):
                dspark_request_experiment.run_loaded_requests(operations, generator, model, collectives,
                    tokenizer, pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary,
                    report, progress, **options)
        if measure_calls != [True, False, False] or finished != [True]:
            raise ValueError('One complete combined audit and two timed cached requests required')
    finally:
        after = fingerprints()
        report['prefix_cache_sources'] = sources
        report['prefix_cache_sources_after'] = after
        report['prefix_cache_closed'] = not hasattr(model, '_qwen_prefix_session')
        if sources != after:
            raise ValueError('Prefix experiment source changed during execution')
