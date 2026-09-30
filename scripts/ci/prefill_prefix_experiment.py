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


def target_capacity(model, pages, report):
    pairs = model._paged_kv_caches
    if len(pairs) != 16 or any(len(pair) != 2 for pair in pairs):
        raise ValueError('All native attention cache allocations required')
    shapes = [tuple(value.shape) for pair in pairs for value in pair]
    if (any(len(shape) != 4 or shape[2] != 64 or any(dimension <= 0 for dimension in shape) for shape in shapes)
            or len({shape[0] for shape in shapes}) != 1 or len(pages.shape) != 2 or pages.shape[0] != 1):
        raise ValueError('Consistent allocated KV blocks and one native page table required')
    capacity = dict(page_count=pages.shape[1], cache_blocks=shapes[0][0], block_size=64)
    if (not 0 < capacity['page_count'] <= capacity['cache_blocks']
            or pages[0].tolist() != list(range(capacity['page_count']))):
        raise ValueError('Exclusive contiguous offline target page allocation required')
    reported = report.get('target_capacity')
    if reported is not None and any(reported.get(key) != value for key, value in capacity.items()):
        raise ValueError('Reported target capacity differs from allocated tensors')
    return capacity


def validate_changed_suffix_audit(requests):
    if len(requests) != 3:
        raise ValueError('One changed-suffix audit and two timed requests required')
    records = requests[0].get('dspark', {}).get('prefix_cache', [])
    if len(records) != 2:
        raise ValueError('Cold-control and changed-suffix candidate records required')
    priming = records[1].get('changed_suffix_priming', {})
    boundary = priming.get('checkpoint_boundary', {})
    if (priming.get('changed_suffix') is not True or priming.get('prefix_tokens') != 2048
            or priming.get('prompt_tokens') != 4096 or boundary.get('position') != 2048
            or any(boundary.get(key) is not True for key in ('captured', 'complete', 'restored'))
            or any('changed_suffix_priming' in value for request in requests[1:]
                for value in request.get('dspark', {}).get('prefix_cache', []))):
        raise ValueError('Changed-suffix priming must be complete and confined to the untimed audit')


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
    capacity = target_capacity(model, pages, report)
    report['prefix_cache_target_capacity'] = capacity
    original_measure = full_dspark_request.measure_dspark_request
    original_finish = frozen_ladder_requests.finish
    measure_calls, finished = [], []

    def finish(result, summarize):
        validate_changed_suffix_audit(result['request_checks'])
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
                def audited_factory(tokens, prefill, ordinal):
                    prime = list(tokens)
                    prime[-1] = 0 if prime[-1] != 0 else 1
                    return factory(tokens, prefill, ordinal, prime_tokens=prime if ordinal == 1 else None)
                selected_factory = audited_factory if kwargs.get('audit_features') is True else factory
                return original_measure(*args, **kwargs, cached_prefill_factory=selected_factory)

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
