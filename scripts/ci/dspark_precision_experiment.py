"""One loaded winning T16 runtime with independent HiFi4/HiFi2 proposal audits."""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from dspark_precision_comparison import SCHEDULE, summarize
from dspark_precision_device import device_type
from dspark_precision_gate import qualify


SOURCES = ('dspark_precision_experiment.py', 'dspark_precision_comparison.py',
    'dspark_precision_device.py', 'dspark_precision_gate.py', 'dspark_layer_precision.py',
    'dspark_projection_precision.py', 'dspark_projection_precision_report.py',
    'dspark-precision-reviewed.json')


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import dspark_request_experiment
    import frozen_ladder_requests
    import full_dspark_request
    import dspark_prepared_proposal
    import dspark_native_cached_layer
    from gdn_shared_qk_variants import validate_route

    if (any(os.environ.get(name) != '1' for name in ('QWEN_DSPARK_PROJECTION_HIFI2',
            'QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS'))
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(options.get('prompt', ())) != 4096 or report.get('streams') != 1
            or options.get('captured_publication') is not True):
        raise ValueError('Explicit allocated unprofiled winning 4K precision comparison required')
    directory = Path(__file__).parent
    reviewed = json.loads((directory / 'dspark-precision-reviewed.json').read_bytes())
    evidence = qualify(directory, directory / 'precision-evidence', reviewed_reports=reviewed)
    candidate_type = device_type(dspark_prepared_proposal.TracedDSparkDevice, dspark_native_cached_layer.execute)
    original_measure = full_dspark_request.measure_dspark_request

    def fingerprints():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCES}

    sources = fingerprints()
    calls, finished = [], []

    def measure(*args, **kwargs):
        if len(calls) >= len(SCHEDULE):
            raise ValueError('Unexpected additional precision request')
        precision, audit = SCHEDULE[len(calls)]
        if (kwargs.get('audit_features') is not audit
                or any(kwargs.get(name) is not True for name in ('gdn_shared_qk', 'fused_t16_mlp',
                    'captured_publication', 'target_attention_t16', 'commit_only_gdn',
                    'proposal_trace', 'native_attention', 'score_layout'))
                or any(kwargs.get(name) for name in ('t32', 'banked_proposal', 'profile_verifier', 'profile_drafter'))):
            raise ValueError('Both precision arms must retain the complete winning T16 runtime')
        calls.append((precision, audit))
        instances = []

        def construct(*arguments, **keywords):
            device = candidate_type(*arguments, **keywords)
            instances.append(device)
            return device

        try:
            with (patch.object(dspark_prepared_proposal, 'TracedDSparkDevice', construct)
                    if precision == 'hifi2' else nullcontext()):
                result = original_measure(*args, **kwargs)
            if precision == 'hifi2' and (len(instances) != 1 or not instances[0].closed
                    or instances[0].prepared is None or not instances[0].prepared.closed
                    or instances[0].precision_layer_calls <= 0 or instances[0].precision_layer_calls % 5):
                raise ValueError('One completely executed and closed five-layer precision device required')
            result.update(drafter_precision=precision, drafter_precision_evidence=dict(
                report_sha256=reviewed if precision == 'hifi2' else None,
                layer_calls=instances[0].precision_layer_calls if instances else 0,
                closed=True))
            return result
        finally:
            for device in instances:
                if not device.closed:
                    device.close()

    def finish(result, summarize_requests):
        comparison = summarize(result['request_checks'], summarize_requests,
            frozen_ladder_requests.validate_audit, validate_route)
        result.update(drafter_precision_comparison=comparison, pp=None, committed_tg=None,
            ctx_tokens=4096, fresh_context_audit=True,
            scope='Same winning T16 target; HiFi4 versus HiFi2 drafter projection matmuls only')
        finished.append(True)

    try:
        with patch.object(full_dspark_request, 'measure_dspark_request', measure), \
                patch.object(frozen_ladder_requests, 'finish', finish):
            dspark_request_experiment.run_loaded_requests(operations, generator, model, collectives, tokenizer,
                pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary, report, progress, **options)
        if calls != list(SCHEDULE) or finished != [True]:
            raise ValueError('Two independent audits and complete ABBA timing required')
    finally:
        report['drafter_precision_sources'] = sources
        report['drafter_precision_sources_after'] = fingerprints()
        if (report['drafter_precision_sources_after'] != sources
                or qualify(directory, directory / 'precision-evidence', reviewed_reports=reviewed) != evidence):
            raise ValueError('Precision source or simulator evidence changed during execution')
