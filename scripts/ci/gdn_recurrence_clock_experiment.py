"""One native-audited combined request with all-layer recurrence phase samples."""

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

from gdn_recurrence_clock_capture import RecurrenceClockCapture
from gdn_recurrence_clock_combined import CombinedRecurrenceCapture
from gdn_recurrence_clock_gate import HELPERS, qualify


FILES = (*HELPERS, 'gdn_recurrence_clock_gate.py', 'gdn_recurrence_clock_report.py',
    'gdn_recurrence_clock_combined.py', 'gdn_recurrence_clock_experiment.py')


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import dspark_request_experiment
    import frozen_ladder_requests
    import full_dspark_request
    import gdn_shared_qk_pipeline
    import gdn_recurrence_clock_pipeline
    from verifier_engine import VerifierEngine
    from gdn_multitoken_conv import release_owned

    if (any(os.environ.get(name) != '1' for name in ('QWEN_GDN_RECURRENCE_CLOCK_COMBINED',
            'QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS'))
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(options.get('prompt', ())) != 4096 or report.get('streams') != 1
            or options.get('captured_publication') is not True):
        raise ValueError('Explicit allocated unprofiled 4K combined diagnostic required')
    directory = Path(__file__).parent
    admission = qualify(directory / 'recurrence-clock-evidence', directory, os.environ['TT_METAL_HOME'])

    def fingerprints():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES}

    sources = fingerprints()
    original_measure = full_dspark_request.measure_dspark_request
    owned, calls, finished = [], [], []
    bank = None

    def finish(result, summarize):
        requests = result['request_checks']
        if len(requests) != 1 or requests[0].get('arm') != 'publication':
            raise ValueError('Exactly one combined publication audit required')
        frozen_ladder_requests.validate_audit(requests[0])
        result.update(pp=None, committed_tg=None, fresh_context_audit=True, ctx_tokens=4096,
            scope='Bounded combined recurrence attribution; no performance qualification',
            gdn_recurrence_clock_combined=bank.summary())
        finished.append(True)

    try:
        progress('allocate_recurrence_clock_pages_before_request_warmup')
        captures = [RecurrenceClockCapture(operations, model.mesh_device, owned) for unused in range(48)]
        bank = CombinedRecurrenceCapture(operations, model, captures, gdn_recurrence_clock_pipeline, admission)

        def measure(*args, **kwargs):
            required = ('audit_features', 'fused_t16_mlp', 'gdn_shared_qk', 'captured_publication',
                'target_attention_t16', 'commit_only_gdn', 'proposal_trace', 'native_attention', 'score_layout')
            if (calls or any(kwargs.get(name) is not True for name in required)
                    or any(kwargs.get(name) for name in ('t32', 'banked_proposal', 'profile_verifier', 'profile_drafter'))):
                raise ValueError('Exactly one complete winning T16 audit required')
            calls.append(True)
            with bank.install(gdn_shared_qk_pipeline, VerifierEngine):
                result = original_measure(*args, **kwargs)
            if (result.get('instrumented_timing') is not True
                    or result.get('committed_tokens_per_second') is not None
                    or result.get('gdn_norm_prefetch', {}).get('enabled') is not True
                    or result['gdn_norm_prefetch']['builds'] != len(bank.builds)):
                raise ValueError('Audited all-layer norm-prefetched diagnostic must not publish throughput')
            result['gdn_recurrence_clock_combined'] = bank.summary()
            report['gdn_recurrence_clock_combined'] = result['gdn_recurrence_clock_combined']
            return result

        with patch.object(full_dspark_request, 'measure_dspark_request', measure), \
                patch.object(frozen_ladder_requests, 'finish', finish):
            dspark_request_experiment.run_loaded_requests(operations, generator, model, collectives, tokenizer,
                pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary, report, progress, **options)
        if calls != [True] or finished != [True]:
            raise ValueError('Combined recurrence diagnostic did not finish exactly once')
    finally:
        if bank is not None:
            try:
                bank.assert_releasable()
            except BaseException:
                model._qwen_recurrence_clock_failed_owned = owned
                raise
        release_owned(operations, owned)
        report['gdn_recurrence_clock_sources'] = sources
        report['gdn_recurrence_clock_sources_after'] = fingerprints()
        if report['gdn_recurrence_clock_sources_after'] != sources:
            raise ValueError('Recurrence diagnostic source changed during execution')
