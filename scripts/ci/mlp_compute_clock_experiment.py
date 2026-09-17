"""One fully audited winning T16 request with bounded per-layer diagnostics."""

import hashlib
import json
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from mlp_compute_clock_capture import ComputeClockCapture
from mlp_compute_clock_combined import CombinedComputeClockCapture
from mlp_compute_clock_hardware import retained
from mlp_compute_clock_hardware import SIMULATOR_SHA256
from mlp_compute_clock_projection import instrument_projection
from mlp_compute_clock_report import validate


HARDWARE_SHA256 = 'f95a0a568250b00e338accfc0904e3f2e98646de52e76e7b0d3b6348db143634'
HELPERS = ('mlp_compute_clock_capture.py', 'mlp_compute_clock_projection.py', 'mlp_compute_clock.py', 'mlp_clock_samples.py',
    'frozen_mlp_wait_zones.py', 'frozen_recipe_context.py')


def candidate_class(directory, evidence):
    directory, evidence = Path(directory), Path(evidence)
    simulator = retained(evidence / 'simulator')
    raw = (evidence / 'fused-batch.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != HARDWARE_SHA256:
        raise ValueError('Exact qualified hardware diagnostic required')
    hardware = json.loads(raw)
    validate(hardware, backend='hardware')
    if ((evidence / 'fused-batch.exit-status').read_text().strip() != '0'
            or hardware['kernels'] != simulator['kernels']):
        raise ValueError('Clean hardware exit and simulator-identical kernels required')
    expected = json.loads((evidence / 'compute-clock-hardware-sources.json').read_text())
    for name in HELPERS:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected[name]:
            raise ValueError('Qualified diagnostic helper changed: ' + name)
    for name, digest in hardware['kernels'][0]['reader_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Qualified original reader changed: ' + name)
    source = instrument_projection((directory / 'fused_1d.py').read_text())
    if hashlib.sha256(source.encode()).hexdigest() != expected['fused_1d.py']:
        raise ValueError('Projection differs from qualified hardware diagnostic')
    module = ModuleType('qualified_mlp_compute_clock_candidate')
    module.__file__ = str(directory / 'fused_1d.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    module.FusedProjection.diagnostic_evidence = dict(passed=True,
        report_sha256=SIMULATOR_SHA256, kernels=simulator['kernels'])
    return module.FusedProjection


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    import dspark_request_experiment
    import frozen_ladder_requests
    import full_dspark_request
    import fused_t16_scope
    from verifier_engine import VerifierEngine
    from gdn_multitoken_conv import release_owned

    if (os.environ.get('QWEN_MLP_COMPUTE_CLOCK_COMBINED') != '1'
            or os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_DEVICE_PROFILER')
            or len(options.get('prompt', ())) != 4096 or report.get('streams') != 1
            or options.get('captured_publication') is not True):
        raise ValueError('Explicit allocated unprofiled 4K combined diagnostic required')
    directory = Path(__file__).parent
    candidate = candidate_class(directory, directory / 'compute-clock-evidence')
    source_names = (*HELPERS, 'mlp_compute_clock_experiment.py', 'mlp_compute_clock_combined.py', 'mlp_compute_clock_hardware.py',
        'mlp_compute_clock_report.py', 'mlp_clock_report.py', 'fused_1d.py', 'fused_1d_input.cpp', 'fused_1d_weights.cpp')
    def fingerprints():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in source_names}
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
            scope='Bounded combined verifier attribution; no performance qualification',
            mlp_compute_clock_combined=bank.summary())
        finished.append(True)

    try:
        progress('allocate_all_layer_clock_pages_before_request_warmup')
        captures = [ComputeClockCapture(operations, model.mesh_device, owned) for unused in range(64)]
        bank = CombinedComputeClockCapture(model, captures, candidate)
        def measure(*args, **kwargs):
            required = ('audit_features', 'fused_t16_mlp', 'gdn_shared_qk', 'captured_publication',
                'target_attention_t16', 'commit_only_gdn', 'proposal_trace', 'native_attention')
            if calls or any(kwargs.get(name) is not True for name in required):
                raise ValueError('Exactly one complete winning T16 audit required')
            calls.append(True)
            with bank.install(fused_t16_scope, VerifierEngine):
                result = original_measure(*args, **kwargs)
            if result.get('instrumented_timing') is not True or result.get('committed_tokens_per_second') is not None:
                raise ValueError('Diagnostic must not publish throughput')
            result['mlp_compute_clock_combined'] = bank.summary()
            return result
        with patch.object(full_dspark_request, 'measure_dspark_request', measure), \
                patch.object(frozen_ladder_requests, 'finish', finish):
            dspark_request_experiment.run_loaded_requests(operations, generator, model, collectives, tokenizer,
                pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary, report, progress, **options)
        if calls != [True] or finished != [True]:
            raise ValueError('Combined diagnostic did not finish exactly once')
    finally:
        if bank is not None:
            try:
                bank.assert_releasable()
            except BaseException:
                model._qwen_clock_failed_owned = owned
                raise
        release_owned(operations, owned)
        report['mlp_compute_clock_sources'] = sources
        report['mlp_compute_clock_sources_after'] = fingerprints()
        if report['mlp_compute_clock_sources_after'] != sources:
            raise ValueError('Diagnostic source changed during combined execution')
