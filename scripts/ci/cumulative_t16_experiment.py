"""Native versus direct-window plus compact-score complete T16 requests."""

from contextlib import contextmanager, nullcontext, ExitStack
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import gdn_direct_window_experiment as direct_experiment
from compact_score_gate import qualify, REPORT_SHA256
from cumulative_t16_scope import scoped_cumulative_t16, validate_request
from gdn_direct_window_comparison import repeatability
from mlp_down_grid_gate import qualify as qualify_down, REPORT_SHA256 as DOWN_SHA256


COMPACT_FILES = ('compact_score_gate.py', 'compact_score_report.py', 'compact_score_hardware_sources.py',
    'compact_score_scope.py', 'compact_score_device.py', 'compact_markov.py', 'compact-score-probe.py',
    'compact_score_hardware_device.py', 'compact_score_hardware_markov.py',
    'compact_score_io.cpp', 'compact_score_compute.cpp', 'compact_score_reduce.cpp')
DOWN_FILES = ('mlp_down_grid_gate.py', 'mlp_down_grid_scope.py', 'mlp_down_grid.py', 'mlp-down-grid-probe.py')
NORM_FILES = ('cumulative_norm_runtime.py', 'cumulative_norm_validation.py',
    'shared_qk_norm_scatter_gate.py', 'shared_qk_norm_scatter.py', 'gdn_norm_scatter.py')
REGISTER_FILES = ('cumulative_register_runtime.py', 'cumulative_register_scope.py',
    'cumulative_fusion_validation.py', 'mlp_register_epilogue_gate.py', 'mlp_register_epilogue.py',
    'mlp_rounding_policy.py', 'mlp_weight_pipeline_gate.py', 'mlp_weight_pipeline.py',
    'mlp_weight_pipeline_report.py', 'mlp_weight_pipeline_comparison.py')
REGISTER_PAYLOADS = tuple('mlp-register-epilogue-candidate/' + name
    for name in ('fused_1d.py', 'fused_1d_input.cpp', 'fused_1d_weights.cpp'))
FILES = tuple(dict.fromkeys(direct_experiment.FILES + COMPACT_FILES + DOWN_FILES + NORM_FILES + REGISTER_FILES +
    ('cumulative_t16_experiment.py', 'cumulative_t16_scope.py')))


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, **options):
    if any(os.environ.get(name) != '1' for name in ('QWEN_CUMULATIVE_T16', 'QWEN_COMPACT_SCORE_HARDWARE')):
        raise ValueError('Explicit cumulative T16 and compact-score opt-ins required')
    directory = Path(__file__).parent
    evidence = qualify(directory, directory / 'compact-score-evidence')
    down_flag = os.environ.get('QWEN_CUMULATIVE_MLP_DOWN', '0')
    if down_flag not in ('0', '1'):
        raise ValueError('Explicit zero/one cumulative MLP-down policy required')
    down = qualify_down(directory / 'mlp-down-grid-evidence', directory, '/opt/tt-metal') if down_flag == '1' else None
    norm_flag = os.environ.get('QWEN_CUMULATIVE_NORM', '0')
    if norm_flag not in ('0', '1'):
        raise ValueError('Explicit zero/one cumulative normalization policy required')
    if norm_flag == '1':
        from cumulative_norm_runtime import require_active, select_norm
        require_active()
    components = ['direct_windows', 'compact_scores'] + (['wider_mlp_down'] if down is not None else [])
    if norm_flag == '1':
        components.append('norm_scatter')
    register_flag = os.environ.get('QWEN_CUMULATIVE_REGISTER', '0')
    if register_flag not in ('0', '1'):
        raise ValueError('Explicit zero/one cumulative register policy required')
    if register_flag == '1':
        components.append('register_epilogue')

    def fingerprints():
        names = FILES + (REGISTER_PAYLOADS if register_flag == '1' else ())
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}

    sources, audits = fingerprints(), []
    register_candidate = None

    @contextmanager
    def combined(direct_admission, source_directory):
        if Path(source_directory).resolve() != directory.resolve():
            raise ValueError('Both components must come from the same staged runtime')
        with (select_norm('scatter') if norm_flag == '1' else nullcontext()), \
                scoped_cumulative_t16(direct_admission, evidence, directory,
                **(dict(down_admission=down) if down is not None else {})) as audit, \
                (register_candidate() if register_flag == '1' else nullcontext()) as register_audit:
            if register_flag == '1':
                audit['register'] = register_audit
            audits.append(audit)
            yield audit['direct']

    try:
        with ExitStack() as stack:
            if register_flag == '1':
                from cumulative_register_runtime import runtime_scope
                register_candidate = stack.enter_context(runtime_scope(directory, runtime_root='/opt/tt-metal'))
            stack.enter_context(patch.object(direct_experiment, 'scoped_direct_windows', combined))
            direct_experiment.run_loaded_requests(operations, generator, model, collectives, tokenizer,
                pages, kv_cache, parameters, layer_weights, predecessor, successor, rotary, report, progress, **options)
        requests = report.get('request_checks', [])
        if len(requests) != 6 or len(audits) != 3:
            raise ValueError('Complete two-audit ABBA cumulative schedule required')
        pending = iter(audits)
        for request, (enabled, audited) in zip(requests, direct_experiment.SCHEDULE, strict=True):
            if (request.get('gdn_direct_window', {}).get('direct') is not enabled
                    or request.get('instrumented_timing') is not audited):
                raise ValueError('Cumulative schedule identity changed')
            audit = next(pending) if enabled else None
            if enabled:
                validate_request(request, audit)
            if register_flag == '1':
                from cumulative_fusion_validation import validate_fusion_policy
                validate_fusion_policy(request, 'register' if enabled else 'baseline')
            if norm_flag == '1':
                from cumulative_norm_validation import validate_norm_history
                validate_norm_history(request, 'scatter' if enabled else 'prefetch')
            request['compact_score'] = dict(compact=enabled,
                hits=audit['compact']['calls'] if enabled else 0,
                report_sha256=REPORT_SHA256 if enabled else None, restored=True)
            if down is not None:
                request['mlp_down_grid'] = dict(wider_down=enabled,
                    hits=list(audit['down']['hits']) if enabled else [0] * 64,
                    report_sha256=DOWN_SHA256 if enabled else None, restored=True)
        report.update(cumulative_t16=True, cumulative_components=components,
            cumulative_route_diagnostics=audits,
            cumulative_measurement_quality=repeatability(report['gdn_direct_window_comparison']),
            scope='Native retained T16 versus cumulative components: ' + ', '.join(components))
    finally:
        report['cumulative_sources'] = sources
        report['cumulative_sources_after'] = fingerprints()
        if report['cumulative_sources_after'] != sources or qualify(directory, directory / 'compact-score-evidence') != evidence:
            raise ValueError('Cumulative sources or compact admission changed')
        if down is not None and qualify_down(directory / 'mlp-down-grid-evidence', directory, '/opt/tt-metal') != down:
            raise ValueError('Cumulative down-grid admission changed')
