"""Opt-in combined T32 target and cached native proposals; not serving promotion."""

from contextlib import ExitStack
import importlib
from pathlib import Path
from unittest.mock import patch

from dflash_t32_native_scope import scoped_native_t32, require_active
from gdn_shared_qk_t32_gate import qualify as qualify_gdn, REPORT_SHA256 as GDN_SHA256
from gdn_shared_qk_t32_scope import scoped_shared_qk_t32
from gdn_direct_window_t32_gate import REPORT_SHA256 as WINDOW_SHA256
from gdn_direct_window_t32_scope import scoped_direct_windows_t32
from mlp_down_grid_t32_gate import qualify as qualify_down, REPORT_SHA256 as DOWN_SHA256
from mlp_down_grid_t32_scope import scoped_down_grid_t32
from mlp_block_stream_t32_gate import REPORT_SHA256 as STREAM_SHA256
from mlp_block_stream_runtime import scoped_block_stream
from mlp_weight_pipeline_report import validate_fusion


def validate_request(request):
    if (request.get('selected_drafter') != 'dflash2' or request.get('dflash', {}).get('block_rows') != 32
            or any(request.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact'))):
        raise ValueError('Complete exact single-request T32 DFlash target audit required')
    validate_fusion(request, STREAM_SHA256, expected_extra_weight_allocations=64,
        expected_rows=32, fusion_key='fused_t32_mlp')
    hits = request['fused_t32_mlp']['hits']
    stream = request.get('block_stream', {})
    if (stream.get('report_sha256') != STREAM_SHA256 or stream.get('rows') != 32
            or stream.get('restored') is not True or stream.get('constructions') != 64
            or stream.get('stream_allocations') != 64 or stream.get('calls') != sum(hits)
            or sorted(stream.get('constructed_layers', [])) != list(range(64))
            or stream.get('serving_defaults_changed') is not False):
        raise ValueError('All 64 qualified T32 stream projections must execute and restore')
    down = request.get('mlp_down_grid', {})
    if (down.get('report_sha256') != DOWN_SHA256 or down.get('rows') != 32
            or down.get('restored') is not True or down.get('hits') != hits):
        raise ValueError('Every fused T32 MLP must use its qualified down grid')
    shared, windows = request.get('gdn_shared_qk', {}), request.get('gdn_direct_window', {})
    loads = shared.get('loads', [])
    if (shared.get('admission', {}).get('report_sha256') != GDN_SHA256
            or shared.get('restored') is not True or shared.get('released') is not True
            or not loads or len(loads) % 48
            or any(load != dict(rows=32, programs=3, retained_preparation_buffers=2) for load in loads)
            or windows.get('admission', {}).get('report_sha256') != WINDOW_SHA256
            or windows.get('restored') is not True or windows.get('rows') != 32
            or windows.get('hits') != len(loads)):
        raise ValueError('All 48 GDN layers require matched T32 windows, recurrence and scatter')


def measure_combined_t32(operations, model, sampler, prompt, pages, helpers, *, directory,
                        runtime_root, evidence, streams, **options):
    from full_dflash_request import measure_dflash_request
    from fused_t16_scope import FusedT32Arm
    from models.tt_transformers.tt.ccl import tt_all_reduce
    import gdn_shared_qk_t32_adapter

    if len(prompt) != 4096 or options.get('max_new_tokens') != 256:
        raise ValueError('Initial combined T32 comparison requires CTX4096 and a 256-output budget')
    directory = Path(directory)
    gdn = qualify_gdn(evidence['gdn'], directory, runtime_root)
    down = qualify_down(evidence['down'], directory, runtime_root)
    builder = importlib.import_module('shared_qk_norm_t32_scatter')
    for name in ('shared_qk_norm_t32_scatter', 'gdn_shared_qk_t32_pipeline', 'gdn_shared_qk_t32_program'):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (directory / (name + '.py')).resolve():
            raise ValueError('T32 recurrence builder loaded from another checkout')
    with ExitStack() as stack:
        stack.enter_context(scoped_native_t32(evidence['attention'], evidence['cache'], directory, runtime_root))
        stack.enter_context(patch.object(gdn_shared_qk_t32_adapter, 'require_simulator', require_active))
        window_audit = stack.enter_context(scoped_direct_windows_t32(evidence['windows'], directory, runtime_root))
        down_audit = stack.enter_context(scoped_down_grid_t32(down))
        stream_audit = stack.enter_context(scoped_block_stream(directory, evidence['stream'],
            runtime_root=runtime_root, operations=operations,
            weights=[layer.feed_forward.weights.w_gate_up for layer in model.layers], streams=streams, token_rows=32))
        shared_audit = stack.enter_context(scoped_shared_qk_t32(operations, gdn, builder=builder.build))
        fusion = FusedT32Arm(operations, model, tt_all_reduce)
        stack.enter_context(fusion.install())
        result = measure_dflash_request(operations, model, sampler, prompt, pages, helpers,
            block_rows=32, proposal_capture=True, commit_only_gdn=True, fused_convolution=True,
            cache_history=True, native_proposal_attention=True, target_combined_t32=True, **options)
    result.update(fused_t32_mlp=fusion.audit, block_stream=stream_audit,
        gdn_shared_qk=shared_audit, gdn_direct_window=window_audit, mlp_down_grid=down_audit)
    result['fused_t32_mlp']['extra_weight_allocations'] = 64
    validate_request(result)
    if (qualify_gdn(evidence['gdn'], directory, runtime_root) != gdn
            or qualify_down(evidence['down'], directory, runtime_root) != down):
        raise ValueError('T32 target component admission changed during request')
    return result
