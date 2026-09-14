"""Offline request-scoped 64K history and qualified draft attention bindings."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from dspark_64k_admission import admitted_request, current_admission
from dspark_attention_64k_gate import REPORT_SHA256
import dspark_history as history


def require_scope():
    evidence = current_admission()
    if evidence is None or evidence.get('context') != 65536 or evidence.get('capacity') != 66560:
        raise ValueError('Active exact 64K request admission required')
    return evidence


def target_allocation():
    evidence = require_scope()
    block_size = 64
    capacity = evidence['capacity']
    if capacity % block_size or evidence.get('output_tokens') != 256:
        raise ValueError('Tile-aligned admitted target capacity and output budget required')
    pages = capacity // block_size
    return dict(max_seq_len=capacity, page_count=pages, cache_blocks=pages + 8, block_size=block_size)


def validate_target_kv_prefix(valid, caches):
    allocation = target_allocation()
    if type(valid) is not int or not 1 <= valid <= allocation['max_seq_len']:
        raise ValueError('Explicit valid admitted target KV prefix required')
    pages = (valid + allocation['block_size'] - 1) // allocation['block_size']
    if not caches or any(len(value.shape) != 4 or value.shape[0] < pages
            or value.shape[2] != allocation['block_size'] for value in caches):
        raise ValueError('Target KV storage must cover every audited page')


def validate_ordered_cache_shapes(cache, packed, positions, pages):
    allocation = target_allocation()
    rows = packed[1] if len(packed) == 4 else 0
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32) or tuple(packed) != (1, rows, 32, 256):
        raise ValueError('Native prepared T=1/2/4/8/16/32 KV tiles required')
    if len(cache) != 4 or cache[0] < 1 or tuple(cache[1:]) != (2, 64, 256):
        raise ValueError('Expected two-head 64-row BF8 paged cache')
    if (tuple(positions) != (rows,) or len(pages) != 2 or pages[0] != rows
            or pages[1] != allocation['page_count'] or pages[1] > cache[0]):
        raise ValueError('Paired positions and admitted 1040-page table required')
    return rows


def stable_history_class(original):
    class SixtyFourKHistory(original):
        def __init__(self, operations, mesh, collectives, parameters, layer_weights, chunks, rotary, *, position, capacity):
            require_scope()
            if type(position) is not int or position != 65536 or type(capacity) is not int or capacity != 66560:
                raise ValueError('Exact 65536/66560 full-history allocation required')
            self.operations, self.mesh, self.collectives = operations, mesh, collectives
            self.parameters, self.layer_weights, self.rotary = parameters, layer_weights, rotary
            self.position, self.pending, self.closed = position, None, False
            self.capacity, self.spare_layers, self.layers = capacity, (), ()
            self.layers = history.project_chunks(operations, mesh, collectives, parameters,
                layer_weights, chunks, rotary, start=0, rows=position)
            previous = self.layers
            scope = history.TensorScope(operations, history.leaves(previous))
            try:
                active = tuple(tuple(scope.retain(operations.pad(value,
                    [(0, 0), (0, 0), (0, capacity - position), (0, 0)], 0.0))
                    for value in pair) for pair in previous)
                spare = tuple(tuple(scope.retain(operations.clone(value,
                    memory_config=operations.DRAM_MEMORY_CONFIG)) for value in pair) for pair in active)
                operations.synchronize_device(mesh)
                self.layers, self.spare_layers = active, spare
            except BaseException:
                try:
                    scope.release()
                finally:
                    history.FullHistoryKV.close(self)
                raise
            scope.release(keep=history.leaves(active) + history.leaves(spare))
            self.release_layers(previous)

        def check_prefix(self, prefix, position):
            require_scope()
            return super().check_prefix(prefix, position)

    return SixtyFourKHistory


def prefill_capture_class(original):
    class SixtyFourKCapture(original):
        def __init__(self, operations, model, position):
            require_scope()
            if type(position) is not int or position != 65536:
                raise ValueError('Exact full 64K prefill capture required')
            super().__init__(operations, model, 8192)
            self.position = position

    return SixtyFourKCapture


@contextmanager
def runtime_scope(directory, report_path, *, context, output_tokens, factory_root, build_path):
    import dspark_full_attention
    import dspark_native_cached_layer
    import dspark_native_fixed_gate
    import dspark_prefill
    import dspark_stable_history
    import native_draft_sdpa
    import ordered_cache
    import full_dspark_request
    from dspark_ladder_attention import adapter
    from dspark_ladder_factory import scoped_stats_pack, selector_assert
    from dspark_ladder_normalization import scratch_normalization
    from dspark_ladder_scalar_reciprocal import scalar_reciprocal
    from dspark_ladder_score_center import scalar_score_center
    from dspark_ladder_stage_print import stage_snapshots
    from dspark_ladder_sum_update import scalar_sum_update

    with ExitStack() as stack:
        evidence = stack.enter_context(admitted_request(directory, report_path, context=context,
            output_tokens=output_tokens, factory_root=factory_root, build_path=build_path))
        stack.enter_context(patch.object(ordered_cache, 'validate_shapes', validate_ordered_cache_shapes))
        stack.enter_context(patch.object(dspark_full_attention, 'MAX_CONTEXT', 66560))
        stack.enter_context(patch.object(dspark_stable_history, 'StableHistoryKV',
            stable_history_class(dspark_stable_history.StableHistoryKV)))
        stack.enter_context(patch.object(dspark_prefill, 'FullHistoryCapture',
            prefill_capture_class(dspark_prefill.FullHistoryCapture)))
        stack.enter_context(patch.object(full_dspark_request, 'FullHistoryCapture', dspark_prefill.FullHistoryCapture))
        stack.enter_context(patch.object(full_dspark_request, 'history_limit', lambda: require_scope()['capacity']))
        stack.enter_context(patch.object(dspark_native_cached_layer, 'attend', adapter(65536)))

        def admitted_qualification(requested):
            require_scope()
            if Path(requested).resolve() != Path(directory).resolve():
                raise ValueError('Only the admitted script directory is qualified')
            return {'dspark-ladder-hardware-65536.json': REPORT_SHA256}

        stack.enter_context(patch.object(dspark_native_fixed_gate, 'qualify', admitted_qualification))
        for scope in (scalar_reciprocal(), scalar_sum_update(), scalar_score_center(),
                stage_snapshots(row=3, column=0), scratch_normalization(), scoped_stats_pack()):
            stack.enter_context(scope)
        qualified = native_draft_sdpa.replacements

        def replacements():
            result = qualified()
            result['compute_common.hpp'] += (('#pragma once', '''#pragma once
#ifndef QWEN_DRAFT_EXP_APPROX
#define QWEN_DRAFT_EXP_APPROX true
#endif'''),)
            before, after = result['sdpa.cpp'][0]
            assertion = selector_assert()
            if after.count(assertion) != 1:
                raise ValueError('Exact qualified ladder selector required')
            runtime_assertion = '''static_assert(QWEN_DRAFT_EXP_APPROX ||
    (get_compile_time_arg_val(3) == 2112 && get_compile_time_arg_val(8) == 32),
    "Admitted 64K draft requires qualified geometry");
'''
            result['sdpa.cpp'] = ((before, after.replace(assertion, runtime_assertion)),)
            return result

        stack.enter_context(patch.object(native_draft_sdpa, 'replacements', replacements))
        yield evidence
