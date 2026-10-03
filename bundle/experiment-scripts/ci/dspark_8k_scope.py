"""Request-scoped 8K adapter preserving the simulator-pinned implementation files."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from frozen_context_geometry import selected_geometry, geometry
from unittest.mock import patch

from dspark_8k_admission import admitted_request, history_limit
from frozen_combined_runtime import REPORT_SHA256


def stable_history_class(original):
    from dspark_history import FullHistoryKV, TensorScope, leaves

    class EightKHistory(original):
        def __init__(self, operations, mesh, collectives, parameters, layer_weights, chunks, rotary, *, position, capacity):
            if history_limit() != selected_geometry()['capacity'] or type(position) is not int or position != selected_geometry()['context'] or type(capacity) is not int or capacity != selected_geometry()['capacity']:
                raise ValueError('Admitted exact selected-context fixed history required')
            from frozen_combined_history import initialise_history
            initialise_history(self, operations, mesh, collectives, parameters,
                layer_weights, chunks, rotary, position=position)
            self.capacity, self.spare_layers = capacity, ()
            previous = self.layers
            scope = TensorScope(operations, leaves(previous))
            try:
                active = tuple(tuple(scope.retain(operations.pad(value,
                    [(0, 0), (0, 0), (0, capacity - position), (0, 0)], 0.0))
                    for value in pair) for pair in previous)
                spare = tuple(tuple(scope.retain(operations.clone(value,
                    memory_config=operations.DRAM_MEMORY_CONFIG)) for value in pair) for pair in active)
                operations.synchronize_device(mesh)
                self.layers, self.spare_layers = active, spare
            except BaseException:
                scope.release()
                FullHistoryKV.close(self)
                raise
            scope.release(keep=leaves(active) + leaves(spare))
            self.release_layers(previous)

    return EightKHistory


@contextmanager
def runtime_scope(directory, *, context, output_tokens, factory_root, build_evidence):
    import dspark_full_attention
    import dspark_native_cached_layer
    import dspark_native_fixed_gate
    import dspark_stable_history
    import native_draft_sdpa
    from dspark_attention_chunk_trial import execute
    from dspark_stats_pack import SELECTOR_ASSERT, scoped_stats_pack

    with ExitStack() as stack:
        evidence = stack.enter_context(admitted_request(directory, context=context,
            output_tokens=output_tokens, factory_root=factory_root, build_evidence=build_evidence))
        import dspark_prefill
        import full_dspark_request
        from frozen_combined_history import prefill_capture_class
        capture = prefill_capture_class(dspark_prefill.FullHistoryCapture)
        stack.enter_context(patch.object(dspark_prefill, 'FullHistoryCapture', capture))
        stack.enter_context(patch.object(full_dspark_request, 'FullHistoryCapture', capture))
        stack.enter_context(patch.object(dspark_full_attention, 'MAX_CONTEXT', geometry(context)['capacity']))
        stack.enter_context(patch.object(dspark_stable_history, 'StableHistoryKV',
            stable_history_class(dspark_stable_history.StableHistoryKV)))
        stack.enter_context(patch.object(dspark_native_cached_layer, 'attend', execute))
        def admitted_qualification(requested):
            if Path(requested).resolve() != Path(directory).resolve():
                raise ValueError('Only the admitted script directory is qualified')
            return {'dspark-native-8k-attention.json': REPORT_SHA256}

        stack.enter_context(patch.object(dspark_native_fixed_gate, 'qualify', admitted_qualification))
        from dspark_ladder_scalar_reciprocal import scalar_reciprocal
        stack.enter_context(scalar_reciprocal())
        from frozen_reciprocal_isolation import isolated_reciprocal
        stack.enter_context(isolated_reciprocal())
        stack.enter_context(scoped_stats_pack())
        qualified_replacements = native_draft_sdpa.replacements

        def hardware_replacements():
            replacements = qualified_replacements()
            before, after = replacements['sdpa.cpp'][0]
            if after.count(SELECTOR_ASSERT) != 1:
                raise ValueError('Qualified simulator selector assertion required')
            assertion = f'''static_assert(QWEN_DRAFT_EXP_APPROX ||
    (get_compile_time_arg_val(3) == {geometry(context)['padded_keys'] // 32} && get_compile_time_arg_val(8) == 8),
    "Admitted draft requires qualified 256-key geometry");
'''
            replacements['sdpa.cpp'] = ((before, after.replace(SELECTOR_ASSERT, assertion)),)
            return replacements

        stack.enter_context(patch.object(native_draft_sdpa, 'replacements', hardware_replacements))
        from frozen_draft_tail_scope import runtime_scope as incremental_scope
        stack.enter_context(incremental_scope(directory))
        yield evidence
