"""Request-scoped 8K adapter preserving the simulator-pinned implementation files."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from dspark_8k_admission import admitted_request, history_limit
from dspark_attention_8k_gate import REPORT_SHA256


def stable_history_class(original):
    from dspark_history import FullHistoryKV, TensorScope, leaves

    class EightKHistory(original):
        def __init__(self, operations, mesh, collectives, parameters, layer_weights, chunks, rotary, *, position, capacity):
            if history_limit() != 8448 or type(position) is not int or position != 8192 or type(capacity) is not int or capacity != 8448:
                raise ValueError('Admitted exact 8192/8448 fixed history required')
            FullHistoryKV.__init__(self, operations, mesh, collectives, parameters,
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
        stack.enter_context(patch.object(dspark_full_attention, 'MAX_CONTEXT', 8448))
        stack.enter_context(patch.object(dspark_stable_history, 'StableHistoryKV',
            stable_history_class(dspark_stable_history.StableHistoryKV)))
        stack.enter_context(patch.object(dspark_native_cached_layer, 'attend', execute))
        def admitted_qualification(requested):
            if Path(requested).resolve() != Path(directory).resolve():
                raise ValueError('Only the admitted script directory is qualified')
            return {'dspark-native-8k-attention.json': REPORT_SHA256}

        stack.enter_context(patch.object(dspark_native_fixed_gate, 'qualify', admitted_qualification))
        stack.enter_context(scoped_stats_pack())
        qualified_replacements = native_draft_sdpa.replacements

        def hardware_replacements():
            replacements = qualified_replacements()
            before, after = replacements['sdpa.cpp'][0]
            if after.count(SELECTOR_ASSERT) != 1:
                raise ValueError('Qualified simulator selector assertion required')
            assertion = '''static_assert(QWEN_DRAFT_EXP_APPROX ||
    (get_compile_time_arg_val(3) == 272 && get_compile_time_arg_val(8) == 8),
    "Admitted draft requires qualified 256-key geometry");
'''
            replacements['sdpa.cpp'] = ((before, after.replace(SELECTOR_ASSERT, assertion)),)
            return replacements

        stack.enter_context(patch.object(native_draft_sdpa, 'replacements', hardware_replacements))
        yield evidence
