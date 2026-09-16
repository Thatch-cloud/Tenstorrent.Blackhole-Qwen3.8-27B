"""Matched shared-Q/K requests changing only simulator-qualified MLP FIFO depths."""

from contextlib import contextmanager, nullcontext
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

from frozen_mlp_buffer_gate import REPORT_SHA256, qualify


@contextmanager
def runtime_scope(directory):
    if (os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Allocated admitted 32K hardware required for buffer comparison')
    import full_dspark_request
    import fused_t16_scope
    import dspark_fusion_variants
    import gdn_shared_qk_variants

    directory = Path(directory)
    evidence = qualify(directory, directory / 'frozen-mlp-buffer.json')
    spec = importlib.util.spec_from_file_location('frozen_mlp_buffer_candidate',
        directory / 'frozen_mlp_buffer_candidate.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    measure = full_dspark_request.measure_dspark_request
    original_route = gdn_shared_qk_variants.validate_route

    def measured(*args, **kwargs):
        enabled = kwargs.get('gdn_shared_qk', False)
        if type(enabled) is not bool or not all(kwargs.get(name) is True for name in (
                'captured_publication', 'fused_t16_mlp', 'target_attention_t16', 'score_layout')):
            raise ValueError('Matched complete fused request required')
        kwargs['gdn_shared_qk'] = True
        with (patch.object(fused_t16_scope, 'FusedProjection', candidate.FusedProjection) if enabled else nullcontext()), \
                (patch.object(fused_t16_scope, 'qualify_simulator', lambda: evidence) if enabled else nullcontext()):
            result = measure(*args, **kwargs)
        result['mlp_buffer_blocks'] = 4 if enabled else 2
        return result

    def route(value, arm):
        if arm not in ('control', 'publication') or value.get('mlp_buffer_blocks') != (4 if arm == 'publication' else 2):
            raise ValueError('Matched buffer arm identity required')
        with (patch.object(dspark_fusion_variants, 'REPORT_SHA256', REPORT_SHA256)
                if arm == 'publication' else nullcontext()):
            original_route(value, 'publication')

    with patch.object(full_dspark_request, 'measure_dspark_request', measured), \
            patch.object(gdn_shared_qk_variants, 'validate_route', route):
        yield evidence
