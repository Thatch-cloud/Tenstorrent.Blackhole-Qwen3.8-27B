"""Unqualified four-block fused-MLP buffering candidate; no serving integration."""

import hashlib

from frozen_recipe_context import replace_once


def transform(source):
    result = replace_once(source,
        'cb(0, ttnn.bfloat16, 2048, 16, all_cores)',
        'cb(0, ttnn.bfloat16, 2048, 32, all_cores)')
    result = replace_once(result,
        'cb(1, ttnn.bfloat4_b, 576, 32 * pairs_per_worker, workers)',
        'cb(1, ttnn.bfloat4_b, 576, 64 * pairs_per_worker, workers)')
    result = replace_once(result,
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,',
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,\n'
        '                             input_buffer_blocks=4, weight_buffer_blocks=4,')
    compile(result, 'fused_1d.py', 'exec')
    return result


def manifest(source):
    candidate = transform(source)
    return dict(control_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        token_rows=16, pairs_per_worker=3, input_buffer_blocks=4, weight_buffer_blocks=4,
        extra_input_bytes_per_core=16 * 2048,
        extra_weight_bytes_per_worker=32 * 3 * 576,
        compute_changed=False, readers_changed=False,
        simulator_qualified=False, hardware_qualified=False, performance_qualified=False)
