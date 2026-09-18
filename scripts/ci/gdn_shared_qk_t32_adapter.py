"""Prepare simulator-only T32 shared-Q/K builders without changing kernel arithmetic."""

import os

from frozen_recipe_context import replace_once


def require_simulator():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_HARDWARE_TESTS') or os.environ.get('QWEN_CARDS_ALLOCATED')):
        raise ValueError('T32 shared-Q/K requires its own simulator qualification')


def payloads(originals):
    program = originals['gdn_shared_qk_program.py']
    for before, after in (
        ('    if type(serial) is not bool',
            '    from gdn_shared_qk_t32_adapter import require_simulator\n'
            '    require_simulator()\n    if type(serial) is not bool'),
        ('(1, 16, 5120)', '(1, 32, 5120)'),
        ('(query, (1, 16, 1024)', '(query, (1, 32, 1024)'),
        ('(key, (1, 16, 1024)', '(key, (1, 32, 1024)'),
        ('Expected T16 packed BF16 input', 'Expected T32 packed BF16 input'),
        ('[head, 16, addresses[0]]', '[head, 32, addresses[0]]'),
        ('[head, 16, *addresses[1:]]', '[head, 32, *addresses[1:]]'),
        ("if role == 'writer' else [16]", "if role == 'writer' else [32]")):
        program = replace_once(program, before, after)
    pipeline = originals['gdn_shared_qk_pipeline.py']
    for before, after in (
        ('from gdn_shared_qk_program import build as build_normalization',
            'from gdn_shared_qk_t32_program import build as build_normalization'),
        ("    stage, rows, prefetch_inputs = 'recurrence', 16, False",
            "    from gdn_shared_qk_t32_adapter import require_simulator\n"
            "    require_simulator()\n"
            "    stage, rows, prefetch_inputs = 'recurrence', 32, False"),
        ('    if len(tensors) != 11:',
            '    from gdn_shared_qk_t32_adapter import require_simulator\n'
            '    require_simulator()\n    if len(tensors) != 11:'),
        ('    expected = ((1, 16, 5120), (1, 16, 24), (1, 16, 24), (1, 24, 128, 128),\n'
            '        (16, 1, 96, 32), (16, 24, 128, 128), (1, 16, 3072), (1, 1, 128),\n'
            '        (1, 16, 3072), (1, 16, 1024), (1, 16, 1024))',
            '    expected = ((1, 32, 5120), (1, 32, 24), (1, 32, 24), (1, 24, 128, 128),\n'
            '        (32, 1, 96, 32), (32, 24, 128, 128), (1, 32, 3072), (1, 1, 128),\n'
            '        (1, 32, 3072), (1, 32, 1024), (1, 32, 1024))'),
        ('kernels, "norm_gate", 16)', 'kernels, "norm_gate", 32)')):
        pipeline = replace_once(pipeline, before, after)
    scatter = replace_once(originals['shared_qk_norm_scatter.py'],
        '    import gdn_shared_qk_pipeline as pipeline',
        '    import gdn_shared_qk_t32_pipeline as pipeline')
    sources = {'gdn_shared_qk_t32_program.py': program,
        'gdn_shared_qk_t32_pipeline.py': pipeline, 'shared_qk_norm_t32_scatter.py': scatter}
    for name, source in sources.items():
        compile(source, name, 'exec')
    return sources
