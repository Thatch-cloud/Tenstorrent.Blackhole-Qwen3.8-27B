"""Prepare simulator-only T32 shared-Q/K builders without changing kernel arithmetic."""

import os

def require_simulator():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_HARDWARE_TESTS') or os.environ.get('QWEN_CARDS_ALLOCATED')):
        raise ValueError('T32 shared-Q/K requires its own simulator qualification')


def payloads(originals):
    from frozen_recipe_context import replace_once

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


def adapt_probe(source):
    from frozen_recipe_context import replace_once
    from frozen_gdn_cache_stage import adapt_probe as scatter_probe

    source = scatter_probe(source, norm_scatter=True)
    source = replace_once(source, 'from shared_qk_norm_scatter import build as build_pipeline',
        'from shared_qk_norm_t32_scatter import build as build_pipeline')
    replacements = (
        ('rows=16, norm_unchanged=True', 'rows=32, norm_unchanged=True', 1),
        ('(2, 16, 5120)', '(2, 32, 5120)', 1),
        ('(2, 16, 24)', '(2, 32, 24)', 2),
        ('(2, 16, 3072)', '(2, 32, 3072)', 1),
        ('mask.reshape(16, -1)', 'mask.reshape(32, -1)', 1),
        ('allocate((16, 1, 96, 32)', 'allocate((32, 1, 96, 32)', 1),
        ('allocate((16, 24, 128, 128)', 'allocate((32, 24, 128, 128)', 1),
        ('allocate((1, 16, 3072)', 'allocate((1, 32, 3072)', 1),
        ('allocate((1, 16, 1024)', 'allocate((1, 32, 1024)', 2))
    for before, after, count in replacements:
        if source.count(before) != count:
            raise ValueError('Exact T16 probe geometry required: ' + before)
        source = source.replace(before, after)
    source = replace_once(source, "        for seed in (1, 2, 0):",
        "        report['stale_controls'] = []\n"
        "        for seed in (1, 2, 0):\n"
        "            previous_states = read(actual[1])")
    source = replace_once(source, "            check(reference, actual, host, 'replay_' + str(seed))",
        "            check(reference, actual, host, 'replay_' + str(seed))\n"
        "            for chip, (previous, current) in enumerate(zip(previous_states, read(actual[1]), strict=True)):\n"
        "                if torch.equal(previous, current):\n"
        "                    raise AssertionError('Changed-input T32 replay retained stale recurrent states')\n"
        "                report['stale_controls'].append(dict(seed=seed, chip=chip, detected=True))")
    source = replace_once(source, "        if len(report['checks']) != 24 or len(report['immutable_checks']) != 48:",
        "        if (len(report['checks']) != 24 or len(report['immutable_checks']) != 48\n"
        "                or len(report['stale_controls']) != 6):")
    compile(source, 'gdn-shared-qk-t32-probe.py', 'exec')
    return source
