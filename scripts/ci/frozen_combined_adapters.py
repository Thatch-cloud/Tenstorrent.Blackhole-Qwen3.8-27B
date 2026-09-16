"""Explicit candidate admission changes; never applied to default serving files."""

from frozen_recipe_context import replace_once


FILES = ('dspark_8k_admission.py', 'dspark_8k_entry.py')


def adapt_combined_sources(sources):
    result = dict(sources)
    result['dspark_8k_admission.py'] = adapt_admission(result['dspark_8k_admission.py'])
    replacements = {
        'dspark_8k_entry.py': (
            ('request_context() != 8192', 'request_context() != 32768'),
            ("    from dspark_8k_scope import runtime_scope",
                "    if os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1' or os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1':\n"
                "        raise ValueError('Explicit combined candidate and compact scratch required')\n"
                '    from dspark_8k_scope import runtime_scope'),
            ('context=8192, output_tokens=256', 'context=32768, output_tokens=256')),
        'dspark_8k_scope.py': (
            ('from dspark_attention_8k_gate import REPORT_SHA256',
                'from frozen_combined_runtime import REPORT_SHA256'),
            ('        stack.enter_context(scoped_stats_pack())',
                '        from dspark_ladder_scalar_reciprocal import scalar_reciprocal\n'
                '        stack.enter_context(scalar_reciprocal())\n'
                '        stack.enter_context(scoped_stats_pack())')),
        'dspark_context_selection.py': (
            ('if history_limit() == 8448:', 'if history_limit() == 33024:'),),
    }
    for name, changes in replacements.items():
        for before, after in changes:
            result[name] = replace_once(result[name], before, after)
    return result


def adapt_admission(source):
    changes = (
        ('from dspark_attention_8k_gate import qualify', 'from frozen_combined_runtime import qualify'),
        ('return 8448 if _ADMISSION.get() is not None else 8192',
            "return _ADMISSION.get()['capacity'] if _ADMISSION.get() is not None else 8192"),
        ('context != 8192 or output_tokens != 256', 'context != 32768 or output_tokens != 256'),
        ('8K trial requires exactly 8192 prompt rows and 256 output-token headroom',
            'Qualified candidate requires exactly 32768 prompt rows and 256 output-token headroom'),
        ('output_tokens=output_tokens, capacity=8448, key_chunk_size=256',
            'output_tokens=output_tokens, capacity=33024, key_chunk_size=256'),
    )
    for before, after in changes:
        source = replace_once(source, before, after)
    compile(source, 'dspark_8k_admission.py', 'exec')
    return source
