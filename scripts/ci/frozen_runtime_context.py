"""Historical runtime plumbing; component admission remains independently mandatory."""

from frozen_recipe_context import replace_once


FILES = ('run-dspark-hardware.sh', 'dspark-hardware-suite.sh',
    'dspark_context_selection.py', 'dspark-target-hardware.py', 'dspark_8k_scope.py')


def adapt_runtime_sources(sources):
    result = dict(sources)
    changes = {
        'dspark_8k_scope.py': (
            ('from pathlib import Path',
                'from pathlib import Path\nfrom frozen_context_geometry import selected_geometry, geometry'),
            ('history_limit() != 8448 or type(position) is not int or position != 8192 or type(capacity) is not int or capacity != 8448',
                "history_limit() != selected_geometry()['capacity'] or type(position) is not int or position != selected_geometry()['context'] or type(capacity) is not int or capacity != selected_geometry()['capacity']"),
            ('Admitted exact 8192/8448 fixed history required',
                'Admitted exact selected-context fixed history required'),
            ("patch.object(dspark_full_attention, 'MAX_CONTEXT', 8448)",
                "patch.object(dspark_full_attention, 'MAX_CONTEXT', geometry(context)['capacity'])"),
            ("            assertion = '''static_assert", "            assertion = f'''static_assert"),
            ('get_compile_time_arg_val(3) == 272 && get_compile_time_arg_val(8) == 8',
                "get_compile_time_arg_val(3) == {geometry(context)['padded_keys'] // 32} && get_compile_time_arg_val(8) == 8")),
        'run-dspark-hardware.sh': ((
            '    -e "QWEN_DSPARK_MODE=$mode"',
            '    -e "QWEN_DSPARK_REQUEST_CONTEXT=${QWEN_DSPARK_REQUEST_CONTEXT:-8192}" \\\n    -e "QWEN_DSPARK_MODE=$mode"'),),
        'dspark-hardware-suite.sh': ((
            'export QWEN_DSPARK_REQUEST_CONTEXT=8192',
            'export QWEN_DSPARK_REQUEST_CONTEXT=${QWEN_DSPARK_REQUEST_CONTEXT:-8192}'),),
        'dspark_context_selection.py': (
            ('import os', 'import os\nfrom frozen_context_geometry import CONTEXTS'),
            ("'QWEN_DSPARK_REQUEST_CONTEXT', '4096'", "'QWEN_DSPARK_REQUEST_CONTEXT', '8192'"),
            ("value not in ('4096', '8192')", 'value not in tuple(map(str, CONTEXTS))'),
            ('Only explicit 4096 or 8192 request contexts are supported',
                'Explicit same-recipe ladder context required; admission is checked separately')),
        'dspark-target-hardware.py': (
            ('import os', 'import os\nfrom frozen_context_geometry import selected_geometry'),
            ('max_batch_size=8, max_seq_len=65536',
                "max_batch_size=8, max_seq_len=selected_geometry()['target_sequence_capacity']")),
    }
    for name, replacements in changes.items():
        for before, after in replacements:
            result[name] = replace_once(result[name], before, after)
    return result
