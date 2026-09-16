"""Historical runtime plumbing; component admission remains independently mandatory."""

from frozen_recipe_context import replace_once


FILES = ('run-dspark-hardware.sh', 'dspark-hardware-suite.sh',
    'dspark_context_selection.py', 'dspark-target-hardware.py')


def adapt_runtime_sources(sources):
    result = dict(sources)
    changes = {
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
