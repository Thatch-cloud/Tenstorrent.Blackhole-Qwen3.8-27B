"""Explicit candidate admission changes; never applied to default serving files."""

from frozen_recipe_context import replace_once


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
