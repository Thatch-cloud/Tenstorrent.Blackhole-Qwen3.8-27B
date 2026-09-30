"""Unqualified independent scratch tiles for four convolution-window writes."""

from frozen_recipe_context import replace_once


def transform_reader(source):
    source = replace_once(source,
        'const uint32_t scratch = source_tiles + 5 * 2048;',
        'const uint32_t scratch = source_tiles + (5 + slot) * 2048;')
    source = replace_once(source,
        '        noc_async_write_tile(page, destination, scratch);\n        noc_async_write_barrier();',
        '        noc_async_write_tile(page, destination, scratch);')
    return replace_once(source,
        '        window<output2_args.next_compile_time_args_offset()>(3, page, rows, scratch);',
        '        window<output2_args.next_compile_time_args_offset()>(3, page, rows, scratch);\n'
        '        noc_async_write_barrier();')


def transform_builder(source):
    return replace_once(source, 'total_size=6 * 2048,', 'total_size=9 * 2048,')


def schedule(pages):
    if type(pages) is not int or pages < 1:
        raise ValueError('Positive page count required')
    return [(page, operation, slot) for page in range(pages)
            for operation, slot in [('issue', slot) for slot in range(4)] + [('barrier', None)]]
