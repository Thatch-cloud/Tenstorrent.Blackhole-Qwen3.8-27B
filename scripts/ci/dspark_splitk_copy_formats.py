"""Explicit format transitions for the unqualified hybrid decode candidate."""

import re


def transform(source):
    pattern = re.compile(r'^( +)(move_block<true>\((cb_\w+), (cb_\w+), (\w+)\);)$', re.MULTILINE)
    matches = list(pattern.finditer(source))
    if len(matches) != 11:
        raise ValueError('Expected ten native moves and one decomposed correction move')

    def replace(match):
        indent, operation, source_cb, destination_cb, unused_count = match.groups()
        return (f'{indent}reconfig_data_format_srca({source_cb});\n'
                f'{indent}pack_reconfig_data_format({destination_cb});\n'
                f'{indent}{operation}')

    return pattern.sub(replace, source)
