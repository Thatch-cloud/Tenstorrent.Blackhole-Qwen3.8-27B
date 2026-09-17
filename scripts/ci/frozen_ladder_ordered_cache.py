"""Explicit page-table geometry for offline ladder cache writes; kernel math unchanged."""

from contextlib import contextmanager
from unittest.mock import patch

from frozen_context_geometry import geometry


def validate_shapes(cache, packed, positions, pages, *, context):
    specification = geometry(context)
    rows = packed[1] if len(packed) == 4 else 0
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32) or tuple(packed) != (1, rows, 32, 256):
        raise ValueError('Native prepared T=1/2/4/8/16/32 KV tiles required')
    if (len(cache) != 4 or type(cache[0]) is not int or cache[0] < 1
            or tuple(cache[1:]) != (2, 64, 256)):
        raise ValueError('Expected two-head 64-row paged cache')
    if (tuple(positions) != (rows,) or len(pages) != 2 or pages[0] != rows
            or type(pages[1]) is not int
            or not 1 <= pages[1] <= min(specification['target_page_count'], cache[0])):
        raise ValueError('Paired positions and page-table rows within selected context capacity required')
    return rows


@contextmanager
def page_geometry(context):
    import ordered_cache
    specification = geometry(context)
    evidence = dict(context=context, max_pages=specification['target_page_count'],
        calls=0, restored=False, kernel_math_changed=False, geometry_qualified=False)

    def validate(*arguments):
        result = validate_shapes(*arguments, context=context)
        evidence['calls'] += 1
        return result

    original = ordered_cache.validate_shapes
    try:
        with patch.object(ordered_cache, 'validate_shapes', validate):
            yield evidence
    finally:
        evidence['restored'] = ordered_cache.validate_shapes is original
