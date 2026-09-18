"""Host oracle for face-aligned DMA segments; device qualification remains separate."""

from draft_kv_slide import geometry


def segments(history_rows, prefix, tile):
    shape = geometry(history_rows, prefix)
    if type(tile) is not int or not 0 <= tile < 64:
        raise ValueError('Bounded history tile required')
    result, row = [], 0
    while row < 32:
        destination = tile * 32 + row
        if destination >= shape['rows']:
            result.append(('zero', 0, row, 32 - row))
            break
        logical = destination + shape['drop']
        historical = logical < history_rows
        source = logical if historical else logical - history_rows
        remaining = history_rows - source if historical else prefix - source
        count = min(16 - row % 16, 16 - source % 16, remaining, shape['rows'] - destination)
        if count <= 0:
            raise ValueError('DMA segment must advance')
        result.append(('active' if historical else 'delta', source, row, count))
        row += count
    return result
