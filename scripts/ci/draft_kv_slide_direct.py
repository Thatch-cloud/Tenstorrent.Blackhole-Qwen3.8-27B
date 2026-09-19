"""Host oracle for face-aligned DMA segments; device qualification remains separate."""

from draft_kv_slide import geometry


def transfers(source_row, destination_row, count, face, buffer_base=0, source_base=0):
    if (not 1 <= count <= 16 or face not in (0, 1)
            or source_row % 16 + count > 16 or destination_row % 16 + count > 16
            or buffer_base % 16 or source_base % 64):
        raise ValueError('Face-bounded segment and aligned buffers required')
    source = source_base + (source_row % 32 // 16) * 1024 + face * 512 + source_row % 16 * 32
    destination = buffer_base + 6144 + (destination_row // 16) * 1024 + face * 512 + destination_row % 16 * 32
    length = count * 32
    if source % 64 == destination % 64:
        return [('dram', source, destination, length)]
    scratch = ((buffer_base + 63) & ~63) + face * 1024 + source % 64
    return [('dram', source, scratch, length), ('l1', scratch, destination, length)]


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
