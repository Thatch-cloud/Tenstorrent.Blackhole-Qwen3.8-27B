"""Unqualified direct-scatter norm reader; no serving integration."""

import gdn_vsplit_norm_batch as batch


START = '    for (uint32_t token = 0; token < n_inst; ++token) {'
END = '    pre.push_back(4);'
SCATTER = '''    for (uint32_t token = 0; token < n_inst; ++token) {
        const uint32_t offset = (512 * (token / 16) + 16 * (token % 16)) * 4;
        for (uint32_t partition = 0; partition < 4; ++partition) {
            const uint32_t page = token * 96 + bh_start * 4 + partition;
            const uint32_t dst = destination + partition * 4096 + offset;
            noc_async_read(pre_acc.get_noc_addr(page, 0), dst, 64);
            noc_async_read(pre_acc.get_noc_addr(page, 64), dst + 1024, 64);
        }
    }
    noc_async_read_barrier();
'''


def replace_reader(source):
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('Norm scatter reader anchors changed')
    start, end = source.index(START), source.index(END)
    if end <= start or source[start:end] != batch.READER_BODY[
            batch.READER_BODY.index(START):batch.READER_BODY.index(END)]:
        raise ValueError('Norm scatter control reader changed')
    return source[:start] + SCATTER + source[end:]


def load_kernels(root=batch.split.DEFAULT_ROOT):
    kernels = batch.load_kernels(root)
    kernels['norm_gate']['reader'] = replace_reader(kernels['norm_gate']['reader'])
    return kernels


def transfers(rows, head):
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Unsupported token count')
    if type(head) is not int or not 0 <= head < 24:
        raise ValueError('Expected head in [0,24)')
    return [(token * 96 + head * 4 + partition, half * 64,
             partition * 4096 + batch.tile_element(token, half * 16) * 4, 64)
            for token in range(rows) for partition in range(4) for half in range(2)]
