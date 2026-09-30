"""Two in-flight bulk BF4 pages, retaining the existing two-block compute buffer."""

from frozen_recipe_context import replace_once


SERIAL = '''    static_assert(pairs_per_worker == 3);
    for (uint32_t block = 0; block < 20; ++block) {
        cb_reserve_back(1, 48);
        const uint32_t destination = get_write_ptr(1);
        noc_async_read(weights.get_noc_addr(block * 91 + first_pair / 3), destination, 27648);
        noc_async_read_barrier();
        cb_push_back(1, 48);
    }
'''
PIPELINED = '''    static_assert(pairs_per_worker == 3);
    constexpr uint32_t block_tiles = 48;
    constexpr uint32_t block_bytes = 27648;
    cb_reserve_back(1, 2 * block_tiles);
    const uint32_t first_slot = get_write_ptr(1);
    auto issue_block = [&](uint32_t block) {
        const uint32_t destination = first_slot + (block % 2) * block_bytes;
        const uint32_t transaction = 1 + block % 2;
        const uint64_t page = weights.get_noc_addr(block * 91 + first_pair / 3);
        for (uint32_t offset = 0; offset < block_bytes; offset += NOC_MAX_BURST_SIZE) {
            const uint32_t remaining = block_bytes - offset;
            const uint32_t bytes = remaining < NOC_MAX_BURST_SIZE ? remaining : NOC_MAX_BURST_SIZE;
            const uint64_t source = page + offset;
            noc_async_read_one_packet_set_state(source, bytes);
            noc_async_read_set_trid(transaction);
            noc_async_read_one_packet_with_state_with_trid(
                0, static_cast<uint32_t>(source), destination + offset, transaction);
        }
    };
    issue_block(0);
    for (uint32_t block = 0; block < 20; ++block) {
        if (block + 1 < 20) {
            cb_reserve_back(1, 2 * block_tiles);
            issue_block(block + 1);
        }
        noc_async_read_barrier_with_trid(1 + block % 2);
        cb_push_back(1, block_tiles);
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
'''


def transform(reader):
    return replace_once(reader, SERIAL, PIPELINED)


def packet_ranges(max_burst):
    if type(max_burst) is not int or max_burst <= 0 or max_burst % 32:
        raise ValueError('Positive aligned native maximum burst required')
    return tuple((offset, min(max_burst, 27648 - offset)) for offset in range(0, 27648, max_burst))
