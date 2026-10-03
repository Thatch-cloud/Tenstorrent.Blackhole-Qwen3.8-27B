"""Experimental row-parallel whole-head norm/gate after exact recurrence."""

import hashlib

import gdn_vsplit as split
from gdn_vsplit_prefetch import validate_runtime


READER_BODY = '''    static_assert(Vt == 4 && FNG);
    CircularBuffer weights(cb_w);
    weights.reserve_back(2 * Vt);
    const uint32_t weight_base = weights.get_write_ptr();
    for (uint32_t tile = 0; tile < Vt; ++tile) {
        noc.async_read(w_acc, weights, tb_io, {.page_id = tile}, {.offset_bytes = tile * tb_io});
    }
    noc.async_read_barrier();
    for (uint32_t tile = 0; tile < Vt; ++tile) {
        const uint32_t base = weight_base + tile * tb_io;
        for (uint32_t row = 1; row < 32; ++row) {
            const uint32_t offset = (512 * (row / 16) + 16 * (row % 16)) * elem;
            copy_chunk(base, base + offset);
            copy_chunk(base + 256 * elem, base + offset + 256 * elem);
        }
    }
    weights.push_back(2 * Vt);

    CircularBuffer gates(cb_v);
    gates.reserve_back(Vt);
    for (uint32_t tile = 0; tile < Vt; ++tile) {
        noc.async_read(z_acc, gates, tb_io,
            {.page_id = bh_start * Vt + tile}, {.offset_bytes = tile * tb_io});
    }
    noc.async_read_barrier();
    gates.push_back(Vt);

    const auto pre_acc = TensorAccessor(q_a, q_addr, 128);
    constexpr uint32_t cb_pre = 15, cb_stick = 5;
    CircularBuffer pre(cb_pre);
    pre.reserve_back(4);
    const uint32_t destination = pre.get_write_ptr();
    zero(destination, 4 * 4096 / 4);
    for (uint32_t token = 0; token < n_inst; ++token) {
        const uint32_t offset = (512 * (token / 16) + 16 * (token % 16)) * 4;
        for (uint32_t partition = 0; partition < 4; ++partition) {
            CircularBuffer stick(cb_stick);
            stick.reserve_back(1);
            noc.async_read(pre_acc, stick, 128,
                {.page_id = token * 96 + bh_start * 4 + partition}, {.offset_bytes = 0});
            noc.async_read_barrier();
            stick.push_back(1);
            stick.wait_front(1);
            const uint32_t src = stick.get_read_ptr();
            const uint32_t dst = destination + partition * 4096 + offset;
            asm volatile("" ::: "memory");
            auto input = CoreLocalMem<volatile uint32_t>(src);
            auto output = CoreLocalMem<volatile uint32_t>(dst);
            for (uint32_t word = 0; word < 16; ++word) {
                output[word] = input[word];
                output[256 + word] = input[16 + word];
            }
            asm volatile("" ::: "memory");
            stick.pop_front(1);
        }
    }
    pre.push_back(4);
}
'''

WRITER_BODY = '''    static_assert(Vt == 4 && FNG);
    CircularBuffer output(cb_out);
    output.wait_front(Vt);
    const uint32_t base = output.get_read_ptr();
    for (uint32_t tile = 0; tile < Vt; ++tile) {
        for (uint32_t row = n_inst; row < 32; ++row) {
            const uint32_t offset = (512 * (row / 16) + 16 * (row % 16)) * elem;
            zero(base + tile * tb_io + offset, 16 * elem / 4);
            zero(base + tile * tb_io + offset + 256 * elem, 16 * elem / 4);
        }
        noc.async_write(CoreLocalMem<uint32_t>(base), o_acc, tb_io,
            {.offset_bytes = tile * tb_io}, {.page_id = bh_start * Vt + tile});
    }
    noc.async_write_barrier();
    output.pop_front(Vt);
}
'''


def before_weight_load(source):
    anchor = '    if constexpr (FNG) {'
    if source.count(anchor) != 1:
        raise ValueError('Norm reader weight-load anchor changed')
    return source[:source.index(anchor)]


def load_kernels(root=split.DEFAULT_ROOT):
    kernels = split.load_kernels(root)
    norm = kernels['norm_gate']
    norm['reader'] = before_weight_load(norm['reader']) + READER_BODY
    norm['compute'] = split.native.replace_once(norm['compute'], split.LOOP,
        '    for (uint32_t it = 0; it < 1; ++it) {')
    anchor = '    if constexpr (FNG) { zero(asm_base, Vt * tb_io / 4); }'
    if norm['writer'].count(anchor) != 1:
        raise ValueError('Norm writer anchor changed')
    norm['writer'] = norm['writer'][:norm['writer'].index(anchor)] + WRITER_BODY
    return kernels


def tile_element(token, column):
    if type(token) is not int or type(column) is not int or not 0 <= token < 32 or not 0 <= column < 32:
        raise ValueError('Expected physical norm tile coordinates')
    return ((token // 16) * 2 + column // 16) * 256 + token % 16 * 16 + column % 16


def audit(root):
    kernels = load_kernels(root)
    return dict(scope='Uncertified token-row-parallel whole128-wide norm/gate; recurrence unchanged',
        norm_iterations_per_block=1, norm_workers=24, cb_bytes=204800,
        generated_hashes={role: hashlib.sha256(source.encode()).hexdigest()
                          for role, source in kernels['norm_gate'].items()})
