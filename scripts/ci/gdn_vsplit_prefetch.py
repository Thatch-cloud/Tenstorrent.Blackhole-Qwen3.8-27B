"""Experimental full-block L1 input cache for the96-worker recurrence reader."""

import hashlib
from pathlib import Path

import gdn_vsplit as split


RUNTIME_HASHES = {
    'tt_metal/hw/inc/api/dataflow/circular_buffer.h':
        'cbce48d248143c87ed0bc29305a4206f149341a22d723d189c92a6a946969a46',
    'tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_unpack.h':
        '2f56110ad41c400dc5be49cc0eb18338f6294c79804dcda8840d049a281704cb',
}

CACHE_AND_GATHER = '''    static_assert(Kt == 4 && Vt == 1 && PACKED && !FNG);
    constexpr uint32_t cb_cache = 31;
    CircularBuffer cache(cb_cache);
    cache.reserve_back(11);
    for (uint32_t tile = 0; tile < 4; ++tile) {
        noc.async_read(q_acc, cache, tb_io,
            {.page_id = 4 * (bh_start / 12) + tile}, {.offset_bytes = tile * tb_io});
        noc.async_read(k_acc, cache, tb_io,
            {.page_id = 32 + 4 * (bh_start / 12) + tile}, {.offset_bytes = (4 + tile) * tb_io});
    }
    noc.async_read(v_acc, cache, tb_io, {.page_id = 64 + bh_start}, {.offset_bytes = 8 * tb_io});
    noc.async_read(beta_acc, cache, tb_io, {.page_id = 0}, {.offset_bytes = 9 * tb_io});
    noc.async_read(g_acc, cache, tb_io, {.page_id = 0}, {.offset_bytes = 10 * tb_io});
    noc.async_read_barrier();
    cache.push_back(11);
    cache.wait_front(11);
    const uint32_t cached_base = cache.get_read_ptr();

    auto gather_cached_row = [&](uint32_t cb_id, uint32_t tiles, uint32_t first, uint32_t row) {
        CircularBuffer destination(cb_id);
        destination.reserve_back(tiles);
        const uint32_t base = destination.get_write_ptr();
        if (row == 0) {
            zero(base, tiles * tb_io / 4);
        }
        const uint32_t offset = (512 * (row / 16) + 16 * (row % 16)) * elem;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            const uint32_t source = cached_base + (first + tile) * tb_io + offset;
            const uint32_t target = base + tile * tb_io;
            copy_chunk(source, target);
            copy_chunk(source + 256 * elem, target + 256 * elem);
        }
        destination.push_back(tiles);
    };

    auto gather_cached_scalar = [&](uint32_t cb_id, uint32_t page, uint32_t row) {
        CircularBuffer destination(cb_id);
        destination.reserve_back(1);
        const uint32_t base = destination.get_write_ptr();
        if (row == 0) {
            zero(base, tb_io / 4);
        }
        const uint32_t column = bh_start / 4;
        const uint32_t element = ((row / 16) * 2 + column / 16) * 256
            + (row % 16) * 16 + column % 16;
        const uint32_t address = cached_base + page * tb_io + element * elem;
        asm volatile("" ::: "memory");
        auto source = CoreLocalMem<volatile uint32_t>(address & ~3u);
        auto target = CoreLocalMem<volatile uint32_t>(base);
        target[0] = (source[0] >> ((address & 2u) * 8)) & 0xffffu;
        asm volatile("" ::: "memory");
        destination.push_back(1);
    };

'''

CACHED_INPUTS = '''        gather_cached_row(cb_q, 4, 0, token);
        gather_cached_row(cb_k, 4, 4, token);
        gather_cached_row(cb_v, 1, 8, token);
        gather_cached_scalar(cb_beta, 9, token);
        gather_cached_scalar(cb_g, 10, token);

'''


def validate_runtime(root):
    split.validate_runtime(root)
    for relative, expected in RUNTIME_HASHES.items():
        if hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Prefetch CB ownership runtime changed: {relative}')


def transform_reader(source):
    source = split.native.replace_section(source, '    auto gather_row =',
        '    // fp32 all-ones tile', CACHE_AND_GATHER)
    source = split.native.replace_section(source, '        if constexpr (PACKED) {',
        '        // State [B,H,K,V]', CACHED_INPUTS)
    source = split.native.replace_once(source,
        '            gather_row(z_acc, cb_v, Vt, (b / 32) * WTZ + ZOT + h * Vt, b % 32);',
        '            static_assert(!FNG);')
    if not source.endswith('    }\n}\n'):
        raise ValueError('Recurrence reader tail changed')
    return source[:-2] + '    cache.pop_front(11);\n}\n'


def load_kernels(root=split.DEFAULT_ROOT):
    kernels = split.load_kernels(root)
    kernels['recurrence']['reader'] = transform_reader(kernels['recurrence']['reader'])
    return kernels


def source_element(row, column):
    if type(row) is not int or type(column) is not int or not 0 <= row < 32 or not 0 <= column < 32:
        raise ValueError('Expected physical BF16 tile coordinates')
    return ((row // 16) * 2 + column // 16) * 256 + row % 16 * 16 + column % 16


def audit(root):
    kernels = load_kernels(root)
    return dict(scope='Uncertified recurrence-only input prefetch; no norm cache or compute changes',
        extra_cb_bytes_per_worker=22528, cache_pages_per_worker=11,
        reader_sha256=hashlib.sha256(kernels['recurrence']['reader'].encode()).hexdigest(),
        runtime_hashes=RUNTIME_HASHES,
        issued_input_bytes_per_chip={str(rows): dict(control=rows * 96 * 11 * 2048,
            candidate=96 * 11 * 2048) for rows in (1, 2, 4, 8, 16, 32)})
