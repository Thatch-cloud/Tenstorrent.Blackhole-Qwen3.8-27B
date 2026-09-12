"""Full-page dataflow for block Q/K normalization and its serial reference."""

from gdn_shared_qk_compute import buffer_plan as compute_buffers


COMMON = '''#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void fill_words(uint32_t base, uint32_t count, uint32_t value) {
    auto target = CoreLocalMem<volatile uint32_t>(base);
    for (uint32_t word = 0; word < count; ++word) { target[word] = value; }
    asm volatile("" ::: "memory");
}

void copy_words(uint32_t source, uint32_t destination, uint32_t count) {
    asm volatile("" ::: "memory");
    auto input = CoreLocalMem<volatile uint32_t>(source);
    auto output = CoreLocalMem<volatile uint32_t>(destination);
    for (uint32_t word = 0; word < count; ++word) { output[word] = input[word]; }
    asm volatile("" ::: "memory");
}

uint32_t row_element(uint32_t row) {
    return 512 * (row / 16) + 16 * (row % 16);
}

'''

READER = '''void kernel_main() {
    constexpr bool serial = get_compile_time_arg_val(0);
    constexpr auto input_args = TensorAccessorArgs<1>();
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(2), 2048);
    Noc noc;
    CircularBuffer ones(6);
    ones.reserve_back(1);
    fill_words(ones.get_write_ptr(), 1024, 0x3f800000);
    ones.push_back(1);
    if constexpr (serial) {
        CircularBuffer cache(31);
        cache.reserve_back(8);
        for (uint32_t kind = 0; kind < 2; ++kind) {
            for (uint32_t tile = 0; tile < 4; ++tile) {
                noc.async_read(input, cache, 2048,
                    {.page_id = kind * 32 + head * 4 + tile},
                    {.offset_bytes = (kind * 4 + tile) * 2048});
            }
        }
        noc.async_read_barrier();
        cache.push_back(8);
        cache.wait_front(8);
        const uint32_t cached = cache.get_read_ptr();
        for (uint32_t token = 0; token < rows; ++token) {
            for (uint32_t kind = 0; kind < 2; ++kind) {
                CircularBuffer destination(kind);
                destination.reserve_back(4);
                const uint32_t base = destination.get_write_ptr();
                fill_words(base, 4 * 512, 0);
                for (uint32_t tile = 0; tile < 4; ++tile) {
                    const uint32_t source = cached + (kind * 4 + tile) * 2048 + row_element(token) * 2;
                    copy_words(source, base + tile * 2048, 8);
                    copy_words(source + 512, base + tile * 2048 + 512, 8);
                }
                destination.push_back(4);
            }
        }
        cache.pop_front(8);
    } else {
        for (uint32_t kind = 0; kind < 2; ++kind) {
            CircularBuffer destination(kind);
            destination.reserve_back(4);
            for (uint32_t tile = 0; tile < 4; ++tile) {
                noc.async_read(input, destination, 2048,
                    {.page_id = kind * 32 + head * 4 + tile}, {.offset_bytes = tile * 2048});
            }
            noc.async_read_barrier();
            const uint32_t base = destination.get_write_ptr();
            for (uint32_t tile = 0; tile < 4; ++tile) {
                for (uint32_t row = rows; row < 32; ++row) {
                    const uint32_t offset = base + tile * 2048 + row_element(row) * 2;
                    fill_words(offset, 8, 0);
                    fill_words(offset + 512, 8, 0);
                }
            }
            destination.push_back(4);
        }
    }
}
'''

WRITER = '''void kernel_main() {
    constexpr bool serial = get_compile_time_arg_val(0);
    constexpr auto query_args = TensorAccessorArgs<1>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const auto query_output = TensorAccessor(query_args, get_arg_val<uint32_t>(2), 4096);
    const auto key_output = TensorAccessor(key_args, get_arg_val<uint32_t>(3), 4096);
    Noc noc;
    if constexpr (serial) {
        for (uint32_t kind = 0; kind < 2; ++kind) {
            CircularBuffer assembly(12 + kind);
            assembly.reserve_back(4);
            fill_words(assembly.get_write_ptr(), 4096, 0);
        }
        for (uint32_t token = 0; token < rows; ++token) {
            for (uint32_t kind = 0; kind < 2; ++kind) {
                CircularBuffer normalized(10 + kind);
                normalized.wait_front(4);
                const uint32_t source = normalized.get_read_ptr();
                const uint32_t target = CircularBuffer(12 + kind).get_write_ptr();
                for (uint32_t tile = 0; tile < 4; ++tile) {
                    const uint32_t offset = target + tile * 4096 + row_element(token) * 4;
                    copy_words(source + tile * 4096, offset, 16);
                    copy_words(source + tile * 4096 + 1024, offset + 1024, 16);
                }
                normalized.pop_front(4);
            }
        }
        for (uint32_t kind = 0; kind < 2; ++kind) {
            CircularBuffer assembly(12 + kind);
            assembly.push_back(4);
        }
    }
    for (uint32_t kind = 0; kind < 2; ++kind) {
        CircularBuffer result((serial ? 12 : 10) + kind);
        result.wait_front(4);
        const uint32_t base = result.get_read_ptr();
        for (uint32_t tile = 0; tile < 4; ++tile) {
            if (kind == 0) {
                noc.async_write(CoreLocalMem<uint32_t>(base + tile * 4096), query_output, 4096,
                    {}, {.page_id = head * 4 + tile});
            } else {
                noc.async_write(CoreLocalMem<uint32_t>(base + tile * 4096), key_output, 4096,
                    {}, {.page_id = head * 4 + tile});
            }
        }
        noc.async_write_barrier();
        result.pop_front(4);
    }
}
'''


def kernels():
    return dict(reader=COMMON + READER, writer=COMMON + WRITER)


def buffer_plan(*, serial):
    if type(serial) is not bool:
        raise ValueError('Explicit serial ownership plan required')
    io, fp32 = compute_buffers()
    if serial:
        io[31] = 8
        fp32.update({12: 4, 13: 4})
    return io, fp32


def row_element(row, column=0):
    if type(row) is not int or type(column) is not int or not 0 <= row < 32 or not 0 <= column < 32:
        raise ValueError('Expected physical 32x32 tile coordinate')
    return (row // 16 * 2 + column // 16) * 256 + row % 16 * 16 + column % 16
