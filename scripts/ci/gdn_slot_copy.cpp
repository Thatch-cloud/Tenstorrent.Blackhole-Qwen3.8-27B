#include "api/dataflow/dataflow_api.h"

template <uint32_t source_offset, uint32_t destination_offset, bool row_only>
void transfer(uint32_t argument_offset, uint32_t worker, uint32_t scratch) {
    constexpr auto source_args = TensorAccessorArgs<source_offset>();
    constexpr auto destination_args = TensorAccessorArgs<destination_offset>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(argument_offset), 2048);
    const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(argument_offset + 1), 2048);
    const uint32_t pages = get_arg_val<uint32_t>(argument_offset + 2);
    const uint32_t source_slot = get_arg_val<uint32_t>(argument_offset + 3);
    const uint32_t destination_slot = get_arg_val<uint32_t>(argument_offset + 4);
    for (uint32_t page = worker; page < pages; page += 48) {
        if constexpr (row_only) {
            const uint32_t source_row = source_slot * 32;
            const uint32_t destination_row = destination_slot * 32;
            noc_async_read(source.get_noc_addr(page, source_row), scratch + source_row, 32);
            noc_async_read(source.get_noc_addr(page, 512 + source_row), scratch + 512 + source_row, 32);
            noc_async_read_barrier();
            noc_async_write(scratch + source_row, destination.get_noc_addr(page, destination_row), 32);
            noc_async_write(scratch + 512 + source_row, destination.get_noc_addr(page, 512 + destination_row), 32);
        } else {
            noc_async_read_tile(source_slot * pages + page, source, scratch);
            noc_async_read_barrier();
            noc_async_write_tile(destination_slot * pages + page, destination, scratch);
        }
        noc_async_write_barrier();
    }
}

void kernel_main() {
    constexpr auto first = TensorAccessorArgs<0>();
    constexpr auto second = TensorAccessorArgs<first.next_compile_time_args_offset()>();
    constexpr auto third = TensorAccessorArgs<second.next_compile_time_args_offset()>();
    constexpr auto fourth = TensorAccessorArgs<third.next_compile_time_args_offset()>();
    constexpr auto fifth = TensorAccessorArgs<fourth.next_compile_time_args_offset()>();
    constexpr auto sixth = TensorAccessorArgs<fifth.next_compile_time_args_offset()>();
    constexpr auto seventh = TensorAccessorArgs<sixth.next_compile_time_args_offset()>();
    constexpr auto eighth = TensorAccessorArgs<seventh.next_compile_time_args_offset()>();
    constexpr auto ninth = TensorAccessorArgs<eighth.next_compile_time_args_offset()>();
    const uint32_t worker = get_arg_val<uint32_t>(25);
    const uint32_t scratch = (get_write_ptr(0) + 63) & ~63u;
    transfer<0, first.next_compile_time_args_offset(), false>(0, worker, scratch);
    transfer<second.next_compile_time_args_offset(), third.next_compile_time_args_offset(), true>(5, worker, scratch);
    transfer<fourth.next_compile_time_args_offset(), fifth.next_compile_time_args_offset(), true>(10, worker, scratch);
    transfer<sixth.next_compile_time_args_offset(), seventh.next_compile_time_args_offset(), true>(15, worker, scratch);
    transfer<eighth.next_compile_time_args_offset(), ninth.next_compile_time_args_offset(), true>(20, worker, scratch);
}
