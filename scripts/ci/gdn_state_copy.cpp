#include "api/dataflow/dataflow_api.h"

template <uint32_t source_offset, uint32_t destination_offset, bool row_only>
void transfer(uint32_t argument_offset, uint32_t worker, uint32_t scratch) {
    constexpr auto source_args = TensorAccessorArgs<source_offset>();
    constexpr auto destination_args = TensorAccessorArgs<destination_offset>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(argument_offset), 2048);
    const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(argument_offset + 1), 2048);
    const uint32_t pages = get_arg_val<uint32_t>(argument_offset + 2);
    for (uint32_t page = worker; page < pages; page += 48) {
        if constexpr (row_only) {
            noc_async_read(source.get_noc_addr(page, 0), scratch, 32);
            noc_async_read(source.get_noc_addr(page, 512), scratch + 512, 32);
            noc_async_read_barrier();
            noc_async_write(scratch, destination.get_noc_addr(page, 0), 32);
            noc_async_write(scratch + 512, destination.get_noc_addr(page, 512), 32);
        } else {
            noc_async_read_tile(page, source, scratch);
            noc_async_read_barrier();
            noc_async_write_tile(page, destination, scratch);
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
    const uint32_t worker = get_arg_val<uint32_t>(15);
    const uint32_t scratch = get_write_ptr(0);
    transfer<0, first.next_compile_time_args_offset(), false>(0, worker, scratch);
    transfer<second.next_compile_time_args_offset(), third.next_compile_time_args_offset(), true>(3, worker, scratch);
    transfer<fourth.next_compile_time_args_offset(), fifth.next_compile_time_args_offset(), true>(6, worker, scratch);
    transfer<sixth.next_compile_time_args_offset(), seventh.next_compile_time_args_offset(), true>(9, worker, scratch);
    transfer<eighth.next_compile_time_args_offset(), ninth.next_compile_time_args_offset(), true>(12, worker, scratch);
}
