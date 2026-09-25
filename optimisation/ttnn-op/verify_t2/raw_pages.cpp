// verify_t2 card-M harness: copy every page of one interleaved buffer to the SAME page index of
// another with the same page size (RISCV_0 only). Used both ways:
//   raw_pages   a tile tensor -> a row-major uint32 (pages, page_bytes / 4) DRAM tensor, so
//               to_torch returns the raw page bytes - padding rows, -0, NaN payloads and bf8
//               exponent sections included;
//   put_pages   a row-major uint32 tensor of host-built pages -> a tile tensor, so an input's
//               padding rows can hold a poison pattern.
// Compile-time args: [page_bytes] + TensorAccessorArgs(source) + TensorAccessorArgs(destination).
// Common runtime args: [source address, destination address]. Per core: [page_start, page_count].

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr auto source_args = TensorAccessorArgs<1>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(source_args, get_common_arg_val<uint32_t>(0), page_bytes);
    const auto destination = TensorAccessor(destination_args, get_common_arg_val<uint32_t>(1), page_bytes);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t staging = get_write_ptr(0);
    for (uint32_t page = start; page < start + count; ++page) {
        noc_async_read(source.get_noc_addr(page), staging, page_bytes);
        noc_async_read_barrier();
        noc_async_write(staging, destination.get_noc_addr(page), page_bytes);
        noc_async_write_barrier();
    }
}
