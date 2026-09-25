#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto packed_args = TensorAccessorArgs<0>();
    constexpr auto separate_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<separate_args.next_compile_time_args_offset()>();
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(0), 576);
    const auto separate = TensorAccessor(separate_args, get_arg_val<uint32_t>(1), 576);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(2), 4096);
    const uint32_t worker = get_arg_val<uint32_t>(3);
    const uint32_t workers = get_arg_val<uint32_t>(4);
    const uint32_t columns = get_arg_val<uint32_t>(5);
    const uint32_t pages = get_arg_val<uint32_t>(6);
    const uint32_t offset = get_arg_val<uint32_t>(7);
    const uint32_t scratch = get_write_ptr(0);
    auto first = reinterpret_cast<volatile const uint32_t*>(scratch);
    auto second = reinterpret_cast<volatile const uint32_t*>(scratch + 576);
    auto result = reinterpret_cast<volatile uint32_t*>(scratch + 4096);
    for (uint32_t index = 0; index < 1024; index++) { result[index] = 0; }
    result[4] = 0x514B5631;
    for (uint32_t page = worker; page < pages; page += workers) {
        const uint32_t paired_page = (page / columns) * columns * 2 + (page % columns) * 2 + offset;
        noc_async_read_tile(paired_page, packed, scratch);
        noc_async_read_tile(page, separate, scratch + 576);
        noc_async_read_barrier();
        for (uint32_t word = 0; word < 144; word++) {
            if (first[word] != second[word]) {
                result[0]++;
                if (result[2] == 0) { result[2] = page + 1; result[3] = word + 1; }
            }
        }
        result[1]++;
    }
    result[5] = ~result[0];
    noc_async_write_tile(worker, output, scratch + 4096);
    noc_async_write_barrier();
}
