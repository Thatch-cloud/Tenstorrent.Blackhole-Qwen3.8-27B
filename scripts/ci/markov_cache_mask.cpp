#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto decision_args = TensorAccessorArgs<0>();
    constexpr auto mask_args = TensorAccessorArgs<decision_args.next_compile_time_args_offset()>();
    const auto decision = TensorAccessor(decision_args, get_arg_val<uint32_t>(0), 32);
    const auto mask = TensorAccessor(mask_args, get_arg_val<uint32_t>(1), 2);
    const uint32_t scratch = (get_write_ptr(0) + 127) & ~127u;
    const uint64_t source = get_noc_addr(0, decision);
    const uint32_t input_address = scratch + static_cast<uint32_t>(source & 127);
    noc_async_read(source, input_address, 32);
    noc_async_read_barrier();
    const auto words = reinterpret_cast<volatile uint32_t*>(input_address);
    const bool miss = words[0] == 1 && words[1] == 0 && words[2] < 64 &&
        words[3] < 248320 && words[4] != 0 && words[5] != 0;
    const uint64_t destination = get_noc_addr(0, mask);
    const uint32_t output_address = scratch + 256 + static_cast<uint32_t>(destination & 127);
    auto output = reinterpret_cast<volatile uint16_t*>(output_address);
    for (uint32_t word = 0; word < 16; ++word) {
        output[word] = 0;
    }
    output[0] = miss ? 0x3f80 : 0;
    noc_async_write(output_address, destination, 32);
    noc_async_write_barrier();
}
