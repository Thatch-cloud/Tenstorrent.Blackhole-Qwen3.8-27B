#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto entry_rec_args = TensorAccessorArgs<0>();
    constexpr auto entry_conv_args = TensorAccessorArgs<entry_rec_args.next_compile_time_args_offset()>();
    constexpr auto history_rec_args = TensorAccessorArgs<entry_conv_args.next_compile_time_args_offset()>();
    constexpr auto history_conv_args = TensorAccessorArgs<history_rec_args.next_compile_time_args_offset()>();
    constexpr auto native_rec_args = TensorAccessorArgs<history_conv_args.next_compile_time_args_offset()>();
    constexpr auto native_conv_args = TensorAccessorArgs<native_rec_args.next_compile_time_args_offset()>();
    constexpr auto checkpoint_rec_args = TensorAccessorArgs<native_conv_args.next_compile_time_args_offset()>();
    constexpr auto checkpoint_conv_args = TensorAccessorArgs<checkpoint_rec_args.next_compile_time_args_offset()>();
    const uint32_t prefix = get_arg_val<uint32_t>(20);
    const uint32_t worker = get_arg_val<uint32_t>(21);
    const uint32_t staged = get_write_ptr(0);
    const uint32_t output = staged + 2048;
    const auto entry_rec = TensorAccessor(entry_rec_args, get_arg_val<uint32_t>(0), 2048);
    const auto history_rec = TensorAccessor(history_rec_args, get_arg_val<uint32_t>(5), 2048);
    const auto native_rec = TensorAccessor(native_rec_args, get_arg_val<uint32_t>(10), 2048);
    const auto checkpoint_rec = TensorAccessor(checkpoint_rec_args, get_arg_val<uint32_t>(15), 2048);
    for (uint32_t page = worker; page < 384; page += 2) {
        if (prefix == 0) {
            noc_async_read_tile(page, entry_rec, staged);
        } else {
            noc_async_read_tile((prefix - 1) * 384 + page, history_rec, staged);
        }
        noc_async_read_barrier();
        noc_async_write_tile(page, native_rec, staged);
        noc_async_write_tile(page, checkpoint_rec, staged);
        noc_async_write_barrier();
    }
    const uint32_t token = prefix == 0 ? 0 : prefix - 1;
    const uint32_t offset = ((token / 16) * 512 + (token % 16) * 16) * 2;
    for (uint32_t task = worker; task < 640; task += 2) {
        const uint32_t slot = task / 160;
        const uint32_t page = task % 160;
        const auto entry = TensorAccessor(entry_conv_args, get_arg_val<uint32_t>(1 + slot), 2048);
        const auto history = TensorAccessor(history_conv_args, get_arg_val<uint32_t>(6 + slot), 2048);
        const auto native = TensorAccessor(native_conv_args, get_arg_val<uint32_t>(11 + slot), 2048);
        const auto checkpoint = TensorAccessor(checkpoint_conv_args, get_arg_val<uint32_t>(16 + slot), 2048);
        if (prefix == 0) {
            noc_async_read_tile(page, entry, staged);
        } else {
            noc_async_read_tile(page, history, staged);
        }
        noc_async_read_barrier();
        auto words = reinterpret_cast<volatile uint32_t*>(output);
        for (uint32_t word = 0; word < 512; ++word) { words[word] = 0; }
        for (uint32_t face = 0; face < 2; ++face) {
            const auto source = reinterpret_cast<volatile const uint32_t*>(staged + offset + face * 512);
            auto destination = reinterpret_cast<volatile uint32_t*>(output + face * 512);
            for (uint32_t word = 0; word < 8; ++word) { destination[word] = source[word]; }
        }
        asm volatile("" ::: "memory");
        noc_async_write_tile(page, checkpoint, output);
        noc_async_write(output, native.get_noc_addr(page, 0), 32);
        noc_async_write(output + 512, native.get_noc_addr(page, 512), 32);
        noc_async_write_barrier();
    }
}
