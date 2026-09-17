#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0), 32);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1), 32);
    const uint32_t workers = get_arg_val<uint32_t>(2);
    const uint32_t tiles = get_arg_val<uint32_t>(3);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t worker = 0; worker < workers; ++worker) {
        noc_async_read(input.get_noc_addr(worker), scratch + worker * 64, 32);
    }
    noc_async_read_barrier();
    uint32_t best_key = 0;
    uint32_t best_token = 0xffffffff;
    uint32_t best_bits = 0;
    uint32_t invalid = 0;
    for (uint32_t worker = 0; worker < workers; ++worker) {
        const auto record = reinterpret_cast<volatile const uint32_t*>(scratch + worker * 64);
        const uint32_t bits = record[0];
        const uint32_t token = record[1];
        if (record[2] != 0 || (bits & 0x7f800000) == 0x7f800000 ||
            token < (worker * tiles / workers) * 32 || token >= ((worker + 1) * tiles / workers) * 32) {
            invalid = 1;
            continue;
        }
        for (uint32_t word = 3; word < 8; ++word) { invalid |= record[word] != 0; }
        const uint32_t canonical = (bits & 0x7fffffff) == 0 ? 0 : bits;
        const uint32_t key = (canonical & 0x80000000) ? ~canonical : canonical ^ 0x80000000;
        if (best_token == 0xffffffff || key > best_key || (key == best_key && token < best_token)) {
            best_key = key;
            best_token = token;
            best_bits = bits;
        }
    }
    auto result = reinterpret_cast<volatile uint32_t*>(scratch + workers * 64);
    result[0] = invalid ? 0xffffffff : best_token;
    result[1] = invalid;
    result[2] = best_bits;
    for (uint32_t word = 3; word < 8; ++word) { result[word] = 0; }
    asm volatile("" ::: "memory");
    noc_async_write(scratch + workers * 64, output.get_noc_addr(0), 32);
    noc_async_write_barrier();
}
