// gdn_prefill_conv_exact writer (RISCV_0). See gdn_prefill_conv_exact.py.
//
// Routes each conv output tile (column ct, row ht) to q, k or v - the three ttnn.slice the
// served path makes of the conv output - and writes the column's new_state tile after the ht_v
// tile. Exactly one strip per column holds ht_v, so every state page has exactly one writer.
// All writes are full pages. PCX_MB_SHIFT_ONLY (section 7.1 movement microbench): no compute
// kernel; c_0 (x shifted by 3) goes to the one [1, T, C] output, c_1..c_3 are drained.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace pcx {
constexpr uint32_t cb_x3 = 0;
constexpr uint32_t cb_x2 = 1;
constexpr uint32_t cb_x1 = 2;
constexpr uint32_t cb_x0 = 3;
constexpr uint32_t cb_out = 6;
constexpr uint32_t cb_state = 8;
constexpr uint32_t page = 2048;
}  // namespace pcx

void kernel_main() {
    using namespace pcx;
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t strips = get_compile_time_arg_val(3);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t v_tiles = Ct - 2 * q_tiles;
    constexpr auto qkv_args = TensorAccessorArgs<6>();
    constexpr auto carry_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<carry_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const auto q = TensorAccessor(q_args, get_common_arg_val<uint32_t>(6), page);
    const auto k = TensorAccessor(k_args, get_common_arg_val<uint32_t>(7), page);
    const auto v = TensorAccessor(v_args, get_common_arg_val<uint32_t>(8), page);
    const auto state = TensorAccessor(state_args, get_common_arg_val<uint32_t>(9), page);
    const uint32_t valid_len = get_common_arg_val<uint32_t>(10);
    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);
    const uint32_t ht_v = (valid_len - 1) / 32;

    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        const uint32_t ct = unit / strips;
        const uint32_t ht0 = (unit % strips) * R;
        for (uint32_t ht = ht0; ht < ht0 + R; ++ht) {
#ifdef PCX_MB_SHIFT_ONLY
            cb_wait_front(cb_x3, 1);
            noc_async_write_tile(ht * Ct + ct, q, get_read_ptr(cb_x3));
            noc_async_writes_flushed();
            cb_pop_front(cb_x3, 1);
            cb_wait_front(cb_x2, 1);
            cb_pop_front(cb_x2, 1);
            cb_wait_front(cb_x1, 1);
            cb_pop_front(cb_x1, 1);
            cb_wait_front(cb_x0, 1);
            cb_pop_front(cb_x0, 1);
#else
            cb_wait_front(cb_out, 1);
            const uint32_t source = get_read_ptr(cb_out);
            if (ct < q_tiles) {
                noc_async_write_tile(ht * q_tiles + ct, q, source);
            } else if (ct < 2 * q_tiles) {
                noc_async_write_tile(ht * q_tiles + (ct - q_tiles), k, source);
            } else {
                noc_async_write_tile(ht * v_tiles + (ct - 2 * q_tiles), v, source);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
            if (ht == ht_v) {
                cb_wait_front(cb_state, 1);
                noc_async_write_tile(ct, state, get_read_ptr(cb_state));
                noc_async_writes_flushed();
                cb_pop_front(cb_state, 1);
            }
#endif
        }
    }
    noc_async_write_barrier();
}
