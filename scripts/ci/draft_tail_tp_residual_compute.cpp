// The COMPUTE kernel of the drafter's residual tail (QWEN_FAST_DRAFT_TAIL, draft_tail_tp.py; tp4/fx-wp6, F-F3a).
//
// The served tail of a branch is four launches on the bf16 convolution output `finished` and the bf16 block `hidden`:
//     wide_finished = fp32(finished)    wide_hidden = fp32(hidden)    summed = wide_finished + wide_hidden (fp32, binary_ng's SFPU add)
//     output = bf16(summed)
// A bf16 value widened to fp32 is exact, so the two widenings are the copy into the destination (the unpacker widens bf16 for the
// fp32 destination, as draft_convolution_fused_compute.cpp relies on); the add is add_binary_tile; the last launch is typecast_tile<Float32,
// Float16_b>. One tile at a time, packed once, as bf16, into CB 16.
//
// Runtime args: [tile count].
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    init_sfpu(0, 16);
    for (uint32_t tile = 0; tile < count; ++tile) {
        cb_wait_front(0, 1);
        cb_wait_front(1, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(0);
        copy_tile(0, 0, 0);
        copy_tile_to_dst_init_short(1);
        copy_tile(1, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit();
        cb_pop_front(0, 1);
        cb_pop_front(1, 1);
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
    }
}
