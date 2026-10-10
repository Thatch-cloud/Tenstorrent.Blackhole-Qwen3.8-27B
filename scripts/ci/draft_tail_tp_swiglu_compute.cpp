// The COMPUTE kernel of the drafter's SwiGLU launch (QWEN_FAST_DRAFT_TAIL, draft_tail_tp.py; tp4/fx-wp6, F-F3a).
//
// draft_mlp.swiglu_device runs nine launches on the fp32 gate and up projections, each a DRAM tensor in and out:
//     rounded_gate = bf16(gate)          rounded_up = bf16(up)
//     wide_gate = fp32(rounded_gate)     activated = silu(wide_gate)             (fp32)
//     rounded_activation = bf16(activated)
//     wide_activation = fp32(rounded_activation)   wide_up = fp32(rounded_up)
//     product = wide_activation * wide_up   (fp32, binary_ng's SFPU multiply)
//     activation = bf16(product)
// Every fp32(bf16(x)) is the value bf16(x) itself widened, so a destination register that holds the bf16-rounded value as an fp32 lane IS
// wide_gate / wide_activation / wide_up; only the four fp32 -> bf16 roundings (typecast_tile<Float32, Float16_b>, the same SFPU primitive as
// ttnn.typecast) and the silu and the multiply do work. They run here in that order on destination tiles 0 (gate) and 1 (up) and the result is
// packed once, as bf16, into CB 16. The fp32 inputs (CB 0, CB 1) are unpacked straight to the destination as the served ops do. The same
// rounding-then-widening pattern is draft_convolution_fused_compute.cpp's, which the card audit holds byte-equal to its composed ops.
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
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        silu_tile_init();
        silu_tile(0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
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
