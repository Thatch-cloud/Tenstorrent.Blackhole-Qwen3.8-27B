// The COMPUTE kernel of the drafter's local reduce (QWEN_FAST_DRAFT_REDUCE, draft_reduce_tp.py; tp4/fx-wp6, F-F1).
//
// For each output tile: the chips' partial tiles p0..p3 (CB 0, four fp32 pages, unpacked straight to the destination) go to destination
// tiles 0..3 and are summed in the served order with the SFPU fp32 add - ((p0 + p1) + p2) + p3, each sum back in destination tile 0 - then
// packed as one fp32 tile (CB 16). The served adds are binary_ng's SFPU fp32 add (ttnn.add of two fp32 tensors with dtype fp32), whose
// kernel is exactly add_binary_tile_init(); add_binary_tile(a, b, out); the same primitive as draft_convolution_fused_compute.cpp's, which
// the card audit holds byte-equal to the composed ops. Nothing is rounded between the adds in either path (fp32 DRAM in, fp32 DRAM out), so
// keeping the running sum in the destination is the same value as the served write-back and re-read.
//
// Compile-time args: [chips (2..4)]. Runtime args: [tile count].
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    constexpr uint32_t chips = get_compile_time_arg_val(0);
    const uint32_t count = get_arg_val<uint32_t>(0);
    init_sfpu(0, 16);
    for (uint32_t tile = 0; tile < count; ++tile) {
        cb_wait_front(0, chips);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(0);
        for (uint32_t chip = 0; chip < chips; ++chip) {
            copy_tile(0, chip, chip);
        }
        add_binary_tile_init();
        for (uint32_t chip = 1; chip < chips; ++chip) {
            add_binary_tile(0, chip, 0);
        }
        tile_regs_commit();
        cb_pop_front(0, chips);
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
    }
}
