// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// #include "api/debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {
    Noc noc;

    uint32_t in_tile_offset_by_head = get_arg_val<uint32_t>(0);
    uint32_t q_start_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_q_out = get_compile_time_arg_val(2);
    constexpr uint32_t head_size = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t head_size_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t PHASES_TO_READ =
        get_compile_time_arg_val(6);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

    constexpr uint32_t num_x = get_compile_time_arg_val(7);
    constexpr uint32_t num_y = get_compile_time_arg_val(8);
    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + num_x));

    CircularBuffer cb_q_out(cb_id_q_out);
    UnicastEndpoint src_ep;

    // Q
    uint32_t qkv_x = 0;
    uint32_t qkv_y = 0;
    uint32_t total_input_cores = num_x * num_y;
    uint32_t num_tiles_per_core = (head_size_num_tiles * batch) / total_input_cores;

    uint32_t qkv_noc_x = in0_mcast_noc_x[qkv_x];
    uint32_t qkv_noc_y = in0_mcast_noc_y[qkv_y];
    uint32_t qkv_read_addr = q_start_addr + in_tile_offset_by_head;
    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    uint32_t tile_size = head_size / head_size_num_tiles;
    const uint32_t cb_write_ptr_base = cb_q_out.get_write_ptr();

    // Rows of the output shard per tile row. The offsets below (16, 512) already assume a 32x32 tile
    // with 16x16 faces, so one tile row of the output shard is 32 users tall.
    constexpr uint32_t TILE_ROWS = 32;

    for (uint32_t q = 0; q < batch; ++q) {
        // The output shard is (batch x head_dim) in tile layout: ceil(batch / 32) tile rows of
        // head_size_num_tiles tiles each, tiles row-major, so one whole tile row is head_size bytes.
        // User q therefore lives at row (q % 32) of batch tile (q / 32). Within one 32x32 tile, rows
        // 0..15 are face 0 and rows 16..31 are face 2, which starts 512 elements into the tile.
        // At batch <= 32 the batch tile index is always 0 and this is byte-for-byte the original offset.
        uint32_t batch_tile = q / TILE_ROWS;
        uint32_t row_in_tile = q - batch_tile * TILE_ROWS;
        uint32_t wptr_offset =
            batch_tile * head_size + (row_in_tile < 16
                                          ? row_in_tile * SUBTILE_LINE_BYTES
                                          : (row_in_tile - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE);
        uint32_t q_write_addr = cb_write_ptr_base + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc.async_read(
                    src_ep,
                    CoreLocalMem<uint32_t>(q_write_addr),
                    SUBTILE_LINE_BYTES,
                    {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr},
                    {});
                // noc.async_read_barrier();
            }
            // Read second phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc.async_read(
                    src_ep,
                    CoreLocalMem<uint32_t>(q_write_addr + 256 * ELEMENT_SIZE),
                    SUBTILE_LINE_BYTES,
                    {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr + 256 * ELEMENT_SIZE},
                    {});
                // noc.async_read_barrier();
            }
            // noc.async_read_barrier();

            qkv_read_addr += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                qkv_x++;
                if (qkv_x == num_x) {
                    qkv_x = 0;
                    qkv_y++;
                }
                qkv_noc_x = in0_mcast_noc_x[qkv_x];
                qkv_noc_y = in0_mcast_noc_y[qkv_y];
                qkv_read_addr = q_start_addr + in_tile_offset_by_head;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    noc.async_read_barrier();
}
