// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"
#include "windowed_mask_gen.hpp"

// [QWEN-SDPA-KS] the slices of one q chunk's K stream: the same integer arithmetic in the reader, the writer and the compute kernel (ksplit_planner.py holds the Python twin).
namespace qwen_ks {
constexpr uint32_t kMinBlocksPerSlice = 4;
FORCE_INLINE uint32_t effective_slices(uint32_t blocks, uint32_t slices) {
    const uint32_t by_blocks = blocks / kMinBlocksPerSlice;
    const uint32_t taking_part = by_blocks < slices ? by_blocks : slices;
    return taking_part > 0 ? taking_part : 1;
}
FORCE_INLINE uint32_t slice_begin(uint32_t blocks, uint32_t slice, uint32_t eff) { return (slice * blocks) / eff; }
}  // namespace qwen_ks

// [QWEN-SDPA-KS] the hand-off of a slice's partial result to the next slice (the pinned compute kernel's ring merge reads it from cb_prev_out / cb_lse_in there).
namespace qwen_ks_xfer {
// Bounded wait for the receiving slice's credit: the semantics of the chain reader's wd_wait_credit (2^28 polls, then waypoint + watcher assert + a spin that never falls through).
constexpr uint32_t kWdPolls = 1u << 28;
FORCE_INLINE void wd_wait_credit(uint32_t sem_id) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    for (uint32_t n = 0;; ++n) {
        invalidate_l1_cache();
        if (*p == 1) {
            return;
        }
        if (n == kWdPolls) {
            WAYPOINT("QKDC");
            ASSERT(false);
            for (;;) {
                (void)*p;
            }
        }
    }
}
FORCE_INLINE void send_partial(
    Noc noc,
    uint32_t cb_out_id,
    uint32_t cb_lse_out_id,
    uint32_t cb_prev_out_id,
    uint32_t cb_lse_in_id,
    uint32_t out_tiles,
    uint32_t lse_tiles,
    uint32_t out_bytes,
    uint32_t lse_bytes,
    uint32_t credit_sem_id,
    uint32_t valid_src_sem_id,
    uint32_t valid_dst_sem_id,
    uint32_t nx,
    uint32_t ny) {
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_lse(cb_lse_out_id);
    cb_out.wait_front(out_tiles);  // the compute kernel made the whole normalised output and its log-sum-exp
    cb_lse.wait_front(lse_tiles);
    wd_wait_credit(credit_sem_id);  // the next slice has reserved cb_prev_out and cb_lse_in
    Semaphore<>(credit_sem_id).set(0);
    const uint32_t out_dst = CircularBuffer(cb_prev_out_id).get_write_ptr();  // the same address on every core (one CB layout)
    const uint32_t lse_dst = CircularBuffer(cb_lse_in_id).get_write_ptr();
    noc.async_write(
        CoreLocalMem<uint32_t>(cb_out.get_read_ptr()), UnicastEndpoint{}, out_bytes, {}, {.noc_x = nx, .noc_y = ny, .addr = out_dst});
    noc.async_write(
        CoreLocalMem<uint32_t>(cb_lse.get_read_ptr()), UnicastEndpoint{}, lse_bytes, {}, {.noc_x = nx, .noc_y = ny, .addr = lse_dst});
    noc.async_write_barrier();  // both writes ACKED on the next slice before its flag (the chain's order)
    Semaphore<>(valid_src_sem_id).relay_unicast(noc, Semaphore<>(valid_dst_sem_id), nx, ny);
    noc.async_writes_flushed();
    cb_out.pop_front(out_tiles);
    cb_lse.pop_front(lse_tiles);
}
}  // namespace qwen_ks_xfer

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(4);
    constexpr uint32_t unpadded_Sk = get_compile_time_arg_val(5);
    constexpr uint32_t DHt = get_compile_time_arg_val(6);
    constexpr uint32_t vDHt = get_compile_time_arg_val(7);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(12);
    constexpr uint32_t scale_val = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t is_causal = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(19);
    constexpr bool use_lightweight_mask = get_compile_time_arg_val(20) == 1;
    constexpr bool use_streaming_compute = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(22);
    constexpr uint32_t k_partial_col = get_compile_time_arg_val(23);
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    // Windowed (block-diagonal) mask generation flags. Fixed scalar slots BEFORE the tensor-accessor
    // block so the accessor offset chain stays intact for all configs.
    constexpr bool use_windowed_mask = get_compile_time_arg_val(25) == 1;

    // out accessor, then the cu_window accessor chained immediately after it (before the CB-id block).
    constexpr auto out_args = TensorAccessorArgs<26>();
    constexpr auto cu_window_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t num_phases = get_arg_val<uint32_t>(2);
    const uint32_t use_chunk_start_idx_tensor = get_arg_val<uint32_t>(3);
    uint32_t chunk_start_t_in_q_chunks_phase_1 = get_arg_val<uint32_t>(4);
    const uint32_t write_offset_phase_1 = get_arg_val<uint32_t>(5);
    uint32_t chunk_start_t_in_q_chunks_phase_2 = 0;
    uint32_t write_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunk_start_t_in_q_chunks_phase_2 = get_arg_val<uint32_t>(6);
        write_offset_phase_2 = get_arg_val<uint32_t>(7);
    }

    // Global Q scheduling args follow phase_2 args.
    const uint32_t global_q_start = get_arg_val<uint32_t>(8);
    const uint32_t global_q_count = get_arg_val<uint32_t>(9);
    const uint32_t cu_window_seqlens_addr = get_arg_val<uint32_t>(10);
    const uint32_t cu_window_seqlens_eles = get_arg_val<uint32_t>(11);
    // [QWEN-SDPA-KS] W3: the slice of this core and the next slice's core (the partial goes there); factory PK13
    const uint32_t ks_slice = get_arg_val<uint32_t>(12);
    const uint32_t ks_next_x = get_arg_val<uint32_t>(13);
    const uint32_t ks_next_y = get_arg_val<uint32_t>(14);

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;  // non-streaming drain only

    constexpr uint32_t cb_arg_offset = cu_window_args.next_compile_time_args_offset();
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_chunk_start_idx = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 4);
    // cu_window CB id lives in the CB-id block (appended by CBIds for windowed mode; inactive otherwise).
    constexpr uint32_t cb_cu_window_in = get_compile_time_arg_val(cb_arg_offset + 5);
    // [QWEN-SDPA-KS] W4: the suffix of the ksplit factory (PK6): slices, Skt, the CBs of the hand-off and of the merge scale, the three semaphores, the flags
    constexpr uint32_t ks_slices = get_compile_time_arg_val(cb_arg_offset + 6);
    constexpr uint32_t ks_Skt = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t ks_cb_lse_out = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t ks_cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 9);
    constexpr uint32_t ks_cb_lse_in = get_compile_time_arg_val(cb_arg_offset + 10);
    constexpr uint32_t ks_cb_scale_in = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t ks_credit_sem = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t ks_valid_src_sem = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t ks_valid_dst_sem = get_compile_time_arg_val(cb_arg_offset + 14);
    constexpr uint32_t ks_flags = get_compile_time_arg_val(cb_arg_offset + 15);
    static_assert((ks_flags & 0x19u) == 0x19u, "[QWEN-SDPA-KS] the ksplit writer needs the flags 0x1, 0x8 and 0x10");
    static_assert(!use_streaming_compute, "[QWEN-SDPA-KS] the ring merge is the non-streaming path");
    // the scale as the packed bf16 pair generate_bcast_col_scalar takes (the factory refuses a scale that is not exact in bf16: PK8)
    constexpr uint32_t ks_scale_packed = (scale_val & 0xFFFF0000u) | (scale_val >> 16);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    const auto out_writer = TensorAccessor(out_args, out_addr);

    const auto out_tile_shape = TensorTileShape(B, NQH, valid_Sqt, vDHt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);
    generate_bcast_col_scalar(CircularBuffer(ks_cb_scale_in), ks_scale_packed);  // [QWEN-SDPA-KS] W5: the ring merge multiplies the running max by this scalar

    // Lightweight mask: generate template tiles once, leave permanently fronted.
    // Sliding layout: [neginf, trailing_primary, leading_prev, leading_current, trailing_next, k_partial?].
    // Non-sliding layout: [neginf, causal_diag?, k_partial?].
    if constexpr (use_lightweight_mask) {
        // is_causal handles K-partial via causal stamp; skip emitting partial tile in causal mode.
        constexpr uint32_t writer_partial_col = is_causal ? 0u : k_partial_col;
        generate_lightweight_mask_tiles<
            writer_partial_col,
            /*joint_l*/ 0u,
            cb_mask_in,
            is_causal,
            sliding_window_size>(noc);
    }

    // Windowed: load cu_window_seqlens into L1 once; the writer synthesizes the block-diagonal mask per
    // Q chunk from it (so the reader streams Q/K/V only).
    if constexpr (use_windowed_mask) {
        const auto cu_window_reader = TensorAccessor(cu_window_args, cu_window_seqlens_addr);
        constexpr uint32_t cu_tile_bytes = get_tile_size(cb_cu_window_in);
        CircularBuffer cb_cu(cb_cu_window_in);
        cb_cu.reserve_back(1);
        noc.async_read(
            cu_window_reader, CoreLocalMem<uint32_t>(cb_cu.get_write_ptr()), cu_tile_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
        cb_cu.push_back(1);
    }

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            CircularBuffer cb_chunk_start(cb_chunk_start_idx);
            cb_chunk_start.wait_front(1);
            auto chunk_start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_chunk_start.get_read_ptr());
            uint32_t chunk_start_idx = chunk_start_ptr[0];
            cb_chunk_start.pop_front(1);
            const uint32_t q_chunk_size = Sq_chunk_t * tt::constants::TILE_HEIGHT;
            chunk_start_t_in_q_chunks_phase_1 = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunk_start_t_in_q_chunks_phase_2 = chunk_start_t_in_q_chunks_phase_1;
            }
        }
    }

    uint32_t chunk_start_t_in_q_chunks = 0;
    uint32_t write_offset = 0;
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_1;
            write_offset = write_offset_phase_1;
        } else {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_2;
            write_offset = write_offset_phase_2;
        }
        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {
            const auto decoded =
                decompose_global_q_index(global_q_start + global_q_iter, q_num_chunks, NQH, use_zigzag_balancing);
            const uint32_t nb = decoded.nb;
            const uint32_t nq = decoded.nq;
            const uint32_t q_chunk = decoded.q_chunk;

            // Generate masks only for legacy generated-mask variants. No-mask variants do not allocate cb_mask_in.
            if constexpr (
                !use_provided_mask && !use_lightweight_mask &&
                (is_causal || sliding_window_size > 0 || use_padded_mask)) {
                generate_mask<is_chunked, sliding_window_size, use_padded_mask, cb_mask_in>(
                    noc,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    q_chunk,
                    chunk_start_t_in_q_chunks,
                    true,
                    false,
                    unpadded_Sk,
                    0,
                    is_causal);
            }

            // Windowed: synthesize this Q chunk's block-diagonal mask (all K chunks) before draining its
            // output. The call is a template wrapper that only instantiates the generator when
            // use_windowed_mask is true (kernel_main is not a template, so a bare `if constexpr` here
            // would still compile the discarded body). valid_Skt derived from the unpadded K length.
            constexpr uint32_t windowed_valid_Skt =
                (unpadded_Sk + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
            windowed_generate_if_enabled<use_windowed_mask, cb_mask_in, cb_cu_window_in>(
                noc,
                q_chunk,
                Sq_chunk_t,
                Sk_chunk_t,
                valid_Sqt,
                windowed_valid_Skt,
                k_num_chunks,
                cu_window_seqlens_eles);

            // Determine how many rows of OUT will be written. Both start and end rows are
            // capped by valid_Sqt, since Sq padding is independent of Sk padding.
            const uint32_t out_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
            const uint32_t out_row_end_tile = std::min(out_row_start_tile + Sq_chunk_t, valid_Sqt);
            const uint32_t out_row_tile_count = out_row_end_tile - out_row_start_tile;
            uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile, 0);
            // [QWEN-SDPA-KS] W6: B blocks in this q chunk (the chunk start is the one the reader pushed), E slices take part: an idle slice made nothing; a slice that is not the last hands
            // its partial to the next one and writes nothing; only the last slice (the one with the diagonal block) writes the output below
            const uint32_t ks_q_high = std::min((chunk_start_t_in_q_chunks + q_chunk + 1) * Sq_chunk_t, ks_Skt);
            const uint32_t ks_blocks = (ks_q_high + Sk_chunk_t - 1) / Sk_chunk_t;
            const uint32_t ks_eff = qwen_ks::effective_slices(ks_blocks, ks_slices);
            if (ks_slice >= ks_eff) {
                continue;
            }
            if (ks_slice + 1 < ks_eff) {
                qwen_ks_xfer::send_partial(
                    noc,
                    cb_out,
                    ks_cb_lse_out,
                    ks_cb_prev_out,
                    ks_cb_lse_in,
                    out_chunk_tiles,
                    Sq_chunk_t,
                    out_chunk_tiles * tile_bytes,
                    Sq_chunk_t * get_tile_size(ks_cb_lse_out),
                    ks_credit_sem,
                    ks_valid_src_sem,
                    ks_valid_dst_sem,
                    ks_next_x,
                    ks_next_y);
                continue;
            }
            if constexpr (use_streaming_compute) {
                // Streaming: drain per row-group (cb_out is a 2-slot ping-pong).
                // Compute always pushes Sq_chunk_t rows; rows past out_row_tile_count
                // are padding and get popped without being written.
                write_block_row_grouped(
                    noc,
                    out_writer,
                    cb_out,
                    Sq_chunk_t,
                    out_row_tile_count,
                    vDHt,
                    out_tile_id,
                    tile_bytes,
                    out_subblock_h,
                    barrier_threshold);
            } else {
                write_block(
                    noc,
                    out_writer,
                    cb_out,
                    out_chunk_tiles,
                    out_row_tile_count,
                    vDHt,
                    out_tile_id,
                    tile_bytes,
                    barrier_threshold);
            }
            // [QWEN-SDPA-KS] W7: the ring path of the compute kernel always makes the log-sum-exp; the last slice has no use for it
            CircularBuffer ks_cb_lse_drop(ks_cb_lse_out);
            ks_cb_lse_drop.wait_front(Sq_chunk_t);
            ks_cb_lse_drop.pop_front(Sq_chunk_t);
        }
    }  // close phase
}
