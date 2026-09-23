// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "dataflow_common.hpp"

// [QWEN-SDPA-PF] reader_interleaved.cpp (sha256 f97f5490...) + R1-R9 of optimisation/ttnn-op/sdpa_prefill_chain
// (make_pf_reader.py; sdpa-prefill-share-spec.md 3.3). Selected by the factory only for
// SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0000 | flags with flag 0x1: the G6 unicast K/V chain.
namespace qwen_pf {
// Bounded equality waits: the semantics of Semaphore<>::wait (spin with an L1 invalidate) plus a poll
// bound. On expiry they NEVER fall through: waypoint + watcher assert, then spin forever. A hang stays a
// hang, so a stale slot is never pushed. The bound (2^28 polls, about 1-3 s) is orders of magnitude above
// any legal wait. Two functions so the waypoint names the wait: QWDC credit, QWDV VALID.
constexpr uint32_t kWdPolls = 1u << 28;
FORCE_INLINE void wd_wait_credit(uint32_t sem_id) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    for (uint32_t n = 0;; ++n) {
        invalidate_l1_cache();
        if (*p == 1) {
            return;
        }
        if (n == kWdPolls) {
            WAYPOINT("QWDC");
            ASSERT(false);
            for (;;) {  // never fall through; the volatile read keeps the loop well-defined
                (void)*p;
            }
        }
    }
}
FORCE_INLINE void wd_wait_valid(uint32_t sem_id) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    for (uint32_t n = 0;; ++n) {
        invalidate_l1_cache();
        if (*p == VALID) {
            return;
        }
        if (n == kWdPolls) {
            WAYPOINT("QWDV");
            ASSERT(false);
            for (;;) {
                (void)*p;
            }
        }
    }
}
// One round = one K chunk AND its V chunk (the K64f stage-3 round shape: one credit, one flag per chunk).
// The flag is ALWAYS reset (served R:416-418 order: reset before crediting), so the VALID wait that follows
// really waits for this round. withhold (test flag 0x200 only, the planted hang) skips just the credit: the
// sink then waits in QWDV and its upstream in QWDC, and no stale slot is ever pushed.
FORCE_INLINE void credit_prev(
    Noc noc, uint32_t receiver_sem_id, uint32_t sender_sem_id, uint32_t px, uint32_t py, bool withhold) {
    Semaphore<>(receiver_sem_id).set(INVALID);
    if (!withhold) {
        Semaphore<>(sender_sem_id).up(noc, px, py, 1);
    }
}
FORCE_INLINE void forward_round(
    Noc noc,
    uint32_t sender_sem_id,
    uint32_t receiver_sem_id,
    uint32_t valid_sem_id,
    uint32_t nx,
    uint32_t ny,
    uint32_t k_addr,
    uint32_t k_bytes,
    uint32_t v_addr,
    uint32_t v_bytes) {
    wd_wait_credit(sender_sem_id);  // next has reserved both slots of this round
    Semaphore<>(sender_sem_id).set(0);
    noc.async_write(
        CoreLocalMem<uint32_t>(k_addr), UnicastEndpoint{}, k_bytes, {}, {.noc_x = nx, .noc_y = ny, .addr = k_addr});
    noc.async_write(
        CoreLocalMem<uint32_t>(v_addr), UnicastEndpoint{}, v_bytes, {}, {.noc_x = nx, .noc_y = ny, .addr = v_addr});
    noc.async_write_barrier();  // data ACKED on next before the flag (K64f order)
    Semaphore<>(valid_sem_id).relay_unicast(noc, Semaphore<>(receiver_sem_id), nx, ny);
    noc.async_writes_flushed();  // no write left in flight before any later read barrier
}
}  // namespace qwen_pf

// Fetch a KV chunk into L1 for forwarding. No CB lifecycle — caller manages
// cb_reserve_back / cb_push_back. Single read barrier at end for lower latency.
template <uint32_t tile_bytes, bool transpose, typename ReaderType>
FORCE_INLINE void read_chunk_for_forwarding(
    const ReaderType& reader,
    const uint32_t cb_id,
    const uint32_t dst_addr,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t skip_src_cols = 0) {
    Noc noc;
    const uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    const uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t tile_id = start_tile_id;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = dst_addr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < src_cols; ++col) {
            noc.async_read(reader, CoreLocalMem<uint32_t>(write_ptr), tile_bytes, {.page_id = tile_id++}, {});
            write_ptr += inner_ptr_stride;
        }
        tile_id += skip_src_cols;
    }
    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_idx = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_zeros_async(noc, cb_id, tile_bytes, tile_idx * tile_bytes);
        }
    }
    // On WH/BH, async_write_zeros is implemented via noc_async_read from MEM_ZEROS_BASE,
    // so async_read_barrier() covers both real reads and zero-fills on the same path.
    // On Quasar, iDMA zero uses a separate completion path (iDMA ack) that needs its own barrier.
    noc.async_read_barrier();
#ifdef ARCH_QUASAR
    noc.write_zeros_l1_barrier();
#endif
}

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t NVH = get_compile_time_arg_val(3);
    constexpr uint32_t Sqt = get_compile_time_arg_val(4);
    constexpr uint32_t Skt = get_compile_time_arg_val(5);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(6);
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(7);
    constexpr uint32_t DHt = get_compile_time_arg_val(8);
    constexpr uint32_t vDHt = get_compile_time_arg_val(9);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(12);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t is_causal = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t broadcast_provided_mask_batch = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t broadcast_provided_mask_heads = get_compile_time_arg_val(18) == 1;
    [[maybe_unused]] constexpr uint32_t use_padded_mask = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t block_size_t = get_compile_time_arg_val(21);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(22);
    constexpr uint32_t use_attention_sink = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t use_mla = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t mla_kv_overlap = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(26);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(27);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(28) == 1;

    // Semaphore IDs for KV chain forwarding (non-causal only, but always present in compile args)
    constexpr uint32_t sender_semaphore_id = get_compile_time_arg_val(29);
    constexpr uint32_t receiver_semaphore_id = get_compile_time_arg_val(30);
    constexpr uint32_t valid_semaphore_id = get_compile_time_arg_val(31);
    constexpr bool mcast_enabled = get_compile_time_arg_val(32) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(33) == 1;
    // [QWEN-SDPA-PF] R2: this reader is only ever the causal flexible-chunked paged prefill chain reader.
    constexpr bool kQwenKvChain = true;
    static_assert(
        is_causal && is_chunked && !use_provided_mask && !use_attention_sink && !use_mla && !mcast_enabled &&
            sliding_window_size == 0 && !use_streaming_compute,
        "[QWEN-SDPA-PF] chain reader: causal flexible-chunked paged prefill only");

    constexpr auto q_args = TensorAccessorArgs<34>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto chunk_start_idx_args = TensorAccessorArgs<attention_sink_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t attention_sink_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunk_start_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_phases = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunked_q_chunk_offset_phase_1 = get_arg_val<uint32_t>(argidx++);
    const uint32_t read_offset_phase_1 = get_arg_val<uint32_t>(argidx++);
    uint32_t chunked_q_chunk_offset_phase_2 = 0;
    uint32_t read_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunked_q_chunk_offset_phase_2 = get_arg_val<uint32_t>(argidx++);
        read_offset_phase_2 = get_arg_val<uint32_t>(argidx++);
    }
    uint32_t chunked_q_chunk_offset_phase_1_local = chunked_q_chunk_offset_phase_1;
    uint32_t chunked_q_chunk_offset_phase_2_local = chunked_q_chunk_offset_phase_2;

    // Global Q scheduling runtime args (parsed after chain metadata so the host-side ordering
    // aligns for both causal and non-causal).
    uint32_t global_q_start = 0;
    uint32_t global_q_count = 0;

    // Parse chain metadata for KV forwarding (non-causal only)
    uint32_t is_chain_participant = 0;
    uint32_t is_injector = 0;
    uint32_t is_sink = 0;
    [[maybe_unused]] uint32_t chain_batch = 0;
    [[maybe_unused]] uint32_t chain_head = 0;
    uint32_t prev_physical_x = 0;
    uint32_t prev_physical_y = 0;
    uint32_t next_physical_x = 0;
    uint32_t next_physical_y = 0;
    [[maybe_unused]] uint32_t next_core_q_chunks = 0;
    [[maybe_unused]] uint32_t mcast_num_dests = 0;
    uint32_t mcast_sender_wait = 0;

    // Initialize NOC/semaphore state for chain forwarding
    [[maybe_unused]] uint32_t sender_wait_count = 1;

    if constexpr (!is_causal || kQwenKvChain) {  // [QWEN-SDPA-PF] R3: the causal G6 chain block (factory F5)
        is_chain_participant = get_arg_val<uint32_t>(argidx++);
        is_injector = get_arg_val<uint32_t>(argidx++);
        is_sink = get_arg_val<uint32_t>(argidx++);
        chain_batch = get_arg_val<uint32_t>(argidx++);
        chain_head = get_arg_val<uint32_t>(argidx++);
        argidx += 2;  // skip chain_q_chunk_start, chain_q_chunk_count (host-only metadata)
        prev_physical_x = get_arg_val<uint32_t>(argidx++);
        prev_physical_y = get_arg_val<uint32_t>(argidx++);
        next_physical_x = get_arg_val<uint32_t>(argidx++);
        next_physical_y = get_arg_val<uint32_t>(argidx++);
        next_core_q_chunks = get_arg_val<uint32_t>(argidx++);
        mcast_num_dests = get_arg_val<uint32_t>(argidx++);
        mcast_sender_wait = get_arg_val<uint32_t>(argidx++);

        if (is_chain_participant) {
            Semaphore<>(valid_semaphore_id).set(VALID);

            if constexpr (mcast_enabled) {
                if (is_injector) {
                    sender_wait_count = mcast_sender_wait;
                }
            }
        }
    }

    // Global Q scheduling runtime args sit right after chain metadata.
    global_q_start = get_arg_val<uint32_t>(argidx++);
    global_q_count = get_arg_val<uint32_t>(argidx++);

    // When chunked: only process K/V up to (chunk_start_idx + Q_chunk_length) tokens.
    // valid_Skt_bound = min(offset_tiles + valid_Sqt, valid_Skt); cap at valid_Skt for callers that pass
    // different valid_Sqt (e.g. ring_distributed uses full Q length in tiles).

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;

    constexpr uint32_t cb_arg_offset = chunk_start_idx_args.next_compile_time_args_offset();
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_id_page_table = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_id_chunk_start_idx_compute = get_compile_time_arg_val(cb_arg_offset + 6);
    constexpr uint32_t cb_id_chunk_start_idx_writer = get_compile_time_arg_val(cb_arg_offset + 7);
    // [QWEN-SDPA-PF] R4: the flags, a suffix CT arg after the CB ids (factory F7).
    constexpr uint32_t qwen_pf_flags = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr bool pf_inj_batch = (qwen_pf_flags & 0x2u) != 0;
    constexpr bool pf_test_mutate = (qwen_pf_flags & 0x100u) != 0;
    constexpr bool pf_test_hang = (qwen_pf_flags & 0x200u) != 0;
    static_assert((qwen_pf_flags & 0x1u) != 0, "[QWEN-SDPA-PF] the chain reader needs flag 0x1");
    static_assert(!pf_test_mutate || NKH >= 2, "[QWEN-SDPA-PF] the mutation flips the KV head");

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    [[maybe_unused]] constexpr uint32_t mask_tile_bytes = use_provided_mask ? get_tile_size(cb_mask_in) : 0;
    constexpr uint32_t attention_sink_tile_bytes = use_attention_sink ? get_tile_size(cb_attention_sink) : 0;

    constexpr uint32_t q_heads_per_k = NQH / NKH;
    constexpr uint32_t q_heads_per_v = NQH / NVH;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    // [QWEN-SDPA-PF] R5: only the injector's K/V DRAM reads change cadence under flag 0x2 (one barrier per
    // chunk, as read_chunk_for_forwarding). DMA batching only: same addresses, same bytes, same placement.
    const uint32_t kv_bt = (pf_inj_batch && is_injector) ? k_chunk_tiles : barrier_threshold;
    // A participant's Q read must be the subblock read, which R7 puts behind an atomic barrier (its credit
    // has landed); the whole-chunk Q read before the k loop has no such barrier (the BH read-barrier hazard).
    // The factory envelope refuses q_num_subblocks == 1 too (qk_in0_num_subblocks > 1).
    static_assert(use_q_subblock_push, "[QWEN-SDPA-PF] the chain reader needs the Q subblock push");

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);
    [[maybe_unused]] const auto mask_reader = TensorAccessor(mask_args, mask_addr);
    const auto attention_sink_reader = TensorAccessor(attention_sink_args, attention_sink_addr);
    const auto chunk_start_idx_reader = TensorAccessor(chunk_start_idx_args, chunk_start_idx_addr);

    constexpr uint32_t skip_src_cols = (use_mla && mla_kv_overlap) ? DHt - vDHt : 0;

    const auto q_tile_shape = TensorTileShape(B, NQH, valid_Sqt, DHt);
    [[maybe_unused]] const auto k_tile_shape = TensorTileShape(B, NKH, valid_Skt, DHt);

    // If we have MLA:
    // - if k and v tensors are overlapped, we want to read from the k tensor, but just a portion of it, hence setting
    // the v tile shape dim to DHt (and skip accordingly based on skip_src_cols)
    // - if k and v tensors are not overlapped, we want to read from the v tensor, hence setting the v tile shape dim to
    // vDHt Otherwise head dim of k and v are same
    [[maybe_unused]] const auto v_tile_shape = TensorTileShape(B, NVH, valid_Skt, use_mla && !mla_kv_overlap ? vDHt : DHt);
    const auto attention_sink_tile_shape = TensorTileShape(B, NQH, 1, 1);

    volatile tt_l1_ptr uint32_t* page_table_ptr;

    CircularBuffer cb_k(cb_k_in);
    CircularBuffer cb_v(cb_v_in);
    [[maybe_unused]] CircularBuffer cb_mask(cb_mask_in);
    CircularBuffer cb_attn_sink(cb_attention_sink);
    CircularBuffer cb_page_table(cb_id_page_table);

    uint32_t chunked_q_chunk_offset = 0;
    if constexpr (is_chunked) {
        if (chunk_start_idx_addr != 0) {
            CircularBuffer cb_chunk_compute(cb_id_chunk_start_idx_compute);
            cb_chunk_compute.reserve_back(1);
            uint32_t chunk_start_write_ptr = cb_chunk_compute.get_write_ptr();
            noc.async_read(
                chunk_start_idx_reader, CoreLocalMem<uint32_t>(chunk_start_write_ptr), 4, {.page_id = 0}, {});
            noc.async_read_barrier();
            uint32_t chunk_start_idx = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chunk_start_write_ptr);
            cb_chunk_compute.push_back(1);

            CircularBuffer cb_chunk_writer(cb_id_chunk_start_idx_writer);
            cb_chunk_writer.reserve_back(1);
            uint32_t chunk_start_write_ptr_2 = cb_chunk_writer.get_write_ptr();
            noc.async_read(
                chunk_start_idx_reader, CoreLocalMem<uint32_t>(chunk_start_write_ptr_2), 4, {.page_id = 0}, {});
            noc.async_read_barrier();
            cb_chunk_writer.push_back(1);

            const uint32_t q_chunk_size = Sq_chunk_t * tt::constants::TILE_HEIGHT;
            chunked_q_chunk_offset_phase_1_local = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunked_q_chunk_offset_phase_2_local = chunked_q_chunk_offset_phase_1_local;
            }
        }
    }
    uint32_t read_offset = 0;
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_1_local;
            read_offset = read_offset_phase_1;
        } else {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_2_local;
            read_offset = read_offset_phase_2;
        }
        uint32_t valid_Skt_bound;
        if (chunk_start_idx_addr != 0) {
            // Flexible or ring: cap at valid_Skt so we never read past K/V extent.
            valid_Skt_bound = std::min(chunked_q_chunk_offset * Sq_chunk_t + valid_Sqt, valid_Skt);
        } else {
            // Legacy: extend by offset so one program can serve all chunks (valid_Skt is chunk 0's).
            valid_Skt_bound = valid_Skt + chunked_q_chunk_offset * Sq_chunk_t;
        }

        // Global Q scheduling: iterate over a linear range of B*NQH*q_num_chunks chunks.
        // - per_head_q_iter resets on (nb, nq) transition: chain forwarding's
        //   `q_iter < next_core_q_chunks` gate expects this (chains are non-causal only).
        // - is_chunked: page-table read on nb transition, single-entry CB rotated forward.
        // - use_attention_sink: pushed every iter, since compute pops Sq_chunk_t per
        //   sdpa_inner_loop call and under global_q each iter is exactly one call.
        uint32_t prev_nb = static_cast<uint32_t>(-1);
        uint32_t prev_nq = static_cast<uint32_t>(-1);
        uint32_t per_head_q_iter = 0;
        [[maybe_unused]] uint32_t mask_batch_offset = 0;
        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {
            const auto decoded =
                decompose_global_q_index(global_q_start + global_q_iter, q_num_chunks, NQH, use_zigzag_balancing);
            if (decoded.nb != prev_nb) {
                if constexpr (!broadcast_provided_mask_batch) {
                    if constexpr (broadcast_provided_mask_heads) {
                        mask_batch_offset = decoded.nb * valid_Sqt * valid_Skt;
                    } else {
                        mask_batch_offset = decoded.nb * valid_Sqt * valid_Skt * NQH;
                    }
                }
                if constexpr (is_chunked) {
                    if (prev_nb != static_cast<uint32_t>(-1)) {
                        cb_page_table.pop_front(1);
                    }
                    cb_page_table.reserve_back(1);
                    page_table_ptr = read_page_table_for_batch(
                        noc, cb_id_page_table, decoded.nb, page_table_args, page_table_addr, page_table_stick_size);
                    cb_page_table.push_back(1);
                }
            }
            if (decoded.nb != prev_nb || decoded.nq != prev_nq) {
                per_head_q_iter = 0;
                prev_nb = decoded.nb;
                prev_nq = decoded.nq;
            }
            if constexpr (use_attention_sink) {
                constexpr uint32_t sink_tiles = use_streaming_compute ? 1 : Sq_chunk_t;
                cb_attn_sink.reserve_back(sink_tiles);
                uint32_t attention_sink_write_ptr = cb_attn_sink.get_write_ptr();
                const uint32_t sink_tile_id = attention_sink_tile_shape.id_of(0, decoded.nq, 0, 0);
                noc.async_read(
                    attention_sink_reader,
                    CoreLocalMem<uint32_t>(attention_sink_write_ptr),
                    attention_sink_tile_bytes,
                    {.page_id = sink_tile_id},
                    {});
                noc.async_read_barrier();
                if constexpr (!use_streaming_compute) {
                    fill_attention_sink_tiles<attention_sink_tile_bytes>(
                        cb_attention_sink, sink_tiles, attention_sink_write_ptr);
                }
                cb_attn_sink.push_back(sink_tiles);
            }

            const uint32_t nb = decoded.nb;
            const uint32_t nq = decoded.nq;
            uint32_t q_chunk = decoded.q_chunk;
            [[maybe_unused]] const uint32_t q_iter = per_head_q_iter;
            ++per_head_q_iter;

            /*
            Determine how many rows of Q will be read. Both start and end rows are
            capped by valid_Sqt, since Sq padding is independent of Sk padding.
            */
            const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
            const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
            const uint32_t q_row_tile_count = q_row_end_tile - q_row_start_tile;
            uint32_t q_read_tile_id = q_tile_shape.id_of(nb, nq, read_offset + q_row_start_tile, 0);

            // Q read is deferred into the K loop (k_chunk==0) for subblock interleaving.
            // When use_q_subblock_push is false, Q is read in full before the K loop (original behavior).
            if constexpr (!use_q_subblock_push) {
                read_chunk_with_padding<q_tile_bytes>(
                    q_reader, cb_q_in, q_read_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);
            }

            q_chunk = chunked_q_chunk_offset + q_chunk;
            uint32_t q_low_idx = q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
            uint32_t q_high_idx;
            if constexpr (is_causal) {
                // Clamp to total K-tile extent (Skt = k_num_chunks * Sk_chunk_t). Without
                // this, when Q-chunk extends past K (e.g., Sq_chunk_t > k_num_chunks*Sk_chunk_t),
                // the K-loop pushes more chunks than compute consumes → CB deadlock.
                const uint32_t q_high_unclamped = q_low_idx + Sq_chunk_t;
                q_high_idx = q_high_unclamped < Skt ? q_high_unclamped : Skt;
            } else {
                q_high_idx = Skt;
            }
            uint32_t k_loop_start = 0;
            if constexpr (use_streaming_compute && sliding_window_size > 0) {
                // Must match the compute kernel's K-loop bounds (see sliding_window_geometry.hpp).
                using window_geom =
                    SlidingWindowLoopGeometry<sliding_window_size, is_causal, tt::constants::TILE_HEIGHT>;
                constexpr uint32_t left_window_tiles = window_geom::left_window_tiles;
                constexpr uint32_t right_window_tiles = window_geom::right_window_tiles;
                if (q_low_idx > left_window_tiles) {
                    k_loop_start = (q_low_idx - left_window_tiles) / Sk_chunk_t;
                }
                if constexpr (!is_causal) {
                    const uint32_t window_high_unclamped = q_low_idx + Sq_chunk_t + right_window_tiles;
                    q_high_idx = window_high_unclamped < Skt ? window_high_unclamped : Skt;
                }
            }

            const uint32_t k_head = nq / q_heads_per_k;
            const uint32_t v_head = nq / q_heads_per_v;

            // [QWEN-SDPA-PF] R6: G6 - every member has the same unit list (factory F4), so every round is shared.
            const bool should_forward = is_chain_participant && !is_sink;
            const bool should_receive = is_chain_participant && !is_injector;
            (void)q_iter;

            // loop while k_low < q_high
            for (uint32_t k_chunk = k_loop_start; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                const uint32_t kv_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt_bound);
                const uint32_t kv_row_end_tile = std::min(kv_row_start_tile + Sk_chunk_t, valid_Skt_bound);
                const uint32_t kv_row_tile_count = kv_row_end_tile - kv_row_start_tile;
                // [QWEN-SDPA-PF] R7: one round per k_chunk (K and its V). Push order is the served one: K, (Q
                // subblocks at the unit's first chunk), V. Every member pushes to its own compute BEFORE
                // forwarding (served R:420 and the K64f leader order), which is safe because only this reader
                // ever rewrites these slots, two rounds later, after this forward's barrier.
                const uint32_t k_slot = cb_k.get_write_ptr();  // the reserves below target exactly these slots
                const uint32_t v_slot = cb_v.get_write_ptr();  // (reserve_back never moves the write pointer)
                const bool last_round =
                    (global_q_iter + 1 == global_q_count) && ((k_chunk + 1) * Sk_chunk_t >= q_high_idx);
                if (should_receive) {
                    cb_k.reserve_back(k_chunk_tiles);
                    cb_v.reserve_back(v_chunk_tiles);
                    qwen_pf::credit_prev(  // planted hang (0x200): the sink withholds its last credit, not the reset
                        noc,
                        receiver_semaphore_id,
                        sender_semaphore_id,
                        prev_physical_x,
                        prev_physical_y,
                        pf_test_hang && is_sink && last_round);
                    qwen_pf::wd_wait_valid(receiver_semaphore_id);  // K and V both landed (acked write, then flag)
                    cb_k.push_back(k_chunk_tiles);
                } else {
                    // The served K read (R:425-439): only the barrier cadence (kv_bt) and, under the test-only
                    // mutation flag, the injector's chunk-0 KV head differ.
                    const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
                    const uint32_t k_head_rd = (pf_test_mutate && is_injector && k_chunk == 0) ? (k_head ^ 1u) : k_head;
                    read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                        k_reader,
                        cb_k_in,
                        k_head_rd,
                        k_chunk_start_row_num,
                        kv_row_tile_count,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        k_tile_bytes,
                        kv_bt,
                        page_table_ptr,
                        true  // transpose=true for K reads
                    );
                }

                // Q subblock push (R:584-599), plus one barrier: a participant's credit atomic has completed
                // before the read barriers inside read_q_subblock.
                if constexpr (use_q_subblock_push) {
                    if (k_chunk == k_loop_start) {
                        if (is_chain_participant) {
                            noc.async_atomic_barrier();
                        }
                        for (uint32_t q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {
                            read_q_subblock<q_tile_bytes>(
                                q_reader,
                                cb_q_in,
                                q_read_tile_id,
                                q_sub * qk_subblock_h,
                                qk_subblock_h,
                                q_row_tile_count,
                                DHt,
                                DHt,
                                barrier_threshold);
                        }
                    }
                }

                if (should_receive) {
                    cb_v.push_back(v_chunk_tiles);
                } else {
                    // The served V read (R:617-632), kv_bt cadence.
                    const uint32_t kv_chunk_start_row_num = k_chunk * Sk_chunk_t;
                    constexpr uint32_t head_dim = (use_mla && !mla_kv_overlap) ? vDHt : DHt;
                    read_paged_chunk_with_padding<NVH, block_size_t, head_dim>(
                        v_reader,
                        cb_v_in,
                        v_head,
                        kv_chunk_start_row_num,
                        kv_row_tile_count,
                        vDHt,
                        Sk_chunk_t,
                        vDHt,
                        v_tile_bytes,
                        kv_bt,
                        page_table_ptr,
                        false,
                        skip_src_cols);
                }

                // Forward both slots of this round to the next member once it has reserved them.
                if (should_forward) {
                    qwen_pf::forward_round(
                        noc,
                        sender_semaphore_id,
                        receiver_semaphore_id,
                        valid_semaphore_id,
                        next_physical_x,
                        next_physical_y,
                        k_slot,
                        k_chunk_tiles * k_tile_bytes,
                        v_slot,
                        v_chunk_tiles * v_tile_bytes);
                }
            }  // close k_chunk
        }  // close global_q_iter
        if constexpr (is_chunked) {
            if (prev_nb != static_cast<uint32_t>(-1)) {
                cb_page_table.pop_front(1);
            }
        }
    }  // close phase
    if (is_chain_participant) {  // [QWEN-SDPA-PF] R8: nothing in flight at exit; semaphores back to their initial values
        noc.async_write_barrier();
        noc.async_atomic_barrier();
        Semaphore<>(receiver_semaphore_id).set(INVALID);
        Semaphore<>(sender_semaphore_id).set(INVALID);
    }
}
