// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include <vector>

#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    /*
    In DRAM, Q is (B, PNHt, DHt), K is (B, St, DHt), V is (B, St, DHt), mask is (B, PNHt, PSt)
    We want to read for a particular batch cur_batch, and sequence length up to padded layer length.
    We read Q: (cur_batch, PNHt, DHt), K: (cur_batch, PSt, DHt), V: (cur_batch, PSt, DHt), mask: (cur_batch, PNHt, PSt)
    */
    constexpr uint32_t B = get_compile_time_arg_val(0);           // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);        // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);          // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);         // head dim
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);        // head dim of V
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr bool is_q_sharded = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(8);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(9);
    constexpr uint32_t index_stick_size_B = get_compile_time_arg_val(10);
    constexpr bool is_paged_attention = get_compile_time_arg_val(11) == 1;
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(12);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(13);
    constexpr uint32_t Bkv = get_compile_time_arg_val(14);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(15);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(16);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(17);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(18);
    constexpr bool is_causal = get_compile_time_arg_val(19) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(20) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(22);
    constexpr bool tilize_q = get_compile_time_arg_val(23) == 1;
    constexpr bool reuse_k = get_compile_time_arg_val(24) == 1;
    constexpr bool use_half_tile = get_compile_time_arg_val(25);
    constexpr uint32_t q_chunk_size_bytes = get_compile_time_arg_val(26);
    constexpr bool is_cur_pos_tensor_sharded = get_compile_time_arg_val(27);
    constexpr bool is_page_table_sharded = get_compile_time_arg_val(28);
    constexpr uint32_t q_page_size_bytes = get_compile_time_arg_val(29);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(30);
    constexpr uint32_t original_block_size = get_compile_time_arg_val(31);
    constexpr bool has_block_padding = is_paged_attention && original_block_size > 0 && original_block_size < 32;
    constexpr uint32_t k_mcast_semaphore_id = get_compile_time_arg_val(32);
    constexpr bool q_locally_available = get_compile_time_arg_val(33) == 1;
    constexpr bool use_k_mcast = get_compile_time_arg_val(34) == 1;
    constexpr uint32_t Bmask = get_compile_time_arg_val(35);
    // 0 = unbounded cache (legacy); nonzero = wrap virtual tile index mod this value
    // before page_table lookup. Value is in TILE rows (= cache_position_modulo /
    // TILE_HEIGHT). Validated to be a multiple of block_size_t at op level.
    constexpr uint32_t capacity_t = get_compile_time_arg_val(36);

    constexpr auto q_args = TensorAccessorArgs<37>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    // [QWEN-SDPA] suffix compile-time args (factory F6), after every TensorAccessorArgs block, so the
    // legacy offsets never move. This file is reader_decode_all.cpp (sha256 49a05926) plus the edits
    // R1-R3 of optimisation/ttnn-op/sdpa_decode_qwen; the factory selects it only in [QWEN-SDPA] mode.
    constexpr uint32_t qwen_cta = attention_sink_args.next_compile_time_args_offset();
    constexpr bool mask_tail = get_compile_time_arg_val(qwen_cta + 0) == 1;
    constexpr uint32_t mask_width_t = get_compile_time_arg_val(qwen_cta + 1);
    constexpr bool kv_share = get_compile_time_arg_val(qwen_cta + 2) == 1;
    constexpr uint32_t kv_ready_semaphore_id = get_compile_time_arg_val(qwen_cta + 3);
    static_assert(!(kv_share && use_k_mcast), "KV share and the MLA K multicast are exclusive");
    static_assert(!(kv_share && reuse_k), "KV share needs an explicit V tensor");
    static_assert(!mask_tail || (Sk_chunk_t > 0 && St % Sk_chunk_t == 0), "Tail mask needs a fixed chunk size");
    // [QWEN-SDPA] stage-4 suffix compile-time args (factory F16), after F6's four. This file is the stage-3
    // reader_decode_qwen.cpp (sha256 280a847f) plus the edits R6-R9 of optimisation/ttnn-op/sdpa_decode_slice;
    // the factory selects it for flag 0x4 (q-slice) or 0x8 (K/V read-ahead) only. PNHt (CTA 1) is the
    // slice's row tiles under 0x4; Q and the mask keep pnht_full row tiles per entry.
    constexpr uint32_t pnht_full = get_compile_time_arg_val(qwen_cta + 4);
    constexpr uint32_t rows_per_kv = get_compile_time_arg_val(qwen_cta + 5);
    constexpr bool kv_readahead = get_compile_time_arg_val(qwen_cta + 6) == 1;
    constexpr bool q_slice = get_compile_time_arg_val(qwen_cta + 7) == 1;
    static_assert(get_compile_time_arg_val(qwen_cta + 8) == 0x51CE, "Slice reader ABI tag: factory F16 and R6 disagree");
    static_assert(q_slice || PNHt == pnht_full, "Without the slice every row tile is read");
    static_assert(!q_slice || (num_kv_heads - 1) * rows_per_kv / 32 + PNHt <= pnht_full,
                  "The last KV head's slice runs past Q");
    static_assert(!kv_readahead || kv_share, "K/V read-ahead needs KV share");
    // [QWEN-SDPA] K64j R10 (optimisation/ttnn-op/k64j): the runtime extent (flag 0x20), the last reader suffix
    // (factory F21: +9, after F16's five). Under it this non-causal kernel takes the causal block's cur_pos read
    // below: each entry's E - 1 from the cur_pos tensor, or UINT32_MAX to skip the entry.
    constexpr bool runtime_extent = get_compile_time_arg_val(qwen_cta + 9) == 1;
    static_assert(!runtime_extent || mask_tail, "The runtime extent needs the tail mask");
    static_assert(!runtime_extent || mask_width_t == Sk_chunk_t, "The runtime extent reads the narrow one-chunk mask");
    static_assert(!runtime_extent || !is_causal, "The runtime extent is a non-causal mode");
    static_assert(!runtime_extent || !is_cur_pos_tensor_sharded, "The runtime extent reads an interleaved cur_pos tensor");

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    // #44366: cur_pos is consumed by both the writer (c_8) and compute (c_15).
    // Using one shared CB races — whichever consumer pops first drains the
    // count and the other hangs waiting for tiles. Each consumer gets its own CB.
    constexpr uint32_t cb_writer_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_9;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_compute_cur_pos = tt::CBIndex::c_15;

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t attention_sink_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(arg_idx++);
    const bool is_worker = get_arg_val<uint32_t>(arg_idx++) == 0;
    const bool is_output_core = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);
    const bool do_k_mcast = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_y0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_y1 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_dests = get_arg_val<uint32_t>(arg_idx++);
    // [QWEN-SDPA] R6: this KV head's first Q row tile under the slice (the factory's F14 rule); 0 without it.
    const uint32_t q_tile_start = q_slice ? (cur_head_group * rows_per_kv) >> 5 : 0;

    // idle core
    if (q_addr == 0) {
        return;
    }

    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    // [QWEN-SDPA] K64j R10: a runtime-extent program (non-causal) takes this causal block as it is. Both CB copies
    // (c_8 for the writer, c_15 for compute) are pushed before the UINT32_MAX test, and the skip returns before any
    // Q, K or V read and before the KV-share handshake: nothing below has run yet.
    if constexpr (is_causal || runtime_extent) {
        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            // Reader fills cb_writer_cur_pos (c_8) first (from DRAM, or via the
            // aliased sharded buffer) then copies the same stick into
            // cb_compute_cur_pos (c_15) via an L1->L1 read.
            CircularBuffer cb_writer(cb_writer_cur_pos);
            cb_writer.reserve_back(1);
            uint32_t index_cb_wr_ptr = cb_writer.get_write_ptr();
            if constexpr (!is_cur_pos_tensor_sharded) {
                const auto addrg = TensorAccessor(pos_args, pos_addr);
                // index_tensor has one page to read
                noc.async_read(addrg, CoreLocalMem<uint32_t>(index_cb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
                noc.async_read_barrier();
            }
            CircularBuffer cb_compute(cb_compute_cur_pos);
            cb_compute.reserve_back(1);
            uint32_t index_cb_compute_wr_ptr = cb_compute.get_write_ptr();
            const uint8_t noc_id = noc.get_noc_id();
            const uint32_t my_noc_x = my_x[noc_id];
            const uint32_t my_noc_y = my_y[noc_id];
            UnicastEndpoint pos_src;
            noc.async_read(
                pos_src,
                CoreLocalMem<uint32_t>(index_cb_compute_wr_ptr),
                index_stick_size_B,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = index_cb_wr_ptr},
                {});
            noc.async_read_barrier();
            if constexpr (runtime_extent && kv_share) {
                // [QWEN-SDPA] K64j R10: under KV share the leader and its twins run one READY/VALID round per
                // chunk, so every entry must split the same extent. Each entry takes slot 0 and writes it into its
                // own slot of both copies before they are pushed: its writer (c_8) and compute (c_15) split slot 0
                // too, and a skipped slot 0 skips the whole bundle.
                volatile tt_l1_ptr uint32_t* writer_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
                volatile tt_l1_ptr uint32_t* compute_words =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_compute_wr_ptr);
                const uint32_t slot0 = writer_words[0];
                writer_words[cur_batch / q_heads_parallel_factor] = slot0;
                compute_words[cur_batch / q_heads_parallel_factor] = slot0;
            }
            cb_writer.push_back(1);
            cb_compute.push_back(1);
            volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
            cur_pos = index_ptr[cur_batch / q_heads_parallel_factor];
        }
        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    // When block_size < TILE_HEIGHT, each tile has zero-padded rows. Convert cur_pos from
    // the original sequence space to the padded tile space so get_runtime_args computes the
    // correct number of tiles to process. Only needed for causal mode where cur_pos comes
    // from user input; non-causal uses cur_pos_base which is already in the padded space.
    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            cur_batch,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
    uint32_t output_core_noc_y = all_output_noc_y[cur_batch];

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;
    uint32_t k_chunk_tiles = Sk_chunk_t_dynamic * DHt;
    uint32_t v_chunk_tiles = Sk_chunk_t_dynamic * vDHt;
    uint32_t mask_chunk_tiles = PNHt * Sk_chunk_t_dynamic;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr uint32_t attention_sink_tile_bytes = get_tile_size(cb_attention_sink);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    // Read Q entirely - always read into cb_q_in
    // When tilize_q is true, compute will tilize back to cb_q_in
    // When tilize_q is false, Q is already tilized
    // [QWEN-SDPA] R7: entry cur_batch's Q holds pnht_full row tiles (q_chunk_tiles is the slice's); this
    // head's slice starts q_tile_start row tiles in.
    const uint32_t q_batch_offset = cur_batch * pnht_full * DHt + q_tile_start * DHt;

    // Read Q
    read_q<cb_q_in, cb_q_rm, q_tile_bytes, q_chunk_tiles, is_q_sharded, tilize_q, use_half_tile, barrier_threshold>(
        q_locally_available,
        is_output_core,
        q_addr,
        output_core_noc_x,
        output_core_noc_y,
        q_chunk_size_bytes,
        q_args,
        q_page_size_bytes,
        q_batch_offset);

    const auto k_reader = TensorAccessor(k_args, k_addr);

    const auto v_reader = TensorAccessor(v_args, v_addr);

    const auto mask_reader = TensorAccessor(mask_args, mask_addr);

    // Read attention sink
    if constexpr (use_attention_sink) {
        const auto attention_sink_reader = TensorAccessor(attention_sink_args, attention_sink_addr);

        CircularBuffer cb_sink(cb_attention_sink);
        cb_sink.reserve_back(PNHt);
        uint32_t attention_sink_write_ptr = cb_sink.get_write_ptr();

        for (uint32_t tile = 0; tile < PNHt; ++tile) {
            // Use noc.async_read with explicit size instead of noc.async_read_page because
            // the CB may use half tiles (16x32) while the DRAM buffer stores full tiles (32x32).
            // noc.async_read_page would read buffer->aligned_page_size() bytes, overflowing the CB.
            noc.async_read(
                attention_sink_reader,
                CoreLocalMem<uint32_t>(attention_sink_write_ptr),
                attention_sink_tile_bytes,
                {.page_id = tile},
                {});
            attention_sink_write_ptr += attention_sink_tile_bytes;
        }
        noc.async_read_barrier();
        cb_sink.push_back(PNHt);
    }

    // Read page table
    volatile tt_l1_ptr uint32_t* page_table_ptr;
    uint32_t page_table_cb_wr_ptr = 0;
    volatile tt_l1_ptr uint16_t* page_table_ptr_u16 = nullptr;
    volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
    if constexpr (is_paged_attention) {
        CircularBuffer cb_page_table(cb_id_page_table);
        uint32_t num_pages_to_read = is_page_table_sharded ? B : 1;
        cb_page_table.reserve_back(num_pages_to_read);
        // Read page table from DRAM
        if constexpr (!is_page_table_sharded) {
            page_table_ptr = read_page_table_for_batch(
                noc,
                cb_id_page_table,
                cur_batch / q_heads_parallel_factor,
                page_table_args,
                page_table_addr,
                page_table_page_size);
            page_table_ptr_u32 = page_table_ptr;
        } else {  // Read page table from dynamically allocated L1 buffer
            page_table_cb_wr_ptr =
                cb_page_table.get_write_ptr() + (cur_batch / q_heads_parallel_factor) * page_table_page_size;
            page_table_ptr_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_cb_wr_ptr);
        }
        cb_page_table.push_back(num_pages_to_read);
    }

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        // Mask row and batch strides are the mask's own width: St for a full-width mask (== PSt
        // for a non-causal full-window call), Sk_chunk_t for a narrow tail mask. Tail mode reads
        // one fixed chunk: the last Sk_chunk_t columns of the mask.
        // [QWEN-SDPA] R8: the mask's batch stride is its full row count (pnht_full); this head's rows start
        // q_tile_start row tiles in, and read_mask_chunk<PNHt> then reads the slice's rows.
        const uint32_t mask_batch_offset =
            ((cur_batch / q_heads_parallel_factor) % Bmask) * pnht_full * mask_width_t + q_tile_start * mask_width_t;
        const uint32_t mask_chunk_offset = mask_tail ? (mask_width_t - Sk_chunk_t) : k_chunk_start * Sk_chunk_t_dynamic;
        uint32_t mask_start_tile_id = mask_batch_offset + mask_chunk_offset;
        // Setup multicast parameters for K streaming (vertical multicast)
        KMcastParams k_mcast_params = {
            .do_mcast = do_k_mcast,
            .mcast_x = mcast_x,
            .mcast_y0 = mcast_y0,
            .mcast_y1 = mcast_y1,
            .num_dests = num_dests,
            .mcast_sem_id = k_mcast_semaphore_id};

        if constexpr (is_paged_attention) {
            // [QWEN-SDPA] R9 (K1b): the read-ahead leader's K and V slots of the chunk it multicasts next.
            uint32_t ra_k_slot = 0;
            uint32_t ra_v_slot = 0;
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
                const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t_dynamic;
                uint64_t k_base_read_ptr;

                if constexpr (kv_share) {
                    // [QWEN-SDPA] KV share (factory F10-F12, spec 7.2). The entries of this bundle read the
                    // same page-table row, so entry 0 (the LEADER, do_k_mcast) reads each K and V chunk from
                    // DRAM exactly as the legacy path does and multicasts both CB slots to its B-1 twins, which
                    // sit directly below it in this column and never read K or V themselves. One READY/VALID
                    // round per chunk covers K and V. Every core has the same CB layout (factory: every CB
                    // spans the whole grid) and the same chunk range, so k_slot / v_slot are the same L1
                    // address on every twin.
                    CircularBuffer cb_k(cb_k_in);
                    CircularBuffer cb_v(cb_v_in);
                    Semaphore<> kv_valid(k_mcast_semaphore_id);
                    if (kv_readahead && do_k_mcast) {
                        // [QWEN-SDPA] R9, K1b (flag 0x8): the READ-AHEAD LEADER. The stage-3 leader (the next
                        // branch) reads chunk n, waits for READY, multicasts it and only then reads n+1, so its
                        // period is T_read + T_mcast + the handshake. Here chunk n was read one iteration early
                        // (by the prologue for the first chunk), so the read of n+1 runs while n's multicast is
                        // in flight: wait READY(n); multicast K(n) and V(n) (non-blocking); read chunk n's mask
                        // if it has one; read K(n+1) and V(n+1) into the other ring slot (never past
                        // k_chunk_end); write barrier; VALID(n). Slot n % 2 is next written by chunk n+2's read,
                        // in the next iteration, after this iteration's write barrier. The bytes, the CB order,
                        // the READY/VALID rounds and the twins are the stage-3 ones; only when reads land moves.
                        if (k_chunk == k_chunk_start) {
                            ra_k_slot = read_k<
                                cb_k_in,
                                DHt,
                                num_kv_heads,
                                block_size_t,
                                k_tile_bytes,
                                barrier_threshold,
                                is_page_table_sharded,
                                false,
                                capacity_t>(
                                k_chunk_tiles,
                                cur_head,
                                Sk_chunk_t_dynamic,
                                k_chunk_start_row_num,
                                k_reader,
                                page_table_ptr_u16,
                                page_table_ptr_u32,
                                barrier_count);
                            ra_v_slot = cb_v.get_write_ptr();  // read_v reserves exactly this slot
                            read_v<
                                cb_v_in,
                                vDHt,
                                num_kv_heads,
                                block_size_t,
                                v_tile_bytes,
                                barrier_threshold,
                                is_page_table_sharded,
                                false,
                                capacity_t>(
                                v_chunk_tiles,
                                cur_head,
                                Sk_chunk_t_dynamic,
                                k_chunk_start_row_num,
                                v_reader,
                                page_table_ptr_u16,
                                page_table_ptr_u32,
                                barrier_count,
                                ra_k_slot,
                                k_tile_bytes);
                        }
                        const uint32_t k_slot = ra_k_slot;  // chunk n, read by the prologue or the last iteration
                        const uint32_t v_slot = ra_v_slot;
                        Semaphore<> kv_ready(kv_ready_semaphore_id);
                        kv_ready.wait(num_dests);  // every twin has reserved this chunk's K and V slots
                        kv_ready.set(0);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(k_slot),
                            MulticastEndpoint{},
                            k_chunk_tiles * k_tile_bytes,
                            num_dests,
                            {},
                            {.noc_x_start = mcast_x,
                             .noc_y_start = mcast_y0,
                             .noc_x_end = mcast_x,
                             .noc_y_end = mcast_y1,
                             .addr = k_slot},
                            false);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(v_slot),
                            MulticastEndpoint{},
                            v_chunk_tiles * v_tile_bytes,
                            num_dests,
                            {},
                            {.noc_x_start = mcast_x,
                             .noc_y_start = mcast_y0,
                             .noc_x_end = mcast_x,
                             .noc_y_end = mcast_y1,
                             .addr = v_slot},
                            false);
                        if constexpr (use_attention_mask) {  // this chunk's mask before chunk n+1's K/V
                            if (!mask_tail || k_chunk == k_num_chunks - 1) {
                                mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
                                    mask_width_t, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);
                            }
                        }
                        if (k_chunk + 1 < k_chunk_end) {
                            const uint32_t next_row_num = k_chunk_start_row_num + Sk_chunk_t_dynamic;
                            ra_k_slot = read_k<
                                cb_k_in,
                                DHt,
                                num_kv_heads,
                                block_size_t,
                                k_tile_bytes,
                                barrier_threshold,
                                is_page_table_sharded,
                                false,
                                capacity_t>(
                                k_chunk_tiles,
                                cur_head,
                                Sk_chunk_t_dynamic,
                                next_row_num,
                                k_reader,
                                page_table_ptr_u16,
                                page_table_ptr_u32,
                                barrier_count);
                            ra_v_slot = cb_v.get_write_ptr();  // read_v reserves exactly this slot
                            read_v<
                                cb_v_in,
                                vDHt,
                                num_kv_heads,
                                block_size_t,
                                v_tile_bytes,
                                barrier_threshold,
                                is_page_table_sharded,
                                false,
                                capacity_t>(
                                v_chunk_tiles,
                                cur_head,
                                Sk_chunk_t_dynamic,
                                next_row_num,
                                v_reader,
                                page_table_ptr_u16,
                                page_table_ptr_u32,
                                barrier_count,
                                ra_k_slot,
                                k_tile_bytes);
                        }
                        noc.async_write_barrier();  // chunk n has landed on every twin before its flag
                        kv_valid.set(1);
                        kv_valid.set_multicast(noc, mcast_x, mcast_y0, mcast_x, mcast_y1, num_dests);
                    } else if (do_k_mcast) {
                        // LEADER: the legacy DRAM reads byte for byte (same functions, no MLA multicast),
                        // then one multicast of both slots to the twins once every twin has reserved them.
                        const uint32_t k_slot = read_k<
                            cb_k_in,
                            DHt,
                            num_kv_heads,
                            block_size_t,
                            k_tile_bytes,
                            barrier_threshold,
                            is_page_table_sharded,
                            false,
                            capacity_t>(
                            k_chunk_tiles,
                            cur_head,
                            Sk_chunk_t_dynamic,
                            k_chunk_start_row_num,
                            k_reader,
                            page_table_ptr_u16,
                            page_table_ptr_u32,
                            barrier_count);
                        const uint32_t v_slot = cb_v.get_write_ptr();  // read_v reserves exactly this slot
                        read_v<
                            cb_v_in,
                            vDHt,
                            num_kv_heads,
                            block_size_t,
                            v_tile_bytes,
                            barrier_threshold,
                            is_page_table_sharded,
                            false,
                            capacity_t>(
                            v_chunk_tiles,
                            cur_head,
                            Sk_chunk_t_dynamic,
                            k_chunk_start_row_num,
                            v_reader,
                            page_table_ptr_u16,
                            page_table_ptr_u32,
                            barrier_count,
                            k_slot,
                            k_tile_bytes);
                        Semaphore<> kv_ready(kv_ready_semaphore_id);
                        kv_ready.wait(num_dests);  // every twin has reserved this chunk's K and V slots
                        kv_ready.set(0);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(k_slot),
                            MulticastEndpoint{},
                            k_chunk_tiles * k_tile_bytes,
                            num_dests,
                            {},
                            {.noc_x_start = mcast_x,
                             .noc_y_start = mcast_y0,
                             .noc_x_end = mcast_x,
                             .noc_y_end = mcast_y1,
                             .addr = k_slot},
                            false);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(v_slot),
                            MulticastEndpoint{},
                            v_chunk_tiles * v_tile_bytes,
                            num_dests,
                            {},
                            {.noc_x_start = mcast_x,
                             .noc_y_start = mcast_y0,
                             .noc_x_end = mcast_x,
                             .noc_y_end = mcast_y1,
                             .addr = v_slot},
                            false);
                        noc.async_write_barrier();  // the data has landed before the flag
                        kv_valid.set(1);
                        kv_valid.set_multicast(noc, mcast_x, mcast_y0, mcast_x, mcast_y1, num_dests);
                    } else {
                        // TWIN (entry 1..B-1): reserve both slots (its compute has popped chunk n-2), reset
                        // VALID, signal READY to the leader (mcast_x/mcast_y0 = the leader's NoC coordinate),
                        // wait for the bytes, then hand them to compute as if read here.
                        cb_k.reserve_back(k_chunk_tiles);
                        cb_v.reserve_back(v_chunk_tiles);
                        kv_valid.set(0);
                        Semaphore<>(kv_ready_semaphore_id).up(noc, mcast_x, mcast_y0, 1);
                        kv_valid.wait(1);
                        cb_k.push_back(k_chunk_tiles);
                        cb_v.push_back(v_chunk_tiles);
                    }
                    if constexpr (use_attention_mask) {  // each entry still reads its own mask, after K and V
                        // [QWEN-SDPA] R9: a read-ahead leader read this chunk's mask above, before chunk n+1's K/V.
                        if ((!mask_tail || k_chunk == k_num_chunks - 1) && !(kv_readahead && do_k_mcast)) {
                            mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
                                mask_width_t, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);
                        }
                    }
                } else {
                    // Read K chunk - supports both multicast and non-multicast paths
                    k_base_read_ptr = read_k<
                        cb_k_in,
                        DHt,
                        num_kv_heads,
                        block_size_t,
                        k_tile_bytes,
                        barrier_threshold,
                        is_page_table_sharded,
                        use_k_mcast,
                        capacity_t>(
                        k_chunk_tiles,
                        cur_head,
                        Sk_chunk_t_dynamic,
                        k_chunk_start_row_num,
                        k_reader,
                        page_table_ptr_u16,
                        page_table_ptr_u32,
                        barrier_count,
                        k_mcast_params);

                    if constexpr (use_attention_mask) {
                        // Tail: only the head's final k-chunk (always core_num_in_reduce 0) is masked;
                        // compute applies the mask under the identical predicate (C2).
                        if (!mask_tail || k_chunk == k_num_chunks - 1) {
                            mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
                                mask_width_t, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);
                        }
                    }
                    // Read V chunk - either from DRAM or from K's L1 buffer (transpose) when reuse_k is true
                    read_v<
                        cb_v_in,
                        vDHt,
                        num_kv_heads,
                        block_size_t,
                        v_tile_bytes,
                        barrier_threshold,
                        is_page_table_sharded,
                        reuse_k,
                        capacity_t>(
                        v_chunk_tiles,
                        cur_head,
                        Sk_chunk_t_dynamic,
                        k_chunk_start_row_num,
                        v_reader,
                        page_table_ptr_u16,
                        page_table_ptr_u32,
                        barrier_count,
                        k_base_read_ptr,
                        k_tile_bytes);
                }
            }
        } else {
            // Offset for current batch
            const uint32_t k_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * DHt;
            const uint32_t k_head_offset = cur_head * St * DHt;

            // Then, read K, V, Mask k_chunk_tiles at a time
            const uint32_t k_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic * DHt;
            uint32_t k_start_tile_id = k_batch_offset + k_head_offset + k_chunk_offset;

            // V has its own layout when it's an independent tensor (width = vDHt, not DHt)
            const uint32_t v_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * vDHt;
            const uint32_t v_head_offset = cur_head * St * vDHt;
            const uint32_t v_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic * vDHt;
            uint32_t v_start_tile_id = v_batch_offset + v_head_offset + v_chunk_offset;

            read_kv_mask_chunks<
                DHt,
                vDHt,
                barrier_threshold,
                mask_tile_bytes,
                PNHt,
                use_attention_mask,
                cb_k_in,
                cb_v_in,
                cb_mask_in,
                reuse_k>(
                k_chunk_start,
                k_chunk_end,
                k_start_tile_id,
                v_start_tile_id,
                mask_start_tile_id,
                Sk_chunk_t_dynamic,
                k_chunk_tiles,
                v_chunk_tiles,
                mask_chunk_tiles,
                k_reader,
                v_reader,
                mask_reader,
                k_tile_bytes,
                v_tile_bytes,
                PSt);
        }
    }
    if constexpr (kv_share) {
        // [QWEN-SDPA] KV share (spec 7.2 R5): no NoC transaction left in flight and VALID back at 0
        // when the kernel ends (the fused_1d_input lesson: an unbarriered multicast exit hung the
        // stack). The early returns above all come before any semaphore traffic.
        if (do_k_mcast) {
            noc.async_write_barrier();
        } else {
            noc.async_atomic_barrier();
        }
        Semaphore<>(k_mcast_semaphore_id).set(0);
    }
}
