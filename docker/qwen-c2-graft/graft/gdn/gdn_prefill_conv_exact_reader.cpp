// gdn_prefill_conv_exact reader (RISCV_1). See gdn_prefill_conv_exact.py.
//
// Per unit (R tile rows of one tile column ct): each qkv tile is read ONCE into a 3-slot ring
// (cur / prev / prefetch); the tile above the unit's first row (or the carry / zero tile at the
// top of the column) is its halo. For every tile the reader pushes four tiles to compute:
//   c_0 = x shifted down by 3 (tap 0)   c_1 = by 2 (tap 1)   c_2 = by 1 (tap 2)   c_3 = x (tap 3)
// A shifted tile is 8 face-row copies (gdn_prefill_conv_exact.shift_copies): rows s..31 from
// the current tile, rows 0..s-1 from the halo. On the column's ht_v tile it also builds
// new_state (rows x[vl-3..vl-1]) into c_8 for the writer. Taps (and the carry tile) are read
// once per column the core visits and pushed to compute through c_4.
//
// Every DRAM / L1-bank read is a full 2048-byte page; sub-page traffic is L1-local only.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace pcx {
constexpr uint32_t cb_x3 = 0;
constexpr uint32_t cb_x2 = 1;
constexpr uint32_t cb_x1 = 2;
constexpr uint32_t cb_x0 = 3;
constexpr uint32_t cb_taps = 4;
constexpr uint32_t cb_ring = 7;
constexpr uint32_t cb_state = 8;
constexpr uint32_t cb_halo = 10;
constexpr uint32_t page = 2048;
constexpr uint32_t face = 512;
constexpr uint32_t row = 32;
constexpr uint32_t state_rows = 3;

constexpr uint32_t face_offset(uint32_t face_row, uint32_t half) { return (2 * face_row + half) * face; }

// L1 -> L1 copy of n bytes (a multiple of 32). NOC loopback by default; PCX_SHIFT_WORDS copies
// with the RISC instead (the section 7.1 microbench keeps the faster).
FORCE_INLINE void lcopy(uint32_t src, uint32_t dst, uint32_t n) {
#ifdef PCX_SHIFT_WORDS
    auto s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    auto d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < n / 4; ++i) {
        d[i] = s[i];
    }
#else
    noc_async_read(get_noc_addr(src), dst, n);
#endif
}

// x shifted down by s rows into dst: rows s..31 from cur, rows 0..s-1 from prev (the last s rows
// of the x tile above when prev_is_x, else rows 3-s..2 of the carry / zero tile).
FORCE_INLINE void shift_tile(uint32_t s, uint32_t cur, uint32_t prev, bool prev_is_x, uint32_t dst) {
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t upper = face_offset(0, half);
        const uint32_t lower = face_offset(1, half);
        const uint32_t halo = prev_is_x ? lower + row * (16 - s) : upper + row * (state_rows - s);
        lcopy(cur + upper, dst + upper + row * s, (16 - s) * row);
        lcopy(prev + halo, dst + upper, row * s);
        lcopy(cur + lower, dst + lower + row * s, (16 - s) * row);
        lcopy(cur + upper + row * (16 - s), dst + lower, row * s);
    }
}

FORCE_INLINE void zero_tile(uint32_t address) {
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t i = 0; i < page / 4; ++i) {
        words[i] = 0;
    }
}

// Word copy of one 16-element face row (32 bytes) - the state tile is built by the RISC, after
// every source has landed, so no NOC ordering is involved.
FORCE_INLINE void copy_row(uint32_t src, uint32_t dst) {
    auto s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    auto d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < row / 4; ++i) {
        d[i] = s[i];
    }
}

// The FIR's state is a one-hot matmul: -0 (0x8000) comes out +0; under CANON_DENORM a denormal
// (exponent bits zero) is flushed to +0 as well. Applied to both bf16 halves of a word.
FORCE_INLINE uint32_t canonical_half(uint32_t value) {
    if (value == 0x8000u) {
        return 0;
    }
#ifdef CANON_DENORM
    if ((value & 0x7F80u) == 0) {
        return 0;
    }
#endif
    return value;
}

FORCE_INLINE uint32_t canonical_pair(uint32_t word) {
    return canonical_half(word & 0xFFFFu) | (canonical_half(word >> 16) << 16);
}

// PCX_FLUSH_X_DENORM (1: a denormal -> +0, 2: -> a zero of its own sign). In the FIR, x_padded
// goes through untilize / concat / tilize (and taps 1-3 and the static-slice state through a
// second untilize / slice / tilize). If that round trip flushes bf16 denormals, every x the op
// reads must be flushed the same way BEFORE it is shifted, multiplied or copied into new_state -
// on both state paths, independent of `canon`. The op flushes each x / carry tile in L1 once,
// after it lands; every staging tile (c_0..c_3) and state row is copied from those. The card-M
// run records which setting matches (gdn_prefill_conv_exact.FLUSH_X_DENORM_DEFAULT). Exact +-0
// and normal values are never touched.
#ifdef PCX_FLUSH_X_DENORM
FORCE_INLINE uint32_t flush_half(uint32_t value) {
    if ((value & 0x7F80u) == 0 && (value & 0x7Fu) != 0) {
#if PCX_FLUSH_X_DENORM == 2
        return value & 0x8000u;
#else
        return 0;
#endif
    }
    return value;
}

// Idempotent, so a tile flushed twice (a halo reused from the ring) is unchanged.
FORCE_INLINE void flush_tile(uint32_t address) {
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t i = 0; i < page / 4; ++i) {
        const uint32_t word = words[i];
        words[i] = flush_half(word & 0xFFFFu) | (flush_half(word >> 16) << 16);
    }
    asm volatile("" ::: "memory");
}
#define PCX_FLUSH(address) flush_tile(address)
#else
#define PCX_FLUSH(address) ((void)0)
#endif
}  // namespace pcx

void kernel_main() {
    using namespace pcx;
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t strips = get_compile_time_arg_val(3);
#ifdef NEG_STALE
    constexpr bool has_carry = false;
#else
    constexpr bool has_carry = get_compile_time_arg_val(5) != 0;
#endif
    constexpr auto qkv_args = TensorAccessorArgs<6>();
    constexpr auto carry_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<carry_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();

    const auto qkv = TensorAccessor(qkv_args, get_common_arg_val<uint32_t>(0), page);
    const auto carry = TensorAccessor(carry_args, get_common_arg_val<uint32_t>(1), page);
    const auto tap0 = TensorAccessor(tap0_args, get_common_arg_val<uint32_t>(2), page);
    const auto tap1 = TensorAccessor(tap1_args, get_common_arg_val<uint32_t>(3), page);
    const auto tap2 = TensorAccessor(tap2_args, get_common_arg_val<uint32_t>(4), page);
    const auto tap3 = TensorAccessor(tap3_args, get_common_arg_val<uint32_t>(5), page);
    const uint32_t valid_len = get_common_arg_val<uint32_t>(10);
    const uint32_t canon = get_common_arg_val<uint32_t>(11);
    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);

    const uint32_t ring_base = get_write_ptr(cb_ring);
    const uint32_t zero = get_write_ptr(cb_halo);
    const uint32_t carry_tile = zero + page;
    zero_tile(zero);
    const uint32_t ht_v = (valid_len - 1) / 32;

    uint32_t last_ct = 0xFFFFFFFFu;  // column and row of the last x tile left in the ring
    uint32_t last_ht = 0xFFFFFFFFu;
    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        const uint32_t ct = unit / strips;
        const uint32_t ht0 = (unit % strips) * R;
#ifndef PCX_MB_SHIFT_ONLY
        if (ct != last_ct) {
            cb_reserve_back(cb_taps, 4);
            const uint32_t taps = get_write_ptr(cb_taps);
            noc_async_read_tile(ct, tap0, taps);
            noc_async_read_tile(ct, tap1, taps + page);
            noc_async_read_tile(ct, tap2, taps + 2 * page);
            noc_async_read_tile(ct, tap3, taps + 3 * page);
            if constexpr (has_carry) {
                noc_async_read_tile(ct, carry, carry_tile);
            }
            noc_async_read_barrier();
            if constexpr (has_carry) {
                PCX_FLUSH(carry_tile);
            }
            cb_push_back(cb_taps, 4);
        }
#else
        if (has_carry && ct != last_ct) {
            noc_async_read_tile(ct, carry, carry_tile);
            noc_async_read_barrier();
            PCX_FLUSH(carry_tile);
        }
#endif
        const bool top = ht0 == 0;
        const uint32_t halo = top ? (has_carry ? carry_tile : zero) : ring_base + ((ht0 - 1) % 3) * page;
        if (!top && !(ct == last_ct && ht0 - 1 == last_ht)) {
            noc_async_read_tile((ht0 - 1) * Ct + ct, qkv, halo);
        }
        noc_async_read_tile(ht0 * Ct + ct, qkv, ring_base + (ht0 % 3) * page);
        for (uint32_t ht = ht0; ht < ht0 + R; ++ht) {
            noc_async_read_barrier();  // cur (and the halo) have landed
            if (ht + 1 < ht0 + R) {
                // Slot ht+1 is neither cur (ht) nor prev (ht-1).
                noc_async_read_tile((ht + 1) * Ct + ct, qkv, ring_base + ((ht + 1) % 3) * page);
            }
            const uint32_t cur = ring_base + (ht % 3) * page;
            const uint32_t prev = ht == ht0 ? halo : ring_base + ((ht - 1) % 3) * page;
            const bool prev_is_x = ht == ht0 ? !top : true;
            // Under PCX_FLUSH_X_DENORM: cur (and, at a unit's first row, its x halo) has landed and
            // the in-flight prefetch targets the third slot, so flushing here races nothing.
            PCX_FLUSH(cur);
            if (ht == ht0 && !top) {
                PCX_FLUSH(halo);
            }

            cb_reserve_back(cb_x0, 1);
            lcopy(cur, get_write_ptr(cb_x0), page);
            cb_reserve_back(cb_x3, 1);
            shift_tile(3, cur, prev, prev_is_x, get_write_ptr(cb_x3));
            cb_reserve_back(cb_x2, 1);
            shift_tile(2, cur, prev, prev_is_x, get_write_ptr(cb_x2));
            cb_reserve_back(cb_x1, 1);
#ifdef NEG_SHIFT
            shift_tile(2, cur, prev, prev_is_x, get_write_ptr(cb_x1));  // negative control: tap 2 gets shift 2
#else
            shift_tile(1, cur, prev, prev_is_x, get_write_ptr(cb_x1));
#endif
            noc_async_read_barrier();  // the local copies are done (the prefetch too; 2 KB)

#ifndef PCX_MB_SHIFT_ONLY
            if (ht == ht_v) {
                // new_state row j <- x[valid_len - 3 + j] (gdn_prefill_conv_exact.state_rows).
                cb_reserve_back(cb_state, 1);
                const uint32_t state = get_write_ptr(cb_state);
                zero_tile(state);
                for (uint32_t j = 0; j < state_rows; ++j) {
                    const int32_t source = static_cast<int32_t>(valid_len) - static_cast<int32_t>(state_rows) +
                                           static_cast<int32_t>(j);
                    uint32_t base;
                    uint32_t r;
                    if (source >= static_cast<int32_t>(32 * ht_v)) {
                        base = cur;
                        r = static_cast<uint32_t>(source) - 32 * ht_v;
                    } else if (source >= 0) {
                        base = prev;  // ht_v > 0 here, and prev is the x tile above
                        r = static_cast<uint32_t>(source) - 32 * (ht_v - 1);
                    } else {
                        base = has_carry ? carry_tile : zero;
                        r = static_cast<uint32_t>(static_cast<int32_t>(state_rows) + source);
                    }
                    for (uint32_t half = 0; half < 2; ++half) {
                        copy_row(base + face_offset(r / 16, half) + row * (r % 16), state + face_offset(0, half) + row * j);
                    }
                }
                if (canon) {
                    // The FIR's state is a one-hot matmul: an exact -0 comes out +0 (and, if the
                    // device test says so, a denormal is flushed). Rows 0..2 of both upper faces.
                    // (The x round-trip flush, PCX_FLUSH_X_DENORM, already applied on both paths.)
                    // Whole 32-bit words (two bf16 each), so no sub-word L1 stores.
                    for (uint32_t half = 0; half < 2; ++half) {
                        auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state + face_offset(0, half));
                        for (uint32_t i = 0; i < state_rows * 8; ++i) {
                            words[i] = canonical_pair(words[i]);
                        }
                    }
                }
                asm volatile("" ::: "memory");
                cb_push_back(cb_state, 1);
            }
#endif
            asm volatile("" ::: "memory");
            cb_push_back(cb_x3, 1);
            cb_push_back(cb_x2, 1);
            cb_push_back(cb_x1, 1);
            cb_push_back(cb_x0, 1);
        }
        last_ct = ct;
        last_ht = ht0 + R - 1;
    }
}
