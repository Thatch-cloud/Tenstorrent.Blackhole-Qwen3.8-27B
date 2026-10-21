"""Make writer_interleaved_qwen_ksplit.cpp (edits W1-W7) from the pinned writer_interleaved.cpp (sha 2a0959ff..., tt-metal 9f9cd4fd; committed as fixtures/).

Prefill lever PF3-K (flag 0x10 of apply_factory_pk.py). The factory (PK11) selects this file only for flag 0x10. It is the pinned writer with the hand-off added:

    W1  the two API headers the chain reader already includes (noc_semaphore.h for Semaphore<>, endpoints.h for UnicastEndpoint)
    W2  namespace qwen_ks (the slice arithmetic, ks_common.RANGE_BLOCK byte for byte) and namespace qwen_ks_xfer: the bounded credit wait (a hang stays a hang: waypoint QKDC +
        watcher assert, then a spin) and send_partial, which moves the slice's normalised output (32 tiles) and log-sum-exp (4 fp32 tiles) from its cb_out / cb_lse_out into the NEXT
        slice's cb_prev_out / cb_lse_in at the same L1 addresses (every core has the same CB layout: the chain forwards K/V slots the same way), after the receiving slice's credit,
        then acks the writes, then raises the receiver's VALID flag (the chain's own order: data acked, then the flag)
    W3  the runtime words after the served twelve: the slice and the next slice's core (factory PK13)
    W4  the suffix compile-time args (factory PK6: cb_arg_offset + 6 ..) and the scale tile's packed bf16 pair
    W5  after the scale tile of the reduce: the ring merge's scale tile (cb_scale_in), made by the writer's own generate_bcast_col_scalar (element [0][0] is the scale; the merge reads it with a
        scalar broadcast)
    W6  per unit: B, E from the chunk start the writer already reads; an idle slice skips the unit, a slice that is not the last sends its partial and skips the DRAM write
    W7  the last slice writes the output as served, then drops its (unused) log-sum-exp

The mask, the reduce scaler and the drain of the served writer are untouched. Not a line of the served unit loop changes for the last slice.

Rules (as make_pf_reader.py): the input must be the pinned writer; every anchor must occur exactly once and start at the recorded line; the output must hash to WRITER_OUTPUT
(--record prints it); reverting the edits must give the input back; LF only.

    python3 make_ks_writer.py                  # write writer_interleaved_qwen_ksplit.cpp beside this file (the pinned writer from fixtures/)
    python3 make_ks_writer.py --check .        # compare with the committed file, write nothing
    python3 make_ks_writer.py --writer FILE    # another copy of the pinned writer (the build container's)
"""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ks_common  # noqa: E402

NL = chr(10)
lines = ks_common.lines
PINNED_WRITER = HERE / 'fixtures' / 'writer_interleaved.2a0959ff.cpp'
OUTPUT_NAME = 'writer_interleaved_qwen_ksplit.cpp'
BASE_SHA = '2a0959ff7cce507eab39f2e063f4dbc264d8b51f4db9f479b8c2419435c745c2'
WRITER_OUTPUT = '9b9ef47f9af44cfe83dc82d587568cfd82c4881021306f7af677cb672fd589ee'

# ---- W1: headers ----------------------------------------------------------------------------------------------------------------------------------
W1_OLD = lines('#include "api/core_local_mem.h"')
W1_NEW = W1_OLD + lines('#include "api/dataflow/noc_semaphore.h"', '#include "api/dataflow/endpoints.h"')

# ---- W2: the slice arithmetic and the hand-off ---------------------------------------------------------------------------------------------------
W2_OLD = lines('void kernel_main() {')
W2_NEW = ks_common.RANGE_BLOCK + lines(
    '',
    '// [QWEN-SDPA-KS] the hand-off of a slice\'s partial result to the next slice (the pinned compute kernel\'s ring merge reads it from cb_prev_out / cb_lse_in there).',
    'namespace qwen_ks_xfer {',
    '// Bounded wait for the receiving slice\'s credit: the semantics of the chain reader\'s wd_wait_credit (2^28 polls, then waypoint + watcher assert + a spin that never falls through).',
    'constexpr uint32_t kWdPolls = 1u << 28;',
    'FORCE_INLINE void wd_wait_credit(uint32_t sem_id) {',
    '    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));',
    '    for (uint32_t n = 0;; ++n) {',
    '        invalidate_l1_cache();',
    '        if (*p == 1) {',
    '            return;',
    '        }',
    '        if (n == kWdPolls) {',
    '            WAYPOINT("QKDC");',
    '            ASSERT(false);',
    '            for (;;) {',
    '                (void)*p;',
    '            }',
    '        }',
    '    }',
    '}',
    'FORCE_INLINE void send_partial(',
    '    Noc noc,',
    '    uint32_t cb_out_id,',
    '    uint32_t cb_lse_out_id,',
    '    uint32_t cb_prev_out_id,',
    '    uint32_t cb_lse_in_id,',
    '    uint32_t out_tiles,',
    '    uint32_t lse_tiles,',
    '    uint32_t out_bytes,',
    '    uint32_t lse_bytes,',
    '    uint32_t credit_sem_id,',
    '    uint32_t valid_src_sem_id,',
    '    uint32_t valid_dst_sem_id,',
    '    uint32_t nx,',
    '    uint32_t ny) {',
    '    CircularBuffer cb_out(cb_out_id);',
    '    CircularBuffer cb_lse(cb_lse_out_id);',
    '    cb_out.wait_front(out_tiles);  // the compute kernel made the whole normalised output and its log-sum-exp',
    '    cb_lse.wait_front(lse_tiles);',
    '    wd_wait_credit(credit_sem_id);  // the next slice has reserved cb_prev_out and cb_lse_in',
    '    Semaphore<>(credit_sem_id).set(0);',
    '    const uint32_t out_dst = CircularBuffer(cb_prev_out_id).get_write_ptr();  // the same address on every core (one CB layout)',
    '    const uint32_t lse_dst = CircularBuffer(cb_lse_in_id).get_write_ptr();',
    '    noc.async_write(',
    '        CoreLocalMem<uint32_t>(cb_out.get_read_ptr()), UnicastEndpoint{}, out_bytes, {}, {.noc_x = nx, .noc_y = ny, .addr = out_dst});',
    '    noc.async_write(',
    '        CoreLocalMem<uint32_t>(cb_lse.get_read_ptr()), UnicastEndpoint{}, lse_bytes, {}, {.noc_x = nx, .noc_y = ny, .addr = lse_dst});',
    '    noc.async_write_barrier();  // both writes ACKED on the next slice before its flag (the chain\'s order)',
    '    Semaphore<>(valid_src_sem_id).relay_unicast(noc, Semaphore<>(valid_dst_sem_id), nx, ny);',
    '    noc.async_writes_flushed();',
    '    cb_out.pop_front(out_tiles);',
    '    cb_lse.pop_front(lse_tiles);',
    '}',
    '}  // namespace qwen_ks_xfer',
    '') + W2_OLD

# ---- W3: the runtime words ------------------------------------------------------------------------------------------------------------------------
W3_OLD = lines('    const uint32_t cu_window_seqlens_eles = get_arg_val<uint32_t>(11);')
W3_NEW = W3_OLD + lines(
    '    // [QWEN-SDPA-KS] W3: the slice of this core and the next slice\'s core (the partial goes there); factory PK13',
    '    const uint32_t ks_slice = get_arg_val<uint32_t>(12);',
    '    const uint32_t ks_next_x = get_arg_val<uint32_t>(13);',
    '    const uint32_t ks_next_y = get_arg_val<uint32_t>(14);')

# ---- W4: the suffix compile-time args ---------------------------------------------------------------------------------------------------------------
W4_OLD = lines('    constexpr uint32_t cb_cu_window_in = get_compile_time_arg_val(cb_arg_offset + 5);')
W4_NEW = W4_OLD + lines(
    '    // [QWEN-SDPA-KS] W4: the suffix of the ksplit factory (PK6): slices, Skt, the CBs of the hand-off and of the merge scale, the three semaphores, the flags',
    '    constexpr uint32_t ks_slices = get_compile_time_arg_val(cb_arg_offset + 6);',
    '    constexpr uint32_t ks_Skt = get_compile_time_arg_val(cb_arg_offset + 7);',
    '    constexpr uint32_t ks_cb_lse_out = get_compile_time_arg_val(cb_arg_offset + 8);',
    '    constexpr uint32_t ks_cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 9);',
    '    constexpr uint32_t ks_cb_lse_in = get_compile_time_arg_val(cb_arg_offset + 10);',
    '    constexpr uint32_t ks_cb_scale_in = get_compile_time_arg_val(cb_arg_offset + 11);',
    '    constexpr uint32_t ks_credit_sem = get_compile_time_arg_val(cb_arg_offset + 12);',
    '    constexpr uint32_t ks_valid_src_sem = get_compile_time_arg_val(cb_arg_offset + 13);',
    '    constexpr uint32_t ks_valid_dst_sem = get_compile_time_arg_val(cb_arg_offset + 14);',
    '    constexpr uint32_t ks_flags = get_compile_time_arg_val(cb_arg_offset + 15);',
    '    static_assert((ks_flags & 0x19u) == 0x19u, "[QWEN-SDPA-KS] the ksplit writer needs the flags 0x1, 0x8 and 0x10");',
    '    static_assert(!use_streaming_compute, "[QWEN-SDPA-KS] the ring merge is the non-streaming path");',
    '    // the scale as the packed bf16 pair generate_bcast_col_scalar takes (the factory refuses a scale that is not exact in bf16: PK8)',
    '    constexpr uint32_t ks_scale_packed = (scale_val & 0xFFFF0000u) | (scale_val >> 16);')

# ---- W5: the scale tile -------------------------------------------------------------------------------------------------------------------------
W5_OLD = lines('    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);')
W5_NEW = W5_OLD + lines(
    '    generate_bcast_col_scalar(CircularBuffer(ks_cb_scale_in), ks_scale_packed);  // [QWEN-SDPA-KS] W5: the ring merge multiplies the running max by this scalar')

# ---- W6: per unit, before the served write ---------------------------------------------------------------------------------------------------------
W6_OLD = lines('            uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile, 0);')
W6_NEW = W6_OLD + lines(
    '            // [QWEN-SDPA-KS] W6: B blocks in this q chunk (the chunk start is the one the reader pushed), E slices take part: an idle slice made nothing; a slice that is not the last hands',
    '            // its partial to the next one and writes nothing; only the last slice (the one with the diagonal block) writes the output below',
    '            const uint32_t ks_q_high = std::min((chunk_start_t_in_q_chunks + q_chunk + 1) * Sq_chunk_t, ks_Skt);',
    '            const uint32_t ks_blocks = (ks_q_high + Sk_chunk_t - 1) / Sk_chunk_t;',
    '            const uint32_t ks_eff = qwen_ks::effective_slices(ks_blocks, ks_slices);',
    '            if (ks_slice >= ks_eff) {',
    '                continue;',
    '            }',
    '            if (ks_slice + 1 < ks_eff) {',
    '                qwen_ks_xfer::send_partial(',
    '                    noc,',
    '                    cb_out,',
    '                    ks_cb_lse_out,',
    '                    ks_cb_prev_out,',
    '                    ks_cb_lse_in,',
    '                    out_chunk_tiles,',
    '                    Sq_chunk_t,',
    '                    out_chunk_tiles * tile_bytes,',
    '                    Sq_chunk_t * get_tile_size(ks_cb_lse_out),',
    '                    ks_credit_sem,',
    '                    ks_valid_src_sem,',
    '                    ks_valid_dst_sem,',
    '                    ks_next_x,',
    '                    ks_next_y);',
    '                continue;',
    '            }')

# ---- W7: the last slice drops its log-sum-exp ---------------------------------------------------------------------------------------------------
W7_OLD = lines(
    '            }',
    '        }',
    '    }  // close phase')
W7_NEW = lines(
    '            }',
    '            // [QWEN-SDPA-KS] W7: the ring path of the compute kernel always makes the log-sum-exp; the last slice has no use for it',
    '            CircularBuffer ks_cb_lse_drop(ks_cb_lse_out);',
    '            ks_cb_lse_drop.wait_front(Sq_chunk_t);',
    '            ks_cb_lse_drop.pop_front(Sq_chunk_t);',
    '        }',
    '    }  // close phase')

# (label, pinned-writer line the anchor starts on, old text, new text), in file order.
EDITS = (
    ('W1', 8, W1_OLD, W1_NEW),
    ('W2', 15, W2_OLD, W2_NEW),
    ('W3', 68, W3_OLD, W3_NEW),
    ('W4', 80, W4_OLD, W4_NEW),
    ('W5', 95, W5_OLD, W5_NEW),
    ('W6', 194, W6_OLD, W6_NEW),
    ('W7', 221, W7_OLD, W7_NEW),
)


class Refusal(ValueError):
    """The input is not the pinned writer, or an anchor drifted."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_anchors(text):
    for label, line, old, _new in EDITS:
        count = text.count(old)
        if count != 1:
            raise Refusal('anchor %s occurs %d times (need exactly 1)' % (label, count))
        found = text[:text.index(old)].count(NL) + 1
        if found != line:
            raise Refusal('anchor %s starts at line %d, recorded %d; refusing on drift' % (label, found, line))


def patch(source):
    if sha(source) != BASE_SHA:
        raise Refusal('unexpected writer %s (need the pinned writer %s)' % (sha(source), BASE_SHA))
    text = source.decode('utf-8')
    check_anchors(text)
    for _label, _line, old, new in EDITS:
        text = text.replace(old, new)
    out = text.encode('utf-8')
    if chr(13).encode() in out:
        raise Refusal('output has a CR; files must be LF')
    return out


def unpatch(patched):
    text = patched.decode('utf-8')
    for label, _line, old, new in reversed(EDITS):
        if text.count(new) != 1:
            raise Refusal('edit %s occurs %d times' % (label, text.count(new)))
        text = text.replace(new, old)
    return text.encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('--check', metavar='DIR')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--writer', default=str(PINNED_WRITER))
    args = parser.parse_args(argv)
    base = Path(args.writer).read_bytes()
    try:
        generated = patch(base)
    except Refusal as error:
        print('refused: %s' % error)
        return 1
    if unpatch(generated) != base:
        print('edits do not invert')
        return 1
    digest = sha(generated)
    if args.dump:
        sys.stdout.write(generated.decode('utf-8'))
        return 0
    if args.record:
        print(digest)
        return 0
    if WRITER_OUTPUT != '__RECORD__' and digest != WRITER_OUTPUT:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, WRITER_OUTPUT))
        return 1
    if args.check:
        committed = (Path(args.check) / OUTPUT_NAME).read_bytes()
        if committed != generated:
            print('%s differs from the generated writer' % (Path(args.check) / OUTPUT_NAME))
            return 1
        print('ok %s %s' % (OUTPUT_NAME, digest))
        return 0
    (HERE / OUTPUT_NAME).write_bytes(generated)
    print('wrote %s %s' % (HERE / OUTPUT_NAME, digest))
    return 0


if __name__ == '__main__':
    sys.exit(main())
