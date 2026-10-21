"""Make sdpa_qwen_ksplit.cpp (edits C1-C6) from the pinned compute kernel sdpa.cpp (sha a3f48af8..., tt-metal 9f9cd4fd; committed as fixtures/).

Prefill lever PF3-K (flag 0x10 of apply_factory_pk.py). The factory (PK11) selects this file only for flag 0x10. It is a new INSTANTIATION of the pinned template, not new arithmetic:
`sdpa_inner_loop<RING, ...>` of compute_common.hpp (fixtures/compute_common.3fb5da24.hpp, written for ring attention) instead of the `STANDARD` loop that sdpa_standard wraps. In the
RING path a pass emits its NORMALISED output and its log-sum-exp (cb_out, cb_lse_out: lse = scale x max + log(sum)) and, with ring_iter > 0, folds the previous pass's pair in:

    out = prev_out - sigmoid(cur_lse - prev_lse) x (prev_out - cur_out)        lse = prev_lse - logsigmoid(prev_lse - cur_lse)

(cb_prev_out, cb_lse_in). A K-split slice is such a pass over its K block range [iter_k_chunk_start, iter_k_chunk_end): slice 0 with ring_iter = 0, every later slice with ring_iter = 1 and
the previous slice's pair, which the reader pushed into cb_prev_out / cb_lse_in once the writer of the previous slice had delivered it. So one pinned merge per hop and no new compute code.
The parameters that make the RING path the chunked-prefill causal pass of the flexible op:

    chunked_enabled = true, chunked_q_local_padded_Nt = chunked_chunk_size_t = 2^20 tiles, ring_index = 0, ring_id = 0
        kv_global_tile_for_local<true, 0, N, N>(0, t) = (t / N) * N + t % N = t for every K tile t < N: the K frame is the global one; no straddle (local_start + Sk <= N)
        q_start_tile = q_start_idx_t + q_chunk * Sq_chunk_t with q_start_idx_t = chunk offset (read from the device tensor, as sdpa.cpp does) x Sq_chunk_t: the STANDARD path's absolute q tile
    logical_nt = num_local_k_chunks = 0xFFFFFFFF, every padding / joint mask flag false: no K chunk is skipped as padding or treated as joint
    is_causal = true, lightweight_mask_enabled = true: the diagonal stamp of the served path, applied to the one K chunk (the last of the last slice) that overlaps the diagonal
    use_zigzag_balancing = false and iter_q_start = the q chunk of the unit: q_chunk = remap_q_index(q_iter) % q_num_chunks = the per-head q chunk
    is_last_ring_iter = true: Q is popped at the end of the pass

    C1  after the includes: namespace qwen_ks (ks_common.RANGE_BLOCK byte for byte) and the two ring constants
    C2  the slice (runtime word 7, factory PK13)
    C3  the suffix compile-time args (factory PK6: 52 ..): slices, cb_lse_in, cb_lse_out, cb_prev_out, cb_scale_in, flags
    C4  the streaming branch is refused (the ring merge is the non-streaming path)
    C5  the ring pass of this core's unit, then return (an unused core, or an idle slice, returns without a pass); the served standard loop below it is unreachable for this kernel
    C6  the tail of the file is untouched

The reader pushes a dummy K/V pair after an even number of K chunks, because the pinned ring loop pops one at its end (KV_chunks_processed_in_iter % 2 == 0).

NOT COMPILED. tt-metal's compute API headers are not in this repository, so no stub compile of this file exists; the JIT compile on the build container (or card M) is the first compile.
Everything here is checked against the committed copy of the pinned header by text (test_ksplit_sources: the template's parameter list, the function's argument count, the ring merge's CB uses).

Rules (as make_pf_reader.py): the input must be the pinned kernel; every anchor must occur exactly once and start at the recorded line; the output must hash to COMPUTE_OUTPUT
(--record prints it); reverting the edits must give the input back; LF only.

    python3 make_ks_compute.py                  # write sdpa_qwen_ksplit.cpp beside this file (the pinned kernel from fixtures/)
    python3 make_ks_compute.py --check .        # compare with the committed file, write nothing
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
PINNED_COMPUTE = HERE / 'fixtures' / 'sdpa.a3f48af8.cpp'
OUTPUT_NAME = 'sdpa_qwen_ksplit.cpp'
BASE_SHA = 'a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0'
COMPUTE_OUTPUT = 'fc9571d178639f139b1da0f6aa385b5b10963f647df83860e763a7caff893913'

# ---- C1: namespace qwen_ks and the ring constants ---------------------------------------------------------------------------------------------------
C1_OLD = lines('#include "compute_streaming.hpp"')
C1_NEW = C1_OLD + lines('') + ks_common.RANGE_BLOCK + lines(
    '',
    '// [QWEN-SDPA-KS] the ring instantiation\'s constants: a K tile index below kKvIdentityTiles maps to itself (kv_global_tile_for_local), and nothing is ever "beyond the logical end"',
    'namespace qwen_ks_ring {',
    'constexpr uint32_t kKvIdentityTiles = %du;' % ks_common.KV_IDENTITY_TILES,
    'constexpr uint32_t kNever = 0x%XU;' % ks_common.KV_NEVER_TILES,
    '}  // namespace qwen_ks_ring')

# ---- C2: the slice ---------------------------------------------------------------------------------------------------------------------------------
C2_OLD = lines('    const uint32_t global_q_count = get_arg_val<uint32_t>(6);')
C2_NEW = C2_OLD + lines('    const uint32_t ks_slice = get_arg_val<uint32_t>(7);  // [QWEN-SDPA-KS] C2: this core\'s slice of its q chunk\'s K stream (factory PK13)')

# ---- C3: the suffix compile-time args --------------------------------------------------------------------------------------------------------------
C3_OLD = lines('    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(cb_arg_offset + 17);')
C3_NEW = C3_OLD + lines(
    '    // [QWEN-SDPA-KS] C3: the suffix of the ksplit factory (PK6): slices, the CBs of the ring merge (previous output, previous and own log-sum-exp, the scale tile), the flags',
    '    constexpr uint32_t ks_slices = get_compile_time_arg_val(cb_arg_offset + 18);',
    '    constexpr uint32_t ks_cb_lse_in = get_compile_time_arg_val(cb_arg_offset + 19);',
    '    constexpr uint32_t ks_cb_lse_out = get_compile_time_arg_val(cb_arg_offset + 20);',
    '    constexpr uint32_t ks_cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 21);',
    '    constexpr uint32_t ks_cb_scale_in = get_compile_time_arg_val(cb_arg_offset + 22);',
    '    constexpr uint32_t ks_flags = get_compile_time_arg_val(cb_arg_offset + 23);',
    '    static_assert((ks_flags & 0x19u) == 0x19u, "[QWEN-SDPA-KS] the ksplit compute kernel needs the flags 0x1, 0x8 and 0x10");')

# ---- C4: no streaming ------------------------------------------------------------------------------------------------------------------------------
C4_OLD = lines('    if constexpr (use_streaming_compute) {')
C4_NEW = lines(
    '    static_assert(!use_streaming_compute, "[QWEN-SDPA-KS] C4: the ring merge is the non-streaming path (the factory refuses streaming compute under ksplit)");',
    '    if constexpr (use_streaming_compute) {')

# ---- C5: the ring pass --------------------------------------------------------------------------------------------------------------------------------
C5_OLD = lines('        for (uint32_t phase = 0; phase < num_phases; ++phase) {')
C5_NEW = lines(
    '        // [QWEN-SDPA-KS] C5: this core\'s ONE unit (the factory gives a ksplit core 0 or 1 flat positions) as one slice of its q chunk\'s K stream through the pinned RING path',
    '        static_assert(use_lightweight_causal_mask, "[QWEN-SDPA-KS] C5: the ring pass takes the served causal lightweight-mask path");',
    '        if (global_q_count == 0) {',
    '            return;',
    '        }',
    '        const uint32_t ks_q_chunk = remap_q_index(global_q_start, q_num_chunks, use_zigzag_balancing) % q_num_chunks;  // the per-head q chunk, as decompose_global_q_index finds it',
    '        const uint32_t ks_q_start_tile = (chunked_q_chunk_offset_phase_1 + ks_q_chunk) * Sq_chunk_t;',
    '        const uint32_t ks_q_high_unclamped = ks_q_start_tile + Sq_chunk_t;',
    '        const uint32_t ks_q_high = ks_q_high_unclamped < Skt ? ks_q_high_unclamped : Skt;',
    '        const uint32_t ks_blocks = (ks_q_high + Sk_chunk_t - 1) / Sk_chunk_t;',
    '        const uint32_t ks_eff = qwen_ks::effective_slices(ks_blocks, ks_slices);',
    '        if (ks_slice >= ks_eff) {',
    '            return;  // an idle slice: its reader streamed nothing and pushed no Q',
    '        }',
    '        ChunkedContext ks_chunked;',
    '        ks_chunked.q_start_idx_t = chunked_q_chunk_offset_phase_1 * Sq_chunk_t;',
    '        ks_chunked.ring_index = 0;',
    '        sdpa_inner_loop<',
    '            RING,',
    '            cb_qk_im,',
    '            cb_identity_scale_in,',
    '            0,  // cb_attention_sink (not used)',
    '            ks_cb_scale_in,',
    '            Sq_chunk_t,',
    '            Sk_chunk_t,',
    '            0,  // NH (not used)',
    '            DHt,',
    '            vDHt,',
    '            false,  // use_attention_sink',
    '            false,  // use_provided_mask',
    '            false,  // use_padded_mask',
    '            false,  // use_joint_mask',
    '            false,  // is_chunked (the STANDARD path\'s flag; the ring path takes chunked_enabled below)',
    '            scale_fp32,',
    '            0,     // sliding_window_size',
    '            true,  // lightweight_mask_enabled',
    '            true,  // chunked_enabled',
    '            qwen_ks_ring::kKvIdentityTiles,',
    '            qwen_ks_ring::kKvIdentityTiles>(',
    '            0,  // Skt (not used by the ring path)',
    '            qk_in0_block_w,',
    '            qk_subblock_w,',
    '            qk_subblock_h,',
    '            qk_in0_num_subblocks,',
    '            qk_in1_num_subblocks,',
    '            qk_num_blocks,',
    '            out_in0_block_w,',
    '            out_subblock_w,',
    '            out_subblock_h,',
    '            out_in0_num_subblocks,',
    '            out_in1_num_subblocks,',
    '            out_num_blocks,',
    '            ks_q_chunk,      // iter_q_start: with use_zigzag_balancing false the ring path\'s q_chunk is this',
    '            ks_q_chunk + 1,  // iter_q_end',
    '            q_num_chunks,',
    '            0,  // local_q_start (not used)',
    '            0,  // chunked_q_chunk_offset (the ring path takes it from ks_chunked)',
    '            qwen_ks::slice_begin(ks_blocks, ks_slice, ks_eff),      // iter_k_chunk_start',
    '            qwen_ks::slice_begin(ks_blocks, ks_slice + 1, ks_eff),  // iter_k_chunk_end',
    '            q_chunk_tiles,',
    '            k_chunk_tiles,',
    '            v_chunk_tiles,',
    '            qk_chunk_tiles,',
    '            out_chunk_tiles,',
    '            0,  // mask_chunk_0 (not used)',
    '            0,  // mask_chunk_1 (not used)',
    '            ks_slice > 0 ? 1 : 0,  // ring_iter: the first slice has no previous partial, every later one folds the previous slice\'s in',
    '            0,                     // ring_id',
    '            qwen_ks_ring::kNever,  // num_local_k_chunks: no K chunk is a joint chunk',
    '            0,                     // local_padded_Nt (the chunked map above replaces it)',
    '            qwen_ks_ring::kNever,  // logical_nt: no K chunk lies beyond the logical end',
    '            false,                 // ring_iter_needs_global_n_mask',
    '            false,                 // ring_iter_needs_joint_n_mask',
    '            false,                 // local_n_needs_masking',
    '            0,                     // global_n_mask_chunk_id',
    '            0,                     // local_n_mask_chunk_id',
    '            0,                     // joint_n_mask_chunk_id',
    '            cb_q_in,',
    '            cb_k_in,',
    '            cb_v_in,',
    '            cb_mask_in,',
    '            cb_col_identity,',
    '            cb_out_im_A,',
    '            cb_out_im_B,',
    '            cb_max_A,',
    '            cb_max_B,',
    '            cb_sum_A,',
    '            cb_sum_B,',
    '            cb_exp_max_diff,',
    '            ks_cb_lse_in,',
    '            ks_cb_lse_out,',
    '            ks_cb_prev_out,',
    '            cb_out,',
    '            lw_mask,',
    '            is_causal,',
    '            false,  // is_balanced',
    '            false,  // use_zigzag_balancing: ks_q_chunk is already the per-head chunk',
    '            true,   // is_last_ring_iter: pop Q at the end of the pass',
    '            ks_chunked);',
    '        return;') + C5_OLD

# (label, pinned-kernel line the anchor starts on, old text, new text), in file order.
EDITS = (
    ('C1', 13, C1_OLD, C1_NEW),
    ('C2', 66, C2_OLD, C2_NEW),
    ('C3', 92, C3_OLD, C3_NEW),
    ('C4', 113, C4_OLD, C4_NEW),
    ('C5', 209, C5_OLD, C5_NEW),
)


class Refusal(ValueError):
    """The input is not the pinned kernel, or an anchor drifted."""


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
        raise Refusal('unexpected compute kernel %s (need the pinned sdpa.cpp %s)' % (sha(source), BASE_SHA))
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
    parser.add_argument('--kernel', default=str(PINNED_COMPUTE))
    args = parser.parse_args(argv)
    base = Path(args.kernel).read_bytes()
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
    if COMPUTE_OUTPUT != '__RECORD__' and digest != COMPUTE_OUTPUT:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, COMPUTE_OUTPUT))
        return 1
    if args.check:
        committed = (Path(args.check) / OUTPUT_NAME).read_bytes()
        if committed != generated:
            print('%s differs from the generated kernel' % (Path(args.check) / OUTPUT_NAME))
            return 1
        print('ok %s %s' % (OUTPUT_NAME, digest))
        return 0
    (HERE / OUTPUT_NAME).write_bytes(generated)
    print('wrote %s %s' % (HERE / OUTPUT_NAME, digest))
    return 0


if __name__ == '__main__':
    sys.exit(main())
