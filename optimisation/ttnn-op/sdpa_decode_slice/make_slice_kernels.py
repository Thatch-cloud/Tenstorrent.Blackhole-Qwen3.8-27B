"""Build the two stage-4 [QWEN-SDPA] kernels (K1, graft K64i) from the served originals, byte for byte.

  reader_decode_qwen_slice.cpp = the stage-3 reader_decode_qwen.cpp (280a847f, itself reader_decode_all.cpp
                                 49a05926 + R1-R5 by ../sdpa_decode_qwen/make_qwen_kernels.py --stage 3)
                                 + R6  the five stage-4 compile-time args (factory F16), the ABI tag and
                                       coverage static_asserts, and q_tile_start (this KV head's first Q row
                                       tile under the slice, 0 without it)
                                 + R7  Q from entry b's full layout, starting at q_tile_start
                                 + R8  the mask's batch stride is its full row count; rows from q_tile_start
                                 + R9  K1b, behind kv_readahead (flag 0x8): the read-ahead share leader
                                       (R9a its slot state, R9b the leader branch, R9c the shared mask read
                                       skipped for it, since it reads its mask before chunk n+1)
  writer_decode_qwen_slice.cpp = writer_decode_all.cpp (734c90c0)
                                 + W1  the three stage-4 compile-time args, the tag and shape asserts, and
                                       q_tile_start from cur_head_group
                                 + W2  entry b's output tiles start at b * pnht_full * vDHt
                                 + W3  write_partial_tiles_sliced: write_partial_tiles_to_memory's loops and
                                       face-line writes, same DRAM page and offset, L1 tile rebased by
                                       q_tile_start (W3a the function, W3b the call)

(k1-sdpa-head-slice-design.md sections 3 and 5.2-5.4.) The compute kernel is reused unchanged
(sdpa_flash_decode_qwen.cpp, 8776fcc7). The originals come from the probe dump or from the files
themselves (--reader/--writer: docker cp'd out of ttbuild on the rig); the base shas must match, every
anchor must occur exactly once, the results must hash to the recorded OUTPUTS, and the edits must invert
(the slice reader back to the stage-3 reader, the slice writer back to 734c90c0).

    py -3.11 make_slice_kernels.py --dump sdpa-decode-sources.txt --out .
    py -3.11 make_slice_kernels.py --dump sdpa-decode-sources.txt --check .
    python3 make_slice_kernels.py --reader reader_decode_all.cpp --writer writer_decode_all.cpp --check .   # rig
"""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
QWEN_DIR = HERE.parent / 'sdpa_decode_qwen'
if str(QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_DIR))

import make_qwen_kernels as qwen  # noqa: E402 - the stage-1/3 kernels, read-only

NL = chr(10)

READER_SLICE_NAME = 'reader_decode_qwen_slice.cpp'
WRITER_SLICE_NAME = 'writer_decode_qwen_slice.cpp'
WRITER_BASE = '734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8'     # writer_decode_all.cpp
READER_STAGE3 = qwen.OUTPUTS_STAGE3[qwen.READER_NAME]                                   # 280a847f
DUMP_WRITER = qwen.DUMP_PREFIX + 'dataflow/writer_decode_all.cpp'
BASES = {READER_SLICE_NAME: READER_STAGE3, WRITER_SLICE_NAME: WRITER_BASE}
# The recorded results; a changed edit list must update these deliberately.
OUTPUTS = {
    READER_SLICE_NAME: '0f5a019ccc06ca603bb4ed44c77cc66f9cb3e5bd35193810eb127f13c9f5631f',
    WRITER_SLICE_NAME: 'ac6cf815c34df85a9d39593d95f28eb2da0cb5b37c232633e65bf3ed116925f4',
}
ABI_TAG = 0x51CE


def lines(*parts):
    return ''.join(part + NL for part in parts)


# ---------------------------------------------------------------------------------------------
# The reader: R6-R9 on the stage-3 reader.
# ---------------------------------------------------------------------------------------------

R6_ANCHOR = lines(
    '    static_assert(!mask_tail || (Sk_chunk_t > 0 && St % Sk_chunk_t == 0), "Tail mask needs a fixed chunk size");')
R6 = R6_ANCHOR + lines(
    '    // [QWEN-SDPA] stage-4 suffix compile-time args (factory F16), after F6\'s four. This file is the stage-3',
    '    // reader_decode_qwen.cpp (sha256 280a847f) plus the edits R6-R9 of optimisation/ttnn-op/sdpa_decode_slice;',
    '    // the factory selects it for flag 0x4 (q-slice) or 0x8 (K/V read-ahead) only. PNHt (CTA 1) is the',
    '    // slice\'s row tiles under 0x4; Q and the mask keep pnht_full row tiles per entry.',
    '    constexpr uint32_t pnht_full = get_compile_time_arg_val(qwen_cta + 4);',
    '    constexpr uint32_t rows_per_kv = get_compile_time_arg_val(qwen_cta + 5);',
    '    constexpr bool kv_readahead = get_compile_time_arg_val(qwen_cta + 6) == 1;',
    '    constexpr bool q_slice = get_compile_time_arg_val(qwen_cta + 7) == 1;',
    '    static_assert(get_compile_time_arg_val(qwen_cta + 8) == 0x51CE, "Slice reader ABI tag: factory F16 and R6 disagree");',
    '    static_assert(q_slice || PNHt == pnht_full, "Without the slice every row tile is read");',
    '    static_assert(!q_slice || (num_kv_heads - 1) * rows_per_kv / 32 + PNHt <= pnht_full,',
    '                  "The last KV head\'s slice runs past Q");',
    '    static_assert(!kv_readahead || kv_share, "K/V read-ahead needs KV share");')

R6B_ANCHOR = lines('    const uint32_t num_dests = get_arg_val<uint32_t>(arg_idx++);')
R6B = R6B_ANCHOR + lines(
    '    // [QWEN-SDPA] R6: this KV head\'s first Q row tile under the slice (the factory\'s F14 rule); 0 without it.',
    '    const uint32_t q_tile_start = q_slice ? (cur_head_group * rows_per_kv) >> 5 : 0;')

R7_OLD = lines('    const uint32_t q_batch_offset = cur_batch * q_chunk_tiles;')
R7_NEW = lines(
    '    // [QWEN-SDPA] R7: entry cur_batch\'s Q holds pnht_full row tiles (q_chunk_tiles is the slice\'s); this',
    '    // head\'s slice starts q_tile_start row tiles in.',
    '    const uint32_t q_batch_offset = cur_batch * pnht_full * DHt + q_tile_start * DHt;')

R8_OLD = lines(
    '        const uint32_t mask_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bmask) * PNHt * mask_width_t;')
R8_NEW = lines(
    '        // [QWEN-SDPA] R8: the mask\'s batch stride is its full row count (pnht_full); this head\'s rows start',
    '        // q_tile_start row tiles in, and read_mask_chunk<PNHt> then reads the slice\'s rows.',
    '        const uint32_t mask_batch_offset =',
    '            ((cur_batch / q_heads_parallel_factor) % Bmask) * pnht_full * mask_width_t + q_tile_start * mask_width_t;')

R9A_ANCHOR = lines(
    '        if constexpr (is_paged_attention) {',
    '            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {')
R9A = lines(
    '        if constexpr (is_paged_attention) {',
    '            // [QWEN-SDPA] R9 (K1b): the read-ahead leader\'s K and V slots of the chunk it multicasts next.',
    '            uint32_t ra_k_slot = 0;',
    '            uint32_t ra_v_slot = 0;',
    '            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {')


def read_k_call(target, row_expr, indent):
    """The stage-3 leader's read_k call (no MLA multicast), assigning its slot to `target`."""
    pad = ' ' * indent
    return lines(
        pad + '%s = read_k<' % target,
        pad + '    cb_k_in,',
        pad + '    DHt,',
        pad + '    num_kv_heads,',
        pad + '    block_size_t,',
        pad + '    k_tile_bytes,',
        pad + '    barrier_threshold,',
        pad + '    is_page_table_sharded,',
        pad + '    false,',
        pad + '    capacity_t>(',
        pad + '    k_chunk_tiles,',
        pad + '    cur_head,',
        pad + '    Sk_chunk_t_dynamic,',
        pad + '    %s,' % row_expr,
        pad + '    k_reader,',
        pad + '    page_table_ptr_u16,',
        pad + '    page_table_ptr_u32,',
        pad + '    barrier_count);')


def read_v_call(slot, row_expr, indent):
    """The stage-3 leader's read_v call (explicit V), noting the V slot it reserves in `slot` first."""
    pad = ' ' * indent
    return lines(
        pad + '%s = cb_v.get_write_ptr();  // read_v reserves exactly this slot' % slot,
        pad + 'read_v<',
        pad + '    cb_v_in,',
        pad + '    vDHt,',
        pad + '    num_kv_heads,',
        pad + '    block_size_t,',
        pad + '    v_tile_bytes,',
        pad + '    barrier_threshold,',
        pad + '    is_page_table_sharded,',
        pad + '    false,',
        pad + '    capacity_t>(',
        pad + '    v_chunk_tiles,',
        pad + '    cur_head,',
        pad + '    Sk_chunk_t_dynamic,',
        pad + '    %s,' % row_expr,
        pad + '    v_reader,',
        pad + '    page_table_ptr_u16,',
        pad + '    page_table_ptr_u32,',
        pad + '    barrier_count,',
        pad + '    ra_k_slot,',
        pad + '    k_tile_bytes);')


def multicast(slot, tiles, tile_bytes, indent):
    pad = ' ' * indent
    return lines(
        pad + 'noc.async_write_multicast(',
        pad + '    CoreLocalMem<uint32_t>(%s),' % slot,
        pad + '    MulticastEndpoint{},',
        pad + '    %s * %s,' % (tiles, tile_bytes),
        pad + '    num_dests,',
        pad + '    {},',
        pad + '    {.noc_x_start = mcast_x,',
        pad + '     .noc_y_start = mcast_y0,',
        pad + '     .noc_x_end = mcast_x,',
        pad + '     .noc_y_end = mcast_y1,',
        pad + '     .addr = %s},' % slot,
        pad + '    false);')


R9B_ANCHOR = lines(
    '                    Semaphore<> kv_valid(k_mcast_semaphore_id);',
    '                    if (do_k_mcast) {')
R9B = lines(
    '                    Semaphore<> kv_valid(k_mcast_semaphore_id);',
    '                    if (kv_readahead && do_k_mcast) {',
    '                        // [QWEN-SDPA] R9, K1b (flag 0x8): the READ-AHEAD LEADER. The stage-3 leader (the next',
    '                        // branch) reads chunk n, waits for READY, multicasts it and only then reads n+1, so its',
    '                        // period is T_read + T_mcast + the handshake. Here chunk n was read one iteration early',
    '                        // (by the prologue for the first chunk), so the read of n+1 runs while n\'s multicast is',
    '                        // in flight: wait READY(n); multicast K(n) and V(n) (non-blocking); read chunk n\'s mask',
    '                        // if it has one; read K(n+1) and V(n+1) into the other ring slot (never past',
    '                        // k_chunk_end); write barrier; VALID(n). Slot n % 2 is next written by chunk n+2\'s read,',
    '                        // in the next iteration, after this iteration\'s write barrier. The bytes, the CB order,',
    '                        // the READY/VALID rounds and the twins are the stage-3 ones; only when reads land moves.',
    '                        if (k_chunk == k_chunk_start) {') + \
    read_k_call('ra_k_slot', 'k_chunk_start_row_num', 28) + \
    read_v_call('ra_v_slot', 'k_chunk_start_row_num', 28) + lines(
    '                        }',
    '                        const uint32_t k_slot = ra_k_slot;  // chunk n, read by the prologue or the last iteration',
    '                        const uint32_t v_slot = ra_v_slot;',
    '                        Semaphore<> kv_ready(kv_ready_semaphore_id);',
    '                        kv_ready.wait(num_dests);  // every twin has reserved this chunk\'s K and V slots',
    '                        kv_ready.set(0);') + \
    multicast('k_slot', 'k_chunk_tiles', 'k_tile_bytes', 24) + \
    multicast('v_slot', 'v_chunk_tiles', 'v_tile_bytes', 24) + lines(
    '                        if constexpr (use_attention_mask) {  // this chunk\'s mask before chunk n+1\'s K/V',
    '                            if (!mask_tail || k_chunk == k_num_chunks - 1) {',
    '                                mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(',
    '                                    mask_width_t, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);',
    '                            }',
    '                        }',
    '                        if (k_chunk + 1 < k_chunk_end) {',
    '                            const uint32_t next_row_num = k_chunk_start_row_num + Sk_chunk_t_dynamic;') + \
    read_k_call('ra_k_slot', 'next_row_num', 28) + \
    read_v_call('ra_v_slot', 'next_row_num', 28) + lines(
    '                        }',
    '                        noc.async_write_barrier();  // chunk n has landed on every twin before its flag',
    '                        kv_valid.set(1);',
    '                        kv_valid.set_multicast(noc, mcast_x, mcast_y0, mcast_x, mcast_y1, num_dests);',
    '                    } else if (do_k_mcast) {')

R9C_OLD = lines(
    '                    if constexpr (use_attention_mask) {  // each entry still reads its own mask, after K and V',
    '                        if (!mask_tail || k_chunk == k_num_chunks - 1) {')
R9C_NEW = lines(
    '                    if constexpr (use_attention_mask) {  // each entry still reads its own mask, after K and V',
    '                        // [QWEN-SDPA] R9: a read-ahead leader read this chunk\'s mask above, before chunk n+1\'s K/V.',
    '                        if ((!mask_tail || k_chunk == k_num_chunks - 1) && !(kv_readahead && do_k_mcast)) {')

READER_EDITS = (('R6', R6_ANCHOR, R6), ('R6 q_tile_start', R6B_ANCHOR, R6B), ('R7', R7_OLD, R7_NEW),
                ('R8', R8_OLD, R8_NEW), ('R9a', R9A_ANCHOR, R9A), ('R9b', R9B_ANCHOR, R9B), ('R9c', R9C_OLD, R9C_NEW))

# ---------------------------------------------------------------------------------------------
# The writer: W1-W3 on writer_decode_all.cpp.
# ---------------------------------------------------------------------------------------------

W1_ANCHOR = lines('    constexpr auto out_args = TensorAccessorArgs<28>();')
W1 = W1_ANCHOR + lines(
    '    // [QWEN-SDPA] stage-4 suffix compile-time args (factory F16), after the output accessor. This file is',
    '    // writer_decode_all.cpp (sha256 734c90c0) plus the edits W1-W3 of optimisation/ttnn-op/sdpa_decode_slice;',
    '    // the factory selects it for flag 0x4 (q-slice) only. PNHt (CTA 1) is the slice\'s row tiles: the tree',
    '    // traffic below already moves the slice. The output keeps pnht_full row tiles per entry.',
    '    constexpr uint32_t qwen_slice_cta = out_args.next_compile_time_args_offset();',
    '    constexpr uint32_t pnht_full = get_compile_time_arg_val(qwen_slice_cta + 0);',
    '    constexpr uint32_t rows_per_kv = get_compile_time_arg_val(qwen_slice_cta + 1);',
    '    static_assert(get_compile_time_arg_val(qwen_slice_cta + 2) == 0x51CE, "Slice writer ABI tag: factory F16 and W1 disagree");',
    '    static_assert(num_kv_heads > 1 && rows_per_kv == num_q_heads / num_kv_heads, "The slice writes each KV head\'s own rows");',
    '    static_assert(!is_out_sharded && num_heads_per_core == 1, "The slice writer writes DRAM output, one KV head per core");',
    '    static_assert((num_kv_heads - 1) * rows_per_kv / 32 + PNHt <= pnht_full, "The last KV head\'s slice runs past the output");')

W1B_ANCHOR = lines('    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);')
W1B = W1B_ANCHOR + lines(
    '    const uint32_t q_tile_start = (cur_head_group * rows_per_kv) >> 5;  // [QWEN-SDPA] W1: the slice\'s first row tile')

W2_OLD = lines('        uint32_t out_tile_id = cur_batch * out_chunk_tiles;')
W2_NEW = lines(
    '        // [QWEN-SDPA] W2: entry cur_batch\'s output is pnht_full row tiles (out_chunk_tiles is the slice\'s).',
    '        uint32_t out_tile_id = cur_batch * pnht_full * vDHt;')

W3A_ANCHOR = lines('#define MAX_TREE_REDUCTION_ROUNDS 6')
W3A = W3A_ANCHOR + lines(
    '',
    '// [QWEN-SDPA] W3: write_partial_tiles_to_memory (dataflow_common.hpp) for a head-sliced output. The same loops',
    '// and the same two face-line writes per kept row, to the same DRAM page and byte offset; only the L1 source is',
    '// rebased: cb_out holds row tiles [q_tile_start, q_tile_start + PNHt) of the entry, so row tile head_tile is L1',
    '// tile (head_tile - q_tile_start) * vDHt + d, while its DRAM tile stays out_tile_id + head_tile * vDHt + d.',
    'template <uint32_t cb_out, uint32_t ELEMENT_SIZE, uint32_t barrier_threshold, uint32_t PNHt, typename WriterType>',
    'uint32_t write_partial_tiles_sliced(',
    '    uint32_t& out_tile_id,  // base tile index in DRAM for this batch (the full layout)',
    '    const WriterType& out_writer,',
    '    uint32_t& barrier_count,',
    '    uint32_t cur_head,            // kv-head group index 0..num_kv_heads-1',
    '    uint32_t num_heads_to_write,  // q-heads per kv-head group (folded rows per KV head)',
    '    uint32_t out_chunk_tiles,     // tiles in cb_out = PNHt (the slice) * vDHt',
    '    uint32_t q_tile_start) {      // the slice\'s first row tile',
    '    Noc noc;',
    '    constexpr uint32_t FACE_HW = 16;',
    '    constexpr uint32_t TILE_HW = 32;',
    '    constexpr uint32_t FACE_ELEMENT_CNT = FACE_HW * FACE_HW;',
    '    constexpr uint32_t tile_bytes = get_tile_size(cb_out);',
    '    constexpr uint32_t FACE_LINE_BYTES = FACE_HW * ELEMENT_SIZE;',
    '    const uint32_t num_hidden_tiles = out_chunk_tiles / PNHt;',
    '',
    '    CircularBuffer cb(cb_out);',
    '    uint32_t l1_base_addr = cb.get_read_ptr();',
    '',
    '    for (uint32_t hidden_tile = 0; hidden_tile < num_hidden_tiles; ++hidden_tile) {',
    '        for (uint32_t head = 0; head < num_heads_to_write; ++head) {',
    '            uint32_t starting_row = cur_head * num_heads_to_write + head;',
    '            uint32_t tile_row = starting_row % TILE_HW;',
    '            uint32_t head_tile = starting_row / TILE_HW;',
    '',
    '            uint32_t in_tile_offset = (tile_row < FACE_HW) ? tile_row * FACE_LINE_BYTES',
    '                                                           : (tile_row + FACE_HW) * FACE_LINE_BYTES;  // skip face',
    '',
    '            // L1: cb_out holds the slice\'s row tiles only; DRAM: the full layout, as the legacy writer',
    '            const uint32_t l1_tile_index = (head_tile - q_tile_start) * num_hidden_tiles + hidden_tile;',
    '            const uint32_t dram_tile_id = out_tile_id + head_tile * num_hidden_tiles + hidden_tile;',
    '            uint32_t l1_read_addr_head = l1_base_addr + l1_tile_index * tile_bytes + in_tile_offset;',
    '',
    '            noc.async_write(',
    '                CoreLocalMem<uint32_t>(l1_read_addr_head),',
    '                out_writer,',
    '                FACE_LINE_BYTES,',
    '                {},',
    '                {.page_id = dram_tile_id, .offset_bytes = in_tile_offset});',
    '            noc.async_write(',
    '                CoreLocalMem<uint32_t>(l1_read_addr_head + FACE_ELEMENT_CNT * ELEMENT_SIZE),',
    '                out_writer,',
    '                FACE_LINE_BYTES,',
    '                {},',
    '                {.page_id = dram_tile_id, .offset_bytes = in_tile_offset + FACE_ELEMENT_CNT * ELEMENT_SIZE});',
    '',
    '            if (++barrier_count == barrier_threshold) {',
    '                noc.async_writes_flushed();',
    '                barrier_count = 0;',
    '            }',
    '        }',
    '    }',
    '    return barrier_count;',
    '}')

W3B_OLD = lines(
    '                barrier_count = write_partial_tiles_to_memory<cb_out, ELEMENT_SIZE, barrier_threshold, PNHt>(',
    '                    out_tile_id, out_writer, barrier_count, cur_head, num_heads_to_write, out_chunk_tiles);')
W3B_NEW = lines(
    '                barrier_count = write_partial_tiles_sliced<cb_out, ELEMENT_SIZE, barrier_threshold, PNHt>(',
    '                    out_tile_id, out_writer, barrier_count, cur_head, num_heads_to_write, out_chunk_tiles, q_tile_start);')

WRITER_EDITS = (('W1', W1_ANCHOR, W1), ('W1 q_tile_start', W1B_ANCHOR, W1B), ('W2', W2_OLD, W2_NEW),
                ('W3a', W3A_ANCHOR, W3A), ('W3b', W3B_OLD, W3B_NEW))

EDITS = {READER_SLICE_NAME: READER_EDITS, WRITER_SLICE_NAME: WRITER_EDITS}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stage3_reader(stock_reader):
    """The stage-3 reader (280a847f) from the stock reader_decode_all.cpp (49a05926)."""
    built = qwen.apply_edits(qwen.READER_NAME, stock_reader, 3)
    if sha256(built) != READER_STAGE3:
        raise ValueError('make_qwen_kernels stage 3 built %s, recorded %s' % (sha256(built)[:16], READER_STAGE3[:16]))
    return built


def apply_edits(name, base):
    """The stage-4 kernel `name` from its base bytes (the stage-3 reader, or the stock writer)."""
    if sha256(base) != BASES[name]:
        raise ValueError('%s base is %s, expected %s' % (name, sha256(base)[:16], BASES[name][:16]))
    text = base.decode('utf-8')
    for label, old, new in EDITS[name]:
        if text.count(old) != 1:
            raise ValueError('%s anchor %s occurs %d times' % (name, label, text.count(old)))
        text = text.replace(old, new)
    return text.encode('utf-8')


def revert_edits(name, built):
    """The inverse, last edit first: the slice reader back to the stage-3 reader, the writer to 734c90c0."""
    text = built.decode('utf-8')
    for label, old, new in reversed(EDITS[name]):
        if text.count(new) != 1:
            raise ValueError('%s edit %s occurs %d times' % (name, label, text.count(new)))
        text = text.replace(new, old)
    return text.encode('utf-8')


def bases_from(dump=None, reader=None, writer=None):
    """{name: base bytes}: the stage-3 reader (built from the stock reader) and the stock writer."""
    if dump is not None:
        text = Path(dump).read_bytes().decode('utf-8')
        stock_reader = qwen.from_dump(text, qwen.DUMP_PATHS[qwen.READER_NAME])
        stock_writer = qwen.from_dump(text, DUMP_WRITER)
    else:
        stock_reader, stock_writer = Path(reader).read_bytes(), Path(writer).read_bytes()
    return {READER_SLICE_NAME: stage3_reader(stock_reader), WRITER_SLICE_NAME: stock_writer}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dump', help='probe_sdpa_decode_sources.py output')
    source.add_argument('--reader', help='the stock reader_decode_all.cpp (then --writer too)')
    parser.add_argument('--writer', help='the stock writer_decode_all.cpp')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--out', help='directory to write the two slice kernels into')
    target.add_argument('--check', help='directory whose slice kernels must equal the build')
    parser.add_argument('--record', action='store_true', help='print the output shas instead of enforcing them')
    args = parser.parse_args(argv)
    if args.reader and not args.writer:
        parser.error('--reader needs --writer')
    try:
        bases = bases_from(args.dump, args.reader, args.writer)
        built = {name: apply_edits(name, base) for name, base in bases.items()}
    except ValueError as error:
        print('refused: %s' % error)
        return 1
    status = 0
    for name, data in built.items():
        digest = sha256(data)
        if not args.record and digest != OUTPUTS[name]:
            print('%s built to %s, recorded %s' % (name, digest, OUTPUTS[name]))
            return 1
        if revert_edits(name, data) != bases[name]:
            print('%s edits do not invert' % name)
            return 1
        if args.out:
            target = Path(args.out) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            print('%s base %s -> %s written %s' % (name, BASES[name][:16], digest, target))
        else:
            target = Path(args.check) / name
            present = target.read_bytes() if target.is_file() else None
            same = present == data
            print('%s base %s -> %s %s' % (name, BASES[name][:16], digest, 'matches' if same else 'DIFFERS at ' + str(target)))
            status |= 0 if same else 1
    return status


if __name__ == '__main__':
    sys.exit(main())
