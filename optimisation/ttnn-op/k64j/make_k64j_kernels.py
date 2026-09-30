"""Build the four K64j [QWEN-SDPA] kernels (the runtime extent, flag 0x20) from K64i's, byte for byte.

  dataflow/reader_decode_qwen.cpp        = the stage-3 reader (280a847f, ../sdpa_decode_qwen/stage3)      + R10
  dataflow/reader_decode_qwen_slice.cpp  = the stage-4 slice reader (0f5a019c, ../sdpa_decode_slice)     + R10
  compute/sdpa_flash_decode_qwen.cpp     = the qwen compute kernel (8776fcc7, ../sdpa_decode_qwen/stage3) + C3
  dataflow/writer_decode_qwen_slice.cpp  = the stage-4 slice writer (ac6cf815, ../sdpa_decode_slice)     + W4

  R10  the runtime_extent suffix CTA (factory F21: qwen_cta + 4 in the qwen reader, + 9 in the slice reader) with
       static_assert(!runtime_extent || mask_tail), the narrow one-chunk mask, non-causal and an interleaved
       cur_pos tensor; the causal block's cur_pos read under `is_causal || runtime_extent`, so both CB copies are
       pushed and the UINT32_MAX skip returns before any Q, K or V read and before the KV-share handshake; under
       KV share every entry takes slot 0 and writes it into its own slot of both copies (its writer and compute
       then split slot 0's extent too: the leader and its twins run one READY/VALID round per chunk)
  C3   CTA 33 and the same condition on the compute kernel's c_15 read; apply_mask_at_last_chunk stays causal-only
       and the tail predicate (k_chunk == k_num_chunks - 1) is already the runtime one
  W4   CTAs +3 q_slice and +4 runtime_extent after K64i's three; K64i's slice asserts hold under q_slice only, and
       q_tile_start is 0 without it, so W1-W3 reduce to writer_decode_all.cpp at q_slice = 0 (test_k64j_sources
       proves it); c_8 read under `is_causal || runtime_extent`; generate_mask never under runtime_extent

K64j never edits a stock kernel or a shared header (reader_decode_all.cpp, writer_decode_all.cpp,
sdpa_flash_decode.cpp, dataflow_common.hpp, rt_args_common.hpp): they also build the exact profile's programs, and
the kernel-cache key hashes only the graft's *qwen*.cpp kernels (k64j_probe/README.md).

The bases come from the committed K64i kernels (the default), the probe dump (--dump), or the stock originals
(--reader/--writer/--compute: docker cp'd out of ttbuild on the rig), which are first built into stages 3 and 4 by
make_qwen_kernels.py and make_slice_kernels.py (read-only). The base shas must match, every anchor must occur
exactly once, the results must hash to the recorded OUTPUTS, and the edits must invert to the K64i kernels.

    py -3.11 make_k64j_kernels.py --out kernels                    # from the committed K64i kernels
    py -3.11 make_k64j_kernels.py --dump sdpa-decode-sources.txt --check kernels
    python3 make_k64j_kernels.py --reader R.cpp --writer W.cpp --compute C.cpp --check kernels   # rig
"""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
QWEN_DIR = OPS / 'sdpa_decode_qwen'
SLICE_DIR = OPS / 'sdpa_decode_slice'
for _path in (str(QWEN_DIR), str(SLICE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import make_qwen_kernels as qwen  # noqa: E402 - stages 1 and 3, read-only
import make_slice_kernels as slice_kernels  # noqa: E402 - stage 4 (K64i), read-only

NL = chr(10)

READER_QWEN = 'dataflow/reader_decode_qwen.cpp'
READER_SLICE = 'dataflow/reader_decode_qwen_slice.cpp'
COMPUTE_QWEN = 'compute/sdpa_flash_decode_qwen.cpp'
WRITER_SLICE = 'dataflow/writer_decode_qwen_slice.cpp'
KERNELS = (READER_QWEN, READER_SLICE, COMPUTE_QWEN, WRITER_SLICE)
# K64i's served kernels (every one is in ~/opgraft-K64i/sdpa_decode/device/kernels).
BASES = {
    READER_QWEN: qwen.OUTPUTS_STAGE3[qwen.READER_NAME],                         # 280a847f
    READER_SLICE: slice_kernels.OUTPUTS[slice_kernels.READER_SLICE_NAME],       # 0f5a019c
    COMPUTE_QWEN: qwen.OUTPUTS_STAGE3[qwen.COMPUTE_NAME],                       # 8776fcc7
    WRITER_SLICE: slice_kernels.OUTPUTS[slice_kernels.WRITER_SLICE_NAME],       # ac6cf815
}
COMMITTED = {
    READER_QWEN: QWEN_DIR / 'stage3' / qwen.READER_NAME,
    READER_SLICE: SLICE_DIR / slice_kernels.READER_SLICE_NAME,
    COMPUTE_QWEN: QWEN_DIR / 'stage3' / qwen.COMPUTE_NAME,
    WRITER_SLICE: SLICE_DIR / slice_kernels.WRITER_SLICE_NAME,
}
# The recorded results; a changed edit list must update these deliberately.
OUTPUTS = {
    READER_QWEN: 'adb6091878ba3f0a0805846ff56f05352610d7fe779b5ae96320c437c095db49',
    READER_SLICE: '518d8096e3cceb160eaef8ab4f0ae976ccbffd3904d31176b7f9d02828c37f8a',
    COMPUTE_QWEN: '409a1aafc3ffaaca2c6afba0e999525b7d141491f0a52e5583efa70447ee5c0e',
    WRITER_SLICE: '642c36f809be0f1ad1664deb405dc310dabaa32d5628710320a118eb37a6cc7a',
}
# Where each reader reads runtime_extent (factory F21): after every other suffix of that reader.
READER_EXTENT_OFFSET = {READER_QWEN: 4, READER_SLICE: 9}


def lines(*parts):
    return ''.join(part + NL for part in parts)


# ---------------------------------------------------------------------------------------------
# R10: both qwen readers.
# ---------------------------------------------------------------------------------------------

# The last compile-time line each reader has before R10's: R1's tail assert (the qwen reader), R6's read-ahead
# assert (the slice reader).
R10_CTA_ANCHORS = {
    READER_QWEN: lines(
        '    static_assert(!mask_tail || (Sk_chunk_t > 0 && St % Sk_chunk_t == 0), "Tail mask needs a fixed chunk size");'),
    READER_SLICE: lines('    static_assert(!kv_readahead || kv_share, "K/V read-ahead needs KV share");'),
}
R10_CTA_AFTER = {READER_QWEN: "F6's four", READER_SLICE: "F16's five"}


def r10_cta(name):
    offset = READER_EXTENT_OFFSET[name]
    return R10_CTA_ANCHORS[name] + lines(
        '    // [QWEN-SDPA] K64j R10 (optimisation/ttnn-op/k64j): the runtime extent (flag 0x20), the last reader suffix',
        '    // (factory F21: +%d, after %s). Under it this non-causal kernel takes the causal block\'s cur_pos read'
        % (offset, R10_CTA_AFTER[name]),
        "    // below: each entry's E - 1 from the cur_pos tensor, or UINT32_MAX to skip the entry.",
        '    constexpr bool runtime_extent = get_compile_time_arg_val(qwen_cta + %d) == 1;' % offset,
        '    static_assert(!runtime_extent || mask_tail, "The runtime extent needs the tail mask");',
        '    static_assert(!runtime_extent || mask_width_t == Sk_chunk_t, "The runtime extent reads the narrow one-chunk mask");',
        '    static_assert(!runtime_extent || !is_causal, "The runtime extent is a non-causal mode");',
        '    static_assert(!runtime_extent || !is_cur_pos_tensor_sharded, "The runtime extent reads an interleaved cur_pos tensor");')


R10_READ_OLD = lines(
    '    if constexpr (is_causal) {',
    '        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list')
R10_READ_NEW = lines(
    '    // [QWEN-SDPA] K64j R10: a runtime-extent program (non-causal) takes this causal block as it is. Both CB copies',
    '    // (c_8 for the writer, c_15 for compute) are pushed before the UINT32_MAX test, and the skip returns before any',
    '    // Q, K or V read and before the KV-share handshake: nothing below has run yet.',
    '    if constexpr (is_causal || runtime_extent) {',
    '        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list')

R10_SHARE_OLD = lines(
    '            cb_writer.push_back(1);',
    '            cb_compute.push_back(1);')
R10_SHARE_NEW = lines(
    '            if constexpr (runtime_extent && kv_share) {',
    '                // [QWEN-SDPA] K64j R10: under KV share the leader and its twins run one READY/VALID round per',
    '                // chunk, so every entry must split the same extent. Each entry takes slot 0 and writes it into its',
    '                // own slot of both copies before they are pushed: its writer (c_8) and compute (c_15) split slot 0',
    '                // too, and a skipped slot 0 skips the whole bundle.',
    '                volatile tt_l1_ptr uint32_t* writer_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);',
    '                volatile tt_l1_ptr uint32_t* compute_words =',
    '                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_compute_wr_ptr);',
    '                const uint32_t slot0 = writer_words[0];',
    '                writer_words[cur_batch / q_heads_parallel_factor] = slot0;',
    '                compute_words[cur_batch / q_heads_parallel_factor] = slot0;',
    '            }') + R10_SHARE_OLD


def reader_edits(name):
    return (('R10 cta', R10_CTA_ANCHORS[name], r10_cta(name)), ('R10 read', R10_READ_OLD, R10_READ_NEW),
            ('R10 share', R10_SHARE_OLD, R10_SHARE_NEW))


# ---------------------------------------------------------------------------------------------
# C3: the qwen compute kernel.
# ---------------------------------------------------------------------------------------------

C3_CTA_ANCHOR = lines('    constexpr bool mask_tail = get_compile_time_arg_val(32) == 1;')
C3_CTA = C3_CTA_ANCHOR + lines(
    '    // [QWEN-SDPA] K64j C3 (optimisation/ttnn-op/k64j): CTA 33 (factory F21), the runtime extent (flag 0x20). Under',
    "    // it this non-causal kernel reads each entry's position from c_15 in the causal block below. The tail predicate",
    '    // (k_chunk == k_num_chunks - 1) then counts the runtime chunks, and apply_mask_at_last_chunk stays causal-only.',
    '    constexpr bool runtime_extent = get_compile_time_arg_val(33) == 1;',
    '    static_assert(!runtime_extent || (mask_tail && !is_causal), "The runtime extent is the non-causal tail mode");')

C3_READ_OLD = R10_READ_OLD
C3_READ_NEW = lines(
    '    if constexpr (is_causal || runtime_extent) {  // [QWEN-SDPA] K64j C3: a 0x20 program reads c_15 as the causal path does',
    '        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list')

COMPUTE_EDITS = (('C3 cta', C3_CTA_ANCHOR, C3_CTA), ('C3 read', C3_READ_OLD, C3_READ_NEW))

# ---------------------------------------------------------------------------------------------
# W4: the slice writer, which F21 selects for every 0x20 program.
# ---------------------------------------------------------------------------------------------

W4_CTA_OLD = lines(
    "    static_assert(num_kv_heads > 1 && rows_per_kv == num_q_heads / num_kv_heads, \"The slice writes each KV head's own rows\");",
    '    static_assert(!is_out_sharded && num_heads_per_core == 1, "The slice writer writes DRAM output, one KV head per core");',
    "    static_assert((num_kv_heads - 1) * rows_per_kv / 32 + PNHt <= pnht_full, \"The last KV head's slice runs past the output\");")
W4_CTA_NEW = lines(
    '    // [QWEN-SDPA] K64j W4 (optimisation/ttnn-op/k64j): +3 q_slice, +4 runtime_extent (factory F21), which selects',
    '    // this writer for every 0x20 program, sliced or not. Without the slice PNHt is pnht_full and q_tile_start is 0,',
    '    // so W1-W3 reduce to writer_decode_all.cpp; the slice\'s own asserts hold under it only.',
    '    constexpr bool q_slice = get_compile_time_arg_val(qwen_slice_cta + 3) == 1;',
    '    constexpr bool runtime_extent = get_compile_time_arg_val(qwen_slice_cta + 4) == 1;',
    '    static_assert(q_slice || runtime_extent, "The factory selects this writer under 0x4 or 0x20 only (F21)");',
    "    static_assert(q_slice ? PNHt < pnht_full : PNHt == pnht_full, \"PNHt is the slice under 0x4 and Q's own row tiles otherwise\");",
    '    static_assert(!runtime_extent || !is_causal, "The runtime extent is a non-causal mode");',
    "    static_assert(!q_slice || (num_kv_heads > 1 && rows_per_kv == num_q_heads / num_kv_heads), \"The slice writes each KV head's own rows\");",
    '    static_assert(!is_out_sharded && num_heads_per_core == 1, "The slice writer writes DRAM output, one KV head per core");',
    "    static_assert(!q_slice || (num_kv_heads - 1) * rows_per_kv / 32 + PNHt <= pnht_full, \"The last KV head's slice runs past the output\");")

W4_START_OLD = lines(
    "    const uint32_t q_tile_start = (cur_head_group * rows_per_kv) >> 5;  // [QWEN-SDPA] W1: the slice's first row tile")
W4_START_NEW = lines(
    "    const uint32_t q_tile_start = q_slice ? (cur_head_group * rows_per_kv) >> 5 : 0;  // [QWEN-SDPA] W1, W4: the slice's first row tile")

W4_READ_OLD = lines(
    '    if constexpr (is_causal) {',
    '        if (cur_pos_arg != UINT32_MAX) {')
W4_READ_NEW = lines(
    '    if constexpr (is_causal || runtime_extent) {  // [QWEN-SDPA] K64j W4: a 0x20 program reads c_8 as the causal path does',
    '        if (cur_pos_arg != UINT32_MAX) {')

W4_MASK_OLD = lines(
    '    if constexpr (is_causal) {',
    '        // These helper functions respect tile size of CBs (ie. no need for special handling of tiny tiles)',
    '        generate_mask<cb_mask_in, PNHt>(k_num_chunks, Sk_chunk_t_dynamic, cur_pos);')
W4_MASK_NEW = lines(
    '    if constexpr (is_causal && !runtime_extent) {  // [QWEN-SDPA] K64j W4: never under the runtime extent (R10 reads the tail mask)',
    '        // These helper functions respect tile size of CBs (ie. no need for special handling of tiny tiles)',
    '        generate_mask<cb_mask_in, PNHt>(k_num_chunks, Sk_chunk_t_dynamic, cur_pos);')

WRITER_EDITS = (('W4 cta', W4_CTA_OLD, W4_CTA_NEW), ('W4 start', W4_START_OLD, W4_START_NEW),
                ('W4 read', W4_READ_OLD, W4_READ_NEW), ('W4 mask', W4_MASK_OLD, W4_MASK_NEW))

EDITS = {READER_QWEN: reader_edits(READER_QWEN), READER_SLICE: reader_edits(READER_SLICE),
         COMPUTE_QWEN: COMPUTE_EDITS, WRITER_SLICE: WRITER_EDITS}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def apply_edits(name, base):
    """The K64j kernel `name` from its K64i base bytes."""
    if sha256(base) != BASES[name]:
        raise ValueError('%s base is %s, expected %s' % (name, sha256(base)[:16], BASES[name][:16]))
    text = base.decode('utf-8')
    for label, old, new in EDITS[name]:
        if text.count(old) != 1:
            raise ValueError('%s anchor %s occurs %d times' % (name, label, text.count(old)))
        text = text.replace(old, new)
    return text.encode('utf-8')


def revert_edits(name, built):
    """The inverse, last edit first: a K64j kernel back to its K64i base."""
    text = built.decode('utf-8')
    for label, old, new in reversed(EDITS[name]):
        if text.count(new) != 1:
            raise ValueError('%s edit %s occurs %d times' % (name, label, text.count(new)))
        text = text.replace(new, old)
    return text.encode('utf-8')


def bases_from(dump=None, reader=None, writer=None, compute=None):
    """{name: K64i base bytes}: the committed kernels, or stages 3 and 4 rebuilt from the stock originals."""
    if dump is None and reader is None:
        return {name: path.read_bytes() for name, path in COMMITTED.items()}
    if dump is not None:
        text = Path(dump).read_bytes().decode('utf-8')
        stock_reader = qwen.from_dump(text, qwen.DUMP_PATHS[qwen.READER_NAME])
        stock_compute = qwen.from_dump(text, qwen.DUMP_PATHS[qwen.COMPUTE_NAME])
        stock_writer = qwen.from_dump(text, slice_kernels.DUMP_WRITER)
    else:
        stock_reader, stock_writer, stock_compute = (Path(reader).read_bytes(), Path(writer).read_bytes(),
                                                     Path(compute).read_bytes())
    stage3_reader = slice_kernels.stage3_reader(stock_reader)
    stage3_compute = qwen.apply_edits(qwen.COMPUTE_NAME, stock_compute, 3)
    if sha256(stage3_compute) != BASES[COMPUTE_QWEN]:
        raise ValueError('make_qwen_kernels stage 3 built the compute kernel %s, recorded %s'
                         % (sha256(stage3_compute)[:16], BASES[COMPUTE_QWEN][:16]))
    stage4 = {slice_kernels.READER_SLICE_NAME: slice_kernels.apply_edits(slice_kernels.READER_SLICE_NAME, stage3_reader),
              slice_kernels.WRITER_SLICE_NAME: slice_kernels.apply_edits(slice_kernels.WRITER_SLICE_NAME, stock_writer)}
    bases = {READER_QWEN: stage3_reader, COMPUTE_QWEN: stage3_compute,
             READER_SLICE: stage4[slice_kernels.READER_SLICE_NAME], WRITER_SLICE: stage4[slice_kernels.WRITER_SLICE_NAME]}
    for name, data in bases.items():
        if sha256(data) != BASES[name]:
            raise ValueError('the K64i %s rebuilt to %s, recorded %s' % (name, sha256(data)[:16], BASES[name][:16]))
    return bases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--dump', help='probe_sdpa_decode_sources.py output (the stock originals)')
    source.add_argument('--reader', help='the stock reader_decode_all.cpp (then --writer and --compute too)')
    parser.add_argument('--writer', help='the stock writer_decode_all.cpp')
    parser.add_argument('--compute', help='the stock sdpa_flash_decode.cpp')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--out', help='directory to write the four kernels into (dataflow/, compute/)')
    target.add_argument('--check', help='directory whose four kernels must equal the build')
    parser.add_argument('--record', action='store_true', help='print the output shas instead of enforcing them')
    args = parser.parse_args(argv)
    if args.reader and not (args.writer and args.compute):
        parser.error('--reader needs --writer and --compute')
    try:
        bases = bases_from(args.dump, args.reader, args.writer, args.compute)
        built = {name: apply_edits(name, bases[name]) for name in KERNELS}
    except ValueError as error:
        print('refused: %s' % error)
        return 1
    status = 0
    for name in KERNELS:
        data = built[name]
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
