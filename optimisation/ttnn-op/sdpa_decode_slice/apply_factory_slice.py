"""Apply the [QWEN-SDPA] stage-4 factory edits (K1: F13-F18) to sdpa_decode_program_factory.cpp.

Stage 4 (k1-sdpa-head-slice-design.md section 5.1, with the review's F16/F17 fix) is applied to the
stage-3 factory (06167779..., apply_factory_qwen.py --stage 3), and adds two per-call flags to the
q_chunk_size sentinel:

  0x4  q-slice (K1a): every core of KV head h computes only the Q row tiles that hold that head's
       G = num_q_heads / num_kv_heads folded rows, [floor(h*G/32), floor(h*G/32) + slice), where
       slice is the widest head's span. PNHt becomes the slice everywhere downstream (CBs, subblocks,
       the compute, writer and reader CTA 1, MUL_BCAST_GRANULARITY, the F4 log line).
  0x8  K/V read-ahead (K1b): the share leader reads chunk n+1 while chunk n's multicast is in flight.
       Inert without share at B = 1, as 0x2 is; refused without 0x2.

  F13  the flag constants and the slice kernels' ABI tag (0x51CE)
  F14  PNHt = the slice under 0x4 (qwen_pnht_full keeps Q's own row tiles)
  F15  admits 0x4 / 0x8; refuses 0x8 without 0x2, and under 0x4 a slice that saves no tile, heads that do
       not divide, a head the slice does not cover, and a mask whose row tiles are not Q's
  F16  the reader suffix (+4 pnht_full, +5 rows_per_kv, +6 kv_readahead, +7 q_slice, +8 tag) under 0x4 OR
       0x8, the writer suffix (+0 pnht_full, +1 rows_per_kv, +2 tag) under 0x4 only
  F17  reader_decode_qwen_slice.cpp under 0x4 or 0x8; writer_decode_qwen_slice.cpp under 0x4 only (0x8
       alone keeps the stock writer: its rows are not sliced)
  F18  one log line per stage-4 program: '[QWEN-SDPA] q-slice rows_per_kv=.. pnht_full=.. slice_tiles=..
       readahead=..' (slice_tiles is the program's PNHt: pnht_full on a read-ahead-only program)

Input: the tree-scratch base factory 3e0a69af (stage 3 is applied first, by apply_factory_qwen.py), or
the stage-3 factory 06167779 itself; anything else is refused. Every anchor must occur exactly once, the
result must hash to STAGE4_FACTORY, and the edits must invert: stage 4 -> stage 3 -> stage 1 -> 3e0a69af.
Run on an already patched file it reports and exits 0.

    python3 apply_factory_slice.py <factory.cpp> --out X.cpp   # into X.cpp, the input untouched
    python3 apply_factory_slice.py <factory.cpp>               # in place, keeping .orig-<sha8>
    python3 apply_factory_slice.py <factory.cpp> --record      # print the output sha instead of enforcing it

It imports ../sdpa_decode_qwen/apply_factory_qwen.py (the stages 1 and 3) read-only.
"""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
QWEN_DIR = HERE.parent / 'sdpa_decode_qwen'
if str(QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_DIR))

import apply_factory_qwen as qwen  # noqa: E402 - stages 1 and 3, read-only

NL = chr(10)

BASE_FACTORY = qwen.BASE_FACTORY                  # 3e0a69af, the tree-scratch factory K64d onward serve
STAGE3_FACTORY = qwen.QWEN_FACTORY_STAGE3         # 06167779, K64f / K64g
STAGE4_FACTORY = '1634369677ae0247a387abb5610b8ee19f2b7a87d476534dd1e05bf6714529e3'   # K64i (recorded)

FLAG_TAIL, FLAG_SHARE, FLAG_SLICE, FLAG_READAHEAD = 0x1, 0x2, 0x4, 0x8
ABI_TAG = 0x51CE
READER_SLICE_NAME = 'reader_decode_qwen_slice.cpp'
WRITER_SLICE_NAME = 'writer_decode_qwen_slice.cpp'
# The reader's stage-4 compile-time args after F6's four (mask_tail, mask_width_t, kv_share, kv_ready id),
# and the writer's after its output accessor. The kernels (R6, W1) read them at these offsets.
READER_SUFFIX = ('pnht_full', 'rows_per_kv', 'kv_readahead', 'q_slice', 'tag')   # qwen_cta + 4 .. + 8
WRITER_SUFFIX = ('pnht_full', 'rows_per_kv', 'tag')                              # out_args.next() + 0 .. + 2


def lines(*parts):
    return ''.join(part + NL for part in parts)


F13_ANCHOR = lines('    constexpr uint32_t kQwenKvShare = 0x2u;')
F13 = F13_ANCHOR + lines(
    '    // [QWEN-SDPA] stage 4 (K1, optimisation/ttnn-op/sdpa_decode_slice): 0x4 head-sliced Q (the cores of a',
    '    // KV head compute only the Q row tiles that hold its rows), 0x8 the share leader\'s K/V read-ahead (K1b).',
    '    constexpr uint32_t kQwenQSlice = 0x4u;',
    '    constexpr uint32_t kQwenKvReadahead = 0x8u;',
    '    constexpr uint32_t kQwenSliceAbiTag = 0x51CEu;  // the slice kernels\' compile-time-arg ABI tag (R6, W1)')

F14_OLD = lines('    const uint32_t PNHt = PNH / q_heads_parallel_factor / TILE_HEIGHT;')
F14_NEW = lines(
    '    // [QWEN-SDPA] stage 4 (K1a, flag 0x4): every core of KV head h computes only the Q row tiles',
    '    // [floor(h*G/32), floor(h*G/32) + slice) that hold its G = num_q_heads / num_kv_heads folded rows;',
    '    // slice is the widest head\'s span (8-row groups: 3 -> 2 tiles). Every later PNHt use (CB sizes,',
    '    // subblocks, the compute, writer and reader CTA 1, MUL_BCAST_GRANULARITY, the F4 line) follows it.',
    '    // Without 0x4 PNHt is qwen_pnht_full, the value it always had, so those programs are K64g\'s.',
    '    const uint32_t qwen_pnht_full = PNH / q_heads_parallel_factor / TILE_HEIGHT;',
    '    const bool qwen_q_slice = (qwen_flags & kQwenQSlice) != 0;',
    '    const uint32_t qwen_rows_per_kv = num_q_heads / num_kv_heads;',
    '    uint32_t qwen_slice_tiles = 0;',
    '    if (qwen_q_slice) {',
    '        for (uint32_t h = 0; h < num_kv_heads; ++h) {',
    '            const uint32_t first = (h * qwen_rows_per_kv) / TILE_HEIGHT;',
    '            const uint32_t end = ((h + 1) * qwen_rows_per_kv + TILE_HEIGHT - 1) / TILE_HEIGHT;',
    '            qwen_slice_tiles = std::max(qwen_slice_tiles, end - first);',
    '        }',
    '    }',
    '    const uint32_t PNHt = qwen_q_slice ? qwen_slice_tiles : qwen_pnht_full;')

F15_DECL_ANCHOR = lines('    const bool qwen_kv_share = (qwen_flags & kQwenKvShare) != 0 && B > 1;')
F15_DECL = F15_DECL_ANCHOR + lines(
    '    // [QWEN-SDPA] K1b (flag 0x8): the share leader reads chunk n+1 while chunk n\'s multicast is in flight.',
    '    // Like 0x2 it is inert at B == 1; F15 refuses it without 0x2.',
    '    const bool qwen_kv_readahead = (qwen_flags & kQwenKvReadahead) != 0 && qwen_kv_share;')

F15_OLD = lines(
    '        TT_FATAL((qwen_flags & ~(kQwenMaskTail | kQwenKvShare)) == 0, "[QWEN-SDPA] unknown flags {:#x}", qwen_flags);')
F15_NEW = lines(
    '        TT_FATAL((qwen_flags & ~(kQwenMaskTail | kQwenKvShare | kQwenQSlice | kQwenKvReadahead)) == 0,',
    '                 "[QWEN-SDPA] unknown flags {:#x}", qwen_flags);',
    '        TT_FATAL((qwen_flags & kQwenKvReadahead) == 0 || (qwen_flags & kQwenKvShare) != 0,',
    '                 "[QWEN-SDPA] KV read-ahead needs KV share (0x2), flags {:#x}", qwen_flags);',
    '        if (qwen_q_slice) {',
    '            TT_FATAL(num_kv_heads > 1 && num_q_heads % num_kv_heads == 0,',
    '                     "[QWEN-SDPA] q-slice needs num_q_heads ({}) a multiple of num_kv_heads ({}) > 1",',
    '                     num_q_heads, num_kv_heads);',
    '            TT_FATAL(qwen_slice_tiles < qwen_pnht_full,',
    '                     "[QWEN-SDPA] q-slice saves no tile: {} rows per KV head span {} of {} row tiles",',
    '                     qwen_rows_per_kv, qwen_slice_tiles, qwen_pnht_full);',
    '            for (uint32_t h = 0; h < num_kv_heads; ++h) {',
    '                const uint32_t first = (h * qwen_rows_per_kv) / TILE_HEIGHT;',
    '                const uint32_t end = ((h + 1) * qwen_rows_per_kv + TILE_HEIGHT - 1) / TILE_HEIGHT;',
    '                TT_FATAL(first + qwen_slice_tiles <= qwen_pnht_full && end <= first + qwen_slice_tiles,',
    '                         "[QWEN-SDPA] q-slice does not cover KV head {}: tiles [{}, {}) of {}, slice {}",',
    '                         h, first, end, qwen_pnht_full, qwen_slice_tiles);',
    '            }',
    '            TT_FATAL(!use_attention_mask || attn_mask->padded_shape()[2] / TILE_HEIGHT == qwen_pnht_full,',
    '                     "[QWEN-SDPA] q-slice needs a mask with Q\'s {} row tiles", qwen_pnht_full);',
    '        }')

F16R_ANCHOR = lines(
    '        reader_compile_time_args_common.push_back(kv_ready_semaphore_id);',
    '    }')
F16R = F16R_ANCHOR + lines(
    '    if (qwen_q_slice || qwen_kv_readahead) {',
    '        // [QWEN-SDPA] stage 4 suffix for reader_decode_qwen_slice.cpp (R6), after F6\'s four. The slice reader',
    '        // serves 0x8 without 0x4 too (review F16/F17): q_slice tells it whether Q and the mask are sliced.',
    '        reader_compile_time_args_common.push_back(qwen_pnht_full);                            // +4',
    '        reader_compile_time_args_common.push_back(qwen_rows_per_kv);                          // +5',
    '        reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_kv_readahead));  // +6',
    '        reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_q_slice));       // +7',
    '        reader_compile_time_args_common.push_back(kQwenSliceAbiTag);                          // +8',
    '    }')

F16W_ANCHOR = lines('    tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args_common);')
F16W = F16W_ANCHOR + lines(
    '    if (qwen_q_slice) {',
    '        // [QWEN-SDPA] stage 4 suffix for writer_decode_qwen_slice.cpp (W1), after the output accessor.',
    '        writer_compile_time_args_common.push_back(qwen_pnht_full);',
    '        writer_compile_time_args_common.push_back(qwen_rows_per_kv);',
    '        writer_compile_time_args_common.push_back(kQwenSliceAbiTag);',
    '    }')

F17R_OLD = qwen.F8R_NEW
F17R_NEW = lines(
    '    reader_desc.kernel_source =',
    '        kernel_path + ((qwen_q_slice || qwen_kv_readahead) ? "dataflow/reader_decode_qwen_slice.cpp"',
    '                       : qwen_mode                         ? "dataflow/reader_decode_qwen.cpp"',
    '                                                           : "dataflow/reader_decode_all.cpp");')
F17W_OLD = lines('    writer_desc.kernel_source = kernel_path + "dataflow/writer_decode_all.cpp";')
F17W_NEW = lines(
    '    writer_desc.kernel_source =',
    '        kernel_path + (qwen_q_slice ? "dataflow/writer_decode_qwen_slice.cpp" : "dataflow/writer_decode_all.cpp");')

F18_ANCHOR = lines(
    '                 intermed_output_tiles / (out_tiles + 2 * PNHt), qwen_cb_bytes);',
    '    }')
F18 = F18_ANCHOR + lines(
    '    if (qwen_q_slice || qwen_kv_readahead) {',
    '        log_info(tt::LogOp, "[QWEN-SDPA] q-slice rows_per_kv={} pnht_full={} slice_tiles={} readahead={}",',
    '                 qwen_rows_per_kv, qwen_pnht_full, PNHt, qwen_kv_readahead);',
    '    }')

STAGE4_EDITS = (('F13', F13_ANCHOR, F13), ('F14', F14_OLD, F14_NEW), ('F15 decl', F15_DECL_ANCHOR, F15_DECL),
                ('F15', F15_OLD, F15_NEW), ('F16 reader', F16R_ANCHOR, F16R), ('F16 writer', F16W_ANCHOR, F16W),
                ('F17 reader', F17R_OLD, F17R_NEW), ('F17 writer', F17W_OLD, F17W_NEW), ('F18', F18_ANCHOR, F18))

# Format literals of the stage-4 factory; they end up in _ttnncpp.so's strings (build_k64i.sh step 7,
# the card-B harness's binary check, the arm's pooled_attention_replay binary markers).
SLICE_LOG_MARKER = '[QWEN-SDPA] q-slice rows_per_kv='
READAHEAD_REFUSAL = '[QWEN-SDPA] KV read-ahead needs KV share'
NO_SAVING_REFUSAL = '[QWEN-SDPA] q-slice saves no tile'
COVER_REFUSAL = '[QWEN-SDPA] q-slice does not cover KV head'
HEADS_REFUSAL = '[QWEN-SDPA] q-slice needs num_q_heads'
MASK_REFUSAL = "[QWEN-SDPA] q-slice needs a mask with Q's"
STAGE4_MARKERS = qwen.STAGE_MARKERS[3] + (SLICE_LOG_MARKER, READAHEAD_REFUSAL, NO_SAVING_REFUSAL, COVER_REFUSAL,
                                          HEADS_REFUSAL, MASK_REFUSAL, READER_SLICE_NAME, WRITER_SLICE_NAME)
STAGE4_ABSENT = qwen.STAGE_ABSENT[3]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stage3(source):
    """The stage-3 factory bytes from the base or the stage-3 factory itself; ValueError otherwise."""
    digest = sha256(source)
    if digest == STAGE3_FACTORY:
        return source
    if digest == BASE_FACTORY:
        built = qwen.patch(source, 3)
        if sha256(built) != STAGE3_FACTORY:
            raise ValueError('apply_factory_qwen stage 3 built %s, recorded %s' % (sha256(built), STAGE3_FACTORY))
        return built
    raise ValueError('unexpected factory %s (need the base %s or stage 3 %s)' % (digest, BASE_FACTORY, STAGE3_FACTORY))


def patch(source):
    """The stage-4 factory from the base or stage-3 bytes; ValueError on any other input."""
    text = stage3(source).decode('utf-8')
    for label, old, new in STAGE4_EDITS:
        count = text.count(old)
        if count != 1:
            raise ValueError('anchor %s occurs %d times' % (label, count))
        text = text.replace(old, new)
    for marker in STAGE4_MARKERS:
        if marker not in text:
            raise ValueError('stage-4 factory lacks %r' % marker)
    for marker in STAGE4_ABSENT:
        if marker in text:
            raise ValueError('stage-4 factory carries %r' % marker)
    return text.encode('utf-8')


def unpatch(patched, *, to_stage=3):
    """The inverse, last edit first: stage 4 -> stage 3 (to_stage=3), -> stage 1 (1) or -> the base (0)."""
    text = patched.decode('utf-8')
    for label, old, new in reversed(STAGE4_EDITS):
        if text.count(new) != 1:
            raise ValueError('edit %s occurs %d times' % (label, text.count(new)))
        text = text.replace(new, old)
    data = text.encode('utf-8')
    if to_stage == 3:
        return data
    return qwen.unpatch(data, 3, to_stage=1 if to_stage == 1 else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('factory', help='the 3e0a69af base or the 06167779 stage-3 factory')
    parser.add_argument('--out', help='write here instead of patching in place')
    parser.add_argument('--record', action='store_true', help='print the output sha instead of enforcing it')
    args = parser.parse_args(argv)
    path = Path(args.factory)
    source = path.read_bytes()
    if sha256(source) == STAGE4_FACTORY:
        print('factory already stage 4 %s' % STAGE4_FACTORY)
        if args.out:
            Path(args.out).write_bytes(source)
        return 0
    try:
        patched = patch(source)
    except ValueError as error:
        print('refused: %s' % error)
        return 1
    digest = sha256(patched)
    if not args.record and digest != STAGE4_FACTORY:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, STAGE4_FACTORY))
        return 1
    for stage, expected in ((3, STAGE3_FACTORY), (1, qwen.QWEN_FACTORY), (0, BASE_FACTORY)):
        if sha256(unpatch(patched, to_stage=stage)) != expected:
            print('stage-4 edits do not invert to %s (%s)' % ('the base' if stage == 0 else 'stage %d' % stage, expected[:16]))
            return 1
    if args.out:
        Path(args.out).write_bytes(patched)
        target = Path(args.out)
    else:
        backup = path.with_name(path.name + '.orig-' + sha256(source)[:8])
        if not backup.exists():
            backup.write_bytes(source)
        path.write_bytes(patched)
        target = path
    print('factory %s -> stage 4 %s written %s' % (sha256(source)[:16], digest, target))
    print(digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
