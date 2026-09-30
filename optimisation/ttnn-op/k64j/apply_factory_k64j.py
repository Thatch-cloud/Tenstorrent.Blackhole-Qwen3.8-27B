"""Apply the K64j factory edits (F19-F22) to sdpa_decode_program_factory.cpp: the runtime extent, flag 0x20.

K64j (optimisation/ttnn-op/k64j_probe/README.md, "K64j itself: the edit list and the build") is applied to K64i's
stage-4 factory (1634369677ae0247..., ../sdpa_decode_slice/apply_factory_slice.py) and adds one per-call flag to the
q_chunk_size sentinel:

  0x20 runtime extent: the reader, compute and writer take each entry's position from the cur_pos tensor (E - 1,
       E = the entry's 256-key family, or UINT32_MAX to skip the entry) instead of the compile-time St * 32 - 1.
       The card-B P0 probe showed the split, the tree, the trace re-read and the skip on the causal path equal the
       compile-time call at capacity E (run 36218136852, 70/70; pass 2 run 36218529407, 638/638).

  F19  kQwenRuntimeExtent = 0x20; the unknown-flag mask admits it (0x10 stays unknown: the card tests' control);
       qwen_runtime_extent, declared beside qwen_kv_readahead
  F20  the K64i refusal of a cur_pos tensor spares 0x20 (its literal kept byte for byte, so every QWEN string of
       K64i stays in the binary); under 0x20: the tail flag (0x1) and the narrow one-chunk mask, an interleaved
       cur_pos tensor, int32 and row-major, of B entries. Causal stays refused for every qwen mode.
  F21  compile-time args: runtime_extent after every other reader suffix (reader_decode_qwen.cpp +4,
       reader_decode_qwen_slice.cpp +9), compute CTA 33; writer_decode_qwen_slice.cpp for EVERY 0x20 program
       (the stock writer never reads the runtime word, and a writer left on St * 32 - 1 hangs: k64j_probe R1),
       with its suffix +3 q_slice, +4 runtime_extent after K64i's three. c_8 and c_15 already exist whenever a
       cur_pos tensor is passed, so every 0x20 program's cb_bytes grows by two sticks.
  F22  one log line per 0x20 program: '[QWEN-SDPA] runtime-extent entries=.. kv_share=.. q_slice=.. writer=..
       cur_pos_stick_bytes=..'; its format literal is the binary marker the extent mode and the gate require.

Input: the tree-scratch base 3e0a69af, the stage-3 factory 06167779 or the stage-4 factory 16343696 (stages 3 and 4
are applied first, by the K64i generators, read-only); anything else is refused. Every anchor must occur exactly
once, the result must hash to K64J_FACTORY, and the edits must invert: K64j -> stage 4 -> stage 3 -> stage 1 -> the
base. Run on an already patched file it reports and exits 0.

    python3 apply_factory_k64j.py <factory.cpp> --out X.cpp   # into X.cpp, the input untouched
    python3 apply_factory_k64j.py <factory.cpp>               # in place, keeping .orig-<sha8>
    python3 apply_factory_k64j.py <factory.cpp> --record      # print the output sha instead of enforcing it
"""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SLICE_DIR = HERE.parent / 'sdpa_decode_slice'
QWEN_DIR = HERE.parent / 'sdpa_decode_qwen'
for _path in (str(SLICE_DIR), str(QWEN_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import apply_factory_qwen as qwen  # noqa: E402 - stages 1 and 3, read-only
import apply_factory_slice as slice_factory  # noqa: E402 - stage 4 (K64i), read-only

NL = chr(10)

BASE_FACTORY = slice_factory.BASE_FACTORY           # 3e0a69af, the tree-scratch factory
STAGE3_FACTORY = slice_factory.STAGE3_FACTORY       # 06167779, K64f / K64g
STAGE4_FACTORY = slice_factory.STAGE4_FACTORY       # 16343696, K64i
K64J_FACTORY = 'bb4dc6a759d40054d31089792f617a1d66f5837da2065fa12f61438d09a55080'   # K64j (recorded)

FLAG_TAIL, FLAG_SHARE, FLAG_SLICE, FLAG_READAHEAD = 0x1, 0x2, 0x4, 0x8
FLAG_EXTENT = 0x20
UNKNOWN_FLAG_CONTROL = 0x10          # the card tests' unknown-flag control: F19 must keep refusing it
KNOWN_FLAGS = FLAG_TAIL | FLAG_SHARE | FLAG_SLICE | FLAG_READAHEAD | FLAG_EXTENT
WRITER_SLICE_NAME = slice_factory.WRITER_SLICE_NAME
READER_QWEN_NAME = 'reader_decode_qwen.cpp'
READER_SLICE_NAME = slice_factory.READER_SLICE_NAME
# Where each kernel reads runtime_extent (F21): the reader's offset from qwen_cta, the compute CTA index, and the
# slice writer's suffix after its output accessor (K64i's three, then K64j's two).
READER_EXTENT_OFFSET = {READER_QWEN_NAME: 4, READER_SLICE_NAME: 9}
COMPUTE_EXTENT_CTA = 33
WRITER_SUFFIX = slice_factory.WRITER_SUFFIX + ('q_slice', 'runtime_extent')


def lines(*parts):
    return ''.join(part + NL for part in parts)


F19_ANCHOR = lines("    constexpr uint32_t kQwenSliceAbiTag = 0x51CEu;  // the slice kernels' compile-time-arg ABI tag (R6, W1)")
F19 = F19_ANCHOR + lines(
    '    // [QWEN-SDPA] K64j (optimisation/ttnn-op/k64j): 0x20 the runtime extent. The reader, compute and writer take',
    "    // each entry's position from the cur_pos tensor (E - 1, E the entry's 256-key family; UINT32_MAX skips it)",
    '    // instead of the compile-time St * 32 - 1. 0x10 stays unknown: it is the card tests\' unknown-flag control.',
    '    constexpr uint32_t kQwenRuntimeExtent = 0x20u;')

F19_MASK_OLD = lines(
    '        TT_FATAL((qwen_flags & ~(kQwenMaskTail | kQwenKvShare | kQwenQSlice | kQwenKvReadahead)) == 0,')
F19_MASK_NEW = lines(
    '        TT_FATAL((qwen_flags & ~(kQwenMaskTail | kQwenKvShare | kQwenQSlice | kQwenKvReadahead | kQwenRuntimeExtent)) == 0,')

F19_DECL_ANCHOR = lines('    const bool qwen_kv_readahead = (qwen_flags & kQwenKvReadahead) != 0 && qwen_kv_share;')
F19_DECL = F19_DECL_ANCHOR + lines(
    '    // [QWEN-SDPA] K64j (flag 0x20): F20 checks its preconditions, F21 hands it to the kernels, F22 logs it.',
    '    const bool qwen_runtime_extent = (qwen_flags & kQwenRuntimeExtent) != 0;')

# The K64i text of the cur_pos refusal. Its literal stays byte for byte (build step 6 requires every QWEN string of
# K64i in the new binary); only the condition spares 0x20.
CUR_POS_REFUSAL = '[QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor'
F20_OLD = lines(
    '        TT_FATAL(!is_causal && !use_cur_pos_tensor && sliding_window_size == 0,',
    '                 "%s");' % CUR_POS_REFUSAL)
EXTENT_MASK_REFUSAL = '[QWEN-SDPA] runtime extent (0x20) needs the tail flag (0x1) and a narrow'
EXTENT_CUR_POS_REFUSAL = '[QWEN-SDPA] runtime extent (0x20) needs an interleaved cur_pos tensor'
EXTENT_LAYOUT_REFUSAL = '[QWEN-SDPA] runtime extent (0x20) needs an int32 row-major cur_pos tensor of B='
F20_NEW = lines(
    '        // [QWEN-SDPA] K64j F20: no cur_pos tensor unless 0x20. The literal is K64i\'s, byte for byte, so every QWEN',
    '        // string of K64i stays in the binary; causal stays refused for every qwen mode.',
    '        TT_FATAL(!is_causal && (!use_cur_pos_tensor || qwen_runtime_extent) && sliding_window_size == 0,',
    '                 "%s");' % CUR_POS_REFUSAL,
    '        if (qwen_runtime_extent) {',
    "            // [QWEN-SDPA] K64j F20: R10 reads the tail mask's one chunk at offset 0 (a wide mask would be read at",
    '            // [C - 256, C), not [E - 256, E)), and one int32 word per entry from page 0 of the cur_pos tensor.',
    '            TT_FATAL(qwen_mask_tail && use_attention_mask && qwen_mask_width_t == Sk_chunk_t,',
    '                     "%s {}-tile mask, got flags {:#x} and mask width {}",' % EXTENT_MASK_REFUSAL,
    '                     Sk_chunk_t, qwen_flags, qwen_mask_width_t);',
    '            TT_FATAL(use_cur_pos_tensor && !is_cur_pos_tensor_sharded,',
    '                     "%s");' % EXTENT_CUR_POS_REFUSAL,
    '            TT_FATAL(cur_pos_tensor->dtype() == DataType::INT32 && cur_pos_tensor->layout() == Layout::ROW_MAJOR &&',
    '                         cur_pos_tensor->padded_shape()[-1] == B,',
    '                     "%s{} entries", B);' % EXTENT_LAYOUT_REFUSAL,
    '        }')

# F16's reader suffix ends with the ABI tag at +8; runtime_extent follows every reader suffix.
F21_READER_ANCHOR = lines(
    '        reader_compile_time_args_common.push_back(kQwenSliceAbiTag);                          // +8',
    '    }')
F21_READER = F21_READER_ANCHOR + lines(
    '    if (qwen_mode) {',
    '        // [QWEN-SDPA] K64j F21: runtime_extent after every other reader suffix, so no K64i offset moves: +4 of',
    "        // reader_decode_qwen.cpp (after F6's four), +9 of reader_decode_qwen_slice.cpp (after F16's five) (R10).",
    '        reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_runtime_extent));',
    '    }')

F21_WRITER_OLD = slice_factory.F16W[len(slice_factory.F16W_ANCHOR):]
F21_WRITER_NEW = lines(
    '    if (qwen_q_slice || qwen_runtime_extent) {',
    '        // [QWEN-SDPA] stage 4 suffix for writer_decode_qwen_slice.cpp (W1), after the output accessor. K64j F21: F17',
    '        // selects that writer for every 0x20 program too (the stock writer never reads the runtime word), and W4',
    '        // reads +3 q_slice (0: PNHt is pnht_full and q_tile_start 0) and +4 runtime_extent.',
    '        writer_compile_time_args_common.push_back(qwen_pnht_full);                             // +0',
    '        writer_compile_time_args_common.push_back(qwen_rows_per_kv);                           // +1',
    '        writer_compile_time_args_common.push_back(kQwenSliceAbiTag);                           // +2',
    '        writer_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_q_slice));        // +3',
    '        writer_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_runtime_extent)); // +4',
    '    }')

F21_COMPUTE_OLD = lines(
    '        compute_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_mask_tail));  // index 32')
F21_COMPUTE_NEW = F21_COMPUTE_OLD + lines(
    '        compute_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_runtime_extent));  // index 33 (K64j F21, C3)')

F21_KERNEL_OLD = slice_factory.F17W_NEW
F21_KERNEL_NEW = lines(
    '    writer_desc.kernel_source =',
    '        kernel_path + ((qwen_q_slice || qwen_runtime_extent) ? "dataflow/writer_decode_qwen_slice.cpp"',
    '                                                             : "dataflow/writer_decode_all.cpp");')

EXTENT_LOG_MARKER = '[QWEN-SDPA] runtime-extent entries='
F22_ANCHOR = lines(
    '                 qwen_rows_per_kv, qwen_pnht_full, PNHt, qwen_kv_readahead);',
    '    }')
F22 = F22_ANCHOR + lines(
    '    if (qwen_runtime_extent) {',
    '        log_info(tt::LogOp,',
    '                 "%s{} kv_share={} q_slice={} writer=writer_decode_qwen_slice.cpp cur_pos_stick_bytes={}",'
    % EXTENT_LOG_MARKER,
    '                 B, qwen_kv_share, qwen_q_slice, cur_pos_stick_size);',
    '    }')

K64J_EDITS = (('F19', F19_ANCHOR, F19), ('F19 mask', F19_MASK_OLD, F19_MASK_NEW), ('F19 decl', F19_DECL_ANCHOR, F19_DECL),
              ('F20', F20_OLD, F20_NEW), ('F21 reader', F21_READER_ANCHOR, F21_READER),
              ('F21 writer', F21_WRITER_OLD, F21_WRITER_NEW), ('F21 compute', F21_COMPUTE_OLD, F21_COMPUTE_NEW),
              ('F21 kernel', F21_KERNEL_OLD, F21_KERNEL_NEW), ('F22', F22_ANCHOR, F22))

# Format literals of the K64j factory; they end up in _ttnncpp.so's strings (build_k64j.sh step 6, the extent
# mode of pooled_attention_replay, the gate).
K64J_MARKERS = slice_factory.STAGE4_MARKERS + (EXTENT_LOG_MARKER, EXTENT_MASK_REFUSAL, EXTENT_CUR_POS_REFUSAL,
                                               EXTENT_LAYOUT_REFUSAL, CUR_POS_REFUSAL)
K64J_ABSENT = slice_factory.STAGE4_ABSENT


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stage4(source):
    """The stage-4 factory bytes from the base, the stage-3 or the stage-4 factory; ValueError otherwise."""
    digest = sha256(source)
    if digest == STAGE4_FACTORY:
        return source
    if digest in (BASE_FACTORY, STAGE3_FACTORY):
        built = slice_factory.patch(source)
        if sha256(built) != STAGE4_FACTORY:
            raise ValueError('apply_factory_slice built %s, recorded %s' % (sha256(built), STAGE4_FACTORY))
        return built
    raise ValueError('unexpected factory %s (need the base %s, stage 3 %s or stage 4 %s)'
                     % (digest, BASE_FACTORY, STAGE3_FACTORY, STAGE4_FACTORY))


def patch(source):
    """The K64j factory from the base, stage-3 or stage-4 bytes; ValueError on any other input."""
    text = stage4(source).decode('utf-8')
    for label, old, new in K64J_EDITS:
        count = text.count(old)
        if count != 1:
            raise ValueError('anchor %s occurs %d times' % (label, count))
        text = text.replace(old, new)
    for marker in K64J_MARKERS:
        if marker not in text:
            raise ValueError('K64j factory lacks %r' % marker)
    for marker in K64J_ABSENT:
        if marker in text:
            raise ValueError('K64j factory carries %r' % marker)
    return text.encode('utf-8')


def unpatch(patched, *, to_stage=4):
    """The inverse, last edit first: K64j -> stage 4 (to_stage=4), -> stage 3, -> stage 1 or -> the base (0)."""
    text = patched.decode('utf-8')
    for label, old, new in reversed(K64J_EDITS):
        if text.count(new) != 1:
            raise ValueError('edit %s occurs %d times' % (label, text.count(new)))
        text = text.replace(new, old)
    data = text.encode('utf-8')
    if to_stage == 4:
        return data
    return slice_factory.unpatch(data, to_stage=to_stage)


INVERSES = ((4, STAGE4_FACTORY), (3, STAGE3_FACTORY), (1, qwen.QWEN_FACTORY), (0, BASE_FACTORY))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('factory', help='the 3e0a69af base, the 06167779 stage-3 or the 16343696 stage-4 factory')
    parser.add_argument('--out', help='write here instead of patching in place')
    parser.add_argument('--record', action='store_true', help='print the output sha instead of enforcing it')
    args = parser.parse_args(argv)
    path = Path(args.factory)
    source = path.read_bytes()
    if sha256(source) == K64J_FACTORY:
        print('factory already K64j %s' % K64J_FACTORY)
        if args.out:
            Path(args.out).write_bytes(source)
        return 0
    try:
        patched = patch(source)
    except ValueError as error:
        print('refused: %s' % error)
        return 1
    digest = sha256(patched)
    if not args.record and digest != K64J_FACTORY:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, K64J_FACTORY))
        return 1
    for stage, expected in INVERSES:
        if sha256(unpatch(patched, to_stage=stage)) != expected:
            print('K64j edits do not invert to %s (%s)' % ('the base' if stage == 0 else 'stage %d' % stage, expected[:16]))
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
    print('factory %s -> K64j %s written %s' % (sha256(source)[:16], digest, target))
    print(digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
