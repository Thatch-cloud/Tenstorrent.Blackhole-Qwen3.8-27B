"""Apply the [QWEN-SDPA] stage-1 factory edits F1-F8 to sdpa_decode_program_factory.cpp.

Input must be the tree-scratch patched factory, sha256 3e0a69af... (the one ~/opgraft-K64d
was built from); any other input is refused. Every anchor must occur exactly once, and the
result must hash to QWEN_FACTORY (recorded here, so a changed edit list has to update it
deliberately). Run twice on an already patched file it reports and exits 0.

    python3 apply_factory_qwen.py <factory.cpp>             # patch in place, keep <factory.cpp>.orig-3e0a69af
    python3 apply_factory_qwen.py <factory.cpp> --out X.cpp # write X.cpp, leave the input alone

Stage 1 only (sdpa-onepass-spec.md 4.1): the tail-only mask, forced compact tree scratch
and the per-call sentinel in SDPAProgramConfig.q_chunk_size. The KV-share flag (0x2) is
refused by TT_FATAL; F9-F12 (stage 3) are not in this build.
"""

import argparse
import hashlib
from pathlib import Path
import sys

NL = chr(10)

BASE_FACTORY = '3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a'
QWEN_FACTORY = '1b54abd3fe466a058046939e2ec0366505f80fde631dba387c591985fa3da301'


def lines(*parts):
    return ''.join(part + NL for part in parts)


F1_ANCHOR = lines('    const bool share_cache = operation_attributes.share_cache.value_or(false);')
F1 = F1_ANCHOR + lines(
    '    // [QWEN-SDPA] per-call decode modes, carried in SDPAProgramConfig.q_chunk_size. The decode',
    '    // path never reads q_chunk_size (sdpa_decode.cpp:131-135 only logs it) and this op has no',
    '    // custom program hash, so the sentinel keys its own program-cache entry.',
    '    constexpr std::size_t kQwenMagicMask = 0xFFFFFF00u;',
    '    constexpr std::size_t kQwenMagic = 0x51DEC000u;',
    '    constexpr uint32_t kQwenMaskTail = 0x1u;',
    '    constexpr uint32_t kQwenKvShare = 0x2u;',
    '    const bool qwen_mode =',
    '        program_config.has_value() && (program_config->q_chunk_size & kQwenMagicMask) == kQwenMagic;',
    '    const uint32_t qwen_flags = qwen_mode ? static_cast<uint32_t>(program_config->q_chunk_size & 0xFFu) : 0u;')

F2_ANCHOR = lines(
    '    const tt::DataFormat page_table_df =',
    '        is_paged_attention ? tt_metal::datatype_to_dataformat_converter(page_table_tensor.value().dtype())',
    '                           : tt::DataFormat::Invalid;')
F2 = F2_ANCHOR + lines(
    '',
    '    // ========== [QWEN-SDPA] preconditions ==========',
    '    const bool qwen_mask_tail = (qwen_flags & kQwenMaskTail) != 0;',
    '    const bool qwen_kv_share = (qwen_flags & kQwenKvShare) != 0 && B > 1;',
    '    const uint32_t qwen_mask_width_t = use_attention_mask ? attn_mask->padded_shape()[3] / TILE_WIDTH : St;',
    '    if (qwen_mode) {',
    '        TT_FATAL((qwen_flags & ~(kQwenMaskTail | kQwenKvShare)) == 0, "[QWEN-SDPA] unknown flags {:#x}", qwen_flags);',
    '        TT_FATAL((qwen_flags & kQwenKvShare) == 0, "[QWEN-SDPA] KV share is not in this build");  // STAGE 1 ONLY; F9 removes it',
    '        TT_FATAL(is_paged_attention && !is_page_table_sharded, "[QWEN-SDPA] needs an interleaved paged page table");',
    '        TT_FATAL(!is_causal && !use_cur_pos_tensor && sliding_window_size == 0,',
    '                 "[QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor");',
    '        TT_FATAL(!use_mla && !is_q_sharded && !is_output_sharded && !on_subcoregrid && !use_attention_sink && !tilize_q,',
    '                 "[QWEN-SDPA] needs DRAM tiled Q and output, no MLA, sink or sub-core grids");',
    '        TT_FATAL(tensor_args.v.has_value() && !apply_geometry_override && capacity_t == 0 && !has_block_padding,',
    '                 "[QWEN-SDPA] needs explicit V, native cache geometry, no wrap and no block padding");',
    '        TT_FATAL(q_heads_parallel_factor == 1 && num_heads_per_core == 1, "[QWEN-SDPA] needs one KV head per core");',
    '        TT_FATAL(Sk_chunk_t > 0 && St % Sk_chunk_t == 0, "[QWEN-SDPA] needs a fixed chunk that divides the cache");',
    '        if (qwen_mask_tail) {',
    '            TT_FATAL(use_attention_mask && (qwen_mask_width_t == St || qwen_mask_width_t == Sk_chunk_t),',
    '                     "[QWEN-SDPA] tail mask must be full width ({}) or one chunk ({}), got {}", St, Sk_chunk_t, qwen_mask_width_t);',
    '        } else {',
    '            TT_FATAL(!use_attention_mask || qwen_mask_width_t == St, "[QWEN-SDPA] a narrow mask needs the tail flag");',
    '        }',
    '    }')

F3_OLD = lines(
    '    const bool compact_tree_scratch = scratch_rounds_override && std::string(scratch_rounds_override) == "1";')
F3_NEW = lines(
    '    // Compact scratch drops only never-used slots (writer indexes slots by round < rounds), so',
    '    // forcing it in [QWEN-SDPA] mode changes no data flow; it is what lets PNHt=3 fit L1.',
    '    const bool compact_tree_scratch =',
    '        qwen_mode || (scratch_rounds_override && std::string(scratch_rounds_override) == "1");')

F4_ANCHOR = lines(
    '    add_cb(',
    '        CBIndex::c_20,',
    '        out_tiles * out_tile_size,',
    '        out_df,',
    '        out_tile_size,',
    '        &out_tile,',
    '        is_output_sharded ? out_buffer : nullptr);')
F4 = F4_ANCHOR + lines(
    '    if (qwen_mode) {',
    '        uint32_t qwen_cb_bytes = 0;',
    '        for (const auto& cb : desc.cbs) {',
    '            qwen_cb_bytes += cb.total_size;',
    '        }',
    '        log_info(tt::LogOp, "[QWEN-SDPA] flags={:#x} B={} PNHt={} St={} mask_width_t={} kv_share={} scratch_slots={} cb_bytes={}",',
    '                 qwen_flags, B, PNHt, St, qwen_mask_width_t, qwen_kv_share,',
    '                 intermed_output_tiles / (out_tiles + 2 * PNHt), qwen_cb_bytes);',
    '    }')

F5_ANCHOR = lines(
    '    desc.semaphores.push_back(SemaphoreDescriptor{',
    '        .id = k_mcast_semaphore_id, .core_type = tt::CoreType::WORKER, .core_ranges = core_grid, .initial_value = 0});')
F5 = F5_ANCHOR + lines(
    '    const uint32_t kv_ready_semaphore_id = 3;  // [QWEN-SDPA] KV share (created in stage 3, F11)')

F6_ANCHOR = lines(
    '    if (use_attention_sink) {',
    '        tt_metal::TensorAccessorArgs(*attention_sink->buffer()).append_to(reader_compile_time_args_common);',
    '    } else {',
    '        tt_metal::TensorAccessorArgs(static_cast<const Buffer*>(nullptr)).append_to(reader_compile_time_args_common);',
    '    }')
F6 = F6_ANCHOR + lines(
    '    if (qwen_mode) {',
    '        // [QWEN-SDPA] suffix compile-time args, after every TensorAccessorArgs block (legacy <37> never moves).',
    '        reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_mask_tail));',
    '        reader_compile_time_args_common.push_back(qwen_mask_width_t);',
    '        reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_kv_share));',
    '        reader_compile_time_args_common.push_back(kv_ready_semaphore_id);',
    '    }')

F7_ANCHOR = lines(
    '        sliding_window_size,',
    '        num_tree_reduction_rounds,',
    '        original_block_size,',
    '    };',
    '',
    '    // ========== Compute Defines ==========')
F7 = lines(
    '        sliding_window_size,',
    '        num_tree_reduction_rounds,',
    '        original_block_size,',
    '    };',
    '    if (qwen_mode) {',
    '        compute_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_mask_tail));  // index 32',
    '    }',
    '',
    '    // ========== Compute Defines ==========')

F8R_OLD = lines('    reader_desc.kernel_source = kernel_path + "dataflow/reader_decode_all.cpp";')
F8R_NEW = lines(
    '    reader_desc.kernel_source =',
    '        kernel_path + (qwen_mode ? "dataflow/reader_decode_qwen.cpp" : "dataflow/reader_decode_all.cpp");')
F8C_OLD = lines('    compute_desc.kernel_source = kernel_path + "compute/sdpa_flash_decode.cpp";')
F8C_NEW = lines(
    '    compute_desc.kernel_source =',
    '        kernel_path + (qwen_mode ? "compute/sdpa_flash_decode_qwen.cpp" : "compute/sdpa_flash_decode.cpp");')

EDITS = (('F1', F1_ANCHOR, F1), ('F2', F2_ANCHOR, F2), ('F3', F3_OLD, F3_NEW), ('F4', F4_ANCHOR, F4),
         ('F5', F5_ANCHOR, F5), ('F6', F6_ANCHOR, F6), ('F7', F7_ANCHOR, F7), ('F8 reader', F8R_OLD, F8R_NEW),
         ('F8 compute', F8C_OLD, F8C_NEW))

# In the output, the only places the legacy path can differ: every new branch is gated on qwen_mode.
MARKERS = ('[QWEN-SDPA] flags=', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS', 'reader_decode_qwen.cpp', 'sdpa_flash_decode_qwen.cpp')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def patch(source):
    """The F1-F8 factory from the 3e0a69af bytes; ValueError on any other input."""
    digest = sha256(source)
    if digest != BASE_FACTORY:
        raise ValueError('unexpected factory %s (need %s)' % (digest, BASE_FACTORY))
    text = source.decode('utf-8')
    for label, old, new in EDITS:
        count = text.count(old)
        if count != 1:
            raise ValueError('anchor %s occurs %d times' % (label, count))
        text = text.replace(old, new)
    for marker in MARKERS:
        if marker not in text:
            raise ValueError('patched factory lacks %r' % marker)
    return text.encode('utf-8')


def unpatch(patched):
    """The inverse, last edit first: proves the output is the base plus exactly these edits."""
    text = patched.decode('utf-8')
    for label, old, new in reversed(EDITS):
        if text.count(new) != 1:
            raise ValueError('edit %s occurs %d times' % (label, text.count(new)))
        text = text.replace(new, old)
    return text.encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('factory')
    parser.add_argument('--out', help='write here instead of patching in place')
    parser.add_argument('--record', action='store_true', help='print the output sha instead of enforcing it')
    args = parser.parse_args(argv)
    path = Path(args.factory)
    source = path.read_bytes()
    if sha256(source) == QWEN_FACTORY:
        print('factory already qwen-patched %s' % QWEN_FACTORY)
        if args.out:
            Path(args.out).write_bytes(source)
        return 0
    try:
        patched = patch(source)
    except ValueError as error:
        print('refused: %s' % error)
        return 1
    digest = sha256(patched)
    if not args.record and digest != QWEN_FACTORY:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, QWEN_FACTORY))
        return 1
    if unpatch(patched) != source:
        print('edits do not invert')
        return 1
    if args.out:
        Path(args.out).write_bytes(patched)
        target = Path(args.out)
    else:
        backup = path.with_name(path.name + '.orig-' + BASE_FACTORY[:8])
        if not backup.exists():
            backup.write_bytes(source)
        path.write_bytes(patched)
        target = path
    print('factory %s -> %s written %s' % (BASE_FACTORY[:16], digest, target))
    print(digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
