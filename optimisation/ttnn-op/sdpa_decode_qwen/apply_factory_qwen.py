"""Apply the [QWEN-SDPA] factory edits to sdpa_decode_program_factory.cpp: stage 1 (F1-F8) or stage 3 (F1-F12).

Input must be the tree-scratch patched factory, sha256 3e0a69af... (the one ~/opgraft-K64d
was built from); any other input is refused. Every anchor must occur exactly once, and the
result must hash to that stage's recorded output (QWEN_FACTORIES, so a changed edit list has
to update it deliberately). Run twice on an already patched file it reports and exits 0.

    python3 apply_factory_qwen.py <factory.cpp>                       # stage 1, in place, keep .orig-3e0a69af
    python3 apply_factory_qwen.py <factory.cpp> --stage 3 --out X.cpp # stage 3 into X.cpp, input untouched

Stage 1 (sdpa-onepass-spec.md 4.1, ~/opgraft-K64e, served): the tail-only mask, forced compact
tree scratch and the per-call sentinel in SDPAProgramConfig.q_chunk_size. The KV-share flag
(0x2) is refused by TT_FATAL. The default, so build_k64e.sh reproduces K64e unchanged.

Stage 3 (spec 7.1, ~/opgraft-K64f): F1-F8 plus F9-F12, the K/V leader multicast across twin
bundle entries. F9 replaces the stage-1 'KV share is not in this build' TT_FATAL with the
twin-band fit check, F10 places entry b's core p directly below entry 0's (twins vertically
adjacent, leader on top), F11 creates the READY semaphore (id 3), F12 gives each leader its
twins' column span and each twin its leader's NoC coordinate in the existing K-multicast
runtime-arg slots 15-19. Stage 3's edits apply to stage 1's output, so reverting them (last
first) gives the stage-1 factory byte for byte.
"""

import argparse
import hashlib
from pathlib import Path
import sys

NL = chr(10)

BASE_FACTORY = '3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a'
QWEN_FACTORY = '1b54abd3fe466a058046939e2ec0366505f80fde631dba387c591985fa3da301'      # stage 1 (K64e)
QWEN_FACTORY_STAGE3 = '06167779a979ba1f35c78d1c002f9956215ca52a4ccd4531e0f5f5fe4e44191d'  # stage 3 (K64f)
QWEN_FACTORIES = {1: QWEN_FACTORY, 3: QWEN_FACTORY_STAGE3}
STAGES = (1, 3)


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
F2_SHARE_REFUSED = lines(
    '        TT_FATAL((qwen_flags & kQwenKvShare) == 0, "[QWEN-SDPA] KV share is not in this build");  // STAGE 1 ONLY; F9 removes it')
F2 = F2_ANCHOR + lines(
    '',
    '    // ========== [QWEN-SDPA] preconditions ==========',
    '    const bool qwen_mask_tail = (qwen_flags & kQwenMaskTail) != 0;',
    '    const bool qwen_kv_share = (qwen_flags & kQwenKvShare) != 0 && B > 1;',
    '    const uint32_t qwen_mask_width_t = use_attention_mask ? attn_mask->padded_shape()[3] / TILE_WIDTH : St;',
    '    if (qwen_mode) {',
    '        TT_FATAL((qwen_flags & ~(kQwenMaskTail | kQwenKvShare)) == 0, "[QWEN-SDPA] unknown flags {:#x}", qwen_flags);') + \
    F2_SHARE_REFUSED + lines(
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
F5_LINE = lines(
    '    const uint32_t kv_ready_semaphore_id = 3;  // [QWEN-SDPA] KV share (created in stage 3, F11)')
F5 = F5_ANCHOR + F5_LINE

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

# ---------------------------------------------------------------------------------------------
# Stage 3 (spec 7.1): applied to the stage-1 output, in this order.
# ---------------------------------------------------------------------------------------------

# F9: the stage-1 refusal of flag 0x2 becomes the twin-band fit check. B twins of each of the
# num_cores_per_batch positions stack vertically, so the grid needs B bands of
# ceil(num_cores_per_batch / grid.x) rows: B=3 needs 9 of 10 rows, B=2 6, B=4 would need 12.
F9_NEW = lines(
    '        if (qwen_kv_share) {',
    '            TT_FATAL(((num_cores_per_batch + grid_size.x - 1) / grid_size.x) * B <= grid_size.y,',
    '                     "[QWEN-SDPA] KV-share twin bands do not fit the {}x{} grid for B={}", grid_size.x, grid_size.y, B);',
    '        }')

F10_ANCHOR = lines(
    '    } else {',
    '        // Q in DRAM, no sharding: simple linear assignment')
F10 = lines(
    '    } else if (qwen_kv_share) {',
    '        // [QWEN-SDPA] twin placement. Linear index i = b * num_cores_per_batch + p keeps every role',
    '        // (cur_batch, cur_head, core_num_in_reduce, tree params); only the coordinate moves. Entry',
    "        // b's core p sits at column p % grid.x, row (p / grid.x) * B + b: the B twins of each p are",
    '        // vertically adjacent with entry 0 (the leader) on top. B=3: rows 0-8 (14 cores idle); B=2: rows 0-5.',
    '        std::vector<uint8_t> used(num_cores_available, 0);',
    '        for (uint32_t i = 0; i < num_active_cores; ++i) {',
    '            const uint32_t b = i / num_cores_per_batch;',
    '            const uint32_t p = i % num_cores_per_batch;',
    '            const CoreCoord core = {p % grid_size.x, (p / grid_size.x) * B + b};',
    '            used[core.y * grid_size.x + core.x] = 1;',
    '            core_group.push_back(core);',
    '        }',
    '        for (uint32_t idx = 0; idx < num_cores_available; ++idx) {',
    '            if (!used[idx]) {',
    '                core_group_idle.push_back(CoreCoord{idx % grid_size.x, idx / grid_size.x});',
    '            }',
    '        }') + F10_ANCHOR

# F11: the READY semaphore; id 2 (k_mcast, unused in qwen mode: q_heads_parallel_factor == 1)
# serves as VALID.
F11 = F5_LINE + lines(
    '    if (qwen_kv_share) {',
    '        desc.semaphores.push_back(SemaphoreDescriptor{',
    '            .id = kv_ready_semaphore_id, .core_type = tt::CoreType::WORKER, .core_ranges = core_grid, .initial_value = 0});',
    '    }')

F12_ANCHOR = lines(
    '            core_num_in_output = i % num_cores_per_batch;',
    '        }')
F12 = lines(
    '            core_num_in_output = i % num_cores_per_batch;',
    '            if (qwen_kv_share) {',
    "                // [QWEN-SDPA] the existing K-multicast slots 15-19: a leader (entry 0) multicasts to its",
    "                // B-1 twins directly below; a twin gets its leader's NoC coordinate in mcast_x/mcast_y0.",
    '                const uint32_t p = i % num_cores_per_batch;',
    '                const CoreCoord leader{p % grid_size.x, (p / grid_size.x) * B};',
    '                num_dests = B - 1;',
    '                if (cur_batch == 0) {',
    '                    do_k_mcast = true;',
    '                    const auto first = device->worker_core_from_logical_core(CoreCoord{leader.x, leader.y + 1});',
    '                    const auto last = device->worker_core_from_logical_core(CoreCoord{leader.x, leader.y + B - 1});',
    '                    TT_FATAL(first.x == last.x && last.y >= first.y && last.y - first.y == B - 2,',
    '                             "[QWEN-SDPA] KV-share twins not vertically contiguous: ({},{})..({},{})", first.x, first.y, last.x, last.y);',
    '                    mcast_x = first.x;',
    '                    mcast_y0 = first.y;',
    '                    mcast_y1 = last.y;',
    '                } else {',
    '                    const auto phys = device->worker_core_from_logical_core(leader);',
    "                    mcast_x = phys.x;  // follower: the leader's NoC coordinate",
    '                    mcast_y0 = phys.y;',
    '                    mcast_y1 = phys.y;',
    '                }',
    '            }',
    '        }')

STAGE3_EDITS = (('F9', F2_SHARE_REFUSED, F9_NEW), ('F10', F10_ANCHOR, F10), ('F11', F5_LINE, F11),
                ('F12', F12_ANCHOR, F12))
STAGE_EDITS = {1: EDITS, 3: EDITS + STAGE3_EDITS}

# In the output, the only places the legacy path can differ: every new branch is gated on qwen_mode.
MARKERS = ('[QWEN-SDPA] flags=', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS', 'reader_decode_qwen.cpp', 'sdpa_flash_decode_qwen.cpp')
# Format literals that end up in _ttnncpp.so's strings: the stage-3 build has the first and not
# the second; the stage-1 build the reverse. build_k64f.sh and the card-M test read them.
SHARE_MARKER = '[QWEN-SDPA] KV-share twin bands'
STAGE1_SHARE_REFUSAL = '[QWEN-SDPA] KV share is not in this build'
STAGE_MARKERS = {1: MARKERS + (STAGE1_SHARE_REFUSAL,), 3: MARKERS + (SHARE_MARKER,)}
STAGE_ABSENT = {1: (SHARE_MARKER,), 3: (STAGE1_SHARE_REFUSAL,)}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def patch(source, stage=1):
    """The stage's factory from the 3e0a69af bytes; ValueError on any other input."""
    if stage not in STAGES:
        raise ValueError('unknown stage %r' % (stage,))
    digest = sha256(source)
    if digest != BASE_FACTORY:
        raise ValueError('unexpected factory %s (need %s)' % (digest, BASE_FACTORY))
    text = source.decode('utf-8')
    for label, old, new in STAGE_EDITS[stage]:
        count = text.count(old)
        if count != 1:
            raise ValueError('anchor %s occurs %d times' % (label, count))
        text = text.replace(old, new)
    for marker in STAGE_MARKERS[stage]:
        if marker not in text:
            raise ValueError('patched factory lacks %r' % marker)
    for marker in STAGE_ABSENT[stage]:
        if marker in text:
            raise ValueError('stage %d factory carries %r' % (stage, marker))
    return text.encode('utf-8')


def unpatch(patched, stage=1, *, to_stage=0):
    """The inverse, last edit first: proves the output is the base plus exactly these edits.
    to_stage=1 stops after undoing the stage-3 edits (stage 3 -> stage 1)."""
    text = patched.decode('utf-8')
    edits = STAGE_EDITS[stage]
    keep = len(STAGE_EDITS[to_stage]) if to_stage else 0
    for label, old, new in reversed(edits[keep:]):
        if text.count(new) != 1:
            raise ValueError('edit %s occurs %d times' % (label, text.count(new)))
        text = text.replace(new, old)
    return text.encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('factory')
    parser.add_argument('--stage', type=int, choices=STAGES, default=1,
                        help='1: F1-F8 (K64e, the default); 3: F1-F12 (K64f)')
    parser.add_argument('--out', help='write here instead of patching in place')
    parser.add_argument('--record', action='store_true', help='print the output sha instead of enforcing it')
    args = parser.parse_args(argv)
    target_sha = QWEN_FACTORIES[args.stage]
    path = Path(args.factory)
    source = path.read_bytes()
    if sha256(source) == target_sha:
        print('factory already qwen-patched (stage %d) %s' % (args.stage, target_sha))
        if args.out:
            Path(args.out).write_bytes(source)
        return 0
    try:
        patched = patch(source, args.stage)
    except ValueError as error:
        print('refused: %s' % error)
        return 1
    digest = sha256(patched)
    if not args.record and digest != target_sha:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, target_sha))
        return 1
    if unpatch(patched, args.stage) != source:
        print('edits do not invert')
        return 1
    if args.stage == 3 and sha256(unpatch(patched, 3, to_stage=1)) != QWEN_FACTORY:
        print('stage-3 edits do not invert to the stage-1 factory')
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
    print('factory %s -> stage %d %s written %s' % (BASE_FACTORY[:16], args.stage, digest, target))
    print(digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
