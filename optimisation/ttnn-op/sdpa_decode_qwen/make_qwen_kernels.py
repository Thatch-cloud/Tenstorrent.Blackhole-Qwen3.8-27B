"""Build the two [QWEN-SDPA] kernels (stage 1 or stage 3) from the served originals, byte for byte.

stage 1 (K64e, served; the default, in this directory):
  reader_decode_qwen.cpp     = reader_decode_all.cpp  (sha256 49a05926...) + R1, R2, R3
  sdpa_flash_decode_qwen.cpp = sdpa_flash_decode.cpp  (sha256 d24769bd...) + C1, C2
stage 3 (K64f, in stage3/):
  reader_decode_qwen.cpp     = the stage-1 reader + R4 (K/V leader multicast), R5 (exit barrier)
  sdpa_flash_decode_qwen.cpp = the stage-1 compute, unchanged

(spec: sdpa-onepass-spec.md sections 4.2, 4.3 and 7.2). The originals are taken either from
the probe dump (probe_sdpa_decode_sources.py output: every file printed with '%5d  '
line prefixes and its sha256) or from the files themselves (--reader/--compute, e.g.
docker cp'd out of ttbuild). Whatever the source, the base sha must match exactly, every
anchor must occur exactly once, and the result must hash to the recorded output sha, so
the committed kernels cannot drift from what this script produces.

    py -3.11 make_qwen_kernels.py --dump sdpa-decode-sources.txt --out .                  # stage 1
    py -3.11 make_qwen_kernels.py --dump sdpa-decode-sources.txt --stage 3 --out stage3   # stage 3
    py -3.11 make_qwen_kernels.py --dump sdpa-decode-sources.txt --check .                # verify
    python3 make_qwen_kernels.py --stage 3 --reader R.cpp --compute C.cpp --check stage3  # on the rig

The inverse (every edit's new text back to its old text) is also checked by
test_sdpa_decode_qwen_sources.py against the committed files alone, which proves the
committed kernels are the recorded bases plus exactly these edits without needing the
bases in the repository, and that reverting R5 and R4 from the stage-3 reader gives the
stage-1 reader byte for byte.
"""

import argparse
import hashlib
from pathlib import Path
import re
import sys

NL = chr(10)

READER_BASE = '49a05926b437e2ca90d7e01c60e85a6b11f333375a6159fa02c9ff6f78af764e'
COMPUTE_BASE = 'd24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f'
READER_NAME, COMPUTE_NAME = 'reader_decode_qwen.cpp', 'sdpa_flash_decode_qwen.cpp'
DUMP_PREFIX = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/'
DUMP_PATHS = {READER_NAME: DUMP_PREFIX + 'dataflow/reader_decode_all.cpp',
              COMPUTE_NAME: DUMP_PREFIX + 'compute/sdpa_flash_decode.cpp'}
BASES = {READER_NAME: READER_BASE, COMPUTE_NAME: COMPUTE_BASE}
# The recorded results; a changed edit list must update these deliberately.
OUTPUTS = {
    READER_NAME: '55d8fe5e1bc87d9ada56f4a6279afa523865028c163b36803bfd07013f488702',
    COMPUTE_NAME: '8776fcc7420c6f27a9c7ae06c54c391225a00ce78322c5397970d74a5063ca8a',
}
OUTPUTS_STAGE3 = {
    READER_NAME: '280a847fae833891dffff1057d67b999288a183386cd058e3e1614755ce3499b',
    COMPUTE_NAME: OUTPUTS[COMPUTE_NAME],  # stage 3 leaves the compute kernel as it is
}


def lines(*parts):
    return ''.join(part + NL for part in parts)


R1_ANCHOR = lines('    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();')
R1 = R1_ANCHOR + lines(
    '    // [QWEN-SDPA] suffix compile-time args (factory F6), after every TensorAccessorArgs block, so the',
    '    // legacy offsets never move. This file is reader_decode_all.cpp (sha256 49a05926) plus the edits',
    '    // R1-R3 of optimisation/ttnn-op/sdpa_decode_qwen; the factory selects it only in [QWEN-SDPA] mode.',
    '    constexpr uint32_t qwen_cta = attention_sink_args.next_compile_time_args_offset();',
    '    constexpr bool mask_tail = get_compile_time_arg_val(qwen_cta + 0) == 1;',
    '    constexpr uint32_t mask_width_t = get_compile_time_arg_val(qwen_cta + 1);',
    '    constexpr bool kv_share = get_compile_time_arg_val(qwen_cta + 2) == 1;',
    '    constexpr uint32_t kv_ready_semaphore_id = get_compile_time_arg_val(qwen_cta + 3);',
    '    static_assert(!(kv_share && use_k_mcast), "KV share and the MLA K multicast are exclusive");',
    '    static_assert(!(kv_share && reuse_k), "KV share needs an explicit V tensor");',
    '    static_assert(!mask_tail || (Sk_chunk_t > 0 && St % Sk_chunk_t == 0), "Tail mask needs a fixed chunk size");')

R2_OLD = lines(
    '        const uint32_t mask_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bmask) * PNHt * St;',
    '        const uint32_t mask_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic;',
    '        uint32_t mask_start_tile_id = mask_batch_offset + mask_chunk_offset;')
R2_NEW = lines(
    "        // Mask row and batch strides are the mask's own width: St for a full-width mask (== PSt",
    '        // for a non-causal full-window call), Sk_chunk_t for a narrow tail mask. Tail mode reads',
    '        // one fixed chunk: the last Sk_chunk_t columns of the mask.',
    '        const uint32_t mask_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bmask) * PNHt * mask_width_t;',
    '        const uint32_t mask_chunk_offset = mask_tail ? (mask_width_t - Sk_chunk_t) : k_chunk_start * Sk_chunk_t_dynamic;',
    '        uint32_t mask_start_tile_id = mask_batch_offset + mask_chunk_offset;')

R3_OLD = lines(
    '                if constexpr (use_attention_mask) {',
    '                    mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(',
    '                        PSt, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);',
    '                }')
R3_NEW = lines(
    '                if constexpr (use_attention_mask) {',
    "                    // Tail: only the head's final k-chunk (always core_num_in_reduce 0) is masked;",
    '                    // compute applies the mask under the identical predicate (C2).',
    '                    if (!mask_tail || k_chunk == k_num_chunks - 1) {',
    '                        mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(',
    '                            mask_width_t, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);',
    '                    }',
    '                }')

C1_ANCHOR = lines('    constexpr bool has_block_padding = original_block_size > 0 && original_block_size < 32;')
C1 = C1_ANCHOR + lines(
    '    // [QWEN-SDPA] factory F7. This file is sdpa_flash_decode.cpp (sha256 d24769bd) plus the edits',
    '    // C1-C2 of optimisation/ttnn-op/sdpa_decode_qwen; the factory selects it only in [QWEN-SDPA] mode.',
    '    constexpr bool mask_tail = get_compile_time_arg_val(32) == 1;')

C2_OLD = lines(
    '                        if constexpr (use_attention_mask) {',
    '                            reconfig_data_format(cb_qk_im, cb_mask_in);',
    '                            add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);',
    '                        }')
C2_NEW = lines(
    '                        if constexpr (use_attention_mask) {',
    '                            // [QWEN-SDPA] tail: the reader (R3) supplied a mask for the final chunk only.',
    '                            if (!mask_tail || k_chunk == k_num_chunks - 1) {',
    '                                reconfig_data_format(cb_qk_im, cb_mask_in);',
    '                                add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);',
    '                            }',
    '                        }')


EDITS = {
    READER_NAME: (('R1', R1_ANCHOR, R1), ('R2', R2_OLD, R2_NEW), ('R3', R3_OLD, R3_NEW)),
    COMPUTE_NAME: (('C1', C1_ANCHOR, C1), ('C2', C2_OLD, C2_NEW)),
}

# ---------------------------------------------------------------------------------------------
# Stage 3 (spec 7.2): R4 and R5 on top of R1-R3; the compute kernel is unchanged.
# ---------------------------------------------------------------------------------------------

# The stage-1 reader's paged K, mask, V block (original lines 299-344 with R3 applied).
R4_K = lines(
    '                // Read K chunk - supports both multicast and non-multicast paths',
    '                k_base_read_ptr = read_k<',
    '                    cb_k_in,',
    '                    DHt,',
    '                    num_kv_heads,',
    '                    block_size_t,',
    '                    k_tile_bytes,',
    '                    barrier_threshold,',
    '                    is_page_table_sharded,',
    '                    use_k_mcast,',
    '                    capacity_t>(',
    '                    k_chunk_tiles,',
    '                    cur_head,',
    '                    Sk_chunk_t_dynamic,',
    '                    k_chunk_start_row_num,',
    '                    k_reader,',
    '                    page_table_ptr_u16,',
    '                    page_table_ptr_u32,',
    '                    barrier_count,',
    '                    k_mcast_params);',
    '')
R4_V = lines(
    "                // Read V chunk - either from DRAM or from K's L1 buffer (transpose) when reuse_k is true",
    '                read_v<',
    '                    cb_v_in,',
    '                    vDHt,',
    '                    num_kv_heads,',
    '                    block_size_t,',
    '                    v_tile_bytes,',
    '                    barrier_threshold,',
    '                    is_page_table_sharded,',
    '                    reuse_k,',
    '                    capacity_t>(',
    '                    v_chunk_tiles,',
    '                    cur_head,',
    '                    Sk_chunk_t_dynamic,',
    '                    k_chunk_start_row_num,',
    '                    v_reader,',
    '                    page_table_ptr_u16,',
    '                    page_table_ptr_u32,',
    '                    barrier_count,',
    '                    k_base_read_ptr,',
    '                    k_tile_bytes);')
R4_OLD = R4_K + R3_NEW + R4_V


def indent(text, spaces=4):
    return ''.join((' ' * spaces + row if row else row) + NL for row in text.split(NL)[:-1])


R4_NEW = lines(
    '                if constexpr (kv_share) {',
    '                    // [QWEN-SDPA] KV share (factory F10-F12, spec 7.2). The entries of this bundle read the',
    '                    // same page-table row, so entry 0 (the LEADER, do_k_mcast) reads each K and V chunk from',
    "                    // DRAM exactly as the legacy path does and multicasts both CB slots to its B-1 twins, which",
    '                    // sit directly below it in this column and never read K or V themselves. One READY/VALID',
    '                    // round per chunk covers K and V. Every core has the same CB layout (factory: every CB',
    '                    // spans the whole grid) and the same chunk range, so k_slot / v_slot are the same L1',
    '                    // address on every twin.',
    '                    CircularBuffer cb_k(cb_k_in);',
    '                    CircularBuffer cb_v(cb_v_in);',
    '                    Semaphore<> kv_valid(k_mcast_semaphore_id);',
    '                    if (do_k_mcast) {',
    '                        // LEADER: the legacy DRAM reads byte for byte (same functions, no MLA multicast),',
    '                        // then one multicast of both slots to the twins once every twin has reserved them.',
    '                        const uint32_t k_slot = read_k<',
    '                            cb_k_in,',
    '                            DHt,',
    '                            num_kv_heads,',
    '                            block_size_t,',
    '                            k_tile_bytes,',
    '                            barrier_threshold,',
    '                            is_page_table_sharded,',
    '                            false,',
    '                            capacity_t>(',
    '                            k_chunk_tiles,',
    '                            cur_head,',
    '                            Sk_chunk_t_dynamic,',
    '                            k_chunk_start_row_num,',
    '                            k_reader,',
    '                            page_table_ptr_u16,',
    '                            page_table_ptr_u32,',
    '                            barrier_count);',
    '                        const uint32_t v_slot = cb_v.get_write_ptr();  // read_v reserves exactly this slot',
    '                        read_v<',
    '                            cb_v_in,',
    '                            vDHt,',
    '                            num_kv_heads,',
    '                            block_size_t,',
    '                            v_tile_bytes,',
    '                            barrier_threshold,',
    '                            is_page_table_sharded,',
    '                            false,',
    '                            capacity_t>(',
    '                            v_chunk_tiles,',
    '                            cur_head,',
    '                            Sk_chunk_t_dynamic,',
    '                            k_chunk_start_row_num,',
    '                            v_reader,',
    '                            page_table_ptr_u16,',
    '                            page_table_ptr_u32,',
    '                            barrier_count,',
    '                            k_slot,',
    '                            k_tile_bytes);',
    '                        Semaphore<> kv_ready(kv_ready_semaphore_id);',
    '                        kv_ready.wait(num_dests);  // every twin has reserved this chunk\'s K and V slots',
    '                        kv_ready.set(0);',
    '                        noc.async_write_multicast(',
    '                            CoreLocalMem<uint32_t>(k_slot),',
    '                            MulticastEndpoint{},',
    '                            k_chunk_tiles * k_tile_bytes,',
    '                            num_dests,',
    '                            {},',
    '                            {.noc_x_start = mcast_x,',
    '                             .noc_y_start = mcast_y0,',
    '                             .noc_x_end = mcast_x,',
    '                             .noc_y_end = mcast_y1,',
    '                             .addr = k_slot},',
    '                            false);',
    '                        noc.async_write_multicast(',
    '                            CoreLocalMem<uint32_t>(v_slot),',
    '                            MulticastEndpoint{},',
    '                            v_chunk_tiles * v_tile_bytes,',
    '                            num_dests,',
    '                            {},',
    '                            {.noc_x_start = mcast_x,',
    '                             .noc_y_start = mcast_y0,',
    '                             .noc_x_end = mcast_x,',
    '                             .noc_y_end = mcast_y1,',
    '                             .addr = v_slot},',
    '                            false);',
    '                        noc.async_write_barrier();  // the data has landed before the flag',
    '                        kv_valid.set(1);',
    '                        kv_valid.set_multicast(noc, mcast_x, mcast_y0, mcast_x, mcast_y1, num_dests);',
    '                    } else {',
    "                        // TWIN (entry 1..B-1): reserve both slots (its compute has popped chunk n-2), reset",
    "                        // VALID, signal READY to the leader (mcast_x/mcast_y0 = the leader's NoC coordinate),",
    '                        // wait for the bytes, then hand them to compute as if read here.',
    '                        cb_k.reserve_back(k_chunk_tiles);',
    '                        cb_v.reserve_back(v_chunk_tiles);',
    '                        kv_valid.set(0);',
    '                        Semaphore<>(kv_ready_semaphore_id).up(noc, mcast_x, mcast_y0, 1);',
    '                        kv_valid.wait(1);',
    '                        cb_k.push_back(k_chunk_tiles);',
    '                        cb_v.push_back(v_chunk_tiles);',
    '                    }',
    '                    if constexpr (use_attention_mask) {  // each entry still reads its own mask, after K and V',
    '                        if (!mask_tail || k_chunk == k_num_chunks - 1) {',
    '                            mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(',
    '                                mask_width_t, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);',
    '                        }',
    '                    }',
    '                } else {') + indent(R4_OLD) + lines(
    '                }')

R5_ANCHOR = lines(
    '                PSt);',
    '        }',
    '    }',
    '}')
R5 = lines(
    '                PSt);',
    '        }',
    '    }',
    '    if constexpr (kv_share) {',
    '        // [QWEN-SDPA] KV share (spec 7.2 R5): no NoC transaction left in flight and VALID back at 0',
    '        // when the kernel ends (the fused_1d_input lesson: an unbarriered multicast exit hung the',
    "        // stack). The early returns above all come before any semaphore traffic.",
    '        if (do_k_mcast) {',
    '            noc.async_write_barrier();',
    '        } else {',
    '            noc.async_atomic_barrier();',
    '        }',
    '        Semaphore<>(k_mcast_semaphore_id).set(0);',
    '    }',
    '}')

STAGES = (1, 3)
STAGE_EDITS = {
    1: EDITS,
    3: {READER_NAME: EDITS[READER_NAME] + (('R4', R4_OLD, R4_NEW), ('R5', R5_ANCHOR, R5)),
        COMPUTE_NAME: EDITS[COMPUTE_NAME]},
}
STAGE_OUTPUTS = {1: OUTPUTS, 3: OUTPUTS_STAGE3}
# Where each stage's committed kernel set lives, relative to this directory.
STAGE_DIRS = {1: '.', 3: 'stage3'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def from_dump(text, path):
    """One file's exact bytes from the probe dump: the header line names its path, sha and
    line count; the body follows a rule line, each line prefixed '%5d  '."""
    rows = text.split(NL)
    header = re.compile('^' + re.escape(path) + r'  sha256=([0-9a-f]{64})  lines=([0-9]+)$')
    found = [(index, header.match(row)) for index, row in enumerate(rows) if header.match(row)]
    if len(found) != 1:
        raise ValueError('%s appears %d times in the dump' % (path, len(found)))
    index, match = found[0]
    expected, count = match.group(1), int(match.group(2))
    body = []
    for number, row in enumerate(rows[index + 2:index + 2 + count], 1):
        prefix = '%5d  ' % number
        if row.startswith(prefix):
            body.append(row[len(prefix):])
        elif row == prefix.rstrip():
            body.append('')
        else:
            raise ValueError('%s line %d is not in dump form: %r' % (path, number, row[:60]))
    data = NL.join(body).encode('utf-8')
    if sha256(data) != expected:
        raise ValueError('%s does not reconstruct to its recorded sha %s' % (path, expected[:16]))
    return data


def apply_edits(name, base, stage=1):
    if stage not in STAGES:
        raise ValueError('unknown stage %r' % (stage,))
    if sha256(base) != BASES[name]:
        raise ValueError('%s base is %s, expected %s' % (name, sha256(base)[:16], BASES[name][:16]))
    text = base.decode('utf-8')
    for label, old, new in STAGE_EDITS[stage][name]:
        if text.count(old) != 1:
            raise ValueError('%s anchor %s occurs %d times' % (name, label, text.count(old)))
        text = text.replace(old, new)
    return text.encode('utf-8')


def revert_edits(name, built, stage=1, *, to_stage=0):
    """The inverse: every edit's new text back to its old text, last edit first. to_stage=1
    stops once the stage-3 edits are undone (the stage-3 kernel back to the stage-1 one)."""
    text = built.decode('utf-8')
    edits = STAGE_EDITS[stage][name]
    keep = len(STAGE_EDITS[to_stage][name]) if to_stage else 0
    for label, old, new in reversed(edits[keep:]):
        if text.count(new) != 1:
            raise ValueError('%s edit %s occurs %d times' % (name, label, text.count(new)))
        text = text.replace(new, old)
    return text.encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dump', help='probe_sdpa_decode_sources.py output')
    source.add_argument('--reader', help='reader_decode_all.cpp (then --compute too)')
    parser.add_argument('--compute', help='sdpa_flash_decode.cpp')
    parser.add_argument('--stage', type=int, choices=STAGES, default=1,
                        help='1: R1-R3/C1-C2 (K64e, the default); 3: plus R4-R5 (K64f)')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--out', help='directory to write the two qwen kernels into')
    target.add_argument('--check', help='directory whose qwen kernels must equal the build')
    parser.add_argument('--record', action='store_true', help='print the output shas instead of enforcing them')
    args = parser.parse_args(argv)
    if args.dump:
        text = Path(args.dump).read_bytes().decode('utf-8')
        bases = {name: from_dump(text, path) for name, path in DUMP_PATHS.items()}
    else:
        if not args.compute:
            parser.error('--reader needs --compute')
        bases = {READER_NAME: Path(args.reader).read_bytes(), COMPUTE_NAME: Path(args.compute).read_bytes()}
    outputs = STAGE_OUTPUTS[args.stage]
    status = 0
    for name, base in bases.items():
        built = apply_edits(name, base, args.stage)
        digest = sha256(built)
        if not args.record and digest != outputs[name]:
            print('%s built to %s, recorded %s' % (name, digest, outputs[name]))
            return 1
        if revert_edits(name, built, args.stage) != base:
            print('%s edits do not invert' % name)
            return 1
        if args.stage == 3 and sha256(revert_edits(name, built, 3, to_stage=1)) != OUTPUTS[name]:
            print('%s stage-3 edits do not invert to the stage-1 kernel' % name)
            return 1
        if args.out:
            target = Path(args.out) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(built)
            print('%s stage %d base %s -> %s written %s' % (name, args.stage, BASES[name][:16], digest, target))
        else:
            target = Path(args.check) / name
            present = target.read_bytes() if target.is_file() else None
            same = present == built
            print('%s stage %d base %s -> %s %s' % (name, args.stage, BASES[name][:16], digest,
                                                     'matches' if same else 'DIFFERS at ' + str(target)))
            status |= 0 if same else 1
    return status


if __name__ == '__main__':
    sys.exit(main())
