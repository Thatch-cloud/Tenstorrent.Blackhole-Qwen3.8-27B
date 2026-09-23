"""Build the two [QWEN-SDPA] stage-1 kernels from the served originals, byte for byte.

reader_decode_qwen.cpp    = reader_decode_all.cpp  (sha256 49a05926...) + R1, R2, R3
sdpa_flash_decode_qwen.cpp = sdpa_flash_decode.cpp (sha256 d24769bd...) + C1, C2

(spec: sdpa-onepass-spec.md sections 4.2 and 4.3). The originals are taken either from
the probe dump (probe_sdpa_decode_sources.py output: every file printed with '%5d  '
line prefixes and its sha256) or from the files themselves (--reader/--compute, e.g.
docker cp'd out of ttbuild). Whatever the source, the base sha must match exactly, every
anchor must occur exactly once, and the result must hash to the recorded output sha, so
the committed kernels cannot drift from what this script produces.

    py -3.11 make_qwen_kernels.py --dump sdpa-decode-sources.txt --out .      # regenerate
    py -3.11 make_qwen_kernels.py --dump sdpa-decode-sources.txt --check .    # verify
    python3 make_qwen_kernels.py --reader R.cpp --compute C.cpp --check .     # on the rig

The inverse (every edit's new text back to its old text) is also checked by
test_sdpa_decode_qwen_sources.py against the committed files alone, which proves the
committed kernels are the recorded bases plus exactly these edits without needing the
bases in the repository.
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


def apply_edits(name, base):
    if sha256(base) != BASES[name]:
        raise ValueError('%s base is %s, expected %s' % (name, sha256(base)[:16], BASES[name][:16]))
    text = base.decode('utf-8')
    for label, old, new in EDITS[name]:
        if text.count(old) != 1:
            raise ValueError('%s anchor %s occurs %d times' % (name, label, text.count(old)))
        text = text.replace(old, new)
    return text.encode('utf-8')


def revert_edits(name, built):
    """The inverse: every edit's new text back to its old text, last edit first."""
    text = built.decode('utf-8')
    for label, old, new in reversed(EDITS[name]):
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
    status = 0
    for name, base in bases.items():
        built = apply_edits(name, base)
        digest = sha256(built)
        if not args.record and digest != OUTPUTS[name]:
            print('%s built to %s, recorded %s' % (name, digest, OUTPUTS[name]))
            return 1
        if revert_edits(name, built) != base:
            print('%s edits do not invert' % name)
            return 1
        if args.out:
            target = Path(args.out) / name
            target.write_bytes(built)
            print('%s base %s -> %s written %s' % (name, BASES[name][:16], digest, target))
        else:
            target = Path(args.check) / name
            present = target.read_bytes() if target.is_file() else None
            same = present == built
            print('%s base %s -> %s %s' % (name, BASES[name][:16], digest, 'matches' if same else 'DIFFERS at ' + str(target)))
            status |= 0 if same else 1
    return status


if __name__ == '__main__':
    sys.exit(main())
