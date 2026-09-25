"""Make the four K0 probe readers (sdpa-prefill-share-spec.md section 7.1) from the served reader.

K0 is the no-build first step of prefill lever #1: one card-M session in which a variant of the
served causal chunked SDPA reader is bind-mounted over the image's
    /opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp
(run_m1.sh M1_READER=...), and the served factory JIT-compiles it. No C++ build, no factory change.

    k0a    K/V are read from DRAM only for k_chunk 0 and 1; every later k chunk reserves and pushes
           the stale slot (real, finite bf8 from an earlier read). Output wrong by design.
           t_c = slope x 0.064 ms is the compute floor per step.
    k0b4   as k0a, but the 16 would-be G6 injectors, (core_id / 8) % 6 == 0, read every chunk at
           the served barrier_threshold (4). The G6 DRAM ceiling without forwarding.
    k0b32  as k0b4 with a K/V barrier every 32 tiles on those 16 cores (flag 0x2's cadence).
    k0c    every core reads every chunk, K/V barrier every 32 tiles (Q keeps 4). Output MUST equal
           stock byte for byte (the batching premise behind flag 0x2).

Facts from the served source (f97f5490):
  - core_id is reader runtime arg 7 (R:115); the factory pushes the linear core index i there
    (sdpa_program_factory.cpp fd8c0676, `reader_args.push_back(i)`, core = {i % 11, i / 11}). Core i
    < 96 works on Q head i / 8, so (core_id / 8) % 6 == 0 is i in 0-7 and 48-55: one core per
    (KV head, m) group. Cores 96-103 also satisfy it but own no Q chunks (global_q_count 0).
  - barrier_threshold is the compile-time constant at R:209,
    get_barrier_read_threshold<q_tile_bytes, num_cores>() = ((512 / 110) * 1152) / 1088 = 4 for bf8 Q.
    It is passed to read_paged_chunk_with_padding as a runtime parameter (DC:265), which reserves
    dst_rows * dst_cols tiles (DC:273), reads with a barrier every `barrier_threshold` tiles, barriers
    and pushes the same count (DC:314).

Hang safety: every variant keeps the served CB protocol per k chunk. The served K read is
`read_paged_chunk_with_padding(..., dst_rows=Sk_chunk_t, dst_cols=DHt, ...)`, i.e. reserve then push
Sk_chunk_t * DHt tiles on cb_k_in; V is the same with vDHt on cb_v_in. A skipped chunk runs
cb_k.reserve_back(Sk_chunk_t * DHt); cb_k.push_back(Sk_chunk_t * DHt) (cb_k is cb_k_in, R:233), so
only the NoC reads and their barriers are skipped. Q, the page table, chunk_start and the loop bounds
are untouched, and no variant waits on anything the served reader does not wait on. Only the paged
(is_chunked) branch is edited; the non-paged branches keep the served code.

Rules (as make_qwen_kernels.py): the input sha must be the served f97f5490 file; every anchor must
occur exactly once and start at the served line the spec cites; each output must hash to the value
recorded below; reverting the edits must give the input back (checked here and, from the committed
outputs alone, by test_k0_readers.py). Files are written with LF endings.

    py -3.11 make_k0_readers.py                         # read the probe-v25 reader, write k0/
    py -3.11 make_k0_readers.py --reader R.cpp --check k0
    python3 make_k0_readers.py --reader /path/reader_interleaved.cpp --out /tmp/k0   # on the rig
"""

import argparse
import hashlib
import sys
from pathlib import Path

NL = chr(10)
HERE = Path(__file__).resolve().parent
DEFAULT_READER = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp/probe-v25/src/device/kernels/dataflow/'
                      'reader_interleaved.cpp')
DEFAULT_OUT = HERE / 'k0'
IMAGE_READER = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp'

BASE_SHA = 'f97f5490cf476db92d575de33c85f8707f96d7474896ee086ee864c23a3efa27'
VARIANTS = ('k0a', 'k0b4', 'k0b32', 'k0c')
OUTPUT_NAMES = {variant: 'reader_%s.cpp' % variant for variant in VARIANTS}
# The recorded results; a changed edit list must update these deliberately (--record prints them).
OUTPUTS = {
    'k0a': 'fbbd325213d0f3d77c4ea700679112bb959d7f4182cacab815a848dbbb78df38',
    'k0b4': '0fe631aaae0a9e894c865a1623cde031b7828ee0eef1f531ea0d648a0d25fd06',
    'k0b32': 'c66b03c6c9fc8739443addf81058579c0a3a04b47b1874ffbc760d8645681504',
    'k0c': '817bf36b9a28fffe603c012c6a32cca7cd5e709469173ded98ac1cc895f14747',
}

KV_BT_BATCH = 32          # K/V barrier cadence of k0b32 (injectors) and k0c (everyone): one per chunk
INJECTOR_PREDICATE = '((core_id / 8) % 6) == 0'
READ_PREDICATE = '(qwen_k0_full_kv || k_chunk < 2)'


def lines(*parts):
    return ''.join(part + NL for part in parts)


def sha(data):
    return hashlib.sha256(data).hexdigest()


# --- anchors: the served text each edit replaces, and the 1-based served line it starts on ----------

HEADER_ANCHOR = lines('#include "dataflow_common.hpp"')                                        # R:13
HEADER_LINE = 13
BT_ANCHOR = lines('    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();')
BT_LINE = 209                                                                                  # R:209
K_CALL = lines(                                                                                # R:426-439
    '                        read_paged_chunk_with_padding<NKH, block_size_t, DHt>(',
    '                            k_reader,',
    '                            cb_k_in,',
    '                            k_head,',
    '                            k_chunk_start_row_num,',
    '                            kv_row_tile_count,',
    '                            DHt,',
    '                            Sk_chunk_t,',
    '                            DHt,',
    '                            k_tile_bytes,',
    '                            barrier_threshold,',
    '                            page_table_ptr,',
    '                            true  // transpose=true for K reads',
    '                        );')
K_LINE = 426
V_CALL = lines(                                                                                # R:619-632
    '                        read_paged_chunk_with_padding<NVH, block_size_t, head_dim>(',
    '                            v_reader,',
    '                            cb_v_in,',
    '                            v_head,',
    '                            kv_chunk_start_row_num,',
    '                            kv_row_tile_count,',
    '                            vDHt,',
    '                            Sk_chunk_t,',
    '                            vDHt,',
    '                            v_tile_bytes,',
    '                            barrier_threshold,',
    '                            page_table_ptr,',
    '                            false,',
    '                            skip_src_cols);')
V_LINE = 619
BT_ARG = '                            barrier_threshold,' + NL
BT_ARG_NEW = '                            qwen_k0_kv_bt,' + NL

DESCRIPTIONS = {
    'k0a': ('K/V read from DRAM only for k_chunk 0 and 1; later chunks reserve and push the stale',
            'slot. Timing probe (t_c = slope x 0.064 ms); the output is wrong by design.'),
    'k0b4': ('as k0a, but the 16 would-be G6 injectors, (core_id / 8) % 6 == 0, read every K/V chunk',
             'at the served barrier_threshold (4). G6 DRAM ceiling without forwarding; output wrong.'),
    'k0b32': ('as k0b4 with a K/V read barrier every 32 tiles on those 16 cores (flag 0x2 cadence).',
              'G6 DRAM ceiling at bt 32; output wrong by design.'),
    'k0c': ('every core reads every K/V chunk (served), with a K/V read barrier every 32 tiles instead',
            'of barrier_threshold (4); Q keeps 4. Output must equal stock byte for byte.'),
}


def header(variant):
    first, second = DESCRIPTIONS[variant]
    return lines(
        '// [QWEN-SDPA-K0] %s: K0 probe reader, NOT for serving (sdpa-prefill-share-spec.md 7.1).' % variant,
        '// reader_interleaved.cpp (sha256 %s...) + the K0 edits of' % BASE_SHA[:8],
        '// optimisation/ttnn-op/sdpa_prefill_bench/make_k0_readers.py; mounted over the image reader.',
        '// %s' % first,
        '// %s' % second)


def constants(variant):
    if variant == 'k0a':
        return lines(
            '    // [QWEN-SDPA-K0] k0a: no core reads K/V past k_chunk 1; the reads keep the served cadence.',
            '    const bool qwen_k0_full_kv = false;',
            '    const uint32_t qwen_k0_kv_bt = barrier_threshold;')
    if variant == 'k0b4':
        return lines(
            '    // [QWEN-SDPA-K0] k0b4: core_id is RT arg 7, the linear core index i. (i / 8) % 6 == 0 are the',
            '    // 16 would-be G6 injectors (one per KV head and m); they read every K/V chunk, the rest only',
            '    // k_chunk 0 and 1. Served barrier_threshold everywhere.',
            '    const bool qwen_k0_full_kv = %s;' % INJECTOR_PREDICATE,
            '    const uint32_t qwen_k0_kv_bt = barrier_threshold;')
    if variant == 'k0b32':
        return lines(
            '    // [QWEN-SDPA-K0] k0b32: core_id is RT arg 7, the linear core index i. (i / 8) % 6 == 0 are the',
            '    // 16 would-be G6 injectors; they read every K/V chunk with a barrier every %d tiles, the rest' % KV_BT_BATCH,
            '    // read only k_chunk 0 and 1 at the served barrier_threshold.',
            '    const bool qwen_k0_full_kv = %s;' % INJECTOR_PREDICATE,
            '    const uint32_t qwen_k0_kv_bt = qwen_k0_full_kv ? %du : barrier_threshold;' % KV_BT_BATCH)
    if variant == 'k0c':
        return lines(
            '    // [QWEN-SDPA-K0] k0c: the K/V read barrier cadence only (Q, mask and everything else keep',
            '    // barrier_threshold). Same addresses, bytes and placement: the output must equal stock.',
            '    constexpr uint32_t qwen_k0_kv_bt = %d;' % KV_BT_BATCH)
    raise ValueError('unknown variant %r' % variant)


def indent(text, pad='    '):
    return ''.join((pad + line if line else line) + NL for line in text.split(NL)[:-1])


def skip_branch(call, cb_object, dst_cols):
    """The if/else around one served K or V read: the served call (bt swapped) when the chunk is
    read, else exactly its CB lifecycle (reserve then push dst_rows * dst_cols tiles), no NoC."""
    count = 'Sk_chunk_t * %s' % dst_cols
    return (lines('                        if %s {   // [QWEN-SDPA-K0] DRAM read: the served call' % READ_PREDICATE)
            + indent(call.replace(BT_ARG, BT_ARG_NEW))
            + lines('                        } else {',
                    '                            // [QWEN-SDPA-K0] no DRAM read: the served call\'s own CB lifecycle only,',
                    '                            // reserve then push dst_rows * dst_cols = %s tiles over the stale slot.' % count,
                    '                            %s.reserve_back(%s);' % (cb_object, count),
                    '                            %s.push_back(%s);' % (cb_object, count),
                    '                        }'))


def edits(variant):
    """(name, served line, old text, new text), in file order."""
    if variant not in VARIANTS:
        raise ValueError('unknown variant %r' % variant)
    out = [('E0 header', HEADER_LINE, HEADER_ANCHOR, HEADER_ANCHOR + header(variant)),
           ('E1 constants', BT_LINE, BT_ANCHOR, BT_ANCHOR + constants(variant))]
    if variant == 'k0c':
        out.append(('E2 K read', K_LINE, K_CALL, K_CALL.replace(BT_ARG, BT_ARG_NEW)))
        out.append(('E3 V read', V_LINE, V_CALL, V_CALL.replace(BT_ARG, BT_ARG_NEW)))
    else:
        out.append(('E2 K read', K_LINE, K_CALL, skip_branch(K_CALL, 'cb_k', 'DHt')))
        out.append(('E3 V read', V_LINE, V_CALL, skip_branch(V_CALL, 'cb_v', 'vDHt')))
    return out


class Refusal(Exception):
    """The input is not the served reader, or an anchor drifted: nothing is written."""


def check_base(data):
    digest = sha(data)
    if digest != BASE_SHA:
        raise Refusal('input sha256 %s is not the served reader f97f5490 (%s); refusing' % (digest, BASE_SHA))
    return digest


def check_anchors(text, variant):
    """Every anchor exactly once in the served text, at the served line the spec cites."""
    for name, line, old, _ in edits(variant):
        if old.count(BT_ARG) > 1:
            raise Refusal('%s: the barrier argument occurs more than once in its anchor' % name)
        count = text.count(old)
        if count != 1:
            raise Refusal('%s: anchor occurs %d times (need exactly 1); refusing on drift' % (name, count))
        found = text[:text.index(old)].count(NL) + 1
        if found != line:
            raise Refusal('%s: anchor starts at line %d, the spec cites R:%d; refusing on drift' % (name, found, line))


def apply_edits(text, variant):
    check_anchors(text, variant)
    for _, _, old, new in edits(variant):
        text = text.replace(old, new)
    return text


def revert_edits(text, variant):
    """Every edit's new text back to its old text, last edit first: must give the served reader."""
    for name, _, old, new in reversed(edits(variant)):
        count = text.count(new)
        if count != 1:
            raise Refusal('%s: edited text occurs %d times in the variant (need exactly 1)' % (name, count))
        text = text.replace(new, old)
    return text


def build(data, variant):
    """Served reader bytes -> the variant's bytes (LF, UTF-8). Refuses a wrong base or drift."""
    check_base(data)
    text = data.decode('utf-8')
    out = apply_edits(text, variant).encode('utf-8')
    if revert_edits(out.decode('utf-8'), variant).encode('utf-8') != data:
        raise Refusal('%s: reverting the edits does not give the served reader back' % variant)
    if chr(13).encode() in out:
        raise Refusal('%s: output has a CR; files must be LF' % variant)
    return out


def build_all(data, record=False):
    outputs = {variant: build(data, variant) for variant in VARIANTS}
    if not record:
        for variant, out in outputs.items():
            if sha(out) != OUTPUTS[variant]:
                raise Refusal('%s: output sha256 %s is not the recorded %s (edit list changed? rerun with '
                              '--record and update OUTPUTS deliberately)' % (variant, sha(out), OUTPUTS[variant]))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('--reader', type=Path, default=DEFAULT_READER,
                        help='the served reader_interleaved.cpp (sha256 f97f5490...)')
    parser.add_argument('--out', type=Path, help='write reader_k0*.cpp here (default: k0/ next to this script)')
    parser.add_argument('--check', type=Path, help='compare against reader_k0*.cpp in this directory; write nothing')
    parser.add_argument('--record', action='store_true', help='skip the recorded-output check and print the shas')
    options = parser.parse_args(argv)
    try:
        data = options.reader.read_bytes()
    except OSError as error:
        print('REFUSING: cannot read the served reader %s (%s)' % (options.reader, error), file=sys.stderr)
        return 2
    print('input %s sha256=%s' % (options.reader, sha(data)))
    try:
        outputs = build_all(data, record=options.record)
    except Refusal as error:
        print('REFUSING: %s' % error, file=sys.stderr)
        return 2
    status = 0
    for variant in VARIANTS:
        out = outputs[variant]
        name = OUTPUT_NAMES[variant]
        line = '%-16s sha256=%s' % (name, sha(out))
        if options.check is not None:
            path = options.check / name
            ok = path.is_file() and path.read_bytes() == out
            line += '  %s' % ('matches ' + str(path) if ok else 'DIFFERS from ' + str(path))
            status = status or (0 if ok else 1)
        print(line)
    if options.record:
        print('OUTPUTS = {')
        for variant in VARIANTS:
            print("    '%s': '%s'," % (variant, sha(outputs[variant])))
        print('}')
    if options.check is None:
        out_dir = options.out or DEFAULT_OUT
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            (out_dir / OUTPUT_NAMES[variant]).write_bytes(outputs[variant])
        print('wrote %d readers to %s (mount one with M1_READER=<file> over %s)' % (len(VARIANTS), out_dir,
                                                                                  IMAGE_READER))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
