"""Make reader_interleaved_qwen_ksplit.cpp (edits K1-K7) from the committed chain reader (eecc1166...).

Prefill lever PF3-K (flag 0x10 of apply_factory_pk.py). The factory (PK11) selects this file only for SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0000 | flags with flag 0x10.
It is the chain reader (the G6 K/V chain, one round per K chunk) with ONE change of meaning: a core streams only its SLICE of its q chunk's K blocks, and a core that is not the first
slice receives the previous slice's partial result (the normalised output and the log-sum-exp) into the two CBs the pinned compute kernel's ring merge reads.

    K1  after the qwen_pf helpers: namespace qwen_ks, the slice arithmetic (ks_common.RANGE_BLOCK, byte for byte the writer's and the compute kernel's)
    K2  the suffix compile-time args (factory PK6: cb_arg_offset + 9 ..) and the test-hang flag 0x400
    K3  the runtime words after the global-Q pair: the slice and the previous slice's core (factory PK12/PK13)
    K4  per unit: B, the effective slices E, whether this slice takes part, its K range; a receiving slice reserves cb_prev_out / cb_lse_in, credits the previous slice's core
        (an atomic barrier right after: no atomic is in flight when the first read barrier of the unit runs)
    K5  the k loop runs [begin, end) of the slice; the last round is the last of the slice
    K6  after the loop: a receiving slice waits for the previous slice's VALID flag (the partial landed: acked write, then flag), pushes the two CBs and resets the flag; a slice with an
        even number of K chunks pushes the one dummy K/V pair the pinned ring loop pops at its end (compute_common.hpp: KV_chunks_processed_in_iter % 2 == 0)

An idle slice (s >= E) runs an empty loop and does nothing else; every member of a chain has the same q chunk and slice, so a chain is idle or active as a whole.
The chain protocol (credit, VALID, forward) is untouched.

Rules (as make_pf_reader.py): the input must be the committed chain reader; every anchor must occur exactly once and start at the recorded line; the output must hash to READER_OUTPUT
(--record prints it); reverting the edits must give the input back; LF only.

    python3 make_ks_reader.py                  # write reader_interleaved_qwen_ksplit.cpp beside this file
    python3 make_ks_reader.py --check .        # compare with the committed file, write nothing
    python3 make_ks_reader.py --dump           # print the generated reader
    python3 make_ks_reader.py --record         # print the output sha
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
CHAIN_READER = HERE.parent / 'sdpa_prefill_chain' / 'reader_interleaved_qwen_chain.cpp'
OUTPUT_NAME = 'reader_interleaved_qwen_ksplit.cpp'
BASE_SHA = 'eecc1166a209e61dc8498149b5a4338cc278d7106f4ca68d6434f620e942f8d8'
READER_OUTPUT = '16438818a762c699d929c579903925eba175b6f0e61171efa01dc0113a8dc128'

# ---- K1: namespace qwen_ks (before the first helper that follows the qwen_pf namespace) -----------------------------------------------------------
K1_OLD = lines('// Fetch a KV chunk into L1 for forwarding. No CB lifecycle \u2014 caller manages')
K1_NEW = ks_common.RANGE_BLOCK + K1_OLD

# ---- K2: the suffix compile-time args -----------------------------------------------------------------------------------------------------------
K2_OLD = lines('    static_assert(!pf_test_mutate || NKH >= 2, "[QWEN-SDPA-PF] the mutation flips the KV head");')
K2_NEW = K2_OLD + lines(
    '    // [QWEN-SDPA-KS] K2: the suffix of the ksplit factory (PK6): slices, the CBs the previous slice\'s partial lands in, the hand-off semaphores; 0x400 = the planted hang',
    '    constexpr uint32_t ks_slices = get_compile_time_arg_val(cb_arg_offset + 9);',
    '    constexpr uint32_t ks_cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 10);',
    '    constexpr uint32_t ks_cb_lse_in = get_compile_time_arg_val(cb_arg_offset + 11);',
    '    constexpr uint32_t ks_credit_sem = get_compile_time_arg_val(cb_arg_offset + 12);',
    '    constexpr uint32_t ks_valid_sem = get_compile_time_arg_val(cb_arg_offset + 13);',
    '    constexpr bool ks_test_hang = (qwen_pf_flags & 0x400u) != 0;',
    '    constexpr uint32_t ks_out_tiles = Sq_chunk_t * vDHt;',
    '    static_assert((qwen_pf_flags & 0x19u) == 0x19u, "[QWEN-SDPA-KS] the ksplit reader needs the flags 0x1, 0x8 and 0x10");')

# ---- K3: the runtime words ---------------------------------------------------------------------------------------------------------------------
K3_OLD = lines('    global_q_count = get_arg_val<uint32_t>(argidx++);')
K3_NEW = K3_OLD + lines(
    '    // [QWEN-SDPA-KS] K3: the slice of this core and the previous slice\'s core (the hand-off credit goes there); factory PK13',
    '    const uint32_t ks_slice = get_arg_val<uint32_t>(argidx++);',
    '    const uint32_t ks_prev_x = get_arg_val<uint32_t>(argidx++);',
    '    const uint32_t ks_prev_y = get_arg_val<uint32_t>(argidx++);')

# ---- K4: per unit: the range, and the receiver's reservation and credit --------------------------------------------------------------------------
K4_OLD = lines('            const uint32_t k_head = nq / q_heads_per_k;')
K4_NEW = lines(
    '            // [QWEN-SDPA-KS] K4: B K blocks in this q chunk, E slices take part, this slice streams [k_loop_start, k_loop_end) (empty when it is idle)',
    '            const uint32_t ks_blocks = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;',
    '            const uint32_t ks_eff = qwen_ks::effective_slices(ks_blocks, ks_slices);',
    '            const bool ks_active = ks_slice < ks_eff;',
    '            const bool ks_receiver = ks_active && ks_slice > 0;',
    '            k_loop_start = qwen_ks::slice_begin(ks_blocks, ks_slice, ks_eff);',
    '            const uint32_t k_loop_end = ks_active ? qwen_ks::slice_begin(ks_blocks, ks_slice + 1, ks_eff) : k_loop_start;',
    '            CircularBuffer ks_cb_prev(ks_cb_prev_out);',
    '            CircularBuffer ks_cb_lse(ks_cb_lse_in);',
    '            if (ks_receiver) {',
    '                ks_cb_prev.reserve_back(ks_out_tiles);  // room for the previous slice\'s output and lse, then say so (the planted hang withholds only the credit)',
    '                ks_cb_lse.reserve_back(Sq_chunk_t);',
    '                if (!ks_test_hang) {',
    '                    Semaphore<>(ks_credit_sem).up(noc, ks_prev_x, ks_prev_y, 1);',
    '                }',
    '                noc.async_atomic_barrier();  // no atomic in flight when the unit\'s first read barrier runs (the BH read-barrier hazard of R7)',
    '            }',
    '') + K4_OLD

# ---- K5: the loop runs the slice ----------------------------------------------------------------------------------------------------------------
K5A_OLD = lines('            for (uint32_t k_chunk = k_loop_start; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {')
K5A_NEW = lines('            for (uint32_t k_chunk = k_loop_start; k_chunk < k_loop_end; ++k_chunk) {  // [QWEN-SDPA-KS] K5: the slice, not the whole causal range')
K5B_OLD = lines('                    (global_q_iter + 1 == global_q_count) && ((k_chunk + 1) * Sk_chunk_t >= q_high_idx);')
K5B_NEW = lines('                    (global_q_iter + 1 == global_q_count) && (k_chunk + 1 >= k_loop_end);')

# ---- K6: after the loop ---------------------------------------------------------------------------------------------------------------------------
K6_OLD = lines('            }  // close k_chunk')
K6_NEW = K6_OLD + lines(
    '            // [QWEN-SDPA-KS] K6: the previous slice\'s partial has landed (acked write, then VALID); hand it to the compute kernel\'s ring merge, then reset the flag',
    '            if (ks_receiver) {',
    '                qwen_pf::wd_wait_valid(ks_valid_sem);',
    '                ks_cb_prev.push_back(ks_out_tiles);',
    '                ks_cb_lse.push_back(Sq_chunk_t);',
    '                Semaphore<>(ks_valid_sem).set(INVALID);',
    '            }',
    '            if (ks_active && ((k_loop_end - k_loop_start) % 2) == 0) {  // the pinned ring loop pops one K/V pair at its end after an even number of chunks',
    '                cb_k.reserve_back(k_chunk_tiles);',
    '                cb_k.push_back(k_chunk_tiles);',
    '                cb_v.reserve_back(v_chunk_tiles);',
    '                cb_v.push_back(v_chunk_tiles);',
    '            }')

# (label, chain-reader line the anchor starts on, old text, new text), in file order.
EDITS = (
    ('K1', 90, K1_OLD, K1_NEW),
    ('K3', 259, K3_OLD, K3_NEW),
    ('K2', 284, K2_OLD, K2_NEW),
    ('K4', 484, K4_OLD, K4_NEW),
    ('K5A', 493, K5A_OLD, K5A_NEW),
    ('K5B', 504, K5B_OLD, K5B_NEW),
    ('K6', 596, K6_OLD, K6_NEW),
)


class Refusal(ValueError):
    """The input is not the chain reader, or an anchor drifted."""


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
    """The ksplit reader bytes from the chain reader bytes."""
    if sha(source) != BASE_SHA:
        raise Refusal('unexpected reader %s (need the chain reader %s)' % (sha(source), BASE_SHA))
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
    parser.add_argument('--check', metavar='DIR', help='compare DIR/%s with the generated reader and write nothing' % OUTPUT_NAME)
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--reader', default=str(CHAIN_READER), help='the chain reader (default: the committed one)')
    args = parser.parse_args(argv)
    try:
        generated = patch(Path(args.reader).read_bytes())
    except Refusal as error:
        print('refused: %s' % error)
        return 1
    if unpatch(generated) != Path(args.reader).read_bytes():
        print('edits do not invert')
        return 1
    digest = sha(generated)
    if args.dump:
        sys.stdout.write(generated.decode('utf-8'))
        return 0
    if args.record:
        print(digest)
        return 0
    if READER_OUTPUT != '__RECORD__' and digest != READER_OUTPUT:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, READER_OUTPUT))
        return 1
    if args.check:
        committed = (Path(args.check) / OUTPUT_NAME).read_bytes()
        if committed != generated:
            print('%s differs from the generated reader' % (Path(args.check) / OUTPUT_NAME))
            return 1
        print('ok %s %s' % (OUTPUT_NAME, digest))
        return 0
    (HERE / OUTPUT_NAME).write_bytes(generated)
    print('wrote %s %s' % (HERE / OUTPUT_NAME, digest))
    return 0


if __name__ == '__main__':
    sys.exit(main())
