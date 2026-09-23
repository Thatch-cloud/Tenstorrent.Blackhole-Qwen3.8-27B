"""Make reader_interleaved_qwen_chain.cpp (edits R1-R9) from the served prefill SDPA reader.

Prefill lever #1 (sdpa-prefill-share-spec.md section 3.3): the G6 unicast K/V chain reader. The
factory (apply_factory_pf.py F6) selects this file only for max_cores_per_head_batch =
0x5EFA0000 | flags with flag 0x1; the served reader_interleaved.cpp is NOT modified.

    R1  after `#include "dataflow_common.hpp"`: namespace qwen_pf, the bounded waits (never fall
        through: waypoint + watcher assert, then spin forever) and the credit / forward helpers
        (credit_prev always resets the flag; its withhold argument, the 0x200 planted hang, skips
        only the credit, so the hung sink waits in QWDV instead of pushing a stale slot)
    R2  after the zigzag CT line: kQwenKvChain and the static envelope
    R3  the chain runtime block is parsed in chain mode (`!is_causal || kQwenKvChain`)
    R4  after cb_id_chunk_start_idx_writer: the flags CT arg (cb_arg_offset + 8)
    R5  after barrier_threshold: kv_bt, the injector's K/V barrier cadence under flag 0x2, and a
        static_assert that Q is read by subblocks (the read R7 puts behind the atomic barrier)
    R6  the should_forward / should_receive block: every member shares every round
    R7  the K/V round body (R:406-709): one credit/VALID round per k_chunk covering K and V
    R8  before the end of kernel_main: barriers, then the semaphores back to their initial values
    R9  [[maybe_unused]] on every declaration R6/R7 orphan (the spec names mask_batch_offset and
        q_iter; R7 also removes the only reads of the mask locals, the tile shapes and four chain
        words, listed in R9_ORPHANS)

Rules (as make_k0_readers.py / make_qwen_kernels.py): the input must be the served f97f5490
file; every anchor must occur exactly once and start at the recorded served line; the output must
hash to READER_OUTPUT (--record prints it); reverting the edits must give the input back; LF only.

    py -3.11 make_pf_reader.py                                   # probe-v25 reader -> reader_interleaved_qwen_chain.cpp here
    py -3.11 make_pf_reader.py --check .                         # compare with the committed file, write nothing
    python3 make_pf_reader.py --reader /tmp/reader_interleaved.cpp --check DIR   # on the rig (build_k64g.sh)
    py -3.11 make_pf_reader.py --dump                            # print the generated reader
"""

import argparse
import hashlib
import sys
from pathlib import Path

NL = chr(10)
HERE = Path(__file__).resolve().parent
DEFAULT_READER = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp/probe-v25/src/device/kernels/dataflow/'
                      'reader_interleaved.cpp')
OUTPUT_NAME = 'reader_interleaved_qwen_chain.cpp'

BASE_SHA = 'f97f5490cf476db92d575de33c85f8707f96d7474896ee086ee864c23a3efa27'
READER_OUTPUT = 'eecc1166a209e61dc8498149b5a4338cc278d7106f4ca68d6434f620e942f8d8'

WATCHDOG_POLLS_LOG2 = 28     # kWdPolls = 1 << 28 (spec R1: about 1-3 s, far above any legal wait)
WAYPOINTS = ('QWDC', 'QWDV')  # credit wait, VALID wait


def lines(*parts):
    return ''.join(part + NL for part in parts)


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ---- R1: helpers (after R:13) ------------------------------------------------------------------
R1_ANCHOR = lines('#include "dataflow_common.hpp"')
R1 = R1_ANCHOR + lines(
    '',
    '// [QWEN-SDPA-PF] reader_interleaved.cpp (sha256 f97f5490...) + R1-R9 of optimisation/ttnn-op/sdpa_prefill_chain',
    '// (make_pf_reader.py; sdpa-prefill-share-spec.md 3.3). Selected by the factory only for',
    '// SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0000 | flags with flag 0x1: the G6 unicast K/V chain.',
    'namespace qwen_pf {',
    '// Bounded equality waits: the semantics of Semaphore<>::wait (spin with an L1 invalidate) plus a poll',
    '// bound. On expiry they NEVER fall through: waypoint + watcher assert, then spin forever. A hang stays a',
    '// hang, so a stale slot is never pushed. The bound (2^28 polls, about 1-3 s) is orders of magnitude above',
    '// any legal wait. Two functions so the waypoint names the wait: QWDC credit, QWDV VALID.',
    'constexpr uint32_t kWdPolls = 1u << %d;' % WATCHDOG_POLLS_LOG2,
    'FORCE_INLINE void wd_wait_credit(uint32_t sem_id) {',
    '    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));',
    '    for (uint32_t n = 0;; ++n) {',
    '        invalidate_l1_cache();',
    '        if (*p == 1) {',
    '            return;',
    '        }',
    '        if (n == kWdPolls) {',
    '            WAYPOINT("%s");' % WAYPOINTS[0],
    '            ASSERT(false);',
    '            for (;;) {  // never fall through; the volatile read keeps the loop well-defined',
    '                (void)*p;',
    '            }',
    '        }',
    '    }',
    '}',
    'FORCE_INLINE void wd_wait_valid(uint32_t sem_id) {',
    '    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));',
    '    for (uint32_t n = 0;; ++n) {',
    '        invalidate_l1_cache();',
    '        if (*p == VALID) {',
    '            return;',
    '        }',
    '        if (n == kWdPolls) {',
    '            WAYPOINT("%s");' % WAYPOINTS[1],
    '            ASSERT(false);',
    '            for (;;) {',
    '                (void)*p;',
    '            }',
    '        }',
    '    }',
    '}',
    '// One round = one K chunk AND its V chunk (the K64f stage-3 round shape: one credit, one flag per chunk).',
    '// The flag is ALWAYS reset (served R:416-418 order: reset before crediting), so the VALID wait that follows',
    '// really waits for this round. withhold (test flag 0x200 only, the planted hang) skips just the credit: the',
    '// sink then waits in QWDV and its upstream in QWDC, and no stale slot is ever pushed.',
    'FORCE_INLINE void credit_prev(',
    '    Noc noc, uint32_t receiver_sem_id, uint32_t sender_sem_id, uint32_t px, uint32_t py, bool withhold) {',
    '    Semaphore<>(receiver_sem_id).set(INVALID);',
    '    if (!withhold) {',
    '        Semaphore<>(sender_sem_id).up(noc, px, py, 1);',
    '    }',
    '}',
    'FORCE_INLINE void forward_round(',
    '    Noc noc,',
    '    uint32_t sender_sem_id,',
    '    uint32_t receiver_sem_id,',
    '    uint32_t valid_sem_id,',
    '    uint32_t nx,',
    '    uint32_t ny,',
    '    uint32_t k_addr,',
    '    uint32_t k_bytes,',
    '    uint32_t v_addr,',
    '    uint32_t v_bytes) {',
    '    wd_wait_credit(sender_sem_id);  // next has reserved both slots of this round',
    '    Semaphore<>(sender_sem_id).set(0);',
    '    noc.async_write(',
    '        CoreLocalMem<uint32_t>(k_addr), UnicastEndpoint{}, k_bytes, {}, {.noc_x = nx, .noc_y = ny, .addr = k_addr});',
    '    noc.async_write(',
    '        CoreLocalMem<uint32_t>(v_addr), UnicastEndpoint{}, v_bytes, {}, {.noc_x = nx, .noc_y = ny, .addr = v_addr});',
    '    noc.async_write_barrier();  // data ACKED on next before the flag (K64f order)',
    '    Semaphore<>(valid_sem_id).relay_unicast(noc, Semaphore<>(receiver_sem_id), nx, ny);',
    '    noc.async_writes_flushed();  // no write left in flight before any later read barrier',
    '}',
    '}  // namespace qwen_pf')

# ---- R2: mode constant and static envelope (after R:97) ----------------------------------------
R2_ANCHOR = lines('    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(33) == 1;')
R2 = R2_ANCHOR + lines(
    '    // [QWEN-SDPA-PF] R2: this reader is only ever the causal flexible-chunked paged prefill chain reader.',
    '    constexpr bool kQwenKvChain = true;',
    '    static_assert(',
    '        is_causal && is_chunked && !use_provided_mask && !use_attention_sink && !use_mla && !mcast_enabled &&',
    '            sliding_window_size == 0 && !use_streaming_compute,',
    '        "[QWEN-SDPA-PF] chain reader: causal flexible-chunked paged prefill only");')

# ---- R3: parse the chain block (R:150-151) -----------------------------------------------------
R3_OLD = lines(
    '    if constexpr (!is_causal) {',
    '        is_chain_participant = get_arg_val<uint32_t>(argidx++);')
R3_NEW = lines(
    '    if constexpr (!is_causal || kQwenKvChain) {  // [QWEN-SDPA-PF] R3: the causal G6 chain block (factory F5)',
    '        is_chain_participant = get_arg_val<uint32_t>(argidx++);')

# ---- R4: flags (after R:196) -------------------------------------------------------------------
R4_ANCHOR = lines('    constexpr uint32_t cb_id_chunk_start_idx_writer = get_compile_time_arg_val(cb_arg_offset + 7);')
R4 = R4_ANCHOR + lines(
    '    // [QWEN-SDPA-PF] R4: the flags, a suffix CT arg after the CB ids (factory F7).',
    '    constexpr uint32_t qwen_pf_flags = get_compile_time_arg_val(cb_arg_offset + 8);',
    '    constexpr bool pf_inj_batch = (qwen_pf_flags & 0x2u) != 0;',
    '    constexpr bool pf_test_mutate = (qwen_pf_flags & 0x100u) != 0;',
    '    constexpr bool pf_test_hang = (qwen_pf_flags & 0x200u) != 0;',
    '    static_assert((qwen_pf_flags & 0x1u) != 0, "[QWEN-SDPA-PF] the chain reader needs flag 0x1");',
    '    static_assert(!pf_test_mutate || NKH >= 2, "[QWEN-SDPA-PF] the mutation flips the KV head");')

# ---- R5: barrier cadence (after R:209) ---------------------------------------------------------
R5_ANCHOR = lines('    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();')
R5 = R5_ANCHOR + lines(
    "    // [QWEN-SDPA-PF] R5: only the injector's K/V DRAM reads change cadence under flag 0x2 (one barrier per",
    '    // chunk, as read_chunk_for_forwarding). DMA batching only: same addresses, same bytes, same placement.',
    '    const uint32_t kv_bt = (pf_inj_batch && is_injector) ? k_chunk_tiles : barrier_threshold;',
    "    // A participant's Q read must be the subblock read, which R7 puts behind an atomic barrier (its credit",
    '    // has landed); the whole-chunk Q read before the k loop has no such barrier (the BH read-barrier hazard).',
    '    // The factory envelope refuses q_num_subblocks == 1 too (qk_in0_num_subblocks > 1).',
    '    static_assert(use_q_subblock_push, "[QWEN-SDPA-PF] the chain reader needs the Q subblock push");')

# ---- R6: roles (R:392-399) ---------------------------------------------------------------------
R6_OLD = lines(
    '            // Chain forwarding conditions are loop-invariant \u2014 compute once',
    '            bool should_forward = false;',
    '            bool should_receive = false;',
    '            if constexpr (!is_causal) {',
    '                should_forward = is_chain_participant && !is_sink && (nb == chain_batch && nq == chain_head) &&',
    '                                 (q_iter < next_core_q_chunks);',
    '                should_receive = is_chain_participant && !is_injector && (nb == chain_batch && nq == chain_head);',
    '            }')
R6_NEW = lines(
    '            // [QWEN-SDPA-PF] R6: G6 - every member has the same unit list (factory F4), so every round is shared.',
    '            const bool should_forward = is_chain_participant && !is_sink;',
    '            const bool should_receive = is_chain_participant && !is_injector;',
    '            (void)q_iter;')

# ---- R7: one round per k_chunk (R:406-709) -----------------------------------------------------
R7_FIRST = '                const uint32_t k_start_tile_id = k_tile_shape.id_of(nb, k_head, kv_row_start_tile, 0);' + NL
R7_LAST = lines(
    '                    if (!should_receive) {',
    '                        cb_v.push_back(v_chunk_tiles);',
    '                    }',
    '                }')
R7_NEW = lines(
    '                // [QWEN-SDPA-PF] R7: one round per k_chunk (K and its V). Push order is the served one: K, (Q',
    "                // subblocks at the unit's first chunk), V. Every member pushes to its own compute BEFORE",
    '                // forwarding (served R:420 and the K64f leader order), which is safe because only this reader',
    "                // ever rewrites these slots, two rounds later, after this forward's barrier.",
    '                const uint32_t k_slot = cb_k.get_write_ptr();  // the reserves below target exactly these slots',
    '                const uint32_t v_slot = cb_v.get_write_ptr();  // (reserve_back never moves the write pointer)',
    '                const bool last_round =',
    '                    (global_q_iter + 1 == global_q_count) && ((k_chunk + 1) * Sk_chunk_t >= q_high_idx);',
    '                if (should_receive) {',
    '                    cb_k.reserve_back(k_chunk_tiles);',
    '                    cb_v.reserve_back(v_chunk_tiles);',
    '                    qwen_pf::credit_prev(  // planted hang (0x200): the sink withholds its last credit, not the reset',
    '                        noc,',
    '                        receiver_semaphore_id,',
    '                        sender_semaphore_id,',
    '                        prev_physical_x,',
    '                        prev_physical_y,',
    '                        pf_test_hang && is_sink && last_round);',
    '                    qwen_pf::wd_wait_valid(receiver_semaphore_id);  // K and V both landed (acked write, then flag)',
    '                    cb_k.push_back(k_chunk_tiles);',
    '                } else {',
    '                    // The served K read (R:425-439): only the barrier cadence (kv_bt) and, under the test-only',
    "                    // mutation flag, the injector's chunk-0 KV head differ.",
    '                    const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;',
    '                    const uint32_t k_head_rd = (pf_test_mutate && is_injector && k_chunk == 0) ? (k_head ^ 1u) : k_head;',
    '                    read_paged_chunk_with_padding<NKH, block_size_t, DHt>(',
    '                        k_reader,',
    '                        cb_k_in,',
    '                        k_head_rd,',
    '                        k_chunk_start_row_num,',
    '                        kv_row_tile_count,',
    '                        DHt,',
    '                        Sk_chunk_t,',
    '                        DHt,',
    '                        k_tile_bytes,',
    '                        kv_bt,',
    '                        page_table_ptr,',
    '                        true  // transpose=true for K reads',
    '                    );',
    '                }',
    '',
    '                // Q subblock push (R:584-599), plus one barrier: a participant\'s credit atomic has completed',
    '                // before the read barriers inside read_q_subblock.',
    '                if constexpr (use_q_subblock_push) {',
    '                    if (k_chunk == k_loop_start) {',
    '                        if (is_chain_participant) {',
    '                            noc.async_atomic_barrier();',
    '                        }',
    '                        for (uint32_t q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {',
    '                            read_q_subblock<q_tile_bytes>(',
    '                                q_reader,',
    '                                cb_q_in,',
    '                                q_read_tile_id,',
    '                                q_sub * qk_subblock_h,',
    '                                qk_subblock_h,',
    '                                q_row_tile_count,',
    '                                DHt,',
    '                                DHt,',
    '                                barrier_threshold);',
    '                        }',
    '                    }',
    '                }',
    '',
    '                if (should_receive) {',
    '                    cb_v.push_back(v_chunk_tiles);',
    '                } else {',
    '                    // The served V read (R:617-632), kv_bt cadence.',
    '                    const uint32_t kv_chunk_start_row_num = k_chunk * Sk_chunk_t;',
    '                    constexpr uint32_t head_dim = (use_mla && !mla_kv_overlap) ? vDHt : DHt;',
    '                    read_paged_chunk_with_padding<NVH, block_size_t, head_dim>(',
    '                        v_reader,',
    '                        cb_v_in,',
    '                        v_head,',
    '                        kv_chunk_start_row_num,',
    '                        kv_row_tile_count,',
    '                        vDHt,',
    '                        Sk_chunk_t,',
    '                        vDHt,',
    '                        v_tile_bytes,',
    '                        kv_bt,',
    '                        page_table_ptr,',
    '                        false,',
    '                        skip_src_cols);',
    '                }',
    '',
    '                // Forward both slots of this round to the next member once it has reserved them.',
    '                if (should_forward) {',
    '                    qwen_pf::forward_round(',
    '                        noc,',
    '                        sender_semaphore_id,',
    '                        receiver_semaphore_id,',
    '                        valid_semaphore_id,',
    '                        next_physical_x,',
    '                        next_physical_y,',
    '                        k_slot,',
    '                        k_chunk_tiles * k_tile_bytes,',
    '                        v_slot,',
    '                        v_chunk_tiles * v_tile_bytes);',
    '                }')

# ---- R8: exit (before the final brace) ---------------------------------------------------------
R8_ANCHOR = lines(
    '    }  // close phase',
    '}')
R8_NEW = lines(
    '    }  // close phase',
    '    if (is_chain_participant) {  // [QWEN-SDPA-PF] R8: nothing in flight at exit; semaphores back to their initial values',
    '        noc.async_write_barrier();',
    '        noc.async_atomic_barrier();',
    '        Semaphore<>(receiver_semaphore_id).set(INVALID);',
    '        Semaphore<>(sender_semaphore_id).set(INVALID);',
    '    }',
    '}')

# ---- R9: [[maybe_unused]] on the orphaned declarations -----------------------------------------
# (name, served line, served declaration line). The spec's two first; then the ones R6/R7 orphan by
# removing their only reads (the mask block, the non-paged branches, the non-causal role test and
# the mcast forward).
R9_ORPHANS = (
    ('mask_batch_offset', 293, '        uint32_t mask_batch_offset = 0;'),
    ('q_iter', 342, '            const uint32_t q_iter = per_head_q_iter;'),
    ('use_padded_mask', 81, '    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(19) == 1;'),
    ('chain_batch', 137, '    uint32_t chain_batch = 0;'),
    ('chain_head', 138, '    uint32_t chain_head = 0;'),
    ('next_core_q_chunks', 143, '    uint32_t next_core_q_chunks = 0;'),
    ('mcast_num_dests', 144, '    uint32_t mcast_num_dests = 0;'),
    ('sender_wait_count', 148, '    uint32_t sender_wait_count = 1;'),
    ('mask_tile_bytes', 201, '    constexpr uint32_t mask_tile_bytes = use_provided_mask ? get_tile_size(cb_mask_in) : 0;'),
    ('mask_reader', 214, '    const auto mask_reader = TensorAccessor(mask_args, mask_addr);'),
    ('k_tile_shape', 221, '    const auto k_tile_shape = TensorTileShape(B, NKH, valid_Skt, DHt);'),
    ('v_tile_shape', 228, '    const auto v_tile_shape = TensorTileShape(B, NVH, valid_Skt, use_mla && !mla_kv_overlap ? vDHt : DHt);'),
    ('cb_mask', 235, '    CircularBuffer cb_mask(cb_mask_in);'),
)


def maybe_unused(line):
    stripped = line.lstrip(' ')
    return line[:len(line) - len(stripped)] + '[[maybe_unused]] ' + stripped


def r7_old(text):
    """The served R:406-709 text: from k_start_tile_id through the end of the V forward block."""
    start = text.index(R7_FIRST)
    end = text.index(R7_LAST, start) + len(R7_LAST)
    return text[start:end]


def edits(text):
    """(name, served line, old text, new text), in file order. R7's old text is cut from `text`."""
    out = [('R1 helpers', 13, R1_ANCHOR, R1)]
    for name, line, decl in sorted(R9_ORPHANS, key=lambda entry: entry[1]):
        if line < 97:
            out.append(('R9 %s' % name, line, decl + NL, maybe_unused(decl) + NL))
    out.append(('R2 envelope', 97, R2_ANCHOR, R2))
    for name, line, decl in sorted(R9_ORPHANS, key=lambda entry: entry[1]):
        if 97 < line < 150:
            out.append(('R9 %s' % name, line, decl + NL, maybe_unused(decl) + NL))
    out.append(('R3 chain block', 150, R3_OLD, R3_NEW))
    out.append(('R4 flags', 196, R4_ANCHOR, R4))
    for name, line, decl in sorted(R9_ORPHANS, key=lambda entry: entry[1]):
        if 196 < line < 209:
            out.append(('R9 %s' % name, line, decl + NL, maybe_unused(decl) + NL))
    out.append(('R5 cadence', 209, R5_ANCHOR, R5))
    for name, line, decl in sorted(R9_ORPHANS, key=lambda entry: entry[1]):
        if line > 209:
            out.append(('R9 %s' % name, line, decl + NL, maybe_unused(decl) + NL))
    out.append(('R6 roles', 392, R6_OLD, R6_NEW))
    out.append(('R7 rounds', 406, r7_old(text), R7_NEW))
    out.append(('R8 exit', 717, R8_ANCHOR, R8_NEW))
    return out


class Refusal(Exception):
    """The input is not the served reader, or an anchor drifted: nothing is written."""


def check_base(data):
    digest = sha(data)
    if digest != BASE_SHA:
        raise Refusal('input sha256 %s is not the served reader f97f5490 (%s); refusing' % (digest, BASE_SHA))


def check_anchors(text):
    for name, line, old, _new in edits(text):
        count = text.count(old)
        if count != 1:
            raise Refusal('%s: anchor occurs %d times (need exactly 1); refusing on drift' % (name, count))
        found = text[:text.index(old)].count(NL) + 1
        if found != line:
            raise Refusal('%s: anchor starts at line %d, recorded R:%d; refusing on drift' % (name, found, line))
    first, last = R7_FIRST, R7_LAST
    old = r7_old(text)
    end_line = text[:text.index(old) + len(old)].count(NL)
    if end_line != 709 or not old.startswith(first) or not old.endswith(last):
        raise Refusal('R7: the replaced span ends at line %d, recorded R:709' % end_line)


def apply_edits(text):
    check_anchors(text)
    for _name, _line, old, new in edits(text):
        text = text.replace(old, new)
    return text


def revert_edits(text, served_r7):
    """Every edit's new text back to its old text, last edit first: must give the served reader.
    R7 replaces 300 served lines, so its old text is not in the chain reader: the caller passes the
    served R:406-709 span (r7_old() of the served reader; the CPU tests rebuild the served reader
    from the committed K0 probe readers, which revert to f97f5490 on their own)."""
    for name, _line, old, new in reversed(edits(served_r7)):
        count = text.count(new)
        if count != 1:
            raise Refusal('%s: edited text occurs %d times in the chain reader (need exactly 1)' % (name, count))
        text = text.replace(new, old)
    return text


def build(data):
    """Served reader bytes -> the chain reader bytes (LF, UTF-8). Refuses a wrong base or drift."""
    check_base(data)
    text = data.decode('utf-8')
    out = apply_edits(text)
    if revert_edits(out, r7_old(text)) != text:
        raise Refusal('reverting the edits does not give the served reader back')
    encoded = out.encode('utf-8')
    if chr(13).encode() in encoded:
        raise Refusal('output has a CR; files must be LF')
    return encoded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('--reader', type=Path, default=DEFAULT_READER,
                        help='the served reader_interleaved.cpp (sha256 f97f5490...)')
    parser.add_argument('--out', type=Path, help='write %s into this directory (default: next to this script)' % OUTPUT_NAME)
    parser.add_argument('--check', type=Path, help='compare against %s in this directory; write nothing' % OUTPUT_NAME)
    parser.add_argument('--dump', action='store_true', help='print the generated reader; write nothing')
    parser.add_argument('--record', action='store_true', help='skip the recorded-output check and print the sha')
    options = parser.parse_args(argv)
    try:
        data = options.reader.read_bytes()
    except OSError as error:
        print('REFUSING: cannot read the served reader %s (%s)' % (options.reader, error), file=sys.stderr)
        return 2
    print('input %s sha256=%s' % (options.reader, sha(data)), file=sys.stderr if options.dump else sys.stdout)
    try:
        out = build(data)
    except Refusal as error:
        print('REFUSING: %s' % error, file=sys.stderr)
        return 2
    digest = sha(out)
    if not options.record and digest != READER_OUTPUT:
        print('REFUSING: output sha256 %s is not the recorded %s (edit list changed? rerun with --record and '
              'update READER_OUTPUT deliberately)' % (digest, READER_OUTPUT), file=sys.stderr)
        return 2
    if options.dump:
        sys.stdout.write(out.decode('utf-8'))
        return 0
    line = '%s sha256=%s' % (OUTPUT_NAME, digest)
    if options.check is not None:
        path = options.check / OUTPUT_NAME
        ok = path.is_file() and path.read_bytes() == out
        print(line + '  ' + ('matches ' + str(path) if ok else 'DIFFERS from ' + str(path)))
        return 0 if ok else 1
    target = (options.out or HERE) / OUTPUT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(out)
    print(line + '  written ' + str(target))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
