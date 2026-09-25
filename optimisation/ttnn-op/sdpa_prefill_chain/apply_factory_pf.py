"""Apply the [QWEN-SDPA-PF] edits F0-F7 to the prefill SDPA factory sdpa_program_factory.cpp.

Prefill lever #1 (sdpa-prefill-share-spec.md section 3.2): the G6 unicast K/V chain for the served
causal, flexible-chunked, paged SDPA. A call opts in per call with
SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0000 | flags (this factory never reads that
field; its default 16 never carries the tag), and only then does the factory group the cores
whose whole unit list maps to the same (batch, kv_head, q_chunk) sequence into chains, create the
three chain semaphores, pass the 14-word chain block and the flags, and select
reader_interleaved_qwen_chain.cpp. Every other call builds the served descriptor unchanged.

    F0  includes <algorithm> <cstdlib> <numeric> <vector> after <cmath>
    F1  sentinel parse after `use_zigzag_balancing = is_causal` (tag, flags, refusals)
    F2  semaphore ids 0/1/2 into reader CT 29/30/31 also in chain mode
    F7  the flags as a reader suffix CT arg (cb_arg_offset + 8), after the reader CB-id insert
    F3  the three semaphore descriptors (INVALID / INVALID / VALID over core_grid) in chain mode
    F4  the envelope TT_FATAL, the G6 topology and the '[QWEN-SDPA-PF] flags=' log line, before the
        mcast_enabled update
    F6  the chain reader's kernel source in chain mode
    F5  the 14-word runtime chain block also in chain mode

Input must be the served factory, sha256 fd8c0676... (probe-v25, images 1b9b6445 / 0648ca9a /
the ttbuild tree); any other input is refused. Every anchor must occur exactly once and start at
the line recorded below; the result must hash to PF_FACTORY (so a changed edit list has to update
it deliberately, --record prints it); reverting the edits must give fd8c0676 back; the
qwen_draft_fp32_intermediates block (the image patch F:706-714) must be byte-identical in the
output; and every new line outside F0 lies inside create_descriptor (nothing is added at
namespace scope: the unity-build collision trap). Run on an already patched file it reports and
exits 0.

    python3 apply_factory_pf.py <factory.cpp> --out X.cpp     # write the patched factory to X.cpp
    python3 apply_factory_pf.py <factory.cpp>                 # in place, keeps .orig-fd8c0676
    python3 apply_factory_pf.py <factory.cpp> --record        # print the output sha, do not enforce it
"""

import argparse
import hashlib
from pathlib import Path
import sys

NL = chr(10)

BASE_FACTORY = 'fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783'
PF_FACTORY = 'bfab8558d889ad215f0e9ee732c75a4142be1a1e7f37810f4ca8c5e3c73bdf65'

# Constants that cross file boundaries (reader, Python opt-in, bench, card-M test; the CPU tests
# check each against the C++ text).
PF_TAG = 0x5EFA0000
PF_TAG_MASK = 0xFFFF0000
FLAG_KV_CHAIN = 0x1
FLAG_INJ_BATCH = 0x2
FLAG_NOC_ORDER = 0x4
FLAG_TEST_MUTATE = 0x100
FLAG_TEST_HANG = 0x200
FLAG_RESERVED_PAIR_FUSION = 0x1000
PRODUCTION_FLAGS = FLAG_KV_CHAIN | FLAG_INJ_BATCH | FLAG_NOC_ORDER
KNOWN_FLAGS = PRODUCTION_FLAGS | FLAG_TEST_MUTATE | FLAG_TEST_HANG
TEST_FLAGS = FLAG_TEST_MUTATE | FLAG_TEST_HANG
TEST_ENV = 'QWEN_SDPA_PF_TEST'
SEMAPHORE_IDS = dict(sender=0, receiver=1, valid=2)
FLAGS_CT_OFFSET = 8            # reader CT index cb_arg_offset + 8
CHAIN_RT_WORDS = ('participates', 'is_injector', 'is_sink', 'batch', 'head', 'q_chunk_start', 'q_chunk_count',
                  'prev_physical.x', 'prev_physical.y', 'next_physical.x', 'next_physical.y',
                  'next_core_q_chunks', 'mcast_num_dests', 'mcast_sender_wait')
NOC_GRID = (17, 12)            # BH NoC grid: the flag-0x4 order-cost proxy only
NOC_ORDER_MAX_MEMBERS = 8      # F4 refuses the 0x4 exhaustive search above this (8! = 40,320 orders)
NOC_ORDER_MARKER = '[QWEN-SDPA-PF] noc_order searches at most'
# The row counts (S) the Python opt-in may send and the card-M Q1 sweep qualifies (the opt-in is
# scripts/ci/lever_n_m3native_patch.py section I, SDPA_PF_ROWS, re-exported by pf_optin.py; and
# test_sdpa_prefill_chain_card_m.py ROWS; the CPU tests keep the three equal). Any other S takes the
# served path: 256-1792-row tail chunks and S > 2048 (unequal groups) are not qualified.
QUALIFIED_ROWS = (512, 1024, 2048)
QUALIFIED_CHUNK = 128

LOG_MARKER = '[QWEN-SDPA-PF] flags='
ENVELOPE_MARKER = '[QWEN-SDPA-PF] kv_chain outside its qualified envelope'
FLAGS_MARKER = '[QWEN-SDPA-PF] unknown or incomplete flags'
TEST_MARKER = '[QWEN-SDPA-PF] test-only flags'
READER_NAME = 'reader_interleaved_qwen_chain.cpp'
SERVED_READER_NAME = 'reader_interleaved.cpp'
KERNEL_DIR = 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/'
MARKERS = (LOG_MARKER, ENVELOPE_MARKER, FLAGS_MARKER, TEST_MARKER, READER_NAME, NOC_ORDER_MARKER)
# The F4 envelope, predicate by predicate: the spec 3.2 eighteen plus qk_in0_num_subblocks > 1 (the reader's
# Q subblock push, R5's static_assert; the CPU tests check each is in the TT_FATAL).
ENVELOPE = ('is_causal', 'flexible_chunked', '!has_sliding_window', '!use_provided_mask', '!is_windowed',
            '!use_attention_sink', '!use_mla', '!use_streaming_compute', 'lightweight_causal', 'B == 1',
            'NKH == NVH', 'NQH % NKH == 0', 'global_q_pair_distribute', 'Sq_chunk_t == Sk_chunk_t', 'DHt == vDHt',
            'num_phases == 1', 'k_df == tt::DataFormat::Bfp8_b', 'v_df == k_df', 'qk_in0_num_subblocks > 1')
# The image patch a factory graft must carry through untouched (spec 3.2 rules, F:706-714).
PROTECTED_START = '    const bool qwen_draft_fp32_intermediates =' + NL
PROTECTED_END = '    tt::DataFormat stats_df = qwen_draft_fp32_intermediates ? tt::DataFormat::Float32 : im_df;' + NL
FUNCTION_LINE = 'ProgramDescriptor SDPAOperation::SDPAProgramFactory::create_descriptor(' + NL


def lines(*parts):
    return ''.join(part + NL for part in parts)


# ---- F0: includes (F:21) -----------------------------------------------------------------------
F0_ANCHOR = lines('#include <cmath>')
F0 = F0_ANCHOR + lines(
    '#include <algorithm>',
    '#include <cstdlib>',
    '#include <numeric>',
    '#include <vector>')

# ---- F1: sentinel parse (F:527) ----------------------------------------------------------------
F1_ANCHOR = lines('    const bool use_zigzag_balancing = is_causal;')
F1 = F1_ANCHOR + lines(
    '    // [QWEN-SDPA-PF] per-call opt-in carried in SDPAProgramConfig.max_cores_per_head_batch (this factory',
    '    // never reads that field). Tag 0x5EFA in the high half, flags in the low half. Spec: sdpa-prefill-share-spec.md',
    '    // 3.1-3.2; edits F0-F7 of optimisation/ttnn-op/sdpa_prefill_chain/apply_factory_pf.py.',
    '    constexpr uint32_t kQwenPfTagMask = 0xFFFF0000u, kQwenPfTag = 0x5EFA0000u;',
    '    constexpr uint32_t kQwenPfKvChain = 0x1u, kQwenPfInjBatch = 0x2u, kQwenPfNocOrder = 0x4u;',
    '    constexpr uint32_t kQwenPfTestMutate = 0x100u, kQwenPfTestHang = 0x200u;',
    '    const uint32_t qwen_pf_word =',
    '        program_config.has_value() ? static_cast<uint32_t>(program_config->max_cores_per_head_batch) : 0u;',
    '    const bool qwen_pf_mode = (qwen_pf_word & kQwenPfTagMask) == kQwenPfTag;',
    '    const uint32_t qwen_pf_flags = qwen_pf_mode ? (qwen_pf_word & ~kQwenPfTagMask) : 0u;',
    '    const bool qwen_kv_chain = qwen_pf_mode && (qwen_pf_flags & kQwenPfKvChain) != 0;',
    '    if (qwen_pf_mode) {',
    '        TT_FATAL(qwen_kv_chain && (qwen_pf_flags & ~(kQwenPfKvChain | kQwenPfInjBatch | kQwenPfNocOrder |',
    '                                                    kQwenPfTestMutate | kQwenPfTestHang)) == 0,',
    '                 "[QWEN-SDPA-PF] unknown or incomplete flags {:#x}", qwen_pf_flags);',
    '        const char* qwen_pf_test_env = std::getenv("QWEN_SDPA_PF_TEST");',
    '        TT_FATAL((qwen_pf_flags & (kQwenPfTestMutate | kQwenPfTestHang)) == 0 ||',
    '                     (qwen_pf_test_env != nullptr && std::string(qwen_pf_test_env) == "1"),',
    '                 "[QWEN-SDPA-PF] test-only flags {:#x} need QWEN_SDPA_PF_TEST=1", qwen_pf_flags);',
    '    }')

# ---- F2: semaphore ids (F:586-587) -------------------------------------------------------------
F2_OLD = lines(
    '    if (!is_causal) {',
    '        sender_semaphore_id = 0;')
F2_NEW = lines(
    '    if (!is_causal || qwen_kv_chain) {  // [QWEN-SDPA-PF] ids 0/1/2 into reader CT 29/30/31',
    '        sender_semaphore_id = 0;')

# ---- F7: flags as a reader suffix CT arg (F:832-833) -------------------------------------------
F7_ANCHOR = lines(
    '    reader_compile_time_args.insert(',
    '        reader_compile_time_args.end(), reader_cb_compile_time_args.begin(), reader_cb_compile_time_args.end());')
F7 = F7_ANCHOR + lines(
    '    if (qwen_kv_chain) {',
    '        reader_compile_time_args.push_back(qwen_pf_flags);  // [QWEN-SDPA-PF] reader CT cb_arg_offset + 8',
    '    }')

# ---- F3: semaphore descriptors (F:842-843) -----------------------------------------------------
F3_OLD = lines(
    '    if (!is_causal) {',
    '        desc.semaphores.push_back(SemaphoreDescriptor{')
F3_NEW = lines(
    '    if (!is_causal || qwen_kv_chain) {  // [QWEN-SDPA-PF] INVALID / INVALID / VALID over core_grid',
    '        desc.semaphores.push_back(SemaphoreDescriptor{')

# ---- F4: envelope, G6 topology, log line (before F:1347) ---------------------------------------
F4_ANCHOR = lines('    // Update mcast_enabled compile-time arg now that chain construction is complete')
F4_BLOCK = lines(
    '    // [QWEN-SDPA-PF] G6 K/V chains. Cores whose whole unit list maps to the same (batch, kv_head, q_chunk)',
    '    // sequence read byte-identical K/V streams (k ranges depend only on q_chunk and the device-read',
    '    // chunk_start; tile ids only on kv_head, row and page table), and so stay at the same K/V CB phase.',
    '    if (qwen_kv_chain) {',
    '        TT_FATAL(is_causal && flexible_chunked && !has_sliding_window && !use_provided_mask && !is_windowed &&',
    '                     !use_attention_sink && !use_mla && !use_streaming_compute && lightweight_causal && B == 1 &&',
    '                     NKH == NVH && NQH % NKH == 0 && global_q_pair_distribute && Sq_chunk_t == Sk_chunk_t &&',
    '                     DHt == vDHt && num_phases == 1 && k_df == tt::DataFormat::Bfp8_b && v_df == k_df &&',
    '                     qk_in0_num_subblocks > 1,  // the reader Q subblock push (its read follows an atomic barrier)',
    '                 "[QWEN-SDPA-PF] kv_chain outside its qualified envelope");',
    '        const uint32_t q_per_kv = NQH / NKH;',
    '        constexpr uint32_t kQwenNocX = 17, kQwenNocY = 12;  // BH NoC grid; order-cost proxy only',
    '        std::map<std::vector<uint32_t>, std::vector<uint32_t>> qwen_groups;  // unit list -> linear core ids',
    '        for (uint32_t i = 0; i < num_cores; ++i) {',
    '            // The per-core range of the reader runtime-arg loop below, including its clamp.',
    '            uint32_t g_start = i * global_q_base_chunks_per_core +',
    '                               std::min(i, global_q_cores_doing_extra) * global_q_extra_chunks_per_core;',
    '            uint32_t g_count = global_q_base_chunks_per_core +',
    '                               ((i < global_q_cores_doing_extra) ? global_q_extra_chunks_per_core : 0u);',
    '            if (g_start >= total_q_chunks) {',
    '                continue;',
    '            }',
    '            if (g_start + g_count > total_q_chunks) {',
    '                g_count = total_q_chunks - g_start;',
    '            }',
    '            if (g_count == 0) {',
    '                continue;',
    '            }',
    '            std::vector<uint32_t> key;',
    '            for (uint32_t u = 0; u < g_count; ++u) {  // host mirror of q_chunk_remapping.hpp (zigzag == is_causal)',
    '                const uint32_t lin = g_start + u, head_idx = lin / q_num_chunks, pos = lin % q_num_chunks;',
    '                const uint32_t q = (pos % 2 == 0) ? pos / 2 : q_num_chunks - 1 - pos / 2;',
    '                const uint32_t remapped = head_idx * q_num_chunks + q;',
    '                const uint32_t nb = remapped / (NQH * q_num_chunks), nq = (remapped / q_num_chunks) % NQH;',
    '                key.insert(key.end(), {nb, nq / q_per_kv, q});',
    '            }',
    '            qwen_groups[key].push_back(i);',
    '        }',
    '        uint32_t qwen_chains = 0, qwen_members = 0;',
    '        for (const auto& [key, members] : qwen_groups) {',
    '            if (members.size() < 2) {',
    '                continue;  // a singleton reads privately (the served path in the chain reader)',
    '            }',
    '            std::vector<CoreCoord> phys;',
    '            for (uint32_t i : members) {',
    '                phys.push_back(device->worker_core_from_logical_core(CoreCoord{i % grid_size.x, i / grid_size.x}));',
    '            }',
    '            std::vector<uint32_t> order(members.size());',
    '            std::iota(order.begin(), order.end(), 0u);  // raster: ascending linear index',
    '            if (qwen_pf_flags & kQwenPfNocOrder) {',
    '                // Exhaustive: minimise the sum over consecutive members of the direction-agnostic torus',
    '                // Manhattan distance on NoC coordinates; the first minimum wins. A group has at most',
    '                // NQH / NKH members (6 here: 720 orders); the bound keeps a wider model from stalling',
    '                // the program build in an n! search.',
    '                TT_FATAL(members.size() <= 8, "[QWEN-SDPA-PF] noc_order searches at most 8 members, a chain has {}",',
    '                         members.size());',
    '                auto cost = [&](const std::vector<uint32_t>& o) {',
    '                    uint32_t c = 0;',
    '                    for (size_t p = 0; p + 1 < o.size(); ++p) {',
    '                        const auto a = phys[o[p]], b = phys[o[p + 1]];',
    '                        const uint32_t dx = a.x > b.x ? a.x - b.x : b.x - a.x, dy = a.y > b.y ? a.y - b.y : b.y - a.y;',
    '                        c += std::min<uint32_t>(dx, kQwenNocX - dx) + std::min<uint32_t>(dy, kQwenNocY - dy);',
    '                    }',
    '                    return c;',
    '                };',
    '                std::vector<uint32_t> probe = order, best = order;',
    '                uint32_t best_cost = cost(order);',
    '                while (std::next_permutation(probe.begin(), probe.end())) {',
    '                    const uint32_t c = cost(probe);',
    '                    if (c < best_cost) {',
    '                        best_cost = c;',
    '                        best = probe;',
    '                    }',
    '                }',
    '                order = best;',
    '            }',
    '            const uint32_t units = static_cast<uint32_t>(key.size() / 3);',
    '            for (size_t p = 0; p < order.size(); ++p) {',
    '                auto& c = core_chain_info[members[order[p]]];',
    '                c.participates = true;',
    '                c.is_injector = (p == 0);',
    '                c.is_sink = (p + 1 == order.size());',
    '                c.batch = key[0];',
    '                c.head = key[1];',
    '                c.q_chunk_start = key[2];',
    '                c.q_chunk_count = units;  // host-only words',
    '                c.prev_physical = (p > 0) ? phys[order[p - 1]] : CoreCoord{0, 0};',
    '                c.next_physical = c.is_sink ? CoreCoord{0, 0} : phys[order[p + 1]];',
    '                c.next_core_q_chunks = c.is_sink ? 0u : units;',
    '                c.use_mcast = false;',
    '                c.mcast_num_dests = 0;',
    '                c.mcast_sender_wait = 0;',
    '            }',
    '            ++qwen_chains;',
    '            qwen_members += static_cast<uint32_t>(members.size());',
    '        }',
    '        log_info(tt::LogOp, "[QWEN-SDPA-PF] flags={:#x} kv_chain=1 chains={} members={} order={}", qwen_pf_flags,',
    '                 qwen_chains, qwen_members, (qwen_pf_flags & kQwenPfNocOrder) ? "noc" : "raster");',
    '    }',
    '')
F4 = F4_BLOCK + F4_ANCHOR

# ---- F6: kernel source (F:1353-1354) -----------------------------------------------------------
F6_OLD = lines(
    '    reader_desc.kernel_source =',
    '        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp";')
F6_NEW = lines(
    '    reader_desc.kernel_source =',
    '        qwen_kv_chain',
    '            ? "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp"',
    '            : "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp";')

# ---- F5: runtime chain block (F:1424-1425) -----------------------------------------------------
F5_OLD = lines(
    '        // Add chain metadata for non-causal case',
    '        if (!is_causal) {')
F5_NEW = lines(
    '        // Add chain metadata for non-causal case ([QWEN-SDPA-PF]: and for the causal G6 chain)',
    '        if (!is_causal || qwen_kv_chain) {')

# (label, served line the anchor starts on, old text, new text), in file order.
EDITS = (
    ('F0', 21, F0_ANCHOR, F0),
    ('F1', 527, F1_ANCHOR, F1),
    ('F2', 586, F2_OLD, F2_NEW),
    ('F7', 832, F7_ANCHOR, F7),
    ('F3', 842, F3_OLD, F3_NEW),
    ('F4', 1347, F4_ANCHOR, F4),
    ('F6', 1353, F6_OLD, F6_NEW),
    ('F5', 1424, F5_OLD, F5_NEW),
)


def decode_word(word, test_env=None):
    """F1 in Python: (chain, flags) for SDPAProgramConfig.max_cores_per_head_batch = word, or
    ValueError with the TT_FATAL text F1 raises. test_env is the value of QWEN_SDPA_PF_TEST."""
    word &= 0xFFFFFFFF
    if (word & PF_TAG_MASK) != PF_TAG:
        return False, 0
    flags = word & ~PF_TAG_MASK & 0xFFFFFFFF
    if not flags & FLAG_KV_CHAIN or flags & ~KNOWN_FLAGS:
        raise ValueError('%s %#x' % (FLAGS_MARKER, flags))
    if flags & TEST_FLAGS and test_env != '1':
        raise ValueError('%s %#x need %s=1' % (TEST_MARKER, flags, TEST_ENV))
    return True, flags


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Refusal(ValueError):
    """The input is not the served factory, or an anchor drifted: nothing is written."""


def check_anchors(text):
    for label, line, old, _new in EDITS:
        count = text.count(old)
        if count != 1:
            raise Refusal('anchor %s occurs %d times (need exactly 1)' % (label, count))
        found = text[:text.index(old)].count(NL) + 1
        if found != line:
            raise Refusal('anchor %s starts at line %d, recorded F:%d; refusing on drift' % (label, found, line))


def protected_block(text):
    start = text.index(PROTECTED_START)
    end = text.index(PROTECTED_END, start) + len(PROTECTED_END)
    return text[start:end]


def patch(source):
    """The F0-F7 factory from the fd8c0676 bytes; Refusal on any other input or on drift."""
    digest = sha256(source)
    if digest != BASE_FACTORY:
        raise Refusal('unexpected factory %s (need %s)' % (digest, BASE_FACTORY))
    text = source.decode('utf-8')
    check_anchors(text)
    for _label, _line, old, new in EDITS:
        text = text.replace(old, new)
    for marker in MARKERS:
        if marker not in text:
            raise Refusal('patched factory lacks %r' % marker)
    if protected_block(text) != protected_block(source.decode('utf-8')):
        raise Refusal('the qwen_draft_fp32_intermediates block moved')
    body = text.index(FUNCTION_LINE)
    for label, _line, _old, new in EDITS[1:]:
        if text.index(new) < body:
            raise Refusal('edit %s lies outside create_descriptor' % label)
    out = text.encode('utf-8')
    if chr(13).encode() in out:
        raise Refusal('output has a CR; files must be LF')
    return out


def unpatch(patched):
    """The inverse, last edit first: proves the output is the base plus exactly these edits."""
    text = patched.decode('utf-8')
    for label, _line, old, new in reversed(EDITS):
        if text.count(new) != 1:
            raise Refusal('edit %s occurs %d times' % (label, text.count(new)))
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
    if sha256(source) == PF_FACTORY:
        print('factory already [QWEN-SDPA-PF]-patched %s' % PF_FACTORY)
        if args.out:
            Path(args.out).write_bytes(source)
        return 0
    try:
        patched = patch(source)
    except Refusal as error:
        print('refused: %s' % error)
        return 1
    digest = sha256(patched)
    if not args.record and digest != PF_FACTORY:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, PF_FACTORY))
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
    print('factory %s -> [QWEN-SDPA-PF] %s written %s' % (BASE_FACTORY[:16], digest, target))
    print(digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
