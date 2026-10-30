"""Apply the [QWEN-SDPA-PF] ksplit edits PK0-PK13 to the prefill SDPA factory: K-SPLIT of a q chunk's K stream over S slices (prefill lever, flag 0x10).

Prefill lever PF3-K (optimisation/ttnn-op/sdpa_prefill_ksplit/README.md, docs/tp4-prefill3-ksplit.md). The oneq call (apply_factory_ps.py, flag 0x8) gives each of the
`total = 6 x R / 128` q chunks of a pass its own core. A pass of 1,024 rows keeps 48 of the grid's 130 cores busy, one of 512 rows 24: the rest idle, and a q chunk's K stream is
serial on its core (13.51 us a 128x128 block). This patch adds one flag to the per-call word:

    SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0000 | flags       flags: 0x1 chain, 0x2 inj_batch, 0x4 noc_order, 0x8 oneq, 0x10 ksplit (NEW; needs 0x1 and 0x8),
                                                                          0x400 ksplit_hang (NEW, test only: needs 0x10 and QWEN_SDPA_PF_TEST=1)

With 0x10 the factory gives every q chunk S = min(4, cores / total) consecutive SLICES: cores [s * total, (s + 1) * total) are slice s (the oneq layout, repeated), the F4 grouping
keys a chain by (batch, kv_head, q chunk, SLICE), and the three kernels are replaced by the ksplit twins (a reader derived from the chain reader, the pinned writer with the hand-off,
the pinned compute kernel with the RING instantiation): make_ks_reader.py, make_ks_writer.py, make_ks_compute.py. The K ranges of the slices are computed ON THE DEVICE from the chunk
start (the flexible path's tensor), by one integer function the three kernels carry alike. The statistics CBs become fp32 (the log-sum-exp is stored there; bf16 would be
6x the error of the output's own rounding). Nothing else moves; a call without 0x10 is the oneq call, byte for byte (the other edits are guarded by qwen_ks).

    PK0   the decode of the ksplit bit, S and the busy core count (after PS0)
    PK1   constants kQwenPfKsplit, kQwenPfKsHang and the two semaphore ids (F1's block)
    PK2   0x10 and 0x400 in F1's accepted set, 0x10 only with 0x8, 0x400 only with 0x10
    PK3   0x400 joins the test-only flags (QWEN_SDPA_PF_TEST=1)
    PK4   stats_df is fp32 under ksplit
    PK5   the extra CBs: previous output, previous lse, own lse, scale tile
    PK6   the suffix compile-time args of the reader, the writer and the compute kernel
    PK7   the hand-off semaphores (credit, VALID), over the core grid
    PK8   the ksplit envelope (at least 2 slices, fp32 dest and non-approx exp, a scale exact in bf16)
    PK9   the F4 range of a ksplit core (slice-major) and the slice in the chain key
    PK10  the log line `[QWEN-SDPA-PF] ksplit=1 ...`
    PK11  the kernel sources
    PK12  the per-core slice, previous and next core (slice-major)
    PK13  the runtime args of the three kernels

Input must be the PS factory (apply_factory_ps.py output, sha256 37b9d966...), the PF factory or the served fd8c0676 (PF and PS are applied first); any other input is refused. Every
anchor occurs exactly once and starts at the line recorded below (in the PS file); the result must hash to PK_FACTORY; the edits invert; every new line lies inside create_descriptor
(nothing at namespace scope: the unity-build collision trap); the qwen_draft_fp32_intermediates block is byte-identical; no CR.

    python3 apply_factory_pk.py <factory.cpp> --out X.cpp     # write the patched factory to X.cpp
    python3 apply_factory_pk.py <factory.cpp>                 # in place, keeps .orig-<sha8>
    python3 apply_factory_pk.py <factory.cpp> --record        # print the output sha, do not enforce it
    python3 apply_factory_pk.py <factory.cpp> --lines         # print where each anchor starts in the PS file (to record them)

NOT COMPILED. The anchors, the inversion and the stub compile of the touched regions (stub_compile_ksplit.py) are checked on the CPU; the ttbuild ninja run is the proof.
"""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sdpa_prefill_oneq'))
sys.path.insert(0, str(HERE.parent / 'sdpa_prefill_chain'))

import apply_factory_pf as pf  # noqa: E402
import apply_factory_ps as ps  # noqa: E402

NL = chr(10)

SERVED_FACTORY = pf.BASE_FACTORY
PF_FACTORY = pf.PF_FACTORY
PS_FACTORY = ps.PS_FACTORY
PK_FACTORY = '692c79ee9cbc2f29ac3ab88bb274828afcf3ce76461930be61bd6edd07c45e68'

# Constants that cross file boundaries (the planner, the kernels' generators, the card harness, the Python hook; the CPU tests check each against the C++ text).
PF_TAG = pf.PF_TAG
FLAG_KSPLIT = 0x10
FLAG_KS_HANG = 0x400
KS_BITS = pf.FLAG_KV_CHAIN | ps.FLAG_ONEQ | FLAG_KSPLIT          # 0x19: the early decode needs all three
PRODUCTION_FLAGS = (0x1, 0x3, 0x5, 0x7)
KSPLIT_FLAGS = tuple(flags | ps.FLAG_ONEQ | FLAG_KSPLIT for flags in PRODUCTION_FLAGS)     # 0x19, 0x1B, 0x1D, 0x1F
CREDIT_SEM = 3
VALID_SEM = 4
SLICE_CAP = 4
# Compile-time argument layout the kernels read (reader: after the flags word at cb_arg_offset + 8; writer: after the cu_window CB id at cb_arg_offset + 5; compute: after the 18 CB ids).
READER_CT = ('slices', 'cb_prev_out', 'cb_lse_in', 'credit_sem', 'valid_sem')                       # cb_arg_offset + 9 ...
WRITER_CT = ('slices', 'Skt', 'cb_lse_out', 'cb_prev_out', 'cb_lse_in', 'cb_scale_in', 'credit_sem', 'valid_src_sem', 'valid_dst_sem', 'flags')   # cb_arg_offset + 6 ...
COMPUTE_CT = ('slices', 'cb_lse_in', 'cb_lse_out', 'cb_prev_out', 'cb_scale_in', 'flags')            # 52 ...
# Runtime arguments appended after the base list: reader (slice, previous core x, y), writer (slice, next core x, y), compute (slice).
FATAL_SLICES = '[QWEN-SDPA-PF] ksplit needs at least 2 slices: {} q chunks on {} cores'
FATAL_SCALE = '[QWEN-SDPA-PF] ksplit needs a scale exact in bf16'
FATAL_MODE = '[QWEN-SDPA-PF] ksplit needs fp32 dest accumulation and exact exp'
LOG_TEXT = '[QWEN-SDPA-PF] ksplit=1 slices={} q_chunks={} cores={} busy={} chains={} members={}'
NEW_QWEN_STRINGS = (FATAL_SLICES, FATAL_SCALE, FATAL_MODE, LOG_TEXT)          # the only QWEN strings this patch adds to the binary
KERNEL_DIR = 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/'
READER_PATH = KERNEL_DIR + 'dataflow/reader_interleaved_qwen_ksplit.cpp'
WRITER_PATH = KERNEL_DIR + 'dataflow/writer_interleaved_qwen_ksplit.cpp'
COMPUTE_PATH = KERNEL_DIR + 'compute/sdpa_qwen_ksplit.cpp'
MARKERS = (FATAL_SLICES, FATAL_SCALE, FATAL_MODE, LOG_TEXT, 'qwen_ks', 'kQwenPfKsplit', 'kQwenKsCreditSem')
FUNCTION_LINE = pf.FUNCTION_LINE


def lines(*parts):
    return ''.join(part + NL for part in parts)


# ---- PK0: the decode (after PS0's pair switch) ----------------------------------------------------------------------------------------------------
PK0_OLD = lines('    const bool global_q_pair_distribute = is_causal && (q_num_chunks % 2 == 0) && !qwen_one_q;')
PK0_NEW = PK0_OLD + lines(
    '    // [QWEN-SDPA-PF] ksplit (flag 0x10 on top of the oneq word 0x9): every q chunk gets S = min(4, cores / q chunks) consecutive SLICES of its K stream; cores',
    '    // [s * total, (s + 1) * total) are slice s, each laid out as the oneq split (core u of a slice = flat position u). S < 2 is refused with the envelope checks.',
    '    constexpr uint32_t kQwenKsBit = 0x10u;',
    '    const bool qwen_ks = qwen_one_q && (qwen_oq_word & kQwenKsBit) != 0;',
    '    const uint32_t qwen_ks_slices = qwen_ks ? std::min<uint32_t>(4u, num_cores / std::max<uint32_t>(total_q_chunks, 1u)) : 1u;',
    '    const uint32_t qwen_ks_cores = qwen_ks ? total_q_chunks * qwen_ks_slices : 0u;')

# ---- PK1: the constants (F1's block) --------------------------------------------------------------------------------------------------------------
PK1_OLD = lines('    constexpr uint32_t kQwenPfOneQ = 0x8u;')
PK1_NEW = PK1_OLD + lines(
    '    constexpr uint32_t kQwenPfKsplit = 0x10u, kQwenPfKsHang = 0x400u;',
    '    constexpr uint32_t kQwenKsCreditSem = %du, kQwenKsValidSem = %du;' % (CREDIT_SEM, VALID_SEM))

# ---- PK2: the accepted set; 0x10 needs 0x8, 0x400 needs 0x10 -----------------------------------------------------------------------------------------
PK2_OLD = lines('                                                    kQwenPfTestMutate | kQwenPfTestHang)) == 0,')
PK2_NEW = lines(
    '                                                    kQwenPfTestMutate | kQwenPfTestHang | kQwenPfKsplit | kQwenPfKsHang)) == 0 &&',
    '                     ((qwen_pf_flags & kQwenPfKsplit) == 0 || (qwen_pf_flags & kQwenPfOneQ) != 0) &&',
    '                     ((qwen_pf_flags & kQwenPfKsHang) == 0 || (qwen_pf_flags & kQwenPfKsplit) != 0),')

# ---- PK3: 0x400 is a test flag --------------------------------------------------------------------------------------------------------------------
PK3_OLD = lines('        TT_FATAL((qwen_pf_flags & (kQwenPfTestMutate | kQwenPfTestHang)) == 0 ||')
PK3_NEW = lines('        TT_FATAL((qwen_pf_flags & (kQwenPfTestMutate | kQwenPfTestHang | kQwenPfKsHang)) == 0 ||')

# ---- PK4: fp32 statistics under ksplit (the log-sum-exp lives in the max CBs; set after the protected draft block, which stays byte-identical) -------------
PK4_OLD = lines('    tt::DataFormat qk_im_df = fp32_dest_intermediate_dataformat(fp32_dest_acc_en);')
PK4_NEW = lines(
    '    if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit: the statistics CBs (max, exp_max_diff and the lse CBs) are fp32; the draft block above is untouched',
    '        stats_df = tt::DataFormat::Float32;',
    '    }') + PK4_OLD

# ---- PK5: the extra CBs ---------------------------------------------------------------------------------------------------------------------------
PK5_OLD = lines('    cb_ids.out = allocate_tile_cb(out0_t, out_tile_size, out_df);')
PK5_NEW = PK5_OLD + lines(
    '    uint32_t qwen_ks_cb_prev_out = 0, qwen_ks_cb_lse_in = 0, qwen_ks_cb_lse_out = 0, qwen_ks_cb_scale_in = 0;',
    '    if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit: the previous slice\'s partial lands in the first two; the third holds this slice\'s lse; the fourth the scale tile of the ring merge',
    '        qwen_ks_cb_prev_out = allocate_tile_cb(out0_t, out_tile_size, out_df);',
    '        qwen_ks_cb_lse_in = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);',
    '        qwen_ks_cb_lse_out = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);',
    '        qwen_ks_cb_scale_in = allocate_tile_cb(scale_tiles, scalar_tile_size, scalar_df);',
    '    }')

# ---- PK6: the suffix compile-time args ------------------------------------------------------------------------------------------------------------
PK6_OLD = lines(
    '    compute_compile_time_args.insert(',
    '        compute_compile_time_args.end(), compute_cb_compile_time_args.begin(), compute_cb_compile_time_args.end());')
PK6_NEW = PK6_OLD + lines(
    '    if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit suffix args: reader cb_arg_offset + 9.., writer cb_arg_offset + 6.., compute 52.. (before the output accessor args)',
    '        reader_compile_time_args.insert(reader_compile_time_args.end(),',
    '                                        {qwen_ks_slices, qwen_ks_cb_prev_out, qwen_ks_cb_lse_in, kQwenKsCreditSem, kQwenKsValidSem});',
    '        writer_compile_time_args.insert(writer_compile_time_args.end(),',
    '                                        {qwen_ks_slices, Skt, qwen_ks_cb_lse_out, qwen_ks_cb_prev_out, qwen_ks_cb_lse_in, qwen_ks_cb_scale_in,',
    '                                         kQwenKsCreditSem, valid_semaphore_id, kQwenKsValidSem, qwen_pf_flags});',
    '        compute_compile_time_args.insert(compute_compile_time_args.end(),',
    '                                         {qwen_ks_slices, qwen_ks_cb_lse_in, qwen_ks_cb_lse_out, qwen_ks_cb_prev_out, qwen_ks_cb_scale_in, qwen_pf_flags});',
    '    }')

# ---- PK7: the hand-off semaphores ------------------------------------------------------------------------------------------------------------------
PK7_OLD = lines(
    '            .initial_value = VALID,',
    '        });',
    '    }')
PK7_NEW = PK7_OLD + lines(
    '    if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit: the credit (raised on the sending slice by the receiving one) and the VALID flag (raised on the receiving slice by the sender)',
    '        desc.semaphores.push_back(SemaphoreDescriptor{',
    '            .id = kQwenKsCreditSem,',
    '            .core_type = tt::CoreType::WORKER,',
    '            .core_ranges = core_grid,',
    '            .initial_value = INVALID,',
    '        });',
    '        desc.semaphores.push_back(SemaphoreDescriptor{',
    '            .id = kQwenKsValidSem,',
    '            .core_type = tt::CoreType::WORKER,',
    '            .core_ranges = core_grid,',
    '            .initial_value = INVALID,',
    '        });',
    '    }')

# ---- PK9: the F4 range of a ksplit core and the slice in the chain key ---------------------------------------------------------------------------
PK9A_OLD = lines(
    '            if (g_start >= total_q_chunks) {',
    '                continue;',
    '            }')
PK9A_NEW = lines(
    '            if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit: slice-major, one unit per core, the oneq layout repeated per slice',
    '                g_start = (i < qwen_ks_cores) ? i % total_q_chunks : total_q_chunks;',
    '                g_count = (i < qwen_ks_cores) ? 1u : 0u;',
    '            }') + PK9A_OLD
PK9B_OLD = lines('            qwen_groups[key].push_back(i);')
PK9B_NEW = lines(
    '            if (qwen_ks) {  // the slice joins the key (4 words: key.size() / 3 below is still the 1 unit of the core)',
    '                key.push_back(i / total_q_chunks);',
    '            }') + PK9B_OLD

# ---- PK8: the envelope ----------------------------------------------------------------------------------------------------------------------------
PK8_OLD = lines(
    '                     "[QWEN-SDPA-PF] oneq needs one q chunk per core: {} q chunks on {} cores", total_q_chunks, num_cores);',
    '        }')
PK8_NEW = PK8_OLD + lines(
    '        if (qwen_ks) {',
    '            TT_FATAL(qwen_ks_slices >= 2, "%s", total_q_chunks, num_cores);' % FATAL_SLICES.replace('%', '%%'),
    '            TT_FATAL(fp32_dest_acc_en && !exp_approx_mode, "%s");' % FATAL_MODE.replace('%', '%%'),
    '            TT_FATAL((scale_packed & 0xFFFFu) == 0u, "%s");' % FATAL_SCALE.replace('%', '%%'),
    '        }')

# ---- PK10: the log line ---------------------------------------------------------------------------------------------------------------------------
PK10_OLD = lines(
    '                     total_q_chunks, num_cores, qwen_chains, qwen_members);',
    '        }')
PK10_NEW = PK10_OLD + lines(
    '        if (qwen_ks) {',
    '            log_info(tt::LogOp, "%s",' % LOG_TEXT,
    '                     qwen_ks_slices, total_q_chunks, num_cores, qwen_ks_cores, qwen_chains, qwen_members);',
    '        }')

# ---- PK11: the kernel sources -------------------------------------------------------------------------------------------------------------------
PK11_OLD = lines('    // Set reader rt args')
PK11_NEW = lines(
    '    if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit: the twins of the three kernels (make_ks_reader.py, make_ks_writer.py, make_ks_compute.py), same compile-time args plus the suffixes',
    '        reader_desc.kernel_source = "%s";' % READER_PATH,
    '        writer_desc.kernel_source = "%s";' % WRITER_PATH,
    '        compute_desc.kernel_source = "%s";' % COMPUTE_PATH,
    '    }') + PK11_OLD

# ---- PK12: the slice, previous and next core of a ksplit core --------------------------------------------------------------------------------------
PK12_OLD = lines(
    '            global_q_count = total_q_chunks - global_q_start;',
    '        }')
PK12_NEW = PK12_OLD + lines(
    '        uint32_t qwen_ks_slice = 0;',
    '        CoreCoord qwen_ks_prev{0, 0}, qwen_ks_next{0, 0};',
    '        if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit: slice-major cores; the partner of core i in the next slice is core i + total (same head and q chunk)',
    '            const bool qwen_ks_busy = i < qwen_ks_cores;',
    '            global_q_start = qwen_ks_busy ? i % total_q_chunks : total_q_chunks;',
    '            global_q_count = qwen_ks_busy ? 1u : 0u;',
    '            qwen_ks_slice = qwen_ks_busy ? i / total_q_chunks : 0u;',
    '            if (qwen_ks_busy && qwen_ks_slice > 0) {',
    '                const uint32_t p = i - total_q_chunks;',
    '                qwen_ks_prev = device->worker_core_from_logical_core(CoreCoord{p % grid_size.x, p / grid_size.x});',
    '            }',
    '            if (qwen_ks_busy && qwen_ks_slice + 1 < qwen_ks_slices) {',
    '                const uint32_t n = i + total_q_chunks;',
    '                qwen_ks_next = device->worker_core_from_logical_core(CoreCoord{n % grid_size.x, n / grid_size.x});',
    '            }',
    '        }')

# ---- PK13: the runtime args ----------------------------------------------------------------------------------------------------------------------
PK13A_OLD = lines(
    '        reader_args.push_back(global_q_start);',
    '        reader_args.push_back(global_q_count);')
PK13A_NEW = PK13A_OLD + lines(
    '        if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit: the slice and the previous slice\'s core (the credit goes there)',
    '            reader_args.push_back(qwen_ks_slice);',
    '            reader_args.push_back(static_cast<uint32_t>(qwen_ks_prev.x));',
    '            reader_args.push_back(static_cast<uint32_t>(qwen_ks_prev.y));',
    '        }')
PK13B_OLD = lines(
    '        writer_desc.emplace_runtime_args(',
    '            core,',
    '            {out0_buffer,')
PK13B_NEW = lines(
    '        if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit writer words: the served twelve, then the slice and the next slice\'s core',
    '            KernelDescriptor::RTArgList qwen_ks_writer_args;',
    '            qwen_ks_writer_args.push_back(out0_buffer);',
    '            qwen_ks_writer_args.push_back(i);',
    '            qwen_ks_writer_args.push_back(num_phases);',
    '            qwen_ks_writer_args.push_back(static_cast<uint32_t>(flexible_chunked ? 1 : 0));',
    '            qwen_ks_writer_args.push_back(chunked_q_chunk_offset);',
    '            qwen_ks_writer_args.push_back(write_offset);',
    '            qwen_ks_writer_args.push_back(0u);',
    '            qwen_ks_writer_args.push_back(0u);',
    '            qwen_ks_writer_args.push_back(global_q_start);',
    '            qwen_ks_writer_args.push_back(global_q_count);',
    '            qwen_ks_writer_args.push_back(cu_window_buffer);',
    '            qwen_ks_writer_args.push_back(cu_window_seqlens_eles);',
    '            qwen_ks_writer_args.push_back(qwen_ks_slice);',
    '            qwen_ks_writer_args.push_back(static_cast<uint32_t>(qwen_ks_next.x));',
    '            qwen_ks_writer_args.push_back(static_cast<uint32_t>(qwen_ks_next.y));',
    '            writer_desc.emplace_runtime_args(core, qwen_ks_writer_args);',
    '        } else') + PK13B_OLD
PK13C_OLD = lines(
    '        compute_desc.emplace_runtime_args(',
    '            core,',
    '            {i,')
PK13C_NEW = lines(
    '        if (qwen_ks) {  // [QWEN-SDPA-PF] ksplit compute words: the served seven, then the slice',
    '            KernelDescriptor::RTArgList qwen_ks_compute_args;',
    '            qwen_ks_compute_args.push_back(i);',
    '            qwen_ks_compute_args.push_back(num_phases);',
    '            qwen_ks_compute_args.push_back(static_cast<uint32_t>(flexible_chunked ? 1 : 0));',
    '            qwen_ks_compute_args.push_back(chunked_q_chunk_offset);',
    '            qwen_ks_compute_args.push_back(0u);',
    '            qwen_ks_compute_args.push_back(global_q_start);',
    '            qwen_ks_compute_args.push_back(global_q_count);',
    '            qwen_ks_compute_args.push_back(qwen_ks_slice);',
    '            compute_desc.emplace_runtime_args(core, qwen_ks_compute_args);',
    '        } else') + PK13C_OLD

# (label, PS line the anchor starts on, old text, new text), in file order.
EDITS = (
    ('PK0', 403, PK0_OLD, PK0_NEW),
    ('PK1', 544, PK1_OLD, PK1_NEW),
    ('PK2', 552, PK2_OLD, PK2_NEW),
    ('PK3', 555, PK3_OLD, PK3_NEW),
    ('PK4', 745, PK4_OLD, PK4_NEW),
    ('PK5', 858, PK5_OLD, PK5_NEW),
    ('PK6', 870, PK6_OLD, PK6_NEW),
    ('PK7', 893, PK7_OLD, PK7_NEW),
    ('PK8', 1394, PK8_OLD, PK8_NEW),
    ('PK9A', 1405, PK9A_OLD, PK9A_NEW),
    ('PK9B', 1422, PK9B_OLD, PK9B_NEW),
    ('PK10', 1486, PK10_OLD, PK10_NEW),
    ('PK11', 1528, PK11_OLD, PK11_NEW),
    ('PK12', 1541, PK12_OLD, PK12_NEW),
    ('PK13A', 1588, PK13A_OLD, PK13A_NEW),
    ('PK13B', 1593, PK13B_OLD, PK13B_NEW),
    ('PK13C', 1608, PK13C_OLD, PK13C_NEW),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Refusal(ValueError):
    """The input is not the PS (or PF, or served) factory, or an anchor drifted: nothing is written."""


def anchor_lines(text):
    """{label: line} where each anchor starts in `text` (the --lines mode that records them), or raises when an anchor is not unique."""
    found = {}
    for label, _line, old, _new in EDITS:
        count = text.count(old)
        if count != 1:
            raise Refusal('anchor %s occurs %d times (need exactly 1)' % (label, count))
        found[label] = text[:text.index(old)].count(NL) + 1
    return found


def check_anchors(text):
    for label, line, old, _new in EDITS:
        count = text.count(old)
        if count != 1:
            raise Refusal('anchor %s occurs %d times (need exactly 1)' % (label, count))
        found = text[:text.index(old)].count(NL) + 1
        if found != line:
            raise Refusal('anchor %s starts at line %d, recorded PS:%d; refusing on drift' % (label, found, line))


def to_ps(source):
    """The PS factory bytes from `source` (the PS factory itself, the PF factory, or the served fd8c0676 with PF and PS applied)."""
    digest = sha256(source)
    if digest == PS_FACTORY:
        return source
    try:
        return ps.patch(source)
    except ps.Refusal as error:
        raise Refusal('PS edits refused: %s' % error)


def patch(source):
    """The PK factory from the PS (or PF, or served) bytes; Refusal on any other input or on drift."""
    base = to_ps(source)
    text = base.decode('utf-8')
    check_anchors(text)
    for _label, _line, old, new in EDITS:
        text = text.replace(old, new)
    for marker in MARKERS:
        if marker not in text:
            raise Refusal('patched factory lacks %r' % marker)
    if pf.protected_block(text) != pf.protected_block(base.decode('utf-8')):
        raise Refusal('the qwen_draft_fp32_intermediates block moved')
    body = text.index(FUNCTION_LINE)
    for label, _line, _old, new in EDITS:
        if text.index(new) < body:
            raise Refusal('edit %s lies outside create_descriptor' % label)
    out = text.encode('utf-8')
    if chr(13).encode() in out:
        raise Refusal('output has a CR; files must be LF')
    return out


def unpatch(patched):
    """The inverse, last edit first: proves the output is the PS factory plus exactly these edits."""
    text = patched.decode('utf-8')
    for label, _line, old, new in reversed(EDITS):
        if text.count(new) != 1:
            raise Refusal('edit %s occurs %d times' % (label, text.count(new)))
        text = text.replace(new, old)
    return text.encode('utf-8')


KNOWN_FLAGS = ps.KNOWN_FLAGS | FLAG_KSPLIT | FLAG_KS_HANG
TEST_FLAGS = pf.TEST_FLAGS | FLAG_KS_HANG


def decode_word(word, test_env=None):
    """F1 plus PS2 plus PK2/PK3 in Python: (chain, flags, oneq, ksplit) for SDPAProgramConfig.max_cores_per_head_batch = word, or ValueError with the TT_FATAL text F1 raises.
    test_env is the value of QWEN_SDPA_PF_TEST."""
    word &= 0xFFFFFFFF
    if (word & pf.PF_TAG_MASK) != PF_TAG:
        return False, 0, False, False
    flags = word & ~pf.PF_TAG_MASK & 0xFFFFFFFF
    if not flags & pf.FLAG_KV_CHAIN or flags & ~KNOWN_FLAGS:
        raise ValueError('%s %#x' % (pf.FLAGS_MARKER, flags))
    if flags & FLAG_KSPLIT and not flags & ps.FLAG_ONEQ:
        raise ValueError('%s %#x' % (pf.FLAGS_MARKER, flags))
    if flags & FLAG_KS_HANG and not flags & FLAG_KSPLIT:
        raise ValueError('%s %#x' % (pf.FLAGS_MARKER, flags))
    if flags & TEST_FLAGS and test_env != '1':
        raise ValueError('%s %#x need %s=1' % (pf.TEST_MARKER, flags, pf.TEST_ENV))
    return True, flags, bool(flags & ps.FLAG_ONEQ), bool(flags & FLAG_KSPLIT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('factory')
    parser.add_argument('--out', help='write here instead of patching in place')
    parser.add_argument('--record', action='store_true', help='print the output sha instead of enforcing it')
    parser.add_argument('--lines', action='store_true', help='print where each anchor starts in the PS factory and exit')
    args = parser.parse_args(argv)
    path = Path(args.factory)
    source = path.read_bytes()
    if args.lines:
        try:
            found = anchor_lines(to_ps(source).decode('utf-8'))
        except Refusal as error:
            print('refused: %s' % error)
            return 1
        for label, line, _old, _new in EDITS:
            print('%-6s recorded %-5s found %d' % (label, line, found[label]))
        return 0
    if PK_FACTORY != '__RECORD__' and sha256(source) == PK_FACTORY:
        print('factory already [QWEN-SDPA-PF]-ksplit-patched %s' % PK_FACTORY)
        if args.out:
            Path(args.out).write_bytes(source)
        return 0
    try:
        patched = patch(source)
    except Refusal as error:
        print('refused: %s' % error)
        return 1
    digest = sha256(patched)
    if not args.record and digest != PK_FACTORY:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, PK_FACTORY))
        return 1
    if unpatch(patched) != to_ps(source):
        print('edits do not invert')
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
    print('factory %s -> [QWEN-SDPA-PF] ksplit %s written %s' % (sha256(source)[:16], digest, target))
    print(digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
