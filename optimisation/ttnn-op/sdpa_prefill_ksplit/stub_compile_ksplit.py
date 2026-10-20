"""g++ -fsyntax-only of the [QWEN-SDPA-PF] ksplit C++ against syntax stubs (no tt-metal tree needed).

Three checks, all -Wall -Wextra -Werror (plus -Wno-unused-parameter, which the stubs' empty declarations would trip):

  decision   the three regions of the patched factory that DECIDE (the work split with PK0, F1 with PK1-PK3, F4 with PK8-PK10), as stub_compile_oneq.py compiles the oneq ones;
             host_exec_ksplit.py then RUNS them. The same regions in a harness that also takes the RT loop's PK12 block.
  plumbing   the regions that only plumb: PK4 (fp32 statistics), PK5 (the four CBs), PK6 (the suffix compile-time args), PK7 (the two semaphores) and the kernel / runtime-argument
             section PK11-PK13 around its unchanged neighbours, each in a stub function whose locals carry the factory's own declared types. The descriptor stubs (KernelDescriptor, its
             RTArgList, SemaphoreDescriptor) are written from the call sites of the served factory: an assumption the ttbuild ninja run is the proof of.
  kernels    the ksplit reader and writer against the real dataflow_common.hpp / q_chunk_remapping.hpp / sliding_window_geometry.hpp of the pinned tt-metal (a staged "probe" tree,
             QWEN_SDPA_PREFILL_SRC; skipped without it) and the API stubs of ../sdpa_prefill_chain/stubcheck plus stubcheck/include (the writer's four extra headers). The pinned reader and
             writer must compile first (they prove the stubs cover the surface); the ksplit twins must then compile with no warning beyond theirs. The COMPUTE kernel has no stub
             check: tt-metal's compute API is not in this repository.

It is evidence, not proof: the stubs are declarations. The JIT compile and the ttbuild ninja run are the proofs.

    python3 -B stub_compile_ksplit.py [--gxx g++] [--probe DIR]        # QWEN_PF_GXX, QWEN_SDPA_PREFILL_SRC override
    python3 -B stub_compile_ksplit.py --dump-plumbing
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ONEQ = HERE.parent / 'sdpa_prefill_oneq'
CHAIN = HERE.parent / 'sdpa_prefill_chain'
sys.path.insert(0, str(ONEQ))
sys.path.insert(0, str(CHAIN / 'stubcheck'))
sys.path.insert(0, str(HERE))

import apply_factory_pk as pk  # noqa: E402
import stub_compile  # noqa: E402
import stub_compile_oneq as oneq  # noqa: E402

NL = chr(10)
FLAGS = oneq.FLAGS

DECISION_SIGNATURE = oneq.SIGNATURE.replace('const uint32_t qk_in0_num_subblocks) {',
                                            'const uint32_t qk_in0_num_subblocks, const uint32_t scale_packed, const bool fp32_dest_acc_en,'
                                            ' const bool exp_approx_mode) {')

PLUMBING_PRELUDE = r'''
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

namespace tt {
enum class DataFormat { Bfp8_b, Float16_b, Float32, Int32 };
enum class CoreType { WORKER };
uint32_t tile_size(DataFormat format);
namespace tt_metal {
struct Buffer {};
}  // namespace tt_metal
}  // namespace tt
struct CoreCoord {
    std::size_t x = 0;
    std::size_t y = 0;
};
struct IDevice {
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical) const;
};
constexpr uint32_t INVALID = 0;
constexpr uint32_t VALID = 1;
struct SemaphoreDescriptor {
    uint32_t id = 0;
    tt::CoreType core_type = tt::CoreType::WORKER;
    int core_ranges = 0;
    uint32_t initial_value = 0;
};
struct RTArg {
    RTArg(uint32_t);
    RTArg(tt::tt_metal::Buffer*);
};
struct KernelDescriptor {
    struct RTArgList {
        RTArgList() = default;
        RTArgList(std::initializer_list<RTArg>);
        void push_back(RTArg);
    };
    const char* kernel_source = nullptr;
    void emplace_runtime_args(CoreCoord core, RTArgList args);
};
struct StubProgramDescriptor {
    std::vector<SemaphoreDescriptor> semaphores;
};
'''

# One function carrying the locals the plumbing regions read, in the types the factory declares them.
PLUMBING_LOCALS = r'''
uint32_t stub_plumbing(IDevice* device, const bool qwen_ks, const uint32_t qwen_ks_slices, const uint32_t qwen_ks_cores, const uint32_t total_q_chunks,
                       const uint32_t Skt, const uint32_t qwen_pf_flags, uint32_t valid_semaphore_id, CoreCoord grid_size, const uint32_t num_cores,
                       const bool flexible_chunked, const uint32_t out0_t, const uint32_t out_tile_size, const tt::DataFormat out_df,
                       const uint32_t statistics_tiles, const uint32_t scale_tiles, const uint32_t scalar_tile_size, const tt::DataFormat scalar_df,
                       tt::tt_metal::Buffer* out0_buffer, tt::tt_metal::Buffer* cu_window_buffer, const uint32_t cu_window_seqlens_eles) {
    constexpr uint32_t kQwenKsCreditSem = 3u, kQwenKsValidSem = 4u;
    int core_grid = 0;
    StubProgramDescriptor desc;
    uint32_t next_cb_index = 0;
    const auto allocate_tile_cb = [&](uint32_t num_tiles, uint32_t tile_size, tt::DataFormat data_format) -> uint32_t {
        (void)num_tiles;
        (void)tile_size;
        (void)data_format;
        return next_cb_index++;
    };
    struct {
        uint32_t out = 0;
    } cb_ids;
    const bool fp32_dest_acc_en = true;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;
    const auto fp32_dest_intermediate_dataformat = [](bool fp32) { return fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b; };
'''
PLUMBING_MIDDLE = r'''
    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args, compute_compile_time_args;
    const std::vector<uint32_t> compute_cb_compile_time_args(18, 0u);
'''
PLUMBING_KERNELS = r'''
    KernelDescriptor reader_desc, writer_desc, compute_desc;
    uint32_t num_phases = 1, chunked_q_chunk_offset = 0, read_offset = 0, write_offset = 0;
    tt::tt_metal::Buffer *q_buffer = nullptr, *k_buffer = nullptr, *v_buffer = nullptr, *mask_buffer = nullptr, *page_table_buffer = nullptr,
                         *attention_sink_buffer = nullptr, *chunk_start_idx_buffer = nullptr;
    const bool is_causal = true, qwen_kv_chain = true;
    struct ChainInfo {
        bool participates = false, is_injector = false, is_sink = false;
        uint32_t batch = 0, head = 0, q_chunk_start = 0, q_chunk_count = 0, next_core_q_chunks = 0, mcast_num_dests = 0, mcast_sender_wait = 0;
        CoreCoord prev_physical, next_physical;
    };
    std::vector<ChainInfo> core_chain_info(num_cores);
    uint32_t global_q_base_chunks_per_core = 0, global_q_cores_doing_extra = 0, global_q_extra_chunks_per_core = 1;
'''
PLUMBING_RT_OPEN = r'''
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        uint32_t global_q_start = i * global_q_base_chunks_per_core + std::min(i, global_q_cores_doing_extra) * global_q_extra_chunks_per_core;
        uint32_t global_q_count = global_q_base_chunks_per_core + ((i < global_q_cores_doing_extra) ? global_q_extra_chunks_per_core : 0u);
        if (global_q_start >= total_q_chunks) {
            global_q_start = total_q_chunks;
            global_q_count = 0;
        } else if (global_q_start + global_q_count > total_q_chunks) {
'''
PLUMBING_RT_ARGS = r'''
        const auto& chain = core_chain_info[i];
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(q_buffer);
        reader_args.push_back(k_buffer);
        reader_args.push_back(v_buffer);
        reader_args.push_back(mask_buffer);
        reader_args.push_back(page_table_buffer);
        reader_args.push_back(attention_sink_buffer);
        reader_args.push_back(chunk_start_idx_buffer);
        reader_args.push_back(i);
        reader_args.push_back(num_phases);
        reader_args.push_back(chunked_q_chunk_offset);
        reader_args.push_back(read_offset);
        if (qwen_kv_chain) {
            reader_args.push_back(static_cast<uint32_t>(chain.participates));
        }
'''
PLUMBING_CLOSE = r'''
    }
    return next_cb_index + static_cast<uint32_t>(desc.semaphores.size()) + static_cast<uint32_t>(reader_compile_time_args.size() + writer_compile_time_args.size()
           + compute_compile_time_args.size()) + (reader_desc.kernel_source != nullptr) + (is_causal ? 1u : 0u) + stats_tile_size + (valid_semaphore_id > 0);
}
'''


def decision_harness(text=None):
    """The decision regions of the patched factory in a stub create_descriptor (PK0-PK3, PK8-PK10)."""
    import host_exec_ksplit

    text = host_exec_ksplit.patched_text() if text is None else text
    prelude = stub_compile.FACTORY_PRELUDE.split('uint32_t stub_create_descriptor(')[0]
    split = oneq.cut(text, oneq.SPLIT_BEGIN, oneq.SPLIT_END, 'work split') + oneq.SPLIT_END
    f1 = oneq.cut(text, oneq.F1_BEGIN, oneq.F1_END, 'F1')
    f4 = oneq.cut(text, oneq.F4_BEGIN, oneq.F4_END, 'F4')
    return prelude + DECISION_SIGNATURE + split + f1 + oneq.BETWEEN_F1_F4.replace('(void)use_zigzag_balancing;', '(void)use_zigzag_balancing;' + NL + '    (void)kQwenKsCreditSem;' + NL + '    (void)kQwenKsValidSem;') + f4 + oneq.CLOSE


def region(text, new, label):
    if text.count(new) != 1:
        raise ValueError('%s: the region text is not unique in the patched factory' % label)
    return new


def plumbing_harness(text=None):
    """PK4, PK5, PK6, PK7 and the kernel / runtime-argument section PK11-PK13 in a stub function."""
    import host_exec_ksplit

    text = host_exec_ksplit.patched_text() if text is None else text
    pieces = [region(text, getattr(pk, '%s_NEW' % name), name) for name in ('PK4', 'PK5', 'PK6', 'PK7')]
    pk4, pk5, pk6, pk7 = pieces
    pk7 = pk7[len(pk.PK7_OLD):]          # the new text starts with the served block's closing lines (the anchor): only the addition is the lever's
    # the kernel / rt section: the new text of PK11, then the loop open, PK12, the reader words (PK13A), the writer / compute statements with their served neighbours
    pk11 = region(text, pk.PK11_NEW, 'PK11')
    pk12 = region(text, pk.PK12_NEW, 'PK12')
    pk13a = region(text, pk.PK13A_NEW, 'PK13A')
    start = text.index(pk.PK13B_NEW)
    stop = text.index('    desc.kernels.push_back(std::move(reader_desc));', start)
    rt_tail = text[start:stop]
    rt_tail = rt_tail.rstrip(NL)
    # the served tail ends with the compute statement of the loop and the loop's closing brace
    if not rt_tail.rstrip().endswith('}'):
        raise ValueError('the runtime-argument tail does not end with the loop brace')
    rt_tail = rt_tail.rstrip()[:-1]    # the stub's PLUMBING_CLOSE closes the loop
    return (PLUMBING_PRELUDE + PLUMBING_LOCALS + pk4 + '    (void)qk_im_df;' + NL + '    uint32_t stats_tile_size = tt::tile_size(stats_df);' + NL + pk5 + PLUMBING_MIDDLE + pk6 + pk7 + PLUMBING_KERNELS
            + pk11.replace('    // Set reader rt args' + NL, '') + PLUMBING_RT_OPEN + pk12 + PLUMBING_RT_ARGS + pk13a.split('        reader_args.push_back(global_q_count);' + NL)[1]
            + rt_tail + PLUMBING_CLOSE)


def compile_text(gxx, source_text, name='blocks.cpp', flags=None, include_dirs=()):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / name
        source.write_text(source_text, encoding='utf-8', newline=NL)
        command = [gxx] + (flags or FLAGS) + ['-I' + str(path) for path in include_dirs] + [str(source)]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return result.returncode, (result.stdout + result.stderr).strip()


# ---- the kernels ---------------------------------------------------------------------------------------------------------------------------------

READER_CT_TAIL = (4, 20, 21, pk.CREDIT_SEM, pk.VALID_SEM)                      # slices, cb_prev_out, cb_lse_in, credit, valid
WRITER_BASE = (1, 6, 1, 32, 32, 1152, 8, 8, 4, 8, 4, 1024, 0x3F803F80, 0x3D800000, 130, 1, 0, 0, 1, 0, 1, 0, 2, 0, 1, 0)   # the served writer's 26 fixed args at 1,024 rows
WRITER_ACCESSORS = (0, 0)                                                        # out, cu_window
WRITER_CBS = (3, 4, 5, 7, 8, 0)                                                  # mask, identity scale, col identity, chunk start, out, cu_window
WRITER_CT_TAIL = (2, 544, 21, 20, 19, 22, pk.CREDIT_SEM, 2, pk.VALID_SEM)       # slices, Skt, cb_lse_out, cb_prev_out, cb_lse_in, cb_scale_in, credit, valid src, valid dst (+ flags)
WARNING = stub_compile.WARNING


def reader_ct(flags):
    return stub_compile.ct_vector(flags) + list(READER_CT_TAIL)


def writer_ct(flags=None):
    values = list(WRITER_BASE) + list(WRITER_ACCESSORS) + list(WRITER_CBS)
    if flags is not None:
        values += list(WRITER_CT_TAIL) + [flags]
    return values


def stage_writer(root, probe, text):
    return stub_compile.stage_reader_tree(root, probe, text, 'writer_interleaved.cpp')


def check_kernels(gxx, probe, flags=0x1B):
    """[(name, returncode, output)] for the pinned reader, the ksplit reader, the pinned writer, the ksplit writer."""
    probe = Path(probe)
    results = []
    include = [HERE / 'stubcheck' / 'include', CHAIN / 'stubcheck' / 'include']
    cases = (
        ('reader_interleaved.cpp', (probe / 'kernels' / 'dataflow' / 'reader_interleaved.cpp').read_text(encoding='utf-8'), stub_compile.ct_vector(flags & 0x7)),
        ('reader_interleaved_qwen_ksplit.cpp', (HERE / 'reader_interleaved_qwen_ksplit.cpp').read_text(encoding='utf-8'), reader_ct(flags)),
        ('writer_interleaved.cpp', (probe / 'kernels' / 'dataflow' / 'writer_interleaved.cpp').read_text(encoding='utf-8'), writer_ct()),
        ('writer_interleaved_qwen_ksplit.cpp', (HERE / 'writer_interleaved_qwen_ksplit.cpp').read_text(encoding='utf-8'), writer_ct(flags)),
    )
    for name, text, ct in cases:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            placed = 'writer_interleaved.cpp' if name.startswith('writer') else name
            source = stub_compile.stage_reader_tree(root, probe, text, placed)
            code, output = stub_compile.run(gxx, source, include + [root], [('STUB_CT_ARGS', ','.join(str(value) for value in ct))], stub_compile.READER_FLAGS)
        results.append((name, code, output))
    return results


def warning_sets(results):
    """{name: {(flag, variable)}} of the unused-variable style warnings."""
    return {name: stub_compile.warning_set(output) for name, _code, output in results}


def kernel_verdict(results):
    """(ok, {twin: warnings the twin adds beyond its pinned original}): all four compile and no twin adds a warning."""
    sets = warning_sets(results)
    codes = {name: code for name, code, _output in results}
    extra = {'reader_interleaved_qwen_ksplit.cpp': sets['reader_interleaved_qwen_ksplit.cpp'] - sets['reader_interleaved.cpp'],
             'writer_interleaved_qwen_ksplit.cpp': sets['writer_interleaved_qwen_ksplit.cpp'] - sets['writer_interleaved.cpp']}
    return all(code == 0 for code in codes.values()) and not any(extra.values()), extra


def find_gxx(explicit=None):
    return oneq.find_gxx(explicit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('--gxx')
    parser.add_argument('--probe', default=os.environ.get('QWEN_SDPA_PREFILL_SRC', stub_compile.DEFAULT_PROBE))
    parser.add_argument('--dump-plumbing', action='store_true')
    options = parser.parse_args(argv)
    if options.dump_plumbing:
        sys.stdout.write(plumbing_harness())
        return 0
    gxx = find_gxx(options.gxx)
    if gxx is None:
        print('no g++ found (set QWEN_PF_GXX)', file=sys.stderr)
        return 2
    print('compiler %s' % gxx)
    status = 0
    for label, text in (('decision regions (PK0-PK3, PK8-PK10)', decision_harness()), ('plumbing regions (PK4-PK7, PK11-PK13)', plumbing_harness())):
        code, output = compile_text(gxx, text)
        print('%s: %s' % (label, 'ok' if code == 0 else 'FAILED'))
        if output:
            print(output)
        status |= code != 0
    probe = Path(options.probe)
    if not (probe / 'kernels' / 'dataflow' / 'dataflow_common.hpp').is_file():
        print('kernels: skipped (no probe tree at %s)' % probe)
        return int(status)
    results = check_kernels(gxx, probe)
    for name, code, output in results:
        print('%s: %s' % (name, 'compiles' if code == 0 else 'FAILED'))
        if code != 0:
            print(output)
    ok, extra = kernel_verdict(results)
    print('ksplit kernels: %s' % ('ok, no warning beyond the pinned ones' if ok else 'FAILED, new warnings %r' % extra))
    status |= not ok
    return int(status)


if __name__ == '__main__':
    sys.exit(main())
