"""g++ -fsyntax-only of the [QWEN-SDPA-PF] C++ against syntax stubs (no tt-metal tree needed).

Two checks, both with -Wall -Wextra -Werror (plus -Wno-unused-parameter, which the stubs' empty
declarations would otherwise trip):

  reader    the served reader_interleaved.cpp (f97f5490) AND the chain reader, each against the real
            probe-v25 dataflow_common.hpp / q_chunk_remapping.hpp / sliding_window_geometry.hpp and the
            API stubs in include/ (stub_tt_api.hpp, shaped after the call sites). The served reader
            passing proves the stubs cover the surface; the chain reader must then pass too, with the
            served shape's CT vector (causal, flexible chunked, 12/2 heads, q/k chunk 4 tiles, ...,
            the flags suffix at cb_arg_offset + 8). Any warning (an orphaned local, a narrowing) fails.
  factory   every [QWEN-SDPA-PF] block of apply_factory_pf.py (F1-F7) compiled inside a stub
            create_descriptor whose locals carry the factory's own declared types, so the new code's
            names, types, lambdas, structured bindings and format strings are checked.

It is evidence, not proof: the stubs are declarations written from the call sites. The JIT compile
on card M (spec Q-12) and the ttbuild ninja build are the proofs.

    py -3.11 stubcheck/stub_compile.py                       # both, with the default compiler
    py -3.11 stubcheck/stub_compile.py --gxx g++ --flags 0x3 # another compiler / flags word
Env: QWEN_PF_GXX (compiler), QWEN_SDPA_PREFILL_SRC (probe-v25 src/device).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
sys.path.insert(0, str(CHAIN))

import apply_factory_pf as factory  # noqa: E402

LOCAL_ARM_GXX = 'C:/Users/liamb/AppData/Local/stm32cube/bundles/gnu-tools-for-stm32/13.3.1+st.9/bin/arm-none-eabi-g++.exe'
DEFAULT_PROBE = 'C:/Users/liamb/.claude/jobs/8376c877/tmp/probe-v25/src/device'
FLAGS = ['-std=c++20', '-fsyntax-only', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter']
# The served reader itself has four unused locals (Sqt, k_num_chunks, core_id, q_chunk_tiles) and
# compiles in the JIT, so unused-variable warnings are not fatal there; the readers are compiled
# with them as warnings and the chain reader's warning set must be a subset of the served one's.
READER_FLAGS = FLAGS + ['-Wno-error=unused-variable', '-Wno-error=unused-but-set-variable']

# The served shape's reader CT vector (factory fd8c0676, rows 2048, q/k chunk 128, 12 Q heads,
# 2 KV heads, head dim 256, 64-token pages, page table 2016 blocks, bf8, fp32 dest on). Accessor
# blocks take one slot each in the stubs (TensorAccessorArgs<Base>::next = Base + 1).
CT_NAMES = ('B', 'NQH', 'NKH', 'NVH', 'Sqt', 'Skt', 'valid_Sqt', 'valid_Skt', 'DHt', 'vDHt', 'Sq_chunk_t',
            'q_num_chunks', 'Sk_chunk_t', 'k_num_chunks', 'num_cores', 'is_causal', 'use_provided_mask',
            'broadcast_mask_batch', 'broadcast_mask_heads', 'use_padded_mask', 'is_chunked', 'block_size_t',
            'page_table_stick_size', 'use_attention_sink', 'use_mla', 'mla_kv_overlap', 'qk_subblock_h',
            'sliding_window_size', 'use_streaming_compute', 'sender_sem', 'receiver_sem', 'valid_sem',
            'mcast_enabled', 'zigzag')
CT_VALUES = (1, 12, 2, 2, 64, 4096, 64, 4096, 8, 8, 4, 16, 4, 1024, 110, 1, 0, 0, 0, 0, 1, 2, 8064, 0, 0, 0, 2, 0,
             0, 0, 1, 2, 0, 1)
ACCESSORS = 7            # q, k, v, mask, page_table, attention_sink, chunk_start_idx
CB_IDS = (0, 1, 2, 3, 0xFFFFFFFF, 6, 7, 8)   # q, k, v, mask, sink (inactive), page table, chunk start x2


def ct_vector(flags, overrides=None):
    """The reader CT vector; overrides {CT name: value} replaces named fixed args (tests: a shape the
    chain must refuse at compile time, e.g. qk_subblock_h = Sq_chunk_t)."""
    values = list(CT_VALUES)
    for name, value in (overrides or {}).items():
        values[CT_NAMES.index(name)] = value
    return values + [0] * ACCESSORS + list(CB_IDS) + [flags]


def find_gxx(explicit=None):
    for candidate in (explicit, os.environ.get('QWEN_PF_GXX'), shutil.which('g++'), LOCAL_ARM_GXX):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate
    return None


def run(gxx, source, include_dirs, defines, flags=None):
    command = [gxx] + (flags or FLAGS) + ['-I' + str(path) for path in include_dirs] + ['-D%s=%s' % pair for pair in defines]
    command.append(str(source))
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return result.returncode, (result.stdout + result.stderr).strip()


def stage_reader_tree(root, probe, reader_text, name):
    """<root>/kernels/dataflow/<name> next to the real dataflow_common.hpp, and the two real
    headers it includes under their tree paths."""
    kernels = Path(probe) / 'kernels'
    dataflow = root / 'kernels' / 'dataflow'
    dataflow.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(kernels / 'dataflow' / 'dataflow_common.hpp', dataflow / 'dataflow_common.hpp')
    tree = root / 'cpp' / 'ttnn' / 'operations' / 'transformer' / 'sdpa' / 'device' / 'kernels'
    tree.mkdir(parents=True, exist_ok=True)
    for header in ('q_chunk_remapping.hpp', 'sliding_window_geometry.hpp'):
        shutil.copyfile(kernels / header, tree / header)
    target = dataflow / name
    target.write_text(reader_text, encoding='utf-8', newline=chr(10))
    return target


def check_readers(gxx, probe, chain_path, flags, overrides=None):
    """[(name, returncode, output)] for the served and the chain reader."""
    served = (Path(probe) / 'kernels' / 'dataflow' / 'reader_interleaved.cpp').read_text(encoding='utf-8')
    chain = Path(chain_path).read_text(encoding='utf-8')
    defines = [('STUB_CT_ARGS', ','.join(str(value) for value in ct_vector(flags, overrides)))]
    results = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, text in (('reader_interleaved.cpp', served), ('reader_interleaved_qwen_chain.cpp', chain)):
            source = stage_reader_tree(root, probe, text, name)
            code, output = run(gxx, source, [HERE / 'include', root], defines, READER_FLAGS)
            results.append((name, code, output))
    return results


WARNING = re.compile(r"warning: (?:unused variable|variable) '([A-Za-z_0-9]+)'.*\[(-W[a-z-]+)\]")


def warning_set(output):
    """{(flag, name)} of the unused-variable style warnings in a g++ output."""
    return {(match.group(2), match.group(1)) for match in WARNING.finditer(output)}


def reader_verdict(results):
    """(ok, served warnings, chain-only warnings): both compile and the chain reader adds no warning."""
    (served_name, served_code, served_out), (chain_name, chain_code, chain_out) = results
    served, chain = warning_set(served_out), warning_set(chain_out)
    return served_code == 0 and chain_code == 0 and not (chain - served), served, chain - served


# --- the factory blocks -------------------------------------------------------------------------

FACTORY_PRELUDE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

// Stubs for the host names the [QWEN-SDPA-PF] blocks touch (types as the factory declares them).
struct CoreCoord {
    std::size_t x = 0;
    std::size_t y = 0;
};
struct IDevice {
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical) const;
};
namespace tt {
enum class DataFormat { Bfp8_b, Float16_b, Float32, Int32 };
enum LogType { LogOp };
}  // namespace tt
template <typename... Args>
void log_info(tt::LogType, const char* format, Args&&... args);
template <typename... Args>
void stub_fatal(bool condition, const char* format, Args&&... args);
#define TT_FATAL(condition, ...) stub_fatal(static_cast<bool>(condition), __VA_ARGS__)
struct SDPAProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t q_chunk_size = 0;
    std::size_t k_chunk_size = 0;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;  // sdpa_config.hpp (decode-sources dump, line 20)
};
constexpr uint32_t INVALID = 0;
constexpr uint32_t VALID = 1;
struct SemaphoreDescriptor {
    uint32_t id = 0;
    int core_type = 0;
    int core_ranges = 0;
    uint32_t initial_value = 0;
};
struct StubDescriptor {
    std::vector<SemaphoreDescriptor> semaphores;
};
struct StubKernelDescriptor {
    const char* kernel_source = nullptr;
};
// CoreChainInfo exactly as sdpa_program_factory.cpp fd8c0676 declares it (lines 49-63).
struct CoreChainInfo {
    bool participates = false;
    bool is_injector = false;
    bool is_sink = false;
    uint32_t batch = 0;
    uint32_t head = 0;
    uint32_t q_chunk_start = 0;
    uint32_t q_chunk_count = 0;
    CoreCoord prev_physical = CoreCoord{0, 0};
    CoreCoord next_physical = CoreCoord{0, 0};
    uint32_t next_core_q_chunks = 0;
    bool use_mcast = false;
    uint32_t mcast_num_dests = 0;
    uint32_t mcast_sender_wait = 0;
};

uint32_t stub_create_descriptor(const std::optional<SDPAProgramConfig>& program_config, IDevice* device,
                                const bool is_causal, const bool flexible_chunked, const bool has_sliding_window,
                                const bool use_provided_mask, const bool is_windowed, bool use_attention_sink,
                                const bool use_mla, const bool use_streaming_compute, const bool lightweight_causal,
                                const uint32_t B, const uint32_t NQH, const uint32_t NKH, const uint32_t NVH,
                                const uint32_t Sq_chunk_t, const uint32_t Sk_chunk_t, const uint32_t DHt,
                                const uint32_t vDHt, const uint32_t q_num_chunks, CoreCoord grid_size,
                                uint32_t num_cores, const uint32_t total_q_chunks,
                                const bool global_q_pair_distribute, uint32_t global_q_base_chunks_per_core,
                                uint32_t global_q_cores_doing_extra, uint32_t global_q_extra_chunks_per_core,
                                tt::DataFormat k_df, tt::DataFormat v_df, const uint32_t qk_in0_num_subblocks) {
'''

FACTORY_MIDDLE = r'''
    std::vector<uint32_t> reader_compile_time_args(34, 0u);
    const std::vector<uint32_t> reader_cb_compile_time_args(8, 0u);
    const auto sem_args_offset = 29u;
    uint32_t sender_semaphore_id = 0;
    uint32_t receiver_semaphore_id = 0;
    uint32_t valid_semaphore_id = 0;
    (void)use_zigzag_balancing;
'''

FACTORY_TAIL = r'''
    StubDescriptor desc;
    uint32_t num_phases = 1;
    std::vector<CoreChainInfo> core_chain_info(num_cores);
    uint32_t mcast_chains = 0;
'''

FACTORY_END = r'''
    reader_compile_time_args[sem_args_offset + 3] = (mcast_chains > 0) ? 1 : 0;
    StubKernelDescriptor reader_desc;
'''

FACTORY_RT = r'''
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& chain = core_chain_info[i];
        std::vector<uint32_t> reader_args;
'''

FACTORY_CLOSE = r'''
            reader_args.push_back(static_cast<uint32_t>(chain.participates));
        }
    }
    return static_cast<uint32_t>(reader_compile_time_args.size() + desc.semaphores.size()) + sender_semaphore_id +
           receiver_semaphore_id + valid_semaphore_id + (reader_desc.kernel_source != nullptr);
}
'''


def new_text(label):
    for name, _line, _old, new in factory.EDITS:
        if name == label:
            return new
    raise KeyError(label)


def factory_harness():
    """The F1-F7 new texts in file order inside a stub create_descriptor. F2/F3/F5 are the opening
    lines of blocks whose served bodies are elided, so each is closed right after its first line."""
    f2 = new_text('F2') + '        receiver_semaphore_id = 1;' + chr(10) + '        valid_semaphore_id = 2;' + chr(10) + \
        '        reader_compile_time_args[sem_args_offset + 0] = sender_semaphore_id;' + chr(10) + '    }' + chr(10)
    f3 = new_text('F3') + '            .id = sender_semaphore_id, .core_type = 0, .core_ranges = 0, .initial_value = INVALID});' + \
        chr(10) + '    }' + chr(10)
    f6 = new_text('F6')
    f5 = new_text('F5')
    return (FACTORY_PRELUDE + new_text('F1') + FACTORY_MIDDLE + f2 + new_text('F7') + FACTORY_TAIL + f3
            + new_text('F4') + FACTORY_END + f6 + FACTORY_RT + f5 + FACTORY_CLOSE)


def check_factory(gxx):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'factory_blocks.cpp'
        source.write_text(factory_harness(), encoding='utf-8', newline=chr(10))
        code, output = run(gxx, source, [], [])
    return code, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--gxx', help='compiler (default: QWEN_PF_GXX, g++ on PATH, the local arm-none-eabi-g++)')
    parser.add_argument('--probe', default=os.environ.get('QWEN_SDPA_PREFILL_SRC', DEFAULT_PROBE),
                        help='probe-v25 src/device (the served reader and dataflow headers)')
    parser.add_argument('--chain', default=str(CHAIN / 'reader_interleaved_qwen_chain.cpp'))
    parser.add_argument('--flags', default='0x1', help='the flags word in the CT vector (0x1, 0x3, 0x303, ...)')
    parser.add_argument('--dump-factory', action='store_true', help='print the factory harness and exit')
    options = parser.parse_args(argv)
    if options.dump_factory:
        sys.stdout.write(factory_harness())
        return 0
    gxx = find_gxx(options.gxx)
    if gxx is None:
        print('no g++ found (set QWEN_PF_GXX)', file=sys.stderr)
        return 2
    print('compiler %s' % gxx)
    status = 0
    code, output = check_factory(gxx)
    print('factory blocks F1-F7: %s' % ('ok' if code == 0 else 'FAILED'))
    if output:
        print(output)
    status |= code != 0
    probe = Path(options.probe)
    if not (probe / 'kernels' / 'dataflow' / 'dataflow_common.hpp').is_file():
        print('readers: skipped (no probe tree at %s)' % probe)
        return int(status)
    results = check_readers(gxx, probe, options.chain, int(options.flags, 0))
    for name, code, output in results:
        print('%s (flags %s): %s' % (name, options.flags, 'compiles' if code == 0 else 'FAILED'))
        if code != 0:
            print(output)
    ok, served, extra = reader_verdict(results)
    print('served reader warnings (JIT-tolerated): %s' % ', '.join('%s %s' % pair for pair in sorted(served)))
    print('chain reader: %s' % ('ok, no warning beyond the served ones' if ok else 'FAILED, new warnings %r' % sorted(extra)))
    status |= not ok
    return int(status)


if __name__ == '__main__':
    raise SystemExit(main())
