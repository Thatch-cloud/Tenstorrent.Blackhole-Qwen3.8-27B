"""Run the REAL work-split and chain-grouping C++ of the patched prefill factory on the host.

The regions of sdpa_program_factory.cpp the oneq edits touch (the work split with the PS0 decode, F1's flag parse, F4's
envelope / grouping / log, and the reader runtime-argument loop's per-core range) are cut out of the patched file text
(apply_factory_ps.py applied to the served fd8c0676) and compiled with g++ against the stubs of stub_compile_oneq.py,
with real definitions for TT_FATAL (throws) and log_info (renders the {} placeholders). The compiled program takes one
call's shape and program word, runs those very lines and prints, per core, the range and the chain words the reader
would get. The CPU tests compare that output with oneq_planner.plan(): the planner is the factory, line for line, on
every shape the tests name - without a card and without a tt-metal tree.

    python3 -B host_exec_oneq.py 6 1 2048 128 13 10 0xb          # NQH NKH rows q_chunk grid_x grid_y word [qk_subblocks]
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stub_compile_oneq as stubs  # noqa: E402

NL = chr(10)
RT_BEGIN = '        uint32_t global_q_start = i * global_q_base_chunks_per_core +' + NL
RT_END = '            global_q_count = total_q_chunks - global_q_start;' + NL + '        }' + NL

DEFINITIONS = r'''
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <type_traits>

struct StubArg {
    bool number = false;
    unsigned long long value = 0;
    std::string text;
};
template <typename T>
StubArg stub_arg(const T& value) {
    StubArg arg;
    if constexpr (std::is_arithmetic_v<T>) {
        arg.number = true;
        arg.value = static_cast<unsigned long long>(value);
    } else {
        arg.text = std::string(value);
    }
    return arg;
}
inline std::string stub_render(const char* format, const std::vector<StubArg>& args) {
    std::string out;
    std::size_t next = 0;
    for (const char* p = format; *p; ++p) {
        if (*p == '{') {
            const char* q = std::strchr(p, '}');
            const std::string spec(p + 1, q);
            if (next < args.size()) {
                const StubArg& arg = args[next];
                if (!arg.number) {
                    out += arg.text;
                } else if (spec == ":#x") {
                    char buffer[32];
                    std::snprintf(buffer, sizeof buffer, "0x%llx", arg.value);
                    out += buffer;
                } else {
                    out += std::to_string(arg.value);
                }
            }
            ++next;
            p = q;
        } else {
            out += *p;
        }
    }
    return out;
}
template <typename... Args>
void log_info(tt::LogType, const char* format, Args&&... args) {
    std::vector<StubArg> rendered{stub_arg(args)...};
    std::printf("LOG %s\n", stub_render(format, rendered).c_str());
}
template <typename... Args>
void stub_fatal(bool condition, const char* format, Args&&... args) {
    if (!condition) {
        std::vector<StubArg> rendered{stub_arg(args)...};
        throw std::runtime_error(stub_render(format, rendered));
    }
}
CoreCoord IDevice::worker_core_from_logical_core(const CoreCoord& logical) const {
    return CoreCoord{logical.x + 1, logical.y + 2};
}
'''

SIGNATURE = r'''
int run_case(const std::optional<SDPAProgramConfig>& program_config, IDevice* device, const bool is_causal,
             const uint32_t NQH, const uint32_t NKH, const uint32_t q_num_chunks, CoreCoord grid_size,
             const uint32_t Sq_chunk_t, const uint32_t qk_in0_num_subblocks) {
    const bool flexible_chunked = true, has_sliding_window = false, use_provided_mask = false, is_windowed = false;
    const bool use_attention_sink = false, use_mla = false, use_streaming_compute = false, lightweight_causal = true;
    const uint32_t B = 1, NVH = NKH, Sk_chunk_t = Sq_chunk_t, DHt = 8, vDHt = 8;
    const tt::DataFormat k_df = tt::DataFormat::Bfp8_b, v_df = tt::DataFormat::Bfp8_b;
    uint32_t num_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
'''
AFTER_SPLIT = r'''
    std::printf("SPLIT total=%u max_per_core=%u q_buffer_factor=%u\n", total_q_chunks, max_global_q_chunks_per_core,
                q_buffer_factor);
'''
AFTER_F4 = r'''
    for (uint32_t i = 0; i < num_cores; ++i) {
'''
RT_PRINT = r'''
        const auto& chain = core_chain_info[i];
        std::printf("CORE %u start=%u count=%u participates=%d injector=%d sink=%d batch=%u head=%u qstart=%u qcount=%u "
                    "prev=%zu,%zu next=%zu,%zu nextq=%u\n",
                    i, global_q_start, global_q_count, chain.participates, chain.is_injector, chain.is_sink, chain.batch,
                    chain.head, chain.q_chunk_start, chain.q_chunk_count, chain.prev_physical.x, chain.prev_physical.y,
                    chain.next_physical.x, chain.next_physical.y, chain.next_core_q_chunks);
    }
    return 0;
}
'''
MAIN = r'''
int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: NQH NKH rows q_chunk grid_x grid_y word [qk_subblocks] [causal]\n");
        return 2;
    }
    const uint32_t NQH = std::strtoul(argv[1], nullptr, 0), NKH = std::strtoul(argv[2], nullptr, 0);
    const uint32_t rows = std::strtoul(argv[3], nullptr, 0), q_chunk = std::strtoul(argv[4], nullptr, 0);
    CoreCoord grid{std::strtoul(argv[5], nullptr, 0), std::strtoul(argv[6], nullptr, 0)};
    SDPAProgramConfig config;
    config.max_cores_per_head_batch = static_cast<uint32_t>(std::strtoull(argv[7], nullptr, 0));
    const uint32_t Sq_chunk_t = q_chunk / 32;
    const uint32_t subblocks = argc > 8 ? std::strtoul(argv[8], nullptr, 0) : Sq_chunk_t / 2;
    const bool causal = argc > 9 ? std::strtoul(argv[9], nullptr, 0) != 0 : true;
    IDevice device;
    try {
        return run_case(config, &device, causal, NQH, NKH, rows / q_chunk, grid, Sq_chunk_t, subblocks);
    } catch (const std::exception& error) {
        std::printf("FATAL %s\n", error.what());
        return 3;
    }
}
'''


def source(text=None):
    text = stubs.patched_text() if text is None else text
    prelude = stubs.stub_compile.FACTORY_PRELUDE.split('uint32_t stub_create_descriptor(')[0]
    split = stubs.cut(text, stubs.SPLIT_BEGIN, stubs.SPLIT_END, 'work split') + stubs.SPLIT_END
    f1 = stubs.cut(text, stubs.F1_BEGIN, stubs.F1_END, 'F1')
    f4 = stubs.cut(text, stubs.F4_BEGIN, stubs.F4_END, 'F4')
    rt = stubs.cut(text, RT_BEGIN, RT_END, 'reader runtime-argument range') + RT_END
    return (prelude + DEFINITIONS + SIGNATURE + split + AFTER_SPLIT + f1 + stubs.BETWEEN_F1_F4 + f4
            + AFTER_F4 + rt + RT_PRINT + MAIN)


def build(directory, gxx=None, text=None):
    """Compile the harness into `directory`; -> the executable's path (cached by source hash)."""
    gxx = stubs.find_gxx(gxx)
    if gxx is None:
        raise RuntimeError('no g++ found (set QWEN_PF_GXX)')
    code = source(text)
    tag = hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / ('oneq_exec_%s' % tag)
    if exe.exists():
        return exe
    src = directory / ('oneq_exec_%s.cpp' % tag)
    src.write_text(code, encoding='utf-8', newline=NL)
    result = subprocess.run([gxx, '-std=c++20', '-O0', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter', '-o', str(exe),
                             str(src)], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        raise RuntimeError('host-exec harness failed to compile:' + NL + (result.stdout + result.stderr)[:4000])
    return exe


def run(exe, nqh, nkh, rows, q_chunk, grid, word, subblocks=None, causal=True):
    """-> dict(fatal=text or None, logs=[...], split=dict, cores=[dict per core])."""
    argv = [str(exe), str(nqh), str(nkh), str(rows), str(q_chunk), str(grid[0]), str(grid[1]), hex(word)]
    if subblocks is not None or not causal:
        argv += [str(subblocks if subblocks is not None else q_chunk // 64), '1' if causal else '0']
    result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace',
                            env={k: v for k, v in os.environ.items() if k != 'QWEN_SDPA_PF_TEST'})
    out = dict(fatal=None, logs=[], split=None, cores=[], returncode=result.returncode)
    for line in result.stdout.splitlines():
        if line.startswith('FATAL '):
            out['fatal'] = line[len('FATAL '):]
        elif line.startswith('LOG '):
            out['logs'].append(line[len('LOG '):])
        elif line.startswith('SPLIT '):
            out['split'] = dict(pair.split('=') for pair in line[len('SPLIT '):].split())
        elif line.startswith('CORE '):
            fields = line.split()
            core = {'core': int(fields[1])}
            for item in fields[2:]:
                key, value = item.split('=')
                core[key] = tuple(int(v) for v in value.split(',')) if ',' in value else int(value)
            out['cores'].append(core)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    with tempfile.TemporaryDirectory() as directory:
        exe = build(directory)
        nqh, nkh, rows, q_chunk, gx, gy = (int(value, 0) for value in argv[:6])
        outcome = run(exe, nqh, nkh, rows, q_chunk, (gx, gy), int(argv[6], 0))
    print(outcome['fatal'] or '\n'.join(outcome['logs']))
    busy = sum(1 for core in outcome['cores'] if core['count'])
    print('busy cores', busy)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
