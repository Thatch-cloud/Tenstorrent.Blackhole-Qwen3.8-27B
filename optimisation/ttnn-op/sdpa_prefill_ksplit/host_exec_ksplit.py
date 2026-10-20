"""Run the REAL decision C++ of the ksplit factory on the host (the twin of ../sdpa_prefill_oneq/host_exec_oneq.py).

The regions of sdpa_program_factory.cpp the ksplit edits touch and that DECIDE something are cut out of the patched file text (apply_factory_pk.py applied to the served fd8c0676, through
PF and PS): the work split with the PS0 / PK0 decode (S, the busy core count), F1's flag parse with PK1-PK3, F4's envelope / grouping / log with PK8-PK10, and the reader runtime-argument
loop's per-core range with PK12 (slice, previous and next core). They are compiled with g++ against the stubs of ../sdpa_prefill_oneq/stub_compile_oneq.py, with real definitions for
TT_FATAL (throws) and log_info (renders the {} placeholders), and run for one call's shape and program word. The CPU tests compare the output with ksplit_planner.plan(): the planner is
the factory, line for line, on every shape the tests name - without a card and without a tt-metal tree.

    python3 -B host_exec_ksplit.py 6 1 1024 128 13 10 0x1b          # NQH NKH rows q_chunk grid_x grid_y word [qk_subblocks]
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ONEQ = HERE.parent / 'sdpa_prefill_oneq'
sys.path.insert(0, str(ONEQ))
sys.path.insert(0, str(HERE))

import apply_factory_pk as pk  # noqa: E402
import host_exec_oneq as base  # noqa: E402
import stub_compile_oneq as stubs  # noqa: E402

NL = chr(10)
RT_BEGIN = base.RT_BEGIN
SIGNATURE = r'''
int run_case(const std::optional<SDPAProgramConfig>& program_config, IDevice* device, const bool is_causal,
             const uint32_t NQH, const uint32_t NKH, const uint32_t q_num_chunks, CoreCoord grid_size,
             const uint32_t Sq_chunk_t, const uint32_t qk_in0_num_subblocks, const uint32_t scale_packed,
             const bool fp32_dest_acc_en, const bool exp_approx_mode) {
    const bool flexible_chunked = true, has_sliding_window = false, use_provided_mask = false, is_windowed = false;
    const bool use_attention_sink = false, use_mla = false, use_streaming_compute = false, lightweight_causal = true;
    const uint32_t B = 1, NVH = NKH, Sk_chunk_t = Sq_chunk_t, DHt = 8, vDHt = 8;
    const tt::DataFormat k_df = tt::DataFormat::Bfp8_b, v_df = tt::DataFormat::Bfp8_b;
    uint32_t num_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
'''
AFTER_SPLIT = r'''
    std::printf("SPLIT total=%u max_per_core=%u q_buffer_factor=%u ks=%d slices=%u cores=%u\n", total_q_chunks, max_global_q_chunks_per_core,
                q_buffer_factor, static_cast<int>(qwen_ks), qwen_ks_slices, qwen_ks_cores);
'''
RT_PRINT = r'''
        const auto& chain = core_chain_info[i];
        std::printf("CORE %u start=%u count=%u participates=%d injector=%d sink=%d batch=%u head=%u qstart=%u qcount=%u "
                    "prev=%zu,%zu next=%zu,%zu nextq=%u slice=%u ksprev=%zu,%zu ksnext=%zu,%zu\n",
                    i, global_q_start, global_q_count, chain.participates, chain.is_injector, chain.is_sink, chain.batch,
                    chain.head, chain.q_chunk_start, chain.q_chunk_count, chain.prev_physical.x, chain.prev_physical.y,
                    chain.next_physical.x, chain.next_physical.y, chain.next_core_q_chunks, qwen_ks_slice, qwen_ks_prev.x,
                    qwen_ks_prev.y, qwen_ks_next.x, qwen_ks_next.y);
    }
    return 0;
}
'''
MAIN = r'''
int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: NQH NKH rows q_chunk grid_x grid_y word [qk_subblocks] [causal] [scale_packed] [fp32_dest] [exp_approx]\n");
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
    const uint32_t scale_packed = argc > 10 ? static_cast<uint32_t>(std::strtoull(argv[10], nullptr, 0)) : 0x3D800000u;
    const bool fp32_dest = argc > 11 ? std::strtoul(argv[11], nullptr, 0) != 0 : true;
    const bool approx = argc > 12 ? std::strtoul(argv[12], nullptr, 0) != 0 : false;
    IDevice device;
    try {
        return run_case(config, &device, causal, NQH, NKH, rows / q_chunk, grid, Sq_chunk_t, subblocks, scale_packed, fp32_dest, approx);
    } catch (const std::exception& error) {
        std::printf("FATAL %s\n", error.what());
        return 3;
    }
}
'''


def patched_text():
    served = (HERE.parent / 'sdpa_prefill_chain' / 'fixtures' / 'sdpa_program_factory.fd8c0676.cpp').read_bytes()
    return pk.patch(served).decode('utf-8')


def source(text=None):
    text = patched_text() if text is None else text
    prelude = stubs.stub_compile.FACTORY_PRELUDE.split('uint32_t stub_create_descriptor(')[0]
    split = stubs.cut(text, stubs.SPLIT_BEGIN, stubs.SPLIT_END, 'work split') + stubs.SPLIT_END
    f1 = stubs.cut(text, stubs.F1_BEGIN, stubs.F1_END, 'F1')
    f4 = stubs.cut(text, stubs.F4_BEGIN, stubs.F4_END, 'F4')
    begin = text.index(RT_BEGIN)
    end_marker = pk.PK12_NEW
    end = text.index(end_marker, begin) + len(end_marker)
    rt = text[begin:end]
    return (prelude + base.DEFINITIONS + SIGNATURE + split + AFTER_SPLIT + f1 + stubs.BETWEEN_F1_F4 + f4 + base.AFTER_F4 + rt + RT_PRINT + MAIN)


def build(directory, gxx=None, text=None):
    """Compile the harness into `directory`; -> the executable's path (cached by source hash)."""
    gxx = stubs.find_gxx(gxx)
    if gxx is None:
        raise RuntimeError('no g++ found (set QWEN_PF_GXX)')
    code = source(text)
    tag = hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / ('ksplit_exec_%s' % tag)
    if exe.exists():
        return exe
    src = directory / ('ksplit_exec_%s.cpp' % tag)
    src.write_text(code, encoding='utf-8', newline=NL)
    result = subprocess.run([gxx, '-std=c++20', '-O0', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter', '-Wno-unused-variable', '-o', str(exe), str(src)],
                            capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        raise RuntimeError('host-exec harness failed to compile:' + NL + (result.stdout + result.stderr)[:4000])
    return exe


def run(exe, nqh, nkh, rows, q_chunk, grid, word, subblocks=None, causal=True, scale_packed=0x3D800000, fp32_dest=True, approx=False):
    """-> dict(fatal=text or None, logs=[...], split=dict, cores=[dict per core])."""
    argv = [str(exe), str(nqh), str(nkh), str(rows), str(q_chunk), str(grid[0]), str(grid[1]), hex(word),
            str(subblocks if subblocks is not None else q_chunk // 64), '1' if causal else '0', hex(scale_packed), '1' if fp32_dest else '0', '1' if approx else '0']
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
    print(outcome['fatal'] or NL.join(outcome['logs']))
    busy = sum(1 for core in outcome['cores'] if core['count'])
    print('busy cores', busy)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
