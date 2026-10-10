"""g++ -fsyntax-only of the [QWEN-SDPA-PF] oneq C++ (apply_factory_ps.py PS0-PS5) against syntax stubs.

The edits are not compiled in isolation: the three regions of the patched factory they touch are cut out of the
PS factory text itself (the work split with the PS0 decode, F1's flag parse with PS1/PS2, F4's envelope, grouping and
log with PS3-PS5) and compiled in order inside a stub create_descriptor whose parameters carry the factory's own
declared types. The stubs are the ones of ../sdpa_prefill_chain/stubcheck/stub_compile.py (its type declarations for
CoreCoord, IDevice, SDPAProgramConfig, CoreChainInfo, TT_FATAL and log_info), so a misspelt name, a missing
declaration or a wrong type in the new code fails here instead of in ttbuild.

It is evidence, not proof: the stubs are declarations written from the call sites. The ttbuild ninja run is the proof.

    python3 -B stub_compile_oneq.py               # with g++ on PATH (QWEN_PF_GXX overrides)
    python3 -B stub_compile_oneq.py --dump        # print the harness and exit
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'sdpa_prefill_chain' / 'stubcheck'))

import apply_factory_ps as ps  # noqa: E402
import stub_compile  # noqa: E402

NL = chr(10)
FLAGS = ['-std=c++20', '-fsyntax-only', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter']

SPLIT_BEGIN = '    const uint32_t total_q_chunks = B * NQH * q_num_chunks;' + NL
SPLIT_END = '    const uint32_t q_buffer_factor = (max_global_q_chunks_per_core > 1) ? 2 : 1;' + NL
F1_BEGIN = '    const bool use_zigzag_balancing = is_causal;' + NL
F1_END = NL + '    std::vector<uint32_t> reader_compile_time_args = {'
F4_BEGIN = '    // [QWEN-SDPA-PF] G6 K/V chains.'
F4_END = '    // Update mcast_enabled compile-time arg now that chain construction is complete' + NL

SIGNATURE = r'''
uint32_t stub_create_descriptor(const std::optional<SDPAProgramConfig>& program_config, IDevice* device,
                                const bool is_causal, const bool flexible_chunked, const bool has_sliding_window,
                                const bool use_provided_mask, const bool is_windowed, bool use_attention_sink,
                                const bool use_mla, const bool use_streaming_compute, const bool lightweight_causal,
                                const uint32_t B, const uint32_t NQH, const uint32_t NKH, const uint32_t NVH,
                                const uint32_t Sq_chunk_t, const uint32_t Sk_chunk_t, const uint32_t DHt,
                                const uint32_t vDHt, const uint32_t q_num_chunks, CoreCoord grid_size,
                                uint32_t num_cores, tt::DataFormat k_df, tt::DataFormat v_df,
                                const uint32_t qk_in0_num_subblocks) {
'''
BETWEEN_F1_F4 = r'''
    (void)use_zigzag_balancing;
    uint32_t num_phases = 1;
    std::vector<CoreChainInfo> core_chain_info(num_cores);
'''
CLOSE = r'''
    return q_buffer_factor + static_cast<uint32_t>(core_chain_info.size());
}
'''


def cut(text, begin, end, label):
    start = text.index(begin)
    stop = text.index(end, start)
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError('%s: the region markers are not unique' % label)
    return text[start:stop]


def patched_text():
    from_pf = stub_compile.factory  # noqa: F841 - the module is imported for its side effects (path), kept explicit
    served = (HERE.parent / 'sdpa_prefill_chain' / 'fixtures' / 'sdpa_program_factory.fd8c0676.cpp').read_bytes()
    return ps.patch(served).decode('utf-8')


def harness(text=None):
    text = patched_text() if text is None else text
    prelude = stub_compile.FACTORY_PRELUDE.split('uint32_t stub_create_descriptor(')[0]
    split = cut(text, SPLIT_BEGIN, SPLIT_END, 'work split') + SPLIT_END
    f1 = cut(text, F1_BEGIN, F1_END, 'F1')
    f4 = cut(text, F4_BEGIN, F4_END, 'F4')
    return prelude + SIGNATURE + split + f1 + BETWEEN_F1_F4 + f4 + CLOSE


def compile_harness(gxx, source_text):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'factory_oneq_blocks.cpp'
        source.write_text(source_text, encoding='utf-8', newline=NL)
        result = subprocess.run([gxx] + FLAGS + [str(source)], capture_output=True, text=True, encoding='utf-8',
                                errors='replace')
    return result.returncode, (result.stdout + result.stderr).strip()


def find_gxx(explicit=None):
    for candidate in (explicit, os.environ.get('QWEN_PF_GXX'), shutil.which('g++')):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('--gxx')
    parser.add_argument('--dump', action='store_true')
    options = parser.parse_args(argv)
    text = harness()
    if options.dump:
        sys.stdout.write(text)
        return 0
    gxx = find_gxx(options.gxx)
    if gxx is None:
        print('no g++ found (set QWEN_PF_GXX)', file=sys.stderr)
        return 2
    code, output = compile_harness(gxx, text)
    print('compiler %s' % gxx)
    print('oneq factory blocks (work split, F1, F4): %s' % ('ok' if code == 0 else 'FAILED'))
    if output:
        print(output)
    return int(code != 0)


if __name__ == '__main__':
    raise SystemExit(main())
