"""g++ -fsyntax-only of the K64i kernels and factory blocks against syntax stubs (no tt-metal tree needed).

Three checks, each with -Wall -Wextra -Werror (unused-variable style warnings are compared, not fatal: the
served kernels carry some and compile in the JIT):

  readers   the SERVED stage-3 reader_decode_qwen.cpp (280a847f) and reader_decode_qwen_slice.cpp, against the
            real sdpa_decode dataflow_common.hpp / rt_args_common.hpp and the real prefill dataflow_common.hpp
            it includes (with q_chunk_remapping.hpp and sliding_window_geometry.hpp), plus the API stubs of
            ../../sdpa_prefill_chain/stubcheck/include and include/ here. The served reader passing proves the
            stubs cover the surface; the slice reader must then pass at every flag combination the factory can
            build (0x7, 0xF, 0xB, 0x5, G16's 0x5) with no warning the served reader lacks, and FAIL where its
            static_asserts must fire (a wrong ABI tag; a slice that runs past Q; read-ahead without share).
  writers   the stock writer_decode_all.cpp (734c90c0) and writer_decode_qwen_slice.cpp, likewise (the slice
            writer at G8 and G16, and failing on a wrong tag and a slice past the output).
  factory   the stage-4 blocks of apply_factory_slice.py (F13-F18) compiled inside a stub create_descriptor
            whose locals carry the factory's own declared types.

The compile-time-arg vectors are the factory's for the served call (G8, B=2, 131,328 keys, 16 cores per
head); accessor blocks take one slot each in the stubs (TensorAccessorArgs<Base>::next = Base + 1).
Evidence, not proof: the stubs are declarations written from the call sites. The ttbuild ninja build and
the card-B run's JIT compile are the proofs.

    py -3.11 stubcheck/stub_compile.py
Env: QWEN_PF_GXX (compiler), QWEN_SDPA_SOURCES_DUMP (the decode sources dump), QWEN_SDPA_PREFILL_SRC
(probe-v25 src/device: the prefill kernel headers).
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SLICE = HERE.parent
OPS = SLICE.parent
PREFILL_STUBS = OPS / 'sdpa_prefill_chain' / 'stubcheck' / 'include'
for path in (str(SLICE), str(OPS / 'sdpa_decode_qwen')):
    if path not in sys.path:
        sys.path.insert(0, path)

import apply_factory_slice as factory  # noqa: E402
import make_qwen_kernels as qwen_kernels  # noqa: E402
import make_slice_kernels as kernels  # noqa: E402

JOB = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp')
DEFAULT_DUMP = Path(os.environ.get('QWEN_SDPA_SOURCES_DUMP', JOB / 'sdpa-decode-sources.txt'))
DEFAULT_PREFILL = Path(os.environ.get('QWEN_SDPA_PREFILL_SRC', JOB / 'probe-v25' / 'src' / 'device'))
LOCAL_ARM_GXX = 'C:/Users/liamb/AppData/Local/stm32cube/bundles/gnu-tools-for-stm32/13.3.1+st.9/bin/arm-none-eabi-g++.exe'
FLAGS = ['-std=c++20', '-fsyntax-only', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
         '-Wno-error=unused-variable', '-Wno-error=unused-but-set-variable']
DECODE_KERNELS = 'ttnn/operations/transformer/sdpa_decode/device/kernels'
PREFILL_KERNELS = 'ttnn/operations/transformer/sdpa/device/kernels'
PREFILL_TREE = 'cpp/ttnn/operations/transformer/sdpa/device/kernels'
PREFILL_DATAFLOW_COMMON = '554a0b282d2a36c7b129eef04df67a5becbb14f98220246e43e47fdd56118aa6'

# The served call (G8 B=2, 131,328 keys, 64 active cores, 16 per head) as the stage-3/4 factory builds it.
READER_CT = dict(B=2, PNHt=3, St=4104, DHt=8, vDHt=8, Sk_chunk_t=8, num_cores=64, is_q_sharded=0,
                 num_cores_per_batch=32, k_chunk_size=256, index_stick_size_B=0, is_paged_attention=1,
                 num_kv_heads=2, block_size_t=2, Bkv=2116, q_heads_parallel_factor=1, num_cores_per_head=16,
                 num_heads_per_core=1, num_output_cores=2, is_causal=0, use_attention_mask=1,
                 use_attention_sink=0, max_dynamic_chunk_size=8, tilize_q=0, reuse_k=0, use_half_tile=0,
                 q_chunk_size_bytes=3 * 8 * 2048, is_cur_pos_tensor_sharded=0, is_page_table_sharded=0,
                 q_page_size_bytes=2048, sliding_window_size=0, original_block_size=64, k_mcast_semaphore_id=2,
                 q_locally_available=0, use_k_mcast=0, Bmask=2, capacity_t=0)
READER_ACCESSORS = 7            # q, k, v, mask, cur_pos, page_table, attention_sink
WRITER_CT = dict(B=2, PNHt=3, St=4104, DHt=8, vDHt=8, Sk_chunk_t=8, identity_scalar_packed=0x3F803F80,
                 zero_scalar_packed=0, scale_val=0x3D800000, num_cores_per_batch=32, num_cores=64,
                 reducer_semaphore_id=0, output_semaphore_id=1, is_out_sharded=0, k_chunk_size=256, num_q_heads=96,
                 num_kv_heads=2, num_cores_per_head=16, num_heads_per_core=1, num_reducer_cores=4,
                 num_output_cores=2, ELEMENT_SIZE=2, is_causal=0, max_dynamic_chunk_size=8,
                 q_heads_parallel_factor=1, sliding_window_size=0, num_tree_reduction_rounds=4, original_block_size=64)
WRITER_ACCESSORS = 1            # out


def reader_ct(pnht, qwen=(1, 4104, 1, 3), slice_suffix=None, **overrides):
    values = dict(READER_CT, PNHt=pnht, q_chunk_size_bytes=pnht * 8 * 2048, **overrides)
    vector = list(values.values()) + [0] * READER_ACCESSORS + list(qwen)
    return vector + list(slice_suffix or ())


def writer_ct(pnht, slice_suffix=None, **overrides):
    values = dict(WRITER_CT, PNHt=pnht, **overrides)
    return list(values.values()) + [0] * WRITER_ACCESSORS + list(slice_suffix or ())


TAG = kernels.ABI_TAG
# (label, kernel, CT vector, True (must compile) or the static_assert text that must stop it). The reader's slice
# suffix: pnht_full, rows_per_kv, kv_readahead, q_slice, tag; the writer's: pnht_full, rows_per_kv, tag.
READER_CASES = (
    ('stage-3 reader 0x3 (served)', 'stage3', reader_ct(3), True),
    ('slice reader 0x7 (K1a)', 'slice', reader_ct(2, slice_suffix=(3, 48, 0, 1, TAG)), True),
    ('slice reader 0xF (K1a+K1b)', 'slice', reader_ct(2, slice_suffix=(3, 48, 1, 1, TAG)), True),
    ('slice reader 0xB (K1b, no slice)', 'slice', reader_ct(3, slice_suffix=(3, 48, 1, 0, TAG)), True),
    ('slice reader 0x5 (B=1, no share)', 'slice', reader_ct(2, qwen=(1, 4104, 0, 3), slice_suffix=(3, 48, 0, 1, TAG),
                                                             B=1, num_output_cores=1, Bmask=1), True),
    ('slice reader G16 0x5', 'slice', reader_ct(3, qwen=(1, 4104, 0, 3), slice_suffix=(6, 96, 0, 1, TAG),
                                                B=1, num_output_cores=1, Bmask=1), True),
    ('slice reader, wrong ABI tag', 'slice', reader_ct(2, slice_suffix=(3, 48, 1, 1, 0x51CF)), 'ABI tag'),
    ('slice reader, slice past Q', 'slice', reader_ct(3, slice_suffix=(3, 48, 0, 1, TAG)), 'runs past Q'),
    ('slice reader, read-ahead without share', 'slice', reader_ct(2, qwen=(1, 4104, 0, 3),
                                                                  slice_suffix=(3, 48, 1, 1, TAG)), 'needs KV share'),
    ('slice reader, 0xB with a sliced PNHt', 'slice', reader_ct(2, slice_suffix=(3, 48, 1, 0, TAG)),
     'every row tile is read'),
)
WRITER_CASES = (
    ('stock writer (served)', 'stock', writer_ct(3), True),
    ('slice writer 0x7', 'slice', writer_ct(2, slice_suffix=(3, 48, TAG)), True),
    ('slice writer G16', 'slice', writer_ct(3, slice_suffix=(6, 96, TAG), num_q_heads=192), True),
    ('slice writer 7-row groups', 'slice', writer_ct(2, slice_suffix=(3, 42, TAG), num_q_heads=84), True),
    ('slice writer, wrong ABI tag', 'slice', writer_ct(2, slice_suffix=(3, 48, 0x51CF)), 'ABI tag'),
    ('slice writer, slice past the output', 'slice', writer_ct(3, slice_suffix=(3, 48, TAG)), 'runs past the output'),
    ('slice writer, rows_per_kv not num_q_heads / num_kv_heads', 'slice', writer_ct(2, slice_suffix=(3, 40, TAG)),
     "each KV head's own rows"),
)


def find_gxx(explicit=None):
    for candidate in (explicit, os.environ.get('QWEN_PF_GXX'), shutil.which('g++'), LOCAL_ARM_GXX):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate
    return None


def sources(dump, prefill):
    """The real headers and the served kernels: {relative path: bytes}. ValueError if a sha is off."""
    text = Path(dump).read_bytes().decode('utf-8')
    decode = {name: qwen_kernels.from_dump(text, qwen_kernels.DUMP_PREFIX + name)
              for name in ('dataflow/dataflow_common.hpp', 'rt_args_common.hpp', 'dataflow/writer_decode_all.cpp')}
    out = {DECODE_KERNELS + '/' + name: data for name, data in decode.items()}
    common = (Path(prefill) / 'kernels' / 'dataflow' / 'dataflow_common.hpp').read_bytes()
    if qwen_kernels.sha256(common) != PREFILL_DATAFLOW_COMMON:
        raise ValueError('the prefill dataflow_common.hpp at %s is not the served %s' % (prefill, PREFILL_DATAFLOW_COMMON[:16]))
    out[PREFILL_KERNELS + '/dataflow/dataflow_common.hpp'] = common
    for header in ('q_chunk_remapping.hpp', 'sliding_window_geometry.hpp'):
        out[PREFILL_TREE + '/' + header] = (Path(prefill) / 'kernels' / header).read_bytes()
    return out


def kernel_text(kind):
    return {
        'stage3': (OPS / 'sdpa_decode_qwen' / 'stage3' / qwen_kernels.READER_NAME).read_bytes(),
        'slice_reader': (SLICE / kernels.READER_SLICE_NAME).read_bytes(),
        'slice_writer': (SLICE / kernels.WRITER_SLICE_NAME).read_bytes(),
    }[kind]


def compile_case(gxx, root, name, data, ct):
    target = root / DECODE_KERNELS / 'dataflow' / name
    target.write_bytes(data)
    command = [gxx] + FLAGS + ['-include', str(HERE / 'include' / 'stub_decode_api.hpp'),
                               '-I' + str(HERE / 'include'), '-I' + str(PREFILL_STUBS), '-I' + str(root),
                               '-DSTUB_CT_ARGS=' + ','.join(str(value) for value in ct), str(target)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return result.returncode, (result.stdout + result.stderr).strip()


WARNING = re.compile(r"warning: (?:unused variable|variable) '([A-Za-z_0-9]+)'.*\[(-W[a-z-]+)\]")


def warning_set(output):
    return {(match.group(2), match.group(1)) for match in WARNING.finditer(output)}


def check_kernels(gxx, dump, prefill):
    """[(label, ok, detail)]: every case compiles or fails as it must; the slice kernels add no warning."""
    results = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for relative, data in sources(dump, prefill).items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_bytes(data)
        stock_writer = (root / DECODE_KERNELS / 'dataflow' / 'writer_decode_all.cpp').read_bytes()
        texts = {('reader', 'stage3'): (qwen_kernels.READER_NAME, kernel_text('stage3')),
                 ('reader', 'slice'): (kernels.READER_SLICE_NAME, kernel_text('slice_reader')),
                 ('writer', 'stock'): ('writer_decode_all.cpp', stock_writer),
                 ('writer', 'slice'): (kernels.WRITER_SLICE_NAME, kernel_text('slice_writer'))}
        baseline = {}
        for family, cases in (('reader', READER_CASES), ('writer', WRITER_CASES)):
            for label, kind, ct, expect in cases:
                name, data = texts[(family, kind)]
                code, output = compile_case(gxx, root, name, data, ct)
                warnings = warning_set(output)
                if kind in ('stage3', 'stock'):
                    baseline[family] = warnings
                if expect is True:
                    extra = warnings - baseline.get(family, set())
                    ok = code == 0 and not extra
                    detail = output if code else ('new warnings %r' % sorted(extra) if extra else 'compiles')
                else:
                    ok = code != 0 and 'static assertion failed' in output and expect in output
                    detail = ('refused by its static_assert (%s)' % expect if ok
                              else 'compiled, but must not' if code == 0 else output)
                results.append((label, ok, detail))
    return results


# --- the factory blocks -------------------------------------------------------------------------

FACTORY_PRELUDE = r'''
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Stubs for the host names the stage-4 blocks touch, typed as sdpa_decode_program_factory.cpp declares them.
namespace tt {
enum LogType { LogOp };
}  // namespace tt
template <typename... Args>
void log_info(tt::LogType, const char* format, Args&&... args);
template <typename... Args>
void stub_fatal(bool condition, const char* format, Args&&... args);
#define TT_FATAL(condition, ...) stub_fatal(static_cast<bool>(condition), __VA_ARGS__)
struct Shape {
    uint32_t operator[](int index) const;
};
struct Tensor {
    Shape padded_shape() const;
};
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

uint32_t stub_create_descriptor(uint32_t q_chunk_size, bool has_program_config, uint32_t B, uint32_t PNH,
                                uint32_t q_heads_parallel_factor, uint32_t num_q_heads, uint32_t num_kv_heads,
                                const std::optional<Tensor>& attn_mask, bool use_attention_mask, uint32_t St,
                                uint32_t out_tiles, uint32_t intermed_output_tiles, uint32_t qwen_cb_bytes) {
    constexpr std::size_t kQwenMagicMask = 0xFFFFFF00u;
    constexpr std::size_t kQwenMagic = 0x51DEC000u;
    constexpr uint32_t kQwenMaskTail = 0x1u;
'''

FACTORY_MIDDLE = r'''
    const bool qwen_mode = has_program_config && (q_chunk_size & kQwenMagicMask) == kQwenMagic;
    const uint32_t qwen_flags = qwen_mode ? static_cast<uint32_t>(q_chunk_size & 0xFFu) : 0u;
'''

FACTORY_TAIL = r'''
    const bool qwen_mask_tail = (qwen_flags & kQwenMaskTail) != 0;
    const uint32_t qwen_mask_width_t = use_attention_mask ? attn_mask->padded_shape()[3] / TILE_WIDTH : St;
    if (qwen_mode) {
'''

FACTORY_AFTER_CHECKS = r'''
    }
    if (qwen_mode) {
        log_info(tt::LogOp, "[QWEN-SDPA] flags={:#x} B={} PNHt={} St={} mask_width_t={} kv_share={} scratch_slots={} cb_bytes={}",
                 qwen_flags, B, PNHt, St, qwen_mask_width_t, qwen_kv_share,
'''

FACTORY_CTAS = r'''
    const uint32_t kv_ready_semaphore_id = 3;
    std::vector<uint32_t> reader_compile_time_args_common(44, 0u);
    std::vector<uint32_t> writer_compile_time_args_common(29, 0u);
    if (qwen_mode) {
        reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_mask_tail));
        reader_compile_time_args_common.push_back(qwen_mask_width_t);
        reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_kv_share));
'''

FACTORY_KERNELS = r'''
    const std::string kernel_path = "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/";
    struct {
        std::string kernel_source;
    } reader_desc, writer_desc;
'''

FACTORY_CLOSE = r'''
    return static_cast<uint32_t>(reader_compile_time_args_common.size() + writer_compile_time_args_common.size() +
                                 reader_desc.kernel_source.size() + writer_desc.kernel_source.size()) + PNHt;
}
'''


def edit(label):
    for name, _old, new in factory.STAGE4_EDITS:
        if name == label:
            return new
    raise KeyError(label)


def factory_harness():
    """The F13-F18 new texts in file order inside a stub create_descriptor. Anchors that are the tail of a
    served statement (F16 reader's '    }', F18's continuation line) are completed by the stub text around
    them; F17's new texts replace whole statements."""
    f16_reader = edit('F16 reader')
    f16_writer = 'writer_compile_time_args_common.push_back(0u);' + chr(10)
    f16_writer = edit('F16 writer').replace(factory.F16W_ANCHOR, '    ' + f16_writer)
    f18 = edit('F18')
    return (FACTORY_PRELUDE + edit('F13') + FACTORY_MIDDLE + edit('F14')
            + '    const bool qwen_kv_share = (qwen_flags & kQwenKvShare) != 0 && B > 1;' + chr(10)
            + edit('F15 decl').replace(factory.F15_DECL_ANCHOR, '')
            + FACTORY_TAIL + edit('F15') + FACTORY_AFTER_CHECKS + f18
            + FACTORY_CTAS + f16_reader + f16_writer + FACTORY_KERNELS + edit('F17 reader') + edit('F17 writer')
            + FACTORY_CLOSE)


def check_factory(gxx):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'factory_blocks.cpp'
        source.write_text(factory_harness(), encoding='utf-8', newline=chr(10))
        command = [gxx] + ['-std=c++20', '-fsyntax-only', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
                           str(source)]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return result.returncode, (result.stdout + result.stderr).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--gxx', help='compiler (default: QWEN_PF_GXX, g++ on PATH, the local arm-none-eabi-g++)')
    parser.add_argument('--dump', default=str(DEFAULT_DUMP), help='the decode sources dump')
    parser.add_argument('--prefill', default=str(DEFAULT_PREFILL), help='probe-v25 src/device')
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
    print('factory blocks F13-F18: %s' % ('ok' if code == 0 else 'FAILED'))
    if output:
        print(output)
    status |= code != 0
    if not Path(options.dump).is_file() or not (Path(options.prefill) / 'kernels').is_dir():
        print('kernels: skipped (no dump at %s or no prefill tree at %s)' % (options.dump, options.prefill))
        return int(status)
    for label, ok, detail in check_kernels(gxx, options.dump, options.prefill):
        print('%-58s %s' % (label, 'ok (%s)' % detail if ok else 'FAILED: ' + detail))
        status |= not ok
    return int(status)


if __name__ == '__main__':
    raise SystemExit(main())
