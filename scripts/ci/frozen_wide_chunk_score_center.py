"""Parameterized port of dspark_ladder_score_center.py's
scalar_score_center() candidate - a TR0 scalar-FP32 mask-add and
max-centering pass on the raw QK scores before the exponential, replacing
three bf16 broadcast-op call sites (`add_block_inplace` for the mask add,
`sub_bcast_cols_init`/`sub_tiles_bcast_cols` for the max-centering) with a
shared `qwen_scalar_score_transform` helper operating on the raw score bits.

dspark_ladder_score_center.scalar_score_center(key_tiles=...) IS itself
parameterized by key_tiles/Skt (unlike scalar_sum_update, which has no
parameter at all) - the condition text is a clean substitution
(`.replace('== 2112', f'== {key_tiles}')`, dspark_ladder_score_center.py:90).
But its own validation guard (`if key_tiles not in (40, 2112): raise
ValueError(...)`, line 68-69) hardcodes an allowlist that excludes 2080, and
that module is not something this port modifies (it is the ladder's own,
separately-qualified-as-experimental candidate). So rather than call it
directly and be refused for Skt==2080, this module reproduces its
substitution logic with the same `.replace`-style parameterization but no
value restriction - generated for both accepted Skt values (2080 and 2112),
not refused for 2080.

Does NOT import or modify dspark_ladder_score_center.py itself.

UNVALIDATED at this Skt value / in this composition: hardware-proven only as
part of the ladder's own four-fix combination at Skt==2112
(run 34797353681); never run in isolation, and never run combined with this
port's other three fixes until a build+requalification cycle confirms it.
"""


SKT = 2080


HELPER_ANCHOR = 'void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {'
HELPER = '''
void qwen_scalar_score_transform(uint32_t scores_cb, uint32_t operand_cb, uint32_t tiles, uint32_t columns, bool mask) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    static uint32_t progress_calls = 0;
    const uint32_t progress_call = progress_calls++;
    const bool report_progress = progress_call < 2 || progress_call % 16 == 0;
    if (report_progress) DEVICE_PRINT("QWEN_SCORE_ENTER call={} mask={} tiles={}\\n", progress_call, mask, tiles);
    CircularBuffer(scores_cb).wait_front(tiles);
    if (report_progress) DEVICE_PRINT("QWEN_SCORE_SCORES_READY mask={}\\n", mask);
    CircularBuffer(operand_cb).wait_front(mask ? tiles : tiles / columns);
    if (report_progress) DEVICE_PRINT("QWEN_SCORE_OPERAND_READY mask={}\\n", mask);
    const auto scores_interface = get_local_cb_interface(scores_cb);
    const auto operand_interface = get_local_cb_interface(operand_cb);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        volatile float* scores = reinterpret_cast<volatile float*>(
            (scores_interface.fifo_rd_ptr + tile * scores_interface.fifo_page_size) << cb_addr_shift);
        const uint32_t operand_address = (operand_interface.fifo_rd_ptr +
            (mask ? tile : tile / columns) * operand_interface.fifo_page_size) << cb_addr_shift;
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t first = row < 16 ? row * 16 : 512 + (row - 16) * 16;
            for (uint32_t column = 0; column < 32; ++column) {
                const uint32_t offset = first + (column < 16 ? column : 256 + column - 16);
                if (mask) {
                    const volatile uint16_t* masks = reinterpret_cast<const volatile uint16_t*>(operand_address);
                    const uint32_t mask_bits = static_cast<uint32_t>(masks[offset]) << 16;
                    if ((mask_bits & 0x7fffffffU) == 0) continue;
                    union { uint32_t bits; float value; } converted;
                    converted.bits = mask_bits;
                    scores[offset] = scores[offset] + converted.value;
                } else {
                    const volatile float* maxima = reinterpret_cast<const volatile float*>(operand_address);
                    if (scores[offset] == -__builtin_inff() && maxima[first] == -__builtin_inff()) continue;
                    scores[offset] = scores[offset] - maxima[first];
                }
            }
        }
    }
    if (mask) CircularBuffer(operand_cb).pop_front(tiles);
    if (report_progress) DEVICE_PRINT("QWEN_SCORE_DONE mask={}\\n", mask);
#endif
}
'''
MASK = '                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);'
INIT = '    sub_bcast_cols_init(in0_cb, in1_cb);'
SUBTRACT = '                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);'


def _mask_after(skt):
    return (f'                    if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == {skt}) {{\n'
        '                        qwen_scalar_score_transform(cb_qk_im, cb_mask_in, qk_chunk_tiles, Sk_chunk_t, true);\n'
        '                    } else {\n' + MASK + '\n                    }')


def _init_after(skt):
    return (f'    if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == {skt}) {{\n'
        '        qwen_scalar_score_transform(in0_cb, in1_cb, rows * cols, cols, false);\n'
        '        copy_tile_to_dst_init_short(in0_cb);\n'
        '    } else {\n' + INIT + '\n    }')


def _subtract_after(skt):
    return (f'                if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == {skt}) {{\n'
        '                    copy_tile(in0_cb, j, j);\n'
        '                } else {\n' + SUBTRACT + '\n                }')


def score_center_scope():
    """Composed into dspark-native-8k-attention-probe.py's main() right
    after scalar_sum_update(), before kernel_scope() - see
    frozen_wide_chunk_normalization.py's _patch_probe() and
    docs/numerics-65536-attention.md section 7 for the ladder's own
    composition order this mirrors. Includes the DEVICE_PRINT progress
    instrumentation unconditionally, matching what
    dspark_ladder_score_center.scalar_score_center always does for either of
    its two accepted key_tiles values (the `if key_tiles in (40, 2112):`
    guard around it is unreachable-false given that function's own earlier
    validation already restricts key_tiles to exactly those two)."""
    from contextlib import contextmanager
    from unittest.mock import patch
    import native_draft_sdpa

    @contextmanager
    def scope():
        original = native_draft_sdpa.replacements

        def replacements():
            substitutions = original()
            substitutions['compute_common.hpp'] += (
                ('#include <cstdint>', '#include <cstdint>\n#include "api/debug/dprint.h"'),
                (HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
                (MASK, _mask_after(SKT)),
                (INIT, _init_after(SKT)),
                (SUBTRACT, _subtract_after(SKT)))
            return substitutions

        with patch.object(native_draft_sdpa, 'replacements', replacements):
            yield

    return scope()
