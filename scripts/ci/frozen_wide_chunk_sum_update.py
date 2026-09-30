"""Parameterized port of dspark_ladder_sum_update.py's scalar_sum_update()
candidate - a TR0 scalar-FP32 correction to the running-sum (denominator)
update, replacing the kernel's bf16 `mul_tiles_bcast_cols_inplace` +
`add_block_inplace` combine with a per-row, per-column FP32
`current[offset] += previous[offset] * correction` loop over the raw
partial-sum CBs.

The ladder module hardcodes this to `get_compile_time_arg_val(3) == 2112`
(Skt) in the AFTER text (dspark_ladder_sum_update.py:38) and takes no
parameter at all - scalar_sum_update() has no key_tiles/skt argument,
unlike scalar_score_center(). So unlike frozen_wide_chunk_score_center.py
(which reuses the ladder's own function, since it IS parameterized), this
module is a parallel, independent copy of the substitution text with the
literal Skt swapped for the module-level SKT constant, mirroring how
frozen_wide_chunk_scratch.py already handles factory_transform(). Does NOT
import or modify dspark_ladder_sum_update.py itself.

The condition text itself is a clean parameter (only the compile-time-arg
comparison literal differs), so this is generated for both accepted Skt
values (2080 and 2112), not refused for 2080.

UNVALIDATED at this Skt value / in this composition: hardware-proven only as
part of the ladder's own four-fix combination at Skt==2112
(run 34797353681); never run in isolation, and never run combined with this
port's other three fixes until a build+requalification cycle confirms it.
"""


SKT = 2080


BEFORE = ('                mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);\n'
    '\n'
    '                /* cb_cur_sum += cb_prev_sum */\n'
    '                add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);')
BODY = '''
                    CircularBuffer(alias_prev_sum).wait_front(Sq_chunk_t);
                    CircularBuffer(alias_cur_sum).wait_front(Sq_chunk_t);
                    CircularBuffer(cb_exp_max_diff).wait_front(Sq_chunk_t);
                    for (uint32_t tile = 0; tile < Sq_chunk_t; ++tile) {
                        const auto previous_cb = get_local_cb_interface(alias_prev_sum);
                        const auto current_cb = get_local_cb_interface(alias_cur_sum);
                        const auto factor_cb = get_local_cb_interface(cb_exp_max_diff);
                        const volatile float* previous = reinterpret_cast<const volatile float*>(
                            (previous_cb.fifo_rd_ptr + tile * previous_cb.fifo_page_size) << cb_addr_shift);
                        volatile float* current = reinterpret_cast<volatile float*>(
                            (current_cb.fifo_rd_ptr + tile * current_cb.fifo_page_size) << cb_addr_shift);
                        const volatile float* factor = reinterpret_cast<const volatile float*>(
                            (factor_cb.fifo_rd_ptr + tile * factor_cb.fifo_page_size) << cb_addr_shift);
                        for (uint32_t row = 0; row < 32; ++row) {
                            const uint32_t first = row < 16 ? row * 16 : 512 + (row - 16) * 16;
                            const float correction = factor[first];
                            for (uint32_t column = 0; column < 32; ++column) {
                                const uint32_t offset = first + (column < 16 ? column : 256 + column - 16);
                                current[offset] = current[offset] + previous[offset] * correction;
                            }
                        }
                    }
                    CircularBuffer(alias_prev_sum).pop_front(Sq_chunk_t);
'''


def sum_update_scope():
    """Composed into dspark-native-8k-attention-probe.py's main() right
    after scalar_reciprocal(), before scalar_score_center() - see
    frozen_wide_chunk_normalization.py's _patch_probe() and
    docs/numerics-65536-attention.md section 7 for the ladder's own
    composition order this mirrors."""
    from contextlib import contextmanager
    from unittest.mock import patch
    import native_draft_sdpa

    after = (f'                if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == {SKT}) {{\n'
        '#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0\n' + BODY + '#endif\n'
        '                } else {\n' + BEFORE + '\n                }')

    @contextmanager
    def scope():
        original = native_draft_sdpa.replacements

        def replacements():
            substitutions = original()
            substitutions['compute_common.hpp'] += ((BEFORE, after),)
            return substitutions

        with patch.object(native_draft_sdpa, 'replacements', replacements):
            yield

    return scope()
