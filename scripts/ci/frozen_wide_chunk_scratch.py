"""Parameterized port of dspark_ladder_normalization.py's dedicated-scratch-CB
FP32 reciprocal candidate (hardware-validated at context 65536, run
34797353681, "Hardware 64K correctness passes: 34797353681",
docs/context-ladder-investigation.md:915-933), targeting the frozen-recipe's
own Skt value for the 65536 rung instead of the ladder module's Skt == 2112.

The ladder module reserves 1024 output rows in its own capacity formula
(dspark_ladder_geometry.py:7,12: output_tokens=1024), giving native_keys =
67584 (Skt == 2112). The fixed-storage frozen-recipe probe this module is
ported into (dspark-native-8k-attention-probe.py) reserves only 15 proposal
rows (capacity = context + 256, frozen_context_geometry.py:12), so the same
1024-key chunk width gives a different padded-key count: 66560 keys, Skt ==
2080 (see frozen_wide_chunk_normalization.py for the derivation). The
technique - a dedicated one-tile FP32 scratch CB for the reciprocal, unpacked
via UnpackToDestFp32 instead of sharing a CB with the BF16 numerator and
being silently narrowed to TF32 on unpack - is otherwise unchanged from the
validated candidate: same three anchors, same substitution shape, only the
literal Skt value differs.

Deliberately does NOT import or modify dspark_ladder_normalization.py: this
module is a parallel, independent copy so the already-hardware-qualified
ladder candidate (used by dspark-ladder-attention-probe.py /
dspark_ladder_build.py) is left completely untouched by this port.

UNVALIDATED at this Skt value: the technique is hardware-proven at Skt ==
2112 under the ladder's own geometry; applying it verbatim at Skt == 2080
under the frozen recipe's geometry has not itself been built, simulated or
run on hardware. See frozen_wide_chunk_normalization.py's module docstring
for the required next step (context-build-65536-v1) before any numerical or
hardware requalification.

CORRECTION, discovered while porting the sum-update/score-center fixes
(docs/numerics-65536-attention.md section 7): dspark_ladder_normalization.py
has TWO independent halves, and earlier revisions of this port carried only
the first. factory_transform() (above) patches sdpa_program_factory.cpp -
the FACTORY - to allocate the dedicated recip_scratch CB and set its unpack
mode. scratch_normalization() (the ladder's own name; ported below as
kernel_scope()) patches compute_common.hpp - the KERNEL - via
native_draft_sdpa.replacements, inserting the qwen_normalize_scratch helper
AND, critically, replacing the actual call site
(`mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(...)`) that performs
the final normalization multiply, so it routes through the dedicated CB
instead of the original shared one. Without this second half, the factory
allocates recip_scratch but nothing in the kernel ever reads or writes it -
the original, shared, TF32-truncating path stays in use regardless. This
means every prior evidence run of this port (v2, v3 - context-build-65536-v2
and v3, experiment/frozen-64k-reciprocal-replay-v3) never actually exercised
the scratch-CB technique at all; whatever improvement those runs showed
(384 -> 9 failed elements) is attributable entirely to the widened
fp32-stats condition (stats_df now Float32 at Sk_chunk_t==32, which never
matched before this port existed) and the reduced iteration count
(258 -> 65/66), not to the reciprocal fix this module is named for. kernel_scope()
below closes that gap.
"""


SKT = 2080


SUBSTITUTIONS = (
    (b'    if (use_streaming_compute) {\n        cb_ids.recip_scratch = allocate_tile_cb(1, im_tile_size, im_df);\n    }',
     b'    if (use_streaming_compute) {\n        cb_ids.recip_scratch = allocate_tile_cb(1, im_tile_size, im_df);\n'
     b'    } else if (qwen_draft_fp32_intermediates && Skt == ' + str(SKT).encode() + b') {\n'
     b'        cb_ids.recip_scratch = allocate_tile_cb(1, tt::tile_size(tt::DataFormat::Float32), tt::DataFormat::Float32);\n    }'),
    (b'    compute_desc.config = ComputeConfigDescriptor{',
     b'    std::vector<tt::tt_metal::UnpackToDestMode> qwen_normalization_modes;\n'
     b'    if (qwen_draft_fp32_intermediates && Skt == ' + str(SKT).encode() + b') {\n'
     b'        qwen_normalization_modes.resize(64, tt::tt_metal::UnpackToDestMode::Default);\n'
     b'        qwen_normalization_modes.at(cb_ids.recip_scratch) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;\n'
     b'    }\n    compute_desc.config = ComputeConfigDescriptor{'),
    (b'        .dst_full_sync_en = dst_full_sync_en,\n        .math_approx_mode = math_approx_mode,',
     b'        .dst_full_sync_en = dst_full_sync_en,\n'
     b'        .unpack_to_dest_mode = qwen_normalization_modes,\n        .math_approx_mode = math_approx_mode,'),
)


MARKER = b'qwen_normalization_modes'  # present iff the scratch-CB layer below is applied


def factory_transform(source, *, reverse=False):
    if not isinstance(source, bytes):
        raise ValueError('Explicit factory source bytes required')
    for before, after in (reversed(SUBSTITUTIONS) if reverse else SUBSTITUTIONS):
        if reverse:
            before, after = after, before
        if source.count(before) != 1:
            raise ValueError('Exact scratch allocation and FP32 unpack anchor required: ' + before[:64].decode())
        source = source.replace(before, after)
    return source


def factory_scope():
    """Context manager wrapping dspark_fp32_build.transform so the scratch-CB
    substitutions above apply on top of the base ANCHOR/REPLACEMENT swap
    during a build, and reverse cleanly during validate_manifest()'s
    reconstruction check.

    Only patches `transform` - not a `restore_factory_source` function, which
    does not exist at the pinned revision (8c102b20): dspark_fp32_build.py
    there inlines that reversal directly inside validate_manifest() as
    `original = source.replace(replacement.encode(), ANCHOR.encode())`, with
    no separate, patchable name for it. (The current working tree's own copy
    of dspark_fp32_build.py does define restore_factory_source, but this
    module is staged into, and must work against, the tree the pinned lane
    actually executes - the historical one.) Instead, the patched `transform`
    is made self-detecting: called with input that already carries the
    scratch-CB marker (validate_manifest's inlined reversal only ever strips
    the condition-text swap, never the scratch-CB layer, since it has no
    knowledge of it), it reverses that layer itself before doing anything
    else. This works identically whether the caller's own reversal step is a
    separate function, inlined, or absent, and requires no assumption about
    which shape of dspark_fp32_build.py is staged. Mirrors
    dspark_ladder_build.factory_scope()'s structure otherwise."""
    from contextlib import contextmanager
    from unittest.mock import patch
    import dspark_fp32_build as baseline

    original_transform = baseline.transform

    def selected(source, *, enabled=True):
        if enabled and MARKER in source:
            source = factory_transform(source, reverse=True)
        result = original_transform(source, enabled=enabled)
        if enabled:
            result = factory_transform(result)
        return result

    @contextmanager
    def scope():
        with patch.object(baseline, 'transform', selected):
            yield

    return scope()


KERNEL_HELPER = r'''
template <uint32_t columns>
void qwen_normalize_scratch(uint32_t numerator_cb, uint32_t reciprocal_cb,
                            uint32_t scratch_cb, uint32_t output_cb) {
    CircularBuffer(reciprocal_cb).wait_front(1);
    CircularBuffer(scratch_cb).reserve_back(1);
    reconfig_data_format_srca(reciprocal_cb);
    copy_tile_init(reciprocal_cb);
    pack_reconfig_data_format(scratch_cb);
    tile_regs_acquire();
    copy_tile(reciprocal_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch_cb);
    tile_regs_release();
    CircularBuffer(scratch_cb).push_back(1);
    CircularBuffer(scratch_cb).wait_front(1);
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    const auto source_address = get_local_cb_interface(reciprocal_cb).fifo_rd_ptr << cb_addr_shift;
    const auto scratch_address = get_local_cb_interface(scratch_cb).fifo_rd_ptr << cb_addr_shift;
    const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(source_address);
    volatile uint32_t* scratch = reinterpret_cast<volatile uint32_t*>(scratch_address);
    for (uint32_t index = 0; index < 1024; ++index) {
        scratch[index] = source[index];
    }
#endif
    CircularBuffer(numerator_cb).wait_front(columns);
    CircularBuffer(output_cb).reserve_back(columns);
    pack_reconfig_data_format(output_cb);
    PACK((llk_pack_reconfig_l1_acc(false)));
    sfpu_mul_bcast_col_init();
    for (uint32_t tile = 0; tile < columns; ++tile) {
        tile_regs_acquire();
        reconfig_data_format_srca(numerator_cb);
        copy_tile_init(numerator_cb);
        copy_tile(numerator_cb, tile, 0);
        reconfig_data_format_srca(scratch_cb);
        copy_tile_init(scratch_cb);
        copy_tile(scratch_cb, 0, 1);
        sfpu_mul_bcast_col(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output_cb);
        tile_regs_release();
    }
    CircularBuffer(numerator_cb).pop_front(columns);
    CircularBuffer(reciprocal_cb).pop_front(1);
    CircularBuffer(scratch_cb).pop_front(1);
    CircularBuffer(output_cb).push_back(columns);
}
'''
KERNEL_FINAL_CALL = ('            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>'
    '(alias_mm2_prev_out, alias_prev_sum, cb_out);')


def kernel_scope():
    """Port of dspark_ladder_normalization.py's scratch_normalization() -
    the kernel-level half this module was missing until now (see the module
    docstring's CORRECTION). Patches native_draft_sdpa.replacements to insert
    KERNEL_HELPER and replace the final normalization multiply's call site so
    it routes through the dedicated scratch CB (allocated by factory_transform
    above, at compile-time-arg index 42 - the same fixed slot the ladder's own
    validated candidate uses, since factory_transform's insertion point and
    structure are byte-identical to the ladder's, only the Skt guard literal
    differs) when the resolved Skt matches, gated by the module-level SKT
    constant (patched to match at staging time, same as factory_transform's
    embedded literal). Composed into dspark-native-8k-attention-probe.py's
    main() alongside scalar_reciprocal()/scalar_sum_update()/
    scalar_score_center(), in that order - see
    frozen_wide_chunk_normalization.py's _patch_probe()."""
    from contextlib import contextmanager
    from unittest.mock import patch
    import native_draft_sdpa

    @contextmanager
    def scope():
        original = native_draft_sdpa.replacements

        def replacements():
            substitutions = original()
            substitutions['compute_common.hpp'] += (
                ('#include "api/compute/bcast.h"',
                 '#include "api/compute/bcast.h"\n#include "api/compute/sfpu_binary_bcast.h"'),
                ('enum SDPAType {', KERNEL_HELPER + '\nenum SDPAType {'),
                (KERNEL_FINAL_CALL,
                 f'            if constexpr (!QWEN_DRAFT_EXP_APPROX && get_compile_time_arg_val(3) == {SKT}) {{\n'
                 '                static_assert(Sq_chunk_t == 1);\n'
                 '                qwen_normalize_scratch<vDHt>(alias_mm2_prev_out, alias_prev_sum, '
                 'get_compile_time_arg_val(42), cb_out);\n'
                 '            } else {\n' + KERNEL_FINAL_CALL + '\n            }'))
            return substitutions

        with patch.object(native_draft_sdpa, 'replacements', replacements):
            yield

    return scope()


def validate_manifest(root, output):
    """Drop-in replacement for dspark_fp32_build.validate_manifest, re-entering
    factory_scope() for the duration of the call so the reconstruction check
    inside it (source.replace(...) then transform(), inlined at the pinned
    revision - see factory_scope()'s docstring) sees and correctly
    reverses/reapplies the scratch-CB substitutions, whichever process or
    call site invokes it (the build-time call in frozen_sim_build_cache.py
    and the probe's own runtime calls are separate Python processes; each
    must re-enter this scope independently since unittest.mock.patch does
    not persist across process boundaries). Only the 65536-context staged
    dspark-native-8k-attention-probe.py imports this in place of
    dspark_fp32_build.validate_manifest; every other context is unaffected."""
    import dspark_fp32_build as baseline
    with factory_scope():
        return baseline.validate_manifest(root, output)
