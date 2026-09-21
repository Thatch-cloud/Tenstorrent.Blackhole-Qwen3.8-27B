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
