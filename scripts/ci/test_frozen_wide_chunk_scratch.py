"""Tests for frozen_wide_chunk_scratch.py's scratch-CB substitution and its
factory_scope()/validate_manifest() round-trip, against a synthetic factory
source containing the three anchor sites (not the real, unreadable-without-
tt-metal C++ file). This is the highest-risk piece of the wide-chunk port:
dspark_fp32_build.validate_manifest()'s self-consistency check
(restore_factory_source then re-transform, and compare to the built source)
was not designed to know about a second, independent set of substitutions
layered on top of its own ANCHOR/REPLACEMENT swap. These tests prove the
factory_scope() wrapper reverses and reapplies both layers correctly, in the
same order they were applied, and that skipping factory_scope() fails loudly
rather than silently miscomparing.
"""

import hashlib
import sys
import types
import unittest


ANCHOR = '''    tt::DataFormat im_df =
        tt::DataFormat::Float16_b;  // Keep most intermediates in bf16 to save L1; opt-in fp32 per-CB below.
    tt::DataFormat stats_df = im_df;'''
REPLACEMENT = '''    const bool qwen_draft_fp32_intermediates =
        B == 1 && NQH == 16 && NKH == 4 && DHt == 4 && vDHt == 4 &&
        Skt == 272 && Sq_chunk_t == 1 && Sk_chunk_t == 8 &&
        !is_causal && compute_use_provided_mask && !is_chunked &&
        !use_attention_sink && !is_windowed && !use_streaming_compute &&
        fp32_dest_acc_en && !exp_approx_mode;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = qwen_draft_fp32_intermediates ? tt::DataFormat::Float32 : im_df;'''

PRISTINE = (
    b'// filler before\n' + ANCHOR.encode() + b'\n// filler after\n'
    b'    if (use_streaming_compute) {\n        cb_ids.recip_scratch = allocate_tile_cb(1, im_tile_size, im_df);\n    }\n'
    b'    compute_desc.config = ComputeConfigDescriptor{\n'
    b'        .dst_full_sync_en = dst_full_sync_en,\n        .math_approx_mode = math_approx_mode,\n'
    b'    };\n'
)
SOURCE_SHA256 = hashlib.sha256(PRISTINE).hexdigest()


def _install_fakes():
    """Fresh fake dspark_fp32_intermediates / dspark_fp32_build modules per
    test, mirroring the real modules' relevant shape exactly (transform,
    restore_factory_source, REPLACEMENT/ANCHOR), so frozen_wide_chunk_scratch
    (imported once, real) can be exercised against them without any real
    tt-metal checkout."""
    intermediates = types.ModuleType('dspark_fp32_intermediates')
    intermediates.SOURCE = 'fake/source.cpp'
    intermediates.SOURCE_SHA256 = SOURCE_SHA256
    intermediates.ANCHOR = ANCHOR
    intermediates.REPLACEMENT = REPLACEMENT

    def transform(source, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('Explicit factory variant selection required')
        if not isinstance(source, bytes) or hashlib.sha256(source).hexdigest() != intermediates.SOURCE_SHA256:
            raise ValueError('Exact pinned SDPA factory required')
        if source.count(intermediates.ANCHOR.encode()) != 1:
            raise ValueError('Unique intermediate-format anchor required')
        replacement = intermediates.REPLACEMENT if enabled else intermediates.REPLACEMENT.replace(
            'qwen_draft_fp32_intermediates =\n', 'qwen_draft_fp32_intermediates = false &&\n')
        return source.replace(intermediates.ANCHOR.encode(), replacement.encode())

    intermediates.transform = transform
    sys.modules['dspark_fp32_intermediates'] = intermediates

    build = types.ModuleType('dspark_fp32_build')
    build.SOURCE, build.SOURCE_SHA256 = intermediates.SOURCE, intermediates.SOURCE_SHA256
    build.ANCHOR, build.REPLACEMENT, build.transform = intermediates.ANCHOR, intermediates.REPLACEMENT, transform

    def restore_factory_source(source, replacement):
        return source.replace(replacement.encode(), build.ANCHOR.encode())

    build.restore_factory_source = restore_factory_source

    def validate_manifest(root, output):
        enabled = True
        replacement = build.REPLACEMENT if enabled else build.REPLACEMENT.replace(
            'qwen_draft_fp32_intermediates =\n', 'qwen_draft_fp32_intermediates = false &&\n')
        source = build._current_built_source
        if source.count(replacement.encode()) != 1:
            raise ValueError('Unique rebuilt factory variant required')
        original = build.restore_factory_source(source, replacement)
        if build.transform(original, enabled=enabled) != source:
            raise ValueError('Rebuilt factory differs from exact transformation')
        return dict(passed=True)

    build.validate_manifest = validate_manifest
    sys.modules['dspark_fp32_build'] = build
    return intermediates, build


class FactoryScopeRoundTripTests(unittest.TestCase):

    def setUp(self):
        self.intermediates, self.build = _install_fakes()
        sys.modules.pop('frozen_wide_chunk_scratch', None)
        import frozen_wide_chunk_scratch
        self.scratch = frozen_wide_chunk_scratch

    def tearDown(self):
        for name in ('dspark_fp32_intermediates', 'dspark_fp32_build', 'frozen_wide_chunk_scratch'):
            sys.modules.pop(name, None)

    def test_factory_transform_forward_reverse_is_identity(self):
        forward = self.scratch.factory_transform(PRISTINE)
        self.assertNotEqual(forward, PRISTINE)
        self.assertIn(f'Skt == {self.scratch.SKT}'.encode(), forward)
        reversed_back = self.scratch.factory_transform(forward, reverse=True)
        self.assertEqual(reversed_back, PRISTINE)

    def test_factory_transform_reports_missing_anchor(self):
        mangled = PRISTINE.replace(b'compute_desc.config', b'compute_desc_config')
        with self.assertRaises(ValueError):
            self.scratch.factory_transform(mangled)

    def test_build_time_transform_applies_both_layers_in_order(self):
        with self.scratch.factory_scope():
            built = self.build.transform(PRISTINE, enabled=True)
        # Condition swap present (base layer)...
        self.assertIn(b'qwen_draft_fp32_intermediates', built)
        # ...and scratch-CB layer present, applied on top of it.
        self.assertEqual(built.count(f'Skt == {self.scratch.SKT}'.encode()), 2)
        self.assertNotIn(ANCHOR.encode(), built)

    def test_validate_manifest_reconstructs_build_time_output_exactly(self):
        with self.scratch.factory_scope():
            built = self.build.transform(PRISTINE, enabled=True)
        self.build._current_built_source = built
        report = self.scratch.validate_manifest('/fake/root', '/fake/output')
        self.assertEqual(report, dict(passed=True))

    def test_validate_manifest_without_factory_scope_fails_loudly(self):
        with self.scratch.factory_scope():
            built = self.build.transform(PRISTINE, enabled=True)
        self.build._current_built_source = built
        with self.assertRaises(ValueError):
            self.build.validate_manifest('/fake/root', '/fake/output')

    def test_independent_factory_scope_entries_are_deterministic(self):
        """Simulates the build-time call and the probe's later, separate-
        process call producing identical results from independently entered
        scopes - the property frozen_wide_chunk_scratch.validate_manifest's
        docstring depends on."""
        with self.scratch.factory_scope():
            built_first = self.build.transform(PRISTINE, enabled=True)
        with self.scratch.factory_scope():
            built_second = self.build.transform(PRISTINE, enabled=True)
        self.assertEqual(built_first, built_second)

    def test_scope_does_not_leak_patches_after_exit(self):
        original_transform = self.build.transform
        with self.scratch.factory_scope():
            pass
        self.assertIs(self.build.transform, original_transform)


if __name__ == '__main__':
    unittest.main()
