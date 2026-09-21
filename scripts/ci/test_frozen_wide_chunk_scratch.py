"""Tests for frozen_wide_chunk_scratch.py's scratch-CB substitution and its
factory_scope()/validate_manifest() round-trip.

FactoryScopeRoundTripTests exercises the logic against a synthetic factory
source containing the three anchor sites, using fake dspark_fp32_build /
dspark_fp32_intermediates modules shaped EXACTLY like the pinned revision
(8c102b20) - no separate restore_factory_source function; validate_manifest
inlines that reversal directly as `source.replace(replacement.encode(),
ANCHOR.encode())`. An earlier version of these fakes had a
restore_factory_source function (matching the CURRENT WORKING TREE's copy of
dspark_fp32_build.py, not the pinned one the lane actually executes) and so
did not catch factory_scope() patching an attribute that doesn't exist at
8c102b20 - exactly the AttributeError run 35591662531 hit. RealStagedModuleTests
below closes that gap for real: it imports the actual, staged dspark_fp32_build
module from a real dry-run checkout, not a hand-built fake, so any future
mismatch between what factory_scope() assumes and what the pinned module
actually exports fails here instead of in a build lane.
"""

import hashlib
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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
    test, shaped exactly like the pinned revision (8c102b20): transform is
    module-level on both; validate_manifest INLINES the ANCHOR/REPLACEMENT
    reversal (`source.replace(replacement.encode(), ANCHOR.encode())`)
    directly in its own body rather than calling out to a separate,
    patchable restore_factory_source function - because there isn't one."""
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
    # No restore_factory_source attribute on this module at all - matching
    # 8c102b20 exactly. If frozen_wide_chunk_scratch.factory_scope() ever
    # tries to patch.object(build, 'restore_factory_source', ...) again,
    # these tests reproduce the real AttributeError immediately.

    def validate_manifest(root, output):
        enabled = True
        replacement = build.REPLACEMENT if enabled else build.REPLACEMENT.replace(
            'qwen_draft_fp32_intermediates =\n', 'qwen_draft_fp32_intermediates = false &&\n')
        source = build._current_built_source
        if source.count(replacement.encode()) != 1:
            raise ValueError('Unique rebuilt factory variant required')
        original = source.replace(replacement.encode(), build.ANCHOR.encode())  # inlined, as at 8c102b20
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

    def test_entering_scope_does_not_touch_a_nonexistent_attribute(self):
        """The regression test for the actual bug: entering/exiting
        factory_scope() against a module with no restore_factory_source at
        all must not raise AttributeError."""
        with self.scratch.factory_scope():
            pass

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


REVISION = '8c102b20df22329106955b4006bf4d650bb94e40'
ROOT = Path(__file__).resolve().parents[2]


class RealStagedModuleTestsBase(unittest.TestCase):
    """Imports the ACTUAL dspark_fp32_build module from a real dry-run
    checkout staged for context 65536 under this class's KNOB value
    (frozen_recipe_context.main(), the same technique used elsewhere in this
    port) - not a hand-built fake - and exercises
    frozen_wide_chunk_scratch.factory_scope() against it directly. This is
    what would have caught run 35591662531's AttributeError: that bug was
    invisible to FactoryScopeRoundTripTests above only because its OLD fakes
    (before this file's fix) had a restore_factory_source function the real,
    pinned module does not. Concrete subclasses below set KNOB to each
    accepted QWEN_FROZEN_65536_SKT value, so both staged-module round trips
    run for real, not just one.

    Does not attempt to run dspark_fp32_build.main() or
    frozen_sim_build_cache.main() themselves: both hardcode absolute paths
    (/opt/tt-metal, /experiment/results/...) and dspark_fp32_build.main()
    additionally calls restore_registrations(), which hashes a real tt-metal
    checkout's implementation-source files via sdpa_graft_build.audit -
    reproducing that structure would mean building a large, fragile mock of
    a real tt-metal tree, which risks false confidence more than it buys
    coverage. What actually matters for this bug class - whether
    factory_scope()'s patching matches the real module's real exported
    symbols - is exercised directly and completely below, against
    dspark_fp32_build.transform and .validate_manifest, the two functions
    factory_scope() wraps."""

    KNOB = None  # set by concrete subclasses

    @classmethod
    def setUpClass(cls):
        if cls is RealStagedModuleTestsBase:
            raise unittest.SkipTest('base class; see the per-knob subclasses below')
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        checkout = Path(cls._tmp.name) / 'checkout'
        subprocess.run(['git', 'clone', '--quiet', str(ROOT), str(checkout)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--quiet', REVISION], check=True)
        manifest = Path(cls._tmp.name) / 'manifest.json'
        subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('frozen_recipe_context.py')),
            '--checkout', str(checkout), '--context', '65536', '--target-replay', '--scalar-reciprocal',
            '--probe-seconds', '1020', '--manifest', str(manifest)],
            env=dict(os.environ, QWEN_FROZEN_65536_SKT=cls.KNOB), check=True)
        cls.staged = checkout / 'scripts/ci'

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        # Import fresh from the staged directory each test, so no earlier
        # test's sys.modules entry (or this process's own working-tree copy,
        # imported by other test modules in the same run) can substitute for
        # the actual staged/pinned module under test.
        for name in ('dspark_fp32_build', 'dspark_fp32_intermediates', 'frozen_wide_chunk_scratch',
                'dspark_hardware_gate', 'sdpa_graft_build'):
            sys.modules.pop(name, None)
        self._old_path = list(sys.path)
        sys.path.insert(0, str(self.staged))
        import dspark_fp32_build as staged_build
        import frozen_wide_chunk_scratch as staged_scratch
        self.build, self.scratch = staged_build, staged_scratch

    def tearDown(self):
        sys.path[:] = self._old_path
        for name in ('dspark_fp32_build', 'dspark_fp32_intermediates', 'frozen_wide_chunk_scratch',
                'dspark_hardware_gate', 'sdpa_graft_build'):
            sys.modules.pop(name, None)

    def test_staged_dspark_fp32_build_has_no_restore_factory_source(self):
        """Documents the actual shape being tested against, so this test
        file itself proves the bug's precondition rather than assuming it."""
        self.assertFalse(hasattr(self.build, 'restore_factory_source'))
        self.assertTrue(hasattr(self.build, 'transform'))
        self.assertTrue(hasattr(self.build, 'validate_manifest'))

    def test_staged_scratch_module_skt_matches_this_class_knob(self):
        """Proves the QWEN_FROZEN_65536_SKT env var set for this class's
        staging run actually landed in the staged frozen_wide_chunk_scratch.py
        - not just that staging succeeded with some value."""
        self.assertEqual(self.scratch.SKT, int(self.KNOB))

    def test_entering_factory_scope_against_the_real_staged_module_does_not_raise(self):
        """The exact call frozen_sim_build_cache.py makes
        (`with frozen_wide_chunk_scratch.factory_scope(), patch.object(...)`)
        reduced to its first step. This alone reproduces run 35591662531's
        AttributeError against the old factory_scope() implementation."""
        with self.scratch.factory_scope():
            pass
        # And leaves no patch behind afterwards.
        self.assertNotEqual(self.build.transform.__module__, self.scratch.__name__)

    def test_transform_and_validate_manifest_round_trip_on_the_real_staged_module(self):
        """Monkeypatches only what a real tt-metal checkout would otherwise
        supply (SOURCE_SHA256/ANCHOR/REPLACEMENT and the report file the real
        build process writes) onto the REAL staged modules, then calls their
        REAL transform and validate_manifest functions under factory_scope()
        - so this test's assertions are about the pinned modules' actual code
        path, not a description of it.

        Patches both dspark_fp32_build and dspark_fp32_intermediates: transform
        is imported by name into dspark_fp32_build (`from dspark_fp32_intermediates
        import ... transform`), but its function BODY still resolves
        SOURCE_SHA256/ANCHOR/REPLACEMENT against dspark_fp32_intermediates's own
        module globals (where it is defined), not the caller's - patching only
        dspark_fp32_build's copies of those names has no effect on it."""
        import dspark_fp32_intermediates as staged_intermediates
        with patch.object(staged_intermediates, 'SOURCE_SHA256', SOURCE_SHA256), \
                patch.object(staged_intermediates, 'ANCHOR', ANCHOR), \
                patch.object(staged_intermediates, 'REPLACEMENT', REPLACEMENT), \
                patch.object(self.build, 'SOURCE_SHA256', SOURCE_SHA256), \
                patch.object(self.build, 'ANCHOR', ANCHOR), \
                patch.object(self.build, 'REPLACEMENT', REPLACEMENT):
            with self.scratch.factory_scope():
                built = self.build.transform(PRISTINE, enabled=True)
            self.assertIn(f'Skt == {self.scratch.SKT}'.encode(), built)
            self.assertNotIn(ANCHOR.encode(), built)

            import json
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / self.build.SOURCE).parent.mkdir(parents=True, exist_ok=True)
                (root / self.build.SOURCE).write_bytes(built)
                output = root / 'dspark-fp32-build.json'
                output.write_text(json.dumps(dict(passed=True, source_before=SOURCE_SHA256, factory_enabled=True,
                    source_after=hashlib.sha256(built).hexdigest())))
                with self.scratch.factory_scope():
                    with self.assertRaises(ValueError) as caught:
                        self.build.validate_manifest(root, output)
                # Reaches the binaries_after check (past the reconstruction
                # check this test targets) rather than failing reconstruction
                # itself - proving the scratch-CB layer round-tripped.
                self.assertIn('binar', str(caught.exception).lower())


class RealStagedModuleTestsSkt2080(RealStagedModuleTestsBase):
    """This port's own first derivation (frozen recipe's context+256 basis)."""
    KNOB = '2080'


class RealStagedModuleTestsSkt2112(RealStagedModuleTestsBase):
    """The ladder's own Skt - the value with an actual zero-failure hardware
    result behind it (run 34797353681), and the v4 lane's default."""
    KNOB = '2112'


if __name__ == '__main__':
    unittest.main()
