"""Tests for the 65536-only wide-chunk normalization port.

Covers: the geometry derivation for both accepted Skt values, the
QWEN_FROZEN_65536_SKT knob (default, explicit selection, invalid refusal),
that every substitution targets the actual post-adapt_probe_sources() text
(not a guess) under EACH knob value, that the generated
dspark_fp32_intermediates.py text carries exactly one Skt branch for 65536
per staging, that every other context is an exact no-op regardless of the
knob, the manifest geometry-honesty override, and that the resulting
per-context source text (and therefore the content-addressed build-cache
key) differs between 65536 and 32768 - the property that makes a manual
binary pin unnecessary (see the module docstring in
frozen_wide_chunk_normalization.py and the port report).
"""

import hashlib
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from frozen_context_geometry import CONTEXTS
import frozen_wide_chunk_normalization as wide
from frozen_recipe_context import REVISION, adapt_cache_launcher, adapt_probe_sources


ROOT = Path(__file__).resolve().parents[2]


def historical(name):
    """Exact historical source for scripts/ci/<name> at the pinned frozen
    revision, the same content frozen_recipe_context.main() reads via
    `git show {REVISION}:scripts/ci/{name}`."""
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{REVISION}:scripts/ci/{name}'],
        check=True, capture_output=True, text=True).stdout


def adapted_sources(context, probe_seconds=1020):
    """Reproduces the exact pipeline state frozen_wide_chunk_normalization
    receives inside frozen_recipe_context.main(): adapt_probe_sources() then
    adapt_cache_launcher() on real historical file content (before any
    scalar-reciprocal/target-replay/combined-runtime adapters), plus
    frozen_sim_build_cache.py - staged verbatim from the working tree by
    main()'s later unconditional-copy loop, which runs before this module's
    adapter (frozen_recipe_context.py wires it in right after that loop)."""
    names = ('dspark_attention_chunk_trial.py', 'dspark-native-8k-attention-probe.py',
        'dspark_stats_pack.py', 'dspark_fp32_intermediates.py', 'run-simulator.sh', 'simulator-suite.sh')
    sources = {name: historical(name) for name in names}
    result = adapt_cache_launcher(adapt_probe_sources(sources, context), probe_seconds)
    result['frozen_sim_build_cache.py'] = Path(__file__).with_name('frozen_sim_build_cache.py').read_text()
    return result


class GeometryTests(unittest.TestCase):

    def test_module_level_constants_never_vary_by_knob(self):
        self.assertEqual(wide.CONTEXT, 65536)
        self.assertEqual(wide.CAPACITY, 65792)
        self.assertEqual(wide.STORAGE_KEYS, 65856)
        self.assertEqual(wide.KEY_CHUNK, 1024)
        self.assertEqual(wide.SK_CHUNK_T, 32)

    def test_geometry_for_2080(self):
        self.assertEqual(wide.geometry_for_skt(2080), dict(
            skt=2080, padded_keys=66560, key_chunk=1024, sk_chunk_t=32, iterations=65, poison_rows=704))

    def test_geometry_for_2112(self):
        self.assertEqual(wide.geometry_for_skt(2112), dict(
            skt=2112, padded_keys=67584, key_chunk=1024, sk_chunk_t=32, iterations=66, poison_rows=1728))

    def test_both_geometries_have_exact_multiple_padded_keys(self):
        for skt in wide.ACCEPTED_SKT:
            with self.subTest(skt=skt):
                geometry = wide.geometry_for_skt(skt)
                self.assertEqual(geometry['padded_keys'] % geometry['key_chunk'], 0)

    def test_2080_is_this_ports_own_derivation_2112_is_the_ladders(self):
        # 2064 is the currently-generated (failing) Skt at the standard
        # 256-key chunk under this same context+256 capacity formula - must
        # not equal either accepted value.
        self.assertNotIn(2064, wide.ACCEPTED_SKT)

    def test_rejects_unsupported_skt(self):
        for bad in (2064, 2111, 2113, 0, -2112, 1040):
            with self.subTest(skt=bad):
                with self.assertRaises(ValueError):
                    wide.geometry_for_skt(bad)


class KnobResolutionTests(unittest.TestCase):

    def test_default_is_2112(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(wide.KNOB, None)
            self.assertEqual(wide.resolve_skt(), 2112)
        self.assertEqual(wide.DEFAULT_SKT, 2112)

    def test_explicit_2080_selects_this_ports_own_geometry(self):
        with patch.dict(os.environ, {wide.KNOB: '2080'}):
            self.assertEqual(wide.resolve_skt(), 2080)

    def test_explicit_2112_selects_the_ladders_geometry(self):
        with patch.dict(os.environ, {wide.KNOB: '2112'}):
            self.assertEqual(wide.resolve_skt(), 2112)

    def test_invalid_values_refused(self):
        for bad in ('2064', '2111', '', 'abc', '2112 ', '65536'):
            with self.subTest(value=bad):
                with patch.dict(os.environ, {wide.KNOB: bad}):
                    with self.assertRaises(ValueError):
                        wide.resolve_skt()


class NoOpForOtherContextsTests(unittest.TestCase):

    def test_returns_equal_dict_for_every_context_other_than_65536_regardless_of_knob(self):
        sources = dict(a='x', b='y')
        for knob in ('2080', '2112', None):
            with patch.dict(os.environ, {} if knob is None else {wide.KNOB: knob}):
                if knob is None:
                    os.environ.pop(wide.KNOB, None)
                for context in CONTEXTS:
                    if context == wide.CONTEXT:
                        continue
                    with self.subTest(context=context, knob=knob):
                        result = wide.adapt_wide_chunk_normalization(sources, context)
                        self.assertEqual(result, sources)
                        self.assertIsNot(result, sources)  # returns a copy, never the same object

    def test_32768_real_pipeline_output_is_byte_identical_regardless_of_knob(self):
        without_wide_chunk = adapted_sources(32768)
        for knob in ('2080', '2112'):
            with self.subTest(knob=knob), patch.dict(os.environ, {wide.KNOB: knob}):
                with_wide_chunk_noop = wide.adapt_wide_chunk_normalization(dict(without_wide_chunk), 32768)
                self.assertEqual(with_wide_chunk_noop, without_wide_chunk)
                for name in without_wide_chunk:
                    self.assertEqual(hashlib.sha256(with_wide_chunk_noop[name].encode()).hexdigest(),
                        hashlib.sha256(without_wide_chunk[name].encode()).hexdigest(), name)


class ManifestGeometryOverrideTests(unittest.TestCase):

    def test_none_for_every_other_context(self):
        for context in CONTEXTS:
            if context == wide.CONTEXT:
                continue
            with self.subTest(context=context):
                self.assertIsNone(wide.manifest_geometry_override(context))

    def test_65536_reports_the_resolved_geometry_not_the_standard_formula(self):
        for knob, expected in (('2080', dict(padded_keys=66560, key_chunk=1024)),
                ('2112', dict(padded_keys=67584, key_chunk=1024))):
            with self.subTest(knob=knob), patch.dict(os.environ, {wide.KNOB: knob}):
                self.assertEqual(wide.manifest_geometry_override(65536), expected)


class WideChunkPatchTests(unittest.TestCase):
    """Exercise each _patch_* function against the REAL post-adapt_probe_sources()
    text (fetched from the pinned historical revision), not a hand-written
    guess at what that text looks like - anchor drift shows up as a
    ValueError from _once(), which these tests would surface immediately.
    Parameterized over both accepted Skt values."""

    @classmethod
    def setUpClass(cls):
        cls.sources = adapted_sources(65536)

    def _geometries(self):
        return [wide.geometry_for_skt(skt) for skt in wide.ACCEPTED_SKT]

    def test_probe_patch_report_literals_and_validate_manifest_redirect(self):
        for geometry in self._geometries():
            with self.subTest(skt=geometry['skt']):
                patched = wide._patch_probe(self.sources['dspark-native-8k-attention-probe.py'], geometry)
                self.assertIn(f"key_chunk_size={geometry['key_chunk']}", patched)
                self.assertIn(f"native_padded_keys={geometry['padded_keys']}", patched)
                self.assertIn(f"added_masked_poison_rows={geometry['poison_rows']}", patched)
                self.assertNotIn('key_chunk_size=256', patched)
                self.assertNotIn("native_padded_keys=SHAPE['padded_keys']", patched)
                self.assertEqual(patched.count('from frozen_wide_chunk_scratch import validate_manifest'), 2)
                self.assertNotIn('from dspark_fp32_build import validate_manifest', patched)
                self.assertIn('from frozen_wide_chunk_sum_update import sum_update_scope', patched)
                self.assertIn('from frozen_wide_chunk_score_center import score_center_scope', patched)
                self.assertIn('from frozen_wide_chunk_scratch import kernel_scope', patched)
                self.assertIn('sum_update_scope(), score_center_scope(), kernel_scope(), scoped_stats_pack(),',
                    patched)
                compile(patched, 'dspark-native-8k-attention-probe.py', 'exec')

    def test_probe_patch_works_without_scalar_reciprocal_too(self):
        """context==65536 staging runs unconditionally for any flag
        combination frozen_recipe_context.py accepts - including the real
        experiment/frozen-64k-target-replay-* lanes, which do not pass
        --scalar-reciprocal (qwen-frozen-32k-numerical.yml's sequential,
        non-elif if-blocks: only the reciprocal-eager/replay/diagnostics
        patterns add that flag). _patch_probe must not assume
        adapt_scalar_reciprocal() already ran."""
        source = self.sources['dspark-native-8k-attention-probe.py']
        self.assertNotIn('scalar_reciprocal()', source)  # precondition: --scalar-reciprocal was not applied here
        geometry = wide.geometry_for_skt(2112)
        patched = wide._patch_probe(source, geometry)
        self.assertIn('with sum_update_scope(), score_center_scope(), kernel_scope(), scoped_stats_pack(),', patched)
        compile(patched, 'dspark-native-8k-attention-probe.py', 'exec')

    def test_chunk_trial_patch(self):
        for geometry in self._geometries():
            with self.subTest(skt=geometry['skt']):
                patched = wide._patch_chunk_trial(self.sources['dspark_attention_chunk_trial.py'], geometry)
                self.assertIn(f"KEY_CHUNK = {geometry['key_chunk']}", patched)
                self.assertIn(f"PADDED_KEYS = {geometry['padded_keys']}", patched)
                self.assertNotIn('KEY_CHUNK = 256', patched)
                compile(patched, 'dspark_attention_chunk_trial.py', 'exec')

    def test_stats_pack_patch_assert_and_message(self):
        for geometry in self._geometries():
            with self.subTest(skt=geometry['skt']):
                patched = wide._patch_stats_pack(self.sources['dspark_stats_pack.py'], geometry)
                self.assertIn(f"get_compile_time_arg_val(3) == {geometry['skt']} && "
                    f"get_compile_time_arg_val(8) == {geometry['sk_chunk_t']}", patched)
                self.assertIn(f"{geometry['padded_keys']} keys and {geometry['key_chunk']}-key chunks", patched)
                self.assertNotIn('== 272 &&', patched)
                self.assertNotIn('8704 keys and 256-key chunks', patched)
                compile(patched, 'dspark_stats_pack.py', 'exec')

    def test_fp32_intermediates_patch_widens_condition_without_touching_factory_selector_call(self):
        for geometry in self._geometries():
            with self.subTest(skt=geometry['skt']):
                patched = wide._patch_fp32_intermediates(self.sources['dspark_fp32_intermediates.py'], geometry)
                self.assertIn('{factory_selector()}', patched)  # call site itself untouched
                self.assertIn(f"(Skt == {geometry['skt']} && Sk_chunk_t == {geometry['sk_chunk_t']}))", patched)
                self.assertIn('{factory_selector()} && Sk_chunk_t == 8)', patched)
                # Exactly one 65536 Skt branch, whichever geometry resolved to -
                # never both accepted values' branches in the same staging.
                for other in wide.ACCEPTED_SKT:
                    if other != geometry['skt']:
                        self.assertNotIn(f'Skt == {other} &&', patched)
                compile(patched, 'dspark_fp32_intermediates.py', 'exec')

    def test_kernel_fix_modules_skt_patched_to_match(self):
        for module_name in wide.KERNEL_FIX_MODULES:
            real_source = Path(__file__).with_name(module_name).read_text()
            for geometry in self._geometries():
                with self.subTest(module=module_name, skt=geometry['skt']):
                    patched = wide._patch_skt_constant(real_source, geometry)
                    self.assertIn(f"SKT = {geometry['skt']}", patched)
                    if geometry['skt'] != 2080:
                        self.assertNotIn('SKT = 2080', patched)
                    compile(patched, module_name, 'exec')

    def test_build_cache_patch(self):
        real_build_cache = Path(__file__).with_name('frozen_sim_build_cache.py').read_text()
        patched = wide._patch_build_cache(real_build_cache)
        self.assertIn('import frozen_wide_chunk_scratch', patched)
        self.assertIn("'frozen_wide_chunk_scratch.py'", patched)
        self.assertIn('with frozen_wide_chunk_scratch.factory_scope(), patch.object(baseline.subprocess', patched)
        self.assertIn('with frozen_wide_chunk_scratch.factory_scope():\n        baseline.validate_manifest', patched)
        compile(patched, 'frozen_sim_build_cache.py', 'exec')

    def test_full_adapter_output_compiles_and_stages_the_new_module_under_each_knob(self):
        for knob in ('2080', '2112'):
            with self.subTest(knob=knob), patch.dict(os.environ, {wide.KNOB: knob}):
                result = wide.adapt_wide_chunk_normalization(dict(self.sources), 65536)
                self.assertIn('frozen_wide_chunk_scratch.py', result)
                self.assertIn(f'SKT = {knob}', result['frozen_wide_chunk_scratch.py'])
                for name, source in result.items():
                    if name.endswith('.py'):
                        compile(source, name, 'exec')


def target_probe_source():
    """Real historical target-t16-attention-8k-probe.py after
    frozen_target_replay.adapt_target_probe() - the exact text
    adapt_wide_chunk_normalization() receives whenever --target-replay was
    passed (frozen_recipe_context.py applies adapt_target_probe() at
    line 259, before this module's adapter runs at line 346; the
    verbatim-copy loop in between does not touch this filename)."""
    from frozen_target_replay import adapt_target_probe
    return adapt_target_probe(historical('target-t16-attention-8k-probe.py'))


class TargetProbePatchTests(unittest.TestCase):
    """_patch_target_probe: the target-replay-scratch factory-pin fix for
    run 35598585759 ('Exact pinned SDPA factory required', QWEN_FROZEN_
    TARGET_SCRATCH=1 lanes). Only staged when --target-replay was passed
    (target-t16-attention-8k-probe.py present in `sources`); a pure no-op
    omission otherwise, and for every context other than 65536 regardless
    of --target-replay."""

    @classmethod
    def setUpClass(cls):
        cls.source = target_probe_source()

    def test_anchor_appears_exactly_once_in_the_real_adapted_text(self):
        # Precondition for _once() inside _patch_target_probe: if this ever
        # drifts to 0 or >1, _patch_target_probe raises ValueError instead
        # of silently mismatching - this just documents why that's expected.
        self.assertEqual(self.source.count('        from dspark_fp32_build import validate_manifest\n'), 1)

    def test_redirects_the_compact_scratch_validate_manifest_import(self):
        patched = wide._patch_target_probe(self.source)
        self.assertEqual(patched.count('from frozen_wide_chunk_scratch import validate_manifest'), 1)
        self.assertNotIn('from dspark_fp32_build import validate_manifest', patched)
        # The unrelated report['factory_build'] call site itself is untouched -
        # only the import line changes.
        self.assertIn("report['factory_build'] = validate_manifest("
            "'/opt/tt-metal', '/experiment/results/dspark-fp32-build.json')", patched)
        compile(patched, 'target-t16-attention-8k-probe.py', 'exec')

    def test_full_adapter_applies_the_redirect_when_target_replay_key_present(self):
        sources = adapted_sources(65536)
        sources['target-t16-attention-8k-probe.py'] = self.source
        result = wide.adapt_wide_chunk_normalization(sources, 65536)
        patched = result['target-t16-attention-8k-probe.py']
        self.assertEqual(patched.count('from frozen_wide_chunk_scratch import validate_manifest'), 1)
        self.assertNotIn('from dspark_fp32_build import validate_manifest', patched)
        compile(patched, 'target-t16-attention-8k-probe.py', 'exec')

    def test_full_adapter_is_a_no_op_when_target_replay_key_absent(self):
        # Real eager-only / reciprocal-only lanes never pass --target-replay,
        # so target-t16-attention-8k-probe.py is never staged at all - the
        # conditional wiring in adapt_wide_chunk_normalization must not add it.
        sources = adapted_sources(65536)
        self.assertNotIn('target-t16-attention-8k-probe.py', sources)
        result = wide.adapt_wide_chunk_normalization(sources, 65536)
        self.assertNotIn('target-t16-attention-8k-probe.py', result)

    def test_32768_leaves_target_probe_source_untouched_even_when_present(self):
        sources = adapted_sources(32768)
        sources['target-t16-attention-8k-probe.py'] = self.source
        result = wide.adapt_wide_chunk_normalization(sources, 32768)
        self.assertEqual(result['target-t16-attention-8k-probe.py'], self.source)


class ContentAddressedCacheDifferentiationTests(unittest.TestCase):
    """No manual per-context binary pin is added (see the port report): the
    existing content-addressed build cache (frozen_binary_cache.cache_key,
    keyed on a digest of the staged factory and builder files) already
    produces a different key whenever the staged text differs. These tests
    are the evidence for that claim: the 65536-adapted files must never hash
    the same as the untouched 32768 files, under either knob value, and the
    two knob values must themselves differ."""

    def test_fp32_intermediates_digest_differs_between_contexts_and_between_knob_values(self):
        sources_32768 = adapted_sources(32768)
        digest_32768 = hashlib.sha256(sources_32768['dspark_fp32_intermediates.py'].encode()).hexdigest()
        digests_65536 = {}
        for knob in ('2080', '2112'):
            with patch.dict(os.environ, {wide.KNOB: knob}):
                sources_65536 = wide.adapt_wide_chunk_normalization(adapted_sources(65536), 65536)
            digests_65536[knob] = hashlib.sha256(sources_65536['dspark_fp32_intermediates.py'].encode()).hexdigest()
            self.assertNotEqual(digest_32768, digests_65536[knob])
        self.assertNotEqual(digests_65536['2080'], digests_65536['2112'])

    def test_build_cache_digest_differs_between_contexts_after_wide_chunk_patch(self):
        real_build_cache = Path(__file__).with_name('frozen_sim_build_cache.py').read_text()
        unpatched_digest = hashlib.sha256(real_build_cache.encode()).hexdigest()
        patched_digest = hashlib.sha256(wide._patch_build_cache(real_build_cache).encode()).hexdigest()
        self.assertNotEqual(unpatched_digest, patched_digest)


if __name__ == '__main__':
    unittest.main()
