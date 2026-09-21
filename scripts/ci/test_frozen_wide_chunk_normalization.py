"""Tests for the 65536-only wide-chunk normalization port.

Covers: the geometry derivation, that every substitution targets the actual
post-adapt_probe_sources() text (not a guess), that the 65536 branch carries
the scratch-CB substitution and 1024-key chunking, that every other context
is an exact no-op, and that the resulting per-context source text (and
therefore the content-addressed build-cache key) differs between 65536 and
32768 - the property that makes a manual binary pin unnecessary (see the
module docstring in frozen_wide_chunk_normalization.py and the port report).
"""

import hashlib
import subprocess
import unittest
from pathlib import Path

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

    def test_derived_constants(self):
        self.assertEqual(wide.CONTEXT, 65536)
        self.assertEqual(wide.CAPACITY, 65792)
        self.assertEqual(wide.STORAGE_KEYS, 65856)
        self.assertEqual(wide.KEY_CHUNK, 1024)
        self.assertEqual(wide.PADDED_KEYS, 66560)
        self.assertEqual(wide.SKT, 2080)
        self.assertEqual(wide.SK_CHUNK_T, 32)
        self.assertEqual(wide.ITERATIONS, 65)
        self.assertEqual(wide.ADDED_MASKED_POISON_ROWS, 704)

    def test_padded_keys_is_exact_multiple_of_key_chunk(self):
        self.assertEqual(wide.PADDED_KEYS % wide.KEY_CHUNK, 0)

    def test_differs_from_current_unfixed_and_from_ladder_geometry(self):
        # 2064 is the currently-generated (failing) Skt at the standard
        # 256-key chunk under this same context+256 capacity formula; 2112 is
        # dspark_ladder_normalization.py's own Skt, under its different
        # (output_tokens=1024) capacity formula. Both must NOT equal SKT -
        # if either does, the module docstring's derivation claim is wrong.
        self.assertNotEqual(wide.SKT, 2064)
        self.assertNotEqual(wide.SKT, 2112)


class NoOpForOtherContextsTests(unittest.TestCase):

    def test_returns_equal_dict_for_every_context_other_than_65536(self):
        sources = dict(a='x', b='y')
        for context in CONTEXTS:
            if context == wide.CONTEXT:
                continue
            with self.subTest(context=context):
                result = wide.adapt_wide_chunk_normalization(sources, context)
                self.assertEqual(result, sources)
                self.assertIsNot(result, sources)  # returns a copy, never the same object

    def test_32768_real_pipeline_output_is_byte_identical_with_and_without_this_module(self):
        without_wide_chunk = adapted_sources(32768)
        with_wide_chunk_noop = wide.adapt_wide_chunk_normalization(dict(without_wide_chunk), 32768)
        self.assertEqual(with_wide_chunk_noop, without_wide_chunk)
        # Not merely equal - every value byte-identical, which is what the
        # staged-file and build-cache-key guarantee actually depends on.
        for name in without_wide_chunk:
            self.assertEqual(hashlib.sha256(with_wide_chunk_noop[name].encode()).hexdigest(),
                hashlib.sha256(without_wide_chunk[name].encode()).hexdigest(), name)


class WideChunkPatchTests(unittest.TestCase):
    """Exercise each _patch_* function against the REAL post-adapt_probe_sources()
    text (fetched from the pinned historical revision), not a hand-written
    guess at what that text looks like - anchor drift shows up as a
    ValueError from _once(), which these tests would surface immediately."""

    @classmethod
    def setUpClass(cls):
        cls.sources = adapted_sources(65536)

    def test_probe_patch_report_literals_and_validate_manifest_redirect(self):
        patched = wide._patch_probe(self.sources['dspark-native-8k-attention-probe.py'])
        self.assertIn(f'key_chunk_size={wide.KEY_CHUNK}', patched)
        self.assertIn(f'native_padded_keys={wide.PADDED_KEYS}', patched)
        self.assertIn(f'added_masked_poison_rows={wide.ADDED_MASKED_POISON_ROWS}', patched)
        self.assertNotIn('key_chunk_size=256', patched)
        self.assertNotIn("native_padded_keys=SHAPE['padded_keys']", patched)
        # Both validate_manifest call sites (runner_fingerprints and run())
        # must be redirected, and dspark_fp32_build must no longer be
        # imported for this name anywhere in the file.
        self.assertEqual(patched.count('from frozen_wide_chunk_scratch import validate_manifest'), 2)
        self.assertNotIn('from dspark_fp32_build import validate_manifest', patched)
        compile(patched, 'dspark-native-8k-attention-probe.py', 'exec')

    def test_chunk_trial_patch(self):
        patched = wide._patch_chunk_trial(self.sources['dspark_attention_chunk_trial.py'])
        self.assertIn(f'KEY_CHUNK = {wide.KEY_CHUNK}', patched)
        self.assertIn(f'PADDED_KEYS = {wide.PADDED_KEYS}', patched)
        self.assertNotIn('KEY_CHUNK = 256', patched)
        compile(patched, 'dspark_attention_chunk_trial.py', 'exec')

    def test_stats_pack_patch_assert_and_message(self):
        patched = wide._patch_stats_pack(self.sources['dspark_stats_pack.py'])
        self.assertIn(f'get_compile_time_arg_val(3) == {wide.SKT} && get_compile_time_arg_val(8) == {wide.SK_CHUNK_T}',
            patched)
        self.assertIn(f'{wide.PADDED_KEYS} keys and {wide.KEY_CHUNK}-key chunks', patched)
        self.assertNotIn('== 272 &&', patched)
        self.assertNotIn('8704 keys and 256-key chunks', patched)
        compile(patched, 'dspark_stats_pack.py', 'exec')

    def test_fp32_intermediates_patch_widens_condition_without_touching_factory_selector_call(self):
        patched = wide._patch_fp32_intermediates(self.sources['dspark_fp32_intermediates.py'])
        self.assertIn('{factory_selector()}', patched)  # call site itself untouched
        self.assertIn(f'(Skt == {wide.SKT} && Sk_chunk_t == {wide.SK_CHUNK_T}))', patched)
        self.assertIn('{factory_selector()} && Sk_chunk_t == 8)', patched)
        compile(patched, 'dspark_fp32_intermediates.py', 'exec')

    def test_build_cache_patch(self):
        real_build_cache = Path(__file__).with_name('frozen_sim_build_cache.py').read_text()
        patched = wide._patch_build_cache(real_build_cache)
        self.assertIn('import frozen_wide_chunk_scratch', patched)
        self.assertIn("'frozen_wide_chunk_scratch.py'", patched)
        self.assertIn('with frozen_wide_chunk_scratch.factory_scope(), patch.object(baseline.subprocess', patched)
        self.assertIn('with frozen_wide_chunk_scratch.factory_scope():\n        baseline.validate_manifest', patched)
        compile(patched, 'frozen_sim_build_cache.py', 'exec')

    def test_full_adapter_output_compiles_and_stages_the_new_module(self):
        result = wide.adapt_wide_chunk_normalization(self.sources, 65536)
        self.assertIn('frozen_wide_chunk_scratch.py', result)
        self.assertEqual(result['frozen_wide_chunk_scratch.py'],
            Path(__file__).with_name('frozen_wide_chunk_scratch.py').read_text())
        for name, source in result.items():
            if name.endswith('.py'):
                compile(source, name, 'exec')


class ContentAddressedCacheDifferentiationTests(unittest.TestCase):
    """No manual per-context binary pin is added (see the port report): the
    existing content-addressed build cache (frozen_binary_cache.cache_key,
    keyed on a digest of the staged factory and builder files) already
    produces a different key whenever the staged text differs. These tests
    are the evidence for that claim: the 65536-adapted files must never hash
    the same as the untouched 32768 files."""

    def test_fp32_intermediates_digest_differs_between_contexts(self):
        sources_32768 = adapted_sources(32768)
        sources_65536 = wide.adapt_wide_chunk_normalization(adapted_sources(65536), 65536)
        digest_32768 = hashlib.sha256(sources_32768['dspark_fp32_intermediates.py'].encode()).hexdigest()
        digest_65536 = hashlib.sha256(sources_65536['dspark_fp32_intermediates.py'].encode()).hexdigest()
        self.assertNotEqual(digest_32768, digest_65536)

    def test_build_cache_digest_differs_between_contexts_after_wide_chunk_patch(self):
        real_build_cache = Path(__file__).with_name('frozen_sim_build_cache.py').read_text()
        unpatched_digest = hashlib.sha256(real_build_cache.encode()).hexdigest()
        patched_digest = hashlib.sha256(wide._patch_build_cache(real_build_cache).encode()).hexdigest()
        self.assertNotEqual(unpatched_digest, patched_digest)


if __name__ == '__main__':
    unittest.main()
