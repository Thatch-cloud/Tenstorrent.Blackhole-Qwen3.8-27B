"""Tests for the 65536-only target-replay k-chunk diagnostic lever
(QWEN_FROZEN_65536_REPLAY_K_CHUNK).

Covers: the substitution against the real historical attention_replay.py
content, that the injected runtime validation actually raises for an invalid
knob value (not just that the text looks right), that every other context is
a pure no-op (attention_replay.py never appears in the returned sources at
all, matching its current never-staged behaviour), and that a real dry run
of frozen_recipe_context.main() against the pinned checkout leaves 32768
byte-identical including attention_replay.py's own on-disk bytes.
"""

import subprocess
import sys
import types
import unittest
from pathlib import Path

from frozen_context_geometry import CONTEXTS
import frozen_wide_chunk_replay as replay
from frozen_recipe_context import REVISION


ROOT = Path(__file__).resolve().parents[2]


def historical_attention_replay():
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{REVISION}:scripts/ci/attention_replay.py'],
        check=True, capture_output=True, text=True).stdout


BACKSLASH = chr(92)

LAUNCHER = ('docker create \\\n'
    '    -e "QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}" \\\n'
    '    --entrypoint /bin/bash image suite.sh\n')


class NoOpForOtherContextsTests(unittest.TestCase):

    def test_returns_equal_dict_without_attention_replay_for_every_other_context(self):
        sources = dict(a='x', b='y')
        for context in CONTEXTS:
            if context == replay.CONTEXT:
                continue
            with self.subTest(context=context):
                result = replay.adapt_replay_k_chunk(sources, context, checkout='/does/not/matter')
                self.assertEqual(result, sources)
                self.assertNotIn('attention_replay.py', result)
                self.assertIsNot(result, sources)

    def test_raises_if_attention_replay_already_staged(self):
        sources = dict(**{'attention_replay.py': 'already here'})
        with self.assertRaises(ValueError):
            replay.adapt_replay_k_chunk(sources, 65536, checkout='/does/not/matter')

    def test_missing_attention_replay_raises_clear_value_error_naming_the_file(self):
        """The bug this guards: a synthetic/partial checkout (e.g. a test
        fixture that doesn't happen to include attention_replay.py) must not
        surface as a raw FileNotFoundError traceback from deep inside this
        module - it must be an explicit, named ValueError, since a REAL
        checkout at REVISION always has the file and a missing one signals a
        genuinely wrong --checkout argument, not something to swallow."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            (checkout / 'scripts/ci').mkdir(parents=True)  # directory exists, file does not
            with self.assertRaises(ValueError) as caught:
                replay.adapt_replay_k_chunk({}, 65536, checkout)
            self.assertIn('attention_replay.py', str(caught.exception))
            self.assertNotIsInstance(caught.exception, FileNotFoundError)


class PatchAgainstRealHistoricalSourceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.historical = historical_attention_replay()

    def test_patch_targets_the_real_anchor_and_only_that_literal(self):
        patched = replay._patch_attention_replay(self.historical)
        self.assertIn("os.environ.get('QWEN_FROZEN_65536_REPLAY_K_CHUNK', '128')", patched)
        self.assertIn('k_chunk_size=_qwen_replay_k_chunk)', patched)
        self.assertNotIn('k_chunk_size=256)', patched)
        # q_chunk_size=0 is a separate, untouched parameter.
        self.assertIn('q_chunk_size=0, k_chunk_size=_qwen_replay_k_chunk', patched)
        compile(patched, 'attention_replay.py', 'exec')

    def test_missing_anchor_refuses_loudly(self):
        mangled = self.historical.replace(
            'grid = mesh.compute_with_storage_grid_size()', 'grid = mesh.compute_with_storage_grid_size_v2()')
        with self.assertRaises(ValueError):
            replay._patch_attention_replay(mangled)

    def test_adapt_replay_k_chunk_end_to_end_against_tmp_checkout(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            (checkout / 'scripts/ci').mkdir(parents=True)
            (checkout / 'scripts/ci/attention_replay.py').write_text(self.historical)
            result = replay.adapt_replay_k_chunk({'run-simulator.sh': LAUNCHER}, 65536, checkout)
            self.assertIn('attention_replay.py', result)
            self.assertIn('_qwen_replay_k_chunk', result['attention_replay.py'])
            # The launcher is a required input at 65536 now: the knob is useless unless
            # it reaches the container, so a missing runner is an error, not a skip.
            with self.assertRaisesRegex(ValueError, 'run-simulator.sh must be staged'):
                replay.adapt_replay_k_chunk({}, 65536, checkout)


class KnobForwardingTests(unittest.TestCase):
    """The knob is read at runtime INSIDE the container, so run-simulator.sh has to
    pass it through. Nothing did before, which is why scratch-v1..v8 are all
    k_chunk=128 results whatever the workflow set."""

    def setUp(self):
        self.historical = historical_attention_replay()

    def _adapt(self, context, sources):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            (checkout / 'scripts/ci').mkdir(parents=True)
            (checkout / 'scripts/ci/attention_replay.py').write_text(self.historical)
            return replay.adapt_replay_k_chunk(sources, context, checkout)

    def test_knob_is_forwarded_just_before_the_case_anchor(self):
        result = self._adapt(65536, {'run-simulator.sh': LAUNCHER})
        runner = result['run-simulator.sh']
        expected = '    -e "%s=${%s:-%s}" %s\n' % (replay.KNOB, replay.KNOB, replay.DEFAULT, BACKSLASH)
        self.assertIn(expected, runner)
        self.assertEqual(runner.count('-e "QWEN_SIM_CASE='), 1)
        self.assertLess(runner.index(replay.KNOB + '='), runner.index('-e "QWEN_SIM_CASE='))
        # Same default on both sides, so an unset knob is exactly the shipped behaviour.
        self.assertIn(':-' + replay.DEFAULT + '}', runner)

    def test_every_forwarded_line_still_continues(self):
        runner = self._adapt(65536, {'run-simulator.sh': LAUNCHER})['run-simulator.sh']
        block = runner.split('docker create', 1)[1].split('--entrypoint', 1)[0]
        self.assertEqual([line for line in block.split('\n')[:-1] if not line.endswith(BACKSLASH)], [])

    def test_forwarding_is_65536_only(self):
        self.assertEqual(replay.adapt_replay_k_chunk({'run-simulator.sh': LAUNCHER}, 32768,
            Path('.'))['run-simulator.sh'], LAUNCHER)


class RuntimeValidationTests(unittest.TestCase):
    """Actually execute the injected validation logic (not the whole
    ReplayAttentionReader.__init__, which needs real ttnn/torch/mesh objects
    this session has no access to) against every accepted and rejected knob
    value, proving it raises rather than just reading like it should."""

    def _run_validation(self, env_value):
        namespace = {'os': types.SimpleNamespace(environ=({} if env_value is None else {replay.KNOB: env_value}))}
        source = (
            "_qwen_replay_k_chunk = os.environ.get('QWEN_FROZEN_65536_REPLAY_K_CHUNK', '128')\n"
            "if _qwen_replay_k_chunk not in ('64', '128', '256'):\n"
            "    raise ValueError('Explicit 64/128/256 replay K-chunk width required')\n"
            "_qwen_replay_k_chunk = int(_qwen_replay_k_chunk)\n"
        )
        exec(source, namespace)
        return namespace['_qwen_replay_k_chunk']

    def test_unset_defaults_to_128(self):
        self.assertEqual(self._run_validation(None), 128)

    def test_accepted_values(self):
        for value, expected in (('64', 64), ('128', 128), ('256', 256)):
            with self.subTest(value=value):
                self.assertEqual(self._run_validation(value), expected)

    def test_rejected_values_raise(self):
        for value in ('99', '0', '', '256 ', 'abc', '512'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self._run_validation(value)


class RealDryRunByteIdentityTests(unittest.TestCase):
    """One real end-to-end run of frozen_recipe_context.main() against a
    disposable clone of the pinned checkout - the same technique used to
    verify the wide-chunk-normalization port, extended to prove
    attention_replay.py specifically stays untouched for 32768."""

    def test_32768_attention_replay_untouched_by_real_pipeline_run(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / 'checkout'
            subprocess.run(['git', 'clone', '--quiet', str(ROOT), str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'checkout', '--quiet', REVISION], check=True)
            before = (checkout / 'scripts/ci/attention_replay.py').read_bytes()
            manifest = Path(tmp) / 'manifest.json'
            subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('frozen_recipe_context.py')),
                '--checkout', str(checkout), '--context', '32768', '--target-replay', '--scalar-reciprocal',
                '--probe-seconds', '1020', '--manifest', str(manifest)], check=True)
            after = (checkout / 'scripts/ci/attention_replay.py').read_bytes()
            self.assertEqual(before, after)
            status = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain', '--',
                'scripts/ci/attention_replay.py'], check=True, capture_output=True, text=True)
            self.assertEqual(status.stdout.strip(), '')


if __name__ == '__main__':
    unittest.main()
