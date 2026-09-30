import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from frozen_combined_adapters import adapt_admission, adapt_combined_sources, FILES
from frozen_runtime_context import FILES as RUNTIME_FILES, adapt_runtime_sources
from frozen_combined_runtime import (qualify, validate_target_option, prepare_scratch,
    qualified_native_reference, staged_combined_context)
from frozen_recipe_context import REVISION, COMBINED_RUNTIME_CONTEXTS


class CombinedAdmissionTests(unittest.TestCase):
    def test_native_reference_only_replaces_audited_tree_sources(self):
        from sdpa_tree_scratch import HASHES, ROOT, PATCHED_FACTORY_SHA256
        original = {(ROOT / name).as_posix(): checksum for name, checksum in HASHES.items()}
        changed = dict(HASHES, **{'sdpa_decode_program_factory.cpp': PATCHED_FACTORY_SHA256})
        with patch('sdpa_tree_scratch.audit', return_value=changed):
            result = qualified_native_reference('.', dict(original, unrelated='retained'))
            self.assertEqual(result['unrelated'], 'retained')
            self.assertEqual(result[(ROOT / 'sdpa_decode_program_factory.cpp').as_posix()], PATCHED_FACTORY_SHA256)
            original[(ROOT / 'sdpa_decode_program_factory.cpp').as_posix()] = 'unknown'
            with self.assertRaisesRegex(ValueError, 'Unexpected original'):
                qualified_native_reference('.', original)

    def test_hardware_scratch_cannot_apply_without_allocation(self):
        with patch.dict(os.environ, {}, clear=True), patch('frozen_combined_runtime.subprocess.run') as execute:
            with self.assertRaisesRegex(ValueError, 'Explicit offline hardware'):
                prepare_scratch('.')
            execute.assert_not_called()

    def test_target_request_geometry_remains_strict(self):
        options = dict(rows=16, position=32768, remaining=256, replay=True,
            norm_batch=True, native_sampling=True, group_rows=4, short_context=False)
        validate_target_option(True, **options)
        for name, value in (('rows', 8), ('position', 8192), ('remaining', 257), ('remaining', 0),
                ('replay', False), ('norm_batch', False), ('native_sampling', False),
                ('group_rows', 8), ('short_context', True)):
            with self.assertRaises(ValueError):
                validate_target_option(True, **dict(options, **{name: value}))

    def test_candidate_scope_retains_guards_and_enters_reciprocal(self):
        sources = {name: subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/{name}'], text=True) for name in FILES + RUNTIME_FILES}
        adapted = adapt_combined_sources(adapt_runtime_sources(sources))
        for name, source in adapted.items():
            if name.endswith('.py'):
                compile(source, name, 'exec')
        entry = adapted['dspark_8k_entry.py']
        self.assertIn('request_context() != 32768', entry)
        for guard in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED', 'TT_METAL_SIMULATOR',
                '--captured-publication', 'QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS'):
            self.assertIn(guard, entry)
        self.assertIn('stack.enter_context(scalar_reciprocal())', adapted['dspark_8k_scope.py'])
        self.assertIn('if history_limit() == 33024:', adapted['dspark_context_selection.py'])
        self.assertIn('from frozen_combined_runtime import qualify_target', adapted['target_t16_attention_gate.py'])
        self.assertIn('from frozen_combined_runtime import validate_target_option', adapted['target_t16_attention_gate.py'])
        self.assertIn('scratch = prepare_scratch(root)', adapted['dspark_8k_build.py'])
        self.assertIn("verify_scratch(root, inputs.get('target_tree_scratch'))", adapted['dspark_8k_build.py'])
        self.assertIn('enabled=request_context() == 32768', adapted['dspark_runtime_cache.py'])
        self.assertIn('if options.preflight and request_context() == 32768:', adapted['dspark-target-hardware.py'])
        self.assertIn('require_compatible_native(native,', adapted['dspark-target-hardware.py'])
        namespace = {'__file__': str(Path(__file__).with_name('coding_context_request.py'))}
        exec(adapted['coding_context_request.py'], namespace)
        from coding_request import TASK
        def encode(messages, **options):
            self.assertTrue(messages[1]['content'].endswith(TASK))
            self.assertIs(options['enable_thinking'], False)
            return [ord(character) for character in messages[1]['content']]
        tokenizer = SimpleNamespace(apply_chat_template=encode)
        tokens, report = namespace['make_context_prompt'](tokenizer, context_tokens=32768)
        self.assertEqual(len(tokens), 32768)
        self.assertEqual(report['actual_context'], 32768)
        self.assertEqual(len(report['sources']), 12)

    def test_selected_admission_preserves_binary_and_factory_checks(self):
        source = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/dspark_8k_admission.py'], text=True)
        changed = adapt_admission(source)
        namespace = {}
        exec(changed, namespace)
        namespace['validate_request'](32768, 256)
        for context in (8192, 16384, 65536, 131072, 262144):
            with self.assertRaises(ValueError):
                namespace['validate_request'](context, 256)
        self.assertEqual(namespace['history_limit'](), 8192)
        token = namespace['_ADMISSION'].set({'capacity': 33024})
        try:
            self.assertEqual(namespace['history_limit'](), 33024)
        finally:
            namespace['_ADMISSION'].reset(token)
        start, end = '    factory_sha256 = verify_factory', '    admission = dict('
        self.assertEqual(source.split(start)[1].split(end)[0], changed.split(start)[1].split(end)[0])

    def test_default_or_simulator_runtime_cannot_enter_hardware_admission(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Explicit allocated offline'):
                qualify('.')
        with patch.dict(os.environ, {'QWEN_FROZEN_COMBINED_RUNTIME': '1',
                'QWEN_DSPARK_REQUEST_CONTEXT': '32768', 'QWEN_HARDWARE_TESTS': '1',
                'QWEN_CARDS_ALLOCATED': '1', 'TT_METAL_SIMULATOR': '1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Explicit allocated offline'):
                qualify('.')


class Rung65536StagingTests(unittest.TestCase):
    """T16 recipe rung 65536: the six combined-adapter files and the admission module
    must carry 65536-derived literals when staged at that context, while staging at the
    default 32768 stays byte-identical to the pre-widening hardcoded literals (docs/
    t16-recipe-rung-65k.md). Covers frozen_combined_adapters.py and the parts of
    frozen_combined_runtime.py that must accept whichever context was staged."""

    def setUp(self):
        self.sources = {name: subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/{name}'], text=True) for name in FILES + RUNTIME_FILES}

    def test_context_choices_match_frozen_recipe_context_cli(self):
        self.assertEqual(COMBINED_RUNTIME_CONTEXTS, (32768, 65536))

    def test_32768_default_adapter_output_is_byte_identical_to_before_widening(self):
        # Snapshot of the exact hardcoded literals adapt_combined_sources/adapt_admission
        # emitted before they were made geometry-derived. Calling with no context argument
        # (the pre-widening call shape, still used by
        # test_candidate_scope_retains_guards_and_enters_reciprocal above) and calling
        # with context=32768 explicitly must both still produce these exact strings.
        before_widening = {
            'dspark_8k_build.py': 'key_chunk_size=256, capacity=33024, target_tree_scratch=scratch)',
            'dspark-target-hardware.py':
                "if history_limit() != 33024 or gate['native_reference'].get(SOURCE) != SOURCE_SHA256:",
            'coding_context_request.py': 'context_tokens not in (4096, 8192, 32768)',
            'dspark_context_selection.py': 'if history_limit() == 33024:',
        }
        for default_call in (lambda sources: adapt_combined_sources(adapt_runtime_sources(sources)),
                lambda sources: adapt_combined_sources(adapt_runtime_sources(sources), context=32768)):
            with self.subTest(default_call=default_call):
                adapted = default_call(dict(self.sources))
                for name, snippet in before_widening.items():
                    self.assertIn(snippet, adapted[name])

        admission_source = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/dspark_8k_admission.py'], text=True)
        for default_call in (lambda source: adapt_admission(source),
                lambda source: adapt_admission(source, context=32768)):
            with self.subTest(default_call=default_call):
                changed = default_call(admission_source)
                self.assertIn('context != 32768 or output_tokens != 256', changed)
                self.assertIn('Qualified candidate requires exactly 32768 prompt rows and 256 output-token headroom',
                    changed)
                self.assertIn('output_tokens=output_tokens, capacity=33024, key_chunk_size=256', changed)

    def test_65536_staging_emits_65536_derived_literals_not_32768(self):
        adapted = adapt_combined_sources(adapt_runtime_sources(dict(self.sources)), context=65536)
        for name, source in adapted.items():
            if name.endswith('.py'):
                compile(source, name, 'exec')
        self.assertIn('request_context() != 65536', adapted['dspark_8k_entry.py'])
        self.assertIn('context=65536, output_tokens=256', adapted['dspark_8k_entry.py'])
        self.assertIn('enabled=request_context() == 65536', adapted['dspark_runtime_cache.py'])
        self.assertIn('if options.preflight and request_context() == 65536:', adapted['dspark-target-hardware.py'])
        self.assertIn('if history_limit() != 65792', adapted['dspark-target-hardware.py'])
        self.assertIn('if history_limit() == 65792:', adapted['dspark_context_selection.py'])
        self.assertIn('key_chunk_size=256, capacity=65792, target_tree_scratch=scratch)', adapted['dspark_8k_build.py'])
        self.assertIn('context_tokens not in (4096, 8192, 65536)', adapted['coding_context_request.py'])
        self.assertIn('if context_tokens == 65536:', adapted['coding_context_request.py'])
        self.assertIn('from frozen_combined_runtime import validate_target_option', adapted['target_t16_attention_gate.py'])
        self.assertIn('if request_context() == 65536:\n        from frozen_combined_runtime import qualify_target as qualify_8k',
            adapted['target_t16_attention_gate.py'])
        for name in ('32768', '33024'):
            self.assertNotIn(name, adapted['dspark_runtime_cache.py'])

        admission_source = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/dspark_8k_admission.py'], text=True)
        changed = adapt_admission(admission_source, context=65536)
        namespace = {}
        exec(changed, namespace)
        namespace['validate_request'](65536, 256)
        for context in (8192, 16384, 32768, 131072, 262144):
            with self.subTest(context=context), self.assertRaises(ValueError):
                namespace['validate_request'](context, 256)
        self.assertIn('output_tokens=output_tokens, capacity=65792, key_chunk_size=256', changed)
        self.assertIn('Qualified candidate requires exactly 65536 prompt rows and 256 output-token headroom', changed)
        # The no-admission fallback (dspark_8k_admission.history_limit()'s base case) is a
        # fixed bucket unrelated to the combined-runtime candidate context - unchanged.
        self.assertEqual(namespace['history_limit'](), 8192)

    def test_out_of_range_context_is_rejected_not_silently_staged(self):
        for context in (4096, 8192, 16384, 131072, 262144):
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, 'Only 32768 or 65536'):
                    adapt_combined_sources(dict(self.sources), context=context)
                with self.assertRaisesRegex(ValueError, 'Only 32768 or 65536'):
                    adapt_admission('unused', context=context)

    def test_runtime_accepts_staged_65536_tree_only_when_explicitly_selected(self):
        options = dict(rows=16, remaining=256, replay=True, norm_batch=True,
            native_sampling=True, group_rows=4, short_context=False)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(staged_combined_context(), 32768)
            validate_target_option(True, position=32768, **options)
            with self.assertRaises(ValueError):
                validate_target_option(True, position=65536, **options)
        with patch.dict(os.environ, {'QWEN_FROZEN_COMBINED_CONTEXT': '65536'}, clear=True):
            self.assertEqual(staged_combined_context(), 65536)
            validate_target_option(True, position=65536, **options)
            with self.assertRaises(ValueError):
                validate_target_option(True, position=32768, **options)
        with patch.dict(os.environ, {'QWEN_FROZEN_COMBINED_CONTEXT': '131072'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Unsupported QWEN_FROZEN_COMBINED_CONTEXT'):
                staged_combined_context()

    def test_qualify_reads_staged_context_from_environment_not_a_32768_constant(self):
        with patch.dict(os.environ, {'QWEN_FROZEN_COMBINED_RUNTIME': '1',
                'QWEN_FROZEN_COMBINED_CONTEXT': '65536', 'QWEN_DSPARK_REQUEST_CONTEXT': '32768',
                'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1',
                'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}, clear=True):
            # Explicitly selected 65536 but the request geometry env still says 32768:
            # must refuse as a mismatch, not silently fall back to either side.
            with self.assertRaisesRegex(ValueError, 'Explicit allocated offline staged combined'):
                qualify('.')
        with patch.dict(os.environ, {'QWEN_FROZEN_COMBINED_RUNTIME': '1',
                'QWEN_FROZEN_COMBINED_CONTEXT': '65536', 'QWEN_DSPARK_REQUEST_CONTEXT': '65536',
                'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1',
                'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}, clear=True):
            # Matching, explicit selection all the way through: reaches the gate, which
            # then refuses on its own separate "no qualified evidence" ground (covered in
            # test_frozen_combined_gate.py), never the environment/geometry guard above.
            with self.assertRaisesRegex(ValueError, 'no qualified combined-runtime evidence'):
                qualify('.')


if __name__ == '__main__':
    unittest.main()
