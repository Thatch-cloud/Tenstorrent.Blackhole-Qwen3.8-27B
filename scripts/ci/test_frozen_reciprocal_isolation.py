import unittest
import shutil
import subprocess

import native_draft_sdpa
from dspark_ladder_scalar_reciprocal import scalar_reciprocal, BEFORE, BODY
from frozen_reciprocal_isolation import isolate, isolated_reciprocal


class ReciprocalIsolationTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'), 'Host C++ compiler required')
    def test_preprocessed_draft_is_identical_and_target_omits_scalar_body(self):
        from test_dspark_ladder_scalar_reciprocal import STUB
        original = STUB.replace('#define QWEN_DRAFT_EXP_APPROX false\n', '')
        guarded = original.replace(BODY, '\n#if defined(QWEN_DRAFT_EXP_APPROX)\n' + BODY + '#endif\n')
        for processor in range(3):
            flags = ['g++', '-std=c++17', '-x', 'c++', '-DCOMPILE_FOR_TRISC=' + str(processor)]
            for macro in ('false', 'true'):
                command = flags + ['-DQWEN_DRAFT_EXP_APPROX=' + macro, '-E', '-P', '-']
                before = subprocess.run(command, input=original, text=True, capture_output=True, check=True, timeout=15)
                after = subprocess.run(command, input=guarded, text=True, capture_output=True, check=True, timeout=15)
                self.assertEqual(before.stdout, after.stdout)
            target = subprocess.run(flags + ['-E', '-P', '-'], input=guarded,
                text=True, capture_output=True, check=True, timeout=15)
            self.assertNotIn('values[offset] = 1.0f / values[offset]', target.stdout)
            self.assertNotIn('QWEN_DRAFT_EXP_APPROX', target.stdout)
            subprocess.run(flags + ['-fsyntax-only', '-'], input=guarded,
                text=True, capture_output=True, check=True, timeout=15)

    def test_only_preprocessor_boundary_changes(self):
        original = native_draft_sdpa.replacements
        with scalar_reciprocal():
            before = native_draft_sdpa.replacements()
            with isolated_reciprocal():
                after = native_draft_sdpa.replacements()
            self.assertEqual(native_draft_sdpa.replacements(), before)
        self.assertIs(native_draft_sdpa.replacements, original)
        self.assertEqual(before['sdpa.cpp'], after['sdpa.cpp'])
        changed = [(old, new) for old, new in zip(before['compute_common.hpp'], after['compute_common.hpp']) if old != new]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0][0], (BEFORE, BEFORE + BODY))
        self.assertEqual(changed[0][1], (BEFORE, BEFORE + '\n#if defined(QWEN_DRAFT_EXP_APPROX)\n' + BODY + '#endif\n'))

    def test_missing_scalar_candidate_rejected(self):
        with self.assertRaises(ValueError):
            isolate(native_draft_sdpa.replacements())


if __name__ == '__main__':
    unittest.main()
