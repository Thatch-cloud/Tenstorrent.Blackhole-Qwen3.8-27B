from unittest.mock import patch
from pathlib import Path
import os
import subprocess
import unittest

import test_dspark_ladder_score_center as baseline
from dspark_score_bitwise import bitwise_infinity_checks, candidate_entrypoint, transform
import native_draft_sdpa


class BitwiseScoreTests(baseline.ScoreCenterTests):
    def setUp(self):
        self.enterContext(patch.object(baseline, 'STUB', transform(baseline.STUB)))
        self.enterContext(bitwise_infinity_checks())

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            transform('')

    def test_child_reenters_candidate_and_preserves_native_audit(self):
        with patch.object(native_draft_sdpa, 'run_precise_probe', return_value={'audited': True}) as original:
            with candidate_entrypoint('candidate.py'):
                self.assertEqual(native_draft_sdpa.run_precise_probe('baseline.py'), {'audited': True})
            original.assert_called_once_with('candidate.py')
            self.assertIs(native_draft_sdpa.run_precise_probe, original)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned Blackhole compiler required')
    def test_blackhole_assembly_removes_soft_float_comparisons(self):
        compiler = Path(os.environ['TT_NATIVE_TEST_ROOT']) / 'runtime/sfpi/compiler/bin/riscv-tt-elf-g++'
        result = subprocess.run([str(compiler), '-std=c++17', '-O2', '-mcpu=tt-bh',
            '-DCOMPILE_FOR_TRISC=0', '-x', 'c++', '-S', '-o', '-', '-'],
            input=baseline.STUB, capture_output=True, text=True, check=True, timeout=15)
        self.assertNotIn('__eqsf2', result.stdout)
        self.assertIn('__subsf3', result.stdout)
        self.assertIn('__addsf3', result.stdout)

    def test_smoke_exits_before_full_ladder(self):
        directory = Path(__file__).parent
        suite = (directory / 'simulator-suite.sh').read_text()
        start = suite.index('if [ "${QWEN_SCORE_BITWISE:-0}" = 1 ]; then',
            suite.index('export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=60000'))
        end = suite.index('fi', start)
        branch = suite[start:end]
        self.assertIn('timeout -k 15 105', branch)
        self.assertIn('exit 0', branch)
        self.assertLess(end, suite.index('for context in 65536'))
        workflow = (directory.parents[1] / '.github/workflows/qwen-ttsim.yml').read_text()
        self.assertIn('timeout -k 15 465 bash scripts/ci/run-simulator.sh', workflow)
