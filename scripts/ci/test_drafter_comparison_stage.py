import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from drafter_comparison_stage import stage
import drafter_comparison_stage as staging
import dflash_t16_native_attention_gate as gate
import test_proposal_native_attention as fixtures


class ComparisonStageTests(unittest.TestCase):
    def test_native_overlay_qualifies_reports_before_and_after_copy(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch('dspark_hardware_gate.simulator_preflight'):
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            evidence = root / 'evidence'
            evidence.mkdir()
            directory = Path(staging.__file__).parent
            for name in gate.SOURCES:
                (scripts / name).write_bytes((directory / name).read_bytes())
            for name in ('compact-score-evidence', 'mlp-down-grid-evidence', 'register-epilogue-evidence'):
                (scripts / name).mkdir()
            (scripts / 'dspark-target-hardware.py').write_text('from cumulative_t16_experiment import run_loaded_requests\n')
            (scripts / 'dspark-hardware-suite.sh').write_text(
                '    > /experiment/results/dspark-build-time.json\nset +e\nload_weights\n')
            (scripts / 'run-dspark-hardware.sh').write_text(
                '    -e "QWEN_DSPARK_MODE=$mode"\ndocker start -a "$test_id" | tee "$output/dspark-console.log"\n')
            reports = {}
            for context in (31, 2048):
                report = fixtures.ProposalNativeAttentionGateTests().fixture(context)
                report.update(policy=gate.POLICY, sources=gate.hashes(directory, gate.SOURCES),
                    block_rows=16, fixture_sha256=None,
                    runtime_binaries=dict.fromkeys(gate.BINARIES, gate.BINARY_SHA256),
                    runtime_binaries_after=dict.fromkeys(gate.BINARIES, gate.BINARY_SHA256))
                for name in ('native_sources', 'native_sources_after'):
                    report[name][gate.FACTORY] = gate.COMBINED_FACTORY
                data = json.dumps(report).encode()
                (evidence / f'dflash-t16-{context}.json').write_bytes(data)
                (evidence / f'dflash-t16-{context}.exit-status').write_text('0\n')
                reports[context] = hashlib.sha256(data).hexdigest()
            with patch('dflash_t16_native_scope.REPORTS', reports):
                result = stage(root, root / 'manifest.json', native_evidence=evidence)
            self.assertEqual(result['comparison'], 'dflash-composed-versus-native')
            self.assertEqual(len(result['native_t16_simulator_reports']), 2)
            self.assertIn('dflash_native_comparison_experiment', (scripts / 'dspark-target-hardware.py').read_text())
            shell = (scripts / 'run-dspark-hardware.sh').read_text()
            self.assertIn('QWEN_DFLASH_NATIVE_COMPARISON=1', shell)
            self.assertNotIn('QWEN_DRAFTER_COMPARISON=1', shell)
            suite = (scripts / 'dspark-hardware-suite.sh').read_text()
            self.assertLess(suite.index('dflash_t16_native_scope.py'), suite.index('set +e'))
            self.assertLess(suite.index('dflash_t16_native_scope.py'), suite.index('load_weights'))
            self.assertFalse(result['hardware_qualified'])

    def test_overlay_keeps_control_and_copies_only_pinned_cached_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch('dspark_hardware_gate.simulator_preflight') as preflight:
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name in ('compact-score-evidence', 'mlp-down-grid-evidence', 'register-epilogue-evidence'):
                (scripts / name).mkdir()
            (scripts / 'dspark-target-hardware.py').write_text('from cumulative_t16_experiment import run_loaded_requests\n')
            (scripts / 'run-dspark-hardware.sh').write_text(
                '    -e "QWEN_DSPARK_MODE=$mode"\ndocker start -a "$test_id" | tee "$output/dspark-console.log"\n')
            manifest = root / 'manifest.json'
            result = stage(root, manifest)
            self.assertEqual(json.loads(manifest.read_text()), result)
            shell = (scripts / 'run-dspark-hardware.sh').read_text()
            self.assertIn('QWEN_DRAFTER_COMPARISON=1', shell)
            self.assertIn('copy_dflash_fixtures', shell)
            self.assertNotIn('prepare_dflash_fixtures', shell)
            self.assertLess(shell.index('copy_dflash_fixtures'), shell.index('docker start'))
            self.assertFalse(result['hardware_qualified'])
            self.assertEqual(preflight.call_count, 2)
            self.assertNotIn('draft_attention.py', result['after'])
            with self.assertRaises(ValueError):
                stage(root, manifest)

    def test_missing_target_evidence_fails_before_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'component evidence'):
                stage(root, root / 'manifest.json')
