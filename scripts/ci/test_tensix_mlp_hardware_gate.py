from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sampling_link_policy import SOURCES as FABRIC_SOURCES
from tensix_mlp_gate import ORIGINAL_PACKER, PACKER, qualify
from tensix_mlp_hardware_gate import HARDWARE_SOURCES, MODEL_SOURCES, qualify_hardware, validate_evidence
import test_tensix_mlp_gate


class TensixMlpHardwareGateTests(unittest.TestCase):
    def fixture(self, faster=True):
        candidate = 0.3 if faster else 0.5
        return dict(passed=True, closed_cleanly=True, backend='hardware', stage='complete', rows=8,
            streams=1, layer=0, collective_links=4, pool_buffers=2, repeats_per_sample=50,
            seeds=[1659, 2670, 3781], dram_boundary=True, native_collective=True, all_samples_retained=True,
            native_requested_links=dict(default=2, axis0=2, axis1=2),
            matched_requested_links=dict(default=4, axis0=4, axis1=4),
            eager_checks=[dict(pattern=pattern, chip=chip, exact=True) for pattern in range(3) for chip in range(2)],
            trace_checks=[dict(pattern=pattern, arm=arm, chip=chip, exact=True)
                for pattern in range(3) for arm in range(2) for chip in range(2)],
            negative_controls=[dict(arm=arm, chip=chip, stale_detected=True) for arm in range(2) for chip in range(2)],
            input_checks=[dict(pattern=pattern, tensor=tensor, chip=chip, packed_words_unchanged=True,
                bindings_stable=True, pool_reused=True) for pattern in range(3) for tensor in range(4) for chip in range(2)],
            timed_checks=[dict(pattern=pattern, block=block, sample=sample, chip=chip, exact=True)
                for pattern in range(3) for block in range(3) for sample in range(4) for chip in range(2)],
            blocks=[dict(pattern=pattern, block=block, samples_ms=[0.4, candidate, candidate, 0.4],
                order=['control', 'candidate', 'candidate', 'control'], control_ms=0.4, candidate_ms=candidate,
                ratio=0.4 / candidate) for pattern in range(3) for block in range(3)],
            control_ms=0.4, candidate_ms=candidate, eligible_for_full_model_gate=faster)

    def test_correctness_pass_does_not_imply_performance_promotion(self):
        for faster in (False, True):
            result = qualify_hardware(self.fixture(faster))
            self.assertTrue(result['passed'])
            self.assertEqual(result['eligible_for_full_model_gate'], faster)

    def test_native_link_count_is_recorded_without_confusing_it_with_the_matched_override(self):
        for links in (1, 2, 4):
            report = self.fixture()
            report['native_requested_links'] = dict(default=links, axis0=links, axis1=links)
            self.assertTrue(qualify_hardware(report)['passed'])
        for field in ('native_requested_links', 'matched_requested_links'):
            for value in (None, {}, dict(default=4, axis0=True, axis1=4), dict(default=4, axis0=4.0, axis1=4)):
                report = {**self.fixture(), field: value}
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    qualify_hardware(report)
        report = self.fixture()
        report['matched_requested_links']['axis0'] = 2
        with self.assertRaises(ValueError):
            qualify_hardware(report)

    def test_all_comparisons_and_timed_outputs_are_required(self):
        original = self.fixture()
        for field, result in dict(eager_checks='exact', trace_checks='exact', timed_checks='exact',
                negative_controls='stale_detected', input_checks='packed_words_unchanged').items():
            for mutation in ('missing', 'duplicate', 'boolean_chip', 'false', 'nonboolean'):
                report = deepcopy(original)
                checks = report[field]
                if mutation == 'missing':
                    checks.pop()
                elif mutation == 'duplicate':
                    checks[1] = checks[0]
                elif mutation == 'boolean_chip':
                    checks[0]['chip'] = False
                else:
                    checks[0][result] = False if mutation == 'false' else 1
                with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
                    qualify_hardware(report)

    def test_no_timing_sample_can_be_removed_reordered_or_relabelled(self):
        original = self.fixture()
        for mutation in ('missing_block', 'duplicate_block', 'missing_sample', 'order', 'nonfinite',
                'zero', 'bool', 'summary', 'ratio', 'aggregate', 'eligibility'):
            report = deepcopy(original)
            block = report['blocks'][0]
            if mutation == 'missing_block':
                report['blocks'].pop()
            elif mutation == 'duplicate_block':
                report['blocks'][1] = block
            elif mutation == 'missing_sample':
                block['samples_ms'].pop()
            elif mutation == 'order':
                block['order'] = ['candidate', 'control', 'control', 'candidate']
            elif mutation in ('nonfinite', 'zero', 'bool'):
                block['samples_ms'][0] = dict(nonfinite=float('nan'), zero=0, bool=True)[mutation]
            elif mutation == 'summary':
                block['candidate_ms'] = 0.1
            elif mutation == 'ratio':
                block['ratio'] = 2
            elif mutation == 'aggregate':
                report['candidate_ms'] = 0.1
            else:
                report['eligible_for_full_model_gate'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify_hardware(report)

    def test_backend_cleanup_boundary_shape_and_pool_are_strict(self):
        for name, value in dict(passed=1, closed_cleanly=False, backend='simulator', stage='preflight_complete',
                rows=8.0, streams=True, layer=False, collective_links=1, pool_buffers=6, repeats_per_sample=49,
                seeds=[1659], dram_boundary=False, native_collective=False, all_samples_retained=False,
                error='teardown failed').items():
            report = {**self.fixture(), name: value}
            with self.subTest(name=name), self.assertRaises(ValueError):
                qualify_hardware(report)

    def test_evidence_is_bound_to_sources_simulator_exit_and_native_model(self):
        simulator = test_tensix_mlp_gate.TensixMlpGateTests().fixture()
        report = self.fixture()
        hardware_sources = {name: 'c' * 64 for name in HARDWARE_SOURCES}
        report.update(sources=simulator['sources'], hardware_sources=hardware_sources, model_sources=MODEL_SOURCES,
            fabric_sources=FABRIC_SOURCES, native_sources={**simulator['native_sources'], PACKER: ORIGINAL_PACKER})
        report['simulator_gate'] = qualify(simulator, report['sources'], report['native_sources'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulator_path, status_path = root / 'tensix-mlp-simulator.json', root / 'tensix-mlp-simulator.exit-status'
            simulator_path.write_text(json.dumps(simulator))
            status_path.write_text('0\n')
            report['simulator_report_sha256'] = hashlib.sha256(simulator_path.read_bytes()).hexdigest()
            report['simulator_exit_sha256'] = hashlib.sha256(status_path.read_bytes()).hexdigest()
            def validate(value):
                with patch('tensix_mlp_hardware_gate.hashes', side_effect=[report['sources'], hardware_sources]):
                    return validate_evidence(value, root)
            self.assertTrue(validate(report)['passed'])
            for name in ('sources', 'hardware_sources', 'model_sources', 'fabric_sources', 'simulator_gate',
                    'simulator_report_sha256', 'simulator_exit_sha256', 'native_sources'):
                changed = {**report, name: {}}
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validate(changed)
            status_path.write_text('1\n')
            with self.assertRaisesRegex(ValueError, 'wrapper exit'):
                validate(report)

    def test_preflight_returns_before_any_accelerator_import(self):
        path = Path(__file__).with_name('tensix-stream-mlp-hardware.py')
        spec = importlib.util.spec_from_file_location('stream_mlp_hardware_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        simulator = test_tensix_mlp_gate.TensixMlpGateTests().fixture()
        native = {**simulator['native_sources'], PACKER: ORIGINAL_PACKER}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path, status_path, output_path = root / 'sim.json', root / 'sim.exit-status', root / 'hardware.json'
            report_path.write_text(json.dumps(simulator))
            status_path.write_text('0\n')
            arguments = ['hardware', '--simulator-report', str(report_path), '--simulator-exit-status',
                str(status_path), '--output', str(output_path), '--preflight']
            with patch('sys.argv', arguments), patch.dict('os.environ', {'TT_METAL_HOME': str(root)}, clear=True), \
                    patch.object(module, 'require_projection_environment'), patch.object(module, 'audit', return_value=FABRIC_SOURCES), \
                    patch.object(module, 'hashes', side_effect=[simulator['sources'], native, {}, MODEL_SOURCES]), \
                    patch.dict('sys.modules', {'ttnn': None, 'torch': None}):
                module.main()
            result = json.loads(output_path.read_text())
            self.assertTrue(result['preflight_passed'])
            self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
