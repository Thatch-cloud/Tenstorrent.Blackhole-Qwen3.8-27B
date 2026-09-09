from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tensix_mlp_gate import COMPONENTS, NATIVE_SOURCES, ORIGINAL_PACKER, PACKER, SIMULATOR_PACKER, SOURCES
from tensix_mlp_gate import main, qualify, qualify_producers
from tensix_weight_stream import stream_geometry
from tiny_mlp_gate import HARDWARE_TP_COMMON, SIMULATOR_TP_COMMON, TP_COMMON


class TensixMlpGateTests(unittest.TestCase):
    def test_producer_count_mapping_and_runtime_must_match(self):
        report = self.fixture(16)
        self.assertEqual(qualify_producers(report, 16), 16)
        self.assertTrue(qualify(report, report['sources'], report['native_sources'])['passed'])
        with self.assertRaises(ValueError):
            qualify_producers(report, 8)
        for value in (None, True, 16.0, 8, 12):
            with self.subTest(value=value), self.assertRaises(ValueError):
                qualify_producers({**report, 'producers_per_card': value})
        for failure in ('missing', 'coordinate', 'duplicate'):
            changed = deepcopy(report)
            mapping = changed['producer_mappings']['gate']
            if failure == 'missing':
                mapping.pop()
            elif failure == 'coordinate':
                mapping[0] = ((0, 0), mapping[0][1])
            else:
                mapping[0][1].append(1)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                qualify_producers(changed)

    def fixture(self, producers=8):
        return dict(passed=True, closed_cleanly=True, backend='simulator', packer_zero_graft=True,
            shared_pool=True, shared_workspace=True, full_mlp=True, dram_boundary=True, rows=8, compared_rows=32,
            fixtures=2, pool_buffers=2, components=list(COMPONENTS),
            producers_per_card=producers, producer_mappings={name: stream_geometry(name,
                34 if name == 'down' else 20, producers)['mapping'] for name in ('gate', 'up', 'down')},
            sources={name: 'a' * 64 for name in SOURCES},
            native_sources={**{name: 'b' * 64 for name in NATIVE_SOURCES}, PACKER: SIMULATOR_PACKER,
                TP_COMMON: SIMULATOR_TP_COMMON},
            control_checks=[dict(pattern=pattern, fixture=fixture, component=component, chip=chip,
                exact_all_32_rows=True) for pattern in range(2) for fixture in range(2)
                for component in range(4) for chip in range(2)],
            eager_checks=[dict(pattern=pattern, fixture=fixture, component=component, chip=chip,
                exact_all_32_rows=True) for pattern in range(2) for fixture in range(2)
                for component in range(4) for chip in range(2)],
            replay_checks=[dict(repetition=repetition, pattern=pattern, fixture=fixture, component=component,
                chip=chip, exact_all_32_rows=True, bindings_stable=True, pool_reused=True)
                for repetition, pattern in enumerate((0, 1, 0)) for fixture in range(2)
                for component in range(4) for chip in range(2)],
            input_checks=[dict(phase=phase, repetition=repetition, tensor=tensor, chip=chip, packed_words_unchanged=True)
                for phase, repeats in ((0, 2), (1, 3)) for repetition in range(repeats)
                for tensor in range(7) for chip in range(2)],
            negative_controls=[dict(fixture=fixture, chip=chip, stale_detected=True)
                for fixture in range(2) for chip in range(2)],
            fixture_checks=[dict(chip=chip, different_weights_detected=True) for chip in range(2)],
            weight_bindings=[[4096 * (index + 1), 8192 * (index + 1)] for index in range(6)])

    def test_complete_gate_accepts_restored_runtime_and_pinned_decode_equivalence(self):
        report = self.fixture()
        for packer in (ORIGINAL_PACKER, SIMULATOR_PACKER):
            for common in (SIMULATOR_TP_COMMON, HARDWARE_TP_COMMON):
                native = {**report['native_sources'], PACKER: packer, TP_COMMON: common}
                result = qualify(report, report['sources'], native)
                self.assertTrue(result['passed'])
                self.assertEqual('native_equivalence' in result, common == HARDWARE_TP_COMMON)

    def test_native_drift_or_missing_source_fails_closed(self):
        original = self.fixture()
        for field, names in (('sources', SOURCES), ('native_sources', NATIVE_SOURCES)):
            for name in names:
                report = deepcopy(original)
                report[field][name] = 'c' * 64
                with self.subTest(field=field, name=name), self.assertRaises(ValueError):
                    qualify(report, original['sources'], original['native_sources'])
        for field in ('sources', 'native_sources'):
            for replacement in (None, [], {}):
                report = deepcopy(original)
                report[field] = replacement
                with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                    qualify(report, original['sources'], original['native_sources'])

    def test_unreviewed_runtime_graft_or_prefill_change_fails(self):
        report = self.fixture()
        for name in (PACKER, TP_COMMON):
            native = {**report['native_sources'], name: 'c' * 64}
            with self.subTest(name=name), self.assertRaises(ValueError):
                qualify(report, report['sources'], native)
        for field in ('sources', 'native_sources'):
            sources = deepcopy(report['sources'])
            native = deepcopy(report['native_sources'])
            (sources if field == 'sources' else native).pop(next(iter(sources if field == 'sources' else native)))
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualify(report, sources, native)

    def test_complete_matrix_and_strict_boolean_results_required(self):
        original = self.fixture()
        fields = dict(control_checks='exact_all_32_rows', eager_checks='exact_all_32_rows',
            replay_checks='pool_reused', input_checks='packed_words_unchanged',
            negative_controls='stale_detected', fixture_checks='different_weights_detected')
        for field, result in fields.items():
            for mutation in ('missing', 'duplicate', 'false', 'bool_result', 'bool_chip'):
                report = deepcopy(original)
                checks = report[field]
                if mutation == 'missing':
                    checks.pop()
                elif mutation == 'duplicate':
                    checks[1] = checks[0]
                elif mutation == 'bool_chip':
                    checks[0]['chip'] = False
                else:
                    checks[0][result] = False if mutation == 'false' else 1
                with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
                    qualify(report, original['sources'], original['native_sources'])

    def test_shape_pool_cleanup_and_backend_metadata_required(self):
        original = self.fixture()
        fields = dict(passed=1, closed_cleanly=False, backend='hardware', packer_zero_graft=False,
            shared_pool=False, shared_workspace=False, full_mlp=False, dram_boundary=False, rows=8.0, compared_rows=8,
            fixtures=1, pool_buffers=6, components=list(reversed(COMPONENTS)), error='teardown failed')
        for field, value in fields.items():
            report = {**deepcopy(original), field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualify(report, original['sources'], original['native_sources'])

    def test_each_fixture_requires_distinct_integer_weight_addresses(self):
        original = self.fixture()
        for mutation in ('missing', 'single_chip', 'duplicate', 'zero', 'boolean', 'float'):
            report = deepcopy(original)
            bindings = report['weight_bindings']
            if mutation == 'missing':
                bindings.pop()
            elif mutation == 'single_chip':
                bindings[0].pop()
            else:
                bindings[1][1] = dict(duplicate=bindings[0][1], zero=0, boolean=True, float=8192.0)[mutation]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify(report, original['sources'], original['native_sources'])

    def test_json_success_does_not_override_failed_or_missing_wrapper_exit(self):
        report = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path, status_path = root / 'probe.json', root / 'probe.exit-status'
            report_path.write_text(json.dumps(report))
            arguments = ['tensix_mlp_gate.py', '--report', str(report_path), '--native-root', str(root),
                '--exit-status', str(status_path)]
            with patch('sys.argv', arguments), patch('tensix_mlp_gate.hashes',
                    side_effect=[report['sources'], report['native_sources']]) as source_hashes:
                for status in ('1', '124', ''):
                    status_path.write_text(status)
                    with self.subTest(status=status), self.assertRaises(ValueError):
                        main()
                status_path.unlink()
                with self.assertRaises(FileNotFoundError):
                    main()
                source_hashes.assert_not_called()
                status_path.write_text('0\n')
                with patch('builtins.print') as output:
                    main()
                self.assertTrue(json.loads(output.call_args.args[0])['passed'])


if __name__ == '__main__':
    unittest.main()
