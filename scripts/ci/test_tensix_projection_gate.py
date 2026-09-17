from copy import deepcopy
import json
import unittest

from tensix_projection_gate import NATIVE_SOURCES, ORIGINAL_PACKER, PACKER, SIMULATOR_PACKER, SOURCES, qualify
from tensix_weight_stream import stream_geometry


class TensixProjectionGateTests(unittest.TestCase):
    def fixture(self):
        return dict(passed=True, closed_cleanly=True, backend='simulator', packer_zero_graft=True,
            native_compute_unchanged=True, token_rows=8, compared_rows=32,
            sources={name: 'a' * 64 for name in SOURCES},
            native_sources={**{name: 'b' * 64 for name in NATIVE_SOURCES}, PACKER: SIMULATOR_PACKER},
            geometry=json.loads(json.dumps(stream_geometry('gate', 5))),
            control_checks=[dict(pattern=pattern, chip=chip, exact_all_32_rows=True)
                for pattern in range(2) for chip in range(2)],
            eager_checks=[dict(pattern=pattern, arm=arm, chip=chip, exact_all_32_rows=True, inputs_unchanged=True)
                for pattern in range(2) for arm in range(2) for chip in range(2)],
            replay_checks=[dict(repetition=repetition, pattern=pattern, arm=arm, chip=chip,
                exact_all_32_rows=True, inputs_unchanged=True, bindings_stable=True)
                for repetition, pattern in enumerate((0, 1, 0)) for arm in range(2) for chip in range(2)],
            negative_controls=[dict(arm=arm, chip=chip, stale_detected=True) for arm in range(2) for chip in range(2)])

    def test_complete_projection_matrix_can_be_checked_after_runtime_restore(self):
        report = self.fixture()
        for packer in (ORIGINAL_PACKER, SIMULATOR_PACKER):
            native = {**report['native_sources'], PACKER: packer}
            self.assertFalse(qualify(report, report['sources'], native)['full_projection'])
        for projection, blocks in (('gate', 20), ('up', 20), ('down', 34)):
            report['geometry'] = json.loads(json.dumps(stream_geometry(projection, blocks)))
            self.assertTrue(qualify(report, report['sources'], report['native_sources'])['full_projection'])

    def test_native_changes_or_hidden_graft_cannot_pass(self):
        original = self.fixture()
        for mutation in ('simulator_hash', 'native_hash', 'compute_source', 'missing_source', 'hidden_graft'):
            report = deepcopy(original)
            native = deepcopy(original['native_sources'])
            if mutation == 'simulator_hash':
                report['native_sources'][PACKER] = ORIGINAL_PACKER
            elif mutation == 'native_hash':
                native[PACKER] = 'c' * 64
            elif mutation == 'compute_source':
                report['native_sources'][NATIVE_SOURCES[0]] = 'c' * 64
            elif mutation == 'missing_source':
                report['sources'].pop(SOURCES[0])
            else:
                report['packer_zero_graft'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify(report, original['sources'], native)

    def test_incomplete_false_duplicate_or_boolean_index_audits_fail(self):
        original = self.fixture()
        for field in ('control_checks', 'eager_checks', 'replay_checks', 'negative_controls'):
            for mutation in ('missing', 'duplicate', 'false', 'bool_index'):
                report = deepcopy(original)
                rows = report[field]
                if mutation == 'missing':
                    rows.pop()
                elif mutation == 'duplicate':
                    rows[1] = rows[0]
                elif mutation == 'false':
                    rows[0]['stale_detected' if field == 'negative_controls' else 'exact_all_32_rows'] = 1
                else:
                    rows[0]['chip'] = False
                with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
                    qualify(report, original['sources'], original['native_sources'])

    def test_padding_geometry_and_cleanup_are_mandatory(self):
        original = self.fixture()
        for mutation in ('compared_rows', 'token_rows', 'closed_cleanly', 'native_compute_unchanged', 'geometry'):
            report = deepcopy(original)
            if mutation == 'geometry':
                report['geometry']['coordinates'][0][0] = False
            else:
                report[mutation] = 8 if mutation == 'compared_rows' else False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify(report, original['sources'], original['native_sources'])


if __name__ == '__main__':
    unittest.main()
