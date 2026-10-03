from pathlib import Path
import copy
import unittest
from unittest.mock import patch

from dspark_score_layout_hardware_gate import qualify, validate_hardware
from dspark_score_layout_hardware_audit import audit


class HardwareAdmissionTests(unittest.TestCase):
    def evidence(self):
        stages = [(pattern, 'eager') for pattern in range(2)] + [
            (pattern, f'replay_{repetition}') for repetition, pattern in enumerate((0, 1, 0))]
        return dict(passed=True, released_cleanly=True, vocabulary=248320, proposals=15,
            admission=qualify(Path(__file__).parent),
            eager_checks=[dict(pattern=pattern, repetition=None, step=step, chip=chip,
                token_exact=True, scores_exact=True) for pattern in range(2) for step in range(15) for chip in range(2)],
            replay_checks=[dict(pattern=pattern, repetition=repetition, step=step, chip=chip,
                token_exact=True, scores_exact=True) for repetition, pattern in enumerate((0, 1, 0))
                for step in range(15) for chip in range(2)],
            input_checks=[dict(pattern=pattern, stage=stage, operand=operand, chip=chip, exact=True)
                for pattern, stage in stages for operand in range(2) for chip in range(2)],
            weight_hashes_before=[['a' * 64, 'a' * 64], ['b' * 64, 'b' * 64]],
            weight_hashes_after=[['a' * 64, 'a' * 64], ['b' * 64, 'b' * 64]])

    def test_complete_matrix_and_reject_missing_or_failed_checks(self):
        report = self.evidence()
        self.assertEqual(len(validate_hardware(report, Path(__file__).parent)), 64)
        for mutation in (
                lambda value: value['eager_checks'].pop(),
                lambda value: value['replay_checks'].pop(),
                lambda value: value['input_checks'].pop(),
                lambda value: value['eager_checks'][0].update(scores_exact=False),
                lambda value: value['replay_checks'][0].update(token_exact=False),
                lambda value: value.update(released_cleanly=False),
                lambda value: value.update(admission={}),
                lambda value: value['weight_hashes_after'][0].__setitem__(0, 'c' * 64)):
            changed = copy.deepcopy(report)
            mutation(changed)
            with self.assertRaises(ValueError):
                validate_hardware(changed, Path(__file__).parent)

    def test_admission_is_not_hardware_or_speed_qualification(self):
        result = qualify(Path(__file__).parent)
        self.assertTrue(result['hardware_correctness_required'])
        self.assertFalse(result['timing_qualified'])
        self.assertEqual(result['small_feedback']['replay_checks'], 120)
        self.assertEqual(result['score_layout']['results'][1]['vocabulary'], 248320)

    def test_simulator_cannot_claim_hardware_audit(self):
        with patch.dict('os.environ', {'TT_METAL_SIMULATOR': 'simulator'}, clear=True):
            with self.assertRaises(RuntimeError):
                audit(None, None, None, None)


if __name__ == '__main__':
    unittest.main()
