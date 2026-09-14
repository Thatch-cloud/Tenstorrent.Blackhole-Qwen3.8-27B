from pathlib import Path
import subprocess
import unittest


class HardwareEntryTests(unittest.TestCase):
    def test_incompatible_trials_fail_before_evidence_or_device_access(self):
        script = Path(__file__).with_name('run-dspark-hardware.sh')
        environment = dict(PATH='/usr/bin:/bin', QWEN_CARDS_ALLOCATED='1',
            RUNNER_NAME='thatch-build-amd64-02-cp-temp',
            QWEN_DSPARK_64K_TRIAL='1', QWEN_DSPARK_MODE='request-norm-scatter',
            QWEN_DSPARK_CAPTURED_PUBLICATION='1')
        cases = dict(QWEN_DSPARK_MODE='request-target-attention',
            QWEN_DSPARK_CAPTURED_PUBLICATION='0', QWEN_DSPARK_FUSION_T16='1',
            QWEN_DSPARK_SCORE_LAYOUT='1', QWEN_DSPARK_BANKED_PROPOSAL='1',
            QWEN_DSPARK_NATIVE_SLOT='1', QWEN_DSPARK_MLP_DOWN='1',
            QWEN_DSPARK_BIAS_CACHE='1', QWEN_DSPARK_HISTORY_PROFILE='1',
            QWEN_DSPARK_DRAFT_PROFILE='1', QWEN_DSPARK_MLP_FOOTPRINT='1',
            QWEN_DSPARK_CODING_TASK='rotate_right_v1')
        for name, value in cases.items():
            with self.subTest(option=name):
                result = subprocess.run(['bash', str(script)],
                    env={**environment, name: value}, capture_output=True,
                    text=True, timeout=5)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, '')
                self.assertEqual(result.stderr, '')
