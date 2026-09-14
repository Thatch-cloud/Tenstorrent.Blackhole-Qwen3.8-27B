import shutil
import subprocess
import unittest

from dspark_splitk_unfused_correction import guard_candidate


class NativeGuardTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cpp'), 'C preprocessor required')
    def test_native_does_not_reference_experiment_buffers(self):
        source = guard_candidate('native_decode;', 'experiment_cb_32; experiment_cb_33;')
        for enabled in (False, True):
            command = ['cpp', '-P'] + (['-DQWEN_SPLITK_NATIVE_EXPERIMENT=1'] if enabled else [])
            result = subprocess.run(command, input=source, text=True, capture_output=True, check=True).stdout
            self.assertEqual('experiment_cb_33' in result, enabled)
            self.assertEqual('native_decode' in result, not enabled)


if __name__ == '__main__':
    unittest.main()
