from pathlib import Path
import unittest

from frozen_wait_zone_capture import adapt_capture


class CaptureTests(unittest.TestCase):
    def test_preserves_request_gate_and_enables_raw_export(self):
        source = Path(__file__).with_name('dspark-combined-profile.sh').read_text()
        candidate = adapt_capture(source)
        self.assertNotIn('--disable-device-data-dump-to-files', candidate)
        self.assertIn('unset TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES', candidate)
        self.assertIn('--disable-device-data-push-to-tracy', candidate)
        self.assertIn('trap preserve_metadata EXIT', candidate)
        self.assertIn('test -s "$output/metadata/profile_log_device.csv"', candidate)
        self.assertIn('request_verifier_profile_report.py "$output" dspark', candidate)
        self.assertIn('--max-new-tokens 64', candidate)
        self.assertIn('timeout -k 30 4200', candidate)

    def test_double_adaptation_rejected(self):
        source = Path(__file__).with_name('dspark-combined-profile.sh').read_text()
        with self.assertRaises(ValueError):
            adapt_capture(adapt_capture(source))


if __name__ == '__main__':
    unittest.main()
