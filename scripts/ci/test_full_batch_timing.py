import unittest

from full_batch_timing import summarize


class TimingTests(unittest.TestCase):
    def test_paired_means_not_mixed_sample_ratio(self):
        samples = [dict(arm=arm, milliseconds=value) for arm, value in
                   zip(("serial", "batch", "batch", "serial"), (10, 3, 5, 14))]
        self.assertEqual(summarize(samples), dict(serial_ms=12, batch_ms=4, speedup=3))

    def test_rejects_incomplete_or_reordered_blocks(self):
        for arms in (("serial",), ("serial", "serial", "batch", "batch")):
            with self.assertRaises(ValueError):
                summarize([dict(arm=arm, milliseconds=1) for arm in arms])

    def test_rejects_nonpositive_latency(self):
        samples = [dict(arm=arm, milliseconds=0) for arm in ("serial", "batch", "batch", "serial")]
        with self.assertRaises(ValueError):
            summarize(samples)


if __name__ == "__main__":
    unittest.main()
