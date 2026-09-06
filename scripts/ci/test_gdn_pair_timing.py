import unittest

from gdn_pair_timing import paired_replays


class PairTimingTests(unittest.TestCase):
    def test_restores_before_each_timer_and_uses_abba(self):
        events = []
        now = 0
        def clock():
            nonlocal now
            events.append("clock")
            now += 1
            return now
        result = paired_replays(lambda: events.append("restore"), lambda: events.append("sync"),
                                lambda arm: events.append(arm), lambda arm: events.append("validate:" + arm), clock)
        expected = []
        for _ in range(3):
            for arm in ("control", "candidate", "candidate", "control"):
                for _ in range(10):
                    expected.extend(["restore", "sync", "clock", arm, "sync", "clock"])
                expected.append("validate:" + arm)
        self.assertEqual(events, expected)
        self.assertEqual(result["median_paired_ratio"], 1)
        self.assertEqual(len(result["samples"]), 12)

    def test_validation_failure_stops_measurement(self):
        replays = []
        def fail(arm):
            raise AssertionError("mismatch")
        with self.assertRaisesRegex(AssertionError, "mismatch"):
            paired_replays(lambda: None, lambda: None, replays.append, fail)
        self.assertEqual(replays, ["control"] * 10)


if __name__ == "__main__":
    unittest.main()
