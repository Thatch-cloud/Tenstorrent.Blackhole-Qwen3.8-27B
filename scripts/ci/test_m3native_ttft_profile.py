"""The admission profile, checked against the two runs it was derived from.

v34 (35658854824) is the serial-prefill case with no synchronised stall; v35
(35667198294) is the same serial prefill PLUS a synchronised mid-decode stall that
raising MAX_PREFILL_CHUNK_SIZE introduced. A profile that cannot tell those apart is
useless, so both are pinned here with their real numbers.
"""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from m3native_ttft_profile import FLAG_MAX_STALL, FLAG_MAX_TTFT, profile, render, synchronised_stalls


def stream(ttft, first_gap, steady=253.0, rounds=40, extra=None, wall=65.0):
    gaps = [first_gap * 1000.0] + [steady] * rounds
    for index, seconds in (extra or {}).items():
        gaps[index] = seconds * 1000.0
    return dict(ttft_s=ttft, gaps_ms=gaps, wall_s=wall, tokens=rounds + 1)


# Run 35658854824, rounded to the recorded precision.
V34 = [stream(13.5, 39.5, wall=65.5), stream(39.6, 13.4, steady=256.8, wall=62.9),
       stream(26.5, 26.5, steady=253.5, wall=65.1), stream(52.5, 0.5, steady=258.0, wall=63.9)]

# Run 35667198294: same shape, plus all four users stalling on rounds 27-30.
SYNC = {27: 1.3, 28: 1.7, 29: 2.2, 30: 1.2}
V35 = [stream(14.3, 41.9, extra=SYNC, wall=71.0), stream(41.8, 14.4, extra=SYNC, wall=68.0),
       stream(27.9, 28.3, extra=SYNC, wall=70.0), stream(55.5, 0.7, extra=SYNC, wall=69.0)]


class SerialPrefillTests(unittest.TestCase):
    def test_v34_is_serial_and_fully_explained_by_admission_order(self):
        with patch.dict(os.environ, {}, clear=True):
            result = profile(V34)
        self.assertEqual(result['users'], 4)
        self.assertEqual(result['ttft_s'], [13.5, 26.5, 39.6, 52.5])
        self.assertEqual(result['ttft_deltas_s'], [13.0, 13.1, 12.9])
        self.assertTrue(result['prefill_serial'])
        self.assertAlmostEqual(result['prefill_interval_s'], 13.0, places=1)
        # Each user's first gap IS the remaining prefill time.
        self.assertEqual(result['first_gap_s'], [39.5, 26.5, 13.4, 0.5])
        self.assertEqual(result['predicted_stall_s'], [39.0, 26.0, 13.0, 0.0])
        self.assertTrue(result['stall_explained_by_admission'])
        self.assertAlmostEqual(result['stall_total_s'], 79.4, places=1)
        self.assertGreater(result['stall_share'], 0.30)
        self.assertEqual(result['synchronised_stalls'], [])

    def test_v35_is_still_serial_but_carries_a_synchronised_stall(self):
        with patch.dict(os.environ, {}, clear=True):
            result = profile(V35)
        self.assertTrue(result['prefill_serial'])
        self.assertTrue(result['stall_explained_by_admission'])
        # The signature the chunk-size regression produced, which admission order cannot explain.
        self.assertEqual([e['round'] for e in result['synchronised_stalls']], [27, 28, 29, 30])
        for entry in result['synchronised_stalls']:
            self.assertEqual(entry['users'], 4)
        self.assertGreater(result['stall_total_s'], profile(V34)['stall_total_s'])

    def test_interleaved_prefill_is_not_reported_as_serial(self):
        """What a fixed run should look like: everyone admitted close together, and no
        first gap anywhere near a prefill interval."""
        interleaved = [stream(14.0 + 0.3 * i, 0.4) for i in range(4)]
        with patch.dict(os.environ, {}, clear=True):
            result = profile(interleaved)
        self.assertFalse(result['prefill_serial'])
        self.assertLess(result['stall_total_s'], 1.0)
        self.assertEqual(result['synchronised_stalls'], [])

    def test_first_gap_is_never_counted_as_a_synchronised_stall(self):
        """Every user's gap 0 is the admission stall. Counting it under the synchronised
        name would report the serial signature twice as if it were two findings."""
        self.assertEqual(synchronised_stalls(V34), [])
        self.assertTrue(all(e['round'] for e in synchronised_stalls(V35)))


class ThresholdTests(unittest.TestCase):
    def test_reporting_is_unconditional_and_assertion_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            result = profile(V34)
        self.assertFalse(result['thresholds_checked'])
        self.assertEqual(result['failures'], [])

    def test_stall_ceiling_fails_when_exceeded(self):
        with patch.dict(os.environ, {FLAG_MAX_STALL: '20'}, clear=True):
            result = profile(V34)
        self.assertTrue(result['thresholds_checked'])
        self.assertEqual(len(result['failures']), 1)
        self.assertIn('79.4', result['failures'][0])

    def test_ttft_ceiling_uses_the_worst_user(self):
        with patch.dict(os.environ, {FLAG_MAX_TTFT: '30'}, clear=True):
            result = profile(V34)
        self.assertIn('52.5', result['failures'][0])
        with patch.dict(os.environ, {FLAG_MAX_TTFT: '60'}, clear=True):
            self.assertEqual(profile(V34)['failures'], [])

    def test_malformed_thresholds_raise_rather_than_silently_skipping(self):
        for bad in ('yes', '', '0', '-5'):
            with self.subTest(value=bad), patch.dict(os.environ, {FLAG_MAX_STALL: bad}, clear=True):
                with self.assertRaises(ValueError):
                    profile(V34)

    def test_empty_streams_refused(self):
        with self.assertRaises(ValueError):
            profile([])
        with self.assertRaises(ValueError):
            profile([None, None])


class RenderTests(unittest.TestCase):
    def test_render_names_the_signatures_it_found(self):
        with patch.dict(os.environ, {}, clear=True):
            text = render(profile(V34))
        self.assertIn('SERIAL', text)
        self.assertIn('explained by admission order', text)
        self.assertIn('79.4 s of', text)
        self.assertNotIn('SYNCHRONISED', text)

    def test_render_surfaces_a_synchronised_stall(self):
        with patch.dict(os.environ, {}, clear=True):
            text = render(profile(V35))
        self.assertIn('SYNCHRONISED    round 27', text)
        self.assertIn('4 users', text)

    def test_render_surfaces_failures(self):
        with patch.dict(os.environ, {FLAG_MAX_STALL: '20'}, clear=True):
            text = render(profile(V34))
        self.assertIn('FAIL', text)


if __name__ == '__main__':
    unittest.main()
