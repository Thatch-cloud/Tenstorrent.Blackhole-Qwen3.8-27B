"""a0_power: deterministic, monotone in the true R, in line with the registered claims, aggregates only."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a0_power as power  # noqa: E402
import tau_lab_report as rep  # noqa: E402


def screen_shaped_meta():
    """The screen's size: 240 swe turns in 60 clusters, 96 own turns in 35 clusters (about 2.7 turns each), real-looking lengths."""
    rows, k = [], 0
    for name, turns, clusters, length in (('swe', 240, 60, 400), ('own', 96, 35, 700)):
        for at in range(turns):
            rows.append(dict(k=k, set=name, cluster=at % clusters, weight=1.0 + (at % 3), answer_tokens=length + 13 * (at % 7)))
            k += 1
    return rows


class PowerTests(unittest.TestCase):
    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        meta = screen_shaped_meta()
        a = power.simulate(meta, (1.10,), trials=20, seed=1, resamples=60)
        self.assertEqual(a, power.simulate(meta, (1.10,), trials=20, seed=1, resamples=60))
        self.assertNotEqual(a, power.simulate(meta, (1.10,), trials=20, seed=2, resamples=60))

    def test_the_go_share_rises_with_the_true_ratio_and_the_kill_share_falls(self):
        table = power.simulate(screen_shaped_meta(), (1.00, 1.10, 1.25), trials=60, seed=3, resamples=80)
        go = [row['go_point'] for row in table]
        kill = [row['kill'] for row in table]
        self.assertTrue(go[0] < go[1] < go[2])
        self.assertTrue(kill[0] > kill[1] >= kill[2])
        self.assertLess(go[0], 0.1)
        self.assertGreater(go[2], 0.95)

    def test_the_registered_power_claims(self):
        """A true R of 1.10 passes 'point >= 1.10' about half the time; a true R of 1.15 passes with high probability."""
        table = dict((row['r'], row) for row in power.simulate(screen_shaped_meta(), (1.10, 1.15), trials=120, seed=4, resamples=80))
        self.assertTrue(0.3 <= table[1.10]['go_point'] <= 0.7, table[1.10])
        self.assertGreater(table[1.15]['go_point'], 0.85)
        # the interval half-width sits in the registered +/-2-7% band
        self.assertTrue(0.01 <= table[1.10]['halfwidth'] <= 0.08, table[1.10])

    def test_public_summary_is_numbers_only(self):
        table = power.simulate(screen_shaped_meta(), (1.10,), trials=10, seed=5, resamples=40)
        summary = power.public_summary(table, 10, 5)
        rep.assert_public(summary)
        self.assertIn('r_1_10', summary)

    def test_cli_writes_the_summary(self):
        root = tempfile.mkdtemp()
        try:
            meta = os.path.join(root, 'meta.jsonl')
            with open(meta, 'w') as handle:
                for row in screen_shaped_meta()[:80]:
                    handle.write(json.dumps(row) + '\n')
            out = os.path.join(root, 'power.json')
            lines = []
            self.assertEqual(power.main(['--meta', meta, '--out', out, '--trials', '5', '--resamples', '30'], say=lines.append), 0)
            with open(out) as handle:
                self.assertIn('r_1_00', json.load(handle))
            self.assertEqual(len(lines), len(power.DEFAULT_R))
        finally:
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
