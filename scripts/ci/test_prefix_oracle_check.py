"""prefix_oracle_check: the gates' oracle held to what the REAL scheduler graft committed, on CPU.

GOLDEN was recorded by driving vLLM 0.25.1's real TTScheduler and KVCacheManager with
prefix_scheduler_graft installed (prefix_oracle_check.run_graft; the probe step runs it in the
serving image). The oracle must agree with it request for request: a sequential gate arm FAILS on a
Q the oracle did not predict, so an oracle that drifted from the graft would fail good engines."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_judge as judge  # noqa: E402
import prefix_oracle_check as check  # noqa: E402


class GoldenTests(unittest.TestCase):
    def test_the_oracle_agrees_with_the_real_graft_request_for_request(self):
        rows = check.run_oracle()
        self.assertEqual([rid for rid, _, _, _ in rows], [rid for rid, _, _ in check.GOLDEN])
        for (rid, q, h, plan), (_, graft_q, graft_h) in zip(rows, check.GOLDEN):
            self.assertTrue(check.agrees(q, h, graft_q, graft_h), '%s: oracle Q=%d h=%d, graft Q=%d h=%s' % (
                rid, q, h, graft_q, graft_h))

    def test_the_golden_run_covers_the_design_cases(self):
        golden = dict((rid, (q, h)) for rid, q, h in check.GOLDEN)
        self.assertEqual(golden['chain-4'][0], 10240, 'a chain hits at the previous prompt\'s boundary')
        self.assertEqual(golden['chain-early'][0], 2048, 'an early divergence falls back to an older checkpoint')
        self.assertEqual((golden['chain-fresh-salt'][0], golden['chain-unsalted']), (0, (0, None)))
        self.assertEqual([golden['b%d-2' % n][0] for n in (2047, 2048, 2049, 4096)], [0, 2048, 2048, 4096])
        self.assertEqual(golden['b4096-3'], (0, 4032), 'the num_tokens-1 cap: P - 64, no checkpoint below it')
        self.assertEqual(golden['shared-2'][0], 4096, 'the gap capture of a shared block')
        self.assertEqual(golden['lru-main-3'], (0, 4096), 'the KV is cached but its checkpoint was evicted')

    def test_a_missing_grant_line_only_excuses_h(self):
        self.assertTrue(check.agrees(0, 1984, 0, None))
        self.assertFalse(check.agrees(2048, 2048, 0, None))
        self.assertFalse(check.agrees(2048, 2048, 2048, 2112))

    def test_scenarios_are_deterministic_and_chunk_sized(self):
        a, b = check.scenarios(), check.scenarios()
        self.assertEqual([(n, c, [(r, p, s) for r, p, s in steps]) for n, c, steps in a],
                         [(n, c, [(r, p, s) for r, p, s in steps]) for n, c, steps in b])
        self.assertEqual(judge.CHUNK, check.CHUNK)


if __name__ == '__main__':
    unittest.main()
