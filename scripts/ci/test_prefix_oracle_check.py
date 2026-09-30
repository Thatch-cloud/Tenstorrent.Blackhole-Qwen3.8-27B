"""prefix_oracle_check: the gates' oracle held to what the REAL scheduler graft committed, on CPU.

GOLDEN was recorded by driving vLLM 0.25.1's real TTScheduler and KVCacheManager with the scheduler
graft (qwen_prefix_scheduler_patch over qwen_prefix_registry) installed (prefix_oracle_check.run_graft;
the probe step runs it in the serving image). The oracle must agree with it request for request: a sequential gate arm FAILS on a
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

    def test_a_changed_golden_fails_even_when_the_oracle_agrees(self):
        said = []
        self.assertEqual(check.verdict(list(check.GOLDEN), said.append), 0)
        self.assertIn('GOLDEN unchanged', said[-1])
        rows = list(check.GOLDEN)
        index = [rid for rid, _, _ in rows].index('b2047-1')
        oracle_h = dict((rid, h) for rid, _, h, _ in check.run_oracle())['b2047-1']
        rows[index] = ('b2047-1', 0, oracle_h)    # a grant line now printed where there was none
        said = []
        self.assertEqual(check.verdict(rows, said.append), 1)
        self.assertTrue(any('0 mismatches; GOLDEN CHANGED' in line for line in said), said)
        rows[index] = ('b2047-1', 2048, oracle_h)
        self.assertEqual(check.verdict(rows, lambda line: None), 1)

    def test_the_graft_driven_is_the_images_when_it_has_one(self):
        shas = dict(('/%s/%s' % (where, name), 'aa') for where in ('c2', 'img') for name in check.GRAFT_FILES)
        self.assertEqual(check.choose_graft('/c2', '/img', exists=lambda path: False, sha=shas.get), ('/c2', None))
        self.assertEqual(check.choose_graft('/c2', '/img', exists=lambda path: True, sha=shas.get), ('/img', None))
        shas['/img/qwen_prefix_registry.py'] = 'bb'
        path, problem = check.choose_graft('/c2', '/img', exists=lambda path: True, sha=shas.get)
        self.assertEqual(path, '/img')
        self.assertIn('does not run', problem)
        self.assertIn('qwen_prefix_registry.py', problem)
        path, problem = check.choose_graft('/c2', '/img', exists=lambda path: path.endswith('registry.py'),
                                           sha=shas.get)
        self.assertIn('could not install the graft', problem)
        self.assertEqual(check.GRAFT_FILES, ('qwen_prefix_registry.py', 'qwen_prefix_scheduler_patch.py'))

    def test_the_image_graft_is_where_the_overlay_lays_the_plugin_copies(self):
        """The patched TTScheduler imports the graft from its own package: the overlay manifest lays both
        files there (test_qwen_prefix_image_closure), and that is the copy the engine runs."""
        import c2_overlay

        manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docker', 'qwen-c2-overlay.txt')
        laid = set()
        for entry in c2_overlay.read_manifest(manifest):
            laid.update(entry.destinations)
        for name in check.GRAFT_FILES:
            self.assertIn(check.graft_file(check.IMAGE_GRAFT, name), laid)
        self.assertEqual(check.IMAGE_GRAFT + '/', c2_overlay.PLUGIN)

    def test_scenarios_are_deterministic_and_chunk_sized(self):
        a, b = check.scenarios(), check.scenarios()
        self.assertEqual([(n, c, [(r, p, s) for r, p, s in steps]) for n, c, steps in a],
                         [(n, c, [(r, p, s) for r, p, s in steps]) for n, c, steps in b])
        self.assertEqual(judge.CHUNK, check.CHUNK)


if __name__ == '__main__':
    unittest.main()
