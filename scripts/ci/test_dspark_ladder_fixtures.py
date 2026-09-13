import unittest

from dspark_ladder_fixtures import fixture_probe
from dspark_ladder_geometry import CONTEXTS


class LadderFixtureTests(unittest.TestCase):
    def test_all_contexts_detect_missing_history_proposals_and_frontier_changes(self):
        import torch
        with torch.inference_mode():
            for context in CONTEXTS:
                with self.subTest(context=context), fixture_probe(context) as probe:
                    patterns = probe.fixtures()
                    expected = [[probe.reference(values, chip) for chip in range(2)] for values in patterns]
                    checks = probe.controls(patterns, expected)
                    self.assertEqual(len(checks), 8)
                    self.assertTrue(all(check['detected'] for check in checks))

    def test_context_limit_restores_after_failure(self):
        import dspark_full_attention
        original = dspark_full_attention.MAX_CONTEXT
        with self.assertRaises(RuntimeError):
            with fixture_probe(65536):
                raise RuntimeError('abort')
        self.assertEqual(dspark_full_attention.MAX_CONTEXT, original)
