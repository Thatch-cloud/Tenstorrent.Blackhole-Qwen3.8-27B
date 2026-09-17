from types import SimpleNamespace
import unittest

from dspark_request_runtime import DSparkRequestRuntime
from dspark_t32_runtime import T32DSparkRequestRuntime
import test_dflash_request_runtime as baseline_tests


class T32RuntimeTests(unittest.TestCase):
    def fixture(self):
        _, drafter, session, engine = baseline_tests.DFlashRequestRuntimeTests().fixture()
        drafter.max_drafts = 31
        drafter.propose.return_value = tuple(range(11, 42))
        engine.verified_features_for_publication.return_value = tuple(range(32))
        runtime = T32DSparkRequestRuntime(drafter, position=170)
        runtime.bind(session, engine)
        session.phase = 'drafting'
        self.assertEqual(runtime('request', (9, 10), 31), tuple(range(11, 42)))
        ticket = SimpleNamespace(position=170, tokens=(10, *range(11, 42)))
        session.pending = engine.pending = ticket
        session.phase, engine.phase = 'committing', 'verified'
        return runtime, drafter, session, engine

    def test_every_prefix_updates_only_committed_frontier(self):
        for prefix in range(33):
            with self.subTest(prefix=prefix):
                runtime, drafter, _, engine = self.fixture()
                runtime.publish(prefix)
                self.assertEqual(runtime.position, 170 + prefix)
                self.assertEqual(drafter.position, 170 + prefix)
                self.assertEqual(runtime.committed_feature_rows, prefix)
                engine.publish.assert_called_once_with(prefix)
                if prefix:
                    drafter.prepare_publication.assert_called_once_with(tuple(range(32)), prefix, position=170)
                else:
                    drafter.prepare_publication.assert_not_called()

    def test_failed_target_never_commits_draft_features(self):
        runtime, drafter, _, engine = self.fixture()
        engine.publish.side_effect = RuntimeError('target failed')
        with self.assertRaisesRegex(RuntimeError, 'target failed'):
            runtime.publish(32)
        drafter.commit_publication.assert_not_called()
        drafter.discard_publication.assert_called_once()
        self.assertEqual(drafter.position, 170)

    def test_invalid_prefix_and_original_runtime_remain_rejected(self):
        for prefix in (-1, 33, True):
            runtime, drafter, _, engine = self.fixture()
            with self.assertRaises(ValueError):
                runtime.publish(prefix)
            engine.publish.assert_not_called()
            drafter.prepare_publication.assert_not_called()
        _, drafter, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            DSparkRequestRuntime(drafter, position=170)


if __name__ == '__main__':
    unittest.main()
