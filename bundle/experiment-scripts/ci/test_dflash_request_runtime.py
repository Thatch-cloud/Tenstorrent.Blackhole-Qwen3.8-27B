from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dflash_request_runtime import DFlashRequestRuntime, TARGET_TAPS


class DFlashRequestRuntimeTests(unittest.TestCase):
    def fixture(self):
        drafter = SimpleNamespace(position=170, propose=Mock(return_value=(11, 12, 13)),
            prepare_publication=Mock(side_effect=lambda features, prefix, **kwargs: (features, prefix)),
            discard_publication=Mock())
        def commit(publication):
            drafter.position += publication[1]
        drafter.commit_publication = Mock(side_effect=commit)
        session = SimpleNamespace(phase='idle', position=170, request_id='request', seed=10, vocab_size=100, pending=None)
        engine = SimpleNamespace(phase='idle', session=session, retain_feature_taps=TARGET_TAPS, pending=None,
            verified_features_for_publication=Mock(return_value=('seed-feature', 'accepted-feature', 'rejected-feature')),
            publish=Mock())
        runtime = DFlashRequestRuntime(drafter, position=170)
        runtime.bind(session, engine)
        return runtime, drafter, session, engine

    def pending(self, runtime, session, engine):
        session.phase = 'drafting'
        self.assertEqual(runtime('request', (9, 10), 3), (11, 12, 13))
        ticket = SimpleNamespace(position=170, tokens=(10, 11, 12, 13))
        session.pending = engine.pending = ticket
        session.phase, engine.phase = 'committing', 'verified'
        return ticket

    def test_rejection_publishes_only_processed_input_prefix(self):
        runtime, drafter, session, engine = self.fixture()
        self.pending(runtime, session, engine)
        runtime.publish(2)
        drafter.prepare_publication.assert_called_once_with(engine.verified_features_for_publication.return_value, 2, position=170)
        engine.publish.assert_called_once_with(2)
        self.assertEqual(runtime.position, 172)
        self.assertEqual(runtime.committed_feature_rows, 2)
        self.assertEqual(drafter.position, 172)

    def test_abort_has_no_feature_publication(self):
        runtime, drafter, session, engine = self.fixture()
        self.pending(runtime, session, engine)
        runtime.publish(0)
        engine.publish.assert_called_once_with(0)
        engine.verified_features_for_publication.assert_not_called()
        drafter.prepare_publication.assert_not_called()
        self.assertEqual(runtime.position, 170)

    def test_feature_preparation_failure_never_commits_target(self):
        runtime, drafter, session, engine = self.fixture()
        self.pending(runtime, session, engine)
        drafter.prepare_publication.side_effect = RuntimeError('projection failed')
        with self.assertRaisesRegex(RuntimeError, 'projection failed'):
            runtime.publish(2)
        engine.publish.assert_not_called()
        self.assertEqual((runtime.phase, engine.phase), ('failed', 'failed'))

    def test_target_failure_discards_unpublished_features(self):
        runtime, drafter, session, engine = self.fixture()
        self.pending(runtime, session, engine)
        engine.publish.side_effect = RuntimeError('target failed')
        with self.assertRaisesRegex(RuntimeError, 'target failed'):
            runtime.publish(2)
        drafter.commit_publication.assert_not_called()
        drafter.discard_publication.assert_called_once()
        self.assertEqual(drafter.position, 170)

    def test_stale_frontier_foreign_tickets_and_invalid_prefixes_rejected(self):
        for change in ('frontier', 'ticket', 'negative', 'overflow', 'bool'):
            runtime, drafter, session, engine = self.fixture()
            self.pending(runtime, session, engine)
            prefix = 2
            if change == 'frontier':
                runtime.position += 1
            elif change == 'ticket':
                engine.pending = object()
            else:
                prefix = {'negative': -1, 'overflow': 5, 'bool': True}[change]
            with self.assertRaises(ValueError):
                runtime.publish(prefix)
            drafter.prepare_publication.assert_not_called()

    def test_bad_proposal_ids_fail_before_verification(self):
        for result in ((11, 12), (11, True, 13), (11, 100, 13)):
            runtime, drafter, session, engine = self.fixture()
            drafter.propose.return_value = result
            session.phase = 'drafting'
            with self.assertRaises(ValueError):
                runtime('request', (10,), 3)
            self.assertEqual(runtime.phase, 'failed')
