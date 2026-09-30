from types import SimpleNamespace
import unittest

from dspark_request_runtime import DSparkRequestRuntime
import test_dflash_request_runtime as baseline_tests


class ConcurrentProtocolTests(unittest.TestCase):
    def fixture(self, request_id, position):
        _, drafter, session, engine = baseline_tests.DFlashRequestRuntimeTests().fixture()
        drafter.max_drafts = 15
        drafter.position = session.position = position
        session.request_id = request_id
        drafter.propose.return_value = tuple(range(11, 26))
        engine.verified_features_for_publication.return_value = (request_id, position)
        runtime = DSparkRequestRuntime(drafter, position=position)
        runtime.bind(session, engine)
        return SimpleNamespace(runtime=runtime, drafter=drafter, session=session, engine=engine)

    def propose(self, owner):
        owner.session.phase = 'drafting'
        tokens = owner.runtime(owner.session.request_id, (owner.session.seed,), 15)
        ticket = SimpleNamespace(position=owner.runtime.position, tokens=(owner.session.seed, *tokens))
        owner.session.pending = owner.engine.pending = ticket
        owner.session.phase, owner.engine.phase = 'committing', 'verified'
        return ticket

    def test_interleaved_prefixes_keep_independent_frontiers(self):
        for prefix in range(17):
            with self.subTest(prefix=prefix):
                first = self.fixture('first', 4096)
                second = self.fixture('second', 8192)
                self.propose(first)
                self.propose(second)
                second.runtime.publish(16 - prefix)
                self.assertEqual(first.runtime.position, 4096)
                first.engine.publish.assert_not_called()
                first.drafter.commit_publication.assert_not_called()
                first.runtime.publish(prefix)
                self.assertEqual(first.runtime.position, 4096 + prefix)
                self.assertEqual(second.runtime.position, 8208 - prefix)
                for owner, count, position in ((first, prefix, 4096), (second, 16 - prefix, 8192)):
                    self.assertEqual(owner.drafter.position, position + count)
                    if count:
                        owner.drafter.prepare_publication.assert_called_once_with(
                            (owner.session.request_id, position), count, position=position)
                    else:
                        owner.drafter.prepare_publication.assert_not_called()

    def test_foreign_request_rejected_before_drafting(self):
        owner = self.fixture('first', 4096)
        owner.session.phase = 'drafting'
        with self.assertRaises(ValueError):
            owner.runtime('second', (owner.session.seed,), 15)
        owner.drafter.propose.assert_not_called()
        self.assertEqual(owner.runtime.phase, 'idle')

    def test_foreign_ticket_rejected_even_at_identical_position(self):
        first = self.fixture('first', 4096)
        second = self.fixture('second', 4096)
        self.propose(first)
        foreign_ticket = self.propose(second)
        first.session.pending = foreign_ticket
        with self.assertRaises(ValueError):
            first.runtime.publish(8)
        first.engine.publish.assert_not_called()
        first.drafter.prepare_publication.assert_not_called()
        second.runtime.publish(8)
        self.assertEqual(second.runtime.position, 4104)

    def test_failed_publication_does_not_poison_other_request(self):
        first = self.fixture('first', 4096)
        second = self.fixture('second', 4096)
        self.propose(first)
        self.propose(second)
        first.engine.publish.side_effect = RuntimeError('target failure')
        with self.assertRaisesRegex(RuntimeError, 'target failure'):
            first.runtime.publish(8)
        first.drafter.commit_publication.assert_not_called()
        first.drafter.discard_publication.assert_called_once()
        second.runtime.publish(16)
        self.assertEqual(first.runtime.position, 4096)
        self.assertEqual(second.runtime.position, 4112)


if __name__ == '__main__':
    unittest.main()
