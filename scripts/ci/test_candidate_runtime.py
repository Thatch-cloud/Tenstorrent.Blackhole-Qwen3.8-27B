from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from candidate_runtime import CandidateRuntime


class CandidateRuntimeTests(unittest.TestCase):
    def sampler(self):
        collective = SimpleNamespace(get_num_links=lambda axis=None: 2)
        return SimpleNamespace(tt_sampling=SimpleNamespace(mesh_device=SimpleNamespace(shape=(1, 2)),
            num_argmax_gather_links=1, tt_ccl=collective))

    def test_fixed_profile_lifecycle_and_delegation(self):
        sampler = self.sampler()
        engine = Mock(buckets={1: {}, 2: {}, 4: {}, 8: {}}, setup_ms=12, phase='idle')
        with patch('candidate_runtime.audit', return_value={'audited': True}), \
                patch('candidate_runtime.VerifierEngine', return_value=engine) as factory:
            runtime = CandidateRuntime('model', 'session', 'pages', 'helpers', sampler=sampler)
        factory.assert_called_once_with('model', 'session', 'pages', 'helpers', sampler=sampler,
            norm_batch=True, max_verify_rows=8)
        self.assertEqual(sampler.tt_sampling.num_argmax_gather_links, 4)
        self.assertEqual(runtime.setup_ms, 12)
        self.assertEqual(len(runtime.buckets), 4)
        self.assertIs(runtime.verify('ticket'), engine.verify.return_value)
        runtime.publish(3)
        engine.publish.assert_called_once_with(3)
        runtime.close()
        runtime.close()
        engine.close.assert_called_once()
        self.assertEqual(sampler.tt_sampling.num_argmax_gather_links, 1)
        self.assertEqual(sampler.tt_sampling.tt_ccl.get_num_links(), 2)
        with self.assertRaises(RuntimeError):
            runtime.verify('ticket')

    def test_construction_failure_restores_sampler(self):
        sampler = self.sampler()
        with patch('candidate_runtime.audit', return_value={}), \
                patch('candidate_runtime.VerifierEngine', side_effect=RuntimeError('capture failed')):
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                CandidateRuntime(None, None, None, None, sampler=sampler)
        self.assertEqual(sampler.tt_sampling.num_argmax_gather_links, 1)

    def test_unqualified_combinations_fail_before_audit(self):
        for options in (dict(norm_batch=False), dict(max_verify_rows=32), dict(attention_replay=True),
                dict(attention_mask_once=True), dict(replay_group_rows=8)):
            with patch('candidate_runtime.audit') as audit:
                with self.assertRaises(ValueError):
                    CandidateRuntime(None, None, None, None, sampler=self.sampler(), **options)
                audit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
