from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from prefill_prefix_experiment import run_loaded_requests
import test_dspark_request_experiment


class PrefixExperimentTests(unittest.TestCase):
    def fixture(self):
        report = dict(streams=1, target_index_sha256='a' * 64, target_config_sha256='b' * 64,
            target_sources={'target.py': 'c' * 64}, parameter_sha256={'layer': 'd' * 64},
            sources={'recipe.py': 'e' * 64}, native_sources={'native.cpp': 'f' * 64},
            target_capacity=dict(page_count=1024, cache_blocks=1032, block_size=64))
        arguments = (None, None, SimpleNamespace(), None, None,
            torch.arange(1024).reshape(1, -1), None, None, None, None, None, None, report, Mock())
        environment = dict(QWEN_PREFIX_CACHE_EXPERIMENT='1', QWEN_FROZEN_COMBINED_RUNTIME='1',
            QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        return arguments, environment

    def test_offline_route_wraps_complete_ladder_and_keeps_cached_pp_separate(self):
        import dspark_request_experiment
        import frozen_ladder_requests
        import full_dspark_request

        arguments, environment = self.fixture()
        report = arguments[-2]
        events, factory = [], object()
        requests = test_dspark_request_experiment.DSparkRequestExperimentTests().cached_requests()

        @contextmanager
        def session(*args, **kwargs):
            self.assertEqual(kwargs['prefix_position'], 2048)
            self.assertEqual(kwargs['inactive_pages'], list(range(1024, 1032)))
            events.append('allocate')
            try:
                yield factory
            finally:
                events.append('release')

        def finish(result, summarize):
            result['request_summary'] = summarize(requests)
            result['pp'] = result['request_summary']['pp']

        def run(*args, **kwargs):
            self.assertEqual(events, ['allocate'])
            for audit in (True, False, False):
                full_dspark_request.measure_dspark_request(audit_features=audit)
            frozen_ladder_requests.finish(report, dspark_request_experiment.summarize)

        with patch.dict('os.environ', environment, clear=True), \
                patch('prefill_prefix_experiment.offline_prefix_session', session), \
                patch.object(dspark_request_experiment, 'run_loaded_requests', side_effect=run), \
                patch.object(frozen_ladder_requests, 'finish', side_effect=finish), \
                patch.object(full_dspark_request, 'measure_dspark_request') as measure:
            run_loaded_requests(*arguments, prompt=list(range(4096)), captured_publication=True)
            self.assertEqual(measure.call_count, 3)
            self.assertTrue(all(call.kwargs['cached_prefill_factory'] is factory for call in measure.call_args_list))
        self.assertEqual(events, ['allocate', 'release'])
        self.assertTrue(report['prefix_cache_closed'])
        self.assertTrue(report['prefix_cache_experiment'])
        self.assertEqual(report['prefix_cache_sources'], report['prefix_cache_sources_after'])
        self.assertIsNone(report['pp'])
        self.assertEqual(report['effective_cached_pp'], 4096)

    def test_no_opt_in_and_partial_run_are_rejected(self):
        arguments, environment = self.fixture()
        with patch.dict('os.environ', {}, clear=True), self.assertRaisesRegex(ValueError, 'Explicit'):
            run_loaded_requests(*arguments, prompt=list(range(4096)), captured_publication=True)
        @contextmanager
        def session(*args, **kwargs):
            yield object()
        with patch.dict('os.environ', environment, clear=True), \
                patch('prefill_prefix_experiment.offline_prefix_session', session), \
                patch('dspark_request_experiment.run_loaded_requests'), \
                self.assertRaisesRegex(ValueError, 'complete combined'):
            run_loaded_requests(*arguments, prompt=list(range(4096)), captured_publication=True)


if __name__ == '__main__':
    unittest.main()
