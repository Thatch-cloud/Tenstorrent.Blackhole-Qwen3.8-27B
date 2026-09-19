from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import t32_combined_experiment as experiment


class CombinedExperimentTests(unittest.TestCase):
    def test_combined_request_owns_installation_and_requires_native_comparison(self):
        for compare in (True, False):
            events, report = [], dict(streams=1)
            installation, audit = {}, dict(native_proposal_checks=[])

            @contextmanager
            def installation_scope(*args, **kwargs):
                events.append('install')
                try:
                    yield installation
                finally:
                    events.append('restore')

            @contextmanager
            def admission(*args):
                events.append('admit')
                try:
                    yield
                finally:
                    events.append('revoke')

            @contextmanager
            def hardware(*args):
                events.append('mesh')
                try:
                    yield audit
                finally:
                    events.append('close')

            def run(*args, **kwargs):
                events.append('request')
                self.assertEqual(kwargs['max_new_tokens'], 65)
                self.assertEqual(kwargs['t32_attention_evidence'], 'attention.json')
                if compare:
                    audit['native_proposal_checks'].append(dict(exact=True))

            with patch.dict('os.environ', TT_METAL_HOME='/runtime'), \
                    patch.object(experiment, 'installed', installation_scope), \
                    patch.object(experiment, 'request_admission', admission), \
                    patch.object(experiment, 'hardware_scope', hardware), \
                    patch('dspark_request_experiment.run_loaded_requests', side_effect=run):
                def invoke():
                    experiment.run_loaded_requests(None, None, SimpleNamespace(mesh_device=object()),
                        None, None, None, None, None, None, None, None, None, report, None,
                        prompt=[1] * 4096, context={}, proposal_evidence='proposal.json',
                        score_evidence='score.json', attention_evidence='attention.json')
                if compare:
                    invoke()
                else:
                    with self.assertRaisesRegex(ValueError, 'native-score proposal'):
                        invoke()
            self.assertEqual(events, ['install', 'admit', 'mesh', 'request', 'close', 'revoke', 'restore'])
