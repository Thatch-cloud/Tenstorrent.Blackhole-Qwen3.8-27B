import copy
import unittest

from cumulative_norm_runtime import PREFETCH_SHA256, SCATTER_SHA256
from cumulative_norm_validation import HISTORY_SHA256, validate_norm_history


def fixture(policy):
    checksum = PREFETCH_SHA256 if policy == 'prefetch' else SCATTER_SHA256
    return dict(incremental_history=dict(enabled=True, report_sha256=HISTORY_SHA256),
        gdn_norm_prefetch=dict(enabled=policy == 'prefetch', builds=48 if policy == 'prefetch' else 0,
            report_sha256=PREFETCH_SHA256 if policy == 'prefetch' else None),
        norm_reader=dict(policy=policy, builds=48, report_sha256=checksum, restored=True),
        gdn_shared_qk=dict(admission=dict(report_sha256=checksum), loads=[{}] * 48))


class NormValidationTests(unittest.TestCase):
    def test_legacy_prefetch_and_explicit_scatter(self):
        prefetch = fixture('prefetch')
        validate_norm_history(prefetch)
        del prefetch['norm_reader']
        validate_norm_history(prefetch)
        validate_norm_history(fixture('scatter'), 'scatter')
        with self.assertRaises(ValueError):
            validate_norm_history(fixture('scatter'))
        with self.assertRaises(ValueError):
            validate_norm_history(prefetch, 'scatter')

    def test_scatter_does_not_relax_history_or_source_identity(self):
        mutations = (
            lambda request: request['incremental_history'].update(enabled=False),
            lambda request: request['incremental_history'].update(report_sha256='changed'),
            lambda request: request['norm_reader'].update(restored=False),
            lambda request: request['norm_reader'].update(builds=96),
            lambda request: request['norm_reader'].update(report_sha256=PREFETCH_SHA256),
            lambda request: request['gdn_shared_qk']['admission'].update(report_sha256=PREFETCH_SHA256),
            lambda request: request['gdn_norm_prefetch'].update(enabled=True),
        )
        for mutate in mutations:
            request = copy.deepcopy(fixture('scatter'))
            mutate(request)
            with self.assertRaises(ValueError):
                validate_norm_history(request, 'scatter')
