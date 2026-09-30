from types import SimpleNamespace
import unittest

from drafter_request_metadata import record_request
from sampling_link_policy import SOURCES


class RequestMetadataTests(unittest.TestCase):
    def test_completion_is_derived_and_request_retained_before_validation(self):
        for drafter in ('dspark', 'dflash2'):
            for last, ended in ((42, True), (43, False)):
                report = dict(sampling_link_sources=dict(SOURCES), request_checks=[])
                result = dict(emitted=[1, last], eos_ids=[42])
                record_request(report, result, drafter, SimpleNamespace(num_argmax_gather_links=4), dict(SOURCES))
                self.assertIs(result['ended_with_eos'], ended)
                self.assertIs(report['request_checks'][0], result)
                self.assertEqual(result['fabric_sources'], SOURCES)

    def test_wrong_sampler_or_source_identity_cannot_be_annotated_as_four_links(self):
        for links, sources in ((1, SOURCES), (True, SOURCES), (4, {})):
            report = dict(sampling_link_sources=dict(SOURCES), request_checks=[])
            with self.assertRaises(ValueError):
                record_request(report, dict(emitted=[1], eos_ids=[1]), 'dflash2',
                               SimpleNamespace(num_argmax_gather_links=links), sources)
            self.assertEqual(report['request_checks'], [])
