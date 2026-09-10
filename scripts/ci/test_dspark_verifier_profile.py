import copy
import unittest

from request_verifier_profile_report import validate_request
from test_request_verifier_profile import RequestProfileReportTests


class DSparkVerifierProfileTests(unittest.TestCase):
    def fixture(self):
        records, unused = RequestProfileReportTests().fixture()
        for record in records:
            record['rows'] = 16
            record['label'] = f'qwen_request_verify_{record["block"]}_pos{record["position"]}_t16_trace23'
        request = dict(length=4096, lookup_max_rows=16, commit_only_gdn=True, norm_batch=True,
            native_sampling_rows=True, exact=True, state_exact=True, inactive_exact=True,
            instrumented_timing=True, committed_tokens_per_second=None,
            dspark=dict(native_attention=True, proposal_trace=True, audit_features=True,
                feature_checks=[dict(exact=True)] * 30),
            blocks=[dict(position=record['position'], rows=16) for record in records],
            verifier_profile=dict(records=records, trace_counts={'23': 3}, full_rows=16,
                host_calls=[dict(function='verify')]))
        report = dict(passed=True, instrumented_timing=True, correctness_only=True, closed_cleanly=True,
            profile_family='dspark', committed_tg=None, sources={'code': 'same'}, sources_after={'code': 'same'},
            native_sources={'runtime': 'same'}, native_sources_after={'runtime': 'same'}, request_checks=[request])
        console = '\n'.join(f'QWEN_REQUEST_VERIFY_BEGIN {record["label"]}\nQWEN_REQUEST_VERIFY_END {record["label"]}'
            for record in records)
        return report, console

    def test_complete_t16_profile_qualifies_without_throughput(self):
        report, console = self.fixture()
        request, records = validate_request(report, console, family='dspark')
        self.assertEqual(len(records), 3)
        self.assertEqual(request['verifier_profile']['full_rows'], 16)

    def test_incomplete_changed_or_timed_request_is_rejected(self):
        original, console = self.fixture()
        for key, value in (('committed_tg', 100), ('closed_cleanly', False), ('sources_after', {'code': 'changed'})):
            report = copy.deepcopy(original)
            report[key] = value
            with self.assertRaises(ValueError):
                validate_request(report, console, family='dspark')
        report = copy.deepcopy(original)
        report['request_checks'][0]['dspark']['feature_checks'].pop()
        with self.assertRaises(ValueError):
            validate_request(report, console, family='dspark')
