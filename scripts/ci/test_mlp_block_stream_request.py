import unittest

from mlp_block_stream_request import validate_request, REPORT_SHA256
from test_cumulative_fusion_validation import fixture


class BlockStreamRequestTests(unittest.TestCase):
    def fixture(self):
        request = fixture(True)
        request['fused_t16_mlp'].update(passed_simulator=REPORT_SHA256, extra_weight_allocations=64)
        request['block_stream'] = dict(report_sha256=REPORT_SHA256, restored=True, constructions=64,
            stream_allocations=64, constructed_layers=list(range(64)), calls=128, serving_defaults_changed=False)
        return request

    def test_all_target_weight_checks_and_explicit_allocation_required(self):
        validate_request(self.fixture())
        for mutate in (
                lambda value: value['fused_t16_mlp'].update(extra_weight_allocations=0),
                lambda value: value['fused_t16_mlp']['weight_audit']['checks'].pop(),
                lambda value: value['block_stream'].update(calls=127),
                lambda value: value['block_stream'].update(constructed_layers=[0] * 64),
                lambda value: value['block_stream'].update(restored=False),
                lambda value: value['register_epilogue'].update(calls=127)):
            request = self.fixture()
            mutate(request)
            with self.assertRaises(ValueError):
                validate_request(request)

    def test_pipeline_requires_its_own_reader_and_simulator_identity(self):
        from mlp_block_stream_pipeline_gate import REPORT_SHA256 as PIPELINE_SHA256, READER_SHA256

        request = self.fixture()
        request['block_stream'].update(bulk_pipeline=True, pipeline_reader_sha256=READER_SHA256)
        with self.assertRaises(ValueError):
            validate_request(request)
        request['block_stream']['report_sha256'] = PIPELINE_SHA256
        request['fused_t16_mlp']['passed_simulator'] = PIPELINE_SHA256
        validate_request(request)
        request['block_stream']['pipeline_reader_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            validate_request(request)


if __name__ == '__main__':
    unittest.main()
