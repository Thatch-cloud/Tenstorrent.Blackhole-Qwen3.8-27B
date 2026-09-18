"""Full target-route validation with explicit additional weight allocations."""

from mlp_block_stream_gate import REPORT_SHA256
from mlp_register_epilogue_gate import REPORT_SHA256 as REGISTER_SHA256
from mlp_weight_pipeline_report import validate_fusion


def validate_request(request):
    audit = request.get('block_stream', {})
    expected = REPORT_SHA256
    pipeline = audit.get('bulk_pipeline', False)
    progressive = audit.get('progressive_input', False)
    if type(progressive) is not bool or (progressive and pipeline):
        raise ValueError('Progressive input requires the separately admitted serial weight path')
    if type(pipeline) is not bool:
        raise ValueError('Explicit bulk-pipeline route identity required')
    if pipeline:
        from mlp_block_stream_pipeline_gate import REPORT_SHA256 as expected, READER_SHA256

        if audit.get('pipeline_reader_sha256') != READER_SHA256:
            raise ValueError('Exact simulator-qualified pipeline reader required')
    elif 'pipeline_reader_sha256' in audit:
        raise ValueError('Serial reader cannot claim pipeline execution')
    if progressive:
        from mlp_progressive_input_gate import REPORT_SHA256 as expected, READER_SHA256

        if (audit.get('input_reader_sha256') != READER_SHA256
                or audit.get('extra_l1_bytes_per_multicast_core') != 144 * 2048):
            raise ValueError('Exact simulator-qualified progressive input route required')
    elif 'input_reader_sha256' in audit:
        raise ValueError('Control cannot claim progressive activation delivery')
    validate_fusion(request, expected, expected_extra_weight_allocations=64)
    register = request.get('register_epilogue', {})
    hits = request['fused_t16_mlp']['hits']
    if (audit.get('report_sha256') != expected or audit.get('restored') is not True
            or audit.get('constructions') != 64 or audit.get('stream_allocations') != 64
            or sorted(audit.get('constructed_layers', [])) != list(range(64))
            or audit.get('calls') != sum(hits) or audit.get('serving_defaults_changed') is not False
            or register != dict(register_resident=True, report_sha256=REGISTER_SHA256,
                constructions=64, calls=sum(hits), restored=True)):
        raise ValueError('Complete executed block-stream route and unchanged register arithmetic required')
