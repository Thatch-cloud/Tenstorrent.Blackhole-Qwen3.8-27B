"""Full target-route validation with explicit additional weight allocations."""

from mlp_block_stream_gate import REPORT_SHA256
from mlp_register_epilogue_gate import REPORT_SHA256 as REGISTER_SHA256
from mlp_weight_pipeline_report import validate_fusion


def validate_request(request):
    validate_fusion(request, REPORT_SHA256, expected_extra_weight_allocations=64)
    audit = request.get('block_stream', {})
    register = request.get('register_epilogue', {})
    hits = request['fused_t16_mlp']['hits']
    if (audit.get('report_sha256') != REPORT_SHA256 or audit.get('restored') is not True
            or audit.get('constructions') != 64 or audit.get('stream_allocations') != 64
            or sorted(audit.get('constructed_layers', [])) != list(range(64))
            or audit.get('calls') != sum(hits) or audit.get('serving_defaults_changed') is not False
            or register != dict(register_resident=True, report_sha256=REGISTER_SHA256,
                constructions=64, calls=sum(hits), restored=True)):
        raise ValueError('Complete executed block-stream route and unchanged register arithmetic required')
