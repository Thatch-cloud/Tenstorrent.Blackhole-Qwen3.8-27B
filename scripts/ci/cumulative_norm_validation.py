"""Validate explicit norm alternatives without relaxing incremental-history identity."""

from cumulative_norm_runtime import validate_identity
from frozen_gdn_norm_gate import REPORT_SHA256 as PREFETCH_SHA256
from history_append_hardware_gate import REPORT_SHA256 as HISTORY_SHA256


def validate_norm_history(request, policy='prefetch'):
    if policy not in ('prefetch', 'scatter'):
        raise ValueError('Known normalization policy required')
    history = request.get('incremental_history', {})
    if history.get('enabled') is not True or history.get('report_sha256') != HISTORY_SHA256:
        raise ValueError('Unchanged qualified incremental history required')
    if policy == 'scatter' or 'norm_reader' in request:
        validate_identity(request)
        if request['norm_reader']['policy'] != policy:
            raise ValueError('Normalization execution differs from the declared cumulative arm')
        if request['norm_reader']['builds'] != len(request.get('gdn_shared_qk', {}).get('loads', [])):
            raise ValueError('Normalization must cover every shared-Q/K build')
        return
    norm = request.get('gdn_norm_prefetch', {})
    if (norm.get('enabled') is not True or norm.get('report_sha256') != PREFETCH_SHA256
            or type(norm.get('builds')) is not int or norm['builds'] < 48 or norm['builds'] % 48):
        raise ValueError('Unchanged qualified norm prefetch required')
