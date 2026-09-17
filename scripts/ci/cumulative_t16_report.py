"""Independently validate both cumulative components and full-request measurements."""

import argparse
import hashlib
import json
from pathlib import Path

from compact_score_combined_report import validate_route as validate_compact
from cumulative_t16_scope import validate_request
from gdn_direct_window_combined_report import validate as validate_direct
from mlp_down_grid_report import validate_route as validate_down


def validate(report):
    comparison = validate_direct(report)
    if (report.get('cumulative_t16') is not True
            or report.get('cumulative_components') not in (['direct_windows', 'compact_scores'],
                ['direct_windows', 'compact_scores', 'wider_mlp_down'])
            or not report.get('cumulative_sources')
            or report['cumulative_sources'] != report.get('cumulative_sources_after')
            or report.get('cumulative_measurement_quality') != comparison['measurement_quality']):
        raise ValueError('Complete source-stable two-component cumulative report required')
    audits = report.get('cumulative_route_diagnostics', [])
    if len(audits) != 3:
        raise ValueError('Three cumulative candidate route audits required')
    pending = iter(audits)
    with_down = 'wider_mlp_down' in report['cumulative_components']
    for request in report['request_checks']:
        validate_compact(request, 'publication')
        enabled = request['gdn_direct_window']['direct']
        if request['compact_score']['compact'] is not enabled:
            raise ValueError('Both components must be enabled on exactly the same requests')
        if with_down:
            validate_down(request, 'publication')
            if request['mlp_down_grid']['wider_down'] is not enabled:
                raise ValueError('Down-grid selection must match the cumulative arm')
        elif 'mlp_down_grid' in request:
            raise ValueError('Undeclared cumulative down-grid identity')
        if enabled:
            audit = next(pending)
            if ('down' in audit) != with_down:
                raise ValueError('Down-grid route audit must match the declared components')
            validate_request(request, audit)
    return dict(comparison, components=report['cumulative_components'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
