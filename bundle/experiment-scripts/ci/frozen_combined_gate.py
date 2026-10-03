"""Pinned 32K component evidence for an explicit combined-runtime candidate."""

import hashlib
import json
from pathlib import Path

from frozen_probe_evidence import validate_pair
from frozen_target_replay import validate_target_report


REPORTS = {
    'draft-numerical.json': '091cbdd2ad2b01a9c529083ef1ac11bbc78784fae59f471b4daf075420fbccef',
    'draft-diagnostics.json': '1dc508f7235cea9a52f6c34cb928581895dc6376a3cf70e7ff7644697df5820b',
    'target-replay.json': '92c5187536e53739d00536216d0d00d9c901b4963648bb6bfda5195b593007e0',
}


def load_reports(directory):
    reports = {}
    for name, expected in REPORTS.items():
        payload = (Path(directory) / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Pinned component report required: ' + name)
        reports[name] = json.loads(payload)
    return reports


def qualify(directory, *, draft_sources, target_sources, context):
    if type(context) is not int or context != 32768:
        raise ValueError('Only 32768 has both retained candidate qualifications')
    reports = load_reports(directory)
    draft = validate_pair(reports['draft-numerical.json'], reports['draft-diagnostics.json'], context)
    if draft['reciprocal_variant'] != 'scalar-fp32':
        raise ValueError('Explicit qualified draft reciprocal required')
    for name, expected in reports['draft-numerical.json']['sources'].items():
        if hashlib.sha256((Path(draft_sources) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Qualified draft source changed: ' + name)
    target = validate_target_report(reports['target-replay.json'], target_sources, context,
        compact_scratch=True)
    return dict(context=context, capacity=context + 256, reports=dict(REPORTS),
        draft=draft, target=target, component_sources_verified=True,
        combined_runtime_qualified=False, performance_qualified=False)
