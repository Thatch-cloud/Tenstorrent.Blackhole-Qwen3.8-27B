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

# Per-context evidence-report pins. 32768 is REPORTS above (the retained, qualified
# candidate - never touched by widening this table). 65536 stages the same shape via
# frozen_recipe_context.py --context 65536, but has no qualified draft-numerical,
# draft-diagnostics or target-replay evidence yet: None here is an explicit
# "unqualified" placeholder, not a missing entry, and qualify() refuses it with a
# message naming the three runs still needed. Fill it in with the real three SHA256s
# only once that evidence exists (see docs/t16-recipe-rung-65k.md section 2 for the
# qwen-frozen-32k-numerical.yml / qwen-frozen-recover.yml / qwen-frozen-combined.yml
# tag lanes that produce them - they need new 65536 tags/run-IDs first, since today
# they are hardcoded to 32768).
CONTEXT_REPORTS = {
    32768: REPORTS,
    65536: None,
}


def load_reports(directory, reports=None):
    if reports is None:
        reports = REPORTS
    result = {}
    for name, expected in reports.items():
        payload = (Path(directory) / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Pinned component report required: ' + name)
        result[name] = json.loads(payload)
    return result


def qualify(directory, *, draft_sources, target_sources, context):
    if type(context) is not int or context not in CONTEXT_REPORTS:
        raise ValueError('Only 32768 has both retained candidate qualifications')
    reports_pins = CONTEXT_REPORTS[context]
    if reports_pins is None:
        raise ValueError(
            f'{context} has no qualified combined-runtime evidence yet: run the draft-numerical.json '
            'and draft-diagnostics.json simulator qualification (qwen-frozen-32k-numerical.yml / '
            'qwen-frozen-recover.yml lanes) and the target-replay.json hardware qualification '
            '(qwen-frozen-32k-numerical.yml --target-replay lane, staged through qwen-frozen-combined.yml) '
            f'at context {context}, then record the three report SHA256 hashes in '
            f'frozen_combined_gate.CONTEXT_REPORTS[{context}] before this gate can qualify it')
    reports = load_reports(directory, reports_pins)
    draft = validate_pair(reports['draft-numerical.json'], reports['draft-diagnostics.json'], context)
    if draft['reciprocal_variant'] != 'scalar-fp32':
        raise ValueError('Explicit qualified draft reciprocal required')
    for name, expected in reports['draft-numerical.json']['sources'].items():
        if hashlib.sha256((Path(draft_sources) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Qualified draft source changed: ' + name)
    target = validate_target_report(reports['target-replay.json'], target_sources, context,
        compact_scratch=True)
    return dict(context=context, capacity=context + 256, reports=dict(reports_pins),
        draft=draft, target=target, component_sources_verified=True,
        combined_runtime_qualified=False, performance_qualified=False)
