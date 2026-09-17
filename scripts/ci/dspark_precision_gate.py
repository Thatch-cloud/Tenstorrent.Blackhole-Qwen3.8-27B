"""Source-bound component admission; not target-quality or hardware qualification."""

import hashlib
import json
from pathlib import Path

from dspark_projection_precision_report import SOURCE_NAMES, validate_set
from dspark_projection_precision_stage import PROJECTIONS


RUNTIME = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
CHECKPOINT = '9e98811de3c111aa93a4d2477e15ca2ac6d52609d062956f269255e3fa05b043'


def qualify(directory, evidence, *, reviewed_reports):
    directory, evidence = Path(directory), Path(evidence)
    if (set(reviewed_reports) != set(PROJECTIONS)
            or any(not isinstance(value, str) or len(value) != 64 or value == '0' * 64
                or any(character not in '0123456789abcdef' for character in value)
                for value in reviewed_reports.values())):
        raise ValueError('All seven independently reviewed report hashes required')
    sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    reports = []
    for projection in PROJECTIONS:
        folder = evidence / projection
        raw = (folder / 'dspark-projection-hifi2.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != reviewed_reports[projection]:
            raise ValueError('Projection report differs from independently reviewed artifact: ' + projection)
        if ((folder / 'dspark-projection-hifi2.exit-status').read_text().strip() != '0'
                or (folder / 'simulator-runtime.txt').read_text().strip() != RUNTIME):
            raise ValueError('Clean exit on pinned simulator runtime required')
        if json.loads((folder / 'container-cleanup.json').read_bytes()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0):
            raise ValueError('Clean simulator teardown required')
        candidate = json.loads((folder / 'precision-candidate.json').read_bytes())
        if (candidate.get('projection') != projection or candidate.get('component_execution_only') is not True
                or any(candidate.get('sources', {}).get(name) != sources[name] for name in (
                    'dspark-projection-precision-probe.py', 'dspark_projection_precision.py'))):
            raise ValueError('Staged precision policy differs from reviewed simulator candidate')
        reports.append(json.loads(raw))
    result = validate_set(reports, expected_sources=sources, checkpoint_sha256=CHECKPOINT)
    return dict(result, report_sha256=dict(reviewed_reports), runtime=RUNTIME,
        hardware_qualified=False, performance_qualified=False)
