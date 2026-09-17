"""Source-bound component admission; not target-quality or hardware qualification."""

import hashlib
import json
from pathlib import Path

from dspark_projection_precision_report import SOURCE_NAMES, validate_set
from dspark_projection_precision_stage import PROJECTIONS


RUNTIME = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
WEIGHTS = dict(zip(PROJECTIONS, (
    '1f5b77ee27b80ddec91a8599d27868d8b4ed7052b202be35db013f989fcb9be5',
    '8797b20921e603dc2aa25fbfdef10a9ed1b00c17857bd22b381744cf247034d3',
    '9e98811de3c111aa93a4d2477e15ca2ac6d52609d062956f269255e3fa05b043',
    '620b4bba6611c6c7944aa9bea8d6748004647ffdbef44075b9080deefdfbcd33',
    'aa832838a0dbd01249295bb60b651c4d119dac910aa2d788c997ff10ed915ee2',
    'cf0f815c5bc3e1ca64e3a16918d1aebe586a7b6d540ceda887b42b7c642d70b5',
    'd9eee90bff8d04e7bcebcf0b9b9fe580d1628680b48025c898eab6e6318cb746',
), strict=True))
QUERY_REPORT = '361ace678704220f0ed138d2aa6878924187a92ef59d9b2dd4b0a6fd0d1e1e6c'
QUERY_MANIFEST = 'b7f563038887922825f77e45920d63cc07da5c45fdbd886f8f9745fec2acf934'


def candidate_manifest(folder, projection, report_sha256):
    current = folder / 'precision-candidate.json'
    if current.exists():
        return json.loads(current.read_bytes())
    if projection != 'self_attn.q_proj.weight' or report_sha256 != QUERY_REPORT:
        raise ValueError('Only the reviewed initial query screen has a legacy manifest')
    raw = (folder / 'weight-pipeline-candidate.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != QUERY_MANIFEST:
        raise ValueError('Legacy query manifest differs from reviewed artifact')
    candidate = json.loads(raw)
    if 'projection' in candidate:
        raise ValueError('Unexpected legacy query manifest schema')
    return dict(candidate, projection=projection)


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
        candidate = candidate_manifest(folder, projection, reviewed_reports[projection])
        if (candidate.get('projection') != projection or candidate.get('component_execution_only') is not True
                or any(candidate.get('sources', {}).get(name) != sources[name] for name in (
                    'dspark-projection-precision-probe.py', 'dspark_projection_precision.py'))):
            raise ValueError('Staged precision policy differs from reviewed simulator candidate')
        reports.append(json.loads(raw))
    result = validate_set(reports, expected_sources=sources, expected_weights=WEIGHTS)
    return dict(result, report_sha256=dict(reviewed_reports), runtime=RUNTIME,
        hardware_qualified=False, performance_qualified=False)
