"""Source-pinned exact simulator admission for the fused K/V publication writer."""

import hashlib
import json
from pathlib import Path

from draft_kv_slide_report import validate


REPORT_SHA256 = '7eb9560c661f77bc422d9551d513bf0e264f88eb2a0c7ab0ac0ce6a7a8099a38'


def qualify(directory, evidence):
    evidence = Path(evidence)
    raw = (evidence / 'history-append.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Pinned complete sliding K/V replay report required')
    if (evidence / 'history-append.exit-status').read_text().strip() != '0':
        raise ValueError('Successful simulator process required')
    report = json.loads(raw)
    result = validate(report, directory, staged_probe=False)
    return dict(result, report_sha256=REPORT_SHA256, sources=report['sources'])


if __name__ == '__main__':
    directory = Path(__file__).parent
    print(json.dumps(qualify(directory, directory / 'draft-kv-slide-evidence')))
