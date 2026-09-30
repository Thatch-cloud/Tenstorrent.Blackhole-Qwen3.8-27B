"""Source-pinned exact simulator admission for the fused K/V publication writer."""

import hashlib
import json
from pathlib import Path

from draft_kv_slide_report import validate


REPORT_SHA256 = '7eb9560c661f77bc422d9551d513bf0e264f88eb2a0c7ab0ac0ce6a7a8099a38'
DIRECT_REPORT_SHA256 = '221d4e5fc003f55a690cd583e3a4ea55d5269bec611b47dc21fb2fbfcf087c6e'
QUALIFIED_KERNELS = {
    REPORT_SHA256: 'bc45d47257c844aff4bf17f478b536a48763578e544597884b7d1620083b6ba1',
    DIRECT_REPORT_SHA256: '1679bbd779add56b4bd445a6b4c51bd3e49c39a8a520dfaddfc3bd9f36667d47',
}


def validate_record(admission):
    directory = Path(__file__).parent
    sources = {name: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for name, filename in (('history-append-probe.py', 'draft-kv-slide-probe.py'),
            ('draft_kv_slide.py', 'draft_kv_slide.py'), ('draft_kv_slide.cpp', 'draft_kv_slide.cpp'))}
    if (not isinstance(admission, dict) or admission.get('report_sha256') not in QUALIFIED_KERNELS
            or admission.get('passed') is not True or admission.get('checks') != 120
            or admission.get('sources') != sources
            or sources['draft_kv_slide.cpp'] != QUALIFIED_KERNELS[admission['report_sha256']]):
        raise ValueError('Complete source-bound K/V simulator admission required')


def qualify(directory, evidence):
    evidence = Path(evidence)
    raw = (evidence / 'history-append.json').read_bytes()
    report_sha256 = hashlib.sha256(raw).hexdigest()
    if report_sha256 not in QUALIFIED_KERNELS:
        raise ValueError('Pinned complete sliding K/V replay report required')
    if (evidence / 'history-append.exit-status').read_text().strip() != '0':
        raise ValueError('Successful simulator process required')
    report = json.loads(raw)
    result = validate(report, directory, staged_probe=False)
    if report['sources']['draft_kv_slide.cpp'] != QUALIFIED_KERNELS[report_sha256]:
        raise ValueError('Report must match its qualified kernel')
    return dict(result, report_sha256=report_sha256, sources=report['sources'])


if __name__ == '__main__':
    directory = Path(__file__).parent
    print(json.dumps(qualify(directory, directory / 'draft-kv-slide-evidence')))
