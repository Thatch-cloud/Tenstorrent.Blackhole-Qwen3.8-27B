"""Reject incomplete full-proposal simulation; never qualify target throughput."""

import argparse
import hashlib
import json
from pathlib import Path


def validate(report):
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.iterdir()) if path.suffix in ('.py', '.cpp', '.sh')}
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('score_layout') != 'fused'
            or report.get('learned_layers') != 5 or report.get('proposals') != 31
            or report.get('full_request_qualified') is not False
            or report.get('sources') != sources or report.get('sources_after') != sources
            or not report.get('attention') or report.get('attention_after') != report['attention']):
        raise ValueError('Complete source-matched fused proposal and clean runtime required')
    expected = dict(position=4096, tensors=6, exact=True, reference='native-score-eager')
    if report.get('score_reference_checks') != [expected] * 3:
        raise ValueError('Initial and both changed-input native-score comparisons required')
    replay = dict(position=4096, tensors=6, exact=True)
    if report.get('replay_checks') != [replay] * 2:
        raise ValueError('Both changed-input replay comparisons required')
    checks = report.get('checks', [])
    if len(checks) != 2:
        raise ValueError('Two complete token trajectories required')
    for check, anchor in zip(checks, (20, 10), strict=True):
        tokens = check.get('tokens', [])
        if (check.get('anchor') != anchor or check.get('exact') is not True or len(tokens) != 31
                or any(type(token) is not int or not 0 <= token < 248320 for token in tokens)):
            raise ValueError('Full-vocabulary trajectories for both anchors required')
    return dict(passed=True, full_request_qualified=False, performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    options = parser.parse_args()
    if options.report.with_suffix('.exit-status').read_text().strip() != '0':
        raise ValueError('Successful simulator exit required')
    print(json.dumps(validate(json.loads(options.report.read_text()))))


if __name__ == '__main__':
    main()
