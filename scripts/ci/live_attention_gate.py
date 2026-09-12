"""Source-pinned simulator qualification for the opt-in learned attention integration."""

import argparse
import hashlib
import json
from pathlib import Path

from live_qk_gate import SOURCES as QK_SOURCES, native_hashes, require_matrix


SOURCES = (*QK_SOURCES, 'draft_live_attention.py', 'draft_attention_branch.py', 'dflash_device.py',
    'dflash_proposal_inputs.py', 'dflash_proposal_trace.py', 'draft-live-attention-probe.py', 'live_attention_gate.py')
FIXTURE_SHA256 = '0974cf572f0db9291f56b4ee322829d60f61484e2521b4b69c8393035b541e49'


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCES}


def qualify_integration(report, context, sources, native):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or type(report.get('context')) is not int
            or report['context'] != context or context not in (31, 2048)
            or report.get('sources') != sources or report.get('native_sources') != native
            or report.get('fixture_sha256') != (FIXTURE_SHA256 if context == 31 else None)):
        raise ValueError('Complete source-pinned learned/synthetic integration simulation required')
    require_matrix(report.get('eager_checks'), ('pattern', 'chip'),
        {(pattern, chip) for pattern in range(2) for chip in range(2)}, lambda row: ('exact_all_rows',))
    require_matrix(report.get('replay_checks'), ('repetition', 'pattern', 'arm', 'chip'),
        {(repetition, pattern, arm, chip) for repetition, pattern in enumerate((0, 1, 0))
            for arm in range(2) for chip in range(2)}, lambda row: ('exact_all_rows', 'inputs_unchanged', 'bindings_stable'))
    require_matrix(report.get('negative_controls'), ('arm', 'chip'),
        {(arm, chip) for arm in range(2) for chip in range(2)}, lambda row: ('stale_detected',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(31, 2048), required=True)
    parser.add_argument('--metal-root', type=Path, required=True)
    options = parser.parse_args()
    qualify_integration(json.loads(options.report.read_text()), options.context,
        source_hashes(), native_hashes(options.metal_root))
    print(json.dumps(dict(passed=True, context=options.context, scope='Simulator integration, not throughput')))
