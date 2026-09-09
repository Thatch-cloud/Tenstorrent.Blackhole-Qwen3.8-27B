"""Require qualified attention integration and a complete isolated 4K request ABBA."""

import argparse
import hashlib
import json
from pathlib import Path

from live_attention_gate import source_hashes, native_hashes, qualify_integration


def simulator_digests():
    return {str(context): hashlib.sha256(Path(__file__).with_name(
        f'live-attention-simulator-{context}.json').read_bytes()).hexdigest() for context in (31, 2048)}


def qualify_inputs(metal_root):
    sources, native = source_hashes(), native_hashes(metal_root)
    for context in (31, 2048):
        report = json.loads(Path(__file__).with_name(f'live-attention-simulator-{context}.json').read_text())
        qualify_integration(report, context, sources, native)
    return simulator_digests()


def qualify_request(report):
    from dflash_benchmark_report import report_rows
    from full_dflash_request import summarize_dflash_live_query_requests

    if report.get('passed') is not True or report.get('context_lengths') != [4096]:
        raise ValueError('Passed complete 4K live-query hardware request required')
    if report.get('live_attention_simulator_reports') != simulator_digests():
        raise ValueError('Hardware must retain both source-pinned integration report identities')
    expected = source_hashes()
    expected['full_dflash_request.py'] = hashlib.sha256(Path(__file__).with_name('full_dflash_request.py').read_bytes()).hexdigest()
    requests = report.get('request_checks', [])
    if any(any(entry.get('sources', {}).get(name) != digest for name, digest in expected.items()) for entry in requests):
        raise ValueError('Hardware request integration sources changed')
    summarize_dflash_live_query_requests(requests)
    return report_rows(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--metal-root', type=Path)
    selection.add_argument('--hardware-result', type=Path)
    options = parser.parse_args()
    result = qualify_inputs(options.metal_root) if options.metal_root else qualify_request(
        json.loads(options.hardware_result.read_text()))
    print(json.dumps(dict(passed=True, result=result)))
