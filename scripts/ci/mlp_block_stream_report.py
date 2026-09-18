"""Independently validate complete transport ABBA, not isolated kernel timing."""

import hashlib
import json
from pathlib import Path
import sys

from dflash_native_comparison_report import summarize


def qualify_report(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True or report.get('error')
            or report.get('streams') != 1 or report.get('ctx_tokens') != 4096
            or report.get('sampler_links') != 4 or not report.get('drafter_comparison_sources')
            or report['drafter_comparison_sources'] != report.get('drafter_comparison_sources_after')):
        raise ValueError('Clean matched four-link single-stream run with unchanged sources required')
    pool = report.get('block_stream_pool', {})
    if (pool.get('released') is not True or pool.get('native_bindings_unchanged') is not True
            or pool.get('allocated_layers') != 64 or pool.get('serving_defaults_changed') is not False
            or len(pool.get('admission', [])) != 2 or pool.get('setup_ms', 0) <= 0):
        raise ValueError('Explicitly admitted complete weight pool and successful release required')
    result = summarize(report.get('request_checks', []), weight_transport=True)
    if result != report.get('block_stream_comparison'):
        raise ValueError('Recomputed complete-request result differs from saved summary')
    return result


if __name__ == '__main__':
    raw = Path(sys.argv[1]).read_bytes()
    print(json.dumps(dict(qualify_report(json.loads(raw)), raw_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
