"""Require the entire synthetic cache and T32 native replay matrix, not throughput."""

import argparse
import hashlib
import json
from pathlib import Path

from dflash_t16_native_attention_gate import native_hashes
from dflash_combined_sim_runtime import binary_hashes


def qualify(report, sources, native, binaries):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('block_rows') != 32 or report.get('learned_qualified') is not False
            or report.get('hardware_qualified') is not False or report.get('performance_qualified') is not False
            or report.get('sources') != sources or report.get('native_sources') != native
            or report.get('native_sources_after') != native or report.get('runtime_binaries') != binaries
            or report.get('runtime_binaries_after') != binaries):
        raise ValueError('Clean unchanged simulator cache replay evidence required')
    expected = [dict(stage='initial', chip=chip, position=2048, exact=True, finite=True, rows=32) for chip in range(2)]
    position = 2048
    for prefix in (1, 16, 32):
        expected.append(dict(stage='pending-rejected', prefix=prefix, exact=True))
        for stage, frontier in ((f'discard-{prefix}', position), (f'commit-{prefix}', position + prefix)):
            expected.extend(dict(stage=stage, chip=chip, position=frontier, exact=True, finite=True, rows=32) for chip in range(2))
        position += prefix
    if report.get('checks') != expected:
        raise ValueError('Complete ordered two-chip commit/discard and replay matrix required')
    return dict(cache_boundary_qualified=True, learned_qualified=False, hardware_qualified=False, performance_qualified=False)


def main():
    import importlib.util
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    options = parser.parse_args()
    directory = Path(__file__).parent
    spec = importlib.util.spec_from_file_location('probe', directory / 'dflash-t32-cache-replay-probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    root = os.environ['TT_METAL_HOME']
    sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in probe.SOURCES}
    print(json.dumps(qualify(json.loads(options.report.read_text()), sources,
        native_hashes(root, simulator=True), binary_hashes(root))))


if __name__ == '__main__':
    main()
