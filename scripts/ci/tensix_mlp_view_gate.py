"""Independent source-bound simulator gate for native rank expansion without compute or repacking."""

import hashlib
import json

from tensix_mlp_gate import ORIGINAL_PACKER, PACKER, hashes
from tensix_mlp_weight_views import WEIGHTS, qualify_views


SOURCES = ('tensix_mlp_weight_views.py', 'tensix-mlp-view-probe.py', 'tensix_mlp_view_gate.py')
NATIVE_SOURCES = {
    'ttnn/cpp/ttnn/operations/experimental/reshape/view.cpp': '849995c8d16235e176266cf678ad618ea279bc5edd62d418ce456900ca252eee',
    'ttnn/cpp/ttnn/operations/experimental/reshape/view_nanobind.cpp': '7b162e832a0370805556e78f982408c47bbbba88a15d866951ba059387f33c14',
    'ttnn/core/tensor/tensor_ops.cpp': '8fc326c22555a2a67909ec80d20fa583c7938f97ea1951a81f73d6de1679b39b',
    'ttnn/api/ttnn/tensor/tensor_ops.hpp': 'fadab5265114bcd0e86baec0fc38f979469ea42e4979d400da770934be2daf25',
    PACKER: ORIGINAL_PACKER,
}


def qualify(report, sources, native_sources):
    if (not isinstance(report, dict) or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('error') or any(report.get(name) is not True for name in ('passed', 'closed_cleanly'))
            or report.get('sources') != sources or set(sources) != set(SOURCES)
            or report.get('native_sources') != NATIVE_SOURCES or native_sources != NATIVE_SOURCES):
        raise ValueError('Clean source-matched full-size metadata-view simulation with original native runtime required')
    qualify_views(report.get('weight_views'))
    for name, (inner, width, unused_dtype) in WEIGHTS.items():
        if report['weight_views'][name]['native']['shape'] != [inner, width]:
            raise ValueError('Simulator must exercise the actual native 2D loader layout')
    if report.get('shard_axes') != dict(gate=1, up=1, down=0):
        raise ValueError('Native gate/up column and down row sharding required')
    checks = report.get('checks')
    if (not isinstance(checks, list) or len(checks) != 6
            or any(not isinstance(check, dict) or type(check.get('chip')) is not int
                or not isinstance(check.get('weight'), str) or any(check.get(flag) is not True for flag in
                    ('contents_exact', 'native_unchanged_after_view_release', 'canonical_identity', 'distinct_shards'))
                for check in checks)
            or {(check['weight'], check['chip']) for check in checks}
                != {(name, chip) for name in WEIGHTS for chip in range(2)}):
        raise ValueError('Exact contents, original lifetime and distinct shards must be checked for all six chip weights')
    counts = report.get('program_cache')
    if (not isinstance(counts, list) or len(counts) != 2
            or any(type(count) is not int or count < 0 for count in counts) or counts[0] != counts[1]):
        raise ValueError('Metadata aliasing must not add a device program')
    return dict(passed=True, weights=3, chip_checks=6, added_programs=0)


def prerequisite(root, native_sources):
    report_path, status_path = root / 'tensix-mlp-view-simulator.json', root / 'tensix-mlp-view-simulator.exit-status'
    if status_path.read_text().strip() != '0':
        raise ValueError('Metadata-view simulator wrapper exit must succeed')
    result = qualify(json.loads(report_path.read_text()), hashes(root, SOURCES), native_sources)
    return dict(gate=result, report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        exit_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest(), native_sources=native_sources)
