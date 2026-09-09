"""Source-bound mask/replay gate, not draft accuracy, target correctness or speed certification."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from live_qk_gate import require_matrix
from proposal_native_attention import POLICY


SOURCES = ('proposal_native_attention.py', 'proposal-native-attention-probe.py',
    'proposal_native_attention_gate.py', 'draft_attention.py', 'draft_live_qk.py',
    'attention_batch.py', 'gdn_multitoken_conv.py', 'feature_projection.py', 'live_qk_gate.py')
SDPA = 'ttnn/cpp/ttnn/operations/transformer/sdpa/'
PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
NATIVE_SOURCES = (PACKER, *(SDPA + name for name in ('sdpa.cpp', 'sdpa.hpp', 'sdpa_nanobind.cpp',
    'device/sdpa_device_operation.cpp', 'device/sdpa_device_operation.hpp',
    'device/sdpa_device_operation_types.hpp', 'device/sdpa_program_factory.cpp',
    'device/sdpa_interleaved_cb_ids.hpp', 'device/sdpa_subblock_utils.hpp',
    'device/kernels/compute/compute_common.hpp', 'device/kernels/compute/compute_streaming.hpp',
    'device/kernels/compute/sdpa.cpp', 'device/kernels/dataflow/reader_interleaved.cpp',
    'device/kernels/dataflow/writer_interleaved.cpp')))
ORIGINAL = {
    PACKER: '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181',
    SDPA + 'device/kernels/compute/compute_common.hpp': '3fb5da2440c3bf90ebceb8acd55424c7739339de4c6b02db83836e1e3414fa19',
    SDPA + 'device/kernels/compute/sdpa.cpp': 'a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0',
}
FIXTURE_SHA256 = '0974cf572f0db9291f56b4ee322829d60f61484e2521b4b69c8393035b541e49'


def hashes(root, names):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in names}


def native_hashes(root):
    result = hashes(root, NATIVE_SOURCES)
    if any(result[name] != digest for name, digest in ORIGINAL.items()):
        raise ValueError('Original native SDPA and packer required; no precision or simulator graft')
    if (Path(root) / SDPA / 'device/kernels/compute/.qwen-precise-draft.lock').exists():
        raise ValueError('An active precise-draft graft owner excludes this experiment')
    return result


def qualify(report, context, sources, native):
    if (not isinstance(report, dict) or report.get('error') or report.get('passed') is not True
            or report.get('closed_cleanly') is not True or report.get('backend') != 'simulator'
            or report.get('stage') != 'complete'
            or report.get('policy') != POLICY or type(context) is not int or context not in (31, 2048)
            or type(report.get('context')) is not int or report['context'] != context
            or report.get('target_integrated') is not False or report.get('accuracy_qualified') is not False
            or not isinstance(sources, dict) or not isinstance(native, dict)
            or report.get('sources') != sources or set(sources) != set(SOURCES)
            or report.get('native_sources') != native or report.get('native_sources_after') != native
            or set(native) != set(NATIVE_SOURCES) or any(native[name] != digest for name, digest in ORIGINAL.items())
            or report.get('fixture_sha256') != (FIXTURE_SHA256 if context == 31 else None)):
        raise ValueError('Clean source-bound proposal-only simulation with unchanged native runtime required')
    for digest in (*sources.values(), *native.values()):
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest):
            raise ValueError('Complete SHA256 source fingerprints required')
    require_matrix(report.get('eager_checks'), ('pattern', 'chip'),
        {(pattern, chip) for pattern in range(3) for chip in range(2)}, lambda row: ('finite_all_rows',))
    for check in report['eager_checks']:
        metrics = check.get('numerical_difference')
        if (not isinstance(metrics, dict) or set(metrics) != {'max_abs', 'mean_abs', 'rms', 'reference_max_abs', 'legacy_close'}
                or type(metrics['legacy_close']) is not bool or any(type(metrics[name]) not in (int, float)
                    or not math.isfinite(metrics[name]) or metrics[name] < 0
                    for name in ('max_abs', 'mean_abs', 'rms', 'reference_max_abs'))
                or not metrics['mean_abs'] <= metrics['rms'] + 1e-6 <= metrics['max_abs'] + 2e-6):
            raise ValueError('Record finite numerical differences; the legacy accuracy outcome is not relaxed')
    require_matrix(report.get('replay_checks'), ('repetition', 'pattern', 'chip'),
        {(repetition, pattern, chip) for repetition, pattern in enumerate((0, 1, 2, 0)) for chip in range(2)},
        lambda row: ('exact_eager_all_rows', 'bindings_stable'))
    require_matrix(report.get('input_checks'), ('phase', 'pattern', 'tensor', 'chip'),
        {(phase, pattern, tensor, chip) for phase, patterns in ((0, range(3)), (1, range(4)))
            for pattern in patterns for tensor in range(4) for chip in range(2)}, lambda row: ('unchanged',))
    require_matrix(report.get('negative_controls'), ('chip',), {(0,), (1,)}, lambda row: ('stale_detected',))
    require_matrix(report.get('masked_input_checks'), ('phase', 'chip'),
        {(phase, chip) for phase in range(2) for chip in range(2)}, lambda row: ('masked_changes_ignored',))
    return dict(passed=True, context=context, policy=POLICY, accuracy_qualified=False, target_integrated=False,
        scope='Finite native proposal attention with exact mask isolation and replay; not coding quality or TG')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--exit-status', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(31, 2048), required=True)
    options = parser.parse_args()
    if options.exit_status.read_text().strip() != '0':
        raise ValueError('Successful outer simulator wrapper exit required')
    print(json.dumps(qualify(json.loads(options.report.read_text()), options.context,
        hashes(Path(__file__).parent, SOURCES), native_hashes(options.native_root))))


if __name__ == '__main__':
    main()
