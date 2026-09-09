"""Fail-closed source and comparison manifest for the small-tile MLP gate."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCES = (
    'tiny_mlp.py', 'tiny-mlp-probe.py', 'tiny_tile_dma.py', 'tiny_tile_dma.cpp', 'tiny_tile_matmul.py',
    'attention_batch.py', 'tiny_tile_product.py', 'tiny_tile_product_compute.cpp', 'tiny_tile_product_io.cpp')
PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
ORIGINAL_PACKER = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
SIMULATOR_PACKER = '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7'
TP_COMMON = 'models/demos/blackhole/qwen36/tt/tp_common.py'
SIMULATOR_TP_COMMON = 'bb43f0cde336c3f84725d47a64ed2b506b5287bdd0e910cd24b13feed0a0826a'
HARDWARE_TP_COMMON = '5419361f26071b388fd58768f003b11c704b40d508524b968aab78362843aa66'
UNCHANGED_TP_COMMON_AST = '1d3374593f1caf77b453ef22f442779f9bd4da15b4b78d86d7a2178ed5216762'
NATIVE_SOURCES = (
    TP_COMMON, 'models/demos/blackhole/qwen36/tt/mlp.py', PACKER,
    'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp',
    'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp',
    'tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h')


def qualify(report, sources, native_sources):
    if (report.get('passed') is not True or report.get('stage') != 'complete' or report.get('rows') != 8
            or report.get('packer_zero_graft') is not True or report.get('packer_l1_acc') is not True):
        raise ValueError('Completed T8 simulator gate with audited packer graft required')
    if set(sources) != set(SOURCES) or report.get('sources') != sources:
        raise ValueError('Simulator and current small-tile sources must match')
    if set(native_sources) != set(NATIVE_SOURCES) or native_sources[PACKER] != ORIGINAL_PACKER:
        raise ValueError('Pinned native sources and original hardware packer required')
    expected_native = dict(native_sources, **{PACKER: SIMULATOR_PACKER})
    recorded = report.get('native_sources', {})
    prefill_only_difference = (recorded.get(TP_COMMON) == SIMULATOR_TP_COMMON
        and native_sources[TP_COMMON] == HARDWARE_TP_COMMON)
    if prefill_only_difference:
        expected_native[TP_COMMON] = SIMULATOR_TP_COMMON
    if report.get('native_sources') != expected_native:
        changed = sorted(name for name in set(recorded) | set(expected_native)
            if recorded.get(name) != expected_native.get(name))
        raise ValueError(f'Native configuration or operation changed since simulation: {changed}')
    components = ('gate', 'up', 'hidden', 'partial')
    eager = report.get('eager_checks', [])
    trace = report.get('trace_checks', [])
    negative = report.get('negative_controls', [])
    if (len(eager) != 16 or any(check.get('exact') is not True for check in eager)
            or {(check.get('pattern'), check.get('chip'), check.get('component')) for check in eager}
            != {(pattern, chip, component) for pattern in range(2) for chip in range(2) for component in components}):
        raise ValueError('Complete exact eager comparison matrix required')
    if (len(trace) != 32 or any(check.get('exact') is not True for check in trace)
            or {(check.get('pattern'), check.get('arm'), check.get('chip'), check.get('component')) for check in trace}
            != {(pattern, arm, chip, component) for pattern in range(2) for arm in range(2)
                for chip in range(2) for component in components}):
        raise ValueError('Complete changed-input trace comparison matrix required')
    if (len(negative) != 2 or {check.get('chip') for check in negative} != {0, 1}
            or any(check.get('stale_input_detected') is not True for check in negative)):
        raise ValueError('Both stale-input negative controls required')
    result = dict(passed=True, eager_checks=len(eager), trace_checks=len(trace), negative_controls=len(negative))
    if prefill_only_difference:
        result['native_equivalence'] = dict(scope='T8 MLP decode only; prefill placement functions differ',
            simulator_tp_common=SIMULATOR_TP_COMMON, hardware_tp_common=HARDWARE_TP_COMMON,
            unchanged_module_ast_sha256=UNCHANGED_TP_COMMON_AST)
    return result


def qualify_hardware(report):
    if (report.get('passed') is not True or report.get('stage') != 'complete'
            or (report.get('rows'), report.get('streams'), report.get('layer'), report.get('collective_links')) != (8, 1, 0, 4)
            or report.get('seeds') != [1659, 2670, 3781] or report.get('repeats_per_sample') != 50):
        raise ValueError('Completed real-weight T8 hardware comparison required')
    eager, trace, blocks = (report.get(name, []) for name in ('eager_checks', 'trace_checks', 'blocks'))
    if (len(eager) != 6 or any(check.get('exact') is not True for check in eager)
            or {(check.get('pattern'), check.get('chip')) for check in eager}
            != {(pattern, chip) for pattern in range(3) for chip in range(2)}):
        raise ValueError('Complete native-reference eager comparisons required')
    if (len(trace) != 12 or any(check.get('exact') is not True for check in trace)
            or {(check.get('pattern'), check.get('arm'), check.get('chip')) for check in trace}
            != {(pattern, arm, chip) for pattern in range(3) for arm in range(2) for chip in range(2)}):
        raise ValueError('Complete native-reference changed-input traces required')
    if (len(blocks) != 9 or {(block.get('pattern'), block.get('block')) for block in blocks}
            != {(pattern, block) for pattern in range(3) for block in range(3)}):
        raise ValueError('All nine matched ABBA blocks required')
    def matching(actual, expected):
        return (type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
    for block in blocks:
        samples = block.get('samples_ms', [])
        if len(samples) != 4 or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError('Four positive finite ABBA samples required')
        baseline, candidate = statistics.mean((samples[0], samples[3])), statistics.mean(samples[1:3])
        if not all(matching(block.get(name), expected) for name, expected in (
                ('control_ms', baseline), ('candidate_ms', candidate), ('ratio', baseline / candidate))):
            raise ValueError('ABBA block summaries must match raw samples')
    for name in ('control_ms', 'candidate_ms'):
        if not matching(report.get(name), statistics.mean(block[name] for block in blocks)):
            raise ValueError('Hardware summary must match complete ABBA samples')
    eligible = all(block['ratio'] > 1.02 for block in blocks)
    if report.get('eligible_for_full_model_gate') is not eligible:
        raise ValueError('Full-model eligibility must match every timing block')
    return dict(passed=True, control_ms=report['control_ms'], candidate_ms=report['candidate_ms'],
        eligible_for_full_model_gate=eligible, scope='Single-layer hardware gate, not PP/CTX/TG')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware-result', type=Path, required=True)
    options = parser.parse_args()
    root = Path(__file__).parent
    simulator_path = root / 'tiny-mlp-simulator.json'
    simulator = json.loads(simulator_path.read_text())
    report = json.loads(options.hardware_result.read_text())
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    if (report.get('sources') != sources
            or report.get('simulator_report_sha256') != hashlib.sha256(simulator_path.read_bytes()).hexdigest()
            or report.get('hardware_script_sha256') != hashlib.sha256((root / 'tiny-mlp-hardware.py').read_bytes()).hexdigest()):
        raise ValueError('Hardware evidence must match current code and simulator report')
    gate = qualify(simulator, sources, report.get('native_sources', {}))
    if report.get('simulator_gate') != gate:
        raise ValueError('Recorded simulator prerequisite must match independently validated gate')
    print(json.dumps(qualify_hardware(report)))


if __name__ == '__main__':
    main()
