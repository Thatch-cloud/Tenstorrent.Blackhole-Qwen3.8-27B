"""Validate the composed DRAM MLP simulator evidence before hardware use."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tensix_stream_gate import matrix
from tiny_mlp_gate import (PACKER, ORIGINAL_PACKER, SIMULATOR_PACKER,
    TP_COMMON, HARDWARE_TP_COMMON, SIMULATOR_TP_COMMON)


SOURCES = ["dram-mlp-probe.py","dram_mlp.py","dram_sharded_projection.py","dram_projection_reload.py","attention_batch.py","gdn_multitoken_conv.py","tensix_projection_raw.py","tensix_projection_raw.cpp","tiny_tile_matmul.py"]
NATIVE_SOURCES = ["models/demos/blackhole/qwen36/tt/tp_common.py","ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp","ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp","ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp","tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h"]


def variant_sources(sharded):
    if type(sharded) is not bool:
        raise ValueError('Explicit sharded-product selection required')
    replacements = {'dram_mlp.py': 'dram_mlp_sharded.py',
        'dram-mlp-probe.py': 'dram-mlp-sharded-probe.py'} if sharded else {}
    return tuple(replacements.get(name, name) for name in SOURCES)


def qualify(report, sources, native_sources, exit_status, *, hardware=False, sharded=False):
    if (exit_status != '0' or not isinstance(report, dict) or report.get('stage') != 'complete'
            or report.get('backend') != 'simulator' or report.get('error') or report.get('cleanup_error')
            or any(report.get(field) is not True for field in ('passed', 'closed_cleanly', 'full_local_mlp'))
            or any(type(report.get(field)) is not int or report[field] != expected
                for field, expected in (('rows', 16), ('compared_rows', 32), ('streams', 1)))):
        raise ValueError('Clean terminal T16 complete local MLP simulator pass and zero wrapper exit required')
    if type(hardware) is not bool:
        raise ValueError('Explicit simulator or hardware source comparison required')
    if set(sources) != set(variant_sources(sharded)) or set(native_sources) != set(NATIVE_SOURCES):
        raise ValueError('Complete current experiment and native source sets required')
    for values in (sources, native_sources):
        if any(not isinstance(value, str) or len(value) != 64
               or any(character not in '0123456789abcdef' for character in value) for value in values.values()):
            raise ValueError('SHA256 fingerprints required')
    if report.get('sources') != sources or report.get('sources_after') != sources:
        raise ValueError('Experiment sources differ from the completed simulation')
    expected_native = dict(native_sources)
    if hardware:
        if expected_native[PACKER] != ORIGINAL_PACKER:
            raise ValueError('Original hardware packer required')
        expected_native[PACKER] = SIMULATOR_PACKER
        if expected_native[TP_COMMON] == HARDWARE_TP_COMMON:
            expected_native[TP_COMMON] = SIMULATOR_TP_COMMON
    if report.get('native_sources') != expected_native or report.get('native_sources_after') != expected_native:
        raise ValueError('Native sources differ beyond the audited packer and prefill-only compatibility differences')
    matrix(report.get('eager_checks'), ('pattern', 'chip'),
        {(pattern, chip) for pattern in range(2) for chip in range(2)}, ('exact', 'input_unchanged'))
    matrix(report.get('replay_checks'), ('repetition', 'pattern', 'chip'),
        {(repeat, pattern, chip) for repeat, pattern in enumerate((0, 1, 0)) for chip in range(2)},
        ('exact', 'input_unchanged', 'output_poison_replaced'))
    return dict(passed=True, scope='T16 complete local MLP only; hardware collective and timing still required',
        eager_checks=4, replay_checks=6)


def qualify_hardware(report):
    if (report.get('closed_cleanly') is not True or report.get('backend') != 'hardware'
            or report.get('error') or report.get('sources_after') != report.get('sources')
            or report.get('native_sources_after') != report.get('native_sources')):
        raise ValueError('Clean unchanged-source hardware completion required')
    if (report.get('passed') is not True or report.get('stage') != 'complete'
            or (report.get('rows'), report.get('streams'), report.get('layer'), report.get('collective_links')) != (16, 1, 0, 4)
            or report.get('seeds') != [1659, 2670, 3781] or report.get('repeats_per_sample') != 50):
        raise ValueError('Completed real-weight T16 hardware comparison required')
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
    report = json.loads(options.hardware_result.read_text())
    sharded = report.get('sharded_product', False)
    names = variant_sources(sharded)
    basename = 'dram-mlp-sharded-simulator' if sharded else 'dram-mlp-simulator'
    simulator_path = root / (basename + '.json')
    simulator = json.loads(simulator_path.read_text())
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    if (report.get('sources') != sources
            or report.get('simulator_report_sha256') != hashlib.sha256(simulator_path.read_bytes()).hexdigest()
            or report.get('hardware_script_sha256') != hashlib.sha256((root / 'dram-mlp-hardware.py').read_bytes()).hexdigest()):
        raise ValueError('Matching hardware harness and simulator artifact required')
    qualify(simulator, sources, report['native_sources'],
        (root / (basename + '.exit-status')).read_text().strip(), hardware=True, sharded=sharded)
    print(json.dumps(qualify_hardware(report)))


if __name__ == '__main__':
    main()
