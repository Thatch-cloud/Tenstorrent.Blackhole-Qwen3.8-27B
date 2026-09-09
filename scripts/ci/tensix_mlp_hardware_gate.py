"""Independent complete-MLP hardware ABBA gate; component latency is not model token throughput."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from sampling_link_policy import SOURCES as FABRIC_SOURCES
from tensix_mlp_gate import NATIVE_SOURCES, ORIGINAL_PACKER, PACKER, SOURCES, hashes, qualify
from tensix_stream_gate import matrix
from tensix_mlp_view_gate import prerequisite as view_prerequisite
from tensix_mlp_weight_views import qualify_views


HARDWARE_SOURCES = ('tensix-stream-mlp-hardware.py', 'tensix_mlp_hardware_gate.py',
    'tensix_mlp_collective.py', 'sampling_link_policy.py', 'tensix_mlp_weight_views.py', 'tensix_mlp_view_gate.py')
MODEL_SOURCES = {
    'models/demos/blackhole/qwen36/tt/mlp.py': 'b9c8193ee4e9b0151646a58573641e3343b1ae9f51240c472cea78cd857a2257',
    'models/tt_transformers/tt/ccl.py': 'b901f03f960eeb552b1a3dca30e63f66abb4530144efe4a45cdb5edb55bc5a9d',
}


def read_prerequisite(report_path, status_path):
    if status_path.read_text().strip() != '0':
        raise ValueError('Simulator JSON is insufficient without a successful outer wrapper exit')
    return json.loads(report_path.read_text())


def qualify_hardware(report):
    if (not isinstance(report, dict) or report.get('stage') != 'complete' or report.get('backend') != 'hardware'
            or report.get('error') or any(report.get(field) is not True for field in
                ('passed', 'closed_cleanly', 'dram_boundary', 'native_collective', 'all_samples_retained'))
            or any(type(report.get(field)) is not int or report[field] != value for field, value in
                (('rows', 8), ('streams', 1), ('layer', 0), ('collective_links', 4),
                    ('pool_buffers', 2), ('repeats_per_sample', 50)))
            or report.get('seeds') != [1659, 2670, 3781]):
        raise ValueError('Clean real-weight T8 complete MLP comparison including copy and four-link CCL required')
    native_links = report.get('native_requested_links')
    if (not isinstance(native_links, dict) or set(native_links) != {'default', 'axis0', 'axis1'}
            or any(type(value) is not int or value not in (1, 2, 4) for value in native_links.values())
            or report.get('matched_requested_links') != dict(default=4, axis0=4, axis1=4)
            or any(type(value) is not int for value in report['matched_requested_links'].values())):
        raise ValueError('Record original link requests and match all control/candidate collective requests at four links')
    qualify_views(report.get('weight_views'))
    if (report.get('native_weights') != {name: check['native'] for name, check in report['weight_views'].items()}
            or report.get('native_weights_after') != report['native_weights']):
        raise ValueError('Native control weights must retain original metadata and buffers throughout the experiment')
    matrix(report.get('eager_checks'), ('pattern', 'chip'),
        {(pattern, chip) for pattern in range(3) for chip in range(2)}, ('exact',))
    matrix(report.get('trace_checks'), ('pattern', 'arm', 'chip'),
        {(pattern, arm, chip) for pattern in range(3) for arm in range(2) for chip in range(2)}, ('exact',))
    matrix(report.get('negative_controls'), ('arm', 'chip'),
        {(arm, chip) for arm in range(2) for chip in range(2)}, ('stale_detected',))
    matrix(report.get('input_checks'), ('pattern', 'tensor', 'chip'),
        {(pattern, tensor, chip) for pattern in range(3) for tensor in range(4) for chip in range(2)},
        ('packed_words_unchanged', 'bindings_stable', 'pool_reused'))
    matrix(report.get('timed_checks'), ('pattern', 'block', 'sample', 'chip'),
        {(pattern, block, sample, chip) for pattern in range(3) for block in range(3)
            for sample in range(4) for chip in range(2)}, ('exact',))
    blocks = report.get('blocks')
    matrix(blocks, ('pattern', 'block'), {(pattern, block) for pattern in range(3) for block in range(3)}, ())
    def matching(actual, expected):
        return (type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
    for block in blocks:
        samples = block.get('samples_ms')
        if (not isinstance(samples, list) or len(samples) != 4 or block.get('order') != ['control', 'candidate', 'candidate', 'control']
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples)):
            raise ValueError('Every ordered ABBA sample must be retained and positive finite')
        control, candidate = statistics.mean((samples[0], samples[3])), statistics.mean(samples[1:3])
        if not all(matching(block.get(name), expected) for name, expected in
                (('control_ms', control), ('candidate_ms', candidate), ('ratio', control / candidate))):
            raise ValueError('ABBA summaries must match original samples')
    for name in ('control_ms', 'candidate_ms'):
        if not matching(report.get(name), statistics.mean(block[name] for block in blocks)):
            raise ValueError('Aggregate must retain all nine timing blocks')
    eligible = all(block['ratio'] > 1.02 for block in blocks)
    if report.get('eligible_for_full_model_gate') is not eligible:
        raise ValueError('Full-model eligibility requires a greater-than-two-percent gain in every block')
    return dict(passed=True, control_ms=report['control_ms'], candidate_ms=report['candidate_ms'],
        eligible_for_full_model_gate=eligible, scope='One real-weight MLP with TP2 CCL; not PP, CTX or TG')


def validate_evidence(report, root):
    simulator_path = root / 'tensix-mlp-simulator.json'
    status_path = root / 'tensix-mlp-simulator.exit-status'
    simulator = read_prerequisite(simulator_path, status_path)
    sources = hashes(root, SOURCES)
    if (report.get('sources') != sources or report.get('hardware_sources') != hashes(root, HARDWARE_SOURCES)
            or report.get('model_sources') != MODEL_SOURCES or report.get('fabric_sources') != FABRIC_SOURCES
            or report.get('simulator_report_sha256') != hashlib.sha256(simulator_path.read_bytes()).hexdigest()
            or report.get('simulator_exit_sha256') != hashlib.sha256(status_path.read_bytes()).hexdigest()
            or report.get('native_sources', {}).get(PACKER) != ORIGINAL_PACKER):
        raise ValueError('Hardware evidence must match current code, exact simulator result, native model and original packer')
    gate = qualify(simulator, sources, report.get('native_sources'))
    if gate != report.get('simulator_gate'):
        raise ValueError('Recorded prerequisite differs from independent simulator qualification')
    view_evidence = report.get('view_prerequisite')
    if (not isinstance(view_evidence, dict)
            or view_evidence != view_prerequisite(root, view_evidence.get('native_sources'))):
        raise ValueError('Exact simulator-qualified metadata views required alongside unchanged MLP kernel evidence')
    return qualify_hardware(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-result', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(validate_evidence(json.loads(options.hardware_result.read_text()), Path(__file__).parent)))


if __name__ == '__main__':
    main()
