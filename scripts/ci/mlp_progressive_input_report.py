"""Review progressive-input simulator evidence; never authorize hardware execution."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from mlp_block_stream_gate import REPORT_SHA256 as SERIAL_SHA256, validate_report
from mlp_block_stream_pipeline_gate import validate_weights
from mlp_block_stream_projection import adapt_projection
from mlp_progressive_input import projection, reader
from mlp_register_epilogue import adapt_projection as register_projection


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_record(directory):
    directory = Path(directory)
    original = adapt_projection(register_projection((directory / 'fused_1d.py').read_text(), nearest_away=True))
    input_source = (directory / 'fused_1d_input.cpp').read_text()
    payloads = {'fused_1d.py': projection(original), 'fused_1d_input.cpp': reader(input_source),
        'mlp_progressive_input.py': (directory / 'mlp_progressive_input.py').read_text()}
    return dict(before={'fused_1d.py': digest(original.encode()),
        'fused_1d_input.cpp': digest(input_source.encode())},
        after={name: digest(source.encode()) for name, source in payloads.items()},
        extra_l1_bytes_per_multicast_core=144 * 2048, arithmetic_changed=False,
        simulator_qualified=False, hardware_qualified=False, performance_qualified=False)


def validate_candidate(report, serial, staged, directory):
    validate_report(report)
    validate_weights(report)
    generated = source_record(directory)
    if staged != generated:
        raise ValueError('Progressive source differs from simulator staging')
    expected, = copy.deepcopy(serial['kernels'])
    expected.update(progressive_input=True, input_buffer_tiles=160)
    expected['reader_sha256']['fused_1d_input.cpp'] = generated['after']['fused_1d_input.cpp']
    if (report.get('kernels') != [expected]
            or report.get('buffer_candidate_sha256') != generated['after']['fused_1d.py']
            or report.get('trace_source_sha256') != digest((Path(directory) / 'fusion_trace.py').read_bytes())):
        raise ValueError('Only admitted activation delivery and capacity may change')
    for key in ('packer_header_sha256', 'packer_zero_graft', 'precision', 'control_epilogue',
            'weight_check_backend', 'weight_check_sources'):
        if key not in serial or report.get(key) != serial[key]:
            raise ValueError('Native numerical control changed: ' + key)
    before, after = serial['stream_binding'], report.get('stream_binding', {})
    if {key: value for key, value in before.items() if key != 'addresses'} != {
            key: value for key, value in after.items() if key != 'addresses'}:
        raise ValueError('Serial weight stream or arithmetic binding changed')
    addresses = after.get('addresses', [])
    if (len(addresses) != 2 or any(len(pair) != 2 or pair[0] == pair[1]
            or any(type(value) is not int or value <= 0 for value in pair) for pair in addresses)):
        raise ValueError('Two distinct positive native/stream address pairs required')
    return expected


def inspect(evidence, serial_report, directory):
    evidence = Path(evidence)
    serial_raw = Path(serial_report).read_bytes()
    if digest(serial_raw) != SERIAL_SHA256:
        raise ValueError('Pinned serial block-stream simulator reference required')
    serial = json.loads(serial_raw)
    validate_report(serial)
    validate_weights(serial)
    raw = (evidence / 'fused-batch.json').read_bytes()
    staged_raw = (evidence / 'progressive-input-candidate.json').read_bytes()
    report = json.loads(raw)
    kernel = validate_candidate(report, serial, json.loads(staged_raw), directory)
    if ((evidence / 'fused-batch.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_text()) !=
            dict(stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Pinned runtime, zero exit and clean simulator teardown required')
    base = json.loads((evidence / 'block-stream-candidate.json').read_text())
    for name in ('mlp_block_stream.py', 'mlp_block_stream.cpp', 'mlp_block_stream_projection.py',
            'mlp_block_stream_probe.py', 'mlp_register_epilogue.py', 'mlp_rounding_policy.py'):
        if digest((Path(directory) / name).read_bytes()) != base['after'].get(name):
            raise ValueError('Serial transport or arithmetic source changed: ' + name)
    return dict(passed=True, report_sha256=digest(raw), staging_sha256=digest(staged_raw),
        kernel=kernel, extra_l1_bytes_per_multicast_core=144 * 2048,
        hardware_qualified=False, performance_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--serial-report', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(inspect(options.evidence, options.serial_report, Path(__file__).parent), indent=2))
