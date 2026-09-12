"""Independently check the simulator's compressed-word transport evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from tensix_weight_stream import stream_geometry


SOURCES = ('tensix-weight-stream-probe.py', 'tensix_weight_stream.py', 'tensix_weight_stream_reader.cpp',
    'tensix_weight_stream_writer.cpp', 'tensix_weight_stream_sink.cpp', 'tensix_weight_stream_raw.cpp',
    'tiny_tile_matmul.py', 'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py',
    'tensix_weight_packet.py', 'tensix_weight_packet_reader.cpp')
NATIVE_SOURCES = ('tt_metal/hw/inc/api/remote_circular_buffer.h',
    'tt_metal/impl/buffers/global_circular_buffer.cpp', 'ttnn/core/global_circular_buffer.cpp',
    'ttnn/cpp/ttnn-nanobind/program_descriptors.cpp',
    'tt_metal/hw/inc/api/dataflow/dataflow_api.h', 'tt_metal/hw/inc/api/tensor/tensor_accessor.h',
    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h',
    'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')


def hashes(root, names):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def matrix(rows, fields, expected, flags):
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('Incomplete transport check matrix')
    actual = []
    for row in rows:
        if (not isinstance(row, dict) or any(type(row.get(field)) is not int for field in fields)
                or any(row.get(flag) is not True for flag in flags)):
            raise ValueError('Transport check must contain exact integer indices and true audit flags')
        actual.append(tuple(row[field] for field in fields))
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Transport check matrix contains a duplicate or wrong coordinate')


def qualify(report, sources, native_sources, *, single_packet=False, exit_status=None):
    if (not isinstance(report, dict) or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('error')):
        raise ValueError('Clean terminal simulator pass required')
    if (type(single_packet) is not bool or report.get('arm_policies', ['generic', 'generic'])
            != ['generic', 'single-packet' if single_packet else 'generic']):
        raise ValueError('Explicit matching single-packet versus generic transport policy required')
    for field, names, expected in (('sources', SOURCES, sources), ('native_sources', NATIVE_SOURCES, native_sources)):
        if set(expected) != set(names) or report.get(field) != expected:
            raise ValueError('Current experiment and native source hashes required')
        if any(not isinstance(value, str) or len(value) != 64
                or any(character not in '0123456789abcdef' for character in value) for value in expected.values()):
            raise ValueError('SHA256 source hashes required')
    geometry = report.get('geometry', {})
    if geometry.get('projection') not in ('gate', 'down'):
        raise ValueError('Only gate and down transport probes are qualified')
    expected_geometry = json.loads(json.dumps(stream_geometry(geometry['projection'], geometry.get('blocks'),
        geometry.get('producers', 8))))
    if (geometry != expected_geometry or type(geometry.get('full_projection')) is not bool
            or any(type(geometry.get(field)) is not int for field in ('blocks', 'rows', 'width', 'receivers',
                'tile_bytes', 'per_receiver', 'key_block_tiles', 'page_bytes'))):
        raise ValueError('Exact target transport geometry required')
    if single_packet:
        expected_readers = [dict(tile_bytes=geometry['tile_bytes'], block_rows=8,
            receiver_columns=geometry['per_receiver'], total_columns=geometry['width'] // 32)] * 2
        if (exit_status is None or exit_status.strip() != '0' or geometry.get('producers') != 16
                or geometry['full_projection'] is not True or report.get('stage') != 'complete'
                or report.get('reader_engagements') != expected_readers
                or report.get('target_integrated') is not False or report.get('eligible_for_hardware') is not False
                or report.get('sources_after') != sources or report.get('native_sources_after') != native_sources):
            raise ValueError('Complete full-size single-packet run, exact engagements, unchanged sources and outer exit0 required')
    elif report.get('reader_engagements', []):
        raise ValueError('Generic transport must not contain candidate reader engagements')
    matrix(report.get('eager_checks'), ('pattern', 'arm', 'chip'),
        {(pattern, arm, chip) for pattern in range(2) for arm in range(2) for chip in range(2)},
        ('exact_packed_words', 'inputs_unchanged'))
    matrix(report.get('replay_checks'), ('repetition', 'pattern', 'arm', 'chip'),
        {(repetition, pattern, arm, chip) for repetition, pattern in enumerate((0, 1, 0))
            for arm in range(2) for chip in range(2)},
        ('exact_packed_words', 'inputs_unchanged', 'bindings_stable'))
    matrix(report.get('negative_controls'), ('arm', 'chip'), {(arm, chip) for arm in range(2) for chip in range(2)},
        ('stale_detected',))
    return dict(passed=True, projection=geometry['projection'], blocks=geometry['blocks'],
        full_projection=geometry['full_projection'], scope='Transport correctness only; not matmul or hardware speed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--native-root', required=True, type=Path)
    parser.add_argument('--single-packet', action='store_true')
    parser.add_argument('--exit-status', type=Path)
    options = parser.parse_args()
    print(json.dumps(qualify(json.loads(options.report.read_text()), hashes(Path(__file__).parent, SOURCES),
        hashes(options.native_root, NATIVE_SOURCES), single_packet=options.single_packet,
        exit_status=options.exit_status.read_text() if options.exit_status else None)))


if __name__ == '__main__':
    main()
