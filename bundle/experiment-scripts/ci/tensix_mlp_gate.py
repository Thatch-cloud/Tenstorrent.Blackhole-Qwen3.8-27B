"""Independent source and audit gate for full-size pooled streamed MLP simulation."""

import argparse
import json
from pathlib import Path

from tensix_projection_gate import NATIVE_SOURCES as PROJECTION_NATIVE, ORIGINAL_PACKER, PACKER, SIMULATOR_PACKER
from tensix_projection_gate import SOURCES as PROJECTION_SOURCES, hashes
from tensix_stream_gate import matrix
from tiny_mlp_gate import HARDWARE_TP_COMMON, SIMULATOR_TP_COMMON, TP_COMMON
from tensix_weight_stream import stream_geometry


COMPONENTS = ('gate', 'up', 'hidden', 'partial')
SOURCES = (*PROJECTION_SOURCES, 'tensix_stream_mlp.py', 'tensix-stream-mlp-probe.py',
    'tensix_mlp_gate.py', 'tiny_mlp.py', 'tiny_mlp_gate.py', 'tensix_weight_packet.py', 'tensix_weight_packet_reader.cpp')
NATIVE_SOURCES = (*PROJECTION_NATIVE, 'ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp',
    'ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp',
    'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp',
    'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp',
    'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp',
    'ttnn/cpp/ttnn-nanobind/tensor.cpp',
    'ttnn/cpp/ttnn/operations/data_movement/copy/copy.cpp',
    'ttnn/cpp/ttnn/operations/data_movement/copy/copy_nanobind.cpp',
    'ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_device_operation.cpp',
    'ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_default_tilized_program_factory.cpp',
    'ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_start_id.cpp',
    'ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp',
    'ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp',
    'ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp',
    'ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp',
    'tt_metal/hw/inc/api/dataflow/dataflow_api.h', 'tt_metal/hw/inc/api/tensor/tensor_accessor.h')


def qualify_producers(report, expected=None):
    count = report.get('producers_per_card')
    if type(count) is not int or count not in (8, 16) or (expected is not None and count != expected):
        raise ValueError('Simulator and runtime must match the explicit producer count')
    mappings = {name: stream_geometry(name, 34 if name == 'down' else 20, count)['mapping']
        for name in ('gate', 'up', 'down')}
    encoded = json.dumps(mappings, sort_keys=True)
    if json.dumps(report.get('producer_mappings'), sort_keys=True) != encoded:
        raise ValueError('Complete exact producer coordinates and receiver assignments required')
    return count


def qualify_reader(report, single_packet=False, *, fixtures=2):
    if type(single_packet) is not bool or report.get('single_packet', False) is not single_packet:
        raise ValueError('Explicit matching generic or single-packet weight-reader policy required')
    expected = []
    if single_packet:
        if report.get('producers_per_card') != 16:
            raise ValueError('Single-packet candidate retains exactly sixteen producers')
        expected = [dict(tile_bytes=size, block_rows=8, receiver_columns=columns, total_columns=width)
            for unused_fixture in range(fixtures) for size, columns, width in ((576, 4, 272), (576, 4, 272), (1088, 2, 160))
            for unused_chip in range(2)]
    if report.get('reader_engagements', []) != expected:
        raise ValueError('Every candidate gate/up/down reader on both chips must be engaged exactly once per fixture')
    return single_packet


def qualify(report, sources, native_sources, *, single_packet=False):
    if (not isinstance(report, dict) or report.get('backend') != 'simulator' or report.get('error')
            or any(report.get(field) is not True for field in ('passed', 'closed_cleanly', 'packer_zero_graft',
                'shared_pool', 'shared_workspace', 'full_mlp', 'dram_boundary'))
            or any(type(report.get(field)) is not int or report[field] != expected for field, expected in (
                ('rows', 8), ('compared_rows', 32), ('fixtures', 2), ('pool_buffers', 2)))
            or report.get('components') != list(COMPONENTS)):
        raise ValueError('Clean full-size T8 MLP simulation with two pooled FIFOs and shared workspace required')
    qualify_producers(report)
    qualify_reader(report, single_packet)
    if single_packet and (report.get('sources_after') != report.get('sources')
            or report.get('native_sources_after') != report.get('native_sources')):
        raise ValueError('Single-packet probe and native sources must remain unchanged through teardown')
    if not isinstance(sources, dict) or set(sources) != set(SOURCES) or report.get('sources') != sources:
        raise ValueError('Current complete MLP sources required')
    if (not isinstance(native_sources, dict) or not isinstance(report.get('native_sources'), dict)
            or set(native_sources) != set(NATIVE_SOURCES)
            or native_sources.get(PACKER) not in (ORIGINAL_PACKER, SIMULATOR_PACKER)):
        raise ValueError('Audited native operation sources required')
    expected_native = {**native_sources, PACKER: SIMULATOR_PACKER}
    equivalence = (report.get('native_sources', {}).get(TP_COMMON) == SIMULATOR_TP_COMMON
        and native_sources[TP_COMMON] == HARDWARE_TP_COMMON)
    if equivalence:
        expected_native[TP_COMMON] = SIMULATOR_TP_COMMON
    if report.get('native_sources') != expected_native:
        raise ValueError('Native matmul, pooling, product or simulator graft provenance changed')
    for digest in [*sources.values(), *native_sources.values()]:
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest):
            raise ValueError('Full SHA256 provenance required')
    coordinates = {(pattern, fixture, component, chip) for pattern in range(2) for fixture in range(2)
        for component in range(4) for chip in range(2)}
    for field in ('control_checks', 'eager_checks'):
        matrix(report.get(field), ('pattern', 'fixture', 'component', 'chip'), coordinates, ('exact_all_32_rows',))
    matrix(report.get('replay_checks'), ('repetition', 'pattern', 'fixture', 'component', 'chip'),
        {(repetition, pattern, fixture, component, chip) for repetition, pattern in enumerate((0, 1, 0))
            for fixture in range(2) for component in range(4) for chip in range(2)},
        ('exact_all_32_rows', 'bindings_stable', 'pool_reused'))
    matrix(report.get('input_checks'), ('phase', 'repetition', 'tensor', 'chip'),
        {(phase, repetition, tensor, chip) for phase, repeats in ((0, 2), (1, 3)) for repetition in range(repeats)
            for tensor in range(7) for chip in range(2)}, ('packed_words_unchanged',))
    matrix(report.get('negative_controls'), ('fixture', 'chip'),
        {(fixture, chip) for fixture in range(2) for chip in range(2)}, ('stale_detected',))
    matrix(report.get('fixture_checks'), ('chip',), {(0,), (1,)}, ('different_weights_detected',))
    bindings = report.get('weight_bindings')
    if (not isinstance(bindings, list) or len(bindings) != 6
            or any(not isinstance(pair, list) or len(pair) != 2
                or any(type(address) is not int or address <= 0 for address in pair) for pair in bindings)
            or any(len({pair[chip] for pair in bindings}) != 6 for chip in range(2))):
        raise ValueError('Both synthetic fixtures need six distinct physical weight buffers per chip')
    result = dict(passed=True, full_mlp=True, scope='Exact simulated pre-collective MLP; not hardware speed or model TG')
    if single_packet:
        result['single_packet'] = True
    if equivalence:
        result['native_equivalence'] = 'Pinned prefill-only tp_common difference; T8 decode helper unchanged'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--exit-status', type=Path, required=True)
    parser.add_argument('--single-packet', action='store_true')
    options = parser.parse_args()
    if options.exit_status.read_text().strip() != '0':
        raise ValueError('Clean outer simulator wrapper exit required')
    print(json.dumps(qualify(json.loads(options.report.read_text()), hashes(Path(__file__).parent, SOURCES),
        hashes(options.native_root, NATIVE_SOURCES), single_packet=options.single_packet)))


if __name__ == '__main__':
    main()
