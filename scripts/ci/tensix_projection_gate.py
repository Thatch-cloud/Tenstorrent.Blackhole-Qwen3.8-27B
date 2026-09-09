"""Source-bound simulator evidence for the experimental streamed native projection."""

import argparse
import hashlib
import json
from pathlib import Path

from tensix_stream_gate import matrix
from tensix_stream_projection import COMPUTE
from tensix_weight_stream import stream_geometry


PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
ORIGINAL_PACKER = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
SIMULATOR_PACKER = '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7'
SOURCES = ('tensix-stream-projection-probe.py', 'tensix_stream_projection.py', 'tensix_stream_activation.cpp',
    'tensix_stream_matmul_sink.cpp', 'tensix_projection_raw.cpp', 'tensix_projection_raw.py',
    'tensix_projection_gate.py', 'tensix_stream_gate.py', 'tensix_weight_stream_reader.cpp',
    'tensix_weight_stream_writer.cpp', 'tensix_weight_stream.py', 'tiny_tile_matmul.py',
    'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py')
NATIVE_SOURCES = (COMPUTE, 'ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp',
    'ttnn/cpp/ttnn/operations/matmul/shared_with_host/activation_type.hpp',
    'ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp',
    'ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp',
    PACKER, 'tt_metal/hw/inc/api/remote_circular_buffer.h', 'tt_metal/hw/inc/internal/circular_buffer_init.h',
    'tt_metal/hw/inc/internal/circular_buffer_interface.h', 'tt_metal/impl/buffers/global_circular_buffer.cpp',
    'ttnn/cpp/ttnn-nanobind/program_descriptors.cpp', 'tt_metal/api/tt-metalium/program_descriptors.hpp',
    'ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp',
    'models/demos/blackhole/qwen36/tt/tp_common.py')


def hashes(root, names):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def qualify(report, sources, native_sources):
    if (not isinstance(report, dict) or report.get('backend') != 'simulator' or report.get('error')
            or any(report.get(field) is not True for field in ('passed', 'closed_cleanly',
                'packer_zero_graft', 'native_compute_unchanged'))
            or type(report.get('token_rows')) is not int or report['token_rows'] != 8
            or type(report.get('compared_rows')) is not int or report['compared_rows'] != 32):
        raise ValueError('Clean simulator pass covering native T8 and all32 physical rows required')
    if set(sources) != set(SOURCES) or report.get('sources') != sources:
        raise ValueError('Current experiment source hashes required')
    if set(native_sources) != set(NATIVE_SOURCES) or native_sources.get(PACKER) not in (ORIGINAL_PACKER, SIMULATOR_PACKER):
        raise ValueError('Audited native sources and original or scoped simulator packer required')
    expected_native = {**native_sources, PACKER: SIMULATOR_PACKER}
    if report.get('native_sources') != expected_native:
        raise ValueError('Native source drift or missing explicit simulator graft provenance')
    for value in [*sources.values(), *native_sources.values()]:
        if not isinstance(value, str) or len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
            raise ValueError('Full SHA256 provenance required')
    geometry = report.get('geometry', {})
    if not isinstance(geometry, dict) or geometry.get('projection') not in ('gate', 'up', 'down'):
        raise ValueError('Explicit native MLP projection required')
    expected = stream_geometry(geometry['projection'], geometry.get('blocks'))
    if json.dumps(geometry, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('Reviewed stream geometry required')
    matrix(report.get('control_checks'), ('pattern', 'chip'),
        {(pattern, chip) for pattern in range(2) for chip in range(2)}, ('exact_all_32_rows',))
    matrix(report.get('eager_checks'), ('pattern', 'arm', 'chip'),
        {(pattern, arm, chip) for pattern in range(2) for arm in range(2) for chip in range(2)},
        ('exact_all_32_rows', 'inputs_unchanged'))
    matrix(report.get('replay_checks'), ('repetition', 'pattern', 'arm', 'chip'),
        {(repetition, pattern, arm, chip) for repetition, pattern in enumerate((0, 1, 0))
            for arm in range(2) for chip in range(2)}, ('exact_all_32_rows', 'inputs_unchanged', 'bindings_stable'))
    matrix(report.get('negative_controls'), ('arm', 'chip'),
        {(arm, chip) for arm in range(2) for chip in range(2)}, ('stale_detected',))
    return dict(passed=True, projection=geometry['projection'], blocks=geometry['blocks'],
        full_projection=geometry['full_projection'], scope='Simulated projection exactness, not complete MLP or hardware speed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(qualify(json.loads(options.report.read_text()), hashes(Path(__file__).parent, SOURCES),
        hashes(options.native_root, NATIVE_SOURCES))))


if __name__ == '__main__':
    main()
