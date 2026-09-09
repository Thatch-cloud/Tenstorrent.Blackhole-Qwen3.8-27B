"""Fail-closed source and comparison manifest for the small-tile MLP gate."""


SOURCES = (
    'tiny_mlp.py', 'tiny-mlp-probe.py', 'tiny_tile_dma.py', 'tiny_tile_dma.cpp', 'tiny_tile_matmul.py',
    'attention_batch.py', 'tiny_tile_product.py', 'tiny_tile_product_compute.cpp', 'tiny_tile_product_io.cpp')
PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
ORIGINAL_PACKER = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
SIMULATOR_PACKER = '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7'
NATIVE_SOURCES = (
    'models/demos/blackhole/qwen36/tt/tp_common.py', 'models/demos/blackhole/qwen36/tt/mlp.py', PACKER,
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
    if report.get('native_sources') != expected_native:
        raise ValueError('Native configuration or operation changed since simulation')
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
    return dict(passed=True, eager_checks=len(eager), trace_checks=len(trace), negative_controls=len(negative))
