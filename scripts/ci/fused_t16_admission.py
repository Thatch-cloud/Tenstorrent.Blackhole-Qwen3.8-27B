"""Require retained target-mode simulation and byte-exact existing target weights."""

import hashlib
import json
from pathlib import Path

from gdn_multitoken_conv import release_owned


REPORT_SHA256 = 'f155635711e3617cbdb9566a3fdd19e8ab62d9dc7d9eacbf684b9343ab8bd9bb'


def qualify_simulator(root=None):
    root = Path(root) if root is not None else Path(__file__).parent
    raw = (root / 'fused-t16-target-simulator.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Reviewed target-mode simulator report required')
    if (root / 'fused-t16-target-simulator.exit-status').read_text().strip() != '0':
        raise ValueError('Clean simulator exit required')
    report = json.loads(raw)
    if (report.get('passed') is not True or report.get('backend') != 'simulator'
            or report.get('math_approx_mode') is not True
            or [item['rows'] for item in report['trace_replays']] != [1, 8, 16, 32]):
        raise ValueError('Complete target-mode simulator matrix required')
    if hashlib.sha256((root / 'fusion_trace.py').read_bytes()).hexdigest() != report['trace_source_sha256']:
        raise ValueError('Trace helper changed since simulation')
    for kernel in report['kernels']:
        if kernel.get('math_approx_mode') is not True:
            raise ValueError('Target-native math mode required')
        for name, expected in kernel['reader_sha256'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Fusion reader changed since simulation')
    return dict(report_sha256=REPORT_SHA256, kernels=report['kernels'], passed=True)


def qualify_target_weights(operations, model):
    from packed_weight_check import compare_packed_weights, read_comparison

    if len(model.layers) != 64 or model.args.num_devices != 2:
        raise ValueError('All 64 target layers on two devices required')
    checks = []
    for index, layer in enumerate(model.layers):
        mlp = layer.feed_forward
        compute = mlp.compute_kernel_config_decode
        if (not mlp._mlp_1d_decode or mlp._dram_sharded
                or compute.math_fidelity != operations.MathFidelity.LoFi
                or compute.math_approx_mode is not True
                or not compute.fp32_dest_acc_en or not compute.packer_l1_acc):
            raise ValueError('Unchanged native target decode math required')
        for weight, shape, dtype in (
                (mlp.weights.w1, (5120, 8704), operations.bfloat4_b),
                (mlp.weights.w3, (5120, 8704), operations.bfloat4_b),
                (mlp.weights.w_gate_up, (5120, 17408), operations.bfloat4_b),
                (mlp.weights.w2, (8704, 5120), operations.bfloat8_b)):
            if (weight is None or tuple(weight.shape)[-2:] != shape or weight.dtype != dtype
                    or weight.memory_config() != operations.DRAM_MEMORY_CONFIG):
                raise ValueError('Existing native interleaved target weights required')
        owned = []
        try:
            for offset, separate in enumerate((mlp.weights.w1, mlp.weights.w3)):
                result = compare_packed_weights(model.mesh_device, mlp.weights.w_gate_up, separate, offset, owned)
                for check in read_comparison(operations, result, 43520):
                    if not check['exact']:
                        raise AssertionError(f'Target packed weights differ at layer {index}, offset {offset}')
                    checks.append(dict(layer=index, offset=offset, **check))
        finally:
            release_owned(operations, owned)
    if len(checks) != 256:
        raise AssertionError('Both projections and chips for all 64 layers required')
    return dict(passed=True, checks=checks, extra_weight_allocations=0)
