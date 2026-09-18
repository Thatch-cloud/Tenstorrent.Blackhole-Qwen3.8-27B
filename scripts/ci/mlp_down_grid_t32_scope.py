"""Apply the qualified T32 down grid only within its request-owned fusion arm."""

from contextlib import contextmanager
from unittest.mock import patch

from mlp_down_grid import widen
from mlp_down_grid_t32_gate import REPORT_SHA256


@contextmanager
def scoped_down_grid_t32(admission):
    import fused_t16_scope

    if (admission.get('report_sha256') != REPORT_SHA256 or admission.get('rows') != 32
            or admission.get('simulator_qualified') is not True):
        raise ValueError('Source-qualified T32 MLP-down admission required')
    arm_type = fused_t16_scope.FusedT32Arm
    original = arm_type.forward
    if getattr(original, '_mlp_down_grid', False):
        raise ValueError('Nested MLP-down scopes forbidden')
    audit = dict(report_sha256=REPORT_SHA256, rows=32, hits=[0] * 64, fallbacks=0, restored=False)

    def forward(arm, index, value):
        if tuple(value.shape) != (1, 1, 32, 5120):
            audit['fallbacks'] += 1
            return original(arm, index, value)
        if arm.token_rows != 32 or type(index) is not int or not 0 <= index < 64:
            raise ValueError('Complete T32 fused MLP layer required')
        mlp, native_forward = arm.originals[index]
        operations = arm.operations
        weight = mlp.weights.w2
        shards = operations.get_device_tensors(weight)
        if (weight.dtype != operations.bfloat8_b or len(shards) != 2
                or any(tuple(shard.shape)[-2:] != (8704, 5120) for shard in shards)):
            raise ValueError('Unchanged two-chip BF8 MLP-down weights required')
        config = mlp.compute_kernel_config_decode
        if (config.math_fidelity != operations.MathFidelity.LoFi or not config.fp32_dest_acc_en
                or not config.packer_l1_acc):
            raise ValueError('Simulator-covered native MLP-down arithmetic required')
        current = mlp.args.mlp_w2_decode_1d_progcfg
        candidate = widen(operations, current)
        with patch.object(mlp.args, 'mlp_w2_decode_1d_progcfg', candidate):
            result = original(arm, index, value)
        if mlp.args.mlp_w2_decode_1d_progcfg is not current:
            raise ValueError('Shared model args were not restored')
        audit['hits'][index] += 1
        return result

    forward._mlp_down_grid = True
    try:
        with patch.object(arm_type, 'forward', forward):
            try:
                yield audit
            finally:
                if arm_type.forward is not forward:
                    raise ValueError('T32 MLP binding changed outside its owning scope')
    finally:
        audit['restored'] = arm_type.forward is original
