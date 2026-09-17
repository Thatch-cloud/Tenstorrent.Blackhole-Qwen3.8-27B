"""Change only T16 fused MLP down program construction, restoring shared args immediately."""

from contextlib import contextmanager
from unittest.mock import patch

from mlp_down_grid import widen
from mlp_down_grid_gate import REPORT_SHA256


@contextmanager
def scoped_down_grid(admission):
    import fused_t16_scope

    if admission.get('report_sha256') != REPORT_SHA256:
        raise ValueError('Source-qualified simulator admission required')
    original = fused_t16_scope.FusedT16Arm.forward
    if getattr(original, '_mlp_down_grid', False):
        raise ValueError('Nested MLP-down scopes forbidden')
    audit = dict(report_sha256=REPORT_SHA256, hits=[0] * 64, restored=False)

    def forward(arm, index, value):
        if tuple(value.shape) != (1, 1, 16, 5120):
            return original(arm, index, value)
        if getattr(arm, 'token_rows', 16) != 16 or not 0 <= index < 64:
            raise ValueError('Only complete T16 fused MLP layers supported')
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
            raise ValueError('Simulator-covered native MLP-down compute required')
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
        with patch.object(fused_t16_scope.FusedT16Arm, 'forward', forward):
            try:
                yield audit
            finally:
                if fused_t16_scope.FusedT16Arm.forward is not forward:
                    raise ValueError('MLP forward changed outside its owning scope')
    finally:
        audit['restored'] = fused_t16_scope.FusedT16Arm.forward is original
