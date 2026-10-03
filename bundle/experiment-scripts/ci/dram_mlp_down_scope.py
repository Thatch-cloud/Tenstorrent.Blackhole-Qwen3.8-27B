"""Reversible target-only T16 MLP experiment; native prefill and T1 remain untouched."""

from contextlib import contextmanager
import time

from dram_mlp_down import execute
from dram_sharded_projection import configurations
from gdn_multitoken_conv import addresses, release_owned


class DownOnlyArm:
    def __init__(self, operations, model, collective, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('Explicit target MLP route required')
        if len(model.layers) != 64 or model.args.num_devices != 2:
            raise ValueError('Complete 64-layer TP2 target required')
        self.operations, self.mesh, self.collective = operations, model.mesh_device, collective
        self.originals, self.weights, self.installed = [], [], []
        self.hits, self.fallbacks = [0] * 64, [0] * 64
        self.closed = False
        self.audit = dict(restored=False, layers=64, rows=16, setup_ms=None, enabled=enabled,
            scope='Target T16 only; native prefill/T1 and drafter untouched')
        started = time.perf_counter()
        try:
            config = configurations(operations, self.mesh, 'down')
            for layer in model.layers:
                mlp = layer.feed_forward
                compute = mlp.compute_kernel_config_decode
                if (not mlp._mlp_1d_decode or mlp._dram_sharded
                        or compute.math_fidelity != operations.MathFidelity.LoFi
                        or not compute.fp32_dest_acc_en or not compute.packer_l1_acc
                        or tuple(mlp.weights.w2.shape)[-2:] != (8704, 5120)
                        or mlp.weights.w2.dtype != operations.bfloat8_b
                        or mlp.weights.w2.memory_config() != operations.DRAM_MEMORY_CONFIG):
                    raise ValueError('Unchanged native decode precision and interleaved down weights required')
                for weight in (mlp.weights.w1, mlp.weights.w3):
                    if (tuple(weight.shape)[-2:] != (5120, 8704) or weight.dtype != operations.bfloat4_b
                            or weight.memory_config() != operations.DRAM_MEMORY_CONFIG):
                        raise ValueError('Unchanged native BF4 gate and up weights required')
                self.originals.append((mlp, mlp.forward))
            self.native_bindings = [[addresses(operations, weight) for weight in
                (mlp.weights.w1, mlp.weights.w3, mlp.weights.w2)] for mlp, unused in self.originals]
            for mlp, unused in self.originals:
                self.weights.append(operations.to_memory_config(mlp.weights.w2, config['weights']))
            self.audit['prepared_bindings'] = [addresses(operations, weight) for weight in self.weights]
            for index, (mlp, original) in enumerate(self.originals):
                weights = dict(gate=mlp.weights.w1, up=mlp.weights.w3, down=self.weights[index])
                configs = dict(gate=dict(program=mlp.args.mlp_w1_decode_1d_progcfg),
                    up=dict(program=mlp.args.mlp_w3_decode_1d_progcfg), down=config)
                def forward(value, mlp=mlp, original=original, index=index, weights=weights, configs=configs):
                    if self.closed:
                        raise RuntimeError('Closed target MLP experiment cannot execute')
                    if not enabled or tuple(value.shape) != (1, 1, 16, 5120):
                        self.fallbacks[index] += 1
                        return original(value)
                    self.hits[index] += 1
                    local = operations.to_memory_config(value, operations.L1_MEMORY_CONFIG)
                    owned = addresses(operations, local) != addresses(operations, value)
                    try:
                        partial = execute(operations, local, weights, configs, mlp.compute_kernel_config_decode,
                            lambda tensor: tensor)
                    finally:
                        if owned:
                            operations.deallocate(local)
                    return collective(partial, mlp.device, mlp.tt_ccl, cluster_axis=0, dim=3,
                        topology=mlp.args.ccl_topology(), memory_config=operations.DRAM_MEMORY_CONFIG)
                mlp.forward = forward
                self.installed.append((mlp, original))
            operations.synchronize_device(self.mesh)
            self.audit['setup_ms'] = (time.perf_counter() - started) * 1000
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for mlp, original in self.installed:
            mlp.forward = original
        self.audit.update(hits=list(self.hits), fallbacks=list(self.fallbacks),
            restored=all(mlp.forward is original for mlp, original in self.installed))
        try:
            self.operations.synchronize_device(self.mesh)
            if 'prepared_bindings' in self.audit:
                self.audit['prepared_bindings_unchanged'] = self.audit['prepared_bindings'] == [
                    addresses(self.operations, weight) for weight in self.weights]
                if not self.audit['prepared_bindings_unchanged']:
                    raise AssertionError('Prepared down weight bindings changed')
            if hasattr(self, 'native_bindings'):
                current = [[addresses(self.operations, weight) for weight in
                    (mlp.weights.w1, mlp.weights.w3, mlp.weights.w2)] for mlp, unused in self.originals]
                self.audit['native_bindings_unchanged'] = current == self.native_bindings
                if current != self.native_bindings:
                    raise AssertionError('Native target MLP bindings changed')
        finally:
            release_owned(self.operations, self.weights)


@contextmanager
def scoped_down(operations, model, collective, *, enabled=True):
    arm = DownOnlyArm(operations, model, collective, enabled=enabled)
    try:
        yield arm.audit
    finally:
        arm.close()
