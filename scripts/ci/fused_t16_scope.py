"""Opt-in target T16 fusion using existing weights and unchanged down/collective paths."""

from contextlib import contextmanager
import time

from fused_1d import FusedProjection
from fused_t16_admission import qualify_simulator, qualify_target_weights
from gdn_multitoken_conv import addresses
from model_batch import instance_overrides


class FusedT16Arm:
    token_rows = 16

    def __init__(self, operations, model, collective):
        if type(self.token_rows) is not int or self.token_rows not in (16, 32):
            raise ValueError('Explicit simulator-covered fusion width required')
        self.operations, self.model, self.collective = operations, model, collective
        self.active = False
        self.hits, self.fallbacks = [0] * 64, [0] * 64
        started = time.perf_counter()
        self.evidence = qualify_simulator()
        self.weight_audit = qualify_target_weights(operations, model)
        self.originals = [(layer.feed_forward, layer.feed_forward.forward) for layer in model.layers]
        if len({id(mlp) for mlp, original in self.originals}) != 64:
            raise ValueError('Distinct target MLP instances required')
        expected = next(kernel for kernel in self.evidence['kernels'] if kernel['token_rows'] == self.token_rows)
        self.projections = []
        for mlp, original in self.originals:
            projection = FusedProjection(mlp.device, mlp.weights.w_gate_up,
                pairs_per_worker=3, token_rows=self.token_rows, math_approx_mode=True)
            if projection.manifest != expected:
                raise ValueError('Target fusion manifest differs from qualified simulator')
            self.projections.append(projection)
        self.bindings = self.weight_bindings()
        self.audit = dict(rows=self.token_rows, layers=64, restored=False, passed_simulator=self.evidence['report_sha256'],
            weight_audit=self.weight_audit, setup_ms=(time.perf_counter() - started) * 1000,
            extra_weight_allocations=0, serving_defaults_changed=False)

    def weight_bindings(self):
        return [[addresses(self.operations, weight) for weight in
            (mlp.weights.w1, mlp.weights.w3, mlp.weights.w2, mlp.weights.w_gate_up)]
            for mlp, original in self.originals]

    def forward(self, index, value):
        if not self.active:
            raise RuntimeError('Fusion requires an active instance scope')
        mlp, original = self.originals[index]
        if tuple(value.shape) != (1, 1, self.token_rows, 5120):
            self.fallbacks[index] += 1
            return original(value)
        operations = self.operations
        if value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT:
            raise ValueError('Native tiled BF16 target activation required')
        local = operations.to_memory_config(value, operations.L1_MEMORY_CONFIG)
        original_addresses = addresses(operations, value)
        local_addresses = addresses(operations, local)
        aliases = [first == second for first, second in zip(original_addresses, local_addresses, strict=True)]
        if any(aliases) and not all(aliases):
            raise ValueError('Mixed per-chip activation ownership is unsupported')
        hidden = None
        try:
            hidden = self.projections[index](local)
            partial = operations.linear(hidden, mlp.weights.w2,
                compute_kernel_config=mlp.compute_kernel_config_decode,
                program_config=mlp.args.mlp_w2_decode_1d_progcfg,
                memory_config=operations.L1_MEMORY_CONFIG)
        finally:
            if hidden is not None:
                operations.deallocate(hidden)
            if not all(aliases):
                operations.deallocate(local)
        result = self.collective(partial, mlp.device, mlp.tt_ccl, cluster_axis=0, dim=3,
            topology=mlp.args.ccl_topology(), memory_config=operations.DRAM_MEMORY_CONFIG)
        self.hits[index] += 1
        return result

    @contextmanager
    def install(self):
        if self.active:
            raise RuntimeError('Fusion scope already active')
        self.active = True
        try:
            with instance_overrides([(mlp, 'forward', lambda value, index=index: self.forward(index, value))
                    for index, (mlp, original) in enumerate(self.originals)]):
                yield self
        finally:
            self.active = False
            self.audit.update(hits=list(self.hits), fallbacks=list(self.fallbacks),
                restored=all(mlp.forward == original for mlp, original in self.originals),
                native_bindings_unchanged=self.weight_bindings() == self.bindings)
            if not self.audit['restored'] or not self.audit['native_bindings_unchanged']:
                raise AssertionError('Native target bindings changed during fusion scope')


class FusedT32Arm(FusedT16Arm):
    token_rows = 32
