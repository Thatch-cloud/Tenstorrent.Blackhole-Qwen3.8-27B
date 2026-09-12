"""Opt-in full-model fusion helpers; no serving or installed-source changes."""

import math
import statistics

from fused_1d import FusedProjection


def supported_decode(shape, enabled):
    return enabled and list(shape) == [1, 1, 1, 5120]


def paired_block(records):
    if [record["arm"] for record in records] != ["control", "fused", "fused", "control"]:
        raise ValueError("Expected an ABBA block")
    reference = records[0]["token_ids"]
    if len(reference) < 2 or any(record["token_ids"] != reference for record in records):
        raise ValueError("Full-model token divergence")
    if any(not math.isfinite(record["decode_seconds"]) or record["decode_seconds"] <= 0 for record in records):
        raise ValueError("Invalid decode timing")
    control = statistics.mean(records[index]["decode_seconds"] for index in (0, 3))
    fused = statistics.mean(records[index]["decode_seconds"] for index in (1, 2))
    return dict(control_seconds=control, fused_seconds=fused, latency_change=fused / control - 1,
                control_host_tok_s=(len(reference) - 1) / control,
                fused_host_tok_s=(len(reference) - 1) / fused)


class FusionArm:
    def __init__(self, model):
        import ttnn
        from models.tt_transformers.tt.ccl import tt_all_reduce

        if len(model.layers) != 64 or model.args.num_devices != 2:
            raise ValueError("Only the reviewed full 64-layer TP2 model is supported")
        self.enabled = False
        self.hits = [0] * 64
        self.fallbacks = [0] * 64
        self.originals = []
        self.kernels = []
        for index, layer in enumerate(model.layers):
            mlp = layer.feed_forward
            if not mlp._mlp_1d_decode or mlp._dram_sharded or mlp.weights.w_gate_up is None:
                raise ValueError(f"Layer {index} has no compatible existing packed prefill weights")
            if tuple(tensor.dtype for tensor in (mlp.weights.w1, mlp.weights.w2, mlp.weights.w3, mlp.weights.w_gate_up)) != (
                ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat4_b
            ):
                raise ValueError(f"Layer {index} precision changed")
            projection = FusedProjection(mlp.device, mlp.weights.w_gate_up, pairs_per_worker=3)
            if projection.manifest["fused_compute_sha256"] != "97268b7be1054c24edd392263d2d666728e6da3659125903f02d0eba4951c258":
                raise ValueError("Full-model candidate differs from the measured MLP compute")
            if projection.manifest["reader_sha256"] != {
                "fused_1d_input.cpp": "381739e8070b910c61395e5e82970200d96f1de8bff0de25f74d483a1d3258b9",
                "fused_1d_weights.cpp": "c283b6b347045ee3b59363e4e3aa6b0b796e226a5ccbf6df9aa8930a570f46eb",
            }:
                raise ValueError("Full-model reader differs from the measured MLP reader")
            original = mlp.forward
            self.originals.append((mlp, original))
            self.kernels.append(projection.manifest)

            def forward(value, mlp=mlp, projection=projection, original=original, index=index):
                if not supported_decode(value.shape, self.enabled):
                    self.fallbacks[index] += 1
                    return original(value)
                self.hits[index] += 1
                local = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)
                hidden = projection(local)
                ttnn.deallocate(local)
                partial = ttnn.linear(hidden, mlp.weights.w2,
                    compute_kernel_config=mlp.compute_kernel_config_decode,
                    program_config=mlp.args.mlp_w2_decode_1d_progcfg, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(hidden)
                return tt_all_reduce(partial, mlp.device, mlp.tt_ccl, cluster_axis=0, dim=3,
                    topology=mlp.args.ccl_topology(), memory_config=ttnn.DRAM_MEMORY_CONFIG)

            mlp.forward = forward

    def restore(self):
        for mlp, original in self.originals:
            mlp.forward = original
