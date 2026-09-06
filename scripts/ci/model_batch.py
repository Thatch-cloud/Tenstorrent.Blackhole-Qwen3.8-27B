"""Static-fixture full-model batching; no installed or class-global patches."""

from contextlib import contextmanager

from attention_batch import SerialAttentionReader, SerialCacheWriter, serial_tail
from gdn_prefix import decode_projected, gated_decode


@contextmanager
def instance_overrides(bindings):
    previous = []
    try:
        for instance, name, value in bindings:
            previous.append((instance, name, name in instance.__dict__, instance.__dict__.get(name)))
            setattr(instance, name, value)
        yield
    finally:
        for instance, name, existed, value in reversed(previous):
            if existed:
                setattr(instance, name, value)
            else:
                delattr(instance, name)


def validate_checkpoint(rows, prefix):
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16):
        raise ValueError("Expected T=1/2/4/8/16")
    if type(prefix) is not int or not 0 <= prefix <= rows:
        raise ValueError("Checkpoint must be in [0, T]")


class ModelBatch:
    def __init__(self, model, tokens, start, pages, helpers, checkpoints, prefix, serial_sdpa=False):
        import torch
        import ttnn
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        self.rows = len(tokens)
        validate_checkpoint(self.rows, prefix)
        if len(helpers) != 48 or len(checkpoints) != 48 or len(model.layers) != 64:
            raise ValueError("Expected all 64 model layers and 48 GDN checkpoints")
        self.model = model
        self.operations = ttnn
        self.prefix = prefix
        self.buffers = []
        self.bindings = []
        self.writers = []
        self.readers = []
        self.gdn_calls = 0

        def upload(value, dtype):
            result = ttnn.from_torch(value, device=model.mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device))
            self.buffers.append(result)
            return result

        positions = torch.arange(start, start + self.rows, dtype=torch.int32)
        self.tokens = upload(torch.tensor(tokens, dtype=torch.int32).reshape(self.rows, 1), ttnn.uint32)
        self.positions = upload(positions, ttnn.int32)
        self.pages = upload(pages.repeat(self.rows, 1), ttnn.int32)
        singleton_pages = upload(pages, ttnn.int32)
        singleton_positions = [upload(position.reshape(1), ttnn.int32) for position in positions]
        self.cos, self.sin = rot_mats_decode(model.mesh_device, model.args.rope_head_dim,
                                            model.args.max_seq_len, model.args.rope_theta, positions)
        self.buffers.extend([self.cos, self.sin])
        gdn_index = 0
        for layer in model.layers:
            attention = layer.attention
            if layer.is_full_attention:
                writer = SerialCacheWriter(ttnn, singleton_positions, [singleton_pages] * self.rows,
                                           attention._kv_shard_cfg(1))
                self.writers.append(writer)
                reader = SerialAttentionReader(ttnn, singleton_positions, [singleton_pages] * self.rows) if serial_sdpa else None
                if reader is not None:
                    self.readers.append(reader)
                self.bindings.append((attention, "_decode_from_prep", serial_tail(attention, writer, ttnn, reader)))
            else:
                if helpers[gdn_index].gdn is not attention:
                    raise ValueError("GDN checkpoint layer order mismatch")
                forward = self.gdn_forward(attention, helpers[gdn_index], checkpoints[gdn_index])
                self.bindings.append((attention, "forward_decode", forward))
                gdn_index += 1

    def gdn_forward(self, layer, helper, checkpoint):
        operations = self.operations
        native_gated = gated_decode(layer)

        def forward(value):
            from models.tt_transformers.tt.ccl import tt_all_reduce

            if tuple(value.shape) != (1, 1, self.rows, 5120):
                raise ValueError("Unexpected full-model GDN input geometry")
            packed = operations.reshape(value, (1, self.rows, 5120))
            tokens = [operations.slice(packed, (0, index, 0), (1, index + 1, 5120),
                                       memory_config=operations.L1_MEMORY_CONFIG) for index in range(self.rows)]

            def save(prefix):
                if prefix == self.prefix:
                    helper.save(checkpoint)

            save(0)
            outputs = decode_projected(layer, packed, tokens, save, operations, forward=native_gated)
            gated = outputs[0] if self.rows == 1 else operations.concat(outputs, dim=1)
            partial = layer._row_proj(gated, layer.tw["out"])
            if self.rows != 1:
                operations.deallocate(gated)
            for tensor in outputs + tokens:
                operations.deallocate(tensor)
            partial = operations.reshape(partial, (1, 1, self.rows, partial.shape[-1]))
            result = tt_all_reduce(partial, self.model.mesh_device, layer.tt_ccl, cluster_axis=0, dim=3,
                                   topology=self.model.args.ccl_topology(), memory_config=operations.DRAM_MEMORY_CONFIG)
            self.gdn_calls += 1
            return result

        return forward

    def run(self):
        before_gdn = self.gdn_calls
        before_writes = [writer.calls for writer in self.writers]
        before_reads = [reader.calls for reader in self.readers]
        with instance_overrides(self.bindings):
            result = self.model._forward_decode(self.tokens, self.cos, self.sin, self.positions, self.pages)
        if self.gdn_calls - before_gdn != 48 or any(
            writer.calls - before != 2 for writer, before in zip(self.writers, before_writes, strict=True)
        ):
            raise AssertionError("All 48 GDN and 16 attention adapters must engage")
        if any(reader.calls - before != 1 for reader, before in zip(self.readers, before_reads, strict=True)):
            raise AssertionError("Every selected B1 SDPA adapter must engage")
        return result

    def close(self):
        for value in self.buffers:
            self.operations.deallocate(value)
        self.buffers.clear()
