"""Static-fixture full-model batching; no installed or class-global patches."""

from contextlib import contextmanager

from attention_batch import OrderedCacheWriter, SerialAttentionReader, SerialCacheWriter, serial_tail
from gdn_prefix import decode_projected, gated_decode, prepare_token_rows, validate_reused_input


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


def compact_gdn_enabled(rows, requested, serial_sdpa, profiler):
    validate_checkpoint(rows, rows)
    if requested and (not serial_sdpa or profiler is not None):
        raise ValueError("Compact GDN requires the unprofiled B1-SDPA correctness path")
    return bool(requested and rows > 1)


def device_loop_enabled(rows, requested, compact_gdn, hoist_row_layout, compact_prologue=False, packed_checkpoints=False):
    validate_checkpoint(rows, rows)
    if requested and not (compact_gdn and hoist_row_layout):
        raise ValueError('Device loop requires the exact compact row-layout control')
    return bool(requested and rows >= (8 if compact_prologue and not packed_checkpoints else 2))


class ModelBatch:
    def __init__(self, model, tokens, start, pages, helpers, checkpoints, prefix, serial_sdpa=False, profiler=None,
                 compact_gdn=False, reuse_gdn_input=False, skip_row_clones=False, hoist_row_layout=False,
                 device_loop_gdn=False, compact_prologue=False, batch_conv=False, packed_checkpoints=False,
                 retain_records=False, ordered_cache=False):
        import torch
        import ttnn
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        self.rows = len(tokens)
        validate_checkpoint(self.rows, prefix)
        if ordered_cache and (not serial_sdpa or profiler is not None):
            raise ValueError('Ordered cache requires the unprofiled exact B1 SDPA path')
        self.ordered_cache = ordered_cache and self.rows > 1
        cache_kernels = None
        if self.ordered_cache:
            import os
            from ordered_cache import load_kernels
            cache_kernels = load_kernels(os.environ['TT_METAL_HOME'])
        self.compact_gdn = compact_gdn_enabled(self.rows, compact_gdn, serial_sdpa, profiler)
        if reuse_gdn_input and not compact_gdn:
            raise ValueError("Input reuse requires the exact compact GDN control")
        self.reuse_gdn_input = reuse_gdn_input and self.rows > 1
        if skip_row_clones and not reuse_gdn_input:
            raise ValueError("Clone removal requires the exact reused-input control")
        self.skip_row_clones = skip_row_clones and self.rows > 1
        if hoist_row_layout and not skip_row_clones:
            raise ValueError("Layout hoisting requires the selective-clone control")
        self.hoist_row_layout = hoist_row_layout and self.rows > 1
        self.device_loop_gdn = device_loop_enabled(self.rows, device_loop_gdn, compact_gdn, hoist_row_layout,
                                                   compact_prologue, packed_checkpoints)
        if compact_prologue and not device_loop_gdn:
            raise ValueError('Compact prologue requires device-loop GDN')
        self.compact_prologue = compact_prologue and self.device_loop_gdn
        if batch_conv and not compact_prologue:
            raise ValueError('Batched convolution requires compact-prologue control')
        self.batch_conv = batch_conv and self.device_loop_gdn
        if packed_checkpoints and not batch_conv:
            raise ValueError('Packed checkpoints require batched convolution')
        self.packed_checkpoints = packed_checkpoints and self.device_loop_gdn
        if retain_records and not self.packed_checkpoints:
            raise ValueError('Retained records require active packed checkpoints')
        from gdn_records import RetainedGDNBlock
        self.retained = RetainedGDNBlock(self.rows, ttnn) if retain_records else None
        self.working_states = []
        if len(helpers) != 48 or len(checkpoints) != 48 or len(model.layers) != 64:
            raise ValueError("Expected all 64 model layers and 48 GDN checkpoints")
        self.model = model
        self.operations = ttnn
        self.prefix = prefix
        self.profiler = profiler
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
                if self.ordered_cache:
                    writer = OrderedCacheWriter(model.mesh_device, ttnn, cache_kernels)
                self.writers.append(writer)
                reader = SerialAttentionReader(ttnn, singleton_positions, [singleton_pages] * self.rows) if serial_sdpa else None
                if reader is not None:
                    self.readers.append(reader)
                write = profiler.wrap("attention.kv_write", writer) if profiler else writer
                read = profiler.wrap("attention.sdpa_and_row_packing", reader) if profiler and reader else reader
                self.bindings.append((attention, "_decode_from_prep", serial_tail(attention, write, ttnn, read)))
                if profiler:
                    for name, category in (("forward_decode", "attention.block"),
                                           ("_qkv_raw_decode", "attention.input_projection"),
                                           ("_wo_proj", "attention.output_projection")):
                        self.bindings.append((attention, name, profiler.wrap(category, getattr(attention, name))))
            else:
                if helpers[gdn_index].gdn is not attention:
                    raise ValueError("GDN checkpoint layer order mismatch")
                forward = self.gdn_forward(attention, helpers[gdn_index], checkpoints[gdn_index])
                if profiler:
                    forward = profiler.wrap("gdn.block", forward)
                    for name, category in (("_project_qkvzab_raw", "gdn.input_projection"),
                                           ("_row_proj", "gdn.output_projection"),
                                           ("_slice_along", "gdn.active_state_slice"),
                                           ("_write_recurrent_state_prefix", "gdn.active_state_write")):
                        self.bindings.append((attention, name, profiler.wrap(category, getattr(attention, name))))
                self.bindings.append((attention, "forward_decode", forward))
                gdn_index += 1
        if profiler:
            from stage_profile import decoder_bindings
            for index, layer in enumerate(model.layers):
                self.bindings.extend(decoder_bindings(layer, index, profiler))
            for name, category in (("embd", "embedding"), ("_final_norm_decode", "final_norm"), ("_lm_head", "lm_head")):
                self.bindings.append((model, name, profiler.wrap(category, getattr(model, name))))

    def gdn_forward(self, layer, helper, checkpoint):
        operations = self.operations
        if self.device_loop_gdn:
            from pathlib import Path
            from gdn_device_loop_state import DeviceLoopState
            from gdn_multitoken import load_kernels
            from gdn_multitoken_conv import finish_output, release_owned
            state = DeviceLoopState(helper, operations, load_kernels(Path('/opt/tt-metal'), True),
                                    self.compact_prologue, self.batch_conv, self.batch_conv, self.packed_checkpoints)
            self.working_states.append(state)

            def device_forward(value):
                from models.tt_transformers.tt.ccl import tt_all_reduce
                if tuple(value.shape) != (1, 1, self.rows, 5120):
                    raise ValueError('Unexpected full-model GDN input geometry')
                packed = operations.reshape(value, (1, self.rows, 5120))
                result = state.decode(packed, checkpoint, self.prefix)
                finish_output(layer, result, operations, tt_all_reduce)
                output = result['layer_output']
                if self.retained is not None:
                    result['owned'] = [value for value in result['owned'] if value is not output]
                    self.retained.append(state, result, checkpoint)
                else:
                    release_owned(operations, [value for value in result['owned'] if value is not output])
                self.gdn_calls += 1
                return output

            return device_forward
        if self.reuse_gdn_input:
            import inspect
            validate_reused_input(inspect.getsource(type(layer).forward_decode))
        native_gated = gated_decode(layer, profiler=self.profiler)
        snapshot = helper.save
        working = None
        if self.compact_gdn:
            from gdn_working_state import WorkingState
            working = WorkingState(helper, operations, compact_dma=True, skip_row_clones=self.skip_row_clones,
                                   hoist_row_layout=self.hoist_row_layout)
            self.working_states.append(working)
            snapshot = working.save
        if self.profiler:
            native_gated = self.profiler.wrap("gdn.native_row", native_gated)
            snapshot = self.profiler.wrap("gdn.checkpoint", snapshot)

        def forward(value):
            from models.tt_transformers.tt.ccl import tt_all_reduce

            if tuple(value.shape) != (1, 1, self.rows, 5120):
                raise ValueError("Unexpected full-model GDN input geometry")
            packed = operations.reshape(value, (1, self.rows, 5120))
            tokens, owned_tokens = prepare_token_rows(operations, packed, reuse=self.reuse_gdn_input)

            def save(prefix):
                if prefix == self.prefix:
                    snapshot(checkpoint)

            if working:
                outputs = working.decode(packed, tokens, save)
            else:
                save(0)
                outputs = decode_projected(layer, packed, tokens, save, operations, forward=native_gated,
                                           profiler=self.profiler)
            gated = outputs[0] if self.rows == 1 else operations.concat(outputs, dim=1)
            partial = layer._row_proj(gated, layer.tw["out"])
            if self.rows != 1:
                operations.deallocate(gated)
            for tensor in outputs + owned_tokens:
                operations.deallocate(tensor)
            partial = operations.reshape(partial, (1, 1, self.rows, partial.shape[-1]))
            result = tt_all_reduce(partial, self.model.mesh_device, layer.tt_ccl, cluster_axis=0, dim=3,
                                   topology=self.model.args.ccl_topology(), memory_config=operations.DRAM_MEMORY_CONFIG)
            self.gdn_calls += 1
            return result

        return forward

    def run(self):
        if self.retained is not None and (self.retained.closed or self.retained.records):
            raise ValueError('A retained fixture owns exactly one captured or eager block')
        before_gdn = self.gdn_calls
        before_compact = [(state.calls, state.checkpoint_calls) for state in self.working_states]
        before_clones = [state.skipped_clones for state in self.working_states]
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
        if len(self.working_states) != (48 if self.compact_gdn else 0) or any(
            state.calls - before[0] != (1 if self.device_loop_gdn else self.rows) or state.checkpoint_calls - before[1] != 1
            for state, before in zip(self.working_states, before_compact, strict=True)
        ):
            raise AssertionError("Every compact GDN layer must update in place and checkpoint exactly once")
        if any(state.skipped_clones - before != (self.rows - 1 if self.skip_row_clones and not self.device_loop_gdn else 0)
               for state, before in zip(self.working_states, before_clones, strict=True)):
            raise AssertionError("Projected-row clone removal did not engage exactly")
        return result

    def close(self):
        if self.retained is not None:
            self.retained.close()
        for state in self.working_states:
            state.close()
        self.working_states.clear()
        for value in self.buffers:
            self.operations.deallocate(value)
        self.buffers.clear()
