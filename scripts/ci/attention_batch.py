"""Fixture-only batched attention with ordered B1 paged-cache writes."""

from types import FunctionType, MethodType


def capture_operation(operations, mesh, operation):
    trace = operations.begin_trace_capture(mesh, cq_id=0)
    ended = False
    try:
        try:
            result = operation()
        finally:
            operations.end_trace_capture(mesh, trace, cq_id=0)
            ended = True
    except BaseException:
        if ended:
            operations.release_trace(mesh, trace)
        raise
    return trace, result


class Overlay:
    def __init__(self, original, **overrides):
        self.original = original
        self.overrides = overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


class SerialCacheWriter:
    def __init__(self, operations, singleton_positions, singleton_pages, shard_config):
        if len(singleton_positions) not in (1, 2, 4, 8, 16, 32) or len(singleton_pages) != len(singleton_positions):
            raise ValueError("Expected T=1/2/4/8/16/32 singleton position/page pairs")
        self.operations = operations
        self.positions = singleton_positions
        self.pages = singleton_pages
        self.shard_config = shard_config
        self.calls = 0

    def __call__(self, cache, packed, *, update_idxs_tensor, page_table):
        operations = self.operations
        shape = tuple(packed.shape)
        if shape != (1, len(self.positions), 32, 256):
            raise ValueError(f"Unexpected native prepared KV shape: {shape}")
        interleaved = operations.to_memory_config(packed, operations.DRAM_MEMORY_CONFIG)
        try:
            for index, (position, pages) in enumerate(zip(self.positions, self.pages, strict=True)):
                row = operations.slice(interleaved, (0, index, 0, 0), (1, index + 1, 32, 256),
                                       memory_config=operations.DRAM_MEMORY_CONFIG)
                sharded = operations.to_memory_config(row, self.shard_config)
                operations.experimental.paged_update_cache(cache, sharded, update_idxs_tensor=position, page_table=pages)
                operations.deallocate(sharded)
                operations.deallocate(row)
        finally:
            operations.deallocate(interleaved)
        self.calls += 1


class OrderedCacheWriter:
    def __init__(self, mesh, operations, kernels):
        self.mesh = mesh
        self.operations = operations
        self.kernels = kernels
        self.calls = 0

    def __call__(self, cache, packed, *, update_idxs_tensor, page_table):
        from ordered_cache import update, validate_shapes

        validate_shapes(tuple(cache.shape), tuple(packed.shape), tuple(update_idxs_tensor.shape), tuple(page_table.shape))
        operations = self.operations
        converted = packed.memory_config() != operations.DRAM_MEMORY_CONFIG
        interleaved = operations.to_memory_config(packed, operations.DRAM_MEMORY_CONFIG) if converted else packed
        try:
            update(self.mesh, cache, interleaved, update_idxs_tensor, page_table, self.kernels)
        finally:
            if converted:
                operations.deallocate(interleaved)
        self.calls += 1


class SerialAttentionReader:
    def __init__(self, operations, singleton_positions, singleton_pages):
        if len(singleton_positions) not in (1, 2, 4, 8, 16, 32) or len(singleton_pages) != len(singleton_positions):
            raise ValueError("Expected paired singleton attention metadata")
        self.operations = operations
        self.positions = singleton_positions
        self.pages = singleton_pages
        self.calls = 0

    def __call__(self, query, keys, values, *, page_table_tensor, cur_pos_tensor, **kwargs):
        operations = self.operations
        shape = tuple(query.shape)
        if len(shape) != 4 or shape[:2] != (1, len(self.positions)) or not 0 < shape[2] <= 32 or shape[3] != 256:
            raise ValueError(f"Unexpected native query geometry: {shape}")
        outputs = []
        for index, (position, pages) in enumerate(zip(self.positions, self.pages, strict=True)):
            row = operations.slice(query, (0, index, 0, 0), (1, index + 1, shape[2], 256),
                                   memory_config=operations.DRAM_MEMORY_CONFIG)
            outputs.append(operations.transformer.paged_scaled_dot_product_attention_decode(
                row, keys, values, page_table_tensor=pages, cur_pos_tensor=position, **kwargs))
            operations.deallocate(row)
        self.calls += 1
        if len(outputs) == 1:
            return outputs[0]
        result = operations.concat(outputs, dim=1, memory_config=kwargs["memory_config"])
        for output in outputs:
            operations.deallocate(output)
        return result


def serial_tail(attention, writer, operations, reader=None):
    native = type(attention)._decode_from_prep
    namespace = dict(native.__globals__)
    overrides = dict(experimental=Overlay(operations.experimental, paged_update_cache=writer))
    if reader is not None:
        overrides["transformer"] = Overlay(operations.transformer, paged_scaled_dot_product_attention_decode=reader)
    namespace["ttnn"] = Overlay(operations, **overrides)
    isolated = FunctionType(native.__code__, namespace, native.__name__, native.__defaults__, native.__closure__)
    isolated.__kwdefaults__ = native.__kwdefaults__
    return MethodType(isolated, attention)
