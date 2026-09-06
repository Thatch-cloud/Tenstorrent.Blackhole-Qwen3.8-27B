"""Fixture-only batched attention with ordered B1 paged-cache writes."""

from types import FunctionType, MethodType


class Overlay:
    def __init__(self, original, **overrides):
        self.original = original
        self.overrides = overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


class SerialCacheWriter:
    def __init__(self, operations, singleton_positions, singleton_pages, shard_config):
        if len(singleton_positions) not in (1, 2, 4, 8, 16) or len(singleton_pages) != len(singleton_positions):
            raise ValueError("Expected T=1/2/4/8/16 singleton position/page pairs")
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


def serial_tail(attention, writer, operations):
    native = type(attention)._decode_from_prep
    namespace = dict(native.__globals__)
    namespace["ttnn"] = Overlay(operations, experimental=Overlay(operations.experimental, paged_update_cache=writer))
    isolated = FunctionType(native.__code__, namespace, native.__name__, native.__defaults__, native.__closure__)
    isolated.__kwdefaults__ = native.__kwdefaults__
    return MethodType(isolated, attention)
