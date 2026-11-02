"""The KV registry: what the plugin runner tells the connector about the allocated KV tensors.

Built by the plugin runner after the KV caches are allocated and handed to ``TTP2PConnector.register_kv_caches``.
Pure data: this module imports neither ttnn nor the extension (invariant I14).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TTKVRegistry:
    mesh_device: Any                                   # the root (4, 1) mesh device; dp = 1
    tensors: Tuple[Any, ...]                           # the unique KV tensors (K0, V0 .. K4, V4) by tensor index, deduplicated by identity
    layer_to_tensor_idx: Dict[str, int] = field(compare=False)   # from the runner's per-layer specs
    group_layers: Tuple[Tuple[str, ...], ...]          # the KV-cache groups in kv_cache_config order
    num_blocks: int                                    # device blocks per tensor
    pages_per_block: int                               # pages in one block row (row_bytes / page_bytes)
    row_bytes_per_chip: int                            # bytes of one block row of ONE tensor on ONE chip
    shard_bdf: Tuple[str, ...]                         # mesh coordinate d -> PCI address of that card

    @property
    def num_tensors(self) -> int:
        return len(self.tensors)

    @property
    def num_shards(self) -> int:
        return len(self.shard_bdf)

    @property
    def shard_object_bytes(self) -> int:
        """Bytes of one object on one chip: one row of every tensor."""
        return self.num_tensors * self.row_bytes_per_chip

    @property
    def object_bytes(self) -> int:
        """Bytes of one object over all chips (the unit stored on the drive)."""
        return self.num_shards * self.shard_object_bytes

    def problems(self, layout) -> list:
        """What disagrees with the configured layout (empty when the registry is what the spec expects)."""
        found = []
        if self.num_tensors != layout.tensors:
            found.append('tensors: registry has %d, layout %r wants %d' % (self.num_tensors, layout.name, layout.tensors))
        if self.num_shards != layout.chips:
            found.append('shards: registry has %d, layout %r wants %d' % (self.num_shards, layout.name, layout.chips))
        if self.row_bytes_per_chip != layout.row_bytes:
            found.append('row bytes per chip: registry has %d, layout %r wants %d'
                         % (self.row_bytes_per_chip, layout.name, layout.row_bytes))
        if self.pages_per_block != layout.pages_per_block:
            found.append('pages per block: registry has %d, layout %r wants %d'
                         % (self.pages_per_block, layout.name, layout.pages_per_block))
        if self.num_blocks < 2:
            found.append('num_blocks %d leaves no block besides the null block' % self.num_blocks)
        if self.object_bytes != layout.object_bytes:
            found.append('object bytes: registry has %d, layout %r wants %d' % (self.object_bytes, layout.name, layout.object_bytes))
        return found
