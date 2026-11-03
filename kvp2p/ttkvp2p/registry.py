"""The KV registry: what the plugin runner tells the connector about the allocated KV tensors.

Built by the plugin runner after the KV caches are allocated and handed to ``TTP2PConnector.register_kv_caches``.
Pure data: importing this module imports neither ttnn nor the extension (invariant I14).

The shard map (``shard_bdf``: mesh coordinate d -> the PCI address of the card behind it) is read from the extension's
``qwen_p2p_mesh_devices(mesh)``, an op that allocates nothing on a card, so the start-up order of design 2.9 keeps its zero device footprint
until the host checks have passed (amendment A3 of INTERFACE.md). ``resolve_shard_bdf`` is the one place that call is made; the plugin's
``build_registry`` calls it, and ``TTP2PNvmeSpec.get_tt_worker`` calls it for a registry that arrives without a map.
"""

import dataclasses
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


class ShardMapError(ValueError):
    """The mesh -> card map cannot be built or cannot be trusted (an older extension, a card the host cannot name, a duplicate, a disagreement with sysfs).
    The tier stays off and nothing has been allocated on a card."""


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


# --------------------------------------------------------------------------------------------------------------------
# The shard map (amendment A3)
# --------------------------------------------------------------------------------------------------------------------

def shard_bdf_from_devices(devices: Sequence[Any]) -> Tuple[str, ...]:
    """The shard map from the device records of ``qwen_p2p_mesh_devices`` (or ``qwen_p2p_staging_info()['devices']``, the very same records).

    One lower case PCI address per mesh coordinate, in mesh order. Refused (``ShardMapError``) when the list is empty or out of mesh order, when a
    card has no PCI address or no tt-kmd device number (the host must not guess), when two coordinates name the same card or the same device number,
    or when sysfs says the cluster descriptor's address is not that device's (``pci_bus_id_agrees`` False)."""
    records = list(devices or ())
    if not records:
        raise ShardMapError('the mesh reports no device')
    found = []
    for position, record in enumerate(records):
        try:
            bdf, number, index, agrees = record['pci_bus_id'], record['pcie_device'], record.get('mesh_index', position), record.get('pci_bus_id_agrees')
        except (TypeError, KeyError, AttributeError) as error:
            raise ShardMapError('device record %d is not a device record (%s: %s)' % (position, type(error).__name__, error))
        if index != position:
            raise ShardMapError('device record %d is mesh index %r: the records are not in mesh order' % (position, index))
        if not isinstance(bdf, str) or not bdf.strip():
            raise ShardMapError('the card of mesh device %d has no PCI address (the host cannot name it, so it will not guess)' % position)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ShardMapError('the card of mesh device %d has no tt-kmd device number (pcie_device=%r)' % (position, number))
        if agrees is False:
            raise ShardMapError('the PCI address of mesh device %d (%s, from the cluster descriptor) is not the one sysfs gives for tt-kmd device %d (%s)'
                                % (position, bdf, number, record.get('pci_bus_id_sysfs')))
        found.append((bdf.strip().lower(), number))
    if len({bdf for bdf, _ in found}) != len(found) or len({number for _, number in found}) != len(found):
        raise ShardMapError('two mesh devices name the same card: %s' % ([bdf for bdf, _ in found],))
    return tuple(bdf for bdf, _ in found)


def resolve_shard_bdf(mesh_device: Any, ops: Optional[Any] = None) -> Tuple[str, ...]:
    """The mesh's shard map, from the extension. Allocates nothing on a card and enqueues nothing. Raises ``ShardMapError``.

    ``ops`` is the extension module (``qwen_kv_read``); it is imported here, never at module import (I14). A build of the extension that predates
    ``qwen_p2p_mesh_devices`` is refused with that said, so the tier is off at boot instead of failing on the first allocation."""
    try:
        module = ops if ops is not None else importlib.import_module('qwen_kv_read')
    except Exception as error:
        raise ShardMapError('the qwen_kv_read extension cannot be imported (%s: %s)' % (type(error).__name__, error))
    call = getattr(module, 'qwen_p2p_mesh_devices', None)
    if not callable(call):
        raise ShardMapError('this qwen_kv_read build has no qwen_p2p_mesh_devices (it predates the registry fix; the overlay must carry the new build)')
    try:
        devices = call(mesh_device)
    except Exception as error:
        raise ShardMapError('qwen_p2p_mesh_devices failed (%s: %s)' % (type(error).__name__, error))
    return shard_bdf_from_devices(devices)


def with_shard_bdf(registry: TTKVRegistry, ops: Optional[Any] = None) -> TTKVRegistry:
    """``registry`` with its shard map filled. A registry that already carries a map is returned as it is (the plugin's ``build_registry`` fills it;
    this is for a registry that arrives without one). Raises ``ShardMapError``."""
    if registry.shard_bdf:
        return registry
    return dataclasses.replace(registry, shard_bdf=resolve_shard_bdf(registry.mesh_device, ops))
