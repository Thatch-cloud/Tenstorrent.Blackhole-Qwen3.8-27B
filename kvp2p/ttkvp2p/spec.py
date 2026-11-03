"""The offloading spec: configuration, geometry, the NVMe load/store spec and the manager hand-out.

vLLM builds one spec in each connector role (``OffloadingConnector.__init__``), and under the ``uni`` executor both
roles live in the EngineCore process. The spec itself is therefore cheap and side-effect free; the process-wide
runtime is attached only when a role asks for its manager or its worker (invariant I10).

Importing this module touches neither ttnn nor the extension (invariant I14).
"""

import dataclasses
import importlib
import logging
import typing
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.config import OffloadingConfig

logger = logging.getLogger(__name__)

# The runtime states (the frozen interface's KVP2PRuntime.state).
STATE_UNAVAILABLE = 'UNAVAILABLE'
STATE_AVAILABLE = 'AVAILABLE'
STATE_LATCHED = 'LATCHED'
STATE_CLOSING = 'CLOSING'


class TierRefused(RuntimeError):
    """The tier will not start on this configuration; serving continues without it."""


# --------------------------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutGeometry:
    """One object = one block of one group = one row of every tensor on every chip."""

    name: str
    chips: int              # shards per object (the mesh size)
    tensors: int            # KV tensors (K0, V0 .. K4, V4)
    row_bytes: int          # bytes of one block row of ONE tensor on ONE chip
    pages_per_block: int

    @property
    def shard_object_bytes(self) -> int:
        return self.tensors * self.row_bytes

    @property
    def object_bytes(self) -> int:
        return self.chips * self.shard_object_bytes


LAYOUTS: Dict[str, LayoutGeometry] = {
    'gemma-chipmajor-v1': LayoutGeometry('gemma-chipmajor-v1', chips=4, tensors=10, row_bytes=69632, pages_per_block=64),
}


def register_layout(geometry: LayoutGeometry) -> None:
    """Add a layout (tests register a small one; production uses the table above)."""
    LAYOUTS[geometry.name] = geometry


# --------------------------------------------------------------------------------------------------------------------
# Configuration: the keys of kv_connector_extra_config (design 3.1). Attribute names equal the JSON keys.
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class P2PConfig:
    spec_name: str = 'TTP2PNvmeSpec'
    spec_module_path: str = 'ttkvp2p.spec'
    blocks_per_chunk: int = 1
    offload_prompt_only: bool = True
    layout: str = 'gemma-chipmajor-v1'
    persistence: str = 'process'
    retention_ack: bool = False
    nvme_device: str = '/dev/kv-drive'
    nvme_status_dir: str = '/run/kv-drive-status'
    p2pmem_glob: str = '/run/kv-p2pmem/*/allocate'
    p2p_window_id: int = 202
    staging_bytes_per_bank: int = 67108864
    spill_banks: Tuple[int, ...] = (0, 1)
    restore_banks: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    spill_subrun_objects: int = 8
    io_threads: int = 8
    io_queue_depth_objects: int = 8
    device_budget_objects_per_step: int = 96
    idle_wait_ms: float = 3.0
    load_deadline_base_s: float = 2.0
    load_deadline_gbps: float = 1.0
    stall_latch_s: float = 20.0
    strikes_to_latch: int = 3
    aer_latch: str = 'uncorrectable'
    ttl_s: int = 86400
    free_floor_pct: int = 5
    reserve_bytes: int = 8589934592
    store_threshold: int = 1
    min_store_tokens: int = 4096
    daily_write_cap_bytes: int = 329000000000
    require_sed: bool = True
    require_salt: bool = False
    persist_multimodal: bool = False
    control_dir: str = '/var/lib/kvp2p/ctl'
    state_file: str = '/var/lib/kvp2p/state.json'
    kv_epoch: int = 1
    zero_retention_marker: str = ''          # a node-level zero-retention marker file (node.env); '' = only the one beside the control directory
    # Additive knobs (not in the frozen kwarg; the defaults are the design's numbers).
    slot_objects: int = 128                  # cap on objects per staging slot (the digest op takes at most 128 blocks)
    shutdown_drain_s: float = 15.0           # section 2.9: the drain waits 15 s or less for in-flight bios
    store_wait_s: float = 10.0               # a store that finds no spill bank for this long is dropped (design 2.8: a later miss, never a stall)

    @classmethod
    def from_extra_config(cls, extra: Mapping) -> 'P2PConfig':
        """Build from ``kv_connector_extra_config``. Unknown keys are ignored; a malformed value raises ValueError."""
        hints = typing.get_type_hints(cls)
        values: Dict[str, Any] = {}
        for item in dataclasses.fields(cls):
            if item.name not in extra:
                continue
            raw = extra[item.name]
            kind = hints[item.name]
            try:
                if kind is bool:
                    if not isinstance(raw, bool):
                        raise ValueError('expected a boolean')
                    values[item.name] = raw
                elif kind is int:
                    if isinstance(raw, bool) or int(raw) != raw:
                        raise ValueError('expected an integer')
                    values[item.name] = int(raw)
                elif kind is float:
                    if isinstance(raw, bool):
                        raise ValueError('expected a number')
                    values[item.name] = float(raw)
                elif kind is str:
                    if not isinstance(raw, str):
                        raise ValueError('expected a string')
                    values[item.name] = raw
                else:    # Tuple[int, ...]
                    values[item.name] = tuple(int(v) for v in raw)
            except (TypeError, ValueError) as error:
                raise ValueError('kv_connector_extra_config[%r] is malformed: %s' % (item.name, error))
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        problems = []
        if self.blocks_per_chunk != 1:
            problems.append('blocks_per_chunk must be 1 (one object is one block of one group)')
        if not self.spill_banks or not self.restore_banks:
            problems.append('spill_banks and restore_banks must both be non-empty')
        if set(self.spill_banks) & set(self.restore_banks):
            problems.append('spill_banks and restore_banks must be disjoint (a late restore DMA must never reach a spill slot)')
        if len(set(self.spill_banks)) != len(self.spill_banks) or len(set(self.restore_banks)) != len(self.restore_banks):
            problems.append('a bank is listed twice')
        if any(bank < 0 for bank in tuple(self.spill_banks) + tuple(self.restore_banks)):
            problems.append('bank numbers are not negative')
        if self.spill_subrun_objects < 1:
            problems.append('spill_subrun_objects must be at least 1')
        if self.device_budget_objects_per_step < 1:
            problems.append('device_budget_objects_per_step must be at least 1')
        if not (1 <= self.slot_objects <= 128):
            problems.append('slot_objects must be 1..128')
        if self.load_deadline_gbps <= 0 or self.load_deadline_base_s < 0:
            problems.append('the load deadline needs base >= 0 and gbps > 0')
        if self.strikes_to_latch < 1:
            problems.append('strikes_to_latch must be at least 1')
        if self.stall_latch_s <= 0:
            problems.append('stall_latch_s must be positive')
        if self.store_wait_s <= 0:
            problems.append('store_wait_s must be positive')
        if problems:
            raise ValueError('; '.join(problems))


# --------------------------------------------------------------------------------------------------------------------
# Per-request tag (design 3.2): set by the scheduler role, read by the manager
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class ReqTag:
    tenant: Optional[bytes] = None       # blake2b-8 of the request's cache salt, or None
    has_mm: bool = False                 # the request carries multimodal features (never stored: D9)
    admitted: Optional[bool] = None      # the I12 verdict, decided once at the request's first store
    no_retry: bool = False               # a deadline or stall failed this request's load: its recompute must not look up again (I13)
    num_prompt_tokens: Optional[int] = None   # the request's prompt length: the manager sizes the WHOLE boundary from it (I12); None = unknown
    hit_tokens: int = 0                  # tokens already on the device or restored from this tier: a later turn stores only what lies beyond them


# --------------------------------------------------------------------------------------------------------------------
# The NVMe load/store spec
# --------------------------------------------------------------------------------------------------------------------

@dataclass(eq=False)
class NvmeLoadStoreSpec(LoadStoreSpec):
    slot_ids: List[int]                              # 1:1 with the keys and with GPULoadStoreSpec's flat block order (I6)
    generations: List[int]                           # slot generation at prepare time (I5)
    expected: Optional[np.ndarray] = None            # loads only: [n, 4] uint64 digests copied at prepare_load (I4)
    poison: Optional[str] = None                     # additive: set by the scheduler role when the job's pairing is wrong (I6); the worker fails it
    medium: ClassVar[Medium] = Medium.STORAGE


# --------------------------------------------------------------------------------------------------------------------
# A manager for a tier that is off: always misses, drops every store
# --------------------------------------------------------------------------------------------------------------------

class RefusedManager(OffloadingManager):
    """Used when the tier refuses to start (wrong executor, wrong layout, runtime missing): serving continues and nothing is offloaded."""

    def __init__(self, reason: str):
        self.reason = reason

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return LookupResult.MISS

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        raise AssertionError('a refused tier never answers a hit, so it never prepares a load')

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext) -> Optional[PrepareStoreOutput]:
        # An EMPTY keys_to_store is a drop. None would be a warning plus a retry on every step.
        return PrepareStoreOutput(keys_to_store=[], store_spec=NvmeLoadStoreSpec([], []), evicted_keys=[])

    def complete_store(self, keys, req_context, success: bool = True, digests=None) -> None:
        return None         # no store job ever exists for a refused tier; the signature is the real manager's (digests is the additive keyword)


# --------------------------------------------------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------------------------------------------------

class TTP2PNvmeSpec(OffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        self.refusal: Optional[str] = None
        self.p2p_config: Optional[P2PConfig] = None
        self.layout_geometry: Optional[LayoutGeometry] = None
        self.runtime: Optional[Any] = None          # the process-wide runtime the scheduler role attached to (get_manager)
        try:
            self.p2p_config = P2PConfig.from_extra_config(config.extra_config)
            self.layout_geometry = LAYOUTS.get(self.p2p_config.layout)
            if self.layout_geometry is None:
                raise ValueError('unknown layout %r' % (self.p2p_config.layout,))
            if self.blocks_per_chunk != 1:
                raise ValueError('the offloading config chunks %d blocks (the tier stores one block per object)' % self.blocks_per_chunk)
            if config.parallel.world_size != 1:
                raise ValueError('world_size %d (the tier needs exactly one worker)' % config.parallel.world_size)
        except ValueError as error:
            self.refuse(str(error))

    def refuse(self, reason: str) -> None:
        """Turn the tier off for this process: the manager always misses and no worker is built."""
        if self.refusal is None:
            logger.warning('ttkvp2p: the tier refuses to start: %s', reason)
            self.refusal = reason

    @property
    def object_bytes(self) -> int:
        """Bytes of one object on the drive: from the TT geometry, NOT worker_kv_bytes_per_block (which is the GPU-style page size)."""
        if self.layout_geometry is None:
            raise TierRefused(self.refusal or 'no layout')
        return self.layout_geometry.object_bytes

    def get_manager(self) -> OffloadingManager:
        if self.refusal is not None:
            return RefusedManager(self.refusal)
        runtime = None
        try:
            runtime_module = importlib.import_module('ttkvp2p.runtime')
            manager_module = importlib.import_module('ttkvp2p.manager')
            runtime = runtime_module.KVP2PRuntime.acquire(self.p2p_config, 'scheduler')
            manager = manager_module.NvmeKVManager(self.p2p_config, runtime)
        except Exception as error:     # a broken tier must not stop the engine: it stays a no-op that always misses
            logger.error('ttkvp2p: the scheduler role cannot attach to the runtime (%s: %s); the tier stays off',
                         type(error).__name__, error, exc_info=True)
            if runtime is not None:
                # The reference taken above must not leak: the runtime is reference counted, and a leaked 'scheduler' reference
                # would keep it (and the drive claim) open for the life of the process.
                try:
                    runtime.release('scheduler')
                except Exception:
                    logger.exception('ttkvp2p: releasing the runtime after a failed manager build failed')
            self.refusal = 'scheduler role cannot attach: %s' % (error,)
            return RefusedManager(self.refusal)
        self.runtime = runtime
        return manager

    # --- what the connector's scheduler role needs from the runtime (it settles each store job itself; see connector.py)
    def take_store_result(self, job_id: int):
        """Remove and return the worker's ``StoreResult`` for a store job, or None (the worker pruned it or never ran the job)."""
        results = getattr(self.runtime, 'store_results', None)
        if results is None:
            return None
        return results.pop(job_id, None)

    def latch_tier(self, reason: str) -> None:
        """Latch the tier from the scheduler role (the manager misbehaved). Never raises."""
        if self.runtime is None:
            return
        try:
            self.runtime.latch(reason)
        except Exception:
            logger.exception('ttkvp2p: runtime.latch failed')

    def get_worker(self, kv_caches) -> OffloadingWorker:
        raise RuntimeError('TT uses get_tt_worker')

    def get_tt_worker(self, registry) -> 'OffloadingWorker':
        """Build the worker for the registered TT KV tensors. Raises TierRefused when the tier cannot start."""
        if self.refusal is not None:
            raise TierRefused(self.refusal)
        problems = registry.problems(self.layout_geometry)
        if problems:
            raise TierRefused('the registered KV tensors do not match the layout: ' + '; '.join(problems))
        worker_module = importlib.import_module('ttkvp2p.worker')
        return worker_module.build_worker(self, registry)
