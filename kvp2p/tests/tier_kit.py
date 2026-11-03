"""Test kit for the WP3a modules (store, health, runtime, manager, claim): a temporary tree with a CLAIMED file-backed drive, a fresh
cache-drive status file, fake p2pmem files and sysfs, a controllable clock, and a ``HostEnv`` that points the runtime at all of it.

The drive is a regular file (the CPU tests' stand-in for the raw namespace; it is only allowed under ``THATCH_KVP2P_TEST=1``, which the
kit's environment sets), claimed by the real claim tool, so the whole open path runs: host checks, exclusive open, superblock verify, claim
digest, the open-time discard (a punched hole), the TRIM thread (run on the caller's thread when ``sync_io``).
"""

import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import fakes  # noqa: F401  (prepares sys.path and the no-GPU probe before vLLM is imported)
from fakes import TINY

from vllm.v1.kv_offload.base import ReqContext, make_offload_key  # noqa: E402

from ttkvp2p import claim, health, runtime as runtime_module, store  # noqa: E402
from ttkvp2p.manager import NvmeKVManager  # noqa: E402
from ttkvp2p.runtime import KVP2PRuntime  # noqa: E402
from ttkvp2p.spec import P2PConfig, ReqTag, register_layout  # noqa: E402

BY_ID = 'test-drive-by-id'
OBJ = TINY.object_bytes                      # 5,120 B: a tiny layout (not 4 KiB aligned: allowed on a regular file only)
CHIPS = TINY.chips
T0 = 1_700_000_000.0                         # the kit's clock starts here (wall and monotonic alike)


class Clock:
    """One controllable clock for wall time, monotonic time and the manager's TTL."""

    def __init__(self, now: float = T0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def key_for(n: int, group: int = 0) -> bytes:
    return make_offload_key(n.to_bytes(32, 'big'), group)


def req(req_id: str, *, tenant: Optional[bytes] = None, has_mm: bool = False, prompt_tokens: Optional[int] = None,
        no_retry: bool = False, tag: bool = True) -> ReqContext:
    """A request context carrying the connector scheduler's ``ReqTag`` (``num_prompt_tokens`` is the attribute the manager reads for the
    whole-boundary admission size; the WP3a report asks WP4 to set it)."""
    ctx = ReqContext(req_id=req_id)
    if tag:
        reqtag = ReqTag(tenant=tenant, has_mm=has_mm, admitted=None, no_retry=no_retry)
        if prompt_tokens is not None:
            reqtag.num_prompt_tokens = prompt_tokens
        ctx.set_state(reqtag)
    return ctx


class Tree:
    """The files of one test: the drive, the status directory, the control directory, the p2pmem files, a fake sysfs."""

    def __init__(self, root: str, num_slots: int, object_bytes: int, layout_id: str, by_id: str):
        self.root = root
        self.num_slots = num_slots
        self.object_bytes = object_bytes
        self.layout_id = layout_id
        self.by_id = by_id
        self.drive = os.path.join(root, 'drive.img')
        self.by_id_path = os.path.join(root, by_id)          # a symlink named like the by-id entry, as the claim tool wants to be told
        self.status_dir = os.path.join(root, 'status')
        self.ctl_dir = os.path.join(root, 'models', 'ctl')
        self.state_file = os.path.join(root, 'models', 'state.json')
        self.p2pmem_glob = os.path.join(root, 'p2pmem', '*', 'allocate')
        self.sysfs_pci = os.path.join(root, 'sys', 'bus', 'pci', 'devices')
        self.data_off = store.roundup(store.HEADER_BYTES, store.ALIGN)
        self.device_bytes = self.data_off + num_slots * object_bytes

    # --- files
    def write_status(self, clock: Clock, **overrides) -> Dict[str, Any]:
        status = {'schema': 1, 'timestamp': health.iso_utc(clock.now), 'state': 'ok', 'sed': 'opal', 'locking_enabled': True, 'locked': False,
                  'by_id': self.by_id, 'link_speed': '16.0 GT/s PCIe', 'size_bytes': self.device_bytes, 'detail': ''}
        status.update(overrides)
        os.makedirs(self.status_dir, exist_ok=True)
        for key in [k for k, v in status.items() if v is None]:
            del status[key]
        health.write_json_atomic(os.path.join(self.status_dir, 'status.json'), status)
        return status

    def write_p2pmem(self, cards: int = CHIPS, available: Optional[int] = None) -> None:
        size = 4 << 30
        for i in range(cards):
            name = 'card-%d' % i
            os.makedirs(os.path.join(self.root, 'p2pmem', name), exist_ok=True)
            with open(os.path.join(self.root, 'p2pmem', name, 'allocate'), 'w') as handle:
                handle.write('')
            base = os.path.join(self.sysfs_pci, name, 'p2pmem')
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, 'size'), 'w') as handle:
                handle.write('%d\n' % size)
            with open(os.path.join(base, 'available'), 'w') as handle:
                handle.write('%d\n' % (size if available is None else available))

    def command(self, seq: int, verb: str, body: str = '') -> str:
        os.makedirs(self.ctl_dir, exist_ok=True)
        path = os.path.join(self.ctl_dir, '%04d-%s' % (seq, verb))
        with open(path, 'w') as handle:
            handle.write(body)
        return path

    def read_state(self) -> Dict[str, Any]:
        import json
        with open(self.state_file) as handle:
            return json.load(handle)

    def claim(self, clock: Clock, *, confirm: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        args = dict(status_path=os.path.join(self.status_dir, 'status.json'), layout_id=self.layout_id, allow_regular=True,
                    object_bytes=self.object_bytes, env=health.HostEnv(wall=clock))
        args.update(kwargs)
        return claim.claim(self.by_id_path, self.by_id if confirm is None else confirm, **args)


def make_tree(root: str, clock: Clock, *, num_slots: int = 64, object_bytes: int = OBJ, layout_id: str = TINY.name, by_id: str = BY_ID,
              claimed: bool = True) -> Tree:
    tree = Tree(root, num_slots, object_bytes, layout_id, by_id)
    with open(tree.drive, 'wb') as handle:
        handle.truncate(tree.device_bytes)
    if not os.path.lexists(tree.by_id_path):
        os.symlink(tree.drive, tree.by_id_path)
    tree.write_status(clock)
    tree.write_p2pmem()
    os.makedirs(tree.ctl_dir, exist_ok=True)
    if claimed:
        tree.claim(clock)
    return tree


def make_cfg(tree: Tree, **overrides) -> P2PConfig:
    values = dict(
        layout=tree.layout_id, persistence='process', retention_ack=True, nvme_device=tree.drive, nvme_status_dir=tree.status_dir,
        p2pmem_glob=tree.p2pmem_glob, control_dir=tree.ctl_dir, state_file=tree.state_file, reserve_bytes=0, free_floor_pct=0,
        min_store_tokens=0, require_sed=True, ttl_s=86400, daily_write_cap_bytes=10 ** 12, strikes_to_latch=3, shutdown_drain_s=2.0)
    values.update(overrides)
    cfg = P2PConfig(**values)
    cfg.validate()
    return cfg


def make_env(tree: Tree, clock: Clock, *, environ: Optional[Dict[str, str]] = None, sync_io: bool = True, probe=None) -> health.HostEnv:
    base = {'THATCH_KVP2P_TEST': '1'}
    base.update(environ or {})
    return health.HostEnv(environ=base, sysfs_pci=tree.sysfs_pci, wall=clock, monotonic=clock, p2pmem_probe=probe or (lambda path, size: None),
                          sync_io=sync_io)


class TierCase(unittest.TestCase):
    """Base class: ``self.tier(...)`` builds a tree, a clock and a config; ``self.open_runtime()`` opens the runtime as the worker role."""

    def setUp(self) -> None:
        register_layout(TINY)
        store.KNOWN_LAYOUTS.setdefault(TINY.name, TINY.object_bytes)
        self.root = tempfile.mkdtemp(prefix='ttkvp2p-tier-')
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.clock = Clock()
        KVP2PRuntime.reset_registry()
        self.addCleanup(KVP2PRuntime.reset_registry)
        self._saved_env = runtime_module.set_host_env(None)
        self.addCleanup(runtime_module.set_host_env, self._saved_env)
        self.runtimes: List[KVP2PRuntime] = []
        self.addCleanup(self._close_runtimes)

    def _close_runtimes(self) -> None:
        for runtime in self.runtimes:
            try:
                runtime._io.close(1.0)
                if runtime.store is not None:
                    runtime.store.close()
                    runtime.store = None
            except Exception:
                pass

    def tier(self, *, num_slots: int = 64, claimed: bool = True, environ: Optional[Dict[str, str]] = None, sync_io: bool = True,
             probe=None, **cfg_overrides):
        self.tree = make_tree(self.root, self.clock, num_slots=num_slots, claimed=claimed)
        self.cfg = make_cfg(self.tree, **cfg_overrides)
        self.env = make_env(self.tree, self.clock, environ=environ, sync_io=sync_io, probe=probe)
        runtime_module.set_host_env(self.env)
        return self.tree, self.cfg

    def open_runtime(self, *, available: bool = True, settle: bool = True) -> KVP2PRuntime:
        runtime = KVP2PRuntime.acquire(self.cfg, 'worker')
        self.runtimes.append(runtime)
        if runtime.host_ok:
            if settle:
                self.assertTrue(runtime.settle(5.0), 'the open-time discard did not finish')
            if available:
                runtime.set_available()
        return runtime

    def manager(self, runtime: KVP2PRuntime, **kwargs) -> NvmeKVManager:
        kwargs.setdefault('clock', self.clock)
        kwargs.setdefault('groups', None)
        return NvmeKVManager(self.cfg, runtime, **kwargs)


class RealTier:
    """``harness.Rig(real_tier=RealTier())``: drive vLLM's real scheduler against the REAL ``KVP2PRuntime`` and ``NvmeKVManager`` on the rig's
    file drive (the rig's FakeDrive is a second view of the same claimed file; FakeIO does the byte moves, the runtime owns the slots).

    The kit's clock is the manager's and the runtime's clock, so a test can age entries without sleeping."""

    def __init__(self, clock: Optional[Clock] = None, **cfg_extra):
        self.clock = clock or Clock()
        self.cfg_extra = cfg_extra
        self.tree: Optional[Tree] = None
        self.cfg: Optional[P2PConfig] = None
        self._saved_data_off = None
        self._saved_env = None

    def prepare(self, rig, num_slots: int) -> None:
        register_layout(TINY)
        store.KNOWN_LAYOUTS.setdefault(TINY.name, TINY.object_bytes)
        KVP2PRuntime.reset_registry()
        self.tree = make_tree(rig.tmp, self.clock, num_slots=num_slots)
        self._saved_data_off = fakes.DATA_OFF
        fakes.DATA_OFF = self.tree.data_off              # FakeDrive / FakeIO address slot i at DATA_OFF + i * OBJ: the same place the runtime does
        self._saved_env = runtime_module.set_host_env(make_env(self.tree, self.clock, sync_io=True))

    def extra_config(self, rig) -> Dict[str, Any]:
        tree = self.tree
        extra = {'nvme_device': tree.drive, 'nvme_status_dir': tree.status_dir, 'p2pmem_glob': tree.p2pmem_glob, 'control_dir': tree.ctl_dir,
                 'state_file': tree.state_file, 'reserve_bytes': 0, 'free_floor_pct': 0, 'min_store_tokens': 0, 'require_sed': True,
                 'daily_write_cap_bytes': 10 ** 12}
        extra.update(self.cfg_extra)
        return extra

    def after_start(self, rig) -> None:
        runtime = self.runtime(rig)
        if runtime is not None and runtime.host_ok:
            runtime.settle(10.0)                         # the open-time discard of the data region, done before the first request

    def runtime(self, rig) -> Optional[KVP2PRuntime]:
        return KVP2PRuntime._instances.get(self.tree.drive)

    def finish(self, rig) -> None:
        fakes.DATA_OFF = self._saved_data_off
        runtime_module.set_host_env(self._saved_env)
        KVP2PRuntime.reset_registry()
