"""The seam between three work packages, with nothing on it stubbed: the WP4 worker's production factory (``worker.default_backend_factory``) builds
the WP2b ``CardLink`` and ``P2PIO`` over the fake kernel, and hands them the REAL WP3a ``KVP2PRuntime`` (claimed file drive, host checks, exclusive open).

test_wp4_wiring.py proves the factory against a runtime that is only SHAPED like WP3a's (``drive.fd``, ``store.data_off`` / ``num_slots`` / ``object_bytes``,
``canary_offset``, ``latch``). This file proves the names, the types and the semantics are the ones the real runtime has: P2PIO's fence is the data region the
runtime's superblock states; its descriptor is the one the runtime opened (O_DIRECT, exclusive); a P2PIO fault latches the real runtime and the runtime files it
under the closed code ``bounds_violation``; and a slot offset the runtime computes is the offset P2PIO moves bytes to and from.

The geometry is the real Gemma layout (four chips, 696,320 bytes per chip per object); the drive is a sparse file with a handful of slots, so the layout's
minimum-session floor is lowered for the test (``store.MIN_SESSION_OBJECTS``). Needs vLLM (the worker imports it) and numpy; skips where they are not installed.
"""

import fcntl
import os
import shutil
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.environ.get('TTKVP2P_PACKAGE_ROOT') or os.path.dirname(HERE)
for entry in (HERE, PACKAGE_ROOT):
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    import fakes  # noqa: F401  (prepares the environment before vLLM is imported)
    from ttkvp2p import worker as worker_module
    HAVE, WHY = True, ''
except Exception as error:      # noqa: B902
    HAVE, WHY = False, '%s: %s' % (type(error).__name__, error)

import fake_card as fc  # noqa: E402
import tier_kit  # noqa: E402
from ttkvp2p import cardlink as cl  # noqa: E402
from ttkvp2p import p2pio as pio  # noqa: E402
from ttkvp2p import runtime as runtime_module  # noqa: E402
from ttkvp2p import store  # noqa: E402
from ttkvp2p.runtime import STATE_AVAILABLE, STATE_LATCHED  # noqa: E402
from ttkvp2p.spec import LAYOUTS, P2PConfig  # noqa: E402

LAYOUT = 'gemma-chipmajor-v1'
SLOTS = 8


@unittest.skipUnless(HAVE, 'the WP4 worker needs vLLM and numpy: %s' % WHY)
class RealRuntimeBehindP2PIO(tier_kit.TierCase):
    def setUp(self):
        super().setUp()
        object_bytes = LAYOUTS[LAYOUT].object_bytes
        saved_min = store.MIN_SESSION_OBJECTS.get(LAYOUT)
        store.MIN_SESSION_OBJECTS[LAYOUT] = 1
        self.addCleanup(lambda: store.MIN_SESSION_OBJECTS.pop(LAYOUT, None) if saved_min is None else store.MIN_SESSION_OBJECTS.__setitem__(LAYOUT, saved_min))
        self.tree = tier_kit.make_tree(self.root, self.clock, num_slots=SLOTS, object_bytes=object_bytes, layout_id=LAYOUT)
        # the p2pmem files and their sysfs records are named by the cards' addresses, as /run/kv-p2pmem is, so that the runtime's host checks and the
        # factory's CardLink look at the same four entries
        shutil.rmtree(os.path.join(self.root, 'p2pmem'))
        shutil.rmtree(self.tree.sysfs_pci)
        self.p2pmem_root = os.path.join(self.root, 'p2pmem')
        for bdf in fc.BDFS:
            os.makedirs(os.path.join(self.p2pmem_root, bdf))
            open(os.path.join(self.p2pmem_root, bdf, 'allocate'), 'w').close()
            sysfs = os.path.join(self.tree.sysfs_pci, bdf, 'p2pmem')
            os.makedirs(sysfs)
            for name in ('size', 'available'):
                with open(os.path.join(sysfs, name), 'w') as handle:
                    handle.write('%d\n' % (4 << 30))
        self.cfg = tier_kit.make_cfg(self.tree, layout=LAYOUT, slot_objects=3, shutdown_drain_s=2.0)
        self.assertEqual(self.cfg.p2pmem_glob, os.path.join(self.p2pmem_root, '*', 'allocate'))
        self.host = tier_kit.make_env(self.tree, self.clock, sync_io=True)
        runtime_module.set_host_env(self.host)

        self.card = fc.Env(rig_kwargs=dict(shard_object_bytes=fc.GEMMA_SHARD, slot_objects=3, allocate_root=self.p2pmem_root), data_slots=SLOTS, start=False)
        self.addCleanup(self.card.close)
        self.rig = self.card.rig
        self.assertEqual(self.rig.object_bytes, object_bytes, 'the fake cards and the layout agree on the size of an object')
        saved = (cl.SYSFS_CLASS, cl.SYSFS_PCI, cl.default_ops)
        cl.SYSFS_CLASS, cl.SYSFS_PCI, cl.default_ops = fc.CLASS_ROOT, fc.PCI_ROOT, lambda: self.rig.kmd
        self.addCleanup(lambda: setattr(cl, 'SYSFS_CLASS', saved[0]) or setattr(cl, 'SYSFS_PCI', saved[1]) or setattr(cl, 'default_ops', saved[2]))
        qwen = types.ModuleType('qwen_kv_read')
        info = self.rig.staging_info()
        qwen.qwen_p2p_staging_alloc = lambda mesh, size: object()
        qwen.qwen_p2p_staging_info = lambda staging: info
        qwen.qwen_p2p_warm = lambda staging, tensors, null_block=0: None
        qwen.qwen_p2p_freeze = lambda staging: None
        saved_module = sys.modules.get('qwen_kv_read')
        sys.modules['qwen_kv_read'] = qwen
        self.addCleanup(lambda: sys.modules.pop('qwen_kv_read', None) if saved_module is None else sys.modules.__setitem__('qwen_kv_read', saved_module))
        self.registry = types.SimpleNamespace(mesh_device=object(), shard_bdf=fc.BDFS, tensors=list(range(10)))

        self.runtime = self.open_runtime()
        self.assertTrue(self.runtime.host_ok, self.runtime.host_reason)
        self.assertEqual(self.runtime.state, STATE_AVAILABLE)
        flags = fcntl.fcntl(self.runtime.drive.fd, fcntl.F_GETFL)
        if not flags & getattr(os, 'O_DIRECT', 0):
            self.skipTest('the runtime opened the drive file without O_DIRECT (this file system takes none), and P2PIO refuses a buffered descriptor')

    def build(self):
        return worker_module.default_backend_factory(cfg=self.cfg, registry=self.registry, runtime=self.runtime, geometry=None)

    def drive_bytes(self, offset, size):
        fd = os.open(self.tree.drive, os.O_RDONLY)
        try:
            return os.pread(fd, size, offset)
        finally:
            os.close(fd)

    # -------------------------------------------------------------------------------------------------------------------------------------------------
    def test_p2pio_takes_the_descriptor_the_fence_and_the_canary_from_the_real_runtime(self):
        backend = self.build()
        try:
            self.assertIsInstance(backend.io, pio.P2PIO)
            self.assertEqual(backend.io._fd, self.runtime.drive.fd, 'the descriptor is the one the runtime opened, exclusively')
            sb = self.runtime.store.superblock
            self.assertEqual(backend.io._drive_range, (sb.data_off, sb.data_off + sb.num_slots * sb.object_bytes),
                             'the fence is the data region of the superblock: the superblocks and the canary block are never data')
            self.assertEqual((self.runtime.store.data_off, self.runtime.store.num_slots, self.runtime.store.object_bytes), (sb.data_off, SLOTS, sb.object_bytes))
            self.assertLessEqual(self.runtime.canary_offset + self.runtime.canary_bytes, sb.data_off, 'the DMA canary block lies outside the data region')
            self.assertEqual(self.runtime.state, STATE_AVAILABLE, 'the start-up DMA canary through the real drive latched nothing')
        finally:
            backend.close()

    def test_a_slot_offset_of_the_real_runtime_is_where_p2pio_moves_the_bytes(self):
        backend = self.build()
        try:
            for slot in (0, 3, SLOTS - 1):
                for chip in range(4):
                    self.card.put(chip, 0, 0, self.card.card_object(chip, 'slot%d' % slot))
                offset = self.runtime.slot_offset(slot)
                self.assertEqual(backend.io.write_subrun(offset, 0, 1).result(10), self.rig.object_bytes)
                self.assertEqual(self.drive_bytes(offset, self.rig.object_bytes), self.card.drive_object('slot%d' % slot),
                                 'the drive holds the four shards, chip-major, at the slot the runtime names')
            for slot in (0, 3, SLOTS - 1):                      # and the way back: the object of slot s lands in a restore bank
                self.assertEqual(backend.io.read_piece(self.runtime.slot_offset(slot), 2, 1).result(10), self.rig.object_bytes)
                for chip in range(4):
                    self.assertEqual(self.card.get(chip, 2, 0), self.card.card_object(chip, 'slot%d' % slot))
        finally:
            backend.close()

    def test_a_slot_past_the_data_region_is_refused_before_any_byte_moves(self):
        backend = self.build()
        try:
            beyond = self.runtime.store.data_off + SLOTS * self.runtime.store.object_bytes
            before = self.drive_bytes(beyond - self.rig.object_bytes, self.rig.object_bytes)
            with self.assertRaises(cl.BoundsViolation):
                backend.io.write_subrun(beyond, 0, 1)
            with self.assertRaises(cl.BoundsViolation):
                backend.io.read_piece(self.runtime.store.data_off - self.rig.object_bytes, 2, 1)       # the superblock side
            self.assertEqual(self.drive_bytes(beyond - self.rig.object_bytes, self.rig.object_bytes), before)
        finally:
            backend.close()

    def test_a_failed_dma_canary_latches_the_real_runtime_under_the_bounds_code_and_leaves_nothing_open(self):
        original = pio.SystemDriveOps.pwritev
        pio.SystemDriveOps.pwritev = lambda self, fd, buffers, offset: sum(memoryview(b).nbytes for b in buffers)       # a drive that takes the bytes and keeps nothing
        try:
            with self.assertRaises(pio.DmaCanaryError):
                self.build()
        finally:
            pio.SystemDriveOps.pwritev = original
        self.assertEqual(self.rig.kmd.open_fds, 0)
        self.assertEqual(self.runtime.state, STATE_LATCHED)
        self.assertTrue(self.runtime.latch_reason.startswith(pio.FATAL_PREFIX), self.runtime.latch_reason)
        self.assertEqual(self.runtime.latch_code, 'bounds_violation', 'WP3a files the reason WP2b words as a bounds violation, not as an op throw')

    def test_a_p2pio_that_stopped_is_latched_by_the_worker_through_the_real_runtime(self):
        """The thread rule: P2PIO records why it stopped in ``dead`` and never calls the runtime from its own threads; the worker latches on the engine thread."""
        backend = self.build()
        try:
            reason = '%s: StaleAim: I/O planned for bank 2 epoch 3, but the window is at bank 2 epoch 4' % pio.FATAL_PREFIX
            backend.io._dead = reason                           # what a fault found on one of the layer's own threads leaves behind
            self.assertEqual(backend.io.dead, reason)
            self.assertEqual(self.runtime.state, STATE_AVAILABLE, 'nothing latched yet: the layer only recorded the reason')
            latched = []
            worker = types.SimpleNamespace(_latched=False, be=backend, _latch=lambda why: (latched.append(why), self.runtime.latch(why)))
            worker_module.TTP2PWorker._check_io_layer(worker)      # the worker's own method, at its next service pass on the engine thread
            self.assertEqual(latched, [reason])
            self.assertEqual((self.runtime.state, self.runtime.latch_code), (STATE_LATCHED, 'bounds_violation'))
            self.assertEqual(self.runtime.latch_reason, reason)
        finally:
            backend.close()

    def test_closing_the_backend_releases_every_descriptor_and_window_and_the_runtime_still_owns_the_drive(self):
        backend = self.build()
        backend.close()
        self.assertEqual(self.rig.kmd.open_fds, 0)
        for board in self.rig.boards:
            self.assertEqual(board.taken, set())
        self.assertIsNotNone(self.runtime.drive, 'the layer never owned the drive: closing it leaves the runtime\'s descriptor open')
        os.fstat(self.runtime.drive.fd)


if __name__ == '__main__':
    unittest.main()
