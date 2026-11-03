"""The WP4 worker's production wiring (``worker.default_backend_factory``) against the REAL ``CardLink`` and ``P2PIO`` over the fake kmd: the calls the factory
makes (constructor shapes, ``open``, ``aim`` for every bank, ``P2PIO(links=..., runtime=..., config=..., staging_info=...)``, ``dma_canary``, ``close``) are
the ones the modules provide, and the ``Backend`` it returns moves real bytes. This needs vLLM (the worker imports it) and numpy, so it skips where they are
not installed; the vLLM 0.30 job runs it.

The factory builds the card device node from ``staging_info['devices'][d]['pcie_device']`` (the kernel's index), not from ``device_id`` (tt-metal's logical id): this
rig gives the two different values, so a factory that used ``device_id`` would open the wrong cards. The runtime is shaped like WP3a's (``drive.fd``,
``store.data_off`` / ``num_slots`` / ``object_bytes``, ``canary_offset``), because that is what P2PIO reads; the factory is not told about the drive, and the drive
descriptor must be open O_DIRECT (so the test skips on a file system without it). The last class checks the other half of the thread rule: the worker latches the tier
on the engine thread when the I/O layer says it has stopped."""

import logging
import os
import shutil
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.environ.get('TTKVP2P_PACKAGE_ROOT') or os.path.dirname(HERE)
for entry in (HERE, PACKAGE_ROOT):
    if entry not in sys.path:
        sys.path.insert(0, entry)

logging.getLogger('ttkvp2p').addHandler(logging.NullHandler())

try:
    import fakes  # noqa: F401  (prepares the environment before vLLM is imported)
    from harness import RigCase, WorkerKit
    from ttkvp2p import worker as worker_module
    from ttkvp2p.spec import P2PConfig
    HAVE = True
    WHY = ''
except Exception as error:      # noqa: B902
    HAVE, WHY = False, '%s: %s' % (type(error).__name__, error)
    RigCase = unittest.TestCase

import fake_card as fc  # noqa: E402
from ttkvp2p import cardlink as cl  # noqa: E402
from ttkvp2p import p2pio as pio  # noqa: E402


@unittest.skipUnless(HAVE, 'the WP4 worker needs vLLM and numpy: %s' % WHY)
class FactoryWiring(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='ttkvp2p-wiring-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.root = os.path.join(self.dir, 'p2pmem')
        for bdf in fc.BDFS:
            os.makedirs(os.path.join(self.root, bdf))
            open(os.path.join(self.root, bdf, 'allocate'), 'w').close()
        self.env = fc.Env(rig_kwargs=dict(shard_object_bytes=fc.GEMMA_SHARD, slot_objects=3, allocate_root=self.root), data_slots=8, start=False)
        self.addCleanup(self.env.close)
        if not self.env.direct:
            self.skipTest('the drive descriptor must be open O_DIRECT, and this file system does not take it')
        self.rig = self.env.rig
        # the module-level defaults a production link uses, pointed at the fake kernel
        saved = (cl.SYSFS_CLASS, cl.SYSFS_PCI, cl.default_ops)
        cl.SYSFS_CLASS, cl.SYSFS_PCI, cl.default_ops = fc.CLASS_ROOT, fc.PCI_ROOT, lambda: self.rig.kmd
        self.addCleanup(lambda: setattr(cl, 'SYSFS_CLASS', saved[0]) or setattr(cl, 'SYSFS_PCI', saved[1]) or setattr(cl, 'default_ops', saved[2]))
        self.calls = []
        qwen = types.ModuleType('qwen_kv_read')
        info = self.rig.staging_info()
        for entry, board in zip(info['devices'], self.rig.boards):
            self.assertNotEqual(entry['device_id'], entry['pcie_device'], 'the logical id and the kernel index differ, as on a rig')
        qwen.qwen_p2p_staging_alloc = lambda mesh, size: self.calls.append(('alloc', size)) or object()
        qwen.qwen_p2p_staging_info = lambda staging: info
        qwen.qwen_p2p_warm = lambda staging, tensors, null_block=0: self.calls.append(('warm', len(tensors), null_block))
        qwen.qwen_p2p_freeze = lambda staging: self.calls.append(('freeze',))
        saved_module = sys.modules.get('qwen_kv_read')
        sys.modules['qwen_kv_read'] = qwen
        self.addCleanup(lambda: sys.modules.pop('qwen_kv_read', None) if saved_module is None else sys.modules.__setitem__('qwen_kv_read', saved_module))
        self.cfg = P2PConfig(p2pmem_glob=os.path.join(self.root, '*', 'allocate'), nvme_device=self.env.path, slot_objects=3, shutdown_drain_s=2.0)
        self.registry = types.SimpleNamespace(mesh_device=object(), shard_bdf=fc.BDFS, tensors=list(range(10)))
        self.runtime = self.env.wp3a_runtime()

    def build(self):
        return worker_module.default_backend_factory(cfg=self.cfg, registry=self.registry, runtime=self.runtime, geometry=None)

    def test_the_factory_builds_a_backend_that_moves_real_bytes(self):
        backend = self.build()
        try:
            self.assertIsInstance(backend.io, pio.P2PIO)
            self.assertEqual([c[0] for c in self.calls], ['alloc', 'warm', 'freeze'], 'staging, then the links and the DMA canary, then warm and freeze')
            self.assertEqual(self.runtime.latched, [])
            e = self.env
            for d in range(4):
                e.put(d, 0, 0, e.card_object(d, 'w'))
            self.assertEqual(backend.io.write_subrun(e.drive_off(1), 0, 1).result(10), 2785280)
            self.assertEqual(e.drive_read(e.drive_off(1), e.object_bytes), e.drive_object('w'))
            self.assertEqual(backend.io.read_piece(e.drive_off(1), 2, 1, first_obj=2).result(10), 2785280)
            for d in range(4):
                self.assertEqual(e.get(d, 2, 2), e.card_object(d, 'w'))
            backend.flush()
            self.assertTrue(all(link.stats['flushes'] >= 1 for link in self.rig.links))
        finally:
            backend.close()
        self.assertEqual(self.rig.kmd.open_fds, 0, 'closing the backend released every descriptor and window')
        for board in self.rig.boards:
            self.assertEqual(board.taken, set())

    def test_a_failure_during_start_up_leaves_nothing_open(self):
        self.rig.boards[2].alias = {5: 4}
        with self.assertRaises(cl.CanaryMismatch):
            self.build()
        self.assertEqual(self.rig.kmd.open_fds, 0)
        self.assertEqual([c for c in self.calls if c[0] in ('warm', 'freeze')], [], 'no warm-up after a failed canary')

    def test_a_dma_canary_failure_closes_everything(self):
        original = pio.SystemDriveOps.pwritev
        pio.SystemDriveOps.pwritev = lambda self, fd, buffers, offset: sum(memoryview(b).nbytes for b in buffers)       # a drive that takes the bytes and keeps nothing
        try:
            with self.assertRaises(pio.DmaCanaryError):
                self.build()
        finally:
            pio.SystemDriveOps.pwritev = original
        self.assertEqual(self.rig.kmd.open_fds, 0)
        self.assertTrue(self.runtime.latched, 'the tier is latched')
        self.assertTrue(self.runtime.latched[0].startswith(pio.FATAL_PREFIX), self.runtime.latched[0])

    def test_the_factory_hands_the_drive_to_p2pio_from_the_runtime_and_never_opens_it(self):
        backend = self.build()
        try:
            self.assertEqual(backend.io._fd, self.env.fd)
            self.assertEqual(backend.io._drive_range, (fc.Env.DATA_OFF, fc.Env.DATA_OFF + self.env.data_slots * self.env.object_bytes))
            with self.assertRaises(cl.BoundsViolation):
                backend.io.read_piece(self.env.drive_off(self.env.data_slots), 2, 1)       # one slot past the region the runtime states
        finally:
            backend.close()

    def test_a_runtime_that_states_no_data_region_leaves_the_tier_unavailable_with_nothing_open(self):
        del self.runtime.store
        with self.assertRaises(pio.P2PIOError):
            self.build()
        self.assertEqual(self.rig.kmd.open_fds, 0)
        self.assertEqual([c for c in self.calls if c[0] in ('warm', 'freeze')], [])


@unittest.skipUnless(HAVE, 'the WP4 worker needs vLLM and numpy: %s' % WHY)
class WorkerLatchesForTheIoLayer(RigCase):
    """P2PIO never calls the runtime from its own threads; it records why it stopped in ``dead``. The worker, on the engine thread, latches the tier from there."""

    REASON = 'P2PIO bounds violation: StaleAim: I/O planned for bank 2 epoch 3, but the window is at bank 2 epoch 4'

    def test_a_stopped_io_layer_is_latched_at_the_next_service_and_fails_what_is_in_flight(self):
        rig = self.rig(with_scheduler=False)
        kit = WorkerKit(rig)
        kit.fill_blocks([1, 2, 3, 4], 'w')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        self.assertEqual(rig.runtime.state, 'AVAILABLE')
        kit.store(2, [3, 4])
        rig.io.dead = self.REASON                              # found by an I/O thread of P2PIO, which does not call the runtime
        kit.pump_once()
        self.assertEqual(rig.runtime.state, 'LATCHED')
        self.assertEqual(rig.runtime.latch_reason, self.REASON, 'the reason of the layer is the reason of the latch')
        self.assertEqual((kit.results[2].success, kit.results[2].reason), (False, 'latched'))

    def test_a_layer_that_is_not_dead_changes_nothing(self):
        rig = self.rig(with_scheduler=False)
        kit = WorkerKit(rig)
        rig.io.dead = None
        kit.fill_blocks([1, 2], 'w')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        self.assertTrue(kit.results[1].success)
        self.assertEqual(rig.runtime.state, 'AVAILABLE')


# The same tests again on the stand-in runtime (tests/fakes.py), as in the other files.
@unittest.skipUnless(HAVE, 'the WP4 worker needs vLLM and numpy: ' + WHY)
class StandInWorkerLatchesForTheIoLayer(WorkerLatchesForTheIoLayer):
    FAKE_TIER = True


if __name__ == '__main__':
    unittest.main()
