"""The WP4 worker's production wiring (``worker.default_backend_factory``) against the REAL ``CardLink`` and ``P2PIO`` over the fake kmd: the calls the factory
makes (constructor shapes, ``open``, ``aim`` for every bank, ``P2PIO(links=..., runtime=..., config=..., staging_info=...)``, ``dma_canary``, ``close``) are
the ones the modules provide, and the ``Backend`` it returns moves real bytes. This needs vLLM (the worker imports it) and numpy, so it skips where they are
not installed; the vLLM 0.30 job runs it.

The factory builds the card device node from ``staging_info['devices'][d]['device_id']`` (tt-metal's logical id); the staging record of the real extension names the
kernel's index in ``pcie_device``, so this rig makes the two equal. See the WP2b report: the factory should use ``pcie_device`` or ``cardlink.chardev_of_bdf``."""

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
    from ttkvp2p import worker as worker_module
    from ttkvp2p.spec import P2PConfig
    HAVE = True
    WHY = ''
except Exception as error:      # noqa: B902
    HAVE, WHY = False, '%s: %s' % (type(error).__name__, error)

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
        self.rig = self.env.rig
        # the module-level defaults a production link uses, pointed at the fake kernel
        saved = (cl.SYSFS_CLASS, cl.SYSFS_PCI, cl.default_ops)
        cl.SYSFS_CLASS, cl.SYSFS_PCI, cl.default_ops = fc.CLASS_ROOT, fc.PCI_ROOT, lambda: self.rig.kmd
        self.addCleanup(lambda: setattr(cl, 'SYSFS_CLASS', saved[0]) or setattr(cl, 'SYSFS_PCI', saved[1]) or setattr(cl, 'default_ops', saved[2]))
        self.calls = []
        qwen = types.ModuleType('qwen_kv_read')
        info = self.rig.staging_info()
        for entry, board in zip(info['devices'], self.rig.boards):
            entry['device_id'] = board.index
        qwen.qwen_p2p_staging_alloc = lambda mesh, size: self.calls.append(('alloc', size)) or object()
        qwen.qwen_p2p_staging_info = lambda staging: info
        qwen.qwen_p2p_warm = lambda staging, tensors, null_block=0: self.calls.append(('warm', len(tensors), null_block))
        qwen.qwen_p2p_freeze = lambda staging: self.calls.append(('freeze',))
        saved_module = sys.modules.get('qwen_kv_read')
        sys.modules['qwen_kv_read'] = qwen
        self.addCleanup(lambda: sys.modules.pop('qwen_kv_read', None) if saved_module is None else sys.modules.__setitem__('qwen_kv_read', saved_module))
        self.cfg = P2PConfig(p2pmem_glob=os.path.join(self.root, '*', 'allocate'), nvme_device=self.env.path, slot_objects=3, shutdown_drain_s=2.0)
        self.registry = types.SimpleNamespace(mesh_device=object(), shard_bdf=fc.BDFS, tensors=list(range(10)))
        self.runtime = fc.FakeRuntime(data_off=fc.Env.DATA_OFF, num_slots=self.env.data_slots)

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


if __name__ == '__main__':
    unittest.main()
