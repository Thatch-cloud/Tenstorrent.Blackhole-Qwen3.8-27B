"""The production wiring of the device side (``worker.default_backend_factory``) against stand-ins for the three modules it reaches
through the frozen interface: ``qwen_kv_read``, ``ttkvp2p.cardlink`` and ``ttkvp2p.p2pio``. It checks the START-UP ORDER of the design
(host checks, staging, a card link per shard, every bank aimed and verified, the DMA canary, the warm-up of every program to completion,
the freeze, and only then AVAILABLE) and that a failure at any step leaves the tier off with the windows closed and the runtime released."""

import os
import sys
import tempfile
import types
import unittest

from fakes import BANKS, CHIPS, LEAD, SHARD_OBJ, FakeIO, FakeOps, FakeStaging
from harness import SHARDS, RigCase, tokens
from ttkvp2p import worker as worker_module


class StartupSequence(RigCase):
    def stand_ins(self, rig, log, fail_at=None, kmd_index=True):
        """Install the three modules; every call is appended to ``log`` as it happens."""
        card, drive = rig.card, rig.drive
        ops = FakeOps(card, rig.slot_objects)

        def step(name):
            log.append(name)
            if fail_at == name:
                raise RuntimeError('injected failure at %s' % name)

        qwen = types.ModuleType('qwen_kv_read')
        qwen.qwen_blocks_to_staging = ops.qwen_blocks_to_staging
        qwen.qwen_write_blocks_from_staging = ops.qwen_write_blocks_from_staging
        qwen.qwen_blocks_digest_async = ops.qwen_blocks_digest_async
        qwen.qwen_p2p_record_event = ops.qwen_p2p_record_event

        def staging_alloc(mesh, bytes_per_bank):
            step('staging_alloc')
            qwen.bytes_per_bank = bytes_per_bank
            return FakeStaging()

        def staging_info(staging):
            info = {'address': 0x80000000, 'size': card.bank_bytes, 'page_bytes': 1088, 'aligned_4k_offset': LEAD,
                    'aligned_4k_address': 0x80000000 + LEAD, 'aligned_4k_size': card.bank_bytes - LEAD,
                    'banks': [{'bank': b, 'noc_x': 1, 'noc_y': b} for b in range(BANKS)],
                    # device_id is tt-metal's logical id; pcie_device is the tt-kmd index N of /dev/tenstorrent/N. They differ on purpose.
                    'devices': [{'mesh_coord': d, 'device_id': 10 + d, 'pcie_device': 20 + d, 'pci_bus_id': SHARDS[d]}
                                for d in range(CHIPS)]}
            if not kmd_index:
                for entry in info['devices']:
                    entry['pcie_device'] = None            # the host could not resolve the kmd index
            return info

        def warm(staging, tensors, null_block=0):
            step('warm')
            assert null_block == 0 and len(tensors) == 10

        def freeze(staging):
            step('freeze')

        qwen.qwen_p2p_staging_alloc, qwen.qwen_p2p_staging_info = staging_alloc, staging_info
        qwen.qwen_p2p_warm, qwen.qwen_p2p_freeze = warm, freeze

        class CardLink:
            instances = []

            def __init__(self, chardev, bdf, allocate_path, window_id=202):
                self.chardev, self.bdf, self.allocate_path, self.window_id = chardev, bdf, allocate_path, window_id
                self.closed = False
                CardLink.instances.append(self)

            def open(self, staging_info):
                step('open:%s' % self.bdf)
                assert staging_info['banks']

            def aim(self, bank):
                step('aim:%s:%d' % (self.bdf, bank))

            def flush(self):
                log.append('flush')

            def close(self):
                self.closed = True
                log.append('close:%s' % self.bdf)

        class P2PIO:
            def __init__(self, links, runtime, config, staging_info):
                step('p2pio')
                assert len(links) == CHIPS and config.p2p_window_id == 202
                self.io = rig.factory_io = FakeIO(card, drive)

            def read_piece(self, *args, **kwargs):
                return self.io.read_piece(*args, **kwargs)

            def write_subrun(self, *args, **kwargs):
                return self.io.write_subrun(*args, **kwargs)

            def dma_canary(self):
                step('dma_canary')

            def close(self):
                log.append('p2pio_close')
                self.io.close()

        cardlink = types.ModuleType('ttkvp2p.cardlink')
        cardlink.CardLink = CardLink
        p2pio = types.ModuleType('ttkvp2p.p2pio')
        p2pio.P2PIO = P2PIO
        saved = {n: sys.modules.get(n) for n in ('qwen_kv_read', 'ttkvp2p.cardlink', 'ttkvp2p.p2pio')}
        sys.modules.update({'qwen_kv_read': qwen, 'ttkvp2p.cardlink': cardlink, 'ttkvp2p.p2pio': p2pio})

        def restore():
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.addCleanup(restore)
        return CardLink

    def allocate_files(self):
        root = tempfile.mkdtemp(prefix='ttkvp2p-p2pmem-')
        for bdf in SHARDS:
            os.makedirs(os.path.join(root, bdf))
            open(os.path.join(root, bdf, 'allocate'), 'w').close()
        return os.path.join(root, '*', 'allocate')

    def test_the_startup_order_then_available_then_the_worker_does_real_work(self):
        rig = self.rig(start_worker=False, extra={'p2pmem_glob': self.allocate_files()})
        log = []
        CardLink = self.stand_ins(rig, log)
        worker_module.BACKEND_FACTORY = None                          # the production factory
        rig.worker_conn.register_kv_caches(rig.registry)
        self.assertIsNotNone(rig.worker, rig.worker_conn.connector_worker.unavailable_reason)
        self.assertEqual(rig.runtime.state, 'AVAILABLE')
        expected = ['staging_alloc'] + ['open:%s' % b for b in SHARDS]
        expected += ['aim:%s:%d' % (b, bank) for bank in (0, 1, 2, 3, 4, 5, 6, 7) for b in SHARDS]
        expected += ['p2pio', 'dma_canary', 'warm', 'freeze']
        self.assertEqual(log, expected)
        self.assertEqual([link.chardev for link in CardLink.instances], ['/dev/tenstorrent/%d' % (20 + d) for d in range(CHIPS)],
                         'the character device is named by the tt-kmd index (pcie_device), not by tt-metal\'s logical device_id')
        self.assertEqual({link.window_id for link in CardLink.instances}, {202})
        self.assertTrue(all(link.bdf in link.allocate_path for link in CardLink.instances), 'each card gets its own p2pmem file')
        # and the worker it built moves real bytes through those stand-ins, flushing through the card links
        prompt = tokens(2300, 'startup')
        rig.add('r1', prompt)
        rig.drain()
        rig.reset_device_cache()
        rig.add('r2', prompt)
        rig.finish('r2')
        self.assertEqual(rig.worker.stats['done_load'], 1)
        self.assertGreater(log.count('flush'), 0)
        self.assertEqual(log.count('flush') % CHIPS, 0, 'the advisory read goes through every card')
        rig.worker_conn.shutdown()
        self.assertEqual(log[-CHIPS - 1:], ['p2pio_close'] + ['close:%s' % b for b in SHARDS])

    def test_a_failure_at_any_step_leaves_the_tier_off_the_windows_closed_and_the_runtime_released(self):
        for fail_at in ('staging_alloc', 'open:card-2', 'aim:card-1:5', 'p2pio', 'dma_canary', 'warm', 'freeze'):
            with self.subTest(fail_at=fail_at):
                rig = self.rig(start_worker=False, extra={'p2pmem_glob': self.allocate_files()})
                log = []
                CardLink = self.stand_ins(rig, log, fail_at=fail_at)
                worker_module.BACKEND_FACTORY = None
                rig.worker_conn.register_kv_caches(rig.registry)         # never raises
                self.assertIsNone(rig.worker)
                self.assertIn('injected failure', rig.worker_conn.connector_worker.unavailable_reason)
                self.assertEqual(rig.runtime.state, 'UNAVAILABLE', 'AVAILABLE is the last step')
                self.assertEqual(rig.runtime.released, ['worker'])
                self.assertTrue(all(link.closed for link in CardLink.instances), 'every opened window is closed again')
                rig.close()

    def test_a_shard_map_that_disagrees_with_the_staging_devices_is_refused(self):
        import dataclasses
        rig = self.rig(start_worker=False, extra={'p2pmem_glob': self.allocate_files()})
        log = []
        self.stand_ins(rig, log)
        worker_module.BACKEND_FACTORY = None
        swapped = dataclasses.replace(rig.registry, shard_bdf=(SHARDS[1], SHARDS[0]) + SHARDS[2:])
        rig.worker_conn.register_kv_caches(swapped)
        self.assertIsNone(rig.worker)
        self.assertIn('shard map', rig.worker_conn.connector_worker.unavailable_reason)
        self.assertEqual([entry for entry in log if entry.startswith(('open', 'aim'))], [], 'no window was touched')

    def test_a_missing_or_ambiguous_p2pmem_file_is_refused(self):
        rig = self.rig(start_worker=False, extra={'p2pmem_glob': tempfile.mkdtemp(prefix='ttkvp2p-none-') + '/*/allocate'})
        log = []
        self.stand_ins(rig, log)
        worker_module.BACKEND_FACTORY = None
        rig.worker_conn.register_kv_caches(rig.registry)
        self.assertIsNone(rig.worker)
        self.assertIn('p2pmem allocate file', rig.worker_conn.connector_worker.unavailable_reason)
        self.assertEqual(log, [], 'a host check: nothing was allocated on the cards before it failed (design 2.9 step 2)')
        self.assertEqual(rig.card.log, [])

    def test_an_ambiguous_p2pmem_file_is_refused_with_zero_device_footprint(self):
        root = tempfile.mkdtemp(prefix='ttkvp2p-p2pmem-')
        for bdf in SHARDS:
            for variant in ('a', 'b') if bdf == SHARDS[2] else ('a',):          # two files whose paths both name card-2
                os.makedirs(os.path.join(root, bdf, variant))
                open(os.path.join(root, bdf, variant, 'allocate'), 'w').close()
        rig = self.rig(start_worker=False, extra={'p2pmem_glob': os.path.join(root, '*', '*', 'allocate')})
        log = []
        self.stand_ins(rig, log)
        worker_module.BACKEND_FACTORY = None
        rig.worker_conn.register_kv_caches(rig.registry)
        self.assertIsNone(rig.worker)
        self.assertIn('found 2', rig.worker_conn.connector_worker.unavailable_reason)
        self.assertEqual(log, [])

    def test_a_staging_record_without_a_kmd_index_is_refused_before_any_window_is_opened(self):
        rig = self.rig(start_worker=False, extra={'p2pmem_glob': self.allocate_files()})
        log = []
        CardLink = self.stand_ins(rig, log, kmd_index=False)
        worker_module.BACKEND_FACTORY = None
        rig.worker_conn.register_kv_caches(rig.registry)
        self.assertIsNone(rig.worker)
        self.assertIn('pcie_device', rig.worker_conn.connector_worker.unavailable_reason)
        self.assertEqual([entry for entry in log if entry.startswith(('open', 'aim'))], [], 'no window was opened')
        self.assertEqual(CardLink.instances, [])
        self.assertEqual(rig.runtime.state, 'UNAVAILABLE')
        self.assertEqual(rig.runtime.released, ['worker'])


class TopOfWindow(RigCase):
    """The staging slot of every CONFIGURED bank must stay below the top 16 MiB of the window, which is not DRAM."""

    def start_with_bank_addresses(self, rig, addresses):
        real = worker_module.BACKEND_FACTORY

        def factory(**kwargs):
            backend = real(**kwargs)
            for entry in backend.staging_info['banks']:
                entry['address'] = addresses.get(entry['bank'], 0x80000000)
            return backend

        worker_module.BACKEND_FACTORY = factory
        rig.worker_conn.register_kv_caches(rig.registry)

    def test_a_high_bank_other_than_bank_zero_is_refused(self):
        rig = self.rig(start_worker=False)
        self.start_with_bank_addresses(rig, {6: 0xFF000000 - 100})
        self.assertIsNone(rig.worker)
        reason = rig.worker_conn.connector_worker.unavailable_reason
        self.assertIn('bank 6', reason)
        self.assertIn('top 16 MiB', reason)
        self.assertEqual(rig.runtime.released, ['worker'])

    def test_a_high_bank_that_is_not_configured_is_not_checked(self):
        rig = self.rig(start_worker=False, extra={'restore_banks': [2, 3, 4, 5, 7]})
        self.start_with_bank_addresses(rig, {6: 0xFF000000 - 100})
        self.assertIsNotNone(rig.worker, rig.worker_conn.connector_worker.unavailable_reason)

    def test_the_exact_top_is_allowed_and_one_byte_over_is_not(self):
        rig = self.rig(start_worker=False)
        end_of_slot = LEAD + 4096 + rig.slot_objects * SHARD_OBJ                   # bank-relative end of the slot (the fake's layout)
        self.start_with_bank_addresses(rig, {7: 0xFF000000 - end_of_slot})
        self.assertIsNotNone(rig.worker, rig.worker_conn.connector_worker.unavailable_reason)
        rig2 = self.rig(start_worker=False)
        self.start_with_bank_addresses(rig2, {7: 0xFF000000 - end_of_slot + 1})
        self.assertIsNone(rig2.worker)


if __name__ == '__main__':
    unittest.main()
