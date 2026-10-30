"""The NVMe tier's peer-to-peer restore (QWEN_PREFIX_NVME_P2P; docs/tp4-p2p-restore.md), under the real scheduler graft on the fake world of the host-tier tests, with the cards' end of the
path replaced by memory (test_qwen_p2p_restore's FakeMesh and FakeLink) and the drive by a file.

Held:
  * the flag: strict, off unless named, a gate's name in the contract, a new layout (and so a new fingerprint) when on, none of its knobs in the fingerprint;
  * the stored bytes: chip-major, the same bytes in another order, a digest of what is stored, one checksum per chip beside it; with the flag off nothing of this exists;
  * the restore: NVMe blocks land in the device blocks through the cards, the turn is exact; a checksum that differs is a miss and writes nothing; a bad drive is told from a bad path;
    any other failure restores the same blocks through the host path; three failures (or one device-op error) latch the path off and the host records take over; a path that cannot
    start leaves the host backend, which reads the chip-major objects back into the host tier's layout;
  * the sliced restore: the reads run beside the other seats' steps (a slice that has not landed does nothing), host records and NVMe blocks mix, a stalled drive is abandoned;
  * the audit reads the restored blocks back through the model's own IO and compares them with what the drive holds; counters, metrics, the stop.
Run alone: python3 -B -m unittest test_qwen_p2p_tier (python 3.11+, numpy)
"""

import hashlib
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qwen_kv_nvme as kvnvme  # noqa: E402
import qwen_nvme_store as store_module  # noqa: E402
import qwen_p2p_restore as p2p  # noqa: E402
import qwen_prefix_scheduler_patch as patch_module  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import test_qwen_prefix_tiers as tiers_module  # noqa: E402
from qwen_prefix_registry import BLOCK, PrefixRegistry  # noqa: E402
from test_qwen_kv_nvme import ACKED, NvmeCase, first_turn  # noqa: E402
from test_qwen_p2p_restore import FakeMesh, make_links  # noqa: E402
from test_qwen_prefix_scheduler_patch import FakeKvIO, Request, kv_content  # noqa: E402
from test_qwen_prefix_tiers import SMALL_POOL, span  # noqa: E402

P2P = {'QWEN_PREFIX_NVME_P2P': '1'}
KV = store_module.KV


class GeoKvIO(FakeKvIO):
    """FakeKvIO with a chip geometry: 2 cache tensors x 4 chips x 4 bytes = the fake world's 32-byte block (a shard is 8 bytes)."""

    caches = ['k0', 'v0']
    chips = 4
    slice_bytes = 4
    mesh_device = 'fake-mesh'

    def __init__(self):
        FakeKvIO.__init__(self)
        self.corrupt_copy = False

    def sink(self, tensor, chip, block, data):
        buffer = bytearray(self.device.get(block) or bytes(self.block_bytes))
        start = (tensor * self.chips + chip) * self.slice_bytes
        buffer[start:start + self.slice_bytes] = bytes([data[0] ^ 1]) + data[1:] if self.corrupt_copy and tensor == 0 and chip == 0 else data
        self.device[block] = bytes(buffer)


class P2PWorld(object):
    """What the tests reach: the engine factory patched in for default_p2p_engine, the fake mesh, the links, the tiers it built."""

    def __init__(self, threads=2, staging_blocks=8, fail=None, reader=None):
        self.threads, self.staging_blocks, self.fail, self.reader = threads, staging_blocks, fail, reader
        self.meshes, self.links, self.engines = [], [], []

    def factory(self, tier):
        if self.fail is not None:
            raise self.fail
        layout = tier.shards
        mesh = FakeMesh(layout, tier.io.sink)
        link_factory, links = make_links(mesh)
        # (the fake world's blocks are 32 bytes: shards are not 4 KiB-aligned there, so the reads go straight at the store's device file; the real alignment is test_qwen_p2p_restore's)
        reader = self.reader or (lambda offset, view: os.preadv(tier.store.bf.fd, [view], offset))
        engine = p2p.P2PEngine(p2p.StagingApi(mesh, 'fake-mesh', list(range(layout.caches))), layout, reader, threads=self.threads,
                               staging_bytes=layout.shard_bytes * self.staging_blocks, link_factory=link_factory, io_align=1, sleep=lambda seconds: None, logger=tier.log)
        self.meshes.append(mesh)
        self.links.extend(links)
        self.engines.append(engine)
        return engine

    @property
    def mesh(self):
        return self.meshes[-1]


class P2PCase(NvmeCase):
    def setUp(self):
        NvmeCase.setUp(self)
        patcher = mock.patch.object(tiers_module, 'FakeKvIO', GeoKvIO)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.world = P2PWorld()
        self.install(self.world)

    def install(self, world):
        self.world = world
        patcher = mock.patch.object(kvnvme, 'default_p2p_engine', world.factory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def p2p_rig(self, nvme_env=None, **kwargs):
        env = dict(P2P)
        env.update(nvme_env or {})
        return self.rig(nvme_env=env, **kwargs)

    def object_bytes(self, made, key):
        entry = made.nvme.store.lookup(KV, key)
        with open(self.device, 'rb') as handle:
            handle.seek(entry.offset)
            return entry, handle.read(entry.nbytes)

    def corrupt(self, made, key, chip=1, byte=2):
        entry = made.nvme.store.lookup(KV, key)
        position = entry.offset + chip * made.nvme.shards.shard_bytes + byte
        with open(self.device, 'r+b') as handle:
            handle.seek(position)
            old = handle.read(1)
            handle.seek(position)
            handle.write(bytes([old[0] ^ 0xFF]))

    def evict(self, made):
        """Traffic that pushes every earlier session off the device pool and out of the host tier again (their NVMe copies stay)."""
        self.evictions = getattr(self, 'evictions', 0) + 1
        self.push_out(made, first_turn('ev%d' % self.evictions, 'evt%d' % self.evictions, seed=700 + self.evictions), fillers=5)

    def returning_exact(self, made, a, extra=700, name='again'):
        again = self.returning(a, extra, name)
        start, grant = made.serve(again)
        return again, start, grant


# ================================================================================================
# The flag
# ================================================================================================
class FlagTests(unittest.TestCase):
    def test_it_is_off_unless_named_and_its_knobs_are_inert_without_it(self):
        plan, problems = kvnvme.parse(ACKED)
        self.assertEqual((problems, plan.p2p, plan.p2pmem), ([], False, ()))
        for knob in ('QWEN_PREFIX_NVME_P2P_THREADS', 'QWEN_PREFIX_NVME_P2P_STAGING_MIB', 'QWEN_PREFIX_NVME_P2PMEM'):
            plan, problems = kvnvme.parse(dict(ACKED, **{knob: '4' if 'MEM' not in knob else '0000:88:00.0'}))
            self.assertIsNone(plan)
            self.assertTrue(any(knob in problem and 'would ignore it' in problem for problem in problems), (knob, problems))
        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_P2P='0'))
        self.assertEqual((problems, plan.p2p), ([], False))

    def test_on_it_takes_its_knobs_strictly(self):
        plan, problems = kvnvme.parse(dict(ACKED, **P2P))
        self.assertEqual((problems, plan.p2p, plan.p2p_threads, plan.p2p_staging_bytes >> 20, plan.restore), ([], True, 8, 128, 'host'))
        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_P2P='1', QWEN_PREFIX_NVME_P2P_THREADS='16', QWEN_PREFIX_NVME_P2P_STAGING_MIB='256',
                                           QWEN_PREFIX_NVME_P2PMEM='0000:92:00.0, 88:00.0'))
        self.assertEqual((problems, plan.p2p_threads, plan.p2p_staging_bytes, plan.p2pmem), ([], 16, 256 << 20, ('0000:88:00.0', '0000:92:00.0')))
        for name, value in (('QWEN_PREFIX_NVME_P2P', '2'), ('QWEN_PREFIX_NVME_P2P_THREADS', '0'), ('QWEN_PREFIX_NVME_P2P_THREADS', '65'), ('QWEN_PREFIX_NVME_P2P_STAGING_MIB', '1'),
                            ('QWEN_PREFIX_NVME_P2P_STAGING_MIB', '5000'), ('QWEN_PREFIX_NVME_P2P_STAGING_MIB', '1e2'), ('QWEN_PREFIX_NVME_P2PMEM', 'nonsense'),
                            ('QWEN_PREFIX_NVME_P2PMEM', '0000:88:00.0,')):
            plan, problems = kvnvme.parse(dict(ACKED, **dict(P2P, **{name: value})))
            self.assertIsNone(plan, (name, value))
            self.assertTrue(any(name in problem for problem in problems), (name, value, problems))

    def test_the_old_restore_name_points_at_the_new_flag(self):
        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_RESTORE='p2p'))
        self.assertIsNone(plan)
        self.assertTrue(any('QWEN_PREFIX_NVME_P2P=1' in problem for problem in problems), problems)

    def test_the_names_are_in_the_registrys_and_the_contracts_lists(self):
        import qwen_prefix_registry as registry_module

        new = {'QWEN_PREFIX_NVME_P2P', 'QWEN_PREFIX_NVME_P2P_THREADS', 'QWEN_PREFIX_NVME_P2P_STAGING_MIB', 'QWEN_PREFIX_NVME_P2PMEM'}
        self.assertTrue(new <= set(kvnvme.NAMES))
        self.assertEqual(registry_module.ENV_NVME_NAMES, kvnvme.NAMES)
        self.assertEqual(contract.PREFIX_NVME_NAMES, kvnvme.NAMES)
        self.assertEqual(contract.PREFIX_NVME_GATE_ONLY, kvnvme.GATE_ONLY)
        self.assertIn('QWEN_PREFIX_NVME_P2P', kvnvme.GATE_ONLY)
        self.assertNotIn('QWEN_PREFIX_NVME_P2P_THREADS', kvnvme.GATE_ONLY)

    def test_a_traffic_profile_cannot_carry_it_even_under_the_waiver_and_a_gate_profile_can(self):
        env = {'QWEN_PREFIX_REUSE': '1', 'QWEN_PREFIX_HOST_TIER_GIB': '32', 'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': '/dev/nvme1n1', 'QWEN_PREFIX_NVME_RETENTION_ACK': '1',
               'QWEN_PREFIX_NVME_P2P': '1'}
        gate = {'name': 'x-gate', 'env': dict(env), 'gate_only': True}
        problems = [problem for problem in contract.prefix_policy_problems(gate) if 'NVME' in problem or 'NVMe' in problem]
        self.assertEqual(problems, [])
        traffic = {'name': 'x-traffic', 'env': dict(env), 'owner_traffic_waiver': {
            'id': 'w', 'levers': {'QWEN_PREFIX_HOST_TIER_GIB': '32', 'QWEN_PREFIX_NVME': '1'}, 'waives': 'x', 'evidence': ['e'], 'decision': 'PENDING'}}
        problems = [problem for problem in contract.prefix_policy_problems(traffic)]
        self.assertTrue(any('QWEN_PREFIX_NVME_P2P' in problem and "a gate's own" in problem for problem in problems), problems)

    def test_a_knob_without_the_switch_is_a_problem_of_the_profile(self):
        problems = kvnvme.flag_problems(dict(ACKED, QWEN_PREFIX_NVME_P2P_THREADS='4'))
        self.assertTrue(any('QWEN_PREFIX_NVME_P2P_THREADS' in problem for problem in problems), problems)


class FingerprintTests(unittest.TestCase):
    class Io(object):
        fingerprint = 'kv-layout'
        block_bytes = 2228224
        chips = 4

    def components(self, **flags):
        plan, problems = kvnvme.parse(dict(ACKED, **flags))
        self.assertEqual(problems, [])
        return kvnvme.layout_fingerprint(self.Io(), plan, {})

    def test_the_flag_is_a_new_layout_and_so_a_new_fingerprint(self):
        off_fp, off = self.components()
        on_fp, on = self.components(**P2P)
        self.assertEqual((off['layout'], on['layout']), ('packed-v1', 'chipmajor-v1'))
        self.assertNotEqual(on_fp, off_fp)
        self.assertEqual(dict(on, layout=off['layout']), dict(off))

    def test_the_knobs_do_not_move_it(self):
        base, _ = self.components(**P2P)
        for knob in ({'QWEN_PREFIX_NVME_P2P_THREADS': '3'}, {'QWEN_PREFIX_NVME_P2P_STAGING_MIB': '64'}, {'QWEN_PREFIX_NVME_P2PMEM': '0000:88:00.0'}):
            self.assertEqual(self.components(**dict(P2P, **knob))[0], base, knob)


# ================================================================================================
# The stored bytes
# ================================================================================================
class StoredBytesTests(P2PCase):
    def test_with_the_flag_on_the_drive_holds_each_block_chip_major_with_its_checksums(self):
        made = self.p2p_rig()
        a = self.push_out(made, first_turn())
        hashes = a.block_hashes[:64]
        layout = made.nvme.shards
        stored = 0
        for key in hashes:
            if made.nvme.store.lookup(KV, key) is None:
                continue
            entry, data = self.object_bytes(made, key)
            stored += 1
            host = bytearray(kv_content(key))
            expected = bytearray(layout.block_bytes)
            layout.to_chipmajor(host, expected)
            self.assertEqual(data, bytes(expected))
            self.assertEqual(entry.digest, hashlib.sha256(expected).digest())
            self.assertEqual(layout.unpack_checksums(entry.aux), layout.checksums(expected))
            for chip in range(4):
                self.assertEqual(layout.unpack_checksums(entry.aux)[chip], p2p.p2p_checksum(layout.shard(expected, chip)))
        self.assertGreater(stored, 10)

    def test_with_the_flag_off_the_bytes_are_the_host_tiers_and_there_is_nothing_beside_the_digest(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        count = 0
        for key in a.block_hashes[:64]:
            if made.nvme.store.lookup(KV, key) is None:
                continue
            entry, data = self.object_bytes(made, key)
            self.assertEqual(data, kv_content(key))
            self.assertEqual(entry.aux, b'')
            count += 1
        self.assertGreater(count, 10)
        self.assertIsNone(made.nvme.shards)

    def test_the_aux_survives_a_restart_and_a_snapshot(self):
        made = self.p2p_rig()
        a = self.push_all_the_way(made, first_turn())
        key = a.block_hashes[7]
        before = made.nvme.store.lookup(KV, key).aux
        self.assertEqual(len(before), 32)
        made.nvme.store.snapshot()
        made.close()
        again = self.p2p_rig()
        self.assertEqual(again.nvme.store.lookup(KV, key).aux, before)
        self.assertGreater(again.registry.nvme.store.snapshot_stats()['entries'], 0)

    def test_a_drive_of_the_other_layout_is_dropped_at_open_never_restored(self):
        made = self.rig()                                                    # written packed-v1
        a = self.push_all_the_way(made, first_turn())
        made.close()
        again = self.p2p_rig()                                               # opened chipmajor-v1
        self.assertEqual(again.nvme.store.snapshot_stats()['kv_entries'], 0)
        self.assertTrue(any('another layout fingerprint' in line for line in again.lines), again.lines[-8:])

    def test_a_chip_major_block_read_back_through_the_host_path_is_the_host_tiers_block(self):
        made = self.p2p_rig()
        a = self.push_out(made, first_turn())
        keys = [key for key in a.block_hashes[:64] if made.nvme.store.lookup(KV, key) is not None]
        entries = made.nvme.pin_kv(keys)
        views = [bytearray(32) for _ in keys]
        results = made.nvme.read_into(entries, views)
        made.nvme.unpin(entries)
        self.assertEqual(set(results), {None})
        self.assertEqual([bytes(view) for view in views], [kv_content(key) for key in keys])

    def test_a_chip_major_block_that_fails_its_digest_is_refused_on_the_host_path_too(self):
        made = self.p2p_rig()
        a = self.push_out(made, first_turn())
        key = next(key for key in a.block_hashes[:64] if made.nvme.store.lookup(KV, key) is not None)
        self.corrupt(made, key)
        entries = made.nvme.pin_kv([key])
        results = made.nvme.read_into(entries, [bytearray(32)])
        made.nvme.unpin(entries)
        self.assertEqual(results, ['digest'])

    def test_the_batch_is_smaller_than_the_host_layouts_because_the_conversion_needs_a_second_buffer(self):
        self.assertLess(kvnvme.P2P_BATCH_BYTES, kvnvme.MAX_BATCH_BYTES)

    def test_the_tier_needs_the_ios_geometry_to_lay_blocks_out_chip_major(self):
        self.install(self.world)
        with mock.patch.object(tiers_module, 'FakeKvIO', FakeKvIO):
            made = self.p2p_rig()
        self.assertIsNone(made.registry.nvme)
        self.assertTrue(any('needs the tier IO' in line for line in made.lines), made.lines[-4:])
        self.assertEqual(made.registry.snapshot()['nvme_unavailable'], 1)


class RealGeometryTests(NvmeCase):
    """The production block (32 tensors x 4 chips x 17,408 bytes) through the tier's own spill and read paths, over the real 4 KiB alignment."""

    class Io(object):
        caches = ['t%d' % index for index in range(32)]
        chips = 4
        slice_bytes = 17408
        block_bytes = 32 * 4 * 17408
        fingerprint = 'kvtier1:32x4x17408'
        mesh_device = 'mesh'

    def test_the_spill_and_the_read_back_are_exact_and_aligned(self):
        import numpy as np

        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_DEVICE=self.device, QWEN_PREFIX_NVME_P2P='1', QWEN_PREFIX_NVME_REQUIRE_SED='0', QWEN_PREFIX_NVME_DRAIN_S='0',
                                           QWEN_PREFIX_NVME_QUEUE_BYTES=str(64 << 20), QWEN_PREFIX_NVME_THREADS='2'))
        self.assertEqual(problems, [])
        registry = type('R', (), {'tier_on': True, 'tier_config': None, 'tier': None})()
        with mock.patch.object(kvnvme, 'default_p2p_engine', lambda tier: (_ for _ in ()).throw(p2p.P2PUnavailable('no cards in a CPU test'))):
            tier = kvnvme.NvmeTier(plan, self.Io(), registry, logger=lambda *a: None)
        self.addCleanup(tier.close)
        rng = np.random.default_rng(3)
        blocks = [rng.integers(0, 256, self.Io.block_bytes, dtype=np.uint8) for _ in range(6)]
        items = [kvnvme.SpillKV(b'key%d' % index, memoryview(block), 'tenant', b'chain', index, None) for index, block in enumerate(blocks)]
        tier._write_batch(items)
        layout = tier.shards
        self.assertEqual(layout.shard_bytes, 136 * 4096)
        entries = tier.store.pin([(KV, b'key%d' % index) for index in range(6)])
        self.addCleanup(tier.store.unpin, entries)
        for index, entry in enumerate(entries):
            self.assertEqual(entry.offset % 4096, 0)
            for chip in range(4):
                self.assertEqual((entry.offset + chip * layout.shard_bytes) % 4096, 0)
            obj = tier.store.read_bytes(entry)
            expected = bytearray(layout.block_bytes)
            layout.to_chipmajor(blocks[index], expected)
            self.assertEqual(bytes(obj), bytes(expected))
            self.assertEqual(layout.unpack_checksums(entry.aux), layout.checksums(expected))
        views = [bytearray(self.Io.block_bytes) for _ in entries]
        self.assertEqual(set(tier.read_into(entries, views)), {None})
        for block, view in zip(blocks, views):
            self.assertEqual(bytes(view), block.tobytes())
        self.assertEqual(tier.c['nvme_p2p_unavailable'], 1)
        self.assertEqual(tier.backend.name, 'host')


# ================================================================================================
# The restore
# ================================================================================================
class RestoreTests(P2PCase):
    def test_a_returning_session_comes_back_through_the_cards_and_the_turn_is_exact(self):
        made = self.p2p_rig()
        self.assertEqual(made.nvme.backend.name, 'p2p')
        self.assertTrue(made.nvme.backend.direct)
        a = self.push_out(made, first_turn())
        records_before = len(made.registry.tier.records)
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertTrue(made.check_kv())
        snapshot = made.registry.snapshot()
        self.assertGreater(snapshot['nvme_p2p_blocks'], 0)
        self.assertEqual(snapshot['nvme_p2p_blocks'], snapshot['nvme_restore_blocks'])
        self.assertEqual(snapshot['nvme_p2p_mismatch'] + snapshot['nvme_p2p_errors'] + snapshot['nvme_p2p_fallbacks'], 0)
        self.assertEqual(snapshot['nvme_p2p_on'], 1)
        self.assertLessEqual(len(made.registry.tier.records), records_before, 'the NVMe blocks did not become host records')
        self.assertTrue(self.world.mesh.ops('write'))
        self.assertTrue(any(line.startswith('nvme tier p2p restore blocks=') for line in made.lines))

    def test_the_open_line_says_which_backend_and_the_p2p_line_what_it_found(self):
        made = self.p2p_rig()
        open_line = next(line for line in made.lines if line.startswith('nvme tier open '))
        self.assertIn('backend=p2p', open_line)
        on_line = next(line for line in made.lines if line.startswith('nvme tier p2p on '))
        for word in ('cards=4', 'threads=2', 'chunk_blocks=4', 'shard_bytes=8', 'mapping_verified=1', 'window=202'):
            self.assertIn(word, on_line)

    def test_a_flipped_byte_on_the_drive_is_a_miss_the_entry_is_dropped_and_nothing_is_written(self):
        made = self.p2p_rig()
        a = self.push_all_the_way(made, first_turn())
        self.corrupt(made, a.block_hashes[20], chip=2, byte=3)
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 0, 'the turn prefilled cold')
        self.assertTrue(made.check_kv())
        snapshot = made.registry.snapshot()
        self.assertGreaterEqual(snapshot['nvme_p2p_mismatch'], 1)
        self.assertGreaterEqual(snapshot['nvme_digest_failures'], 1)
        self.assertIsNone(made.nvme.store.lookup(KV, a.block_hashes[20]), 'the bad entry was dropped')
        self.assertEqual(made.nvme.backend.strikes, 0, 'a bad drive is not a strike against the path')
        self.assertTrue(made.nvme.backend.direct)
        self.assertTrue(any('CHECKSUM MISMATCH' in line for line in made.lines))

    def test_a_chunk_that_does_not_check_writes_nothing(self):
        made = self.p2p_rig()
        a = self.push_all_the_way(made, first_turn())
        key = a.block_hashes[2]
        self.corrupt(made, key, chip=0, byte=0)
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 0)
        wrote = {block for entry in self.world.mesh.ops('write') for block in entry[3]}
        for block in wrote:
            self.assertIn(block, made.kv.device)
        # the first chunk (blocks 0..3) holds the corrupted block: no write op was enqueued for any of its blocks
        self.assertEqual(self.world.mesh.ops('write'), [])

    def test_good_bytes_the_cards_disagree_with_are_a_strike_against_the_path_not_the_drive(self):
        made = self.p2p_rig()
        a = self.push_all_the_way(made, first_turn())
        self.world.mesh.bad_checksum = lambda chip, index, value: value ^ 4 if chip == 1 else value
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 0)
        self.assertIsNotNone(made.nvme.store.lookup(KV, a.block_hashes[20]), 'the drive\'s bytes verify: the entry stays')
        self.assertEqual(made.nvme.backend.strikes, 1)
        self.assertTrue(made.nvme.backend.direct)
        self.assertEqual(made.registry.snapshot()['nvme_digest_failures'], 0)
        self.world.mesh.bad_checksum = None
        self.evict(made)
        again2, start2, grant2 = self.returning_exact(made, a, name='third')
        self.assertEqual(start2, 4096, 'the next turn restores fine and the strikes start over')
        self.assertEqual(made.nvme.backend.strikes, 0)
        self.assertGreater(made.registry.snapshot()['nvme_p2p_blocks'], 0)
        self.assertTrue(made.check_kv())

    def test_three_mismatches_on_good_bytes_latch_the_path_off_and_the_host_records_take_over(self):
        made = self.p2p_rig()
        a = self.push_all_the_way(made, first_turn())
        self.world.mesh.bad_checksum = lambda chip, index, value: value ^ 1
        for number in range(3):
            self.evict(made)
            self.returning_exact(made, a, name='try%d' % number)
        backend = made.nvme.backend
        self.assertIsNotNone(backend.latched)
        self.assertFalse(backend.direct)
        self.assertEqual(made.registry.snapshot()['nvme_p2p_latched'], 1)
        self.assertEqual(made.registry.snapshot()['nvme_p2p_on'], 0)
        self.assertTrue(any('nvme tier p2p latched off' in line for line in made.lines))
        self.evict(made)
        again, start, grant = self.returning_exact(made, a, name='after')
        self.assertEqual(start, 4096, 'the host path restores it, through host records')
        self.assertTrue(made.check_kv())
        self.assertGreater(len(made.registry.tier.records), 0)
        self.assertTrue(made.registry.tier_on, 'the host tier did not latch: a mismatch is a miss, not three restore failures')

    def test_a_read_error_restores_the_same_blocks_through_the_host_path(self):
        made = self.p2p_rig()
        a = self.push_all_the_way(made, first_turn())
        entry = made.nvme.store.lookup(KV, a.block_hashes[6])
        fail = {'on': True}
        original = made.nvme.p2p_reader

        def reader(offset, view):
            if fail['on'] and entry.offset <= offset < entry.offset + 32:
                raise OSError(5, 'Input/output error')
            os.preadv(made.nvme.store.bf.fd, [view], offset)

        self.world.engines[-1].reader = reader
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 4096, 'never worse than a miss: here not even a miss')
        self.assertTrue(made.check_kv())
        snapshot = made.registry.snapshot()
        self.assertEqual((snapshot['nvme_p2p_errors'], snapshot['nvme_p2p_fallbacks']), (1, 1))
        self.assertEqual(made.nvme.backend.strikes, 1)
        self.assertTrue(original)

    def test_three_failures_in_a_row_latch_the_path_and_a_clean_restore_resets_the_count(self):
        made = self.p2p_rig()
        a = self.push_all_the_way(made, first_turn())
        engine = self.world.engines[-1]
        good = engine.reader
        engine.reader = lambda offset, view: (_ for _ in ()).throw(OSError(5, 'Input/output error'))
        for name in ('x1', 'x2'):
            self.evict(made)
            self.returning_exact(made, a, name=name)
        self.assertEqual(made.nvme.backend.strikes, 2)
        engine.reader = good
        self.evict(made)
        self.returning_exact(made, a, name='ok')
        self.assertEqual(made.nvme.backend.strikes, 0)
        engine.reader = lambda offset, view: (_ for _ in ()).throw(OSError(5, 'Input/output error'))
        for number in range(3):
            self.evict(made)
            self.returning_exact(made, a, name='y%d' % number)
        self.assertIsNotNone(made.nvme.backend.latched)
        self.assertTrue(made.check_kv())

    def test_a_device_op_error_latches_at_once_and_the_blocks_still_come_back(self):
        made = self.p2p_rig()
        a = self.push_all_the_way(made, first_turn())
        self.world.mesh.fail_write = True
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 4096)
        self.assertTrue(made.check_kv())
        self.assertIsNotNone(made.nvme.backend.latched)
        self.assertEqual(made.registry.snapshot()['nvme_p2p_fallbacks'], 1)

    def test_a_path_that_cannot_start_is_the_host_backend_with_one_line_and_a_counter(self):
        self.install(P2PWorld(fail=p2p.P2PUnavailable('the staging op is not in this ttnn')))
        made = self.p2p_rig()
        self.assertEqual(made.nvme.backend.name, 'host')
        self.assertFalse(made.nvme.backend.direct)
        self.assertEqual(made.registry.snapshot()['nvme_p2p_unavailable'], 1)
        self.assertEqual(made.registry.snapshot()['nvme_p2p_on'], 0)
        lines = [line for line in made.lines if 'p2p UNAVAILABLE' in line]
        self.assertEqual(len(lines), 1)
        self.assertIn('the staging op is not in this ttnn', lines[0])
        a = self.push_out(made, first_turn())
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 4096, 'the host backend reads the chip-major objects back into the host layout')
        self.assertTrue(made.check_kv())

    def test_a_surprising_exception_building_the_path_does_not_stop_the_engine(self):
        self.install(P2PWorld(fail=RuntimeError('a ttnn that blew up')))
        made = self.p2p_rig()
        self.assertEqual(made.nvme.backend.name, 'host')
        self.assertTrue(any('p2p UNAVAILABLE' in line and 'a ttnn that blew up' in line for line in made.lines))

    def test_a_session_split_between_the_host_tier_and_the_drive_restores_exactly(self):
        made = self.p2p_rig()
        a = self.push_out(made, first_turn())
        on_host = sum(1 for key in a.block_hashes[:64] if made.registry.tier.has(key))
        on_nvme = sum(1 for key in a.block_hashes[:64] if not made.registry.tier.has(key) and made.nvme.store.lookup(KV, key) is not None)
        self.assertEqual(on_host + on_nvme, 64)
        self.assertGreater(on_nvme, 0)
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 4096)
        self.assertTrue(made.check_kv())
        stats = made.registry.snapshot()
        self.assertEqual((stats['nvme_p2p_blocks'], stats['nvme_restore_blocks']), (on_nvme, on_nvme), 'the host tier\'s own records are written as before, the drive\'s go through the cards')

    def test_the_stop_closes_the_engine_and_its_links(self):
        made = self.p2p_rig()
        self.assertTrue(made.nvme.close())
        self.assertTrue(all(link.closed for link in self.world.links))
        self.assertTrue(self.world.engines[-1].closed)

    def test_the_spill_writer_yields_to_peer_to_peer_reads_in_flight(self):
        made = self.p2p_rig()
        engine = self.world.engines[-1]
        self.assertEqual(made.nvme.backend.reading(), 0)
        with engine.inflight_lock:
            engine.inflight = 3
        self.assertEqual(made.nvme.backend.reading(), 3)
        with engine.inflight_lock:
            engine.inflight = 0

    def test_the_audit_reads_the_restored_blocks_back_and_passes_on_good_bytes(self):
        made = self.p2p_rig(tier_env={'QWEN_PREFIX_HOST_TIER_AUDIT': '1'})
        a = self.push_out(made, first_turn())
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 4096)
        self.assertGreater(made.registry.snapshot()['tier_audit_reads'], 0)
        self.assertEqual(made.registry.snapshot()['tier_audit_mismatches'], 0)

    def test_the_audit_catches_a_copy_that_put_the_wrong_byte_in_the_cache(self):
        made = self.p2p_rig(tier_env={'QWEN_PREFIX_HOST_TIER_AUDIT': '1'})
        a = self.push_out(made, first_turn())
        made.kv.corrupt_copy = True
        with self.assertRaises(patch_module.TierAuditError):
            self.returning_exact(made, a)
        self.assertGreater(made.registry.snapshot()['tier_audit_mismatches'], 0)

    def test_random_traffic_with_the_path_on_stays_exact(self):
        import random

        made = self.p2p_rig(num_blocks=170, budget=1 << 20, tier=60)
        rng = random.Random(5)
        systems = {name: span(2300, 500 + index) for index, name in enumerate(('alpha', 'bravo', 'charlie'))}
        histories = {}
        from test_qwen_prefix_tiers import salt_of

        for step in range(30):
            name = rng.choice(['alpha', 'bravo', 'charlie'])
            base = histories.get(name, systems[name])
            tokens = list(base) + span(rng.randrange(1200, 3200), 4000 + 31 * step)
            made.serve(Request('r%d' % step, tokens, salt=salt_of(name)))
            self.assertTrue(made.check_kv())
            histories[name] = tokens if len(tokens) < 7000 else systems[name]
        made.settle()
        snapshot = made.registry.snapshot()
        self.assertGreater(snapshot['nvme_spill_blocks'], 0)
        self.assertEqual(snapshot['nvme_p2p_mismatch'], 0)
        self.assertEqual(snapshot['nvme_p2p_errors'], 0)


# ================================================================================================
# The sliced restore: beside the other seats
# ================================================================================================
SLICE_ENV = {'QWEN_PREFIX_TIER_RESTORE_SLICE': '16', 'QWEN_PREFIX_TIER_RESTORE_SLICE_MS': '60000'}


class SlicedP2PTests(P2PCase):
    def sliced(self, made, request):
        hooks = made.graft.tier_hooks
        hooks.hold_ready = True
        held, steps = 0, 0
        while hooks.hold(request) and steps < 3000:
            held += 1
            made.graft.step_no += 1
            steps += 1
            time.sleep(0.001)                                # (a scheduler step takes a good deal longer than the reads)
        self.assertLess(steps, 3000, 'the sliced restore did not end')
        return hooks, held, steps

    def test_a_sliced_restore_runs_its_reads_beside_the_steps_and_ends_exact(self):
        made = self.p2p_rig(tier_env=SLICE_ENV)
        a = self.push_out(made, first_turn())
        on_nvme = sum(1 for key in a.block_hashes[:64] if not made.registry.tier.has(key) and made.nvme.store.lookup(KV, key) is not None)
        again = self.returning(a)
        hooks, held, steps = self.sliced(made, again)
        self.assertGreater(held, 1)
        self.assertEqual(made.registry.tier.pinned, set())
        self.assertEqual(hooks.jobs, {})
        stats = made.registry.snapshot()
        self.assertEqual(stats['nvme_p2p_blocks'], on_nvme)
        self.assertGreater(stats['nvme_p2p_restores'], 1, 'one per slice')
        start, grant = made.serve(again)
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertTrue(made.check_kv())
        self.assertTrue(any('host tier restore slice req=again-a1' in line and 'p2p=' in line for line in made.lines))
        self.assertEqual(made.nvme.backend.engine.inflight, 0)

    def test_a_slice_whose_reads_have_not_landed_does_nothing_and_blocks_nobody(self):
        gate = threading.Event()
        world = P2PWorld(reader=None)
        self.install(world)
        made = self.p2p_rig(tier_env=SLICE_ENV)
        a = self.push_out(made, first_turn())
        self.addCleanup(gate.set)
        engine = world.engines[-1]
        direct = engine.reader
        engine.reader = lambda offset, view: (gate.wait(10), os.preadv(made.nvme.store.bf.fd, [view], offset))
        again = self.returning(a)
        hooks = made.graft.tier_hooks
        hooks.hold_ready = True
        self.assertTrue(hooks.hold(again))                              # the job starts, the reads are in flight
        made.graft.step_no += 1
        job = hooks.jobs[again.request_id]
        free_before = made.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        for _ in range(3):
            began = time.perf_counter()
            self.assertTrue(hooks.hold(again))
            made.graft.step_no += 1
            self.assertLess(time.perf_counter() - began, 1.0, 'the pass did not wait for the drive')
        self.assertEqual(len(job.new), 0, 'no block was taken for a chunk that has not landed')
        self.assertEqual(made.scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), free_before)
        self.assertEqual(self.world.mesh.ops('write'), [])
        gate.set()
        steps = 0
        while hooks.hold(again) and steps < 400:
            made.graft.step_no += 1
            steps += 1
        start, grant = made.serve(again)
        self.assertEqual(start, 4096)
        self.assertTrue(made.check_kv())
        self.assertTrue(direct)

    def test_a_mismatch_in_a_slice_abandons_the_job_and_the_turn_goes_cold(self):
        made = self.p2p_rig(tier_env=SLICE_ENV)
        a = self.push_all_the_way(made, first_turn())
        self.corrupt(made, a.block_hashes[40], chip=3, byte=1)
        again = self.returning(a)
        hooks, held, steps = self.sliced(made, again)
        self.assertTrue(any('slice aborted req=again-a1' in line and 'checksums differ' in line for line in made.lines), made.lines[-6:])
        start, grant = made.serve(again)
        self.assertEqual(start, 0)
        self.assertTrue(made.check_kv())
        self.assertEqual(made.registry.tier.pinned, set())
        self.assertGreaterEqual(made.registry.snapshot()['nvme_p2p_mismatch'], 1)

    def test_a_block_that_vanished_abandons_the_job(self):
        made = self.p2p_rig(tier_env=SLICE_ENV)
        a = self.push_out(made, first_turn())
        made.nvme.store.delete([(KV, a.block_hashes[40])])
        again = self.returning(a)
        hooks, held, steps = self.sliced(made, again)
        start, grant = made.serve(again)
        self.assertEqual(start, 0)
        self.assertTrue(made.check_kv())

    def test_a_read_error_in_a_slice_hands_the_rest_to_the_host_path(self):
        made = self.p2p_rig(tier_env=SLICE_ENV)
        a = self.push_all_the_way(made, first_turn())
        entry = made.nvme.store.lookup(KV, a.block_hashes[30])
        engine = self.world.engines[-1]

        def reader(offset, view):
            if entry.offset <= offset < entry.offset + 32:
                raise OSError(5, 'Input/output error')
            os.preadv(made.nvme.store.bf.fd, [view], offset)

        engine.reader = reader
        again = self.returning(a)
        hooks, held, steps = self.sliced(made, again)
        start, grant = made.serve(again)
        self.assertEqual(start, 4096, 'the slices after the failure came through the host records')
        self.assertTrue(made.check_kv())
        self.assertEqual(made.registry.snapshot()['nvme_p2p_errors'], 1)
        self.assertEqual(made.registry.tier.pinned, set())
        self.assertEqual(engine.inflight, 0)

    def test_a_path_that_latches_under_a_job_finishes_it_through_the_host_path(self):
        made = self.p2p_rig(tier_env=SLICE_ENV)
        a = self.push_out(made, first_turn())
        again = self.returning(a)
        hooks = made.graft.tier_hooks
        hooks.hold_ready = True
        self.assertTrue(hooks.hold(again))
        made.graft.step_no += 1
        made.nvme.backend.latch('test')
        steps = 0
        while hooks.hold(again) and steps < 400:
            made.graft.step_no += 1
            steps += 1
        start, grant = made.serve(again)
        self.assertEqual(start, 4096)
        self.assertTrue(made.check_kv())
        self.assertEqual(made.registry.tier.pinned, set())

    def test_a_stalled_drive_is_abandoned_and_the_path_latched(self):
        gate = threading.Event()
        made = self.p2p_rig(tier_env=SLICE_ENV)
        self.addCleanup(gate.set)                                    # (after the rig's own cleanup is registered: cleanups run last in, first out, and closing the rig waits for the reads)
        a = self.push_out(made, first_turn())
        engine = self.world.engines[-1]
        engine.reader = lambda offset, view: (gate.wait(30), os.preadv(made.nvme.store.bf.fd, [view], offset))
        again = self.returning(a)
        hooks = made.graft.tier_hooks
        hooks.hold_ready = True
        self.assertTrue(hooks.hold(again))
        made.graft.step_no += 1
        time.sleep(0.02)
        with mock.patch.object(patch_module, 'P2P_STALL_S', 0.01):
            self.assertFalse(hooks.hold(again), 'the job was abandoned')
        self.assertTrue(any('slice aborted req=again-a1' in line and 'stalled' in line for line in made.lines), made.lines[-4:])
        self.assertIsNotNone(made.nvme.backend.latched)
        gate.set()

    def test_the_trim_finishes_an_unfinished_job_whole_and_waits_for_the_reads(self):
        made = self.p2p_rig(tier_env=SLICE_ENV)
        a = self.push_out(made, first_turn())
        again = self.returning(a)
        hooks = made.graft.tier_hooks
        hooks.hold_ready = True
        self.assertTrue(hooks.hold(again))
        start, grant = made.serve(again)                                   # (the trim reaches a request whose job is unfinished: load() finishes it whole)
        self.assertEqual(start, 4096)
        self.assertTrue(made.check_kv())
        self.assertEqual(hooks.jobs, {})

    def test_the_slice_cap_bounds_the_chunk(self):
        made = self.p2p_rig(tier_env=dict(SLICE_ENV, QWEN_PREFIX_TIER_RESTORE_SLICE='3'))
        a = self.push_out(made, first_turn())
        again = self.returning(a)
        hooks, held, steps = self.sliced(made, again)
        sizes = [int(line.split('blocks=')[1].split()[0]) for line in made.lines if line.startswith('host tier restore slice req=again-a1')]
        self.assertTrue(sizes and max(sizes) <= 3, sizes)

    def test_without_the_slicing_flag_a_direct_backend_restores_whole_in_the_trim(self):
        made = self.p2p_rig()
        a = self.push_out(made, first_turn())
        again, start, grant = self.returning_exact(made, a)
        self.assertEqual(start, 4096)
        self.assertFalse(made.graft.tier_hooks.jobs)


class OtherDirectBackendsKeepTheirOldWay(NvmeCase):
    """A direct backend that does not slice (the double of test_qwen_kv_nvme) still restores whole: the sliced restore is offered only to a backend that says it slices."""

    def test_a_double_has_no_sliced_restore(self):
        made = self.rig(tier_env=dict(SLICE_ENV))
        made.nvme.backend = type('Double', (kvnvme.RestoreBackend,), {'name': 'double', 'direct': True, 'restore_to_device': lambda self, tier, entries, ids, io: None})()
        self.assertFalse(made.graft.tier_hooks.nvme_slices())
        made.nvme.backend = kvnvme.HostBounceBackend()
        self.assertTrue(made.graft.tier_hooks.nvme_slices())


class MetricsTests(P2PCase):
    def test_every_counter_is_present_and_documented_while_the_tier_is_on(self):
        import qwen_prefix_metrics as metrics

        made = self.p2p_rig()
        snapshot = made.registry.snapshot()
        for name in kvnvme.COUNTER_NAMES + kvnvme.GAUGE_NAMES:
            self.assertIn(name, snapshot)
        for name in [name for name in kvnvme.COUNTER_NAMES if name.startswith('nvme_p2p')]:
            self.assertIn(name, metrics.COUNTERS, name)
        for name in [name for name in kvnvme.GAUGE_NAMES if name.startswith('nvme_p2p')]:
            self.assertIn(name, metrics.GAUGES, name)
        self.assertEqual((snapshot['nvme_p2p_on'], snapshot['nvme_p2p_map_verified']), (1, 1))

    def test_the_trouble_markers_see_the_new_lines(self):
        import prefix_markers

        scan = prefix_markers.NVME_TROUBLE
        for line in ('nvme tier p2p UNAVAILABLE, the host backend serves', 'nvme tier p2p latched off: x', 'nvme tier p2p restore failed (OSError: x)', 'nvme tier p2p CHECKSUM MISMATCH on 1 shard'):
            self.assertTrue(any(marker in line for marker in scan), line)
        self.assertFalse(any(marker in 'nvme tier p2p restore blocks=3 bytes=96 ms=1.0' for marker in scan))


if __name__ == '__main__':
    unittest.main()
