"""P3a (tp4_p2p_probe) on the CPU: the drive's identity and R1's readings before a byte is written, then the whole flow on a fake ttnn, fake cards and a regular file for the drive.

Held:
  * the identity: the by-id link of the configured model and serial must resolve to the node, the kernel's own model and serial must agree, a link of that name pointing elsewhere or
    a kernel that says otherwise refuses, an unconfigured model or serial refuses, a plain node under a by-id name proves nothing by its name; the values never reach the log;
  * R1: the controller state and the namespace size are printed, and anything but `live` (or an empty namespace, a superblock of the NVMe tier) refuses with NOTHING written;
  * the flow: one card first, then every published card, byte-exact against what was written to the drive, timings, the negative arm (a flipped byte refused, nothing written, passes
    when put back), the windows released and every p2pmem pool whole again; a restore that corrupts, an arm that does not refuse, a pool left allocated: each is a FAIL;
  * the job wiring: C2_FABRIC_PROBE=p2p is a probe of the fabric action and the workflow step runs the script with the drive and the p2pmem directories, from a private file.
Run alone: python3 -B -m unittest test_tp4_p2p_probe (python 3.11+, numpy)
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import qwen_p2p_restore as p2p  # noqa: E402
import tp4_p2p_probe as probe  # noqa: E402
from test_qwen_p2p_restore import FakeMesh, build_sysfs, make_links  # noqa: E402

MODEL = 'Fake NVMe Model 9000 1TB'
SERIAL = 'FAKESERIAL123'
SLICE_BYTES = 17408


class FakeCache(object):
    def __init__(self, index):
        self.index = index


class FakeMeshDevice(object):
    def __init__(self, chips):
        self.chips = chips

    def get_num_devices(self):
        return self.chips


class ProbeTtnn(FakeMesh):
    """The probe's whole ttnn surface in memory: the mesh calls, the host tier's raw ops, and the device-op extension (FakeMesh)."""

    class FabricConfig(object):
        FABRIC_1D = 'f1d'
        FABRIC_1D_RING = 'f1dr'

    def __init__(self, layout, **kwargs):
        self.rows = {}                                    # (cache index, chip, block) -> bytes
        kwargs.setdefault('bdfs', [bdf for _index, bdf in ProbeCase.CARDS])
        FakeMesh.__init__(self, layout, self._sink, **kwargs)
        self.opened = []
        self.closed = []
        self.corrupt_copy = False

    def _sink(self, tensor, chip, block, data):
        if self.corrupt_copy and tensor.index == 3 and chip == 1:
            data = bytes([data[0] ^ 1]) + data[1:]
        self.rows[(tensor.index, chip, block)] = data

    def MeshShape(self, *shape):
        return shape

    def set_fabric_config(self, value):
        self.fabric = value

    def open_mesh_device(self, shape, **kwargs):
        self.opened.append(shape)
        return FakeMeshDevice(self.layout.chips)

    def close_mesh_device(self, mesh):
        self.closed.append(mesh)

    def qwen_block_bytes(self, cache):
        return SLICE_BYTES

    def qwen_read_blocks_raw(self, cache, out, ids):
        for chip in range(out.shape[0]):
            for position, block in enumerate(ids):
                out[chip, position] = np.frombuffer(self.rows.get((cache.index, chip, block), bytes(SLICE_BYTES)), dtype=np.uint8)

    def qwen_write_blocks_raw(self, cache, data, ids):
        raise AssertionError('the probe does not use the host path to write')


class ProbeCase(unittest.TestCase):
    CARDS = [(2, '0000:41:00.0'), (0, '0000:42:00.0'), (3, '0000:4a:00.0'), (1, '0000:4c:00.0')]

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='p3a-test-')
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.device = os.path.join(self.directory, 'nvme-node.img')
        with open(self.device, 'wb') as handle:
            handle.truncate(64 << 20)
        self.by_id = os.path.join(self.directory, 'by-id')
        os.makedirs(self.by_id)
        os.symlink(self.device, os.path.join(self.by_id, 'nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'))
        self.sysfs = os.path.join(self.directory, 'sys')
        self.sysfs_class, self.sysfs_pci = build_sysfs(self.sysfs, self.CARDS)
        for _index, bdf in self.CARDS:
            with open(os.path.join(self.sysfs_pci, bdf, 'current_link_width'), 'w') as handle:
                handle.write('16\n' if bdf == '0000:41:00.0' else '4\n')
        self.kernel = {'model': MODEL + '   ', 'serial': SERIAL, 'state': 'live', 'size': str(2 * 1024 * 1024 * 1024 // 512)}
        self.states = self.make_block_sysfs()
        patcher = mock.patch.object(probe, 'block_sysfs', lambda device, sysfs='/sys': self.block_dir if self.has_sysfs else None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.has_sysfs = True
        self.logs = []

    def make_block_sysfs(self):
        # /sys/block/nvmeXn1 with its controller above it (device -> ../nvme0): size on the namespace, state/model/serial on the controller
        ctrl = os.path.join(self.sysfs, 'nvme0')
        ns = os.path.join(ctrl, 'nvme0n1')
        os.makedirs(ns)
        self.block_dir = ns
        os.symlink(ctrl, os.path.join(ns, 'device'))
        self.write_kernel()

    def write_kernel(self):
        ctrl = os.path.dirname(self.block_dir)
        for name in ('model', 'serial', 'state'):
            with open(os.path.join(ctrl, name), 'w') as handle:
                handle.write(self.kernel[name] + '\n')
        with open(os.path.join(self.block_dir, 'size'), 'w') as handle:
            handle.write(self.kernel['size'] + '\n')

    def log(self, line):
        self.logs.append(line)

    def alloc(self, ttnn, mesh, blocks, heads):
        self.allocated.append(1)
        return FakeCache(len(self.allocated) - 1)

    def options(self, **overrides):
        args = ['--output', os.path.join(self.directory, 'out.json'), '--drive', os.path.join(self.by_id, 'nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'),
                '--drive-model', MODEL, '--drive-serial', SERIAL, '--by-id-dir', self.by_id, '--sysfs', self.sysfs, '--sysfs-pci', self.sysfs_pci, '--blocks', '24',
                '--caches', '4', '--staging-mib', '2', '--threads', '4', '--scratch-offset', str(1 << 20)]
        options = probe.parse_options(args)
        for name, value in overrides.items():
            setattr(options, name, value)
        return options

    def run_flow(self, options=None, ttnn=None, **kwargs):
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)
        ttnn = ttnn or ProbeTtnn(layout)
        self.ttnn = ttnn
        self.links = []

        def engine_factory(api, lay, drive, opts):
            link_factory, links = make_links(ttnn)
            self.links = links
            engine = p2p.P2PEngine(api, lay, drive.reader, threads=opts.threads, staging_bytes=opts.staging_mib << 20, link_factory=link_factory, io_align=4096,
                                   sleep=lambda seconds: None)
            self.engine = engine
            return engine

        defaults = dict(alloc=self.alloc, drive_factory=lambda path: probe.Drive(path, direct=False), engine_factory=engine_factory)
        self.allocated = []
        defaults.update(kwargs)
        report = probe.run(options or self.options(), ttnn=ttnn, environ={}, log=self.log, **defaults)
        return report


# ================================================================================================
# The drive: identity, R1
# ================================================================================================
class IdentityTests(ProbeCase):
    def check(self, **kwargs):
        args = dict(device=os.path.join(self.by_id, 'nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'), model=MODEL, serial=SERIAL, by_id_dir=self.by_id, sysfs=self.sysfs)
        args.update(kwargs)
        return probe.identity_check(**args)

    def test_the_by_id_link_and_the_kernel_agree(self):
        result = self.check()
        self.assertEqual(result['proof'], ['by-id', 'sysfs'])
        self.assertEqual(len(result['fingerprint']), 6)

    def test_the_by_id_link_alone_is_enough_when_the_kernel_cannot_say(self):
        self.has_sysfs = False
        self.assertEqual(self.check()['proof'], ['by-id'])

    def test_the_kernel_alone_is_enough_for_a_plain_node_in_a_container(self):
        node = os.path.join(self.directory, 'node')
        shutil.copy(self.device, node)
        os.remove(os.path.join(self.by_id, 'nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'))
        self.assertEqual(self.check(device=node)['proof'], ['sysfs'])
        self.has_sysfs = False
        with self.assertRaisesRegex(probe.Refusal, 'cannot be shown'):
            self.check(device=node)

    def test_a_name_without_a_link_proves_nothing(self):
        os.remove(os.path.join(self.by_id, 'nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'))
        shutil.copy(self.device, os.path.join(self.by_id, 'nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'))     # (a node under the name, as a container is handed one)
        self.has_sysfs = False
        with self.assertRaisesRegex(probe.Refusal, 'cannot be shown'):
            self.check(device=os.path.join(self.by_id, 'nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'))

    def test_a_link_of_that_name_that_points_elsewhere_refuses(self):
        other = os.path.join(self.directory, 'other.img')
        open(other, 'wb').close()
        with self.assertRaisesRegex(probe.Refusal, 'names another device'):
            self.check(device=other)

    def test_a_kernel_that_reports_another_model_or_serial_refuses_even_with_a_matching_link(self):
        self.kernel['serial'] = 'SOMEONEELSE'
        self.write_kernel()
        with self.assertRaisesRegex(probe.Refusal, 'another model or serial'):
            self.check()
        self.kernel.update(serial=SERIAL, model='Another Model')
        self.write_kernel()
        with self.assertRaisesRegex(probe.Refusal, 'another model or serial'):
            self.check()

    def test_an_unconfigured_model_or_serial_or_a_missing_node_refuses(self):
        for kwargs, why in (({'model': ''}, 'not configured'), ({'serial': None}, 'not configured'), ({'device': os.path.join(self.directory, 'nope')}, 'does not exist'),
                            ({'model': 'Other Model'}, 'cannot be shown|another')):
            with self.assertRaisesRegex(probe.Refusal, why):
                self.check(**kwargs)

    def test_model_names_are_written_the_way_udev_writes_them(self):
        self.assertEqual(probe.model_token('Fake NVMe Model 9000 1TB '), 'Fake_NVMe_Model_9000_1TB')
        self.assertTrue(probe.accepted_names(MODEL, SERIAL).fullmatch('nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'))
        self.assertTrue(probe.accepted_names(MODEL, SERIAL).fullmatch('nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123_1'))
        self.assertFalse(probe.accepted_names(MODEL, SERIAL).fullmatch('nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL1234'))
        self.assertFalse(probe.accepted_names(MODEL, SERIAL).fullmatch('nvme-eui.0123456789'))


class ReadingTests(ProbeCase):
    def test_the_readings_are_the_controller_state_and_the_namespace_size(self):
        readings = probe.r1_readings(self.options().drive, self.sysfs)
        self.assertEqual((readings['state'], readings['namespace_bytes'], readings['node']), ('live', 2 << 30, 'nvme0n1'))

    def test_a_flow_prints_them_before_anything_else_and_never_the_models_or_serials_name(self):
        report = self.run_flow()
        text = '\n'.join(self.logs)
        self.assertIn('P3A R1 controller_state=live namespace_bytes=2147483648', text)
        self.assertLess(text.index('R1 controller_state'), text.index('wrote '))
        self.assertLess(text.index('drive identity=verified'), text.index('R1 controller_state'))
        for secret in ('FAKESERIAL123', 'Fake NVMe Model', 'Fake_NVMe_Model', MODEL):
            self.assertNotIn(secret, text)
            self.assertNotIn(secret, json.dumps(report, default=str))
        for bdf in [bdf for _, bdf in self.CARDS]:
            self.assertNotIn(bdf, text)
            self.assertNotIn(bdf, json.dumps(report, default=str))

    def test_anything_but_live_is_a_refusal_and_nothing_is_written(self):
        for state in ('dead', 'resetting', 'connecting', 'deleting', 'unreadable'):
            self.kernel['state'] = state
            self.write_kernel()
            before = open(self.device, 'rb').read(1 << 22)
            report = self.run_flow()
            self.assertEqual(report['verdict'], 'NOT-MEASURED', state)
            self.assertIn('not live', report['refusal'])
            self.assertEqual(open(self.device, 'rb').read(1 << 22), before)
            self.assertEqual(self.ttnn.opened, [], 'the mesh was not even opened')
            self.assertTrue(any('REFUSED (nothing was written)' in line for line in self.logs))

    def test_an_empty_namespace_is_a_refusal(self):
        self.kernel['size'] = '0'
        self.write_kernel()
        self.assertIn('0 bytes', self.run_flow()['refusal'])

    def test_a_drive_that_carries_the_tiers_superblock_is_refused_unless_a_gate_says_it_is_its_to_destroy(self):
        with open(self.device, 'r+b') as handle:
            handle.write(b'QNVMSB01')
        report = self.run_flow()
        self.assertIn('superblock', report['refusal'])
        report = self.run_flow(self.options(allow_store_overwrite=True))
        self.assertEqual(report['verdict'], 'PASS', report.get('reasons'))

    def test_the_scratch_region_must_leave_the_tiers_first_mebibyte_alone_and_fit(self):
        for offset, why in ((4096, 'at least 1 MiB'), (1 << 20 | 1, 'multiple'), (3 << 30, 'does not fit')):
            report = self.run_flow(self.options(scratch_offset=offset))
            self.assertEqual(report['verdict'], 'NOT-MEASURED', offset)
            self.assertIn(why, report['refusal'])

    def test_a_missing_identity_refuses_before_the_mesh(self):
        report = self.run_flow(self.options(drive_serial=None))
        self.assertIn('not configured', report['refusal'])
        self.assertEqual(self.ttnn.opened, [])


# ================================================================================================
# The flow
# ================================================================================================
class FlowTests(ProbeCase):
    def test_the_whole_flow_passes_on_good_fakes_and_reports_what_a_gate_reads(self):
        report = self.run_flow()
        self.assertEqual((report['verdict'], report['reasons']), ('PASS', []))
        self.assertEqual(self.ttnn.opened, [(1, 4)])
        self.assertEqual(len(self.ttnn.closed), 1)
        self.assertEqual(report['exact'], dict(tensors=4, blocks=24, mismatched_rows=0))
        self.assertTrue(report['one_card']['verified'])
        self.assertEqual(report['first_card']['position'], 0, 'the widest link first: the x16 card (position 0 in the fake mesh)')
        for key in ('read', 'restore'):
            self.assertGreater(report[key]['gbps'], 0)
        self.assertIn('check_ms_per_chunk', report['restore'])
        self.assertEqual(report['negative'], dict(block=12, chip=3, refused=True, nothing_written=True, passes_after_the_byte_is_back=True))
        self.assertTrue(report['released']['links_closed'] and report['released']['available_equals_size'])
        self.assertEqual(report['engine']['cards'][0]['link_width'], 16)

    def test_the_known_object_is_on_the_drive_chip_major_and_the_cards_read_only_their_own_shards(self):
        report = self.run_flow()
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)
        payloads, arena, sums = probe.make_blocks(layout, 24)
        with open(self.device, 'rb') as handle:
            handle.seek(1 << 20)
            on_drive = handle.read(24 * layout.block_bytes)
        self.assertEqual(on_drive, bytes(arena[:24 * layout.block_bytes]))
        expected = bytearray(layout.block_bytes)
        layout.to_chipmajor(payloads[5], expected)
        self.assertEqual(on_drive[5 * layout.block_bytes:6 * layout.block_bytes], bytes(expected))
        self.assertEqual(report['layout']['shard_bytes'], 4 * SLICE_BYTES)

    def test_one_card_goes_first_and_alone(self):
        order = []
        original = probe.Drive.reader

        def reader(drive, offset, view):
            order.append(offset)
            return original(drive, offset, view)

        with mock.patch.object(probe.Drive, 'reader', reader):
            self.run_flow()
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)
        few = min(24, 2 * self.engine.chunk_blocks)
        chips = {((offset - (1 << 20)) % layout.block_bytes) // layout.shard_bytes for offset in order[:few]}
        self.assertEqual(len(order[:few]), few)
        self.assertEqual(chips, {0}, 'the first reads are one card\'s shards only')
        later = {((offset - (1 << 20)) % layout.block_bytes) // layout.shard_bytes for offset in order[few:]}
        self.assertEqual(later, {0, 1, 2, 3}, 'then every card\'s')

    def test_a_copy_that_puts_a_wrong_byte_in_the_cache_is_a_fail(self):
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)
        ttnn = ProbeTtnn(layout)
        ttnn.corrupt_copy = True
        report = self.run_flow(ttnn=ttnn)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertGreater(report['exact']['mismatched_rows'], 0)

    def test_an_arm_whose_flipped_byte_is_not_refused_is_a_fail(self):
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)
        ttnn = ProbeTtnn(layout)
        with mock.patch.object(probe.Drive, 'flip', lambda drive, offset: 0):                  # (the byte is never flipped: the restore passes where it should have been refused)
            with mock.patch.object(probe.Drive, 'put_byte', lambda drive, offset, value: None):
                report = self.run_flow(ttnn=ttnn)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertIn('negative arm', ' '.join(report['reasons']))
        self.assertFalse(report['negative']['refused'])

    def test_the_windows_are_taken_before_the_first_byte_is_written_and_an_error_names_no_drive_serial_or_card(self):
        before = open(self.device, 'rb').read(8 << 20)

        def refusing(api, lay, drive, opts):
            raise p2p.P2PUnavailable('TLB window 202 of 0000:41:00.0 is held by another process; node %s serial %s (%s)' % (
                opts.drive, SERIAL, os.path.basename(opts.drive)))

        report = self.run_flow(engine_factory=refusing)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertEqual(open(self.device, 'rb').read(8 << 20), before, 'not a byte of the object was written')
        text = json.dumps(report, default=str) + '\n'.join(self.logs)
        for secret in (SERIAL, MODEL, 'Fake_NVMe_Model', '0000:41:00.0', self.by_id, os.path.basename(self.options().drive)):
            self.assertNotIn(secret, text)
        self.assertIn('<pci>', report['error'])
        self.assertIn('<redacted>', report['error'])
        self.assertNotIn('write', report)

    def test_an_os_error_for_the_drive_path_is_scrubbed_too(self):
        def failing(path):
            raise OSError(13, 'Permission denied', path)

        report = self.run_flow(drive_factory=failing)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        text = json.dumps(report, default=str) + '\n'.join(self.logs)
        self.assertNotIn(self.by_id, text)
        self.assertNotIn('FAKESERIAL123', text)

    def test_the_scrubber_takes_out_drive_names_serials_and_pci_addresses_of_any_form(self):
        scrub = probe.scrubber(self.options())
        text = scrub('x /dev/disk/by-id/nvme-Whatever_123456_1 nvme-eui.0025385a12345678 %s at 41:00.0 and 0000:4A:00.3 model %s' % (SERIAL, MODEL))
        for gone in ('Whatever', 'eui.00253', SERIAL, MODEL, '41:00.0', '4A:00.3'):
            self.assertNotIn(gone, text)

    def test_a_mapping_the_kernel_could_not_confirm_is_a_fail_unless_a_gate_allows_it(self):
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)

        def unverified(api, lay, drive, opts):
            ttnn = api.ttnn
            link_factory, links = make_links(ttnn)

            def make(position, index, bdf, staging, ordering):
                link = link_factory(position, index, bdf, staging, ordering)
                link.map_verified = False
                return link

            return p2p.P2PEngine(api, lay, drive.reader, threads=opts.threads, staging_bytes=opts.staging_mib << 20, link_factory=make, io_align=4096, sleep=lambda s: None)

        report = self.run_flow(ttnn=ProbeTtnn(layout), engine_factory=unverified)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertIn('not confirmed against the BAR', ' '.join(report['reasons']))
        report = self.run_flow(self.options(allow_unverified_map=True), ttnn=ProbeTtnn(layout), engine_factory=unverified)
        self.assertEqual(report['verdict'], 'PASS', report.get('reasons'))

    def test_the_default_object_is_several_chunks_so_the_halves_rotate(self):
        options = probe.parse_options(['--output', 'x'])
        layout = p2p.ShardLayout(options.caches, 4, SLICE_BYTES)
        chunk_blocks = ((options.staging_mib << 20) // 2) // layout.shard_bytes
        self.assertGreater(options.blocks, 3 * chunk_blocks)

    def test_a_pool_left_allocated_is_a_fail(self):
        def leaky(engine, options):
            return dict(links_closed=True, available_equals_size=False)

        with mock.patch.object(probe, 'check_released', leaky):
            report = self.run_flow()
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertIn('not released', ' '.join(report['reasons']))

    def test_check_released_reads_the_pool_of_every_card(self):
        self.run_flow()
        released = probe.check_released(self.engine, self.options())
        self.assertTrue(released['available_equals_size'])
        bdf = self.CARDS[1][1]
        with open(os.path.join(self.sysfs_pci, bdf, 'p2pmem', 'available'), 'w') as handle:
            handle.write('0\n')
        self.assertFalse(probe.check_released(self.engine, self.options())['available_equals_size'])

    def test_a_ttnn_without_the_extension_is_a_refusal_before_the_mesh(self):
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)
        ttnn = ProbeTtnn(layout)
        ttnn.qwen_staging_checksum = None
        report = self.run_flow(ttnn=ttnn)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertIn('qwen_staging_checksum', report['refusal'])
        self.assertEqual(ttnn.opened, [])

    def test_the_mesh_is_closed_when_the_data_plane_fails(self):
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)
        ttnn = ProbeTtnn(layout)
        ttnn.fail_write = True
        report = self.run_flow(ttnn=ttnn)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertIn('error', report)
        self.assertEqual(len(ttnn.closed), 1)

    def test_a_shard_that_is_not_whole_pages_is_refused(self):
        report = self.run_flow(self.options(caches=6))
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertIn('multiple of 4', report['refusal'])

    def test_main_writes_the_report_and_exits_by_the_verdict(self):
        layout = p2p.ShardLayout(4, 4, SLICE_BYTES)
        ttnn = ProbeTtnn(layout)

        def engine_factory(api, lay, drive, opts):
            link_factory, links = make_links(ttnn)
            return p2p.P2PEngine(api, lay, drive.reader, threads=opts.threads, staging_bytes=opts.staging_mib << 20, link_factory=link_factory, io_align=4096, sleep=lambda s: None)

        out = os.path.join(self.directory, 'out.json')
        argv = ['--output', out, '--drive', os.path.join(self.by_id, 'nvme-Fake_NVMe_Model_9000_1TB_FAKESERIAL123'), '--drive-model', MODEL, '--drive-serial', SERIAL,
                '--by-id-dir', self.by_id, '--sysfs', self.sysfs, '--sysfs-pci', self.sysfs_pci, '--blocks', '24', '--caches', '4', '--staging-mib', '2', '--scratch-offset', str(1 << 20)]
        self.allocated = []
        status = probe.main(argv, ttnn=ttnn, environ={}, log=self.log, alloc=self.alloc, drive_factory=lambda path: probe.Drive(path, direct=False), engine_factory=engine_factory)
        self.assertEqual(status, 0)
        report = json.load(open(out))
        self.assertEqual((report['kind'], report['verdict']), ('p2p-p3a-quad', 'PASS'))
        self.assertTrue(self.logs[-1].startswith('{'))
        self.assertTrue(any(line.startswith('P3A verdict=PASS') for line in self.logs))

    def test_main_refuses_silly_numbers(self):
        out = os.path.join(self.directory, 'out.json')
        self.assertEqual(probe.main(['--output', out, '--blocks', '2']), 2)
        self.assertEqual(probe.main(['--output', out, '--caches', '6']), 2)


# ================================================================================================
# The partial stream the one-card stage uses
# ================================================================================================
class OneCardStreamTests(unittest.TestCase):
    def test_a_stream_over_some_chips_reads_and_checks_those_and_cannot_write(self):
        from test_qwen_p2p_restore import EngineCase

        case = EngineCase('build')
        case.setUp()
        try:
            case.build(blocks=5, chunk_blocks=3)
            stream = case.engine.start(case.drive.entries, chips=[2])
            chunk = stream.take(wait=True)
            stream.verify(chunk)
            with self.assertRaisesRegex(p2p.P2PError, 'writes every chip'):
                stream.write(chunk, [1, 2, 3])
            stream.release(chunk)
            stream.close()
            chips = {((offset) % case.layout.block_bytes) // case.layout.shard_bytes for offset, _ in case.drive.reads}
            self.assertEqual(chips, {2})
            for bad in ([], [7], [-1]):
                with self.assertRaises(ValueError):
                    case.engine.start(case.drive.entries, chips=bad)
        finally:
            case.doCleanups()

    def test_a_flipped_byte_on_another_chip_does_not_trouble_a_one_card_stream(self):
        from test_qwen_p2p_restore import EngineCase

        case = EngineCase('build')
        case.setUp()
        try:
            case.build(blocks=3, chunk_blocks=3)
            case.drive.flip(block=1, chip=3, byte=4)
            stream = case.engine.start(case.drive.entries, chips=[0])
            chunk = stream.take(wait=True)
            stream.verify(chunk)                               # (chip 3's share was never read: only chip 0's is compared)
            stream.release(chunk)
            stream.close()
        finally:
            case.doCleanups()


# ================================================================================================
# The job key
# ================================================================================================
class JobWiringTests(unittest.TestCase):
    WORKFLOW = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
    TEMPLATE = HERE / 'references' / 'tp4-p2p-jobs' / 'P3a-p2p-restore.env'

    def profiles(self):
        return sorted(json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles'])

    def test_p2p_is_a_probe_of_the_fabric_action(self):
        self.assertIn('p2p', job.FABRIC_PROBES)
        values = job.parse_env('C2_CARDS=quad\nC2_ACTIONS=fabric\nC2_FABRIC_PROBE=p2p\nC2_IMAGE_TAG=tp4-p2p-1\n')
        outputs = job.read_job(values, self.profiles(), root=ROOT)
        self.assertEqual((outputs['actions'], outputs['fabric_probe'], outputs['cards']), ('fabric', 'p2p', 'quad'))

    def test_a_pair_job_cannot_name_it(self):
        with self.assertRaises(job.JobError):
            job.read_job(job.parse_env('C2_CARDS=pair\nC2_ACTIONS=fabric\nC2_FABRIC_PROBE=p2p\nC2_IMAGE_TAG=tp4-p2p-1\n'), self.profiles(), root=ROOT)

    def test_the_template_parses_and_names_no_drive_card_or_host(self):
        text = self.TEMPLATE.read_text(encoding='utf-8')
        outputs = job.read_job(job.parse_env(text), self.profiles(), root=ROOT)
        self.assertEqual((outputs['actions'], outputs['fabric_probe']), ('fabric', 'p2p'))
        self.assertNotRegex(text, r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')
        self.assertNotRegex(text, r'nvme-[A-Za-z0-9]')
        for name in probe.ENV_NAMES:
            self.assertNotIn(name + '=', text, 'the drive and the cards come from the runner\'s private file, not from the job')

    def test_the_workflow_step_runs_the_script_with_the_drive_and_the_p2pmem_directories(self):
        self.assertIn('p2p) script=tp4_p2p_probe.py; report=p2p-p3a.json ;;', self.WORKFLOW)
        step = self.WORKFLOW[self.WORKFLOW.index('p2pargs=()'):]
        step = step[:step.index('timeout -k 30 "$limit" docker run')]
        for word in ('P2P_DRIVE_BY_ID', 'P2P_DRIVE_MODEL', 'P2P_DRIVE_SERIAL', 'P2P_CARDS', '/p2pmem', '--device', 'SYS_ADMIN', 'refusing'):
            self.assertIn(word, step)
        self.assertIn('${p2pargs[@]+"${p2pargs[@]}"}', self.WORKFLOW)
        self.assertIn('P3A', self.WORKFLOW[self.WORKFLOW.index("grep -E 'FABRIC_PROBE"):][:200])

    def test_the_step_reads_a_private_file_with_a_strict_parser_and_checks_the_identity_before_docker(self):
        step = self.WORKFLOW[self.WORKFLOW.index('p2pargs=()'):]
        step = step[:step.index('timeout -k 30 "$limit" docker run')]
        self.assertIn("P2P_ENV_FILE", step)
        self.assertIn('^(P2P_DRIVE_BY_ID|P2P_DRIVE_MODEL|P2P_DRIVE_SERIAL|P2P_CARDS)=', step)       # (a KEY=VALUE line parser, not `source`)
        self.assertNotIn('source ', step)
        self.assertNotIn('. "$', step)
        self.assertLess(step.index('readlink -f'), step.index('p2pargs=(--device'))

    def test_the_shell_block_refuses_what_it_should_before_docker_is_reached(self):
        import subprocess

        block = self.WORKFLOW[self.WORKFLOW.index('          p2pargs=()'):self.WORKFLOW.index('          # split builds the image')]
        import textwrap
        script = 'set -euo pipefail\nPROBE=p2p\n' + textwrap.dedent(block) + '\necho BLOCK-PASSED\n'
        directory = tempfile.mkdtemp(prefix='p3a-shell-')
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'step.sh')
        with open(path, 'w') as handle:
            handle.write(script)
        env_file = os.path.join(directory, 'private.env')

        def run(content=None, mode=0o600):
            if content is not None:
                with open(env_file, 'w') as handle:
                    handle.write(content)
                os.chmod(env_file, mode)
            result = subprocess.run(['bash', path], env={'PATH': os.environ['PATH'], 'HOME': directory, 'P2P_ENV_FILE': env_file}, capture_output=True, text=True, timeout=60)
            return result.returncode, result.stderr.strip()

        good = 'P2P_DRIVE_BY_ID=/dev/disk/by-id/nvme-Some_Model_SER1\nP2P_DRIVE_MODEL=Some Model\nP2P_DRIVE_SERIAL=SER1\nP2P_CARDS=0000:41:00.0\n'
        cases = [
            (None, 0o600, 'no private P3a file'),                                         # (the file does not exist yet)
            (good, 0o666, 'not the runner user'),
            (good.replace('SER1\nP2P_CARDS', 'SER1\nEVIL=$(touch pwned)\nP2P_CARDS'), 0o600, 'not one of its four names'),
            (good.replace('P2P_DRIVE_SERIAL=SER1\n', ''), 0o600, 'P2P_DRIVE_SERIAL is not set'),
            (good.replace('/dev/disk/by-id/nvme-Some_Model_SER1', '/dev/sda'), 0o600, 'must be named by a /dev/disk/by-id'),
            (good.replace('P2P_DRIVE_BY_ID=/dev/disk/by-id/nvme-Some_Model_SER1', 'P2P_DRIVE_BY_ID=/dev/disk/by-id/nvme-Not_There_SER1'), 0o600, 'does not resolve to a block device'),
        ]
        for content, mode, why in cases:
            code, err = run(content, mode)
            self.assertEqual(code, 1, (why, err))
            self.assertIn(why, err)
        self.assertFalse(os.path.exists(os.path.join(directory, 'pwned')))
        self.assertNotIn('BLOCK-PASSED', run(good)[1])

    def test_the_script_is_in_the_image_free_checkout_mount(self):
        # the fabric step mounts the checkout read-only at /c2 and runs /c2/scripts/ci/<script>: the probe needs no image rebuild, only an image with the device-op extension
        self.assertTrue((HERE / 'tp4_p2p_probe.py').exists())
        self.assertIn('-v "$PWD:/c2:ro"', self.WORKFLOW)


if __name__ == '__main__':
    unittest.main()
