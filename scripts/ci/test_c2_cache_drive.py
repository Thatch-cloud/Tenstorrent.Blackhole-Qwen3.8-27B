"""C2_NVME_DEVICE and C2_P2PMEM: the cache drive and the cards' peer-to-peer memory in a job's containers (c2_cache_drive.py; docs/tp4-kv-nvme.md section 12), held on CPU.

* the values: a /dev/disk/by-id path or the word `status`; PCI addresses or the word `auto`; an nvmeN name and anything else refused;
* the R1 readings, taken from a fake sysfs: the controller state the by-id link resolves to NOW and the namespace size, and the refusal of a drive that is not `live` or reads 0 bytes
  (the card reset's PCI rescan that disabled it once), the identity check against the status file, and a log that names neither the drive nor a card;
* the docker arguments each key produces, exactly (the device mapping to the fixed in-container path, the read-only status file, the one-file read-write p2pmem binds, the apparmor
  profile, the env), and nothing at all with both keys absent;
* the job reader (c2_serving_job): the outputs, quad only, only for the steps that use them, the tracked job file untouched;
* the workflow: the fabric, four-card smoke and prefix steps carry the keys and no other step does; the job step masks the literals; the argument file round-trips through bash;
* the prefix gate: the arguments reach `docker run` ahead of the entrypoint, are resolved per arm, a dead drive makes the arm INFRA and starts nothing, the recorded argv is
  redacted, and an arm on an NVMe profile without a drive is refused up front.

Stdlib only; the shell checks skip without bash."""

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c2_cache_drive as cache  # noqa: E402
import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
BASH = shutil.which('bash')
NAME = 'nvme-Example_Model_1TB_SERIAL0001'
POINT = '0000:05:00.0'
CARDS = ('0000:41:00.0', '0000:42:00.0')
BLOCK = os.makedev(259, 0)


class Stat(object):
    """What os.stat says about a block device node (the fake nodes are regular files)."""

    def __init__(self, mode=stat.S_IFBLK | 0o660, rdev=BLOCK):
        self.st_mode, self.st_rdev = mode, rdev


class Rig(object):
    """A fake host: /dev/disk/by-id with one link, a sysfs holding the drive's controller (state, size) and the cards' p2pmem files, and the boot unit's status file."""

    def __init__(self, test, state='live', sectors='1953525168', multipath=False, status=True, cards=CARDS, pci=POINT):
        self.dir = tempfile.mkdtemp(prefix='c2cd-')
        test.addCleanup(shutil.rmtree, self.dir, True)
        self.sysfs = os.path.join(self.dir, 'sys')
        self.by_id_dir = os.path.join(self.dir, 'by-id')
        self.node = os.path.join(self.dir, 'dev-nvme0n1')
        self.status_path = os.path.join(self.dir, 'status.json')
        os.makedirs(self.by_id_dir)
        open(self.node, 'wb').close()
        os.symlink(self.node, os.path.join(self.by_id_dir, NAME))
        self.by_id = os.path.join(self.by_id_dir, NAME)
        pci_dir = os.path.join(self.sysfs, 'devices', 'pci0000:80', '0000:80:01.1', pci)
        controller = os.path.join(pci_dir, 'nvme', 'nvme0')
        self.namespace = os.path.join(controller, 'nvme0n1')
        os.makedirs(self.namespace)
        self.write(os.path.join(controller, 'state'), state)
        self.write(os.path.join(self.namespace, 'size'), sectors)
        os.symlink('..', os.path.join(self.namespace, 'device'))
        os.makedirs(os.path.join(self.sysfs, 'dev', 'block'))
        os.symlink(self.namespace, os.path.join(self.sysfs, 'dev', 'block', '259:0'))
        if multipath:
            # the head namespace lives under the subsystem, whose children are the controllers (symlinks): nothing above it is called nvmeN
            subsystem = os.path.join(self.sysfs, 'devices', 'virtual', 'nvme-subsystem', 'nvme-subsys0')
            head = os.path.join(subsystem, 'nvme0n1')
            os.makedirs(head)
            self.write(os.path.join(head, 'size'), sectors)
            os.symlink(controller, os.path.join(subsystem, 'nvme0'))
            os.remove(os.path.join(self.sysfs, 'dev', 'block', '259:0'))
            os.symlink(head, os.path.join(self.sysfs, 'dev', 'block', '259:0'))
        for address in cards:
            directory = os.path.join(self.sysfs, 'bus', 'pci', 'devices', address, 'p2pmem')
            os.makedirs(directory)
            open(os.path.join(directory, 'allocate'), 'wb').close()
        if status:
            self.write(self.status_path, json.dumps(dict(by_id=NAME, pci_addr=pci, sed='opal', locked=False, locking_enabled=True, state='ok', timestamp='2026-01-02T03:04:05Z')))

    @staticmethod
    def write(path, text):
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)

    def plan(self, nvme='', p2pmem=(), **kwargs):
        return cache.plan(nvme, p2pmem, status_path=self.status_path, sysfs=self.sysfs, stat_fn=lambda path: Stat(), **kwargs)


class ValueTests(unittest.TestCase):
    def test_the_nvme_value_is_a_by_id_path_or_the_word_status_or_empty(self):
        self.assertEqual(cache.read_nvme_spec(''), '')
        self.assertEqual(cache.read_nvme_spec(None), '')
        self.assertEqual(cache.read_nvme_spec(' status '), 'status')
        self.assertEqual(cache.read_nvme_spec('/dev/disk/by-id/' + NAME), '/dev/disk/by-id/' + NAME)
        self.assertEqual(cache.read_nvme_spec('/dev/disk/by-id/nvme-eui.0025385a21404c5b'), '/dev/disk/by-id/nvme-eui.0025385a21404c5b')

    def test_an_nvme_name_or_any_other_path_is_refused_because_the_kernel_renumbers_them(self):
        for bad in ('/dev/nvme0n1', 'nvme0n1', '/dev/nvme0', NAME, '/dev/disk/by-id/', '/dev/disk/by-id/../../nvme0n1', '/dev/disk/by-id/a/b', '/dev/disk/by-path/pci-0000:05:00.0-nvme-1',
                    '/dev/disk/by-id/-x', '/dev/disk/by-id/a b', '/dev/disk/by-id/a;b', 'STATUS'):
            with self.subTest(value=bad):
                with self.assertRaises(cache.CacheDriveError):
                    cache.read_nvme_spec(bad)

    def test_the_p2pmem_value_is_pci_addresses_or_auto_or_empty(self):
        self.assertEqual(cache.read_p2pmem_spec(''), ())
        self.assertEqual(cache.read_p2pmem_spec('auto'), 'auto')
        self.assertEqual(cache.read_p2pmem_spec('0000:41:00.0'), ('0000:41:00.0',))
        self.assertEqual(cache.read_p2pmem_spec('0000:41:00.0, 0000:42:00.0'), ('0000:41:00.0', '0000:42:00.0'))
        self.assertEqual(cache.read_p2pmem_spec('0000:4E:00.0'), ('0000:4e:00.0',), 'lower-cased: the sysfs names are')
        for bad in ('41:00.0', '0000:41:00', '0000:41:00.8', 'all', 'auto,0000:41:00.0', '0000:41:00.0,0000:41:00.0', '0000:41:00.0,', ','.join('0000:41:00.%d' % (i % 8) + str(i) for i in range(3))):
            with self.subTest(value=bad):
                with self.assertRaises(cache.CacheDriveError):
                    cache.read_p2pmem_spec(bad)
        with self.assertRaises(cache.CacheDriveError):
            cache.read_p2pmem_spec(','.join('0000:%02x:00.0' % i for i in range(cache.MAX_PCI_ADDRESSES + 1)))

    def test_the_status_word_takes_the_name_from_the_status_file_as_a_bare_name_or_a_full_path(self):
        rig = Rig(self)
        self.assertEqual(cache.by_id_of('status', rig.status_path), '/dev/disk/by-id/' + NAME)
        Rig.write(rig.status_path, json.dumps(dict(by_id='/dev/disk/by-id/' + NAME)))
        self.assertEqual(cache.by_id_of('status', rig.status_path), '/dev/disk/by-id/' + NAME)
        self.assertEqual(cache.by_id_of('/dev/disk/by-id/other', rig.status_path), '/dev/disk/by-id/other', 'a literal is taken as it is')
        for text in (json.dumps(dict(by_id='')), json.dumps(dict(by_id='/dev/nvme0n1')), json.dumps(dict(by_id='a/b')), json.dumps([1]), 'not json', json.dumps({})):
            Rig.write(rig.status_path, text)
            with self.assertRaises(cache.CacheDriveError):
                cache.by_id_of('status', rig.status_path)
        os.remove(rig.status_path)
        with self.assertRaises(cache.CacheDriveError):
            cache.by_id_of('status', rig.status_path)


class ReadingTests(unittest.TestCase):
    def readings(self, rig):
        return cache.drive_readings(rig.by_id, rig.sysfs, stat_fn=lambda path: Stat())

    def test_a_live_controller_and_a_namespace_above_zero_are_usable(self):
        rig = Rig(self)
        readings = self.readings(rig)
        self.assertEqual((readings['state'], readings['bytes'], readings['node']), ('live', 1953525168 * 512, os.path.realpath(rig.node)))
        self.assertEqual(cache.readings_problems(readings), [])
        self.assertEqual(len(readings['controllers']), 1)

    def test_every_state_but_live_is_refused(self):
        for state in ('dead', 'deleting', 'resetting', 'connecting', 'new', 'adminq'):
            with self.subTest(state=state):
                readings = self.readings(Rig(self, state=state))
                self.assertEqual(readings['state'], state)
                self.assertTrue(any('not live' in text for text in cache.readings_problems(readings)))

    def test_a_namespace_that_reads_zero_bytes_is_refused_as_after_the_rescan(self):
        readings = self.readings(Rig(self, state='dead', sectors='0'))
        problems = cache.readings_problems(readings)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any('0 bytes' in text for text in problems))
        self.assertEqual(cache.readings_problems(self.readings(Rig(self, sectors='0'))), ['the cache drive\'s namespace reads 0 bytes'])
        self.assertEqual(cache.readings_problems(self.readings(Rig(self, sectors='garbage')))[0], 'the cache drive\'s namespace reads 0 bytes')

    def test_a_drive_whose_controller_cannot_be_read_is_not_live(self):
        rig = Rig(self)
        os.remove(os.path.join(os.path.dirname(rig.namespace), 'state'))
        readings = self.readings(rig)
        self.assertEqual((readings['state'], readings['controllers']), ('unreadable', []))
        self.assertTrue(cache.readings_problems(readings))
        shutil.rmtree(os.path.join(rig.sysfs, 'dev'))
        os.makedirs(os.path.join(rig.sysfs, 'dev', 'block'))
        self.assertEqual(self.readings(rig)['state'], 'unreadable', 'no sysfs entry for the device number')

    def test_a_multipath_head_namespace_finds_its_controller_under_the_subsystem(self):
        readings = self.readings(Rig(self, multipath=True))
        self.assertEqual((readings['state'], readings['bytes'] > 0, len(readings['controllers'])), ('live', True, 1))
        dead = self.readings(Rig(self, multipath=True, state='dead'))
        self.assertEqual(dead['state'], 'dead')

    def test_a_link_that_is_not_a_block_device_is_refused(self):
        rig = Rig(self)
        with self.assertRaises(cache.CacheDriveError):
            cache.drive_readings(rig.by_id, rig.sysfs, stat_fn=lambda path: Stat(mode=stat.S_IFREG | 0o644))
        with self.assertRaises(cache.CacheDriveError):
            cache.drive_readings(os.path.join(rig.dir, 'no-such-link'), rig.sysfs)

    def test_the_link_is_resolved_every_time_never_remembered(self):
        rig = Rig(self)
        first = self.readings(rig)
        other = os.path.join(rig.dir, 'dev-nvme3n1')
        open(other, 'wb').close()
        os.remove(rig.by_id)
        os.symlink(other, rig.by_id)                            # the kernel renumbered the drive: the same link names another node now
        second = self.readings(rig)
        self.assertNotEqual(first['node'], second['node'])
        self.assertEqual(second['node'], os.path.realpath(other))

    def test_the_status_files_pci_address_must_be_on_the_resolved_devices_path(self):
        rig = Rig(self)
        readings = self.readings(rig)
        with open(rig.status_path) as handle:
            status = json.loads(handle.read())
        self.assertEqual(cache.identity_problems(status, readings)[0], [])
        self.assertEqual(cache.identity_problems(dict(status, pci_addr='0000:06:00.0'), readings)[1], 'mismatch')
        self.assertTrue(cache.identity_problems(dict(status, pci_addr='0000:06:00.0'), readings)[0])
        self.assertEqual(cache.identity_problems(dict(status, pci_addr=None), readings)[0], [], 'a file with no address checks nothing')
        self.assertEqual(cache.identity_problems(dict(status, pci_addr=None, pci=POINT), readings)[0], [], 'the first design\'s field name')
        self.assertTrue(cache.identity_problems(dict(status, pci_addr=None, pci='0000:06:00.0'), readings)[0])

    def test_a_multipath_drive_is_identified_through_its_controller(self):
        rig = Rig(self, multipath=True)
        readings = self.readings(rig)
        self.assertNotIn(POINT, readings['sysfs_path'], 'the head namespace has no PCI function in its own path')
        status = {'pci_addr': POINT}
        self.assertEqual(cache.identity_problems(status, readings)[0], [])
        self.assertTrue(cache.identity_problems({'pci_addr': '0000:06:00.0'}, readings)[0])

    def test_the_line_for_the_log_names_neither_the_drive_nor_a_card(self):
        rig = Rig(self)
        text = cache.readings_line(self.readings(rig))
        self.assertIn('controller state=live', text)
        self.assertIn('namespace_bytes=1000204886016', text)
        for private in (NAME, 'nvme0', POINT, rig.dir):
            self.assertNotIn(private, text)


class PlanTests(unittest.TestCase):
    def test_both_keys_absent_is_nothing(self):
        rig = Rig(self)
        result = rig.plan()
        self.assertEqual((result['arguments'], result['lines'], result['env']), ([], [], {}))

    def test_the_nvme_key_maps_the_resolved_device_to_the_fixed_path_and_mounts_the_status_file_read_only(self):
        rig = Rig(self)
        result = rig.plan(nvme=rig.by_id)
        node = os.path.realpath(rig.node)
        self.assertEqual(result['arguments'], ['--device', '%s:/dev/c2-cache-drive:rwm' % node,
                                               '--mount', 'type=bind,src=%s,dst=%s,readonly' % (rig.status_path, rig.status_path),
                                               '-e', 'QWEN_PREFIX_NVME_DEVICE=/dev/c2-cache-drive'])
        self.assertEqual(result['env'], {'QWEN_PREFIX_NVME_DEVICE': '/dev/c2-cache-drive'})
        self.assertEqual(cache.IN_CONTAINER_DEVICE, '/dev/c2-cache-drive')
        self.assertEqual(len(result['lines']), 2)
        self.assertTrue(result['lines'][0].startswith('R1 cache drive: controller state=live'))
        self.assertEqual(result['lines'][1], 'R1 identity: the status file\'s pci address is on the resolved device\'s path')

    def test_the_status_word_resolves_through_the_files_name(self):
        rig = Rig(self)
        asked = []

        def readings_of(path):
            asked.append(path)                    # /dev/disk/by-id/<name> does not exist on a test host: the link is handed in
            return cache.drive_readings(rig.by_id, rig.sysfs, stat_fn=lambda p: Stat())

        result = rig.plan(nvme='status', readings_of=readings_of)
        self.assertEqual(asked, ['/dev/disk/by-id/' + NAME])
        self.assertEqual(result['arguments'][0], '--device')

    def test_a_drive_that_is_not_live_refuses_the_job_before_any_argument_exists(self):
        for state, sectors in (('dead', '0'), ('dead', '1953525168'), ('resetting', '1953525168'), ('live', '0')):
            with self.subTest(state=state, sectors=sectors):
                rig = Rig(self, state=state, sectors=sectors)
                with self.assertRaises(cache.CacheDriveError) as caught:
                    rig.plan(nvme=rig.by_id)
                for private in (NAME, POINT, 'nvme0'):
                    self.assertNotIn(private, str(caught.exception))

    def test_a_status_file_about_another_drive_refuses_the_job(self):
        rig = Rig(self)
        Rig.write(rig.status_path, json.dumps(dict(by_id=NAME, pci_addr='0000:06:00.0', sed='opal')))
        with self.assertRaises(cache.CacheDriveError) as caught:
            rig.plan(nvme=rig.by_id)
        self.assertIn('another drive', str(caught.exception))

    def test_without_a_readable_status_file_the_drive_still_maps_but_nothing_is_mounted(self):
        rig = Rig(self, status=False)
        result = rig.plan(nvme=rig.by_id)
        self.assertEqual([token for token in result['arguments'] if token == '--mount'], [])
        self.assertTrue(any('not readable' in line for line in result['lines']))
        with self.assertRaises(cache.CacheDriveError):
            rig.plan(nvme='status')              # the word needs the file

    def test_the_p2pmem_key_binds_one_file_per_card_read_write_and_drops_the_default_apparmor_profile(self):
        rig = Rig(self)
        result = rig.plan(p2pmem=CARDS)
        expected = []
        for address in CARDS:
            path = '/sys/bus/pci/devices/%s/p2pmem/allocate' % address
            expected += ['--mount', 'type=bind,src=%s,dst=%s' % (path, path)]
        expected += ['--security-opt', 'apparmor=unconfined', '-e', 'QWEN_P2PMEM_DEVICES=' + ','.join(CARDS)]
        self.assertEqual(result['arguments'], expected)
        self.assertNotIn('--privileged', result['arguments'])
        self.assertFalse([token for token in result['arguments'] if token.startswith('--cap-add')], 'no capability is added')
        self.assertFalse([token for token in result['arguments'] if 'readonly' in token or 'ro' == token], 'the binds are read-write: the whole point')
        self.assertEqual(result['env'], {'QWEN_P2PMEM_DEVICES': ','.join(CARDS)})
        self.assertEqual(result['lines'], ['p2pmem: 2 card(s), p2pmem/allocate bound read-write (one file each), the container runs with apparmor=unconfined (not privileged)'])
        for address in CARDS:
            self.assertNotIn(address, result['lines'][0])

    def test_auto_takes_every_function_that_publishes_p2pmem(self):
        rig = Rig(self, cards=('0000:42:00.0', '0000:41:00.0'))
        os.makedirs(os.path.join(rig.sysfs, 'bus', 'pci', 'devices', '0000:44:00.0'))          # a card with no p2pmem directory
        os.makedirs(os.path.join(rig.sysfs, 'bus', 'pci', 'devices', 'not-an-address', 'p2pmem'))
        result = rig.plan(p2pmem='auto')
        self.assertEqual(result['env'], {'QWEN_P2PMEM_DEVICES': '0000:41:00.0,0000:42:00.0'}, 'sorted')
        bare = Rig(self, cards=())
        with self.assertRaises(cache.CacheDriveError) as caught:
            bare.plan(p2pmem='auto')
        self.assertIn('no PCI function', str(caught.exception))

    def test_a_named_card_that_does_not_publish_p2pmem_refuses_the_job_without_naming_it(self):
        rig = Rig(self)
        with self.assertRaises(cache.CacheDriveError) as caught:
            rig.plan(p2pmem=('0000:41:00.0', '0000:44:00.0'))
        self.assertIn('1 of the 2 cards', str(caught.exception))
        self.assertNotIn('0000:44:00.0', str(caught.exception))

    def test_both_keys_together_add_both_sets_in_a_fixed_order(self):
        rig = Rig(self)
        result = rig.plan(nvme=rig.by_id, p2pmem=CARDS[:1])
        self.assertEqual(result['arguments'][0], '--device')
        self.assertEqual(result['arguments'][-4:-2], ['--security-opt', 'apparmor=unconfined'])
        self.assertEqual(sorted(result['env']), ['QWEN_P2PMEM_DEVICES', 'QWEN_PREFIX_NVME_DEVICE'])


class CommandTests(unittest.TestCase):
    def run_main(self, rig, *argv):
        out, err = io.StringIO(), io.StringIO()
        code = cache.main(list(argv) + ['--status-file', rig.status_path], out=out, err=err, sysfs=rig.sysfs, stat_fn=lambda path: Stat(), readings_of=lambda path: cache.drive_readings(rig.by_id, rig.sysfs, stat_fn=lambda p: Stat()))
        return code, out.getvalue(), err.getvalue()

    def test_args_writes_nul_separated_arguments_and_prints_only_the_readings(self):
        rig = Rig(self)
        target = os.path.join(rig.dir, 'cache.args')
        code, out, err = self.run_main(rig, 'args', '--nvme-device', '/dev/disk/by-id/' + NAME, '--p2pmem', ','.join(CARDS), '--out', target)
        self.assertEqual((code, err), (0, ''))
        with open(target, 'rb') as handle:
            data = handle.read()
        self.assertTrue(data.endswith(b'\0'))
        tokens = data.decode().split('\0')[:-1]
        self.assertEqual(tokens[:2], ['--device', '%s:/dev/c2-cache-drive:rwm' % os.path.realpath(rig.node)])
        self.assertIn('QWEN_P2PMEM_DEVICES=' + ','.join(CARDS), tokens)
        self.assertTrue(all(line.startswith('[C2-CACHE-DRIVE] ') for line in out.splitlines()))
        for private in (NAME, POINT) + CARDS:
            self.assertNotIn(private, out)

    def test_a_refusal_exits_1_with_the_reason_and_writes_nothing(self):
        rig = Rig(self, state='dead', sectors='0')
        target = os.path.join(rig.dir, 'cache.args')
        code, out, err = self.run_main(rig, 'args', '--nvme-device', '/dev/disk/by-id/' + NAME, '--out', target)
        self.assertEqual(code, 1)
        self.assertIn('REFUSED', err)
        self.assertIn('not live', err)
        self.assertFalse(os.path.exists(target))
        for private in (NAME, POINT):
            self.assertNotIn(private, out + err)
        code, _, err = self.run_main(rig, 'args', '--nvme-device', '/dev/nvme0n1', '--out', target)
        self.assertEqual(code, 1)
        self.assertIn('by-id', err)

    def test_check_prints_and_writes_nothing(self):
        rig = Rig(self)
        code, out, _ = self.run_main(rig, 'check', '--p2pmem', 'auto')
        self.assertEqual(code, 0)
        self.assertIn('p2pmem: 2 card(s)', out)

    def test_mask_emits_a_command_for_each_literal_and_none_for_the_words(self):
        outputs = 'cards=quad\nnvme_device=/dev/disk/by-id/%s\npeer_memory=0000:41:00.0,0000:42:00.0\n' % NAME
        self.assertEqual(cache.mask_lines(outputs), ['::add-mask::/dev/disk/by-id/' + NAME, '::add-mask::' + NAME, '::add-mask::0000:41:00.0', '::add-mask::0000:42:00.0'])
        self.assertEqual(cache.mask_lines('nvme_device=status\npeer_memory=auto\n'), [])
        self.assertEqual(cache.mask_lines('nvme_device=\npeer_memory=\n'), [])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'out')
            with open(path, 'w') as handle:
                handle.write(outputs)
            out = io.StringIO()
            self.assertEqual(cache.main(['mask', path], out=out), 0)
            self.assertEqual(out.getvalue().count('::add-mask::'), 4)
            self.assertEqual(cache.main(['mask', os.path.join(directory, 'missing')], out=io.StringIO()), 0, 'no outputs file: nothing to mask, not a failure')

    def test_redact_removes_the_cards_and_the_host_device_node_from_a_recorded_argv(self):
        argv = ['docker', 'run', '--device', '/dev/nvme3n1:/dev/c2-cache-drive:rwm', '--device', '/dev/tenstorrent/1', '--mount',
                'type=bind,src=/sys/bus/pci/devices/0000:41:00.0/p2pmem/allocate,dst=/sys/bus/pci/devices/0000:41:00.0/p2pmem/allocate', '-e', 'QWEN_P2PMEM_DEVICES=0000:41:00.0,0000:42:00.0']
        redacted = cache.redact(argv)
        text = ' '.join(redacted)
        for private in ('0000:41:00.0', '0000:42:00.0', 'nvme3n1'):
            self.assertNotIn(private, text)
        self.assertIn('<drive>:/dev/c2-cache-drive:rwm', text)
        self.assertIn('/dev/tenstorrent/1', text, 'the cards\' own device nodes are not private and stay')
        self.assertEqual(cache.redact(['docker', 'run', '-e', 'A=1']), ['docker', 'run', '-e', 'A=1'])


class JobReaderTests(unittest.TestCase):
    NAMES = ['coding', 'exact', 'general', 'c2-packed-tp4']

    def read(self, **values):
        base = dict(C2_IMAGE_TAG='v6-teardown', C2_CARDS='quad', C2_ACTIONS='fabric', C2_PROFILE='c2-packed-tp4')
        base.update(values)
        return job.read_job(base, self.NAMES)

    def test_unset_renders_empty_outputs(self):
        outputs = self.read()
        self.assertEqual((outputs['nvme_device'], outputs['peer_memory']), ('', ''))
        self.assertEqual(job.read_cache_drive({}, ['smoke'], 'quad'), ('', ''))

    def test_the_values_pass_through_normalised(self):
        outputs = self.read(C2_NVME_DEVICE='/dev/disk/by-id/' + NAME, C2_P2PMEM='0000:41:00.0, 0000:42:00.0')
        self.assertEqual((outputs['nvme_device'], outputs['peer_memory']), ('/dev/disk/by-id/' + NAME, '0000:41:00.0,0000:42:00.0'))
        outputs = self.read(C2_NVME_DEVICE='status', C2_P2PMEM='auto')
        self.assertEqual((outputs['nvme_device'], outputs['peer_memory']), ('status', 'auto'))
        self.assertEqual(self.read(C2_P2PMEM='auto')['nvme_device'], '', 'p2pmem alone is allowed (a probe of the cards)')

    def test_a_value_the_runner_cannot_honour_is_refused_here_before_any_card_time(self):
        for key, value in (('C2_NVME_DEVICE', '/dev/nvme0n1'), ('C2_NVME_DEVICE', NAME), ('C2_P2PMEM', 'all'), ('C2_P2PMEM', '0000:41:00.0,0000:41:00.0')):
            with self.subTest(key=key, value=value):
                with self.assertRaises(job.JobError):
                    self.read(**{key: value})

    def test_quad_only_and_only_for_the_steps_that_read_them(self):
        with self.assertRaises(job.JobError) as caught:
            job.read_cache_drive(dict(C2_NVME_DEVICE='status'), ['smoke'], 'pair')
        self.assertIn('C2_CARDS=quad', str(caught.exception))
        with self.assertRaises(job.JobError) as caught:
            self.read(C2_NVME_DEVICE='status', C2_ACTIONS='status reset')
        self.assertIn('fabric, smoke, prefix', str(caught.exception))
        for action in job.CACHE_DRIVE_STEPS:
            self.assertIn(action, job.ACTIONS)
            self.assertIn(action, job.QUAD_ACTIONS)

    def test_the_tracked_job_file_is_unchanged_in_effect(self):
        with open(os.path.join(ROOT, '.github', 'c2-serving-job.env'), encoding='utf-8') as handle:
            values = job.parse_env(handle.read())
        self.assertNotIn('C2_NVME_DEVICE', values)
        self.assertNotIn('C2_P2PMEM', values)
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(job.main([os.path.join(ROOT, '.github', 'c2-serving-job.env')]), 0)
        lines = dict(line.split('=', 1) for line in out.getvalue().splitlines())
        self.assertEqual((lines['nvme_device'], lines['peer_memory']), ('', ''))


def step_blocks(text):
    """{step name: its text} of the workflow (a step starts at `      - name:`)."""
    blocks, name, current = {}, None, []
    for line in text.splitlines():
        if line.startswith('      - name: '):
            if name is not None:
                blocks[name] = '\n'.join(current)
            name, current = line[len('      - name: '):], [line]
        elif name is not None:
            current.append(line)
    blocks[name] = '\n'.join(current)
    return blocks


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            self.text = handle.read()
        self.steps = step_blocks(self.text)

    def users(self):
        return sorted(name for name, block in self.steps.items() if 'NVME_DEVICE: ${{ steps.job.outputs.nvme_device }}' in block)

    def test_exactly_the_fabric_quad_smoke_and_prefix_steps_carry_the_keys(self):
        self.assertEqual(self.users(), sorted(['Four-card fabric probe (all four cards, inside the image)', 'Smoke on the four-card set', "Prefix-reuse gates in the agent's container shape"]))
        for name in self.users():
            block = self.steps[name]
            self.assertIn('P2PMEM: ${{ steps.job.outputs.peer_memory }}', block, name)
        for name, block in self.steps.items():
            if name not in self.users():
                self.assertNotIn('NVME_DEVICE: ${{', block, name)
                self.assertNotIn('cache_args', block, name)

    def test_the_two_docker_steps_resolve_the_drive_before_the_container_and_pass_the_arguments_to_docker_run(self):
        for name in ('Four-card fabric probe (all four cards, inside the image)', 'Smoke on the four-card set'):
            block = self.steps[name]
            self.assertIn('cache_args=()', block, name)
            self.assertIn('python3 scripts/ci/c2_cache_drive.py args ${NVME_DEVICE:+--nvme-device "$NVME_DEVICE"} ${P2PMEM:+--p2pmem "$P2PMEM"} --out', block, name)
            self.assertIn("mapfile -d '' -t cache_args <", block, name)
            self.assertIn('${cache_args[@]+"${cache_args[@]}"}', block, name)
            self.assertLess(block.index('c2_cache_drive.py args'), block.index('docker run'), name)
            self.assertEqual(block.count('${cache_args[@]+"${cache_args[@]}"}'), 1, 'one docker run takes it')

    def test_the_prefix_step_hands_the_gate_the_keys_only_when_set(self):
        block = self.steps["Prefix-reuse gates in the agent's container shape"]
        self.assertIn('${NVME_DEVICE:+--nvme-device "$NVME_DEVICE"} ${P2PMEM:+--p2pmem "$P2PMEM"}', block)

    def test_the_job_step_masks_the_literals_before_it_prints_the_outputs(self):
        block = self.steps['Read the job file']
        self.assertIn('python3 scripts/ci/c2_cache_drive.py mask "$GITHUB_OUTPUT"', block)
        self.assertLess(block.index('c2_cache_drive.py mask'), block.index('cat "$GITHUB_OUTPUT"'))
        self.assertLess(block.index('c2_serving_job.py'), block.index('c2_cache_drive.py mask'))

    @unittest.skipUnless(BASH, 'bash')
    def test_the_argument_file_round_trips_through_mapfile_and_an_empty_one_adds_no_word(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'args')
            tokens = ['--device', '/dev/nvme0n1:/dev/c2-cache-drive:rwm', '--mount', 'type=bind,src=/a b,dst=/a b,readonly', '-e', 'X=1 2']
            with open(path, 'wb') as handle:
                handle.write(cache.encode(tokens))
            script = ('set -euo pipefail; cache_args=(); mapfile -d "" -t cache_args < "$1"; printf "%s|" "${cache_args[@]+"${cache_args[@]}"}"; '
                      'empty=(); echo; printf "[%s]" ${empty[@]+"${empty[@]}"} x')
            result = subprocess.run([BASH, '-c', script, 'bash', path], stdout=subprocess.PIPE, universal_newlines=True)
            first, second = result.stdout.split('\n', 1)
            self.assertEqual(first.split('|')[:-1], tokens)
            self.assertEqual(second, '[x]', 'an unset array expands to nothing under set -u')


class GateTests(unittest.TestCase):
    ARMS = dict(arm='nvme-timed')

    def test_server_run_puts_the_arguments_ahead_of_the_entrypoint_and_adds_nothing_without_them(self):
        base = gate.server_run('img', 'name', 'prof', ['/dev/tenstorrent/0'])
        same = gate.server_run('img', 'name', 'prof', ['/dev/tenstorrent/0'], cache_args=())
        self.assertEqual(base, same)
        extra = ['--device', '/dev/nvme0n1:/dev/c2-cache-drive:rwm', '-e', 'QWEN_PREFIX_NVME_DEVICE=/dev/c2-cache-drive']
        argv = gate.server_run('img', 'name', 'prof', ['/dev/tenstorrent/0'], cache_args=extra)
        start = argv.index('--device', argv.index('--device') + 1)
        self.assertEqual(argv[start:start + 4], extra)
        self.assertEqual(argv[:start] + argv[start + 4:], base, 'the arguments are one contiguous run and nothing else moved')
        self.assertLess(start, argv.index('--entrypoint'))

    def test_the_gate_asks_for_the_keys_with_the_options_the_workflow_passes(self):
        parser = gate.build_parser()
        options = parser.parse_args(['--image', 'i', '--results', 'r', '--nvme-device', 'status', '--p2pmem', 'auto'])
        self.assertEqual((options.nvme_device, options.p2pmem), ('status', 'auto'))
        options = parser.parse_args(['--image', 'i', '--results', 'r'])
        self.assertEqual((options.nvme_device, options.p2pmem), ('', ''))

    def test_main_refuses_a_bad_value_and_the_pair_card_set(self):
        with tempfile.TemporaryDirectory() as directory:
            lines = []
            args = ['--image', 'img', '--results', directory, '--dry-run']
            self.assertEqual(gate.main(args + ['--nvme-device', '/dev/nvme0n1'], log=lines.append), 2)
            self.assertIn('by-id', lines[-1])
            self.assertEqual(gate.main(args + ['--nvme-device', 'status'], log=lines.append), 2)
            self.assertIn('--cards quad', lines[-1])
            self.assertEqual(gate.main(args + ['--p2pmem', 'zz'], log=lines.append), 2)

    def test_the_recorded_argv_of_an_arm_is_redacted(self):
        recorded = cache.redact(['--mount', 'type=bind,src=/sys/bus/pci/devices/0000:41:00.0/p2pmem/allocate,dst=x', '--device', '/dev/nvme2n1:/dev/c2-cache-drive:rwm'])
        self.assertNotIn('0000:41:00.0', ' '.join(recorded))
        self.assertNotIn('nvme2n1', ' '.join(recorded))
        with open(os.path.join(HERE, 'c2_prefix_gate.py'), encoding='utf-8') as handle:
            source = handle.read()
        self.assertIn('json.dump(c2_cache_drive.redact(arguments), handle, indent=1)', source, 'docker-run.json goes into the uploaded results')


if __name__ == '__main__':
    unittest.main()
