import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p2p_topology as topo

RR, CR, EC, SV = 0x04, 0x08, 0x20, 0x01


def config(vendor, device, klass, port_type=0, acs=None, speed=5, width=4, mps_code=1, size=4096, pcie_at=0x40, acs_at=0x100):
    """A PCI configuration space with a PCI Express capability at `pcie_at` (behind an MSI capability at 0x40 when it is elsewhere) and, when
    `acs` is not None, an ACS extended capability at `acs_at` (behind an AER capability at 0x100 when it is elsewhere)."""
    blob = bytearray(4096)
    struct.pack_into('<HH', blob, 0, vendor, device)
    struct.pack_into('<H', blob, 6, 0x0010)                      # status: has a capability list
    struct.pack_into('<I', blob, 8, klass << 8)
    blob[0x34] = 0x40
    if pcie_at != 0x40:
        blob[0x40], blob[0x41] = 0x05, pcie_at                   # MSI, pointing on to the PCI Express capability
    blob[pcie_at], blob[pcie_at + 1] = 0x10, 0
    struct.pack_into('<H', blob, pcie_at + 2, 0x2 | (port_type << 4))
    struct.pack_into('<HH', blob, pcie_at + 8, mps_code << 5, 0)        # device control, status
    struct.pack_into('<I', blob, pcie_at + 12, speed | (width << 4))    # link capabilities
    struct.pack_into('<HH', blob, pcie_at + 16, 0, speed | (width << 4))  # link control, status
    if acs is not None:
        if acs_at != 0x100:
            struct.pack_into('<I', blob, 0x100, 0x0001 | (1 << 16) | (acs_at << 20))      # AER, pointing on to ACS
        struct.pack_into('<I', blob, acs_at, 0x000D | (1 << 16))
        struct.pack_into('<HH', blob, acs_at + 4, 0x7F, acs)
    return bytes(blob[:size])


class Tree(object):
    """A fake sysfs: devices/<root bus>/<bdf>/<bdf>/... with bus/pci/devices/<bdf> symlinks, as the kernel lays it out."""

    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, 'bus', 'pci', 'devices'))

    def add(self, path, vendor, device, klass, **kw):
        directory = os.path.join(self.root, 'devices', *path)
        os.makedirs(directory)
        with open(os.path.join(directory, 'vendor'), 'w') as handle:
            handle.write('0x%04x\n' % vendor)
        with open(os.path.join(directory, 'device'), 'w') as handle:
            handle.write('0x%04x\n' % device)
        with open(os.path.join(directory, 'class'), 'w') as handle:
            handle.write('0x%06x\n' % klass)
        with open(os.path.join(directory, 'config'), 'wb') as handle:
            handle.write(config(vendor, device, klass, **kw))
        link = os.path.join(self.root, 'bus', 'pci', 'devices', path[-1])
        os.symlink(os.path.relpath(directory, os.path.dirname(link)), link)


def build(root, card_port_acs=0, nvme_port_acs=0, upstream_acs=0, size=4096, intel_host=None, card_kw=None):
    """root bus 0: root port 00:01.1 -> switch 01:00.0 -> downstream 02:00.0 -> card 03:00.0 and downstream 02:01.0 -> drive 04:00.0;
    root port 00:02.1 -> drive 05:00.0.  root bus 40: root port 40:01.1 -> card 41:00.0 (x16 behind a root port)."""
    t = Tree(root)
    b = 'pci0000:00'
    host_vendor, host_device = intel_host or (0x1022, 0x14a5)
    t.add([b, '0000:00:01.1'], host_vendor, host_device, 0x060400, port_type=4, acs=0, size=size)
    t.add([b, '0000:00:01.1', '0000:01:00.0'], 0x1000, 0xc030, 0x060400, port_type=5, acs=upstream_acs, size=size)
    t.add([b, '0000:00:01.1', '0000:01:00.0', '0000:02:00.0'], 0x1000, 0xc030, 0x060400, port_type=6, acs=card_port_acs, size=size)
    t.add([b, '0000:00:01.1', '0000:01:00.0', '0000:02:00.0', '0000:03:00.0'], 0x1e52, 0xb140, 0x120000, port_type=0, width=4, size=size,
          **(card_kw or {}))
    t.add([b, '0000:00:01.1', '0000:01:00.0', '0000:02:01.0'], 0x1000, 0xc030, 0x060400, port_type=6, acs=nvme_port_acs, size=size)
    t.add([b, '0000:00:01.1', '0000:01:00.0', '0000:02:01.0', '0000:04:00.0'], 0x144d, 0xa810, 0x010802, port_type=0, width=4, size=size)
    t.add([b, '0000:00:02.1'], 0x1022, 0x14a5, 0x060400, port_type=4, acs=0, size=size)
    t.add([b, '0000:00:02.1', '0000:05:00.0'], 0x144d, 0xa810, 0x010802, port_type=0, width=4, size=size)
    far = 'pci0000:40'
    t.add([far, '0000:40:01.1'], host_vendor, host_device, 0x060400, port_type=4, acs=0, size=size)
    t.add([far, '0000:40:01.1', '0000:41:00.0'], 0x1e52, 0xb140, 0x120000, port_type=0, width=16, size=size)
    return t


class Base(unittest.TestCase):
    def tree(self, **kw):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        build(directory.name, **kw)
        return directory.name

    def verdict(self, provider, client, cpu_ok=True, **kw):
        root = self.tree(**kw)
        devices = topo.load_all(root)
        return topo.verdict_for(devices, provider, client, cpu_ok)


class VerdictTest(Base):
    def test_two_devices_under_one_switch_with_no_redirect_stay_in_the_switch(self):
        result = self.verdict('0000:03:00.0', '0000:04:00.0')
        self.assertEqual((result['map'], result['distance'], result['shared_bridge']), (topo.BUS_ADDR, 4, '0000:01:00.0'))
        self.assertEqual(result['acs'], [])
        self.assertNotIn('kernel_parameter', result)

    def test_a_redirect_on_either_downstream_port_sends_it_through_the_host(self):
        for kw, port in (({'card_port_acs': RR}, '0000:02:00.0'), ({'nvme_port_acs': RR}, '0000:02:01.0'), ({'upstream_acs': RR}, '0000:01:00.0'),
                         ({'card_port_acs': CR}, '0000:02:00.0'), ({'nvme_port_acs': EC}, '0000:02:01.0')):
            with self.subTest(kw=kw):
                result = self.verdict('0000:03:00.0', '0000:04:00.0', **kw)
                self.assertEqual((result['map'], result['distance']), (topo.THRU_HOST, 4))
                self.assertEqual(result['acs'], [port])
                self.assertEqual(result['kernel_parameter'], 'pci=disable_acs_redir=' + port)

    def test_bits_that_are_not_a_redirect_do_not_count(self):
        result = self.verdict('0000:03:00.0', '0000:04:00.0', card_port_acs=SV)
        self.assertEqual(result['map'], topo.BUS_ADDR)

    def test_both_ports_redirecting_are_both_listed_for_the_kernel_parameter(self):
        result = self.verdict('0000:03:00.0', '0000:04:00.0', card_port_acs=RR | CR, nvme_port_acs=RR)
        self.assertEqual(result['acs'], ['0000:02:00.0', '0000:02:01.0'])
        self.assertEqual(result['kernel_parameter'], 'pci=disable_acs_redir=0000:02:00.0;0000:02:01.0')

    def test_devices_on_different_root_ports_go_through_the_host(self):
        result = self.verdict('0000:41:00.0', '0000:05:00.0')
        self.assertEqual((result['map'], result['distance'], result['shared_bridge']), (topo.THRU_HOST, 4, None))
        result = self.verdict('0000:03:00.0', '0000:05:00.0')
        self.assertEqual(result['map'], topo.THRU_HOST)
        self.assertIsNone(result['shared_bridge'])

    def test_a_cpu_the_kernel_does_not_trust_blocks_what_must_cross_the_host(self):
        self.assertEqual(self.verdict('0000:41:00.0', '0000:05:00.0', cpu_ok=False)['map'], topo.NOT_SUPPORTED)
        self.assertEqual(self.verdict('0000:03:00.0', '0000:04:00.0', cpu_ok=False)['map'], topo.BUS_ADDR)
        self.assertEqual(self.verdict('0000:41:00.0', '0000:05:00.0', cpu_ok=None)['map'], topo.THRU_HOST + '?')

    def test_the_run_time_fix_clears_the_redirect_bits_and_says_how_to_undo_it(self):
        result = self.verdict('0000:03:00.0', '0000:04:00.0', card_port_acs=RR | CR | SV)
        item = result['clear_acs'][0]
        self.assertEqual(item['bdf'], '0000:02:00.0')
        self.assertEqual(item['clear'], 'setpci -s 0000:02:00.0 106.w=0001')
        self.assertEqual(item['restore'], 'setpci -s 0000:02:00.0 106.w=000d')

    def test_without_root_the_acs_state_is_unknown_and_the_verdict_says_so(self):
        result = self.verdict('0000:03:00.0', '0000:04:00.0', card_port_acs=RR, size=256)
        self.assertEqual(result['map'], topo.BUS_ADDR)           # it cannot see the redirect
        self.assertIn('0000:02:00.0', result['acs_unreadable'])
        self.assertIn('run as root', result['note'])


class WalkTest(Base):
    def test_capabilities_are_found_wherever_the_chains_put_them(self):
        card_kw = dict(pcie_at=0x70, acs=RR, acs_at=0x148)
        root = self.tree(card_kw=card_kw)
        card = topo.load_all(root)['0000:03:00.0']
        self.assertEqual(card.port_type, 0)
        self.assertEqual(card.link['width'], 4)
        self.assertEqual(card.mps, 256)
        self.assertEqual((card.acs, card.acs_offset), (RR, 0x148))
        self.assertTrue(card.has_acs_redirect())

    def test_the_endpoints_own_acs_is_tested_too(self):
        result = self.verdict('0000:03:00.0', '0000:04:00.0', card_kw=dict(acs=RR))
        self.assertEqual(result['acs'], ['0000:03:00.0'])
        self.assertEqual(result['map'], topo.THRU_HOST)

    def test_only_what_the_kernel_reads_can_make_the_verdict_blind(self):
        # a root port above the shared switch is never tested; the card's own downstream port is
        root = self.tree()
        for bdf in ('0000:00:01.1',):
            path = os.path.realpath(os.path.join(root, 'bus', 'pci', 'devices', bdf, 'config'))
            with open(path, 'r+b') as handle:
                handle.truncate(256)
        result = topo.verdict_for(topo.load_all(root), '0000:03:00.0', '0000:04:00.0', True)
        self.assertEqual(result['acs_unreadable'], [])
        path = os.path.realpath(os.path.join(root, 'bus', 'pci', 'devices', '0000:02:00.0', 'config'))
        with open(path, 'r+b') as handle:
            handle.truncate(256)
        result = topo.verdict_for(topo.load_all(root), '0000:03:00.0', '0000:04:00.0', True)
        self.assertEqual(result['acs_unreadable'], ['0000:02:00.0'])

    def test_a_bridge_that_could_not_be_read_does_not_cut_the_chain_short(self):
        root = self.tree()
        os.unlink(os.path.realpath(os.path.join(root, 'bus', 'pci', 'devices', '0000:02:00.0', 'config')))
        devices = topo.load_all(root)
        self.assertNotIn('0000:02:00.0', devices)
        self.assertEqual(devices['0000:03:00.0'].parent.bdf, '0000:01:00.0')
        result = topo.verdict_for(devices, '0000:03:00.0', '0000:04:00.0', True)
        self.assertEqual(result['map'], topo.BUS_ADDR)
        self.assertEqual(result['shared_bridge'], '0000:01:00.0')

    def test_a_device_with_no_iommu_group_says_none(self):
        devices = topo.load_all(self.tree())
        self.assertIsNone(topo.describe(devices['0000:03:00.0'])['iommu_group'])

    def test_the_clear_advice_warns_that_the_kernel_remembers_its_verdict(self):
        result = self.verdict('0000:03:00.0', '0000:04:00.0', card_port_acs=RR)
        self.assertIn('caches its verdict', result['cache_note'])
        self.assertNotIn('cache_note', self.verdict('0000:03:00.0', '0000:04:00.0'))


class WhitelistTest(Base):
    def test_a_listed_intel_host_bridge_allows_what_the_cpu_rule_would_not(self):
        result = self.verdict('0000:41:00.0', '0000:05:00.0', cpu_ok=False, intel_host=(0x8086, 0x2020))
        self.assertEqual(result['map'], topo.THRU_HOST)

    def test_the_older_entries_need_both_devices_under_one_host_bridge(self):
        # 0x3c00 is listed "same host bridge only"; the card and the drive are on different root buses
        result = self.verdict('0000:41:00.0', '0000:05:00.0', cpu_ok=False, intel_host=(0x8086, 0x3c00))
        self.assertEqual(result['map'], topo.NOT_SUPPORTED)
        result = self.verdict('0000:03:00.0', '0000:05:00.0', cpu_ok=False, intel_host=(0x8086, 0x3c00))
        self.assertEqual(result['map'], topo.THRU_HOST)

    def test_an_unlisted_host_bridge_does_not(self):
        result = self.verdict('0000:41:00.0', '0000:05:00.0', cpu_ok=False, intel_host=(0x8086, 0x1234))
        self.assertEqual(result['map'], topo.NOT_SUPPORTED)


class ParseTest(Base):
    def test_link_port_type_and_payload_size_are_read(self):
        devices = topo.load_all(self.tree())
        card = devices['0000:03:00.0']
        self.assertEqual(card.port_type, 0)
        self.assertEqual(card.link, {'speed': 32.0, 'width': 4, 'max_speed': 32.0, 'max_width': 4})
        self.assertEqual(card.mps, 256)
        self.assertEqual(devices['0000:41:00.0'].link['width'], 16)
        self.assertEqual(devices['0000:00:01.1'].port_type, 4)
        self.assertEqual(devices['0000:01:00.0'].port_type, 5)
        self.assertEqual(devices['0000:02:00.0'].port_type, 6)

    def test_the_hierarchy_comes_from_the_sysfs_path(self):
        devices = topo.load_all(self.tree())
        self.assertEqual(devices['0000:04:00.0'].parent.bdf, '0000:02:01.0')
        self.assertEqual(devices['0000:02:01.0'].parent.bdf, '0000:01:00.0')
        self.assertEqual(devices['0000:01:00.0'].parent.bdf, '0000:00:01.1')
        self.assertIsNone(devices['0000:00:01.1'].parent)

    def test_roles_are_found_by_vendor_and_class(self):
        cards, drives = topo.enumerate_roles(topo.load_all(self.tree()))
        self.assertEqual(sorted(cards), ['0000:03:00.0', '0000:41:00.0'])
        self.assertEqual(sorted(drives), ['0000:04:00.0', '0000:05:00.0'])

    def test_the_cpu_rule_is_amd_zen_or_newer(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('platform.machine', return_value='x86_64'):
            for text, expected in (('vendor_id\t: AuthenticAMD\ncpu family\t: 26\n', True), ('vendor_id\t: AuthenticAMD\ncpu family\t: 21\n', False),
                                   ('vendor_id\t: GenuineIntel\ncpu family\t: 6\n', False), ('nonsense', None)):
                path = os.path.join(tmp, 'cpuinfo')
                with open(path, 'w') as handle:
                    handle.write(text)
                self.assertEqual(topo.cpu_supports_p2pdma(path), expected, text)
            self.assertIsNone(topo.cpu_supports_p2pdma(os.path.join(tmp, 'missing')))

    def test_off_x86_the_kernels_answer_is_no(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('platform.machine', return_value='aarch64'):
            path = os.path.join(tmp, 'cpuinfo')
            with open(path, 'w') as handle:
                handle.write('processor\t: 0\nBogoMIPS\t: 2000\n')
            self.assertFalse(topo.cpu_supports_p2pdma(path))


class CommandLineTest(Base):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = topo.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def cpuinfo(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, 'cpuinfo')
        with open(path, 'w') as handle:
            handle.write('vendor_id\t: AuthenticAMD\ncpu family\t: 26\n')
        return path

    def test_list_prints_every_card_and_drive_pair_as_json(self):
        root = self.tree(card_port_acs=RR)
        code, out, _ = self.run_main('--sysfs', root, '--cpuinfo', self.cpuinfo(), '--list', '--json')
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertTrue(report['cpu_supports_p2pdma'])
        self.assertEqual(len(report['pairs']), 4)
        by_pair = {(p['provider']['bdf'], p['client']['bdf']): p['map'] for p in report['pairs']}
        self.assertEqual(by_pair[('0000:03:00.0', '0000:04:00.0')], topo.THRU_HOST)
        self.assertEqual(by_pair[('0000:41:00.0', '0000:04:00.0')], topo.THRU_HOST)

    def test_the_text_report_names_the_boot_parameter_and_the_run_time_command(self):
        root = self.tree(nvme_port_acs=RR)
        code, out, _ = self.run_main('--sysfs', root, '--cpuinfo', self.cpuinfo(), '--provider', '0000:03:00.0', '--client', '0000:04:00.0')
        self.assertEqual(code, 0)
        self.assertIn('pci=disable_acs_redir=0000:02:01.0', out)
        self.assertIn('setpci -s 0000:02:01.0 106.w=', out)
        self.assertIn('cpu_supports_p2pdma (AMD Zen or newer): yes', out)

    def test_an_unknown_device_is_an_error(self):
        root = self.tree()
        code, _, err = self.run_main('--sysfs', root, '--provider', '0000:99:00.0', '--client', '0000:04:00.0')
        self.assertEqual(code, 2)
        self.assertIn('no such PCI device', err)

    def test_it_writes_nothing(self):
        root = self.tree()
        before = sorted(os.path.join(d, f) for d, _, files in os.walk(root) for f in files)
        self.run_main('--sysfs', root, '--list')
        after = sorted(os.path.join(d, f) for d, _, files in os.walk(root) for f in files)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
