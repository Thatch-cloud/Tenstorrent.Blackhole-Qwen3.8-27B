#!/usr/bin/env python3
"""p2p_topology: will the Linux kernel let this PCIe device DMA into that one, and by which route? Read-only; no card or driver is touched.

The answer is the one drivers/pci/p2pdma.c (calc_map_type_and_dist) gives when an NVMe driver is asked to read into a card's memory:

  BUS_ADDR     the two devices share a bridge and no port on the way redirects peer traffic upstream (ACS request / completion redirect or
               egress control): the transaction stays inside the switch, the shortest and fastest route;
  THRU_HOST    the transaction has to reach the root complex (different root ports, or an ACS redirect on the way) and be routed back down; the
               kernel allows it when the CPU is a known-good one (any AMD Zen or newer) or the host bridge is on its short whitelist;
  NOT_SUPPORTED  otherwise.

It also lists, for the ports that redirect, the boot parameter the kernel itself suggests (pci=disable_acs_redir=...) and the setpci command that
clears the bits at run time, with the bits the port has now so they can be put back. It reads sysfs: past its first 64 bytes a device's config space (the PCI Express
capability, and the ACS capability in extended config space) is shown to root only. Run it with sudo, or the ACS state of every port reads as
unknown and the verdict says so.

Usage:
  p2p_topology.py --list
  p2p_topology.py --provider 0000:41:00.0 --client 0000:81:00.0

Stdlib only.
"""

import argparse
import json
import os
import platform
import re
import struct
import sys

VENDOR_TENSTORRENT = 0x1E52     # PCI_VENDOR_ID_TENSTORRENT
PCI_CLASS_NVME = 0x010802

ACS_BITS = {'SV': 0x01, 'TB': 0x02, 'RR': 0x04, 'CR': 0x08, 'UF': 0x10, 'EC': 0x20, 'DT': 0x40}
ACS_REDIRECT_MASK = ACS_BITS['RR'] | ACS_BITS['CR'] | ACS_BITS['EC']      # what the kernel's pci_bridge_has_acs_redir tests

PORT_TYPES = {0: 'endpoint', 1: 'legacy endpoint', 4: 'root port', 5: 'switch upstream port', 6: 'switch downstream port',
              7: 'PCIe-to-PCI bridge', 8: 'PCI-to-PCIe bridge', 9: 'root complex integrated endpoint', 10: 'root complex event collector'}
SPEEDS = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0}

BUS_ADDR = 'BUS_ADDR'
THRU_HOST = 'THRU_HOST_BRIDGE'
NOT_SUPPORTED = 'NOT_SUPPORTED'

# The kernel's host-bridge whitelist (pci_p2pdma_whitelist in drivers/pci/p2pdma.c, v7.0): (vendor, device) -> must be the same host bridge.
HOST_BRIDGE_WHITELIST = {(0x8086, 0x3c00): True, (0x8086, 0x3c01): True, (0x8086, 0x2f00): True, (0x8086, 0x2f01): True,
                         (0x8086, 0x2030): False, (0x8086, 0x2031): False, (0x8086, 0x2032): False, (0x8086, 0x2033): False,
                         (0x8086, 0x2020): False, (0x8086, 0x09a2): False}

BDF = re.compile(r'^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$')


class Device(object):
    """One PCI function as sysfs shows it."""

    def __init__(self, bdf, path, config, vendor, device, klass):
        self.bdf = bdf
        self.path = path
        self.vendor, self.device, self.klass = vendor, device, klass
        self.config = config
        self.port_type = None
        self.link = None
        self.mps = None
        self.acs = None                  # the ACS control register, or None when there is no ACS capability or it cannot be read
        self.acs_offset = None
        self.acs_readable = len(config) > 0x100
        self.parent = None               # the upstream bridge (a Device), or None at a root port
        self._parse()

    def _parse(self):
        cfg = self.config
        if len(cfg) < 0x40 or not struct.unpack_from('<H', cfg, 0x06)[0] & 0x10:
            return
        pointer = cfg[0x34]
        seen = 0
        while pointer and pointer + 2 <= len(cfg) and seen < 48:
            cap_id, nxt = cfg[pointer], cfg[pointer + 1]
            if cap_id == 0x10 and pointer + 0x14 <= len(cfg):
                flags, _, devctl, _, linkcap, _, linksta = struct.unpack_from('<HIHHIHH', cfg, pointer + 2)
                self.port_type = (flags >> 4) & 0xF
                self.mps = 128 << ((devctl >> 5) & 7)
                self.link = {'speed': SPEEDS.get(linksta & 0xF), 'width': (linksta >> 4) & 0x3F,
                             'max_speed': SPEEDS.get(linkcap & 0xF), 'max_width': (linkcap >> 4) & 0x3F}
            pointer = nxt
            seen += 1
        offset = 0x100
        seen = 0
        while offset and offset + 8 <= len(cfg) and seen < 64:
            header = struct.unpack_from('<I', cfg, offset)[0]
            if header in (0, 0xFFFFFFFF):
                break
            if header & 0xFFFF == 0x000D:
                self.acs_offset = offset
                self.acs = struct.unpack_from('<H', cfg, offset + 6)[0]
            offset = (header >> 20) & 0xFFC
            seen += 1

    @property
    def is_bridge(self):
        return self.port_type in (4, 5, 6, 7, 8)

    def has_acs_redirect(self):
        """The kernel's pci_bridge_has_acs_redir: any of request redirect, completion redirect or egress control is on."""
        return bool(self.acs and self.acs & ACS_REDIRECT_MASK)

    def acs_names(self):
        if self.acs is None:
            return None
        return [name for name, bit in ACS_BITS.items() if self.acs & bit]


def read_device(sysfs, bdf):
    path = os.path.realpath(os.path.join(sysfs, 'bus', 'pci', 'devices', bdf))
    if not os.path.isdir(path):
        raise KeyError(bdf)

    def number(name):
        with open(os.path.join(path, name)) as handle:
            return int(handle.read().strip(), 16)

    with open(os.path.join(path, 'config'), 'rb') as handle:
        config = handle.read(4096)
    return Device(bdf, path, config, number('vendor'), number('device'), number('class'))


def link_parents(devices, sysfs):
    """Fill in .parent from the sysfs path: .../pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0 is a chain of upstream bridges."""
    for device in devices.values():
        parts = [p for p in os.path.relpath(device.path, os.path.realpath(sysfs)).split(os.sep) if BDF.match(p)]
        device.chain = parts
        device.root_bus = next((p for p in os.path.relpath(device.path, os.path.realpath(sysfs)).split(os.sep) if p.startswith('pci')), '')
        # the nearest upstream bridge that could be read; one that could not is skipped rather than cutting the chain short
        device.parent = next((devices[name] for name in reversed(parts[:-1]) if name in devices), None)


def load_all(sysfs):
    base = os.path.join(sysfs, 'bus', 'pci', 'devices')
    devices = {}
    for name in sorted(os.listdir(base)):
        if BDF.match(name):
            try:
                devices[name] = read_device(sysfs, name)
            except (KeyError, OSError, ValueError):
                continue
    link_parents(devices, sysfs)
    return devices


def cpu_supports_p2pdma(cpuinfo='/proc/cpuinfo'):
    """The kernel's cpu_supports_p2pdma: any AMD CPU of family 0x17 (Zen) or newer. None when it cannot tell."""
    try:
        with open(cpuinfo) as handle:
            text = handle.read()
    except OSError:
        return None
    vendor = re.search(r'^vendor_id\s*:\s*(\S+)', text, re.M)
    family = re.search(r'^cpu family\s*:\s*(\d+)', text, re.M)
    if not vendor or not family:
        # the kernel's test is compiled for x86 only; anywhere else the answer is no
        return False if platform.machine().lower() not in ('x86_64', 'amd64', 'i686', 'i386') else None
    return vendor.group(1) == 'AuthenticAMD' and int(family.group(1)) >= 0x17


def host_bridge_device(devices, device):
    """The device the kernel looks at to name a host bridge (pci_host_bridge_dev): the first device on the root bus, if it is function 00.0 or a root
    port. Here: the 00.0 device of the root bus, else its first root port."""
    prefix = device.chain[0].rsplit(':', 1)[0] if device.chain else ''
    on_bus = sorted((d for d in devices.values() if d.chain and d.chain[0].rsplit(':', 1)[0] == prefix and d.parent is None), key=lambda d: d.bdf)
    for candidate in on_bus:
        if candidate.bdf.endswith(':00.0'):
            return candidate
    return next((d for d in on_bus if d.port_type == 4), None)


def whitelisted(devices, provider, client):
    """host_bridge_whitelist(): both devices' host bridges are on the kernel's list (and, for the older Intel entries, are the same bridge)."""
    same = provider.root_bus == client.root_bus

    def listed(device):
        bridge = host_bridge_device(devices, device)
        if bridge is None:
            return False
        must_be_same = HOST_BRIDGE_WHITELIST.get((bridge.vendor, bridge.device))
        return must_be_same is not None and (same or not must_be_same)

    return listed(provider) and (same or listed(client))


def predict(provider, client, cpu_ok, whitelist_ok=False):
    """calc_map_type_and_dist, step for step. Returns dict(map, distance, acs, shared_bridge, tested): `tested` is every device whose ACS control
    the kernel reads on the way."""
    acs = []
    tested = []
    a = provider
    dist_a = dist_b = 0
    common = None
    while a is not None:
        dist_b = 0
        tested.append(a)
        if a.has_acs_redirect():
            acs.append(a.bdf)
        bb = client
        while bb is not None:
            if bb is a:
                common = a
                break
            bb = bb.parent
            dist_b += 1
        if common is not None:
            break
        a = a.parent
        dist_a += 1

    if common is not None:
        bb = client
        while bb is not None and bb is not common:
            tested.append(bb)
            if bb.has_acs_redirect():
                acs.append(bb.bdf)
            bb = bb.parent
        distance = dist_a + dist_b
        if not acs:
            return {'map': BUS_ADDR, 'distance': distance, 'acs': [], 'shared_bridge': common.bdf, 'tested': tested}
    else:
        distance = dist_a + dist_b
    if cpu_ok or whitelist_ok:
        verdict = THRU_HOST
    elif cpu_ok is None:
        verdict = THRU_HOST + '?'
    else:
        verdict = NOT_SUPPORTED
    return {'map': verdict, 'distance': distance, 'acs': acs, 'shared_bridge': common.bdf if common else None, 'tested': tested}


def setpci_clear(device):
    """The setpci command that clears the redirect bits of one port, and the one that puts its present value back."""
    if device.acs_offset is None:
        return None
    new = device.acs & ~ACS_REDIRECT_MASK
    register = '%x.w' % (device.acs_offset + 6)          # the ACS control register, by config-space offset
    return {'clear': 'setpci -s %s %s=%04x' % (device.bdf, register, new),
            'restore': 'setpci -s %s %s=%04x' % (device.bdf, register, device.acs)}


def describe(device):
    return {'bdf': device.bdf, 'id': '%04x:%04x' % (device.vendor, device.device), 'class': '%06x' % device.klass,
            'type': PORT_TYPES.get(device.port_type, 'unknown'), 'link': device.link, 'mps': device.mps,
            'acs': device.acs_names(), 'acs_readable': device.acs_readable,
            'iommu_group': iommu_group(device), 'parents': device.chain[:-1]}


def iommu_group(device):
    try:
        link = os.path.join(device.path, 'iommu_group')
        if not os.path.islink(link):
            return None
        group = os.path.realpath(link)
        kind = ''
        try:
            with open(os.path.join(group, 'type')) as handle:
                kind = handle.read().strip()
        except OSError:
            pass
        return {'group': os.path.basename(group), 'type': kind}
    except OSError:
        return None


def verdict_for(devices, provider_bdf, client_bdf, cpu_ok):
    provider, client = devices[provider_bdf], devices[client_bdf]
    result = predict(provider, client, cpu_ok, whitelisted(devices, provider, client))
    result['provider'] = describe(provider)
    result['client'] = describe(client)
    blind = [d.bdf for d in result.pop('tested') if not d.acs_readable]
    result['acs_unreadable'] = blind
    result['clear_acs'] = [dict(bdf=bdf, **setpci_clear(devices[bdf])) for bdf in result['acs'] if setpci_clear(devices[bdf])]
    if result['acs']:
        result['kernel_parameter'] = 'pci=disable_acs_redir=' + ';'.join(result['acs'])
        result['cache_note'] = ('the kernel caches its verdict for a provider and client pair at the first attempt (map_types): after clearing ACS, '
                                'reload the card\'s driver (unload and load tenstorrent) before measuring again')
    if blind:
        result['note'] = ('ACS control of %s could not be read (run as root): the verdict assumes it is off' % ', '.join(blind))
    return result


def enumerate_roles(devices):
    cards = [d.bdf for d in devices.values() if d.vendor == VENDOR_TENSTORRENT]
    drives = [d.bdf for d in devices.values() if d.klass == PCI_CLASS_NVME]
    return cards, drives


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sysfs', default='/sys')
    parser.add_argument('--cpuinfo', default='/proc/cpuinfo')
    parser.add_argument('--list', action='store_true', help='every Tenstorrent card x every NVMe drive')
    parser.add_argument('--provider', help='the memory owner (a card), as 0000:41:00.0')
    parser.add_argument('--client', help='the DMA initiator (an NVMe drive)')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)

    devices = load_all(args.sysfs)
    cpu_ok = cpu_supports_p2pdma(args.cpuinfo)
    pairs = []
    if args.list:
        cards, drives = enumerate_roles(devices)
        pairs = [(card, drive) for card in cards for drive in drives]
        if not pairs:
            print('no Tenstorrent card and NVMe drive pair found under %s' % args.sysfs, file=sys.stderr)
            return 2
    elif args.provider and args.client:
        for name in (args.provider, args.client):
            if name not in devices:
                print('no such PCI device: %s' % name, file=sys.stderr)
                return 2
        pairs = [(args.provider, args.client)]
    else:
        parser.error('give --list, or --provider and --client')

    results = [verdict_for(devices, p, c, cpu_ok) for p, c in pairs]
    if args.json:
        print(json.dumps({'cpu_supports_p2pdma': cpu_ok, 'pairs': results}, indent=2, sort_keys=True))
        return 0
    print('cpu_supports_p2pdma (AMD Zen or newer): %s' % {True: 'yes', False: 'no (host bridge whitelist applies)', None: 'unknown'}[cpu_ok])
    for result in results:
        prov, cli = result['provider'], result['client']
        print('\ncard %s (%s, link %s) <- drive %s (%s, link %s)' % (
            prov['bdf'], prov['type'], _link(prov), cli['bdf'], cli['type'], _link(cli)))
        print('  kernel verdict: %s, distance %d%s' % (result['map'], result['distance'],
                                                       ', shared bridge %s' % result['shared_bridge'] if result['shared_bridge'] else
                                                       ', no shared bridge (different root ports)'))
        if result['acs']:
            print('  ACS redirect on: %s' % ', '.join(result['acs']))
            print('  boot parameter: %s' % result['kernel_parameter'])
            for item in result['clear_acs']:
                print('  run time:       %s     (put back: %s)' % (item['clear'], item['restore']))
            print('  NOTE: %s' % result['cache_note'])
        if result.get('note'):
            print('  NOTE: %s' % result['note'])
    return 0


def _link(info):
    link = info['link']
    if not link:
        return '?'
    return 'x%s gen%s' % (link['width'], {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(link['speed'], '?'))


if __name__ == '__main__':
    sys.exit(main())
