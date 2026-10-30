"""The cache drive and the cards' peer-to-peer memory in a job's containers: C2_NVME_DEVICE and C2_P2PMEM (qwen-c2-serving.yml; docs/tp4-kv-nvme.md section 12).

    python3 scripts/ci/c2_cache_drive.py args [--nvme-device V] [--p2pmem V] --out FILE   print the readings, write the docker arguments (NUL separated) to FILE
    python3 scripts/ci/c2_cache_drive.py check [--nvme-device V] [--p2pmem V]               the same without a file (what a dry run shows)

Host side, stdlib only: it runs on the job's runner (the workflow's steps and c2_prefix_gate.py call it), never in the image. Both keys absent = nothing here is called and the
containers are exactly what they were.

C2_NVME_DEVICE   a /dev/disk/by-id path (or the word `status`: the by_id the host's boot unit wrote into its status file). The device is resolved AT RUN TIME from that link (the
                 kernel renumbers nvmeN at every reset and rescan, so no nvmeN name is ever remembered), and BEFORE a container starts the R1 readings are taken and printed - the state of
                 the controller the link resolves to (/sys/.../nvme*/state) and the namespace's size - and the job is REFUSED unless the state is `live` and the size is above 0
                 (docs/tp4-kv-nvme.md, gate R1: a card reset's PCI rescan can leave the drive disabled, state `dead`, 0 bytes). The status file's pci address, when it has one, must be
                 on the resolved device's sysfs path (the identity check: the file describes THIS drive). The container gets
                     --device <resolved node>:/dev/c2-cache-drive:rwm
                     --mount type=bind,src=<status file>,dst=<status file>,readonly       (the tier's default QWEN_PREFIX_NVME_SED_STATUS: a container has no CAP_SYS_ADMIN,
                                                                                          so the encryption check reads the boot unit's file)
                     -e QWEN_PREFIX_NVME_DEVICE=/dev/c2-cache-drive
C2_P2PMEM        comma-separated PCI addresses of cards that publish p2pmem (or the word `auto`: every PCI function with /sys/bus/pci/devices/<addr>/p2pmem/allocate). The patched
                 tt-kmd (patches/tt-kmd-p2pmem) publishes a slice of BAR4; a process mmaps <addr>/p2pmem/allocate READ-WRITE and O_DIRECT-reads the drive into it. Docker mounts /sys
                 read-only, and that file cannot be opened O_RDWR there (EROFS). The narrowest thing that works - tried with a throwaway container on a sysfs attribute of the same shape -
                 is a bind mount of THAT ONE FILE per card (the host's sysfs is read-write, a bind mount carries it; every sibling stays read-only), but Docker's default AppArmor profile
                 denies writes under /sys whatever the mount flags say, so the container also needs
                     --security-opt apparmor=unconfined
                 which drops that one profile only: not --privileged, no added capability, the default seccomp profile and the device cgroup stay. (A profile of our own that allows
                 `/sys/devices/**/p2pmem/allocate rw` would be narrower still; loading one needs root on the runner and is a platform item, docs/tp4-kv-nvme.md section 12.) Per card:
                     --mount type=bind,src=/sys/bus/pci/devices/<addr>/p2pmem/allocate,dst=<the same path>
                 and once: -e QWEN_P2PMEM_DEVICES=<addr>[,<addr>...]
Nothing here prints a by-id name, a serial or a PCI address (the job log is public): the readings are a state word, a byte count and counts. The values themselves are the job
config's. They come in the job file of the tag's commit, which is pushed: the words `status` and `auto` keep a literal by-id string and PCI addresses out of it.
"""

import argparse
import json
import os
import re
import stat
import sys

IN_CONTAINER_DEVICE = '/dev/c2-cache-drive'
STATUS_FILE = '/run/thatch-cache-drive/status.json'
BY_ID_DIR = '/dev/disk/by-id'
SYSFS = '/sys'
STATUS_WORD = 'status'
AUTO_WORD = 'auto'
NAME = r'[A-Za-z0-9][A-Za-z0-9_.:+@=-]{0,199}'
BY_ID = re.compile(re.escape(BY_ID_DIR) + '/' + NAME)
BY_ID_NAME = re.compile(NAME)
PCI_ADDRESS = re.compile(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')
CONTROLLER = re.compile(r'nvme[0-9]+')
ENV_DEVICE = 'QWEN_PREFIX_NVME_DEVICE'
ENV_P2PMEM = 'QWEN_P2PMEM_DEVICES'
LIVE = 'live'
MAX_PCI_ADDRESSES = 16


class CacheDriveError(ValueError):
    """A key the job must not run with, or a drive that is not usable now: the job is refused before a container starts. The text never names the drive."""


def read_nvme_spec(value):
    """C2_NVME_DEVICE as the string to hand on: '' (unset), STATUS_WORD, or a /dev/disk/by-id path. CacheDriveError for anything else."""
    value = (value or '').strip()
    if not value or value == STATUS_WORD:
        return value
    if not BY_ID.fullmatch(value):
        raise CacheDriveError('C2_NVME_DEVICE must be a /dev/disk/by-id path (or the word %s), not anything else: the drive is found by its by-id link, never by an nvmeN name' % STATUS_WORD)
    return value


def read_p2pmem_spec(value):
    """C2_P2PMEM as a tuple of lower-case PCI addresses, or AUTO_WORD, or () when unset. CacheDriveError for anything else."""
    value = (value or '').strip()
    if not value:
        return ()
    if value == AUTO_WORD:
        return AUTO_WORD
    parts = [part.strip().lower() for part in value.split(',')]
    for part in parts:
        if not PCI_ADDRESS.fullmatch(part):
            raise CacheDriveError('C2_P2PMEM must be comma-separated PCI addresses (dddd:bb:dd.f) or the word %s' % AUTO_WORD)
    if len(set(parts)) != len(parts):
        raise CacheDriveError('C2_P2PMEM names a card twice')
    if len(parts) > MAX_PCI_ADDRESSES:
        raise CacheDriveError('C2_P2PMEM names more than %d cards' % MAX_PCI_ADDRESSES)
    return tuple(parts)


def _read_text(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def read_status(path=STATUS_FILE, read=_read_text):
    """The boot unit's status file as a dict. CacheDriveError when it is missing, unreadable or not a JSON object."""
    try:
        status = json.loads(read(path))
    except (OSError, ValueError):
        raise CacheDriveError('the cache drive\'s status file cannot be read as JSON on this host')
    if not isinstance(status, dict):
        raise CacheDriveError('the cache drive\'s status file is not a JSON object')
    return status


def by_id_of(spec, status_path=STATUS_FILE, read=_read_text):
    """The /dev/disk/by-id path the job names: the literal, or the status file's by_id (a bare name or a full path)."""
    if spec != STATUS_WORD:
        return spec
    name = read_status(status_path, read).get('by_id')
    if not isinstance(name, str) or not name:
        raise CacheDriveError('the status file names no by_id')
    path = name if name.startswith('/') else BY_ID_DIR + '/' + name
    if not BY_ID.fullmatch(path):
        raise CacheDriveError('the status file\'s by_id is not a plain /dev/disk/by-id name')
    return path


def _sysfs_text(path):
    try:
        with open(path, encoding='utf-8') as handle:
            return handle.read().strip()
    except OSError:
        return None


def controllers_of(base):
    """[(controller directory, state text)] of the NVMe controllers behind the namespace's sysfs directory `base` (its realpath under /sys/devices): the namespace's own `device`, the nvme*
    ancestors, and - for a multipath head namespace, whose parent is the subsystem - the subsystem's nvme* children. Only directories that have a `state` file count."""
    found = []
    candidates = [os.path.join(base, 'device')]
    parent = base
    for _level in range(3):
        parent = os.path.dirname(parent)
        if CONTROLLER.fullmatch(os.path.basename(parent)):
            candidates.append(parent)
    subsystem = os.path.dirname(base)
    try:
        candidates += [os.path.join(subsystem, name) for name in sorted(os.listdir(subsystem)) if CONTROLLER.fullmatch(name)]
    except OSError:
        pass
    seen = set()
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real in seen:
            continue
        state = _sysfs_text(os.path.join(real, 'state'))
        if state is not None:
            seen.add(real)
            found.append((real, state))
    return found


def drive_readings(by_id, sysfs=SYSFS, stat_fn=os.stat):
    """The R1 readings of the drive the by-id link resolves to NOW: dict(node, controllers [(directory, state)], state ('live' only when every controller reads live; 'unreadable' with none),
    bytes (the namespace's size, 0 when it reads 0 or cannot be read), sysfs_path). CacheDriveError when the link does not resolve to a block device."""
    node = os.path.realpath(by_id)
    try:
        info = stat_fn(node)
    except OSError:
        raise CacheDriveError('the drive\'s by-id link does not resolve to a device on this host')
    if not stat.S_ISBLK(info.st_mode):
        raise CacheDriveError('the drive\'s by-id link does not resolve to a block device')
    number = '%d:%d' % (os.major(info.st_rdev), os.minor(info.st_rdev))
    base = os.path.realpath(os.path.join(sysfs, 'dev', 'block', number))
    if not os.path.isdir(base):
        return dict(node=node, controllers=[], state='unreadable', bytes=0, sysfs_path=None)
    sectors = _sysfs_text(os.path.join(base, 'size'))
    size = int(sectors) * 512 if sectors and sectors.isdigit() else 0
    controllers = controllers_of(base)
    states = [state for _directory, state in controllers]
    state = 'unreadable' if not states else (LIVE if all(item == LIVE for item in states) else [item for item in states if item != LIVE][0])
    return dict(node=node, controllers=controllers, state=state, bytes=size, sysfs_path=base)


def readings_problems(readings):
    """Why the drive is not usable ([] when it is): the controller must read `live` and the namespace must be above 0 bytes."""
    problems = []
    if readings['state'] != LIVE:
        problems.append('the cache drive\'s controller state reads %r, not live (a card reset\'s PCI rescan can leave it disabled: gate R1 brings it back)' % readings['state'])
    if not readings['bytes'] > 0:
        problems.append('the cache drive\'s namespace reads 0 bytes')
    return problems


def readings_line(readings):
    """The R1 line for the job log: a state word and a size, nothing that names the drive."""
    return 'R1 cache drive: controller state=%s controllers=%d namespace_bytes=%d (resolved from the by-id link at run time)' % (readings['state'], len(readings['controllers']), readings['bytes'])


def identity_problems(status, readings):
    """The status file's pci address (pci_addr, or pci), when it has one, must be a function on the resolved device's sysfs path: the file must describe THIS drive. -> (problems, how)"""
    address = status.get('pci_addr') or status.get('pci')
    if not isinstance(address, str) or not address or not readings.get('sysfs_path'):
        return [], 'not checked (the status file names no pci address, or sysfs cannot say)'
    # (a multipath head namespace has no PCI function in its own path; its controllers' paths have)
    paths = [readings['sysfs_path']] + [directory for directory, _state in readings.get('controllers') or ()]
    if any(('/%s/' % address.lower()) in (path.lower() + '/') for path in paths):
        return [], 'the status file\'s pci address is on the resolved device\'s path'
    return ['the status file describes another drive: its pci address is not on the resolved device\'s sysfs path'], 'mismatch'


def p2pmem_attribute(address):
    return '/sys/bus/pci/devices/%s/p2pmem/allocate' % address


def p2pmem_cards(spec, sysfs=SYSFS, exists=os.path.exists):
    """The PCI addresses to give the container: the named ones, or (auto) every PCI function whose p2pmem/allocate exists. Every one must exist (the patched module publishes it)."""
    if spec == AUTO_WORD:
        try:
            names = sorted(os.listdir(os.path.join(sysfs, 'bus', 'pci', 'devices')))
        except OSError:
            names = []
        cards = [name for name in names if PCI_ADDRESS.fullmatch(name) and exists(os.path.join(sysfs, 'bus', 'pci', 'devices', name, 'p2pmem', 'allocate'))]
        if not cards:
            raise CacheDriveError('C2_P2PMEM=auto: no PCI function on this host publishes p2pmem (the patched tt-kmd is not loaded, or p2pmem_devices names none)')
        return tuple(cards)
    missing = [address for address in spec if not exists(os.path.join(sysfs, 'bus', 'pci', 'devices', address, 'p2pmem', 'allocate'))]
    if missing:
        raise CacheDriveError('p2pmem is not published on %d of the %d cards C2_P2PMEM names (the patched tt-kmd is not loaded for them)' % (len(missing), len(spec)))
    return tuple(spec)


def plan(nvme_spec='', p2pmem_spec=(), status_path=STATUS_FILE, sysfs=SYSFS, read=_read_text, stat_fn=os.stat, exists=os.path.exists, readings_of=None):
    """-> dict(arguments, lines, env): the docker arguments to add for the job's keys, the lines for the log, and the env pairs they set. CacheDriveError (job refused) for a key that
    cannot be honoured. Both keys empty: no arguments, nothing read."""
    arguments, lines, env = [], [], {}
    if nvme_spec:
        by_id = by_id_of(nvme_spec, status_path, read)
        readings = (readings_of or (lambda path: drive_readings(path, sysfs, stat_fn)))(by_id)
        lines.append(readings_line(readings))
        problems = readings_problems(readings)
        try:
            status = read_status(status_path, read)
        except CacheDriveError:
            status = None
            lines.append('status file: not readable on this host, so not mounted (the tier\'s encryption check refuses the drive unless the profile turns it off, a gate\'s own value)')
        if status is not None:
            more, how = identity_problems(status, readings)
            problems += more
            lines.append('R1 identity: %s' % how)
        if problems:
            raise CacheDriveError('; '.join(problems))
        arguments += ['--device', '%s:%s:rwm' % (readings['node'], IN_CONTAINER_DEVICE)]
        if status is not None:
            arguments += ['--mount', 'type=bind,src=%s,dst=%s,readonly' % (status_path, status_path)]
        arguments += ['-e', '%s=%s' % (ENV_DEVICE, IN_CONTAINER_DEVICE)]
        env[ENV_DEVICE] = IN_CONTAINER_DEVICE
    if p2pmem_spec:
        cards = p2pmem_cards(p2pmem_spec, sysfs, exists)
        for address in cards:
            attribute = p2pmem_attribute(address)
            arguments += ['--mount', 'type=bind,src=%s,dst=%s' % (attribute, attribute)]
        arguments += ['--security-opt', 'apparmor=unconfined', '-e', '%s=%s' % (ENV_P2PMEM, ','.join(cards))]
        env[ENV_P2PMEM] = ','.join(cards)
        lines.append('p2pmem: %d card(s), p2pmem/allocate bound read-write (one file each), the container runs with apparmor=unconfined (not privileged)' % len(cards))
    return dict(arguments=arguments, lines=lines, env=env)


def mask_lines(outputs_text):
    """The `::add-mask::` workflow commands for the literals in the job reader's outputs (the file GITHUB_OUTPUT): the by-id path and its name, each PCI address. Nothing for the words
    `status` and `auto`, which are not private. The workflow prints these before it prints the outputs, so a later step's env block and log show asterisks."""
    values = []
    for line in outputs_text.splitlines():
        key, _, value = line.partition('=')
        value = value.strip()
        if key == 'nvme_device' and value.startswith(BY_ID_DIR + '/'):
            values += [value, os.path.basename(value)]
        elif key == 'peer_memory' and value and value != AUTO_WORD:
            values += [part for part in value.split(',') if part]
    return ['::add-mask::%s' % value for value in values]


def redact(arguments):
    """A docker argument list with every PCI address and the host's device node of the drive replaced, for the record a gate writes into its (uploaded) results."""
    out = []
    for token in arguments:
        token = re.sub(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', '<card>', token)
        token = re.sub(r'/dev/nvme[0-9]+n[0-9]+(?=:%s)' % re.escape(IN_CONTAINER_DEVICE), '<drive>', token)
        out.append(token)
    return out


def encode(arguments):
    """The arguments as bytes, NUL separated (bash: mapfile -d '' -t)."""
    return b''.join(token.encode('utf-8') + b'\0' for token in arguments)


def main(argv=None, out=sys.stdout, err=sys.stderr, **kwargs):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('mode', choices=('args', 'check', 'mask'))
    parser.add_argument('outputs', nargs='?', default=None, help='mask: the GITHUB_OUTPUT file')
    parser.add_argument('--nvme-device', default='')
    parser.add_argument('--p2pmem', default='')
    parser.add_argument('--status-file', default=STATUS_FILE)
    parser.add_argument('--out', default=None)
    arguments = parser.parse_args(argv)
    if arguments.mode == 'mask':
        if not arguments.outputs:
            parser.error('mask needs the outputs file')
        try:
            text = _read_text(arguments.outputs)
        except OSError:
            return 0
        for line in mask_lines(text):
            out.write(line + '\n')
        return 0
    try:
        result = plan(read_nvme_spec(arguments.nvme_device), read_p2pmem_spec(arguments.p2pmem), status_path=arguments.status_file, **kwargs)
    except CacheDriveError as error:
        err.write('C2 cache drive: REFUSED: %s\n' % error)
        return 1
    for line in result['lines']:
        out.write('[C2-CACHE-DRIVE] %s\n' % line)
    if arguments.mode == 'args':
        if not arguments.out:
            parser.error('args needs --out')
        with open(arguments.out, 'wb') as handle:
            handle.write(encode(result['arguments']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
