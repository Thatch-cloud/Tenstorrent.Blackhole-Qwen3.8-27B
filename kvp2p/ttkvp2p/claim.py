"""The claim tool: the ONLY writer of a ``gemma-kv`` superblock (design 3.4, owner decision O-D12).

    python3 -m ttkvp2p.claim --device /dev/disk/by-id/<name> --confirm <name> [--size BYTES] --status <status.json>
                             [--layout gemma-chipmajor-v1] [--overwrite=<format seen>] [--dry-run]

It runs only by an owner-approved ``!`` and is never automatic: the tier itself never formats and imports nothing from here. It is gated:

* the device is named by its by-id path, and the operator TYPES THE NAME BACK (``--confirm``): a name that differs from the path's, or a
  path that is not a by-id symlink, is refused;
* the cache-drive status file must be fresh, ``state == ok`` and name the same by-id; the drive's size must equal the size the status file
  records (``size_bytes``) or, when the file records none, the size the operator types (``--size``);
* the drive is opened exclusively (the kernel's claim, plus a lock): a drive another process holds is refused;
* a drive that carries ANY other format is refused unless ``--overwrite=<format seen>`` names exactly what is there. The formats it
  recognises: blank, gemma-kv-v2, qwen-nvme-v1, gpt, mbr, ext, xfs, btrfs, ntfs, luks, swap, lvm, and ``unknown-data`` for anything else
  that is not zero (the P1 probe's random test pattern is that). A refusal prints what it saw.

The claim digest it writes (sha256 of the by-id name and the size) is what the tier recomputes at every start from the status file, so a
cloned or swapped drive does not verify. Standard library only. Exit codes: 0 claimed or already claimed, 2 refused, 3 failed.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from ttkvp2p import health, store

PROBE_BYTES = store.PROBE_BYTES
TAIL_BYTES = 1 << 20
FORMATS = ('blank', 'gemma-kv-v2', 'qwen-nvme-v1', 'gpt', 'mbr', 'ext', 'xfs', 'btrfs', 'ntfs', 'luks', 'swap', 'lvm', 'unknown-data')
BY_ID_DIR = '/dev/disk/by-id'
CLAIM_STATUS_MAX_AGE_S = health.STATUS_MAX_AGE_S


class Refused(Exception):
    """The claim is refused; nothing was written."""


def classify(head: bytes, tail: bytes = b'') -> Tuple[str, str]:
    """(format token, detail) of what the first MiB (and the last) of a drive holds."""
    if not any(head):
        if any(tail):
            if tail[-512:-504] == b'EFI PART':
                return 'gpt', 'a backup GPT header at the end of the drive'
            return 'unknown-data', 'the start is blank but the end of the drive holds data'
        return 'blank', 'the first and last MiB are all zero'
    for offset in (store.SB_SLOT_A, store.SB_SLOT_B):
        if head[offset:offset + 8] == store.SB_MAGIC:
            return 'gemma-kv-v2', 'a gemma-kv superblock (valid or not) at byte %d' % offset
        if head[offset:offset + 8] == store.QWEN_MAGIC:
            return 'qwen-nvme-v1', 'the retired Qwen store\'s superblock at byte %d' % offset
    if head[512:520] == b'EFI PART':
        return 'gpt', 'a GPT partition table'
    if head[510:512] == b'\x55\xaa':
        return 'mbr', 'an MBR boot signature'
    if head[1080:1082] == b'\x53\xef':
        return 'ext', 'an ext2/3/4 superblock'
    if head[0:4] == b'XFSB':
        return 'xfs', 'an XFS superblock'
    if head[3:11] == b'NTFS    ':
        return 'ntfs', 'an NTFS boot sector'
    if head[65536 + 64:65536 + 72] == b'_BHRfS_M':
        return 'btrfs', 'a btrfs superblock'
    if head[0:6] == b'LUKS\xba\xbe':
        return 'luks', 'a LUKS header'
    if head[4086:4096] == b'SWAPSPACE2':
        return 'swap', 'a swap signature'
    if head[512:520] == b'LABELONE':
        return 'lvm', 'an LVM label'
    return 'unknown-data', 'non-zero data with no known signature (the P1 probe\'s random test pattern looks like this)'


def probe(drive: store.DriveFile) -> Tuple[str, str]:
    head = drive.read(0, min(PROBE_BYTES, drive.size // store.ALIGN * store.ALIGN))
    tail = b''
    if drive.size > 2 * PROBE_BYTES:
        tail = drive.read((drive.size - TAIL_BYTES) // store.ALIGN * store.ALIGN, TAIL_BYTES)
    return classify(head, tail)


def check_by_id(device_arg: str, confirm: str, allow_regular: bool) -> Tuple[str, str]:
    """(by-id name, resolved device path). The operator must have typed the name back; the path must be a by-id symlink."""
    name = os.path.basename(device_arg.rstrip('/'))
    if confirm != name:
        raise Refused('the by-id name typed back (%r) is not the name in the device path (%r)' % (confirm, name))
    if not allow_regular:
        if os.path.dirname(os.path.abspath(device_arg)) != BY_ID_DIR or not os.path.islink(device_arg):
            raise Refused('%s is not a symlink in %s: name the drive by its by-id path, never by /dev/nvme*' % (device_arg, BY_ID_DIR))
    return name, os.path.realpath(device_arg)


def check_status(status_path: str, name: str, size: int, typed_size: Optional[int], env: health.HostEnv) -> None:
    try:
        with open(status_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, ValueError) as error:
        raise Refused('the cache-drive status file cannot be read (%s)' % (error,))
    stamp = health.parse_timestamp(data.get('timestamp', data.get('ts')))
    if stamp is None or not (-health.STATUS_CLOCK_SKEW_S <= env.wall() - stamp <= CLAIM_STATUS_MAX_AGE_S):
        raise Refused('the status file is stale or has no timestamp (it must be at most %.0f s old)' % CLAIM_STATUS_MAX_AGE_S)
    if data.get('state') != 'ok':
        raise Refused('the status file says the drive state is %r, not ok' % (data.get('state'),))
    if str(data.get('by_id') or '') != name:
        raise Refused('the status file names a different drive (%r) than the one typed back' % (data.get('by_id'),))
    recorded = data.get('size_bytes', data.get('namespace_bytes'))
    if recorded is not None:
        if int(recorded) != size:
            raise Refused('the status file records %s bytes, the device reads %d' % (recorded, size))
    elif typed_size is None:
        raise Refused('the status file records no size: type the size back with --size %d' % size)
    elif typed_size != size:
        raise Refused('--size %d is not the device\'s size (%d)' % (typed_size, size))


def claim(device_arg: str, confirm: str, *, status_path: str, layout_id: str = 'gemma-chipmajor-v1', mode: str = 'process',
          typed_size: Optional[int] = None, overwrite: Optional[str] = None, dry_run: bool = False, allow_regular: bool = False,
          object_bytes: Optional[int] = None, env: Optional[health.HostEnv] = None) -> Dict[str, Any]:
    """Claim the drive. Returns a report dict; raises Refused (nothing written) or a StoreError. ``object_bytes`` (tests, regular files
    only) names a layout the table does not know."""
    env = env or health.HostEnv()
    if allow_regular and not env.test_mode:
        # the by-id gate is skipped for a regular file, so the flag itself is gated (the same name that gates every other test-only path)
        raise Refused('--allow-regular is for tests: it needs %s=1 in the environment' % health.TEST_ENV)
    if object_bytes is not None and not allow_regular:
        raise Refused('an explicit object size is for regular-file tests only')
    if object_bytes is None:
        if layout_id not in store.KNOWN_LAYOUTS:
            raise Refused('unknown layout %r (known: %s)' % (layout_id, ', '.join(sorted(store.KNOWN_LAYOUTS))))
        object_bytes = store.KNOWN_LAYOUTS[layout_id]
    if mode not in store.MODES:
        raise Refused('unknown mode %r' % (mode,))
    if overwrite is not None and overwrite not in FORMATS:
        raise Refused('--overwrite names no format this tool knows: %r (known: %s)' % (overwrite, ', '.join(FORMATS)))
    name, path = check_by_id(device_arg, confirm, allow_regular)
    if allow_regular and not os.path.isfile(path):
        raise Refused('--allow-regular takes a regular file; %s is not one (a block device is claimed by its by-id path, without the flag)' % path)
    try:
        drive = store.DriveFile(path, allow_regular=allow_regular, exclusive=True)
    except store.DeviceBusy as error:
        raise Refused('the drive is in use: %s' % (error,))
    try:
        if drive.size <= 0:
            raise Refused('the drive reads 0 bytes')
        check_status(status_path, name, drive.size, typed_size, env)
        seen, detail = probe(drive)
        geometry = store.plan_geometry(drive.size, object_bytes, object_align=store.SECTOR if allow_regular else store.ALIGN)
        digest = store.claim_digest(name, drive.size)
        report: Dict[str, Any] = {
            'format_seen': seen, 'detail': detail, 'layout': layout_id, 'mode': mode, 'object_bytes': geometry.object_bytes,
            'num_slots': geometry.num_slots, 'data_off': geometry.data_off, 'device_bytes': geometry.device_bytes,
            'claim': digest.hex(), 'dry_run': dry_run, 'written': False,
        }
        if seen == 'gemma-kv-v2':
            existing = store.read_superblocks(drive)
            if existing and existing[0].claim == digest and existing[0].layout_id == layout_id and existing[0].mode == mode \
                    and existing[0].num_slots == geometry.num_slots and existing[0].data_off == geometry.data_off:
                report['already_claimed'] = True
                return report
        if seen != 'blank' and overwrite != seen:
            raise Refused('the drive carries %s (%s); to replace it type --overwrite=%s' % (seen, detail, seen))
        if dry_run:
            return report
        zero = bytes(min(PROBE_BYTES, drive.size // store.ALIGN * store.ALIGN))
        drive.write(0, zero)                                              # stale signatures out of the head ...
        if drive.size > 2 * PROBE_BYTES:
            drive.write((drive.size - TAIL_BYTES) // store.ALIGN * store.ALIGN, bytes(TAIL_BYTES))     # ... and a backup GPT out of the tail
        drive.sync()
        sb = store.new_superblock(geometry=geometry, layout_id=layout_id, claim=digest, mode=mode, created=int(env.wall()))
        store.write_superblocks(drive, sb)
        written = store.verify_superblock(drive, layout_id=layout_id, object_bytes=geometry.object_bytes, persistence=mode, expect_claim=digest)
        report['written'] = True
        report['uuid'] = written.uuid.hex()
        return report
    finally:
        drive.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='ttkvp2p.claim', description=__doc__.split('\n\n')[0])
    parser.add_argument('--device', required=True, help='the drive\'s by-id path')
    parser.add_argument('--confirm', required=True, help='the by-id NAME, typed back')
    parser.add_argument('--status', required=True, help='the cache-drive status file (status.json) the host writes')
    parser.add_argument('--size', type=int, default=None, help='the drive\'s size in bytes, typed back when the status file records none')
    parser.add_argument('--layout', default='gemma-chipmajor-v1')
    parser.add_argument('--mode', default='process', choices=store.MODES)
    parser.add_argument('--overwrite', default=None, help='the format the drive carries now, named exactly (see the refusal message)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-regular', action='store_true', help='tests only (needs %s=1): a regular file instead of a by-id block device' % health.TEST_ENV)
    args = parser.parse_args(argv)
    try:
        report = claim(args.device, args.confirm, status_path=args.status, layout_id=args.layout, mode=args.mode, typed_size=args.size,
                       overwrite=args.overwrite, dry_run=args.dry_run, allow_regular=args.allow_regular)
    except Refused as error:
        print('REFUSED: %s' % (error,), file=sys.stderr)
        return 2
    except store.StoreError as error:
        print('FAILED: %s' % (error,), file=sys.stderr)
        return 3
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
