"""Pinned artefacts for a 19.12.0 firmware maintenance window, and their verification.

Nothing here flashes. This module only names what must be present on the host,
with the exact sizes and SHA256 digests published by the upstream releases, so a
staging run either reproduces them byte for byte or fails.

tt-flash 3.11.0 is chosen deliberately over 4.0.0: 4.0.0 introduces the board
variable / SPI flash interlock required by firmware 19.15, which is not the target
here. 3.11.0 is comfortably above the 3.6.0 floor that 19.5.0 set for Blackhole
(below it the board ID is lost on flash).
"""

import argparse
import hashlib
import json
from pathlib import Path

FIRMWARE_VERSION = '19.12.0'
TT_FLASH_VERSION = '3.11.0'
FIRMWARE_BASE = ('https://github.com/tenstorrent/tt-system-firmware/releases/download/v%s'
                 % FIRMWARE_VERSION)
TT_FLASH_BASE = 'https://github.com/tenstorrent/tt-flash/releases/download/v%s' % TT_FLASH_VERSION

ARTEFACTS = {
    'fw_pack-19.12.0.fwbundle': dict(
        url='%s/fw_pack-19.12.0.fwbundle' % FIRMWARE_BASE, size=5550608,
        sha256='bf43882ae99b5cd127860f98ab21450d30815b031988b479aaf701c7e3b57a4e',
        role='firmware bundle to flash'),
    'p150a.fwbundle': dict(
        url='%s/p150a.fwbundle' % FIRMWARE_BASE, size=383586,
        sha256='244c74fe01857dc6ea9d8be064b85f61c1ab9c63b7fbbc929ba97eeb33032faa',
        role='board-specific bundle, kept for reference'),
    'fw-pack-v19.12.0-recovery.tar.gz': dict(
        url='%s/fw-pack-v19.12.0-recovery.tar.gz' % FIRMWARE_BASE, size=7743194,
        sha256='96a3fd2558c114a39bfb639571de9192b6beaa1c6b672fffd609d36d2ae7555c',
        role='recovery image for a failed flash'),
    'tt-flash-3.11.0-ubuntu-22.04': dict(
        url='%s/tt-flash-3.11.0-ubuntu-22.04' % TT_FLASH_BASE, size=25588792,
        sha256='ce2712f82e645e83de0893811595ba2a2a7008c3eda85117dd4ae6ee8d3f84ff',
        role='flasher, ubuntu 22.04 host'),
    'tt-flash-3.11.0-ubuntu-24.04': dict(
        url='%s/tt-flash-3.11.0-ubuntu-24.04' % TT_FLASH_BASE, size=25536296,
        sha256='7e3cb4a71d538cc630906c37fde2f73849d4ff5e2c023015394950183bb77f74',
        role='flasher, ubuntu 24.04 host'),
    # The rig runs Ubuntu 26.04, for which upstream publishes no standalone binary.
    # The wheel is py3-none-any, so it installs into a venv on any release; that
    # also sidesteps PEP 668, which blocks pip into the system interpreter.
    'tt_flash-3.11.0-py3-none-any.whl': dict(
        url='%s/tt_flash-3.11.0-py3-none-any.whl' % TT_FLASH_BASE, size=65728,
        sha256='e0bf01bf5b6349abc21afcd829a17561923eb23f9a5b34e30831ad1be1214dc0',
        role='flasher wheel, any host release'),
}

# Upstream publishes standalone flasher binaries only for these releases.
BINARY_HOST_RELEASES = {'22.04': 'tt-flash-3.11.0-ubuntu-22.04',
                        '24.04': 'tt-flash-3.11.0-ubuntu-24.04'}
WHEEL = 'tt_flash-3.11.0-py3-none-any.whl'


def flasher_for_release(version_id):
    """Pick the standalone binary when upstream ships one, else the portable wheel."""
    return BINARY_HOST_RELEASES.get(str(version_id), WHEEL)

# Kept on the host as the pre-upgrade image. Downgrades below v19 are unsupported
# upstream; 19.8.1 is within v19 but a downgrade is still an unproven path.
ROLLBACK_BUNDLE = '/home/thatch/fw_pack-19.8.1.fwbundle'


def digest(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def verify(directory, names=None):
    """Check staged files against the pinned manifest; report, never raise."""
    base = Path(directory)
    results = {}
    for name in (names or ARTEFACTS):
        expected = ARTEFACTS[name]
        path = base / name
        entry = dict(role=expected['role'], expected_sha256=expected['sha256'],
                     expected_size=expected['size'], present=path.is_file())
        if entry['present']:
            entry['actual_size'] = path.stat().st_size
            entry['actual_sha256'] = digest(path)
            entry['size_ok'] = entry['actual_size'] == expected['size']
            entry['sha256_ok'] = entry['actual_sha256'] == expected['sha256']
        else:
            entry.update(actual_size=None, actual_sha256=None, size_ok=False, sha256_ok=False)
        entry['verified'] = bool(entry['present'] and entry['size_ok'] and entry['sha256_ok'])
        results[name] = entry
    return results


def summarize(results):
    required = [name for name, entry in results.items() if entry['verified']]
    return dict(
        firmware_version=FIRMWARE_VERSION, tt_flash_version=TT_FLASH_VERSION,
        artefacts=results, verified_count=len(required), total=len(results),
        all_verified=all(entry['verified'] for entry in results.values()),
        rollback_bundle=ROLLBACK_BUNDLE,
        firmware_modified=False, devices_reset=False, flash_performed=False,
        scope=('Staging only. Files are downloaded and checksummed; no card is flashed, '
               'reset or power-cycled, and the flasher is left outside PATH.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--name', action='append', dest='names')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    report = summarize(verify(options.directory, options.names))
    options.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['all_verified'] else 1)
