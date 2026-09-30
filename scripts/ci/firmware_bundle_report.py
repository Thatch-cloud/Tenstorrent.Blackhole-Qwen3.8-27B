"""Print the firmware bundle each card reports, from a tt-smi snapshot."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--label', default='')
    options = parser.parse_args()
    if not options.snapshot.is_file():
        print('%s: no snapshot' % (options.label or 'firmware'))
        return
    snapshot = json.loads(options.snapshot.read_text())
    for index, device in enumerate(snapshot.get('device_info', [])):
        firmwares = device.get('firmwares') or {}
        print('%-6s device %d  bundle=%s  eth=%s  cm=%s  gddr=%s'
              % (options.label, index, firmwares.get('fw_bundle_version'),
                 firmwares.get('eth_fw'), firmwares.get('cm_fw'),
                 firmwares.get('gddr_fw')))


if __name__ == '__main__':
    main()
