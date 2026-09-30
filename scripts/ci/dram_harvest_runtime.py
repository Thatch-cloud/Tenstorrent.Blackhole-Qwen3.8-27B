"""Open each card briefly so the runtime prints its SoC descriptor, then report DRAM facts.

UMD exposes harvesting_masks only in C++ (cluster_descriptor_types.hpp), and the
pinned image ships no tt_umd python module, so the mask is read from what the
runtime itself reports at init. Run with TT_METAL_LOGGER_LEVEL=Debug and capture
the whole stream: the workflow greps it for harvesting lines.

This loads no model, compiles no kernel beyond device init, and writes no firmware.
"""

import json
import sys

ARCHITECTURAL_DRAM_CHANNELS = 8


def main():
    report = dict(devices={}, errors=[], num_available_devices=None)
    try:
        import ttnn
    except BaseException as error:
        report['errors'].append('import ttnn: %r' % (error,))
        print(json.dumps(report, indent=2))
        return 1

    for name in ('GetNumAvailableDevices', 'get_num_devices'):
        getter = getattr(ttnn, name, None)
        if getter is None:
            continue
        try:
            report['num_available_devices'] = getter()
            break
        except BaseException as error:
            report['errors'].append('%s: %r' % (name, error))

    count = report['num_available_devices'] or 0
    for index in range(count):
        entry = {}
        device = None
        try:
            device = ttnn.open_device(device_id=index)
            for attribute in ('num_dram_channels', 'dram_size_per_channel',
                              'dram_grid_size', 'compute_with_storage_grid_size',
                              'arch', 'id'):
                probe = getattr(device, attribute, None)
                if probe is None:
                    continue
                try:
                    entry[attribute] = str(probe() if callable(probe) else probe)
                except BaseException as error:
                    entry[attribute] = 'error: %r' % (error,)
            channels = entry.get('num_dram_channels')
            if channels is not None and channels.isdigit():
                entry['fewer_than_architectural'] = int(channels) < ARCHITECTURAL_DRAM_CHANNELS
        except BaseException as error:
            entry['error'] = repr(error)
        finally:
            if device is not None:
                try:
                    ttnn.close_device(device)
                except BaseException as error:
                    entry['close_error'] = repr(error)
        report['devices'][str(index)] = entry

    print('=== STRUCTURED REPORT ===')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
