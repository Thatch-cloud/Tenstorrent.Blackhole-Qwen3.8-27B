"""What the ttnn fabric API exposes, read with no device attached.

The packet size that caps collective bandwidth is not an environment variable -
run 35425588508 searched the shipped shared objects and found none. The source
says it comes from `FabricContext::validate_and_apply_packet_size(hal, arch,
requested_size)`, so something requests it and the question is whether anything
reachable from Python can.

Runs in a container WITHOUT --device, so it costs no hardware slot: importing
ttnn and reading signatures does not open a card.
"""

import inspect
import json
import sys

ENUMS = ('FabricConfig', 'FabricTensixConfig', 'FabricRouterConfig',
         'FabricUDMMode', 'FabricReliabilityMode', 'FabricManagerMode')
FUNCS = ('set_fabric_config', 'get_tt_fabric_max_payload_size_bytes',
         'get_tt_fabric_packet_header_size_bytes', 'get_fabric_config')


def main():
    report = {}
    try:
        import ttnn
    except BaseException as error:
        print(json.dumps({'import_error': '%s: %s' % (type(error).__name__, error)}))
        return 1

    for name in FUNCS:
        fn = getattr(ttnn, name, None)
        if fn is None:
            report[name] = 'ABSENT'
            continue
        entry = {'doc': (getattr(fn, '__doc__', '') or '')[:600]}
        try:
            entry['signature'] = str(inspect.signature(fn))
        except BaseException:
            entry['signature'] = 'unavailable (pybind overload)'
        report[name] = entry

    for name in ENUMS:
        cls = getattr(ttnn, name, None)
        report[name] = ('ABSENT' if cls is None else
                        [m for m in dir(cls) if not m.startswith('_')])

    report['packet_symbols'] = sorted(
        n for n in dir(ttnn)
        if 'packet' in n.lower() or 'payload' in n.lower() or 'edm' in n.lower())

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
